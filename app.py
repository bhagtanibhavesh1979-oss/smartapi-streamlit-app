import streamlit as st
import requests
import pyotp
import math
import pandas as pd
import json
from datetime import datetime

# Optional auto-refresh: safe fallback if not installed
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False
    def st_autorefresh(*args, **kwargs):
        return None

st.set_page_config(page_title="Index Options Analyzer", layout="wide")

# ---------------- SESSION STATE ----------------
for key, default in {
    "logged_in": False,
    "jwt_token": None,
    "headers": None,
    "token_data": None,
    "prev_oi": {},
    "prev_prices": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- LOGIN ----------------
def login(api_key, client_code, pin, totp_secret):
    try:
        totp = pyotp.TOTP(totp_secret).now()
        url = "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword"
        payload = {"clientcode": client_code, "password": pin, "totp": totp}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "00-00-00-00-00-00",
            "X-PrivateKey": api_key,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        data = r.json()
        if data.get("status") and data.get("data", {}).get("jwtToken"):
            return data["data"]["jwtToken"]
        return None
    except Exception:
        return None

def get_headers(api_key, jwt_token):
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00-00-00-00-00-00",
        "X-PrivateKey": api_key,
        "Authorization": f"Bearer {jwt_token}",
    }

# ---------------- TOKEN MASTER ----------------
@st.cache
def load_token_master():
    with open("OpenAPIScripMaster.json", "r") as f:
        return json.load(f)

def get_upcoming_expiries(token_data, symbol="NIFTY", lookahead_days=120, count=4):
    expiries = []
    for inst in token_data:
        if inst.get("exch_seg") == "NFO" and inst.get("name") == symbol and inst.get("instrumenttype") == "OPTIDX":
            expiries.append(inst["expiry"])
    expiries = sorted(set(expiries), key=lambda x: datetime.strptime(x, "%d%b%Y"))
    today = datetime.now()
    filtered = []
    for exp in expiries:
        try:
            exp_date = datetime.strptime(exp, "%d%b%Y")
            if 0 <= (exp_date - today).days <= lookahead_days:
                filtered.append(exp)
        except Exception:
            continue
    return filtered[:count] if filtered else expiries[:count]

def find_option_token(token_data, symbol, strike, option_type, expiry):
    strike_in_master = float(strike) * 100  # Angel stores strike*100
    for inst in token_data:
        if (
            inst.get("exch_seg") == "NFO"
            and inst.get("name") == symbol
            and inst.get("instrumenttype") == "OPTIDX"
            and inst.get("expiry") == expiry
            and float(inst.get("strike", 0)) == strike_in_master
            and option_type in inst.get("symbol", "")
        ):
            return inst.get("token")
    return None

# ---------------- MARKET DATA ----------------
INDEX_TOKENS = {
    "NIFTY": ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"),
    "SENSEX": (None, None),  # Update if you have a valid token
}

def quote(headers, exchange, tokens):
    url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote/"
    payload = {"mode": "LTP", "exchangeTokens": {exchange: tokens}}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    return r.json()

def get_spot_price(headers, symbol="NIFTY"):
    exch, token = INDEX_TOKENS.get(symbol, (None, None))
    if not exch or not token:
        return None
    try:
        data = quote(headers, exch, [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            return float(fetched[0].get("ltp"))
    except Exception:
        return None
    return None

def get_option_fields(headers, token):
    # Returns live price and OI if available
    if not token:
        return None, None
    try:
        data = quote(headers, "NFO", [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            row = fetched[0]
            ltp = row.get("ltp")
            oi = row.get("oi") or row.get("openinterest")
            return (float(ltp) if ltp is not None else None), (int(oi) if oi is not None else None)
    except Exception:
        return None, None
    return None, None

# ---------------- BLACK-SCHOLES + GREEKS ----------------
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or S is None or S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def greeks(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or S is None or S <= 0 or K <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma)
    delta = norm_cdf(d1) if option_type == "CE" else norm_cdf(d1) - 1.0
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    theta_call = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm_cdf(d2)
    theta_put = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm_cdf(-d2)
    theta = theta_call if option_type == "CE" else theta_put
    vega = S * norm_pdf(d1) * math.sqrt(T)
    return delta, gamma, theta, vega

# ---------------- Implied volatility (Black–Scholes inversion) ----------------
def implied_vol(S, K, T, r, price, option_type="CE", max_iter=30, tol=1e-6):
    # Guardrails
    if S is None or S <= 0 or K <= 0 or T <= 0 or price is None or price <= 0:
        return None
    # No-arbitrage intrinsic bounds
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    # If price below intrinsic (or absurdly high), fail gracefully
    if price < intrinsic or price > max(S, K):
        return None

    # Newton-Raphson with clamps
    sigma = 0.2  # initial guess
    lower, upper = 0.01, 3.0  # 1% to 300% annualized
    for _ in range(max_iter):
        bs = black_scholes(S, K, T, r, sigma, option_type)
        _, _, _, vega = greeks(S, K, T, r, sigma, option_type)
        diff = bs - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:  # avoid division by near-zero
            break
        sigma = sigma - diff / vega
        # clamp to bounds
        if sigma < lower: sigma = lower
        if sigma > upper: sigma = upper

    # Fallback: bisection if NR didn’t converge
    lo, hi = lower, upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        bs_mid = black_scholes(S, K, T, r, mid, option_type)
        if abs(bs_mid - price) < tol:
            return mid
        if bs_mid > price:
            hi = mid
        else:
            lo = mid
    return None

# ---------------- STRIKE LOGIC ----------------
def get_step_for_index(index_choice):
    if index_choice == "BANKNIFTY":
        return 100
    elif index_choice in ["NIFTY", "FINNIFTY"]:
        return 50
    else:
        return 50

def round_up_to_step(x, step):
    return math.ceil(x / step) * step

def round_down_to_step(x, step):
    return math.floor(x / step) * step

def build_call_strikes_above_spot(spot, step, count):
    if spot is None or spot <= 0:
        return []
    start = round_up_to_step(spot, step)
    return [start + i * step for i in range(count)]

def build_put_strikes_below_spot(spot, step, count):
    if spot is None or spot <= 0:
        return []
    start = round_down_to_step(spot, step) - step
    return [start - i * step for i in range(count)]

# ---------------- SPOT FALLBACK (ATM parity) ----------------
def infer_spot_from_atm(headers, token_data, index_choice, expiry_choice, step):
    # Try a few candidate strikes around round anchors; Spot ≈ K + CE - PE
    anchors = [25000, 40000, 18000]
    candidates = []
    for a in anchors:
        base = round_up_to_step(a, step)
        candidates += [base - step, base, base + step]
    candidates = sorted(set([x for x in candidates if x > 0]))

    for K in candidates[:6]:
        ce_token = find_option_token(token_data, index_choice, K, "CE", expiry_choice)
        pe_token = find_option_token(token_data, index_choice, K, "PE", expiry_choice)
        ce_live, _ = get_option_fields(headers, ce_token)
        pe_live, _ = get_option_fields(headers, pe_token)
        if ce_live is not None and pe_live is not None:
            return K + (ce_live - pe_live)
    return None

# ---------------- UI ----------------
st.title("📈 Index Options Analyzer")

# Login form only if not logged in
if not st.session_state.logged_in:
    with st.form("login_form"):
        api_key = st.text_input("API Key", type="password")
        client_code = st.text_input("Client Code")
        pin = st.text_input("Trading PIN", type="password")
        totp_secret = st.text_input("TOTP Secret", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        jwt_token = login(api_key, client_code, pin, totp_secret)
        if jwt_token:
            st.session_state.jwt_token = jwt_token
            st.session_state.headers = get_headers(api_key, jwt_token)
            try:
                st.session_state.token_data = load_token_master()
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.experimental_rerun()
            except Exception:
                st.error("Instrument master load failed. Please retry.")
        else:
            st.error("Login failed. Please verify API key, client code, PIN, and TOTP secret.")

# Main app if logged in
if st.session_state.logged_in:
    headers = st.session_state.headers
    token_data = st.session_state.token_data

    # Top controls
    top_left, top_right = st.columns([6, 2])
    with top_right:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.jwt_token = None
            st.session_state.headers = None
            st.session_state.token_data = None
            st.experimental_rerun()

    # Refresh toggle + manual button
    st.markdown("### Refresh controls")
    auto_refresh = st.checkbox("Enable auto refresh (15s)", value=False)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=15 * 1000, key="data_refresh")
        st.info("Auto refresh is ON (every 15 seconds).")
    elif auto_refresh and not HAS_AUTOREFRESH:
        st.warning("Auto refresh library not installed. Run: pip install streamlit-autorefresh")
    else:
        if st.button("🔄 Refresh data"):
            st.experimental_rerun()

    # Option chain controls
    st.markdown("### Option chain")
    left, right = st.columns([2, 2])
    with left:
        index_choice = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])
        expiries = get_upcoming_expiries(token_data, index_choice, lookahead_days=120, count=4)
        if not expiries:
            st.warning("No upcoming expiries in 120 days. Showing available expiries.")
            expiries = get_upcoming_expiries(token_data, index_choice, lookahead_days=365, count=4)
        expiry_choice = st.selectbox("Expiry", expiries)
    with right:
        risk_free_rate = st.slider("Risk-Free Rate (%)", 5.0, 10.0, 7.0) / 100.0
        volatility = st.slider("Model Volatility (%)", 10.0, 40.0, 18.0) / 100.0
        strike_count = st.slider("Strikes each side", 3, 15, 8 if index_choice == "BANKNIFTY" else 5)
        alert_misprice = st.slider("Mispricing alert threshold (₹)", 5, 200, 50)
        alert_oi_jump = st.slider("OI surge alert threshold (%)", 5, 500, 50)

    # IV toggle and settings
    st.markdown("### Implied volatility settings")
    show_iv = st.checkbox("Compute implied volatility (from live prices)", value=True)
    iv_max_iter = st.slider("IV max iterations", 10, 60, 30)
    iv_tol = 1e-6  # fixed tolerance for stability

    # Spot + fallback
    step = get_step_for_index(index_choice)
    spot = get_spot_price(headers, index_choice)
    if spot is None and index_choice != "SENSEX":
        inferred_spot = infer_spot_from_atm(headers, token_data, index_choice, expiry_choice, step)
        if inferred_spot:
            spot = inferred_spot
            st.warning(f"Spot unavailable; using inferred spot ≈ ₹{spot:.2f} from ATM parity.")
        else:
            st.warning("Spot unavailable (market closed or rate limit). Live prices may be N/A; fair values still compute.")
    st.info(f"{index_choice} Spot: {'N/A' if spot is None else f'₹{spot:.2f}'}")

    # Build CE/PE strike lists
    ce_strikes = build_call_strikes_above_spot(spot, step, strike_count)
    pe_strikes = build_put_strikes_below_spot(spot, step, strike_count)

    # Time to expiry
    try:
        days_to_expiry = (datetime.strptime(expiry_choice, "%d%b%Y") - datetime.now()).days
    except Exception:
        days_to_expiry = 0
    T = max(days_to_expiry, 0) / 365.0

    # Build Calls table
    call_rows = []
    alerts = []
    for K in ce_strikes:
        ce_token = find_option_token(token_data, index_choice, K, "CE", expiry_choice)
        ce_live, ce_oi = get_option_fields(headers, ce_token)
        ce_fair = black_scholes(spot if spot else K, K, T, risk_free_rate, volatility, "CE")
        ce_delta, ce_gamma, ce_theta, ce_vega = greeks(spot if spot else K, K, T, risk_free_rate, volatility, "CE")

        if ce_live is not None and abs(ce_live - ce_fair) >= alert_misprice:
            alerts.append(f"CE mispricing at {K}: Live {ce_live:.2f} vs Fair {ce_fair:.2f}")

        key = (index_choice, expiry_choice, K, "CE")
        prev = st.session_state.prev_oi.get(key)
        if prev and ce_oi and prev > 0:
            change_pct = ((ce_oi - prev) / prev) * 100.0
            if change_pct >= alert_oi_jump:
                alerts.append(f"CE OI surge at {K}: {change_pct:.1f}% (prev {prev}, now {ce_oi})")
        if ce_oi is not None:
            st.session_state.prev_oi[key] = ce_oi

        # Optional IV
        iv = None
        if show_iv and ce_live is not None and T > 0:
            iv = implied_vol(spot if spot else K, K, T, risk_free_rate, ce_live, "CE", max_iter=iv_max_iter, tol=iv_tol)

        call_rows.append({
            "Strike": K,
            "Live": round(ce_live, 2) if ce_live is not None else None,
            "Fair": round(ce_fair, 2),
            "Δ": round(ce_delta, 4),
            "Γ": round(ce_gamma, 6),
            "Θ (per day)": round(ce_theta / 365.0, 4),
            "Vega": round(ce_vega, 4),
            "OI": ce_oi,
            "IV (%)": round(iv * 100, 2) if iv is not None else None,
        })
    calls_df = pd.DataFrame(call_rows)

    # Build Puts table
    put_rows = []
    for K in pe_strikes:
        pe_token = find_option_token(token_data, index_choice, K, "PE", expiry_choice)
        pe_live, pe_oi = get_option_fields(headers, pe_token)
        pe_fair = black_scholes(spot if spot else K, K, T, risk_free_rate, volatility, "PE")
        pe_delta, pe_gamma, pe_theta, pe_vega = greeks(spot if spot else K, K, T, risk_free_rate, volatility, "PE")

        if pe_live is not None and abs(pe_live - pe_fair) >= alert_misprice:
            alerts.append(f"PE mispricing at {K}: Live {pe_live:.2f} vs Fair {pe_fair:.2f}")

        key = (index_choice, expiry_choice, K, "PE")
        prev = st.session_state.prev_oi.get(key)
        if prev and pe_oi and prev > 0:
            change_pct = ((pe_oi - prev) / prev) * 100.0
            if change_pct >= alert_oi_jump:
                alerts.append(f"PE OI surge at {K}: {change_pct:.1f}% (prev {prev}, now {pe_oi})")
        if pe_oi is not None:
            st.session_state.prev_oi[key] = pe_oi

        iv = None
        if show_iv and pe_live is not None and T > 0:
            iv = implied_vol(spot if spot else K, K, T, risk_free_rate, pe_live, "PE", max_iter=iv_max_iter, tol=iv_tol)

        put_rows.append({
            "Strike": K,
            "Live": round(pe_live, 2) if pe_live is not None else None,
            "Fair": round(pe_fair, 2),
            "Δ": round(pe_delta, 4),
            "Γ": round(pe_gamma, 6),
            "Θ (per day)": round(pe_theta / 365.0, 4),
            "Vega": round(pe_vega, 4),
            "OI": pe_oi,
            "IV (%)": round(iv * 100, 2) if iv is not None else None,
        })
    puts_df = pd.DataFrame(put_rows)

    # Alerts section
    if alerts:
        st.markdown("### Alerts")
        for msg in alerts:
            st.warning(msg)

    # Tables + downloads (wider for readability)
    st.markdown("### Calls (strikes above spot)")
    st.dataframe(calls_df, width=1600, height=420)
    st.download_button("Download Calls CSV", calls_df.to_csv(index=False), "calls.csv", "text/csv")

    st.markdown("### Puts (strikes below spot)")
    st.dataframe(puts_df, width=1600, height=420)
    st.download_button("Download Puts CSV", puts_df.to_csv(index=False), "puts.csv", "text/csv")

    # Footer metrics
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Selected index", index_choice)
    cB.metric("Spot", "N/A" if spot is None else f"₹{spot:.2f}")
    cC.metric("Expiry", expiry_choice)
    cD.metric("Days to expiry", max(days_to_expiry, 0))
