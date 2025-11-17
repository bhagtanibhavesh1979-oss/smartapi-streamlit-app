import streamlit as st
import requests
import pyotp
import math
import pandas as pd
from datetime import datetime, date

# ---------- Optional auto-refresh ----------
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False
    def st_autorefresh(*args, **kwargs):
        return None

def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass

st.set_page_config(page_title="Index Options Analyzer", layout="wide")

# ---------- Session defaults ----------
for key, default in {
    "logged_in": False,
    "login_date": None,
    "jwt_token": None,
    "headers": None,
    "token_data": None,
    "prev_oi": {},
    "index_choice": "NIFTY",
    "expiry_choice": "",
    "risk_free_rate_pct": 7.0,
    "strike_count": 5,
    "auto_refresh": False,
    "use_atm_iv_for_fair": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Persist login until midnight
if st.session_state.get("logged_in"):
    today_str = date.today().isoformat()
    if st.session_state.get("login_date") != today_str:
        st.session_state.logged_in = False
        st.session_state.jwt_token = None
        st.session_state.headers = None
        st.session_state.token_data = None

# ---------- Auth helpers ----------
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

# ---------- Token master ----------
@st.cache  # Streamlit 1.12-compatible
def load_token_master():
    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_upcoming_expiries(token_data, symbol="NIFTY", lookahead_days=120, count=4):
    if not token_data:
        return []
    expiries = []
    for inst in token_data:
        try:
            inst_exch = (inst.get("exch_seg") or "").upper()
            inst_type = (inst.get("instrumenttype") or "").upper()
            inst_name = (inst.get("name") or "").upper()
            inst_symbol = (inst.get("symbol") or "").upper()
            # Accept NFO OPTIDX (regular index options) and BFO OPTIDX (SENSEX on BSE derivatives)
            if inst_type == "OPTIDX" and (inst_exch in ("NFO", "BFO")) and (inst_name == symbol or symbol in inst_symbol):
                e = (inst.get("expiry") or "").strip()
                if e:
                    expiries.append(e)
        except Exception:
            continue
    try:
        expiries = sorted(set(expiries), key=lambda x: datetime.strptime(x, "%d%b%Y"))
    except Exception:
        expiries = list(dict.fromkeys(expiries))
    today = datetime.now()
    filtered = []
    for exp in expiries:
        try:
            exp_date = datetime.strptime(exp, "%d%b%Y")
            days = (exp_date - today).days
            if 0 <= days <= lookahead_days:
                filtered.append(exp)
        except Exception:
            continue
    return filtered[:count] if filtered else expiries[:count]

def find_option_token(token_data, symbol, strike, option_type, expiry):
    if not token_data:
        return None
    try:
        strike_in_master = float(strike) * 100  # master stores strike*100
    except Exception:
        strike_in_master = None
    for inst in token_data:
        try:
            inst_exch = (inst.get("exch_seg") or "").upper()
            inst_type = (inst.get("instrumenttype") or "").upper()
            inst_name = (inst.get("name") or "").upper()
            inst_sym = (inst.get("symbol") or "").upper()
            inst_exp = (inst.get("expiry") or "").strip()
            inst_strike = inst.get("strike")
            if inst_type != "OPTIDX" or inst_exch not in ("NFO", "BFO"):
                continue
            if inst_exp != expiry:
                continue
            if not (inst_name == symbol or symbol in inst_sym):
                continue
            try:
                inst_strike_f = float(inst_strike) if inst_strike is not None else None
            except Exception:
                inst_strike_f = None
            if inst_strike_f is None or strike_in_master is None:
                continue
            if inst_strike_f == strike_in_master and option_type in inst_sym:
                return inst.get("token")
        except Exception:
            continue
    return None

# ---------- Market data ----------
INDEX_TOKENS = {
    "NIFTY": ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"),
    "SENSEX": ("BSE", "99919000"),  # your SENSEX spot token
}

def quote(headers, exchange, tokens):
    url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote/"
    payload = {"mode": "LTP", "exchangeTokens": {exchange: tokens}}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    return r.json()

def get_spot_price(headers, symbol="NIFTY"):
    exch, token = INDEX_TOKENS.get(symbol, (None, None))
    if not headers or not exch or not token:
        return None
    try:
        data = quote(headers, exch, [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            ltp = fetched[0].get("ltp")
            return float(ltp) if ltp is not None else None
    except Exception:
        pass
    return None

def get_option_fields(headers, token, exchange="NFO"):
    if not token or not headers:
        return None, None
    try:
        data = quote(headers, exchange, [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            row = fetched[0]
            ltp = row.get("ltp")
            oi = row.get("oi") or row.get("openinterest")
            return (float(ltp) if ltp is not None else None), (int(oi) if oi is not None else None)
    except Exception:
        pass
    return None, None

# ---------- Black–Scholes + Greeks ----------
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes(S, K, T, r, sigma, option_type="CE"):
    if S is None or S <= 0 or K <= 0:
        return 0.0
    if T <= 0:
        return max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if sigma is None or sigma <= 0:
        sigma = 1e-6
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def delta_only(S, K, T, r, sigma, option_type="CE"):
    if S is None or S <= 0 or K <= 0 or T < 0:
        return 0.0
    if T == 0:
        if option_type == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = d1_d2(S, K, T, r, max(sigma, 1e-6))
    return norm_cdf(d1) if option_type == "CE" else norm_cdf(d1) - 1.0

def implied_vol(S, K, T, r, price, option_type="CE", max_iter=30, tol=1e-6):
    if S is None or S <= 0 or K <= 0 or T <= 0 or price is None or price <= 0:
        return None
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if price < intrinsic or price > max(S, K):
        return None
    sigma = 0.2
    lower, upper = 0.01, 3.0
    for _ in range(max_iter):
        bs = black_scholes(S, K, T, r, sigma, option_type)
        d1, _ = d1_d2(S, K, T, r, sigma)
        vega = S * norm_pdf(d1) * math.sqrt(T)
        diff = bs - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break
        sigma = sigma - diff / vega
        sigma = max(lower, min(upper, sigma))
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

# ---------- Strikes ----------
def get_step_for_index(index_choice):
    if index_choice == "BANKNIFTY":
        return 100
    elif index_choice in ["NIFTY", "FINNIFTY", "SENSEX"]:
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

# ---------- ATM helpers ----------
def nearest_atm_strike(spot, step):
    if spot is None or spot <= 0:
        return None
    up = round_up_to_step(spot, step)
    down = round_down_to_step(spot, step)
    return up if abs(up - spot) <= abs(spot - down) else down

def get_exchange_for_index(index_choice):
    # Options exchange: NIFTY/BANKNIFTY/FINNIFTY on NFO; SENSEX on BFO
    return "BFO" if index_choice == "SENSEX" else "NFO"

# ---------- UI ----------
st.markdown("<h3 style='margin-bottom:6px;'>📈 Index Options Analyzer</h3>", unsafe_allow_html=True)

# Login form
if not st.session_state.get("logged_in", False):
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
                st.session_state.login_date = date.today().isoformat()
                st.success("Login successful!")
                safe_rerun()
            except Exception as e:
                st.error(f"Instrument master load failed: {e}")
        else:
            st.error("Login failed. Verify credentials and TOTP.")

# Stop before chain if not logged in
if not st.session_state.get("logged_in", False):
    st.info("Please log in to load the option chain and market data.")
    st.stop()

headers = st.session_state.get("headers", None)
token_data = st.session_state.get("token_data", None)
if not headers:
    st.warning("Session headers missing. Please log in again or refresh.")
    st.stop()

# Spot strip under logo
spot_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
spots = {sym: get_spot_price(headers, sym) for sym in spot_symbols}
strip_cols = st.columns(len(spot_symbols))
for i, sym in enumerate(spot_symbols):
    val = spots.get(sym, None)
    strip_cols[i].metric(sym, "N/A" if val is None else f"₹{val:.2f}")

# Filters
f1, f2, f3 = st.columns([2, 2, 1])
with f1:
    st.session_state.index_choice = st.selectbox("Index", spot_symbols, index=spot_symbols.index(st.session_state.index_choice))
with f2:
    expiries = get_upcoming_expiries(token_data, st.session_state.index_choice, lookahead_days=120, count=4)
    if not expiries:
        expiries = get_upcoming_expiries(token_data, st.session_state.index_choice, lookahead_days=365, count=4)
    st.session_state.expiry_choice = st.selectbox("Expiry", expiries, index=0 if expiries else 0)
with f3:
    if st.button("🔄 Refresh"):
        safe_rerun()

# Collapsible parameters with number inputs
with st.expander("⚙️ Parameters", expanded=False):
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.session_state.risk_free_rate_pct = st.number_input(
            "Risk-Free Rate (%)", min_value=0.0, max_value=12.0, value=float(st.session_state.risk_free_rate_pct), step=0.1
        )
    with c2:
        default_sc = 8 if st.session_state.index_choice == "BANKNIFTY" else int(st.session_state.strike_count)
        st.session_state.strike_count = st.number_input(
            "Strikes each side", min_value=3, max_value=15, value=int(default_sc), step=1
        )
    with c3:
        st.session_state.auto_refresh = st.checkbox("Auto refresh (15s)", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh and HAS_AUTOREFRESH:
            st_autorefresh(interval=15 * 1000, key="data_refresh")
        elif st.session_state.auto_refresh and not HAS_AUTOREFRESH:
            st.warning("Auto refresh library not installed. Run: pip install streamlit-autorefresh")
    st.session_state.use_atm_iv_for_fair = st.checkbox("Use ATM IV for Fair Value (auto)", value=st.session_state.use_atm_iv_for_fair)

# Variables
index_choice = st.session_state.index_choice
expiry_choice = (st.session_state.expiry_choice or "").strip().upper()
risk_free_rate = float(st.session_state.risk_free_rate_pct) / 100.0
strike_count = int(st.session_state.strike_count)
opt_exchange = get_exchange_for_index(index_choice)

# Days to expiry (inclusive)
try:
    exp_d = datetime.strptime(expiry_choice, "%d%b%Y").date()
    today_d = date.today()
    days_to_expiry = max((exp_d - today_d).days, 0)  # example: 25-16 = 9
except Exception:
    days_to_expiry = 0
T = days_to_expiry / 365.0

# Spot resolution
step = get_step_for_index(index_choice)
spot = get_spot_price(headers, index_choice)
if (spot is None or spot == 0) and token_data:
    inferred_spot = None
    # Try ATM parity only for indices with NFO/BFO options
    K_atm = nearest_atm_strike(spot if spot else 0, step)
    if K_atm and expiry_choice:
        ce_token = find_option_token(token_data, index_choice, K_atm, "CE", expiry_choice)
        pe_token = find_option_token(token_data, index_choice, K_atm, "PE", expiry_choice)
        ce_live, _ = get_option_fields(headers, ce_token, exchange=opt_exchange)
        pe_live, _ = get_option_fields(headers, pe_token, exchange=opt_exchange)
        if ce_live is not None and pe_live is not None:
            inferred_spot = K_atm + (ce_live - pe_live)
    if inferred_spot:
        spot = inferred_spot
        st.warning(f"{index_choice} spot unavailable; using inferred spot ≈ ₹{spot:.2f} from ATM parity.")
    else:
        st.info("Spot unavailable (market closed or rate limit). Using anchor strikes to display table.")

st.markdown(f"**Selected index:** {index_choice}   |   **Spot:** {'N/A' if spot is None else f'₹{spot:.2f}'}   |   **Expiry:** {expiry_choice or '(none)'}   |   **Days left:** {days_to_expiry}")

# Build strikes (fallback to anchors if spot missing)
if spot and spot > 0:
    ce_strikes = build_call_strikes_above_spot(spot, step, strike_count)
    pe_strikes = build_put_strikes_below_spot(spot, step, strike_count)
else:
    anchors = {"NIFTY": 17000, "BANKNIFTY": 46000, "FINNIFTY": 27500, "SENSEX": 77000}
    centre = round_up_to_step(anchors.get(index_choice, 17000), step)
    half = max(3, strike_count // 2)
    strikes = [centre + (i - half) * step for i in range(2 * half + 1)]
    strikes_sorted = sorted(set([s for s in strikes if s > 0]))
    ce_strikes = [s for s in strikes_sorted if s >= centre][:strike_count]
    pe_strikes = [s for s in list(reversed(strikes_sorted)) if s <= centre][:strike_count]

# ATM IV auto-estimation (optional) for Fair Value sigma
sigma_model = 0.18
if st.session_state.use_atm_iv_for_fair and spot and T > 0:
    K_atm = nearest_atm_strike(spot, step)
    if K_atm:
        ce_token = find_option_token(token_data, index_choice, K_atm, "CE", expiry_choice)
        pe_token = find_option_token(token_data, index_choice, K_atm, "PE", expiry_choice)
        ce_live, _ = get_option_fields(headers, ce_token, exchange=opt_exchange)
        pe_live, _ = get_option_fields(headers, pe_token, exchange=opt_exchange)
        ivs = []
        if ce_live:
            iv_ce = implied_vol(spot, K_atm, T, risk_free_rate, ce_live, "CE")
            if iv_ce: ivs.append(iv_ce)
        if pe_live:
            iv_pe = implied_vol(spot, K_atm, T, risk_free_rate, pe_live, "PE")
            if iv_pe: ivs.append(iv_pe)
        if ivs:
            sigma_model = max(0.05, min(1.0, sum(ivs) / len(ivs)))  # clamp 5%–100% for stability

alerts = []

# Calls table
call_rows = []
for K in ce_strikes:
    token = find_option_token(token_data, index_choice, K, "CE", expiry_choice) if token_data else None
    live, oi = get_option_fields(headers, token, exchange=opt_exchange)
    fair = black_scholes(spot if spot else K, K, T, risk_free_rate, sigma_model, "CE")
    if fair == 0.0:
        alerts.append(f"Fair value is 0 at CE {K}. Check inputs (spot/T).")
    if live is not None and fair is not None and fair > live:
        alerts.append(f"CE {K}: Fair ({fair:.2f}) > Live ({live:.2f})")
    dlt = delta_only(spot if spot else K, K, T, risk_free_rate, sigma_model, "CE")
    iv = None
    if live is not None and T > 0 and (spot if spot else K) > 0:
        iv = implied_vol(spot if spot else K, K, T, risk_free_rate, live, "CE")
    call_rows.append({
        "Strike": K,
        "Live": round(live, 2) if live is not None else None,
        "Fair": round(fair, 2) if fair is not None else None,
        "Delta": round(dlt, 4),
        "IV (%)": round(iv * 100, 2) if iv is not None else None,
        "OI": oi,
    })
calls_df = pd.DataFrame(call_rows)

# Puts table
put_rows = []
for K in pe_strikes:
    token = find_option_token(token_data, index_choice, K, "PE", expiry_choice) if token_data else None
    live, oi = get_option_fields(headers, token, exchange=opt_exchange)
    fair = black_scholes(spot if spot else K, K, T, risk_free_rate, sigma_model, "PE")
    if fair == 0.0:
        alerts.append(f"Fair value is 0 at PE {K}. Check inputs (spot/T).")
    if live is not None and fair is not None and fair > live:
        alerts.append(f"PE {K}: Fair ({fair:.2f}) > Live ({live:.2f})")
    dlt = delta_only(spot if spot else K, K, T, risk_free_rate, sigma_model, "PE")
    iv = None
    if live is not None and T > 0 and (spot if spot else K) > 0:
        iv = implied_vol(spot if spot else K, K, T, risk_free_rate, live, "PE")
    put_rows.append({
        "Strike": K,
        "Live": round(live, 2) if live is not None else None,
        "Fair": round(fair, 2) if fair is not None else None,
        "Delta": round(dlt, 4),
        "IV (%)": round(iv * 100, 2) if iv is not None else None,
        "OI": oi,
    })
puts_df = pd.DataFrame(put_rows)

# Alerts
if alerts:
    st.markdown("### Alerts")
    for msg in alerts[:20]:
        st.warning(msg)

# Layout: side-by-side tables
t1, t2 = st.columns(2)
with t1:
    st.markdown("### Calls (CE)")
    st.dataframe(calls_df, height=420)
    st.download_button("Download CE CSV", calls_df.to_csv(index=False), "calls.csv", "text/csv")
with t2:
    st.markdown("### Puts (PE)")
    st.dataframe(puts_df, height=420)
    st.download_button("Download PE CSV", puts_df.to_csv(index=False), "puts.csv", "text/csv")

# Footer metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Index", index_choice)
m2.metric("Spot", "N/A" if spot is None else f"₹{spot:.2f}")
m3.metric("Expiry", expiry_choice or "(none)")
m4.metric("Days left", days_to_expiry)

# Logout
bcols = st.columns([1, 1, 1, 1])
with bcols[-1]:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.jwt_token = None
        st.session_state.headers = None
        st.session_state.token_data = None
        safe_rerun()
