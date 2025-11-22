import streamlit as st
import requests
import pyotp
import math
import pandas as pd
from datetime import datetime, date, time, timezone, timedelta

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

IST = timezone(timedelta(hours=5, minutes=30))
st.set_page_config(page_title="Index Options Analyzer", layout="wide")

# ---------- Session defaults ----------
for key, default in {
    "logged_in": False,
    "login_date": None,
    "jwt_token": None,
    "headers": None,
    "token_data": None,
    "index_choice": "NIFTY",
    "expiry_choice": "",
    "risk_free_rate_pct": 7.0,
    "dividend_yield_pct": 0.0,      # carry/dividend
    "user_sigma_pct": 0.0,          # manual sigma override (%)
    "iv_source": "User sigma",      # sigma source
    "daycount": "Actual/365 (hours)",
    "strike_count": 7,
    "auto_refresh": False,
    "alerts": [],
    "active_tab": "Home",
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
@st.cache
def load_token_master():
    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_upcoming_expiries(token_data, symbol="NIFTY", lookahead_days=180, count=10):
    if not token_data:
        return []
    expiries = []
    for inst in token_data:
        try:
            inst_exch = (inst.get("exch_seg") or "").upper()
            inst_type = (inst.get("instrumenttype") or "").upper()
            inst_name = (inst.get("name") or "").upper()
            inst_symbol = (inst.get("symbol") or "").upper()
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
    today = datetime.now(IST).date()
    filtered = []
    for exp in expiries:
        try:
            exp_date = datetime.strptime(exp, "%d%b%Y").date()
            days = (exp_date - today).days
            if -2 <= days <= lookahead_days:
                filtered.append(exp)
        except Exception:
            continue
    return filtered[:count] if filtered else expiries[:count]

def parse_exp(s):
    try:
        return datetime.strptime(s, "%d%b%Y").date()
    except Exception:
        return None

def pick_active_expiry(expiries):
    if not expiries:
        return ""
    today_d = datetime.now(IST).date()
    now_t = datetime.now(IST).time()
    exps = [e for e in expiries if parse_exp(e)]
    exps_sorted = sorted(exps, key=lambda x: parse_exp(x))
    for e in exps_sorted:
        if parse_exp(e) == today_d:
            return e if now_t <= time(15, 30) else (exps_sorted[exps_sorted.index(e) + 1] if exps_sorted.index(e) + 1 < len(exps_sorted) else e)
    for e in exps_sorted:
        if parse_exp(e) and parse_exp(e) >= today_d:
            return e
    return exps_sorted[0]

def find_option_token(token_data, symbol, strike, option_type, expiry):
    if not token_data or strike is None:
        return None
    try:
        strike_master_int = int(round(float(strike) * 100))  # master stores strike*100
    except Exception:
        return None
    symbol_u = (symbol or "").upper()
    expiry_s = (expiry or "").strip()
    for inst in token_data:
        try:
            inst_type = (inst.get("instrumenttype") or "").upper()
            inst_exch = (inst.get("exch_seg") or "").upper()
            if inst_type != "OPTIDX" or inst_exch not in ("NFO", "BFO"):
                continue
            inst_name = (inst.get("name") or "").upper()
            inst_sym = (inst.get("symbol") or "").upper()
            if not (inst_name == symbol_u or symbol_u in inst_sym):
                continue
            if (inst.get("expiry") or "").strip() != expiry_s:
                continue
            try:
                inst_strike_int = int(round(float(inst.get("strike", "0"))))
            except Exception:
                continue
            if inst_strike_int != strike_master_int:
                continue
            if option_type not in inst_sym:
                continue
            return inst.get("token")
        except Exception:
            continue
    return None

# ---------- Market data ----------
INDEX_TOKENS = {
    "NIFTY": ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"),
    "SENSEX": ("BSE", "99919000"),
}

def quote(headers, exchange, tokens):
    url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote/"
    payload = {"mode": "LTP", "exchangeTokens": {exchange: tokens}}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    return r.json()

# ---------- Unit normalization ----------
def normalize_index_spot(val, index):
    if val is None:
        return None
    v = float(val)
    # NIFTY and FINNIFTY should be ~15k–25k
    if index in ("NIFTY", "FINNIFTY"):
        if v > 100000:
            v = v / 10.0
        elif v < 1000:
            v = v * 10.0
    # BANKNIFTY should be ~40k–50k
    elif index == "BANKNIFTY":
        if v > 200000:
            v = v / 10.0
        elif v < 1000:
            v = v * 10.0
    # SENSEX should be ~70k–90k
    elif index == "SENSEX":
        if v > 200000:
            v = v / 10.0
        elif v < 1000:
            v = v * 10.0
    return round(v, 2)


def normalize_option_ltp(val, K):
    # Options typically < K; if LTP > K*5, likely scaled by 100.
    if val is None:
        return None
    v = float(val)
    if K and v > (5.0 * K) and v % 1 == 0:
        v = v / 100.0
    return v

def get_spot_price(headers, symbol="NIFTY"):
    exch, token = INDEX_TOKENS.get(symbol, (None, None))
    if not headers or not exch or not token:
        return None
    try:
        data = quote(headers, exch, [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            ltp = fetched[0].get("ltp")
            return normalize_index_spot(float(ltp) if ltp is not None else None, symbol)
    except Exception:
        pass
    return None

def get_option_fields(headers, token, exchange="NFO", K=None):
    if not token or not headers:
        return None, None
    try:
        data = quote(headers, exchange, [token])
        fetched = data.get("data", {}).get("fetched", [])
        if data.get("status") and fetched:
            row = fetched[0]
            ltp = row.get("ltp")
            oi = row.get("oi") or row.get("openinterest")
            ltp_n = normalize_option_ltp(float(ltp) if ltp is not None else None, K)
            return (ltp_n if ltp_n is not None else None), (int(oi) if oi is not None else None)
    except Exception:
        pass
    return None, None

# ---------- Black–Scholes + Greeks ----------
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1_d2(S, K, T, r, sigma, q=0.0):
    # q is dividend yield / carry
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes_price(S, K, T, r, sigma, option_type="CE", q=0.0):
    if T <= 0:
        return max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if d1 is None:
        return 0.0
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if option_type == "CE":
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)
    else:
        return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1)

def delta_only(S, K, T, r, sigma, option_type="CE", q=0.0):
    if T <= 0:
        if option_type == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = d1_d2(S, K, T, r, max(sigma, 1e-6), q)
    if d1 is None:
        return 0.0
    return (math.exp(-q * T) * norm_cdf(d1)) if option_type == "CE" else (math.exp(-q * T) * (norm_cdf(d1) - 1.0))

def implied_vol(S, K, T, r, price, option_type="CE", q=0.0, max_iter=30, tol=1e-6):
    if S is None or S <= 0 or K <= 0 or T <= 0 or price is None or price <= 0:
        return None
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if price < intrinsic:
        return None
    sigma = 0.2
    lower, upper = 0.01, 3.0
    for _ in range(max_iter):
        bs = black_scholes_price(S, K, T, r, sigma, option_type, q)
        d1, _ = d1_d2(S, K, T, r, sigma, q)
        if d1 is None:
            break
        vega = math.exp(-q * T) * S * norm_pdf(d1) * math.sqrt(T)
        diff = bs - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break
        sigma = sigma - diff / vega
        sigma = max(lower, min(upper, sigma))
    # bisection fallback
    lo, hi = lower, upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        bs_mid = black_scholes_price(S, K, T, r, mid, option_type, q)
        if abs(bs_mid - price) < tol:
            return mid
        if bs_mid > price:
            hi = mid
        else:
            lo = mid
    return None

# ---------- Strikes ----------
def get_step_for_index(index_choice):
    if index_choice in ("BANKNIFTY", "SENSEX"):
        return 100
    elif index_choice in ("NIFTY", "FINNIFTY"):
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

def nearest_atm_strike(spot, step):
    if spot is None or spot <= 0:
        return None
    up = round_up_to_step(spot, step)
    down = round_down_to_step(spot, step)
    return up if abs(up - spot) <= abs(spot - down) else down

def get_exchange_for_index(index_choice):
    return "BFO" if index_choice == "SENSEX" else "NFO"

# ---------- Precise T and sigma selection ----------
def compute_T_from_expiry(expiry_str, now_dt=None):
    if now_dt is None:
        now_dt = datetime.now(IST)
    try:
        exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
        expiry_dt = datetime.combine(exp_date, time(15, 30)).replace(tzinfo=IST)
        delta = expiry_dt - now_dt
        seconds = max(delta.total_seconds(), 0.0)
        dc = st.session_state.get("daycount", "Actual/365 (hours)")
        if dc == "Actual/365 (hours)":
            return seconds / (365.0 * 24.0 * 3600.0)
        elif dc == "Actual/252":
            return (seconds / (24.0 * 3600.0)) / 252.0
        else:
            days = max((expiry_dt.date() - now_dt.date()).days, 0)
            return days / 365.0
    except Exception:
        return 0.0
        if st.session_state.get("use_manual_iv_home", False) and user_sigma_pct > 0:
            return user_sigma_pct / 100.0

def choose_sigma_for_strike(index_choice, strike, expiry_choice, spot_val, option_ltp, option_type="CE"):
    user_sigma_pct = float(st.session_state.get("user_sigma_pct", 0.0) or 0.0)
    iv_source = st.session_state.get("iv_source", "User sigma")
    default_sigma = 0.18

    # 1) Manual override
    if iv_source == "User sigma" and user_sigma_pct > 0:
        return user_sigma_pct / 100.0

    T_local = compute_T_from_expiry(expiry_choice)
    r_local = float(st.session_state.risk_free_rate_pct) / 100.0
    q_local = float(st.session_state.dividend_yield_pct) / 100.0

    # 2) Per-strike LTP IV
    if iv_source == "Per-strike LTP IV" and option_ltp and option_ltp > 0 and T_local > 0 and spot_val and spot_val > 0:
        iv = implied_vol(spot_val, strike, T_local, r_local, option_ltp, option_type, q_local)
        if iv:
            return max(0.01, min(3.0, iv))

    # 3) ATM IV average
    if iv_source == "ATM IV average" and spot_val and spot_val > 0 and T_local > 0:
        step = get_step_for_index(index_choice)
        atm_k = nearest_atm_strike(spot_val, step)
        if atm_k:
            ex = get_exchange_for_index(index_choice)
            ce_token = find_option_token(token_data, index_choice, atm_k, "CE", expiry_choice)
            pe_token = find_option_token(token_data, index_choice, atm_k, "PE", expiry_choice)
            ce_live, _ = get_option_fields(headers, ce_token, exchange=ex, K=atm_k)
            pe_live, _ = get_option_fields(headers, pe_token, exchange=ex, K=atm_k)
            ivs = []
            if ce_live:
                iv_ce = implied_vol(spot_val, atm_k, T_local, r_local, ce_live, "CE", q_local)
                if iv_ce: ivs.append(iv_ce)
            if pe_live:
                iv_pe = implied_vol(spot_val, atm_k, T_local, r_local, pe_live, "PE", q_local)
                if iv_pe: ivs.append(iv_pe)
            if ivs:
                return max(0.01, min(3.0, sum(ivs) / len(ivs)))

    # 4) Fallback
    if user_sigma_pct and user_sigma_pct > 0:
        return user_sigma_pct / 100.0
    return default_sigma

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown("### 📱 Navigation")
    st.session_state.active_tab = st.radio(
        "Go to",
        ["Home", "Calculator", "Parameters", "Alerts", "Logout"],
        index=["Home", "Calculator", "Parameters", "Alerts", "Logout"].index(st.session_state.active_tab),
        key="nav_radio"
    )

# ---------- Login ----------
st.markdown("<h3 style='margin-bottom:6px;'>📈 Index Options Analyzer</h3>", unsafe_allow_html=True)

if not st.session_state.get("logged_in", False):
    with st.form("login_form"):
        api_key = st.text_input("API Key", type="password", key="api_key")
        client_code = st.text_input("Client Code", key="client_code")
        pin = st.text_input("Trading PIN", type="password", key="trading_pin")
        totp_secret = st.text_input("TOTP Secret", type="password", key="totp_secret")
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

if not st.session_state.get("logged_in", False):
    st.info("Please log in to load the option chain and market data.")
    st.stop()

headers = st.session_state.get("headers", None)
token_data = st.session_state.get("token_data", None)
if not headers:
    st.warning("Session headers missing. Please log in again or refresh.")
    st.stop()

# ---------- Spot strip ----------
spot_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
spots = {sym: get_spot_price(headers, sym) for sym in spot_symbols}
strip_cols = st.columns(len(spot_symbols))
for i, sym in enumerate(spot_symbols):
    val = spots.get(sym, None)
    strip_cols[i].metric(sym, "N/A" if val is None else f"₹{val:.2f}")

def fmt2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return None

# ---------- HOME TAB ----------
if st.session_state.active_tab == "Home":
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        st.session_state.index_choice = st.selectbox(
            "Index",
            spot_symbols,
            index=spot_symbols.index(st.session_state.index_choice),
            key="home_index"
        )
    with f2:
        expiries = get_upcoming_expiries(token_data, st.session_state.index_choice, lookahead_days=180, count=10)
        if not expiries:
            expiries = get_upcoming_expiries(token_data, st.session_state.index_choice, lookahead_days=365, count=10)
        active_exp = pick_active_expiry(expiries)
        st.session_state.expiry_choice = st.selectbox(
            "Expiry",
            expiries,
            index=(expiries.index(active_exp) if active_exp in expiries else 0),
            key="home_expiry"
        )
    with f3:
        if st.button("🔄 Refresh", key="home_refresh"):
            safe_rerun()

    index_choice = st.session_state.index_choice
    expiry_choice = (st.session_state.expiry_choice or "").strip().upper()
    r_rate = float(st.session_state.risk_free_rate_pct) / 100.0
    q_rate = float(st.session_state.dividend_yield_pct) / 100.0
    strike_count = int(st.session_state.strike_count)
    opt_exchange = get_exchange_for_index(index_choice)

    T = compute_T_from_expiry(expiry_choice)

    step = get_step_for_index(index_choice)
    spot = get_spot_price(headers, index_choice)
    if (spot is None or spot == 0) and expiry_choice:
        K_atm = nearest_atm_strike(spot if spot else round_up_to_step(17000, step), step)
        if K_atm:
            ce_token = find_option_token(token_data, index_choice, K_atm, "CE", expiry_choice)
            pe_token = find_option_token(token_data, index_choice, K_atm, "PE", expiry_choice)
            ce_live, _ = get_option_fields(headers, ce_token, exchange=opt_exchange, K=K_atm)
            pe_live, _ = get_option_fields(headers, pe_token, exchange=opt_exchange, K=K_atm)
            if ce_live is not None and pe_live is not None:
                inferred_spot = K_atm + (ce_live - pe_live)
                spot = inferred_spot
                st.warning(f"{index_choice} spot unavailable; using inferred spot ≈ ₹{spot:.2f} from ATM parity.")

    days_to_expiry = 0
    try:
        exp_d = datetime.strptime(expiry_choice, "%d%b%Y").date()
        days_to_expiry = max((exp_d - datetime.now(IST).date()).days, 0)
    except Exception:
        pass

    st.markdown(f"**Selected index:** {index_choice}   |   **Spot:** {'N/A' if spot is None else f'₹{spot:.2f}'}   |   **Expiry:** {expiry_choice or '(none)'}   |   **Days left:** {days_to_expiry}   |   **T (yrs):** {T:.6f}")

    # Build strikes
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

    alerts = st.session_state.alerts

    # Calls
    call_rows = []
    for K in ce_strikes:
        token = find_option_token(token_data, index_choice, K, "CE", expiry_choice) if token_data else None
        live, oi = get_option_fields(headers, token, exchange=opt_exchange, K=K)
        sigma_use = choose_sigma_for_strike(index_choice, K, expiry_choice, spot if spot else K, live, "CE")
        fair = black_scholes_price(spot if spot else K, K, T, r_rate, sigma_use, "CE", q_rate)
        dlt = delta_only(spot if spot else K, K, T, r_rate, sigma_use, "CE", q_rate)
        call_rows.append({
            "Strike": K,
            "Live": fmt2(live) if live is not None else None,
            "Fair": fmt2(fair) if fair is not None else None,
            "Delta": round(dlt, 4),
            "IV (%)": fmt2(sigma_use * 100) if sigma_use is not None else None,
            "OI": oi,
        })
    calls_df = pd.DataFrame(call_rows)

    # Puts
    put_rows = []
    for K in pe_strikes:
        token = find_option_token(token_data, index_choice, K, "PE", expiry_choice) if token_data else None
        live, oi = get_option_fields(headers, token, exchange=opt_exchange, K=K)
        sigma_use = choose_sigma_for_strike(index_choice, K, expiry_choice, spot if spot else K, live, "PE")
        fair = black_scholes_price(spot if spot else K, K, T, r_rate, sigma_use, "PE", q_rate)
        dlt = delta_only(spot if spot else K, K, T, r_rate, sigma_use, "PE", q_rate)
        put_rows.append({
            "Strike": K,
            "Live": fmt2(live) if live is not None else None,
            "Fair": fmt2(fair) if fair is not None else None,
            "Delta": round(dlt, 4),
            "IV (%)": fmt2(sigma_use * 100) if sigma_use is not None else None,
            "OI": oi,
        })
    puts_df = pd.DataFrame(put_rows)

    # Dismissable alerts
    if alerts:
        st.markdown("### Alerts")
        new_alerts = []
        for i, msg in enumerate(alerts[:50]):
            cols = st.columns([10, 1])
            cols[0].warning(msg)
            if cols[1].button("✖", key=f"dismiss_{i}"):
                continue
            new_alerts.append(msg)
        st.session_state.alerts = new_alerts

    # Tables
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Calls (CE)")
        st.dataframe(calls_df, height=420)
        st.download_button("Download CE CSV", calls_df.to_csv(index=False), "calls.csv", "text/csv")
    with t2:
        st.markdown("### Puts (PE)")
        st.dataframe(puts_df, height=420)
        st.download_button("Download PE CSV", puts_df.to_csv(index=False), "puts.csv", "text/csv")

    # Debug inspector
    with st.expander("🔎 Debug inspector (verify BS inputs)", expanded=False):
        dbg_col1, dbg_col2, dbg_col3 = st.columns(3)
        dbg_spot = dbg_col1.number_input("Inspect Spot", value=float(spot if spot else 0.0), step=1.0, key="dbg_spot")
        dbg_strike = dbg_col2.number_input("Inspect Strike", value=int(ce_strikes[0] if ce_strikes else 0), step=get_step_for_index(index_choice), key="dbg_strike")
        dbg_type = dbg_col3.selectbox("Inspect Type", ["CE", "PE"], index=0, key="dbg_type")
        dbg_T = T
        dbg_r = r_rate
        dbg_q = q_rate
        dbg_sigma = choose_sigma_for_strike(index_choice, dbg_strike, expiry_choice, dbg_spot, None, dbg_type)
        st.write({
            "Spot": dbg_spot,
            "Strike": dbg_strike,
            "Expiry": expiry_choice,
            "T (yrs)": round(dbg_T, 8),
            "r": round(dbg_r, 6),
            "q": round(dbg_q, 6),
            "sigma (dec)": None if dbg_sigma is None else round(dbg_sigma, 6),
            "sigma (%)": None if dbg_sigma is None else round(dbg_sigma*100, 4),
        })

# ---------- CALCULATOR TAB ----------
elif st.session_state.active_tab == "Calculator":
    st.markdown("### 🧮 Option price calculator (target spot)")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        calc_index = st.selectbox(
            "Index",
            ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"],
            index=["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"].index(st.session_state.index_choice),
            key="calc_index"
        )
    with c2:
        calc_expiries = get_upcoming_expiries(token_data, calc_index, lookahead_days=365, count=10)
        calc_active = pick_active_expiry(calc_expiries)
        calc_exp = st.selectbox("Expiry", calc_expiries, index=(calc_expiries.index(calc_active) if calc_active in calc_expiries else 0), key="calc_expiry")
    with c3:
        calc_type = st.selectbox("Type", ["CE", "PE"], index=0, key="calc_type")
    with c4:
        step_calc = get_step_for_index(calc_index)
        live_spot_calc = get_spot_price(headers, calc_index)
        default_atm = nearest_atm_strike(live_spot_calc, step_calc) if live_spot_calc else round_up_to_step(17000, step_calc)
        calc_strike = st.number_input("Strike", min_value=0, value=int(default_atm), step=step_calc, key="calc_strike")
    with c5:
        target_spot = st.number_input("Target spot", min_value=0.0, value=float(live_spot_calc or 0.0), step=max(1.0, step_calc/10), key="calc_target_spot")

    # Volatility override (like Zerodha)
    vol_col1, vol_col2 = st.columns([2, 3])
    with vol_col1:
        user_vol_pct_calc = st.number_input("Volatility (%)", min_value=0.0, value=float(st.session_state.user_sigma_pct), step=0.01, help="Manual sigma in percent; leave 0 to use Parameters sigma source", key="calc_vol_pct")
    with vol_col2:
        st.caption("Enter volatility to override sigma for this calculation. If 0, the app uses your Parameters sigma source.")

    calc_T = compute_T_from_expiry(calc_exp)
    calc_r = float(st.session_state.risk_free_rate_pct) / 100.0
    calc_q = float(st.session_state.dividend_yield_pct) / 100.0

    # Determine sigma
    if user_vol_pct_calc and user_vol_pct_calc > 0:
        calc_sigma = user_vol_pct_calc / 100.0
    else:
        calc_sigma = choose_sigma_for_strike(calc_index, calc_strike, calc_exp, target_spot if target_spot > 0 else (live_spot_calc or calc_strike), None, calc_type)

    S_used = target_spot if target_spot > 0 else (live_spot_calc or calc_strike)
    calc_fair = black_scholes_price(S_used, calc_strike, calc_T, calc_r, calc_sigma, calc_type, calc_q)
    calc_delta = delta_only(S_used, calc_strike, calc_T, calc_r, calc_sigma, calc_type, calc_q)

    st.markdown(f"**Sigma used:** {calc_sigma:.4f} ({calc_sigma*100:.2f}%)   |   **T (yrs):** {calc_T:.6f}   |   **Fair:** ₹{fmt2(calc_fair)}   |   **Delta:** {calc_delta:.4f}")

    if live_spot_calc and target_spot:
        dS = target_spot - live_spot_calc
        approx_change = calc_delta * dS
        st.caption(f"Approximate option price change from current spot using delta: {approx_change:.2f} (ignores gamma/theta).")

# ---------- PARAMETERS TAB ----------
elif st.session_state.active_tab == "Parameters":
    st.markdown("### ⚙️ Parameters (affecting BS calculations)")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.session_state.risk_free_rate_pct = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0, max_value=50.0, value=float(st.session_state.risk_free_rate_pct),
            step=0.01, key="param_rfr"
        )
        st.session_state.dividend_yield_pct = st.number_input(
            "Dividend Yield / Carry (%)",
            min_value=0.0, max_value=50.0, value=float(st.session_state.dividend_yield_pct),
            step=0.01, key="param_div"
        )

    with p2:
        st.session_state.user_sigma_pct = st.number_input(
            "Default Volatility (%) (manual override)",
            min_value=0.0, max_value=500.0, value=float(st.session_state.user_sigma_pct),
            step=0.01, key="param_sigma"
        )
        default_sc = 8 if st.session_state.index_choice == "BANKNIFTY" else int(st.session_state.strike_count)
        st.session_state.strike_count = st.number_input(
            "Strikes each side",
            min_value=3, max_value=30, value=int(default_sc), step=1,
            key="param_strikecount"
        )

    with p3:
        st.session_state.iv_source = st.selectbox(
            "Sigma source",
            ["User sigma", "Per-strike LTP IV", "ATM IV average", "Fixed model (0.18)"],
            index=["User sigma", "Per-strike LTP IV", "ATM IV average", "Fixed model (0.18)"].index(st.session_state.iv_source),
            key="param_ivsource"
        )
        st.session_state.use_manual_iv_home = st.checkbox(
            "Use manual IV for Home tab",
            value=False,
            key="param_manual_iv_home"
        )
        st.session_state.daycount = st.selectbox(
            "Day count",
            ["Actual/365 (hours)", "Actual/365", "Actual/252"],
            index=["Actual/365 (hours)", "Actual/365", "Actual/252"].index(st.session_state.daycount),
            key="param_daycount"
        )
        st.session_state.auto_refresh = st.checkbox("Auto refresh (15s)", value=st.session_state.auto_refresh, key="param_autorefresh")
        if st.session_state.auto_refresh and HAS_AUTOREFRESH:
            st_autorefresh(interval=15 * 1000, key="data_refresh")
        elif st.session_state.auto_refresh and not HAS_AUTOREFRESH:
            st.warning("Auto refresh library not installed. Run: pip install streamlit-autorefresh")

    st.caption("Tip: To match a reference calculator (e.g., Zerodha), set Volatility, Interest, and choose 'Actual/365 (hours)' daycount.")

# ---------- ALERTS TAB ----------
elif st.session_state.active_tab == "Alerts":
    st.markdown("### 🔔 Alerts")
    alerts = st.session_state.alerts
    if not alerts:
        st.info("No alerts right now.")
    else:
        new_alerts = []
        for i, msg in enumerate(alerts[:50]):
            cols = st.columns([10, 1])
            cols[0].warning(msg)
            if cols[1].button("✖", key=f"alerts_dismiss_{i}"):
                continue
            new_alerts.append(msg)
        st.session_state.alerts = new_alerts

# ---------- LOGOUT TAB ----------
elif st.session_state.active_tab == "Logout":
    st.markdown("### 🔐 Logout")
    if st.button("Confirm logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.jwt_token = None
        st.session_state.headers = None
        st.session_state.token_data = None
        st.session_state.active_tab = "Home"
        safe_rerun()

