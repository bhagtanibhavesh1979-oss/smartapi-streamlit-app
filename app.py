import streamlit as st
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from smartapi import SmartConnect

# --- Black-Scholes Functions ---
def black_scholes(S, K, T, r, sigma, option_type="C"):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    if option_type == "C":
        price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1)/(S*sigma*sqrt(T))
    vega = S*norm.pdf(d1)*sqrt(T)
    theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2 if option_type=="C" else -d2)
    rho = K*T*exp(-r*T)*(norm.cdf(d2) if option_type=="C" else -norm.cdf(-d2))

    return price, delta, gamma, vega, theta, rho

# --- Streamlit UI ---
st.title("SmartAPI Option Chain with Black-Scholes")

# Login form
api_key = st.text_input("API Key", type="password")
client_id = st.text_input("Client ID")
password = st.text_input("Password", type="password")
totp = st.text_input("TOTP")

if st.button("Login"):
    try:
        obj = SmartConnect(api_key)
        data = obj.generateSession(client_id, password, totp)
        st.success("Login successful!")
        st.session_state['smartapi'] = obj
    except Exception as e:
        st.error(f"Login failed: {e}")

# After login, fetch option chain
if 'smartapi' in st.session_state:
    st.subheader("Option Chain Data")

    # Example: Fetch Nifty FUTIDX LTP
    try:
        # Replace with actual SmartAPI instrument token for Nifty Futures
        nifty_token = "26000"  # Example token, adjust to your JSON instrument master
        ltp_data = st.session_state['smartapi'].ltpData("NSE", "FUTIDX", "NIFTY", nifty_token)
        spot = float(ltp_data['data']['ltp'])
        st.write(f"Spot Index (Nifty): {spot}")

        # ATM strike
        atm_strike = round(spot/50)*50
        strikes = [atm_strike + i*50 for i in range(-5,6)]

        # Parameters for Black-Scholes
        T = 30/365  # 30 days to expiry
        r = 0.05    # risk-free rate
        sigma = 0.2 # assumed volatility

        rows = []
        for K in strikes:
            call_price, c_delta, c_gamma, c_vega, c_theta, c_rho = black_scholes(spot, K, T, r, sigma, "C")
            put_price, p_delta, p_gamma, p_vega, p_theta, p_rho = black_scholes(spot, K, T, r, sigma, "P")
            rows.append({
                "Strike": K,
                "Call Price": round(call_price,2),
                "Put Price": round(put_price,2),
                "Delta (C)": round(c_delta,2),
                "Delta (P)": round(p_delta,2),
                "Gamma": round(c_gamma,4),
                "Vega": round(c_vega,2),
                "Theta (C)": round(c_theta,2),
                "Theta (P)": round(p_theta,2),
                "Rho (C)": round(c_rho,2),
                "Rho (P)": round(p_rho,2),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error fetching option chain: {e}")

