import streamlit as st
import numpy as np
from scipy import stats

# --- Fixed Function ---
def twosided(a, b, alt, alpha=0.05):
    # Fixed variable names: ma -> meana, sa -> stda, etc.
    meana = np.mean(a)
    meanb = np.mean(b)
    stda = np.std(a, ddof=1) # Using numpy std with Bessel's correction
    stdb = np.std(b, ddof=1)
    n1 = len(a)
    n2 = len(b)
    
    output = []
    # Standard Error calculation
    se = np.sqrt((stda**2 / n1) + (stdb**2 / n2))
    tcal = (meana - meanb) / se
    df = n1 + n2 - 2
    
    if alt == 'two':
        tpos = stats.t.ppf(1 - alpha/2, df)
        tneg = stats.t.ppf(alpha/2, df)
        # P-value for two-tailed is usually 2 * (1 - cdf(|t|))
        p = 2 * (1 - stats.t.cdf(np.abs(tcal), df))
        
        output.extend([tcal, tpos, tneg, p])
    else:
        # One-tailed logic
        tcritical = stats.t.ppf(1 - alpha, df)
        p = 1 - stats.t.cdf(tcal, df)
        output.extend([tcal, tcritical, p])
        
    return output

# --- Streamlit UI ---
st.title("📊 T-Test Calculator")

st.sidebar.header("Settings")
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05)
test_type = st.sidebar.selectbox("Test Type", ["two", "one"])

col1, col2 = st.columns(2)

with col1:
    data1_input = st.text_area("Data Set A (comma separated)", "58, 128.6, 12, 123.8, 64.34, 78, 763.3")
    
with col2:
    data2_input = st.text_area("Data Set B (comma separated)", "1.1, 2.9, 4.2")

if st.button("Run Hypothesis Test"):
    try:
        # Convert strings to numpy arrays
        a = np.array([float(x.strip()) for x in data1_input.split(",")])
        b = np.array([float(x.strip()) for x in data2_input.split(",")])
        
        results = twosided(a, b, test_type, alpha)
        
        # Display Results
        st.subheader("Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        res_col1.metric("T-Statistic", f"{results[0]:.4f}")
        res_col2.metric("P-Value", f"{results[-1]:.4f}")
        res_col3.metric("Critical Value (T-Pos)", f"{results[1]:.4f}")

        if results[-1] < alpha:
            st.success("Result: Reject the Null Hypothesis (Statistically Significant)")
        else:
            st.warning("Result: Fail to Reject the Null Hypothesis")
            
    except Exception as e:
        st.error(f"Error: Please check your input format. {e}")