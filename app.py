import requests
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Risk Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS for SaaS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 3. Sidebar Preset Controls
st.sidebar.title("🎛 Customer Persona Presets")
st.sidebar.write("Quick-load test profiles:")

if st.sidebar.button("🔴 High Risk Customer Profile"):
    st.session_state.tenure = 2
    st.session_state.monthly = 95.0
    st.session_state.contract = "month-to-month"

if st.sidebar.button("🟢 Retained Customer Profile"):
    st.session_state.tenure = 48
    st.session_state.monthly = 25.0
    st.session_state.contract = "two-year"

# Defaults
if "tenure" not in st.session_state:
    st.session_state.tenure = 12
if "monthly" not in st.session_state:
    st.session_state.monthly = 65.0
if "contract" not in st.session_state:
    st.session_state.contract = "month-to-month"

# 4. Main Dashboard Header
st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("Powered by FastAPI Microservice & Scikit-Learn Random Forest Pipeline")
st.divider()

# 5. Input Controls Layout (Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Parameters")
    tenure = st.slider(
        "Tenure (Months as Customer)",
        min_value=1,
        max_value=72,
        value=st.session_state.tenure,
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=10.0,
        max_value=150.0,
        value=st.session_state.monthly,
    )
    contract = st.selectbox(
        "Contract Type",
        ["month-to-month", "one-year", "two-year"],
        index=["month-to-month", "one-year", "two-year"].index(
            st.session_state.contract
        ),
    )

    evaluate_btn = st.button("🚀 Run Live Inference", type="primary")

with col2:
    st.subheader("⚡ Real-Time Model Inference")

    if evaluate_btn:
        payload = {
            "tenure": int(tenure),
            "monthly_charges": float(monthly_charges),
            "contract": contract,
        }

        try:
            # Query Live Render FastAPI REST endpoint
            response = requests.post(
                "https://churn-api-siddhesh.onrender.com/predict", json=payload
            )

            if response.status_code == 200:
                data = response.json()
                prob = data["churn_probability"] * 100
                risk = data["risk_level"]

                # Metric Cards
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Churn Risk Level", risk)
                with m2:
                    st.metric("Churn Probability", f"{prob:.1f}%")

                # Visual Progress Bar
                st.write("### Risk Distribution Bar")
                st.progress(int(prob))

                if risk == "High":
                    st.error(
                        "⚠️ High Churn Risk Detected: Recommended retention action -> Offer 1-year contract discount."
                    )
                else:
                    st.success(
                        "✅ Healthy Customer Profile: Low risk of cancellation."
                    )

                # Raw API Response Inspector
                with st.expander("🔍 Inspect Raw JSON Response (FastAPI)"):
                    st.json(data)
            else:
                st.error("Error from API backend.")

        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")
    else:
        st.info("Adjust parameters on the left and click 'Run Live Inference'")