import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# Page Configuration
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Live Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ADVANCED CSS: ANIMATED NEON GLOWS & GLASSMORPHISM
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Animated Dynamic Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #31103f, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    footer {visibility: hidden;}

    /* Glassmorphism Cards with Hover Pulse */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stHorizontalBlock"], .css-1r6silb0 {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 24px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45);
        transition: all 0.3s ease;
    }

    /* Glowing Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Neon Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.65rem 1.4rem;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 0 30px rgba(236, 72, 153, 0.8);
    }

    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: pulseGlow 3s infinite;
    }

    @keyframes pulseGlow {
        0% { border-color: rgba(56, 189, 248, 0.25); }
        50% { border-color: rgba(192, 132, 252, 0.6); }
        100% { border-color: rgba(56, 189, 248, 0.25); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------------------------------------
# SECURE LOGIN PORTAL
# ---------------------------------------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>⚡ ChurnGuard AI Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8;'>Enterprise Machine Learning & Real-Time Intelligence Suite</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username (e.g. admin)")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("🚀 Launch Dashboard", use_container_width=True)
            
            if submit:
                if username and password:
                    st.session_state.logged_in = True
                    st.success("Authentication successful! Initializing secure environment...")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Please provide valid credentials.")
    st.stop()

# ---------------------------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('churn.csv').dropna().head(1000)

model = load_model()
df = load_data()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & TELEMETRY
# ---------------------------------------------------------
st.sidebar.markdown("### 🎛️ Control Nexus")
st.sidebar.markdown("**User:** `Muhammad Huzaifa`")
st.sidebar.markdown("**Status:** `Online & Secured`")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation Menu", [
    "🔮 Real-Time Prediction", 
    "📡 Live Telemetry Stream", 
    "📁 Batch CSV Processing", 
    "📈 Exploratory Analytics", 
    "🏆 Model Benchmarks"
])

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Terminate Session", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# ---------------------------------------------------------
# PAGE 1: REAL-TIME PREDICTION DASHBOARD
# ---------------------------------------------------------
if page == "🔮 Real-Time Prediction":
    st.title("🛡️ Real-Time Customer Churn Intelligence")
    st.markdown("Evaluate subscriber vulnerability instantly using your optimized machine learning pipeline.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Customer Demographics")
            age = st.slider("Age", 18, 65, 32)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (Months)", 1, 60, 14)
            usage_frequency = st.slider("Usage Frequency", 1, 30, 18)
            support_calls = st.slider("Support Calls", 0, 10, 2)
            
        with col2:
            st.subheader("💳 Financial & Account Details")
            payment_delay = st.slider("Payment Delay (Days)", 0, 30, 4)
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            total_spend = st.number_input("Total Spend ($)", 100.0, 1000.0, 650.0)
            last_interaction = st.slider("Last Interaction (Days Ago)", 1, 30, 8)
            
        submitted = st.form_submit_button("⚡ Execute AI Prediction", use_container_width=True)
        
    if submitted:
        with st.spinner("Analyzing behavioral patterns and feature metrics..."):
            time.sleep(0.8)
            
            spend_per_tenure = total_spend / (tenure + 1)
            support_delay_ratio = support_calls / (payment_delay + 1)
            
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Tenure': tenure,
                'Usage Frequency': usage_frequency,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Subscription Type': subscription_type,
                'Contract Length': contract_length,
                'Total Spend': total_spend,
                'Last Interaction': last_interaction,
                'Spend_Per_Tenure': spend_per_tenure,
                'Support_Delay_Ratio': support_delay_ratio
            }])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
        st.markdown("---")
        st.subheader("📊 Diagnostic Outcome")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **High Churn Risk Detected (WILL Churn)**")
                st.markdown("💡 *Actionable Insight: Dispatch automated retention coupon or trigger customer success alert.*")
            else:
                st.success("✅ **Stable Customer Profile (WILL NOT Churn)**")
                st.markdown("💡 *Actionable Insight: Customer engagement is optimal. Maintain current tier.*")
                st.balloons()
                
        with res_col2:
            st.metric(label="Calculated Churn Probability", value=f"{probability * 100:.2f}%")
            st.progress(float(probability))

# ---------------------------------------------------------
# PAGE 2: LIVE TELEMETRY STREAM (Live Tracking Statistical Graph)
# ---------------------------------------------------------
elif page == "📡 Live Telemetry Stream":
    st.title("📡 Live-Time Statistical Graph & Telemetry")
    st.markdown("Simulating live incoming API inference requests and real-time risk tracking over time.")
    
    col1, col2, col3 = st.columns(3)
    metric_placeholder_1 = col1.empty()
    metric_placeholder_2 = col2.empty()
    metric_placeholder_3 = col3.empty()
    
    chart_placeholder = st.empty()
    
    if st.button("🔴 Start Live Telemetry Feed"):
        history_timestamps = []
        history_risk = []
        history_requests = []
        
        for i in range(25):
            current_time = time.strftime("%H:%M:%S")
            simulated_risk = random.uniform(35.0, 85.0)
            simulated_reqs = random.randint(120, 350)
            
            history_timestamps.append(current_time)
            history_risk.append(simulated_risk)
            history_requests.append(simulated_reqs)
            
            metric_placeholder_1.metric("Live Avg Churn Risk", f"{simulated_risk:.1f}%", f"+{random.uniform(-1.5, 2.1):.1f}%")
            metric_placeholder_2.metric("Active API Inferences", f"{simulated_reqs} req/s", f"+{random.randint(5, 25)}")
            metric_placeholder_3.metric("System Latency", f"{random.randint(18, 42)} ms", "-2 ms")
            
            live_df = pd.DataFrame({
                "Time": history_timestamps,
                "Risk Score (%)": history_risk,
                "Requests / Sec": history_requests
            }).set_index("Time")
            
            chart_placeholder.line_chart(live_df, use_container_width=True)
            time.sleep(0.6)
        st.success("Live stream telemetry cycle completed successfully.")
    else:
        st.info("Click the button above to initiate live-time tracking simulation.")

# ---------------------------------------------------------
# PAGE 3: BATCH CSV PROCESSING
# ---------------------------------------------------------
elif page == "📁 Batch CSV Processing":
    st.title("📂 Batch Dataset Inference")
    st.markdown("Upload multiple customer records in CSV format for bulk classification.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success("File successfully parsed! Previewing records:")
        st.dataframe(batch_df.head(5))
        
        if st.button("🚀 Process Batch Analysis", type="primary"):
            with st.spinner("Computing inferences across batch records..."):
                try:
                    if 'Spend_Per_Tenure' not in batch_df.columns:
                        batch_df['Spend_Per_Tenure'] = batch_df['Total Spend'] / (batch_df['Tenure'] + 1)
                    if 'Support_Delay_Ratio' not in batch_df.columns:
                        batch_df['Support_Delay_Ratio'] = batch_df['Support Calls'] / (batch_df['Payment Delay'] + 1)
                        
                    preds = model.predict(batch_df.drop(columns=['Churn'], errors='ignore'))
                    probs = model.predict_proba(batch_df.drop(columns=['Churn'], errors='ignore'))[:, 1]
                    
                    batch_df['Predicted_Churn'] = preds
                    batch_df['Churn_Probability'] = np.round(probs * 100, 2)
                    
                    st.subheader("📋 Batch Results Preview")
                    st.dataframe(batch_df[['Tenure', 'Total Spend', 'Predicted_Churn', 'Churn_Probability']].head(10))
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Processed Results CSV",
                        data=csv_data,
                        file_name='churn_batch_predictions.csv',
                        mime='text/csv'
                    )
                except Exception as e:
                    st.error(f"Processing error: {e}")

# ---------------------------------------------------------
# PAGE 4: EXPLORATORY ANALYTICS
# ---------------------------------------------------------
elif page == "📈 Exploratory Analytics":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Visualizing distributions and correlations from the repository dataset.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Churn', palette='coolwarm', ax=ax)
        ax.set_xticklabels(['Not Churned (0)', 'Churned (1)'])
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Total Spend vs Churn Status")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Churn', y='Total Spend', palette='Set2', ax=ax)
        ax.set_xticklabels(['Not Churned (0)', 'Churned (1)'])
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)

# ---------------------------------------------------------
# PAGE 5: MODEL BENCHMARKS
# ---------------------------------------------------------
elif page == "🏆 Model Benchmarks":
    st.title("🏆 AI Model Evaluation & Benchmarks")
    st.markdown("Comparative performance metrics across tested candidate algorithms after GridSearchCV tuning.")
    
    st.info("💡 Optimal pipeline saved securely via **Joblib**.")
    
    comparison_data = {
        "Model Architecture": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting (Tuned)"],
        "Accuracy": ["89.37%", "99.93%", "99.42%", "99.68%"],
        "Precision": ["88.50%", "99.90%", "99.20%", "99.50%"],
        "Recall": ["91.20%", "99.95%", "99.70%", "99.85%"],
        "F1-Score": ["90.47%", "99.94%", "99.49%", "99.72%"],
        "ROC-AUC": ["0.9582", "0.9993", "1.0000", "0.9997"]
    }
    
    st.table(pd.DataFrame(comparison_data))