import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
        color: #f0ede6;
    }

    .stApp { background-color: #0f1117; color: #c8c3b8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2a2d3a;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px 24px;   /* ← change from padding: 16px to padding: 16px 24px */
    }

    [data-testid="stMetricLabel"] { color: #8a8fa8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #e8e0d0 !important; font-size: 1.6rem !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00b37e, #00d492);
        color: #0f1117;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0, 179, 126, 0.3);
    }

    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(0, 179, 126, 0.2);
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 24px 32px;
        margin: 12px 0;
        width: 100%;
        box-sizing: border-box;
    }

    .cluster-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }

    .badge-0 { background: rgba(201, 168, 76, 0.2); color: #c9a84c; border: 1px solid #c9a84c; }
    .badge-1 { background: rgba(76, 175, 130, 0.2); color: #4caf82; border: 1px solid #4caf82; }
    .badge-2 { background: rgba(207, 102, 88, 0.2); color: #cf6658; border: 1px solid #cf6658; }
    .badge-3 { background: rgba(100, 149, 237, 0.2); color: #6495ed; border: 1px solid #6495ed; }

    .divider {
        border: none;
        border-top: 1px solid #2a2d3a;
        margin: 24px 0;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d27; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8a8fa8; font-family: 'DM Sans', sans-serif; }
    .stTabs [aria-selected="true"] { background-color: #252840 !important; color: #e8e0d0 !important; border-radius: 8px; }

    /* Input fields */
    .stSelectbox > div, .stSlider > div { color: #c8c3b8; }

    /* Success/warning/info */
    .stSuccess { background-color: rgba(76, 175, 130, 0.1) !important; border-color: #4caf82 !important; }
    .stWarning { background-color: rgba(201, 168, 76, 0.1) !important; border-color: #c9a84c !important; }

    .header-accent {
        color: #00b37e;
        font-family: 'DM Serif Display', serif;
    }

    .section-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8a8fa8;
        margin-bottom: 8px;
    }

    /* Input focus color */
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #00b37e !important;
        box-shadow: 0 0 0 1px #00b37e !important;
    }

    /* Selectbox focus */
    [data-baseweb="select"] *:focus-visible,
    [data-baseweb="input"] *:focus-visible {
        border-color: #00b37e !important;
        box-shadow: 0 0 0 2px rgba(0, 179, 126, 0.3) !important;
    }

    /* Active/selected state on inputs */
    [data-baseweb="input"]:focus-within {
        border-color: #00b37e !important;
    }

    [data-baseweb="select"]:focus-within > div {
        border-color: #00b37e !important;
    }

    /* Number input stepper buttons */
    [data-testid="stNumberInput"] button:focus,
    [data-testid="stNumberInput"] button:active {
        border-color: #00b37e !important;
        color: #00b37e !important;
    }

    /* Number input container focus */
    [data-testid="stNumberInput"] > div:focus-within {
        border-color: #00b37e !important;
        box-shadow: 0 0 0 1px #00b37e !important;
    }

    /* Number input inner div */
    [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border-color: #00b37e !important;
        box-shadow: 0 0 0 1px #00b37e !important;
    }

    /* Override the baseweb input border directly */
    [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        border-color: #2e3250 !important;
    }

    [data-testid="stNumberInput"] div[data-baseweb="base-input"]:focus-within {
        border-color: #00b37e !important;
        box-shadow: 0 0 0 1px #00b37e !important;
    }

    /* Selectbox dropdown option highlight */
    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: rgba(0, 179, 126, 0.2) !important;
    }

    [data-baseweb="menu"] [role="option"]:hover {
        background-color: rgba(0, 179, 126, 0.1) !important;
    }

    /* Radio button active state */
    [data-testid="stRadio"] label:has(input:checked) {
        color: #00b37e !important;
    }

    /* File uploader focus */
    [data-testid="stFileUploader"] section {
        border-color: #2e3250 !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #00b37e !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), 'models')
    return {
        'kmeans':     joblib.load(os.path.join(base, 'kmeans_model.pkl')),
        'classifier': joblib.load(os.path.join(base, 'classifier_model.pkl')),
        'regression': joblib.load(os.path.join(base, 'regression_model.pkl')),
        'scaler':     joblib.load(os.path.join(base, 'scaler.pkl')),
    }

models = load_models()

# ── Constants ─────────────────────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "High Value Loyalist",
    1: "Satisfied Passive",
    2: "Dissatisfied Inactive",
    3: "Young Casual"
}

CLUSTER_DESCRIPTIONS = {
    0: "High spending, frequent visits, high income. Your most valuable customers.",
    1: "Older demographic, infrequent but satisfied shoppers. Re-engagement targets.",
    2: "Low engagement and low satisfaction. At risk of churning.",
    3: "Young, satisfied but low spending. Long-term nurture segment."
}

CLUSTER_ACTIONS = {
    0: "🏆 Enroll in premium loyalty program. Offer exclusive early access and rewards.",
    1: "📧 Launch re-engagement campaign. A well-timed promotion could increase visit frequency.",
    2: "🔍 Investigate pain points urgently. Consider a satisfaction survey or service recovery offer.",
    3: "🌱 Nurture with entry-level promotions. Build purchase habits early for long-term value."
}

BADGE_CLASS = {0: "badge-0", 1: "badge-1", 2: "badge-2", 3: "badge-3"}

NUMERICAL_COLS = [
    'Age', 'Total_Spending', 'Visit_Frequency', 'Avg_Session_Duration',
    'Avg_Pages_Viewed', 'Avg_Delivery_Time', 'Avg_Rating',
    'Monthly_Sales', 'Annual_Income'
]

CLUSTERING_COLS = [
    'Age', 'Total_Spending', 'Visit_Frequency', 'Avg_Session_Duration',
    'Avg_Pages_Viewed', 'Avg_Delivery_Time', 'Avg_Rating',
    'Monthly_Sales', 'Annual_Income'
]

REGRESSION_COLS = [
    'Age', 'Gender', 'City', 'Visit_Frequency', 'Avg_Session_Duration',
    'Avg_Pages_Viewed', 'Avg_Delivery_Time', 'Avg_Rating',
    'Is_Returning_Customer', 'Most_Used_Payment',
    'Most_Used_Device', 'Favourite_Category'
]

CLASSIFIER_COLS = [
    'Age', 'Gender', 'City', 'Total_Spending', 'Visit_Frequency',
    'Avg_Session_Duration', 'Avg_Pages_Viewed', 'Avg_Delivery_Time',
    'Avg_Rating', 'Is_Returning_Customer', 'Most_Used_Payment',
    'Most_Used_Device', 'Favourite_Category', 'Monthly_Sales', 'Annual_Income'
]

# ── Encoding Maps ─────────────────────────────────────────────────────────────
GENDER_MAP      = {'Female': 0, 'Male': 1, 'Other': 2}
CITY_MAP        = {'Ankara': 0, 'Antalya': 1, 'Bursa': 2, 'Diyarbakir': 3,
                   'Eskisehir': 4, 'Gaziantep': 5, 'Istanbul': 6, 'Izmir': 7,
                   'Kayseri': 8, 'Konya': 9}
PAYMENT_MAP     = {'Bank Transfer': 0, 'Cash on Delivery': 1,
                   'Credit Card': 2, 'Debit Card': 3, 'Digital Wallet': 4}
DEVICE_MAP      = {'Desktop': 0, 'Mobile': 1, 'Tablet': 2}
CATEGORY_MAP    = {'Beauty': 0, 'Books': 1, 'Electronics': 2, 'Fashion': 3,
                   'Food': 4, 'Home & Garden': 5, 'Sports': 6, 'Toys': 7}

# ── Helper Functions ──────────────────────────────────────────────────────────
def preprocess_single(inputs: dict) -> pd.DataFrame:
    """Encode and scale a single customer input dict."""
    row = {
        'Age':                  inputs['age'],
        'Total_Spending':       inputs['total_spending'],
        'Visit_Frequency':      inputs['visit_frequency'],
        'Avg_Session_Duration': inputs['avg_session'],
        'Avg_Pages_Viewed':     inputs['avg_pages'],
        'Avg_Delivery_Time':    inputs['avg_delivery'],
        'Avg_Rating':           inputs['avg_rating'],
        'Monthly_Sales':        inputs['monthly_sales'],
        'Annual_Income':        inputs['annual_income'],
    }
    num_df = pd.DataFrame([row])[NUMERICAL_COLS]
    scaled = models['scaler'].transform(num_df)
    scaled_df = pd.DataFrame(scaled, columns=NUMERICAL_COLS)

    scaled_df['Gender']               = GENDER_MAP[inputs['gender']]
    scaled_df['City']                 = CITY_MAP[inputs['city']]
    scaled_df['Is_Returning_Customer']= int(inputs['is_returning'])
    scaled_df['Most_Used_Payment']    = PAYMENT_MAP[inputs['payment']]
    scaled_df['Most_Used_Device']     = DEVICE_MAP[inputs['device']]
    scaled_df['Favourite_Category']   = CATEGORY_MAP[inputs['category']]

    return scaled_df

def predict_single(scaled_df: pd.DataFrame) -> dict:
    cluster    = int(models['kmeans'].predict(scaled_df[CLUSTERING_COLS])[0])
    promo      = int(models['classifier'].predict(scaled_df[CLASSIFIER_COLS])[0])
    promo_prob = models['classifier'].predict_proba(scaled_df[CLASSIFIER_COLS])[0][1]
    spending   = float(models['regression'].predict(scaled_df[REGRESSION_COLS])[0])
    return {
        'cluster':    cluster,
        'promo':      promo,
        'promo_prob': promo_prob,
        'spending':   spending
    }

def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess an uploaded CSV for batch prediction."""
    required = ['Age', 'Gender', 'City', 'Total_Spending', 'Visit_Frequency',
                'Avg_Session_Duration', 'Avg_Pages_Viewed', 'Avg_Delivery_Time',
                'Avg_Rating', 'Is_Returning_Customer', 'Most_Used_Payment',
                'Most_Used_Device', 'Favourite_Category', 'Monthly_Sales', 'Annual_Income']

    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    processed = df.copy()
    processed['Gender']            = processed['Gender'].map(GENDER_MAP)
    processed['City']              = processed['City'].map(CITY_MAP)
    processed['Most_Used_Payment'] = processed['Most_Used_Payment'].map(PAYMENT_MAP)
    processed['Most_Used_Device']  = processed['Most_Used_Device'].map(DEVICE_MAP)
    processed['Favourite_Category']= processed['Favourite_Category'].map(CATEGORY_MAP)
    processed['Is_Returning_Customer'] = processed['Is_Returning_Customer'].astype(int)

    processed[NUMERICAL_COLS] = models['scaler'].transform(processed[NUMERICAL_COLS])
    return processed, None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Retail Intelligence")
    st.markdown("<p class='section-label'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio("", ["Single Customer", "Batch Analysis"], label_visibility="collapsed")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Cluster Reference</p>", unsafe_allow_html=True)
    for cid, name in CLUSTER_NAMES.items():
        st.markdown(
            f"<span class='cluster-badge {BADGE_CLASS[cid]}'>{name}</span>",
            unsafe_allow_html=True
        )
        st.caption(CLUSTER_DESCRIPTIONS[cid])

# ── Page: Single Customer ─────────────────────────────────────────────────────
if page == "Single Customer":
    st.markdown("<h1>Customer <span class='header-accent'>Prediction</span></h1>", unsafe_allow_html=True)
    st.markdown("Enter a customer's profile to predict their segment, promotion response, and spending.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 2.6])

    with col1:
        st.markdown("<p class='section-label'>Demographics</p>", unsafe_allow_html=True)
        age            = st.number_input("Age", min_value=18, max_value=75, value=30, step=1)
        gender         = st.selectbox("Gender", list(GENDER_MAP.keys()))
        city           = st.selectbox("City", list(CITY_MAP.keys()))
        is_returning   = st.selectbox("Returning Customer", [True, False])

    with col2:
        st.markdown("<p class='section-label'>Spending Profile</p>", unsafe_allow_html=True)
        total_spending  = st.number_input("Total Spending (TRY)", 0.0, 100000.0, 2000.0, step=100.0)
        annual_income   = st.number_input("Annual Income (TRY)", 0.0, 500000.0, 50000.0, step=1000.0)
        monthly_sales   = st.number_input("Monthly Sales (TRY)", 0.0, 50000.0, 1000.0, step=100.0)
        visit_frequency = st.number_input("Visit Frequency", min_value=1, max_value=10, value=3, step=1)

    with col3:
        st.markdown("<p class='section-label'>Behaviour & Preferences</p>", unsafe_allow_html=True)
        col3a, col3b = st.columns(2)

        with col3a:
            avg_session  = st.number_input("Avg Session Duration (min)", min_value=1, max_value=30, value=14, step=1)
            avg_pages    = st.number_input("Avg Pages Viewed", min_value=1, max_value=20, value=9, step=1)
            avg_delivery = st.number_input("Avg Delivery Time (days)", min_value=1, max_value=30, value=6, step=1)
            avg_rating   = st.number_input("Avg Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.5)

        with col3b:
            payment  = st.selectbox("Preferred Payment", list(PAYMENT_MAP.keys()))
            device   = st.selectbox("Preferred Device", list(DEVICE_MAP.keys()))
            category = st.selectbox("Favourite Category", list(CATEGORY_MAP.keys()))

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    run = st.button("Run Prediction →")

    if run:
        inputs = dict(
            age=age, gender=gender, city=city, is_returning=is_returning,
            total_spending=total_spending, annual_income=annual_income,
            monthly_sales=monthly_sales, visit_frequency=visit_frequency,
            avg_session=avg_session, avg_pages=avg_pages,
            avg_delivery=avg_delivery, avg_rating=avg_rating,
            payment=payment, device=device, category=category
        )
        scaled_df = preprocess_single(inputs)
        results   = predict_single(scaled_df)

        c_id   = results['cluster']
        c_name = CLUSTER_NAMES[c_id]

        st.markdown("### Prediction Results")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Customer Segment", c_name)
        with m2:
            promo_label = "✅ Yes" if results['promo'] == 1 else "❌ No"
            st.metric("Promotion Response", promo_label,
                      delta=f"{results['promo_prob']:.0%} confidence")
        with m3:
            st.metric("Predicted Spending Score", f"{results['spending']:.4f}",
                      help="Normalized spending score (0–1 scale)")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-card'>"
            f"<p class='section-label'>Recommended Action</p>"
            f"<span class='cluster-badge {BADGE_CLASS[c_id]}'>{c_name}</span><br><br>"
            f"<p>{CLUSTER_ACTIONS[c_id]}</p>"
            f"<p style='color:#8a8fa8; font-size:0.85rem'>{CLUSTER_DESCRIPTIONS[c_id]}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── Page: Batch Analysis ──────────────────────────────────────────────────────
else:
    st.markdown("<h1>Batch <span class='header-accent'>Analysis</span></h1>", unsafe_allow_html=True)
    st.markdown("Upload a CSV of customer profiles to run predictions across your entire dataset.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.info("Your CSV must include these columns: Age, Gender, City, Total_Spending, "
            "Visit_Frequency, Avg_Session_Duration, Avg_Pages_Viewed, Avg_Delivery_Time, "
            "Avg_Rating, Is_Returning_Customer, Most_Used_Payment, Most_Used_Device, "
            "Favourite_Category, Monthly_Sales, Annual_Income")

    uploaded = st.file_uploader("Upload Customer CSV", type=['csv'])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw_df):,} customers loaded.** Preview:")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Run Batch Predictions →"):
            with st.spinner("Processing..."):
                processed, error = preprocess_batch(raw_df)

                if error:
                    st.error(f"❌ {error}")
                else:
                    results_df = raw_df.copy()
                    results_df['Cluster_ID']   = models['kmeans'].predict(
                        processed[CLUSTERING_COLS])
                    results_df['Cluster_Name'] = results_df['Cluster_ID'].map(CLUSTER_NAMES)
                    results_df['Promo_Response'] = models['classifier'].predict(
                        processed[CLASSIFIER_COLS])
                    results_df['Promo_Response'] = results_df['Promo_Response'].map(
                        {1: 'Yes', 0: 'No'})
                    results_df['Spending_Score'] = models['regression'].predict(
                        processed[REGRESSION_COLS])

                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("### Results Summary")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Customers", f"{len(results_df):,}")
                    m2.metric("Promo Responders",
                              f"{(results_df['Promo_Response']=='Yes').sum():,}")
                    m3.metric("Largest Segment",
                              results_df['Cluster_Name'].value_counts().index[0])
                    m4.metric("Avg Spending Score",
                              f"{results_df['Spending_Score'].mean():.4f}")

                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["📊 Cluster Distribution", "📋 Full Results"])

                    with tab1:
                        cluster_counts = results_df['Cluster_Name'].value_counts()
                        st.bar_chart(cluster_counts)
                        for cid, name in CLUSTER_NAMES.items():
                            count = (results_df['Cluster_ID'] == cid).sum()
                            pct   = count / len(results_df) * 100
                            st.markdown(
                                f"<span class='cluster-badge {BADGE_CLASS[cid]}'>{name}</span> "
                                f"&nbsp; {count:,} customers ({pct:.1f}%)",
                                unsafe_allow_html=True
                            )
                            st.caption(CLUSTER_ACTIONS[cid])

                    with tab2:
                        st.dataframe(results_df, use_container_width=True)
                        csv_out = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Results CSV",
                            csv_out,
                            "customer_predictions.csv",
                            "text/csv"
                        )
