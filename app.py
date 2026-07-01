import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_loader import generate_synthetic_credit, load_german_credit
from preprocessing import prepare_train_test_data, CreditPreprocessor
from models import train_and_evaluate_models, calculate_credit_score

# Set page config
st.set_page_config(
    page_title="Credit Scoring & Risk Assessment Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Header styling with gradient */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin: 0;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
    }
    
    .metric-card-title {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    
    /* Risk band styling classes */
    .badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-default { background-color: #e9ecef; color: #495057; }
    
    /* Tab headers styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #e9ecef;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e9f2ff;
        border-bottom: 2px solid #2a5298 !important;
        color: #1e3c72 !important;
    }
    
    /* Center align gauges */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">💳 Credit Scoring & Default Prediction</h1>
        <p class="header-subtitle">Analyze credit risk using advanced classification algorithms, engineered financial features, and explainable AI metrics.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize Session State to cache datasets and model training
if 'dataset_type' not in st.session_state:
    st.session_state['dataset_type'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'preprocessor' not in st.session_state:
    st.session_state['preprocessor'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Pipeline Configuration")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset Source",
    options=["Synthetic Credit Dataset (Recommended)", "German Credit Dataset (OpenML)"],
    index=0
)

# Convert choice to key
dataset_key = 'synthetic' if "Synthetic" in dataset_choice else 'german_credit'

# Check if dataset choice changed or is not yet initialized
dataset_changed = st.session_state['dataset_type'] != dataset_key

if dataset_changed:
    st.session_state['dataset_type'] = dataset_key
    with st.spinner("Loading dataset..."):
        if dataset_key == 'synthetic':
            # Setup parameters for synthetic dataset
            n_samples = st.sidebar.slider("Number of Samples", min_value=500, max_value=5000, value=2000, step=100)
            df = generate_synthetic_credit(n_samples=n_samples, random_state=42)
        else:
            df = load_german_credit()
            # If load fails, fallback
            if df is None:
                st.sidebar.error("Failed to download German Credit from OpenML. Falling back to Synthetic.")
                st.session_state['dataset_type'] = 'synthetic'
                df = generate_synthetic_credit(n_samples=2000, random_state=42)
        st.session_state['df'] = df
        st.session_state['model_results'] = None  # Reset models to trigger retrain

# Add parameters for models in sidebar
st.sidebar.markdown("### 📊 Model Tuning")
test_size = st.sidebar.slider("Test Split Size", min_value=0.10, max_value=0.50, value=0.20, step=0.05)
random_seed = st.sidebar.number_input("Random State Seed", min_value=1, max_value=999, value=42)

# Train button
retrain = st.sidebar.button("🚀 Train & Evaluate Models", width="stretch")

# Train/Retrain trigger
if st.session_state['model_results'] is None or retrain:
    with st.spinner("Preprocessing data, engineering features, and training models..."):
        df = st.session_state['df']
        
        # Split & preprocess
        X_train, X_test, y_train, y_test, prep = prepare_train_test_data(
            df, dataset_type=st.session_state['dataset_type'], test_size=test_size, random_state=random_seed
        )
        
        # Train & evaluate
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save to session state
        st.session_state['preprocessor'] = prep
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['model_results'] = results
    st.sidebar.success("Pipeline executed successfully!")

# Retrieve objects
df = st.session_state['df']
results = st.session_state['model_results']
preprocessor = st.session_state['preprocessor']

# Setup Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Exploratory Data Analysis", 
    "⚙️ Model Comparison & Metrics", 
    "🔮 Credit Risk Evaluator"
])

# =====================================================================
# TAB 1: EXPLORATORY DATA ANALYSIS
# =====================================================================
with tab1:
    st.markdown("### 🔍 Dataset Summary & Insights")
    
    # 1. Key Statistics row
    c1, c2, c3, c4 = st.columns(4)
    total_records = len(df)
    default_rate = df['Credit_Default'].mean() * 100
    
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Applicants</div>
                <div class="metric-card-value">{total_records:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dd2c00;">
                <div class="metric-card-title">Default Rate</div>
                <div class="metric-card-value">{default_rate:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        avg_age = df['Age'].mean()
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #00c853;">
                <div class="metric-card-title">Average Applicant Age</div>
                <div class="metric-card-value">{avg_age:.1f} Yrs</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        if st.session_state['dataset_type'] == 'synthetic':
            avg_income = df['Annual_Income'].mean()
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ffd600;">
                    <div class="metric-card-title">Average Annual Income</div>
                    <div class="metric-card-value">${avg_income:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            avg_credit = df['Credit_Amount'].mean()
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ffd600;">
                    <div class="metric-card-title">Average Credit Amount</div>
                    <div class="metric-card-value">DM {avg_credit:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)

    # 2. Interactive Charts
    col_chart_left, col_chart_right = st.columns(2)
    
    with col_chart_left:
        st.markdown("#### Feature Relationships with Default")
        # Let users select a feature to plot against default
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Credit_Default']
        plot_feat = st.selectbox("Select variable to visualize:", options=numeric_cols, index=0)
        
        # Plot distribution boxplot or histogram
        fig_dist = px.box(
            df, 
            x='Credit_Default', 
            y=plot_feat,
            color='Credit_Default',
            color_discrete_map={0: '#2a5298', 1: '#dd2c00'},
            labels={'Credit_Default': 'Default Status (0=Good, 1=Bad)'},
            title=f"Distribution of {plot_feat} by Default Status"
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, width="stretch")

    with col_chart_right:
        st.markdown("#### Correlation Heatmap (Numerical Features)")
        # Calculate correlation matrix
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_corr, width="stretch")

    # 3. Categorical breakdown if synthetic
    if st.session_state['dataset_type'] == 'synthetic':
        c_break_1, c_break_2 = st.columns(2)
        with c_break_1:
            st.markdown("#### Housing Status vs Default Rate")
            housing_default = df.groupby('Housing_Status')['Credit_Default'].mean().reset_index()
            housing_default['Default_Rate_%'] = housing_default['Credit_Default'] * 100
            fig_house = px.bar(
                housing_default,
                x='Housing_Status',
                y='Default_Rate_%',
                color='Housing_Status',
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Default Rate (%) by Housing Status"
            )
            st.plotly_chart(fig_house, width="stretch")
            
        with c_break_2:
            st.markdown("#### Credit Utilization vs Income")
            fig_scatter = px.scatter(
                df,
                x='Annual_Income',
                y='Credit_Utilization_Ratio',
                color='Credit_Default',
                color_discrete_map={0: '#2a5298', 1: '#dd2c00'},
                opacity=0.6,
                title="Credit Utilization vs. Annual Income"
            )
            st.plotly_chart(fig_scatter, width="stretch")

    # Data sample preview
    st.markdown("#### 📁 Raw Data Sample Preview")
    st.dataframe(df.head(20), width="stretch")

# =====================================================================
# TAB 2: MODEL COMPARISON & METRICS
# =====================================================================
with tab2:
    st.markdown("### 🏆 Classifier Performance Comparison")
    
    # 1. Compile metric table
    metric_rows = []
    for model_name, data in results.items():
        row = {'Model': model_name}
        row.update({m: f"{v:.4f}" for m, v in data['metrics'].items()})
        metric_rows.append(row)
    
    metrics_df = pd.DataFrame(metric_rows).set_index('Model')
    st.table(metrics_df)
    
    # 2. Performance curves (ROC and PR Curves)
    curve_col_left, curve_col_right = st.columns(2)
    
    with curve_col_left:
        st.markdown("#### ROC Curves")
        fig_roc = go.Figure()
        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guess (AUC=0.50)'))
        
        for model_name, data in results.items():
            fpr = data['curves']['fpr']
            tpr = data['curves']['tpr']
            auc = data['metrics']['ROC-AUC']
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC={auc:.3f})"))
            
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
            title="Receiver Operating Characteristic (ROC)"
        )
        st.plotly_chart(fig_roc, width="stretch")
        
    with curve_col_right:
        st.markdown("#### Precision-Recall Curves")
        fig_pr = go.Figure()
        
        for model_name, data in results.items():
            precision = data['curves']['precision']
            recall = data['curves']['recall']
            f1 = data['metrics']['F1-Score']
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"{model_name} (F1={f1:.3f})"))
            
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(yanchor="bottom", y=0.05, xanchor="left", x=0.05),
            title="Precision-Recall Curve"
        )
        st.plotly_chart(fig_pr, width="stretch")

    # 3. Feature Importance & Confusion Matrix
    st.markdown("---")
    st.markdown("### 🔍 Model Diagnostic Insights")
    
    selected_model_name = st.selectbox("Select Model for Diagnostics:", options=list(results.keys()), index=2) # Default Random Forest
    model_data = results[selected_model_name]
    
    diag_col_left, diag_col_right = st.columns(2)
    
    with diag_col_left:
        st.markdown(f"#### Feature Importances ({selected_model_name})")
        if model_data['feature_importances']:
            fi_df = pd.DataFrame(list(model_data['feature_importances'].items()), columns=['Feature', 'Importance'])
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(12)  # Top 12 features
            
            fig_fi = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale="Viridis",
                title=f"Relative Feature Weights in {selected_model_name}"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, width="stretch")
        else:
            st.info("Feature importance not supported for this model class.")
            
    with diag_col_right:
        st.markdown("#### Confusion Matrix")
        cm = np.array(model_data['confusion_matrix'])
        labels = ['Good (0)', 'Default (1)']
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix - {selected_model_name}"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_cm, width="stretch")

# =====================================================================
# TAB 3: CREDIT RISK EVALUATOR (INTERACTIVE PREDICTION)
# =====================================================================
with tab3:
    st.markdown("### 🔮 Interactive Credit Risk Assessment")
    st.markdown("Enter financial details for a credit applicant to compute their creditworthiness, risk tier, and predicted FICO credit score.")
    
    # Select which model to use for prediction
    eval_model_name = st.selectbox(
        "Select Scoring Model for Assessment:",
        options=list(results.keys()),
        index=2 # default Random Forest
    )
    selected_model_pipeline = results[eval_model_name]['model_object']
    
    st.markdown("---")
    
    if st.session_state['dataset_type'] == 'synthetic':
        # Create input form for synthetic model features
        form_col1, form_col2, form_col3 = st.columns(3)
        
        with form_col1:
            age_in = st.slider("Applicant Age", min_value=18, max_value=100, value=35)
            income_in = st.number_input("Annual Income ($)", min_value=5000, max_value=1000000, value=65000, step=5000)
            housing_in = st.selectbox("Housing Status", options=["Own", "Mortgage", "Rent"], index=1)
            emp_years_in = st.slider("Employment Duration (Years)", min_value=0, max_value=50, value=5)
            
        with form_col2:
            debt_in = st.number_input("Total Debt ($)", min_value=0, max_value=2000000, value=15000, step=1000)
            payment_in = st.number_input("Monthly Financial Payments ($)", min_value=0, max_value=50000, value=250, step=50)
            savings_in = st.number_input("Savings Account Balance ($)", min_value=0, max_value=1000000, value=8000, step=500)
            
        with form_col3:
            util_in = st.slider("Credit Card Utilization Ratio (%)", min_value=0, max_value=100, value=32, step=1) / 100.0
            open_lines_in = st.slider("Number of Open Credit Lines", min_value=1, max_value=30, value=5)
            late_pay_in = st.selectbox("Late Payments in last 12 months", options=[0, 1, 2, 3, 4, "5+"], index=0)
            
            if late_pay_in == "5+":
                late_pay_in = 5
                
        # Trigger assessment
        assess = st.button("🔍 Assess Creditworthiness", type="primary", width="stretch")
        
        if assess:
            # Construct dictionary
            input_dict = {
                'Age': [age_in],
                'Annual_Income': [income_in],
                'Total_Debt': [debt_in],
                'Monthly_Payment': [payment_in],
                'Credit_Utilization_Ratio': [util_in],
                'Num_Open_Credit_Lines': [open_lines_in],
                'Num_Late_Payments_1Yr': [late_pay_in],
                'Employment_Duration_Years': [emp_years_in],
                'Housing_Status': [housing_in],
                'Savings_Balance': [savings_in]
            }
            
            # Convert to dataframe
            input_df = pd.DataFrame(input_dict)
            
            # Preprocess
            input_processed = preprocessor.transform(input_df)
            
            # Run prediction probability
            prob_default = selected_model_pipeline.predict_proba(input_processed)[0, 1]
            
            # Map score
            score, band, color, description = calculate_credit_score(prob_default)
            
            # Layout predictions output
            st.markdown("---")
            st.markdown("### 📝 Credit Risk Assessment Report")
            
            out_col_left, out_col_right = st.columns([1, 1.5])
            
            with out_col_left:
                st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
                # Create a gauge chart for FICO Score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>Risk Band: {band}</b>", 'font': {'size': 20, 'color': color}},
                    gauge={
                        'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': color},
                        'bgcolor': "#e9ecef",
                        'borderwidth': 2,
                        'bordercolor': "#dee2e6",
                        'steps': [
                            {'range': [300, 580], 'color': 'rgba(221, 44, 0, 0.1)'},
                            {'range': [580, 670], 'color': 'rgba(255, 109, 0, 0.1)'},
                            {'range': [670, 740], 'color': 'rgba(255, 214, 0, 0.1)'},
                            {'range': [740, 800], 'color': 'rgba(100, 221, 23, 0.1)'},
                            {'range': [800, 850], 'color': 'rgba(0, 198, 83, 0.1)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=280, 
                    margin=dict(t=50, b=10, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with out_col_right:
                # Key applicant details
                dti = debt_in / (income_in + 1e-5)
                pti = (payment_in * 12) / (income_in + 1e-5)
                
                # Check status
                decision = "🟢 APPROVED" if score >= 670 else "🔴 DENIED"
                decision_color = "#2e7d32" if score >= 670 else "#c62828"
                
                st.markdown(f"""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;">
                        <h4 style="margin-top:0;">Assessment Summary</h4>
                        <p style="font-size:1.4rem; font-weight:700; color:{decision_color}; margin-bottom: 1rem;">
                            Decision: {decision}
                        </p>
                        <p><b>Description:</b> {description}</p>
                        <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #eee;">
                        <h5 style="margin-bottom: 0.5rem;">Engineered Credit Ratios:</h5>
                        <ul>
                            <li><b>Debt-to-Income (DTI):</b> {dti:.1%} <em>(Healthy limit: &lt; 40%)</em></li>
                            <li><b>Payment-to-Income (PTI):</b> {pti:.1%} <em>(Healthy limit: &lt; 15%)</em></li>
                            <li><b>Credit Card Utilization:</b> {util_in:.1%} <em>(Healthy limit: &lt; 30%)</em></li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
            # Key Warnings & Positive Factors
            st.markdown("#### 💡 Decision Factors Analysis")
            w_col1, w_col2 = st.columns(2)
            
            with w_col1:
                st.markdown("##### ⚠️ Risk Warning Factors")
                warnings = []
                if util_in > 0.5:
                    warnings.append(f"High credit utilization ({util_in:.0%}) represents stretched revolving lines.")
                if late_pay_in > 0:
                    warnings.append(f"Recorded {late_pay_in} late payment(s) in the past 12 months.")
                if dti > 0.45:
                    warnings.append(f"High debt-to-income ratio ({dti:.1%}) limits capacity to take new credit.")
                if pti > 0.20:
                    warnings.append(f"Monthly payment obligations represent a high proportion of income ({pti:.1%}).")
                
                if warnings:
                    for w in warnings:
                        st.warning(w)
                else:
                    st.info("No major credit warning factors identified.")
                    
            with w_col2:
                st.markdown("##### 🌟 Positive Supporting Factors")
                positives = []
                if income_in >= 90000:
                    positives.append("Applicant income level is in the upper quartile.")
                if util_in <= 0.30:
                    positives.append(f"Excellent low credit utilization ({util_in:.0%}).")
                if late_pay_in == 0:
                    positives.append("Clean payment record with no late payments in the last year.")
                if savings_in > (payment_in * 6):
                    positives.append(f"Savings balance (${savings_in:,}) is healthy (covers &gt;6 months of payments).")
                if emp_years_in >= 5:
                    positives.append(f"Stable employment tenure ({emp_years_in} years).")
                    
                if positives:
                    for p in positives:
                        st.success(p)
                else:
                    st.info("No strong positive factors identified.")

    else:
        # German credit dataset predictor
        st.info("Interactive input for the raw German Credit dataset is simplified to its primary numerical indicators. You can modify these values below:")
        
        # Identify German credit numerical columns
        germ_col1, germ_col2 = st.columns(2)
        
        with germ_col1:
            dur_months_in = st.slider("Duration (Months)", min_value=4, max_value=72, value=24)
            credit_amt_in = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=3000, step=100)
            install_rate_in = st.selectbox("Installment Rate (% of Income)", options=[1, 2, 3, 4], index=1)
            
        with germ_col2:
            age_in = st.slider("Applicant Age (Years)", min_value=19, max_value=75, value=35)
            residence_in = st.slider("Residence Since (Years)", min_value=1, max_value=4, value=2)
            existing_credits_in = st.selectbox("Existing Credits at Bank", options=[1, 2, 3, 4], index=0)
            
        # Get categorical default choices based on German credit definition
        # Since we use all columns, let's prefill default choices for all the categorical features
        # we will use the most frequent categories as defaults
        assess_german = st.button("🔍 Assess Creditworthiness (German Credit Model)", type="primary", width="stretch")
        
        if assess_german:
            # Reconstruct row using a template row from the dataset and replacing numeric variables
            # This is a robust way to handle the 20 columns without cluttering the UI with 15 dropdowns
            row_template = df.iloc[[0]].drop(columns=['Credit_Default']).copy()
            
            # Update values
            row_template['Age'] = [age_in]
            row_template['Duration_Months'] = [dur_months_in]
            row_template['Credit_Amount'] = [credit_amt_in]
            row_template['Installment_Rate'] = [install_rate_in]
            row_template['Residence_Since_Years'] = [residence_in]
            row_template['Num_Existing_Credits'] = [existing_credits_in]
            
            # Process and Predict
            input_processed = preprocessor.transform(row_template)
            prob_default = selected_model_pipeline.predict_proba(input_processed)[0, 1]
            
            # Map score
            score, band, color, description = calculate_credit_score(prob_default)
            
            # Layout predictions output
            st.markdown("---")
            st.markdown("### 📝 German Credit Assessment Report")
            
            out_col_left, out_col_right = st.columns([1, 1.5])
            
            with out_col_left:
                st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>Risk Band: {band}</b>", 'font': {'size': 20, 'color': color}},
                    gauge={
                        'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': color},
                        'bgcolor': "#e9ecef",
                        'borderwidth': 2,
                        'bordercolor': "#dee2e6",
                        'steps': [
                            {'range': [300, 580], 'color': 'rgba(221, 44, 0, 0.1)'},
                            {'range': [580, 670], 'color': 'rgba(255, 109, 0, 0.1)'},
                            {'range': [670, 740], 'color': 'rgba(255, 214, 0, 0.1)'},
                            {'range': [740, 800], 'color': 'rgba(100, 221, 23, 0.1)'},
                            {'range': [800, 850], 'color': 'rgba(0, 198, 83, 0.1)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=280, 
                    margin=dict(t=50, b=10, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with out_col_right:
                decision = "🟢 APPROVED" if score >= 670 else "🔴 DENIED"
                decision_color = "#2e7d32" if score >= 670 else "#c62828"
                
                st.markdown(f"""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;">
                        <h4 style="margin-top:0;">Assessment Summary</h4>
                        <p style="font-size:1.4rem; font-weight:700; color:{decision_color}; margin-bottom: 1rem;">
                            Decision: {decision}
                        </p>
                        <p><b>Description:</b> {description}</p>
                        <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #eee;">
                        <p>Evaluating using parameters: <b>Duration:</b> {dur_months_in} Months, <b>Amount:</b> DM {credit_amt_in:,}, <b>Age:</b> {age_in} Years.</p>
                    </div>
                """, unsafe_allow_html=True)
