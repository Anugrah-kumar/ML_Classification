import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix



@st.cache_resource
def load_model(path):
    if not path:
        return None
    return joblib.load(path)

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# --- Custom CSS Theme ---
st.markdown("""
    <style>
        /* Main Background */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #164e63 100%);
            color: #ffffff;
        }
        
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%);
        }
        
        /* Page Background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #164e63 100%);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #e0f2fe;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Main Title */
        h1 {
            background: linear-gradient(90deg, #06b6d4, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 3em;
        }
        
        /* Expandable Sections */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #06b6d4 0%, #0891b2 100%) !important;
            color: #e0f2fe !important;
            border-radius: 8px;
            border: 1px solid #00d4ff;
        }
        
        /* Expander Content Background - Multiple selectors */
        div[data-testid="stExpander"] {
            background: rgba(249, 115, 22, 0.15) !important;
        }
        
        div[data-testid="stExpander"] > div {
            background: rgba(249, 115, 22, 0.15) !important;
            border: 1px solid rgba(249, 115, 22, 0.3) !important;
        }
        
        div[data-testid="stExpander"] > div > div {
            background: rgba(249, 115, 22, 0.15) !important;
        }
        
        .streamlit-expanderContent {
            background: rgba(249, 115, 22, 0.15) !important;
            border: 1px solid rgba(249, 115, 22, 0.3) !important;
            border-radius: 8px;
            padding: 15px !important;
        }
        
        /* Direct div styling for expanders */
        [class*="stExpander"] {
            background: rgba(249, 115, 22, 0.15) !important;
        }
        
        /* Metric Cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(13, 71, 161, 0.3) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 15px;
        }
        
        /* Info Box */
        .stInfo {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%) !important;
            border-left: 4px solid #06b6d4 !important;
        }
        
        /* Success Box */
        .stSuccess {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(74, 222, 128, 0.1) 100%) !important;
            border-left: 4px solid #22c55e !important;
        }
        
        /* Error Box */
        .stError {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(248, 113, 113, 0.1) 100%) !important;
            border-left: 4px solid #ef4444 !important;
        }
        
        /* Warning Box */
        .stWarning {
            background: linear-gradient(135deg, rgba(239, 108, 0, 0.2) 0%, rgba(251, 146, 60, 0.1) 100%) !important;
            border-left: 4px solid #f97316 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #0d47a1 0%, #1565c0 100%) !important;
            color: #e0f2fe !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #06b6d4 0%, #0ea5e9 100%) !important;
            box-shadow: 0 8px 16px rgba(6, 182, 212, 0.3);
        }
        
        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(90deg, #059669 0%, #10b981 100%) !important;
            color: #f0fdf4 !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #047857 0%, #059669 100%) !important;
            box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(15, 23, 42, 0.5) !important;
            border: 1px solid rgba(6, 182, 212, 0.3) !important;
            color: #e0f2fe !important;
            border-radius: 8px;
        }
        
        /* File Uploader */
        .stFileUploader {
            background: rgba(249, 115, 22, 0.2) !important;
            border: 2px solid rgba(249, 115, 22, 0.4) !important;
            border-radius: 8px;
            padding: 15px !important;
        }
        
        /* File uploader drop zone */
        div[data-testid="stFileUploader"] {
            background: rgba(249, 115, 22, 0.15) !important;
            border: 2px solid rgba(249, 115, 22, 0.3) !important;
            border-radius: 8px !important;
        }
        
        /* File name text */
        .stFileUploader div {
            color: #e0f2fe !important;
        }
        
        /* File size value - Green color */
        .stFileUploader span {
            color: #22c55e !important;
            font-weight: 600 !important;
        }
        
        /* File uploader info text */
        .stFileUploader p {
            color: #22c55e !important;
        }
        
        /* Slider */
        .stSlider > div > div > div {
            color: #e0f2fe !important;
        }
        
        /* Divider */
        hr {
            border-color: rgba(6, 182, 212, 0.3) !important;
        }
        
        /* Dataframe */
        .stDataFrame {
            background: rgba(15, 23, 42, 0.3) !important;
            border: 1px solid rgba(6, 182, 212, 0.2) !important;
            border-radius: 8px;
        }
        
        /* Table styling */
        [data-testid="stDataFrame"] > div {
            background: transparent !important;
        }
        
        /* Text */
        body, [data-testid="stMarkdownContainer"] {
            color: #e0f2fe;
        }
        
        /* Tabs */
        [role="tab"] {
            background: rgba(13, 71, 161, 0.2) !important;
            border-radius: 8px;
            color: #e0f2fe !important;
        }
        
        [role="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #0d47a1 0%, #1565c0 100%) !important;
            border-bottom: 3px solid #06b6d4 !important;
        }
        
        /* Markdown links */
        a {
            color: #06b6d4 !important;
        }
        
        a:hover {
            color: #0ea5e9 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Default Prediction System")
st.markdown("Upload a test dataset to evaluate different machine learning models or predict a single loan application.")

# --- Sidebar ---
st.sidebar.header("Configuration")

# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

# 2. Model Selection
model_options = [
    "Logistic_Regression", 
    "Decision_Tree", 
    "kNN", 
    "Naive_Bayes", 
    "Random_Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select Model", model_options)

# --- Main Content ---

if uploaded_file is not None:
    tab1, tab2 = st.tabs(["📊 Batch Prediction", "🔮 Single Input Prediction"])
    
    with tab1:
        # Load Data
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📊 Uploaded Dataset Preview")
            
            # File preview and download section
            with st.expander("📁 Input File Preview & Download", expanded=False):
                st.info("Original uploaded CSV file")
                
                # Add row selection slider
                max_rows_available = len(df)
                num_rows = st.slider("Number of rows to preview:", min_value=5, max_value=max_rows_available, value=min(10, max_rows_available), step=5, key='input_rows')
                st.write(f"**Showing {num_rows} of {max_rows_available} total rows**")
                st.dataframe(df.head(num_rows), use_container_width=True, height=600)
                
                # File info
                col_a, col_b = st.columns(2)
                col_a.metric("Total Rows", len(df))
                col_b.metric("Total Columns", len(df.columns))
                
                csv_input = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Input CSV",
                    data=csv_input,
                    file_name='input_data.csv',
                    mime='text/csv',
                    key='input_download'
                )
            
            #st.dataframe(df.head(), use_container_width=True)
            
            # Check for Target Column
            target_col = 'loan_status'
            if target_col not in df.columns:
                st.error(f"❌ Dataset must contain the target column '{target_col}' for evaluation.")
            else:
                
                # **HANDLE MISSING VALUES FIRST**
                initial_rows = len(df)
                df = df.dropna()
                rows_dropped = initial_rows - len(df)
                if rows_dropped > 0:
                    st.warning(f"⚠️ Dropped {rows_dropped} rows with missing values. Working with {len(df)} rows.")
                
                # Update person_gender column as male=1 and female=0
                df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})
                
                # Update person_education 
                df['person_education'] = df['person_education'].map({'High School': 0, 'Associate': 1,'Bachelor': 2, 'Master': 3,'Doctorate': 4})

                # Update previous_loan_defaults_on_file column as Yes=1 and No=0
                df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
                
                # **HANDLE ANY NaN VALUES CREATED BY MAPPING**
                # Fill NaN values created by mapping with 0 (default/unknown category)
                df = df.fillna(0)
                
                # Use vectorized one-hot encoding for person_home_ownership
                if 'person_home_ownership' in df.columns:
                    home_ownership_dummies = pd.get_dummies(df['person_home_ownership'], prefix='person_home_ownership')
                    df = pd.concat([df, home_ownership_dummies], axis=1)
                    df.drop(['person_home_ownership'], axis=1, inplace=True)

                # Use vectorized one-hot encoding for loan_intent
                if 'loan_intent' in df.columns:
                    loan_intent_dummies = pd.get_dummies(df['loan_intent'], prefix='loan_intent')
                    df = pd.concat([df, loan_intent_dummies], axis=1)
                    df.drop(['loan_intent'], axis=1, inplace=True)
                
                # **FINAL NaN CHECK AND FILL**
                remaining_nans = df.isna().sum().sum()
                if remaining_nans > 0:
                    st.warning(f"⚠️ Found {remaining_nans} NaN values after preprocessing. Filling with 0.")
                    df = df.fillna(0)

                # Prepare X and y
                X_test = df.drop(target_col, axis=1)
                y_test = df[target_col]
                
                # Load Model
                model_path = f"model/{selected_model_name.lower()}.joblib"
                
                try:
                    pipeline = load_model(model_path)
                    
                    # Predict
                    y_pred = pipeline.predict(X_test)
                    
                    # Get Probabilities (if supported) for AUC
                    if hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = None
                    
                    # --- Data Preview ---
                    #st.write("### 👁️ Processed Data Preview")
                    # preview_df = X_test.copy()
                    # preview_df['Actual Status'] = y_test.values
                    # preview_df['Predicted Status'] = y_pred
                    # if y_prob is not None:
                    #     preview_df['Default Probability'] = y_prob
                    #st.dataframe(preview_df.head(10))
                    
                    # --- Metrics Calculation ---
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
                    
                    # --- Display Metrics ---
                    st.write(f"### 🚀 Performance of {selected_model_name}")
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("AUC Score", f"{auc:.4f}")
                    col3.metric("Precision", f"{prec:.4f}")
                    col4.metric("Recall", f"{rec:.4f}")
                    col5.metric("F1 Score", f"{f1:.4f}")
                    col6.metric("MCC", f"{mcc:.4f}")
                    
                    # --- Confusion Matrix ---
                    st.write("### 🧩 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model_name}')
                    st.pyplot(fig)
                    
                    # --- Clinical Interpretation of Confusion Matrix ---
                    st.write("### 📋 Confusion Matrix Interpretation")
                    
                    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                    
                    interpretation_cols = st.columns(4)
                    
                    with interpretation_cols[0]:
                        st.markdown("""
                        <div style="background: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #22c55e;">
                        <b>✅ True Negatives (TN)</b><br>
                        <span style="font-size: 1.5em; color: #22c55e;"><b>{}</b></span><br>
                        <small>Correctly predicted NO DEFAULT</small>
                        </div>
                        """.format(tn), unsafe_allow_html=True)
                    
                    with interpretation_cols[1]:
                        st.markdown("""
                        <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444;">
                        <b>❌ False Positives (FP)</b><br>
                        <span style="font-size: 1.5em; color: #ef4444;"><b>{}</b></span><br>
                        <small>Incorrectly predicted DEFAULT</small>
                        </div>
                        """.format(fp), unsafe_allow_html=True)
                    
                    with interpretation_cols[2]:
                        st.markdown("""
                        <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #fbbf24;">
                        <b>⚠️ False Negatives (FN)</b><br>
                        <span style="font-size: 1.5em; color: #fbbf24;"><b>{}</b></span><br>
                        <small>Missed DEFAULT cases</small>
                        </div>
                        """.format(fn), unsafe_allow_html=True)
                    
                    with interpretation_cols[3]:
                        st.markdown("""
                        <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #f97316;">
                        <b>🚨 True Positives (TP)</b><br>
                        <span style="font-size: 1.5em; color: #f97316;"><b>{}</b></span><br>
                        <small>Correctly predicted DEFAULT</small>
                        </div>
                        """.format(tp), unsafe_allow_html=True)
                    
                    # Interpretation guide
                    with st.expander("📖 Understanding the Confusion Matrix"):
                        st.markdown("""
                        **What each metric means for loan default prediction:**
                        
                        - **True Negatives (TN)**: Loans correctly classified as LOW RISK (will not default)
                          - ✅ Good outcome - Correctly identified safe borrowers
                        
                        - **False Positives (FP)**: Loans incorrectly predicted to default
                          - ⚠️ Business Risk - Rejecting good borrowers, losing revenue
                        
                        - **False Negatives (FN)**: Loans incorrectly predicted as safe but actually defaulted
                          - 🚨 Critical Risk - Approving bad borrowers leads to financial loss
                        
                        - **True Positives (TP)**: Loans correctly classified as HIGH RISK (will default)
                          - ✅ Good outcome - Successfully identified problem loans
                        
                        **Key Insights:**
                        - Minimizing FN (False Negatives) is critical to reduce credit losses
                        - Minimizing FP (False Positives) is important to not reject good customers
                        - The balance depends on your risk tolerance and business strategy
                        """)
                    
                    # --- Additional Visualizations ---
                    st.write("### 📈 Additional Analysis")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    # 1. Predicted Probability Distribution
                    with viz_col1:
                        if y_prob is not None:
                            st.write("#### 📊 Prediction Probability Distribution")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.hist(y_prob, bins=30, edgecolor='black', alpha=0.7, color='#06b6d4')
                            ax.set_xlabel('Predicted Probability of Default')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Predicted Probabilities')
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                    
                    # 2. Prediction Outcome Counts
                    with viz_col2:
                        st.write("#### 🎯 Prediction Outcome Counts")
                        pred_counts = pd.Series(y_pred).value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['#22c55e', '#ef4444']
                        bars = ax.bar(['No Default', 'Default'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], color=colors, edgecolor='black', alpha=0.7)
                        ax.set_ylabel('Count')
                        ax.set_title(f'Prediction Outcome Distribution - {selected_model_name}')
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontweight='bold')
                        st.pyplot(fig)
                    
                    # 3. ROC Curve
                    if y_prob is not None:
                        st.write("#### 📉 ROC Curve")
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr, tpr, color='#06b6d4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                        ax.plot([0, 1], [0, 1], color='#ef4444', lw=2, linestyle='--', label='Random Classifier')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title(f'ROC Curve - {selected_model_name}')
                        ax.legend(loc="lower right")
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                    
                    # 4. Feature Importances (if available)
                    if hasattr(pipeline, 'feature_importances_'):
                        st.write("#### 🔍 Feature Importance (Top 15)")
                        feature_importance = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Importance': pipeline.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#0d47a1', edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Importance Score')
                        ax.set_title(f'Top 15 Feature Importances - {selected_model_name}')
                        ax.invert_yaxis()
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2.,
                                   f'{width:.4f}',
                                   ha='left', va='center', fontsize=9)
                        st.pyplot(fig)
                    elif hasattr(pipeline, 'coef_'):
                        st.write("#### 🔍 Feature Coefficients (Top 15)")
                        feature_coef = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Coefficient': np.abs(pipeline.coef_[0])
                        }).sort_values('Coefficient', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors_coef = ['#0d47a1' if x >= 0 else '#ef4444' for x in pipeline.coef_[0][feature_coef.index]]
                        bars = ax.barh(feature_coef['Feature'], feature_coef['Coefficient'], color=colors_coef, edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Absolute Coefficient Value')
                        ax.set_title(f'Top 15 Feature Coefficients - {selected_model_name}')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    
                    # --- Output File Preview & Download ---
                    st.write("### 📁 Output Files")
                    
                    with st.expander("📊 Output Predictions Preview", expanded=False):
                        st.info("Predictions with probability scores (if available)")
                        output_df = X_test.copy()
                        output_df['Actual_Status'] = y_test.values
                        output_df['Predicted_Status'] = y_pred
                        if y_prob is not None:
                            output_df['Default_Probability'] = y_prob
                        
                        # Add row selection slider
                        max_pred_rows = len(output_df)
                        num_rows_pred = st.slider("Number of rows to preview:", min_value=5, max_value=max_pred_rows, value=min(10, max_pred_rows), step=5, key='output_rows')
                        st.write(f"**Showing {num_rows_pred} of {max_pred_rows} total predictions**")
                        st.dataframe(output_df.head(num_rows_pred), use_container_width=True, height=600)
                        
                        # Data info
                        col_a, col_b = st.columns(2)
                        col_a.metric("Total Predictions", len(output_df))
                        col_b.metric("Features", len(output_df.columns))
                        
                        # Download predictions
                        csv_output = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions (With Probability)",
                            data=csv_output,
                            file_name='predictions_with_probability.csv',
                            mime='text/csv',
                            key='predictions_prob'
                        )
                    
                    with st.expander("📋 Full Results Summary", expanded=False):
                        st.info("Complete dataset with all features and predictions")
                        full_df = df.copy()
                        full_df['Predicted_Status'] = y_pred
                        if y_prob is not None:
                            full_df['Default_Probability'] = y_prob
                        
                        # Add row selection slider
                        max_full_rows = len(full_df)
                        num_rows_full = st.slider("Number of rows to preview:", min_value=5, max_value=max_full_rows, value=min(10, max_full_rows), step=5, key='full_rows')
                        st.write(f"**Showing {num_rows_full} of {max_full_rows} total rows**")
                        st.dataframe(full_df.head(num_rows_full), use_container_width=True, height=600)
                        
                        # Data info
                        col_a, col_b = st.columns(2)
                        col_a.metric("Total Records", len(full_df))
                        col_b.metric("Total Columns", len(full_df.columns))
                        
                        # Download full results
                        csv_full = full_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv_full,
                            file_name='full_results.csv',
                            mime='text/csv',
                            key='full_results'
                        )
                    
                    # Original download button for backward compatibility
                    st.divider()
                    df['Predicted_Status'] = y_pred
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
                    
                except FileNotFoundError:
                    st.error(f"⚠️ Model file '{model_path}' not found. Please run 'model_training.py' first.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    
    with tab2:
        st.write("### 🎯 Predict Loan Default for Single Application")
        
        # Input fields for single prediction
        col1, col2 = st.columns(2)
        
        with col1:
            person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
            person_gender = st.selectbox("Person Gender", ["male", "female"])
            person_education = st.selectbox("Person Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
            person_income = st.number_input("Person Income ($)", min_value=0, value=50000)
            person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=70, value=5)
            person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"])
        
        with col2:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
            loan_int_rate = st.slider("Loan Interest Rate (%)", min_value=0.0, max_value=35.0, value=10.0, step=0.1)
            loan_percent_income = st.slider("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
        
        # Show input preview
        st.write("### 📋 Input Preview")
        preview_data = {
            'Person Age': person_age,
            'Gender': person_gender,
            'Education': person_education,
            'Income': f"${person_income:,}",
            'Employment Experience': f"{person_emp_exp} years",
            'Home Ownership': person_home_ownership,
            'Loan Amount': f"${loan_amnt:,}",
            'Loan Intent': loan_intent,
            'Interest Rate': f"{loan_int_rate}%",
            'Loan % of Income': f"{loan_percent_income:.2%}",
            'Credit History': f"{cb_person_cred_hist_length} years",
            'Credit Score': credit_score,
            'Previous Defaults': previous_loan_defaults_on_file
        }
        
        col1, col2, col3 = st.columns(3)
        for idx, (key, value) in enumerate(preview_data.items()):
            if idx % 3 == 0:
                col1.metric(key, value)
            elif idx % 3 == 1:
                col2.metric(key, value)
            else:
                col3.metric(key, value)
        
        # Predict button
        if st.button("🚀 Predict Loan Default", key="predict_single"):
            try:
                # Create single input dataframe
                single_input = pd.DataFrame({
                    'person_age': [person_age],
                    'person_gender': [person_gender],
                    'person_education': [person_education],
                    'person_income': [person_income],
                    'person_emp_exp': [person_emp_exp],
                    'person_home_ownership': [person_home_ownership],
                    'loan_amnt': [loan_amnt],
                    'loan_intent': [loan_intent],
                    'loan_int_rate': [loan_int_rate],
                    'loan_percent_income': [loan_percent_income],
                    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                    'credit_score': [credit_score],
                    'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
                })
                
                # Apply same preprocessing as batch data
                # **HANDLE MISSING VALUES**
                single_input = single_input.fillna(0)
                
                # Mapping categorical variables
                single_input['person_gender'] = single_input['person_gender'].map({'male': 1, 'female': 0})
                single_input['person_education'] = single_input['person_education'].map({'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4})
                single_input['previous_loan_defaults_on_file'] = single_input['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
                
                # **HANDLE ANY NaN VALUES CREATED BY MAPPING**
                single_input = single_input.fillna(0)
                
                # Use vectorized one-hot encoding for person_home_ownership
                if 'person_home_ownership' in single_input.columns:
                    single_input['person_home_ownership_MORTGAGE'] = 0
                    single_input['person_home_ownership_OTHER'] = 0
                    single_input['person_home_ownership_OWN'] = 0
                    single_input['person_home_ownership_RENT'] = 0
                    home_ownership_value = single_input['person_home_ownership'].iloc[0]
                    if home_ownership_value in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
                        single_input[f'person_home_ownership_{home_ownership_value}'] = 1
                    single_input.drop(['person_home_ownership'], axis=1, inplace=True)
                
                # Use vectorized one-hot encoding for loan_intent
                if 'loan_intent' in single_input.columns:
                    single_input['loan_intent_DEBTCONSOLIDATION'] = 0
                    single_input['loan_intent_EDUCATION'] = 0
                    single_input['loan_intent_HOMEIMPROVEMENT'] = 0
                    single_input['loan_intent_MEDICAL'] = 0
                    single_input['loan_intent_PERSONAL'] = 0
                    single_input['loan_intent_VENTURE'] = 0
                    loan_intent_value = single_input['loan_intent'].iloc[0]
                    if loan_intent_value in ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
                        single_input[f'loan_intent_{loan_intent_value}'] = 1
                    single_input.drop(['loan_intent'], axis=1, inplace=True)
                
                # **FINAL NaN CHECK AND FILL**
                single_input = single_input.fillna(0)
                
                # Show processed data preview
                st.write("### 🔍 Processed Features")
                col_info1, col_info2 = st.columns(2)
                col_info1.metric("Total Features", len(single_input.columns))
                col_info2.metric("Rows", len(single_input))
                st.dataframe(single_input, use_container_width=True)
                
                # Load the selected model
                model_path = f"model/{selected_model_name.lower()}.joblib"
                pipeline = load_model(model_path)
                
                # Make prediction
                prediction = pipeline.predict(single_input)[0]
                
                # Get probability if available
                if hasattr(pipeline, "predict_proba"):
                    prediction_proba = pipeline.predict_proba(single_input)[0]
                    default_prob = prediction_proba[1]
                    no_default_prob = prediction_proba[0]
                else:
                    default_prob = None
                
                # Display results
                st.write("### 📋 Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **Loan Default: YES** - High Risk")
                    else:
                        st.success("✅ **Loan Default: NO** - Low Risk")
                    
                    st.write(f"**Selected Model:** {selected_model_name}")
                
                with col2:
                    if default_prob is not None:
                        st.write(f"**Probability of Default:** {default_prob:.2%}")
                        st.write(f"**Probability of No Default:** {no_default_prob:.2%}")
                    
                    # Display confidence level
                    max_prob = max(default_prob, no_default_prob) if default_prob is not None else 0.5
                    if max_prob >= 0.8:
                        st.info("🎯 **Confidence:** Very High")
                    elif max_prob >= 0.6:
                        st.info("📊 **Confidence:** Moderate")
                    else:
                        st.warning("⚠️ **Confidence:** Low")
                
                # --- Export Input & Output ---
                st.write("### 📁 Export Files")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📥 Export Input Data"):
                        st.info("Input features used for prediction")
                        # Create input dataframe with original values
                        input_export = pd.DataFrame({
                            'Person Age': [person_age],
                            'Gender': [person_gender],
                            'Education': [person_education],
                            'Income': [person_income],
                            'Employment Experience': [person_emp_exp],
                            'Home Ownership': [person_home_ownership],
                            'Loan Amount': [loan_amnt],
                            'Loan Intent': [loan_intent],
                            'Interest Rate': [loan_int_rate],
                            'Loan % Income': [loan_percent_income],
                            'Credit History (Years)': [cb_person_cred_hist_length],
                            'Credit Score': [credit_score],
                            'Previous Defaults': [previous_loan_defaults_on_file]
                        })
                        st.dataframe(input_export, use_container_width=True)
                        csv_input = input_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Input CSV",
                            data=csv_input,
                            file_name='single_input.csv',
                            mime='text/csv',
                            key='single_input_file'
                        )
                
                with col2:
                    with st.expander("📤 Export Prediction Output"):
                        st.info("Prediction results for this application")
                        # Create output dataframe
                        output_export = pd.DataFrame({
                            'Prediction': ['Default' if prediction == 1 else 'No Default'],
                            'Risk Level': ['High Risk' if prediction == 1 else 'Low Risk'],
                            'Model Used': [selected_model_name],
                            'Default Probability': [f"{default_prob:.4f}" if default_prob is not None else 'N/A'],
                            'No Default Probability': [f"{no_default_prob:.4f}" if default_prob is not None else 'N/A'],
                            'Confidence': ['Very High' if max_prob >= 0.8 else ('Moderate' if max_prob >= 0.6 else 'Low')]
                        })
                        st.dataframe(output_export, use_container_width=True)
                        csv_output = output_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📤 Download Prediction Output",
                            data=csv_output,
                            file_name='single_output.csv',
                            mime='text/csv',
                            key='single_output_file'
                        )
                
            except FileNotFoundError:
                st.error(f"⚠️ Model file not found. Please run 'model_training.py' first.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

else:
    st.info("👋 Please upload a CSV file in the sidebar to begin analysis.")
    st.text("Download a sample from here")
    file_path = "loan_test_data.csv"
    with open(file_path, "rb") as file:
        st.download_button(
            label="📥 Download Sample CSV",
            data=file,
            file_name="loan_test_data.csv",
            mime="text/csv"
        )

    st.markdown("""
    **Expected CSV Format:**
    - Must contain feature columns: person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file
    - Must contain target column: `loan_status` (for evaluation metrics).
    """)