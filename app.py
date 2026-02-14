import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.train_models import (
    load_dataset,
    split_dataset,
    build_models,
    evaluate_models,
    CLASS_NAMES,
)


st.set_page_config(
    page_title="Heart Disease Classification - ML Models Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥",
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main content styling */
    .main {
        background-color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Header styling */
    h1 {
        color: #1a202c;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.875rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #4a5568;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1a202c;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #ffffff;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        padding: 18px 36px;
        font-weight: 700;
        color: #718096;
        font-size: 1.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] button {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563eb;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #2563eb;
        border-bottom: 4px solid #2563eb;
        font-weight: 700;
    }
    
    .stTabs [aria-selected="true"] button {
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #1a202c;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1e40af;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background-color: #059669;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        font-size: 0.95rem;
    }
    
    .stDownloadButton>button:hover {
        background-color: #047857;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info/Warning/Error box styling */
    .stAlert {
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        padding: 1rem;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        font-weight: 500;
        color: #4a5568;
        font-size: 0.95rem;
    }
    
    /* File uploader styling */
    .stFileUploader label {
        font-weight: 500;
        color: #4a5568;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_all_models():
    """
    Load data, split into train/test, build and train all models.

    Cached so that it runs only once while the app is running.
    """
    X, y, instances_in_dataset = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    models = build_models()
    metrics_df, trained_models, reports, confusion_matrices = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )
    return X, y, X_train, X_test, y_train, y_test, metrics_df, trained_models, reports, confusion_matrices, instances_in_dataset


def main():
    # Professional header
    st.title("Heart Disease Classification: Machine Learning Model Comparison")
    st.markdown("""
        <p style="font-size: 0.95rem; color: #718096; margin-bottom: 2rem;">
            Created by <strong>Syed Shiraz Ahmed</strong>
        </p>
    """, unsafe_allow_html=True)
    
    # Project overview in a clean box
    with st.container():
        st.markdown("""
            <div style="background-color: #f7fafc; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
                <h3 style="color: #1a202c; margin-top: 0;">Project Overview</h3>
                <p style="color: #4a5568; line-height: 1.6; margin-bottom: 0.5rem;">
                    This application implements and compares six machine learning classification models on the Heart Disease dataset, 
                    which contains 13 clinical features used to predict the presence or absence of heart disease.
                </p>
                <p style="color: #4a5568; line-height: 1.6; margin-bottom: 0;">
                    <strong>Key Features:</strong> Dataset exploration • Model performance comparison • Individual model analysis • Prediction on custom data
                </p>
            </div>
        """, unsafe_allow_html=True)

    (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        metrics_df,
        trained_models,
        reports,
        confusion_matrices,
        instances_in_dataset,
    ) = train_all_models()

    # Sidebar: Professional model selection
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("Model Selection")
    model_name = st.sidebar.selectbox(
        "Choose a classification model:",
        metrics_df["ML Model Name"].tolist()
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Data Upload for Predictions")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file for predictions",
        type=["csv"],
        help="Upload a CSV file with the same feature columns as the training dataset (no target column)"
    )

    st.sidebar.info(
        "**Note:** The uploaded file must have the same feature columns as the original dataset in the same order."
    )

    # Download button for sample test data
    try:
        sample_file_path = "data/to_be_predicted.csv"
        with open(sample_file_path, "rb") as f:
            st.sidebar.download_button(
                label="Download Sample Data for Prediction",
                data=f,
                file_name="to_be_predicted.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.sidebar.warning("Sample test file not found.")

    # Tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dataset Overview", "Metric Comparison", "Model Analysis", "Predictions"]
    )

    # ===== Tab 1: Dataset Overview =====
    with tab1:
        st.header("Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instances", f"{instances_in_dataset:,}")
        with col2:
            st.metric("Total Features", X.shape[1])
        with col3:
            st.metric("Models Trained", 6)
        
        st.markdown("---")
        
        st.subheader("Sample Data")
        st.dataframe(X.head(), use_container_width=True)

        st.subheader("Class Distribution")
        class_counts = y.value_counts().rename(
            index={0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
        )
        st.bar_chart(class_counts)

    # ===== Tab 2: Metrics Comparison Across All Models =====
    with tab2:
        st.header("Model Performance Comparison")        
        st.info("**Note:** The test data is already pre-uploaded to decrease the load on the evaluator. The evaluator can however upload the CSV to make predictions on the Predictions Tab.")
        # Show nicely rounded metrics
        metrics_display = metrics_df.copy()
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            metrics_display[col] = metrics_display[col].round(4)

        st.subheader("Performance Summary")
        st.dataframe(metrics_display, use_container_width=True)
        
        # Bar charts for metric comparison
        st.subheader("Metric Visualizations")
        
        metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        
        # Create a 2x3 grid of bar charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            bars = ax.bar(metrics_df["ML Model Name"], metrics_df[metric], color='steelblue', alpha=0.7)
            ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrices for all models
        st.subheader("Confusion Matrices")
        
        # Determine grid layout based on number of models
        num_models = len(confusion_matrices)
        cols_per_row = 3
        rows = (num_models + cols_per_row - 1) // cols_per_row
        
        fig_cm, axes_cm = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
        if num_models == 1:
            axes_cm = [axes_cm]
        else:
            axes_cm = axes_cm.flatten() if rows > 1 else [axes_cm] if num_models == 1 else axes_cm
        
        for idx, (model_name_cm, cm) in enumerate(confusion_matrices.items()):
            ax = axes_cm[idx]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax,
                cbar=True
            )
            ax.set_title(f"{model_name_cm}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
        
        # Hide any unused subplots
        for idx in range(num_models, len(axes_cm)):
            axes_cm[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig_cm)

    # ===== Tab 3: Selected Model Analysis =====
    with tab3:
        st.header(f"Analysis: {model_name}")

        # Performance Metrics at the top
        st.subheader("Performance Metrics")
        
        # Display the key metrics for the selected model
        model_metrics = metrics_df[metrics_df["ML Model Name"] == model_name].copy()
        model_metrics_values = model_metrics[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].iloc[0]
        
        # Create a styled metrics display with columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{model_metrics_values['Accuracy']:.4f}")
            st.metric("AUC", f"{model_metrics_values['AUC']:.4f}")
        
        with col2:
            st.metric("Precision", f"{model_metrics_values['Precision']:.4f}")
            st.metric("Recall", f"{model_metrics_values['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{model_metrics_values['F1']:.4f}")
            st.metric("MCC", f"{model_metrics_values['MCC']:.4f}")
        
        st.markdown("---")

        st.subheader("Confusion Matrix")
        cm = confusion_matrices[model_name]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            annot_kws={"size": 14},
            cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{model_name}", fontsize=12, pad=15)
        st.pyplot(fig)

    # ===== Tab 4: Predictions on Uploaded Data =====
    with tab4:
        st.header("Make Predictions")
        
        st.info(f"**Active Model:** {model_name}")

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return

            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            # Check columns match the training features
            expected_cols = list(X.columns)
            if list(uploaded_df.columns) != expected_cols:
                st.warning(
                    "**Column Mismatch:** The uploaded file columns do not match the training data. "
                    "Please ensure your CSV has the same feature columns in the same order as shown in the Dataset Overview tab."
                )
            else:
                model = trained_models[model_name]
                preds = model.predict(uploaded_df)

                # Map numeric predictions to readable labels
                label_map = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
                pred_labels = pd.Series(preds).map(label_map)

                result_df = uploaded_df.copy()
                result_df["Prediction"] = pred_labels
                
                st.success(f"Successfully generated {len(result_df)} predictions")
                
                st.subheader("Prediction Results")
                result_df["Prediction"] = pred_labels

                st.write("**Predictions:**")
                st.dataframe(result_df, use_container_width=True)
                
                # Display prediction summary
                st.subheader("Prediction Summary")
                prediction_counts = result_df["Prediction"].value_counts()
                
                col1, col2 = st.columns(2)
                for idx, (label, count) in enumerate(prediction_counts.items()):
                    with [col1, col2][idx]:
                        percentage = (count / len(result_df)) * 100
                        st.metric(label, f"{count} ({percentage:.1f}%)")

                st.markdown("---")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=result_df.to_csv(index=False),
                    file_name="heart_disease_predictions.csv",
                    mime="text/csv",
                )
        else:
            st.info("Please upload a CSV file in the sidebar to generate predictions. Use the Download Sample Data for Prediction button from the sidebar to download the sample CSV to upload.")


if __name__ == "__main__":
    main()

