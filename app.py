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
    page_title="Heart Disease Classification Models",
    layout="wide",
)


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
    st.title("Heart Disease Classification")
    st.subheader("Multiple Classification Models Comparison")

    st.markdown(
        """
This app demonstrates **six different classification models** on the
**Heart Disease** dataset (CSV with 13 features, binary target).

You can:
- View the dataset
- Compare model performance using common evaluation metrics
- Select a model
- Upload your own test CSV file (features only) to get predictions
"""
    )

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

    # Sidebar: model selection and upload
    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox(
        "Select a model", metrics_df["ML Model Name"].tolist()
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with feature columns only (no target column).",
        type=["csv"],
    )

    st.sidebar.markdown(
        """
**Note:**  
The uploaded file should have the **same feature columns**
as the original dataset (same names and order).
"""
    )

    # Download button for sample test data
    try:
        sample_file_path = "data/to_be_predicted.csv"
        with open(sample_file_path, "rb") as f:
            st.sidebar.download_button(
                label="Download Sample CSV To Predict With",
                data=f,
                file_name="to_be_predicted.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.sidebar.warning("Sample test file not found.")

    # Tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dataset Overview", "Metric Comparison", "Selected Model Analysis", "Predictions with Uploaded Data"]
    )

    # ===== Tab 1: Dataset Overview =====
    with tab1:
        st.header("Dataset Overview")
        st.write(f"**Total instances:** {instances_in_dataset}")
        st.write(f"**Total features:** {X.shape[1]}")

        st.write("**First 5 rows of the dataset:**")
        st.dataframe(X.head())

        st.write("**Class distribution (target values):**")
        class_counts = y.value_counts().rename(
            index={0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
        )
        st.bar_chart(class_counts)

    # ===== Tab 2: Metrics Comparison Across All Models =====
    with tab2:
        st.header("Metrics Comparison Across All Models")

        # Show nicely rounded metrics
        metrics_display = metrics_df.copy()
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            metrics_display[col] = metrics_display[col].round(4)

        st.dataframe(metrics_display, use_container_width=True)
        
        # Bar charts for metric comparison
        st.subheader("Visual Comparison of Metrics")
        
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
        st.subheader("Confusion Matrices for All Models")
        
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
        st.header(f"Analysis for {model_name}")

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
        
        st.divider()

        st.subheader("Confusion Matrix")
        cm = confusion_matrices[model_name]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        st.pyplot(fig)

    # ===== Tab 4: Predictions on Uploaded Data =====
    with tab4:
        st.header("Upload Test Data and Get Predictions")
        st.subheader(f"Model: {model_name}")

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV file: {e}")
                return

            st.write("**Uploaded data (first 5 rows):**")
            st.dataframe(uploaded_df.head())

            # Check columns match the training features
            expected_cols = list(X.columns)
            if list(uploaded_df.columns) != expected_cols:
                st.warning(
                    "Column names or order do not match the training data.\n\n"
                    "Please make sure your CSV has exactly the same feature columns "
                    "and in the same order as shown in the Dataset Overview tab."
                )
            else:
                model = trained_models[model_name]
                preds = model.predict(uploaded_df)

                # Map numeric predictions to readable labels
                label_map = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
                pred_labels = pd.Series(preds).map(label_map)

                result_df = uploaded_df.copy()
                result_df["Prediction"] = pred_labels

                st.write("**Predictions:**")
                st.dataframe(result_df.head())

                st.download_button(
                    label="Download predictions as CSV",
                    data=result_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        else:
            st.info("Please upload a CSV file in the sidebar to see predictions.")


if __name__ == "__main__":
    main()

