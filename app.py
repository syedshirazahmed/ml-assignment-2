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
    page_title="ML Assignment 2 - Classification Models",
    layout="wide",
)


@st.cache_resource
def train_all_models():
    """
    Load data, split into train/test, build and train all models.

    Cached so that it runs only once while the app is running.
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    models = build_models()
    metrics_df, trained_models, reports, confusion_matrices = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )
    return X, y, X_train, X_test, y_train, y_test, metrics_df, trained_models, reports, confusion_matrices


def main():
    st.title("Machine Learning Assignment 2")
    st.subheader("Multiple Classification Models with Streamlit Deployment")

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

    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs(
        ["Dataset Overview", "Model Comparison", "Predictions with Uploaded Data"]
    )

    # ===== Tab 1: Dataset Overview =====
    with tab1:
        st.header("Dataset Overview")
        st.write(f"**Total instances:** {X.shape[0]}")
        st.write(f"**Total features:** {X.shape[1]}")

        st.write("**First 5 rows of the dataset:**")
        st.dataframe(X.head())

        st.write("**Class distribution (target values):**")
        class_counts = y.value_counts().rename(
            index={0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
        )
        st.bar_chart(class_counts)

    # ===== Tab 2: Model Comparison and Confusion Matrix =====
    with tab2:
        st.header("Model Comparison Table")

        # Show nicely rounded metrics
        metrics_display = metrics_df.copy()
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            metrics_display[col] = metrics_display[col].round(4)

        st.dataframe(metrics_display, use_container_width=True)

        st.subheader(f"Confusion Matrix: {model_name}")
        cm = confusion_matrices[model_name]

        fig, ax = plt.subplots()
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

        st.subheader(f"Classification Report: {model_name}")
        st.text(reports[model_name])

    # ===== Tab 3: Predictions on Uploaded Data =====
    with tab3:
        st.header("Upload Test Data and Get Predictions")

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

