import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import google.generativeai as genai
import os
from dotenv import load_dotenv

import pickle
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample

# SMOTE needs the optional 'imbalanced-learn' package - degrade gracefully
# if it's not installed rather than crashing the whole app
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# SHAP needs the optional 'shap' package - degrade gracefully if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-3.6-flash")
else:
    gemini_model = None

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Generic AI Data Analyst",
    layout="wide"
)

# -----------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------

if "trained" not in st.session_state:
    st.session_state["trained"] = False

st.title("🤖 Generic AI Data Analyst")
st.write(
    "Upload any CSV dataset, clean missing values, encode categorical columns, "
    "train a model, and find the top 10 important features."
)

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # -----------------------------------
    # DATASET PREVIEW
    # -----------------------------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset shape
    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # -----------------------------------
    # DATA TYPES
    # -----------------------------------

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(dtype_df)

    # -----------------------------------
    # HIGH-CARDINALITY CATEGORICAL COLUMNS
    # -----------------------------------

    cat_cols_check = df.select_dtypes(include=["object", "category"]).columns

    if len(cat_cols_check) > 0:

        cardinality_rows = []

        for col in cat_cols_check:

            unique_count = df[col].nunique()
            unique_pct = round((unique_count / df.shape[0]) * 100, 2)

            cardinality_rows.append({
                "Column": col,
                "Unique Values": unique_count,
                "Unique %": unique_pct,
                "High Cardinality": "⚠️ Yes" if (unique_count > 50 or unique_pct > 50) else "No"
            })

        cardinality_df = pd.DataFrame(cardinality_rows).sort_values(
            by="Unique Values",
            ascending=False
        )

        if (cardinality_df["High Cardinality"] == "⚠️ Yes").any():

            st.subheader("High-Cardinality Categorical Columns")

            st.warning(
                "Some categorical columns have a very large number of unique "
                "values (e.g. IDs, names, free text). Label Encoding these turns "
                "them into arbitrary numbers that can mislead the model. Consider "
                "excluding them or using a different encoding strategy."
            )

            st.dataframe(cardinality_df)

    # -----------------------------------
    # REMOVE UNWANTED COLUMNS
    # -----------------------------------

    st.subheader("Remove Unwanted Columns")

    # Auto-detect likely ID / leakage-prone columns:
    # - name looks like an identifier (strong signal on its own)
    # - fully unique text/object column (e.g. emails, UUIDs)
    # - fully unique integer column that's sequential (auto-increment ID
    #   pattern) - NOT just "every value happens to be unique", since a
    #   genuine numeric feature (income, price, etc.) can coincidentally
    #   have all-unique values without being an identifier
    auto_flagged_columns = []

    for col in df.columns:

        looks_like_id = any(
            keyword in col.lower()
            for keyword in ["id", "uuid", "index", "unnamed"]
        )

        is_fully_unique = df[col].nunique() == df.shape[0]

        is_sequential_int = False
        if is_fully_unique and pd.api.types.is_integer_dtype(df[col]):
            sorted_vals = df[col].sort_values().reset_index(drop=True)
            diffs = sorted_vals.diff().dropna()
            # Auto-increment IDs have a constant step (usually 1)
            is_sequential_int = diffs.nunique() == 1

        is_fully_unique_text = (
            is_fully_unique
            and df[col].dtype == "object"
        )

        if looks_like_id or is_sequential_int or is_fully_unique_text:
            auto_flagged_columns.append(col)

    if auto_flagged_columns:

        st.warning(
            f"These columns look like identifiers (unique per row or named "
            f"like an ID): {auto_flagged_columns}. They rarely help "
            f"prediction and can sometimes leak the answer, causing "
            f"unrealistically high accuracy. They're pre-selected for "
            f"removal below — uncheck if you actually want to keep one."
        )

    excluded_columns = st.multiselect(
        "Columns to exclude from training",
        options=list(df.columns),
        default=auto_flagged_columns,
        help=(
            "Drop ID columns, leaked target columns, or anything else you "
            "don't want the model to see. Excluded columns are removed "
            "before cleaning, encoding, and training."
        )
    )

    # -----------------------------------
    # FEATURE ENGINEERING - INTERACTION FEATURES
    # -----------------------------------

    st.subheader("Feature Engineering (Optional)")

    numeric_cols_for_fe = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols_for_fe) >= 2:

        create_interaction = st.checkbox(
            "Create an interaction feature from two numeric columns",
            value=False,
            help=(
                "Combines two numeric columns into one new feature (e.g. "
                "GPA × University Tier). Useful when the effect of one "
                "feature depends on the level of another - a plain "
                "additive model can miss that on its own."
            )
        )

        if create_interaction:

            fe_col1, fe_col2, fe_op = st.columns(3)

            interaction_feature_1 = fe_col1.selectbox(
                "Feature 1", numeric_cols_for_fe, key="fe_feature_1"
            )
            interaction_feature_2 = fe_col2.selectbox(
                "Feature 2",
                [c for c in numeric_cols_for_fe if c != interaction_feature_1],
                key="fe_feature_2"
            )
            interaction_operation = fe_op.selectbox(
                "Operation",
                ["Multiply (×)", "Add (+)", "Subtract (−)", "Divide (÷)"],
                key="fe_operation"
            )

            op_suffix = {
                "Multiply (×)": "x",
                "Add (+)": "plus",
                "Subtract (−)": "minus",
                "Divide (÷)": "div"
            }[interaction_operation]

            interaction_col_name = (
                f"{interaction_feature_1}_{op_suffix}_{interaction_feature_2}"
            )

            # Mutating df here (not just df_features) so the new column
            # flows through to every downstream view AND to the actual
            # training step, which starts fresh from df.copy()
            if interaction_operation == "Multiply (×)":
                df[interaction_col_name] = (
                    df[interaction_feature_1] * df[interaction_feature_2]
                )
            elif interaction_operation == "Add (+)":
                df[interaction_col_name] = (
                    df[interaction_feature_1] + df[interaction_feature_2]
                )
            elif interaction_operation == "Subtract (−)":
                df[interaction_col_name] = (
                    df[interaction_feature_1] - df[interaction_feature_2]
                )
            else:
                # Guard against divide-by-zero -> NaN, handled later by
                # the normal missing-value imputation step
                df[interaction_col_name] = df[interaction_feature_1] / (
                    df[interaction_feature_2].replace(0, np.nan)
                )

            st.success(f"Created new feature: `{interaction_col_name}`")
            st.dataframe(
                df[[
                    interaction_feature_1,
                    interaction_feature_2,
                    interaction_col_name
                ]].head()
            )

    # From here on, all exploratory analysis (missing values, duplicates,
    # outliers, correlation) reflects the columns the user actually wants
    # to train on - not the raw upload. This keeps what's displayed
    # consistent with what the model will actually see.
    df_features = df.drop(columns=excluded_columns, errors="ignore")

    # -----------------------------------
    # MISSING VALUES BEFORE CLEANING
    # -----------------------------------

    st.subheader("Missing Values Before Cleaning")

    missing_df = pd.DataFrame({
        "Missing Values": df_features.isnull().sum(),
        "Missing Percentage": (
            df_features.isnull().mean() * 100
        ).round(2)
    })

    st.dataframe(missing_df)

    # -----------------------------------
    # DUPLICATE ROWS CHECK
    # -----------------------------------

    st.subheader("Duplicate Rows")

    duplicate_count = df_features.duplicated().sum()

    remove_duplicates = False

    if duplicate_count > 0:

        st.warning(
            f"Found {duplicate_count} duplicate row(s) "
            f"({round((duplicate_count / df_features.shape[0]) * 100, 2)}% "
            f"of data, checked across the columns you're keeping). "
            f"Duplicates can inflate model performance metrics."
        )

        remove_duplicates = st.checkbox(
            "Remove duplicate rows before training",
            value=True
        )

    else:

        st.success(
            "No duplicate rows found (checked across the columns "
            "you're keeping)."
        )

    # -----------------------------------
    # OUTLIER DETECTION (IQR METHOD)
    # -----------------------------------

    st.subheader("Outlier Detection (IQR Method)")

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns

    handle_outliers = False

    if len(numeric_cols) == 0:

        st.info("No numeric columns found for outlier detection.")

    else:

        outlier_rows = []

        for col in numeric_cols:

            Q1 = df_features[col].quantile(0.25)
            Q3 = df_features[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = df_features[
                (df_features[col] < lower_bound) | (df_features[col] > upper_bound)
            ].shape[0]

            outlier_percentage = round(
                (outlier_count / df_features.shape[0]) * 100, 2
            )

            outlier_rows.append({
                "Column": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Outlier Count": outlier_count,
                "Outlier %": outlier_percentage
            })

        outlier_df = pd.DataFrame(outlier_rows).sort_values(
            by="Outlier %",
            ascending=False
        )

        st.dataframe(outlier_df)

        # Boxplots for visual inspection - all columns in one figure
        with st.expander("View Boxplots for Numeric Columns"):

            fig, ax = plt.subplots(
                figsize=(max(6, len(numeric_cols) * 1.2), 5)
            )

            ax.boxplot(
                [df_features[col].dropna() for col in numeric_cols],
                labels=numeric_cols,
                vert=True
            )

            ax.set_title("Boxplot of All Numeric Columns")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()

            st.pyplot(fig)

        handle_outliers = st.checkbox(
            "Cap outliers using IQR bounds before training "
            "(recommended for skewed numeric data)"
        )

    # -----------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------

    if len(numeric_cols) >= 2:

        st.subheader("Correlation Heatmap (Numeric Columns)")

        corr_matrix = df_features[numeric_cols].corr()

        fig, ax = plt.subplots(
            figsize=(
                max(6, len(numeric_cols) * 0.8),
                max(5, len(numeric_cols) * 0.8)
            )
        )

        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols)

        # Annotate cells with correlation values (skip if too many columns)
        if len(numeric_cols) <= 15:
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(
                        j, i,
                        round(corr_matrix.iloc[i, j], 2),
                        ha="center", va="center",
                        fontsize=7,
                        color="black"
                    )

        ax.set_title("Correlation Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        st.pyplot(fig)

        # Flag pairs of highly correlated features (possible redundancy)
        high_corr_pairs = []

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.85:
                    high_corr_pairs.append({
                        "Feature 1": numeric_cols[i],
                        "Feature 2": numeric_cols[j],
                        "Correlation": round(corr_val, 2)
                    })

        if high_corr_pairs:

            st.info(
                "The following feature pairs are highly correlated "
                "(|correlation| ≥ 0.85). Consider dropping one of each pair "
                "to reduce redundancy."
            )

            st.dataframe(pd.DataFrame(high_corr_pairs))

    # -----------------------------------
    # SELECT TARGET COLUMN
    # -----------------------------------

    st.subheader("Select Target Column")

    target_options = list(df_features.columns)

    target_column = st.selectbox(
        "Choose the column you want to predict",
        target_options
    )

    # -----------------------------------
    # TARGET LEAKAGE CHECK
    # -----------------------------------

    leakage_candidates = []

    for col in target_options:

        if col == target_column:
            continue

        # Exact match with target -> the model would just learn to copy it
        try:
            match_rate = (df[col] == df[target_column]).mean()
        except Exception:
            match_rate = 0

        if match_rate >= 0.98:
            leakage_candidates.append(col)
            continue

        # Near-perfect numeric correlation with target
        if (
            pd.api.types.is_numeric_dtype(df[col])
            and pd.api.types.is_numeric_dtype(df[target_column])
        ):
            corr_with_target = df[[col, target_column]].corr().iloc[0, 1]
            if abs(corr_with_target) >= 0.98:
                leakage_candidates.append(col)

    if leakage_candidates:

        st.error(
            f"⚠️ Possible target leakage: {leakage_candidates} nearly "
            f"duplicate or are almost perfectly correlated with "
            f"'{target_column}'. Training on these will likely give "
            f"unrealistically high accuracy (e.g. ~100%) because the model "
            f"is essentially seeing the answer. Consider excluding them "
            f"above in 'Remove Unwanted Columns'."
        )

    # -----------------------------------
    # CLASS IMBALANCE CHECK (CLASSIFICATION TARGETS)
    # -----------------------------------

    target_unique_preview = set(df[target_column].dropna().unique())

    is_binary_numeric_preview = (
        pd.api.types.is_numeric_dtype(df[target_column])
        and len(target_unique_preview) == 2
        and target_unique_preview.issubset({0, 1})
    )

    is_categorical_target = (
        df[target_column].dtype == "object"
        or df[target_column].dtype == "category"
    )

    # Default: no special handling unless a classification target is found below
    imbalance_method = "Do nothing (use raw class distribution)"
    custom_class_weights = {}

    if is_categorical_target or is_binary_numeric_preview:

        st.subheader("Class Distribution")

        if is_binary_numeric_preview:
            st.info(
                "Target has only two values (0 and 1) — treating this as "
                "a **classification** problem rather than regression."
            )

        class_counts = df[target_column].value_counts()
        class_pct = (class_counts / class_counts.sum() * 100).round(2)

        class_dist_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage": class_pct.values
        })

        st.dataframe(class_dist_df)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(class_dist_df["Class"].astype(str), class_dist_df["Count"])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

        if class_pct.max() >= 80 or class_pct.min() <= 5:

            st.warning(
                "This target column looks imbalanced. Accuracy alone can be "
                "misleading here — check the confusion matrix and per-class "
                "precision/recall after training rather than relying on "
                "accuracy alone."
            )

        elif class_pct.max() >= 65:

            st.info(
                "This target column is moderately imbalanced "
                f"({class_pct.max()}% / {class_pct.min()}%). Not extreme, "
                "but balancing can still help if the minority class matters."
            )

        smote_options = (
            ["SMOTE (synthetic oversampling, recommended for real imbalance)"]
            if SMOTE_AVAILABLE
            else []
        )

        if not SMOTE_AVAILABLE:
            st.caption(
                "ℹ️ Install `imbalanced-learn` (`pip install imbalanced-learn`) "
                "to unlock SMOTE as a balancing option here."
            )

        imbalance_method = st.selectbox(
            "How should training handle class balance?",
            [
                "Do nothing (use raw class distribution)",
                "Use class_weight='balanced' (no data change)",
                "Cost-sensitive weighting (custom ratio per class)",
            ] + smote_options + [
                "Oversample minority class(es) to match majority (duplicate rows)",
                "Undersample majority class(es) to match minority (drops rows)",
            ],
            index=0,
            help=(
                "class_weight='balanced' reweights errors on the minority "
                "class during training - no rows are added or removed. "
                "Cost-sensitive weighting lets you set your own weight per "
                "class (e.g. 30:70) instead of the automatic 'balanced' "
                "ratio. SMOTE generates new synthetic minority samples by "
                "interpolating between existing ones (generally better than "
                "plain duplication). Oversampling duplicates minority rows "
                "(only in the training split) to match the majority count. "
                "Undersampling drops majority rows (only in the training "
                "split) to match the minority count - simplest, but throws "
                "away data. All of these only ever touch the training "
                "split - the test split stays untouched so evaluation "
                "reflects real-world proportions."
            )
        )

        # Custom cost-sensitive weights: one number input per class
        custom_class_weights = {}

        if imbalance_method.startswith("Cost-sensitive"):

            if len(class_counts.index) > 10:

                st.warning(
                    "Too many classes for manual weight entry - falling "
                    "back to class_weight='balanced' instead."
                )
                imbalance_method = "Use class_weight='balanced' (no data change)"

            else:

                st.caption(
                    "Set a relative weight for each class (higher = "
                    "mistakes on that class cost more during training). "
                    "Only the ratio between weights matters, not the "
                    "absolute numbers."
                )

                weight_cols = st.columns(len(class_counts.index))

                for i, class_value in enumerate(class_counts.index):

                    # Default: weight inversely proportional to class
                    # frequency, so it starts at a sensible "balanced"-like
                    # ratio the user can then tweak
                    default_weight = round(
                        100 / class_pct.loc[class_value], 1
                    )

                    custom_class_weights[class_value] = weight_cols[i].number_input(
                        f"Weight for class '{class_value}'",
                        min_value=0.1,
                        value=float(default_weight),
                        step=0.1,
                        key=f"class_weight_{class_value}"
                    )

    # -----------------------------------
    # FEATURE SCALING OPTION
    # -----------------------------------

    st.subheader("Feature Scaling")

    apply_scaling = st.checkbox(
        "Apply feature scaling (StandardScaler) to numeric features",
        value=False,
        help=(
            "Random Forest doesn't require scaling to split correctly, but "
            "scaling keeps numeric features on a comparable range — useful "
            "if you compare results across models or care about feature "
            "magnitudes. Categorical (encoded) columns and the target are "
            "left untouched."
        )
    )

    if st.button("Clean Data and Train Model"):

        # Create copy
        data = df.copy()

        # -----------------------------------
        # REMOVE UNWANTED COLUMNS
        # -----------------------------------

        # Safety net: never drop the target column even if it
        # somehow ended up in the excluded list
        columns_to_drop = [
            col for col in excluded_columns if col != target_column
        ]

        if columns_to_drop:
            data.drop(columns=columns_to_drop, inplace=True)

        # -----------------------------------
        # REMOVE DUPLICATE ROWS
        # -----------------------------------

        if remove_duplicates:
            data.drop_duplicates(inplace=True)

        # -----------------------------------
        # HANDLE OUTLIERS (IQR METHOD)
        # -----------------------------------

        if handle_outliers:

            for col in data.select_dtypes(include=[np.number]).columns:

                # Skip target column - don't distort what we're predicting
                if col == target_column:
                    continue

                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap (Winsorize) instead of dropping rows,
                # so we don't lose data from other columns
                data[col] = data[col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )

        # -----------------------------------
        # REMOVE ROWS WHERE TARGET IS MISSING
        # -----------------------------------

        data.dropna(
            subset=[target_column],
            inplace=True
        )

        # -----------------------------------
        # HANDLE MISSING VALUES
        # -----------------------------------

        for col in data.columns:

            # Skip target column
            if col == target_column:
                continue

            # Only process columns with NaN
            if data[col].isnull().sum() > 0:

                # CATEGORICAL / OBJECT COLUMN
                if (
                    data[col].dtype == "object"
                    or data[col].dtype == "category"
                ):

                    # Replace NaN with Mode
                    data[col] = data[col].fillna(
                        data[col].mode()[0]
                    )

                # NUMERICAL COLUMN
                else:

                    # Check skewness
                    skewness = data[col].skew()

                    # Use Median if skewed
                    if abs(skewness) >= 1:
                        data[col] = data[col].fillna(
                            data[col].median()
                        )

                    # Use Mean if approximately normal
                    else:
                        data[col] = data[col].fillna(
                            data[col].mean()
                        )

        # -----------------------------------
        # HANDLE TARGET COLUMN
        # -----------------------------------

        target_encoder = None

        unique_target_values = set(
            data[target_column].dropna().unique()
        )

        # Binary numeric target (only 0 and 1) -> treat as classification,
        # not regression, even though the dtype is numeric
        is_binary_numeric_target = (
            pd.api.types.is_numeric_dtype(data[target_column])
            and len(unique_target_values) == 2
            and unique_target_values.issubset({0, 1})
        )

        if (
            data[target_column].dtype == "object"
            or data[target_column].dtype == "category"
        ):

            target_encoder = LabelEncoder()

            data[target_column] = target_encoder.fit_transform(
                data[target_column]
            )

            problem_type = "Classification"

        elif is_binary_numeric_target:

            # Already 0/1, just make sure it's a clean integer type
            data[target_column] = data[target_column].astype(int)

            problem_type = "Classification"

        else:
            problem_type = "Regression"

        # -----------------------------------
        # ENCODE OBJECT COLUMNS
        # -----------------------------------

        categorical_columns = data.select_dtypes(
            include=["object", "category"]
        ).columns

        feature_encoders = {}

        for col in categorical_columns:

            encoder = LabelEncoder()

            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

            feature_encoders[col] = encoder

        # -----------------------------------
        # SHOW CLEANED DATA
        # -----------------------------------

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(data.head())

        # -----------------------------------
        # MISSING VALUES AFTER CLEANING
        # -----------------------------------

        st.subheader("Missing Values After Cleaning")

        st.dataframe(
            data.isnull().sum().to_frame(
                "Missing Values"
            )
        )

        # -----------------------------------
        # SPLIT FEATURES AND TARGET
        # -----------------------------------

        X = data.drop(
            target_column,
            axis=1
        )

        y = data[target_column]

        # -----------------------------------
        # TRAIN TEST SPLIT
        # -----------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -----------------------------------
        # FEATURE SCALING
        # -----------------------------------

        scaler = None

        if apply_scaling:

            # Only scale columns that were originally numeric
            # (leaves label-encoded categorical columns untouched)
            numeric_feature_cols = [
                col for col in numeric_cols if col in X.columns
            ]

            if len(numeric_feature_cols) > 0:

                scaler = StandardScaler()

                # Fit on train only, then transform both -
                # avoids leaking test-set statistics into training
                X_train.loc[:, numeric_feature_cols] = scaler.fit_transform(
                    X_train[numeric_feature_cols]
                )

                X_test.loc[:, numeric_feature_cols] = scaler.transform(
                    X_test[numeric_feature_cols]
                )

        # -----------------------------------
        # CLASS IMBALANCE HANDLING
        # -----------------------------------

        # Only applies to classification, and only touches the TRAINING
        # split - never the test split, so evaluation stays honest and
        # reflects real-world class proportions.

        if problem_type == "Classification" and imbalance_method.startswith("SMOTE"):

            minority_count = y_train.value_counts().min()

            # SMOTE needs at least k_neighbors+1 samples in the minority
            # class; shrink k_neighbors automatically for small classes
            # instead of crashing
            k_neighbors = min(5, minority_count - 1)

            if k_neighbors < 1:

                st.warning(
                    "SMOTE needs at least 2 samples in the minority class - "
                    "this dataset's minority class is too small, so training "
                    "proceeded without balancing. Try 'class_weight=balanced' "
                    "instead for very small minority classes."
                )

            else:

                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

        elif problem_type == "Classification" and imbalance_method.startswith("Oversample"):

            train_combined = X_train.copy()
            train_combined[target_column] = y_train.values

            majority_size = train_combined[target_column].value_counts().max()

            balanced_parts = []
            for class_value, group in train_combined.groupby(target_column):
                balanced_parts.append(
                    resample(
                        group,
                        replace=True,
                        n_samples=majority_size,
                        random_state=42
                    )
                )

            train_balanced = pd.concat(balanced_parts).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            X_train = train_balanced.drop(columns=[target_column])
            y_train = train_balanced[target_column]

        elif problem_type == "Classification" and imbalance_method.startswith("Undersample"):

            train_combined = X_train.copy()
            train_combined[target_column] = y_train.values

            minority_size = train_combined[target_column].value_counts().min()

            balanced_parts = []
            for class_value, group in train_combined.groupby(target_column):
                balanced_parts.append(
                    resample(
                        group,
                        replace=False,
                        n_samples=minority_size,
                        random_state=42
                    )
                )

            train_balanced = pd.concat(balanced_parts).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            X_train = train_balanced.drop(columns=[target_column])
            y_train = train_balanced[target_column]

        # -----------------------------------
        # SELECT MODEL
        # -----------------------------------

        if problem_type == "Regression":

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

        else:

            classifier_kwargs = {
                "n_estimators": 100,
                "random_state": 42
            }

            # class_weight='balanced' reweights errors on the minority class -
            # no rows added or removed, just applied at model level
            if imbalance_method.startswith("Use class_weight"):
                classifier_kwargs["class_weight"] = "balanced"

            # Custom cost-sensitive ratio - user-supplied weight per class
            elif imbalance_method.startswith("Cost-sensitive") and custom_class_weights:
                classifier_kwargs["class_weight"] = custom_class_weights

            model = RandomForestClassifier(**classifier_kwargs)

        # -----------------------------------
        # TRAIN MODEL
        # -----------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -----------------------------------
        # PREDICTIONS
        # -----------------------------------

        predictions = model.predict(
            X_test
        )

        # -----------------------------------
        # MODEL PERFORMANCE
        # -----------------------------------

        st.subheader("Model Performance")

        if problem_type == "Regression":

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "MAE",
                round(mae, 2)
            )

            metric2.metric(
                "RMSE",
                round(rmse, 2)
            )

            metric3.metric(
                "R² Score",
                round(r2, 4)
            )

            # -----------------------------------
            # RESIDUAL PLOT
            # -----------------------------------

            st.subheader("Residual Plot")

            residuals = y_test - predictions

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

            ax1.scatter(predictions, residuals, alpha=0.5)
            ax1.axhline(y=0, color="red", linestyle="--")
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals (Actual - Predicted)")
            ax1.set_title("Residuals vs Predicted")

            ax2.scatter(y_test, predictions, alpha=0.5)
            min_val = min(y_test.min(), predictions.min())
            max_val = max(y_test.max(), predictions.max())
            ax2.plot([min_val, max_val], [min_val, max_val], "r--")
            ax2.set_xlabel("Actual Values")
            ax2.set_ylabel("Predicted Values")
            ax2.set_title("Actual vs Predicted")

            fig.tight_layout()
            st.pyplot(fig)

        else:

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            st.metric(
                "Accuracy",
                round(accuracy * 100, 2)
            )

            # -----------------------------------
            # ROC-AUC & PR-AUC
            # -----------------------------------
            # More reliable than raw accuracy on imbalanced data -
            # PR-AUC especially, since it focuses on minority-class
            # performance rather than overall correctness.

            try:

                probabilities = model.predict_proba(X_test)
                n_classes = probabilities.shape[1]

                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
                    pr_auc = average_precision_score(
                        y_test, probabilities[:, 1]
                    )
                else:
                    roc_auc = roc_auc_score(
                        y_test, probabilities,
                        multi_class="ovr", average="macro"
                    )
                    y_test_binarized = pd.get_dummies(y_test)
                    pr_auc = average_precision_score(
                        y_test_binarized, probabilities, average="macro"
                    )

                auc_col1, auc_col2 = st.columns(2)
                auc_col1.metric("ROC-AUC", round(roc_auc, 4))
                auc_col2.metric("PR-AUC (Avg Precision)", round(pr_auc, 4))

                st.caption(
                    "ROC-AUC and PR-AUC hold up better than raw accuracy "
                    "on imbalanced data. PR-AUC in particular focuses on "
                    "how well the model finds the minority class."
                )

            except Exception as auc_error:
                st.info(f"Could not compute ROC-AUC/PR-AUC: {auc_error}")

            # -----------------------------------
            # CONFUSION MATRIX
            # -----------------------------------

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, predictions)

            class_labels = (
                target_encoder.classes_
                if target_encoder is not None
                else sorted(y.unique())
            )

            fig, ax = plt.subplots(
                figsize=(max(4, len(class_labels) * 0.8), max(4, len(class_labels) * 0.8))
            )

            im = ax.imshow(cm, cmap="Blues")

            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(class_labels)))
            ax.set_yticklabels(class_labels)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, cm[i, j],
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black"
                    )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            st.pyplot(fig)

            # -----------------------------------
            # CLASSIFICATION REPORT
            # -----------------------------------

            st.subheader("Classification Report")

            report_dict = classification_report(
                y_test,
                predictions,
                target_names=[str(c) for c in class_labels],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report_dict).transpose().round(3)

            st.dataframe(report_df)

        # -----------------------------------
        # FEATURE IMPORTANCE
        # -----------------------------------

        st.subheader("Top 10 Important Features")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        top_10_features = importance_df.head(10)

        # Show table
        st.dataframe(top_10_features)

        # -----------------------------------
        # FEATURE IMPORTANCE CHART
        # -----------------------------------

        fig, ax = plt.subplots()

        ax.barh(
            top_10_features["Feature"],
            top_10_features["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("Top 10 Important Features")

        # Highest feature at top
        ax.invert_yaxis()

        st.pyplot(fig)

        # -----------------------------------
        # SHAP EXPLAINABILITY
        # -----------------------------------
        # Generic for ANY tabular RandomForest model (classifier or
        # regressor, binary or multiclass) - not tied to specific column
        # names or domains.

        st.subheader("Model Explainability (SHAP)")

        if not SHAP_AVAILABLE:

            st.caption(
                "ℹ️ Install `shap` (`pip install shap`) to unlock per-feature "
                "SHAP explainability here."
            )

        else:

            run_shap = st.checkbox(
                "Compute SHAP explainability (may take a moment on large "
                "datasets)"
            )

            if run_shap:

                with st.spinner("Computing SHAP values..."):

                    try:

                        # Cap sample size for speed - SHAP on tree models is
                        # fast but still scales with rows x features x trees
                        shap_sample = X_test.sample(
                            n=min(200, len(X_test)),
                            random_state=42
                        )

                        explainer = shap.TreeExplainer(model)
                        raw_shap_values = explainer.shap_values(shap_sample)

                        # Normalize across shap API versions / binary vs
                        # multiclass vs regression into one
                        # (n_samples, n_features) array of attributions
                        if isinstance(raw_shap_values, list):
                            # Older API: list of per-class arrays
                            if len(raw_shap_values) == 2:
                                shap_matrix = raw_shap_values[1]
                            else:
                                shap_matrix = np.mean(
                                    [np.abs(sv) for sv in raw_shap_values],
                                    axis=0
                                )
                        else:
                            shap_array = np.array(raw_shap_values)
                            if shap_array.ndim == 3:
                                # Newer API: (n_samples, n_features, n_classes)
                                if shap_array.shape[-1] == 2:
                                    shap_matrix = shap_array[:, :, 1]
                                else:
                                    shap_matrix = np.abs(shap_array).mean(axis=-1)
                            else:
                                shap_matrix = shap_array

                        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

                        shap_importance_df = pd.DataFrame({
                            "Feature": shap_sample.columns,
                            "Mean |SHAP value|": mean_abs_shap
                        }).sort_values(
                            by="Mean |SHAP value|",
                            ascending=False
                        )

                        top_shap_features = shap_importance_df.head(10)

                        st.dataframe(top_shap_features)

                        fig, ax = plt.subplots()

                        ax.barh(
                            top_shap_features["Feature"],
                            top_shap_features["Mean |SHAP value|"]
                        )

                        ax.set_xlabel("Mean |SHAP value| (impact on prediction)")
                        ax.set_ylabel("Features")
                        ax.set_title("Top 10 Features by SHAP Impact")
                        ax.invert_yaxis()

                        st.pyplot(fig)

                        st.caption(
                            "SHAP values show each feature's actual impact on "
                            "individual predictions (signed and additive), "
                            "unlike Random Forest's built-in importance which "
                            "only reflects average split quality. The two "
                            "rankings can differ - SHAP is generally the more "
                            "trustworthy explanation for real-world decisions."
                        )

                    except Exception as shap_error:
                        st.info(f"Could not compute SHAP values: {shap_error}")

        # -----------------------------------
        # FINAL INFORMATION
        # -----------------------------------

        st.success(
            f"Analysis completed successfully! "
            f"Problem Type: {problem_type}"
        )

        # -----------------------------------
        # DOWNLOAD RESULTS
        # -----------------------------------

        st.subheader("Download Results")

        download_col1, download_col2, download_col3 = st.columns(3)

        # Cleaned dataset
        cleaned_csv = data.to_csv(index=False).encode("utf-8")

        download_col1.download_button(
            label="⬇️ Cleaned Dataset (CSV)",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        # Predictions - decode labels back to original classes for classification
        if problem_type == "Classification" and target_encoder is not None:
            actual_decoded = target_encoder.inverse_transform(y_test)
            predicted_decoded = target_encoder.inverse_transform(predictions)
        else:
            actual_decoded = y_test
            predicted_decoded = predictions

        results_df = pd.DataFrame({
            "Actual": actual_decoded,
            "Predicted": predicted_decoded
        })

        predictions_csv = results_df.to_csv(index=False).encode("utf-8")

        download_col2.download_button(
            label="⬇️ Predictions (CSV)",
            data=predictions_csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Trained model
        model_buffer = io.BytesIO()
        pickle.dump(model, model_buffer)

        download_col3.download_button(
            label="⬇️ Trained Model (.pkl)",
            data=model_buffer.getvalue(),
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )

        # Scaler (only exists if scaling was applied) - needed to transform
        # new data the same way before feeding it to the trained model
        if scaler is not None:

            scaler_buffer = io.BytesIO()
            pickle.dump(scaler, scaler_buffer)

            st.download_button(
                label="⬇️ Fitted Scaler (.pkl)",
                data=scaler_buffer.getvalue(),
                file_name="feature_scaler.pkl",
                mime="application/octet-stream"
            )

        # -----------------------------------
        # SAVE RESULTS FOR GEMINI
        # -----------------------------------
        # NOTE: this must stay inside the "if st.button(...)" block,
        # indented at this level, so it only runs right after training.
        st.session_state["trained"] = True
        st.session_state["data"] = data
        st.session_state["problem_type"] = problem_type
        st.session_state["target_column"] = target_column
        st.session_state["top_10_features"] = top_10_features
        st.session_state["model"] = model
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["predictions"] = predictions
        st.session_state["df"] = df

    # -----------------------------------
    # GEMINI AI INSIGHTS
    # -----------------------------------
    # NOTE: this is inside "if uploaded_file is not None:" but OUTSIDE the
    # "if st.button(...)" block, so it stays visible (using session_state)
    # across reruns, e.g. when the user clicks "Generate AI Insights".

    st.divider()
    st.subheader("🤖 Generative AI Insights")

    if gemini_model is None:

        st.warning(
            "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
        )

    elif st.session_state.get("trained", False):

        if st.button("Generate AI Insights"):

            data = st.session_state["data"]
            target_column = st.session_state["target_column"]
            problem_type = st.session_state["problem_type"]
            top_10_features = st.session_state["top_10_features"]

            dataset_summary = f"""
Dataset Shape:
{data.shape}

Columns:
{list(data.columns)}

Data Types:
{data.dtypes.to_string()}

Missing Values:
{data.isnull().sum().to_string()}

Statistics:
{data.describe(include='all').to_string()}

Target Column:
{target_column}

Problem Type:
{problem_type}

Top 10 Important Features:
{top_10_features.to_string(index=False)}
"""

            prompt = f"""
You are a Senior Data Scientist.

Analyze this dataset and produce a professional report.

Include:

1. Dataset Overview

2. Data Quality

3. Key Trends

4. Top Features

5. Model Insights

6. Business Insights

7. Recommendations

Dataset:

{dataset_summary}
"""

            with st.spinner("Generating AI Insights..."):

                try:

                    response = gemini_model.generate_content(prompt)

                    st.success("AI Analysis Generated Successfully")

                    st.markdown(response.text)

                except Exception as e:

                    st.error(f"Gemini Error: {e}")

    else:

        st.info("Please click **Clean Data and Train Model** first.")