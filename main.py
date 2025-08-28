
# Put these before heavy imports (best before importing xgboost)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="XGBoost Model Training",
    page_icon="🚀",
    layout="wide"
)

def create_target_classes(y, n_classes=3):
    """
    Convert continuous target to classes for stratified sampling.
    Matches the reference code approach exactly.
    """
    quantiles = np.linspace(0, 1, n_classes + 1)
    thresholds = np.quantile(y, quantiles)
    classes = np.digitize(y, thresholds) - 1
    classes = np.clip(classes, 0, n_classes - 1)
    return classes

def train_xgboost_model(df, random_seed=1471, cpu_threads=1, show_progress=True):
    """
    Train XGBoost with Streamlit-friendly defaults.
    Returns: fold_results (list of dicts), summary_stats (dict), feature_importance (DataFrame), final_model (XGBRegressor)
    """
    import time
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # required minimal feature set (for CV training)
    feature_cols = ["hhis", "hhit", "ccr", "mcr", "ownership", "inflation", "bank_age"]
    target_col = "fs"

    # verify columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.error(f"Available columns: {list(df.columns)}")
        return None

    # prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
        if show_progress:
            st.warning("Missing values found in features - filled with median values")
    if y.isnull().sum() > 0:
        mask = ~y.isnull()
        removed = (~mask).sum()
        X = X.loc[mask]
        y = y.loc[mask]
        if show_progress:
            st.warning(f"Missing values found in target - removed {removed} rows")

    # X_all for final feature importance. keep only existing columns
    all_features = ["inflation", "hhis", "bank_age", "hhit", "hhic", "asset_size", "ownership", "ccr", "mcr", "hhig"]
    all_features = [c for c in all_features if c in df.columns]
    X_all = df[all_features].copy()
    if X_all.isnull().sum().sum() > 0:
        X_all = X_all.fillna(X_all.median())

    # XGBoost params tuned for small/container environments
    param_dict = {
        'n_estimators': 55,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.45,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.015,
        'reg_lambda': 1.5,
        'random_state': random_seed,
        'n_jobs': cpu_threads,
        'tree_method': 'hist',   # faster for CPU
        'verbosity': 0
    }

    np.random.seed(random_seed)

    # stratified classes for continuous y
    try:
        y_classes = create_target_classes(y, n_classes=3)
    except Exception:
        # fallback to simple binning if edge-case occurs
        y_classes = pd.qcut(y.rank(method="first"), q=3, labels=False, duplicates='drop').astype(int)

    # Initialize CV splitter; fallback to KFold if stratification is impossible
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(skf.split(X, y_classes))
    except Exception:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(kf.split(X))

    fold_results = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mse_scores = []

    total_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        t0 = time.perf_counter()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        xgb_model = xgb.XGBRegressor(**param_dict)

        # fit (silent). For small datasets verbose evaluation slows things; avoid eval_set here.
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mse_scores.append(mse)

        fold_results.append({
            'Fold': f'Fold {fold_num}',
            'R²': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MSE': float(mse)
        })

        t1 = time.perf_counter()

    total_time = time.perf_counter() - total_start

    # final model on X_all for feature importance
    # ensure indices align with y used in CV (use rows where target exists)
    final_idx = y.index
    X_all_for_fit = X_all.loc[final_idx] if not X_all.empty else X.loc[final_idx]
    final_model = xgb.XGBRegressor(n_estimators=10, learning_rate=0.2,
                                   tree_method='hist', n_jobs=cpu_threads, random_state=random_seed, verbosity=0)
    final_model.fit(X_all_for_fit, y)

    # feature importance (if X_all exists). if not, fallback to feature_cols
    fi_features = all_features if all_features else feature_cols
    importances = final_model.feature_importances_
    # ensure length match
    if len(importances) != len(fi_features):
        # align available features
        fi_features = [c for c in fi_features if c in X_all_for_fit.columns]
        importances = final_model.feature_importances_[:len(fi_features)]

    feature_importance = pd.DataFrame({
        'Feature': fi_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    summary_stats = {
        'CV R² Mean': float(np.mean(r2_scores)),
        'CV R² Std': float(np.std(r2_scores)),
        'CV RMSE Mean': float(np.mean(rmse_scores)),
        'CV MAE Mean': float(np.mean(mae_scores)),
        'CV MSE Mean': float(np.mean(mse_scores))
    }

    return fold_results, summary_stats, feature_importance, final_model

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def main():
    st.title("XGBoost Model Training Dashboard")
    st.markdown("---")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file containing the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_csv(uploaded_file)
            
            # Display basic info about the dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", df.shape[0])
            with col2:
                st.metric("📋 Total Columns", df.shape[1])
            with col3:
                st.metric("🎯 Target Column", "fs")
            
            # Show dataset preview
            with st.expander("📋 Dataset Preview", expanded=False):
                st.dataframe(df.head(10))
            
            # Show dataset info
#             with st.expander("ℹ️ Dataset Information"):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.subheader("Column Data Types")
#                     st.write(df.dtypes.to_frame('Data Type'))
#                 with col2:
#                     st.subheader("Missing Values")
#                     missing_df = pd.DataFrame({
#                         'Missing Count': df.isnull().sum(),
#                         'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
#                     })
#                     st.write(missing_df[missing_df['Missing Count'] > 0])
            
            # Training section
            st.markdown("---")
            st.header("Model Training")
            
#             # Show training configuration
#             with st.expander("⚙️ Training Configuration", expanded=True):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write("**Training Features:**")
#                     st.code(["hhis", "hhit", "ccr", "mcr", "ownership", "inflation", "bank_age"])
#                     st.write("**Target Variable:** fs")
#                     st.write("**Cross-Validation:** 5-Fold Stratified")
#                 with col2:
#                     st.write("**XGBoost Parameters:**")
#                     st.code("""n_estimators: 55
# max_depth: 5
# learning_rate: 0.1
# subsample: 0.45
# colsample_bytree: 0.5
# reg_alpha: 0.015
# reg_lambda: 1.5
# random_state: 1471""")
            
            # Train button
            if st.button("Use XGBoost Model", type="primary"):
                with st.spinner("Training XGBoost model..."):
                    results = train_xgboost_model(df, random_seed=1471)
                    
                    if results is not None:
                        fold_results, summary_stats, feature_importance, final_model = results
                        
                        # Display results
                        
                        # Summary metrics
                        st.subheader("📊 Cross-Validation Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("R² Mean", f"{summary_stats['CV R² Mean']:.6f}")
                        with col2:
                            st.metric("R² Std", f"{summary_stats['CV R² Std']:.5f}")
                        with col3:
                            st.metric("RMSE Mean", f"{summary_stats['CV RMSE Mean']:.4f}")
                        with col4:
                            st.metric("MAE Mean", f"{summary_stats['CV MAE Mean']:.4f}")
                        with col5:
                            st.metric("MSE Mean", f"{summary_stats['CV MSE Mean']:.4f}")
                        
                        # Fold-wise results table
                        st.subheader("📋 Fold-wise Results")
                        fold_df = pd.DataFrame(fold_results)
                        
                        # Format the dataframe for better display
                        fold_df_display = fold_df.copy()
                        fold_df_display['R²'] = fold_df_display['R²'].apply(lambda x: f"{x:.6f}")
                        fold_df_display['RMSE'] = fold_df_display['RMSE'].apply(lambda x: f"{x:.4f}")
                        fold_df_display['MAE'] = fold_df_display['MAE'].apply(lambda x: f"{x:.4f}")
                        fold_df_display['MSE'] = fold_df_display['MSE'].apply(lambda x: f"{x:.4f}")
                        
                        st.dataframe(fold_df_display, use_container_width=True)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Feature importance plot
                            fig = px.bar(
                                feature_importance.head(15), 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title="Top Feature Importance",
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("**Top Features:**")
                            top_features = feature_importance.head(10).copy()
                            top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.4f}")
                            st.dataframe(top_features, use_container_width=True, hide_index=True)
                        
                        # Performance visualization
                        st.subheader("📈 Performance Visualization")
                        
                        # Create subplots
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('R² Score by Fold', 'RMSE by Fold', 'MAE by Fold', 'MSE by Fold'),
                            vertical_spacing=0.12
                        )
                        
                        # Add traces
                        folds = [f"Fold {i+1}" for i in range(5)]
                        
                        fig.add_trace(go.Scatter(x=folds, y=[r['R²'] for r in fold_results], 
                                               mode='lines+markers', name='R²'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=folds, y=[r['RMSE'] for r in fold_results], 
                                               mode='lines+markers', name='RMSE'), row=1, col=2)
                        fig.add_trace(go.Scatter(x=folds, y=[r['MAE'] for r in fold_results], 
                                               mode='lines+markers', name='MAE'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=folds, y=[r['MSE'] for r in fold_results], 
                                               mode='lines+markers', name='MSE'), row=2, col=2)
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        # Prepare download data
                        results_summary = pd.DataFrame([summary_stats])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            csv1 = fold_df.to_csv(index=False)
                            st.download_button(
                                "📄 Download Fold Results",
                                csv1,
                                "fold_results.csv",
                                "text/csv"
                            )
                        with col2:
                            csv2 = results_summary.to_csv(index=False)
                            st.download_button(
                                "📊 Download Summary Stats",
                                csv2,
                                "summary_stats.csv",
                                "text/csv"
                            )
                        with col3:
                            csv3 = feature_importance.to_csv(index=False)
                            st.download_button(
                                "🎯 Download Feature Importance",
                                csv3,
                                "feature_importance.csv",
                                "text/csv"
                            )
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Please make sure your CSV file contains the required columns.")
    
    else:
        # Show instructions
        st.info("👆 Please upload a CSV file to begin training")
        
        st.markdown("""
        ### 📋 Requirements
        Your CSV file should contain the following columns:
        
        **Training Features:**
        - `hhis` - Feature 1
        - `hhit` - Feature 2  
        - `ccr` - Feature 3
        - `mcr` - Feature 4
        - `ownership` - Feature 5
        - `inflation` - Feature 6
        - `bank_age` - Feature 7
        
        **Target Variable:**
        - `fs` - Target variable (continuous)
        
        ### ⚙️ Model Configuration
        - **Algorithm:** XGBoost Regressor
        - **Cross-Validation:** 5-Fold Stratified
        - **Random State:** 1471 (for reproducibility)
        - **Feature Importance:** Calculated on all available features
        
        ### 📊 Output Metrics
        The app will provide:
        - Cross-validation R², RMSE, MAE, MSE (mean and std)
        - Fold-wise performance metrics
        - Feature importance ranking
        - Interactive visualizations
        """)

if __name__ == "__main__":
    main()
