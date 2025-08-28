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

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
    Returns: fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_shap
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
    mape_scores = []
    std_scores = []

    total_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        t0 = time.perf_counter()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_std = np.std(y_test)

        xgb_model = xgb.XGBRegressor(**param_dict)

        # fit (silent). For small datasets verbose evaluation slows things; avoid eval_set here.
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores = mape_scores if 'mape_scores' in locals() else []
        mape_scores.append(mape)
        std_scores.append(y_std)

        fold_results.append({
            'Fold': fold_num,
            'R²': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MSE': float(mse),
            'MAPE (%)': float(mape),
            'Std': float(y_std)
        })

        t1 = time.perf_counter()

    total_time = time.perf_counter() - total_start

    # Training model on feature_cols for predictions
    training_model = xgb.XGBRegressor(**param_dict)
    training_model.fit(X, y)

    # final model on X_all for feature importance and SHAP
    # ensure indices align with y used in CV (use rows where target exists)
    final_idx = y.index
    X_all_for_fit = X_all.loc[final_idx] if not X_all.empty else X.loc[final_idx]
    final_model = xgb.XGBRegressor(**param_dict)
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

    fold_results.append({
        'Fold': 'Mean',
        'R²': float(np.mean(r2_scores)),
        'RMSE': float(np.mean(rmse_scores)),
        'MAE': float(np.mean(mae_scores)),
        'MSE': float(np.mean(mse_scores)),
        'MAPE (%)': float(np.mean(mape_scores)),
        'Std': float(np.mean(std_scores))
    })

    summary_stats = {
        'CV R² Mean': float(np.mean(r2_scores)),
        'CV RMSE Mean': float(np.mean(rmse_scores)),
        'CV MAE Mean': float(np.mean(mae_scores)),
        'CV MSE Mean': float(np.mean(mse_scores)),
        'CV MAPE Mean': float(np.mean(mape_scores)),
    }

    return fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_fit

def create_shap_plots(model, X_sample, feature_names):
    """Create SHAP plots for model interpretation"""
    if not SHAP_AVAILABLE:
        st.warning("SHAP is not available. Please install it using: pip install shap")
        return None, None, None
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for sample
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot data
        shap_summary = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP': np.abs(shap_values).mean(axis=0)
        }).sort_values('Mean_SHAP', ascending=False)
        
        return explainer, shap_values, shap_summary
    except Exception as e:
        st.error(f"Error creating SHAP analysis: {str(e)}")
        return None, None, None

def create_prediction_interface(training_model, df, feature_cols):
    """Create interface for user input predictions"""
    st.subheader("🎯 Make Predictions")
    st.write("Enter values for all columns to predict the financial soundness (fs) score:")
    
    # Get column statistics for input validation
    stats = df.describe()
    
    # Create input fields for all columns
    user_inputs = {}
    
    # Organize inputs in columns
    input_cols = st.columns(3)
    all_columns = [col for col in df.columns if col != 'fs']
    all_columns = [col for col in all_columns if col != 'year']
    
    for i, col in enumerate(all_columns):
        if col == "bank_no" or col=="year":
            continue
        if col == "bank_age":
            i = 9
        with input_cols[i % 3]:
            if col in stats.columns:
                min_val = float(stats.loc['min', col])
                max_val = float(stats.loc['max', col])
                mean_val = float(stats.loc['mean', col])
                
                user_inputs[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range: {min_val:.3f} to {max_val:.3f}"
                )
            else:
                user_inputs[col] = st.number_input(f"{col}", value=0.0)
    
    # Predict button
    if st.button("🚀 Predict FS Score", type="primary"):
        # Create input dataframe with only the training features
        input_df = pd.DataFrame([user_inputs])
        prediction_input = input_df[feature_cols]
        
        # Make prediction
        prediction = training_model.predict(prediction_input)[0]
        
        # Display prediction
        st.success(f"**Predicted Financial Soundness (FS) Score: {prediction:.4f}**")
        

def create_csv_predictions(training_model, df, feature_cols):
    """Create predictions for all rows in the dataset"""
    st.subheader("📊 Dataset Predictions")
    
    # Check if target column exists
    if 'fs' not in df.columns:
        st.warning("Target column 'fs' not found. Only predictions will be shown.")
        has_target = False
    else:
        has_target = True
    
    # Prepare prediction input
    prediction_input = df[feature_cols].fillna(df[feature_cols].median())
    
    # Make predictions
    predictions = training_model.predict(prediction_input)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_fs'] = predictions
    
    if has_target:
        results_df['fs_difference'] = results_df['fs'] - results_df['predicted_fs']
        results_df['absolute_difference'] = np.abs(results_df['fs_difference'])
        
        # Calculate metrics
        actual_fs = results_df['fs'].dropna()
        pred_fs = results_df.loc[actual_fs.index, 'predicted_fs']
        
        r2 = r2_score(actual_fs, pred_fs)
        rmse = np.sqrt(mean_squared_error(actual_fs, pred_fs))
        mae = mean_absolute_error(actual_fs, pred_fs)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.6f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
    
    # Display preview
    st.write("**Dataset with Predictions:**")
    
    # Show relevant columns
    if has_target:
        display_cols = ['fs', 'predicted_fs', 'fs_difference', 'absolute_difference'] + feature_cols
    else:
        display_cols = ['predicted_fs'] + feature_cols
    
    preview_df = results_df[display_cols].head(10)
    st.dataframe(preview_df, use_container_width=True)
    
    # Download button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Predictions CSV",
        csv_data,
        "predictions_with_differences.csv",
        "text/csv",
        help="Download the complete dataset with predictions and differences"
    )
    
    return results_df

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:  # Excel files
        return pd.read_excel(uploaded_file)

def main():
    st.title("XGBoost Model Training Dashboard")
    st.markdown("---")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your CSV or Excel file containing the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            
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
            
            # Training section
            st.markdown("---")
            st.header("Model Training")
            
            # Train button
            if st.button("🚀 Train XGBoost Model", type="primary"):
                with st.spinner("Training XGBoost model..."):
                    results = train_xgboost_model(df, random_seed=1471)
                    
                    if results is not None:
                        fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_shap = results
                        
                        # Store models in session state for later use
                        st.session_state.training_model = training_model
                        st.session_state.final_model = final_model
                        st.session_state.X_all_for_shap = X_all_for_shap
                        st.session_state.df = df
                        st.session_state.feature_cols = ["hhis", "hhit", "ccr", "mcr", "ownership", "inflation", "bank_age"]
                        
                        # Display results
                        
                        # Summary metrics
                        st.subheader("📊 Cross-Validation Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Mean", f"{summary_stats['CV R² Mean']:.6f}")
                            st.metric("RMSE Mean", f"{summary_stats['CV RMSE Mean']:.4f}")
                        with col2:
                            st.metric("MAE Mean", f"{summary_stats['CV MAE Mean']:.4f}")
                            st.metric("MSE Mean", f"{summary_stats['CV MSE Mean']:.4f}")
                        with col3:
                            st.metric("MAPE Mean", f"{summary_stats['CV MAPE Mean']:.4f}%")
                        
                        # Fold-wise results table
                        st.subheader("📋 Fold-wise Results")
                        fold_df = pd.DataFrame(fold_results)
                        
                        # Format the dataframe for better display
                        fold_df_display = fold_df.copy()
                        fold_df_display['R²'] = fold_df_display['R²'].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x)
                        fold_df_display['RMSE'] = fold_df_display['RMSE'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        fold_df_display['MAE'] = fold_df_display['MAE'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        fold_df_display['MSE'] = fold_df_display['MSE'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        fold_df_display['MAPE (%)'] = fold_df_display['MAPE (%)'].apply(lambda x: f"{x:.4f}%" if isinstance(x, (int, float)) else x)
                        fold_df_display['Std'] = fold_df_display['Std'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        
                        st.dataframe(fold_df_display, use_container_width=True, hide_index=True)
                        
                        # Feature importance
#                         st.subheader("🎯 Feature Importance")
#                         
#                         col1, col2 = st.columns([2, 1])
#                         
#                         with col1:
#                             # Feature importance plot
#                             fig = px.bar(
#                                 feature_importance.head(15), 
#                                 x='Importance', 
#                                 y='Feature',
#                                 orientation='h',
#                                 title="Top Feature Importance",
#                                 color='Importance',
#                                 color_continuous_scale='viridis'
#                             )
#                             fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
#                             st.plotly_chart(fig, use_container_width=True)
#                         
#                         with col2:
#                             st.write("**Top Features:**")
#                             top_features = feature_importance.head(10).copy()
#                             top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.4f}")
#                             st.dataframe(top_features, use_container_width=True, hide_index=True)
                        
                        # SHAP Analysis
                        if SHAP_AVAILABLE:
                            st.subheader("🔍 SHAP Analysis")
                            with st.spinner("Generating SHAP insights..."):
                                explainer, shap_values, shap_summary = create_shap_plots(
                                    final_model, 
                                    X_all_for_shap.head(50),  # Use sample for performance
                                    X_all_for_shap.columns.tolist()
                                )
                                
                                if shap_summary is not None:
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        # SHAP summary plot
                                        fig_shap = px.bar(
                                            shap_summary.head(15),
                                            x='Mean_SHAP',
                                            y='Feature',
                                            orientation='h',
                                            title="SHAP Feature Impact (Mean Absolute SHAP Values)",
                                            color='Mean_SHAP',
                                            color_continuous_scale='plasma'
                                        )
                                        fig_shap.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                                        st.plotly_chart(fig_shap, use_container_width=True)
                                    
                                    with col2:
                                        st.write("**SHAP Impact Ranking:**")
                                        shap_display = shap_summary.head(10).copy()
                                        shap_display['Mean_SHAP'] = shap_display['Mean_SHAP'].apply(lambda x: f"{x:.4f}")
                                        st.dataframe(shap_display, use_container_width=True, hide_index=True)
                                        
                                        st.info("SHAP values show how much each feature contributes to individual predictions, providing more detailed insights than traditional feature importance.")
                        else:
                            st.warning("⚠️ SHAP is not available. Install it with: `pip install shap` to get detailed model interpretability insights.")
                        
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
            
            # Prediction interface (only show if model is trained)
            if 'training_model' in st.session_state:
                st.markdown("---")
                create_prediction_interface(
                    st.session_state.training_model, 
                    st.session_state.df,
                    st.session_state.feature_cols
                )
                
                st.markdown("---")
                create_csv_predictions(
                    st.session_state.training_model, 
                    st.session_state.df,
                    st.session_state.feature_cols
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
        
        
        **Target Variable:**
        - `fs` - Target variable (continuous)
        
        
        
        **🔍 SHAP Analysis:**
        - Get detailed model interpretability insights
        - Understand how each feature contributes to predictions
        - Install SHAP with: `pip install shap`
        
        **🎯 Individual Predictions:**
        - Input custom values for all columns
        - Get instant FS score predictions
        - See which features are used for training vs. display
        
        **📊 Dataset Predictions:**
        - Generate predictions for entire dataset
        - Compare actual vs predicted FS values
        - Download results with difference calculations
        
        ### 🔧 Hyperparameters (Optimized for Small Datasets)
        
        - **`n_estimators: 55`** - Number of boosting rounds. Lower value to prevent overfitting on small datasets
        - **`max_depth: 5`** - Maximum tree depth. Shallow trees reduce complexity for limited data
        - **`learning_rate: 0.1`** - Step size shrinkage. Standard rate balancing training speed and accuracy
        - **`subsample: 0.45`** - Fraction of samples used per tree. Low value prevents overfitting with limited rows
        - **`colsample_bytree: 0.5`** - Fraction of features used per tree. Reduces overfitting and adds regularization
        - **`reg_alpha: 0.015`** - L1 regularization. Light regularization to prevent overfitting
        - **`reg_lambda: 1.5`** - L2 regularization. Stronger L2 penalty for model stability on small data
        - **`tree_method: 'hist'`** - Histogram-based algorithm for faster CPU training
        
        ### ⚙️ Model Configuration
        - **Algorithm:** XGBoost Regressor  
        - **Cross-Validation:** 5-Fold Stratified  
        - **Feature Importance:** Calculated on all available features  
        - **SHAP Analysis:** Uses model trained on all features
        - **Predictions:** Based only on core training features
        """)

if __name__ == "__main__":
    main()
