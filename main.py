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
from sklearn.preprocessing import MinMaxScaler  # Added MinMaxScaler import
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # Added for saving scaler
import time
# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="XGBoost Model Training",
    layout="wide"
)

def optimize_random_seeds(df, start_seed=1, end_seed=100, top_n=10, show_progress=True):
    """
    Optimize random seeds by training models with different seeds and ranking by CV R² mean.
    
    Parameters:
    - df: Input dataframe
    - start_seed: Starting random seed value
    - end_seed: Ending random seed value  
    - top_n: Number of top seeds to return
    - show_progress: Whether to show progress bar
    
    Returns:
    - results_df: DataFrame with seed results ranked by CV R² mean
    - best_seed: The best performing seed
    - best_results: Results from the best seed
    """
    
    # Required features
    feature_cols = ["hhis", "hhit", "ccr", "mcr", "ownership", "inflation", "bank_age"]
    target_col = "fs"
    
    # Verify columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None, None, None
    
    # Prepare data once
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    if y.isnull().sum() > 0:
        mask = ~y.isnull()
        X = X.loc[mask]
        y = y.loc[mask]
    
    # XGBoost parameters
    base_params = {
        'n_estimators': 55,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.45,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.015,
        'reg_lambda': 1.5,
        'n_jobs': 1,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    seed_results = []
    seed_range = range(start_seed, end_seed + 1)
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, seed in enumerate(seed_range):
        if show_progress:
            status_text.text(f"Testing seed {seed}/{end_seed} (Progress: {i+1}/{len(seed_range)})")
            progress_bar.progress((i + 1) / len(seed_range))
        
        try:
            # Initialize scaler for this seed
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            
            # Set seed for reproducibility
            np.random.seed(seed)
            
            # Create stratified classes
            try:
                y_classes = create_target_classes(y, n_classes=3)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(skf.split(X_scaled, y_classes))
            except Exception:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(kf.split(X_scaled))
            
            # Cross-validation
            r2_scores = []
            rmse_scores = []
            mae_scores = []
            mse_scores = []
            
            params_with_seed = base_params.copy()
            params_with_seed['random_state'] = seed
            
            for train_idx, test_idx in splits:
                X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model = xgb.XGBRegressor(**params_with_seed)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                r2_scores.append(r2_score(y_test, y_pred))
                mse = mean_squared_error(y_test, y_pred)
                mse_scores.append(mse)
                rmse_scores.append(np.sqrt(mse))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
            
            # Calculate means and standard deviations
            r2_mean = np.mean(r2_scores)
            r2_std = np.std(r2_scores, ddof=1) if len(r2_scores) > 1 else 0
            rmse_mean = np.mean(rmse_scores)
            mae_mean = np.mean(mae_scores)
            mse_mean = np.mean(mse_scores)
            
            seed_results.append({
                'Seed': seed,
                'CV_R2_Mean': r2_mean,
                'CV_R2_Std': r2_std,
                'CV_RMSE_Mean': rmse_mean,
                'CV_MAE_Mean': mae_mean,
                'CV_MSE_Mean': mse_mean,
                'R2_Scores': r2_scores  # Store individual fold scores
            })
            
        except Exception as e:
            if show_progress:
                st.warning(f"Error with seed {seed}: {str(e)}")
            continue
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    if not seed_results:
        st.error("No successful seed evaluations completed")
        return None, None, None
    
    # Create results DataFrame
    results_df = pd.DataFrame(seed_results)
    
    # Sort by CV R² mean (descending)
    results_df = results_df.sort_values('CV_R2_Mean', ascending=False).reset_index(drop=True)
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    # Best seed and its results
    best_seed = int(top_results.iloc[0]['Seed'])
    best_results = top_results.iloc[0].to_dict()
    
    return results_df, best_seed, best_results

def display_seed_optimization_results(results_df, best_seed, best_results, top_n=10):
    """Display the results of seed optimization"""
    
    st.subheader(f"🏆 Best Random Seed: {best_seed}")
    
    # Display best seed metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CV R² Mean", f"{best_results['CV_R2_Mean']:.6f}")
    with col2:
        st.metric("CV R² Std", f"{best_results['CV_R2_Std']:.6f}")
    with col3:
        st.metric("CV RMSE Mean", f"{best_results['CV_RMSE_Mean']:.4f}")
    with col4:
        st.metric("CV MAE Mean", f"{best_results['CV_MAE_Mean']:.4f}")
    
    # Top seeds table
    st.subheader(f"📊 Top {top_n} Random Seeds")
    
    display_df = results_df.head(top_n).copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    
    # Reorder columns for better display
    display_df = display_df[['Rank', 'Seed', 'CV_R2_Mean', 'CV_R2_Std', 'CV_RMSE_Mean', 'CV_MAE_Mean', 'CV_MSE_Mean']]
    
    # Format for display
    display_df['CV_R2_Mean'] = display_df['CV_R2_Mean'].apply(lambda x: f"{x:.6f}")
    display_df['CV_R2_Std'] = display_df['CV_R2_Std'].apply(lambda x: f"{x:.6f}")
    display_df['CV_RMSE_Mean'] = display_df['CV_RMSE_Mean'].apply(lambda x: f"{x:.4f}")
    display_df['CV_MAE_Mean'] = display_df['CV_MAE_Mean'].apply(lambda x: f"{x:.4f}")
    display_df['CV_MSE_Mean'] = display_df['CV_MSE_Mean'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visualization of seed performance
    st.subheader("📈 Seed Performance Visualization")
    
    # Create visualization of top seeds
    top_seeds_viz = results_df.head(min(20, len(results_df)))
    
    fig = go.Figure()
    
    # Add R² scores
    fig.add_trace(go.Scatter(
        x=top_seeds_viz['Seed'],
        y=top_seeds_viz['CV_R2_Mean'],
        mode='lines+markers',
        name='CV R² Mean',
        line=dict(color='blue', width=2),
        marker=dict(size=8, color='blue'),
        error_y=dict(
            type='data',
            array=top_seeds_viz['CV_R2_Std'],
            visible=True,
            color='blue',
            thickness=1
        )
    ))
    
    # Highlight best seed
    best_row = top_seeds_viz.iloc[0]
    fig.add_trace(go.Scatter(
        x=[best_row['Seed']],
        y=[best_row['CV_R2_Mean']],
        mode='markers',
        name=f'Best Seed ({best_seed})',
        marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='darkred'))
    ))
    
    fig.update_layout(
        title=f'CV R² Performance Across Different Random Seeds (Top 20)',
        xaxis_title='Random Seed',
        yaxis_title='CV R² Mean',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Seed Optimization Results")
    
    # Prepare download data (remove R2_Scores column for CSV)
    download_df = results_df.drop('R2_Scores', axis=1)
    csv_data = download_df.to_csv(index=False)
    
    st.download_button(
        "📥 Download All Seed Results",
        csv_data,
        "seed_optimization_results.csv",
        "text/csv",
        help="Download complete seed optimization results"
    )
    
    return best_seed

def add_seed_optimization_interface():
    """Add seed optimization interface to the main app"""
    
    st.markdown("---")
    st.header("🎲 Random Seed Optimization")
    st.write("Find the best random seed by testing multiple seeds and ranking by CV R² performance.")
    
    # Controls for seed optimization
    col1, col2, col3 = st.columns(3)
    with col1:
        start_seed = st.number_input("Start Seed", min_value=1, max_value=10000, value=1, step=1)
    with col2:
        end_seed = st.number_input("End Seed", min_value=1, max_value=100000, value=100, step=1)
    with col3:
        top_n = st.number_input("Top N Seeds to Show", min_value=5, max_value=50, value=10, step=1)
    
    # Validation
    if start_seed >= end_seed:
        st.error("Start seed must be less than end seed")
        return
    
    if (end_seed - start_seed + 1) > 1000:
        st.warning("Large seed range detected. This may take a long time. Consider reducing the range.")
    
    # Estimate time
    estimated_time = (end_seed - start_seed + 1) * 2  # Rough estimate: 2 seconds per seed
    if estimated_time > 60:
        st.info(f"Estimated time: ~{estimated_time//60} minutes {estimated_time%60} seconds")
    else:
        st.info(f"Estimated time: ~{estimated_time} seconds")
    
    # Optimization button
    if st.button("🔍 Optimize Random Seeds", type="primary"):
        if 'df' not in st.session_state:
            st.error("Please upload data first.")
            return
        
        start_time = time.time()
        
        # Run optimization
        results_df, best_seed, best_results = optimize_random_seeds(
            st.session_state.df,
            start_seed=start_seed,
            end_seed=end_seed,
            top_n=top_n,
            show_progress=True
        )
        
        if results_df is not None:
            elapsed_time = time.time() - start_time
            st.success(f"Optimization completed in {elapsed_time:.1f} seconds!")
            
            # Store results in session state
            st.session_state.seed_results = results_df
            st.session_state.best_seed = best_seed
            st.session_state.best_results = best_results
            
            # Display results
            display_seed_optimization_results(results_df, best_seed, best_results, top_n)
            
            # Option to train with best seed
            st.markdown("---")
            st.subheader("🚀 Train with Best Seed")
            if st.button(f"Train Model with Best Seed ({best_seed})", type="secondary"):
                with st.spinner(f"Training model with seed {best_seed}..."):
                    results = train_xgboost_model(st.session_state.df, random_seed=best_seed)
                    
                    if results is not None:
                        fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_shap, scaler = results
                        
                        # Store in session state
                        st.session_state.training_model = training_model
                        st.session_state.final_model = final_model
                        st.session_state.X_all_for_shap = X_all_for_shap
                        st.session_state.scaler = scaler
                        st.session_state.feature_cols = feature_cols
                        
                        st.success(f"Model trained successfully with seed {best_seed}!")
                        st.balloons()
        else:
            st.error("Seed optimization failed. Please check your data and try again.")

def add_mse_trees_analysis_to_main():
    """
    Add this function to your main() function after the training section.
    This creates the MSE vs Trees analysis interface.
    """
    # Add this after your existing training section in main()
    
    # MSE vs Trees Analysis
    st.markdown("---")
    st.header("📈 MSE vs Number of Trees Analysis")
    st.write("Analyze how MSE changes with the number of trees to detect overfitting and find optimal tree count.")
    
    # Controls for the analysis
    col1, col2 = st.columns(2)
    with col1:
        max_trees = 80

    if st.button("🔍 Analyze MSE vs Trees", type="secondary"):
        if 'df' not in st.session_state or 'scaler' not in st.session_state:
            st.error("Please train a model first or upload data.")
        else:
            # Run the analysis
            result = create_mse_vs_trees_plot(
                st.session_state.df, 
                st.session_state.scaler,  # Pass the scaler
                random_seed=1471, 
                cpu_threads=1, 
                max_trees=max_trees
            )
            
            if result[0] is not None:
                fig, tree_counts, train_mse, cv_mse = result
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Data download
                st.subheader("💾 Download Analysis Data")
                analysis_df = pd.DataFrame({
                    'Trees': tree_counts,
                    'Training_MSE': train_mse,
                    'CV_MSE': cv_mse
                })
                
                csv_data = analysis_df.to_csv(index=False)
                st.download_button(
                    "📥 Download MSE vs Trees Data",
                    csv_data,
                    "mse_vs_trees_analysis.csv",
                    "text/csv",
                    help="Download the MSE values for different tree counts"
                )

def create_detailed_feature_importance_table(model, feature_names, target_features=None):
    """
    Create a comprehensive feature importance table with gain, cover, frequency, and custom order.
    
    Parameters:
    - model: Trained XGBoost model
    - feature_names: List of feature names used in training
    - target_features: Specific features to include (default: all features)
    
    Returns:
    - DataFrame with detailed importance metrics
    """
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    try:
        # Get the booster from the XGBoost model
        booster = model.get_booster()
        
        # Get importance scores for different metrics
        gain_importance = booster.get_score(importance_type='gain')
        cover_importance = booster.get_score(importance_type='cover')
        weight_importance = booster.get_score(importance_type='weight')  # Number of times feature is used
        
        # If target_features is specified, filter to only those features
        if target_features is None:
            target_features = feature_names
        
        # Define the custom ranking order
        custom_order = ["inflation", "hhis", "hhit", "hhig", "hhic", "ccr", "mcr"]
        
        # Create comprehensive importance table
        importance_data = []
        
        for feature in target_features:
            if feature in feature_names:
                # Get scores (default to 0 if feature not used in any splits)
                gain = gain_importance.get(feature, 0.0)
                cover = cover_importance.get(feature, 0.0)
                frequency = weight_importance.get(feature, 0.0)  # Renamed from weight to frequency
                
                # Custom ranking based on predefined order
                if feature in custom_order:
                    custom_rank = custom_order.index(feature) + 1
                else:
                    custom_rank = len(custom_order) + 1
                
                importance_data.append({
                    'Feature': feature,
                    'Gain': gain,
                    'Cover': cover,
                    'Frequency': frequency,
                    'Custom_Rank': custom_rank
                })
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        if len(importance_df) == 0:
            st.warning("No matching features found in the model.")
            return pd.DataFrame()

        importance_df["Custom_Rank"] = importance_df["Custom_Rank"]-1
        # Sort by custom ranking order
        importance_df = importance_df.sort_values('Custom_Rank', ascending=True).reset_index(drop=True)
        
        # Format the values for better display
        importance_df['Gain'] = importance_df['Gain'].apply(lambda x: f"{x:.6f}")
        importance_df['Cover'] = importance_df['Cover'].apply(lambda x: f"{x:.6f}")
        importance_df['Frequency'] = importance_df['Frequency'].apply(lambda x: f"{x:.0f}")
        
        return importance_df
        
    except Exception as e:
        st.error(f"Error creating detailed importance table: {str(e)}")
        return pd.DataFrame()

def display_detailed_importance_analysis(model, feature_names, target_features=None):
    """
    Display comprehensive feature importance analysis with visualizations.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
    import pandas as pd
    
    # Default target features if not specified
    if target_features is None:
        target_features = ["hhis", "hhig", "hhic", "hhit", "ccr", "mcr"]
    
    st.subheader("🎯 Detailed Feature Importance Analysis")
    st.write(f"Analyzing importance metrics for: {', '.join(target_features)}")
    
    # Create the detailed importance table
    importance_df = create_detailed_feature_importance_table(model, feature_names, target_features)
    
    if len(importance_df) == 0:
        st.warning("No data available for importance analysis.")
        return
    
    # Display the comprehensive table
    st.subheader("📊 Complete Feature Importance Metrics")
    
    # Create formatted display table
    display_df = importance_df.copy()
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", help="Feature name"),
            "Gain": st.column_config.TextColumn("Gain", help="Average gain across all splits using this feature"),
            "Cover": st.column_config.TextColumn("Cover", help="Average coverage across all splits using this feature"),
            "Frequency": st.column_config.TextColumn("Frequency", help="Number of times feature is used in splits"),
            "Custom_Rank": st.column_config.NumberColumn("Rank", help="Custom ranking order: inflation, hhis, hhit, hhig, hhic, ccr, mcr")
        }
    )
    
    return importance_df

# Integration function to add to your main Streamlit app
def add_detailed_importance_to_streamlit(final_model, feature_names):
    """
    Add this to your main Streamlit app after training the model.
    """
    
    # Target features for analysis
    target_features = ["hhis", "hhig", "hhic", "hhit", "ccr", "mcr"]
    
    st.markdown("---")
    
    # Display the detailed analysis
    detailed_df = display_detailed_importance_analysis(
        final_model, 
        feature_names, 
        target_features
    )
    
    return detailed_df

def create_mse_vs_trees_plot(df, scaler, random_seed=1471, cpu_threads=1, max_trees=100):
    """
    Create a line plot showing MSE vs number of trees for training and cross-validation.
    Returns training and CV MSE at different tree counts.
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import mean_squared_error
    import plotly.graph_objects as go
    import streamlit as st
    
    # Required feature set
    feature_cols = ["hhis", "hhit", "ccr", "mcr", "ownership", "inflation", "bank_age"]
    target_col = "fs"
    
    # Verify columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None, None, None
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    if y.isnull().sum() > 0:
        mask = ~y.isnull()
        X = X.loc[mask]
        y = y.loc[mask]
    
    # Apply scaling
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    np.random.seed(random_seed)
    
    # Create stratified classes for CV
    try:
        y_classes = create_target_classes(y, n_classes=3)
    except Exception:
        y_classes = pd.qcut(y.rank(method="first"), q=3, labels=False, duplicates='drop').astype(int)
    
    # XGBoost parameters (adjusted for evaluation)
    param_dict = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.45,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.015,
        'reg_lambda': 1.5,
        'random_state': random_seed,
        'n_jobs': cpu_threads,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    # Initialize CV splitter
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(skf.split(X_scaled, y_classes))
    except Exception:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(kf.split(X_scaled))
    
    # Store MSE values for each tree count
    tree_counts = list(range(1, max_trees + 1, 5))  # Every 5 trees
    cv_mse_by_trees = []
    train_mse_by_trees = []
    
    with st.spinner(f"Evaluating MSE for different tree counts (up to {max_trees} trees)..."):
        progress_bar = st.progress(0)
        
        for i, n_trees in enumerate(tree_counts):
            # Cross-validation MSE
            cv_mse_scores = []
            train_mse_scores = []
            
            for train_idx, test_idx in splits:
                X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model with current number of trees
                model = xgb.XGBRegressor(n_estimators=n_trees, **param_dict)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                
                # Calculate MSE
                cv_mse = mean_squared_error(y_test, y_pred_test)
                train_mse = mean_squared_error(y_train, y_pred_train)
                
                cv_mse_scores.append(cv_mse)
                train_mse_scores.append(train_mse)
            
            # Store average MSE across folds
            cv_mse_by_trees.append(np.mean(cv_mse_scores))
            train_mse_by_trees.append(np.mean(train_mse_scores))
            
            # Update progress
            progress_bar.progress((i + 1) / len(tree_counts))
        
        progress_bar.empty()
    
    # Create the plot
    fig = go.Figure()
    
    # Add training MSE line (blue)
    fig.add_trace(go.Scatter(
        x=tree_counts,
        y=train_mse_by_trees,
        mode='lines+markers',
        name='Training MSE',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue')
    ))
    
    # Add cross-validation MSE line (orange)
    fig.add_trace(go.Scatter(
        x=tree_counts,
        y=cv_mse_by_trees,
        mode='lines+markers',
        name='Cross-Validation MSE',
        line=dict(color='orange', width=2),
        marker=dict(size=6, color='orange')
    ))
    
    # Update layout
    fig.update_layout(
        title='MSE vs Number of Trees (Training vs Cross-Validation)',
        xaxis_title='Number of Trees',
        yaxis_title='Mean Squared Error (MSE)',
        width=800,
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig, tree_counts, train_mse_by_trees, cv_mse_by_trees

def create_correlation_heatmap(df):
    """Create correlation matrix heatmap for all variables"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    st.subheader("🔗 Variable Correlation Matrix")
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-relevant columns if they exist
    exclude_cols = ['bank_no', 'year'] if any(col in numeric_cols for col in ['bank_no', 'year']) else []
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric variables to create correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle (to show only lower triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Prepare data for plotly heatmap
    corr_values = corr_matrix.values
    # Apply mask to correlation values
    corr_values_masked = np.where(mask, np.nan, corr_values)
    
    fig_plotly = go.Figure(data=go.Heatmap(
        z=corr_values_masked,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_values_masked, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        showscale=True
    ))
    
    fig_plotly.update_layout(
        title='Interactive Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables',
        height=600,
        width=800
    )
    
    st.plotly_chart(fig_plotly, use_container_width=True)
    
    # Show strongest correlations
    st.subheader("📊 Strongest Correlations")
    
    # Get correlation pairs (excluding self-correlations)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j],
                'Abs Correlation': abs(corr_matrix.iloc[i, j])
            })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(corr_pairs).sort_values('Abs Correlation', ascending=False)
    
    # Display top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strongest Positive Correlations:**")
        positive_corr = corr_df[corr_df['Correlation'] > 0].head(10)
        positive_display = positive_corr[['Variable 1', 'Variable 2', 'Correlation']].copy()
        positive_display['Correlation'] = positive_display['Correlation'].apply(lambda x: f"{x:.3f}")
        st.dataframe(positive_display, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Strongest Negative Correlations:**")
        negative_corr = corr_df[corr_df['Correlation'] < 0].head(10)
        negative_display = negative_corr[['Variable 1', 'Variable 2', 'Correlation']].copy()
        negative_display['Correlation'] = negative_display['Correlation'].apply(lambda x: f"{x:.3f}")
        st.dataframe(negative_display, hide_index=True, use_container_width=True)
    
    # Download correlation matrix
    csv_corr = corr_matrix.to_csv()
    st.download_button(
        "📥 Download Correlation Matrix",
        csv_corr,
        "correlation_matrix.csv",
        "text/csv",
        help="Download the complete correlation matrix as CSV"
    )
    
    return corr_matrix

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
    Returns: fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_shap, scaler
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

    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    

    # X_all for final feature importance. keep only existing columns
    all_features = ["inflation", "hhis", "bank_age", "hhit", "hhic", "asset_size", "ownership", "ccr", "mcr", "hhig"]
    all_features = [c for c in all_features if c in df.columns]
    X_all = df[all_features].copy()
    if X_all.isnull().sum().sum() > 0:
        X_all = X_all.fillna(X_all.median())
    
    # Create separate scaler for all features
    scaler_all = MinMaxScaler()
    X_all_scaled = pd.DataFrame(
        scaler_all.fit_transform(X_all), 
        columns=X_all.columns, 
        index=X_all.index
    )

    # XGBoost params tuned for small/container environments
    param_dict = {
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
        splits = list(skf.split(X_scaled, y_classes))
    except Exception:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(kf.split(X_scaled))

    fold_results = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mse_scores = []
    mape_scores = []

    total_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        t0 = time.perf_counter()

        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        xgb_model = xgb.XGBRegressor(n_estimators=55, **param_dict)

        # fit with scaled data
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # collect fold-level metrics
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

        # For fold-wise rows keep 'Std' blank
        fold_results.append({
            'Fold': fold_num,
            'R²': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MSE': float(mse),
            'MAPE (%)': float(mape),
            'Std': ""   # empty for individual folds
        })

    total_time = time.perf_counter() - total_start

    # Training model on scaled feature_cols for predictions
    training_model = xgb.XGBRegressor(n_estimators=55, **param_dict)
    training_model.fit(X_scaled, y)

    # final model on X_all_scaled for feature importance and SHAP
    final_idx = y.index
    X_all_for_fit = X_all_scaled.loc[final_idx] if not X_all_scaled.empty else X_scaled.loc[final_idx]
    final_model = xgb.XGBRegressor(n_estimators=55, **param_dict)
    final_model.fit(X_all_for_fit, y)

    # feature importance
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

    # compute means
    r2_mean = float(np.mean(r2_scores))
    rmse_mean = float(np.mean(rmse_scores))
    mae_mean = float(np.mean(mae_scores))
    mse_mean = float(np.mean(mse_scores))
    mape_mean = float(np.mean(mape_scores))

    # Compute std of R² across folds
    if len(r2_scores) > 1:
        r2_std = float(np.std(r2_scores, ddof=1))
    else:
        r2_std = 0.0

    # Append mean row
    fold_results.append({
        'Fold': 'Mean',
        'R²': r2_mean,
        'RMSE': rmse_mean,
        'MAE': mae_mean,
        'MSE': mse_mean,
        'MAPE (%)': mape_mean,
        'Std': float(r2_std)
    })

    # Include R² std in summary_stats
    summary_stats = {
        'CV R² Mean': r2_mean,
        'CV R² Std': float(r2_std),
        'CV RMSE Mean': rmse_mean,
        'CV MAE Mean': mae_mean,
        'CV MSE Mean': mse_mean,
        'CV MAPE Mean': mape_mean,
    }

    return fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_fit, scaler

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

def create_prediction_interface(training_model, scaler, df, feature_cols):
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
                
                if col == "asset_size":
                    max_val = 100000.0

                if col == "bank_age":
                    max_val = 300.0
#                 st.write(np.maximum(max_val,1))
                user_inputs[col] = st.number_input(
                    f"{col}",
                    min_value=np.minimum(min_val,0),
                    max_value=np.maximum(max_val,1),
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range: {min_val:.3f} to {max_val:.3f}"
                )
            else:
                user_inputs[col] = st.number_input(f"{col}", value=0.0)
    
    # Show scaling information
    st.write("**Scaling Information:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original input values will be scaled to [0,1] range")
    with col2:
        st.write("Model was trained on scaled features")
    
    # Predict button
    if st.button("🚀 Predict FS Score", type="primary"):
        try:
            # Create input dataframe with only the training features
            input_df = pd.DataFrame([user_inputs])
            prediction_input = input_df[feature_cols]
            
            # Apply MinMax scaling to user input
            prediction_input_scaled = pd.DataFrame(
                scaler.transform(prediction_input),
                columns=prediction_input.columns
            )
            
            # Show before and after scaling
            with st.expander("📊 View Scaling Applied to Your Input"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Values:**")
                    st.dataframe(prediction_input.T, use_container_width=True, column_config={0: "Value"})
                with col2:
                    st.write("**Scaled Values [0,1]:**")
                    st.dataframe(prediction_input_scaled.T, use_container_width=True, column_config={0: "Scaled Value"})
            
            # Make prediction using scaled input
            prediction = training_model.predict(prediction_input_scaled)[0]
            
            # Display prediction
            st.success(f"**Predicted Financial Soundness (FS) Score: {prediction:.4f}**")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure all required features are provided")

def create_csv_predictions(training_model, scaler, df, feature_cols):
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
    
    # Apply MinMax scaling
    prediction_input_scaled = pd.DataFrame(
        scaler.transform(prediction_input),
        columns=prediction_input.columns,
        index=prediction_input.index
    )
    
    
    # Make predictions
    predictions = training_model.predict(prediction_input_scaled)
    
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
    
    # Add scaling information in sidebar
    st.sidebar.markdown("---")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            
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
            
            # ADD SEED OPTIMIZATION INTERFACE
            add_seed_optimization_interface()
            
            # Training section
            st.markdown("---")
            st.header("Model Training")
            
            # Use best seed if available, otherwise use default
            if 'best_seed' in st.session_state:
                default_seed = st.session_state.best_seed
                st.info(f"Using optimized seed: {default_seed}")
            else:
                default_seed = 1471
            
            # Seed input for manual training
            manual_seed = st.number_input("Manual Random Seed", min_value=1, max_value=10000, value=default_seed, step=1)
            
            # Train button
            if st.button("🚀 Train XGBoost Model", type="primary"):
                with st.spinner("Training XGBoost model"):
                    results = train_xgboost_model(df, random_seed=manual_seed)
                    
                    if results is not None:
                        fold_results, summary_stats, feature_importance, final_model, training_model, X_all_for_shap, scaler = results
                        
                        # Store models and scaler in session state for later use
                        st.session_state.training_model = training_model
                        st.session_state.final_model = final_model
                        st.session_state.X_all_for_shap = X_all_for_shap
                        st.session_state.scaler = scaler  # Store the scaler
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
                            # show R² standard deviation here
                            st.metric("R² Std", f"{summary_stats['CV R² Std']:.6f}")
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
                        
                        st.markdown("---")

                        # Detailed Feature Importance Analysis for Specific Features
                        target_features = ["hhis", "hhig", "hhic", "hhit", "ccr", "mcr"]
                        final_feature_names = X_all_for_shap.columns.tolist()

                        detailed_importance_df = display_detailed_importance_analysis(
                            final_model, 
                            final_feature_names, 
                            target_features
                        )

                        # Store in session state
                        st.session_state.detailed_importance = detailed_importance_df

                        # ADD THE CORRELATION HEATMAP HERE:
                        st.markdown("---")
                        create_correlation_heatmap(df)
                        
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
                        
                        fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics Across CV Folds")
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

            add_mse_trees_analysis_to_main()

            # Prediction interface (only show if model is trained)
            if 'training_model' in st.session_state and 'scaler' in st.session_state:
                st.markdown("---")
                create_prediction_interface(
                    st.session_state.training_model,
                    st.session_state.scaler,  # Pass the scaler
                    st.session_state.df,
                    st.session_state.feature_cols
                )
                
                st.markdown("---")
                create_csv_predictions(
                    st.session_state.training_model,
                    st.session_state.scaler,  # Pass the scaler
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
        
        ### 🎲 NEW: Random Seed Optimization
        
        **✅ What's New:**
        - **Automatic Seed Testing:** Test multiple random seeds to find the best performing one
        - **CV R² Optimization:** Seeds are ranked by their cross-validation R² mean performance
        - **Visual Performance Analysis:** See how different seeds perform with interactive plots
        - **Best Seed Integration:** Automatically use the best seed for subsequent training
        - **Comprehensive Results:** Download complete seed optimization results
        
        **🎯 How It Works:**
        1. **Set Seed Range:** Choose start and end seed values (e.g., 1 to 100)
        2. **Automatic Testing:** Each seed is tested with 5-fold cross-validation
        3. **Performance Ranking:** Seeds ranked by CV R² mean (higher is better)
        4. **Best Seed Selection:** Top performing seed is highlighted and recommended
        5. **One-Click Training:** Train your final model with the optimal seed
        
        **📊 Results Provided:**
        - Top N best performing seeds with detailed metrics
        - CV R² mean, standard deviation, RMSE, MAE, MSE for each seed
        - Interactive visualization showing seed performance trends
        - Downloadable results for further analysis
        
        ### 🔧 MinMaxScaler Integration
        
        **✅ What's New:**
        - **Automatic Feature Scaling:** All features are automatically scaled to [0,1] range
        - **User Input Scaling:** When you make predictions, your input values are automatically scaled using the same scaler
        - **Consistent Scaling:** Both training and prediction use the same scaling parameters
        - **Visual Feedback:** See before/after scaling for your inputs
        
        **🎯 Benefits:**
        - **Improved Model Performance:** Prevents features with larger scales from dominating
        - **Better Convergence:** Helps XGBoost converge faster and more reliably  
        - **Consistent Predictions:** User inputs are scaled the same way as training data
        - **Robust to Scale Differences:** Works well even with features of very different magnitudes
        
        **🔍 SHAP Analysis:**
        - Get detailed model interpretability insights
        - Understand how each feature contributes to predictions
        - Install SHAP with: `pip install shap`
        
        **🎯 Individual Predictions:**
        - Input custom values for all columns
        - Values automatically scaled before prediction
        - See scaling transformation applied to your inputs
        - Get instant FS score predictions
        
        **📊 Dataset Predictions:**
        - Generate predictions for entire dataset with automatic scaling
        - Compare actual vs predicted FS values
        - Download results with difference calculations
        
        ### 🔧 Hyperparameters (Optimized for Small Datasets)
        
        - **`n_estimators: 55`** - Number of boosting rounds
        - **`max_depth: 5`** - Maximum tree depth  
        - **`learning_rate: 0.1`** - Step size shrinkage
        - **`subsample: 0.45`** - Fraction of samples used per tree
        - **`colsample_bytree: 0.5`** - Fraction of features used per tree
        - **`reg_alpha: 0.015`** - L1 regularization
        - **`reg_lambda: 1.5`** - L2 regularization
        - **`tree_method: 'hist'`** - Histogram-based algorithm for faster CPU training
        
        ### ⚙️ Model Configuration
        - **Algorithm:** XGBoost Regressor with MinMaxScaler
        - **Cross-Validation:** 5-Fold Stratified  
        - **Feature Scaling:** MinMaxScaler [0,1] applied to all features
        - **Feature Importance:** Calculated on scaled features
        - **SHAP Analysis:** Uses scaled features for interpretation
        - **Predictions:** Automatic scaling applied to user inputs
        - **Seed Optimization:** Find best random seed for maximum performance
        """)

if __name__ == "__main__":
    main()
