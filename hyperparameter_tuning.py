import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the data preprocessing from our improved model
from improved_model import load_and_preprocess_data, create_features

def get_important_features(model, X_train, model_name, threshold='mean'):
    """
    Get important features based on the model's feature importance scores.
    threshold can be 'mean' or a float between 0 and 1
    """
    importances = model.best_estimator_.feature_importances_
    
    # Create DataFrame of features and their importance scores
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    if threshold == 'mean':
        threshold = feature_imp['Importance'].mean()
    
    # Select features above threshold
    important_features = feature_imp[feature_imp['Importance'] > threshold]['Feature'].tolist()
    
    print(f"\nSelected {len(important_features)} important features for {model_name}:")
    for idx, feature in enumerate(important_features, 1):
        importance = feature_imp[feature_imp['Feature'] == feature]['Importance'].values[0]
        print(f"{idx}. {feature}: {importance:.4f}")
    
    return important_features

def plot_feature_importance(model, X_train, model_name):
    # Set figure style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    if model_name == "Random Forest":
        importances = model.best_estimator_.feature_importances_
    else:  # XGBoost
        importances = model.best_estimator_.feature_importances_
    
    # Get feature names and importances
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    sns.barplot(data=feature_imp, x='Importance', y='Feature')
    plt.title(f'Feature Importance - {model_name}', pad=20)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return feature_imp

def plot_cv_results(model, model_name):
    # Set figure style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    # Get cross-validation results
    cv_results = pd.DataFrame(model.cv_results_)
    
    # Calculate mean RMSE for each parameter combination
    rmse_scores = -cv_results['mean_test_score']
    
    # Create violin plot
    sns.violinplot(y=rmse_scores, color='lightblue')
    plt.title(f'Cross-validation RMSE Distribution - {model_name}', pad=20)
    plt.ylabel('RMSE')
    
    # Add mean line
    plt.axhline(y=rmse_scores.mean(), color='red', linestyle='--', 
                label=f'Mean RMSE: {rmse_scores.mean():.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def tune_random_forest():
    print("Tuning Random Forest...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = create_features(df)
    
    # Initial fit to get feature importance
    init_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    init_rf_grid = GridSearchCV(
        estimator=init_rf,
        param_grid={'max_depth': [None]},  # Simple grid for initial fit
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    init_rf_grid.fit(X_train, y_train)
    
    # Get important features
    important_features = get_important_features(init_rf_grid, X_train, "Random Forest")
    
    # Use only important features
    X_train = X_train[important_features]
    X_test = X_test[important_features]
    
    # Define the parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 20]
    }
    
    # Create base model
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Create GridSearchCV object
    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the grid search
    rf_grid.fit(X_train, y_train)
    
    print("\nBest Random Forest Parameters:")
    print(rf_grid.best_params_)
    print(f"Best RMSE: {-rf_grid.best_score_:,.2f}")
    
    # Plot feature importance and CV results
    plot_feature_importance(rf_grid, X_train, "Random Forest")
    plot_cv_results(rf_grid, "Random Forest")
    
    return rf_grid.best_params_, important_features

def tune_xgboost():
    print("\nTuning XGBoost...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = create_features(df)
    
    # Initial fit to get feature importance
    init_xgb = xgb.XGBRegressor(random_state=42)
    init_xgb_grid = GridSearchCV(
        estimator=init_xgb,
        param_grid={'max_depth': [3]},  # Simple grid for initial fit
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    init_xgb_grid.fit(X_train, y_train)
    
    # Get important features
    important_features = get_important_features(init_xgb_grid, X_train, "XGBoost")
    
    # Use only important features
    X_train = X_train[important_features]
    X_test = X_test[important_features]
    
    # Define the parameter grid for XGBoost
    xgb_param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 500],
        'subsample': [0.8, 1.0]
    }
    
    # Create base model
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    # Create GridSearchCV object
    xgb_grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the grid search
    xgb_grid.fit(X_train, y_train)
    
    print("\nBest XGBoost Parameters:")
    print(xgb_grid.best_params_)
    print(f"Best RMSE: {-xgb_grid.best_score_:,.2f}")
    
    # Plot feature importance and CV results
    plot_feature_importance(xgb_grid, X_train, "XGBoost")
    plot_cv_results(xgb_grid, "XGBoost")
    
    return xgb_grid.best_params_, important_features

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    
    # Tune Random Forest
    rf_best_params, rf_features = tune_random_forest()
    
    # Tune XGBoost
    xgb_best_params, xgb_features = tune_xgboost()
    
    print("\nRandom Forest Best Parameters:")
    print(rf_best_params)
    print("\nRandom Forest Important Features:")
    print(rf_features)
    
    print("\nXGBoost Best Parameters:")
    print(xgb_best_params)
    print("\nXGBoost Important Features:")
    print(xgb_features)
    
    print("\nTuning completed. You can now use these parameters and features in improved_model.py") 