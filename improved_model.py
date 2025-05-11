import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_csv('train.csv')
    df.dropna(subset=['value_eur'], inplace=True)
    
    # Drop irrelevant columns
    cols_to_drop = ['Unnamed: 0', 'id', 'short_name', "long_name", "dob", "club_jersey_number", 
                    "nation_jersey_number", "club_name", "club_loaned_from", "nation_position", 
                    "player_traits", "player_tags", "nationality_name"]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Create interaction features
    df['overall_potential'] = df['overall'] * df['potential']
    df['overall_age_ratio'] = df['overall'] / df['age']
    df['wage_value_ratio'] = df['wage_eur'] / df['value_eur'].clip(lower=1)
    df['release_value_ratio'] = df['release_clause_eur'] / df['value_eur'].clip(lower=1)
    
    # Create skill aggregates with weights
    attacking_cols = [col for col in df.columns if col.startswith('attacking_')]
    skill_cols = [col for col in df.columns if col.startswith('skill_')]
    movement_cols = [col for col in df.columns if col.startswith('movement_')]
    power_cols = [col for col in df.columns if col.startswith('power_')]
    mentality_cols = [col for col in df.columns if col.startswith('mentality_')]
    goalkeeping_cols = [col for col in df.columns if col.startswith('goalkeeping_')]
    
    # Weight the aggregates by the player's position
    df['is_forward'] = df['club_position'].isin(['ST', 'CF', 'LW', 'RW'])
    df['is_midfielder'] = df['club_position'].isin(['CAM', 'CM', 'CDM', 'LM', 'RM'])
    df['is_defender'] = df['club_position'].isin(['CB', 'LB', 'RB', 'LWB', 'RWB'])
    df['is_goalkeeper'] = df['club_position'] == 'GK'
    
    # Create weighted skill scores
    df['total_attacking'] = df[attacking_cols].mean(axis=1) * (df['is_forward'] * 1.5 + df['is_midfielder'] * 1.2 + 1)
    df['total_skill'] = df[skill_cols].mean(axis=1) * (df['is_forward'] * 1.3 + df['is_midfielder'] * 1.4 + 1)
    df['total_movement'] = df[movement_cols].mean(axis=1) * (df['is_forward'] * 1.4 + df['is_midfielder'] * 1.3 + 1)
    df['total_power'] = df[power_cols].mean(axis=1) * (df['is_defender'] * 1.3 + 1)
    df['total_mentality'] = df[mentality_cols].mean(axis=1)
    df['total_goalkeeping'] = df[goalkeeping_cols].mean(axis=1) * (df['is_goalkeeper'] * 2 + 1)
    
    # Drop position indicator columns
    df.drop(columns=['is_forward', 'is_midfielder', 'is_defender', 'is_goalkeeper'], inplace=True)
    
    return df

def get_feature_groups(df):
    # Identify different types of features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from features
    if 'value_eur' in numeric_features:
        numeric_features.remove('value_eur')
    
    return numeric_features, categorical_features

def create_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_features(df, X_train=None, X_test=None, first_time=True):
    if first_time:
        # Split features and target
        y = df['value_eur']
        X = df.drop('value_eur', axis=1)
        
        # Get feature groups
        numeric_features, categorical_features = get_feature_groups(X)
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit and transform the data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        numeric_features_out = numeric_features
        categorical_features_out = []
        if categorical_features:
            categorical_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        # Combine all feature names
        feature_names = np.concatenate([numeric_features_out, categorical_features_out])
        
        # Convert to DataFrame
        X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        return X_train, X_test, y_train, y_test, preprocessor
    else:
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        return X_train_transformed, X_test_transformed

def train_models(X_train, X_test, y_train, y_test):
    # Train XGBoost with tuned parameters
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.05,  # Lower learning rate for better generalization
        max_depth=4,         # Reduced to prevent overfitting
        n_estimators=1000,   # Increased due to lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # Increased to reduce overfitting
        gamma=0.1,          # Added to control regularization
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=1.0,     # L2 regularization
        random_state=42
    )
    
    # Train Random Forest with tuned parameters
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    # Create ensemble model
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model)
        ],
        weights=[0.6, 0.4]  # Give more weight to XGBoost based on CV performance
    )
    
    # Perform cross-validation for each model
    print("\nPerforming cross-validation...")
    for name, model in [('XGBoost', xgb_model), ('Random Forest', rf_model)]:
        cv_rmse = -cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='neg_root_mean_squared_error')
        cv_mape = -cross_val_score(model, X_train, y_train,
                                cv=5, scoring='neg_mean_absolute_percentage_error')
        print(f"\n{name}:")
        print(f"CV RMSE: {cv_rmse.mean():,.2f} (+/- {cv_rmse.std() * 2:,.2f})")
        print(f"CV MAPE: {cv_mape.mean():,.2f}% (+/- {cv_mape.std() * 2:,.2f}%)")
    
    # Train final models
    print("\nTraining final models...")
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    train_preds = ensemble.predict(X_train)
    test_preds = ensemble.predict(X_test)
    
    # Calculate metrics
    print("\nFinal Results (Ensemble):")
    print(f"Train RMSE: {root_mean_squared_error(y_train, train_preds):,.2f}")
    print(f"Test RMSE: {root_mean_squared_error(y_test, test_preds):,.2f}")
    print(f"Train MAPE: {mean_absolute_percentage_error(y_train, train_preds):,.2f}%")
    print(f"Test MAPE: {mean_absolute_percentage_error(y_test, test_preds):,.2f}%")
    
    return ensemble

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Create features and split data
    print("\nCreating features and splitting data...")
    X_train, X_test, y_train, y_test, preprocessor = create_features(df)
    
    # Train models
    print("\nTraining models...")
    model = train_models(X_train, X_test, y_train, y_test) 