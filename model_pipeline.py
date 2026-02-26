import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import os

def main():
    print("Loading data...")
    # Load data
    train_features = pd.read_csv('train_hh_features.csv')
    train_gt = pd.read_csv('train_hh_gt.csv')
    train_rates = pd.read_csv('train_rates_gt.csv')
    test_features = pd.read_csv('test_hh_features.csv')

    # Merge features with ground truth on survey_id and hhid
    train_df = pd.merge(train_features, train_gt, on=['survey_id', 'hhid'])
    
    # Identify target and weight
    target_col = 'cons_ppp17'
    weight_col = 'weight'
    
    # Exclude ID columns, target, and weight from features
    exclude_cols = ['survey_id', 'hhid', target_col] # Keep weight in train_df but not as a feature
    feature_cols = [c for c in train_features.columns if c not in exclude_cols and c != weight_col]
    
    # Preprocessing: Handle categorical variables
    # XGBoost can optionally handle categoricals, or we can one-hot encode.
    # We will let XGBoost handle it via enable_categorical=True and category dtype
    categorical_features = []
    for col in feature_cols:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].astype('category')
            test_features[col] = test_features[col].astype('category')
            categorical_features.append(col)
            
    print(f"Features count: {len(feature_cols)}, Categorical count: {len(categorical_features)}")
    
    # Cross Validation Strategy
    # We use GroupKFold based on survey_id to measure out-of-survey generalization
    gkf = GroupKFold(n_splits=3)
    
    oof_preds = np.zeros(len(train_df))
    
    print("Training models...")
    models = []
    
    # Basic XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'n_estimators': 500,
        'random_state': 42,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df['survey_id'])):
        print(f"--- Fold {fold+1} ---")
        X_tr, y_tr, w_tr = train_df.iloc[train_idx][feature_cols], train_df.iloc[train_idx][target_col], train_df.iloc[train_idx][weight_col]
        X_va, y_va, w_va = train_df.iloc[val_idx][feature_cols], train_df.iloc[val_idx][target_col], train_df.iloc[val_idx][weight_col]
        
        # We can pass sample_weight to XGBoost
        model = XGBRegressor(**xgb_params)
        
        # Fit with weights
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[w_va],
            verbose=False
        )
        
        oof_preds[val_idx] = model.predict(X_va)
        models.append(model)
        
        rmse = root_mean_squared_error(y_va, oof_preds[val_idx])
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
    print(f"Overall OOF RMSE: {root_mean_squared_error(train_df[target_col], oof_preds):.4f}")
    
    # Calculate Poverty Rates for OOF
    # Extract threshold columns from train_rates
    rate_cols = [c for c in train_rates.columns if c.startswith('pct_hh_below_')]
    thresholds = [float(c.split('_')[-1]) for c in rate_cols]
    
    print("\nPredicting on Test Set...")
    # Inference on Test data
    test_X = test_features[feature_cols]
    test_preds = np.zeros(len(test_features))
    
    for model in models:
        test_preds += model.predict(test_X) / len(models)
        
    test_features['cons_ppp17'] = test_preds
    
    # Save predicted household consumption
    pred_consumption = test_features[['survey_id', 'hhid', 'cons_ppp17']]
    pred_consumption.to_csv('predicted_household_consumption.csv', index=False)
    print("Saved predicted_household_consumption.csv")
    
    # Calculate Test Poverty Distribution
    test_poverty = []
    
    for survey_id, group in test_features.groupby('survey_id'):
        survey_rates = {'survey_id': survey_id}
        weights = group['weight'].values
        preds = group['cons_ppp17'].values
        
        total_weight = weights.sum()
        for t_name, t_val in zip(rate_cols, thresholds):
            is_poor = (preds < t_val).astype(int)
            poor_rate = (is_poor * weights).sum() / total_weight
            survey_rates[t_name] = poor_rate
            
        test_poverty.append(survey_rates)
        
    pred_poverty_df = pd.DataFrame(test_poverty)
    # Ensure correct column order
    pred_poverty_df = pred_poverty_df[['survey_id'] + rate_cols]
    pred_poverty_df.to_csv('predicted_poverty_distribution.csv', index=False)
    print("Saved predicted_poverty_distribution.csv")
    print("Done!")

if __name__ == '__main__':
    main()
