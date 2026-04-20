import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df, feature_cols, is_train=True, survey_stats=None):
    """Create additional features to improve model accuracy."""
    df = df.copy()
    
    # Identify numeric columns from the feature set
    numeric_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    # 1. Missing value count per row (captures data quality signal)
    df['num_missing'] = df[feature_cols].isnull().sum(axis=1)
    
    # 2. Per-survey statistics for numeric features (captures survey-level patterns)
    if is_train:
        survey_stats = {}
        for col in numeric_cols[:15]:  # Top 15 numeric features to avoid explosion
            stats = df.groupby('survey_id')[col].agg(['mean', 'std']).to_dict()
            survey_stats[col] = stats
            df[f'{col}_survey_mean'] = df['survey_id'].map(stats['mean'])
            df[f'{col}_survey_std'] = df['survey_id'].map(stats['std'])
            # Deviation from survey mean
            df[f'{col}_dev_from_survey'] = df[col] - df[f'{col}_survey_mean']
    else:
        for col in numeric_cols[:15]:
            if col in survey_stats:
                stats = survey_stats[col]
                df[f'{col}_survey_mean'] = df['survey_id'].map(stats['mean']).fillna(0)
                df[f'{col}_survey_std'] = df['survey_id'].map(stats['std']).fillna(0)
                df[f'{col}_dev_from_survey'] = df[col] - df[f'{col}_survey_mean']
    
    # 3. Numeric pairwise ratios between top correlated features
    if len(numeric_cols) >= 2:
        for i in range(min(5, len(numeric_cols))):
            for j in range(i + 1, min(5, len(numeric_cols))):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                df[f'ratio_{c1}_{c2}'] = df[c1] / (df[c2] + 1e-8)
    
    # 4. Row-level numeric aggregations
    if numeric_cols:
        df['row_numeric_mean'] = df[numeric_cols].mean(axis=1)
        df['row_numeric_std'] = df[numeric_cols].std(axis=1)
        df['row_numeric_max'] = df[numeric_cols].max(axis=1)
        df['row_numeric_min'] = df[numeric_cols].min(axis=1)
        df['row_numeric_range'] = df['row_numeric_max'] - df['row_numeric_min']
    
    return df, survey_stats


def main():
    print("Loading data...")
    train_features = pd.read_csv('train_hh_features.csv')
    train_gt = pd.read_csv('train_hh_gt.csv')
    train_rates = pd.read_csv('train_rates_gt.csv')
    test_features = pd.read_csv('test_hh_features.csv')

    # Merge features with ground truth
    train_df = pd.merge(train_features, train_gt, on=['survey_id', 'hhid'])
    
    target_col = 'cons_ppp17'
    weight_col = 'weight'
    
    exclude_cols = ['survey_id', 'hhid', target_col]
    base_feature_cols = [c for c in train_features.columns if c not in exclude_cols and c != weight_col]
    
    # Handle categorical variables
    categorical_features = []
    for col in base_feature_cols:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].astype('category')
            test_features[col] = test_features[col].astype('category')
            categorical_features.append(col)
    
    # ====== KEY IMPROVEMENT 1: Log-transform the target ======
    # Consumption data is heavily right-skewed; log-transform stabilizes variance
    # and dramatically improves model performance
    train_df['log_target'] = np.log1p(train_df[target_col].clip(lower=0.01))
    log_target_col = 'log_target'
    print(f"Target stats — Mean: {train_df[target_col].mean():.2f}, "
          f"Median: {train_df[target_col].median():.2f}, "
          f"Skew: {train_df[target_col].skew():.2f}")
    print(f"Log-target stats — Mean: {train_df[log_target_col].mean():.2f}, "
          f"Skew: {train_df[log_target_col].skew():.2f}")
    
    # ====== KEY IMPROVEMENT 2: Feature Engineering ======
    print("Engineering features...")
    train_df, survey_stats = engineer_features(train_df, base_feature_cols, is_train=True)
    test_features, _ = engineer_features(test_features, base_feature_cols, is_train=False, survey_stats=survey_stats)
    
    # Update feature columns to include engineered features
    new_cols = [c for c in train_df.columns 
                if c not in ['survey_id', 'hhid', target_col, weight_col, log_target_col]
                and c in test_features.columns]
    feature_cols = new_cols
    
    print(f"Total features: {len(feature_cols)} (base: {len(base_feature_cols)}, "
          f"engineered: {len(feature_cols) - len(base_feature_cols)}, "
          f"categorical: {len(categorical_features)})")
    
    # Cross Validation with GroupKFold
    gkf = GroupKFold(n_splits=3)
    
    oof_preds_log = np.zeros(len(train_df))
    
    print("\nTraining models...")
    models = []
    fold_metrics = []
    
    # ====== KEY IMPROVEMENT 3: Better Hyperparameters ======
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.03,        # Lower LR for better convergence
        'n_estimators': 2000,          # More trees (early stopping will pick best)
        'max_depth': 8,                # Deeper trees to capture complex patterns
        'min_child_weight': 5,         # Regularization
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'gamma': 0.1,                  # Min loss reduction for split
        'random_state': 42,
        'colsample_bytree': 0.7,       # Feature sampling per tree
        'colsample_bylevel': 0.7,      # Feature sampling per level
        'subsample': 0.8,              # Row sampling
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 100   # More patience
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df['survey_id'])):
        print(f"\n--- Fold {fold+1} ---")
        X_tr = train_df.iloc[train_idx][feature_cols]
        y_tr = train_df.iloc[train_idx][log_target_col]  # Train on log-target
        w_tr = train_df.iloc[train_idx][weight_col]
        
        X_va = train_df.iloc[val_idx][feature_cols]
        y_va_log = train_df.iloc[val_idx][log_target_col]
        y_va_orig = train_df.iloc[val_idx][target_col]  # Original scale for eval
        w_va = train_df.iloc[val_idx][weight_col]
        
        model = XGBRegressor(**xgb_params)
        
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va_log)],
            sample_weight_eval_set=[w_va],
            verbose=False
        )
        
        # Predict in log space, then inverse-transform
        val_preds_log = model.predict(X_va)
        val_preds_orig = np.expm1(val_preds_log)  # Inverse of log1p
        val_preds_orig = np.clip(val_preds_orig, 0, None)  # No negative consumption
        
        oof_preds_log[val_idx] = val_preds_log
        models.append(model)
        
        # Evaluate on ORIGINAL scale (what matters for poverty rates)
        fold_rmse = root_mean_squared_error(y_va_orig, val_preds_orig)
        fold_r2 = r2_score(y_va_orig, val_preds_orig)
        fold_mae = mean_absolute_error(y_va_orig, val_preds_orig)
        
        # Also report log-scale R² (the model's native performance)
        fold_r2_log = r2_score(y_va_log, val_preds_log)
        
        fold_metrics.append({
            'fold': fold + 1,
            'rmse': round(float(fold_rmse), 4),
            'r2_score': round(float(fold_r2), 4),
            'r2_score_log': round(float(fold_r2_log), 4),
            'mae': round(float(fold_mae), 4)
        })
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f} | R²: {fold_r2:.4f} | "
              f"R²(log): {fold_r2_log:.4f} | MAE: {fold_mae:.4f}")
        print(f"  Best iteration: {model.best_iteration}")
    
    # Overall metrics on original scale
    oof_preds_orig = np.expm1(oof_preds_log)
    oof_preds_orig = np.clip(oof_preds_orig, 0, None)
    
    overall_rmse = root_mean_squared_error(train_df[target_col], oof_preds_orig)
    overall_r2 = r2_score(train_df[target_col], oof_preds_orig)
    overall_mae = mean_absolute_error(train_df[target_col], oof_preds_orig)
    overall_r2_log = r2_score(train_df[log_target_col], oof_preds_log)
    
    print(f"\n{'='*60}")
    print(f"Overall OOF RMSE: {overall_rmse:.4f} | R²: {overall_r2:.4f} | "
          f"R²(log): {overall_r2_log:.4f} | MAE: {overall_mae:.4f}")
    print(f"{'='*60}")
    
    # Save metrics to JSON for the dashboard
    metrics = {
        'fold_metrics': fold_metrics,
        'overall': {
            'rmse': round(float(overall_rmse), 4),
            'r2_score': round(float(overall_r2), 4),
            'r2_score_log': round(float(overall_r2_log), 4),
            'mae': round(float(overall_mae), 4),
            'num_folds': len(fold_metrics),
            'num_training_samples': len(train_df),
            'num_features': len(feature_cols)
        }
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved model_metrics.json")
    
    # Extract threshold columns from train_rates
    rate_cols = [c for c in train_rates.columns if c.startswith('pct_hh_below_')]
    thresholds = [float(c.split('_')[-1]) for c in rate_cols]
    
    print("\nPredicting on Test Set...")
    test_X = test_features[feature_cols]
    test_preds_log = np.zeros(len(test_features))
    
    for model in models:
        test_preds_log += model.predict(test_X) / len(models)
    
    # Inverse-transform from log space
    test_preds = np.expm1(test_preds_log)
    test_preds = np.clip(test_preds, 0, None)
    
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
    pred_poverty_df = pred_poverty_df[['survey_id'] + rate_cols]
    pred_poverty_df.to_csv('predicted_poverty_distribution.csv', index=False)
    print("Saved predicted_poverty_distribution.csv")
    print("Done!")

if __name__ == '__main__':
    main()
