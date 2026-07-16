import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df, feature_cols):
    """Create additional features to improve model accuracy."""
    df = df.copy()
    
    # 1. Missing value count per row (captures data quality signal)
    df['num_missing'] = df[feature_cols].isnull().sum(axis=1)
    
    # Identify numeric columns from the feature set (excluding categorical ones)
    numeric_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']
                    and not c.startswith('consumed') and c not in ['survey_id', 'hhid', 'weight']]
    
    # Food consumption indicators (binary)
    food_cols = [c for c in df.columns if c.startswith('consumed')]
    if food_cols:
        pos_vals = [1, 1.0, '1', '1.0', 'Yes', 'Access']
        df['total_consumed_items'] = df[food_cols].isin(pos_vals).sum(axis=1)
        
        # Define basic staples vs luxury foods based on descriptions
        basic_staples = ['consumed300', 'consumed500', 'consumed600', 'consumed1500', 'consumed1600', 'consumed2800', 'consumed3200', 'consumed3600']
        basic_cols = [c for c in basic_staples if c in df.columns]
        df['basic_consumed_count'] = df[basic_cols].isin(pos_vals).sum(axis=1) if basic_cols else 0
        
        luxury_items = ['consumed200', 'consumed400', 'consumed800', 'consumed900', 'consumed2000', 'consumed2100', 'consumed2200', 'consumed2400', 'consumed2600', 'consumed4100', 'consumed4400', 'consumed4500', 'consumed4700']
        luxury_cols = [c for c in luxury_items if c in df.columns]
        df['luxury_consumed_count'] = df[luxury_cols].isin(pos_vals).sum(axis=1) if luxury_cols else 0
        
        df['luxury_ratio'] = df['luxury_consumed_count'] / (df['total_consumed_items'] + 1e-8)
    
    # Utility access score
    utility_indicators = ['water', 'toilet', 'sewer', 'elect']
    util_cols = [c for c in utility_indicators if c in df.columns]
    pos_vals = [1, 1.0, '1', '1.0', 'Yes', 'Access']
    df['utility_score'] = df[util_cols].isin(pos_vals).sum(axis=1) if util_cols else 0
    
    # Household composition ratios
    if 'hsize' in df.columns:
        child_cols = ['num_children5', 'num_children10', 'num_children18']
        existing_child_cols = [c for c in child_cols if c in df.columns]
        df['total_children'] = df[existing_child_cols].astype(float).sum(axis=1) if existing_child_cols else 0
        df['child_ratio'] = df['total_children'] / (df['hsize'] + 1e-8)
        
        adult_cols = ['num_adult_female', 'num_adult_male']
        existing_adult_cols = [c for c in adult_cols if c in df.columns]
        df['adult_ratio'] = df[existing_adult_cols].astype(float).sum(axis=1) / (df['hsize'] + 1e-8) if existing_adult_cols else 0
        
        df['dependency_ratio'] = (df['total_children'] + df.get('num_elderly', 0)) / (df[existing_adult_cols].astype(float).sum(axis=1) + 1e-8) if existing_adult_cols else 0
    
    # Log utility expenditure
    if 'utl_exp_ppp17' in df.columns:
        df['log_utl_exp'] = np.log1p(df['utl_exp_ppp17'])
        df['utl_exp_per_capita'] = df['utl_exp_ppp17'] / (df['hsize'] + 1e-8)
        
    # List of engineered numeric columns to calculate survey stats for
    target_numeric_cols = numeric_cols + [
        'total_consumed_items', 'basic_consumed_count', 'luxury_consumed_count', 'luxury_ratio',
        'utility_score', 'child_ratio', 'adult_ratio', 'dependency_ratio', 'log_utl_exp', 'utl_exp_per_capita'
    ]
    target_numeric_cols = [c for c in target_numeric_cols if c in df.columns]
    
    # 2. Per-survey statistics (mean, std, and deviation from mean)
    for col in target_numeric_cols:
        df[f'{col}_survey_mean'] = df.groupby('survey_id')[col].transform('mean')
        df[f'{col}_survey_std'] = df.groupby('survey_id')[col].transform('std').fillna(0)
        df[f'{col}_dev_from_survey'] = df[col] - df[f'{col}_survey_mean']
        
    # 3. Pairwise ratios between top correlated features
    if 'utl_exp_ppp17' in df.columns and 'hsize' in df.columns:
        df['ratio_utl_hsize'] = df['utl_exp_ppp17'] / (df['hsize'] + 1e-8)
    if 'total_consumed_items' in df.columns and 'hsize' in df.columns:
        df['ratio_consumed_hsize'] = df['total_consumed_items'] / (df['hsize'] + 1e-8)
        
    # Additional Interaction Features
    if 'hsize' in df.columns:
        if 'basic_consumed_count' in df.columns:
            df['basic_per_capita'] = df['basic_consumed_count'] / (df['hsize'] + 1e-8)
        if 'luxury_consumed_count' in df.columns:
            df['luxury_per_capita'] = df['luxury_consumed_count'] / (df['hsize'] + 1e-8)
    
    if 'utility_score' in df.columns and 'total_consumed_items' in df.columns:
        df['util_x_consumed'] = df['utility_score'] * df['total_consumed_items']
        
    # 4. Row-level numeric aggregations
    if numeric_cols:
        df['row_numeric_mean'] = df[numeric_cols].mean(axis=1)
        df['row_numeric_std'] = df[numeric_cols].std(axis=1)
        df['row_numeric_max'] = df[numeric_cols].max(axis=1)
        df['row_numeric_min'] = df[numeric_cols].min(axis=1)
        df['row_numeric_range'] = df['row_numeric_max'] - df['row_numeric_min']
        
    return df


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
    categorical_cols = ['water_source', 'sanitation_source', 'dweltyp', 'educ_max', 'sector1d', 'strata']
    for col in base_feature_cols:
        if train_df[col].dtype == 'object' or col in categorical_cols:
            train_df[col] = train_df[col].astype('category')
            test_features[col] = test_features[col].astype('category')
            categorical_features.append(col)
    
    # ====== KEY IMPROVEMENT 1: Log-transform the target ======
    # Consumption data is heavily right-skewed; log-transform stabilizes variance
    # and dramatically improves model performance
    # We now also clip the 99.5th percentile extreme outliers to improve RMSE
    upper_clip = train_df[target_col].quantile(0.995)
    train_df['log_target'] = np.log1p(train_df[target_col].clip(lower=0.01, upper=upper_clip))
    log_target_col = 'log_target'
    print(f"Target stats — Mean: {train_df[target_col].mean():.2f}, "
          f"Median: {train_df[target_col].median():.2f}, "
          f"Skew: {train_df[target_col].skew():.2f}")
    print(f"Log-target stats — Mean: {train_df[log_target_col].mean():.2f}, "
          f"Skew: {train_df[log_target_col].skew():.2f}")
    
    # ====== KEY IMPROVEMENT 2: Feature Engineering ======
    print("Engineering features...")
    train_df = engineer_features(train_df, base_feature_cols)
    test_features = engineer_features(test_features, base_feature_cols)
    
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
    models_xgb = []
    models_lgb = []
    models_cat = []
    fold_metrics = []
    
    # Check what packages are available
    has_lgb = False
    try:
        from lightgbm import LGBMRegressor
        from lightgbm import early_stopping, log_evaluation
        has_lgb = True
        print("  LightGBM is available.")
    except ImportError:
        print("  WARNING: LightGBM is not installed. Ensembling with LightGBM is disabled.")
        print("  To enable, run: pip install lightgbm")
        
    has_cat = False
    try:
        from catboost import CatBoostRegressor
        has_cat = True
        print("  CatBoost is available.")
    except ImportError:
        print("  WARNING: CatBoost is not installed. Ensembling with CatBoost is disabled.")
        print("  To enable, run: pip install catboost")
    
    # ====== KEY IMPROVEMENT 3: Better Hyperparameters ======
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.02,        # Lower LR for better convergence
        'n_estimators': 2500,          # More trees (early stopping will pick best)
        'max_depth': 7,                # Slightly shallower trees to prevent overfit
        'min_child_weight': 10,        # Higher regularization
        'reg_alpha': 0.5,              # Stronger L1 regularization
        'reg_lambda': 2.0,             # Stronger L2 regularization
        'gamma': 0.2,                  # Min loss reduction for split
        'random_state': 42,
        'colsample_bytree': 0.6,       # Lower Feature sampling per tree
        'colsample_bylevel': 0.6,      # Lower Feature sampling per level
        'subsample': 0.75,             # Row sampling
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 150   # More patience
    }
    
    if has_lgb:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.02,
            'n_estimators': 2500,
            'max_depth': 7,
            'num_leaves': 45,
            'min_child_samples': 30,
            'subsample': 0.75,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': 4,
            'verbose': -1
        }
        
    if has_cat:
        cat_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'learning_rate': 0.03,
            'iterations': 2500,
            'depth': 6,
            'l2_leaf_reg': 5,
            'random_seed': 42,
            'verbose': 250,
            'task_type': 'CPU',
            'thread_count': 4
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
        
        # Train XGBoost
        print("  Training XGBoost...")
        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va_log)],
            sample_weight_eval_set=[w_va],
            verbose=250
        )
        val_preds_xgb = model_xgb.predict(X_va)
        models_xgb.append(model_xgb)
        
        # Train LightGBM
        val_preds_lgb = np.zeros(len(X_va))
        if has_lgb:
            print("  Training LightGBM...")
            model_lgb = LGBMRegressor(**lgb_params)
            model_lgb.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va_log)],
                eval_sample_weight=[w_va],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(250)]
            )
            val_preds_lgb = model_lgb.predict(X_va)
            models_lgb.append(model_lgb)
            
        # Train CatBoost
        val_preds_cat = np.zeros(len(X_va))
        if has_cat:
            print("  Training CatBoost...")
            X_tr_cat = X_tr.copy()
            X_va_cat = X_va.copy()
            for col in categorical_features:
                X_tr_cat[col] = X_tr_cat[col].astype(str)
                X_va_cat[col] = X_va_cat[col].astype(str)
                
            model_cat = CatBoostRegressor(**cat_params)
            model_cat.fit(
                X_tr_cat, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va_cat, y_va_log)],
                early_stopping_rounds=100,
                cat_features=categorical_features
            )
            val_preds_cat = model_cat.predict(X_va_cat)
            models_cat.append(model_cat)
            
        # Dynamic Ensembling based on individual OOF RMSE
        preds_list = [val_preds_xgb]
        rmse_xgb_val = root_mean_squared_error(y_va_orig, np.clip(np.expm1(val_preds_xgb), 0, None))
        rmse_list = [rmse_xgb_val]
        
        if has_lgb:
            rmse_lgb_val = root_mean_squared_error(y_va_orig, np.clip(np.expm1(val_preds_lgb), 0, None))
            preds_list.append(val_preds_lgb)
            rmse_list.append(rmse_lgb_val)
            
        if has_cat:
            rmse_cat_val = root_mean_squared_error(y_va_orig, np.clip(np.expm1(val_preds_cat), 0, None))
            preds_list.append(val_preds_cat)
            rmse_list.append(rmse_cat_val)
            
        # Calculate weights inversely proportional to RMSE (lower RMSE -> higher weight)
        inv_rmse = [1.0 / r for r in rmse_list]
        total_inv_rmse = sum(inv_rmse)
        fold_weights = [r / total_inv_rmse for r in inv_rmse]
        
        # We save these fold_weights dynamically as attributes on the models so we can use them later
        model_xgb.dynamic_weight = fold_weights[0]
        if has_lgb:
            model_lgb.dynamic_weight = fold_weights[1]
        if has_cat:
            model_cat.dynamic_weight = fold_weights[2]
        
        val_preds_log = np.zeros(len(X_va))
        for p, w in zip(preds_list, fold_weights):
            val_preds_log += p * w
            
        val_preds_orig = np.expm1(val_preds_log)  # Inverse of log1p
        val_preds_orig = np.clip(val_preds_orig, 0, None)  # No negative consumption
        
        oof_preds_log[val_idx] = val_preds_log
        
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
        print(f"Fold {fold+1} Ensemble RMSE: {fold_rmse:.4f} | R²: {fold_r2:.4f} | "
              f"R²(log): {fold_r2_log:.4f} | MAE: {fold_mae:.4f}")
        print(f"  XGB Best iteration: {model_xgb.best_iteration}")
        if has_lgb:
            print(f"  LGB Best iteration: {model_lgb.best_iteration_}")
        if has_cat:
            print(f"  Cat Best iteration: {model_cat.best_iteration_}")
            
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
    
    num_models = len(models_xgb)
    for idx in range(num_models):
        m_xgb = models_xgb[idx]
        preds_xgb = m_xgb.predict(test_X)
        
        preds_lgb = np.zeros(len(test_X))
        if has_lgb:
            m_lgb = models_lgb[idx]
            preds_lgb = m_lgb.predict(test_X)
            
        preds_cat = np.zeros(len(test_X))
        if has_cat:
            m_cat = models_cat[idx]
            test_X_cat = test_X.copy()
            for col in categorical_features:
                test_X_cat[col] = test_X_cat[col].astype(str)
            preds_cat = m_cat.predict(test_X_cat)
            
        fold_pred = preds_xgb * m_xgb.dynamic_weight
        if has_lgb:
            fold_pred += preds_lgb * m_lgb.dynamic_weight
        if has_cat:
            fold_pred += preds_cat * m_cat.dynamic_weight
            
        test_preds_log += fold_pred / num_models
        
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
