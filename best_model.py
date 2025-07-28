import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ━━━━━━━━━━━━━━━━ 1) Load & Preprocess ━━━━━━━━━━━━━━━━
train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")

target_cols = [f"BlendProperty{i}" for i in range(1, 11)]
fraction_cols = [f"Component{i}_fraction" for i in range(1, 6)]
prop_cols = [f"Component{i}_Property{j}" for i in range(1, 6) for j in range(1, 11)]

# Normalize fractions
for df in (train, test):
    df[fraction_cols] = df[fraction_cols].div(df[fraction_cols].sum(axis=1), axis=0)

# Interaction features
for i in range(1, 6):
    frac = f"Component{i}_fraction"
    for j in range(1, 11):
        prop = f"Component{i}_Property{j}"
        name = f"{frac}x{prop}"
        train[name] = train[frac] * train[prop]
        test[name]  = test[frac]  * test[prop]

# PCA
pca = PCA(n_components=10, random_state=42)
train_pca = pca.fit_transform(train[prop_cols])
test_pca  = pca.transform(test[prop_cols])
for k in range(10):
    train[f"pca{k+1}"] = train_pca[:, k]
    test[f"pca{k+1}"] = test_pca[:, k]

# Component-wise stats
for i in range(1, 6):
    props = [f"Component{i}_Property{j}" for j in range(1, 11)]
    train[f"Comp{i}_mean"] = train[props].mean(axis=1)
    train[f"Comp{i}_std"]  = train[props].std(axis=1)
    test[f"Comp{i}_mean"]  = test[props].mean(axis=1)
    test[f"Comp{i}_std"]   = test[props].std(axis=1)

# ━━━━━━━━━━━━━━━━ 2) Prepare Data ━━━━━━━━━━━━━━━━
feature_cols = [c for c in train.columns if c not in target_cols + ['ID']]
X       = train[feature_cols].astype(np.float32)
y       = train[target_cols].astype(np.float32).values
X_test  = test[feature_cols].astype(np.float32)

# ━━━━━━━━━━━━━━━━ 3) Final LightGBM + Stack ━━━━━━━━━━━━━━━━
best_lgb_params = {
    'n_estimators': 460,
    'learning_rate': 0.022516481508345575,
    'num_leaves': 75,
    'max_depth': 9,
    'subsample': 0.7946836881119737,
    'colsample_bytree': 0.667149440377488,
    'reg_alpha': 0.1278353323139978,
    'reg_lambda': 2.5788535510183386,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'random_state': 42,
    'n_jobs': 1,
    'verbosity': -1
}

xgb = XGBRegressor(
    n_estimators=331, max_depth=4, learning_rate=0.08303,
    subsample=0.9404, colsample_bytree=0.6778,
    gamma=0.00284, reg_alpha=2.1732, reg_lambda=0.06053,
    tree_method='hist', device='cuda', verbosity=0,
    random_state=42, n_jobs=1
)

rf = RandomForestRegressor(
    n_estimators=444, max_depth=16,
    min_samples_split=2, min_samples_leaf=3,
    random_state=42, n_jobs=1
)

cat = CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=6,
    verbose=False, random_seed=42, task_type='GPU', devices='0'
)

lgb = LGBMRegressor(**best_lgb_params)

estimators = [
    ('ridge', Ridge(alpha=1.0, random_state=42)),
    ('rf', rf),
    ('xgb', xgb),
    ('lgb', lgb),
    ('cat', cat)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    n_jobs=1,
    passthrough=False
)

model = MultiOutputRegressor(stack)

# ━━━━━━━━━━━━━━━━ 4) 5‑Fold CV & Submit ━━━━━━━━━━━━━━━━
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr, val) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}/5 training...")
    t0 = time.time()

    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[val])
    score = mean_absolute_percentage_error(y[val], pred)

    print(f"Fold {fold} MAPE: {score:.4f} | Time: {time.time() - t0:.1f}s")
    fold_scores.append(score)

print(f"\nOverall CV MAPE: {np.mean(fold_scores):.4f}")

# Final model train on full data
print("Retraining on full data…")
model.fit(X, y)
y_pred = model.predict(X_test)

sub = pd.DataFrame(y_pred, columns=target_cols)
sub.insert(0, 'ID', test['ID'])
sub.to_csv("submission_fixed_lgb.csv", index=False)
print("submission_fixed_lgb.csv saved")
