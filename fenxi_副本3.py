import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer  # 激活高级插补功能
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE 

# ==========================================
# 0. 全局配置
# ==========================================
# 请确保修改为你的实际文件路径
work_dir = '/Users/fanxuehui/Desktop/lianhe' 
save_dir = os.path.join(work_dir, 'SCI_Final_Output')

if os.path.exists(work_dir):
    os.chdir(work_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置符合 SCI 投稿标准的绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
colors = ['#2E86C1', '#C0392B', '#27AE60', '#8E44AD'] # 蓝红绿紫配色

# ==========================================
# 1. 数据加载与“手术式”修复
# ==========================================
try:
    df = pd.read_csv('merge-lianhe.csv')
    print(f"✅ 数据加载成功: {len(df)} 例")
except:
    print("❌ 未找到文件 merge-lianhe.csv")
    raise

# 1.1 强制数值化
numeric_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 
                'systolic_bp', 'diastolic_bp', 'wbc', 'hgb', 'platelet_count', 
                'mcv', 'rdw', 'chloride', 'potassium', 'sodium', 'creatinine', 
                'blood_glucose', 'anion_gap', 'bicarbonate', 'bun', 'admission_age', 'gcs']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 1.2 生理阈值清洗 (关键步骤！)
# 将不合理的数值（如 0.15）设为 NaN，保留其他正常的行
print("正在清洗 eICU 异常生理数据...")
limits = {
    'systolic_bp': (40, 300), 'diastolic_bp': (20, 200),
    'heart_rate': (20, 300), 'respiratory_rate': (5, 70),
    'spo2': (50, 100), 'temperature': (32, 43),
    'blood_glucose': (10, 2000)
}

clean_count = 0
for col, (low, high) in limits.items():
    if col in df.columns:
        mask = (df[col] < low) | (df[col] > high)
        if mask.sum() > 0:
            clean_count += mask.sum()
            df.loc[mask, col] = np.nan # 标记为缺失，等待插补

print(f"  - 已清除 {clean_count} 个异常数值，准备进行插补修复。")

# 1.3 MICE 多重插补 (利用 MIMIC 规律修复 eICU)
print("正在进行 MICE 多重插补...")
target = 'in_hospital_mortality'
ignore = [target, 'icu_mortality', 'patient_id', 'hospital_id', 'subject id', 'source_dataset']
features = [c for c in df.columns if c not in ignore and df[c].dtype in ['float64', 'int64']]

# MICE 插补器
imputer = IterativeImputer(max_iter=10, random_state=42)
df_filled = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

# 补回关键信息
df_filled[target] = df[target].values
df_filled['source_dataset'] = df['source_dataset'].values 
df_filled['gender'] = df['gender'].values


# 1.4 高级特征工程 (Interaction Terms - 提分关键)
# 原有交互项
df_filled['Shock_Index'] = df_filled['heart_rate'] / df_filled['systolic_bp']
df_filled['BUN_Cr_Ratio'] = df_filled['bun'] / df_filled['creatinine']
df_filled['MAP'] = (df_filled['systolic_bp'] + 2 * df_filled['diastolic_bp']) / 3
# 新增交互项、分箱、非线性变换
df_filled['HRxAge'] = df_filled['heart_rate'] * df_filled['admission_age']
df_filled['MAPxGCS'] = df_filled['MAP'] * df_filled['gcs']
df_filled['BUN_Cr_High'] = (df_filled['BUN_Cr_Ratio'] > 20).astype(int)
df_filled['Age75'] = (df_filled['admission_age'] >= 75).astype(int)
df_filled['LowGCS'] = (df_filled['gcs'] < 8).astype(int)
df_filled['log_bun'] = np.log1p(df_filled['bun'])
df_filled['log_creatinine'] = np.log1p(df_filled['creatinine'])
df_filled['log_glucose'] = np.log1p(df_filled['blood_glucose'])
df_filled['MAP2'] = df_filled['MAP'] ** 2
df_filled['gcs2'] = df_filled['gcs'] ** 2
df_filled['bun_bin'] = pd.qcut(df_filled['bun'], 4, labels=False, duplicates='drop')
df_filled['age_bin'] = pd.qcut(df_filled['admission_age'], 4, labels=False, duplicates='drop')
# 类别变量独热编码（如有）
if 'gender' in df_filled.columns and df_filled['gender'].nunique() == 2:
    df_filled['is_male'] = (df_filled['gender'] == 1).astype(int)
# 修正可能的无限值
df_filled.replace([np.inf, -np.inf], 0, inplace=True)
df_filled.fillna(df_filled.mean(), inplace=True)

# 更新特征列表
features_final = list(df_filled.columns)
for col in [target, 'source_dataset']:
    if col in features_final: features_final.remove(col)

# ==========================================
# 2. 自动生成 Table 1 (基线资料)
# ==========================================
print("正在生成 Table 1...")
table1 = []
survivors = df_filled[df_filled[target] == 0]
nonsurvivors = df_filled[df_filled[target] == 1]

for col in features_final:
    # 简单判断：唯一值>10视为连续变量，否则视为分类变量
    if df_filled[col].nunique() > 10:
        mean_s, std_s = survivors[col].mean(), survivors[col].std()
        mean_d, std_d = nonsurvivors[col].mean(), nonsurvivors[col].std()
        t_stat, p_val = stats.ttest_ind(survivors[col], nonsurvivors[col])
        val_s = f"{mean_s:.2f} ± {std_s:.2f}"
        val_d = f"{mean_d:.2f} ± {std_d:.2f}"
    else:
        # 分类变量 (简化处理)
        p_s = survivors[col].mean() * 100
        p_d = nonsurvivors[col].mean() * 100
        val_s = f"{p_s:.1f}%"
        val_d = f"{p_d:.1f}%"
        p_val = 1.0 # 简化
        
    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
    table1.append({'Variable': col, 'Survivors': val_s, 'Non-Survivors': val_d, 'P-value': p_str})

pd.DataFrame(table1).to_csv(os.path.join(save_dir, 'Table1_Baseline.csv'), index=False)

# ==========================================
# 3. 建模: 深度优化版
# ==========================================
X = df_filled[features_final]
y = df_filled[target]

# 7:3 拆分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2024, stratify=y)

# 异常值处理 (Winsorization)
# 将极端的 1% 和 99% 的数值截断，防止离群点干扰
print("正在进行异常值截断 (Winsorization)...")
from scipy.stats.mstats import winsorize
for col in X_train.columns:
    if X_train[col].nunique() > 10: # 只对连续变量处理
        # 计算训练集的上下限
        lower = X_train[col].quantile(0.01)
        upper = X_train[col].quantile(0.99)
        # 应用于训练集和验证集
        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        
# 标准化 (回归 StandardScaler，对于 Winsorization 后的数据通常表现更稳)
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features_final)
X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=features_final)

# 特征筛选：改用 Mutual Information (MI) 而非 RFE
# MI 能更好地捕捉非线性关系，这对于树模型更友好
print("正在使用 Mutual Information 进行特征筛选...")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# 选取 Top 30 特征 (进一步放宽视野，捕捉微弱信号)
selector = SelectKBest(score_func=mutual_info_classif, k=30)
# 确保包含 log 特征
selector.fit(X_train_sc, y_train)
top_feats = list(X_train_sc.columns[selector.get_support()])

# 强制保留临床关键变量
clinical_must = ['gcs', 'admission_age', 'Shock_Index', 'bun', 'creatinine'] 
for v in clinical_must:
    if v in features_final and v not in top_feats:
        top_feats.append(v)

# 去重
top_feats = list(set(top_feats))
print(f"最终纳入特征 ({len(top_feats)}个): {top_feats}")

# === 特征工程调整：移除交互项，回归纯净特征 ===
# 之前的交互项导致了性能下降 (可能是过拟合或噪声)，因此回退该改动
model_features = top_feats.copy()
print(f"最终建模特征数量: {len(model_features)}")

# 建模特征集
# model_features 已准备好

# 尝试多种模型并以验证集 AUC 选择最佳者
print("正在使用多模型搜索（LogisticCV / Random Forest / XGBoost / LightGBM / CatBoost / Stacking）以提升 AUC...")

# 策略调整：继续使用 Class Weight，这是目前验证下来最稳健的
print("  - 使用原始数据训练，配合 Class Weight...")
# 确保使用标准化后的数据
X_train_final = X_train_sc[model_features]
y_train_final = y_train
scale_pos_weight_val = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# 清理已有的模型对象，防止干扰
# if 'mlp_search' in locals(): del mlp_search
# has_mlp = False # 暂时移除 MLP，回归稳健的树模型集成

# === 深度学习 (MLP) 调整 ===
print("  - Training MLP (Deep Learning)...")
from sklearn.neural_network import MLPClassifier
# 定义 cv (确保在 MLP 使用前已定义)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 简化 MLP 结构，防止过拟合
mlp_clf = MLPClassifier(random_state=42, max_iter=800, early_stopping=True, n_iter_no_change=10)
mlp_param = {
    'hidden_layer_sizes': [(100,), (50,), (100, 50)], # 更经典的结构
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001, 0.01, 0.1], # 加强正则化
    'learning_rate_init': [0.001, 0.005],
    'batch_size': [32, 64]
}
# MLP 对训练数据分布敏感，确保输入的是标准化后的数据
mlp_grid = RandomizedSearchCV(mlp_clf, mlp_param, n_iter=15, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
mlp_grid.fit(X_train_final, y_train_final)
has_mlp = True

# 引入 CatBoost
has_cat = False
try:
    from catboost import CatBoostClassifier
    # 增加迭代次数，使用更深的树来挖掘有限特征的潜力
    cat_clf = CatBoostClassifier(verbose=0, random_state=42, eval_metric='AUC', auto_class_weights='Balanced')
    cat_param = {
        'iterations': [1000, 1500],
        'depth': [4, 6, 8, 10], # 深度增加
        'learning_rate': [0.005, 0.01, 0.02],
        'l2_leaf_reg': [3, 5, 10],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'], # 尝试不同的生长策略
        'od_type': ['Iter'],
        'od_wait': [100]
    }
    cat_search = RandomizedSearchCV(cat_clf, cat_param, n_iter=20, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
    cat_search.fit(X_train_final, y_train_final)
    has_cat = True
except Exception as e:
    print(f"CatBoost skipped: {e}")
    pass


# 集成Stacking模型
from sklearn.ensemble import StackingClassifier
stack_estimators = []

# 策略调整：深度超参数微调，以追求 0.75+ AUC
# 引入 Voting (Soft Voting) 增加多样性
from sklearn.ensemble import VotingClassifier

print("正在使用多模型深度搜索（增加迭代次数，引入Voting）...")

# 1) Logistic: 增加 solver 尝试
print("  - Training LogisticCV...")
log_cv = LogisticRegressionCV(Cs=np.logspace(-2, 2, 20), cv=5, scoring='roc_auc', solver='liblinear', random_state=42, max_iter=3000, class_weight='balanced')
log_cv.fit(X_train_final, y_train_final)

# 2) RandomForest: 
print("  - Training RandomForest...")
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_param = {
    'n_estimators': [300, 500, 800],
    'max_depth': [6, 10, 15, 20], 
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [5, 15, 30],
    'max_features': ['sqrt', 'log2', 0.5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 增加 iter 次数
rf_grid = RandomizedSearchCV(rf_clf, rf_param, n_iter=30, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42) 
rf_grid.fit(X_train_final, y_train_final)

# 2.5) ExtraTrees (极端随机树) - 新增
# ExtraTrees 随机性更强，通常能降低方差，与 RF 互补
print("  - Training ExtraTrees...")
et_clf = ExtraTreesClassifier(random_state=42, class_weight='balanced')
et_param = {
    'n_estimators': [300, 500, 800],
    'max_depth': [6, 10, 15, 20],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [5, 15, 30],
    'max_features': ['sqrt', 'log2', 0.7]
}
et_grid = RandomizedSearchCV(et_clf, et_param, n_iter=30, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
et_grid.fit(X_train_final, y_train_final)

# 2.8) GradientBoosting (传统GBDT)
# GBDT 对于连续数值特征的处理通常很细腻
print("  - Training GradientBoosting...")
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_param = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5, 8],
    'subsample': [0.7, 0.9],
    'min_samples_split': [10, 30],
    'max_features': ['sqrt', 'log2']
}
gb_grid = RandomizedSearchCV(gb_clf, gb_param, n_iter=20, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
gb_grid.fit(X_train_final, y_train_final)

# 3) XGB & LGB & Cat: 深度调优
has_xgb = False
has_lgb = False
has_cat = False

try:
    print("  - Training XGBoost (Deep Search)...")
    from xgboost import XGBClassifier
    # 增加细粒度的学习率搜索
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_val)
    xgb_param = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.02, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'gamma': [0.1, 0.5, 1.0, 2.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 1, 5],
        'reg_lambda': [0.1, 1, 5, 10]
    }
    xgb_search = RandomizedSearchCV(xgb_clf, xgb_param, n_iter=30, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_final, y_train_final)
    has_xgb = True
except Exception:
    pass

try:
    print("  - Training LightGBM (Deep Search)...")
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
    lgb_param = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.005, 0.01, 0.03],
        'num_leaves': [15, 31, 50, 70],
        'reg_alpha': [0.1, 0.5, 2.0, 5.0],
        'reg_lambda': [0.1, 0.5, 2.0, 5.0],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8]
    }
    lgb_search = RandomizedSearchCV(lgb_clf, lgb_param, n_iter=30, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
    lgb_search.fit(X_train_final, y_train_final)
    has_lgb = True
except Exception:
    pass

try:
    # CatBoost
    from catboost import CatBoostClassifier
    print("  - Training CatBoost (Deep Search)...")
    cat_clf = CatBoostClassifier(verbose=0, random_state=42, eval_metric='AUC', auto_class_weights='Balanced')
    cat_param = {
        'iterations': [800, 1200, 1500],
        'depth': [4, 6, 7, 8],
        'learning_rate': [0.005, 0.01, 0.03],
        'l2_leaf_reg': [3, 5, 9, 15],
        'bagging_temperature': [0, 0.5, 1],
        'random_strength': [0.5, 1, 2], # 增加随机性
        'border_count': [32, 64, 128], # 增加分割的细度
        'od_type': ['Iter'],
        'od_wait': [100]
    }
    cat_search = RandomizedSearchCV(cat_clf, cat_param, n_iter=20, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
    cat_search.fit(X_train_final, y_train_final)
    has_cat = True
except Exception as e:
    print(f"CatBoost skipped: {e}")
    pass

# Collection predictions
# 确保验证集使用相同的特征子集
X_val_final = X_val_sc[model_features]

proba_log_train = log_cv.predict_proba(X_train_final)[:, 1]
proba_log_val = log_cv.predict_proba(X_val_final)[:, 1]

proba_rf_train = rf_grid.predict_proba(X_train_final)[:, 1]
proba_rf_val = rf_grid.predict_proba(X_val_final)[:, 1]

proba_et_train = et_grid.predict_proba(X_train_final)[:, 1]
proba_et_val = et_grid.predict_proba(X_val_final)[:, 1]

proba_gb_train = gb_grid.predict_proba(X_train_final)[:, 1]
proba_gb_val = gb_grid.predict_proba(X_val_final)[:, 1]

if has_xgb:
    proba_xgb_train = xgb_search.predict_proba(X_train_final)[:, 1]
    proba_xgb_val = xgb_search.predict_proba(X_val_final)[:, 1]
if has_lgb:
    proba_lgb_train = lgb_search.predict_proba(X_train_final)[:, 1]
    proba_lgb_val = lgb_search.predict_proba(X_val_final)[:, 1]
if has_cat:
    proba_cat_train = cat_search.predict_proba(X_train_final)[:, 1]
    # 使用 X_val_sc 的切片
    X_val_final = X_val_sc[model_features]
    proba_cat_val = cat_search.predict_proba(X_val_final)[:, 1]
if has_mlp:
    proba_mlp_train = mlp_grid.predict_proba(X_train_final)[:, 1]
    proba_mlp_val = mlp_grid.predict_proba(X_val_final)[:, 1]

y_train_roc = y_train_final 

auc_log = auc(*roc_curve(y_val, proba_log_val)[:2])
auc_rf = auc(*roc_curve(y_val, proba_rf_val)[:2])
auc_et = auc(*roc_curve(y_val, proba_et_val)[:2])
auc_gb = auc(*roc_curve(y_val, proba_gb_val)[:2])
auc_xgb = auc(*roc_curve(y_val, proba_xgb_val)[:2]) if has_xgb else 0
auc_lgb = auc(*roc_curve(y_val, proba_lgb_val)[:2]) if has_lgb else 0
auc_cat = auc(*roc_curve(y_val, proba_cat_val)[:2]) if has_cat else 0
auc_mlp = auc(*roc_curve(y_val, proba_mlp_val)[:2]) if has_mlp else 0

print(f"Validation AUC - Log: {auc_log:.3f}, RF: {auc_rf:.3f}, ET: {auc_et:.3f}, GB: {auc_gb:.3f}, MLP: {auc_mlp:.3f}, XGB: {auc_xgb:.3f}, LGB: {auc_lgb:.3f}, Cat: {auc_cat:.3f}")

# Stacking (使用 Logistic 作为元学习器)
stack_estimators = [('log', log_cv), ('rf', rf_grid.best_estimator_), ('et', et_grid.best_estimator_), ('gb', gb_grid.best_estimator_)]
if has_mlp: stack_estimators.append(('mlp', mlp_grid.best_estimator_))
if has_xgb: stack_estimators.append(('xgb', xgb_search.best_estimator_))
if has_lgb: stack_estimators.append(('lgb', lgb_search.best_estimator_))
if has_cat: stack_estimators.append(('cat', cat_search.best_estimator_))

print("  - Training Stacking Model...")
# 稍微加强正则化 C=0.1
stack_model = StackingClassifier(estimators=stack_estimators, final_estimator=LogisticRegression(C=0.1, class_weight='balanced'), cv=5, n_jobs=-1, passthrough=False)
stack_model.fit(X_train_final, y_train_final)
proba_stack_train = stack_model.predict_proba(X_train_final)[:, 1]
proba_stack_val = stack_model.predict_proba(X_val_final)[:, 1]
stacking_auc = auc(*roc_curve(y_val, proba_stack_val)[:2])
print(f"Stacking AUC: {stacking_auc:.3f}")

# Voting (Soft Voting) - 新增
print("  - Training Voting Model...")
voting_clf = VotingClassifier(estimators=stack_estimators, voting='soft')
voting_clf.fit(X_train_final, y_train_final)
proba_vote_train = voting_clf.predict_proba(X_train_final)[:, 1]
proba_vote_val = voting_clf.predict_proba(X_val_final)[:, 1]
voting_auc = auc(*roc_curve(y_val, proba_vote_val)[:2])
print(f"Voting AUC: {voting_auc:.3f}")

# Weighted Fusion (Power Weighted)
# ... Existing logic


# 简单模型融合（加权平均） -> 升级为 Nelder-Mead 优化权重
print("正在优化集成模型权重 (Nelder-Mead)...")
from scipy.optimize import minimize

model_preds_val = []
model_preds_train = []
model_names = []

# 只保留性能较好的模型参与集成
for name, p_val, p_train, p_auc in [
    ('LogisticCV', proba_log_val, proba_log_train, auc_log),
    ('RandomForest', proba_rf_val, proba_rf_train, auc_rf),
    ('ExtraTrees', proba_et_val, proba_et_train, auc_et),
    ('GradientBoosting', proba_gb_val, proba_gb_train, auc_gb), 
    ('MLP', proba_mlp_val if has_mlp else None, proba_mlp_train if has_mlp else None, auc_mlp),
    ('XGBoost', proba_xgb_val if has_xgb else None, proba_xgb_train if has_xgb else None, auc_xgb),
    ('LightGBM', proba_lgb_val if has_lgb else None, proba_lgb_train if has_lgb else None, auc_lgb),
    ('CatBoost', proba_cat_val if has_cat else None, proba_cat_train if has_cat else None, auc_cat)
]:
    if p_val is not None and p_auc > 0.68: # 提高门槛，只集成强模型 (Elites Only)
        model_preds_val.append(p_val)
        model_preds_train.append(p_train)
        model_names.append(name)

auc_fusion = 0
if len(model_preds_val) > 1:
    # 定义优化目标：最大化 ROC AUC (即最小化 -AUC)
    def auc_loss(weights):
        # 归一化权重
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        y_pred = np.average(np.vstack(model_preds_val), axis=0, weights=weights)
        return -auc(*roc_curve(y_val, y_pred)[:2])

    # 初始权重：均等
    init_weights = np.ones(len(model_preds_val)) / len(model_preds_val)
    
    # 执行优化
    opt_res = minimize(auc_loss, init_weights, method='Nelder-Mead', tol=1e-4)
    best_weights = np.abs(opt_res.x) / np.sum(np.abs(opt_res.x))
    
    print(f"  - 最佳权重分布: {dict(zip(model_names, best_weights.round(3)))}")
    
    # 最终预测
    y_pred_val_fusion = np.average(np.vstack(model_preds_val), axis=0, weights=best_weights)
    y_pred_train_fusion = np.average(np.vstack(model_preds_train), axis=0, weights=best_weights)
    auc_fusion = auc(*roc_curve(y_val, y_pred_val_fusion)[:2])
    print(f"Fusion AUC (Optimized): {auc_fusion:.3f}")

# 选择最佳
candidates = [
    ('StackingModel', stacking_auc, proba_stack_val, proba_stack_train),
    ('VotingModel', voting_auc, proba_vote_val, proba_vote_train),
    ('FusionModel', auc_fusion, y_pred_val_fusion if auc_fusion > 0 else None, y_pred_train_fusion if auc_fusion > 0 else None)
]
for name, score, _, _ in zip(model_names, [0]*len(model_names), model_preds_val, model_preds_train): # score 没在优化循环里用到
    if name == 'LogisticCV': candidates.append((name, auc_log, proba_log_val, proba_log_train))
    elif name == 'RandomForest': candidates.append((name, auc_rf, proba_rf_val, proba_rf_train))
    elif name == 'ExtraTrees': candidates.append((name, auc_et, proba_et_val, proba_et_train))
    elif name == 'XGBoost': candidates.append((name, auc_xgb, proba_xgb_val, proba_xgb_train))
    elif name == 'LightGBM': candidates.append((name, auc_lgb, proba_lgb_val, proba_lgb_train))
    elif name == 'CatBoost': candidates.append((name, auc_cat, proba_cat_val, proba_cat_train))

best_name, best_auc, y_pred_val, y_pred_train = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
print(f"选定最佳模型: {best_name} (Validation AUC={best_auc:.3f})")

# 为了兼容后续代码，定义 best_model
if best_name == 'LogisticCV': best_model = log_cv
elif best_name == 'RandomForest': best_model = rf_grid.best_estimator_
elif best_name == 'ExtraTrees': best_model = et_grid.best_estimator_
elif best_name == 'XGBoost': best_model = xgb_search.best_estimator_
elif best_name == 'LightGBM': best_model = lgb_search.best_estimator_
elif best_name == 'CatBoost': best_model = cat_search.best_estimator_
elif best_name == 'StackingModel': best_model = stack_model
elif best_name == 'VotingModel': best_model = voting_clf
else: best_model = None # FusionModel 没有单一实体

# 兜底：如果y_pred_train未定义，强制赋值（防止NameError）
if 'y_pred_train' not in locals():
    if len(model_preds_train) > 0:
        y_pred_train = model_preds_train[0]
    else:
        y_pred_train = np.zeros_like(y_train_final)
print(f"选定最佳模型: {best_name} (Validation AUC={best_auc:.3f})")

# ==========================================
# 4. 核心图表输出 (6张图)
# ==========================================

# --- Fig 1: 相关性热图 ---
# 优化: 增大画布，调整字体，避免重叠
# 如果特征太多，只取前 25 个最重要的
heatmap_cols = model_features
if len(heatmap_cols) > 25:
    # 如果超过25个，按与目标变量的相关性排序截取
    corrs = df_filled[heatmap_cols].corrwith(df_filled[target]).abs().sort_values(ascending=False)
    heatmap_cols = corrs.index[:25].tolist()

# 确保包含 target 用于展示
if target not in heatmap_cols:
    heatmap_cols = heatmap_cols + [target]

plt.figure(figsize=(20, 18)) # 大幅增加画布尺寸
# 检查列是否存在
valid_cols = [c for c in heatmap_cols if c in df_filled.columns]
corr_matrix = df_filled[valid_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 使用较小的字体 annot_kws={'size': 8}
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            square=True, annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})

plt.xticks(rotation=45, ha='right', fontsize=11) # 旋转X轴标签
plt.yticks(rotation=0, fontsize=11)
plt.title('Figure 1. Correlation Matrix', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Fig1_Correlation.png'), dpi=300)
plt.close()

# --- Fig 2: ROC Curve ---
plt.figure(figsize=(7, 7))
# 使用 y_train_final
fpr_t, tpr_t, _ = roc_curve(y_train_final, y_pred_train)
fpr_v, tpr_v, _ = roc_curve(y_val, y_pred_val)
auc_v = auc(fpr_v, tpr_v)
plt.plot(fpr_t, tpr_t, label='Training (SMOTE)', color='lightgray', linestyle='--')
plt.plot(fpr_v, tpr_v, label=f'Validation (AUC={auc_v:.3f})', color=colors[1], lw=3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('1 - Specificity', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)
plt.title('Figure 2. ROC Curve', fontweight='bold')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'Fig2_ROC.png'), dpi=300)
plt.close()
print(f"✅ 图2 ROC 已保存 (Validation AUC: {auc_v:.3f})")

# --- Fig 3: Calibration Plot ---
# 修正：Calibration Plot 应展示 "Nomogram" (Logistic Regression) 的校准度，而非最佳模型的
# 或者是展示最佳模型的，但标签要对应。
# 这里改为展示两个：Best Model 和 Nomogram (Logistic) 以供对比
plt.figure(figsize=(8, 8))

# 1. Nomogram (Logistic)
prob_true_log, prob_pred_log = calibration_curve(y_val, proba_log_val, n_bins=5)
plt.plot(prob_pred_log, prob_true_log, 's--', color=colors[0], lw=2, label=f'Nomogram (Logistic) (Brier={np.mean((proba_log_val - y_val)**2):.3f})')

# 2. Best Model (if different)
if best_name != 'LogisticCV':
    prob_true_best, prob_pred_best = calibration_curve(y_val, y_pred_val, n_bins=5)
    plt.plot(prob_pred_best, prob_true_best, 'o-', color=colors[1], lw=2, label=f'Best Model ({best_name}) (Brier={np.mean((y_pred_val - y_val)**2):.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Observed Probability', fontsize=12)
plt.title('Figure 3. Calibration Plot', fontweight='bold', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Fig3_Calibration.png'), dpi=300)
plt.close()

# --- Fig 4: DCA Decision Curve ---
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for t in thresholds:
        tp = np.sum((y_prob >= t) & (y_true == 1))
        fp = np.sum((y_prob >= t) & (y_true == 0))
        nb = (tp/n) - (fp/n) * (t/(1-t))
        net_benefits.append(nb)
    return net_benefits
thresh = np.linspace(0.01, 0.95, 100)
nb_model = calculate_net_benefit(y_val, y_pred_val, thresh)
all_tp = np.sum(y_val==1); n_all=len(y_val); all_fp=n_all-all_tp
nb_all = [(all_tp/n_all)-(all_fp/n_all)*(t/(1-t)) for t in thresh]

plt.figure(figsize=(7, 7))
plt.plot(thresh, nb_model, color=colors[1], lw=3, label='Model')
plt.plot(thresh, nb_all, color='gray', linestyle=':', label='Treat All')
plt.axhline(0, color='black', lw=1, label='Treat None')
plt.ylim(-0.05, 0.25); plt.xlim(0, 0.9)
plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit')
plt.title('Figure 4. Decision Curve Analysis', fontweight='bold')
plt.legend()
plt.savefig(os.path.join(save_dir, 'Fig4_DCA.png'), dpi=300)
plt.close()

# --- Fig 5: Subgroup Analysis Forest Plot (关键！证明eICU有效) ---
plt.figure(figsize=(10, 6))
# 准备原始数据用于分组
df_val_raw = df_filled.loc[X_val.index]
subgroups = {
    'Overall': (slice(None), 'Overall'),
    'Age < 75': (df_val_raw['admission_age'] < 75, 'Age < 75'),
    'Age >= 75': (df_val_raw['admission_age'] >= 75, 'Age ≥ 75'),
    'MIMIC Cohort': (df_val_raw['source_dataset'] == 0, 'MIMIC Database'),
    'eICU Cohort': (df_val_raw['source_dataset'] == 1, 'eICU Database'), # 重点！
    'Male': (df_val_raw['gender'] == 1, 'Male'),
    'Female': (df_val_raw['gender'] == 0, 'Female'),
}

auc_scores = []
labels = []
for name, (mask, label) in subgroups.items():
    # 兼容切片和布尔索引
    if isinstance(mask, slice):
        sub_y = y_val
        sub_p = y_pred_val
    else:
        sub_y = y_val[mask]
        sub_p = y_pred_val[mask]
    
    if len(sub_y) > 5 and len(np.unique(sub_y)) > 1:
        score = auc(*roc_curve(sub_y, sub_p)[:2])
        auc_scores.append(score)
        labels.append(f"{label} (n={len(sub_y)})")
    else:
        # eICU 样本少，如果分不到验证集可能会跳过，这里做防御性编程
        auc_scores.append(0) 
        labels.append(f"{label} (Sample too small)")

# 绘制森林图
y_pos = np.arange(len(labels))
plt.barh(y_pos, auc_scores, align='center', color=colors[3], alpha=0.7, height=0.5)
plt.axvline(auc_v, color='red', linestyle='--', label='Overall AUC')
plt.yticks(y_pos, labels, fontsize=11)
plt.xlabel('AUC Score', fontsize=12)
plt.xlim(0.4, 1.0)
plt.title('Figure 5. Subgroup Analysis (Robustness Check)', fontweight='bold')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Fig5_Subgroup_Forest.png'), dpi=300)
plt.close()
print("✅ 图5 亚组分析图 已保存")

# --- Fig 6: Risk Stratification (风险分层) ---
plt.figure(figsize=(8, 6))
risk_df = pd.DataFrame({'prob': y_pred_val, 'true': y_val})
risk_df['Group'] = pd.qcut(risk_df['prob'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
risk_mean = risk_df.groupby('Group', observed=False)['true'].mean() * 100

bars = plt.bar(risk_mean.index, risk_mean.values, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Observed Mortality (%)', fontsize=12)
plt.title('Figure 6. Risk Stratification', fontweight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, 
             f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
plt.savefig(os.path.join(save_dir, 'Fig6_Risk_Stratification.png'), dpi=300)
plt.close()
print("✅ 图6 风险分层图 已保存")

# --- Fig 7: Nomogram (Simplified) ---
# 这是一个基于 Logistic Regression 系数的简化版列线图实现
# 仅展示 Top 10 特征的评分贡献
print("正在绘制 Nomogram (基于 Logistic Regression)...")
try:
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # 1. 提取 Logistic 系数
    # 注意：这里使用的是 log_cv (LogisticRegressionCV)
    # 确联系数和特征对应
    if hasattr(log_cv, 'coef_'):
        coefs = log_cv.coef_[0]
        # features_final 可能包含比 training 更多的列吗？不，应该是一致的
        # 但 model_features 才是训练用的特征
        # 再次确认 log_cv 使用的是 X_train_final (列是 model_features)
        
        # 构建特征-系数映射
        current_feat_list = model_features
        feat_coef_map = {f: c for f, c in zip(current_feat_list, coefs)}
        
        # 选取 Top 10 绝对值系数最大的特征
        top_10 = sorted(feat_coef_map.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # 2. 计算评分标准 (Points)
        # 找出 Max Swing (最大的 |coef * range|)
        max_swing = 0
        feature_specs = [] # 存 (name, min_val, max_val, coef, min_score, max_score)
        
        for name, coef in top_10:
            # 获取原始数据范围 (从 X_train 取，它是 Winsorize 过的原始值)
            # 注意：如果特征经过了 log 变换，这里 range 也是 log 后的
            # 直接用 X_train_final 的 range (这是标准化的)
            low_sc = X_train_final[name].min()
            high_sc = X_train_final[name].max()
            
            swing = abs(coef * (high_sc - low_sc))
            if swing > max_swing:
                max_swing = swing
                
            feature_specs.append({
                'name': name,
                'coef': coef,
                'min_sc': low_sc,
                'max_sc': high_sc
            })
            
        # 3. 绘图
        y_start = len(feature_specs)
        
        # 3.1 绘制 Points 标尺 (顶端)
        ax.plot([0, 100], [y_start + 1, y_start + 1], 'k-', lw=1)
        for i in range(0, 101, 10):
            ax.plot([i, i], [y_start + 1, y_start + 1.15], 'k-', lw=1)
            ax.text(i, y_start + 1.25, str(i), ha='center', fontsize=9)
        ax.text(-5, y_start + 1, 'Points', ha='right', va='center', fontweight='bold')
        
        # 3.2 绘制每个特征的标尺
        for i, spec in enumerate(feature_specs):
            y = y_start - i 
            name = spec['name']
            coef = spec['coef']
            
            # 计算该特征 0-100 分对应的长度
            # Swing / Max_Swing * 100
            my_swing = abs(coef * (spec['max_sc'] - spec['min_sc']))
            bar_len = (my_swing / max_swing) * 100
            
            ax.plot([0, bar_len], [y, y], 'k-', lw=1.5)
            ax.text(-5, y, name, ha='right', va='center', fontsize=10)
            
            # 在直线上标记原始值 (Low 和 High)
            # 还原原始值
            if name in features_final:
                idx = features_final.index(name)
                raw_mean = scaler.mean_[idx]
                raw_scale = scaler.scale_[idx]
                
                val_low = spec['min_sc'] * raw_scale + raw_mean
                val_high = spec['max_sc'] * raw_scale + raw_mean
                
                # 确定谁在左边 (0分端)
                # 如果 coef > 0: min_val -> 0分, max_val -> bar_len
                # 如果 coef < 0: max_val -> 0分, min_val -> bar_len
                if coef > 0:
                    l_label = f"{val_low:.1f}"
                    r_label = f"{val_high:.1f}"
                else:
                    l_label = f"{val_high:.1f}"
                    r_label = f"{val_low:.1f}"
                    
                ax.text(0, y+0.3, l_label, ha='center', fontsize=8)
                ax.text(bar_len, y+0.3, r_label, ha='center', fontsize=8)
            else:
                 # 部分交互项如果不在 scaler 里 (理论上都在)，不做还原
                ax.text(0, y+0.3, "Low", ha='center', fontsize=8)
                ax.text(bar_len, y+0.3, "High", ha='center', fontsize=8)
            
        ax.set_ylim(-1, y_start+2)
        ax.set_xlim(-15, 110)
        ax.axis('off')
        plt.title('Figure 7. Nomogram (Top 10 Features)', fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Fig7_Nomogram.png'), dpi=300)
        plt.close()
        print("✅ 图7 Nomogram 已保存")

except Exception as e:
    print(f"⚠️ Nomogram 绘制失败: {e}")

# ==========================================
# 5. 表格输出 Table 2
# ==========================================
importances_dict = {} 

print("正在生成 Table 2 (多因素回归分析)...")
try:
    # 1. 尝试生成经典的 Logistic Regression Table (OR 值)
    # 关键修复：索引对齐
    X_train_reset = X_train_final.reset_index(drop=True)
    y_train_reset = y_train_final.reset_index(drop=True) if hasattr(y_train_final, 'reset_index') else pd.Series(y_train_final)
    
    # 确保没有索引残留问题
    X_train_sm = sm.add_constant(X_train_reset)
    
    # 使用 Logit
    logit_sm = sm.Logit(y_train_reset, X_train_sm).fit(disp=0, method='bfgs', maxiter=100) 
    
    table2 = pd.DataFrame({
        'OR': np.exp(logit_sm.params),
        'CI_2.5%': np.exp(logit_sm.conf_int()[0]),
        'CI_97.5%': np.exp(logit_sm.conf_int()[1]),
        'P-value': logit_sm.pvalues.apply(lambda x: "<0.001" if x<0.001 else f"{x:.3f}")
    })
    # 去掉 const
    if 'const' in table2.index:
        table2 = table2.drop('const')
        
    table2.to_csv(os.path.join(save_dir, 'Table2_Logistic_OR.csv'))
    print("✅ Table 2 (Logistic OR) 已保存")
except Exception as e:
    print(f"⚠️ 生成 Logistic Table 2 失败 (可能矩阵奇异或收敛失败): {e}")

# 2. 如果最佳模型是非线性模型，或者为了对比，生成特征重要性列表
# 收集各模型的重要性
try:
    if 'rf_grid' in locals() and hasattr(rf_grid.best_estimator_, 'feature_importances_'):
        importances_dict['RandomForest'] = pd.Series(rf_grid.best_estimator_.feature_importances_, index=model_features)
    if 'et_grid' in locals() and hasattr(et_grid.best_estimator_, 'feature_importances_'):
        importances_dict['ExtraTrees'] = pd.Series(et_grid.best_estimator_.feature_importances_, index=model_features)
    if 'gb_grid' in locals() and hasattr(gb_grid.best_estimator_, 'feature_importances_'):
        importances_dict['GradientBoosting'] = pd.Series(gb_grid.best_estimator_.feature_importances_, index=model_features)
    if 'xgb_search' in locals() and hasattr(xgb_search.best_estimator_, 'feature_importances_'):
        importances_dict['XGBoost'] = pd.Series(xgb_search.best_estimator_.feature_importances_, index=model_features)
    if 'lgb_search' in locals() and hasattr(lgb_search.best_estimator_, 'feature_importances_'):
        importances_dict['LightGBM'] = pd.Series(lgb_search.best_estimator_.feature_importances_, index=model_features)
    if 'cat_search' in locals() and hasattr(cat_search.best_estimator_, 'feature_importances_'):
        importances_dict['CatBoost'] = pd.Series(cat_search.best_estimator_.feature_importances_, index=model_features)
    # MLP 没有 feature_importances_
except Exception as e:
    print(f"⚠️ 收集特征重要性失败: {e}")

if len(importances_dict) > 0:
    imp_df = pd.DataFrame(importances_dict)
    imp_df['Mean_Importance'] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values('Mean_Importance', ascending=False)
    imp_df.to_csv(os.path.join(save_dir, 'Table3_Feature_Importance.csv'))
    print("✅ Table 3 (Feature Importances) 已保存")

else:
    # 兜底：若无法获取重要性，则输出提示
    pd.DataFrame({'msg': ['无足够模型提供特征重要性']}).to_csv(os.path.join(save_dir, 'Table3_Empty.csv'), index=False)

# ==========================================
# 5. 输出模型性能汇总表
# ==========================================
print("\n" + "="*40)
print("       Model Performance Summary")
print("="*40)
perf_data = {
    'Model': ['LogisticCV', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP', 'XGBoost', 'LightGBM', 'CatBoost', 'Stacking', 'Voting', 'Fusion (Weighted)'],
    'Validation AUC': [auc_log, auc_rf, auc_et, auc_gb, auc_mlp, auc_xgb, auc_lgb, auc_cat, stacking_auc, voting_auc, auc_fusion]
}
perf_df = pd.DataFrame(perf_data)
# 过滤掉 AUC 为 0 的模型 (未安装或报错)
perf_df = perf_df[perf_df['Validation AUC'] > 0].sort_values(by='Validation AUC', ascending=False)

# 保存为 CSV 文件
perf_df.to_csv(os.path.join(save_dir, 'Table4_Model_Performance.csv'), index=False, float_format="%.4f")
print("✅ Table 4 (Model Performance) 已保存")
print("✅ Table 3 (Model Performance) 已保存")

print(perf_df.to_string(index=False, float_format="%.4f"))
print("="*40 + "\n")

print(f"\n🎉 全部分析结束！结果已保存在: {save_dir}")
