import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    roc_curve, auc, calibration_curve, roc_auc_score,
    accuracy_score, recall_score, f1_score, confusion_matrix
)
import statsmodels.api as sm

# --- 机器学习模型库 ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    StackingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 尝试导入高级集成模型
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ==========================================
# 0. 全局配置与 SCI 风格设置
# ==========================================
warnings.filterwarnings('ignore')
work_dir = '.' 
save_dir = os.path.join(work_dir, 'SCI_Analysis_Results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ==========================================
# 1. 数据加载与严谨清洗
# ==========================================
print("Step 1: Loading & Cleaning Data...")
try:
    # 自动处理编码或异常行
    df = pd.read_csv('merge-lianhe.csv', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    # 剔除 ID 类与数据泄露类变量 (核心：必须剔除 icu_mortality)
    cols_to_drop = ['icu_mortality', 'subject id', 'patient_id', 'hadm_id', 'hospital_id']
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    # 强制数值化
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 生理极值过滤
    if 'systolic_bp' in df.columns:
        df.loc[(df['systolic_bp'] < 40) | (df['systolic_bp'] > 300), 'systolic_bp'] = np.nan
    if 'heart_rate' in df.columns:
        df.loc[(df['heart_rate'] < 20) | (df['heart_rate'] > 250), 'heart_rate'] = np.nan

    # 1.6 特征工程 (增加医学常用指标)
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        df['Shock_Index'] = df['heart_rate'] / df['systolic_bp']
    if 'bun' in df.columns and 'creatinine' in df.columns:
        df['BUN_Cr_Ratio'] = df['bun'] / df['creatinine']

    print(f"✅ Data Ready. Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

# ==========================================
# 2. MICE 多重插补
# ==========================================
target = 'in_hospital_mortality'
# 这里的 source_dataset 仅在亚组分析时用，不参与插补计算
features_analysis = [c for c in df.columns if c not in [target, 'source_dataset']]

print("Step 2: Running MICE Imputation...")
imputer = IterativeImputer(max_iter=10, random_state=42)
df_filled = df.copy()
df_filled[features_analysis] = imputer.fit_transform(df[features_analysis])

# ==========================================
# 3. Table 1: 基线特征描述
# ==========================================
print("Step 3: Generating Table 1...")
# 此处逻辑同前，保存为 Table1_Baseline.csv
# [此处代码为简洁起见略，逻辑已在您的原代码中成熟运行]

# ==========================================
# 4. 数据划分与特征筛选
# ==========================================
X = df_filled[features_analysis]
y = df_filled[target]
# meta 存储用于亚组分析的原始信息 (确保索引对齐)
meta = df_filled[['source_dataset', 'admission_age', 'gender']] if 'source_dataset' in df_filled.columns else df_filled[['admission_age', 'gender']]

X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
    X, y, meta, test_size=0.3, random_state=2024, stratify=y
)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)

# 筛选 Top 15 特征
selector = SelectKBest(mutual_info_classif, k=15)
selector.fit(X_train_sc, y_train)
selected_feats = X.columns[selector.get_support()].tolist()
X_train_final = X_train_sc[selected_feats]
X_val_final = X_val_sc[selected_feats]

# ==========================================
# 5. 模型训练与 Table 4 (多模型对比)
# ==========================================
print("Step 4: Training Models...")
model_dict = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}
if HAS_LGB: model_dict['LightGBM'] = LGBMClassifier(verbose=-1, random_state=42)

results_list = []
probas = {}

for name, model in model_dict.items():
    model.fit(X_train_final, y_train)
    y_prob = model.predict_proba(X_val_final)[:, 1]
    y_pred = model.predict(X_val_final)
    
    auc_val = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    results_list.append({
        'Model': name, 'AUC': auc_val, 'Accuracy': acc, 
        'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp)
    })
    probas[name] = y_prob

# 保存指标
pd.DataFrame(results_list).to_csv(os.path.join(save_dir, 'Table4_Model_Performance.csv'), index=False)
best_model_name = max(probas, key=lambda k: roc_auc_score(y_val, probas[k]))
best_prob = probas[best_model_name]

# ==========================================
# 6. 生成可视化 (包含修正后的亚组分析)
# ==========================================
print("Step 5: Visualizing Results...")

# Fig 2: ROC Comparison
plt.figure(figsize=(7, 7))
for name, y_prob in probas.items():
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title('Figure 2. ROC Curves')
plt.savefig(os.path.join(save_dir, 'Figure2_ROC.png'), dpi=300)

# --- 核心：Figure 5 优化后的亚组分析 ---
print("   - Generating Subgroup Analysis...")
subgroup_configs = [(slice(None), 'Overall')]

if 'source_dataset' in meta_val.columns:
    subgroup_configs.append((meta_val['source_dataset'] == 0, 'MIMIC Database'))
    subgroup_configs.append((meta_val['source_dataset'] == 1, 'eICU Database'))
if 'admission_age' in meta_val.columns:
    subgroup_configs.append((meta_val['admission_age'] >= 65, 'Age ≥ 65'))
    subgroup_configs.append((meta_val['admission_age'] < 65, 'Age < 65'))
if 'gender' in meta_val.columns:
    subgroup_configs.append((meta_val['gender'] == 1, 'Male'))
    subgroup_configs.append((meta_val['gender'] == 0, 'Female'))

sub_labels, sub_scores = [], []
for mask, label in subgroup_configs:
    # 关键：使用 .values 确保 numpy array 和 pandas series 索引对齐
    if isinstance(mask, slice):
        y_sub, p_sub = y_val, best_prob
    else:
        y_sub = y_val[mask.values]
        p_sub = best_prob[mask.values]
    
    if len(y_sub) > 10 and len(np.unique(y_sub)) > 1:
        sub_scores.append(roc_auc_score(y_sub, p_sub))
        sub_labels.append(f"{label} (n={len(y_sub)})")

plt.figure(figsize=(10, 6))
plt.barh(sub_labels, sub_scores, color='#1f77b4', alpha=0.7)
plt.axvline(sub_scores[0], color='red', linestyle='--', label=f'Overall AUC')
plt.xlim(max(0, min(sub_scores)-0.1), 1.0)
plt.xlabel('AUC Score')
plt.title(f'Figure 5. Subgroup Analysis ({best_model_name})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure5_Subgroup.png'), dpi=300)

# --- Figure 6 & 7: 风险分层与特征贡献 ---
# 风险四分位图
df_risk = pd.DataFrame({'prob': best_prob, 'label': y_val})
df_risk['Risk Group'] = pd.qcut(df_risk['prob'], 4, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4(High)'])
risk_res = df_risk.groupby('Risk Group')['label'].mean() * 100

plt.figure(figsize=(8, 5))
risk_res.plot(kind='bar', color=colors, edgecolor='black')
plt.ylabel('Observed Mortality Rate (%)')
plt.title('Figure 6. Risk Stratification')
plt.savefig(os.path.join(save_dir, 'Figure6_Risk.png'), dpi=300)

# 特征重要性 (Random Forest 为例)
importances = model_dict['Random Forest'].feature_importances_
feat_imp = pd.Series(importances, index=selected_feats).sort_values()
plt.figure(figsize=(10, 8))
feat_imp.plot(kind='barh', color='teal')
plt.title('Figure 7. Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure7_Importance.png'), dpi=300)

print(f"\n🚀 Analysis Complete! All figures and tables are in '{save_dir}'.")