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

# --- 导入机器学习模型 ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    StackingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 尝试导入 XGBoost/LightGBM (如果安装了的话)
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
# 0. 全局配置
# ==========================================
warnings.filterwarnings('ignore')
work_dir = '.' 
save_dir = os.path.join(work_dir, 'SCI_Results_Final')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# SCI 投稿绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 鲜明的对比色板
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# ==========================================
# 1. 数据加载与清洗
# ==========================================
print("Step 1: Loading & Cleaning Data...")
try:
    # on_bad_lines='skip' 自动跳过文件末尾的乱码行
    df = pd.read_csv('merge-lianhe.csv', on_bad_lines='skip')
    
    # 清洗列名空格
    df.columns = [c.strip() for c in df.columns]
    
    # 严格清洗：删除 subject id 不是数字的无效行
    df['subject id'] = pd.to_numeric(df['subject id'], errors='coerce')
    df = df.dropna(subset=['subject id'])
    
    print(f"✅ Data loaded successfully. Valid N = {len(df)}")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

# 1.1 强制数值化
numeric_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 
                'systolic_bp', 'diastolic_bp', 'wbc', 'hgb', 'platelet_count', 
                'mcv', 'rdw', 'chloride', 'potassium', 'sodium', 'creatinine', 
                'blood_glucose', 'anion_gap', 'bicarbonate', 'bun', 'admission_age', 'gcs']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 1.2 生理极值清洗 (去除医学上不可能的数值)
limits = {
    'systolic_bp': (40, 300), 'diastolic_bp': (20, 200),
    'heart_rate': (20, 300), 'respiratory_rate': (5, 70),
    'spo2': (50, 100), 'temperature': (32, 43),
    'blood_glucose': (10, 2000), 'wbc': (0.1, 200)
}
for col, (low, high) in limits.items():
    if col in df.columns:
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

# 1.3 特征工程
if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
    df['Shock_Index'] = df['heart_rate'] / df['systolic_bp']
if 'bun' in df.columns and 'creatinine' in df.columns:
    df['BUN_Cr_Ratio'] = df['bun'] / df['creatinine']
if 'heart_rate' in df.columns and 'admission_age' in df.columns:
    df['HRxAge'] = df['heart_rate'] * df['admission_age']

# 1.4 MICE 多重插补
target = 'in_hospital_mortality'
# 排除 ID 和无关变量
features_raw = [c for c in df.columns if c not in [target, 'patient_id', 'subject id', 'icu_mortality'] and df[c].dtype in ['float64', 'int64']]

print("   - Running MICE Imputation...")
imputer = IterativeImputer(max_iter=10, random_state=42)
df_filled = df.copy()
df_filled[features_raw] = imputer.fit_transform(df[features_raw])

# ==========================================
# 2. Table 1: 基线资料 (含统计检验)
# ==========================================
print("Step 2: Generating Table 1...")
table1 = []
survivors = df_filled[df_filled[target] == 0]
nonsurvivors = df_filled[df_filled[target] == 1]

for col in features_raw:
    if col == 'source_dataset': continue
    
    # 连续变量 (>10个唯一值): T检验
    if df_filled[col].nunique() > 10:
        mean_s, std_s = survivors[col].mean(), survivors[col].std()
        mean_d, std_d = nonsurvivors[col].mean(), nonsurvivors[col].std()
        stat, p_val = stats.ttest_ind(survivors[col], nonsurvivors[col], equal_var=False)
        val_s = f"{mean_s:.2f} ± {std_s:.2f}"
        val_d = f"{mean_d:.2f} ± {std_d:.2f}"
    # 分类变量: 卡方/Fisher
    else:
        ct = pd.crosstab(df_filled[col], df_filled[target])
        if ct.size == 0: p_val = 1.0
        elif ct.min().min() < 5 and ct.shape == (2,2):
            stat, p_val = fisher_exact(ct)
        else:
            stat, p_val, _, _ = chi2_contingency(ct)
        
        n_s = survivors[col].sum()
        n_d = nonsurvivors[col].sum()
        val_s = f"{int(n_s)} ({n_s/len(survivors)*100:.1f}%)"
        val_d = f"{int(n_d)} ({n_d/len(nonsurvivors)*100:.1f}%)"

    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
    table1.append({'Variable': col, 'Survivors': val_s, 'Non-Survivors': val_d, 'P-value': p_str})

pd.DataFrame(table1).to_csv(os.path.join(save_dir, 'Table1_Baseline.csv'), index=False)

# ==========================================
# 3. 特征筛选与数据划分
# ==========================================
print("Step 3: Preparing Data...")

X = df_filled.drop(columns=[target, 'source_dataset'], errors='ignore')
y = df_filled[target]
meta = df_filled[['source_dataset', 'admission_age', 'gender']]

# 7:3 划分
X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
    X, y, meta, test_size=0.3, random_state=2024, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)

# 互信息特征筛选
print("   - Selecting Top Features...")
clinical_must = ['wbc', 'blood_glucose', 'gcs', 'Shock_Index', 'bun', 'admission_age']
selector = SelectKBest(mutual_info_classif, k=15)
selector.fit(X_train_sc, y_train)
selected_feats = list(set(X.columns[selector.get_support()].tolist() + [c for c in clinical_must if c in X.columns]))

X_train_final = X_train_sc[selected_feats]
X_val_final = X_val_sc[selected_feats]
print(f"   - Final Feature Count: {len(selected_feats)}")

# ==========================================
# 4. 核心：多模型训练与 Table 4 生成
# ==========================================
print("Step 4: Training Multiple Models for Table 4...")

# 4.1 定义基模型库
model_dict = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=5),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# 动态添加 XGBoost/LightGBM
if HAS_XGB:
    model_dict['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
if HAS_LGB:
    model_dict['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)

# 4.2 训练基模型并计算指标
results_list = []
trained_models = {}
probas = {} # 存储预测概率用于画图

for name, model in model_dict.items():
    model.fit(X_train_final, y_train)
    
    # 获取概率
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val_final)[:, 1]
    else:
        y_prob = model.decision_function(X_val_final)
        
    y_pred = model.predict(X_val_final)
    
    # 计算五大指标
    auc_val = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    sens = recall_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    spec = tn / (tn + fp)
    f1 = f1_score(y_val, y_pred)
    
    results_list.append({
        'Model': name, 'AUC': auc_val, 'Accuracy': acc, 
        'Sensitivity': sens, 'Specificity': spec, 'F1 Score': f1
    })
    trained_models[name] = model
    probas[name] = y_prob

# 4.3 训练 Stacking 集成模型
print("   - Training Stacking Ensemble...")
base_estimators = [
    ('rf', trained_models['Random Forest']),
    ('gb', trained_models['Gradient Boosting']),
    ('et', trained_models['Extra Trees'])
]
if HAS_LGB: base_estimators.append(('lgb', trained_models['LightGBM']))

stack = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=5)
stack.fit(X_train_final, y_train)
y_prob_stack = stack.predict_proba(X_val_final)[:, 1]
y_pred_stack = stack.predict(X_val_final)

tn, fp, fn, tp = confusion_matrix(y_val, y_pred_stack).ravel()
results_list.append({
    'Model': 'Stacking Ensemble',
    'AUC': roc_auc_score(y_val, y_prob_stack),
    'Accuracy': accuracy_score(y_val, y_pred_stack),
    'Sensitivity': recall_score(y_val, y_pred_stack),
    'Specificity': tn / (tn + fp),
    'F1 Score': f1_score(y_val, y_pred_stack)
})
trained_models['Stacking Ensemble'] = stack
probas['Stacking Ensemble'] = y_prob_stack

# 4.4 生成并保存 Table 4
table4 = pd.DataFrame(results_list).sort_values(by='AUC', ascending=False)
table4 = table4.round(3) # 保留3位小数
table4.to_csv(os.path.join(save_dir, 'Table4_Model_Performance.csv'), index=False)
print("✅ Table 4 Generated (Detailed Model Comparison).")

# 选出最佳模型
best_model_name = table4.iloc[0]['Model']
best_prob = probas[best_model_name]
print(f"🏆 Best Model Selected: {best_model_name}")

# ==========================================
# 5. 生成其他表格 (Table 2 & 3)
# ==========================================
# Table 2: Logistic Regression Results
try:
    logit = sm.Logit(y_train, sm.add_constant(X_train_final)).fit(disp=0)
    conf = logit.conf_int()
    conf['OR'] = logit.params.apply(np.exp)
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    conf['P-value'] = logit.pvalues.apply(lambda x: "<0.001" if x<0.001 else f"{x:.3f}")
    conf.to_csv(os.path.join(save_dir, 'Table2_Logistic_Regression.csv'))
except: pass

# Table 3: Feature Importance (RF)
imp = pd.DataFrame({
    'Feature': selected_feats, 
    'Importance': trained_models['Random Forest'].feature_importances_
}).sort_values(by='Importance', ascending=False)
imp.to_csv(os.path.join(save_dir, 'Table3_Feature_Importance.csv'), index=False)
top_10_feats = imp.head(10)['Feature'].tolist()

# ==========================================
# 6. 生成可视化图表 (Fig 1-7)
# ==========================================
print("Step 5: Generating Figures...")

# Fig 1: Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df_filled[top_10_feats + [target]].corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Figure 1. Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure1_Correlation.png'), dpi=300)
plt.close()

# Fig 2: ROC Curve Comparison (所有模型)
plt.figure(figsize=(8, 8))
# 1. 先画出最佳模型 (加粗红色)
fpr, tpr, _ = roc_curve(y_val, best_prob)
plt.plot(fpr, tpr, color='red', lw=3, label=f"{best_model_name} (AUC={auc(fpr, tpr):.3f})")
# 2. 再画出其他关键模型 (细线)
for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree']:
    if name in probas and name != best_model_name:
        f, t, _ = roc_curve(y_val, probas[name])
        plt.plot(f, t, lw=1.5, alpha=0.7, linestyle='--', label=f"{name} (AUC={auc(f, t):.3f})")

plt.plot([0, 1], [0, 1], 'k:', lw=1)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Figure 2. ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'Figure2_ROC_Comparison.png'), dpi=300)
plt.close()

# Fig 3: Calibration (Best Model)
prob_true, prob_pred = calibration_curve(y_val, best_prob, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, 's-', color=colors[0], label=best_model_name)
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.title('Figure 3. Calibration Plot')
plt.legend()
plt.savefig(os.path.join(save_dir, 'Figure3_Calibration.png'), dpi=300)
plt.close()

# Fig 4: DCA (Best Model)
def net_benefit(y_true, y_prob, t):
    tp = np.sum((y_prob >= t) & (y_true == 1))
    fp = np.sum((y_prob >= t) & (y_true == 0))
    n = len(y_true)
    return (tp/n) - (fp/n) * (t/(1-t))

thresh = np.linspace(0.01, 0.95, 100)
nb_model = [net_benefit(y_val, best_prob, t) for t in thresh]
nb_all = [net_benefit(y_val, np.ones_like(y_val), t) for t in thresh]
plt.figure(figsize=(7, 6))
plt.plot(thresh, nb_model, color='red', lw=2, label=best_model_name)
plt.plot(thresh, nb_all, color='gray', linestyle='--', label='Treat All')
plt.axhline(0, color='k', lw=1, label='Treat None')
plt.ylim(-0.05, 0.3)
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Figure 4. Decision Curve Analysis')
plt.legend()
plt.savefig(os.path.join(save_dir, 'Figure4_DCA.png'), dpi=300)
plt.close()

# Fig 5: Subgroup Analysis
subgroups = {
    'Overall': (slice(None), 'Overall'),
    'Age >= 75': (meta_val['admission_age'] >= 75, 'Age ≥ 75'),
    'Age < 75': (meta_val['admission_age'] < 75, 'Age < 75'),
    'MIMIC Cohort': (meta_val['source_dataset'] == 0, 'MIMIC Database'),
    'eICU Cohort': (meta_val['source_dataset'] == 1, 'eICU Database')
}
labels, scores = [], []
for name, (mask, label) in subgroups.items():
    y_sub = y_val[mask] if not isinstance(mask, slice) else y_val
    p_sub = best_prob[mask.values] if not isinstance(mask, slice) else best_prob
    
    if len(y_sub) > 5:
        scores.append(roc_auc_score(y_sub, p_sub))
        labels.append(f"{label} (n={len(y_sub)})")
    else:
        scores.append(0); labels.append(f"{label} (n<5)")

plt.figure(figsize=(8, 6))
plt.barh(labels, scores, color=colors[0], alpha=0.6)
plt.axvline(table4.iloc[0]['AUC'], color='red', linestyle='--')
plt.xlim(0.4, 1.0)
plt.title('Figure 5. Subgroup Analysis')
plt.xlabel('AUC')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure5_Subgroup.png'), dpi=300)
plt.close()

# Fig 6: Risk Stratification
df_risk = pd.DataFrame({'prob': best_prob, 'target': y_val})
df_risk['group'] = pd.qcut(df_risk['prob'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
risk_mean = df_risk.groupby('group')['target'].mean() * 100
plt.figure(figsize=(8, 6))
risk_mean.plot(kind='bar', color=colors[:4], edgecolor='black', rot=0)
plt.ylabel('Mortality Rate (%)')
plt.title('Figure 6. Risk Stratification')
plt.savefig(os.path.join(save_dir, 'Figure6_Risk_Stratification.png'), dpi=300)
plt.close()

# Fig 7: Nomogram
lr_nomo = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_nomo.fit(X_train_final[top_10_feats], y_train)
coefs = pd.Series(lr_nomo.coef_[0], index=top_10_feats).sort_values(key=abs)
plt.figure(figsize=(10, 7))
c_bar = [colors[2] if x > 0 else colors[0] for x in coefs.values]
plt.barh(coefs.index, coefs.values, color=c_bar)
plt.axvline(0, color='k', lw=1)
plt.title('Figure 7. Nomogram Features (Log-Odds)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure7_Nomogram.png'), dpi=300)
plt.close()

print(f"\n🎉 All Results Generated! Check the '{save_dir}' folder.")