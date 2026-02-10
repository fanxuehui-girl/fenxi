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

# --- 机器学习模型 ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 0. 全局配置
# ==========================================
warnings.filterwarnings('ignore')
save_dir = './SCI_ICU_Mortality_Results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# SCI 绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ==========================================
# 1. 数据加载与清理 (针对 ICU Mortality)
# ==========================================
print("Step 1: Loading & Cleaning Data for ICU Mortality...")
df = pd.read_csv('merge-lianhe.csv', on_bad_lines='skip')
df.columns = [c.strip() for c in df.columns]

# --- 核心修改：in_hospital_mortality 设为泄露变量剔除 ---
target = 'icu_mortality'
leakage_and_ids = ['in_hospital_mortality', 'subject id', 'patient_id', 'hadm_id', 'hospital_id', 'source_dataset']
df.drop(columns=[c for c in leakage_and_ids if c in df.columns], errors='ignore', inplace=True)

# 确保目标变量没有缺失值
df = df.dropna(subset=[target])

# 强制数值化
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 特征工程
if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
    df['Shock_Index'] = df['heart_rate'] / df['systolic_bp']
if 'bun' in df.columns and 'creatinine' in df.columns:
    df['BUN_Cr_Ratio'] = df['bun'] / df['creatinine']

# ==========================================
# 2. MICE 多重插补
# ==========================================
print("Step 2: MICE Imputation...")
features_all = [c for c in df.columns if c != target]
imputer = IterativeImputer(max_iter=10, random_state=42)
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ==========================================
# 3. Table 1: 基线资料 (ICU Survivors vs. Non-Survivors)
# ==========================================
print("Step 3: Generating Table 1...")
table1 = []
survivors = df_filled[df_filled[target] == 0]
nonsurvivors = df_filled[df_filled[target] == 1]

for col in features_all:
    if df_filled[col].nunique() > 10: 
        m_s, s_s = survivors[col].mean(), survivors[col].std()
        m_d, s_d = nonsurvivors[col].mean(), nonsurvivors[col].std()
        _, p_val = stats.ttest_ind(survivors[col], nonsurvivors[col], equal_var=False)
        val_s, val_d = f"{m_s:.2f} ± {s_s:.2f}", f"{m_d:.2f} ± {s_d:.2f}"
    else:
        ct = pd.crosstab(df_filled[col], df_filled[target])
        if ct.size < 4: p_val = 1.0 # 容错
        else:
            _, p_val, _, _ = chi2_contingency(ct) if ct.min().min() >= 5 else (0, fisher_exact(ct)[1], 0, 0)
        n_s, n_d = survivors[col].sum(), nonsurvivors[col].sum()
        val_s, val_d = f"{int(n_s)} ({n_s/len(survivors)*100:.1f}%)", f"{int(n_d)} ({n_d/len(nonsurvivors)*100:.1f}%)"
    
    table1.append({'Variable': col, 'ICU-Survivors': val_s, 'ICU-Non-Survivors': val_d, 'P-value': f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"})

pd.DataFrame(table1).to_csv(os.path.join(save_dir, 'Table1_Baseline.csv'), index=False)

# ==========================================
# 4. 特征筛选与划分
# ==========================================
X = df_filled[features_all]
y = df_filled[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2024, stratify=y)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features_all)
X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=features_all)

selector = SelectKBest(mutual_info_classif, k=15)
selector.fit(X_train_sc, y_train)
selected_feats = X_train_sc.columns[selector.get_support()].tolist()

X_train_final = X_train_sc[selected_feats]
X_val_final = X_val_sc[selected_feats]

# ==========================================
# 5. 模型训练与表格生成 (Table 2, 3, 4)
# ==========================================
print("Step 4: Model Training...")

# Table 4: Metrics
model_dict = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced')
}

results = []
probas = {}
for name, model in model_dict.items():
    model.fit(X_train_final, y_train)
    y_prob = model.predict_proba(X_val_final)[:, 1]
    y_pred = model.predict(X_val_final)
    auc_val = roc_auc_score(y_val, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    results.append({'Model': name, 'AUC': auc_val, 'Accuracy': accuracy_score(y_val, y_pred),
                    'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp)})
    probas[name] = y_prob

pd.DataFrame(results).to_csv(os.path.join(save_dir, 'Table4_Model_Performance.csv'), index=False)

# Table 2: Logistic Regression ORs
logit_model = sm.Logit(y_train, sm.add_constant(X_train_final)).fit(disp=0)
df_lr = pd.DataFrame({'OR': np.exp(logit_model.params), 'P-value': logit_model.pvalues})
df_lr.to_csv(os.path.join(save_dir, 'Table2_Logistic_Results.csv'))

# Table 3: Feature Importance (RF)
importances = pd.DataFrame({'Feature': selected_feats, 'Importance': model_dict['Random Forest'].feature_importances_})
importances.sort_values(by='Importance', ascending=False).to_csv(os.path.join(save_dir, 'Table3_Feature_Importance.csv'), index=False)

# ==========================================
# 6. 生成可视化图表 (Figure 1-6)
# ==========================================
print("Step 5: Visualizing (6 Figures)...")

# Fig 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_filled[selected_feats[:10] + [target]].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Figure 1. Feature Correlation (Target: ICU Mortality)')
plt.savefig(os.path.join(save_dir, 'Figure1_Correlation.png'), dpi=300)

# Fig 2: ROC Comparison
plt.figure(figsize=(8, 7))
for name, prob in probas.items():
    fpr, tpr, _ = roc_curve(y_val, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('1 - Specificity'); plt.ylabel('Sensitivity'); plt.legend(); plt.title('Figure 2. ROC Curves (ICU Mortality)')
plt.savefig(os.path.join(save_dir, 'Figure2_ROC.png'), dpi=300)

# Fig 3: Calibration
plt.figure(figsize=(6, 6))
prob_true, prob_pred = calibration_curve(y_val, probas['Logistic Regression'], n_bins=8)
plt.plot(prob_pred, prob_true, 's-', label='Logistic Regression')
plt.plot([0,1],[0,1], 'k--', label='Perfectly Calibrated')
plt.title('Figure 3. Calibration Curve'); plt.legend()
plt.savefig(os.path.join(save_dir, 'Figure3_Calibration.png'), dpi=300)

# Fig 4: DCA
thresh = np.linspace(0.01, 0.8, 100)
nb = [((np.sum((probas['Logistic Regression'] >= t) & (y_val == 1))/len(y_val)) - 
       (np.sum((probas['Logistic Regression'] >= t) & (y_val == 0))/len(y_val)) * (t/(1-t))) for t in thresh]
plt.figure(figsize=(7, 6))
plt.plot(thresh, nb, color='red', label='Model')
plt.axhline(0, color='black', label='None')
plt.ylim(-0.05, 0.2); plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit')
plt.title('Figure 4. Decision Curve Analysis'); plt.legend()
plt.savefig(os.path.join(save_dir, 'Figure4_DCA.png'), dpi=300)

# Fig 5: Risk Stratification
df_risk = pd.DataFrame({'prob': probas['Logistic Regression'], 'label': y_val})
df_risk['Risk Group'] = pd.qcut(df_risk['prob'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
(df_risk.groupby('Risk Group')['label'].mean()*100).plot(kind='bar', color=colors[:4], edgecolor='black')
plt.ylabel('Observed ICU Mortality (%)'); plt.title('Figure 5. Risk Stratification')
plt.savefig(os.path.join(save_dir, 'Figure5_Risk.png'), dpi=300)

# Fig 6: Nomogram Plot (Feature Contribution)
coefs = pd.Series(logit_model.params[1:], index=selected_feats).sort_values()
plt.figure(figsize=(10, 8))
bar_colors = ['#d62728' if x > 0 else '#1f77b4' for x in coefs]
coefs.plot(kind='barh', color=bar_colors)
plt.axvline(0, color='black', lw=0.8)
plt.title('Figure 6. Model Nomogram (Feature Weights)'); plt.xlabel('Log-Odds Coefficient')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure6_Nomogram.png'), dpi=300)

print(f"✅ Success! Results saved to '{save_dir}'.")