# =============================================================================
# main.py - UAS Data Science: Customer Churn Prediction Pipeline
# EDA -> Direct Modeling -> Preprocessing -> Tuning -> Save Model
# =============================================================================

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

RS = 42
TOP_N = 10                      # jumlah fitur final yang dipakai model & app
OUT  = Path("output_plots"); OUT.mkdir(exist_ok=True)
MDL  = Path("models");       MDL.mkdir(exist_ok=True)
plt.rcParams['figure.dpi'] = 110

print("="*60, "\nUAS DATA SCIENCE - CHURN PREDICTION PIPELINE\n" + "="*60)

# ─────────────────────────────────────────────────────────────
# 0. LOAD / GENERATE DATASET
# ─────────────────────────────────────────────────────────────
def make_synthetic(n=15000, seed=42):
    rng = np.random.default_rng(seed)
    def inj(a, r=0.04):
        a = a.astype(object); a[rng.random(n) < r] = np.nan; return a
    age=rng.integers(18,70,n); spent=rng.exponential(500,n).round(2)
    sat=rng.integers(1,11,n).astype(float); tix=rng.poisson(1.5,n)
    ref=rng.integers(0,6,n); lv=(spent*rng.uniform(1.2,3,n)).round(2)
    freq=rng.integers(0,20,n).astype(float); vis=rng.integers(1,200,n).astype(float)
    st=rng.exponential(5,n).round(2); pg=rng.integers(1,30,n).astype(float)
    eo=rng.uniform(0,1,n).round(3); ec=(eo*rng.uniform(.1,.6,n)).round(3)
    nps=rng.integers(-100,101,n).astype(float); disc=rng.integers(0,20,n).astype(float)
    dd=rng.integers(0,15,n).astype(float); mk=rng.exponential(30,n).round(2)
    ao=(spent/np.maximum(freq+1,1)).round(2)
    churn=((-0.3*(sat-5)+0.4*tix+0.3*ref-0.5*(freq/10)+rng.normal(0,1,n))>0).astype(int)
    return pd.DataFrame({
        'customer_id':[f'C{i:05d}' for i in range(n)],
        'gender':rng.choice(['Male','Female','Other'],n),
        'age':inj(age),'country':rng.choice(['USA','UK','Germany','France','India'],n),
        'city':rng.choice(['New York','London','Berlin','Paris','Mumbai'],n),
        'signup_date':pd.to_datetime('2020-01-01'),'last_purchase_date':pd.to_datetime('2023-06-01'),
        'acquisition_channel':rng.choice(['Organic','Paid','Referral','Social','Email'],n),
        'device_type':rng.choice(['Mobile','Desktop','Tablet'],n),
        'subscription_type':rng.choice(['Basic','Standard','Premium'],n),
        'is_premium_user':rng.integers(0,2,n),'total_visits':inj(vis),
        'avg_session_time':inj(st),'pages_per_session':inj(pg),
        'email_open_rate':inj(eo),'email_click_rate':inj(ec),
        'total_spent':inj(spent),'avg_order_value':inj(ao),
        'discount_used':inj(disc),'coupon_code':rng.choice(['SAVE10','DISC20','NONE','VIP30'],n),
        'support_tickets':inj(tix),'refund_requested':inj(ref),
        'delivery_delay_days':inj(dd),'payment_method':rng.choice(['Credit Card','PayPal','Debit Card','Bank Transfer'],n),
        'satisfaction_score':inj(sat),'nps_score':inj(nps),
        'marketing_spend_per_user':inj(mk),'lifetime_value':inj(lv),
        'last_3_month_purchase_freq':inj(freq),'churn':churn
    })

df = pd.read_csv('sales_marketing.csv') if Path('sales_marketing.csv').exists() else make_synthetic()
if not Path('sales_marketing.csv').exists():
    df.to_csv('sales_marketing.csv', index=False)
print(f"\n[0] Dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────
# 1. EDA
# ─────────────────────────────────────────────────────────────
print("\n[1] EDA")
print(df.head())
df.info()
print(df.describe(include='all'))

miss = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(9,4))
ax.barh(miss.index, miss.values, color=sns.color_palette('Reds_r', len(miss)))
ax.set_title('Missing Value per Kolom (%)'); plt.tight_layout()
plt.savefig(OUT/'1_missing_values.png'); plt.close()

vc = df['churn'].value_counts()
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].bar(['Tidak Churn','Churn'], vc.values, color=['#2ecc71','#e74c3c'])
axes[1].pie(vc.values, labels=['Tidak Churn','Churn'], autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'])
plt.suptitle('Distribusi Target Churn'); plt.tight_layout()
plt.savefig(OUT/'2_distribusi_churn.png'); plt.close()

num_df = df.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(13,10))
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            mask=np.triu(np.ones(num_df.corr().shape,bool)), annot_kws={'size':7}, ax=ax)
ax.set_title('Heatmap Korelasi'); plt.tight_layout()
plt.savefig(OUT/'3_heatmap_korelasi.png'); plt.close()
print("[1] EDA plots disimpan di output_plots/")

# ─────────────────────────────────────────────────────────────
# 2. DIRECT MODELING (tanpa preprocessing)
# ─────────────────────────────────────────────────────────────
print("\n[2] Direct Modeling")
DROP = ['customer_id','signup_date','last_purchase_date','coupon_code','churn']
Xd = df.drop(columns=[c for c in DROP if c in df.columns])
y  = df['churn']
le = LabelEncoder()
for c in Xd.select_dtypes('object').columns:
    Xd[c] = le.fit_transform(Xd[c].astype(str))
for c in Xd.columns:
    Xd[c] = Xd[c].fillna(Xd[c].median())

Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.2, random_state=RS, stratify=y)

models_direct = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=RS),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=-1),
    'Voting Classifier': VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=500, random_state=RS)),
        ('svm', SVC(probability=True, random_state=RS)),
        ('knn', KNeighborsClassifier(n_neighbors=7))], voting='soft')
}
rows = []
for name, m in models_direct.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append({'Model':name,'Stage':'Direct','Accuracy':accuracy_score(yte,p),
                 'Precision':precision_score(yte,p,zero_division=0),
                 'Recall':recall_score(yte,p,zero_division=0),
                 'F1':f1_score(yte,p,zero_division=0)})
df_direct = pd.DataFrame(rows)
print(df_direct.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n[3] Preprocessing")
dfp = df.drop(columns=['customer_id','signup_date','last_purchase_date','coupon_code'], errors='ignore')
dfp.drop_duplicates(inplace=True)

num_c = [c for c in dfp.select_dtypes(include=np.number).columns if c != 'churn']
cat_c = dfp.select_dtypes(include='object').columns.tolist()
dfp[num_c] = SimpleImputer(strategy='median').fit_transform(dfp[num_c])
if cat_c: dfp[cat_c] = SimpleImputer(strategy='most_frequent').fit_transform(dfp[cat_c])

for c in num_c:
    Q1, Q3 = dfp[c].quantile(.25), dfp[c].quantile(.75)
    dfp[c] = dfp[c].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))

le_map = {}
for c in cat_c:
    le_map[c] = LabelEncoder()
    dfp[c] = le_map[c].fit_transform(dfp[c].astype(str))

Xp = dfp.drop(columns=['churn']); yp = dfp['churn']
feat_names = Xp.columns.tolist()
Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(Xp, yp, test_size=0.2, random_state=RS, stratify=yp)
scaler_full = StandardScaler()
Xp_tr_s = scaler_full.fit_transform(Xp_tr); Xp_te_s = scaler_full.transform(Xp_te)

models_prep = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=RS),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=-1),
    'Voting Classifier': VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=500, random_state=RS)),
        ('svm', SVC(probability=True, random_state=RS)),
        ('knn', KNeighborsClassifier(n_neighbors=7))], voting='soft')
}
rows_p = []
trained_prep = {}
for name, m in models_prep.items():
    m.fit(Xp_tr_s, yp_tr); p = m.predict(Xp_te_s)
    trained_prep[name] = m
    rows_p.append({'Model':name,'Stage':'Preprocessed','Accuracy':accuracy_score(yp_te,p),
                   'Precision':precision_score(yp_te,p,zero_division=0),
                   'Recall':recall_score(yp_te,p,zero_division=0),
                   'F1':f1_score(yp_te,p,zero_division=0)})
df_prep = pd.DataFrame(rows_p)
print(df_prep.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE & SELECTION (Top-N fitur konsisten)
# ─────────────────────────────────────────────────────────────
print(f"\n[4] Feature Importance & Top-{TOP_N} Selection")
rf_ref = trained_prep['Random Forest']
importances = pd.Series(rf_ref.feature_importances_, index=feat_names).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9,6))
importances.head(15)[::-1].plot.barh(ax=ax, color=sns.color_palette('YlOrRd',15)[::-1])
ax.set_title(f'Feature Importance (Random Forest)'); plt.tight_layout()
plt.savefig(OUT/'4_feature_importance.png'); plt.close()

top_features = importances.head(TOP_N).index.tolist()
print(f"Top-{TOP_N} fitur: {top_features}")

X_top = Xp[top_features]
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X_top, yp, test_size=0.2, random_state=RS, stratify=yp)
scaler_top = StandardScaler()
Xt_tr_s = scaler_top.fit_transform(Xt_tr); Xt_te_s = scaler_top.transform(Xt_te)

# ─────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (pada Top-N fitur)
# ─────────────────────────────────────────────────────────────
print("\n[5] Hyperparameter Tuning (GridSearchCV)")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=RS),
    {'C':[0.01,0.1,1,10], 'solver':['lbfgs','liblinear']}, cv=cv5, scoring='f1', n_jobs=-1)
gs_lr.fit(Xt_tr_s, yt_tr)
print(f"LR  best: {gs_lr.best_params_} | CV F1: {gs_lr.best_score_:.4f}")

gs_rf = GridSearchCV(RandomForestClassifier(random_state=RS),
    {'n_estimators':[50,100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]},
    cv=cv5, scoring='f1', n_jobs=-1)
gs_rf.fit(Xt_tr_s, yt_tr)
print(f"RF  best: {gs_rf.best_params_} | CV F1: {gs_rf.best_score_:.4f}")

gs_vc = GridSearchCV(VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=RS)),
        ('svm', SVC(probability=True, random_state=RS, kernel='linear')),
        ('knn', KNeighborsClassifier())], voting='soft'),
    {'lr__C':[1,10], 'svm__C':[1], 'knn__n_neighbors':[7,11]},
    cv=3, scoring='f1', n_jobs=1)
gs_vc.fit(Xt_tr_s, yt_tr)
print(f"VC  best: {gs_vc.best_params_} | CV F1: {gs_vc.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. EVALUASI & SIMPAN MODEL TERBAIK
# ─────────────────────────────────────────────────────────────
print("\n[6] Evaluasi Model Hasil Tuning")
tuned = {'Logistic Regression (Tuned)': gs_lr.best_estimator_,
         'Random Forest (Tuned)': gs_rf.best_estimator_,
         'Voting Classifier (Tuned)': gs_vc.best_estimator_}
rows_t = []
for name, m in tuned.items():
    p = m.predict(Xt_te_s)
    rows_t.append({'Model':name,
                   'Accuracy':accuracy_score(yt_te,p),
                   'Precision':precision_score(yt_te,p,zero_division=0),
                   'Recall':recall_score(yt_te,p,zero_division=0),
                   'F1':f1_score(yt_te,p,zero_division=0)})
df_tuned = pd.DataFrame(rows_t).set_index('Model')
print(df_tuned.to_string())

best_name = df_tuned['F1'].idxmax()
best_model = tuned[best_name]
p_best = best_model.predict(Xt_te_s)

cm = confusion_matrix(yt_te, p_best)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(cm, display_labels=['Tidak Churn','Churn']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix - {best_name}'); plt.tight_layout()
plt.savefig(OUT/'5_confusion_matrix_best.png'); plt.close()

# Simpan SEMUA artefak dengan struktur KONSISTEN (top_features = fitur yg dipakai scaler & model)
joblib.dump(best_model, MDL/'best_model.pkl')
joblib.dump(scaler_top, MDL/'scaler.pkl')          # scaler ini fit pada top_features
joblib.dump(scaler_top, MDL/'scaler_top.pkl')      # alias, sama persis
joblib.dump(le_map, MDL/'label_encoders.pkl')
joblib.dump(top_features, MDL/'top_features.pkl')  # WAJIB sama panjang dgn scaler.n_features_in_
joblib.dump(feat_names, MDL/'all_features.pkl')
joblib.dump({
    'best_model_name': best_name,
    'test_accuracy': float(df_tuned.loc[best_name,'Accuracy']),
    'test_precision': float(df_tuned.loc[best_name,'Precision']),
    'test_recall': float(df_tuned.loc[best_name,'Recall']),
    'test_f1': float(df_tuned.loc[best_name,'F1']),
    'top_features': top_features,
    'n_features': len(top_features)
}, MDL/'model_metadata.pkl')

print(f"\n✔ Model terbaik: {best_name}")
print(f"  F1={df_tuned.loc[best_name,'F1']:.4f}  Acc={df_tuned.loc[best_name,'Accuracy']:.4f}")
print(f"  Jumlah fitur tersimpan: {len(top_features)} (scaler & model konsisten)")
print(f"  Artefak tersimpan di: {MDL}/")
print("\n" + "="*60 + "\nPIPELINE SELESAI - Jalankan: streamlit run app.py\n" + "="*60)
