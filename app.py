"""
Sales Lead Qualifier System · MD02
Run:  streamlit run app.py
CSV:  leads_main_1000_final.csv  (same folder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
)
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Sales Lead Qualifier · MD02",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLES ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.block-container { padding: 1.2rem 2rem 3rem; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }

section[data-testid="stSidebar"] { background: #0b0f19 !important; border-right: 1px solid #1a2540; }
section[data-testid="stSidebar"] * { color: #8ba3c0 !important; }
section[data-testid="stSidebar"] label { color: #3d5a78 !important; font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700; }
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-family: 'Outfit', sans-serif !important;
    padding: 0.6rem 1rem !important; font-size: 0.85rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important; transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
section[data-testid="stSidebar"] .stButton>button:hover { transform: translateY(-1px) !important; }

span[data-baseweb="tag"] { background-color: #111827 !important; border: 1px solid #1e3a5f !important; border-radius: 5px !important; }
span[data-baseweb="tag"] span { color: #60a5fa !important; font-weight: 600 !important; font-size: 0.74rem !important; font-family: 'Outfit', sans-serif !important; }
span[data-baseweb="tag"] [role="presentation"] svg { fill: #3b82f6 !important; }

.stSelectbox div, .stMultiSelect div, .stSlider div, .stTextInput input, .stNumberInput input {
    font-family: 'Outfit', sans-serif !important;
}
.stDataFrame, .stDataFrame * { font-family: 'Outfit', sans-serif !important; }
.stMetric label, .stMetric div { font-family: 'Outfit', sans-serif !important; }
button[data-baseweb="tab"] { font-family: 'Outfit', sans-serif !important; }

.hero { background: linear-gradient(135deg, #0b0f19 0%, #0d1526 60%, #091220 100%); border: 1px solid #1a2540; border-radius: 16px; padding: 1.7rem 2.1rem; margin-bottom: 1.4rem; position: relative; overflow: hidden; }
.hero::before { content:''; position:absolute; top:0; right:0; width:300px; height:100%; background: radial-gradient(ellipse at 85% 50%, rgba(37,99,235,0.07) 0%, transparent 70%); pointer-events:none; }
.hero h1 { font-family: 'Outfit', sans-serif !important; font-size: 1.7rem !important; font-weight: 800 !important; color: #e2eaf8 !important; margin: 5px 0 3px !important; letter-spacing: -0.01em; }
.hero .sub { color: #4a7aab; font-size: 0.83rem; margin: 0; font-weight: 400; letter-spacing: 0.01em; }
.hero .badge { display: inline-block; background: rgba(37,99,235,0.15); border: 1px solid rgba(37,99,235,0.3); color: #60a5fa; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; padding: 2px 10px; border-radius: 20px; font-family: 'JetBrains Mono', monospace; }
.hero .acc-badge { display: inline-block; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25); color: #34d399; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 10px; border-radius: 20px; font-family: 'JetBrains Mono', monospace; margin-left: 7px; }
.hero .pills { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 11px; }
.hero .pill { background: rgba(255,255,255,0.04); border: 1px solid #1a2540; color: #5a7fa0; font-size: 0.64rem; padding: 3px 10px; border-radius: 20px; font-weight: 500; font-family: 'Outfit', sans-serif; }

.kpi { background: #fff; border-radius: 12px; padding: 1rem 1.15rem; border: 1px solid #e5ecf5; box-shadow: 0 2px 10px rgba(0,0,0,0.05); transition: box-shadow 0.2s; }
.kpi:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.09); }
.kpi .k-lbl { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 5px; font-family: 'Outfit', sans-serif; }
.kpi .k-val { font-size: 1.85rem; font-weight: 800; color: #0b0f19; font-family: 'Outfit', sans-serif; line-height: 1; letter-spacing: -0.02em; }
.kpi .k-sub { font-size: 0.64rem; color: #94a3b8; margin-top: 4px; font-weight: 400; }
.kpi.high   { border-top: 3px solid #ef4444; }
.kpi.med    { border-top: 3px solid #f59e0b; }
.kpi.low    { border-top: 3px solid #10b981; }
.kpi.blue   { border-top: 3px solid #3b82f6; }
.kpi.purple { border-top: 3px solid #8b5cf6; }
.kpi.teal   { border-top: 3px solid #06b6d4; }

.sec { font-family: 'Outfit', sans-serif; font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.13em; color: #94a3b8; margin: 1.5rem 0 0.75rem; display: flex; align-items: center; gap: 8px; }
.sec::after { content: ''; flex: 1; height: 1px; background: #e5ecf5; }

.ibox { padding: 0.78rem 1.05rem; border-radius: 9px; font-size: 0.81rem; line-height: 1.55; margin-bottom: 0.65rem; font-family: 'Outfit', sans-serif; }
.ibox.blue  { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e40af; }
.ibox.amber { background: #fffbeb; border-left: 4px solid #f59e0b; color: #92400e; }
.ibox.green { background: #f0fdf4; border-left: 4px solid #10b981; color: #065f46; }
.ibox.dark  { background: #0b0f19; border-left: 4px solid #3b82f6; color: #8ba3c0; }

.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #f1f5f9; border-radius: 10px; padding: 3px; border: none; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; padding: 6px 18px !important; font-size: 0.78rem !important; font-weight: 600 !important; color: #64748b !important; background: transparent !important; border: none !important; font-family: 'Outfit', sans-serif !important; letter-spacing: 0.01em !important; }
.stTabs [aria-selected="true"] { background: #fff !important; color: #2563eb !important; box-shadow: 0 1px 5px rgba(0,0,0,0.1) !important; }

.pred-box { background: #0b0f19; border: 1px solid #1a2540; border-radius: 14px; padding: 1.4rem; margin-top: 0.8rem; }
.tier-label { font-family: 'Outfit', sans-serif; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
.bar-bg { background: #131a2e; border-radius: 6px; height: 7px; margin: 6px 0; }
.bar-fg { height: 7px; border-radius: 6px; }
.conf-badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.05em; font-family: 'Outfit', sans-serif; margin-left: 8px; vertical-align: middle; }
.conf-h { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.conf-m { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.conf-l { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.risk { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); color: #b45309; border-radius: 7px; padding: 6px 10px; font-size: 0.78rem; font-weight: 600; margin-top: 7px; font-family: 'Outfit', sans-serif; }

.fu-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 0.4rem; }
.fu-title { font-family: 'Outfit', sans-serif; font-size: 0.92rem; font-weight: 700; color: #0b0f19; margin-bottom: 7px; }
.fu-row { display: flex; gap: 20px; font-size: 0.78rem; color: #64748b; flex-wrap: wrap; font-family: 'Outfit', sans-serif; }
.fu-row span strong { color: #1e293b; }

.model-card { background: #0b1221; border: 1px solid #1a2540; border-radius: 10px; padding: 0.85rem 0.95rem; height: 100%; }

.streamlit-expanderHeader { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; font-size: 0.88rem !important; }
.stCaption { font-family: 'Outfit', sans-serif !important; }
.stAlert { font-family: 'Outfit', sans-serif !important; }
.stDownloadButton button { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; }
.stFormSubmitButton button { font-family: 'Outfit', sans-serif !important; font-weight: 700 !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────
PRIORITY_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
SAMPLE_CSV = "leads_main_1000_final.csv"

CAT_COLS = ["email_response","cart_activity","last_activity","budget_level",
            "company_size","previous_interaction","previous_outcome","lead_source"]
NUM_FEATS = ["number_of_visits","time_spent_on_website","engagement_score",
             "click_rate","email_open_rate","inactivity_period"]
KNOWN_VALUES = {
    "email_response":       ["Clicked","Replied","No_Response","Opened"],
    "cart_activity":        ["Yes","No"],
    "last_activity":        ["Content_Download","Webinar","Demo_Request","Page_View","Pricing_Page","Form_Submit"],
    "budget_level":         ["Low","Medium","High"],
    "company_size":         ["Small","Medium","Large","Enterprise"],
    "previous_interaction": ["Yes","No"],
    "previous_outcome":     ["In_Progress","No_Contact","No_Response","Success","Failure"],
    "lead_source":          ["Social","Referral","Paid_Ads","Website","Email_Campaign","Organic_Search"],
}
FOLLOW_UP = {
    "High":   {"icon":"🔴","timeline":"Within 24 hours","channel":"Phone Call + Email","days":1,
               "msg":"Immediate outreach required — highest conversion potential."},
    "Medium": {"icon":"🟡","timeline":"Within 3–5 days","channel":"Email + LinkedIn","days":4,
               "msg":"Schedule follow-up this week. Nurture with relevant content."},
    "Low":    {"icon":"🟢","timeline":"Within 2 weeks","channel":"Email Newsletter","days":14,
               "msg":"Add to drip campaign. Monitor engagement signals before direct outreach."},
}
ALL_MODELS = {
    "Random Forest":       (RandomForestClassifier,      {"n_estimators":200,"random_state":42,"n_jobs":-1}),
    "Gradient Boosting":   (GradientBoostingClassifier,  {"n_estimators":150,"random_state":42}),
    "Logistic Regression": (LogisticRegression,           {"max_iter":1000,"random_state":42}),
    "Extra Trees":         (ExtraTreesClassifier,         {"n_estimators":200,"random_state":42,"n_jobs":-1}),
    "AdaBoost":            (AdaBoostClassifier,           {"n_estimators":100,"random_state":42}),
    "Decision Tree":       (DecisionTreeClassifier,       {"random_state":42,"max_depth":12}),
    "K-Nearest Neighbors": (KNeighborsClassifier,         {"n_neighbors":7}),
    "Naive Bayes":         (GaussianNB,                   {}),
}

# ── SESSION STATE ────────────────────────────────────────
DEFAULTS = {
    "trained":False,"df":None,"model":None,"scaler":None,"encoders":{},
    "feature_cols":[],"target_le":None,"report":None,"cm":None,"auc":None,
    "accuracy":None,"model_name":None,"logs":[],"comparison":None,"src_mode":None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── ML FUNCTIONS ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    drop = [c for c in ["converted","lead_score","recommended_followup_days","conversion_likelihood"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    feat_cols = (
        [c for c in NUM_FEATS if c in df.columns] +
        [c + "_enc" for c in CAT_COLS if c in df.columns]
    )
    return df, encoders, feat_cols


def train_models(df: pd.DataFrame, feat_cols: list):
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["lead_category"].astype(str))
    X = df[feat_cols].fillna(0)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    progress = st.progress(0, text="Training models…")
    n = len(ALL_MODELS)
    for i, (name, (cls, kw)) in enumerate(ALL_MODELS.items()):
        progress.progress(i / n, text=f"Training {name}…")
        try:
            mdl = cls(**kw)
            mdl.fit(Xtr, ytr)
            yp = mdl.predict(Xte)
            yprob = mdl.predict_proba(Xte)
            acc = accuracy_score(yte, yp)
            f1  = f1_score(yte, yp, average="macro")
            rep = classification_report(yte, yp, target_names=le_target.classes_, output_dict=True)
            cm  = confusion_matrix(yte, yp)
            try:
                auc = (roc_auc_score(yte, yprob[:,1]) if len(le_target.classes_)==2
                       else roc_auc_score(yte, yprob, multi_class="ovr", average="macro"))
            except Exception:
                auc = None
            results[name] = {"model":mdl,"report":rep,"cm":cm,"auc":auc,"accuracy":acc,"f1":f1}
        except Exception as e:
            results[name] = {"model":None,"report":None,"cm":None,"auc":None,"accuracy":0.0,"f1":0.0}
    progress.progress(1.0, text="Done!")
    progress.empty()

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best = results[best_name]

    comparison = pd.DataFrame([{
        "Algorithm": name,
        "Accuracy":  f"{r['accuracy']*100:.2f}%",
        "AUC-ROC":   f"{r['auc']:.4f}" if r["auc"] else "—",
        "F1 (Macro)":f"{r['f1']:.4f}",
        "Best":      "✅" if name == best_name else "",
        "_acc": r["accuracy"], "_auc": r["auc"] or 0, "_f1": r["f1"],
    } for name, r in results.items()])

    return (best["model"], sc, le_target, best["report"], best["cm"],
            best["auc"], best["accuracy"], best_name, comparison)


def predict_all_oof(df: pd.DataFrame, best_model_cls, best_model_kw, sc, feat_cols, le_target):
    """
    Out-of-Fold (OOF) predictions — every row is scored by a model
    that was NOT trained on it, giving honest, realistic probabilities
    instead of inflated in-sample recall scores.
    """
    df = df.copy()
    y = le_target.transform(df["lead_category"].astype(str))
    X = df[feat_cols].fillna(0)
    Xs = sc.transform(X)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probas = np.zeros((len(df), len(le_target.classes_)))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xs, y)):
        mdl_fold = best_model_cls(**best_model_kw)
        mdl_fold.fit(Xs[tr_idx], y[tr_idx])
        oof_probas[val_idx] = mdl_fold.predict_proba(Xs[val_idx])

    oof_preds = oof_probas.argmax(axis=1)
    df["predicted_category"]    = le_target.inverse_transform(oof_preds)
    df["conversion_likelihood"] = oof_probas.max(axis=1).round(4)
    df["lead_score"]            = (df["conversion_likelihood"] * 100).round(0).astype(int)
    return df


def add_followup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _row(score):
        if   score >= 70: return ("🔴 Immediate (24 hrs)", "Phone Call + Email", 1)
        elif score >= 40: return ("🟡 Within 3–5 days",    "Email + LinkedIn",   4)
        else:             return ("🟢 Within 2 weeks",      "Email Newsletter",  14)
    df["followup_timeline"] = df["lead_score"].apply(lambda s: _row(s)[0])
    df["followup_channel"]  = df["lead_score"].apply(lambda s: _row(s)[1])
    df["followup_date"]     = df["lead_score"].apply(
        lambda s: (datetime.today() + timedelta(days=int(_row(s)[2]))).strftime("%Y-%m-%d"))
    return df


def predict_single(inputs, mdl, sc, encoders, feat_cols, le_target):
    row = {}
    for col, le in encoders.items():
        val = inputs.get(col, le.classes_[0])
        try:    row[col + "_enc"] = int(le.transform([str(val)])[0])
        except: row[col + "_enc"] = 0
    for col in NUM_FEATS:
        row[col] = float(inputs.get(col, 0))
    X  = pd.DataFrame([{k: row.get(k, 0) for k in feat_cols}])
    Xs = sc.transform(X)
    pred   = mdl.predict(Xs)[0]
    probas = mdl.predict_proba(Xs)[0]
    label  = le_target.inverse_transform([pred])[0]
    prob_dict = {le_target.inverse_transform([i])[0]: round(float(p)*100, 1) for i, p in enumerate(probas)}
    score = int(max(probas) * 100)
    return label, prob_dict, score


def explain_lead(inputs):
    reasons = []
    v, t = inputs.get("number_of_visits",0), inputs.get("time_spent_on_website",0)
    inact = inputs.get("inactivity_period", 99)
    if   v >= 10: reasons.append(f"Very high visit count ({v} visits) — strong buying interest")
    elif v >=  5: reasons.append(f"Good visit frequency ({v} visits)")
    if t >= 20: reasons.append(f"Significant time on site ({t:.0f} min)")
    if inputs.get("cart_activity") == "Yes":    reasons.append("Added items to cart — high purchase intent 🛒")
    if inputs.get("email_response") == "Replied": reasons.append("Actively replied to emails — highly engaged")
    elif inputs.get("email_response") == "Opened": reasons.append("Opens emails — shows baseline interest")
    last = inputs.get("last_activity","")
    if last in ["Demo_Request","Pricing_Page","Form_Submit"]:
        reasons.append(f"High-intent last activity: {last.replace('_',' ')}")
    if inputs.get("budget_level") == "High":      reasons.append("High budget — strong conversion capability")
    if inputs.get("previous_outcome") == "Success": reasons.append("Previous interaction was successful — warm prospect")
    if inact <= 7: reasons.append(f"Recently active ({inact} days ago) — lead is hot")
    return reasons or ["Low engagement signals across all behavioral dimensions"]


def get_confidence(score):
    if   score >= 75: return "High Confidence",     "conf-h"
    elif score >= 45: return "Moderate Confidence", "conf-m"
    else:             return "Low Confidence",       "conf-l"


def get_risk(inputs):
    inact = inputs.get("inactivity_period", 0)
    if inact >= 45:
        return f"⚠️ Lead may be cold — {inact} days of inactivity"
    if inputs.get("email_response") == "No_Response" and inputs.get("cart_activity") == "No":
        return "⚠️ No email engagement and no cart activity — low intent signals"
    return None


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="All Leads")
        for tier in ["High","Medium","Low"]:
            if "lead_category" in df.columns:
                sub = df[df["lead_category"] == tier]
                if len(sub):
                    sub.to_excel(w, index=False, sheet_name=f"{tier} Priority")
    return buf.getvalue()


# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.8rem 0 0.4rem'>
      <div style='font-family:Outfit,sans-serif;font-size:1.05rem;font-weight:700;color:#e2eaf8;'>🎯 Lead Qualifier</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#1d3a5a;margin-top:2px;letter-spacing:0.1em;'>MD02 · SALES AI</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Data Source**")
    src_mode = st.radio("", ["Sample Dataset","Upload CSV"], label_visibility="collapsed")

    if src_mode != st.session_state.src_mode:
        for k, v in DEFAULTS.items():
            if k != "logs":
                st.session_state[k] = v
        st.session_state.src_mode = src_mode

    uploaded = None
    if src_mode == "Upload CSV":
        uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filters**  *(applied after training)*")
    filter_priority = st.multiselect(
        "Priority", ["High","Medium","Low"],
        default=["High","Medium","Low"], label_visibility="collapsed"
    )
    filter_score = st.slider("Min Lead Score", 0, 100, 0, 5, label_visibility="collapsed")

    st.markdown("---")
    run_btn = st.button("🚀  Analyze Leads", use_container_width=True, type="primary")

    if st.session_state.trained:
        acc = st.session_state.accuracy or 0
        auc = st.session_state.auc or 0
        st.markdown("---")
        _df = st.session_state.df
        _nh = int((_df["lead_category"]=="High").sum())
        _nm = int((_df["lead_category"]=="Medium").sum())
        _nl = int((_df["lead_category"]=="Low").sum())
        _avg_score = _df["lead_score"].mean() if "lead_score" in _df.columns else 0
        st.markdown(f"""
        <div style='background:#0b1221;border-radius:10px;padding:0.9rem 1rem;border:1px solid #1a2540;'>
          <div style='font-size:0.55rem;text-transform:uppercase;letter-spacing:0.11em;color:#1d3a5a;font-weight:700;margin-bottom:9px;font-family:Outfit,sans-serif;'>✅ Model Active</div>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px;align-items:center;'>
            <span style='font-size:0.7rem;color:#3d5a78;font-family:Outfit,sans-serif;'>Accuracy</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#34d399;font-weight:600;'>{acc*100:.1f}%</span>
          </div>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px;align-items:center;'>
            <span style='font-size:0.7rem;color:#3d5a78;font-family:Outfit,sans-serif;'>AUC-ROC</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#60a5fa;font-weight:600;'>{auc:.4f}</span>
          </div>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px;align-items:center;'>
            <span style='font-size:0.7rem;color:#3d5a78;font-family:Outfit,sans-serif;'>Avg Score</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#f59e0b;font-weight:600;'>{_avg_score:.0f}/100</span>
          </div>
          <div style='display:flex;justify-content:space-between;margin-bottom:10px;align-items:center;'>
            <span style='font-size:0.7rem;color:#3d5a78;font-family:Outfit,sans-serif;'>Best Model</span>
            <span style='font-size:0.68rem;color:#7aa8cc;font-weight:600;font-family:Outfit,sans-serif;'>{st.session_state.model_name or "—"}</span>
          </div>
          <div style='border-top:1px solid #1a2540;padding-top:8px;'>
            <div style='font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;color:#1d3a5a;font-weight:700;margin-bottom:6px;font-family:Outfit,sans-serif;'>Lead Breakdown</div>
            <div style='display:flex;gap:6px;'>
              <div style='flex:1;background:#1a0a0a;border-radius:6px;padding:4px 6px;text-align:center;border:1px solid #3a1a1a;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#ef4444;font-weight:700;'>{_nh}</div>
                <div style='font-size:0.55rem;color:#7a3a3a;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;font-family:Outfit,sans-serif;'>High</div>
              </div>
              <div style='flex:1;background:#1a140a;border-radius:6px;padding:4px 6px;text-align:center;border:1px solid #3a2a0a;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#f59e0b;font-weight:700;'>{_nm}</div>
                <div style='font-size:0.55rem;color:#7a5a2a;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;font-family:Outfit,sans-serif;'>Med</div>
              </div>
              <div style='flex:1;background:#0a1a0e;border-radius:6px;padding:4px 6px;text-align:center;border:1px solid #0a3a1a;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#10b981;font-weight:700;'>{_nl}</div>
                <div style='font-size:0.55rem;color:#1a5a3a;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;font-family:Outfit,sans-serif;'>Low</div>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── LOAD DATA ────────────────────────────────────────────
df_raw = None

if src_mode == "Upload CSV":
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            df_raw.columns = df_raw.columns.str.strip().str.lower()
        except Exception as e:
            st.error(f"Could not parse file: {e}")
else:
    try:
        df_raw = pd.read_csv(SAMPLE_CSV)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
    except FileNotFoundError:
        st.error("⚠️ Sample file not found. Place `leads_main_1000_final.csv` in the same folder.")


# ── TRAIN ────────────────────────────────────────────────
if run_btn:
    if df_raw is None:
        st.warning("⚠️ Please upload a CSV file first, then click Analyze Leads.")
    else:
        df_p, enc, fcols = preprocess(df_raw)

        # Step 1: Train all 8 models on 80/20 split to pick the best
        mdl, sc, le_t, rep, cm, auc, acc, best_name, comparison = train_models(df_p, fcols)

        # Step 2: Score the FULL dataset using Out-of-Fold predictions
        # so every row gets a score from a model that never saw it → no inflated 100% values
        best_cls, best_kw = ALL_MODELS[best_name]
        with st.spinner("Generating honest out-of-fold scores for all leads…"):
            df_p = predict_all_oof(df_p, best_cls, best_kw, sc, fcols, le_t)

        df_p = add_followup(df_p)
        st.session_state.update({
            "trained":True,"df":df_p,"model":mdl,"scaler":sc,
            "encoders":enc,"feature_cols":fcols,"target_le":le_t,
            "report":rep,"cm":cm,"auc":auc,"accuracy":acc,
            "model_name":best_name,"comparison":comparison,
        })
        st.success(f"✅ Best model: **{best_name}** · Accuracy: {acc*100:.1f}% · All 8 algorithms compared.")


# ── HERO ─────────────────────────────────────────────────
acc_badge = ""
if st.session_state.trained:
    acc_badge = f'<span class="acc-badge">✓ {st.session_state.accuracy*100:.1f}% · {st.session_state.model_name}</span>'

st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:flex-start;justify-content:space-between">
    <div>
      <div><span class="badge">MD02</span>{acc_badge}</div>
      <h1>Sales Lead Qualifier System</h1>
      <p class="sub">AI-Powered Sales Intelligence · Prioritize leads · Predict conversions · Maximize efficiency</p>
      <div class="pills">
        <span class="pill">✔ Real-time Lead Scoring</span>
        <span class="pill">✔ AI-driven Prioritization</span>
        <span class="pill">✔ Smart Follow-up Recommendations</span>
        <span class="pill">✔ 8-Model Auto Selection</span>
      </div>
    </div>
    <div style="font-size:3rem;opacity:0.07;line-height:1;padding-top:6px">🎯</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── PRE-TRAIN STATE ───────────────────────────────────────
if df_raw is None:
    if src_mode == "Upload CSV":
        st.markdown('<div class="ibox blue">👈 Upload a CSV file in the sidebar, then click <strong>🚀 Analyze Leads</strong>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ibox blue">👈 Click <strong>🚀 Analyze Leads</strong> in the sidebar to get started with the sample dataset.</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state.trained:
    st.markdown(f'<div class="ibox amber">📊 Data loaded — <strong>{len(df_raw):,} rows · {len(df_raw.columns)} columns</strong>. Click <strong>🚀 Analyze Leads</strong> to run all 8 ML algorithms.</div>', unsafe_allow_html=True)
    with st.expander("👁 Preview data"):
        st.dataframe(df_raw.head(30), use_container_width=True)
    st.stop()

# ── FILTERED VIEW ─────────────────────────────────────────
df: pd.DataFrame = st.session_state.df
df_f = df[
    (df["lead_category"].isin(filter_priority)) &
    (df["lead_score"] >= filter_score)
].copy()

# ── DATASET SUMMARY STRIP ─────────────────────────────────
_nh_all = int((df["lead_category"]=="High").sum())
_nm_all = int((df["lead_category"]=="Medium").sum())
_nl_all = int((df["lead_category"]=="Low").sum())
_avg_conv = df["conversion_likelihood"].mean()*100 if "conversion_likelihood" in df.columns else 0
_avg_eng  = df["engagement_score"].mean() if "engagement_score" in df.columns else 0
st.markdown(f"""
<div style='display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap;'>
  <div style='background:#fff;border:1px solid #e5ecf5;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#94a3b8;font-family:Outfit,sans-serif;'>Total Leads</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#0b0f19;'>{len(df):,}</span>
  </div>
  <div style='background:#fff7f7;border:1px solid #fecaca;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#ef4444;font-family:Outfit,sans-serif;'>🔴 High</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#dc2626;'>{_nh_all:,}</span>
  </div>
  <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#f59e0b;font-family:Outfit,sans-serif;'>🟡 Medium</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#d97706;'>{_nm_all:,}</span>
  </div>
  <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#10b981;font-family:Outfit,sans-serif;'>🟢 Low</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#059669;'>{_nl_all:,}</span>
  </div>
  <div style='background:#fff;border:1px solid #e5ecf5;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#94a3b8;font-family:Outfit,sans-serif;'>Avg Conv.</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#2563eb;'>{_avg_conv:.0f}%</span>
  </div>
  <div style='background:#fff;border:1px solid #e5ecf5;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:8px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
    <span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:#94a3b8;font-family:Outfit,sans-serif;'>Avg Engagement</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;font-weight:700;color:#8b5cf6;'>{_avg_eng:.1f}</span>
  </div>
  {"<div style='background:#fef9c3;border:1px solid #fde68a;border-radius:9px;padding:0.5rem 1rem;display:flex;align-items:center;gap:6px;'><span style='font-size:0.7rem;color:#92400e;font-family:Outfit,sans-serif;font-weight:500;'>⚠️ Showing <strong>{len(df_f):,}</strong> of {len(df):,} leads (filtered)</span></div>" if len(df_f) < len(df) else ""}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
T = st.tabs([
    "📊 Overview","🏆 Priority Leads","🔮 New Lead",
    "🤖 Model Insights","📅 Action Plan","📝 Activity Log","📤 Export",
])


# ── TAB 1: OVERVIEW ───────────────────────────────────────
with T[0]:
    if len(df_f) == 0:
        st.markdown('<div class="ibox amber">⚠️ No leads match current filters. Adjust the sidebar filters.</div>', unsafe_allow_html=True)
    else:
        total    = len(df_f)
        n_high   = int((df_f["lead_category"]=="High").sum())
        n_med    = int((df_f["lead_category"]=="Medium").sum())
        n_low    = int((df_f["lead_category"]=="Low").sum())
        avg_score = df_f["lead_score"].mean()
        avg_conv  = df_f["conversion_likelihood"].mean() * 100 if "conversion_likelihood" in df_f else 0
        pct = lambda n: f"{n/total*100:.1f}%"

        st.markdown('<div class="ibox dark">This system identifies high-value leads using behavioral analysis so sales teams focus effort where conversion potential is highest.</div>', unsafe_allow_html=True)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        for col, css, icon, lbl, val, sub in [
            (c1,"blue",  "📊","Total Leads",      f"{total:,}",      "current filter"),
            (c2,"high",  "🔴","High Priority",    f"{n_high:,}",     pct(n_high)),
            (c3,"med",   "🟡","Medium Priority",  f"{n_med:,}",      pct(n_med)),
            (c4,"low",   "🟢","Nurture",          f"{n_low:,}",      pct(n_low)),
            (c5,"purple","🎯","Avg Score",        f"{avg_score:.0f}","out of 100"),
            (c6,"teal",  "📈","Est. Conv. Rate",  f"{avg_conv:.0f}%","model estimate"),
        ]:
            col.markdown(f'<div class="kpi {css}"><div class="k-lbl">{icon} {lbl}</div><div class="k-val">{val}</div><div class="k-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        r1, r2 = st.columns(2)
        with r1:
            fig = px.pie(df_f, names="lead_category", title="Lead Quality Distribution",
                         color="lead_category", color_discrete_map=PRIORITY_COLORS, hole=0.55)
            fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03]*3)
            fig.update_layout(height=300, margin=dict(t=45,b=5,l=5,r=5), showlegend=False,
                              font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            fig2 = px.histogram(df_f, x="engagement_score", color="lead_category", nbins=30,
                                title="Engagement Score by Priority", color_discrete_map=PRIORITY_COLORS,
                                labels={"engagement_score":"Engagement Score"})
            fig2.update_layout(height=300, margin=dict(t=45,b=5,l=5,r=5), bargap=0.06,
                               font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
            st.plotly_chart(fig2, use_container_width=True)

        r3, r4 = st.columns(2)
        with r3:
            if "lead_source" in df_f.columns:
                src = df_f.groupby(["lead_source","lead_category"]).size().reset_index(name="n")
                fig3 = px.bar(src, x="lead_source", y="n", color="lead_category",
                              title="Leads by Source & Priority", barmode="stack",
                              color_discrete_map=PRIORITY_COLORS, labels={"lead_source":"Source","n":"Count"})
                fig3.update_layout(height=280, margin=dict(t=45,b=5,l=5,r=5),
                                   font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
                st.plotly_chart(fig3, use_container_width=True)
        with r4:
            if "company_size" in df_f.columns:
                cs = df_f.groupby(["company_size","lead_category"]).size().reset_index(name="n")
                fig4 = px.bar(cs, x="company_size", y="n", color="lead_category",
                              title="Leads by Company Size", barmode="group",
                              color_discrete_map=PRIORITY_COLORS,
                              labels={"company_size":"Size","n":"Count"},
                              category_orders={"company_size":["Small","Medium","Large","Enterprise"]})
                fig4.update_layout(height=280, margin=dict(t=45,b=5,l=5,r=5),
                                   font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
                st.plotly_chart(fig4, use_container_width=True)

        r5, r6 = st.columns(2)
        with r5:
            fig5 = px.box(df_f, x="lead_category", y="engagement_score", color="lead_category",
                          title="Engagement Score by Priority", color_discrete_map=PRIORITY_COLORS,
                          points="outliers", category_orders={"lead_category":["High","Medium","Low"]})
            fig5.update_layout(height=280, margin=dict(t=45,b=5,l=5,r=5), showlegend=False,
                               font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
            st.plotly_chart(fig5, use_container_width=True)
        with r6:
            if "budget_level" in df_f.columns:
                bl = df_f.groupby(["budget_level","lead_category"]).size().reset_index(name="n")
                fig6 = px.bar(bl, x="budget_level", y="n", color="lead_category",
                              title="Budget Level vs Priority", barmode="group",
                              color_discrete_map=PRIORITY_COLORS,
                              labels={"budget_level":"Budget","n":"Count"},
                              category_orders={"budget_level":["Low","Medium","High"]})
                fig6.update_layout(height=280, margin=dict(t=45,b=5,l=5,r=5),
                                   font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
                st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="sec">📈 Engagement vs Lead Score</div>', unsafe_allow_html=True)
        hover_extra = [c for c in ["company_size","lead_source","budget_level"] if c in df_f.columns]
        sc_size = "number_of_visits" if "number_of_visits" in df_f.columns else None
        fig_sc = px.scatter(
            df_f.sample(min(500, len(df_f)), random_state=42),
            x="engagement_score", y="lead_score", color="lead_category",
            size=sc_size, opacity=0.55, hover_data=hover_extra,
            color_discrete_map=PRIORITY_COLORS,
            title="Engagement Score vs Lead Score",
            labels={"engagement_score":"Engagement Score","lead_score":"Lead Score (0–100)"}
        )
        fig_sc.update_layout(height=360, margin=dict(t=45,b=5,l=5,r=5),
                             font=dict(family="Outfit", size=12), legend_title_text="Priority",
                             title_font=dict(family="Outfit", size=14, color="#1e293b"))
        st.plotly_chart(fig_sc, use_container_width=True)


# ── TAB 2: PRIORITY LEADS ─────────────────────────────────
with T[1]:
    st.markdown('<div class="sec">🏆 Prioritized Lead Rankings</div>', unsafe_allow_html=True)

    if len(df_f) == 0:
        st.markdown('<div class="ibox amber">⚠️ No leads match current filters.</div>', unsafe_allow_html=True)
    else:
        rc1, rc2, rc3 = st.columns([3,2,2])
        search  = rc1.text_input("🔍 Search", placeholder="Filter by any value…", label_visibility="collapsed")
        sort_by = rc2.selectbox("Sort by",
            [c for c in ["lead_score","conversion_likelihood","engagement_score","number_of_visits","time_spent_on_website"] if c in df_f.columns],
            label_visibility="collapsed")
        top_n = rc3.selectbox("Show", [50,100,250,500,"All"], label_visibility="collapsed")

        SHOW_COLS = [c for c in [
            "lead_category","predicted_category","lead_score","conversion_likelihood",
            "engagement_score","number_of_visits","time_spent_on_website",
            "email_open_rate","click_rate","company_size","budget_level",
            "lead_source","cart_activity","email_response",
            "followup_timeline","followup_channel","followup_date",
        ] if c in df_f.columns]

        df_rank = df_f[SHOW_COLS].sort_values(sort_by, ascending=False).reset_index(drop=True)
        df_rank.index += 1

        if search:
            mask = df_rank.astype(str).apply(lambda r: r.str.contains(search, case=False, na=False)).any(axis=1)
            df_rank = df_rank[mask]
        if top_n != "All":
            df_rank = df_rank.head(int(top_n))

        def _style_priority(val):
            colors = {"High":"#dc2626","Medium":"#d97706","Low":"#16a34a"}
            c = colors.get(str(val), "")
            return f"font-weight:800;color:{c};" if c else "font-weight:600;"

        fmt = {}
        if "conversion_likelihood" in df_rank.columns: fmt["conversion_likelihood"] = "{:.1%}"
        if "lead_score"            in df_rank.columns: fmt["lead_score"]            = "{:.0f}"
        if "engagement_score"      in df_rank.columns: fmt["engagement_score"]      = "{:.1f}"

        style_cols = [c for c in ["lead_category","predicted_category"] if c in df_rank.columns]
        styled = df_rank.style.format(fmt)
        for col in style_cols:
            styled = styled.map(_style_priority, subset=[col])

        st.dataframe(styled, use_container_width=True, height=500)
        st.caption(f"Showing {len(df_rank):,} leads · sorted by {sort_by}")
        st.markdown('<div style="font-size:0.66rem;color:#94a3b8;font-style:italic;">💡 Lead Score: out-of-fold model confidence — each row scored by a model that never trained on it, giving honest probability estimates.</div>', unsafe_allow_html=True)


# ── TAB 3: NEW LEAD PREDICTION ────────────────────────────
with T[2]:
    st.markdown('<div class="sec">🔮 Single Lead Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibox blue">Enter lead attributes and click <strong>Generate Prediction</strong> for an instant AI classification.</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Engagement Signals**")
            visits   = st.slider("Number of Visits",         1,   20,    6)
            time_sp  = st.slider("Time on Site (min)",       1.0, 60.0,  15.0, 0.5)
            inact    = st.slider("Inactivity Period (days)", 1,   75,    20)
            click_r  = st.slider("Click Rate",               0.0, 1.0,   0.4,  0.05)
            email_or = st.slider("Email Open Rate",          0.0, 1.0,   0.35, 0.05)
        with col2:
            st.markdown("**Behavioral Data**")
            email_resp = st.selectbox("Email Response",       KNOWN_VALUES["email_response"])
            cart       = st.selectbox("Cart Activity",        KNOWN_VALUES["cart_activity"])
            last_act   = st.selectbox("Last Activity",        KNOWN_VALUES["last_activity"])
            prev_int   = st.selectbox("Previous Interaction", KNOWN_VALUES["previous_interaction"])
            prev_out   = st.selectbox("Previous Outcome",     KNOWN_VALUES["previous_outcome"])
        with col3:
            st.markdown("**Business Profile**")
            budget   = st.selectbox("Budget Level",  KNOWN_VALUES["budget_level"])
            company  = st.selectbox("Company Size",  KNOWN_VALUES["company_size"])
            lead_src = st.selectbox("Lead Source",   KNOWN_VALUES["lead_source"])
        submitted = st.form_submit_button("🔮 Generate Prediction", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "number_of_visits": visits, "time_spent_on_website": time_sp,
            "engagement_score": visits * time_sp, "click_rate": click_r,
            "email_open_rate": email_or, "inactivity_period": inact,
            "email_response": email_resp, "cart_activity": cart,
            "last_activity": last_act, "budget_level": budget,
            "company_size": company, "previous_interaction": prev_int,
            "previous_outcome": prev_out, "lead_source": lead_src,
        }
        label, prob_dict, score = predict_single(
            inputs, st.session_state.model, st.session_state.scaler,
            st.session_state.encoders, st.session_state.feature_cols, st.session_state.target_le)
        reasons  = explain_lead(inputs)
        risk_msg = get_risk(inputs)
        conf_label, conf_cls = get_confidence(score)
        color_map = {"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"}
        tier_color = color_map.get(label,"#888")
        fu = FOLLOW_UP.get(label, FOLLOW_UP["Low"])
        fu_date = (datetime.today() + timedelta(days=fu["days"])).strftime("%B %d, %Y")
        top_prob = max(prob_dict.values())
        prob_sorted = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        prob_bars_html = ""
        for tier, pv in prob_sorted.items():
            c = color_map.get(tier,"#888")
            prob_bars_html += f"""
            <div style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                <span style='font-size:0.72rem;color:#7aa8cc;'>{tier}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{c};font-weight:600;'>{pv:.1f}%</span>
              </div>
              <div class='bar-bg'><div class='bar-fg' style='width:{pv}%;background:{c};'></div></div>
            </div>"""

        res_col, det_col = st.columns([1,1.2])
        with res_col:
            st.markdown(f"""
            <div class="pred-box">
              <div style='color:#3d5a78;font-size:0.58rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:4px;'>Prediction Result</div>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                <span class="tier-label" style='color:{tier_color};'>{label} Priority</span>
                <span class="conf-badge {conf_cls}">{conf_label}</span>
              </div>
              <div class='bar-bg'><div class='bar-fg' style='width:{score}%;background:{tier_color};'></div></div>
              <div style='display:flex;justify-content:space-between;margin-bottom:16px;'>
                <span style='font-size:0.67rem;color:#1d3a5a;'>Lead Score</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{tier_color};font-weight:700;'>{score} / 100</span>
              </div>
              <div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:#1d3a5a;font-weight:700;margin-bottom:5px;'>Conversion Likelihood</div>
              <div style='font-size:1.5rem;font-family:Outfit,sans-serif;font-weight:700;color:{tier_color};margin-bottom:12px;'>{top_prob:.0f}%</div>
              <div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:#1d3a5a;font-weight:700;margin-bottom:7px;'>Class Breakdown</div>
              {prob_bars_html}
              <hr style='border:none;border-top:1px solid #1a2540;margin:12px 0;'>
              <div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:#1d3a5a;font-weight:700;margin-bottom:7px;'>Recommended Action</div>
              <div style='background:#0b1221;border-radius:9px;padding:0.75rem 0.9rem;border:1px solid #1a2540;'>
                <div style='font-size:0.85rem;color:#e2eaf8;font-weight:600;font-family:Outfit,sans-serif;margin-bottom:6px;'>{fu["channel"]}</div>
                <div style='font-size:0.7rem;color:#3d5a78;margin-bottom:2px;'>⏰ <strong style='color:#7aa8cc;'>{fu["timeline"]}</strong></div>
                <div style='font-size:0.7rem;color:#3d5a78;'>📅 Follow up by: <strong style='color:#60a5fa;'>{fu_date}</strong></div>
              </div>
            </div>""", unsafe_allow_html=True)

        with det_col:
            st.markdown("**🧠 Why this prediction?**")
            for r in reasons:
                flag = "⚠️" if any(w in r.lower() for w in ["cold","risk","low"]) else "✅"
                st.markdown(f"- {flag} {r}")
            if risk_msg:
                st.markdown(f'<div class="risk">{risk_msg}</div>', unsafe_allow_html=True)
            st.markdown("**📊 Input Summary**")
            summary = pd.DataFrame({
                "Feature": ["Visits","Time on Site","Inactivity","Cart","Email Response","Last Activity","Budget","Company"],
                "Value":   [visits, f"{time_sp} min", f"{inact} days", cart, email_resp, last_act.replace("_"," "), budget, company]
            }).set_index("Feature")
            st.dataframe(summary, use_container_width=True)


# ── TAB 4: MODEL INSIGHTS ─────────────────────────────────
with T[3]:
    rep      = st.session_state.report
    cm_val   = st.session_state.cm
    auc      = st.session_state.auc
    acc      = st.session_state.accuracy
    mdl      = st.session_state.model
    fcols    = st.session_state.feature_cols
    le_t     = st.session_state.target_le
    cmp_df   = st.session_state.comparison
    best_name = st.session_state.model_name

    st.markdown('<div class="sec">🏅 All 8 Algorithm Comparison</div>', unsafe_allow_html=True)

    if cmp_df is not None:
        ALGO_COLORS = {
            "Random Forest":"#3b82f6","Gradient Boosting":"#8b5cf6",
            "Logistic Regression":"#06b6d4","Extra Trees":"#f59e0b",
            "AdaBoost":"#ec4899","Decision Tree":"#10b981",
            "K-Nearest Neighbors":"#f97316","Naive Bayes":"#a78bfa",
        }
        algo_list = list(cmp_df.iterrows())
        for row_start in range(0, len(algo_list), 4):
            chunk = algo_list[row_start:row_start+4]
            cols  = st.columns(len(chunk))
            for ci, (_, row) in enumerate(chunk):
                alg = row["Algorithm"]
                is_best = alg == best_name
                color  = ALGO_COLORS.get(alg, "#888")
                border = "2px solid #34d399" if is_best else "1px solid #1a2540"
                badge  = '<span style="background:rgba(16,185,129,0.15);color:#34d399;font-size:0.52rem;font-weight:700;padding:1px 6px;border-radius:20px;font-family:JetBrains Mono,monospace;border:1px solid rgba(16,185,129,0.3);margin-left:5px;">BEST</span>' if is_best else ""
                cols[ci].markdown(f"""
                <div class="model-card" style="border:{border};">
                  <div style='font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;color:#1d3a5a;font-weight:700;margin-bottom:3px;'>Algorithm</div>
                  <div style='font-family:Outfit,sans-serif;font-size:0.78rem;font-weight:700;color:{color};margin-bottom:5px;line-height:1.3;'>{alg}{badge}</div>
                  <hr style='border:none;border-top:1px solid #1a2540;margin:6px 0;'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                    <span style='font-size:0.64rem;color:#3d5a78;'>Accuracy</span>
                    <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#e2eaf8;font-weight:700;'>{row["Accuracy"]}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                    <span style='font-size:0.64rem;color:#3d5a78;'>AUC-ROC</span>
                    <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#60a5fa;font-weight:600;'>{row["AUC-ROC"]}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;'>
                    <span style='font-size:0.64rem;color:#3d5a78;'>F1 Macro</span>
                    <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#a78bfa;font-weight:600;'>{row["F1 (Macro)"]}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

        fig_cmp = go.Figure()
        bar_colors = ["#34d399" if a == best_name else "#3b82f6" for a in cmp_df["Algorithm"]]
        fig_cmp.add_trace(go.Bar(name="Accuracy (%)", x=cmp_df["Algorithm"],
            y=cmp_df["_acc"]*100, marker_color=bar_colors,
            text=(cmp_df["_acc"]*100).round(1).astype(str)+"%", textposition="outside"))
        fig_cmp.add_trace(go.Bar(name="AUC-ROC (×100)", x=cmp_df["Algorithm"],
            y=cmp_df["_auc"]*100, marker_color="#8b5cf6", opacity=0.75))
        fig_cmp.add_trace(go.Bar(name="F1 Macro (×100)", x=cmp_df["Algorithm"],
            y=cmp_df["_f1"]*100, marker_color="#06b6d4", opacity=0.75))
        fig_cmp.update_layout(
            barmode="group", height=360,
            title="All Models — Accuracy · AUC · F1",
            margin=dict(t=55,b=25,l=10,r=10), font=dict(family="Outfit", size=12),
            title_font=dict(family="Outfit", size=14, color="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(range=[0,115]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown('<div class="sec">🤖 Best Model Detail</div>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.metric("Best Algorithm", best_name or "—")
    mc2.metric("Accuracy",       f"{acc*100:.1f}%" if acc else "—")
    mc3.metric("AUC-ROC",        f"{auc:.4f}" if auc else "—")
    mc4.metric("Test Split",     "20% holdout")
    mc5.metric("Train Samples",  f"{int(len(df)*0.8):,}")

    sub1, sub2, sub3 = st.tabs(["Feature Importance","Confusion Matrix","Classification Report"])
    with sub1:
        fi_vals = (mdl.feature_importances_ if hasattr(mdl,"feature_importances_")
                   else np.abs(mdl.coef_).mean(axis=0) if hasattr(mdl,"coef_") else None)
        if fi_vals is not None:
            fi = pd.DataFrame({"Feature":fcols,"Importance":fi_vals})
            fi = fi.sort_values("Importance", ascending=True).tail(20)
            fi["Feature"] = fi["Feature"].str.replace("_enc","",regex=False)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            title="Top Feature Importances", color="Importance",
                            color_continuous_scale="Blues")
            fig_fi.update_layout(height=500, margin=dict(t=45,b=5,l=5,r=5),
                                 coloraxis_showscale=False, font=dict(family="Outfit", size=12),
                                 title_font=dict(family="Outfit", size=14, color="#1e293b"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")
    with sub2:
        class_names = le_t.classes_.tolist()
        fig_cm = px.imshow(cm_val, text_auto=True, aspect="auto",
                           x=class_names, y=class_names,
                           title="Confusion Matrix", color_continuous_scale="Blues",
                           labels=dict(x="Predicted",y="Actual"))
        fig_cm.update_layout(height=400, font=dict(family="Outfit", size=12),
                             title_font=dict(family="Outfit", size=14, color="#1e293b"))
        st.plotly_chart(fig_cm, use_container_width=True)
    with sub3:
        rows = [{"Class":k,"Precision":round(v["precision"],3),"Recall":round(v["recall"],3),
                 "F1-Score":round(v["f1-score"],3),"Support":int(v.get("support",0))}
                for k,v in rep.items() if isinstance(v,dict)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── TAB 5: ACTION PLAN ────────────────────────────────────
with T[4]:
    st.markdown('<div class="sec">📅 Recommended Action Plan</div>', unsafe_allow_html=True)

    if len(df_f) == 0:
        st.markdown('<div class="ibox amber">⚠️ No leads match current filters.</div>', unsafe_allow_html=True)
    else:
        for tier in ["High","Medium","Low"]:
            if tier not in filter_priority:
                continue
            cfg = FOLLOW_UP[tier]
            sub = df_f[df_f["lead_category"] == tier]
            if len(sub) == 0:
                continue
            avg_sc = sub["engagement_score"].mean() if "engagement_score" in sub.columns else 0
            avg_ls = sub["lead_score"].mean()

            with st.expander(f"{cfg['icon']} **{tier} Priority** — {len(sub):,} leads · avg score {avg_ls:.0f} · avg engagement {avg_sc:.0f}", expanded=(tier=="High")):
                st.markdown(f"""
                <div class="fu-card">
                  <div class="fu-title">🚀 {cfg['channel']}</div>
                  <div class="fu-row">
                    <span>⏰ Timeline: <strong>{cfg['timeline']}</strong></span>
                    <span>📡 Channel: <strong>{cfg['channel']}</strong></span>
                  </div>
                  <p style='margin:7px 0 0;font-size:0.77rem;color:#64748b;'>{cfg['msg']}</p>
                </div>""", unsafe_allow_html=True)

                lc1, lc2 = st.columns(2)
                with lc1:
                    top_cols = [c for c in ["lead_score","conversion_likelihood","engagement_score",
                                            "company_size","lead_source","budget_level","followup_date"]
                                if c in sub.columns]
                    top8 = sub.sort_values("lead_score", ascending=False)[top_cols].head(8).reset_index(drop=True)
                    top8.index += 1
                    st.markdown("**Top leads in this tier**")
                    fmt_fu = {}
                    if "conversion_likelihood" in top8.columns: fmt_fu["conversion_likelihood"] = "{:.1%}"
                    if "lead_score"            in top8.columns: fmt_fu["lead_score"]            = "{:.0f}"
                    st.dataframe(top8.style.format(fmt_fu), use_container_width=True)
                with lc2:
                    if "lead_source" in sub.columns:
                        src_cnt = sub["lead_source"].value_counts().reset_index()
                        src_cnt.columns = ["Source","Count"]
                        fig_s = px.bar(src_cnt, x="Source", y="Count", title="Source Breakdown",
                                       color_discrete_sequence=[PRIORITY_COLORS[tier]])
                        fig_s.update_layout(height=260, margin=dict(t=35,b=5,l=5,r=5), showlegend=False)
                        st.plotly_chart(fig_s, use_container_width=True)

        if "followup_date" in df_f.columns:
            st.markdown('<div class="sec">📆 Follow-up Calendar</div>', unsafe_allow_html=True)
            tl = df_f.groupby(["followup_date","lead_category"]).size().reset_index(name="n")
            fig_tl = px.bar(tl, x="followup_date", y="n", color="lead_category",
                            title="Leads Due per Follow-up Date", color_discrete_map=PRIORITY_COLORS,
                            labels={"followup_date":"Date","n":"leads"})
            fig_tl.update_layout(height=280, margin=dict(t=45,b=5,l=5,r=5),
                                 font=dict(family="Outfit", size=12), title_font=dict(family="Outfit", size=14, color="#1e293b"))
            st.plotly_chart(fig_tl, use_container_width=True)


# ── TAB 6: ACTIVITY LOG ───────────────────────────────────
with T[5]:
    st.markdown('<div class="sec">📝 Track Lead Interactions</div>', unsafe_allow_html=True)

    with st.form("log_form", clear_on_submit=True):
        fc1,fc2,fc3,fc4 = st.columns([1.5,2,2,3])
        lead_num = fc1.number_input("Lead Row #", min_value=1, max_value=len(df), step=1, value=1)
        action   = fc2.selectbox("Action", ["Email Sent","Phone Call","LinkedIn Message",
                                            "Meeting Scheduled","Demo Completed","Proposal Sent",
                                            "Closed Won","Closed Lost"])
        outcome  = fc3.selectbox("Outcome", ["Positive","Neutral","No Response",
                                             "Not Interested","Callback Requested"])
        notes    = fc4.text_input("Notes", placeholder="Any context…")
        submitted_log = st.form_submit_button("➕ Log Interaction", type="primary")

    if submitted_log:
        row = df.iloc[int(lead_num) - 1]
        entry = {
            "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Lead #":       int(lead_num),
            "Priority":     str(row.get("lead_category", "—")),
            "Lead Score":   int(row.get("lead_score", 0)),
            "Conv. Likely": f"{float(row.get('conversion_likelihood', 0)):.1%}",
            "Action":       action,
            "Outcome":      outcome,
            "Notes":        notes,
        }
        st.session_state.logs.append(entry)
        st.success(f"✅ Logged: **{action}** → {outcome} for Lead #{lead_num}")

    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(log_df, use_container_width=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            ac = log_df["Action"].value_counts().reset_index()
            ac.columns = ["Action","Count"]
            fig_a = px.bar(ac, x="Action", y="Count", title="Actions Taken",
                           color_discrete_sequence=["#3b82f6"])
            fig_a.update_layout(height=260, showlegend=False, margin=dict(t=35,b=5,l=5,r=5))
            st.plotly_chart(fig_a, use_container_width=True)
        with lc2:
            oc = log_df["Outcome"].value_counts().reset_index()
            oc.columns = ["Outcome","Count"]
            fig_o = px.pie(oc, names="Outcome", values="Count", title="Outcome Breakdown", hole=0.45)
            fig_o.update_layout(height=260, margin=dict(t=35,b=5,l=5,r=5))
            st.plotly_chart(fig_o, use_container_width=True)

        st.download_button("📥 Download Activity Log",
                           data=log_df.to_csv(index=False).encode(),
                           file_name="activity_log.csv", mime="text/csv")
    else:
        st.markdown('<div class="ibox blue">No interactions logged yet. Use the form above to track lead activity.</div>', unsafe_allow_html=True)


# ── TAB 7: EXPORT ─────────────────────────────────────────
with T[6]:
    st.markdown('<div class="sec">📤 Export Qualified Leads</div>', unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    exp_tiers = ec1.multiselect("Priority tiers", ["High","Medium","Low"], default=["High","Medium"])
    exp_min   = ec2.slider("Min Lead Score", 0, 100, 0, 5)

    safe_cols = [c for c in df_f.columns if not c.endswith("_enc")]
    default_export = [c for c in [
        "lead_category","predicted_category","lead_score","conversion_likelihood",
        "engagement_score","number_of_visits","time_spent_on_website",
        "company_size","budget_level","lead_source","cart_activity",
        "email_response","followup_timeline","followup_channel","followup_date"
    ] if c in safe_cols]

    exp_cols = st.multiselect("Columns to export", safe_cols, default=default_export)

    df_exp = df_f[
        (df_f["lead_category"].isin(exp_tiers)) &
        (df_f["lead_score"] >= exp_min)
    ]
    valid = [c for c in exp_cols if c in df_exp.columns]

    if len(df_exp) == 0:
        st.markdown('<div class="ibox amber">⚠️ No leads match the current export filters.</div>', unsafe_allow_html=True)
    elif not valid:
        st.warning("Select at least one column to preview and export.")
    else:
        st.markdown(f'<div class="ibox green">✅ <strong>{len(df_exp):,} leads</strong> ready for export ({", ".join(exp_tiers)} priority · min score {exp_min}).</div>', unsafe_allow_html=True)

        sort_col = "lead_score" if "lead_score" in valid else valid[0]
        preview  = df_exp[valid].sort_values(sort_col, ascending=False).reset_index(drop=True)
        preview.index += 1
        fmt2 = {}
        if "conversion_likelihood" in valid: fmt2["conversion_likelihood"] = "{:.1%}"
        if "lead_score"            in valid: fmt2["lead_score"]            = "{:.0f}"
        st.dataframe(preview.style.format(fmt2), use_container_width=True, height=360)

        ex1, ex2 = st.columns(2)
        ex1.download_button("📥 Download CSV",
            data=df_exp[valid].to_csv(index=False).encode(),
            file_name=f"qualified_leads_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True, type="primary")
        try:
            import openpyxl
            export_for_excel = df_exp[valid].copy()
            if "lead_category" not in valid:
                export_for_excel["lead_category"] = df_exp["lead_category"].values
            ex2.download_button("📊 Download Excel (tabs per tier)",
                data=to_excel_bytes(export_for_excel),
                file_name=f"qualified_leads_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except ImportError:
            ex2.info("Install openpyxl for Excel export: `pip install openpyxl`")