"""
Employee Attrition Intelligence - HR Analytics platform (Streamlit).

Run:  streamlit run streamlit_app.py
Deps: streamlit, pandas, numpy, scikit-learn, joblib   (no other packages)
Model lookup order: $MODEL_PATH, models/model.pkl, models/model.joblib
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Attrition Intelligence | HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    *([Path(os.environ["MODEL_PATH"])] if os.environ.get("MODEL_PATH") else []),
    BASE_DIR / "models" / "model.pkl",
    BASE_DIR / "models" / "model.joblib",
    Path.cwd() / "models" / "model.pkl",
    Path.cwd() / "models" / "model.joblib",
]

# NOTE: 'average_montly_hours' keeps the original dataset spelling on purpose.
CORE_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "work_accident",
    "promotion_last_5years",
]
DEPARTMENTS = ["sales", "technical", "support", "IT", "product_mng",
               "marketing", "RandD", "accounting", "hr", "management"]
SALARIES = ["low", "medium", "high"]

VALID_RANGES = {
    "satisfaction_level": (0.0, 1.0),
    "last_evaluation": (0.0, 1.0),
    "number_project": (1, 12),
    "average_montly_hours": (40, 350),
    "time_spend_company": (0, 40),
    "work_accident": (0, 1),
    "promotion_last_5years": (0, 1),
}
ALIASES = [
    ("average_montly_hours", "average_monthly_hours", "avg_monthly_hours"),
    ("sales", "department", "Department"),
    ("time_spend_company", "tenure", "years_at_company"),
]

RISK_LEVELS = {  # label: (color, soft background)
    "Low risk": ("#10b981", "rgba(16,185,129,.14)"),
    "Moderate risk": ("#f59e0b", "rgba(245,158,11,.14)"),
    "High risk": ("#f43f5e", "rgba(244,63,94,.14)"),
}

# --------------------------------------------------------------------------- #
# Design system (CSS)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0f172a; --bg-2:#111c33; --card:#131f38; --line:rgba(148,163,184,.16);
  --text:#e2e8f0; --muted:#94a3b8; --accent:#6366f1; --accent-2:#8b5cf6;
}
html, body, [class*="css"], .stApp, button, input, textarea, select{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif !important;
}
.stApp{
  background:
    radial-gradient(900px 420px at 88% -8%, rgba(99,102,241,.16), transparent 60%),
    radial-gradient(700px 380px at -5% 0%, rgba(139,92,246,.10), transparent 60%),
    var(--bg);
  color:var(--text);
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1240px;}
h1,h2,h3,h4{color:#f8fafc; letter-spacing:-.02em;}
p, label, span, li{color:var(--text);}

/* Hero */
.hero{
  display:flex; align-items:center; gap:18px; padding:26px 28px; margin-bottom:22px;
  border:1px solid var(--line); border-radius:18px;
  background:linear-gradient(135deg, rgba(99,102,241,.16), rgba(19,31,56,.85) 55%);
  box-shadow:0 10px 30px rgba(2,6,23,.35), inset 0 1px 0 rgba(255,255,255,.04);
}
.hero-logo{
  width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 0 0 1px rgba(255,255,255,.12), 0 8px 24px rgba(99,102,241,.45);
}
.hero h1{margin:0; font-size:1.75rem; font-weight:800; line-height:1.15;}
.hero p{margin:4px 0 0; color:var(--muted); font-size:.95rem;}
.pill{
  margin-left:auto; padding:6px 12px; border-radius:999px; font-size:.78rem; font-weight:600;
  border:1px solid var(--line); background:rgba(15,23,42,.6); color:var(--muted); white-space:nowrap;
}
.pill b{color:#10b981;}

/* Section headings */
.sec{display:flex; align-items:center; gap:10px; margin:6px 0 4px;}
.sec .ic{
  width:32px; height:32px; border-radius:9px; display:flex; align-items:center; justify-content:center;
  background:rgba(99,102,241,.16); border:1px solid rgba(99,102,241,.35); font-size:1rem;
}
.sec h3{margin:0; font-size:1.05rem; font-weight:700;}
.sec-sub{color:var(--muted); font-size:.85rem; margin:0 0 12px 42px;}

/* Cards + KPI */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card); border:1px solid var(--line) !important; border-radius:16px;
  box-shadow:0 1px 2px rgba(2,6,23,.4), 0 8px 24px rgba(2,6,23,.22);
}
.kpi{
  background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px;
  box-shadow:0 1px 2px rgba(2,6,23,.4), 0 6px 18px rgba(2,6,23,.22);
  transition:border-color .2s, box-shadow .2s;
}
.kpi:hover{border-color:rgba(99,102,241,.55); box-shadow:0 0 0 3px rgba(99,102,241,.12);}
.kpi .k-top{display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:.82rem; font-weight:500;}
.kpi .k-val{font-size:1.85rem; font-weight:800; color:#f8fafc; margin-top:6px; letter-spacing:-.03em;}
.kpi .k-sub{font-size:.78rem; color:var(--muted); margin-top:2px;}

/* Risk badge, gauge, probability bar */
.badge{
  display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
  font-weight:700; font-size:.88rem; border:1px solid;
}
.badge i{width:8px; height:8px; border-radius:50%; display:inline-block; box-shadow:0 0 10px currentColor; background:currentColor;}
.gauge-wrap{display:flex; flex-direction:column; align-items:center; padding:6px 0 2px;}
.gauge-num{font-size:2.4rem; font-weight:800; letter-spacing:-.04em; margin-top:-58px;}
.gauge-lbl{color:var(--muted); font-size:.8rem; margin-bottom:12px;}
.pbar{height:12px; border-radius:999px; background:rgba(148,163,184,.14); overflow:hidden; display:flex; margin:6px 0 8px;}
.pbar div{height:100%;}
.plegend{display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted);}
.plegend b{color:var(--text);}

/* Recommendation / insight items */
.tip{
  display:flex; gap:12px; padding:12px 14px; margin-bottom:10px; border-radius:12px;
  background:rgba(15,23,42,.55); border:1px solid var(--line); border-left:3px solid var(--accent);
}
.tip .t-ic{font-size:1.15rem; line-height:1.4;}
.tip .t-h{font-weight:650; color:#f1f5f9; font-size:.92rem;}
.tip .t-b{color:var(--muted); font-size:.84rem; margin-top:2px;}
.alert{
  display:flex; gap:14px; padding:18px 20px; border-radius:14px; margin:8px 0 18px;
  border:1px solid rgba(244,63,94,.4); background:linear-gradient(135deg, rgba(244,63,94,.14), rgba(15,23,42,.6));
}
.alert .a-ic{font-size:1.5rem;}
.alert h4{margin:0 0 4px; color:#fecdd3;}
.alert p, .alert li{color:#e2e8f0; font-size:.88rem; margin:2px 0;}
.alert code{background:rgba(2,6,23,.6); padding:2px 6px; border-radius:6px; color:#c7d2fe;}
.note{
  padding:12px 14px; border-radius:12px; font-size:.86rem; margin:8px 0;
  border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.10); color:#fde68a;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{
  height:46px; padding:0 18px; border-radius:10px 10px 0 0; color:var(--muted); font-weight:600;
  background:transparent;
}
.stTabs [aria-selected="true"]{color:#fff !important; background:rgba(99,102,241,.14);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent) !important; height:3px;}

/* Inputs + buttons */
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--accent); box-shadow:0 0 0 4px rgba(99,102,241,.25);}
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input{
  background:#0b1324 !important; border-color:var(--line) !important; border-radius:10px !important; color:var(--text) !important;
}
.stButton > button, .stDownloadButton > button{
  border-radius:10px; font-weight:650; border:1px solid rgba(99,102,241,.6);
  background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:#fff;
  box-shadow:0 6px 18px rgba(99,102,241,.35); transition:transform .12s, box-shadow .12s;
}
.stButton > button:hover, .stDownloadButton > button:hover{transform:translateY(-1px); box-shadow:0 10px 24px rgba(99,102,241,.45); color:#fff; border-color:#a5b4fc;}
[data-testid="stFileUploader"] section{background:#0b1324; border:1.5px dashed rgba(99,102,241,.45); border-radius:14px;}
[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:12px; overflow:hidden;}
section[data-testid="stSidebar"]{background:var(--bg-2); border-right:1px solid var(--line);}
.foot{color:var(--muted); font-size:.78rem; text-align:center; margin-top:34px;}
</style>
"""


def html(markup: str) -> None:
    """Render HTML safely through markdown (strips indentation and blank lines)."""
    clean = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def section(icon: str, title: str, sub: str = "") -> None:
    html(f'<div class="sec"><div class="ic">{icon}</div><h3>{title}</h3></div>')
    if sub:
        html(f'<p class="sec-sub">{sub}</p>')


def kpi(icon: str, label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="k-top"><span>{label}</span><span>{icon}</span></div>'
            f'<div class="k-val">{value}</div><div class="k-sub">{sub}</div></div>')


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Return (model, path, error). Never raises so the UI can degrade gracefully."""
    errors = []
    for path in MODEL_CANDIDATES:
        if path.is_file():
            try:
                return joblib.load(path), str(path), None
            except Exception as exc:  # corrupted file, sklearn version mismatch, etc.
                errors.append(f"{path.name}: {exc}")
    return None, None, "; ".join(errors) if errors else "No model file found."


def expected_columns(model) -> list[str]:
    """Return the feature list the model actually needs.

    The loaded model expects exactly 7 numerical inputs (n_features_in_ = 7)
    and is *not* an end-to-end pipeline that handles categoricals.  Always
    return CORE_FEATURES so that categorical columns like 'sales' (department)
    and 'salary' are never included in the inference payload.
    """
    return list(CORE_FEATURES)


def risk_label(p: float) -> str:
    return "Low risk" if p < 0.35 else "Moderate risk" if p < 0.65 else "High risk"


def predict(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(X))
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return pred, proba[:, idx]
    return pred, pred.astype(float)


# --------------------------------------------------------------------------- #
# Visual components
# --------------------------------------------------------------------------- #
def gauge_html(prob: float) -> str:
    label = risk_label(prob)
    color = RISK_LEVELS[label][0]
    arc = 251.3  # length of a radius-80 half circle
    dash = max(0.0, min(1.0, prob)) * arc
    return f"""
    <div class="gauge-wrap">
      <svg width="240" height="136" viewBox="0 0 200 110">
        <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#10b981"/>
        <stop offset=".5" stop-color="#f59e0b"/><stop offset="1" stop-color="#f43f5e"/></linearGradient></defs>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="rgba(148,163,184,.16)" stroke-width="14" stroke-linecap="round"/>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#g)" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {arc}" opacity=".95"/>
      </svg>
      <div class="gauge-num" style="color:{color}">{prob*100:.1f}%</div>
      <div class="gauge-lbl">Probability of leaving</div>
    </div>"""


def badge_html(label: str) -> str:
    color, bg = RISK_LEVELS[label]
    return f'<span class="badge" style="color:{color}; background:{bg}; border-color:{color}55"><i></i>{label}</span>'


def prob_bar_html(prob: float) -> str:
    return f"""
    <div class="pbar"><div style="width:{(1-prob)*100:.1f}%; background:#10b981"></div>
    <div style="width:{prob*100:.1f}%; background:#f43f5e"></div></div>
    <div class="plegend"><span>Stays <b>{(1-prob)*100:.1f}%</b></span><span>Leaves <b>{prob*100:.1f}%</b></span></div>"""


def recommendations(row: dict) -> list[tuple[str, str, str]]:
    tips = []
    sat, ev = row.get("satisfaction_level", 0.5), row.get("last_evaluation", 0.7)
    proj, hrs = row.get("number_project", 4), row.get("average_montly_hours", 200)
    ten, promo = row.get("time_spend_company", 3), row.get("promotion_last_5years", 0)
    if sat < 0.4:
        tips.append(("💬", "Hold a stay conversation", "Low satisfaction is the strongest warning sign. Ask what would make the role better."))
    if hrs > 250 or proj >= 6:
        tips.append(("⚖️", "Rebalance workload", "High hours or project load often precedes burnout. Redistribute or add support."))
    if proj <= 2 and sat < 0.6:
        tips.append(("🎯", "Increase engagement", "Very few projects can signal under-use. Offer stretch assignments."))
    if ten >= 4 and not promo:
        tips.append(("🚀", "Map a growth path", "Mid-tenure staff without a recent promotion tend to look elsewhere."))
    if ev >= 0.8 and sat < 0.6:
        tips.append(("⭐", "Protect top performers", "High performers who are unhappy are the costliest departures."))
    if str(row.get("salary", "")).lower() == "low":
        tips.append(("💰", "Review compensation", "Benchmark pay against the market for this role."))
    if not tips:
        tips.append(("✅", "Maintain current course", "No major risk drivers detected. Keep regular check-ins going."))
    return tips


def model_missing_banner(err: str | None) -> None:
    html(f"""
    <div class="alert"><div class="a-ic">⚠️</div><div>
      <h4>Model not found</h4>
      <p>The app is running, but predictions are disabled until a trained model is available.</p>
      <ul>
        <li>Train and save your model to <code>models/model.pkl</code> or <code>models/model.joblib</code>
            (for example: <code>joblib.dump(pipeline, "models/model.pkl")</code>).</li>
        <li>Or point to it with the <code>MODEL_PATH</code> environment variable.</li>
        <li>In Docker, mount or copy the <code>models/</code> folder into the image.</li>
      </ul>
      <p style="opacity:.7">Details: {err or "n/a"}</p>
    </div></div>""")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def sample_dataframe(cols: list[str]) -> pd.DataFrame:
    """Generate a small sample DataFrame with only the 7 numerical features."""
    rows = [
        (0.38, 0.53, 2, 157, 3, 0, 0),
        (0.80, 0.86, 5, 262, 6, 0, 0),
        (0.11, 0.88, 7, 272, 4, 0, 0),
        (0.72, 0.87, 5, 223, 5, 0, 0),
        (0.92, 0.55, 3, 160, 3, 1, 1),
        (0.45, 0.50, 2, 135, 3, 0, 0),
    ]
    return pd.DataFrame(rows, columns=cols)


def harmonize(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    for exp in expected:  # case-insensitive match
        if exp not in df.columns and exp.lower() in lower_map:
            df = df.rename(columns={lower_map[exp.lower()]: exp})
    for group in ALIASES:  # spelling / naming variants
        for exp in expected:
            if exp in group and exp not in df.columns:
                for alt in group:
                    if alt in df.columns:
                        df = df.rename(columns={alt: exp})
                        break
    return df


def validate(df: pd.DataFrame, expected: list[str]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(f"`{c}`" for c in missing))
        return errors, warnings
    if df.empty:
        errors.append("The file has no rows.")
    nulls = int(df[expected].isna().any(axis=1).sum())
    if nulls:
        errors.append(f"{nulls} row(s) contain empty values. Fill or remove them and upload again.")
    for col, (lo, hi) in VALID_RANGES.items():
        if col in expected and pd.api.types.is_numeric_dtype(df[col]):
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                warnings.append(f"`{col}`: {bad} value(s) outside the expected range {lo} to {hi}.")
        elif col in expected:
            errors.append(f"`{col}` must be numeric.")
    return errors, warnings


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
html(CSS)
model, model_path, model_error = load_model()
EXPECTED = expected_columns(model) if model is not None else list(CORE_FEATURES)
status = (f'<span class="pill"><b>●</b> Model ready · {Path(model_path).name}</span>' if model
          else '<span class="pill" style="color:#fda4af">● Model unavailable</span>')

html(f"""
<div class="hero">
  <div class="hero-logo"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
    <path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg></div>
  <div><h1>Attrition Intelligence</h1>
  <p>Predict employee turnover risk and act on it before it costs you talent.</p></div>
  {status}
</div>""")

if model is None:
    model_missing_banner(model_error)

tab_single, tab_batch, tab_insights = st.tabs(
    ["🧑‍💼  Risk Simulator", "📂  Batch Analysis", "🧠  Model Insights"]
)

# ------------------------------- TAB 1 ------------------------------------- #
with tab_single:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        section("😊", "Employee profile", "Adjust the inputs to see how the risk changes.")
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            section("📈", "Engagement")
            satisfaction = st.slider("Satisfaction level", 0.0, 1.0, 0.50, 0.01)
            evaluation = st.slider("Last evaluation score", 0.0, 1.0, 0.70, 0.01)
            accident = st.toggle("Had a workplace accident", value=False)
        with c2, st.container(border=True):
            section("🗂️", "Workload")
            projects = st.slider("Number of projects", 1, 10, 4)
            hours = st.slider("Average monthly hours", 80, 320, 200, 5)
            tenure = st.slider("Years at company", 1, 20, 3)
        with st.container(border=True):
            section("🏢", "Career & compensation")
            c3, c4, c5 = st.columns(3)
            promoted = c3.toggle("Promoted in last 5 years", value=False)
            department = c4.selectbox("Department", DEPARTMENTS)
            salary = c5.selectbox("Salary band", SALARIES, index=1)

    # Only the 7 numerical features go to the model.  Department and salary
    # are kept in the profile for context / recommendations but are NOT part
    # of the inference payload (they are not in EXPECTED / CORE_FEATURES).
    profile = {
        "satisfaction_level": satisfaction, "last_evaluation": evaluation,
        "number_project": projects, "average_montly_hours": hours,
        "time_spend_company": tenure, "work_accident": int(accident),
        "promotion_last_5years": int(promoted),
        "department": department, "salary": salary,  # UI-only, not sent to model
    }

    with right:
        section("🎯", "Risk assessment", "Live prediction from the trained model.")
        with st.container(border=True):
            if model is None:
                st.info("Load a model to enable predictions.")
            else:
                try:
                    row = pd.DataFrame([{c: profile.get(c, 0) for c in EXPECTED}])
                    pred, prob = predict(model, row)
                    p = float(prob[0])
                    label = risk_label(p)
                    html(gauge_html(p))
                    html(f'<div style="text-align:center; margin-bottom:14px">{badge_html(label)}</div>')
                    html(prob_bar_html(p))
                    verdict = "likely to leave" if int(pred[0]) == 1 else "likely to stay"
                    st.caption(f"Model classification: **{verdict}** (decision threshold applied by the model).")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.caption("Check that the model's expected columns match the schema in this app.")
        section("💡", "Suggested actions")
        for icon, head, body in recommendations(profile):
            html(f'<div class="tip"><div class="t-ic">{icon}</div><div><div class="t-h">{head}</div>'
                 f'<div class="t-b">{body}</div></div></div>')

# ------------------------------- TAB 2 ------------------------------------- #
with tab_batch:
    section("📂", "Batch analysis", "Score a whole workforce from a CSV file.")
    sample = sample_dataframe(EXPECTED)
    col_a, col_b = st.columns([1.3, 1], gap="large")
    with col_a:
        with st.container(border=True):
            upload = st.file_uploader("Upload employee CSV", type=["csv"], label_visibility="collapsed")
            st.caption("Required columns: " + ", ".join(f"`{c}`" for c in EXPECTED))
    with col_b:
        with st.container(border=True):
            st.markdown("**Need a template?**")
            st.caption("Download the sample file and replace the rows with your data.")
            st.download_button("⬇️  Download sample CSV", sample.to_csv(index=False).encode("utf-8"),
                               file_name="sample_employees.csv", mime="text/csv", use_container_width=True)

    if upload is None:
        with st.expander("Preview the expected format", expanded=True):
            st.dataframe(sample, use_container_width=True, hide_index=True)
    elif model is None:
        st.warning("Batch scoring is unavailable until a model is loaded.")
    else:
        try:
            raw = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read the file as CSV: {exc}")
            raw = None
        if raw is not None:
            data = harmonize(raw, EXPECTED)
            errors, warnings = validate(data, EXPECTED)
            for w in warnings:
                html(f'<div class="note">⚠️ {w}</div>')
            if errors:
                for e in errors:
                    st.error(e)
                st.caption("Fix the issues above and upload the file again.")
            else:
                with st.spinner("Scoring employees..."):
                    _, probs = predict(model, data[EXPECTED])
                out = data.copy()
                out["attrition_probability"] = np.round(probs, 4)
                out["risk_level"] = [risk_label(p) for p in probs]

                n, high = len(out), int((out["risk_level"] == "High risk").sum())
                mod = int((out["risk_level"] == "Moderate risk").sum())
                k1, k2, k3, k4 = st.columns(4)
                k1.markdown(kpi("👥", "Employees scored", f"{n:,}", "rows in file"), unsafe_allow_html=True)
                k2.markdown(kpi("🔴", "High risk", f"{high:,}", f"{high/n:.1%} of workforce"), unsafe_allow_html=True)
                k3.markdown(kpi("🟠", "Moderate risk", f"{mod:,}", f"{mod/n:.1%} of workforce"), unsafe_allow_html=True)
                k4.markdown(kpi("📉", "Average risk", f"{out['attrition_probability'].mean():.1%}", "mean probability"),
                            unsafe_allow_html=True)

                st.write("")
                section("📋", "Results", "Filter by risk level, sort by any column, and export.")
                levels = st.multiselect("Risk level", list(RISK_LEVELS), default=list(RISK_LEVELS))
                view = out[out["risk_level"].isin(levels)].sort_values("attrition_probability", ascending=False)
                st.dataframe(
                    view, use_container_width=True, hide_index=True,
                    column_config={
                        "attrition_probability": st.column_config.ProgressColumn(
                            "Attrition probability", min_value=0.0, max_value=1.0, format="%.2f"),
                        "risk_level": st.column_config.TextColumn("Risk level"),
                    },
                )
                st.download_button("⬇️  Download results (CSV)", view.to_csv(index=False).encode("utf-8"),
                                   file_name="attrition_predictions.csv", mime="text/csv")

# ------------------------------- TAB 3 ------------------------------------- #
with tab_insights:
    section("🧠", "Model insights", "What tends to drive turnover, and how to respond.")

    def feature_importance(m):
        try:
            est = m.steps[-1][1] if hasattr(m, "steps") else m
            names = None
            if hasattr(m, "steps") and hasattr(m[:-1], "get_feature_names_out"):
                names = list(m[:-1].get_feature_names_out())
            if hasattr(est, "feature_importances_"):
                vals = np.asarray(est.feature_importances_)
            elif hasattr(est, "coef_"):
                vals = np.abs(np.ravel(est.coef_))
            else:
                return None
            names = names if names and len(names) == len(vals) else (
                EXPECTED if len(EXPECTED) == len(vals) else [f"feature_{i}" for i in range(len(vals))])
            names = [n.split("__")[-1] for n in names]
            return pd.Series(vals, index=names).sort_values(ascending=False).head(10)
        except Exception:
            return None

    imp = feature_importance(model) if model is not None else None
    ic1, ic2 = st.columns([1.2, 1], gap="large")
    with ic1, st.container(border=True):
        section("📊", "Top drivers in your model")
        if imp is not None:
            st.bar_chart((imp / imp.sum()).sort_values(), horizontal=True, color="#6366f1")
            st.caption("Relative importance, normalized to 100%.")
        else:
            st.info("Feature importance is not available for this model type, or no model is loaded.")
    with ic2, st.container(border=True):
        section("🔎", "Key turnover drivers")
        drivers = [
            ("😟", "Low satisfaction", "The most consistent predictor of exits across HR datasets."),
            ("🔥", "Overwork", "Very high monthly hours with many projects signals burnout."),
            ("🧭", "Stalled growth", "Several years without promotion raises flight risk."),
            ("⭐", "Unrecognized talent", "High evaluations paired with low satisfaction are costly exits."),
            ("📉", "Under-utilization", "Few projects and low hours often mean disengagement."),
        ]
        for icon, head, body in drivers:
            html(f'<div class="tip"><div class="t-ic">{icon}</div><div><div class="t-h">{head}</div>'
                 f'<div class="t-b">{body}</div></div></div>')

    section("🛡️", "Retention strategies")
    s1, s2, s3 = st.columns(3, gap="medium")
    for col, icon, head, body in [
        (s1, "💬", "Listen early", "Run pulse surveys and stay interviews. Act on satisfaction dips within weeks."),
        (s2, "⚖️", "Manage workload", "Cap sustained overtime, balance project allocation, and track hours by team."),
        (s3, "🚀", "Invest in growth", "Publish career paths, review promotion cadence, and tie pay to the market."),
    ]:
        col.markdown(kpi(icon, head, "", body).replace('<div class="k-val"></div>', ""), unsafe_allow_html=True)

    st.write("")
    with st.expander("How risk levels are defined"):
        st.markdown(
            "- **Low risk**: probability below 35%\n"
            "- **Moderate risk**: 35% to 65%\n"
            "- **High risk**: above 65%\n\n"
            "Predictions estimate likelihood, not certainty. Use them to prioritize conversations, "
            "not to make employment decisions."
        )

html('<div class="foot">Attrition Intelligence · Built with Streamlit and scikit-learn</div>')