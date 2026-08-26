"""
SkillBridge - Staffing Intelligence Suite
------------------------------------------
Three tools for an IT staffing recruiter's actual daily workflow:
  1. Shortlist Candidates - given a client requirement, rank candidates instantly
  2. Job Fit Finder        - given a candidate, find their best-fit roles
  3. ATS Resume Scorer     - check/fix a resume before submitting it to a client

Two data modes throughout:
  - DEMO POOL: our synthetic 800-candidate dataset + the trained ML model
    (Random Forest, trained on synthetic placement history - see src/04_model_training.py)
  - YOUR OWN CANDIDATES: recruiter uploads a real candidate list. Since there's
    no placement HISTORY for a brand-new client/candidate pool (a classic
    "cold start" problem), we use a transparent rule-based fit score here
    instead of the ML model - the ML model would be applying learned patterns
    outside the data it was trained on, which isn't honest to present as a
    real prediction. The rule-based score uses the same skill-match +
    experience-fit logic our synthetic data itself was generated from.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile

from ats_scorer import score_resume, extract_keywords
from resume_builder import build_resume_docx

st.set_page_config(page_title="SkillBridge - Staffing Intelligence Suite", layout="wide", page_icon="🌉")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EDU_ORDER = {"Diploma": 1, "B.Sc": 2, "BCA": 2, "B.Tech": 3, "MCA": 4, "M.Tech": 4}

# ---------------------------------------------------------
# Load demo data & trained model (cached)
# ---------------------------------------------------------
@st.cache_data
def load_demo_data():
    candidates = pd.read_csv(f"{DATA_DIR}/candidates.csv")
    jobs = pd.read_csv(f"{DATA_DIR}/jobs.csv")
    return candidates, jobs

@st.cache_resource
def load_model():
    model = joblib.load(f"{DATA_DIR}/best_model.pkl")
    feature_cols = joblib.load(f"{DATA_DIR}/model_feature_columns.pkl")
    return model, feature_cols

demo_candidates, demo_jobs = load_demo_data()
model, feature_cols = load_model()

# Pipeline tracker state - lives for the browser session (resets on refresh, which is
# fine for a demo; a production version would back this with a real database).
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []  # list of dicts: candidate, requirement, stage

PIPELINE_STAGES = ["Submitted", "Client Review", "Interview Scheduled", "Interviewed", "Offer Extended", "Placed", "Rejected"]

# Resume builder state
if "builder_experience" not in st.session_state:
    st.session_state.builder_experience = []
if "builder_education" not in st.session_state:
    st.session_state.builder_education = []

# ---------------------------------------------------------
# Styling - enterprise SaaS look: restrained color use, clear hierarchy,
# a sidebar for branding/navigation instead of everything stacked in the
# main column, and a KPI strip so the tool reads as a product, not a demo.
# ---------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  { font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif; }
#MainMenu, footer {visibility: hidden;}

div.stButton > button {
    background: #1E3A8A; color: white; border: none; border-radius: 6px; font-weight: 600;
    padding: 0.5rem 1.1rem; letter-spacing: 0.01em;
}
div.stButton > button:hover { background: #1E40AF; color: white; }

.sb-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 18px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 18px;
}
.sb-brand { display: flex; align-items: center; gap: 10px; }
.sb-brand-mark {
    width: 34px; height: 34px; border-radius: 8px; background: #1E3A8A;
    display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 16px;
}
.sb-brand-name { font-size: 19px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em; }
.sb-brand-tag { font-size: 12px; color: #6B7280; margin-top: -2px; }

.kpi-strip { display: flex; gap: 14px; margin-bottom: 22px; }
.kpi-box {
    flex: 1; background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 14px 18px;
}
.kpi-value { font-size: 22px; font-weight: 700; color: #0F172A; }
.kpi-label { font-size: 12px; color: #6B7280; margin-top: 2px; }

.match-card {
    background: #ffffff; border: 1px solid #E5E7EB; border-radius: 10px; padding: 18px 22px;
    margin-bottom: 12px; border-left: 4px solid #cccccc;
}
.match-card.tier-high { border-left-color: #0D9488; }
.match-card.tier-mid  { border-left-color: #CA8A04; }
.match-card.tier-low  { border-left-color: #BE123C; }
.match-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.match-card-title { font-size: 16px; font-weight: 600; color: #111827; }
.match-score { font-size: 20px; font-weight: 700; color: #111827; }
.tier-pill { font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 4px; margin-left: 6px; text-transform: uppercase; letter-spacing: 0.03em; }
.tier-pill.tier-high { background: #CCFBF1; color: #0F766E; }
.tier-pill.tier-mid  { background: #FEF3C7; color: #92400E; }
.tier-pill.tier-low  { background: #FFE4E6; color: #9F1239; }
.progress-track { background: #F1F5F9; border-radius: 4px; height: 6px; width: 100%; margin: 8px 0 10px 0; overflow: hidden; }
.progress-fill { height: 6px; border-radius: 4px; }
.tier-high .progress-fill { background: #0D9488; }
.tier-mid  .progress-fill { background: #CA8A04; }
.tier-low  .progress-fill { background: #BE123C; }
.meta-row { font-size: 13px; color: #6B7280; margin-bottom: 8px; }
.explanation-row { font-size: 13.5px; color: #374151; background: #F8FAFC; border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 5px; font-size: 12px; margin: 2px 4px 2px 0; font-weight: 500; }
.chip-match { background: #F0FDFA; color: #0F766E; border: 1px solid #99F6E4; }
.chip-missing { background: #FFF1F2; color: #9F1239; border: 1px solid #FECDD3; }
.chip-label { font-size: 11px; font-weight: 700; color: #9CA3AF; margin-right: 6px; text-transform: uppercase; letter-spacing: 0.03em; }
.mode-badge { display:inline-block; padding: 3px 12px; border-radius: 5px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 10px;}
.mode-ml { background:#EEF2FF; color:#3730A3; border: 1px solid #C7D2FE; }
.mode-rule { background:#FFFBEB; color:#92400E; border: 1px solid #FDE68A; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sb-topbar">
    <div class="sb-brand">
        <div class="sb-brand-mark">SB</div>
        <div>
            <div class="sb-brand-name">SkillBridge</div>
            <div class="sb-brand-tag">Staffing Intelligence Suite</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
for col, value, label in [
    (kpi1, f"{len(demo_candidates)}", "Candidates in Demo Pool"),
    (kpi2, f"{len(demo_jobs)}", "Open Demo Requirements"),
    (kpi3, "4", "Modules"),
    (kpi4, "ML + Rule-Based", "Scoring Engine"),
]:
    col.markdown(f'<div class="kpi-box"><div class="kpi-value">{value}</div>'
                  f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### SkillBridge")
    st.caption("Staffing Intelligence Suite")
    st.markdown("---")
    st.markdown("**Why this tool exists**")
    st.write(
        "A staffing team's daily bottleneck is manual screening: a new client requirement arrives and a "
        "recruiter has to scan resumes by hand, then hope the shortlisted candidate's resume survives the "
        "client's own ATS. SkillBridge automates both steps."
    )
    st.markdown("**Scoring modes**")
    st.write("🔵 **ML Model** — demo pool only, trained on synthetic placement history.")
    st.write("🟡 **Rule-Based** — your own uploaded data. No placement history exists yet for a new "
             "pool, so this uses a transparent, explainable formula instead of an unproven model.")
    st.markdown("---")
    st.caption("SkillBridge · Staffing Intelligence Suite · v1.0")


def tier_for(prob):
    if prob >= 60:
        return "tier-high", "Strong Fit"
    elif prob >= 35:
        return "tier-mid", "Possible Fit"
    else:
        return "tier-low", "Weak Fit"


def generate_explanation(probability, matched_skills, missing_skills, extra_context=""):
    """Turn the raw score into a plain-English sentence a recruiter can act on
    immediately, instead of making them interpret a bare percentage.
    """
    n_matched, n_missing = len(matched_skills), len(missing_skills)
    total = n_matched + n_missing

    if probability >= 60:
        lead = "Strong fit."
    elif probability >= 35:
        lead = "Possible fit, with gaps."
    else:
        lead = "Weak fit."

    if total == 0:
        skill_clause = "No skill requirements could be matched against this profile."
    elif n_missing == 0:
        skill_clause = f"Covers all {n_matched} required skills."
    else:
        top_missing = ", ".join(sorted(missing_skills)[:3])
        skill_clause = f"Covers {n_matched} of {total} required skills; missing {top_missing}."

    sentence = f"{lead} {skill_clause}"
    if extra_context:
        sentence += f" {extra_context}"
    return sentence


def render_match_card(title, probability, meta_items, matched_skills, missing_skills, extra_context=""):
    tier_class, tier_label = tier_for(probability)
    matched_html = "".join(f'<span class="chip chip-match">{s}</span>' for s in sorted(matched_skills)) or "<span style='color:#9ca3af;font-size:13px'>None</span>"
    missing_html = "".join(f'<span class="chip chip-missing">{s}</span>' for s in sorted(missing_skills)) or "<span style='color:#9ca3af;font-size:13px'>None - full match</span>"
    meta_html = " &nbsp;·&nbsp; ".join(meta_items)
    explanation = generate_explanation(probability, matched_skills, missing_skills, extra_context)
    html = f"""
    <div class="match-card {tier_class}">
        <div class="match-card-header">
            <div class="match-card-title">{title} &nbsp;<span class="tier-pill {tier_class}">{tier_label}</span></div>
            <div class="match-score">{probability:.1f}%</div>
        </div>
        <div class="progress-track"><div class="progress-fill" style="width:{min(max(probability,0),100)}%;"></div></div>
        <div class="meta-row">{meta_html}</div>
        <div class="explanation-row">{explanation}</div>
        <div><span class="chip-label">Has</span>{matched_html}</div>
        <div style="margin-top:4px;"><span class="chip-label">Missing</span>{missing_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def build_ml_features(cand_row, job_row):
    """Feature vector for the trained ML model - only valid for our demo/synthetic schema."""
    cand_skills = set(cand_row["skills"].split(", "))
    req_skills = set(job_row["required_skills"].split(", "))
    overlap = cand_skills & req_skills
    missing = req_skills - cand_skills
    extra = cand_skills - req_skills
    jaccard = len(overlap) / len(cand_skills | req_skills) if (cand_skills | req_skills) else 0
    skill_match_ratio = len(overlap) / len(req_skills) if req_skills else 0
    row = {
        "skill_match_ratio": skill_match_ratio, "jaccard_similarity": jaccard,
        "skill_overlap_count": len(overlap), "skill_missing_count": len(missing),
        "skill_extra_count": len(extra),
        "experience_gap": cand_row["experience_years"] - job_row["required_experience"],
        "same_domain": int(cand_row["preferred_domain"] == job_row["domain"]),
        "has_certification": cand_row["has_certification"],
        "education_level": EDU_ORDER.get(cand_row["education"], 2),
    }
    for col in feature_cols:
        if col.startswith("domain_"):
            row[col] = int(col == f"domain_{job_row['domain']}")
        elif col.startswith("client_"):
            row[col] = int(col == f"client_{job_row['client_type']}")
    return pd.DataFrame([row])[feature_cols], missing, overlap


def rule_based_fit(cand_skills, required_skills, cand_experience, required_experience):
    """Transparent, explainable fit score for data with no placement history to learn
    from. Mirrors the same weighting logic used to originally generate our synthetic
    training data (skill match matters most, meeting experience matters, no hidden model).
    """
    overlap = cand_skills & required_skills
    missing = required_skills - cand_skills
    match_ratio = len(overlap) / len(required_skills) if required_skills else 0
    exp_gap = cand_experience - required_experience

    score = 15 + match_ratio * 55
    score += 10 if exp_gap >= 0 else -15
    score = max(2, min(97, score))
    return score, overlap, missing


def extract_experience_years(resume_text):
    """Best-effort heuristic: look for phrases like '3 years', '5+ years experience'
    in resume text and return the largest number found. This is intentionally a
    rough estimate, not a precise extraction - free-text resumes phrase experience
    too many different ways to parse reliably. The UI always shows this as an
    editable, pre-filled value so the recruiter can correct it before relying on it.
    """
    import re
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*years?", resume_text.lower())
    years = [float(m) for m in matches if float(m) <= 40]  # filter obviously-wrong matches (e.g. a year like "2024")
    return max(years) if years else 0.0


def parse_candidate_file(uploaded_file):
    """Expected columns: candidate_id, skills, experience_years, education (optional),
    preferred_domain (optional). Returns a cleaned dataframe or raises with a clear message.
    """
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_cols = {"candidate_id", "skills", "experience_years"}
    missing_cols = required_cols - set(c.strip() for c in df.columns)
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}. "
                          f"Download the template below for the exact format.")
    if "education" not in df.columns:
        df["education"] = "B.Tech"
    if "preferred_domain" not in df.columns:
        df["preferred_domain"] = "General"
    return df


def candidate_template_csv():
    template = pd.DataFrame({
        "candidate_id": ["C001", "C002"],
        "skills": ["Python, SQL, Power BI", "Java, Spring Boot, AWS"],
        "experience_years": [3.5, 5.0],
        "education": ["B.Tech", "MCA"],
        "preferred_domain": ["Data & ML", "Banking/Fintech Software"],
    })
    return template.to_csv(index=False).encode("utf-8")


def rows_to_csv(rows, id_label="id"):
    """Convert a list of match-result dicts into a downloadable CSV (for exporting shortlists)."""
    export_rows = []
    for r in rows:
        export_rows.append({
            id_label: r["id"],
            "fit_score_pct": round(r["prob"], 1),
            "matched_skills": ", ".join(sorted(r["overlap"])),
            "missing_skills": ", ".join(sorted(r["missing"])),
            "details": " | ".join(r["meta"]),
        })
    return pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Shortlist Candidates", "Job Fit Finder", "ATS Resume Scorer", "Compare Candidates",
    "Pipeline & Margin", "Resume Builder"
])

# ============================================================
# TAB 1: Shortlist Candidates (Recruiter View)
# ============================================================
with tab1:
    st.subheader("Shortlist candidates against a job requirement")

    source = st.radio(
        "Candidate source", ["Demo Pool (ML Model)", "Upload My Own Candidates (Rule-Based Fit)"],
        horizontal=True, key="t1_source"
    )

    if source == "Demo Pool (ML Model)":
        st.markdown('<span class="mode-badge mode-ml">ML-predicted placement probability</span>', unsafe_allow_html=True)
        job_id = st.selectbox(
            "Select a demo job posting", demo_jobs["job_id"],
            format_func=lambda jid: f"{jid} — {demo_jobs[demo_jobs.job_id==jid]['domain'].values[0]} "
                                     f"({demo_jobs[demo_jobs.job_id==jid]['client_type'].values[0]})"
        )
        job_row = demo_jobs[demo_jobs["job_id"] == job_id].iloc[0]
        with st.expander("Job details"):
            st.write(f"**Domain:** {job_row['domain']}")
            st.write(f"**Required skills:** {job_row['required_skills']}")
            st.write(f"**Required experience:** {job_row['required_experience']} years")
            st.write(f"**Client type:** {job_row['client_type']}")

        if st.button("Rank Candidates", type="primary", key="t1_demo_btn"):
            rows = []
            for _, cand_row in demo_candidates.iterrows():
                X, missing, overlap = build_ml_features(cand_row, job_row)
                prob = model.predict_proba(X)[0, 1] * 100
                rows.append({"id": cand_row["candidate_id"], "prob": prob, "missing": missing,
                             "overlap": overlap, "meta": [f"{cand_row['experience_years']} yrs exp",
                             cand_row["education"], cand_row["preferred_domain"]]})
            rows.sort(key=lambda r: r["prob"], reverse=True)
            st.write(f"**Top 10 candidates for {job_id}:**")
            st.download_button("⬇️ Export shortlist (CSV)", rows_to_csv(rows[:10], "candidate_id"),
                                f"shortlist_{job_id}.csv", "text/csv", key="t1_demo_dl")
            for r in rows[:10]:
                render_match_card(r["id"], r["prob"], r["meta"], r["overlap"], r["missing"])

    else:
        st.markdown('<span class="mode-badge mode-rule">Rule-based fit score</span>', unsafe_allow_html=True)
        st.download_button("⬇️ Download candidate list template (CSV)", candidate_template_csv(),
                            "candidate_template.csv", "text/csv")
        uploaded = st.file_uploader("Upload your candidate list (CSV or Excel)", type=["csv", "xlsx"], key="t1_upload")

        jd_text = st.text_area("Paste the real client job description", height=160, key="t1_jd",
                                placeholder="Paste the client's job description here...")
        req_exp = st.number_input("Required years of experience (from the JD)", min_value=0.0, max_value=20.0,
                                   value=2.0, step=0.5, key="t1_exp")

        if st.button("Rank Candidates", type="primary", key="t1_upload_btn"):
            if not uploaded or not jd_text.strip():
                st.warning("Please upload a candidate list AND paste the job description.")
            else:
                try:
                    cand_df = parse_candidate_file(uploaded)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                required_skills = set(extract_keywords(jd_text, top_n=25))
                if not required_skills:
                    st.warning("Couldn't detect specific skills in that job description - try pasting "
                               "a more detailed JD with explicit tool/technology names.")
                else:
                    rows = []
                    for _, cand_row in cand_df.iterrows():
                        cand_skills = set(s.strip().lower() for s in str(cand_row["skills"]).split(","))
                        score, overlap, missing = rule_based_fit(
                            cand_skills, required_skills, float(cand_row["experience_years"]), req_exp
                        )
                        rows.append({"id": cand_row["candidate_id"], "prob": score, "missing": missing,
                                     "overlap": overlap, "meta": [f"{cand_row['experience_years']} yrs exp",
                                     str(cand_row.get("education", "-"))]})
                    rows.sort(key=lambda r: r["prob"], reverse=True)
                    st.write(f"**Ranked {len(rows)} candidates ({len(required_skills)} skills detected in JD):**")
                    st.download_button("⬇️ Export shortlist (CSV)", rows_to_csv(rows, "candidate_id"),
                                        "shortlist_custom.csv", "text/csv", key="t1_upload_dl")
                    for r in rows:
                        render_match_card(r["id"], r["prob"], r["meta"], r["overlap"], r["missing"])

# ============================================================
# TAB 2: Job Fit Finder (Candidate View)
# ============================================================
with tab2:
    st.subheader("Find the best-fit role for a candidate")

    source2 = st.radio(
        "Candidate source", ["Demo Pool (ML Model)", "My Own Candidate (Rule-Based Fit)"],
        horizontal=True, key="t2_source"
    )

    if source2 == "Demo Pool (ML Model)":
        st.markdown('<span class="mode-badge mode-ml">ML-predicted placement probability</span>', unsafe_allow_html=True)
        cand_id = st.selectbox(
            "Select a demo candidate", demo_candidates["candidate_id"],
            format_func=lambda cid: f"{cid} — {demo_candidates[demo_candidates.candidate_id==cid]['preferred_domain'].values[0]}"
        )
        cand_row = demo_candidates[demo_candidates["candidate_id"] == cand_id].iloc[0]
        with st.expander("Candidate profile"):
            st.write(f"**Preferred domain:** {cand_row['preferred_domain']}")
            st.write(f"**Skills:** {cand_row['skills']}")
            st.write(f"**Experience:** {cand_row['experience_years']} years")
            st.write(f"**Education:** {cand_row['education']}")

        if st.button("Find Best-Fit Jobs", type="primary", key="t2_demo_btn"):
            rows = []
            for _, job_row in demo_jobs.iterrows():
                X, missing, overlap = build_ml_features(cand_row, job_row)
                prob = model.predict_proba(X)[0, 1] * 100
                rows.append({"id": f"{job_row['job_id']} — {job_row['domain']}", "prob": prob,
                             "missing": missing, "overlap": overlap,
                             "meta": [job_row["client_type"], f"{job_row['required_experience']} yrs required"]})
            rows.sort(key=lambda r: r["prob"], reverse=True)
            st.write(f"**Top 10 job matches for {cand_id}:**")
            st.download_button("⬇️ Export matches (CSV)", rows_to_csv(rows[:10], "job_id"),
                                f"job_matches_{cand_id}.csv", "text/csv", key="t2_demo_dl")
            for r in rows[:10]:
                render_match_card(r["id"], r["prob"], r["meta"], r["overlap"], r["missing"])

    else:
        st.markdown('<span class="mode-badge mode-rule">Rule-based fit score</span>', unsafe_allow_html=True)
        st.write("Enter one candidate's details and paste a target job description to check fit.")

        input_mode = st.radio("How do you want to enter the candidate?",
                               ["Type skills manually", "Upload their resume (auto-detect skills)"],
                               horizontal=True, key="t2_input_mode")

        c1, c2 = st.columns(2)
        with c1:
            if input_mode == "Type skills manually":
                cand_skills_input = st.text_area("Candidate's skills (comma-separated)", height=100, key="t2_skills",
                                                  placeholder="Python, SQL, Power BI, Excel")
                cand_exp_input = st.number_input("Candidate's years of experience", min_value=0.0, max_value=30.0,
                                                  value=2.0, step=0.5, key="t2_exp")
            else:
                resume_upload = st.file_uploader("Upload candidate's resume (PDF or DOCX)", type=["pdf", "docx"], key="t2_resume")
                cand_skills_input, cand_exp_input = "", 0.0
                if resume_upload is not None:
                    suffix = "." + resume_upload.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(resume_upload.read())
                        tmp_path = tmp.name
                    from ats_scorer import extract_text
                    resume_text = extract_text(tmp_path)
                    os.unlink(tmp_path)
                    detected_skills = extract_keywords(resume_text, top_n=30)
                    detected_exp = extract_experience_years(resume_text)
                    st.success(f"Detected {len(detected_skills)} skills and ~{detected_exp} years experience. "
                               f"Review and correct below if needed.")
                    cand_skills_input = st.text_area("Detected skills (edit if needed)", value=", ".join(detected_skills),
                                                      height=80, key="t2_skills_auto")
                    cand_exp_input = st.number_input("Detected experience (edit if needed)", min_value=0.0, max_value=30.0,
                                                      value=float(detected_exp), step=0.5, key="t2_exp_auto")
        with c2:
            jd_text2 = st.text_area("Paste the target job description", height=160, key="t2_jd",
                                     placeholder="Paste a job description here...")
            req_exp2 = st.number_input("Required years of experience", min_value=0.0, max_value=20.0,
                                        value=2.0, step=0.5, key="t2_reqexp")

        if st.button("Check Fit", type="primary", key="t2_upload_btn"):
            if not cand_skills_input.strip() or not jd_text2.strip():
                st.warning("Please provide the candidate's skills (type or upload a resume) AND paste a job description.")
            else:
                cand_skills = set(s.strip().lower() for s in cand_skills_input.split(","))
                required_skills = set(extract_keywords(jd_text2, top_n=25))
                if not required_skills:
                    st.warning("Couldn't detect specific skills in that job description.")
                else:
                    score, overlap, missing = rule_based_fit(cand_skills, required_skills, cand_exp_input, req_exp2)
                    render_match_card("Fit for this role", score,
                                       [f"{cand_exp_input} yrs vs {req_exp2} yrs required"], overlap, missing)

# ============================================================
# TAB 3: ATS Resume Scorer
# ============================================================
with tab3:
    st.subheader("Score a resume against a job description")
    st.caption("Checks resume formatting, keyword coverage, and impact statements the way an "
               "automated applicant-tracking system would, before you submit it to a client.")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_resume = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        jd_text3 = st.text_area("Paste the target job description", height=200,
                                 placeholder="Paste a job description here...", key="t3_jd")

    if st.button("Score Resume", type="primary", key="t3_btn"):
        if not uploaded_resume or not jd_text3.strip():
            st.warning("Please upload a resume AND paste a job description.")
        else:
            suffix = "." + uploaded_resume.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_resume.read())
                tmp_path = tmp.name
            with st.spinner("Analyzing resume..."):
                result = score_resume(tmp_path, jd_text3)
            os.unlink(tmp_path)

            score = result["ats_score"]
            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
            st.markdown(f"### ATS Score: :{color}[{score}/100]")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Keyword Match", f"{result['keyword_match_pct']}%")
            c2.metric("Quantified Bullets", result["quantified_bullets"])
            c3.metric("Action Verb Usage", f"{result['action_verb_ratio']}%")
            c4.metric("Sections Found", f"{len(result['sections_found'])}/6")

            st.write("**Matched keywords:**", ", ".join(result["matched_keywords"]) or "None")
            st.write("**Missing keywords:**", ", ".join(result["missing_keywords"]) or "None")
            if result["sections_missing"]:
                st.write("**Missing sections:**", ", ".join(result["sections_missing"]))

            st.write("### Recommendations")
            for s in result["suggestions"]:
                st.write(f"- {s}")

# ============================================================
# TAB 4: Compare Candidates side-by-side
# ============================================================
with tab4:
    st.subheader("Compare two candidates against the same job")
    st.write("Useful when you've shortlisted a couple of strong options and need to decide who to submit first.")

    compare_source = st.radio("Candidate source", ["Demo Pool (ML Model)", "Enter Manually (Rule-Based Fit)"],
                               horizontal=True, key="t4_source")

    if compare_source == "Demo Pool (ML Model)":
        st.markdown('<span class="mode-badge mode-ml">ML-predicted placement probability</span>', unsafe_allow_html=True)
        job_id4 = st.selectbox(
            "Select a demo job posting", demo_jobs["job_id"],
            format_func=lambda jid: f"{jid} — {demo_jobs[demo_jobs.job_id==jid]['domain'].values[0]}",
            key="t4_job"
        )
        job_row4 = demo_jobs[demo_jobs["job_id"] == job_id4].iloc[0]

        colA, colB = st.columns(2)
        with colA:
            cand_a = st.selectbox("Candidate A", demo_candidates["candidate_id"], key="t4_cand_a")
        with colB:
            cand_b = st.selectbox("Candidate B", demo_candidates["candidate_id"],
                                   index=1, key="t4_cand_b")

        if st.button("Compare", type="primary", key="t4_demo_btn"):
            if cand_a == cand_b:
                st.warning("Pick two different candidates to compare.")
            else:
                row_a = demo_candidates[demo_candidates["candidate_id"] == cand_a].iloc[0]
                row_b = demo_candidates[demo_candidates["candidate_id"] == cand_b].iloc[0]
                X_a, missing_a, overlap_a = build_ml_features(row_a, job_row4)
                X_b, missing_b, overlap_b = build_ml_features(row_b, job_row4)
                prob_a = model.predict_proba(X_a)[0, 1] * 100
                prob_b = model.predict_proba(X_b)[0, 1] * 100

                colA, colB = st.columns(2)
                with colA:
                    render_match_card(cand_a, prob_a, [f"{row_a['experience_years']} yrs", row_a["education"]], overlap_a, missing_a)
                with colB:
                    render_match_card(cand_b, prob_b, [f"{row_b['experience_years']} yrs", row_b["education"]], overlap_b, missing_b)

                winner = cand_a if prob_a > prob_b else cand_b if prob_b > prob_a else None
                if winner:
                    st.info(f"**Head-to-head:** {winner} has the higher predicted placement probability "
                            f"({max(prob_a,prob_b):.1f}% vs {min(prob_a,prob_b):.1f}%).")
                else:
                    st.info("**Head-to-head:** Both candidates score equally on placement probability.")

    else:
        st.markdown('<span class="mode-badge mode-rule">Rule-based fit score</span>', unsafe_allow_html=True)
        jd_text4 = st.text_area("Paste the target job description", height=140, key="t4_jd")
        req_exp4 = st.number_input("Required years of experience", min_value=0.0, max_value=20.0,
                                    value=2.0, step=0.5, key="t4_reqexp")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Candidate A**")
            skills_a = st.text_area("Skills (comma-separated)", key="t4_skills_a", height=80)
            exp_a = st.number_input("Years experience", min_value=0.0, max_value=30.0, value=2.0, step=0.5, key="t4_exp_a")
        with colB:
            st.markdown("**Candidate B**")
            skills_b = st.text_area("Skills (comma-separated)", key="t4_skills_b", height=80)
            exp_b = st.number_input("Years experience", min_value=0.0, max_value=30.0, value=2.0, step=0.5, key="t4_exp_b")

        if st.button("Compare", type="primary", key="t4_manual_btn"):
            if not skills_a.strip() or not skills_b.strip() or not jd_text4.strip():
                st.warning("Please fill in both candidates' skills and the job description.")
            else:
                required_skills4 = set(extract_keywords(jd_text4, top_n=25))
                if not required_skills4:
                    st.warning("Couldn't detect specific skills in that job description.")
                else:
                    set_a = set(s.strip().lower() for s in skills_a.split(","))
                    set_b = set(s.strip().lower() for s in skills_b.split(","))
                    score_a, overlap_a, missing_a = rule_based_fit(set_a, required_skills4, exp_a, req_exp4)
                    score_b, overlap_b, missing_b = rule_based_fit(set_b, required_skills4, exp_b, req_exp4)

                    colA, colB = st.columns(2)
                    with colA:
                        render_match_card("Candidate A", score_a, [f"{exp_a} yrs experience"], overlap_a, missing_a)
                    with colB:
                        render_match_card("Candidate B", score_b, [f"{exp_b} yrs experience"], overlap_b, missing_b)

                    if score_a != score_b:
                        winner = "Candidate A" if score_a > score_b else "Candidate B"
                        st.info(f"**Head-to-head:** {winner} is the stronger fit "
                                f"({max(score_a,score_b):.1f}% vs {min(score_a,score_b):.1f}%).")
                    else:
                        st.info("**Head-to-head:** Both candidates score equally.")

# ============================================================
# TAB 5: Pipeline & Margin - the two things a staffing business actually runs on
# ============================================================
with tab5:
    st.subheader("Bill Rate & Margin Calculator")
    st.caption("What the client pays vs. what the candidate is paid - the margin is the "
               "staffing company's actual revenue on this placement.")

    c1, c2, c3 = st.columns(3)
    with c1:
        bill_rate = st.number_input("Client bill rate (₹/hour)", min_value=0.0, value=1200.0, step=50.0, key="margin_bill")
    with c2:
        pay_rate = st.number_input("Candidate pay rate (₹/hour)", min_value=0.0, value=800.0, step=50.0, key="margin_pay")
    with c3:
        hours = st.number_input("Billable hours / month", min_value=0.0, value=160.0, step=8.0, key="margin_hours")

    if bill_rate > 0:
        margin_per_hour = bill_rate - pay_rate
        margin_pct = (margin_per_hour / bill_rate) * 100 if bill_rate else 0
        monthly_margin = margin_per_hour * hours
        annual_margin = monthly_margin * 12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Margin / Hour", f"₹{margin_per_hour:,.0f}")
        m2.metric("Margin %", f"{margin_pct:.1f}%")
        m3.metric("Monthly Margin", f"₹{monthly_margin:,.0f}")
        m4.metric("Annual Margin", f"₹{annual_margin:,.0f}")

        if margin_pct < 15:
            st.warning(f"Margin is {margin_pct:.1f}% - below the typical 15-25% healthy range for IT "
                       f"staffing placements. Consider renegotiating the bill rate or candidate pay rate.")
        elif margin_pct > 40:
            st.info(f"Margin is {margin_pct:.1f}% - unusually high. Worth checking the bill rate is "
                    f"market-competitive so the client doesn't churn to a cheaper vendor.")
        else:
            st.success(f"Margin is {margin_pct:.1f}% - within a healthy range for IT staffing placements.")

        rate_df = pd.DataFrame({"Rate Type": ["Client Bill Rate", "Candidate Pay Rate"],
                                 "₹/hour": [bill_rate, pay_rate]}).set_index("Rate Type")
        st.bar_chart(rate_df, height=220)

    st.markdown("---")
    st.subheader("Candidate Pipeline Tracker")
    st.caption("Track candidates through the submission funnel - Submitted → Client Review → "
               "Interview → Offer → Placed - the same stages a real staffing CRM (Bullhorn, "
               "JobDiva, Ceipal) tracks.")

    with st.form("pipeline_add_form", clear_on_submit=True):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_candidate = st.text_input("Candidate name/ID")
        with pc2:
            p_requirement = st.text_input("Requirement / Client")
        with pc3:
            p_stage = st.selectbox("Current stage", PIPELINE_STAGES)
        submitted = st.form_submit_button("Add to Pipeline", type="primary")
        if submitted:
            if not p_candidate.strip() or not p_requirement.strip():
                st.warning("Please fill in both candidate and requirement.")
            else:
                st.session_state.pipeline.append({
                    "Candidate": p_candidate.strip(), "Requirement": p_requirement.strip(), "Stage": p_stage
                })
                st.success(f"Added {p_candidate} → {p_requirement} ({p_stage})")

    if st.session_state.pipeline:
        pipeline_df = pd.DataFrame(st.session_state.pipeline)

        stage_counts = pipeline_df["Stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        st.write("**Funnel overview:**")
        st.bar_chart(stage_counts, height=260)

        placed = stage_counts.get("Placed", 0)
        total = len(pipeline_df)
        rejected = stage_counts.get("Rejected", 0)
        active = total - placed - rejected
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Total in Pipeline", total)
        f2.metric("Active", active)
        f3.metric("Placed", placed)
        f4.metric("Conversion Rate", f"{(placed/total*100 if total else 0):.0f}%")

        st.write("**All pipeline entries:**")
        st.dataframe(pipeline_df, width="stretch", hide_index=True)

        if st.button("Clear Pipeline", key="clear_pipeline"):
            st.session_state.pipeline = []
            st.rerun()
    else:
        st.info("No candidates in the pipeline yet - add one above to see the funnel view.")

# ============================================================
# TAB 6: Resume Builder - generates an ATS-friendly resume, then scores it
# with our own ATS scorer to prove the score, not just claim it.
# ============================================================
with tab6:
    st.subheader("Resume Builder")
    st.caption("Generates a clean, single-column .docx built to the same rules our ATS Scorer checks "
               "for - correct section headers, no tables/text boxes (a common reason ATS systems fail "
               "to parse a resume at all), and keyword coverage against a target role.")

    st.markdown("**Personal details**")
    p1, p2 = st.columns(2)
    with p1:
        rb_name = st.text_input("Full name", key="rb_name")
        rb_email = st.text_input("Email", key="rb_email")
    with p2:
        rb_phone = st.text_input("Phone", key="rb_phone")
        rb_linkedin = st.text_input("LinkedIn (optional)", key="rb_linkedin")

    rb_target_jd = st.text_area("Target job description (optional, but strongly recommended)", height=120,
                                 key="rb_jd", placeholder="Paste the job description you're applying to - "
                                 "this tunes keyword suggestions and lets us show your live ATS score.")

    if rb_target_jd.strip():
        suggested_skills = extract_keywords(rb_target_jd, top_n=15)
        if suggested_skills:
            st.info(f"**Skills detected in this JD** (include any you genuinely have): {', '.join(suggested_skills)}")

    rb_summary = st.text_area("Professional summary (2-3 lines)", height=80, key="rb_summary",
                               placeholder="e.g. Data Analyst with experience in Python, SQL, and Power BI...")
    rb_skills = st.text_area("Skills (comma-separated)", height=70, key="rb_skills",
                              placeholder="Python, SQL, Power BI, Excel")

    st.markdown("---")
    st.markdown("**Experience**")
    with st.form("rb_exp_form", clear_on_submit=True):
        e1, e2, e3 = st.columns(3)
        with e1:
            exp_title = st.text_input("Job title")
        with e2:
            exp_company = st.text_input("Company")
        with e3:
            exp_dates = st.text_input("Dates (e.g. Jan 2025 - Jun 2025)")
        exp_bullets = st.text_area("Achievements (one per line - include numbers where possible, "
                                    "e.g. 'Improved load time by 40%')", height=100)
        if st.form_submit_button("Add Experience Entry"):
            if exp_title.strip() and exp_company.strip():
                st.session_state.builder_experience.append({
                    "title": exp_title.strip(), "company": exp_company.strip(), "dates": exp_dates.strip(),
                    "bullets": [b.strip() for b in exp_bullets.split("\n") if b.strip()]
                })
                st.success(f"Added: {exp_title} at {exp_company}")
            else:
                st.warning("Job title and company are required.")

    if st.session_state.builder_experience:
        for i, exp in enumerate(st.session_state.builder_experience):
            st.write(f"- **{exp['title']}** at {exp['company']} ({exp['dates']}) - {len(exp['bullets'])} bullet(s)")
        if st.button("Clear experience entries", key="rb_clear_exp"):
            st.session_state.builder_experience = []
            st.rerun()

    st.markdown("---")
    st.markdown("**Education**")
    with st.form("rb_edu_form", clear_on_submit=True):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            edu_degree = st.text_input("Degree")
        with d2:
            edu_institution = st.text_input("Institution")
        with d3:
            edu_year = st.text_input("Year")
        with d4:
            edu_score = st.text_input("CGPA/Score")
        if st.form_submit_button("Add Education Entry"):
            if edu_degree.strip() and edu_institution.strip():
                st.session_state.builder_education.append({
                    "degree": edu_degree.strip(), "institution": edu_institution.strip(),
                    "year": edu_year.strip(), "score": edu_score.strip()
                })
                st.success(f"Added: {edu_degree}, {edu_institution}")
            else:
                st.warning("Degree and institution are required.")

    if st.session_state.builder_education:
        for edu in st.session_state.builder_education:
            st.write(f"- {edu['degree']}, {edu['institution']} ({edu['year']})")
        if st.button("Clear education entries", key="rb_clear_edu"):
            st.session_state.builder_education = []
            st.rerun()

    st.markdown("---")
    rb_certs = st.text_area("Certifications (one per line, optional)", height=70, key="rb_certs")

    st.markdown("---")
    if st.button("Generate Resume", type="primary", key="rb_generate"):
        if not rb_name.strip() or not rb_skills.strip():
            st.warning("At minimum, please enter your name and skills.")
        else:
            data = {
                "name": rb_name.strip(), "email": rb_email.strip(), "phone": rb_phone.strip(),
                "linkedin": rb_linkedin.strip(), "summary": rb_summary.strip(),
                "skills": [s.strip() for s in rb_skills.split(",") if s.strip()],
                "experience": st.session_state.builder_experience,
                "education": st.session_state.builder_education,
                "certifications": [c.strip() for c in rb_certs.split("\n") if c.strip()],
            }
            out_path = os.path.join(tempfile.gettempdir(), f"resume_{rb_name.strip().replace(' ', '_')}.docx")
            build_resume_docx(data, out_path)

            with open(out_path, "rb") as f:
                resume_bytes = f.read()
            st.download_button("⬇️ Download Resume (.docx)", resume_bytes,
                                file_name=f"{rb_name.strip().replace(' ', '_')}_Resume.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            if rb_target_jd.strip():
                result = score_resume(out_path, rb_target_jd)
                st.write(f"### Live ATS Score: {result['ats_score']}/100")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Keyword Match", f"{result['keyword_match_pct']}%")
                sc2.metric("Quantified Bullets", result["quantified_bullets"])
                sc3.metric("Action Verb Usage", f"{result['action_verb_ratio']}%")
                sc4.metric("Sections Found", f"{len(result['sections_found'])}/6")
                if result["missing_keywords"]:
                    st.write("**Consider adding (if genuinely applicable):**", ", ".join(result["missing_keywords"]))
                for s in result["suggestions"]:
                    st.write(f"- {s}")
            else:
                st.caption("Paste a target job description above and regenerate to see a live ATS score.")

            os.unlink(out_path)
