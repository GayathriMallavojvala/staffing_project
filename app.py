"""
IT Candidate-Job Matching & Placement Predictor - Dashboard
----------------------------------------------------------------
Three views:
  1. Recruiter View  - pick a job, see ranked candidates with placement probability
  2. Candidate View   - pick a candidate, see their best-fit jobs
  3. ATS Resume Score - upload a resume + job description, get a score + fixes
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile

from ats_scorer import score_resume

st.set_page_config(page_title="GenZ Infotech - Talent Match & ATS Tool", layout="wide")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Load data & model (cached so it doesn't reload on every interaction)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    candidates = pd.read_csv(f"{DATA_DIR}/candidates.csv")
    jobs = pd.read_csv(f"{DATA_DIR}/jobs.csv")
    return candidates, jobs

@st.cache_resource
def load_model():
    model = joblib.load(f"{DATA_DIR}/best_model.pkl")
    feature_cols = joblib.load(f"{DATA_DIR}/model_feature_columns.pkl")
    return model, feature_cols

candidates, jobs = load_data()
model, feature_cols = load_model()

EDU_ORDER = {"Diploma": 1, "B.Sc": 2, "BCA": 2, "B.Tech": 3, "MCA": 4, "M.Tech": 4}


def build_features(cand_row, job_row):
    """Recreate the exact same features used in training, for one candidate-job pair."""
    cand_skills = set(cand_row["skills"].split(", "))
    req_skills = set(job_row["required_skills"].split(", "))

    overlap = cand_skills & req_skills
    missing = req_skills - cand_skills
    extra = cand_skills - req_skills
    jaccard = len(overlap) / len(cand_skills | req_skills) if (cand_skills | req_skills) else 0
    skill_match_ratio = len(overlap) / len(req_skills) if req_skills else 0

    row = {
        "skill_match_ratio": skill_match_ratio,
        "jaccard_similarity": jaccard,
        "skill_overlap_count": len(overlap),
        "skill_missing_count": len(missing),
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


st.title("🎯 IT Candidate-Job Matching & Placement Predictor")
st.caption("Built for GenZ Infotech's IT Staffing vertical")

tab1, tab2, tab3 = st.tabs(["👔 Recruiter View", "🧑‍💻 Candidate View", "📄 ATS Resume Scorer"])

# ============================================================
# TAB 1: Recruiter View - pick a job, rank all candidates
# ============================================================
with tab1:
    st.subheader("Find the best-fit candidates for a job")
    job_id = st.selectbox(
        "Select a job posting",
        jobs["job_id"],
        format_func=lambda jid: f"{jid} — {jobs[jobs.job_id==jid]['domain'].values[0]} "
                                 f"({jobs[jobs.job_id==jid]['client_type'].values[0]})"
    )
    job_row = jobs[jobs["job_id"] == job_id].iloc[0]

    with st.expander("Job details"):
        st.write(f"**Domain:** {job_row['domain']}")
        st.write(f"**Required skills:** {job_row['required_skills']}")
        st.write(f"**Required experience:** {job_row['required_experience']} years")
        st.write(f"**Client type:** {job_row['client_type']}")

    if st.button("Rank Candidates", type="primary"):
        rows = []
        for _, cand_row in candidates.iterrows():
            X, missing, overlap = build_features(cand_row, job_row)
            prob = model.predict_proba(X)[0, 1]
            rows.append({
                "candidate_id": cand_row["candidate_id"],
                "placement_probability": round(prob * 100, 1),
                "skill_match": f"{len(overlap)}/{len(overlap)+len(missing)}",
                "missing_skills": ", ".join(sorted(missing)) if missing else "None",
                "experience_years": cand_row["experience_years"],
                "education": cand_row["education"],
            })
        results = pd.DataFrame(rows).sort_values("placement_probability", ascending=False).head(15)
        st.write(f"**Top 15 candidates for {job_id}:**")
        st.dataframe(
            results.style.background_gradient(subset=["placement_probability"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )

# ============================================================
# TAB 2: Candidate View - pick a candidate, see best-fit jobs
# ============================================================
with tab2:
    st.subheader("Find the best-fit jobs for a candidate")
    cand_id = st.selectbox(
        "Select a candidate",
        candidates["candidate_id"],
        format_func=lambda cid: f"{cid} — {candidates[candidates.candidate_id==cid]['preferred_domain'].values[0]}"
    )
    cand_row = candidates[candidates["candidate_id"] == cand_id].iloc[0]

    with st.expander("Candidate profile"):
        st.write(f"**Preferred domain:** {cand_row['preferred_domain']}")
        st.write(f"**Skills:** {cand_row['skills']}")
        st.write(f"**Experience:** {cand_row['experience_years']} years")
        st.write(f"**Education:** {cand_row['education']}")
        st.write(f"**Certified:** {'Yes' if cand_row['has_certification'] else 'No'}")

    if st.button("Find Best-Fit Jobs", type="primary"):
        rows = []
        for _, job_row in jobs.iterrows():
            X, missing, overlap = build_features(cand_row, job_row)
            prob = model.predict_proba(X)[0, 1]
            rows.append({
                "job_id": job_row["job_id"],
                "domain": job_row["domain"],
                "placement_probability": round(prob * 100, 1),
                "skills_to_gain": ", ".join(sorted(missing)) if missing else "None - full match!",
            })
        results = pd.DataFrame(rows).sort_values("placement_probability", ascending=False).head(10)
        st.write(f"**Top 10 job matches for {cand_id}:**")
        st.dataframe(
            results.style.background_gradient(subset=["placement_probability"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )

# ============================================================
# TAB 3: ATS Resume Scorer
# ============================================================
with tab3:
    st.subheader("Score a resume against a job description")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_resume = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        jd_text = st.text_area("Paste the target job description", height=200,
                                placeholder="Paste a job description here...")

    if st.button("Score Resume", type="primary"):
        if not uploaded_resume or not jd_text.strip():
            st.warning("Please upload a resume AND paste a job description.")
        else:
            suffix = "." + uploaded_resume.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_resume.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing resume..."):
                result = score_resume(tmp_path, jd_text)
            os.unlink(tmp_path)

            score = result["ats_score"]
            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
            st.markdown(f"### ATS Score: :{color}[{score}/100]")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Keyword Match", f"{result['keyword_match_pct']}%")
            c2.metric("Quantified Bullets", result["quantified_bullets"])
            c3.metric("Action Verb Usage", f"{result['action_verb_ratio']}%")
            c4.metric("Sections Found", f"{len(result['sections_found'])}/6")

            st.write("**✅ Matched keywords:**", ", ".join(result["matched_keywords"]) or "None")
            st.write("**❌ Missing keywords:**", ", ".join(result["missing_keywords"]) or "None")
            if result["sections_missing"]:
                st.write("**⚠️ Missing sections:**", ", ".join(result["sections_missing"]))

            st.write("### 💡 Suggestions to improve")
            for s in result["suggestions"]:
                st.write(f"- {s}")
