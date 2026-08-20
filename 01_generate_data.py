"""
Step 1: Synthetic Data Generation
-----------------------------------
We simulate an IT staffing agency's data:
  - candidates.csv   : candidate profiles (skills, experience, education)
  - jobs.csv         : job postings from clients (required skills, domain)
  - placements.csv   : candidate-job matches with outcome (placed / not placed)

Design principle: outcomes are NOT random. Placement probability depends
on skill overlap + experience fit + a bit of realistic noise, so the
patterns are learnable by an ML model later (just like real data would be).
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------
# 1. Define the skill universe, grouped by domain
# ---------------------------------------------------------
SKILL_POOL = {
    "Web Development": ["HTML", "CSS", "JavaScript", "React", "Angular", "Node.js", "PHP", "Django"],
    "Mobile Development": ["Flutter", "React Native", "Swift", "Kotlin", "Java", "Objective-C"],
    "Cloud & DevOps": ["AWS", "Azure", "Docker", "Kubernetes", "CI/CD", "Terraform", "Linux"],
    "Data & ML": ["Python", "SQL", "Machine Learning", "Pandas", "TensorFlow", "Power BI", "Excel"],
    "Banking/Fintech Software": ["Java", "Spring Boot", "SQL", "Core Banking Systems", "REST APIs"],
    "Healthcare IT": ["HL7", "FHIR", "SQL", "Python", "Java", "HIPAA Compliance"],
}
ALL_SKILLS = sorted(set(s for skills in SKILL_POOL.values() for s in skills))
DOMAINS = list(SKILL_POOL.keys())

EDUCATION_LEVELS = ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc", "Diploma"]
EDU_WEIGHT = [0.4, 0.15, 0.15, 0.1, 0.1, 0.1]

# ---------------------------------------------------------
# 2. Generate Candidates
# ---------------------------------------------------------
N_CANDIDATES = 800

def generate_candidate(cid):
    domain_pref = random.choice(DOMAINS)
    n_skills = np.random.randint(3, 7)
    # 70% of skills come from their preferred domain, 30% random (realistic overlap)
    domain_skills = random.sample(SKILL_POOL[domain_pref], min(n_skills, len(SKILL_POOL[domain_pref])))
    extra_skills = random.sample(ALL_SKILLS, max(0, n_skills - len(domain_skills)))
    skills = list(set(domain_skills + extra_skills))

    experience_years = round(np.random.exponential(scale=3), 1)
    experience_years = min(experience_years, 15)

    education = np.random.choice(EDUCATION_LEVELS, p=EDU_WEIGHT)
    has_certification = np.random.choice([0, 1], p=[0.6, 0.4])

    return {
        "candidate_id": f"C{cid:04d}",
        "preferred_domain": domain_pref,
        "skills": ", ".join(sorted(skills)),
        "num_skills": len(skills),
        "experience_years": experience_years,
        "education": education,
        "has_certification": has_certification,
    }

candidates = pd.DataFrame([generate_candidate(i) for i in range(1, N_CANDIDATES + 1)])

# ---------------------------------------------------------
# 3. Generate Job Postings (from clients across GenZ's verticals)
# ---------------------------------------------------------
N_JOBS = 150

def generate_job(jid):
    domain = random.choice(DOMAINS)
    n_required = np.random.randint(3, 6)
    required_skills = random.sample(SKILL_POOL[domain], min(n_required, len(SKILL_POOL[domain])))
    required_experience = round(np.random.choice([0.5, 1, 2, 3, 5, 7], p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.1]), 1)

    return {
        "job_id": f"J{jid:04d}",
        "domain": domain,
        "required_skills": ", ".join(sorted(required_skills)),
        "num_required_skills": len(required_skills),
        "required_experience": required_experience,
        "client_type": random.choice(["Banking", "Healthcare", "Startup", "Enterprise"]),
    }

jobs = pd.DataFrame([generate_job(i) for i in range(1, N_JOBS + 1)])

# ---------------------------------------------------------
# 4. Generate Candidate-Job Matches + Placement Outcomes
#    (this is the core table our ML model will learn from)
# ---------------------------------------------------------
records = []
for _, cand in candidates.iterrows():
    cand_skills = set(cand["skills"].split(", "))
    # each candidate is matched against 2-4 random job postings (like a recruiter shortlisting)
    matched_jobs = jobs.sample(np.random.randint(2, 5), random_state=None)

    for _, job in matched_jobs.iterrows():
        req_skills = set(job["required_skills"].split(", "))

        skill_overlap = len(cand_skills & req_skills)
        skill_match_ratio = skill_overlap / len(req_skills)  # 0 to 1

        experience_gap = cand["experience_years"] - job["required_experience"]

        # ---- Core logic: placement probability depends on real factors ----
        base_prob = 0.15
        base_prob += skill_match_ratio * 0.55          # skill match matters most
        base_prob += 0.1 if experience_gap >= 0 else -0.15  # meeting experience requirement helps
        base_prob += 0.05 if cand["has_certification"] else 0
        base_prob += 0.03 if cand["preferred_domain"] == job["domain"] else 0
        base_prob += np.random.normal(0, 0.08)          # realistic noise
        placement_prob = np.clip(base_prob, 0.02, 0.97)

        placed = np.random.binomial(1, placement_prob)

        # time to placement (only meaningful if placed) - better matches placed faster
        days_to_place = None
        if placed:
            days_to_place = int(np.clip(np.random.normal(30 - skill_match_ratio * 15, 6), 5, 60))

        records.append({
            "candidate_id": cand["candidate_id"],
            "job_id": job["job_id"],
            "skill_match_ratio": round(skill_match_ratio, 2),
            "experience_gap": round(experience_gap, 1),
            "same_domain": int(cand["preferred_domain"] == job["domain"]),
            "has_certification": cand["has_certification"],
            "placed": placed,
            "days_to_place": days_to_place,
        })

placements = pd.DataFrame(records)

# ---------------------------------------------------------
# 5. Save all three tables
# ---------------------------------------------------------
candidates.to_csv("../data/candidates.csv", index=False)
jobs.to_csv("../data/jobs.csv", index=False)
placements.to_csv("../data/placements.csv", index=False)

print("candidates.csv:", candidates.shape)
print("jobs.csv:", jobs.shape)
print("placements.csv:", placements.shape)
print("\nOverall placement rate:", round(placements["placed"].mean() * 100, 1), "%")
print("\nSample placements:")
print(placements.head())
