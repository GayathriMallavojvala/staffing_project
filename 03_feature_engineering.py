"""
Step 3: Feature Engineering + Statistical Testing
-----------------------------------------------------
Goal:
  1. Build richer features than plain skill_match_ratio (extra skill detail,
     encoded categoricals) so the ML model has more signal to learn from.
  2. Statistically CONFIRM what EDA suggested visually - a chart showing a
     difference isn't proof; a hypothesis test is.
"""

import pandas as pd
import numpy as np
from scipy import stats

candidates = pd.read_csv("../data/candidates.csv")
jobs = pd.read_csv("../data/jobs.csv")
placements = pd.read_csv("../data/placements.csv").drop(columns=["has_certification"])

df = placements.merge(candidates, on="candidate_id").merge(jobs, on="job_id", suffixes=("_cand", "_job"))

# ---------------------------------------------------------
# 1. Richer skill-matching features (beyond the simple ratio)
# ---------------------------------------------------------
def skill_features(row):
    cand_skills = set(row["skills"].split(", "))
    req_skills = set(row["required_skills"].split(", "))

    overlap = cand_skills & req_skills
    missing = req_skills - cand_skills          # what the candidate LACKS for this job
    extra = cand_skills - req_skills             # skills beyond what's required

    jaccard = len(overlap) / len(cand_skills | req_skills) if (cand_skills | req_skills) else 0

    return pd.Series({
        "skill_overlap_count": len(overlap),
        "skill_missing_count": len(missing),
        "skill_extra_count": len(extra),
        "jaccard_similarity": round(jaccard, 3),
    })

df = pd.concat([df, df.apply(skill_features, axis=1)], axis=1)

# ---------------------------------------------------------
# 2. Encode categorical variables for ML
# ---------------------------------------------------------
# Education: ordinal encoding (higher degree = higher number) - this is a
# meaningful order, not just arbitrary categories, so ordinal encoding fits
# better here than one-hot encoding.
EDU_ORDER = {"Diploma": 1, "B.Sc": 2, "BCA": 2, "B.Tech": 3, "MCA": 4, "M.Tech": 4}
df["education_level"] = df["education"].map(EDU_ORDER)

# Domain and client_type: one-hot encoding, since there's no natural order
df = pd.get_dummies(df, columns=["domain", "client_type"], prefix=["domain", "client"])

# ---------------------------------------------------------
# 3. Statistical significance testing
# ---------------------------------------------------------
print("=" * 60)
print("HYPOTHESIS TESTS - confirming EDA findings are real, not noise")
print("=" * 60)

placed = df[df["placed"] == 1]
not_placed = df[df["placed"] == 0]

# Test 1: Does skill_match_ratio differ significantly between placed vs not placed?
t_stat, p_val = stats.ttest_ind(placed["skill_match_ratio"], not_placed["skill_match_ratio"])
print(f"\n[T-test] skill_match_ratio: placed vs not placed")
print(f"  Placed mean: {placed['skill_match_ratio'].mean():.3f}  |  Not placed mean: {not_placed['skill_match_ratio'].mean():.3f}")
print(f"  t-statistic: {t_stat:.2f}, p-value: {p_val:.2e}")
print(f"  -> {'STATISTICALLY SIGNIFICANT' if p_val < 0.05 else 'not significant'} (threshold p < 0.05)")

# Test 2: Is certification independent of placement outcome? (Chi-square test)
contingency = pd.crosstab(df["has_certification"], df["placed"])
chi2, p_val2, dof, expected = stats.chi2_contingency(contingency)
print(f"\n[Chi-square test] has_certification vs placed")
print(contingency)
print(f"  chi2: {chi2:.2f}, p-value: {p_val2:.4f}")
print(f"  -> {'STATISTICALLY SIGNIFICANT' if p_val2 < 0.05 else 'NOT statistically significant'} (threshold p < 0.05)")

# Test 3: Does experience_gap differ significantly between placed vs not placed?
t_stat3, p_val3 = stats.ttest_ind(placed["experience_gap"], not_placed["experience_gap"])
print(f"\n[T-test] experience_gap: placed vs not placed")
print(f"  Placed mean: {placed['experience_gap'].mean():.2f}  |  Not placed mean: {not_placed['experience_gap'].mean():.2f}")
print(f"  t-statistic: {t_stat3:.2f}, p-value: {p_val3:.2e}")
print(f"  -> {'STATISTICALLY SIGNIFICANT' if p_val3 < 0.05 else 'not significant'} (threshold p < 0.05)")

# ---------------------------------------------------------
# 4. Save the final engineered feature table for modeling
# ---------------------------------------------------------
feature_cols = [
    "skill_match_ratio", "jaccard_similarity", "skill_overlap_count",
    "skill_missing_count", "skill_extra_count", "experience_gap",
    "same_domain", "has_certification", "education_level",
] + [c for c in df.columns if c.startswith("domain_") or c.startswith("client_")]

model_df = df[feature_cols + ["placed", "days_to_place"]].copy()
model_df.to_csv("../data/model_features.csv", index=False)

print("\n" + "=" * 60)
print(f"Final feature table saved: {model_df.shape[0]} rows, {len(feature_cols)} features")
print("=" * 60)
print("Feature columns:", feature_cols)
