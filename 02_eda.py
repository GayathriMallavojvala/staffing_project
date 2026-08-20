"""
Step 2: Exploratory Data Analysis (EDA)
------------------------------------------
Goal: Understand what actually drives placement success before building
any ML model. This is the "data analytics" phase - finding patterns,
not predicting yet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

candidates = pd.read_csv("../data/candidates.csv")
jobs = pd.read_csv("../data/jobs.csv")
placements = pd.read_csv("../data/placements.csv")

# Merge everything into one analysis-ready table
# (drop the duplicate has_certification from placements - keep the one from candidates
# to avoid pandas silently renaming it to has_certification_x/_y on merge)
df = placements.drop(columns=["has_certification"]).merge(candidates, on="candidate_id").merge(
    jobs, on="job_id", suffixes=("_cand", "_job")
)

print("=" * 60)
print("1. OVERALL PLACEMENT RATE")
print("=" * 60)
print(f"Total matches: {len(df)}")
print(f"Overall placement rate: {df['placed'].mean()*100:.1f}%")

# ---------------------------------------------------------
# 2. Does skill match ratio actually drive placement?
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("2. PLACEMENT RATE BY SKILL MATCH RATIO")
print("=" * 60)
df["skill_match_bucket"] = pd.cut(
    df["skill_match_ratio"], bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
)
skill_placement = df.groupby("skill_match_bucket", observed=True)["placed"].mean() * 100
print(skill_placement.round(1))

fig, ax = plt.subplots(figsize=(7, 4.5))
skill_placement.plot(kind="bar", ax=ax, color="#2E5090")
ax.set_ylabel("Placement Rate (%)")
ax.set_xlabel("Skill Match Ratio")
ax.set_title("Placement Rate Increases Sharply With Skill Match")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../data/eda_skill_match_vs_placement.png")
plt.close()

# ---------------------------------------------------------
# 3. Experience gap effect
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("3. PLACEMENT RATE BY EXPERIENCE GAP")
print("=" * 60)
df["exp_gap_bucket"] = pd.cut(
    df["experience_gap"], bins=[-20, -2, 0, 2, 20],
    labels=["Underqualified (>2yr short)", "Slightly short", "Meets/exceeds", "Well exceeds"]
)
exp_placement = df.groupby("exp_gap_bucket", observed=True)["placed"].mean() * 100
print(exp_placement.round(1))

fig, ax = plt.subplots(figsize=(7, 4.5))
exp_placement.plot(kind="bar", ax=ax, color="#C0392B")
ax.set_ylabel("Placement Rate (%)")
ax.set_xlabel("Experience vs Requirement")
ax.set_title("Being Underqualified Hurts Placement Chances")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("../data/eda_experience_gap_vs_placement.png")
plt.close()

# ---------------------------------------------------------
# 4. Domain-wise placement rates
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("4. PLACEMENT RATE BY DOMAIN")
print("=" * 60)
domain_placement = df.groupby("domain")["placed"].mean().sort_values(ascending=False) * 100
print(domain_placement.round(1))

fig, ax = plt.subplots(figsize=(8, 4.5))
domain_placement.plot(kind="barh", ax=ax, color="#27AE60")
ax.set_xlabel("Placement Rate (%)")
ax.set_title("Placement Rate by Job Domain")
plt.tight_layout()
plt.savefig("../data/eda_domain_vs_placement.png")
plt.close()

# ---------------------------------------------------------
# 5. Certification impact
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("5. CERTIFICATION IMPACT")
print("=" * 60)
cert_placement = df.groupby("has_certification")["placed"].mean() * 100
print(cert_placement.round(1))

# ---------------------------------------------------------
# 6. Most in-demand skills (from job postings)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("6. TOP 10 MOST IN-DEMAND SKILLS (across job postings)")
print("=" * 60)
all_required = jobs["required_skills"].str.split(", ").explode()
top_skills = all_required.value_counts().head(10)
print(top_skills)

fig, ax = plt.subplots(figsize=(8, 5))
top_skills.sort_values().plot(kind="barh", ax=ax, color="#8E44AD")
ax.set_xlabel("Number of Job Postings Requiring This Skill")
ax.set_title("Top 10 In-Demand Skills")
plt.tight_layout()
plt.savefig("../data/eda_top_skills.png")
plt.close()

# ---------------------------------------------------------
# 7. Days to placement vs skill match (for placed candidates)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("7. DAYS TO PLACEMENT vs SKILL MATCH (placed candidates only)")
print("=" * 60)
placed_df = df[df["placed"] == 1]
corr = placed_df["skill_match_ratio"].corr(placed_df["days_to_place"])
print(f"Correlation between skill match and days-to-place: {corr:.2f}")
print("(negative = better matches get placed faster)")

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=placed_df, x="skill_match_ratio", y="days_to_place", alpha=0.4, ax=ax, color="#2E5090")
sns.regplot(data=placed_df, x="skill_match_ratio", y="days_to_place", scatter=False, ax=ax, color="#C0392B")
ax.set_title("Better Skill Match -> Faster Placement")
ax.set_xlabel("Skill Match Ratio")
ax.set_ylabel("Days to Placement")
plt.tight_layout()
plt.savefig("../data/eda_skillmatch_vs_days.png")
plt.close()

print("\nAll charts saved to data/ folder.")
