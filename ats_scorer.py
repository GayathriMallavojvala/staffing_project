"""
ATS Resume Scorer
-------------------
Given a resume file (PDF/DOCX) and a job description, this module:
  1. Extracts raw text from the resume
  2. Scores it against the job description like a real ATS would
  3. Returns specific, actionable suggestions to improve it

Score components (each is genuinely how real ATS/recruiter screening works):
  - Keyword Match (50%)      : does the resume contain the skills/terms the job needs?
  - Section Completeness (20%): are standard resume sections present?
  - Quantified Impact (15%)  : do bullet points contain numbers/metrics?
  - Action Verbs (15%)       : do bullets start with strong action verbs (not "responsible for")?
"""

import re
import string
from collections import Counter

import pdfplumber
import docx
import wordninja

# ---------------------------------------------------------
# Reference lists used for scoring
# ---------------------------------------------------------
STOPWORDS = set("""
a an the and or but if is are was were be been being of to in on for with as at by from
this that these those it its into your you we our their his her they i he she them us
will can could should would may might must shall not no do does did doing have has had
looking strong candidate plus years experienced skilled proficient knowledge understanding
excellent good great ability strong good working across various including etc using use
job role team teams company work environment opportunity responsibilities requirements required
preferred qualifications must nice degree year plus who what where when why how also
gather collection cleaning fix errors missing run queries patterns hidden trends share
findings advice command clear explain related field basic non complex sources numbers
values reporting stakeholders comfortable translating raw actionable business insights
""".split())

SECTION_KEYWORDS = {
    "Contact Info": ["email", "phone", "linkedin", "@"],
    "Skills": ["skills", "technical skills", "technologies"],
    "Experience": ["experience", "work history", "employment"],
    "Education": ["education", "degree", "university", "college"],
    "Projects": ["projects", "project experience"],
    "Certifications": ["certification", "certified", "certificate"],
}

ACTION_VERBS = set("""
led managed built developed designed implemented created improved increased reduced
optimized launched delivered achieved automated streamlined architected engineered
analyzed deployed migrated integrated collaborated spearheaded drove established
conducted performed generated structured translated trained evaluated executed
coordinated facilitated authored devised formulated pioneered initiated resolved
enhanced accelerated strengthened refined validated tested debugged configured
constructed assembled produced published presented mentored supervised organized
""".split())

WEAK_PHRASES = ["responsible for", "worked on", "helped with", "involved in", "duties included"]

# Known multi-word technical phrases - matched directly instead of naive word-pairing,
# so we never produce meaningless bigrams like "power pandas" from adjacent words.
KNOWN_PHRASES = [
    "machine learning", "deep learning", "power bi", "data visualization",
    "data analysis", "statistical analysis", "natural language processing",
    "computer vision", "data science", "artificial intelligence", "rest apis",
    "core banking", "ci/cd", "cloud computing", "big data", "data engineering",
    "software development", "agile methodology", "version control", "unit testing",
    "object oriented programming", "database management", "data structures",
    "project management", "business intelligence", "hypothesis testing",
    "feature engineering", "a/b testing", "react native", "node.js",
]

# Curated technical/professional skill & tool vocabulary. Final keywords are
# ONLY ever pulled from this list or KNOWN_PHRASES above - never from raw word
# frequency. This is a deliberate precision-over-recall choice: a shorter,
# 100%-relevant keyword list is more useful to a recruiter than a longer list
# padded with generic English words like "find", "like", or "managers".
SKILL_VOCABULARY = set("""
python sql java javascript typescript php ruby golang rust swift kotlin scala
matlab html css react angular vue django flask spring excel tableau looker
pandas numpy tensorflow pytorch keras opencv nltk spacy hadoop spark kafka
airflow dbt snowflake redshift postgresql mysql mongodb oracle sqlite nosql
aws azure gcp docker kubernetes terraform ansible jenkins git github gitlab
linux unix bash devops api graphql microservices statistics forecasting
optimization etl jira confluence salesforce sap sas hipaa hl7 fhir
blockchain cybersecurity networking flutter selenium pytest junit unity
scikit-learn scikitlearn r c c++ c#
""".split())


# ---------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------
def _fix_squished_words(text, length_threshold=13):
    """Some PDFs (certain resume-builder templates) don't embed real space
    characters between words, so extraction tools merge multiple words into
    one long token (e.g. 'MachineLearning,StatisticalModeling'). We split each
    token on punctuation to isolate pure-alphabetic runs, then run
    dictionary-based word segmentation only on runs longer than the
    threshold — leaving normally-spaced text and short tokens untouched so we
    don't risk breaking things that are already correct (e.g. 'TensorFlow').
    """
    # split into alternating (alpha-run, non-alpha separator) pieces, keeping separators
    pattern = re.compile(r"([A-Za-z]+)")

    def fix_token(tok):
        if "@" in tok or "http" in tok.lower():  # never touch emails/URLs
            return tok

        def replace(match):
            run = match.group(1)
            if len(run) >= length_threshold:
                return " " + " ".join(wordninja.split(run)) + " "
            return run
        return pattern.sub(replace, tok)

    fixed_lines = []
    for line in text.split("\n"):
        fixed_lines.append(" ".join(fix_token(tok) for tok in line.split(" ")))
    return "\n".join(fixed_lines)


def extract_text(file_path):
    """Extract raw text from a PDF or DOCX resume file."""
    if file_path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return _fix_squished_words(text)
    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")


# ---------------------------------------------------------
# 2. Keyword extraction from job description
# ---------------------------------------------------------
def extract_keywords(job_description, top_n=20):
    """Pull skill/tool keywords out of a job description.

    Deliberately precision-first: keywords are ONLY accepted if they match our
    curated SKILL_VOCABULARY or KNOWN_PHRASES list - never from raw word
    frequency. An earlier version extracted "frequent words" instead of "real
    skills" and surfaced generic English words like "find", "like", "managers"
    just because they appeared often in JD prose. A recruiter needs a keyword
    list they can trust is actually made of skills, not JD filler.
    """
    job_description = _fix_squished_words(job_description)
    text_lower = job_description.lower()

    # 1. Known multi-word phrases first (higher signal than single words)
    found_phrases = [p for p in KNOWN_PHRASES if p in text_lower]
    for phrase in found_phrases:
        text_lower = text_lower.replace(phrase, " ")

    # 2. Single-word skills - only accepted if in our curated vocabulary.
    # Punctuation is replaced with a SPACE (not deleted) - otherwise sentences
    # like "field.Soft Skills" (no space after the period, common in text
    # copied from web pages) get fused into "fieldsoft".
    punct_to_space = str.maketrans(
        string.punctuation.replace("+", "").replace("#", ""),
        " " * len(string.punctuation.replace("+", "").replace("#", ""))
    )
    text_clean = text_lower.translate(punct_to_space)
    words = text_clean.split()
    found_skills = [w for w in dict.fromkeys(words) if w in SKILL_VOCABULARY]  # dedupe, keep order

    combined = found_phrases + found_skills
    return combined[:top_n]


# ---------------------------------------------------------
# 3. Scoring sub-components
# ---------------------------------------------------------
def score_keyword_match(resume_text, keywords):
    resume_lower = resume_text.lower()
    # fallback: also compare with all spaces removed, so PDFs with irregular/missing
    # spacing (a known extraction issue, see _fix_squished_words) can't cause a false
    # "missing" keyword just because a space landed in an unexpected place
    resume_despaced = resume_lower.replace(" ", "")

    found, missing = [], []
    for kw in keywords:
        kw_despaced = kw.replace(" ", "")
        if kw in resume_lower or kw_despaced in resume_despaced:
            found.append(kw)
        else:
            missing.append(kw)

    match_pct = len(found) / len(keywords) if keywords else 0
    return match_pct, found, missing


def score_sections(resume_text):
    resume_lower = resume_text.lower()
    found_sections, missing_sections = [], []
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in resume_lower for kw in keywords):
            found_sections.append(section)
        else:
            missing_sections.append(section)
    score = len(found_sections) / len(SECTION_KEYWORDS)
    return score, found_sections, missing_sections


CONTACT_LINE_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\+?\d[\d\-\s]{8,}\d")

SECTION_HEADER_TRIGGERS = {
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "education": ["education", "academic background"],
    "projects": ["projects", "project experience", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
}


def classify_lines_by_section(resume_text):
    """Tag each line with which resume section it likely belongs to, by detecting
    header lines (short lines matching common section-title phrasing). This lets
    quantification/action-verb checks look only at Experience/Projects lines -
    treating a skills list or a contact line as if it were an "achievement bullet"
    (an earlier bug) produced nonsense suggestions like asking someone to add a
    metric to their email address.
    """
    lines = [l.strip() for l in resume_text.split("\n")]
    current = None
    classified = []
    for line in lines:
        if not line:
            continue
        low = line.lower().strip(":").strip()
        matched_header = None
        if len(line.split()) <= 4:
            for sec, triggers in SECTION_HEADER_TRIGGERS.items():
                if low in triggers:
                    matched_header = sec
                    break
        if matched_header:
            current = matched_header
            continue  # the header line itself isn't content
        classified.append((line, current))
    return classified


def get_bullet_candidate_lines(resume_text):
    """Lines worth checking for quantification/action-verb quality: content under
    Experience or Projects, excluding contact-info lines. Falls back to the old
    broad heuristic (any sufficiently long line) if section headers couldn't be
    detected at all, so atypically-formatted resumes still get scored.
    """
    classified = classify_lines_by_section(resume_text)
    targeted = [line for line, sec in classified
                if sec in ("experience", "projects") and len(line) > 15
                and not CONTACT_LINE_PATTERN.search(line)]
    if targeted:
        return targeted
    # fallback: no recognizable section headers found at all
    return [line for line, _ in classified if len(line) > 15 and not CONTACT_LINE_PATTERN.search(line)]


def score_quantification(resume_text):
    bullet_lines = get_bullet_candidate_lines(resume_text)
    if not bullet_lines:
        return 0, 0, 0, []
    quantified = [l for l in bullet_lines if re.search(r"\d+%|\$\d+|\d+x|\d+\+|\b\d{2,}\b", l)]
    unquantified = [l for l in bullet_lines if l not in quantified]
    ratio = len(quantified) / len(bullet_lines)
    return ratio, len(quantified), len(bullet_lines), unquantified


def score_action_verbs(resume_text):
    lines = get_bullet_candidate_lines(resume_text)
    if not lines:
        return 0, 0, []
    cleaned_pairs = [(l, l.lstrip("•-*·▪◦ \t")) for l in lines]
    cleaned_pairs = [(orig, cl) for orig, cl in cleaned_pairs if cl]
    strong_start = [orig for orig, cl in cleaned_pairs if cl.split() and cl.split()[0].lower() in ACTION_VERBS]
    weak_lines = [orig for orig, cl in cleaned_pairs
                  if not (cl.split() and cl.split()[0].lower() in ACTION_VERBS)]
    weak_phrase_lines = [orig for orig, cl in cleaned_pairs
                          for phrase in WEAK_PHRASES if phrase in cl.lower()]
    ratio = len(strong_start) / len(cleaned_pairs) if cleaned_pairs else 0
    return ratio, len(weak_phrase_lines), weak_lines


def check_contact_info(resume_text):
    """Real ATS systems reject resumes where basic contact info can't be parsed."""
    issues = []
    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text):
        issues.append("No email address detected - ATS systems and recruiters both need this to reach you.")
    if not re.search(r"(\+?\d[\d\-\s]{8,}\d)", resume_text):
        issues.append("No phone number detected in a standard format.")
    return issues


def check_length(resume_text):
    word_count = len(resume_text.split())
    if word_count < 150:
        return word_count, "Resume looks very short (under 150 words) - ATS systems and recruiters may read this as underqualified or incomplete, even if the content is strong."
    if word_count > 1000:
        return word_count, "Resume looks long (over 1000 words) - for most roles, 1-2 pages is the sweet spot; consider trimming older or less relevant experience."
    return word_count, None


# ---------------------------------------------------------
# 4. Master scoring function
# ---------------------------------------------------------
def score_resume(file_path, job_description):
    resume_text = extract_text(file_path)
    keywords = extract_keywords(job_description)

    kw_match, found_kw, missing_kw = score_keyword_match(resume_text, keywords)
    sec_score, found_sec, missing_sec = score_sections(resume_text)
    quant_ratio, n_quant, n_bullets, unquantified_lines = score_quantification(resume_text)
    verb_ratio, weak_phrase_count, weak_verb_lines = score_action_verbs(resume_text)
    contact_issues = check_contact_info(resume_text)
    word_count, length_issue = check_length(resume_text)

    final_score = round(
        (kw_match * 0.50 + sec_score * 0.20 + quant_ratio * 0.15 + verb_ratio * 0.15) * 100, 1
    )

    # ---- Build specific, prioritized, example-driven suggestions ----
    # Each suggestion is (priority, text) - priority 1 = fix this first (biggest score impact).
    scored_suggestions = []

    if missing_kw:
        impact = round((len(missing_kw) / len(keywords)) * 50) if keywords else 0
        scored_suggestions.append((1,
            f"**Keywords ({impact} points on the table):** add these if genuinely part of your experience — "
            f"{', '.join(missing_kw[:8])}. This is the single biggest lever on your score; keyword match is "
            f"weighted 50%."))

    if missing_sec:
        scored_suggestions.append((1,
            f"**Missing section(s):** add a clearly labeled {', '.join(missing_sec)} section. ATS parsers "
            f"look for these exact headers to categorize your content — without them, correctly-written "
            f"content can still get mis-filed or ignored."))

    if unquantified_lines:
        examples = unquantified_lines[:3]
        example_text = "\n".join(f'  - "{e[:90]}{"..." if len(e) > 90 else ""}"' for e in examples)
        scored_suggestions.append((2,
            f"**Quantify your impact ({n_quant}/{n_bullets} lines have numbers):** these lines could use a "
            f"metric (%, count, $, time saved):\n{example_text}\n"
            f"  Rewrite pattern: \"[Did X] resulting in [Y% / $Y / Y hours saved]\"."))

    if weak_verb_lines:
        examples = weak_verb_lines[:3]
        example_text = "\n".join(f'  - "{e[:90]}{"..." if len(e) > 90 else ""}"' for e in examples)
        scored_suggestions.append((2,
            f"**Weak opening verbs ({verb_ratio*100:.0f}% of lines start strong):** these lines don't open "
            f"with an action verb:\n{example_text}\n"
            f"  Try opening with: Built, Led, Designed, Automated, Reduced, Improved, Delivered."))

    if contact_issues:
        for issue in contact_issues:
            scored_suggestions.append((1, f"**Contact info:** {issue}"))

    if length_issue:
        scored_suggestions.append((3, f"**Length ({word_count} words):** {length_issue}"))

    scored_suggestions.sort(key=lambda x: x[0])
    suggestions = [text for _, text in scored_suggestions]

    if not suggestions:
        suggestions = ["Strong resume — keyword coverage, structure, quantified impact, and contact "
                        "info all look solid for this job description."]

    # Top-3 quick-priority summary for at-a-glance action
    top_priority = suggestions[:3] if len(suggestions) > 1 else []

    return {
        "ats_score": final_score,
        "keyword_match_pct": round(kw_match * 100, 1),
        "matched_keywords": found_kw,
        "missing_keywords": missing_kw,
        "sections_found": found_sec,
        "sections_missing": missing_sec,
        "quantified_bullets": f"{n_quant}/{n_bullets}",
        "action_verb_ratio": round(verb_ratio * 100, 1),
        "word_count": word_count,
        "contact_issues": contact_issues,
        "top_priority_fixes": top_priority,
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    # Quick manual test - see the test resume + job description below
    pass
