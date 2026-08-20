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
job role team company work environment opportunity responsibilities requirements required
preferred qualifications must nice degree year plus who what where when why how also
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
]


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
    """Pull the most meaningful keywords/phrases out of a job description.

    Multi-word technical terms (e.g. "machine learning") are matched against a
    curated phrase list rather than naive adjacent-word pairing, which avoids
    producing meaningless bigrams like "power pandas" from words that just
    happen to sit next to each other.
    """
    text_lower = job_description.lower()

    # 1. Detect known multi-word phrases first, and remove them from the text
    #    so their component words aren't double-counted as separate unigrams.
    found_phrases = [p for p in KNOWN_PHRASES if p in text_lower]
    for phrase in found_phrases:
        text_lower = text_lower.replace(phrase, " ")

    # 2. Extract single-word keywords from what remains
    text_clean = text_lower.translate(str.maketrans("", "", string.punctuation.replace("+", "").replace("#", "")))
    words = [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words)
    top_unigrams = [kw for kw, _ in freq.most_common(top_n)]

    # phrases first (they're higher-signal), then unigrams, capped at top_n total
    combined = found_phrases + [w for w in top_unigrams if w not in found_phrases]
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


def score_quantification(resume_text):
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    bullet_lines = [l for l in lines if len(l) > 15]  # rough proxy for content lines
    if not bullet_lines:
        return 0, 0, 0
    quantified = [l for l in bullet_lines if re.search(r"\d+%|\$\d+|\d+x|\d+\+|\b\d{2,}\b", l)]
    ratio = len(quantified) / len(bullet_lines)
    return ratio, len(quantified), len(bullet_lines)


def score_action_verbs(resume_text):
    lines = [l.strip().lower() for l in resume_text.split("\n") if len(l.strip()) > 15]
    if not lines:
        return 0, 0
    # strip leading bullet characters (•, -, *, etc.) before checking the first word,
    # otherwise every bullet line's "first word" is just the bullet symbol
    cleaned_lines = [l.lstrip("•-*·▪◦ \t") for l in lines]
    cleaned_lines = [l for l in cleaned_lines if l]
    strong_start = sum(1 for l in cleaned_lines if l.split() and l.split()[0] in ACTION_VERBS)
    weak_count = sum(1 for l in lines for phrase in WEAK_PHRASES if phrase in l)
    ratio = strong_start / len(cleaned_lines) if cleaned_lines else 0
    return ratio, weak_count


# ---------------------------------------------------------
# 4. Master scoring function
# ---------------------------------------------------------
def score_resume(file_path, job_description):
    resume_text = extract_text(file_path)
    keywords = extract_keywords(job_description)

    kw_match, found_kw, missing_kw = score_keyword_match(resume_text, keywords)
    sec_score, found_sec, missing_sec = score_sections(resume_text)
    quant_ratio, n_quant, n_bullets = score_quantification(resume_text)
    verb_ratio, weak_count = score_action_verbs(resume_text)

    final_score = round(
        (kw_match * 0.50 + sec_score * 0.20 + quant_ratio * 0.15 + verb_ratio * 0.15) * 100, 1
    )

    # ---- Build actionable suggestions ----
    suggestions = []
    if missing_kw:
        suggestions.append(f"Add these missing keywords if relevant to your experience: {', '.join(missing_kw[:8])}")
    if missing_sec:
        suggestions.append(f"Your resume is missing these sections: {', '.join(missing_sec)}")
    if quant_ratio < 0.3:
        suggestions.append("Add numbers to your bullet points (e.g. 'improved load time by 40%') — only "
                            f"{n_quant}/{n_bullets} lines currently have measurable impact.")
    if verb_ratio < 0.3:
        suggestions.append("Start more bullet points with strong action verbs (Built, Led, Optimized, "
                            "Automated) instead of passive phrasing.")
    if weak_count > 0:
        suggestions.append(f"Replace weak phrases like 'responsible for' / 'worked on' found in {weak_count} "
                            "line(s) with specific accomplishments.")
    if not suggestions:
        suggestions.append("Strong resume! Keyword coverage, structure, and impact statements all look solid.")

    return {
        "ats_score": final_score,
        "keyword_match_pct": round(kw_match * 100, 1),
        "matched_keywords": found_kw,
        "missing_keywords": missing_kw,
        "sections_found": found_sec,
        "sections_missing": missing_sec,
        "quantified_bullets": f"{n_quant}/{n_bullets}",
        "action_verb_ratio": round(verb_ratio * 100, 1),
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    # Quick manual test - see the test resume + job description below
    pass
