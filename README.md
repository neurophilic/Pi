# Scholarπ (ScholarPi)

**Automated Algorithmic Academic Rigor Analytics**

Scholarπ is a production-ready, free-to-host web application that evaluates academic papers (PDFs) and calculates a comprehensive "π-Index" score. Moving beyond subjective LLM "vibes," Scholarπ uses Large Language Models strictly as **Data Extractors**, pulling boolean and integer variables from the text. The system then applies strict, transparent Python algorithms to calculate reliable, deterministic scores across 15 distinct evaluation criteria.

---

## Key Features

* **AI as a Data Extractor:** The LLM reads the paper and extracts hard data (e.g., "Number of authors", "Mentions funding: True/False", "Citation count") via a strict JSON schema.
* **Algorithmic Scoring:** Python functions map the AI-extracted variables into clamped 0.0–10.0 scores, guaranteeing mathematical transparency and preventing "rogue" AI scoring.
* **Dynamic Few-Shot Learning:** The system queries its local SQLite database for its most recent successful run and injects it into the prompt. The AI continuously "learns" optimal JSON formatting from its own history.
* **Smart Rate-Limit Fallback:** Deployed on Groq's free tier, the system catches `RateLimitError` exceptions and seamlessly downgrades from a 70B parameter model to an 8B instant model to ensure the UI never crashes.
* **Persistent SQLite Caching:** Papers are hashed (`SHA-256`) and cached for 30 days. Re-evaluating the same paper is instantaneous and costs zero API credits.
* **External Validation:** Integrates with the Semantic Scholar API to verify the novelty and saturation of the research topic in the real world.

---

## The 15 Evaluation Metrics

Scholarπ calculates scores across two main categories:

**Core Criteria**
* **S1 (CharDensity):** Vocabulary depth and sentence complexity.
* **S2 (NumDensity):** Presence of empirical data, tables, and statistical tests.
* **S3 (Reasoning):** Logical flow and handling of counter-arguments.
* **S4 (CitationIntegration):** Literature review depth and citation support.
* **S4b (CitationVolume):** Total citation volume.
* **S5 (AuthorDiversity):** Collaborative spread (authors and institutions).
* **S6 (Expertise):** Domain terminology and methodological rigor.
* **S7 (Novelty):** Explicitly stated innovations and new contributions.
* **S8 (Suggestions):** Actionability of future research directions.
* **S9 (Fees):** Transparency regarding funding and conflicts of interest.
* **S10 (Recency):** Timeliness of the paper's citations.
* **S11 (FieldDiversity):** Interdisciplinary scope.
* **S12 (Validation):** Sample sizes, methodology, and reproducibility.
* **S13 (LogicalCoherence):** Document structure and readability.

**External Discovery Metrics**
* **S14 (WebGroundedUniqueness):** Objective rarity based on a live Semantic Scholar API sweep.
* **S15 (AuthorHIndex):** Evaluated author prominence.

---

## Installation & Local Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/scholar-pi.git](https://github.com/yourusername/scholar-pi.git)
cd scholar-pi
