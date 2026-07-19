# Dynamic Epistemic Cartography

**A Real-Time Multi-Criteria Scientometric Framework for Emerging Trend Detection and Institutional Assessment**

Dynamic Epistemic Cartography is an advanced, AI-driven evaluation engine designed to map emerging scientific trends and dynamically assess research quality. Evolving from the static ScholarPi algorithmic system, this framework addresses the limitations of traditional, lagging bibliometric indicators (like citation counts and h-index) by leveraging Large Language Models (LLMs) and advanced Multi-Criteria Decision Analysis (MCDA).

The system aligns directly with stringent national evaluation metrics, such as the Italian ANVUR Research Quality Evaluation (VQR), making it a powerful tool for institutions (e.g., Politecnico di Milano) to perform continuous, real-time portfolio optimization.

---

## Core Features

*   **VQR-Aligned Semantic Extraction:** Utilizes Groq's fast inference (Llama 3 models) to read PDF manuscripts and assess 8 critical dimensions of research quality, including Originality (Proxy for the Disruption Index $CD_t$), Methodological Rigor, and Societal Impact (Third Mission).
*   **Entropy Weight Method (EWM):** Autonomously recalculates the mathematical weight of each evaluation criterion every 30 days. By measuring data dispersion (Shannon Entropy), the system rewards high-variance, disruptive frontiers and penalizes saturated metrics, effectively neutralizing Goodhart's Law.
*   **Topological Trend Mapping:** Automatically generates a living spatial network graph of LLM-extracted keywords. Tracks keyword co-occurrence over 30-day epochs to visually detect the early convergence of disparate fields and emerging interdisciplinary frontiers.
*   **Decentralized Caching:** Built-in SQLite database ensures that previously processed manuscripts are instantly retrieved without redundant API calls.

---

## The Evaluation Matrix

The framework synthesizes scientific evaluation into the following 8 dynamic criteria:

1.  **$C_1$ Originality:** Disruption index proxy.
2.  **$C_2$ Methodological Rigor:** Robustness and reproducibility.
3.  **$C_3$ Interdisciplinary Synthesis:** Integration of distant domains.
4.  **$C_4$ Societal Impact:** Real-world applicability (Third Mission).
5.  **$C_5$ Open Science:** Transparent protocols and open data availability.
6.  **$C_6$ Literature Integration:** Grounding claims in relevant literature.
7.  **$C_7$ Empirical Density:** Data density relative to narrative.
8.  **$C_8$ Future Actionability:** Clarity of future research vectors.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your machine. You will also need a free [Groq API Key](https://console.groq.com/keys) to power the LLM extraction.

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/dynamic-epistemic-cartography.git](https://github.com/yourusername/dynamic-epistemic-cartography.git)
cd dynamic-epistemic-cartography
