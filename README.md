# Pi-Index Assessment Engine

The Pi-Index Assessment Engine is a scientometric framework designed to provide objective, transparent, and verifiable assessments of academic research papers. By combining Large Language Models (LLMs) with a blockchain-inspired Proof-of-Stake (PoS) ledger, the system automates the evaluation of scholarly work while ensuring the integrity of the assessment history.

## Overview

This application streamlines the peer review and self-assessment process. It extracts key metrics from PDF research papers, calculates performance scores based on predefined criteria, and tracks these assessments within a local, immutable database. The system is designed to reduce bias in academic evaluation through algorithmic rigor and persistent data tracking.

## Key Features

- **Automated Peer Review**: Utilizes Llama-3.3 LLM integration to analyze papers across eight distinct academic criteria, including Originality, Methodological Rigor, and Future Actionability.
- **Dynamic Weighting**: Implements a self-evolving evaluation standard where the importance of assessment criteria is adjusted dynamically based on the system's current epoch, reflecting evolving research standards.
- **Blockchain-Inspired Integrity**: Each assessment is recorded as a unique, hashed transaction in an internal Proof-of-Stake (PoS) ledger, ensuring that assessment history is tamper-evident.
- **Scope Cartography**: Generates interactive topological maps using graph-based visualization, allowing users to see their research landscape and identify fields and subfields with high epistemic density.
- **ORCID Integration**: Securely connects with the ORCID public registry to authenticate researchers and isolate assessment histories to specific user profiles.

## Technical Architecture

- **Frontend**: Built with Streamlit for a responsive, interactive user interface.
- **Language Models**: Powered by the Groq API (Llama-3.3 / Llama-3.1).
- **Database**: SQLite manages both the papers assessment history and the PoS blockchain ledger.
- **Visualization**: Leverages PyVis for interactive, physics-based network graphing.
- **PDF Processing**: PyMuPDF is utilized for high-fidelity text extraction.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Scientometric_Pi_Index
