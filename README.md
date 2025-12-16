# Data extraction using LLMs

# Overview

Pipeline to extract structured data from scientific literature using large language models (LLMs), as part of our [MR-KG work (medRxiv preprint)](http://dx.doi.org/10.64898/2025.12.14.25342218) for extracting Mendelian Randomization (MR) studies published on PubMed.
The pipeline processes abstracts through multiple LLMs, validates outputs against JSON schemas, and generates manuscript-ready performance assessments.

# Quick start

See DEV.md for setup instructions and development workflow.

# Documentation

- DEV.md - Development setup, environment configuration, and task runners
- DATA.md - Data organization, file structure, and workflow outputs
- ANALYSIS.md - Complete analysis workflow from preprocessing to manuscript assets
- docs/manuscript-assets.md - Manuscript figures and tables generation

# Citation

```bibtex
@article{liu2025mr-kg,
  title = {MR-KG: A knowledge graph of Mendelian randomization evidence powered by large language models},
  url = {http://dx.doi.org/10.64898/2025.12.14.25342218},
  DOI = {10.64898/2025.12.14.25342218},
  publisher = {openRxiv},
  author = {Liu, Yi and Burton, Joshua and Gatua, Winfred and Hemani, Gibran and Gaunt, Tom R},
  year = {2025},
  month = dec
}
```
