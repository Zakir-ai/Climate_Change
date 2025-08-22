# Climate Change Sentiment Analysis (NASA Climate Conversation)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-green)
![NLP](https://img.shields.io/badge/NLP-TextBlob%20%7C%20NLTK-lightgrey)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Approach & Methods](#approach--methods)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Requirements (inline)](#requirements-inline)
- [Reproducibility Notes](#reproducibility-notes)
- [Visualizations](#visualizations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## Overview
This project performs end-to-end Natural Language Processing (NLP) and machine learning on a climate conversation dataset to understand public sentiment around climate change. It includes robust text preprocessing, sentiment labeling, exploratory analysis, TF-IDF feature extraction, and a supervised baseline classifier. The deliverable is a clean, reproducible Jupyter workflow suitable for technical review by engineering and data science teams.

## Dataset
- **File:** `climate_nasa.csv`  
- **Rows × Columns:** 522 × 5  
- **Columns:**
  - `date` *(ISO timestamp)*  
  - `likesCount` *(integer)*  
  - `profileName` *(hashed/opaque id)*  
  - `commentsCount` *(integer; may be NaN)*  
  - `text` *(raw user comment)*  
- **Missing values (selected):** `commentsCount` = 278 missing; `text` = 18 missing.  
- **Intended use:** Analyze public discussion and sentiment in responses to climate content.

## Objectives
1. Clean and normalize user-generated text for analysis.  
2. Generate rule-based sentiment labels using continuous polarity.  
3. Explore sentiment distribution, common tokens, and topic cues.  
4. Train a baseline classifier on TF-IDF features and evaluate performance.  
5. Provide a transparent, reproducible workflow and clear documentation.

## Approach & Methods
- **Preprocessing:**  
  - Lowercasing; URL removal; non-alphabetic filtering.  
  - Tokenization (NLTK), stopword removal (NLTK), lemmatization (WordNet).  
  - Construct `CleanText` as the canonical processed field.
- **Sentiment Labeling (Rule-Based):**  
  - Polarity via **TextBlob**.  
  - Thresholding: polarity > 0.1 → *Positive*; polarity < -0.1 → *Negative*; otherwise *Neutral*.  
  - Derived fields: `Polarity`, `Sentiment`.
- **Feature Engineering:**  
  - `TfidfVectorizer(max_features=1000)` on `CleanText`.
- **Modeling:**  
  - Train/test split: 80/20 (stratified by `Sentiment` if available; otherwise random).  
  - Baseline classifier: **LogisticRegression(max_iter=200)**.  
- **Evaluation:**  
  - `classification_report` (precision/recall/F1).  
  - Confusion matrix heatmap for error analysis.  
- **Exploratory Analysis & Plots:**  
  - Sentiment distribution bar chart.  
  - Top-N frequent words overall and by sentiment.  
  - WordCloud per sentiment class.

## Results
- **Baseline (Logistic Regression on TF-IDF, 20% test split ≈ 101 samples):**  
  - *Weighted precision:* ~0.54  
  - *Weighted recall:* ~0.63  
  - *Weighted F1:* ~0.56  
These results establish a transparent reference point for future model improvements (e.g., class rebalancing, hyperparameter tuning, contextual embeddings).

## Repository Structure
.
├─ Climate_Change.ipynb # Main analysis notebook (EDA + NLP + ML)  
├─ climate_nasa.csv # Dataset (522 comments with metadata)  
├─ images/ # (Optional) saved plots for README/GitHub  
└─ README.md # This file  

## Setup & Usage
1. **Clone**
   ```bash
   git clone https://github.com/Zakir-ai/Climate_Change.git
   cd Climate_Change
   ```

2. **Create environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
   If you are not using a requirements.txt, see the Requirements (inline) section below.

4. **Open the notebook**
   ```bash
   jupyter notebook Climate_Change.ipynb
   ```

5. **Update paths if needed**  
   Ensure climate_nasa.csv path in the notebook matches your project layout.

6. **Run all cells**  
   This will: load data → preprocess → label sentiment → EDA → vectorize → train → evaluate → plot.

## Requirements (inline)
If you prefer not to maintain a separate requirements.txt, the minimal stack used by the notebook is:
```
pandas
scikit-learn
nltk
textblob
wordcloud
matplotlib
seaborn
```
NLTK data (first run): the notebook includes `nltk.download('punkt')`, `nltk.download('stopwords')`, and `nltk.download('wordnet')`.  
If your environment cannot access the internet, pre-download these corpora and place them under your nltk_data directory.

## Reproducibility Notes
- Random seeds: Set `random_state=42` where applicable (e.g., train/test split, model).  
- Determinism: Some NLP tokenization steps may vary slightly across library versions; pin versions for strict reproducibility.  
- Data integrity: The pipeline expects non-empty text. Rows with empty or null text are safely handled by preprocessing.  

## Visualizations
Typical figures generated by the notebook:
- Sentiment distribution (Positive / Negative / Neutral).  
- Top-15 most frequent words (bar chart).  
- WordClouds per sentiment class.  
- Confusion matrix for the classifier.  

You may export these as `.png` into `images/` and embed them in this README if desired.

## Limitations
- Rule-based sentiment labeling (TextBlob) can be sensitive to domain-specific phrasing, sarcasm, and negation.  
- Class imbalance (if present) can bias baseline performance.  
- TF-IDF + Logistic Regression is a strong baseline but not state-of-the-art for nuanced sentiment.  

## Future Work
- Replace rule-based labels with human annotations or weak supervision.  
- Try modern text encoders (e.g., transformers) and compare to TF-IDF baselines.  
- Hyperparameter tuning (e.g., class weights, C grid) and cross-validation.  
- Augment with topic modeling and aspect-based sentiment for richer insights.  
- Build a lightweight API or dashboard for interactive analysis.  

## Acknowledgements
- Open-source libraries: pandas, NLTK, TextBlob, scikit-learn, matplotlib, seaborn, wordcloud.  
- Data fields and structure inspired by public climate conversation datasets.  



