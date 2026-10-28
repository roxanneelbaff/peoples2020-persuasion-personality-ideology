# Persuasiveness of News Editorials depending on Ideology and Personality

Code for the paper (PEOPLES Workshop @ COLING 2020):

```bibtex
@inproceedings{el-baff-etal-2020-persuasiveness,
    title = "Persuasiveness of News Editorials depending on Ideology and Personality",
    author = "El Baff, Roxanne  and
      Al Khatib, Khalid  and
      Stein, Benno  and
      Wachsmuth, Henning",
    editor = "Nissim, Malvina  and
      Patti, Viviana  and
      Plank, Barbara  and
      Durmus, Esin",
    booktitle = "Proceedings of the Third Workshop on Computational Modeling of People's Opinions, Personality, and Emotion's in Social Media",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.peoples-1.4/",
    pages = "29--40"
}
```

## Overview

**Question:** how do readers with different ideologies and personalities react to the content and style of persuasive text?

- **Data:** 1,000 New York Times editorials, each annotated by 3 liberal and 3 conservative readers as *challenging*, *reinforcing*, or *no effect*. Annotators have a Pew political-typology result and Big Five scores (0–100).
- **Reader profiles:**
  - *Personality traits* – low / average / high per Big Five trait (14 profiles; agreeableness has no "average").
  - *Personality groups* – k-means (k=2) on the five traits: **a** (role models: low neuroticism, high others) vs. **b** ("other").
  - *Ideology intensity* – **extreme** (solid liberal / core conservative) vs. **lean**.
- **Features:**
  - *Style* – LIWC, NRC emotion/polarity lexicons, argumentative discourse units (ADUs), MPQA arguing lexicon, subjectivity.
  - *Content* – TF-IDF of 1–3-gram lemmas.
- **Part 1 – content vs. style models:** per profile, a linear SVM predicts challenging / reinforcing / no effect over all feature combinations (macro-F1, compared against a baseline with significance tests).
- **Part 2 – style × geographic scope:** editorials are topic-modelled (Mallet LDA), topics are manually grouped into **global / national / state**, then per profile, scope and effect pair we test style-feature differences and compute effect sizes (heatmaps).

## Setup

Python 3.8, managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`).

```bash
uv sync
uv run jupyter lab    # open notebooks/
```

Run notebooks from `notebooks/` (they use `../data` and `../out`). Mallet LDA requires a local [Mallet](https://mimno.github.io/Mallet/) install.

## Repository layout

| Path | Content |
|---|---|
| `textmining/` | Library: feature extraction (`features.py`, `lexicons.py`), topic modelling (`topic_modeler.py`), classification (`machine_learning.py`, `news_editorials_experiments.py`), significance / effect sizes (`significance_testing.py`, `cross_models_significance.py`), plots (`analysis.py`, `visualization.py`) |
| `notebooks/` | Experiments (below) |
| `corpus/` | Editorial texts and annotator explanations (`corpus_explanation.csv`) |
| `data/` | Extracted features and annotation labels ( editorial content were removed) |
| `out/` | Generated results |
| `docs/` | Figures (personality clusters) |


## Notebooks

Suggested order: topics → profile notebooks.

| Notebook | What it does | Main outputs |
|---|---|---|
| `lda_cluster_by_topics.ipynb` | Mallet-LDA topic modelling (coherence over k, k=18 used); assigns each editorial a topic and scope group | `out/feature_extraction/lda/*`, `out/editorials_topic_w_group.csv` |
| `nmf_cluster_by_topics.ipynb` (not in paper) | NMF (TF-IDF) topic modelling, alternative to LDA | `out/feature_extraction/nmf/*` |
| `lda_vs_nmf.ipynb` (we chose LDA for the paper) | Compares LDA and NMF coherence and topic distributions | `out/feature_extraction/lda/topics_dstr_18.csv` |
| `personality_analysis.ipynb` | k-means clustering of Big Five into groups a/b; effect-size heatmaps and model training/comparison for each group | `data/personality_all_features.csv`, `out/heatmaps/personality_group*`, `out/significance/*personality*`, `out/style_content_models_results/personality_*` |
| `intensity_analysis.ipynb` | Same pipeline for extreme vs. lean ideology | `data/intensity_all_features.csv`, `out/style_content_models_results/intensity/`, `out/effect-size/`, `out/heatmaps/heatmap_effectsize_{extreme,lean}.pdf` |
| `ideology_analysis.ipynb` | Effect-size analysis for liberal vs. conservative readers | `out/significance/{liberal,conservative}_trainset*`, `out/effect-size/heatmap_effectsize_{liberal,conservative}.pdf` |
| `agreeableness_analysis.ipynb`<br>`conscientiousness_analysis.ipynb`<br>`extraversion_analysis.ipynb`<br>`neuroticism_analysis.ipynb`<br>`openness_analysis.ipynb` | Per-trait (low/average/high) feature building, model training and cross-model significance | `data/<trait>_HAL_all_features.csv` (`_HL_` for neuroticism), `out/style_content_models_results/<trait>_*`, `out/model_pair_comparison/*` |
| `features_effect_sizes.ipynb` | Style-feature significance + Cohen's-d effect sizes per profile × scope (global / national / state / all); paper heatmaps | `out/significance/*`, `out/heatmaps/*` |



## Output files (`out/`)

| Path | Description |
|---|---|
| `editorials_topic_w_group.csv` | Editorial → dominant LDA topic and scope group (global / national / state) |
| `topic_*_content_*.csv`, `topic_5_imm.csv` | Editorial text of selected topics (trade, drug/health, immigration), for manual inspection |
| `feature_extraction/lda/` | Mallet-LDA: coherence per k, topic words, document-topic matrix, dominant topic per document, topic distribution (k = 2, 17, 18) |
| `feature_extraction/nmf/` (not in paper) | Same files for NMF (k = 22, 25) |
| `style_content_models_results/` | Cross-validated SVM scores (macro-F1) per profile, scaling (standard / log / sqrt / minmax) and feature combination; `intensity/` holds extreme/lean |
| `models/` | Per-profile result folders (e.g. `extreme_majority`, `personality_a_majority`, `extraversion_high_majority`) |
| `model_pair_comparison/` | Significance tests between model pairs (e.g. style vs. content vs. baseline), per profile and scaling |
| `significance/` | Feature-level significance tests and effect sizes per profile and scope (`all_`, `global_`, `national_`, `state_` prefixes); `_diss` = dissertation version |
| `effect-size/` | Effect-size heatmaps for liberal / conservative readers |
| `heatmaps/` | Effect-size heatmaps (PDF) per profile and scope; `personality_group*` for groups a/b; `dissertation/` for thesis versions |
| `personaity-clusters-cropped.pdf` | Personality clustering figure (also in `docs/`) |

## Data (`data/`)

| File | Description |
|---|---|
| `*_classes.csv` | Per-editorial majority effect (challenging / no effect / reinforcing) and vote distribution for liberal / conservative readers |
| `articles_features.json`, `pars_features.json` | Extracted features per article / paragraph |
| `articles_with_adu_liwc_lexicons*.json`, `paragraphs_with_adu_liwc_lexicons.json` | Articles/paragraphs with ADU, LIWC and lexicon features (`_content` = with text - not included due to license requirement) |
| `articles_with_majority_adus.json`, `*_adus_annotations.json` | ADU annotations |
| `personality_all_features.csv`, `intensity_all_features.csv`, `<trait>_*_all_features.csv` | Feature matrix and labels per reader-profile experiment |
| `lead_all_comb_clustering.csv` | Feature combinations used for the top-combination selection |
