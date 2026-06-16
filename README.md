# Titanic EDA — Exploratory Data Analysis

Exploratory data analysis on the Titanic dataset to uncover survival patterns using Python.

## Overview

This project analyzes the RMS Titanic passenger dataset (891 records, 12 features) to identify key factors influencing survival through statistical summaries and visualizations.

## Key Findings

- **Gender** — Females survived at 74.2% vs males at 18.9%
- **Class** — 1st class 63% vs 3rd class 24% survival rate
- **Age** — Children had highest survival (58%), seniors lowest (22.7%)
- **Fare** — Survivors paid 2.2x more on average ($48 vs $22)

## Tech Stack

- Python 3
- Pandas
- Matplotlib
- Seaborn
- NumPy

## Files

| File | Description |
|------|-------------|
| `titanic_eda.py` | Main EDA script |
| `eda_distributions.png` | Feature distribution charts |
| `eda_survival.png` | Survival analysis charts |
| `eda_correlations.png` | Correlation heatmap |
| `eda_class_gender.png` | Class × gender breakdown |

## How to Run

```bash
pip install pandas matplotlib seaborn numpy
python titanic_eda.py
```

## Dataset

Titanic passenger dataset — [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets)
