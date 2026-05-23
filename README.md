# Airline Passenger Satisfaction Prediction

## Overview
This project predicts airline passenger satisfaction from demographic, travel, and service-rating features. It uses a Random Forest classifier on a public airline satisfaction dataset and evaluates the model with accuracy and class-level metrics. The project is best understood as a focused applied machine learning notebook for tabular classification.

## Motivation
Passenger satisfaction prediction is useful for practicing supervised learning on mixed categorical and numeric data. The project demonstrates preprocessing choices, model training, evaluation, and interpretation for a real-world style survey dataset. It also provides a foundation for later work on model robustness, subgroup performance, and feature importance.

## Dataset
- **Source:** Kaggle Airline Passenger Satisfaction dataset.
- **File:** `data/airline_satisfaction.csv`
- **Size:** 25,976 passenger records.
- **Target variable:** `satisfaction`, converted to a binary prediction target.
- **Important features:** passenger age, flight distance, customer type, type of travel, cabin class, delays, and service ratings such as online boarding, seat comfort, cleanliness, baggage handling, and inflight service.
- **Known limitations:** The dataset is a public benchmark and may not reflect current airline operations, regional differences, sampling bias, or longitudinal customer behavior.

## Methods
- Loaded the dataset with pandas.
- Selected passenger demographics, travel context, delay fields, and service-rating fields.
- Encoded categorical variables for model training.
- Trained a Random Forest classifier.
- Evaluated model performance using accuracy and a classification report.

## Results
The notebook reports a Random Forest accuracy of **0.9349** on the test split.

Classification report from the notebook:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `False` | 0.93 | 0.95 | 0.94 | 2890 |
| `True` | 0.94 | 0.91 | 0.93 | 2289 |

## Key Insights
- Survey-based service ratings are strong predictors of passenger satisfaction.
- Random Forest performs well on this structured tabular dataset.
- Both positive and negative satisfaction classes are predicted with balanced performance.
- The project could be extended by analyzing which service categories contribute most to dissatisfaction.

## Limitations
- The notebook does not yet include cross-validation or hyperparameter tuning.
- The project does not test fairness or subgroup robustness across passenger groups.
- The dataset is observational and does not prove causal effects of specific service improvements.
- Some feature engineering and preprocessing choices should be documented more deeply in future iterations.

## Future Improvements
- Add feature importance plots and permutation importance.
- Compare Random Forest with logistic regression, gradient boosting, and calibrated classifiers.
- Add cross-validation and threshold analysis.
- Evaluate subgroup performance by travel type, class, and customer type.
- Convert the final notebook workflow into a lightweight `src/` training script.

## How to Run
```bash
git clone https://github.com/BobbY-24/Airline-Prediction-Random-Forrest.git
cd Airline-Prediction-Random-Forrest
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
jupyter notebook notebooks/airline_satisfaction_random_forest.ipynb
```

Run the notebook cells from top to bottom. The notebook expects the dataset at `data/airline_satisfaction.csv`.
