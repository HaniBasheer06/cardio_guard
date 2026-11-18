# UCI Heart Disease Dataset (Cleveland)

## Overview
This directory contains dataset resources for the **Heart Disease Prediction Using Machine Learning** project.

- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Donor:** Cleveland Clinic Foundation (Robert Detrano, M.D., Ph.D.)
- **Instances:** 303 patient records
- **Input Features:** 13 clinical features
- **Target Variable:** `num` (Presence/Absence of Heart Disease)

---

## Feature Dictionary

| # | Feature Name | Description | Data Type | Permissible / Example Values |
|---|---|---|---|---|
| 1 | `age` | Patient age in years | Numerical | 29 to 77 |
| 2 | `sex` | Biological sex | Categorical | `1` = Male, `0` = Female |
| 3 | `cp` | Chest pain type | Categorical | `1`: Typical angina, `2`: Atypical angina, `3`: Non-anginal pain, `4`: Asymptomatic |
| 4 | `trestbps` | Resting blood pressure | Numerical | 94 to 200 mm Hg (on admission) |
| 5 | `chol` | Serum cholestoral | Numerical | 126 to 564 mg/dl |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Categorical | `1` = True, `0` = False |
| 7 | `restecg` | Resting electrocardiographic results | Categorical | `0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | Numerical | 71 to 202 bpm |
| 9 | `exang` | Exercise-induced angina | Categorical | `1` = Yes, `0` = No |
| 10 | `oldpeak` | ST depression induced by exercise | Numerical | 0.0 to 6.2 (relative to rest) |
| 11 | `slope` | Slope of the peak exercise ST segment | Categorical | `1`: Upsloping, `2`: Flat, `3`: Downsloping |
| 12 | `ca` | Number of major vessels colored by fluoroscopy | Numerical / Discrete | 0, 1, 2, 3 (4 missing entries imputed during preprocessing) |
| 13 | `thal` | Thalassemia status | Categorical | `3`: Normal, `6`: Fixed defect, `7`: Reversible defect (2 missing entries imputed) |

---

## Target Binarization
The original dataset's `num` target has values `0`, `1`, `2`, `3`, and `4`.
`0` represents no disease (<50% diameter narrowing); values `1` through `4`
represent disease at increasing severity levels (>50% diameter narrowing).

For binary classification, the project maps **`num == 0` to class 0** and
**every `num > 0` to class 1**. The binary labels are educational benchmark
targets, not diagnoses.

---

## Local Data Cache
The dataset is downloaded automatically using `ucimlrepo.fetch_ucirepo(id=45)` and cached locally as:
`data/heart_disease.csv` to ensure high availability and offline training capability.
