# IT Candidate-Job Matching & Placement Predictor

Built for GenZ Infotech's IT Staffing vertical.

## How to run the dashboard
```
pip install -r dashboard/requirements.txt
cd dashboard
streamlit run app.py
```

## Project pipeline (run in order if regenerating from scratch)
1. `src/01_generate_data.py` - synthetic candidate/job/placement data
2. `src/02_eda.py` - exploratory analysis + charts
3. `src/03_feature_engineering.py` - feature engineering + hypothesis tests
4. `src/04_model_training.py` - trains Logistic Regression, Random Forest, XGBoost; saves best model
5. `dashboard/app.py` - Streamlit app (Recruiter view, Candidate view, ATS Resume Scorer)

## Data
All data in `data/` is synthetically generated (see 01_generate_data.py for the generation logic)
since real GenZ Infotech candidate/client data isn't accessible for a student project.
