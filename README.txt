PHARMASIGHT AI - UNIFIED DASHBOARD (INTEGRATION)
=================================================

WHAT THIS IS
A single Streamlit app that combines the outputs of all three intern
workstreams into one Executive Dashboard, per the FINAL INTEGRATION TEAM
TASKS (steps 1-15) in the project brief:

  1-2.  Repository + common project structure   -> this folder
  3.    Import Trial Prediction output           -> data/clinical_trial_predictions.csv
  4.    Import Disease Prediction output          -> data/disease_risk_predictions.csv
  5.    Import Demand Forecast output              -> data/monthly_forecast.csv, quarterly_forecast.csv,
                                                       30_day_forecast_daily_totals.csv, inventory_recommendation.csv
  6.    Shared data layer                          -> load_* functions in app.py (cached)
  7-8.  API endpoints / connect all models         -> not exposed as a separate REST API; Streamlit
                                                       reads model outputs directly (this was run
                                                       locally / offline, no live model-serving API
                                                       was required for the brief's Streamlit-only
                                                       deployment target)
  9.    Centralized analytics engine               -> compute_kpis() / enterprise_health_score() in app.py
  10.   Executive KPI calculations                 -> same functions, feeding the top KPI cards
  11.   Dashboard backend services                  -> data loaders + KPI functions
  12.   Frontend UI screens                         -> 4 tabs (Trial / Disease / Demand / CEO view)
  13.   Integrate model outputs                     -> all 3 CSV/JSON outputs wired into the tabs
  14.   End-to-end testing                           -> smoke-tested locally (streamlit run, verified
                                                       no runtime errors, all tabs render)
  15.   Deploy local Streamlit application           -> `streamlit run app.py`

HOW TO RUN
  1. cd PharmaSightAI
  2. pip install -r requirements.txt
  3. streamlit run app.py
  4. Open the local URL Streamlit prints (usually http://localhost:8501)

FOLDER STRUCTURE
  PharmaSightAI/
    app.py                                  - main Streamlit app (all 4 tabs + KPIs)
    requirements.txt
    README.txt                              - this file
    data/
      clinical_trial_predictions.csv        - Intern 1 output
      clinical_trial_model_comparison.csv   - Intern 1 output
      clinical_trial_feature_importance.csv - Intern 1 output
      disease_risk_predictions.csv          - Intern 2 output (mine)
      disease_risk_model_comparison.csv     - Intern 2 output (mine)
      disease_risk_shap_importance.csv      - Intern 2 output (mine)
      monthly_forecast.csv                  - Intern 3 output
      quarterly_forecast.csv                - Intern 3 output
      30_day_forecast_daily_totals.csv      - Intern 3 output
      monthly_forecast_by_product.csv       - Intern 3 output
      inventory_recommendation.csv          - Intern 3 output

KPI DEFINITIONS (as computed in app.py)
  - Active Trials        = row count of clinical_trial_predictions.csv
  - High Risk Patients    = count of Predicted_Risk_Category == "High Risk"
  - Demand Growth         = % change from first to last FULL forecast month
                            (the first calendar month in the forecast, 2026-09,
                            only has 4 days of data, so it's excluded to avoid
                            a misleading growth number)
  - Products Needing Reorder = count of Stock_Status == "REORDER NOW"
  - Enterprise Health Score  = 0.40 x Trial Success Rate
                             + 0.35 x (% patients NOT High Risk)
                             + 0.25 x (% products NOT flagged REORDER NOW)

HONEST CAVEATS CARRIED FORWARD FROM THE MODELING STAGES
  - Drug Demand Forecast: per the original forecast README, this dataset has
    one row per product with no real time-series history, so per-product
    lag/rolling/Prophet models don't apply as intended. The forecast uses a
    calendar+category+region seasonal index applied to each product's last
    observed level. Model R2 is near zero (~-0.02) because the underlying
    data has almost no correlation between sales and available features.
  - Disease Risk Prediction: feature-target correlation is near zero
    (max |r| ~ 0.03) in this dataset. The best model (Random Forest) beats
    the majority-class baseline on F1 but not on raw accuracy. Risk Scores
    should be read as directional, not diagnostic.
  - Both caveats are surfaced directly in the dashboard (captions on the
    relevant tabs) rather than hidden, so the CEO view doesn't overstate
    model confidence.
