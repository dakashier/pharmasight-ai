"""
PHARMASIGHT AI - Life Sciences Intelligence Platform
Unified Executive Dashboard integrating:
  - Intern 1: Clinical Trial Success Prediction
  - Intern 2: Disease Risk Prediction
  - Intern 3: Drug Demand Forecasting
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaSight AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# ------------------------------------------------------------------
# DATA LAYER (cached loaders)
# ------------------------------------------------------------------
@st.cache_data
def load_clinical_trials():
    df = pd.read_csv(DATA_DIR / "clinical_trial_predictions.csv")
    fi = pd.read_csv(DATA_DIR / "clinical_trial_feature_importance.csv")
    mc = pd.read_csv(DATA_DIR / "clinical_trial_model_comparison.csv")
    return df, fi, mc

@st.cache_data
def load_disease_risk():
    df = pd.read_csv(DATA_DIR / "disease_risk_predictions.csv")
    shap_imp = pd.read_csv(DATA_DIR / "disease_risk_shap_importance.csv")
    mc = pd.read_csv(DATA_DIR / "disease_risk_model_comparison.csv")
    return df, shap_imp, mc

@st.cache_data
def load_demand_forecast():
    monthly = pd.read_csv(DATA_DIR / "monthly_forecast.csv")
    quarterly = pd.read_csv(DATA_DIR / "quarterly_forecast.csv")
    daily = pd.read_csv(DATA_DIR / "30_day_forecast_daily_totals.csv", parse_dates=["Date"])
    inventory = pd.read_csv(DATA_DIR / "inventory_recommendation.csv")
    by_product = pd.read_csv(DATA_DIR / "monthly_forecast_by_product.csv")
    return monthly, quarterly, daily, inventory, by_product

trials_df, trial_feat_imp, trial_model_comp = load_clinical_trials()
risk_df, risk_shap_imp, risk_model_comp = load_disease_risk()
monthly_fc, quarterly_fc, daily_fc, inventory_df, by_product_fc = load_demand_forecast()

# ------------------------------------------------------------------
# CENTRALIZED ANALYTICS ENGINE / KPI CALCULATIONS
# ------------------------------------------------------------------
def compute_kpis():
    active_trials = len(trials_df)

    high_risk_patients = int((risk_df["Predicted_Risk_Category"] == "High Risk").sum())

    # Use only full forecast months (skip partial first month) for a fair growth read
    full_months = monthly_fc.iloc[1:].reset_index(drop=True)
    if len(full_months) >= 2:
        growth = (
            (full_months["Total_Predicted_Sales_Volume"].iloc[-1]
             - full_months["Total_Predicted_Sales_Volume"].iloc[0])
            / full_months["Total_Predicted_Sales_Volume"].iloc[0]
        ) * 100
    else:
        growth = 0.0

    reorder_now = int((inventory_df["Stock_Status"] == "REORDER NOW").sum())

    return {
        "active_trials": active_trials,
        "high_risk_patients": high_risk_patients,
        "demand_growth_pct": growth,
        "inventory_risk_products": reorder_now,
    }

kpis = compute_kpis()

def enterprise_health_score(kpis):
    """Blend the three domains into a single 0-100 health indicator."""
    trial_success_rate = (trials_df["Outcome"] == "Success").mean() * 100
    low_medium_share = (risk_df["Predicted_Risk_Category"] != "High Risk").mean() * 100
    inventory_ok_share = (1 - kpis["inventory_risk_products"] / len(inventory_df)) * 100
    score = (trial_success_rate * 0.4 + low_medium_share * 0.35 + inventory_ok_share * 0.25)
    return round(score, 1), round(trial_success_rate, 1)

health_score, trial_success_rate = enterprise_health_score(kpis)

# ------------------------------------------------------------------
# HEADER + KPI CARDS
# ------------------------------------------------------------------
st.title("💊 PharmaSight AI Dashboard")
st.caption("Unified Clinical Trial, Disease Risk & Drug Demand Intelligence")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Active Trials", f"{kpis['active_trials']:,}")
k2.metric("High Risk Patients", f"{kpis['high_risk_patients']:,}")
k3.metric("Demand Growth (Oct→Dec)", f"{kpis['demand_growth_pct']:.1f}%")
k4.metric("Products Needing Reorder", f"{kpis['inventory_risk_products']:,}")

st.divider()

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Clinical Trial Intelligence",
    "🩺 Disease Risk Intelligence",
    "📦 Drug Demand Forecasting",
    "📊 CEO Executive View",
])

# ==================== TAB 1: CLINICAL TRIAL ====================
with tab1:
    st.subheader("Clinical Trial Success Rate")
    c1, c2 = st.columns([1, 2])
    with c1:
        outcome_counts = trials_df["Outcome"].value_counts()
        fig = px.pie(
            values=outcome_counts.values, names=outcome_counts.index,
            color=outcome_counts.index,
            color_discrete_map={"Success": "#2ca02c", "Failure": "#d62728"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**Success rate: {trial_success_rate:.1f}%** across {len(trials_df):,} trials")
        st.dataframe(trial_model_comp, use_container_width=True, hide_index=True)

    st.subheader("Top Risk Trials")
    top_risk = trials_df.sort_values("Risk_Score", ascending=False).head(15)
    st.dataframe(
        top_risk[["Trial_ID", "Disease_Area", "Trial_Phase", "Success_Probability",
                  "Risk_Score", "Trial_Risk_Category"]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Feature Importance")
    fig2 = px.bar(
        trial_feat_imp.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig2.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Filter trials by disease area / phase"):
        areas = st.multiselect("Disease Area", sorted(trials_df["Disease_Area"].unique()))
        phases = st.multiselect("Trial Phase", sorted(trials_df["Trial_Phase"].unique()))
        filtered = trials_df.copy()
        if areas:
            filtered = filtered[filtered["Disease_Area"].isin(areas)]
        if phases:
            filtered = filtered[filtered["Trial_Phase"].isin(phases)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

# ==================== TAB 2: DISEASE RISK ====================
with tab2:
    st.subheader("Patient Risk Distribution")
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_counts = risk_df["Predicted_Risk_Category"].value_counts().reindex(
            ["High Risk", "Medium Risk", "Low Risk"]
        )
        fig3 = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            labels={"x": "Risk Category", "y": "Patients"},
        )
        fig3.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.dataframe(risk_model_comp, use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ Note: feature-target correlation in this dataset is very low "
            "(max |r| ≈ 0.03). Model outperforms the majority-class baseline "
            "on F1 but not on raw accuracy — treat Risk Scores as directional "
            "planning signals, not clinical diagnoses."
        )

    st.subheader("Risk Factors (SHAP Importance)")
    fig4 = px.bar(
        risk_shap_imp.sort_values("Mean_Abs_SHAP_Value"),
        x="Mean_Abs_SHAP_Value", y="Feature", orientation="h",
        color="Mean_Abs_SHAP_Value", color_continuous_scale="Oranges",
    )
    fig4.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Patient Search")
    search_id = st.text_input("Enter Patient ID (e.g. PAT00001)")
    if search_id:
        match = risk_df[risk_df["Patient_ID"].str.upper() == search_id.strip().upper()]
        if len(match):
            row = match.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Score", f"{row['Risk_Score']:.1f}")
            m2.metric("Predicted Category", row["Predicted_Risk_Category"])
            m3.metric("Recommendation", row["Recommendation"])
        else:
            st.warning("Patient ID not found.")
    else:
        st.dataframe(risk_df.head(20), use_container_width=True, hide_index=True)

# ==================== TAB 3: DRUG DEMAND ====================
with tab3:
    st.subheader("Revenue & Demand Forecast")
    fig5 = px.line(
        daily_fc, x="Date", y="Total_Predicted_Sales_Volume",
        markers=True, labels={"Total_Predicted_Sales_Volume": "Predicted Sales Volume"},
    )
    fig5.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly Forecast**")
        st.dataframe(monthly_fc, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Quarterly Forecast**")
        st.dataframe(quarterly_fc, use_container_width=True, hide_index=True)
        st.caption("2026Q3 covers only 4 forecast days — not comparable to a full quarter.")

    st.subheader("Top Forecasted Drugs")
    top_products = (
        by_product_fc.groupby("Product_ID")["Predicted_Sales_Volume_Month"].sum()
        .sort_values(ascending=False).head(15).reset_index()
        .rename(columns={"Predicted_Sales_Volume_Month": "Total_Predicted_Sales_Volume"})
    )
    top_products = top_products.merge(
        inventory_df[["Product_ID", "Current_Inventory_Level"]], on="Product_ID", how="left"
    )
    st.dataframe(top_products, use_container_width=True, hide_index=True)

    st.subheader("Stock Out Prediction")
    stockout = inventory_df[inventory_df["Stock_Status"] == "REORDER NOW"]
    c3, c4 = st.columns([1, 2])
    with c3:
        by_region = stockout["Region"].value_counts()
        fig6 = px.pie(values=by_region.values, names=by_region.index, hole=0.45)
        fig6.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig6, use_container_width=True)
    with c4:
        st.dataframe(
            stockout[["Product_ID", "Drug_Category", "Region", "Stock_Status",
                      "Suggested_Order_Qty"]].head(20),
            use_container_width=True, hide_index=True,
        )

# ==================== TAB 4: CEO EXECUTIVE VIEW ====================
with tab4:
    st.subheader("Enterprise Health Score")
    st.markdown(f"### {health_score}/100")
    st.progress(min(int(health_score), 100) / 100)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Clinical Trial Success Rate", f"{trial_success_rate:.1f}%")
    e2.metric("High Risk Patients", f"{kpis['high_risk_patients']:,}")
    e3.metric("Drug Demand Growth", f"{kpis['demand_growth_pct']:.1f}%")
    e4.metric("Potential Stockouts", f"{kpis['inventory_risk_products']:,}")

    st.subheader("AI Recommendations")
    recs = []
    if trial_success_rate < 70:
        recs.append("Prioritize high success-probability trials; review low-phase, high-adverse-event trials.")
    else:
        recs.append("Continue prioritizing high-probability trials to sustain success rate.")
    if kpis["high_risk_patients"] / len(risk_df) > 0.1:
        recs.append("Monitor high-risk patient segments closely; escalate clinical follow-up capacity.")
    else:
        recs.append("Maintain current monitoring cadence for high-risk patient segments.")
    if kpis["demand_growth_pct"] > 0:
        recs.append("Increase inventory for fast-moving drug categories ahead of projected demand growth.")
    else:
        recs.append("Demand is trending down over the forecast window — avoid overstocking; right-size orders.")
    recs.append(f"Address {kpis['inventory_risk_products']:,} products flagged REORDER NOW to avoid stockouts.")

    for r in recs:
        st.markdown(f"- {r}")

    st.divider()
    st.markdown(f"## Overall Life Sciences Intelligence Score: **{health_score}/100**")
    st.caption(
        "Score blends clinical trial success rate (40%), share of patients "
        "not flagged High Risk (35%), and share of products not flagged "
        "REORDER NOW (25%)."
    )

st.divider()
st.caption("PharmaSight AI — Intern 1 (Clinical Trials) · Intern 2 (Disease Risk) · Intern 3 (Drug Demand) · Integration by Intern 2")