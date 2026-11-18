"""
Interactive Data Visualization Engine using Plotly.
Provides interactive figures for exploratory analysis, feature distributions,
correlation heatmaps, multi-model ROC curves, and SHAP explainability.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color standards
BLUE_PRIMARY = "#0284c7"
NAVY_DARK = "#0f172a"
RED_ALERT = "#ef4444"
GREEN_SAFE = "#10b981"
AMBER_WARN = "#f59e0b"
FONT_FAMILY = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Applies clean, high-contrast healthcare styling to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, color="#0f172a", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_target_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing class balance (Healthy vs Heart Disease)."""
    target = (df["num"] > 0).astype(int)
    counts = target.value_counts().sort_index()

    labels = ["No Heart Disease (0)", "Heart Disease Present (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [GREEN_SAFE, RED_ALERT]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2.5)),
                textinfo="label+percent+value",
                textfont=dict(family=FONT_FAMILY, size=12, color="#0f172a"),
                hoverinfo="label+percent+value",
            )
        ]
    )
    fig.update_layout(
        title="<b>Target Class Distribution (UCI Cleveland Cohort)</b>",
        showlegend=True,
        margin=dict(t=50, b=30, l=20, r=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return apply_plotly_theme(fig)


def create_age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Interactive histogram of age segmented by disease outcome."""
    target = (df["num"] > 0).astype(int)
    plot_df = df.copy()
    plot_df["Outcome"] = ["Heart Disease" if t == 1 else "Healthy" for t in target]

    fig = px.histogram(
        plot_df,
        x="age",
        color="Outcome",
        barmode="overlay",
        nbins=20,
        color_discrete_map={"Healthy": GREEN_SAFE, "Heart Disease": RED_ALERT},
        opacity=0.8,
        labels={"age": "Age (years)", "count": "Patient Frequency"},
        title="<b>Patient Age Distribution by Diagnostic Outcome</b>",
    )
    fig.update_layout(
        xaxis=dict(title="Age (years)", tickmode="linear", dtick=5, gridcolor="#e2e8f0"),
        yaxis=dict(title="Patient Frequency", gridcolor="#e2e8f0"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def create_cholesterol_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Interactive cholesterol distribution with clinical 200 mg/dl threshold."""
    target = (df["num"] > 0).astype(int)
    plot_df = df.copy()
    plot_df["Outcome"] = ["Heart Disease" if t == 1 else "Healthy" for t in target]

    fig = px.histogram(
        plot_df,
        x="chol",
        color="Outcome",
        barmode="overlay",
        nbins=25,
        color_discrete_map={"Healthy": GREEN_SAFE, "Heart Disease": RED_ALERT},
        opacity=0.8,
        labels={"chol": "Serum Cholesterol (mg/dl)", "count": "Patient Frequency"},
        title="<b>Serum Cholesterol Levels & Clinical Risk Threshold</b>",
    )
    fig.add_vline(
        x=200,
        line_width=2.5,
        line_dash="dash",
        line_color=AMBER_WARN,
        annotation_text="High Cholesterol (> 200 mg/dl)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def create_blood_pressure_chart(df: pd.DataFrame) -> go.Figure:
    """Interactive resting blood pressure distribution with clinical hypertension bounds."""
    target = (df["num"] > 0).astype(int)
    plot_df = df.copy()
    plot_df["Outcome"] = ["Heart Disease" if t == 1 else "Healthy" for t in target]

    fig = px.histogram(
        plot_df,
        x="trestbps",
        color="Outcome",
        barmode="overlay",
        nbins=20,
        color_discrete_map={"Healthy": GREEN_SAFE, "Heart Disease": RED_ALERT},
        opacity=0.8,
        labels={"trestbps": "Resting Blood Pressure (mm Hg)", "count": "Patient Count"},
        title="<b>Resting Blood Pressure Distribution (mm Hg)</b>",
    )
    fig.add_vline(
        x=120,
        line_width=2,
        line_dash="dot",
        line_color=GREEN_SAFE,
        annotation_text="Normal (< 120)",
        annotation_position="top left",
    )
    fig.add_vline(
        x=140,
        line_width=2,
        line_dash="dash",
        line_color=RED_ALERT,
        annotation_text="Hypertension Stage 2 (>= 140)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def create_max_heart_rate_chart(df: pd.DataFrame) -> go.Figure:
    """Interactive maximum heart rate achieved distribution."""
    target = (df["num"] > 0).astype(int)
    plot_df = df.copy()
    plot_df["Outcome"] = ["Heart Disease" if t == 1 else "Healthy" for t in target]

    fig = px.histogram(
        plot_df,
        x="thalach",
        color="Outcome",
        barmode="overlay",
        nbins=20,
        color_discrete_map={"Healthy": GREEN_SAFE, "Heart Disease": RED_ALERT},
        opacity=0.8,
        labels={"thalach": "Max Heart Rate Achieved (bpm)", "count": "Patient Count"},
        title="<b>Maximum Heart Rate Achieved (Stress Test)</b>",
    )
    fig.update_layout(
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation matrix heatmap for all 13 features + binary target."""
    work_df = df.copy()
    if "num" in work_df.columns:
        work_df["target_binary"] = (work_df["num"] > 0).astype(int)
        corr_cols = [c for c in work_df.columns if c != "num"]
    else:
        corr_cols = work_df.columns

    corr_matrix = work_df[corr_cols].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#0f172a"),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="<b>Clinical Feature Correlation Matrix (Pearson Correlation)</b>",
        margin=dict(t=50, b=50, l=80, r=20),
        height=520,
    )
    return apply_plotly_theme(fig)


def create_feature_importance_chart(feature_importances: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing relative Gini feature importances."""
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = [round(item[1] * 100, 2) for item in sorted_items]

    name_map = {
        "thal": "Thalassemia Status (thal)",
        "cp": "Chest Pain Type (cp)",
        "thalach": "Max Heart Rate (thalach)",
        "ca": "Major Vessels Colored (ca)",
        "oldpeak": "ST Depression (oldpeak)",
        "exang": "Exercise Angina (exang)",
        "age": "Age in Years (age)",
        "slope": "ST Slope (slope)",
        "chol": "Serum Cholesterol (chol)",
        "trestbps": "Resting Blood Pressure (trestbps)",
        "sex": "Biological Sex (sex)",
        "restecg": "Resting ECG (restecg)",
        "fbs": "Fasting Blood Sugar (fbs)",
    }
    labels = [name_map.get(f, f) for f in features]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=BLUE_PRIMARY),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(color="#0f172a", size=11, family=FONT_FAMILY),
            )
        ]
    )
    fig.update_layout(
        title="<b>Random Forest Feature Importance (Mean Decrease in Impurity)</b>",
        xaxis=dict(title="Relative Importance (%)", gridcolor="#e2e8f0"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=20, l=180, r=40),
        height=480,
    )
    return apply_plotly_theme(fig)


def create_roc_curves_comparison(models_comparison: Dict[str, Any]) -> go.Figure:
    """Overlaid ROC curves comparing Random Forest, Logistic Regression, and Decision Tree."""
    fig = go.Figure()

    color_map = {
        "Random Forest": "#0284c7",
        "Logistic Regression": "#10b981",
        "Decision Tree": "#f59e0b",
    }

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8"),
            name="Random Guess (AUC = 0.50)",
        )
    )

    for m_name, res in models_comparison.items():
        roc = res.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = res.get("roc_auc", 0.0)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{m_name} (AUC = {auc:.3f})",
                line=dict(color=color_map.get(m_name, "#64748b"), width=2.8),
            )
        )

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Comparison (Held-out Test Cohort)</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1], gridcolor="#e2e8f0"),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0, 1.05], gridcolor="#e2e8f0"),
        margin=dict(t=50, b=20, l=40, r=20),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return apply_plotly_theme(fig)


def create_shap_local_waterfall(shap_analysis: Dict[str, Any]) -> go.Figure:
    """Waterfall or horizontal impact plot showing local SHAP contributions for a specific patient."""
    contributions = shap_analysis.get("contributions", {})
    if not contributions:
        return go.Figure()

    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    sorted_items.reverse()

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [RED_ALERT if v > 0 else GREEN_SAFE for v in values]
    text_labels = [f"{v:+.3f}" for v in values]

    name_map = {
        "thal": "Thalassemia Status",
        "cp": "Chest Pain Type",
        "thalach": "Max Heart Rate",
        "ca": "Major Vessels",
        "oldpeak": "ST Depression",
        "exang": "Exercise Angina",
        "age": "Age",
        "slope": "ST Slope",
        "chol": "Cholesterol",
        "trestbps": "Resting BP",
        "sex": "Sex",
        "restecg": "Resting ECG",
        "fbs": "Fasting Sugar",
    }
    display_names = [name_map.get(f, f) for f in features]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=display_names,
                orientation="h",
                marker=dict(color=colors),
                text=text_labels,
                textposition="outside",
                textfont=dict(color="#0f172a", size=11, family=FONT_FAMILY),
            )
        ]
    )
    fig.update_layout(
        title="<b>Individual Patient SHAP Feature Contributions</b><br><sup>Red = Pushes risk up | Green = Protective / Pushes risk down</sup>",
        xaxis=dict(title="SHAP Value (Log-Odds Impact)", gridcolor="#e2e8f0"),
        margin=dict(t=60, b=30, l=150, r=50),
        height=360,
    )
    return apply_plotly_theme(fig)
