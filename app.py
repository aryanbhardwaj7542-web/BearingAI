
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.io import loadmat
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

st.set_page_config(
    page_title="BearingAI | Machine Intelligence",
    page_icon="⚙️",
    layout="wide",
)

DATA_FILE = "bearing_features.csv"
MODEL_FILE = "bearing_model.joblib"

FEATURES = [
    "rms",
    "std",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "crest_factor",
]

SIGNALS = {
    "Normal": ("Normal_0.mat", "X097_DE_time"),
    "Ball Fault": ("B007_0.mat", "X118_DE_time"),
    "Inner Race Fault": ("IR007_0.mat", "X105_DE_time"),
    "Outer Race Fault": ("OR007_0.mat", "X130_DE_time"),
}

COLORS = {
    "Normal": "#22c55e",
    "Ball Fault": "#f59e0b",
    "Inner Race Fault": "#ef4444",
    "Outer Race Fault": "#8b5cf6",
}


@st.cache_data
def load_features():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_signal(label):
    filename, key = SIGNALS[label]
    data = loadmat(filename)
    return np.asarray(data[key]).ravel()


@st.cache_data
def evaluate_models(data):
    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = data["fault_type"]

    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced"),
        ),
    }

    output = {}

    for name, estimator in models.items():
        predictions = cross_val_predict(estimator, X, y, cv=cv)

        output[name] = {
            "predictions": predictions,
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(
                y, predictions, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y, predictions, average="macro", zero_division=0
            ),
            "f1": f1_score(
                y, predictions, average="macro", zero_division=0
            ),
            "report": classification_report(
                y, predictions, output_dict=True, zero_division=0
            ),
        }

    return y, output


if not os.path.exists(DATA_FILE):
    st.error("bearing_features.csv not found. Run extract_data.py first.")
    st.stop()

if not os.path.exists(MODEL_FILE):
    st.error("bearing_model.joblib not found. Run train_model.py first.")
    st.stop()

df = load_features()
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

st.sidebar.title("⚙️ BearingAI")
st.sidebar.caption("Intelligent Machine Fault Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Machine Overview",
        "Vibration Analysis",
        "AI Fault Prediction",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset: CWRU Bearing Data")
st.sidebar.caption("Educational prototype")

st.title("⚙️ BearingAI")
st.caption("AI-powered bearing condition monitoring and fault classification")


# --------------------------------------------------
# MACHINE OVERVIEW
# --------------------------------------------------
if page == "Machine Overview":
    st.subheader("Machine Health Dashboard")

    counts = df["fault_type"].value_counts()
    total = len(df)
    normal = int(counts.get("Normal", 0))
    faulty = total - normal

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signal Windows", f"{total:,}")
    c2.metric("Normal Windows", f"{normal:,}")
    c3.metric("Fault Windows", f"{faulty:,}")
    c4.metric("Fault Classes", df["fault_type"].nunique())

    st.info(
        "These are historical dataset counts, not live measurements "
        "from a physical machine."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Dataset Distribution")
        distribution = counts.rename_axis("Fault Type").reset_index(
            name="Samples"
        )
        fig = px.bar(
            distribution,
            x="Fault Type",
            y="Samples",
            color="Fault Type",
            color_discrete_map=COLORS,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(
            fig, width="stretch", key="overview_distribution"
        )

    with right:
        st.markdown("#### Feature Overview")
        selected_feature = st.selectbox(
            "Choose a signal feature",
            FEATURES,
        )
        fig = px.box(
            df,
            x="fault_type",
            y=selected_feature,
            color="fault_type",
            color_discrete_map=COLORS,
            labels={"fault_type": "Fault Type"},
        )
        st.plotly_chart(
            fig, width="stretch", key="overview_feature_box"
        )

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10), width="stretch", hide_index=True)


# --------------------------------------------------
# VIBRATION ANALYSIS
# --------------------------------------------------
elif page == "Vibration Analysis":
    st.subheader("Vibration Signal Analysis")

    label = st.selectbox("Select machine condition", list(SIGNALS.keys()))
    signal = load_signal(label)

    max_start = max(0, len(signal) - 2048)
    step = 2048 if max_start >= 2048 else 1

    start = st.slider(
        "Signal start position",
        min_value=0,
        max_value=max_start,
        value=0,
        step=step,
    )

    segment = signal[start:start + 2048]
    sample_indices = np.arange(start, start + len(segment))

    st.markdown(f"#### Time-Domain Waveform — {label}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_indices,
            y=segment,
            mode="lines",
            name="Vibration",
            line=dict(color=COLORS[label], width=1),
        )
    )
    fig.update_layout(
        xaxis_title="Sample index",
        yaxis_title="Amplitude",
    )
    st.plotly_chart(
        fig, width="stretch", key="vibration_waveform"
    )

    st.markdown("#### Frequency Spectrum")

    sampling_rate = 12000
    frequencies, power = welch(
        segment,
        fs=sampling_rate,
        nperseg=min(512, len(segment)),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=10 * np.log10(power + 1e-15),
            mode="lines",
            name="Power Spectrum",
        )
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power spectral density (dB/Hz)",
    )
    st.plotly_chart(
        fig, width="stretch", key="vibration_frequency"
    )

    st.caption(
        "The frequency plot assumes a 12 kHz sampling rate. Verify "
        "acquisition metadata before interpreting engineering results."
    )


# --------------------------------------------------
# AI FAULT PREDICTION
# --------------------------------------------------
elif page == "AI Fault Prediction":
    st.subheader("AI Fault Prediction")

    st.write(
        "Select a window from the labelled dataset to inspect "
        "the model prediction."
    )

    selected = st.selectbox(
        "Known dataset condition",
        list(SIGNALS.keys()),
    )
    subset = df[df["fault_type"] == selected]

    index = st.number_input(
        "Sample index within this class",
        min_value=0,
        max_value=len(subset) - 1,
        value=0,
        step=1,
    )

    sample = subset.iloc[[int(index)]]
    X_sample = sample[FEATURES]

    predicted = model.predict(X_sample)[0]
    probabilities = model.predict_proba(X_sample)[0]
    confidence = float(np.max(probabilities)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Label", selected)
    c2.metric("Predicted Class", predicted)
    c3.metric("Model Confidence", f"{confidence:.1f}%")

    if predicted == selected:
        st.success("Prediction matches this sample's dataset label.")
    else:
        st.warning("Prediction differs from this sample's dataset label.")

    probability_df = pd.DataFrame(
        {
            "Fault Type": model.classes_,
            "Probability": probabilities,
        }
    )

    fig = px.bar(
        probability_df,
        x="Fault Type",
        y="Probability",
        color="Fault Type",
        color_discrete_map=COLORS,
        title="Predicted Class Probabilities",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig, width="stretch", key="prediction_probabilities"
    )

    st.markdown("#### Input Features")
    st.dataframe(
        sample[FEATURES],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Confidence is a model estimate, not a guarantee of real-world "
        "fault certainty."
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance")

    st.warning(
        "Initial evaluation randomly splits windows from the same recordings. "
        "Metrics may be optimistic and are not an independent real-world "
        "performance estimate."
    )

    if st.button("Calculate Evaluation Metrics"):
        with st.spinner("Evaluating Random Forest and SVM..."):
            labels, results = evaluate_models(df)

        comparison_rows = []

        for name, result in results.items():
            st.markdown(f"### {name}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{result['accuracy']:.3f}")
            m2.metric("Precision", f"{result['precision']:.3f}")
            m3.metric("Recall", f"{result['recall']:.3f}")
            m4.metric("F1-score", f"{result['f1']:.3f}")

            report_df = pd.DataFrame(result["report"]).T.round(3)
            st.markdown("#### Classification Report")
            st.dataframe(report_df, width="stretch")

            matrix = confusion_matrix(
                labels,
                result["predictions"],
                labels=list(model.classes_),
            )

            fig = px.imshow(
                matrix,
                x=list(model.classes_),
                y=list(model.classes_),
                text_auto=True,
                color_continuous_scale="Blues",
                labels={
                    "x": "Predicted Class",
                    "y": "Actual Class",
                    "color": "Samples",
                },
                title=f"{name} — Confusion Matrix",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"confusion_matrix_{name.lower().replace(' ', '_')}",
            )

            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": result["accuracy"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1-score": result["f1"],
                }
            )

        st.markdown("### Model Comparison")
        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison.round(3),
            width="stretch",
            hide_index=True,
        )

        fig = px.bar(
            comparison.melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score",
            ),
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Random Forest vs SVM",
            range_y=[0, 1],
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="model_comparison_chart",
        )

        st.info(
            "For a reliable performance estimate, evaluate on separate "
            "recordings or operating conditions. These preliminary scores "
            "should not be presented as validated real-world accuracy."
        )


st.divider()
st.caption(
    "BearingAI | Python • Streamlit • Plotly • scikit-learn"
)
