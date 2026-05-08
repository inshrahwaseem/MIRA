"""
MIRA Mood Tracking & Clustering Pipeline.

Includes:
  - EmotionFeatureEngineer: sklearn transformer producing 20-dim feature vectors
  - MoodClusteringEngine: KMeans (auto-k) + DBSCAN for personal mood profiles
  - MoodDriftAnalyzer: LSTM + linear regression + EWMA for 7-day mood drift
  - EmotionTriggerMiner: Apriori for trigger-pattern discovery
  - CrisisClassifier: VotingClassifier (RF + GBM) + SHAP explainability
  - MoodVisualizer: Plotly dark-themed charts (t-SNE, UMAP, radar, calendar)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DriftReport:
    """Output of MoodDriftAnalyzer.detect_drift()."""

    trend_direction: str   # "improving" | "declining" | "stable"
    rate_of_change: float
    predicted_next_day: float
    alert_level: int       # 0–3
    explanation: str


@dataclass
class TriggerRule:
    """Output of EmotionTriggerMiner.mine_triggers()."""

    antecedent_words: list[str]
    consequent_emotion: str
    confidence: float
    lift: float
    explanation: str


@dataclass
class ClusterResult:
    """Output of MoodClusteringEngine.predict()."""

    cluster_id: int
    cluster_name: str
    confidence: float


# ---------------------------------------------------------------------------
# 1. EmotionFeatureEngineer — sklearn TransformerMixin
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "valence", "arousal", "dominance",
    "joy_score", "sadness_score", "fear_score",
    "anger_score", "surprise_score", "disgust_score",
    "trust_score", "anticipation_score",
    "voice_energy", "voice_pitch_mean",
    "face_confidence", "posture_score", "fidget_score",
    "distortion_count", "masking_probability",
    "hour_of_day_sin", "day_of_week_sin",
]


class EmotionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms raw session dicts into a 20-dim numeric matrix."""

    def __init__(self) -> None:
        self._imputer = KNNImputer(n_neighbors=3)

    def fit(self, session_dicts: list[dict[str, Any]], y: Any = None):
        """Fit the KNN imputer on historical sessions."""
        feature_matrix = self._extract_matrix(session_dicts)
        self._imputer.fit(feature_matrix)
        return self

    def transform(self, session_dicts: list[dict[str, Any]]) -> np.ndarray:
        """Return (n_sessions, 20) imputed feature matrix."""
        feature_matrix = self._extract_matrix(session_dicts)
        return self._imputer.transform(feature_matrix)

    def _extract_matrix(self, session_dicts: list[dict[str, Any]]) -> np.ndarray:
        """Build raw matrix — NaN for missing modality values."""
        rows: list[list[float]] = []
        for session in session_dicts:
            emotion_vector = session.get("emotion_vector", [0.0] * 64)
            voice = session.get("voice", {})
            camera = session.get("camera", {})
            timestamp = session.get("timestamp")

            hour_sin = math.sin(2 * math.pi * session.get("hour", 12) / 24)
            dow_sin = math.sin(2 * math.pi * session.get("day_of_week", 0) / 7)

            row = [
                emotion_vector[0] if len(emotion_vector) > 0 else np.nan,  # valence
                emotion_vector[1] if len(emotion_vector) > 1 else np.nan,  # arousal
                emotion_vector[2] if len(emotion_vector) > 2 else np.nan,  # dominance
                emotion_vector[3] if len(emotion_vector) > 3 else np.nan,  # joy
                emotion_vector[7] if len(emotion_vector) > 7 else np.nan,  # sadness
                emotion_vector[5] if len(emotion_vector) > 5 else np.nan,  # fear
                emotion_vector[9] if len(emotion_vector) > 9 else np.nan,  # anger
                emotion_vector[6] if len(emotion_vector) > 6 else np.nan,  # surprise
                emotion_vector[8] if len(emotion_vector) > 8 else np.nan,  # disgust
                emotion_vector[4] if len(emotion_vector) > 4 else np.nan,  # trust
                emotion_vector[10] if len(emotion_vector) > 10 else np.nan,  # anticipation
                voice.get("energy", np.nan),
                voice.get("pitch_mean", np.nan),
                camera.get("face_confidence", np.nan),
                camera.get("posture_score", np.nan),
                camera.get("fidget_score", np.nan),
                session.get("distortion_count", 0),
                session.get("masking_probability", 0.0),
                hour_sin,
                dow_sin,
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. MoodClusteringEngine — KMeans (auto-k) + DBSCAN
# ---------------------------------------------------------------------------
_CLUSTER_LABEL_EMOTIONS = [
    "joy", "sadness", "fear", "anger",
    "surprise", "disgust", "trust", "anticipation",
]


class MoodClusteringEngine:
    """Personal mood-profile clustering with auto-k KMeans and DBSCAN."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._kmeans: KMeans | None = None
        self._dbscan: DBSCAN | None = None
        self._cluster_labels: dict[int, str] = {}
        self._fitted = False

    def fit(self, feature_matrix: np.ndarray) -> dict[str, Any]:
        """
        Fit scaler → PCA → KMeans (auto-k) + DBSCAN.
        Returns training metrics dict for MLflow.
        """
        scaled = self._scaler.fit_transform(feature_matrix)

        # PCA — retain 95 % variance
        self._pca = PCA(n_components=0.95)
        reduced = self._pca.fit_transform(scaled)
        explained_variance = float(np.sum(self._pca.explained_variance_ratio_))

        # Auto-k via silhouette
        best_k, best_score = 3, -1.0
        for candidate_k in range(3, min(9, len(feature_matrix))):
            candidate_model = KMeans(n_clusters=candidate_k, n_init=10, random_state=42)
            labels = candidate_model.fit_predict(reduced)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_k, best_score = candidate_k, score

        self._kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        self._kmeans.fit(reduced)

        # DBSCAN — auto-eps via k-distance elbow
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(reduced)
        distances, _ = neighbors.kneighbors(reduced)
        sorted_distances = np.sort(distances[:, -1])
        knee_idx = max(1, len(sorted_distances) // 5)
        auto_eps = float(sorted_distances[knee_idx])

        self._dbscan = DBSCAN(eps=auto_eps, min_samples=3)
        self._dbscan.fit(reduced)

        # Label clusters by dominant emotions
        self._cluster_labels = self._label_clusters(feature_matrix, self._kmeans.labels_)
        self._fitted = True

        metrics = {
            "silhouette_score": best_score,
            "k_chosen": best_k,
            "pca_explained_variance": explained_variance,
            "pca_n_components": int(self._pca.n_components_),
            "dbscan_eps": auto_eps,
            "cluster_sizes": {
                str(cluster_id): int(count)
                for cluster_id, count in zip(*np.unique(self._kmeans.labels_, return_counts=True))
            },
        }
        logger.info(f"MoodClustering fit complete: {metrics}")
        return metrics

    def predict(self, feature_row: np.ndarray) -> ClusterResult:
        """Predict cluster for a single feature vector."""
        if not self._fitted or self._kmeans is None or self._pca is None:
            return ClusterResult(cluster_id=-1, cluster_name="unfitted", confidence=0.0)

        scaled = self._scaler.transform(feature_row.reshape(1, -1))
        reduced = self._pca.transform(scaled)

        cluster_id = int(self._kmeans.predict(reduced)[0])
        distances = self._kmeans.transform(reduced)[0]
        confidence = 1.0 / (1.0 + distances[cluster_id])
        cluster_name = self._cluster_labels.get(cluster_id, f"cluster_{cluster_id}")

        return ClusterResult(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            confidence=float(confidence),
        )

    def _label_clusters(self, feature_matrix: np.ndarray, labels: np.ndarray) -> dict[int, str]:
        """Name each cluster by its top-2 dominant emotion dimensions."""
        result: dict[int, str] = {}
        for cluster_id in set(labels):
            mask = labels == cluster_id
            cluster_data = feature_matrix[mask]
            # Columns 3-10 map to the 8 primary emotions
            emotion_means = cluster_data[:, 3:11].mean(axis=0)
            top_indices = np.argsort(emotion_means)[-2:][::-1]
            top_names = [_CLUSTER_LABEL_EMOTIONS[i] for i in top_indices if i < len(_CLUSTER_LABEL_EMOTIONS)]
            result[cluster_id] = " + ".join(top_names) if top_names else f"cluster_{cluster_id}"
        return result


# ---------------------------------------------------------------------------
# 3. MoodDriftAnalyzer — LSTM + linear regression + EWMA
# ---------------------------------------------------------------------------
class MoodDriftAnalyzer:
    """7-day mood-drift detection using LSTM, polyfit, and EWMA."""

    _SEQUENCE_LENGTH = 7

    def __init__(self) -> None:
        self._lstm_model: Any | None = None

    def _build_lstm(self) -> Any:
        """Build TF/Keras LSTM: 7-step → next-day valence."""
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self._SEQUENCE_LENGTH, 3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train_lstm(self, sessions_df: pd.DataFrame) -> dict[str, float]:
        """
        Train the LSTM on historical data.
        sessions_df columns: valence, arousal, crisis_level (sorted by date).
        Returns metrics dict.
        """
        values = sessions_df[["valence", "arousal", "crisis_level"]].values
        sequences_x: list[np.ndarray] = []
        targets_y: list[float] = []

        for start_idx in range(len(values) - self._SEQUENCE_LENGTH):
            sequences_x.append(values[start_idx : start_idx + self._SEQUENCE_LENGTH])
            targets_y.append(values[start_idx + self._SEQUENCE_LENGTH, 0])  # next valence

        if len(sequences_x) < 5:
            return {"error": "insufficient_data", "rows": len(values)}

        array_x = np.array(sequences_x, dtype=np.float32)
        array_y = np.array(targets_y, dtype=np.float32)

        self._lstm_model = self._build_lstm()
        history = self._lstm_model.fit(
            array_x, array_y,
            epochs=30, batch_size=8, validation_split=0.2, verbose=0,
        )

        return {
            "final_loss": float(history.history["loss"][-1]),
            "final_mae": float(history.history["mae"][-1]),
            "val_loss": float(history.history.get("val_loss", [0])[-1]),
        }

    def detect_drift(self, sessions: list[dict[str, Any]], window: int = 7) -> DriftReport:
        """
        Analyse the last `window` sessions for mood drift.
        Each session dict needs at least: valence, arousal, crisis_level.
        """
        if len(sessions) < 3:
            return DriftReport(
                trend_direction="stable",
                rate_of_change=0.0,
                predicted_next_day=0.0,
                alert_level=0,
                explanation="Not enough sessions for drift analysis.",
            )

        recent = sessions[-window:]
        valence_series = pd.Series([s.get("valence", 0.0) for s in recent])

        # Linear regression slope
        coefficients = np.polyfit(range(len(valence_series)), valence_series.values, deg=1)
        slope = float(coefficients[0])

        # EWMA smoothed trend
        ewma = valence_series.ewm(span=3).mean()
        ewma_latest = float(ewma.iloc[-1])

        # LSTM prediction (if available)
        predicted_next_day = ewma_latest  # fallback
        if self._lstm_model is not None and len(recent) >= self._SEQUENCE_LENGTH:
            lstm_input = np.array([[
                [s.get("valence", 0.0), s.get("arousal", 0.0), s.get("crisis_level", 0)]
                for s in recent[-self._SEQUENCE_LENGTH:]
            ]], dtype=np.float32)
            predicted_next_day = float(self._lstm_model.predict(lstm_input, verbose=0)[0, 0])

        # Determine direction and alert
        if slope < -0.08:
            direction = "declining"
        elif slope > 0.08:
            direction = "improving"
        else:
            direction = "stable"

        alert_level = 0
        if direction == "declining":
            alert_level = 1
            if slope < -0.15:
                alert_level = 2
            if predicted_next_day < -0.5:
                alert_level = 3

        explanation_parts = [
            f"7-day valence slope: {slope:+.3f} ({direction}).",
            f"EWMA latest: {ewma_latest:.2f}.",
            f"Predicted next session valence: {predicted_next_day:.2f}.",
        ]
        if alert_level >= 2:
            explanation_parts.append(
                "Significant downward mood trend detected — consider proactive check-in."
            )

        return DriftReport(
            trend_direction=direction,
            rate_of_change=slope,
            predicted_next_day=predicted_next_day,
            alert_level=alert_level,
            explanation=" ".join(explanation_parts),
        )


# ---------------------------------------------------------------------------
# 4. EmotionTriggerMiner — Apriori
# ---------------------------------------------------------------------------
class EmotionTriggerMiner:
    """Discovers trigger patterns using the Apriori algorithm."""

    def mine_triggers(
        self,
        sessions: list[dict[str, Any]],
        min_support: float = 0.3,
        min_confidence: float = 0.6,
    ) -> list[TriggerRule]:
        """
        Discretise text tokens + emotion labels into transactions,
        then run Apriori to find frequent patterns.
        """
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        if len(sessions) < 5:
            return []

        transactions: list[list[str]] = []
        for session in sessions:
            items: list[str] = []
            # Add text keywords (top tokens)
            for keyword in session.get("keywords", []):
                items.append(f"word:{keyword}")
            # Add dominant emotion
            dominant = session.get("dominant_emotion", "")
            if dominant:
                items.append(f"emotion:{dominant}")
            if items:
                transactions.append(items)

        if not transactions:
            return []

        encoder = TransactionEncoder()
        encoded_array = encoder.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(encoded_array, columns=encoder.columns_)

        try:
            frequent = apriori(transaction_df, min_support=min_support, use_colnames=True)
            if frequent.empty:
                return []

            rules_df = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
        except Exception as apriori_error:
            logger.warning(f"Apriori mining failed: {apriori_error}")
            return []

        trigger_rules: list[TriggerRule] = []
        for _, row in rules_df.iterrows():
            antecedent_words = [
                item.replace("word:", "")
                for item in row["antecedents"]
                if item.startswith("word:")
            ]
            consequent_emotions = [
                item.replace("emotion:", "")
                for item in row["consequents"]
                if item.startswith("emotion:")
            ]
            if antecedent_words and consequent_emotions:
                trigger_rules.append(TriggerRule(
                    antecedent_words=antecedent_words,
                    consequent_emotion=consequent_emotions[0],
                    confidence=float(row["confidence"]),
                    lift=float(row["lift"]),
                    explanation=(
                        f"When topics {antecedent_words} appear, "
                        f"'{consequent_emotions[0]}' follows with "
                        f"{row['confidence']:.0%} confidence."
                    ),
                ))
        return trigger_rules


# ---------------------------------------------------------------------------
# 5. CrisisClassifier — VotingClassifier + SHAP
# ---------------------------------------------------------------------------
class CrisisClassifier:
    """Ensemble crisis detector with SHAP explainability."""

    FEATURE_NAMES: list[str] = [
        "valence", "arousal_drop_delta", "dbscan_is_outlier",
        "crisis_keyword_count", "voice_flat_affect", "masking_prob",
    ]

    def __init__(self) -> None:
        self._model: Any | None = None
        self._explainer: Any | None = None

    def build_and_train(self, feature_matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Train a calibrated VotingClassifier."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)

        voting = VotingClassifier(
            estimators=[("rf", rf_estimator), ("gb", gb_estimator)],
            voting="soft",
        )

        self._model = CalibratedClassifierCV(voting, cv=3)
        self._model.fit(feature_matrix, labels)

        # Cross-val metrics
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        predictions = cross_val_predict(voting, feature_matrix, labels, cv=stratified_kfold)

        metrics = {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
        try:
            probabilities = cross_val_predict(voting, feature_matrix, labels, cv=stratified_kfold, method="predict_proba")
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        except Exception:
            metrics["roc_auc"] = 0.0

        # SHAP explainer
        try:
            import shap
            self._explainer = shap.TreeExplainer(rf_estimator.fit(feature_matrix, labels))
        except Exception as shap_error:
            logger.warning(f"SHAP init failed: {shap_error}")

        logger.info(f"CrisisClassifier trained: {metrics}")
        return metrics

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Return (crisis_label, probability)."""
        if self._model is None:
            return 0, 0.0
        prediction = int(self._model.predict(features.reshape(1, -1))[0])
        probability = float(self._model.predict_proba(features.reshape(1, -1))[0, 1])
        return prediction, probability

    def explain_prediction(self, features: dict[str, float]) -> dict[str, float]:
        """SHAP feature-importance for a single prediction."""
        if self._explainer is None:
            return {}
        feature_array = np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES]).reshape(1, -1)
        shap_values = self._explainer.shap_values(feature_array)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = crisis
        return {
            name: float(importance)
            for name, importance in zip(self.FEATURE_NAMES, shap_values[0])
        }


# ---------------------------------------------------------------------------
# 6. MoodVisualizer — Plotly dark-themed
# ---------------------------------------------------------------------------
class MoodVisualizer:
    """Dark-themed Plotly chart generators for the MIRA dashboard."""

    _BG_COLOR = "#0C0A1A"
    _ACCENT = "#7B6EF6"
    _TEXT_COLOR = "#EDE9FF"
    _GRID_COLOR = "rgba(123,110,246,0.15)"

    def _dark_layout(self, title: str) -> dict[str, Any]:
        """Shared dark-theme layout config."""
        return dict(
            template="plotly_dark",
            paper_bgcolor=self._BG_COLOR,
            plot_bgcolor=self._BG_COLOR,
            font=dict(color=self._TEXT_COLOR, family="Lato"),
            title=dict(text=title, font=dict(family="DM Serif Display", size=22)),
            xaxis=dict(gridcolor=self._GRID_COLOR),
            yaxis=dict(gridcolor=self._GRID_COLOR),
        )

    def tsne_2d(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        """2D t-SNE scatter colored by cluster."""
        import plotly.graph_objects as go
        from sklearn.manifold import TSNE

        vectors = np.array([s.get("emotion_vector", [0]*64)[:20] for s in sessions])
        if len(vectors) < 5:
            return {}

        coords = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), random_state=42).fit_transform(vectors)
        clusters = [s.get("cluster_id", 0) for s in sessions]
        dates = [s.get("date", "") for s in sessions]
        emotions = [s.get("dominant_emotion", "?") for s in sessions]

        fig = go.Figure(data=go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=8, color=clusters, colorscale="Viridis", showscale=True),
            text=[f"{date}<br>{emo}" for date, emo in zip(dates, emotions)],
            hoverinfo="text",
        ))
        fig.update_layout(**self._dark_layout("Emotion Map (t-SNE)"))
        return fig.to_dict()

    def umap_3d(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        """3D UMAP scatter colored by cluster, sized by arousal."""
        import plotly.graph_objects as go
        import umap

        vectors = np.array([s.get("emotion_vector", [0]*64)[:20] for s in sessions])
        if len(vectors) < 10:
            return {}

        reducer = umap.UMAP(n_components=3, random_state=42)
        coords = reducer.fit_transform(vectors)
        clusters = [s.get("cluster_id", 0) for s in sessions]
        arousal_sizes = [max(5, s.get("arousal", 0.5) * 15) for s in sessions]

        fig = go.Figure(data=go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=arousal_sizes, color=clusters, colorscale="Plasma", opacity=0.8),
        ))
        fig.update_layout(**self._dark_layout("3D Emotion Universe (UMAP)"))
        return fig.to_dict()

    def mood_calendar_heatmap(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        """GitHub-style 30-day mood heatmap."""
        import plotly.graph_objects as go

        dates = [s.get("date", "") for s in sessions]
        valences = [s.get("valence", 0.0) for s in sessions]
        emotions = [s.get("dominant_emotion", "?") for s in sessions]

        fig = go.Figure(data=go.Heatmap(
            x=dates, y=["Mood"] * len(dates), z=[valences],
            colorscale=[
                [0.0, "#D50000"], [0.25, "#FF6B6B"],
                [0.5, "#9B95C9"], [0.75, "#7B6EF6"], [1.0, "#6BCB77"],
            ],
            text=[emotions],
            hoverinfo="text+z",
            showscale=True,
        ))
        fig.update_layout(**self._dark_layout("30-Day Mood Calendar"))
        return fig.to_dict()

    def emotion_radar(self, session: dict[str, Any]) -> dict[str, Any]:
        """Filled radar chart of 8 primary emotions."""
        import plotly.graph_objects as go

        categories = [
            "Joy", "Trust", "Fear", "Surprise",
            "Sadness", "Disgust", "Anger", "Anticipation",
        ]
        vector = session.get("emotion_vector", [0]*64)
        scores = [vector[i] if i < len(vector) else 0 for i in range(3, 11)]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(123,110,246,0.25)",
            line=dict(color=self._ACCENT, width=2),
        ))
        fig.update_layout(
            **self._dark_layout("Emotion Radar"),
            polar=dict(bgcolor=self._BG_COLOR, radialaxis=dict(visible=True, range=[0, 1])),
        )
        return fig.to_dict()

    def drift_timeline(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        """Valence line chart with EWMA overlay and drift markers."""
        import plotly.graph_objects as go

        dates = [s.get("date", str(i)) for i, s in enumerate(sessions)]
        valence_series = pd.Series([s.get("valence", 0.0) for s in sessions])
        ewma = valence_series.ewm(span=3).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=valence_series.tolist(),
            mode="lines+markers", name="Valence",
            line=dict(color=self._ACCENT, width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=ewma.tolist(),
            mode="lines", name="EWMA Trend",
            line=dict(color="#F5A623", width=2, dash="dash"),
        ))
        fig.update_layout(**self._dark_layout("Mood Drift Timeline"))
        return fig.to_dict()
