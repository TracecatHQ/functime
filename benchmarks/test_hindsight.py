import polars as pl
import polars.selectors as ps
import pytest
import logging
import os
import numpy as np

from typing import List, Tuple
from functime.preprocessing import reindex
from functime_backend.classification import HindsightClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

# Test split size
TEST_FRACTION = 0.20

# Path to store temporal embedding chunks
STORAGE_PATH = os.environ.get("EMBEDDINGS_STORAGE_PATH", ".data/embs")

# Classification metrics
CLASSIFICATION_METRICS = [
    accuracy_score,
    balanced_accuracy_score
]


def preview_dataset(
    name: str,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame
):
    """Log memory usage and first 10 rows given train-split splits."""
    logging.info("🔍 Preview %s dataset", name)
    # Log memory
    logging.info("🔍 y_train mem: %s", f'{y_train.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 X_train mem: %s", f'{X_train.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 y_test mem: %s", f'{y_test.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 X_test mem: %s", f'{X_test.estimated_size("mb"):,.4f} mb')
    # Preview dataset
    logging.debug("🔍 X_train preview:\n%s", X_train)
    logging.debug("🔍 y_train preview:\n%s", y_train)
    logging.debug("🔍 X_test preview:\n%s", X_test)
    logging.debug("🔍 y_test preview:\n%s", y_test)


def split_iid_data(
    data: pl.DataFrame,
    label_cols: List[str],
    test_size: float = TEST_FRACTION,
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    

    # Sample session ids
    entity_col, time_col = data.columns[:2]
    entities = set(data.get_column(entity_col).unique())
    n_samples = int(len(entities) * test_size)
    np.random.seed(seed)
    entity_ids = np.random.choice(sorted(entities), size=n_samples)

    # Train test split
    data = data.lazy()
    # Cannot combine 'streaming' with 'common_subplan_elimination'. CSE will be turned off.
    # NOTE: Filter does not maintain order!
    X_y_train, X_y_test = pl.collect_all([
        data.filter(~pl.col(entity_col).is_in(entity_ids)).sort([entity_col, time_col]),
        data.filter(pl.col(entity_col).is_in(entity_ids)).sort([entity_col, time_col])
    ])

    # Split into X, y
    exog = [entity_col, time_col, pl.all().exclude([entity_col, time_col, *label_cols])]
    endog = [entity_col, time_col, pl.col(label_cols)]

    # Splits
    X_train, y_train = X_y_train.select(exog), X_y_train.select(endog)
    X_test, y_test = X_y_test.select(exog), X_y_test.select(endog)

    return X_train, X_test, y_train, y_test


@pytest.fixture
def classifier():
    return LogisticRegression(max_iter=200)


@pytest.fixture(params=[64, 256, 512])
def parkinsons_dataset(request):
    """Parkinson's EEG brain scans.

    31 subjects (15 in control, 16 in test), 18535 timestamps.
    Predict { "off", "hc" } status in the "session" column, where "off" is the control group.

    Relevant to healthcare and disease detection.
    """
    label_col = "is_control"
    max_timestamp = request.param
    data = (
        pl.scan_parquet("data/parkinsons_eeg.parquet")
        # Ignore collinear features
        .select(pl.all().exclude(["disease_duration", "MMSE", "NAART"]))
        .filter(pl.col("timestamp") < max_timestamp)
        .select(["subject", "timestamp", ps.numeric().exclude("timestamp"), (pl.col("session") == "off").alias("is_control").cast(pl.Int8)])
        .collect(streaming=True)
    )
    X_train, X_test, y_train, y_test = split_iid_data(data, label_cols=[label_col])
    preview_dataset("behacom", X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def parkinsons_baseline(parkinsons_dataset, classifier):
    X_train, X_test, y_train, y_test = parkinsons_dataset
    idx_cols = X_train.columns[:2]
    model = classifier
    model.fit(
        X=X_train.select(pl.all().exclude(idx_cols)).to_numpy(),
        y=y_train.select(pl.all().exclude(idx_cols)).to_numpy(),
    )
    y_pred = model.predict(X=X_test.select(pl.all().exclude(idx_cols)).to_numpy())
    y_test = y_test.select(pl.all().exclude(idx_cols)).to_numpy()
    scores = {}
    for metric in CLASSIFICATION_METRICS:
        metric_name = metric.__name__
        score = metric(y_true=y_test, y_pred=y_pred)
        scores[metric_name] = score
    return scores


@pytest.fixture
def behacom_dataset():
    """Hourly user laptop behavior every week.

    Time-series regression. Predict current app CPU usage at each timestamp.
    Compare temporal embeddings accuracy versus 

    11 users, ~12,000 dimensions (e.g. RAM usage, CPU usage, mouse location), ~5000 timestamps.
    Drop users with less than one week of observations: i.e. users [2, 5].

    Relevant to IoT, productivity, and infosec.
    """
    label_col = "current_app_average_cpu"
    cache_path = ".data/behacom.arrow"
    if os.path.exists(cache_path):
        data = pl.read_ipc(cache_path)
    else:
        data = (
            pl.scan_parquet("data/behacom.parquet")
            .filter(~pl.col("user").is_in([2, 5]))
            .groupby_dynamic("timestamp", every="1h", by="user")
            .agg(ps.numeric().fill_null(0).max())  # Get maximum usage during that hour
            # Drop cpu columns other than label to prevent data leakage
            .select(~ps.contains(["system_cpu", "current_app_std"]))
            .lazy()
            # Create sessions
            .with_columns(year_week=pl.col("timestamp").dt.strftime("%Y%w").cast(pl.Categorical).to_physical().cast(pl.Int32))
            .with_columns(session_id=pl.col("year_week") - pl.col("year_week").min())
            .drop("year_week")
            .select(["user", "session_id", ps.numeric().exclude(["user", "session_id"])])
            .collect(streaming=True)
            .pipe(reindex)
            .fill_null(0)
        )
        data.write_ipc(cache_path)
    X_train, X_test, y_train, y_test = split_iid_data(data, label_cols=[label_col])
    preview_dataset("behacom", X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


@pytest.fixture(params=[0.05], ids=lambda x: f"fraction:{x}")
def elearn_dataset(request):
    """Massive e-learning exam score prediction from Kaggle.

    Multi-output classification problem. Predict 18 questions win / loss across sessions.
    Dataset contains 23,562 sessions intotal. We take a random sample of 0.2, 0.4, 0.8, and 1.0 sessions.
    This example is relevant to education, IoT, and online machine learning.

    Note: we currently discard all string features (e.g. `text`).
    Hindsight will eventually support nested embeddings, in which case the `text` column can be
    transformed into BERT / CLIP / OpenAI embeddings.

    Link: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data
    """
    # Game Walkthrough:
    # https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384796
    entity_col = "session_id"
    time_col = "index"
    label_cols = [f"q{i+1}" for i in range(18)]
    sample_fraction = request.param
    cache_path = ".data/elearn.arrow"

    if os.path.exists(cache_path):
        data = pl.read_ipc(cache_path)
    else:
        labels = (
            pl.read_parquet("data/elearn_labels.parquet")
            .pivot(index="session_id", columns="question_id", values="correct")
            .select(["session_id", pl.all().exclude("session_id").prefix("q")])
        )
        sampled_session_ids = labels.get_column("session_id").unique().sample(fraction=sample_fraction)
        logging.info("🎲 Selected %s / %s sessions (%.2f)", len(sampled_session_ids), len(labels), sample_fraction)
        data = (
            pl.scan_parquet("data/elearn.parquet")
            # Sample session IDs
            .filter(pl.col(entity_col).is_in(sampled_session_ids))
            # NOTE: We groupby max numeric columns to remove duplicates
            .groupby([entity_col, time_col])
            .agg(ps.numeric().max())
            # Join with multioutput labels
            .join(labels.lazy(), how="left", on="session_id")
            # Forward fill
            .select([
                pl.col(entity_col),
                pl.col(time_col),
                pl.all().exclude([entity_col, time_col]).forward_fill().fill_null(0)
            ])
            .collect(streaming=True)
            .pipe(reindex)
            .fill_null(0)
        )
        data.write_ipc(cache_path)

    # Check uniqueness
    if data.n_unique([entity_col, time_col]) < data.height:
        raise ValueError("data contains duplicate (entity, time) rows")

    X_train, X_test, y_train, y_test = split_iid_data(data, label_cols=label_cols)
    preview_dataset("elearn", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def test_regression(behacom_dataset):
    X_train, X_test, y_train, y_test = behacom_dataset


def test_binary_classification(parkinsons_dataset, parkinsons_baseline, classifier):
    X_train, X_test, y_train, y_test = parkinsons_dataset
    baseline = parkinsons_baseline
    logging.info("💯 Baseline Score: %s", baseline)
    model = HindsightClassifier(
        estimator=classifier,
        storage_path=STORAGE_PATH,
        random_state=42,
        max_iter=200,
    )
    model.fit(X=X_train, y=y_train)
    scores = model.score(X=X_test, y=y_test, metrics=CLASSIFICATION_METRICS)["label"]
    logging.info("💯 Hindsight Score: %s", scores)
    for metric_name in scores.keys():
        assert scores[metric_name] > parkinsons_baseline[metric_name]


def test_multioutput_classification(elearn_dataset, classifier):
    X_train, X_test, y_train, y_test = elearn_dataset
    model = HindsightClassifier(
        estimator=classifier,
        storage_path=STORAGE_PATH,
        random_state=42,
        max_iter=200,
    )
    model.fit(X=X_train, y=y_train)
    score = model.score(X=X_test, y=y_test, keep_pred=True, metrics=CLASSIFICATION_METRICS)
    assert score > 0.9


def test_video_classification():
    pass
