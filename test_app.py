"""
Tests for app.py – verifies compatibility fixes made for Python 3.12,
gensim 4.x, and pandas 2.x.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Import-level compatibility: all fixed imports must be importable
# ---------------------------------------------------------------------------

def test_gensim_annoy_indexer_import():
    """AnnoyIndexer must come from gensim.similarities.annoy (gensim 4.x)."""
    from gensim.similarities.annoy import AnnoyIndexer  # noqa: F401


def test_gensim_keyed_vectors_import():
    """KeyedVectors must be importable from both paths used in app.py."""
    from gensim.models import KeyedVectors  # noqa: F401
    import gensim.models.keyedvectors as word2vec  # noqa: F401


def test_py_thesaurus_import():
    """Thesaurus class must be importable from py_thesaurus."""
    from py_thesaurus import Thesaurus  # noqa: F401


def test_app_module_imports():
    """The whole app module must import without errors."""
    import app  # noqa: F401


# ---------------------------------------------------------------------------
# 2. percentile_rank – exercises the pandas .items() fix (was iteritems())
# ---------------------------------------------------------------------------

from app import percentile_rank


def test_percentile_rank_positive():
    values = pd.Series([0.9, 0.6, 0.3], index=[0, 1, 2])
    result = percentile_rank(values, negative=False)
    assert len(result) == 3
    # All values should be in (0, 1]
    assert all(0 < v <= 1 for v in result)


def test_percentile_rank_negative():
    values = pd.Series([0.9, 0.6, 0.3], index=[0, 1, 2])
    result = percentile_rank(values, negative=True)
    # All values should be in [-1, 0)
    assert all(-1 <= v < 0 for v in result)


def test_percentile_rank_returns_series():
    values = pd.Series([1.0, 0.5], index=[10, 20])
    result = percentile_rank(values)
    assert isinstance(result, pd.Series)
    assert list(result.index) == [10, 20]


# ---------------------------------------------------------------------------
# 3. gensim 4.x API: key_to_index replaces vocab
# ---------------------------------------------------------------------------

def test_keyed_vectors_key_to_index():
    """gensim 4.x KeyedVectors uses key_to_index, not vocab."""
    from gensim.models import KeyedVectors

    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["hello", "world"], [np.array([1., 2., 3.]), np.array([4., 5., 6.])])

    assert hasattr(kv, "key_to_index"), "key_to_index attribute missing from KeyedVectors"
    assert "hello" in kv.key_to_index
    assert "world" in kv.key_to_index
    assert list(kv.key_to_index.keys()) == ["hello", "world"]


def test_keyed_vectors_membership():
    """'word in model' membership check works with gensim 4.x KeyedVectors."""
    from gensim.models import KeyedVectors

    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["apple"], [np.array([0.1, 0.2, 0.3])])

    assert "apple" in kv
    assert "banana" not in kv


# ---------------------------------------------------------------------------
# 4. Flask routes – using test client with mocked global state
# ---------------------------------------------------------------------------

def _make_mock_model(words=None):
    """Return a gensim KeyedVectors stub with a small vocabulary."""
    from gensim.models import KeyedVectors

    if words is None:
        words = ["king", "queen", "man", "woman"]
    dim = 10
    vecs = [np.random.rand(dim).astype(np.float32) for _ in words]
    kv = KeyedVectors(vector_size=dim)
    kv.add_vectors(words, vecs)
    return kv


def _make_mock_df(words=None):
    if words is None:
        words = ["king", "queen", "man", "woman"]
    return pd.DataFrame({
        "word": words,
        "gender": [0.8, -0.7, 0.5, -0.6],
        "race": [0.1, 0.2, -0.3, 0.4],
    })


@pytest.fixture()
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c, app_module


def test_fetch_data_all(client):
    """POST /fetch_data with hist_type=ALL returns filtered JSON."""
    c, _ = client
    payload = {
        "slider_sel": [[-1.0, 1.0]],
        "hist_type": "ALL",
        "data": [
            {"word": "king", "gender": 0.8, "race": 0.1},
            {"word": "queen", "gender": -0.7, "race": 0.2},
        ],
    }
    resp = c.post("/fetch_data", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    result = json.loads(resp.get_json())
    assert isinstance(result, list)


def test_fetch_data_specific_bias(client):
    """POST /fetch_data with a specific hist_type column filters correctly."""
    c, _ = client
    payload = {
        "slider_sel": [[0.5, 1.0]],
        "hist_type": "gender",
        "data": [
            {"word": "king", "gender": 0.8, "race": 0.1},
            {"word": "queen", "gender": -0.7, "race": 0.2},
            {"word": "man", "gender": 0.6, "race": -0.3},
        ],
    }
    resp = c.post("/fetch_data", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    result = json.loads(resp.get_json())
    # Only words with gender in [0.5, 1.0] should be returned (king, man)
    returned_words = [r["word"] for r in result]
    assert "king" in returned_words
    assert "man" in returned_words
    assert "queen" not in returned_words


def test_get_all_words(client):
    """GET /get_all_words returns a JSON list of vocabulary words."""
    c, app_module = client
    mock_model = _make_mock_model()
    with patch.object(app_module, "model", mock_model):
        resp = c.get("/get_all_words")
    assert resp.status_code == 200
    words = resp.get_json()
    assert isinstance(words, list)
    assert "king" in words


def test_get_histogram(client):
    """GET /get_histogram/<type> returns values, min and max."""
    c, app_module = client
    mock_df = _make_mock_df()
    with patch.object(app_module, "df", mock_df):
        resp = c.get("/get_histogram/gender")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "values" in data
    assert "min" in data
    assert "max" in data
    assert data["min"] <= data["max"]


def test_get_histogram_all(client):
    """GET /get_histogram/ALL uses the mean of all bias columns."""
    c, app_module = client
    mock_df = _make_mock_df()
    with patch.object(app_module, "df", mock_df):
        resp = c.get("/get_histogram/ALL")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["values"]) == len(mock_df)


def test_get_filenames(client):
    """GET /getFileNames/ returns [group_files, target_files] for English."""
    c, app_module = client
    with patch.object(app_module, "language", "en"):
        resp = c.get("/getFileNames/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 2  # [group_files, target_files]
    assert isinstance(data[0], list)
    assert isinstance(data[1], list)
