"""Regression tests for inclusive user-facing market-data date boundaries.

The Yahoo/yfinance API treats its end argument as exclusive. These tests use
synthetic prices and never make network requests.
"""
import pandas as pd

from src.benchmark import downloadBenchmarkPrices
from src.data import getData


def test_get_data_includes_selected_final_date(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {"Close": [100.0, 102.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

    monkeypatch.setattr("src.data.yf.download", fake_download)
    getData.clear()
    result = getData(["AAPL"], "2026-01-02", "2026-01-05")

    assert calls[0]["start"] == "2026-01-02"
    assert calls[0]["end"] == "2026-01-06"
    assert calls[0]["auto_adjust"] is True
    assert list(result.columns) == ["AAPL"]
    assert result.index[-1] == pd.Timestamp("2026-01-05")


def test_benchmark_uses_same_inclusive_price_boundary(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {"Close": [500.0, 503.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

    monkeypatch.setattr("src.data.yf.download", fake_download)
    getData.clear()
    benchmark = downloadBenchmarkPrices("spy", "2026-01-02", "2026-01-05")

    assert calls[0]["end"] == "2026-01-06"
    assert benchmark.name == "SPY"
    assert benchmark.iloc[-1] == 503.0
