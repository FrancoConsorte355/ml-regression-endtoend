import pandas as pd
import pytest
from pathlib import Path

from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.preprocess import (
    clean_and_merge, drop_duplicates, remove_outliers, preprocess_split
)
from src.feature_pipeline.feature_engineering import (
    add_date_features, frequency_encode, target_encode, drop_unused_columns, run_feature_engineering
)

# =========================
# load.py – unit test
# =========================
# Confirms time-based splitting works.
def test_load_and_split_data_creates_splits(tmp_path):
    dummy_path = tmp_path / "raw.csv"
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 3000, 4000],
        "city_full": ["A", "B", "A", "B", "C", "D"],
    })
    df.to_csv(dummy_path, index=False)

    train, eval, holdout = load_and_split_data(raw_path=str(dummy_path), output_dir=tmp_path)

    assert not train.empty and not eval.empty and not holdout.empty
    assert train["date"].max() < pd.to_datetime("2020-01-01")
    assert eval["date"].min() >= pd.to_datetime("2020-01-01")
    assert holdout["date"].min() >= pd.to_datetime("2022-01-01")
    assert (tmp_path / "train.csv").exists()
    print("✅ Data splitting test passed")


# =========================
# preprocess.py – unit tests
# =========================
# Confirms preprocessing functions behave as intended.
def test_remove_outliers_drops_high_prices():
    df = pd.DataFrame({"median_list_price": [100_000, 500_000, 20_000_000]})
    cleaned = remove_outliers(df)
    assert cleaned["median_list_price"].max() <= 19_000_000
    print("✅ Outlier removal test passed")


def test_drop_duplicates_removes_dupes():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "year": [2020, 2020],
        "median_list_price": [100, 100]
    })
    cleaned = drop_duplicates(df)
    assert cleaned.shape[0] == 0
    print("✅ Duplicate removal test passed")


def test_clean_and_merge_skips_when_city_missing():
    df = pd.DataFrame({"date": ["2020-01-01"], "price": [100]})
    result = clean_and_merge(df, metros_path=None)  # should skip gracefully
    assert "date" in result.columns and "price" in result.columns
    print("✅ Clean-and-merge (no city_full) passed")


# =========================
# feature_engineering – unit tests
# =========================
# Confirms feature functions create consistent numeric features and avoid leakage.
def test_add_date_features_extracts_parts():
    df = pd.DataFrame({"date": ["2020-01-15"]})
    df = add_date_features(df)
    assert df.loc[0, "year"] == 2020 and df.loc[0, "month"] == 1 and df.loc[0, "quarter"] == 1
    print("✅ Date feature extraction test passed")


def test_frequency_encode_counts_values():
    train = pd.DataFrame({"zipcode": [1000, 1000, 2000]})
    eval = pd.DataFrame({"zipcode": [1000, 3000]})
    train, eval, freq_map = frequency_encode(train, eval, "zipcode")
    assert train["zipcode_freq"].tolist() == [2, 2, 1]
    assert eval["zipcode_freq"].tolist() == [2, 0]
    assert isinstance(freq_map, pd.Series)
    print("✅ Frequency encoding test passed")


def test_target_encode_applies_mapping():
    train = pd.DataFrame({"city_full": ["A", "B", "A"], "price": [100, 200, 300]})
    eval = pd.DataFrame({"city_full": ["A", "B"]})
    train, eval, te = target_encode(train, eval, "city_full", "price")
    assert "city_full_encoded" in train.columns
    assert eval["city_full_encoded"].notnull().all()
    assert te is not None
    print("✅ Target encoding test passed")


def test_drop_unused_columns_removes_leakage():
    # 1. Fabricamos el "veneno": un DataFrame sintético que incluye 
    # tanto las columnas inútiles como TODAS las variables de Leakage.
    tr = pd.DataFrame({
        "date": ["2020-01-01"], 
        "city_full": ["A"], 
        "zipcode": [1000],
        "median_sale_price": [200], 
        "median_list_price": [250],        # 🚨 Leakage 1
        "Median Home Value": [240],        # 🚨 Leakage 2
        "median_ppsf": [150],              # 🚨 Leakage 3
        "median_list_ppsf": [155],         # 🚨 Leakage 4
        "price": [300]                     # 🎯 Target (El único que debe sobrevivir)
    })
    
    # Copiamos para simular el set de evaluación
    ev = tr.copy()
    
    # 2. Ejecutamos la función que estamos testeando
    tr2, ev2 = drop_unused_columns(tr, ev)
    
    # 3. Lista exhaustiva de todo el material tóxico/inútil
    leaky_and_unused_cols = [
        "date", "city_full", "zipcode", "median_sale_price",
        "median_list_price", "Median Home Value", "median_ppsf", "median_list_ppsf"
    ]
    
    # 4. Las Afirmaciones (Asserts): Comprobamos que el camión de basura se llevó todo
    for col in leaky_and_unused_cols:
        assert col not in tr2.columns, f"Fallo en Train: La columna tóxica '{col}' sobrevivió."
        assert col not in ev2.columns, f"Fallo en Eval: La columna tóxica '{col}' sobrevivió."
        
    # 5. Comprobamos que no se haya borrado la variable objetivo por error
    assert "price" in tr2.columns and "price" in ev2.columns
    
    print("✅ Drop unused columns test passed (Data Leakage totalmente purgado)")


# =========================
# integration test
# =========================
# Confirms the whole feature pipeline works together.
def test_full_pipeline_integration(tmp_path):
    raw = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 1000, 2000],
        "city_full": ["A", "B", "A", "B", "A", "B"],
        "median_list_price": [150_000]*6
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    train, eval, holdout = load_and_split_data(raw_path=str(raw_path), output_dir=tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(exist_ok=True)
    train.to_csv(tmp_path / "train.csv", index=False)
    eval.to_csv(tmp_path / "eval.csv", index=False)

    preprocess_split("train", raw_dir=tmp_path, processed_dir=processed_dir, metros_path=None)
    preprocess_split("eval", raw_dir=tmp_path, processed_dir=processed_dir, metros_path=None)

    out_train, out_eval, out_holdout, freq_map, te = run_feature_engineering(
        in_train_path=processed_dir / "cleaning_train.csv",
        in_eval_path=processed_dir / "cleaning_eval.csv",
        output_dir=processed_dir,
    )


    assert {"year", "zipcode_freq", "city_full_encoded"}.issubset(out_train.columns)
    assert {"year", "zipcode_freq", "city_full_encoded"}.issubset(out_eval.columns)
    assert freq_map is not None
    assert te is not None
    print("✅ Full pipeline integration test passed")
