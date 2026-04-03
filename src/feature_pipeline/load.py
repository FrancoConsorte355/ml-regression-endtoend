"""
Load & time-split the raw dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str = "data/raw/untouched_raw_original.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    # Define la función con dos parámetros opcionales:
    # - raw_path: ruta del archivo CSV original
    # - output_dir: directorio donde guardar los splits
    
    df = pd.read_csv(raw_path)
    # Carga el CSV en un DataFrame de pandas
    
    df["date"] = pd.to_datetime(df["date"])
    # Convierte la columna "date" a formato datetime (para comparaciones temporales)
    
    df = df.sort_values("date")
    # Ordena el DataFrame por fecha ascendente
    
    cutoff_date_eval = pd.Timestamp("2020-01-01")     # eval starts
    # Define fecha de corte: evaluación comienza el 1 de enero 2020
    
    cutoff_date_holdout = pd.Timestamp("2022-01-01")  # holdout starts
    # Define fecha de corte: datos de prueba comienzan el 1 de enero 2022
    
    train_df = df[df["date"] < cutoff_date_eval]
    # Filtra datos anteriores a 2020 para **entrenamiento**
    
    eval_df = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]
    # Filtra datos entre 2020-2022 (inclusive inicio, exclusivo fin) para **evaluación**
    
    holdout_df = df[df["date"] >= cutoff_date_holdout]
    # Filtra datos desde 2022 en adelante para **prueba final (holdout)**
    
    outdir = Path(output_dir)
    # Convierte la ruta a un objeto Path (manejo seguro de rutas)
    
    outdir.mkdir(parents=True, exist_ok=True)
    # Crea el directorio si no existe (con directorios padres si es necesario)
    
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)
    # Guarda los tres splits como archivos CSV separados
    
    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")
    # Muestra mensaje de confirmación con las dimensiones de cada split
    
    return train_df, eval_df, holdout_df
    # Retorna los tres DataFrames para uso posterior


if __name__ == "__main__":
    load_and_split_data()
