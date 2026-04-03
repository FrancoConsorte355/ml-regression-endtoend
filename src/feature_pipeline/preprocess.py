"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell, GA',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos, TX',
    'Baltimore-Columbia-Towson': 'Baltimore-Columbia-Towson, MD',
    'Boston-Cambridge-Newton': 'Boston-Cambridge-Newton, MA-NH',
    'Charlotte-Concord-Gastonia': 'Charlotte-Concord-Gastonia, NC-SC',
    'Chicago-Naperville-Elgin': 'Chicago-Naperville-Elgin, IL-IN',
    'Cincinnati': 'Cincinnati, OH-KY-IN',
    'DC_Metro': 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
    'Dallas-Fort Worth-Arlington': 'Dallas-Fort Worth-Arlington, TX',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial, CO',
    'Detroit-Warren-Dearborn': 'Detroit-Warren-Dearborn, MI',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands, TX',
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas, NV',
    'Los Angeles-Long Beach-Anaheim': 'Los Angeles-Long Beach-Anaheim, CA',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach, FL',
    'Minneapolis-St. Paul-Bloomington': 'Minneapolis-St. Paul-Bloomington, MN-WI',
    'New York-Newark-Jersey City': 'New York-Newark-Jersey City, NY-NJ',
    'Orlando-Kissimmee-Sanford': 'Orlando-Kissimmee-Sanford, FL',
    'Philadelphia-Camden-Wilmington': 'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD',
    'Phoenix-Mesa-Chandler': 'Phoenix-Mesa-Chandler, AZ',
    'Pittsburgh': 'Pittsburgh, PA',
    'Portland-Vancouver-Hillsboro': 'Portland-Vancouver-Hillsboro, OR-WA',
    'Riverside-San Bernardino-Ontario': 'Riverside-San Bernardino-Ontario, CA',
    'Sacramento-Roseville-Folsom': 'Sacramento-Roseville-Folsom, CA',
    'San Antonio-New Braunfels': 'San Antonio-New Braunfels, TX',
    'San Diego-Chula Vista-Carlsbad': 'San Diego-Chula Vista-Carlsbad, CA',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont, CA',
    'Seattle-Tacoma-Bellevue': 'Seattle-Tacoma-Bellevue, WA',
    'St. Louis': 'St. Louis, MO-IL',
    'Tampa-St. Petersburg-Clearwater': 'Tampa-St. Petersburg-Clearwater, FL',
}


def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    # Maneja valores nulos (NaN) devolviéndolos sin modificar
    if pd.isna(s):
        return s
    
    # Convierte a string, elimina espacios al inicio/final y transforma a minúsculas
    s = str(s).strip().lower()
    
    # Unifica diferentes tipos de guiones (–, —, -) en un solo guión estándar (-)
    # Importante para normalizar ciudades que pueden tener distintos guiones en sus nombres
    s = re.sub(r"[–—-]", "-", s)
    
    # Colapsa espacios múltiples consecutivos en un solo espacio
    # Elimina espacios extra que podrían haber entre palabras
    s = re.sub(r"\s+", " ", s)
    
    return s


def clean_and_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """
    Función principal: Normaliza nombres de ciudades y opcionalmente fusiona 
    coordenadas lat/lng desde un dataset de metros.
    
    Flujo:
    1. Verifica que existe la columna 'city_full'
    2. Normaliza los nombres de ciudades (minúsculas, guiones, espacios)
    3. Aplica mapeo manual de ciudades conocidas
    4. Si lat/lng ya existen, retorna sin cambios
    5. Si archivo metros no existe, retorna sin cambios
    6. Fusiona (merge) coordenadas desde el archivo metros
    7. Reporta ciudades sin coordenadas encontradas
    """

    # Verifica si la columna 'city_full' existe en el DataFrame
    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df

    # Normaliza nombres de ciudades usando la función normalize_city
    # .apply() = aplica una función a cada elemento de la Serie (columna)
    # En este caso, cada valor en df["city_full"] pasa por normalize_city()
    df["city_full"] = df["city_full"].apply(normalize_city)
    
    # Crea un diccionario con ciudades normalizadas como claves y ciudades corregidas como valores
    # Ejemplo: {'atlanta-sandy springs-alpharetta': 'Atlanta-Sandy Springs-Roswell, GA'}
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
    
    # .replace() = reemplaza valores en la columna según el diccionario norm_mapping
    # Corrige nombres de ciudades conocidos que tienen inconsistencias
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # 🚨 Verifica si las columnas 'lat' y 'lng' ya existen en el DataFrame
    # .issubset() = método de conjunto (set) que retorna True si todos los elementos
    #              del primer conjunto están contenidos en el segundo
    # Equivalente a: if "lat" in df.columns and "lng" in df.columns
    if {"lat", "lng"}.issubset(df.columns):
        print("⚠️ Skipping lat/lng merge: already present in DataFrame.")
        return df

    # Verifica si el archivo metros existe o si metros_path es None
    # Si no existe o no se proporcionó, retorna el DataFrame sin cambios
    if not metros_path or not Path(metros_path).exists():
        print("⚠️ Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Lee el archivo CSV de metros en un nuevo DataFrame
    metros = pd.read_csv(metros_path)
    
    # Verifica que el archivo metros tiene las columnas requeridas
    # .issubset() aquí verifica que {'lat', 'lng'} está contenido en metros.columns
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("⚠️ Skipping lat/lng merge: metros file missing required columns.")
        return df

    # Normaliza los nombres de metros del dataset metros (misma lógica que con city_full)
    # .apply() = aplica normalize_city a cada valor de metros["metro_full"]
    metros["metro_full"] = metros["metro_full"].apply(normalize_city)
    
    # .merge() = fusiona dos DataFrames basándose en columnas coincidentes
    # how="left" = mantiene todas las filas del DataFrame izquierdo (df) aunque no tengan coincidencia
    # left_on="city_full" = columna en df que se usa para el match
    # right_on="metro_full" = columna en metros que se usa para el match
    # Resultado: agrega columnas 'lat' y 'lng' a df donde haya coincidencias
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how="left", left_on="city_full", right_on="metro_full")
    
    # Elimina la columna temporal 'metro_full' (ya no la necesitamos)
    # errors="ignore" = no lanza error si la columna no existe
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    # Identifica ciudades que no encontraron coincidencia (lat/lng = NaN)
    # df[df["lat"].isnull()] = filtra filas donde lat es nulo
    # ["city_full"].unique() = obtiene lista única de ciudades sin coordenadas
    missing = df[df["lat"].isnull()]["city_full"].unique()
    
    # Si hay ciudades sin coordenadas, muestra un warning
    if len(missing) > 0:
        print("⚠️ Still missing lat/lng for:", missing)
    else:
        # Si todas las ciudades encontraron coordenadas, confirma éxito
        print("✅ All cities matched with metros dataset.")
    
    # Retorna el DataFrame con ciudades normalizadas y coordenadas agregadas
    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas duplicadas basándose en todas las columnas excepto 'date' y 'year'.
    
    Permite mantener la misma propiedad en diferentes fechas/años, pero elimina
    registros exactamente idénticos en todas las demás columnas.
    
    Ejemplo:
    - Si la misma propiedad aparece en enero y febrero, se MANTIENEN ambas (diferentes dates)
    - Si la misma propiedad aparece 2 veces en enero, se ELIMINA una (duplicados exactos)
    """
    
    # Obtiene el número de filas ANTES de eliminar duplicados
    # .shape[0] retorna el número de filas del DataFrame
    before = df.shape[0]
    
    # Elimina duplicados:
    # .columns.difference(["date", "year"]) = todas las columnas EXCEPTO 'date' y 'year'
    # subset=... = especifica qué columnas usar para detectar duplicados
    # keep=False = ELIMINA TODAS las filas duplicadas (incluyendo la primera)
    # .copy() = crea una copia para evitar modificar el original
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    
    # Obtiene el número de filas DESPUÉS de eliminar duplicados
    after = df.shape[0]
    
    # Calcula y muestra cuántas filas fueron eliminadas
    # (before - after) = número de filas removidas
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    
    # Retorna el DataFrame sin filas duplicadas
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    run_preprocess()
