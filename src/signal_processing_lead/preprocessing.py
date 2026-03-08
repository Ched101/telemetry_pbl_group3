from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "PM2_5": "PM2_5_RAW_LCSs_TURDATA.xlsx",
    "PM10": "PM10_RAW_LCSs_TURDATA.xlsx",
    "NO2": "NO2_RAW_LCSs_TURDATA.xlsx",
    "O3": "O3_RAW_LCSs_TURDATA.xlsx",
}

UNITS = {
    "PM2_5": "ug/m3",
    "PM10": "ug/m3",
    "NO2": "ppb",
    "O3": "ppb",
}

REQUIRED_META_COLS = ["dt_beg_utc", "dt_end_utc", "location", "measurement_program"]


def extract_sensor_id(col_name: str) -> str:
    """
    Example:
    PM2_5_S2R -> S2
    NO2_S10R  -> S10
    """
    match = re.search(r"_S(\d+)R$", col_name)
    if not match:
        raise ValueError(f"Could not extract sensor ID from column: {col_name}")
    return f"S{match.group(1)}"


def load_one_pollutant(pollutant: str, filename: str) -> pd.DataFrame:
    """
    Load one workbook and convert from wide format to tidy long format.
    """
    path = RAW_DIR / filename
    df = pd.read_excel(path, engine="openpyxl")

    # -------- 1. Basic structural validation --------
    missing_meta = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"{filename} is missing required columns: {missing_meta}")

    sensor_cols = [c for c in df.columns if re.search(r"_S\d+R$", c)]
    if not sensor_cols:
        raise ValueError(f"No sensor columns found in {filename}")

    # -------- 2. Timestamp standardization --------
    df["dt_beg_utc"] = pd.to_datetime(df["dt_beg_utc"], utc=True, errors="coerce")
    df["dt_end_utc"] = pd.to_datetime(df["dt_end_utc"], utc=True, errors="coerce")

    # Remove rows with broken timestamps
    df = df.dropna(subset=["dt_beg_utc", "dt_end_utc"]).copy()

    # -------- 3. Convert wide -> long --------
    tidy = df.melt(
        id_vars=REQUIRED_META_COLS,
        value_vars=sensor_cols,
        var_name="raw_sensor_column",
        value_name="value",
    )

    # -------- 4. Add standard fields --------
    tidy["pollutant"] = pollutant
    tidy["unit"] = UNITS[pollutant]
    tidy["sensor_id"] = tidy["raw_sensor_column"].apply(extract_sensor_id)

    # -------- 5. Known data-quality flags --------
    tidy["is_missing"] = tidy["value"].isna()
    tidy["is_invalid_known_issue"] = False
    tidy["invalid_reason"] = pd.NA

    # Known issue from TURDATA readme:
    # NO2_S9R is invalid during Legerova campaign and final comparative measurement
    if pollutant == "NO2":
        mask_no2_s9 = (
            (tidy["sensor_id"] == "S9")
            & (tidy["measurement_program"].isin(
                ["Legerova_campaign", "Final_comparative_measurement"]
            ))
        )
        tidy.loc[mask_no2_s9, "is_invalid_known_issue"] = True
        tidy.loc[mask_no2_s9, "invalid_reason"] = "Known drift in NO2_S9R"

    # -------- 6. Interval sanity check --------
    tidy["interval_hours"] = (
        (tidy["dt_end_utc"] - tidy["dt_beg_utc"]).dt.total_seconds() / 3600.0
    )

    # Reorder columns for consistency
    tidy = tidy[
        [
            "dt_beg_utc",
            "dt_end_utc",
            "interval_hours",
            "location",
            "measurement_program",
            "pollutant",
            "unit",
            "sensor_id",
            "raw_sensor_column",
            "value",
            "is_missing",
            "is_invalid_known_issue",
            "invalid_reason",
        ]
    ].sort_values(["pollutant", "sensor_id", "dt_beg_utc"]).reset_index(drop=True)

    return tidy


def build_master_dataset() -> pd.DataFrame:
    parts = []
    for pollutant, filename in FILES.items():
        tidy = load_one_pollutant(pollutant, filename)
        parts.append(tidy)

    master = pd.concat(parts, ignore_index=True)

    # Optional: remove impossible duplicate records if any appear
    master = master.drop_duplicates(
        subset=["dt_beg_utc", "dt_end_utc", "pollutant", "sensor_id"]
    ).reset_index(drop=True)

    return master


def build_qc_summary(master: pd.DataFrame) -> pd.DataFrame:
    qc = (
        master.groupby(["pollutant", "sensor_id"], dropna=False)
        .agg(
            total_rows=("value", "size"),
            missing_rows=("is_missing", "sum"),
            known_invalid_rows=("is_invalid_known_issue", "sum"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            mean_value=("value", "mean"),
        )
        .reset_index()
    )

    qc["missing_pct"] = 100 * qc["missing_rows"] / qc["total_rows"]
    qc["known_invalid_pct"] = 100 * qc["known_invalid_rows"] / qc["total_rows"]
    return qc.sort_values(["pollutant", "sensor_id"]).reset_index(drop=True)


if __name__ == "__main__":
    master = build_master_dataset()
    qc = build_qc_summary(master)

    master.to_csv(PROCESSED_DIR / "turdata_master_tidy.csv", index=False)
    qc.to_csv(PROCESSED_DIR / "turdata_qc_summary.csv", index=False)

    print("Saved:")
    print(" - data/processed/turdata_master_tidy.csv")
    print(" - data/processed/turdata_qc_summary.csv")
    print()
    print(f"Master shape: {master.shape}")
    print(qc.head(10).to_string(index=False))