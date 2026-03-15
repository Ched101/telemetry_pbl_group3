import subprocess
import sys
from pathlib import Path
import pandas as pd


COMBINED_FILE = "data/processed/high_priority_segments_combined.csv"
RESULTS_DIR = Path("results/modulation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd):
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def main():
    combined_path = Path(COMBINED_FILE)
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined file not found: {combined_path}")

    df = pd.read_csv(combined_path)

    required_cols = ["pollutant", "sensor_id", "segment_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in combined file: {col}")

    unique_segments = (
        df[["pollutant", "sensor_id", "segment_id"]]
        .drop_duplicates()
        .sort_values(["pollutant", "sensor_id", "segment_id"])
        .reset_index(drop=True)
    )

    print("Segments to process:")
    print(unique_segments.to_string(index=False))

    summary_rows = []

    for _, row in unique_segments.iterrows():
        pollutant = str(row["pollutant"])
        sensor_id = str(row["sensor_id"])
        segment_id = int(row["segment_id"])

        print(f"\n=== Processing {pollutant} | {sensor_id} | segment {segment_id} ===")

        # Run AM
        run_command([
            sys.executable,
            "-m",
            "src.modulation_lead.am_modulation",
            "--input", COMBINED_FILE,
            "--combined",
            "--pollutant", pollutant,
            "--sensor_id", sensor_id,
            "--segment_id", str(segment_id),
        ])

        # Run FM
        run_command([
            sys.executable,
            "-m",
            "src.modulation_lead.fm_modulation",
            "--input", COMBINED_FILE,
            "--combined",
            "--pollutant", pollutant,
            "--sensor_id", sensor_id,
            "--segment_id", str(segment_id),
        ])

        # Run ASK
        run_command([
            sys.executable,
            "-m",
            "src.modulation_lead.digital_modulation",
            "--input", COMBINED_FILE,
            "--combined",
            "--pollutant", pollutant,
            "--sensor_id", sensor_id,
            "--segment_id", str(segment_id),
            "--scheme", "ASK",
        ])

        # Run FSK
        run_command([
            sys.executable,
            "-m",
            "src.modulation_lead.digital_modulation",
            "--input", COMBINED_FILE,
            "--combined",
            "--pollutant", pollutant,
            "--sensor_id", sensor_id,
            "--segment_id", str(segment_id),
            "--scheme", "FSK",
        ])

        # Run PSK
        run_command([
            sys.executable,
            "-m",
            "src.modulation_lead.digital_modulation",
            "--input", COMBINED_FILE,
            "--combined",
            "--pollutant", pollutant,
            "--sensor_id", sensor_id,
            "--segment_id", str(segment_id),
            "--scheme", "PSK",
        ])

        # Collect metrics from the segment-specific CSV files
        label = f"{pollutant}_{sensor_id}_seg{segment_id}"

        am_file = RESULTS_DIR / f"am_metrics_{label}.csv"
        fm_file = RESULTS_DIR / f"fm_metrics_{label}.csv"
        ask_file = RESULTS_DIR / f"ask_metrics_{label}.csv"
        fsk_file = RESULTS_DIR / f"fsk_metrics_{label}.csv"
        psk_file = RESULTS_DIR / f"psk_metrics_{label}.csv"

        if am_file.exists():
            am_df = pd.read_csv(am_file)
            summary_rows.append(am_df.iloc[0].to_dict())

        if fm_file.exists():
            fm_df = pd.read_csv(fm_file)
            summary_rows.append(fm_df.iloc[0].to_dict())

        if ask_file.exists():
            ask_df = pd.read_csv(ask_file)
            summary_rows.append(ask_df.iloc[0].to_dict())

        if fsk_file.exists():
            fsk_df = pd.read_csv(fsk_file)
            summary_rows.append(fsk_df.iloc[0].to_dict())

        if psk_file.exists():
            psk_df = pd.read_csv(psk_file)
            summary_rows.append(psk_df.iloc[0].to_dict())

    if not summary_rows:
        raise ValueError("No summary rows were collected.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "all_modulation_results_summary.csv", index=False)

    print("\nBatch analysis complete.")
    print(f"Saved summary to: {RESULTS_DIR / 'all_modulation_results_summary.csv'}")
    print("\nSummary preview:")
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()