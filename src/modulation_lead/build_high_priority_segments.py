from pathlib import Path
import pandas as pd


def main():
    base_dir = Path("data/processed/selected_segments")
    inventory_path = base_dir / "segment_inventory_all_pollutants.csv"
    output_path = Path("data/processed/high_priority_segments_combined.csv")

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

    inventory = pd.read_csv(inventory_path)

    # Normalize text columns just in case
    inventory["recommended_for_modulation"] = (
        inventory["recommended_for_modulation"].astype(str).str.strip().str.lower()
    )
    inventory["priority"] = inventory["priority"].astype(str).str.strip().str.lower()

    # Keep only high-priority, recommended segments
    selected = inventory[
        (inventory["recommended_for_modulation"] == "yes") &
        (inventory["priority"] == "high")
    ].copy()

    if selected.empty:
        raise ValueError("No high-priority recommended segments found in inventory.")

    combined_frames = []

    for _, row in selected.iterrows():
        # Try to find the segment filename
        if "filename" in row.index:
            segment_file = base_dir / str(row["filename"])
        else:
            pollutant = str(row["pollutant"])
            sensor_id = str(row["sensor_id"])
            segment_id = str(row["segment_id"])
            segment_file = base_dir / f"{pollutant}_{sensor_id}_segment_{segment_id}.csv"

        if not segment_file.exists():
            print(f"Skipping missing file: {segment_file}")
            continue

        seg_df = pd.read_csv(segment_file)

        required_cols = {"pollutant", "sensor_id", "segment_id", "value_sg"}
        missing = required_cols - set(seg_df.columns)
        if missing:
            print(f"Skipping {segment_file.name} because columns are missing: {missing}")
            continue

        seg_df = seg_df.copy()
        seg_df["source_file"] = segment_file.name
        seg_df["sample_index"] = range(len(seg_df))

        # Keep useful columns if they exist
        preferred_cols = [
            "pollutant",
            "sensor_id",
            "segment_id",
            "dt_beg_utc",
            "dt_end_utc",
            "sample_index",
            "value_raw",
            "value_sg",
            "source_file",
        ]
        keep_cols = [c for c in preferred_cols if c in seg_df.columns]
        seg_df = seg_df[keep_cols]

        combined_frames.append(seg_df)

    if not combined_frames:
        raise ValueError("No segment files were successfully combined.")

    combined = pd.concat(combined_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    summary = (
        combined.groupby(["pollutant", "sensor_id", "segment_id"])
        .size()
        .reset_index(name="num_samples")
    )

    print("Combined file created successfully.")
    print(f"Saved to: {output_path}")
    print("\nIncluded segments:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()