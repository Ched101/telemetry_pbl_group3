from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    input_file = Path("results/modulation/all_modulation_results_summary.csv")
    output_dir = Path("results/modulation")
    figure_dir = Path("results/figures")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Summary file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Split analog and digital schemes
    analog_df = df[df["scheme"].isin(["AM", "FM"])].copy()
    digital_df = df[df["scheme"].isin(["ASK", "FSK", "PSK"])].copy()

    # -----------------------------
    # Analog summary
    # -----------------------------
    analog_summary = (
        analog_df.groupby("scheme")[["mse", "correlation", "recovered_snr_db"]]
        .mean()
        .reset_index()
        .sort_values("mse")
    )

    analog_summary.to_csv(output_dir / "analog_modulation_summary.csv", index=False)

    # -----------------------------
    # Digital summary
    # -----------------------------
    digital_summary = (
        digital_df.groupby("scheme")[["ber"]]
        .mean()
        .reset_index()
        .sort_values("ber")
    )

    digital_summary.to_csv(output_dir / "digital_modulation_summary.csv", index=False)

    # -----------------------------
    # Combined ranking table
    # -----------------------------
    combined_rows = []

    for _, row in analog_summary.iterrows():
        combined_rows.append({
            "scheme": row["scheme"],
            "category": "Analog",
            "mean_mse": row["mse"],
            "mean_correlation": row["correlation"],
            "mean_recovered_snr_db": row["recovered_snr_db"],
            "mean_ber": None,
        })

    for _, row in digital_summary.iterrows():
        combined_rows.append({
            "scheme": row["scheme"],
            "category": "Digital",
            "mean_mse": None,
            "mean_correlation": None,
            "mean_recovered_snr_db": None,
            "mean_ber": row["ber"],
        })

    combined_summary = pd.DataFrame(combined_rows)
    combined_summary.to_csv(output_dir / "modulation_comparison_table.csv", index=False)

    # -----------------------------
    # Plot 1: Analog comparison
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(analog_summary["scheme"], analog_summary["mse"])
    plt.title("Average MSE for Analog Modulation Schemes")
    plt.xlabel("Scheme")
    plt.ylabel("Mean MSE")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_analog_mean_mse_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(analog_summary["scheme"], analog_summary["correlation"])
    plt.title("Average Correlation for Analog Modulation Schemes")
    plt.xlabel("Scheme")
    plt.ylabel("Mean Correlation")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_analog_mean_correlation_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(analog_summary["scheme"], analog_summary["recovered_snr_db"])
    plt.title("Average Recovered SNR for Analog Modulation Schemes")
    plt.xlabel("Scheme")
    plt.ylabel("Mean Recovered SNR [dB]")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_analog_mean_snr_comparison.png", dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: Digital BER comparison
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(digital_summary["scheme"], digital_summary["ber"])
    plt.title("Average BER for Digital Modulation Schemes")
    plt.xlabel("Scheme")
    plt.ylabel("Mean BER")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_digital_mean_ber_comparison.png", dpi=300)
    plt.close()

    print("Summary analysis complete.")
    print("\nAnalog summary:")
    print(analog_summary.to_string(index=False))
    print("\nDigital summary:")
    print(digital_summary.to_string(index=False))
    print(f"\nSaved summary tables to: {output_dir}")
    print(f"Saved summary figures to: {figure_dir}")


if __name__ == "__main__":
    main()