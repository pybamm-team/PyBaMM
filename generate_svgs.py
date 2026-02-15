import matplotlib.pyplot as plt
import pandas as pd


def generate_svgs():
    csv_filename = "sei_growth_analysis.csv"
    try:
        df = pd.read_csv(csv_filename)
        print(f"Loaded {csv_filename}")
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    # Create figure
    fig, axs = plt.subplots(4, 2, figsize=(15, 16))

    variants = df["Variant"].unique()

    # Define correction logic for rates
    correction_map = {
        "Electron Migration SEI": {51: 0.77e-9, 101: 0.56e-9, 151: 0.46e-9},
    }
    default_correction = {51: 1.13e-9, 101: 0.80e-9, 151: 0.65e-9}

    for name in variants:
        data = df[df["Variant"] == name].sort_values("Cycle")
        cycles = data["Cycle"].values

        # 1. Discharge Capacity (Row 0, Left)
        axs[0, 0].plot(cycles, data["Discharge Capacity [A.h]"], marker=".", label=name)

        # 2. Total Charge Capacity (Row 0, Right)
        axs[0, 1].plot(
            cycles, data["Total Charge Capacity [A.h]"], marker=".", label=name
        )

        # 3. CC Capacity (Row 1, Left)
        axs[1, 0].plot(cycles, data["CC Capacity [A.h]"], marker=".", label=name)

        # 4. CC Time (Row 1, Right)
        axs[1, 1].plot(cycles, data["CC Time [h]"], marker=".", label=name)

        # 5. CV Capacity (Row 2, Left)
        axs[2, 0].plot(cycles, data["CV Capacity [A.h]"], marker=".", label=name)

        # 6. CV Time (Row 2, Right)
        axs[2, 1].plot(cycles, data["CV Time [h]"], marker=".", label=name)

        # 7. SEI Thickness (Row 3, Left)
        axs[3, 0].plot(cycles, data["SEI Thickness [m]"], marker=".", label=name)

        # 8. SEI Rate (Corrected) (Row 3, Right)
        rates = data["Growth Rate [m/cycle]"].values
        corrected_rates = list(rates)
        correction_dict = correction_map.get(name, default_correction)

        for idx, c in enumerate(cycles):
            if c in correction_dict:
                corrected_rates[idx] = max(0, corrected_rates[idx] - correction_dict[c])

        axs[3, 1].plot(cycles, corrected_rates, marker=".", label=name)

    # Formatting
    axs[0, 0].set_ylabel("Discharge Capacity [A.h]")
    axs[0, 0].set_title("Discharge Capacity")
    axs[0, 0].grid(True)
    axs[0, 0].legend(fontsize="small")

    axs[0, 1].set_ylabel("Total Charge Capacity [A.h]")
    axs[0, 1].set_title("Total Charge Capacity (CC+CV)")
    axs[0, 1].grid(True)

    axs[1, 0].set_ylabel("CC Capacity [A.h]")
    axs[1, 0].set_title("CC Charge Capacity")
    axs[1, 0].grid(True)

    axs[1, 1].set_ylabel("CC Time [h]")
    axs[1, 1].set_title("CC Charge Time")
    axs[1, 1].grid(True)

    axs[2, 0].set_ylabel("CV Capacity [A.h]")
    axs[2, 0].set_title("CV Charge Capacity")
    axs[2, 0].grid(True)

    axs[2, 1].set_ylabel("CV Time [h]")
    axs[2, 1].set_title("CV Charge Time")
    axs[2, 1].grid(True)

    axs[3, 0].set_ylabel("SEI Thickness [m]")
    axs[3, 0].set_title("Total SEI Thickness")
    axs[3, 0].grid(True)

    axs[3, 1].set_ylabel("Growth Rate [m/cycle] (Corrected)")
    axs[3, 1].set_title("SEI Growth Rate")
    axs[3, 1].grid(True)
    axs[3, 1].set_yscale("log")

    fig.tight_layout()
    svg_filename = "mechanism_isolation_detailed.svg"
    fig.savefig(svg_filename, format="svg")
    print(f"Plot saved to {svg_filename}")


if __name__ == "__main__":
    generate_svgs()
