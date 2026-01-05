
import pandas as pd
import matplotlib.pyplot as plt

def replot_sei():
    # Load the CSV data
    try:
        df = pd.read_csv("sei_growth_analysis.csv")
    except FileNotFoundError:
        print("Error: sei_growth_analysis.csv not found.")
        return

    # Define corrections for Rest Growth (from simulation logs)
    # {Variant: {Cycle_Index: Correction_Value}}
    # Note: Cycle 51 in CSV corresponds to the "Post-Rest" growth.
    corrections = {
        # Solvent Diffusion Variants (Baseline, No LAM, No Plating, No Cracking, Only SEI)
        "Solvent-Diffusion": {51: 1.13e-9, 101: 0.80e-9, 151: 0.65e-9},
        # Electron Migration
        "Electron Migration SEI": {51: 0.77e-9, 101: 0.56e-9, 151: 0.46e-9}
    }

    # Map variant names to correction types
    variant_map = {
        "Baseline": "Solvent-Diffusion",
        "No LAM": "Solvent-Diffusion",
        "No Plating": "Solvent-Diffusion",
        "No Cracking": "Solvent-Diffusion",
        "Only SEI": "Solvent-Diffusion",
        "Electron Migration SEI": "Electron Migration SEI"
    }

    # Create figure
    # We want to replicate the 3x2 layout or just focus on SEI?
    # The user complained about "mechanism_isolation_detailed.png", which is the full grid.
    # But I only have data for SEI in the CSV.
    # To fully replicate the image, I would need CC/CV data which is NOT in the CSV (based on previous view).
    # Wait, check CSV columns?
    # CSV cols: Variant, Cycle, SEI Thickness [m], Growth Rate [m/cycle], Crack Length [m], Period
    # It does NOT have CC/CV capacity/time.
    # So I cannot recreate the EXACT full 6-panel plot perfectly without that data.
    # However, I can create a corrected "SEI Summary" plot which is what matters for "Growth Rate".
    # Or I can try to modify just the SEI panels? No, I can't edit an existing pixel image.
    
    # STRATEGY CHANGE:
    # Since I cannot recreate the CC/CV plots, I will create a NEW plot focused on the SEI correctness:
    # "sei_mechanism_corrected.png"
    # And I will explain to the user that I've isolated the SEI plots.
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    variants = df["Variant"].unique()
    
    for name in variants:
        data = df[df["Variant"] == name].sort_values("Cycle")
        cycles = data["Cycle"].values
        thickness = data["SEI Thickness [m]"].values
        # Recalculate rates to be sure
        rates = [0.0] * len(thickness)
        for i in range(1, len(thickness)):
            rates[i] = thickness[i] - thickness[i-1]
            
            # Apply Correction if applicable
            current_cycle = cycles[i]
            if current_cycle in [51, 101, 151]:
                corr_type = variant_map.get(name)
                if corr_type and current_cycle in corrections[corr_type]:
                    correction = corrections[corr_type][current_cycle]
                    # Subtract the rest component from the rate
                    rates[i] -= correction
                    if rates[i] < 0: rates[i] = 0.0 # Safety
        
        # Plot Thickness
        axs[0].plot(cycles, thickness, marker='.', label=name)
        
        # Plot Corrected Rate
        axs[1].plot(cycles, rates, marker='.', label=name)

    axs[0].set_ylabel("SEI Thickness [m]")
    axs[0].set_title("Total SEI Thickness")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_ylabel("Growth Rate [m/cycle] (Corrected)")
    axs[1].set_title("SEI Growth Rate (Cycling Only)")
    axs[1].grid(True)
    axs[1].set_yscale('log')
    
    plt.tight_layout()
    plot_filename = "mechanism_isolation_corrected.png"
    plt.savefig(plot_filename)
    print(f"Corrected plot saved to {plot_filename}")

if __name__ == "__main__":
    replot_sei()
