#!/usr/bin/env python3
"""
Test script to demonstrate MSMR module usage
"""

# Import the setup script to add MSMR utilities to Python path

# Now import the MSMR modules
import matplotlib.pyplot as plt
import numpy as np
import utilities.msmr as msmr
import utilities.plotting as plotting

print("MSMR modules successfully imported!")
print("Available functions in msmr module:")
print([func for func in dir(msmr) if not func.startswith("_")])

print("\nAvailable functions in plotting module:")
print([func for func in dir(plotting) if not func.startswith("_")])

# Test a simple MSMR calculation
print("\nTesting MSMR calculation...")

# Example parameters for a single reaction
U = np.linspace(3.0, 4.2, 100)  # Voltage range
U0 = 3.7  # Standard electrode potential
Xj = 1.0  # Maximum fractional occupancy
w = 0.1  # Thermodynamic factor
T = 298.15  # Temperature (K)

# Calculate individual reaction response
xj, dxjdu = msmr.individual_reactions(U, U0, Xj, w, T)

print(f"Calculated {len(xj)} points for individual reaction")
print(f"Voltage range: {U.min():.2f} to {U.max():.2f} V")
print(f"Fractional occupancy range: {xj.min():.3f} to {xj.max():.3f}")

# Create a simple plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(U, xj, "b-", linewidth=2)
plt.xlabel("Voltage (V)")
plt.ylabel("Fractional Occupancy")
plt.title("MSMR Individual Reaction")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(U, -dxjdu, "r-", linewidth=2)
plt.xlabel("Voltage (V)")
plt.ylabel("-dX/dU")
plt.title("Differential Capacity")
plt.grid(True)

plt.tight_layout()
plt.savefig("msmr_test_plot.png", dpi=150, bbox_inches="tight")
print("Test plot saved as 'msmr_test_plot.png'")

print("\nMSMR installation and test completed successfully!")
