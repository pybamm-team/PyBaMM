"""
Smoke test: run 100-cycle aging simulations for both a standard single-phase
SPM and a composite (graphite+silicon) SPM and report pass/fail.
"""

import sys
import importlib.util
import traceback

# Use the local development version of pybamm
sys.path.insert(0, "src")
import pybamm

# Load Chen2020_composite_mod directly from the repo source file
_spec = importlib.util.spec_from_file_location(
    "Chen2020_composite_mod",
    "src/pybamm/input/parameters/lithium_ion/Chen2020_composite_mod.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_composite_params = _mod.get_parameter_values

# ---------------------------------------------------------------------------
# Experiment: simple CC discharge / CC-CV charge, repeated N times
# ---------------------------------------------------------------------------
N_CYCLES = 10

experiment = pybamm.Experiment(
    [
        (
            "Discharge at 0.5C until 2.5 V",
            "Charge at 0.5C until 4.2 V",
            "Hold at 4.2 V until C/10",
        )
    ]
    * N_CYCLES
)

idaklu_solver = pybamm.IDAKLUSolver()



# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def run(label, build_model, build_params, solver):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        model = build_model()
        param_values = build_params()
        sim = pybamm.Simulation(
            model,
            parameter_values=param_values,
            experiment=experiment,
            solver=solver,
        )
        sim.solve()
        print(f"  PASS — solved {N_CYCLES} cycles successfully")
        return True
    except Exception:
        print("  FAIL")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 1. Standard single-phase SPM with aging
# ---------------------------------------------------------------------------
normal_options = {
    "SEI": "solvent-diffusion limited",
    "SEI film resistance": "average",
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "particle mechanics": "swelling and cracking",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "false",
}

result_normal = run(
    "Standard SPM — single-phase aging",
    lambda: pybamm.lithium_ion.SPM(options=normal_options),
    lambda: pybamm.ParameterValues("Chen2020"),
    idaklu_solver,
)


# ---------------------------------------------------------------------------
# 2. Composite SPM (graphite + silicon negative electrode) with aging
# ---------------------------------------------------------------------------
composite_options = {
    "SEI": (("solvent-diffusion limited", "solvent-diffusion limited"), "none"),
    "SEI film resistance": "average",
    "SEI on cracks": (("true", "false"), "false"),
    "loss of active material": (("stress-driven", "stress-driven"), "none"),
    "particle mechanics": (("swelling and cracking", "swelling only"), "none"),
    "lithium plating": (("irreversible", "none"), "none"),
    "lithium plating porosity change": "false",
    "particle phases": ("2", "1"),
    "open-circuit potential": (("single", "current sigmoid"), "single"),
}

result_composite = run(
    "Composite SPM — graphite + silicon aging",
    lambda: pybamm.lithium_ion.SPM(options=composite_options),
    lambda: pybamm.ParameterValues(get_composite_params()),
    idaklu_solver,
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Standard SPM:  {'PASS' if result_normal    else 'FAIL'}")
print(f"  Composite SPM: {'PASS' if result_composite else 'FAIL'}")
print()

sys.exit(0 if (result_normal and result_composite) else 1)
