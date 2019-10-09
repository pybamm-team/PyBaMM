#
# Parameter sets from papers
#
"""
Parameter sets from papers:

Sulzer2019
    > V. Sulzer, S. J. Chapman, C. P. Please, D. A.Howey, and C. W.Monroe, “Faster lead-acid
    > battery simulations from porous-electrode theory: Part I. Physical model,” [Journal of
    > the Electrochemical Society](https://doi.org/10.1149/2.0301910jes), 166(12), 2363 (2019).

Marquis2019

"""

# Lead-acid

Sulzer2019 = {
    "chemistry": "lead-acid",
    "cell": "BBOXX_Sulzer2019",
    "anode": "lead_Sulzer2019",
    "separator": "agm_Sulzer2019",
    "cathode": "lead_dioxide_Sulzer2019",
    "electrolyte": "sulfuric_acid_Sulzer2019",
    "thermal": "default",
}

# Lithium-ion

Marquis2019 = {
    "chemistry": "lithium-ion",
    "cell": "BBOXX_Marquis2019",
    "anode": "lead_Marquis2019",
    "separator": "agm_Marquis2019",
    "cathode": "lead_dioxide_Marquis2019",
    "electrolyte": "sulfuric_acid_Marquis2019",
    "thermal": "default",
}
