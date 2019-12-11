#
# Parameter sets from papers
#
"""
Parameter sets from papers:

Lithium-ion
-----------
Marquis2019
    Scott G. Marquis, Valentin Sulzer, Robert Timms, Colin P. Please, and S. Jon
    Chapman. "An asymptotic derivation of a single particle model with electrolyte."
    `arXiv preprint arXiv:1905.12553 <https://arxiv.org/abs/1905.12553>`_ (2019).


Lead-acid
---------
Sulzer2019
    V. Sulzer, S. J. Chapman, C. P. Please, D. A. Howey, and C. W.Monroe, “Faster
    lead-acid battery simulations from porous-electrode theory: Part I. Physical model.”
    `Journal of the Electrochemical Society <https://doi.org/10.1149/2.0301910jes>`_,
    166(12), 2363 (2019).
"""

#
# Lithium-ion
#
Marquis2019 = {
    "chemistry": "lithium-ion",
    "cell": "kokam_Marquis2019",
    "anode": "graphite_mcmb2528_Marquis2019",
    "separator": "separator_Marquis2019",
    "cathode": "lico2_Marquis2019",
    "electrolyte": "lipf6_Marquis2019",
    "experiment": "1C_discharge_from_full_Marquis2019",
}

#
# Lead-acid
#
Sulzer2019 = {
    "chemistry": "lead-acid",
    "cell": "BBOXX_Sulzer2019",
    "anode": "lead_Sulzer2019",
    "separator": "agm_Sulzer2019",
    "cathode": "lead_dioxide_Sulzer2019",
    "electrolyte": "sulfuric_acid_Sulzer2019",
    "experiment": "1C_discharge_from_full",
}
