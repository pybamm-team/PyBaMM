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

NCA_Kim2011
    Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

Chen2020
    Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W. Dhammika Widanage, and Emma Kendrick. "The development of accurate experimental techniques for parameterization of multi-scale lithium-ion battery models." In preparation (2020).


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

NCA_Kim2011 = {
    "chemistry": "lithium-ion",
    "cell": "Kim2011",
    "anode": "graphite_Kim2011",
    "separator": "separator_Kim2011",
    "cathode": "nca_Kim2011",
    "electrolyte": "lipf6_Kim2011",
    "experiment": "1C_discharge_from_full_Kim2011",
}

Chen2020 = {
    "chemistry": "lithium-ion",
    "cell": "LGM50_Chen2020",
    "anode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "cathode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",
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
