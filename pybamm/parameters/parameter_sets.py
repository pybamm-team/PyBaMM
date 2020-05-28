#
# Parameter sets from papers
#
"""
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.
"""

#
# Lithium-ion
#
NCA_Kim2011 = {
    "chemistry": "lithium-ion",
    "cell": "Kim2011",
    "anode": "graphite_Kim2011",
    "separator": "separator_Kim2011",
    "cathode": "nca_Kim2011",
    "electrolyte": "lipf6_Kim2011",
    "experiment": "1C_discharge_from_full_Kim2011",
    "sei": "example",
    "citation": "kim2011multi",
}

Ecker2015 = {
    "chemistry": "lithium-ion",
    "cell": "kokam_Ecker2015",
    "anode": "graphite_Ecker2015",
    "separator": "separator_Ecker2015",
    "cathode": "LiNiCoO2_Ecker2015",
    "electrolyte": "lipf6_Ecker2015",
    "experiment": "1C_discharge_from_full_Ecker2015",
    "sei": "example",
    "citation": ["ecker2015i", "ecker2015ii", "richardson2020"],
}

Marquis2019 = {
    "chemistry": "lithium-ion",
    "cell": "kokam_Marquis2019",
    "anode": "graphite_mcmb2528_Marquis2019",
    "separator": "separator_Marquis2019",
    "cathode": "lico2_Marquis2019",
    "electrolyte": "lipf6_Marquis2019",
    "experiment": "1C_discharge_from_full_Marquis2019",
    "sei": "example",
    "citation": "marquis2019asymptotic",
}

Chen2020 = {
    "chemistry": "lithium-ion",
    "cell": "LGM50_Chen2020",
    "anode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "cathode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",
    "sei": "example",
    "citation": "Chen2020",
}

Mohtat2020 = {
    "chemistry": "lithium-ion",
    "cell": "UMBL_Mohtat2020",
    "anode": "graphite_UMBL_Mohtat2020",
    "separator": "separator_Mohtat2020",
    "cathode": "NMC_UMBL_Mohtat2020",
    "electrolyte": "LiPF6_Mohtat2020",
    "experiment": "1C_charge_from_empty_Mohtat2020",
    "sei": "example",
    "citation": "Mohtat2020",
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
    "citation": "sulzer2019physical",
}
