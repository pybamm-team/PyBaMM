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
Marquis2019 = {
    "chemistry": "lithium-ion",
    "cell": "kokam_Marquis2019",
    "anode": "graphite_mcmb2528_Marquis2019",
    "separator": "separator_Marquis2019",
    "cathode": "lico2_Marquis2019",
    "electrolyte": "lipf6_Marquis2019",
    "experiment": "1C_discharge_from_full_Marquis2019",
    "citation": "marquis2019asymptotic",
}

Xu2019 = {
    "chemistry": "lithium-ion",
    "cell": "li_metal_Xu2019",
    "anode": "li_metal_Xu2019",
    "separator": "separator_Xu2019",
    "cathode": "NMC532_Xu2019",
    "electrolyte": "lipf6_Valoen2005",
    "experiment": "1C_discharge_from_full_Xu2019",
}

NCA_Kim2011 = {
    "chemistry": "lithium-ion",
    "cell": "Kim2011",
    "anode": "graphite_Kim2011",
    "separator": "separator_Kim2011",
    "cathode": "nca_Kim2011",
    "electrolyte": "lipf6_Kim2011",
    "experiment": "1C_discharge_from_full_Kim2011",
    "citation": "kim2011multi",
}

Chen2020 = {
    "chemistry": "lithium-ion",
    "cell": "LGM50_Chen2020",
    "anode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "cathode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",
    "citation": "Chen2020",
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
