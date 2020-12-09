#
# Parameter sets from papers
#
"""
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.

Lithium-ion parameter sets
--------------------------
    * Chen2020 :
        C.-H. Chen, F. Brosa Planella, K. O’Regan, D. Gastol, W. D. Widanage, and E.
        Kendrick. "Development of Experimental Techniques for Parameterization of
        Multi-scale Lithium-ion Battery Models." Journal of the Electrochemical Society,
        167(8), 080534 (2020).
    * Ecker2015 :
        M. Ecker, T. K. D. Tran, P. Dechent, S. Käbitz, A. Warnecke, and D. U. Sauer.
        "Parameterization of a Physico-Chemical Model of a Lithium-Ion Battery. I.
        Determination of Parameters." Journal of the Electrochemical Society, 162(9),
        A1836-A1848 (2015).
    * Marquis2019 :
        S. G. Marquis, V. Sulzer, R. Timms, C. P. Please and S. J. Chapman. "An
        asymptotic derivation of a single particle model with electrolyte." Journal of
        the Electrochemical Society, 166(15), A3693–A3706 (2019).
    * Mohtat2020 :
        P. Mohtat, S. Lee, V. Sulzer, J. B. Siegel & A. G. Stefanopoulou. "Differential
        Expansion and Voltage Model for Li-ion Batteries at Practical Charging Rates".
        Journal of The Electrochemical Society, 167(11), 110561 (2020).
    * NCA_Kim2011 :
        G. H. Kim, K. Smith, K. J. Lee, S. Santhanagopalan, and A. Pesaran.
        "Multi-domain modeling of lithium-ion batteries encompassing multi-physics in
        varied length scales." Journal of The Electrochemical Society, 158(8), A955-A969
        (2011).
    * Ramadass2004 :
        P. Ramadass, B. Haran, P. M. Gomadam, R. White, and B. N. Popov. "Development
        of First Principles Capacity Fade Model for Li-Ion Cells." Journal of the
        Electrochemical Society, 151(2), A196-A203 (2004).
    * Ai 2020 :
        > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
        Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
        in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
        167(1), 013512. DOI: 10.1149/2.0122001JES.
    * Prada2013 :
        C.-H. Chen, F. Brosa Planella, K. O’Regan, D. Gastol, W. D. Widanage, and E.
        Kendrick. "Development of Experimental Techniques for Parameterization of
        Multi-scale Lithium-ion Battery Models." Journal of the Electrochemical Society,
        167(8), 080534 (2020).
        E. Prada, D., Di Domenico, Y., Creff, J., Bernard, V., Sauvant-Moynot, and F.
        Huet. "A simplified electrochemical and thermal aging model of LiFePO4-graphite
        Li-ion batteries: power and capacity fade simulations". Journal of The
        Electrochemical Society, 160(4), A616 (2013).
        M. J. Lain, J., Brandon, and E. Kendrick. "Design Strategies for High Power vs.
        High Energy Lithium Ion Cells". Batteries, 5(4), 64 (2019).

Lead-acid parameter sets
--------------------------
    * Sulzer2019 :
        V. Sulzer, S. J. Chapman, C. P. Please, D. A. Howey, and C. W. Monroe, "Faster
        lead-acid battery simulations from porous-electrode theory: Part I. Physical
        model." Journal of the Electrochemical Society, 166(12), 2363 (2019).

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
    "citation": "mohtat2020differential",
}

Ramadass2004 = {
    "chemistry": "lithium-ion",
    "cell": "sony_Ramadass2004",
    "anode": "graphite_Ramadass2004",
    "separator": "separator_Ecker2015",  # no values found, relevance?
    "cathode": "lico2_Ramadass2004",
    "electrolyte": "lipf6_Ramadass2004",
    "experiment": "1C_discharge_from_full_Ramadass2004",
    "sei": "ramadass2004",
    "citation": "marquis2019asymptotic",
}

Prada2013 = {
    "chemistry": "lithium-ion",
    "cell": "A123_Lain2019",
    "anode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "cathode": "LFP_Prada2013",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "4C_discharge_from_full_Prada2013",
    "citation": ["Chen2020", "lain2019design", "prada2013simplified"],
}

Ai2020 = {
    "chemistry": "lithium-ion",
    "cell": "Enertech_Ai2020",
    "anode": "graphite_Ai2020",
    "separator": "separator_Ai2020",
    "cathode": "lico2_Ai2020",
    "electrolyte": "lipf6_Enertech_Ai2020",
    "experiment": "1C_discharge_from_full_Ai2020",
    "citation": "Ai2020JES",
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
