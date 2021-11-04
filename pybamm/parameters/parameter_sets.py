#
# Parameter sets from papers
#
"""
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.

Lead-acid parameter sets
------------------------
    * Sulzer2019 :
       - Valentin Sulzer, S. Jon Chapman, Colin P. Please, David A. Howey, and Charles
         W. Monroe. Faster Lead-Acid Battery Simulations from Porous-Electrode Theory:
         Part I. Physical Model. Journal of The Electrochemical Society,
         166(12):A2363–A2371, 2019. doi:10.1149/2.0301910jes.

Lithium-ion parameter sets
--------------------------
    * Ai2020 :
       - Weilong Ai, Ludwig Kraft, Johannes Sturm, Andreas Jossen, and Billy Wu.
         Electrochemical thermal-mechanical modelling of stress inhomogeneity in
         lithium-ion pouch cells. Journal of The Electrochemical Society, 167(1):013512,
         2019. doi:10.1149/2.0122001JES.
    * Chen2020 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
    * Chen2020_plating :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
    * Ecker2015 :
       - Madeleine Ecker, Stefan Käbitz, Izaro Laresgoiti, and Dirk Uwe Sauer.
         Parameterization of a Physico-Chemical Model of a Lithium-Ion Battery: II.
         Model Validation. Journal of The Electrochemical Society, 162(9):A1849–A1857,
         2015. doi:10.1149/2.0541509jes.
       - Madeleine Ecker, Thi Kim Dung Tran, Philipp Dechent, Stefan Käbitz, Alexander
         Warnecke, and Dirk Uwe Sauer. Parameterization of a Physico-Chemical Model of a
         Lithium-Ion Battery: I. Determination of Parameters. Journal of the
         Electrochemical Society, 162(9):A1836–A1848, 2015. doi:10.1149/2.0551509jes.
       - Alastair Hales, Laura Bravo Diaz, Mohamed Waseem Marzook, Yan Zhao, Yatish
         Patel, and Gregory Offer. The cell cooling coefficient: a standard to define
         heat rejection from lithium-ion batteries. Journal of The Electrochemical
         Society, 166(12):A2383, 2019.
       - Giles Richardson, Ivan Korotkin, Rahifa Ranom, Michael Castle, and Jamie M.
         Foster. Generalised single particle models for high-rate operation of graded
         lithium-ion electrodes: systematic derivation and validation. Electrochimica
         Acta, 339:135862, 2020. doi:10.1016/j.electacta.2020.135862.
       - Yan Zhao, Yatish Patel, Teng Zhang, and Gregory J Offer. Modeling the effects
         of thermal gradients induced by tab and surface cooling on lithium ion cell
         performance. Journal of The Electrochemical Society, 165(13):A3169, 2018.
    * Marquis2019 :
       - Scott G. Marquis, Valentin Sulzer, Robert Timms, Colin P. Please, and S. Jon
         Chapman. An asymptotic derivation of a single particle model with electrolyte.
         Journal of The Electrochemical Society, 166(15):A3693–A3706, 2019.
         doi:10.1149/2.0341915jes.
    * Mohtat2020 :
       - Peyman Mohtat, Suhak Lee, Valentin Sulzer, Jason B. Siegel, and Anna G.
         Stefanopoulou. Differential Expansion and Voltage Model for Li-ion Batteries at
         Practical Charging Rates. Journal of The Electrochemical Society,
         167(11):110561, 2020. doi:10.1149/1945-7111/aba5d1.
    * NCA_Kim2011 :
       - Gi-Heon Kim, Kandler Smith, Kyu-Jin Lee, Shriram Santhanagopalan, and Ahmad
         Pesaran. Multi-domain modeling of lithium-ion batteries encompassing
         multi-physics in varied length scales. Journal of the Electrochemical Society,
         158(8):A955–A969, 2011. doi:10.1149/1.3597614.
    * ORegan2021 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Kieran O'Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma Kendrick.
         Thermal-electrochemical parametrisation of a lithium-ion battery: mapping Li
         concentration and temperature dependencies. Journal of The Electrochemical
         Society, ():, 2021. doi:.
    * Prada2013 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Michael J. Lain, James Brandon, and Emma Kendrick. Design strategies for high
         power vs. high energy lithium ion cells. Batteries, 5(4):64, 2019.
         doi:10.3390/batteries5040064.
       - Eric Prada, D. Di Domenico, Y. Creff, J. Bernard, Valérie Sauvant-Moynot, and
         François Huet. A simplified electrochemical and thermal aging model of
         LiFePO4-graphite Li-ion batteries: power and capacity fade simulations. Journal
         of The Electrochemical Society, 160(4):A616, 2013. doi:10.1149/2.053304jes.
    * Ramadass2004 :
       - P Ramadass, Bala Haran, Parthasarathy M Gomadam, Ralph White, and Branko N
         Popov. Development of first principles capacity fade model for li-ion cells.
         Journal of the Electrochemical Society, 151(2):A196, 2004.
         doi:10.1149/1.1634273.
    * Xu2019 :
       - Shanshan Xu, Kuan-Hung Chen, Neil P Dasgupta, Jason B Siegel, and Anna G
         Stefanopoulou. Evolution of dead lithium growth in lithium metal batteries:
         experimentally validated model of the apparent capacity loss. Journal of The
         Electrochemical Society, 166(14):A3456, 2019.
"""

#
# Lithium-ion
#

NCA_Kim2011 = {
    "chemistry": "lithium_ion",
    "cell": "Kim2011",
    "negative electrode": "graphite_Kim2011",
    "separator": "separator_Kim2011",
    "positive electrode": "nca_Kim2011",
    "electrolyte": "lipf6_Kim2011",
    "experiment": "1C_discharge_from_full_Kim2011",
    "sei": "example",
    "citation": "Kim2011",
}

Ecker2015 = {
    "chemistry": "lithium_ion",
    "cell": "kokam_Ecker2015",
    "negative electrode": "graphite_Ecker2015",
    "separator": "separator_Ecker2015",
    "positive electrode": "LiNiCoO2_Ecker2015",
    "electrolyte": "lipf6_Ecker2015",
    "experiment": "1C_discharge_from_full_Ecker2015",
    "sei": "example",
    "citation": [
        "Ecker2015i",
        "Ecker2015ii",
        "Zhao2018",
        "Hales2019",
        "Richardson2020",
    ],
}

Marquis2019 = {
    "chemistry": "lithium_ion",
    "cell": "kokam_Marquis2019",
    "negative electrode": "graphite_mcmb2528_Marquis2019",
    "separator": "separator_Marquis2019",
    "positive electrode": "lico2_Marquis2019",
    "electrolyte": "lipf6_Marquis2019",
    "experiment": "1C_discharge_from_full_Marquis2019",
    "sei": "example",
    "citation": "Marquis2019",
}

Chen2020 = {
    "chemistry": "lithium_ion",
    "cell": "LGM50_Chen2020",
    "negative electrode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "positive electrode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",
    "sei": "example",
    "citation": "Chen2020",
}

Chen2020_plating = {
    "chemistry": "lithium_ion",
    "cell": "LGM50_Chen2020",
    "negative electrode": "graphite_Chen2020_plating",
    "separator": "separator_Chen2020",
    "positive electrode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",
    "sei": "example",
    "lithium plating": "okane2020_Li_plating",
    "citation": "Chen2020",
}

Mohtat2020 = {
    "chemistry": "lithium_ion",
    "cell": "UMBL_Mohtat2020",
    "negative electrode": "graphite_UMBL_Mohtat2020",
    "separator": "separator_Mohtat2020",
    "positive electrode": "NMC_UMBL_Mohtat2020",
    "electrolyte": "LiPF6_Mohtat2020",
    "experiment": "1C_charge_from_empty_Mohtat2020",
    "sei": "example",
    "lithium plating": "yang2017_Li_plating",
    "citation": "Mohtat2020",
}

Ramadass2004 = {
    "chemistry": "lithium_ion",
    "cell": "sony_Ramadass2004",
    "negative electrode": "graphite_Ramadass2004",
    "separator": "separator_Ecker2015",  # no values found, relevance?
    "positive electrode": "lico2_Ramadass2004",
    "electrolyte": "lipf6_Ramadass2004",
    "experiment": "1C_discharge_from_full_Ramadass2004",
    "sei": "ramadass2004",
    "citation": "Ramadass2004",
}

Prada2013 = {
    "chemistry": "lithium_ion",
    "cell": "A123_Lain2019",
    "negative electrode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "positive electrode": "LFP_Prada2013",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "4C_discharge_from_full_Prada2013",
    "citation": ["Chen2020", "Lain2019", "Prada2013"],
}

Ai2020 = {
    "chemistry": "lithium_ion",
    "cell": "Enertech_Ai2020",
    "negative electrode": "graphite_Ai2020",
    "separator": "separator_Ai2020",
    "positive electrode": "lico2_Ai2020",
    "electrolyte": "lipf6_Enertech_Ai2020",
    "experiment": "1C_discharge_from_full_Ai2020",
    "sei": "example",
    "citation": "Ai2019",
}

Xu2019 = {
    "chemistry": "lithium_ion",
    "cell": "li_metal_Xu2019",
    "negative electrode": "li_metal_Xu2019",
    "separator": "separator_Xu2019",
    "positive electrode": "NMC532_Xu2019",
    "electrolyte": "lipf6_Valoen2005",
    "experiment": "1C_discharge_from_full_Xu2019",
    "sei": "example",
    "citation": "Xu2019",
}

ORegan2021 = {
    "chemistry": "lithium_ion",
    "cell": "LGM50_ORegan2021",
    "negative electrode": "graphite_ORegan2021",
    "separator": "separator_ORegan2021",
    "positive electrode": "nmc_ORegan2021",
    "electrolyte": "lipf6_EC_EMC_3_7_Landesfeind2019",
    "experiment": "1C_discharge_from_full_ORegan2021",
    "citation": ["ORegan2021", "Chen2020"],
}

#
# Lead-acid
#

Sulzer2019 = {
    "chemistry": "lead_acid",
    "cell": "BBOXX_Sulzer2019",
    "negative electrode": "lead_Sulzer2019",
    "separator": "agm_Sulzer2019",
    "positive electrode": "lead_dioxide_Sulzer2019",
    "electrolyte": "sulfuric_acid_Sulzer2019",
    "experiment": "1C_discharge_from_full",
    "citation": "Sulzer2019physical",
}
