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
       - Giles Richardson, Ivan Korotkin, Rahifa Ranom, Michael Castle, and Jamie M.
         Foster. Generalised single particle models for high-rate operation of graded
         lithium-ion electrodes: systematic derivation and validation. Electrochimica
         Acta, 339:135862, 2020. doi:10.1016/j.electacta.2020.135862.
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
    * Yang2017 :
       - Madeleine Ecker, Thi Kim Dung Tran, Philipp Dechent, Stefan Käbitz, Alexander
         Warnecke, and Dirk Uwe Sauer. Parameterization of a Physico-Chemical Model of a
         Lithium-Ion Battery: I. Determination of Parameters. Journal of the
         Electrochemical Society, 162(9):A1836–A1848, 2015. doi:10.1149/2.0551509jes.
       - Xiao Guang Yang, Yongjun Leng, Guangsheng Zhang, Shanhai Ge, and Chao Yang
         Wang. Modeling of lithium plating induced aging of lithium-ion batteries:
         transition from linear to nonlinear aging. Journal of Power Sources, 360:28–40,
         2017. doi:10.1016/j.jpowsour.2017.05.110.
"""

#
# Lithium-ion
#

NCA_Kim2011 = {
    "chemistry": "lithium-ion",
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
    "chemistry": "lithium-ion",
    "cell": "kokam_Ecker2015",
    "negative electrode": "graphite_Ecker2015",
    "separator": "separator_Ecker2015",
    "positive electrode": "LiNiCoO2_Ecker2015",
    "electrolyte": "lipf6_Ecker2015",
    "experiment": "1C_discharge_from_full_Ecker2015",
    "sei": "example",
    "citation": ["Ecker2015i", "Ecker2015ii", "Richardson2020"],
}

Yang2017 = {
    "chemistry": "lithium-ion",
    "cell": "Yang2017",
    "negative electrode": "graphite_Yang2017",
    "separator": "separator_Yang2017",
    "positive electrode": "nmc_Yang2017",
    "electrolyte": "lipf6_Ecker2015",
    "experiment": "1C_discharge_from_full_Ecker2015",
    "sei": "yang2017_sei",
    "lithium plating": "yang2017_Li_plating",
    "citation": ["Yang2017", "Ecker2015i"],
}


Marquis2019 = {
    "chemistry": "lithium-ion",
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
    "chemistry": "lithium-ion",
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
    "chemistry": "lithium-ion",
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
    "chemistry": "lithium-ion",
    "cell": "UMBL_Mohtat2020",
    "negative electrode": "graphite_UMBL_Mohtat2020",
    "separator": "separator_Mohtat2020",
    "positive electrode": "NMC_UMBL_Mohtat2020",
    "electrolyte": "LiPF6_Mohtat2020",
    "experiment": "1C_charge_from_empty_Mohtat2020",
    "sei": "example",
    "citation": "Mohtat2020",
}

Ramadass2004 = {
    "chemistry": "lithium-ion",
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
    "chemistry": "lithium-ion",
    "cell": "A123_Lain2019",
    "negative electrode": "graphite_Chen2020",
    "separator": "separator_Chen2020",
    "positive electrode": "LFP_Prada2013",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "4C_discharge_from_full_Prada2013",
    "citation": ["Chen2020", "Lain2019", "Prada2013"],
}

Ai2020 = {
    "chemistry": "lithium-ion",
    "cell": "Enertech_Ai2020",
    "negative electrode": "graphite_Ai2020",
    "separator": "separator_Ai2020",
    "positive electrode": "lico2_Ai2020",
    "electrolyte": "lipf6_Enertech_Ai2020",
    "experiment": "1C_discharge_from_full_Ai2020",
    "sei": "example",
    "citation": "Ai2019",
}

#
# Lead-acid
#

Sulzer2019 = {
    "chemistry": "lead-acid",
    "cell": "BBOXX_Sulzer2019",
    "negative electrode": "lead_Sulzer2019",
    "separator": "agm_Sulzer2019",
    "positive electrode": "lead_dioxide_Sulzer2019",
    "electrolyte": "sulfuric_acid_Sulzer2019",
    "experiment": "1C_discharge_from_full",
    "citation": "Sulzer2019physical",
}
