"""
Parameter sets from papers. To see which parameter sets have been used in
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
    * Chen2020_composite :
       - Weilong Ai, Niall Kirkaldy, Yang Jiang, Gregory Offer, Huizhi Wang, and Billy
         Wu. A composite electrode model for lithium-ion batteries with silicon/graphite
         negative electrodes. Journal of Power Sources, 527:231142, 2022. URL:
         https://www.sciencedirect.com/science/article/pii/S0378775322001604,
         doi:https://doi.org/10.1016/j.jpowsour.2022.231142.
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
    * OKane2022 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Simon E. J. O'Kane, Ian D. Campbell, Mohamed W. J. Marzook, Gregory J. Offer,
         and Monica Marinescu. Physical origin of the differential voltage minimum
         associated with lithium plating in li-ion batteries. Journal of The
         Electrochemical Society, 167(9):090540, may 2020. URL:
         https://doi.org/10.1149/1945-7111/ab90ac, doi:10.1149/1945-7111/ab90ac.
       - Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diego Alonso-Alvarez,
         Robert Timms, Valentin Sulzer, Jacqueline Sophie Edge, Billy Wu, Gregory J.
         Offer, and Monica Marinescu. Lithium-ion battery degradation: how to model it.
         Phys. Chem. Chem. Phys., 24:7909-7922, 2022. URL:
         http://dx.doi.org/10.1039/D2CP00417H, doi:10.1039/D2CP00417H.
    * ORegan2022 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Kieran O'Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma Kendrick.
         Thermal-electrochemical parameters of a high energy lithium-ion cylindrical
         battery. Electrochimica Acta, 425:140700, 2022.
         doi:10.1016/j.electacta.2022.140700.
    * Prada2013 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Michael J. Lain, James Brandon, and Emma Kendrick. Design strategies for high
         power vs. high energy lithium ion cells. Batteries, 5(4):64, 2019.
         doi:10.3390/batteries5040064.
       - Andreas Nyman, Mårten Behm, and Göran Lindbergh. Electrochemical
         characterisation and modelling of the mass transport phenomena in lipf6–ec–emc
         electrolyte. Electrochimica Acta, 53(22):6356–6365, 2008.
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
       - Lars Ole Valøen and Jan N Reimers. Transport properties of lipf6-based li-ion
         battery electrolytes. Journal of The Electrochemical Society, 152(5):A882,
         2005.
       - Shanshan Xu, Kuan-Hung Chen, Neil P Dasgupta, Jason B Siegel, and Anna G
         Stefanopoulou. Evolution of dead lithium growth in lithium metal batteries:
         experimentally validated model of the apparent capacity loss. Journal of The
         Electrochemical Society, 166(14):A3456, 2019.
"""
import warnings


class ParameterSets:
    def __init__(self):
        self.all_parameter_sets = {
            "lead_acid": ["Sulzer2019"],
            "lithium_ion": [
                "Ai2020",
                "Chen2020",
                "Chen2020_composite",
                "Ecker2015",
                "Marquis2019",
                "Mohtat2020",
                "NCA_Kim2011",
                "OKane2022",
                "ORegan2022",
                "Prada2013",
                "Ramadass2004",
                "Xu2019",
            ],
        }
        self.all_parameter_sets_list = [
            *self.all_parameter_sets["lead_acid"],
            *self.all_parameter_sets["lithium_ion"],
        ]

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as error:
            # For backwards compatibility, parameter sets that used to be defined in
            # this file now return the name as a string, which will load the same
            # parameter set as before when passed to `ParameterValues`
            if name in self.all_parameter_sets_list:
                out = name
            else:
                raise error
            warnings.warn(
                f"Parameter sets should be called directly by their name ({name}),"
                f"instead of via pybamm.parameter_sets (pybamm.parameter_sets.{name}).",
                DeprecationWarning,
            )
            return out


parameter_sets = ParameterSets()
