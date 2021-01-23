# Nickel Cobalt Aluminium (NCA) positive electrode parameters

Parameters for an NCA positive electrode, from the paper

> Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A. (2011). Multi-domain modeling of lithium-ion batteries encompassing multi-physics in varied length scales. Journal of The Electrochemical Society, 158(8), A955-A969.

Note, only an effective cell volumetric heat capacity is provided in the paper. We therefore used the values for the density and specific heat capacity reported in the Marquis2019 parameter set in each region and multiplied each density by the ratio of the volumetric heat capacity provided in smith to the calculated value. This ensures that the values produce the same effective cell volumetric heat capacity. This works fine for thermal models that are averaged over the x-direction but not for full (PDE in x direction) thermal models. We do the same for the planar effective thermal conductivity.
