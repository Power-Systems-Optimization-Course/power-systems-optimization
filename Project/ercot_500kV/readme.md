ERCOT 120-bus 500kV simulated system
-

## Description and usage

A complete set of generator parameters, buses, branches, loads, and renewable profiles for the year 2016 in a simulated system covering the Electric Reliability Council of Texas (ERCOT).

Suitable for:
- OPF
- Economic dispatch

## Data sources

The 2016 snapshot of the ERCOT grid is taken from the [U.S. Test System with High Spatial and Temporal Resolution for Renewable Integration Studies](https://zenodo.org/record/3530899#.X6TQ85NKjUK) by Xu et al. The paper describing the methodology: https://arxiv.org/abs/2002.06155

These data are largely adapted from the [ACTIVSg2000 dataset](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg2000/), a 2000-bus synthetic grid on the footprint of Texas in the Electric Grid Test Case Repository published by Texas A&M University. Some changes made with respect to the TAMU system are described in the above paper. MATPOWER and other file formats of the TAMU data are available at the above link. Load profiles for 2016 were taken directly from the TAMU dataset.

To reduce dimensionality, all buses (`bus_id`, 2000 in total) were aggregated to the closest 500kV bus (`bus_agg`, 120 in total) according to the number of branches separating the low and high-voltage buses.

Wind and solar profiles for 2016 were downloaded from [Renewables.ninja](https://www.renewables.ninja/) at the latitude and longitude of a substation connected to the aggregate bus with the largest installed wind and solar capacity, respectively, in each zone. These are stored according to the [zone_id](zone.csv). (Smaller zones with limited wind or solar may not have a profile. In these cases, simply use the nearby zone.)


## License and usage

**U.S. Test System with High Spatial and Temporal Resolution for Renewable Integration Studies**
- Citation: https://arxiv.org/abs/2002.06155
- [Creative Commons 4.0 Attribution International](https://creativecommons.org/licenses/by/4.0/)

**ACTIVSg2000 dataset**
- See the [list of references](https://electricgrids.engr.tamu.edu/references/) describing the methodology, and cite accordingly

**Renewables.ninja**
- [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
