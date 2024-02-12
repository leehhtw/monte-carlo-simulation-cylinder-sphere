# Realistic Microstructure Simulator (RMS): Monte Carlo simulations of diffusion in randomly packed cylinders or spheres of permeable membrane (CUDA C++)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10651812.svg)](https://doi.org/10.5281/zenodo.10651812)

The code implements 3d Monte Carlo simulations originally developed in [Lee, et al., Journal of Neuroscience Methods, 2021](https://doi.org/10.1016/j.jneumeth.2020.109018), demonstrating the effect of water exchange and diffusion in-between randomly packed cylindrs or spheres on the time-dependent kurtosis in [Wu et al., Science Advances 2024 (under revision)]().

* **Demo 1.1, cylinder:** We randomly pack parallel cylinders and perform diffusion simulation in-between cylinders of permeable membrane.
* **Demo 1.2, cylinder:** We analyze the simulation result in Demo 1.1. To simulate the orientation dispersion, we sample 1,000 fiber directions for each simulation based on a Watson distribution in z-axis. Their apparent diffusivity and kurtosis transverse to the main axis are combined based on the formula in [Lee, et al., Communications Biology 2020](https://doi.org/10.1038/s42003-020-1050-x).
* **Demo 2.1, cylinder:** We randomly pack spheres and perform diffusion simulation in-between spheres of permeable membrane.
* **Demo 2.2, cylinder:** We analyze the simulation result in Demo 2.1.
* 
## References
* **Monte Carlo simulation**
  - [Fieremans, et al., NMR Biomed, 2010](https://doi.org/10.1002/nbm.1577)
  - [Novikov, et al., Nature Physics, 2011](https://doi.org/10.1038/nphys1936)
  - [Fieremans and Lee, NeuroImage 2018](https://doi.org/10.1016/j.neuroimage.2018.06.046)
  - [Lee, et al., Communications Biology 2020](https://doi.org/10.1038/s42003-020-1050-x)
  - [Lee, et al., NeuroImage 2020](https://doi.org/10.1016/j.neuroimage.2020.117228)
  - [Lee, et al., Journal of Neuroscience Methods 2021](https://doi.org/10.1016/j.jneumeth.2020.109018)
  - [Lee, et al., NMR in Biomedicine 2024](https://doi.org/10.1002/nbm.5087)

* **Random packing generation**
  - [Donev, et al., J Comput Phys, 2005](https://doi.org/10.1016/j.jcp.2004.08.014) and [code](https://cims.nyu.edu/~donev/Packing/C++/)

* **Watson Distribution sampling**
  - [SphericalDistributionsRand](https://www.mathworks.com/matlabcentral/fileexchange/52398-sphericaldistributionsrand)

## Authors
* [Hong-Hsi Lee](http://www.diffusion-mri.com/people/hong-hsi-lee)
* [Dmitry S Novikov](http://www.diffusion-mri.com/people/dmitry-novikov)
* [Els Fieremans](http://www.diffusion-mri.com/people/els-fieremans)

## Acknowledgement
We would like to thank [Ricardo Coronado-Leija](https://scholar.google.com/citations?user=V5hykxgAAAAJ&hl=en) for the fruitful discussion of simulation implementation.

## License
This project is licensed under the [LICENSE](https://github.com/leehhtw/monte-carlo-simulation-cylinder-sphere/blob/main/LICENSE).
 
