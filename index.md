---
title: MAIN
layout: default
filename: index.md
--- 

CoSimPy is an open source Pyhton library optimised for Magnetic Resonance Imaging (MRI) Radiofrequency (RF) Coil design. The library aims to combine results from electromagnetic (EM) simulations with circuit analysis through a co-simulation environment.

## Summary

  - [Getting Started](#getting-started)
  - [Deployment](#deployment)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

The library has been developed with Python 3.7. and tested on previous versions down to Python 3.5.

### Prerequisites

The library uses the following additional packages:

- [numpy](https://numpy.org/) (>=1.15.2)
- [matplotlib](https://matplotlib.org/) (>=3.0.0)
- [h5py](https://www.h5py.org/) (>=2.8.0)
- [scipy](https://www.scipy.org/) (>=1.1.0)

The package versions reported in brackets represent the oldest releases with which the library has been succesfully tested.

### Installing

With [pip](https://pypi.org/project/pip/) (https://pypi.org/project/cosimpy/):
```
pip install cosimpy
```

With [anaconda](https://www.anaconda.com/products/individual):
```
conda install --channel umbertopy cosimpy
```

## Deployment

After installation, the library can be imported as:

```python
import cosimpy
```

## License

This project is licensed under the MIT
License - see the [LICENSE](LICENSE) file for
details.


[test](./test.md)

## Acknowledgments

The library has been developed in the framework of the Researcher Mobility Grant (RMG) associated with the european project 17IND01 MIMAS. This RMG: 17IND01-RMG1 MIMAS has received funding from the EMPIR programme co-financed by the Participating States and from the European Union's Horizon 2020 research and innovation programme.

[![](./docs/images/EMPIR_logo.jpg)](https://www.euramet.org/research-innovation/research-empir/)
[![](./docs/images/MIMAS_logo.png)](https://www.ptb.de/mimas/home/)
