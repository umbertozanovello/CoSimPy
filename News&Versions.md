---
title: news_versions
layout: default
filename: News&Versions.md
--- 

# News

Version 1.3.2 is now available on [GitHub](https://github.com/umbertozanovello/CoSimPy/tree/main) and [PyPi](https://pypi.org/project/cosimpy/) pages!<br><br>
A bug in `compSensitivities` method of the EM_Field class has been fixed. 

---

# Previous Versions

- v 1.3.1 (Feb 9, 2022):
  * Solved a bug in the `saveRFCoil` method when the RF coil is saved without description;
  * Edited the status print of the saveRFCoil and loadRFCoil methods;
  * Edited the status print of the importFields methods;
  * Solved a bug in the `importFields_cst` method when only the b_field is imported.

- v 1.3.0 (Jan 28, 2021):
  * A new `__sub__` method in the S_Matrix class allows to perform the cascade connection between the last port of a first S_Matrix instance and the first port of  a second just as: `S_res = S_1 - S_2`;
  * The `importTouchstone` method of the SMatrix class has been improved to fully support the v1.1. Touchstone® File Format Specification by IBIS. It will be possible to import also Z- and Y-parameters.  Furthermore, providing to the method the relevant information through a dictionary, the same method will also be  able to import a simple ascii file formatted by columns;
  * An `exportTouchstone` method will be available in the S_Matrix class to export the S-, Z- or Y-parameters either according to the v1.1. Touchstone® File Format Specification by IBIS or to a columns formatted ascii file;
  * Two new methods in the S_Matrix class (`sMatrixOpen`, `sMatrixShort`) will allow to generate an instance of a 1-port open or 1-port short over defined frequency values;
  * The existing methods to generate T and PI circuits accept `None` arguments for longitudinal and transversal parameters respectively, to substitute them with short and open circuits respectively;
  * The original S matrix, S0, is saved in way allowing for the external circuitry losses computation accounting for all the consecutive connections to the S0 matrix;
  * A new method is available in the S_Matrix class (`setAsOriginal`) to set the S_Matrix instance resulting from the connections with external circutries as unconnected. In this way, during powerBalance, the connected circuitry losses will be assigned to the source “other”;
  * A new method of the S_Matrix class (`plotSPanel`) allows to plot, in a single multipanel figure, the S parameters of a multiport device in dB;
  * The importFields_cst method of the EM_Field class is improved to support the import of the electromagnetic field exported by CST as .h5 binary file;
  * Two new methods of the RF_Coil class (`saveRFCoil`, `loadRFCoil`) allow to save and load the RF_Coil instance in proprietary binary files;
  * New methods are available in the RF_Coil class (`duplicatePortsParallel`, `duplicatePortsSeries`) to duplicate existing ports of a simulated device;
  * A new method is available in the RF_Coil class (`connectPorts`) to connect together two or more port pairs, through given circuitries, whose S matrices are properly given by the user.

- v 1.2.0 (Sep 6, 2021):
  * Added three properties to S_Matrix class allowing to store incident powers and original S matrix prior to a connection with external circuitry;
  * Added a method for voltages and currents computation both at the supply ports of the connected and unconnected RF Coil;
  * Added a method for power balance computation;
  * Added a method for power loss density computation
  * Added a method for power deposition computation
  * The S_Matrix method `sMatrixTrLine` now allows for transmission lines with a characteristic impedance different from the port impedances

- v 1.1.1 (May 26, 2021):
  * Solved issues with complex port impedances;
  * Now an error is raised if port impedances have a negative real part.
  
- v 1.1.0 (Apr 22, 2021):
  * Modified the management of the matrix singularities in `S_Matrix.__resSMatrix method`;
  * The `warnings` library is now imported in the S_Matrix module to report non-fatal alerts to the user;
  * An `info_warnings=True` flag can be passed as argument to `S_Matrix.init` and `S_Matrix.importTouchstone` methods to allow for the print of additional information related to warnings;
  * Improved the `S_Matrix.importTouchstone` method;
  * Solved a bug in `EM_Field.importFields_s4l` method;
  * Modified the default value of the parameter 'pkORrms' in `EM_Field importFields_s4l` method.

- v 1.0.0 (Feb 1, 2021): First release
