---
title: news_versions
layout: default
filename: News&Versions.md
--- 

# News

Version 1.2.0 is now available on [GitHub](https://github.com/umbertozanovello/CoSimPy/tree/main) and [PyPi](https://pypi.org/project/cosimpy/) pages!<br><br>
The following improvements have been implemented:
* Added three properties to S_Matrix class allowing to store incident powers and original S matrix prior to a connection with external circuitry;
* Added a method for voltages and currents computation both at the supply ports of the connected and unconnected RF Coil;
* Added a method for power balance computation;
* Added a method for power loss density computation
* Added a method for power deposition computation
* The S_Matrix method `sMatrixTrLine` now allows for transmission lines with a characteristic impedance different from the port impedances

# Previous Versions

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
