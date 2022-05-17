# -*- coding: utf-8 -*-

from cosimpy import *
import numpy as np

L_coil = 300e-9 #Coil inductance
R_coil = 5 #Coil resistance

frequencies = np.linspace(50e6,250e6,1001) #Frequency values at which S parameters are evaluated
nPoints = [20,20,20] #Number of points along x-, y-, z-direction where the magnetic flux density field is evaluated

b_field = np.zeros((1,1,3,np.prod(nPoints))) #b_field is evaluated at one frequency valueand at one port
b_field[:,:,1,:] = 0.1e-6 #Only the y-component is different from zero 

s_coil = S_Matrix.sMatrixRLseries(R_coil,L_coil,frequencies) #S_Matrix instance
em_coil = EM_Field([128e6], nPoints, b_field) #EM_Field instance

rf_coil = RF_Coil(s_coil,em_coil) #RF_Coil instance

tr_line = S_Matrix.sMatrixTrLine(5e-2,frequencies) #5 cm, 50 ohm, lossless transmission line

rf_coil_line = rf_coil.singlePortConnRFcoil([tr_line],True) #Connection of the RF coil to the transmission line

rf_coil_line.s_matrix[128e6].getZMatrix()

#######################

R1 = np.real(rf_coil_line.s_matrix[128e6].getZMatrix())
XL = np.imag(rf_coil_line.s_matrix[128e6].getZMatrix())
Ca = 1/(2*np.pi*128e6*(XL-np.sqrt(50*R1)))
Cb = 1/(2*np.pi*128e6*np.sqrt(50*R1))
L = np.sqrt(50*R1)/(2*np.pi*128e6)

########################

S_Ca = S_Matrix.sMatrixRCseries(0,Ca,frequencies)
S_Cb = S_Matrix.sMatrixRCseries(0,Cb,frequencies)
S_L = S_Matrix.sMatrixRLseries(0,L,frequencies)

match_network = S_Matrix.sMatrixTnetwork(S_Ca,S_L,S_Cb)

rf_coil_line_matched = rf_coil_line.singlePortConnRFcoil([match_network], True) #The RF coil is connected to the matching network

rf_coil_line_matched.s_matrix.plotS(["S1-1"])
