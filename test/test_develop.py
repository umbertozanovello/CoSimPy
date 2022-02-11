#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:24:28 2022

@author: umberto
"""

import numpy as np
from cosimpy import *
import sys, os

test1 = True
test2 = True
test3 = True
test4 = True
test5 = True
test6 = True
test7 = True
test8 = True

if test1:
    def test1():
        """
        TEST 1: Check V,I computation singleLoad, 1 supply port of loaded matrix
        """
        f = np.linspace(100e6,150e6,50)
        
        S_50 = S_Matrix.sMatrixRLseries(50,0, f)
    
        S = S_Matrix.sMatrixPInetwork(None, None, S_50)
        
        rf_coil = RF_Coil(S,None)
        
        loaded_rf_coil = rf_coil.singlePortConnRFcoil([None,S_50],True)
        
        S_loaded = loaded_rf_coil.s_matrix.S[0,0,0]
        
        P_eff = 1*(1-np.abs(S_loaded)**2)
        
        v1_orig = np.sqrt(P_eff*100) #100 ohm
        v2_orig = v1_orig/2
        
        i1_orig = v1_orig/100 #100 ohm
        i2_orig=-i1_orig
        
        v1_sup = v1_orig
        i1_sup = i1_orig
        
        v_sup,i_sup,v_orig,i_orig = loaded_rf_coil.s_matrix.compVI()
        
        assert np.isclose(v1_orig,v_orig[:,0,0]).all()
        assert np.isclose(v2_orig,v_orig[:,1,0]).all()
        assert np.isclose(i1_orig,i_orig[:,0,0]).all()
        assert np.isclose(i2_orig,i_orig[:,1,0]).all()
        assert np.isclose(v1_sup,v_sup[:,0,0]).all()
        assert np.isclose(i1_sup,i_sup[:,0,0]).all()
        
        
if test2:
    def test2():
        """
        TEST 2: Check V,I fullLoad, 2 supply ports of loaded matrix
        """
        f = np.linspace(100e6,150e6,50)
        
        S_50 = S_Matrix.sMatrixRLseries(50,0, f)
    
        S = S_Matrix.sMatrixPInetwork(None, None, S_50)
        
        rf_coil = RF_Coil(S,None)
        
        s_load = np.repeat(np.array([[[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]]),50,axis=0) #Simple port extensions with cables
        S_load = S_Matrix(s_load,f)
        loaded_rf_coil = rf_coil.fullPortsConnRFcoil(S_load,True)
    
        v_sup,i_sup,v_orig,i_orig = loaded_rf_coil.s_matrix.compVI()
        
        assert np.isclose(v_sup,v_orig).all()
        assert np.isclose(i_sup,i_orig).all()
        
        
if test3:
    def test3():
        """
        TEST 3: Check V,I computation singleLoad, 1 supply port of loaded matrix, asymmetric rf_coil circuit 
        """
        f = np.linspace(100e6,150e6,50)
        
        S_50 = S_Matrix.sMatrixRLseries(50,0, f)
        
        S = S_Matrix.sMatrixPInetwork(None, S_50, S_50)
        
        rf_coil = RF_Coil(S,None)
        
        S_load = S_Matrix.sMatrixTnetwork(None, None, S_50)
        loaded_rf_coil = rf_coil.singlePortConnRFcoil([None,S_load],True)
        
        v_sup,i_sup,v_orig,i_orig = loaded_rf_coil.s_matrix.compVI()
        
        v1_sup_1 = np.sqrt(50) * (1 + loaded_rf_coil.s_matrix.S[0,0,0]) #Port 1 is supplied
        R2 = 50*25/(50+25)
        v2_sup_1 = v1_sup_1*R2/(R2+50) #Port 1 is supplied
        
        i1_sup_1 = v1_sup_1 / (50 + R2) #Port 1 is supplied
        i2_sup_1 = -1 * v2_sup_1 / 50 #Port 1 is supplied
        
        v2_sup_2 = np.sqrt(50) * (1 + loaded_rf_coil.s_matrix.S[0,1,1]) #Port 2 is supplied
        v1_sup_2 = v2_sup_2*0.5 #Port 2 is supplied
        
        i2_sup_2 = v2_sup_2 / (25*100/(25+100)) #Port 2 is supplied
        i1_sup_2 = -1 * v1_sup_2 / 50 #Port 2 is supplied
        
        assert np.isclose(v1_sup_1,v_sup[:,0,0]).all()
        assert np.isclose(v2_sup_1,v_sup[:,1,0]).all()
        assert np.isclose(v1_sup_2,v_sup[:,0,1]).all()
        assert np.isclose(v2_sup_2,v_sup[:,1,1]).all()
        assert np.isclose(i1_sup_1,i_sup[:,0,0]).all()
        assert np.isclose(i2_sup_1,i_sup[:,1,0]).all()
        assert np.isclose(i1_sup_2,i_sup[:,0,1]).all()
        assert np.isclose(i2_sup_2,i_sup[:,1,1]).all()
        
        v1_orig_1 = v1_sup_1 #Port 1 supplied
        v2_orig_1 = v2_sup_1 #Port 1 supplied
        
        i1_orig_1 = i1_sup_1 #Port 1 supplied
        i2_orig_1 = -1 * v2_orig_1 / 25 #Port 1 supplied
        
        v1_orig_2 = v1_sup_2 #Port 2 supplied
        v2_orig_2 = v2_sup_2 #Port 2 supplied
        
        i1_orig_2 = -1 * v1_orig_2 / 50 #Port 2 supplied
        i2_orig_2 = v2_orig_2 / (100*50/150) #Port 2 supplied
        
        assert np.isclose(v1_orig_1,v_orig[:,0,0]).all()
        assert np.isclose(v2_orig_1,v_orig[:,1,0]).all()
        assert np.isclose(v1_orig_2,v_orig[:,0,1]).all()
        assert np.isclose(v2_orig_2,v_orig[:,1,1]).all()
        assert np.isclose(i1_orig_1,i_orig[:,0,0]).all()
        assert np.isclose(i2_orig_1,i_orig[:,1,0]).all()
        assert np.isclose(i1_orig_2,i_orig[:,0,1]).all()
        assert np.isclose(i2_orig_2,i_orig[:,1,1]).all()
    
    
if test4:
    def test4():
        """
        TEST 4: Check power balance comparing with Sim4Life supplying all ports with 1W incident power
        """
        
        directory = os.path.join(os.path.dirname(__file__),"filesForTests")
        s_matrix = S_Matrix.importTouchstone(directory+"/S_forPowBalance.s8p", freqUnit="Hz")
        em_field = EM_Field.importFields_s4l(directory+"/FieldForPowerBalance", [123e6], 8, imp_bfield=False)
        rf_coil = RF_Coil(s_matrix, em_field)
        sup = np.ones(8)
        powBal = rf_coil.powerBalance(sup,4e-3**3,[0.6]*np.prod(em_field.nPoints),False)
        
        powFromSim4Life = 0.01026513
        
        assert np.isclose(powBal["P_dep"],powFromSim4Life)
        
        f = rf_coil.s_matrix.frequencies
        
        R = 3
        
        s_open = S_Matrix.sMatrixOpen(f)
        Sd = S_Matrix.sMatrixRCseries(0, 470e-12, f)
        St = S_Matrix.sMatrixRCseries(R, 10e-12, f)
        S_begin = S_Matrix.sMatrixRCseries(R, 5e-12, f)
        Sm = S_Matrix.sMatrixPInetwork(None,None,S_begin)
        rf_coil_loaded = rf_coil.singlePortConnRFcoil([Sm,St,s_open,Sd,Sm,St,s_open,Sd], True)
        powBud = rf_coil_loaded.powerBalance([100,100],4e-3**3,[0.6]*np.prod(em_field.nPoints),False)
                
        posFlag = True
        tol = -1e-10 #It should be <0 but I use -1e-10 for tollerance tollerance
        for value in powBud.values():
            if isinstance(value,list) and value[0] < tol: 
                posFlag = False
            elif value < tol:
                posFlag = False
        assert posFlag
    
    
if test5:
    def test5():
        """
        TEST 5: Check power balance with analytic computation considering the following circuit supplied with 1 W power at all ports
        
             1             3
        ---Rext---R------Rext----
                      |
                      |   2
                      R
                      |
                      |
        -------------------------
        
        Rext are resistances of external circuitry = 50 ohm
        R are resistance of the RF coil = 50 ohm
        """
        
        f = np.linspace(100e6,150e6,50)
        s50 = S_Matrix.sMatrixRLseries(50, 0, f)
        sCoil = S_Matrix.sMatrixPInetwork(None,s50,s50)
        sR = S_Matrix.sMatrixRLseries(50,0,f)
        sR_conn = S_Matrix.sMatrixPInetwork(None,None,sR)
        rf_coil = RF_Coil(sCoil,None)
        rf_coil_loaded = rf_coil.singlePortConnRFcoil([sR_conn,sR_conn],True)
        powBud_1 = rf_coil_loaded.powerBalance([0.5,0])
        powBud_2 = rf_coil_loaded.powerBalance([0,1j])
        powBud_3 = rf_coil_loaded.powerBalance([0.5,1j])
        
        #Port 1 supplied
        
        R_1 = 100 * 50/150 + 100 #Equivalent resistance at port 1
        
        s_1 = (R_1 - 50) / (R_1 + 50)
        
        v_1 = np.sqrt(50*0.5) * (1+s_1)
        
        P_1_1 = v_1**2/R_1 #Power delivered by generator at port 1 into the system and into the port 2 impedance
        
        I_1_1 = v_1 / R_1
        I_1_3 = I_1_1 * 50 / 150
        
        P_1_2 = I_1_3**2 * 50 #Power delivered to the impedance of port 2
        
        P_1_syst = P_1_1 - P_1_2 #Effective power into the system
        
        P_1_circ = I_1_1**2 * 50 + I_1_3**2*50 #Power dissipated into circuitry
        
        assert np.isclose(powBud_1["P_inc_tot"] - powBud_1["P_refl"],P_1_syst).all()
        assert np.isclose(powBud_1["P_circ_loss"],P_1_circ).all()
        
        #Port 2 supplied
    
        R_2 = 150 * 50/200 + 50 #Equivalent resistance at port 1
        
        s_2 = (R_2 - 50) / (R_2 + 50)
        
        v_2 = np.sqrt(50)*1j * (1+s_2)
        
        P_2_2 = v_2*np.conj(v_2)/R_2 #Power delivered by generator at port 2 into the system and into the port 1 impedance
        
        I_2_3 = -v_2 / R_2
        I_2_1 = I_2_3 * 50 / 200
        
        P_2_1 = I_2_1*np.conj(I_2_1) * 50 #Power delivered to the impedance of port 2
        
        P_2_syst = P_2_2 - P_2_1 #Effective power into the system
        
        P_2_circ = I_2_3*np.conj(I_2_3) * 50 + I_2_1*np.conj(I_2_1)*50 #Power dissipated into circuitry
        
        assert np.isclose(powBud_2["P_inc_tot"] - powBud_2["P_refl"],P_2_syst).all()
        assert np.isclose(powBud_2["P_circ_loss"],P_2_circ).all()
        
        #Both ports supplied
    
        I_1_2 = I_1_3 - I_1_1
        I_2_2 = I_2_3 - I_2_1
        
        I_1 = I_1_1 + I_2_1
        I_2 = I_1_2 + I_2_2
        I_3 = I_1_3 + I_2_3
        
        P_tot = 100*I_1*np.conj(I_1) + 50*I_2*np.conj(I_2) + 50*I_3*np.conj(I_3)
        P_circ = 50*I_1*np.conj(I_1) + 50*I_3*np.conj(I_3)
        
        assert np.isclose(powBud_3["P_inc_tot"] - powBud_3["P_refl"],P_tot).all()
        assert np.isclose(powBud_3["P_circ_loss"],P_circ).all()
        
        
if test6:
    def test6():
        """
        TEST 6: Saving and loading
        """
        
        directory = os.path.join(os.path.dirname(__file__),"filesForTests")
        
        s_matrix = S_Matrix.importTouchstone(directory + "/S_forSaveAndLoading.s4p")
        em_field = EM_Field.importFields_s4l(directory + "/FieldForSaveAndLoading",[123e6],4,pkORrms='pk',imp_bfield=True, conductivity=[0.6]*382500,conductivity2=[0.6]*382500)

        orig_rf_coil = RF_Coil(s_matrix,em_field)
        S_open = S_Matrix(np.ones([1,1,1]),[123e6])
        rf_coil = orig_rf_coil.singlePortConnRFcoil([None,S_open,None,S_open],True)
        
        rf_coil.saveRFCoil(directory+"//saveload_test", "Prova\n\n\n")
        loaded_rf_coil = RF_Coil.loadRFCoil(directory+"//saveload_test.cspy")
        
        assert np.allclose(rf_coil.em_field.nPoints, loaded_rf_coil.em_field.nPoints, equal_nan=True)
        assert np.allclose(rf_coil.em_field.frequencies, loaded_rf_coil.em_field.frequencies, equal_nan=True)
        assert np.allclose(rf_coil.em_field.b_field, loaded_rf_coil.em_field.b_field, equal_nan=True)
        assert np.allclose(rf_coil.em_field.e_field, loaded_rf_coil.em_field.e_field, equal_nan=True)
        
        passed = True
        try:
            for prop in rf_coil.em_field.properties:
                if not np.allclose(rf_coil.em_field.properties[prop], loaded_rf_coil.em_field.properties[prop]):
                    passed = False
        except:
            passed = False
            
        assert passed
        
        assert np.allclose(rf_coil.s_matrix.frequencies, loaded_rf_coil.s_matrix.frequencies, equal_nan=True)
        assert np.allclose(rf_coil.s_matrix.S, loaded_rf_coil.s_matrix.S, equal_nan=True)
        
        orig_S0 = rf_coil.s_matrix._S0
        load_S0 = loaded_rf_coil.s_matrix._S0
        
        assert np.allclose(orig_S0.S, load_S0.S, equal_nan=True)
        assert np.allclose(orig_S0._p_incM, load_S0._p_incM, equal_nan=True)
        assert np.allclose(orig_S0._phaseM, load_S0._phaseM, equal_nan=True)
        
        
if test7:
    def test7():
        """
        TEST 7: test connectPorts method
        """

        f = np.linspace(100e6,150e6,50)
        
        sY = S_Matrix.sMatrixDoubleY(f)
        
        sMatConn = S_Matrix(np.repeat(np.array([[[0,1],[1,0]]]),50,axis=0),f)
        
        rf_coil = RF_Coil(sY,None)
        
        res = rf_coil.connectPorts([[2,3]], [sMatConn])
        
        assert np.isclose(res.s_matrix.S,1).all()
    
if test8:
    def test8():
        """
        TEST 8: Test multiple connections randomly
        """
        
        directory = os.path.join(os.path.dirname(__file__),"filesForTests")
        s_matrix = S_Matrix.importTouchstone(directory + "/S_forMultConn.s48p")
        
        rf_coil = RF_Coil(s_matrix)
        
        # First connections
        port_list = np.arange(3,51,3).reshape([8,2])
        s_tl = S_Matrix.sMatrixTrLine(20e-2, s_matrix.frequencies)
        res = rf_coil.connectPorts(port_list, 8*[s_tl],True)
        
        # Second connections
        s_cap = S_Matrix.sMatrixRCseries(0,10e-12,s_matrix.frequencies)
        s_pi_cap = S_Matrix.sMatrixPInetwork(None, None, s_cap)
        # res.s_matrix.setAsOriginal()
        res = res.singlePortConnRFcoil(32*[s_pi_cap],True)
        
        sups = np.eye(32)
        for sup in sups:
            p_bud = res.powerBalance(sup)
            assert np.isclose(p_bud["P_circ_loss"],0)
