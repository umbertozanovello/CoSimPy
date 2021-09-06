# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import NoConvergence
from functools import partial
from copy import copy
import warnings

def warning_format(message, category, filename, lineno, file=None, line=None):
    return '\n%s: Line %s - WARNING - %s\n' % (filename.split("/")[-1], lineno, message)
warnings.formatwarning = warning_format

eps = 1e-10

class S_Matrix():

    def __init__(self, S, freqs, z0=50, **kwarg):
        
        self.__kwarg = kwarg
        if "info_warnings" in self.__kwarg.keys() and self.__kwarg["info_warnings"]==True:
            self.__info_warnings = True
        else:
            self.__info_warnings = False
            
        if not isinstance(S, np.ndarray): 
            raise TypeError("S matrix can only be numpy ndarray")
        elif not isinstance(freqs, np.ndarray) and not isinstance(freqs, list): 
             raise TypeError("Frequencies can only be an Nf list")
        elif (isinstance(z0, np.ndarray) or isinstance(z0, list)) and len(z0) != S.shape[1]:
            raise TypeError("The port impedances list is not compatible with the S matrix ports number")
        elif len(S.shape) != 3:
             raise TypeError("S matrix can only be an Nf x Np x Np matrix (also 1 x 1 x 1 is accepted in case of single load at single frequency")
        elif S.shape[1] != S.shape[2]:
             raise TypeError("S matrix can only be a Nf square matrices")
        elif len(freqs) != S.shape[0]:
             raise TypeError("The frequencies list is not compatible with the S matrix first dimension")
        elif (np.round(np.abs(S),6) > 1).any():
            warnings.warn("An S parameter higher than one has been found. Results could be unphysical at least at one frequency value. Healing the S matrix with the healSMatrix method could solve the problem")
            if self.__info_warnings:
                print("\nMax |S_ij|:\n")
                print(np.max(np.abs(S)))
        
        # Check for positive definiteness of II - (S^H)(S)
        p = np.eye(S.shape[1]) - S @ np.conjugate(np.transpose(S,axes=[0,2,1]))
        not_nan_idxs = np.where(np.logical_not(np.isnan(p).any(axis=(1,2))))[0] #idx of new_P first dimension where no nan values along the other two dimensions are encountered
        if (np.round(np.real(np.linalg.eigvals(p[not_nan_idxs])),6) < 0).any():
            warnings.warn("The S matrix seems to be unphysical at least at one frequency value. Healing the S matrix with the healSMatrix method could solve the problem")
            if self.__info_warnings:
                print("\nEigenvalues of II - S^H @ S:\n")
                print(np.real(np.linalg.eigvals(np.eye(S.shape[1]) - S @ np.conjugate(np.transpose(S,axes=[0,2,1])))))


        self.__S = np.array(S,dtype = "complex")
        self.__f = np.array(freqs) #Hz
        self.__n_f = len(freqs)
        self.__nPorts = S.shape[-1]
        self.__S0 = None
        self.__p_incM = None
        self.__phaseM = None
        
        
        if not isinstance(z0, np.ndarray) and not isinstance(z0, list):
            self.__z0 = z0 * np.ones(self.__nPorts)
        else:
            self.__z0 = np.array(z0)
        
        if (np.real(self.__z0) <= 0).any():
            raise ValueError("The real part of all the port impedances has to be higher than zero")
        elif (np.real(self.__z0) != self.__z0).any():
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            self.__z0 = np.real(self.__z0)
                
    
    
    @property
    def S(self):
        return self.__S
    
    
    @property
    def frequencies(self):
        return self.__f
    
    
    @property
    def n_f(self):
        return self.__n_f
    
    
    @property
    def nPorts(self):
        return self.__nPorts
    
    @property
    def z0(self):
        return self.__z0 
        
    
    def __repr__(self):
        string = '"""""""""""""""\n   S MATRIX\n"""""""""""""""\n\n'
        string += "|V-| = |S||V+|\n|%d x 1| = |%d x %d||%d x 1|\n\nNumber of frequency values = %d\n"%(self.__nPorts,self.__nPorts,self.__nPorts,self.__nPorts, self.__n_f)
        return string
    
    
    def __getitem__(self, key):
        
        if isinstance(key,int) or isinstance(key,float):
            idx = self.__findFreqIndex(key)
            ret_S = np.expand_dims(self.__S[idx], axis = 0)
            ret_frequencies = [self.__f[idx]]
            
        elif isinstance(key,tuple) or isinstance(key,list) or isinstance(key,np.ndarray):
            if isinstance(key,np.ndarray) and len(key.shape) > 1:
                raise SyntaxError
            idxs = list(map(self.__findFreqIndex, key))
            ret_S = self.__S[idxs]
            ret_frequencies = self.__f[idxs]
            
        elif isinstance(key,slice):
            if key.start is None:
                idx0 = 0
            else:
                idx0 = self.__findFreqIndex(key.start)
            if key.stop is None:
                idx1 = self.__n_f
            else:
                idx1 = self.__findFreqIndex(key.stop)
            
            ret_S = self.__S[idx0:idx1]
            ret_frequencies = self.__f[idx0:idx1]
            
        return S_Matrix(ret_S, ret_frequencies)
    
    
    def __add__(self, other):
        
        if not isinstance(other, S_Matrix): 
            raise TypeError("Series operation can be performed only between two 1-port S_Matrix instances")
        elif self.__nPorts != 1 or other.__nPorts != 1:
            raise TypeError("Series operation can be performed only between two 1-port S_Matrix instances")
        elif not np.array_equal(self.__f, other.__f):
            raise ValueError("The S_Matrix instances must be defined over the same frequency values")
        elif not np.array_equal(self.__z0, other.__z0):
            raise ValueError("The port impedances of the S_Matrix instances must be the same")
            
        z0 = self.__z0[0]
        
        z_s = z0 * (1 + self.__S) / (1 - self.__S)
        z_o = z0 * (1 + other.__S) / (1 - other.__S)
        
        z_r = z_s + z_o
        s_r = (z_r - z0) / (z_r + z0)
        
        return S_Matrix(s_r, self.__f, z0)
    
    
    def __mul__(self, other):
        
        if not isinstance(other, S_Matrix): 
            raise TypeError("Parallel operation can be performed only between two 1-port S_Matrix instances")
        elif self.__nPorts != 1 or other.__nPorts != 1:
            raise TypeError("Parallel operation can be performed only between two 1-port S_Matrix instances")
        elif not np.array_equal(self.__f, other.__f):
            raise ValueError("The S_Matrix instances must be defined over the same frequency values")
        elif not np.array_equal(self.__z0, other.__z0):
            raise ValueError("The port impedances of the S_Matrix instances must be the same")
            
        z0 = self.__z0[0]
        
        z_s = z0 * (1 + self.__S) / (1 - self.__S)
        z_o = z0 * (1 + other.__S) / (1 - other.__S)
        
        z_r = (z_s * z_o) / (z_s + z_o)
        s_r = (z_r - z0) / (z_r + z0)
        
        return S_Matrix(s_r, self.__f, z0)
               
    
    def plotS(self, parameters, dB=True, smooth=True):
        
        if not isinstance(parameters,list) and not isinstance(parameters, np.ndarray):
            raise TypeError("Parameters should be list or numpy array")

        eligibleParams = []
        smooth_limit = 1000 #If self.__n_f > smooth_limit it does not perform smooth
        
        for r in range(self.__S.shape[1]):
            for c in range(self.__S.shape[2]):
                eligibleParams.append('S%d-%d'%(r+1,c+1))
        for param in parameters:
            if not param in eligibleParams:
                raise TypeError("One of the parameters is not compliant with the S matrix")
        
        if self.n_f <= 2:
            smooth = False
        elif self.n_f == 3:
            interpOrder = 2
        else:
            interpOrder = 3
        
        if dB:
            plt.figure()
            plt.title("S parameters")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Magnitude (dB)")
            
            for param in parameters:
                i = int(param[1:].split("-")[0]) - 1
                j = int(param[1:].split("-")[1]) - 1
                
                if not smooth or self.__n_f > smooth_limit:
                    x = self.__f
                    y = self.S[:,i,j]
                else:
                    x = np.linspace(self.__f[0], self.__f[-1], 1000)
                    spl = make_interp_spline(self.__f, self.__S[:,i,j], k=interpOrder)
                    y = spl(x)
                
                #if y = 0 I set -50 dB
                dB_values = -50 * np.ones_like(y, dtype=np.float)
                dB_values[y!=0] = 20*np.log10(np.abs(y[y!=0]))

                if self.__n_f == 1:
                    plt.scatter(x*1e-6, dB_values, label=param)
                else:
                    plt.plot(x*1e-6, dB_values, label=param)
            
            plt.legend()
        
        else:
            fig, axs = plt.subplots(2,1)
            fig.suptitle("S parameters")
            axs[0].set_ylabel("Magnitude")
            axs[0].set_xlabel("Frequency (MHz)")
            axs[1].set_ylabel("Phase (rad)")
            axs[1].set_xlabel("Frequency (MHz)")
            
            
            for param in parameters:
                i = int(param[1:].split("-")[0]) - 1
                j = int(param[1:].split("-")[1]) - 1
                
                if not smooth or self.__n_f > smooth_limit:
                    x = self.__f
                    y = self.__S[:,i,j]
                else:
                    x = np.linspace(self.__f[0], self.__f[-1], 1000)
                    spl = make_interp_spline(self.__f, self.__S[:,i,j], k=interpOrder)
                    y = spl(x)
                
                if self.__n_f == 1:
                    axs[0].scatter(x*1e-6, np.abs(y), label=param)
                    axs[1].scatter(x*1e-6, np.angle(y), label=param)                            
                else:
                    axs[0].plot(x*1e-6, np.abs(y), label=param)
                    axs[1].plot(x*1e-6, np.angle(y), label=param)                            
                
            axs[0].legend()
            axs[1].legend()
            
            
    def getZMatrix(self):
        
        z0 = np.diag(self.__z0)
        
        S = np.copy(self.__S)
        
        z0sqrt = np.sqrt(z0)
        
        M = np.eye(self.__nPorts) - S
        
        det_M = np.linalg.det(M)
        
        if (det_M == 0).any():
            warnings.warn("The Z matrix cannot be computed at least at one frequency value. NaN is returned at those frequencies")
        
        M[det_M == 0] = None

        Z = z0sqrt @ (np.eye(self.__nPorts) + S) @ np.linalg.inv(M) @ z0sqrt
            
        return Z
    
    
    def getYMatrix(self):
        
        y0 = np.diag(1/self.__z0)
        
        S = np.copy(self.__S)

        y0sqrt = np.sqrt(y0)
        
        M = np.eye(self.__nPorts) + S
        
        det_M = np.linalg.det(M)
        
        if (det_M == 0).any():
            warnings.warn("The Y matrix cannot be computed at least at one frequency value. NaN is returned at those frequencies")
        
        M[det_M == 0] = None
        
        Y = y0sqrt @ np.linalg.inv(M) @ (np.eye(self.__nPorts) - S) @ y0sqrt
            
        return Y  
    

    def compVI(self):
            
        #Voltages and currents relevant to the S matrix supply ports
        vp_sup = np.diag(np.sqrt(self.__z0))
        vp_sup = np.repeat(np.expand_dims(vp_sup,axis=0),self.__n_f,axis=0) #vp_sup.shape = [self.__n_f, self.nPorts, self.nPorts]
        
        vm_sup = self.__S @ vp_sup
        
        v_sup= vp_sup + vm_sup
        
        y = self.getYMatrix()
        
        if (np.isnan(y)).all(axis=(1,2)).any(): #The Y matrix has not been computed at least at one frequency value
            warnings.warn("The currents cannot be computed at least at one frequency value. NaN is returned at those frequencies")
        
        i_sup = y @ v_sup
        
        if self.__S0 is None:
            
            print("No previous connections to external circuitries or connection data have not been saved. Voltages and currents are computed only at supply ports")
            return v_sup, i_sup
        
        else:
            
            #Voltages and currents relevant to the S matrix ports before the last connection
            vp0 = np.sqrt(self.__S0.__p_incM*self.__S0.__z0) * np.exp(self.__S0.__phaseM*1j) #vp.shape = [self.__f, self.__nPorts, self.__S0.__nPorts]
            vp0 = np.moveaxis(vp0,-1,1) # vp.shape = [self.__f, self.__S0.__nPorts, self.__nPorts]
    
            vm0 = self.__S0.__S @ vp0
            
            v0 = vp0 + vm0
            
            y0 = self.__S0.getYMatrix()
            
            if (np.isnan(y0)).all(axis=(1,2)).any(): #The Y matrix has not been computed at least at one frequency value
                warnings.warn("The currents cannot be computed at least at one frequency value. NaN is returned at those frequencies")
            
            i0 = y0 @ v0
        
        #v_sup or i_sup shape = [self.__f, self.__nPorts, self.__nPorts] --> v_sup[j,k,q] is the voltage at frequency self.__f[j] at port k of the self.__S matrix when 1 W power is incident to port q of the self.__S matrix
        #v0 or i0 shape = [self.__f, self.__S0.__nPorts, self.__nPorts] --> v0[j,k,q] is the voltage at frequency self.__f[j] at port k of the self.__S0.__S matrix when 1 W power is incident to port q of the self.__S matrix

        return v_sup, i_sup ,v0, i0 
          
    
    def powerBalance(self, p_inc):

        if not isinstance(p_inc, np.ndarray) and not isinstance(p_inc, list):
            raise TypeError("p_inc can only be numpy ndarray or a list")
        else:
            p_inc = np.array(p_inc)
        if p_inc.size != self.__nPorts:
            raise TypeError("p_inc has to be a self.nPorts length list or numpy ndarray")
        
        #Computation of power accepted by the whole system: Original S matrix and connected circuitries
        vp_1 = np.sqrt(self.__z0 * np.abs(p_inc)) * np.exp(1j*np.angle(p_inc))
        S_transpose = np.moveaxis(self.__S,-1,-2)
        diag_z0_inv  = np.linalg.inv(np.diag(self.__z0))
        
        p_acc_1 = np.conj(vp_1).T @ (diag_z0_inv - np.conj(S_transpose) @ diag_z0_inv @ self.__S) @ vp_1 # Power accepted by the whole system: Orignal S Matrix and connected circuitries
        p_acc_1 = np.real(p_acc_1)
        
        if self.__S0 is None:
            print("No previous connections to external circuitries or connection data have not been saved. The power balance is computed only at the supply ports")
            return p_acc_1,
        
        #Computation of power accepted by the original S Matrix
        p_inc2 = self.__S0.__p_incM * np.exp(1j*self.__S0.__phaseM) #p_inc2.shape = [self.__n_f, self.__nPorts, self.__S0.__nPorts]
        p_inc2 = np.moveaxis(p_inc2,-1,1) #p_inc2.shape = [self.__n_f, self.__S0.__nPorts, self.__nPorts]
        
        vp_2_temp = np.sqrt(np.abs(p_inc2)) * np.exp(1j*np.angle(p_inc2)) #Still to be multiplied by the sqare root of the port impedances
        vp_2 = np.sqrt(self.__S0.__z0) * (vp_2_temp @  (np.sqrt(np.abs(p_inc)) * np.exp(1j*np.angle(p_inc)))) #p_inc2.shape = [self.__n_f, self.__S0.__nPorts]
        vp_2 = np.expand_dims(vp_2,axis=-1) #vp_2.shape = [self.__n_f, self.__S0.__nPorts,1]
        
        S_transpose = np.moveaxis(self.__S0.__S,-1,-2)
        vp_2_transpose = np.moveaxis(vp_2,-1,-2)
        diag_z0_inv  = np.linalg.inv(np.diag(self.__S0.__z0))
        p_acc_2 = np.conj(vp_2_transpose) @ (diag_z0_inv - np.conj(S_transpose) @ diag_z0_inv @ self.__S0.__S) @ vp_2 # Power accepted by the whole system: Orignal S Matrix and connected circuitries
        p_acc_2 = np.real(p_acc_2)
        
        return p_acc_1, p_acc_2.flatten()
        
    
    def healSMatrix(self, report=False, f_tol=1e-10, rdiff=None, **kwarg):
        
        healed_S = np.zeros_like(self.__S)
        
        def equation_syst(k_t, x):
    
            S = np.reshape(x,[self.__nPorts,self.__nPorts],order='F')
            roots = (np.conj(S).T@S - k_t).flatten(order='F')
    
            return roots
        
        print("\n\n\nHealing S matrix...\n\n")
        
        for i in range(self.__n_f): #I use fo cycle since I want to keep separated the roots finding problems
                
            print("\r%.2f %%" %(i/self.__n_f*100), end='')
            S = self.S[i,:,:]

            P = np.eye(self.__nPorts) - np.conj(S).T@S

            eig_val, eig_vec = np.linalg.eig(P)
            eig_val = np.real(eig_val)
            
            if (eig_val < 0).any():
                
                eig_val[eig_val<0] = 0
                
                new_P = eig_vec @ np.diag(eig_val) @ np.linalg.inv(eig_vec)

                M = np.eye(self.__nPorts) - new_P
                
                try:
                    sol = newton_krylov(partial(equation_syst, M),[S.flatten(order='F')], f_tol=f_tol, rdiff=rdiff)
                    sol=sol.reshape((self.__nPorts,self.__nPorts),order='F')
                    healed_S[i] = sol
                except (NoConvergence):
                    warnings.warn("The Newton Krylov algorithm returned a NoConvergence error at least at one frequency value. NaN is returned at these frequencies")
                    healed_S[i] = np.ones((self.__nPorts,self.__nPorts)) * np.nan
            else:
                healed_S[i] = S
        
        if report:
            rep = {}
            rep["max_abs_Sdiff"] = np.max(np.abs(self.__S - healed_S),axis=(1,2))
            
            new_P = np.repeat(np.expand_dims(np.eye(self.__nPorts,self.__nPorts), axis=0), self.__n_f, axis=0) - np.moveaxis(np.conj(healed_S),-1,1) @ healed_S
            
            rep["min_eig_val"] = np.ones(self.__n_f) * np.nan
            
            not_nan_idxs = np.where(np.logical_not(np.isnan(new_P).any(axis=(1,2))))[0] #idx of new_P first dimension where no nan values along the other two dimensions are encountered

            new_eig_val = np.real(np.linalg.eig(new_P[not_nan_idxs])[0]) #np.linalg.eig works only for not nan values
            
            rep["min_eig_val"][not_nan_idxs] = np.min(new_eig_val, axis=1)
            
            return S_Matrix(healed_S, self.__f, self.__z0, **kwarg), rep
        
        return S_Matrix(healed_S, self.__f, self.__z0, **kwarg)
            
    
    def _singlePortConnSMatrix(self, networks, comp_Pinc=False):
        
        if len(networks) != self.__nPorts:
            raise ValueError("Invalid networks length value")
        
        input_ports = self.__nPorts #Ports to be connected to self.S
        output_ports = 0 #Ports of the returned S Matrix
        output_ports_impedances = np.empty(0)
        for p, network in enumerate(networks):
            if network is None:
                output_ports += 1
                output_ports_impedances = np.append(output_ports_impedances, self.__z0[p])
            elif not isinstance(network, S_Matrix):
                raise TypeError("All the elements of networks have either to be None or an instance of S_Matrix")
            elif not np.array_equal(self.__f, network.__f):
                raise TypeError("All the S_Matrix instances in networks have to be defined over the same frequency values of self.S")
            else:
                if network.__z0[0] != self.__z0[p]:
                    raise TypeError("The input port impedances of the S Matrix elements of networks must be equal to the relevant port impedances of self.S")
                output_ports += network.__nPorts - 1
                output_ports_impedances = np.append(output_ports_impedances, network.__z0[1:])
        
        if output_ports == 0:
            raise TypeError("The returned S_Matrix has to result at least into a 1 port S_Matrix")
        
        #The s matrix that will be used as other by the __resSMatrix method
        s_forComp = np.zeros((self.__n_f, input_ports+output_ports, input_ports+output_ports), dtype='complex')
        
        out_idx = 0
        for in_idx, network in enumerate(networks):
            if network is None:
                s_forComp[:, in_idx, input_ports+out_idx] = 1
                s_forComp[:, input_ports+out_idx, in_idx] = 1
                out_idx += 1
            else:
                n_c = network.__nPorts
                s_forComp[:, in_idx, in_idx] = network.__S[:, 0, 0]
                s_forComp[:, in_idx, input_ports+out_idx:input_ports+out_idx+n_c-1] = network.__S[:, 0, 1:]
                s_forComp[:, input_ports+out_idx:input_ports+out_idx+n_c-1, in_idx] = network.__S[:, 1:, 0]
                s_forComp[:, input_ports+out_idx:input_ports+out_idx+n_c-1, input_ports+out_idx:input_ports+out_idx+n_c-1] = network.__S[:, 1:, 1:]
                out_idx += n_c - 1
        
        s_forComp = S_Matrix(s_forComp, self.__f, np.concatenate((self.__z0, output_ports_impedances)))
        
        S_res = self.__resSMatrix(s_forComp)
        
        
        if comp_Pinc:
            #s matrix for closing s_forComp output ports on matched loads
            s_loads_Pinc = np.zeros((self.__n_f, 2*input_ports+output_ports+1, 2*input_ports+output_ports+1))
            
            for port in range(input_ports+1):
                s_loads_Pinc[:, port, input_ports+output_ports+port] = 1
                s_loads_Pinc[:, input_ports+output_ports+port, port] = 1
            
            p_incM = np.empty((self.__n_f, 0, self.__nPorts))
            phaseM = np.empty_like(p_incM)
            
            if output_ports == 1: #If there is only an output_port, no ports need to be closed on a matched load
                p_inc, phase = self.__load_Pinc(s_forComp)
                p_incM = np.append(p_incM, p_inc, axis=1)
                phaseM = np.append(phaseM, phase, axis=1)
            else:
                for port in range(output_ports):
                    if port != 0:
                        s_forComp = S_Matrix.__movePort(s_forComp, input_ports, -1)

                    s_loads_Pinc_M = S_Matrix(s_loads_Pinc, self.__f, np.concatenate((s_forComp.__z0,s_forComp.__z0[:input_ports+1])))
                    
                    s_Pinc = s_forComp.__resSMatrix(s_loads_Pinc_M)
                    
                    p_inc, phase = self.__load_Pinc(s_Pinc)
                    p_incM = np.append(p_incM, p_inc, axis=1) #P_incM.shape = [self.__f, output_ports, input_ports]
                    phaseM = np.append(phaseM, phase, axis=1) #phaseM.shape = [self.__f, output_ports, input_ports]
                    
            #Save connection data for vi and power budget computations
            self.__p_incM = p_incM
            self.__phaseM = phaseM
            self.__S0 = None #self.__S0 is not None if previous connections already occured. I clear the property to avoid excessive data storing in case of many successive connections.
                             #This can be improved storing nested S0s allowing to compute voltages and currents relevant to all the connected circuitries levels
            S_res.__S0 = self
            
            return S_res, p_incM, phaseM

        return S_res,
    
    
    def _fullPortsConnSMatrix(self, other, comp_Pinc=False):
        
        S_res = self.__resSMatrix(other)
        
        if comp_Pinc:
            
            #s matrix for closing the output ports of other on matched loads
            s_loads_Pinc = np.zeros((self.__n_f, self.__nPorts+other.__nPorts+1, self.__nPorts+other.__nPorts+1))
            
            for port in range(self.__nPorts+1):
                s_loads_Pinc[:, port, other.__nPorts+port] = 1
                s_loads_Pinc[:, other.__nPorts+port, port] = 1

            p_incM = np.empty((self.__n_f, 0, self.__nPorts))
            phaseM = np.empty_like(p_incM)
            
            output_ports = other.__nPorts - self.__nPorts
            if output_ports == 1: #If there is only an output_port, no ports need to be closed on a matched load
                p_inc, phase = self.__load_Pinc(other)
                p_incM = np.append(p_incM, p_inc, axis=1)
                phaseM = np.append(phaseM, phase, axis=1)
            else:
                s_forComp = copy(other)
                
                for port in range(output_ports):
                    if port != 0:
                        s_forComp = S_Matrix.__movePort(s_forComp, self.__nPorts, -1)

                    s_loads_Pinc_M = S_Matrix(s_loads_Pinc, self.__f, np.concatenate((s_forComp.__z0,s_forComp.__z0[:self.__nPorts+1])))
                    
                    s_Pinc = s_forComp.__resSMatrix(s_loads_Pinc_M)
                    
                    p_inc, phase = self.__load_Pinc(s_Pinc)
                    p_incM = np.append(p_incM, p_inc, axis=1) #P_incM.shape = [self.__f, output_ports, input_ports]
                    phaseM = np.append(phaseM, phase, axis=1) #phaseM.shape = [self.__f, output_ports, input_ports]
                
            #Save connection data for vi and power budget computations
            self.__p_incM = p_incM
            self.__phaseM = phaseM
            self.__S0 = None #self.__S0 is not None if previous connections already occured. I clear the property to avoid excessive data storing in case of many successive connections
                             #This can be improved storing nested S0s allowing to compute voltages and currents relevant to all the connected circuitries levels
            S_res.__S0 = self
                    
            return S_res, p_incM, phaseM

        return S_res,
    
    
    def __resSMatrix(self, other):
        
        if not isinstance(other, S_Matrix):
            raise TypeError("The S matrix can be connected only to an instance of S_Matrix")
        elif other.__nPorts < self.__nPorts + 1:
            raise ValueError("The number of ports of other must be higher than (self.nPorts + 1)")
        elif not np.array_equal(other.__f, self.__f):
            raise ValueError("The other S matrix has to be defined on the same frequency values as the S matrix to be loaded")
        elif (other.__z0[:self.__nPorts] != self.__z0).any():
            raise ValueError("The first self.nPorts of other must have the same impedances of the ports of self")
                
        S_C_11 = other.__S[:,:self.nPorts,:self.nPorts]
        S_C_12 = other.__S[:,:self.nPorts,self.nPorts:]
        S_C_21 = other.__S[:,self.nPorts:,:self.nPorts]
        S_C_22 = other.__S[:,self.nPorts:,self.nPorts:]
        
        S_0 = np.copy(self.__S)
        
        
        #Management of singular matrices: Look for singular matrices
        
        #Look for singular matrices
        mask_sing = np.isclose(np.linalg.det(S_0), 0)
        
        if mask_sing.any():
            #Mask of manageble (A procedure) singular matrices: Array of shape equal to S_0.shape[0],S_0.shape[1]. rc_zero_mask[i,j] is True if S_0[i,:,j] == S_0[i,j,:] == 0
            rc_zero_mask = np.logical_and((S_0==0).all(axis=1),(S_0==0).all(axis=2))
            #Indices of S_0 first dimension containing singular matrices which can be managed
            inv_S_0_ixds = np.where((rc_zero_mask).any(axis=1))[0]
                
            #I turn into np.nan S_0 at frequencies at which S_0 = is singualar
            S_0[mask_sing] = np.nan * np.ones((self.nPorts, self.nPorts))
        
        S_0_inv = np.linalg.inv(S_0)
        
        S_res = S_C_21 @ np.linalg.inv(S_0_inv - S_C_11) @ S_C_12 + S_C_22
        
        
        #Management of singular matrices: procedure A
        
        if mask_sing.any():
            S_0 = np.copy(self.__S)
            for idx in inv_S_0_ixds:
                #Indices of the potentially invertible sub-matrix
                rc_idxs = np.where(np.logical_not(rc_zero_mask[idx]))[0]
                sub_S_0 = S_0[idx][rc_idxs][:,rc_idxs]
                
                if np.isclose(np.linalg.det(sub_S_0), 0): #If the sub-matrix is not invertible
                    rc_zero_mask[idx] = False #S matrix is singular and not manageable with A procedure
                else:
                    sub_S_0_inv = np.linalg.inv(sub_S_0) 
                    sub_S_C_11 = S_C_11[idx][rc_idxs][:,rc_idxs]
                    sub_S_C_12 = S_C_12[idx][rc_idxs]
                    sub_S_C_21 = S_C_21[idx][:,rc_idxs]
                
                    S_res[idx] = sub_S_C_21 @ np.linalg.inv(sub_S_0_inv - sub_S_C_11) @ sub_S_C_12 + S_C_22[idx]
            
        #Management of singular matrices: procedure B (I try using Z matrices)
        
        if mask_sing.any():
            #Mask which is True where S0 are singular and not manageable with A procedure
            mask_notSolved = np.logical_and(mask_sing,np.logical_not((rc_zero_mask).any(axis=1)))

            if mask_notSolved.any():
                notSolved_idx = np.where(mask_notSolved)[0]
                for idx in notSolved_idx:
                    S_0_extr = np.expand_dims(S_0[idx],axis=0)
                    S_C_extr = np.copy(other.__S)[idx]
                    
                    #I look for couple of rows i and j where row i is made of zeros except for 1 at j column and
                    #row j is zeros except 1 at column i. This causes the S_Matrix.getZMatrix() method to return error
                    idxs_one = np.vstack(np.where(S_C_extr==1)).T
                    prev_idxs = -1 * np.ones((1,2))

                    for idx_one in idxs_one:

                        if (idx_one[::-1] != prev_idxs).any(axis=1).all():
                            r1 = np.zeros((other.__nPorts))
                            r2 = np.zeros((other.__nPorts))
                            r1[idx_one[1]] = 1
                            r2[idx_one[0]] = 1
                            if (S_C_extr[idx_one[0]] == r1).all() and (S_C_extr[idx_one[1]] == r2).all():
                                S_C_extr[idx_one[0], idx_one[1]] = 1 - eps
                                S_C_extr[idx_one[1], idx_one[0]] = 1 - eps
                                warnings.warn("The matrices are singular at least at one frequency value. Results can be inaccurate at those frequencies")
                                
                        prev_idxs = np.append(prev_idxs, np.expand_dims(idx_one,0), axis=0)
                        
                    S_0_extr = S_Matrix(S_0_extr, [None], self.__z0) #Fictitious S_Matrix instance to use getZMatrix method
                    S_C_extr = np.expand_dims(S_C_extr, axis=0)
                    S_C_extr = S_Matrix(S_C_extr, [None], other.__z0) #Fictitious S_Matrix instance to use getZMatrix method

                    try:
                        Z_0 = S_0_extr.getZMatrix()
                        Z_C = S_C_extr.getZMatrix()

                        
                        Z_C_11 = Z_C[0,:self.nPorts,:self.nPorts]
                        Z_C_12 = Z_C[0,:self.nPorts,self.nPorts:]
                        Z_C_21 = Z_C[0,self.nPorts:,:self.nPorts]
                        Z_C_22 = Z_C[0,self.nPorts:,self.nPorts:]
                        
                        Z_0_inv = np.linalg.inv(Z_0[0])
                        
                        Z_res = Z_C_22 - Z_C_21 @ Z_0_inv @ np.linalg.inv(np.eye(self.nPorts) + Z_C_11 @ Z_0_inv) @ Z_C_12

                        S_res[idx] = S_Matrix.fromZtoS(np.expand_dims(Z_res,axis=0), [None], other.__z0[self.nPorts:]).__S[0]
                    
                    except np.linalg.LinAlgError:

                        warnings.warn("The matrices are singular at least at one frequency value. NaN is returned at those frequencies")
                        if self.__info_warnings:
                                    print("\nS0:\n")
                                    print(S_0_extr)
                                    print("\nSc:\n")
                                    print(S_C_extr)
                                    
        S_res = S_Matrix(S_res, self.__f, other.__z0[self.nPorts:], **self.__kwarg)
        
        return S_res
        

    def __load_Pinc(self, other):
        
        if not isinstance(other, S_Matrix):
            raise TypeError("The S matrix can be loaded only with an instance of S_Matrix")
        elif other.__nPorts != self.__nPorts + 1:
            raise ValueError("other.nPorts must be equal to self.nPorts+1")
        elif not np.array_equal(other.__f, self.__f):
            raise ValueError("The load S matrix has to be defined on the same frequency values as the S matrix to be loaded")
        elif (self.__z0 != other.__z0[:self.__nPorts]).any():
            raise ValueError("The first self.nPorts of other must have the same impedances of the ports of self")

        z0 = self.__z0
        
        S_O = self.__S
        S_CL_11 = other.__S[:,:-1,:-1]
        S_CL_12 = other.__S[:,:-1,-1:]
        
        V_O_p = (np.linalg.inv(np.eye(self.__nPorts) - S_CL_11 @ S_O) @ S_CL_12) * np.sqrt(np.real(other.__z0[-1]))
        
        #For testing
        # V_O_m = S_O @ V_O_p
        # V_O = V_O_p + V_O_m
        # Z_O = self.getZMatrix()
        # I_O = np.linalg.inv(Z_O) @ V_O
        # Vg = V_O + z0m @ I_O
        
        p_inc = np.abs(V_O_p)**2 / (np.repeat(np.expand_dims(np.real(z0),axis=(0,2)), self.__n_f, axis=0))
        phase = np.angle(V_O_p)

        return np.swapaxes(p_inc, 1, 2), np.swapaxes(phase, 1, 2)
    
      
    def __findFreqIndex(self, freq):
        
        if freq in self.__f:
            idx = np.where(self.__f == freq)[0][0]
        else:
            idx = np.argmin(np.abs(self.__f - freq))
            warnings.warn("%e Hz is not contained in the frequencies list. %e Hz is returned instead" %(freq, self.__f[idx]))
        
        return idx
        
        
    @staticmethod
    def fromZtoS(Z, freqs, z0 = 50):
        
        if isinstance(z0, list) or isinstance(z0, np.ndarray):
            if len(z0) != Z.shape[-1]:
                raise ValueError("z0 has to be a list or numpy array with a length equal to the number of ports associated with the S matrix")
        else:
            z0 = z0 * np.ones(Z.shape[-1])
            
        if (np.real(z0) <= 0).any():
            raise ValueError("The real part of all the port impedances has to be higher than zero")
        elif (np.real(z0) != z0).any():
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            z0 = np.real(z0)
            
        y0 = np.diag(1./np.array(z0))
        
        y0sqrt = np.sqrt(y0)
        
        S = (y0sqrt @ Z @ y0sqrt - np.eye(Z.shape[-1])) @ np.linalg.inv(y0sqrt @ Z @ y0sqrt + np.eye(Z.shape[-1]))
    
        return S_Matrix(S,freqs,z0)
    
    
    @staticmethod
    def fromYtoS(Y, freqs, y0 = 0.02):
        
        if isinstance(y0, list) or isinstance(y0, np.ndarray):
            if len(y0) != Y.shape[-1]:
                raise ValueError("y0 has to be a list or numpy array with a length equal to the number of ports associated with the S matrix")
        else:
            y0 = y0 * np.ones(Y.shape[-1])
        
        if (np.real(y0) <= 0).any():
            raise ValueError("The real part of all the port impedances has to be higher than zero")
        elif (np.real(y0) != y0).any():
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            y0 = np.real(y0)    
            
        z0 = np.diag(1./np.array(y0))
        
        z0sqrt = np.sqrt(z0)
        
        S = (np.eye(Y.shape[-1]) - z0sqrt @ Y @ z0sqrt) @ np.linalg.inv(np.eye(Y.shape[-1]) + z0sqrt @ Y @ z0sqrt)
    
        return S_Matrix(S,freqs,np.diag(z0))
    
    
    @staticmethod
    def importTouchstone(filename, fmt='R_I', freqUnit='MHz', z0=50, fix=False, n_f=None, **kwarg):

        if fmt not in ['R_I', 'Mag_Deg', 'dB_Deg', 'Mag_Rad', 'dB_Rad']:
            raise ValueError("fmt is not valid")
        elif freqUnit not in ['Hz', 'MHz']:
            raise ValueError("freqUnit is not valid")
            
        if fix:
            if n_f is None:
                raise ValueError("Please specify the number of frequency values 'n_f' in order to fix the touchstone file")
            with open(filename, 'r') as f:
                data = f.readlines()
            
            comm_rows = 0
            fc = data[0][0]
            while fc.strip() in ["!", "#", ""]:
                comm_rows += 1
                fc = data[comm_rows][0]
            
            comments = "".join(data[:comm_rows])
            data = data[comm_rows:]
            
            for r,d in enumerate(data):
                data[r] = d.split("!")[0]

            if len(data) % n_f != 0: #Check that each frequency value corresponds to the same number of rows
                raise ValueError("Impossible to fix the touchstone file")
            else:
                n_r = int(len(data) / n_f)
                
            new_data = np.array(data[0::n_r])
            
            for i in range(n_r-1):
                new_data = np.core.defchararray.add(new_data, data[i+1::n_r])
            new_data = np.chararray.split(new_data)
            new_data = np.stack(new_data,axis=0)
            
            new_data = new_data.astype(np.float)
            
            filename = filename.split(".")
            filename = ".".join(filename[:-1]) + "_fixed." + filename[-1]            
            
            np.savetxt(filename, new_data, header=comments, fmt="%.6f", delimiter="\t")
            
        data = np.loadtxt(filename, comments=["#","!"])
        data = np.atleast_2d(data)
        
        if freqUnit == 'Hz':
            freq = data[:,0]
        else:
            freq = 1e6*data[:,0]
        
        s_parameters_A = data[:,1:][:,::2]
        s_parameters_B = data[:,2:][:,::2]
        
        if fmt == 'R_I':
            s_parameters = s_parameters_A + 1j*s_parameters_B
        elif fmt == 'Mag_Deg':
            s_parameters = s_parameters_A * np.exp(1j*np.deg2rad(s_parameters_B))
        elif fmt == 'Mag_Rad':
            s_parameters = s_parameters_A * np.exp(1j*s_parameters_B)
        elif fmt == 'dB_Deg':
            s_parameters = (10**(s_parameters_A/20)) * np.exp(1j*np.deg2rad(s_parameters_B))
        elif fmt == 'dB_Rad':
            s_parameters = (10**(s_parameters_A/20)) * np.exp(1j*s_parameters_B)

        s_parameters = np.reshape(s_parameters, (freq.size, np.int(np.sqrt(s_parameters_A.shape[-1])),-1))

        return S_Matrix(s_parameters, freq, z0, **kwarg)
    
    
    @staticmethod
    def sMatrixRCseries(Rvalue, Cvalue, freqs, z0=50):
        
        if Rvalue < 0 or Cvalue <= 0:
            raise ValueError("Rvalue and Cvalue must be equal or higher than zero")
        elif np.real(z0) != z0:
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            z0 = np.real(z0)

        f = np.array(freqs)
        n_f = len(f)
        
        z = (Rvalue - 1.j/(2*np.pi*f*Cvalue)).reshape(n_f,1,1)
        
        s = (z - z0) / (z + z0)
        
        return S_Matrix(s, f, z0)
    
    
    @staticmethod
    def sMatrixRLseries(Rvalue, Lvalue, freqs, z0=50):
        
        if Rvalue < 0 or Lvalue < 0:
            raise ValueError("Rvalue and Lvalue must be equal or higher than zero")
        elif np.real(z0) != z0:
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            z0 = np.real(z0)

        f = np.array(freqs)
        n_f = len(f)
        
        z = (Rvalue + 1.j*2*np.pi*f*Lvalue).reshape(n_f,1,1)
        
        s = (z - z0) / (z + z0)
        
        return S_Matrix(s, f, z0)
    
    
    @staticmethod
    def sMatrixRCparallel(Rvalue, Cvalue, freqs, z0=50):
        
        if Rvalue < 0 or Cvalue <= 0:
            raise ValueError("Rvalue and Cvalue must be equal or higher than zero")
        elif np.real(z0) != z0:
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            z0 = np.real(z0)
    
        f = np.array(freqs)
        n_f = len(f)
        
        tau = Rvalue*Cvalue
        omega = 2*np.pi*f
        z = (Rvalue*(1. - 1j*omega*tau) / ((omega*tau)**2 + 1)).reshape(n_f,1,1)
        
        s = (z - z0) / (z + z0)
             
        return S_Matrix(s, f, z0)
            
    
    @staticmethod
    def sMatrixRLparallel(Rvalue, Lvalue, freqs, z0=50):
        
        if Rvalue < 0 or Lvalue < 0:
            raise ValueError("Rvalue and Lvalue must be equal or higher than zero")
        elif np.real(z0) != z0:
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            z0 = np.real(z0)

        f = np.array(freqs)
        n_f = len(f)
        
        tau = Lvalue / Rvalue
        omega = 2*np.pi*f
       
        z = ((omega*Lvalue) * (omega*tau + 1.j) / ((omega*tau)**2 + 1)).reshape(n_f,1,1)
        
        s = (z - z0) / (z + z0)
        
        return S_Matrix(s, f, z0)
    
    
    @staticmethod
    def sMatrixTnetwork(SL_1, SL_2, ST, z0=50):
        
        if not isinstance(SL_1, S_Matrix) or not isinstance(SL_2, S_Matrix) or not isinstance(ST, S_Matrix):
            raise TypeError("SL_1, SL_2 and ST must be S_Matrix instances")
        elif not np.array_equal(SL_1.__f, SL_2.__f) or not np.array_equal(SL_1.__f, ST.__f):
            raise ValueError("SL_1, SL_2 and ST must be defined over the same frequency values")
        elif (SL_1.nPorts != 1) or (SL_2.nPorts != 1) or (ST.nPorts != 1):
            raise ValueError("SL_1, SL_2 and ST must be one port S_Matrix instances")

        if isinstance(z0,list) or isinstance(z0,np.ndarray):
            if len(z0) != 2:
                raise ValueError("z0 must be a real value or a real values list with length equal to 2")
            z0 = np.array(z0)
        else:
            z0 = z0*np.ones(2)
        
        f = SL_1.__f
        
        #Try to avoid some possible Singular Matrix errors
        if (SL_1.__S==1).any() or (SL_2.__S==1).any() or (ST.__S==1).any():
            warnings.warn("Singular matrix error detected at least at one frequency value. Inaccurate values can be obtained at those frequencies")
            SL_1.__S[SL_1.__S==1] = 1 - eps
            SL_2.__S[SL_2.__S==1] = 1 - eps
            ST.__S[ST.__S==1] = 1 - eps

        ZL_1 = SL_1.getZMatrix()
        ZL_2 = SL_2.getZMatrix()
        ZT = ST.getZMatrix()
        
        Z11 = ZL_1 + ZT
        Z12 = ZT
        Z21 = ZT
        Z22 = ZL_2 + ZT
        
        Z1 = np.concatenate((Z11, Z12), axis=2)
        Z2 = np.concatenate((Z21, Z22), axis=2)
        
        z = np.concatenate((Z1, Z2), axis=1)
        
        return S_Matrix.fromZtoS(z, f, z0)
        
    
    @staticmethod
    def sMatrixPInetwork(ST_1, ST_2, SL, z0=50):

        if not isinstance(ST_1, S_Matrix) or not isinstance(ST_2, S_Matrix) or not isinstance(SL, S_Matrix):
            raise TypeError("ST_1, ST_2 and SL must be S_Matrix instances")
        elif not np.array_equal(ST_1.__f, ST_2.__f) or not np.array_equal(ST_1.__f, SL.__f):
            raise ValueError("ST_1, ST_2 and SL must be defined over the same frequency values")
        
        if isinstance(z0,list) or isinstance(z0,np.ndarray):
            if len(z0) != 2:
                raise ValueError("z0 must be a real value or a real values list with length equal to 2")
            y0 = 1. / np.array(z0)
        else:
            y0 = (1./z0) * np.ones(2)
            
        f = ST_1.__f
        
        #Try to avoid some possible Singular Matrix errors
        if (ST_1.__S==-1).any() or (ST_2.__S==-1).any() or (SL.__S==-1).any():
            warnings.warn("Singular matrix error detected at least at one frequency value. Inaccurate values can be obtained at those frequencies")
            ST_1.__S[ST_1.__S==-1] = -1 + eps
            ST_2.__S[ST_2.__S==-1] = -1 + eps
            SL.__S[SL.__S==-1] = -1 + eps
        
        YT_1 = ST_1.getYMatrix()
        YT_2 = ST_2.getYMatrix()
        YL = SL.getYMatrix()
        
        Y11 = YT_1 + YL
        Y12 = -YL
        Y21 = -YL
        Y22 = YT_2 + YL
        
        Y1 = np.concatenate((Y11, Y12), axis=2)
        Y2 = np.concatenate((Y21, Y22), axis=2)
        
        y = np.concatenate((Y1, Y2), axis=1)
                
        s = S_Matrix.fromYtoS(y, f, y0)
        
        return s
    
    
    @staticmethod
    def sMatrixTrLine(l, freqs, z0_line=50, c_f = 1, alpha=0, z0=50):

        if isinstance(z0,list) or isinstance(z0,np.ndarray):
            z0_arr = z0
            if len(z0) != 2:
                raise ValueError("z0 must be a real value or a real values list with length equal to 2")
        else:
            z0_arr = np.ones(2) * z0
                
        f = np.array(freqs)
        wl = 3e8 * c_f * 1/f
        beta = 2 * np.pi / wl
        
        gamma = alpha + 1j*beta
        
        z0 = z0_arr[0]
        
        #S11
        Sl = (z0 - z0_line) / (z0 + z0_line)
        Zbegin = z0_line * (1 + Sl*np.exp(-2*gamma*l)) / (1 - Sl*np.exp(-2*gamma*l))
        S11 = (Zbegin - z0) / (Zbegin + z0)
        
        #S21
        v_p = 0.5 * (1 + S11) * (1 + z0_line/Zbegin)
        v_m = 0.5 * (1 + S11) * (1 - z0_line/Zbegin)
        S21 = v_p*np.exp(-gamma*l) +  v_m*np.exp(gamma*l)
        
        z0 = z0_arr[1]
        
        #S22
        Sl = (z0 - z0_line) / (z0 + z0_line)
        Zbegin = z0_line * (1 + Sl*np.exp(-2*gamma*l)) / (1 - Sl*np.exp(-2*gamma*l))
        S22 = (Zbegin - z0) / (Zbegin + z0)
        
        #S12
        v_p = 0.5 * (1 + S11) * (1 + z0_line/Zbegin)
        v_m = 0.5 * (1 + S11) * (1 - z0_line/Zbegin)
        S12 = v_p*np.exp(-gamma*l) +  v_m*np.exp(gamma*l)
        
        S11 = S11[:,None,None]
        S22 = S22[:,None,None]
        S21 = S21[:,None,None]
        S12 = S12[:,None,None]
        
        S1 = np.concatenate((S11, S12), axis=2)
        S2 = np.concatenate((S21, S22), axis=2)
        
        s = np.concatenate((S1, S2), axis=1)
        
        return S_Matrix(s, f, z0)
    
    
    @staticmethod
    def __movePort(Smat, idx0, idx1):

        if not isinstance(Smat, S_Matrix):
            raise TypeError("Smat must be an S_Matrix instance")
        
        if idx0 < 0:
            idx0 = Smat.__nPorts+idx0
        if idx1 < 0:
            idx1 = Smat.__nPorts+idx1
        
        tempS = np.copy(Smat.__S)
        tempz0 = np.copy(Smat.__z0)
        
        if idx0 < idx1:
            
            #move row
            tempS[:,idx0:idx1,:] = Smat.__S[:,idx0+1:idx1+1,:]
            tempS[:,idx1,:] = Smat.__S[:,idx0,:]
            tempS[:,idx1+1:,:] = Smat.__S[:,idx1+1:,:]
            
            #move column
            S_new = np.copy(tempS)
            S_new[:,:,idx0:idx1] = tempS[:,:,idx0+1:idx1+1]
            S_new[:,:,idx1] = tempS[:,:,idx0]
            
            #fix z0
            tempz0[idx0:idx1+1] = np.roll(tempz0[idx0:idx1+1],-1)
        
        elif idx0 > idx1:
            
            #move row
            tempS[:,idx1,:] = Smat.__S[:,idx0,:]
            tempS[:,idx1+1:idx0+1,:] = Smat.__S[:,idx1:idx0,:]
            tempS[:,idx0+1:,:] = Smat.__S[:,idx0+1:,:]
            
            #move column
            S_new = np.copy(tempS)
            S_new[:,:,idx1] = tempS[:,:,idx0]
            S_new[:,:,idx1+1:idx0+1] = tempS[:,:,idx1:idx0]
            
            #fix z0
            tempz0[idx0:idx1+1] = np.roll(tempz0[idx0:idx1+1],1)
        
        else:
            
            S_new = tempS

        return S_Matrix(S_new, Smat.__f, tempz0)
