import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.io import loadmat
import warnings
from copy import copy
import re
from os import listdir
from os.path import isfile, join
from ctypes import CDLL, c_int, c_double
from .Exceptions import EM_FieldError, EM_FieldArrayError, EM_FieldFrequenciesError, EM_FieldPointsError,\
    EM_FieldIOError, EM_FieldPropertiesError

def warning_format(message, category, filename, lineno, file=None, line=None):
    return '\n%s: Line %s - WARNING - %s\n' % (filename.split("/")[-1], lineno, message)

warnings.formatwarning = warning_format

class EM_Field():
    
    def __init__(self, freqs, nPoints, b_field=None, e_field=None, props={}):
        
        if e_field is None and b_field is None:
            raise EM_FieldArrayError("At least one among e_field and b_field arguments has to be different from None", "__init__")
        
        if e_field is not None:
            EM_FieldArrayError.check(e_field, "__init__")
            
        if b_field is not None:
            EM_FieldArrayError.check(b_field, "__init__")
                    
        if e_field is not None:
            EM_FieldPointsError.check(nPoints, "__init__", e_field.shape[-1])
            EM_FieldFrequenciesError.check(freqs, "__init__", e_field.shape[0])
        else:
            EM_FieldPointsError.check(nPoints, "__init__", b_field.shape[-1])
            EM_FieldFrequenciesError.check(freqs, "__init__", b_field.shape[0])
        
        if e_field is not None and b_field is not None:
            if e_field.shape[1] != b_field.shape[1]:
                raise EM_FieldArrayError("The second dimension of the e_field array is not consistent with the second dimension of the b_field array", "__init__")

        if e_field is not None:  
            self.__e_field = np.array(e_field,dtype = "complex")
            self.__nPorts = e_field.shape[1]
        else:
            self.__e_field = None
            self.__nPorts = b_field.shape[1]
            
        if b_field is not None:    
            self.__b_field = np.array(b_field,dtype = "complex")
        else:
            self.__b_field = None
            
        self.__f = np.array(freqs)
        self.__nPoints = np.array(nPoints)
        self.__n_f = len(freqs)
        
        EM_FieldPropertiesError.check(props, "__init__", np.prod(nPoints))
        self.__props = props
        for prop in self.__props:
            self.__props[prop] = np.array(self.__props[prop])
            if prop == "idxs": # idxs must be integer type array for proper indexing in other methods
                self.__props[prop] = self.__props[prop].astype(int)
            
    
    @property
    def e_field(self):
        return self.__e_field
    
    
    @property
    def b_field(self):
        return self.__b_field
    
    
    @property
    def nPoints(self):
        return self.__nPoints
    
    
    @property
    def n_f(self):
        return self.__n_f
    
    
    @property
    def nPorts(self):
        return self.__nPorts
    
    
    @property
    def frequencies(self):
        return self.__f
    
    
    @property
    def properties(self):
        return self.__props
    
    
    def __repr__(self):
        string = '"""""""""""""""\n   EM FIELD\n"""""""""""""""\n\n'
        string += "Number of frequency values = %d\nNumber of ports = %d\nNumber of points (nx, ny, nz) = %d, %d, %d\n\n"%(self.__n_f,self.__nPorts, self.__nPoints[0], self.__nPoints[1],self.__nPoints[2])
        if self.e_field is None:
                string += "E field not defined\n\n"
        elif self.b_field is None:
            string += "B field not defined\n\n"
        if self.__props != {}:
            for key in self.__props:
                if key != "idxs":
                    string += "'%s' additional property defined\n" %key
        return string
    
    
    def __getitem__(self, key):
        ret_em_field = copy(self)
        
        if isinstance(key,int) or isinstance(key,float):
            idx = [self.__findFreqIndex(key)] # idx is 1 element list. This guarantees that the shape length of the returned matrices is conserved
            
        elif isinstance(key,tuple) or isinstance(key,list) or isinstance(key,np.ndarray):
            if len(np.array(key).shape) > 1:
                raise IndexError
                
            if (np.unique(np.array(key),return_counts=True)[1] > 1).any():
                raise EM_FieldError("At least one frequency value is repeated among the indices", "__getitem__")
            
            idx = list(map(self.__findFreqIndex, key)) # idx is a list
            
        elif isinstance(key,slice):
            if key.start is None:
                idx0 = 0
            else:
                idx0 = self.__findFreqIndex(key.start)
            if key.stop is None:
                idx1 = self.__n_f
            else:
                idx1 = self.__findFreqIndex(key.stop)
            
            idx = slice(idx0,idx1) # idx is a slice

        if ret_em_field.__e_field is not None:
            ret_em_field.__e_field = ret_em_field.__e_field[idx]
        if ret_em_field.__b_field is not None:
            ret_em_field.__b_field = ret_em_field.__b_field[idx]
        ret_em_field.__f = ret_em_field.__f[idx]

        return ret_em_field
    
    
    def getProperty(self, prop_key):
        
        if not isinstance(prop_key, str):
            raise EM_FieldError("prop_key has to be a string relevant to a key in the properties dictionary", "getProperty")
        if prop_key not in self.__props.keys():
            raise EM_FieldPropertiesError("%s has not been found among the keys of the properties dictionary", "getProperty")
        
        if prop_key == "idxs":
            return self.__props[prop_key]
        else:
            return self.__props[prop_key][self.__props['idxs']]
    
    
    def addProperty(self, prop_key, prop_value):
        
        if prop_key in self.__props:
            raise EM_FieldPropertiesError("The property is already present in the properties dictionary", "addProperty")
            
        new_props = copy(self.__props)
        new_props[prop_key] = prop_value
        
        EM_FieldPropertiesError.check(new_props, "addProperty")
        
        new_props[prop_key] = np.array(new_props[prop_key])
        if prop_key == "idxs": # idxs must be integer type array for proper indexing in other methods
                new_props[prop_key] = new_props[prop_key].astype(int)
                
        self.__props = new_props
        
    
    def maskEMField(self, idx):
        if not isinstance(idx,int):
             raise EM_FieldPropertiesError("idx is supposed to be an integer number", "maskEMField")
        if not self.__props:
            raise EM_FieldPropertiesError("There are no properties defined for the EM_Field instance", "maskEMField")
        if not idx in self.__props["idxs"]:
            raise EM_FieldPropertiesError("The passed index is not present among the indices in the properties dictionary", "maskEMField")
    
        if self.__e_field is not None:
            self.__e_field[:,:,:,self.__props["idxs"]==idx] = np.nan
            
        if self.__b_field is not None:
            self.__b_field[:,:,:,self.__props["idxs"]==idx] = np.nan
            
            
    def compSensitivities(self):
        
        if self.__b_field is None:
            raise EM_FieldError("No b_field property is specified for the EM_Field instance", "compSensitivities")
        
        sens = np.copy(self.__b_field * np.sqrt(2)) # b_field contains rms values of the B field
        sens = np.delete(sens, 2, axis = 2)
        
        b1p = 0.5*(sens[:,:,0,:] + 1j*sens[:,:,1,:])
        b1m = 0.5*np.conj(sens[:,:,0,:] - 1j*sens[:,:,1,:])
        
        sens[:,:,0,:] = b1p
        sens[:,:,1,:] = b1m
        
        return sens
    
    
    def compPowDens(self, elCond_key, p_inc=None):
        
        if not isinstance(elCond_key, str):
            raise EM_FieldError("elCond_key has to be a string relevant to the key of the electrical conductivity in the properties dictionary", "compPowDens")
        if elCond_key not in self.__props.keys():
            raise EM_FieldPropertiesError(f"{elCond_key} has not been found among the keys of the properties dictionary", "compPowDens")
        
        elCond = self.getProperty(elCond_key)
            
        if self.__e_field is None:
            raise EM_FieldError("No e field property is specified for the EM_Field instance. Power density cannot be computed", "compPowDens")
        
        if p_inc is not None: #Power density is computed for a defined supply configuration
            if not isinstance(p_inc, np.ndarray) and not isinstance(p_inc, list):
                raise EM_FieldError("p_inc can only be numpy ndarray or a list", "compPowDens")
            else:
                p_inc = np.array(p_inc)
            if p_inc.size != self.__nPorts:
                raise EM_FieldError("p_inc has to be a self.nPorts length list or numpy ndarray", "compPowDens")
                
            norm = np.sqrt(np.abs(p_inc))*np.exp(1j*np.angle(p_inc))
            efield_new = np.moveaxis(self.e_field,1,-1) #Temporary axis change so field.shape = [self.n_f, 3, self.nPoints, self.nPorts]
            efield_new = efield_new @ norm 
        else:
            efield_new = self.__e_field
        
        powDens = np.linalg.norm(efield_new, axis=-2)**2 * elCond
        
        return powDens
    
    
    def compDepPow(self, voxVols, elCond_key, p_inc=None):
        

        if not isinstance(voxVols, int) and not isinstance(voxVols, float):
            raise EM_FieldError("voxVols has to be int or float", "compDepPow")
            
        powDens = self.compPowDens(elCond_key, p_inc)
            
        depPow = np.nansum(powDens*voxVols, axis=-1)
        
        return depPow
    
    def spatialAverageSAR(self, targetMass, voxVols, elCond_key, massDensity_key, p_inc, freq, step1_libPath, step2_libPath, backgroundIdx = None, additionalBackground=[0,0,0]):

        if not isinstance(elCond_key, str):
            raise EM_FieldError("elCond_key has to be a string relevant to the key of the electrical conductivity in the properties dictionary", "spatialAverageSAR")
        if elCond_key not in self.__props.keys():
            raise EM_FieldPropertiesError("%s has not been found among the keys of the properties dictionary", "spatialAverageSAR")
        
        elCond = self.getProperty(elCond_key)

        if not isinstance(massDensity_key, str):
            raise EM_FieldError("massDensity_key has to be a string relevant to the key of the mass density in the properties dictionary", "spatialAverageSAR")
        if massDensity_key not in self.__props.keys():
            raise EM_FieldPropertiesError(f"{massDensity_key} has not been found among the keys of the properties dictionary", "spatialAverageSAR")
        
        massDensity = self.getProperty(massDensity_key)
            
        if self.__e_field is None:
            raise EM_FieldError("No e field property is specified for the EM_Field instance. Power density cannot be computed", "spatialAverageSAR")
        
        if not isinstance(p_inc, np.ndarray) and not isinstance(p_inc, list):
            raise EM_FieldError("p_inc can only be numpy ndarray or a list", "spatialAverageSAR")
        else:
            p_inc = np.array(p_inc)
        if p_inc.size != self.__nPorts:
            raise EM_FieldError("p_inc has to be a self.nPorts length list or numpy ndarray", "spatialAverageSAR")
        
        f_idx = np.where(self.__f==freq)[0]
        if f_idx.size == 0:
            raise EM_FieldFrequenciesError("No E field for the specified frequency", "compQMatrix")
        else:
            f_idx = f_idx[0]

        # Loading the libraries
        step1_libPath = CDLL(step1_libPath)
        step2_libPath = CDLL(step2_libPath)

        avgSARStep1 = step1_libPath.main
        avgSARStep1.restype = c_int
        avgSARStep2 = step2_libPath.main
        avgSARStep2.restype = c_int

        # Preparing the numpy data
        if backgroundIdx is not None:
            massDensity[self.__props["idxs"]==backgroundIdx] = np.nan # I set to nan density values in background voxels
        massArray = voxVols * massDensity.reshape(self.__nPoints,order='F')

        localSARArray = self.compPowDens(elCond_key, p_inc)[f_idx].reshape(self.__nPoints,order='F')
        localSARArray[np.isnan(massArray)] = np.nan # Not really needeed ...
        additionalBackground = np.array(additionalBackground)
        n_points = self.__nPoints + additionalBackground*2

        voxStatusArray = np.ones_like(massArray, dtype=int) # INVALID=0, UNUSED=1, USED=2, VALID=3

        original_slices = []
        for i in range(3):
            original_slices.append(slice(additionalBackground[i],additionalBackground[i]+self.__nPoints[i]))

        if ((additionalBackground)>0).any():
            
            
            box = np.full(n_points, np.nan)
            box[original_slices[0], original_slices[1], original_slices[2]] = massArray
            massArray = np.copy(box)
            
            box[original_slices[0], original_slices[1], original_slices[2]] = localSARArray
            localSARArray = np.copy(box)
            localSARArray[np.isnan(localSARArray)] = np.nan

            box[original_slices[0], original_slices[1], original_slices[2]] = voxStatusArray
            voxStatusArray = np.copy(box)
        
        voxStatusArray[np.isnan(massArray)] = 0
        voxStatusArray = voxStatusArray.astype(int)

        print(localSARArray.shape)
        print(massArray.shape)
        print(voxStatusArray.shape)

        # ctypes definitions

        voxStatusArray_c = (c_int * voxStatusArray.size)(*voxStatusArray.flatten())
        massArray_c = (c_double * massArray.size)(*massArray.flatten())
        avgSARArray_c = (c_double * massArray.size)(*(np.zeros(massArray.size)))
        localSARArray_c = (c_double * localSARArray.size)(*localSARArray.flatten())
        n_points_c = (c_int * 3)(*n_points)
        targetMass_c = c_double(targetMass)

        # C functions execution
        
        print("Step one in progress ...\n\n")
        ret = avgSARStep1(massArray_c, localSARArray_c, avgSARArray_c, voxStatusArray_c, targetMass_c, n_points_c)

        print("Step two in progress ...\n\n")
        ret = avgSARStep2(massArray_c, localSARArray_c, avgSARArray_c, voxStatusArray_c, targetMass_c, n_points_c)

        avgSARArray = np.array(avgSARArray_c).reshape(n_points)[original_slices[0], original_slices[1], original_slices[2]]
        voxStatusArray = np.array(voxStatusArray_c).reshape(n_points)[original_slices[0], original_slices[1], original_slices[2]].astype(float)
        localSARArray = localSARArray[original_slices[0], original_slices[1], original_slices[2]]

        return avgSARArray, voxStatusArray
    
    def compQMatrix(self, point, freq, z0_ports=50, elCond_key=None):
        
        if self.__e_field is None:
            raise EM_FieldError("No e field property is specified for the EM_Field instance. Power density cannot be computed", "compQMatrix")

        EM_FieldPointsError.check(point, "compQMatrix")
        point = np.array(point)
        

        point_index = point[2]*self.__nPoints[0]*self.__nPoints[1] + point[1]*self.__nPoints[0] + point[0] #index of the selected point according to the 'Fortran' flatten order
        
        f_idx = np.where(self.__f==freq)[0]
        if f_idx.size == 0:
            raise EM_FieldFrequenciesError("No E field for the specified frequency", "compQMatrix")
        else:
            f_idx = f_idx[0]
            

        if elCond_key is not None and not isinstance(elCond_key, str):
            raise EM_FieldError("elCond_key has to be None or a string relevant to the key of the electrical conductivity in the properties dictionary", "compQMatrix")
        if elCond_key is None: #No electrical conductivity is passed as argument and a relevant property is not present
            elCond = 1
        else:
            elCond = self.getProperty(elCond_key)[point_index]
        
        
        if not isinstance(z0_ports, np.ndarray) and not isinstance(z0_ports, list) and not isinstance(z0_ports, int) and not isinstance(z0_ports, float):
            raise EM_FieldError("z0_ports must be a self.nPorts real value elements list or numpy ndarray or single scalar value", "compQMatrix")
        elif isinstance(z0_ports, int) or isinstance(z0_ports, float):
            z0_ports = np.ones(self.__nPorts) * np.abs(z0_ports.real)
        elif len(z0_ports) != self.__nPorts:
            raise EM_FieldError("z0_ports must be a self.nPorts real value elements list or numpy ndarray or single scalar value", "compQMatrix")
        else:
            z0_ports = np.abs(np.array(z0_ports).real)
            
        e_field_pnt = np.copy(self.e_field[f_idx,:,:,point_index]) # e_field in point due to 1 W incident power in relevant ports
        e_field_pnt /= np.sqrt(z0_ports[:,None]) # e_field_pnt is referred to 1 Volt incident voltage in relevant ports
            
        q_matrix = e_field_pnt.conj() @ e_field_pnt.T
        q_matrix *= elCond
        
        return q_matrix

    
    def compVOP(self, freq, u_max_r, z0_ports=50, elCond_key=None, avg_rad=1):
        """
        Implementation of the code suggested in the paper:
        Local Specific Absorption Rate Control for Parallel Transmission by Virtual Observation Points
        (10.1002/mrm.22927)
        
        u_max_r : float
            allowed maximum overestimation of the VOPs with respect to the maximum eigenvalue of the Q matrices of the model (e.g. 0.05 means that the overestimation will be the 5 % of the maximum eigenvalue)
        """
        def volume_average(omega_mat, avg_rad, nPoints):
            """
            
            Parameters
            ----------
            omega_mat : n_points x n_port x n_port numpy ndarray
                n_set of matrices ordered as Fortran (first index faster)
            avg_rad : float
                the ratio between the radius of the averaging sphere and the voxel dimension
            nPoints : 3-element list
                the voxels along each Cartesian direction (prod(nPoints) = n_points)

            Returns
            -------
            avg_omega_mat : n_points x n_port x n_port numpy ndarray
                the Omega matrices averaged inside the given sphere
            """
           
            omega_mat = omega_mat.reshape(list(nPoints)+list(omega_mat.shape[1:]), order='F')
            avg_omega_mat = np.full_like(omega_mat, np.nan)
            index = 0
            x = np.arange(nPoints[0])
            y = np.arange(nPoints[1])
            z = np.arange(nPoints[2])
            
            X, Y, Z = np.meshgrid(x,y,z)
            dist = np.sqrt(X**2 + Y**2 + Z**2)
            
            mask = (dist <= avg_rad)
            print("\nAveraging:\n")
            for k in np.arange(nPoints[2]):
                print("\rPercentage: %d %%" %(k/nPoints[2]*100), end='')
                if k >= avg_rad:
                    k_idxs = slice(k-avg_rad, k+avg_rad)
                else:
                    k_idxs = slice(0, k+avg_rad)
                for j in np.arange(nPoints[1]):
                    if j >= avg_rad:
                        j_idxs = slice(j-avg_rad, j+avg_rad)
                    else:
                        j_idxs = slice(0, j+avg_rad)
                    for i in np.arange(nPoints[0]):
                        if i >= avg_rad:
                            i_idxs = slice(i-avg_rad, i+avg_rad)
                        else:
                            i_idxs = slice(0, i+avg_rad)
                        mask_red = mask[i_idxs, j_idxs, k_idxs]
                        if not np.isnan(omega_mat[i,j,k]).all():
                            avg_omega_mat[i,j,k] = np.nanmean(omega_mat[i_idxs, j_idxs, k_idxs][mask_red], axis=0)
                        index += 1
                        mask = np.roll(mask,1,0)
                    mask = np.roll(mask,1,1)
                mask = np.roll(mask,1,2)
                
            return avg_omega_mat.reshape([-1,omega_mat.shape[-2],omega_mat.shape[-1]], order='F')
        
        def findMatrixA(B_ask, sorted_Omega_mat, u_max):
            """

            Parameters
            ----------
            B_ask : n_port x n_port numpy ndarray
                B* of the reference paper: The matrix of the whole set Omega of the cluster iteration 
                with the highest spectral norm
            sorted_Omega_mat : n_set x n_port x n_port numpy ndarray
                n_set matrices of the Omega set relevant to the cluster iteration and sorted accoridng lambda_min(B*-B) (step 2 of reference paper)
            u_max : float
                the maximum spectral norm of Z* before a new cluster investigation should be performed

            Returns
            -------
            A : n_port x n_port numpy ndarray representing the matrix A that dominates all the matrices in the cluster
            
            cluster_last_index : index of Omega_mat representing the first matrix that cannot be added to the cluster since
                                 it would entail ||Z*||>u

            """

            Z_ask = np.zeros_like(B_ask)
            i = 0
            u = 0
            while u <= u_max and i < sorted_Omega_mat.shape[0]:
                eig_vals, eig_vec = np.linalg.eigh(B_ask - sorted_Omega_mat[i])
                eig_vals[eig_vals>0] = 0
                eig_vals = -1 * eig_vals
                Q_m = eig_vec @ np.diag(eig_vals) @ eig_vec.conj().T
                Z_ask_temp = Z_ask + Q_m
                u = np.linalg.eigh(Z_ask_temp)[0][-1]
                if u <= u_max:
                    Z_ask = Z_ask_temp
                    i +=1
                    
            return B_ask+Z_ask, i

        def sortOmega_mat(unsorted_Omega_mat):
            """

            Parameters
            ----------
            unsorted_Omega_mat : n_set x n_port x n_port numpy ndarray
                n_set matrices of the Omega set relevant to the cluster iteration still unsorted

            Returns
            -------
            B_ask : n_port x n_port numpy ndarray
                B* of the reference paper: The matrix of the whole set Omega of the cluster iteration 
                with the highest spectral norm
            
            sorted_Omega_mat : n_set x n_port x n_port numpy ndarray
               the unsorted_Omega_mat matrices sorted according lambda_min(B*-B) (step 2 of reference paper)
            
            sort_indices : n_set numpy ndarray
                if i=sort_indices[n], sorted_Omega_mat[n] = unsorted_Omega_mat[i]
            """
            
            max_eigvals = np.linalg.eigvalsh(unsorted_Omega_mat)[:,-1] # The array of maximum eigenvalues of the matrices in unsorted_Omega_mat
            B_ask = unsorted_Omega_mat[np.argmax(max_eigvals)]
                
            serv_mats = B_ask[None,...] - unsorted_Omega_mat
            min_serv_eigvals = np.linalg.eigvalsh(serv_mats)[:,0] # The array of minimum eigenvalues of the matrices in unsorted_Omega_mat
            sort_indices = np.argsort(min_serv_eigvals)[::-1]
            sorted_Omega_mat = unsorted_Omega_mat[sort_indices]
            
            return B_ask, sorted_Omega_mat, sort_indices
        
        
        if self.__e_field is None:
            raise EM_FieldError("No e field property is specified for the EM_Field instance. Power density cannot be computed", "compQMatrix")

        f_idx = np.where(self.__f==freq)[0]
        if f_idx.size == 0:
            raise EM_FieldFrequenciesError("No E field for the specified frequency", "compVOP")
        else:
            f_idx = f_idx[0]
             
        if elCond_key is not None and not isinstance(elCond_key, str):
            raise EM_FieldError("elCond_key has to be None or a string relevant to the key of the electrical conductivity in the properties dictionary", "compQMatrix")
        if elCond_key is None: #No electrical conductivity is passed as argument and a relevant property is not present
            elCond = np.ones(np.prod(self.nPoints))
        else:
            elCond = self.getProperty(elCond_key)
            
        if not isinstance(z0_ports, np.ndarray) and not isinstance(z0_ports, list) and not isinstance(z0_ports, int) and not isinstance(z0_ports, float):
            raise EM_FieldError("z0_ports must be a self.nPorts real value elements list or numpy ndarray or single scalar value", "compQMatrix")
        elif isinstance(z0_ports, int) or isinstance(z0_ports, float):
            z0_ports = np.ones(self.__nPorts) * np.abs(z0_ports.real)
        elif len(z0_ports) != self.__nPorts:
            raise EM_FieldError("z0_ports must be a self.nPorts real value elements list or numpy ndarray or single scalar value", "compQMatrix")
        else:
            z0_ports = np.abs(np.array(z0_ports).real)
             
        e_field = np.copy(self.e_field[f_idx,...])
        e_field /= np.sqrt(z0_ports[:,None, None]) # e_field_pnt is referred to 1 Volt incident voltage in relevant ports
            
        q_mats = np.einsum("nkj,mkj->jnm", e_field, e_field.conj())
        
        q_mats *= elCond[:,None, None]
        
        if avg_rad != 1:
            q_mats = volume_average(q_mats, avg_rad, self.nPoints)
        
        u_max = np.nanmax(np.linalg.eigvalsh(q_mats)[:,-1]) * u_max_r
        
        clustered_points = 0 # Number of clustered points
        n_cluster = 0 # Number of identified clusters
        A_mats = np.empty([0,q_mats.shape[1], q_mats.shape[2]])
        cluster_array  = np.full(q_mats.shape[0], np.nan)
        
        not_nan_indices = np.any(np.logical_not(np.isnan(q_mats)),axis=(1,2))
        sorted_Omega_mat = q_mats[not_nan_indices]
        not_nan_cluster_array = np.zeros(sorted_Omega_mat.shape[0], dtype=int)
        
        sort_indices_orig = np.arange(q_mats.shape[0]) # Indices of the sorted Omega matrices with respect to the element of the original_Omega_mat
        last_index_k0 = 0 # Index of the first matrix not clustered in the previous iteration
        
        print("\n\nClusterisation:\n")
        while clustered_points < q_mats[not_nan_indices].shape[0]:
            print("\rCluster n. %d" %n_cluster, end='')
            
            B_ask, sorted_Omega_mat, sort_indices = sortOmega_mat(sorted_Omega_mat[last_index_k0:])
            
            sort_indices_orig = sort_indices_orig[last_index_k0:][sort_indices]
            
            A, last_index_k1 = findMatrixA(B_ask, sorted_Omega_mat, u_max)
            
            A_mats = np.append(A_mats, A[None,...],axis=0)
            
            not_nan_cluster_array[sort_indices_orig[:last_index_k1]] = n_cluster
            clustered_points += last_index_k1
            n_cluster += 1
            last_index_k0= last_index_k1

        cluster_array[not_nan_indices] = not_nan_cluster_array
        
        print("\n\n%d total clusters identified" %n_cluster)
        
        return A_mats, cluster_array
    
    
    def plotProperty(self, prop_key, plane, sliceIdx, vmin=None, vmax=None):
        
        prop = self.getProperty(prop_key) # All the checks are perfomed in this method
        
        if self.__props[prop_key].dtype != int and self.__props[prop_key].dtype != float:
            raise EM_FieldError("Only integer or float properties can be plotted", "plotProperty")
            
        prop = prop.reshape(self.__nPoints, order='F')
        
        fig, ax = plt.subplots(1,1)
        fig.canvas.manager.set_window_title("%s_%s_%d" %(prop_key, plane, sliceIdx))
        fig.suptitle("%s, Index: %d" %(prop_key, sliceIdx))
        
        if plane.lower() == 'xy' or  plane.lower() == 'yz':
            im = ax.imshow(prop[:,:,sliceIdx].T,vmin=vmin,vmax=vmax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif plane.lower() == 'xz' or plane.lower() == 'zx':
            im = ax.imshow(prop[:,sliceIdx,:].T,vmin=vmin,vmax=vmax)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        elif plane.lower() == 'yz' or plane.lower() == 'zy':
            im = ax.imshow(prop[sliceIdx,:,:].T,vmin=vmin,vmax=vmax)
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            
        fig.colorbar(im)
        
        return fig
    

    def plotEMField(self, em_field, comp, freq, ports, plane, sliceIdx, vmin=None, vmax=None):
        
        if em_field.lower() not in ["b_field", "e_field"]:
            raise EM_FieldError("em_field has to be either 'e_field' or 'b_field'", "plotEMField")
        
        if em_field.lower() == "b_field" and self.__b_field is None:
            raise EM_FieldError("No b_field property is specified for the EM_Field instance", "plotEMField")
        if em_field.lower() == "e_field" and self.__e_field is None:
            raise EM_FieldError("No e_field property is specified for the EM_Field instance", "plotEMField")
            
        f_idx = np.where(self.__f==freq)[0]
        if f_idx.size == 0:
            raise EM_FieldError("No EM field is specified for the specified frequency", "plotEMField")
        else:
            f_idx = f_idx[0]
        
        if not isinstance(ports, list) and not isinstance(ports, np.ndarray):
            raise EM_FieldError("ports has to be a 1D list or numpy ndarray", "plotEMField")
        ports = np.sort(np.array(ports) - 1) # -1 since I want to use ports to index the EM field array
        if len(ports.shape) != 1:
            raise EM_FieldError("ports has to be a 1D list or numpy ndarray", "plotEMField")
        for port in ports:
            if port not in np.arange(self.__nPorts):
                raise EM_FieldError("No EM field for the specified port", "plotEMField")

        if comp.lower() not in ['b1+', 'b1-']:
            
            if em_field.lower() == "b_field":
                
                field = self.__b_field[f_idx, ports,:,:]
                
                if comp.lower() == "x":
                    field = 1e6*np.abs(field[:,0,:])
                elif comp.lower() == 'y':
                    field = 1e6*np.abs(field[:,1,:])
                elif comp.lower() == 'z':
                    field = 1e6*np.abs(field[:,2,:])
                elif comp.lower() == 'mag':
                    field = 1e6*np.linalg.norm(field,axis=1)   
                else:
                    raise EM_FieldError("comp must take one of the following values: 'mag', 'x', 'y', 'z', 'b1+', 'b1-'", "plotEMField")
            
            if em_field.lower() == "e_field":
                
                field = self.__e_field[f_idx, ports,:,:]
                
                if comp.lower() == "x":
                    field = np.abs(field[0,:])
                elif comp.lower() == 'y':
                    field = np.abs(field[1,:])
                elif comp.lower() == 'z':
                    field = np.abs(field[2,:])
                elif comp.lower() == 'mag':
                    field = np.linalg.norm(field,axis=1)   
                else:
                    raise EM_FieldError("comp must take one of the following values: 'mag', 'x', 'y', 'z'", "plotEMField")

        else:
            
            if em_field.lower() == "b_field":
                field = self.compSensitivities()
                field = field[f_idx, ports,:,:]
                
                if comp.lower() == "b1+":
                    field = 1e6*np.abs(field[:,0,:])
                else:
                    field = 1e6*np.abs(field[:,1,:])

            else:
                raise EM_FieldError("comp must take one of the following values: 'mag', 'x', 'y', 'z'", "plotEMField")
            
        
        field = field.reshape(np.concatenate( ((ports.size,), self.__nPoints) ), order='F')
        
        n_cols = int(np.ceil(np.sqrt(ports.size)))
        n_rows = int(np.ceil(ports.size/n_cols))
        fig, axs = plt.subplots(n_rows,n_cols)
        axs = np.array(axs).flatten()
        
        fig.canvas.manager.set_window_title("%s_%.2f MHz_%s_%d" %(em_field, freq*1e-6, plane, sliceIdx))
        fig.suptitle("%s, Component: %s, Frequency: %.2f MHz, Index: %d" %(em_field, comp, freq*1e-6, sliceIdx))
        
        for i, ax in enumerate(axs):
            
            if i < ports.size:
                ax = axs[i]
                ax.set_title("Port %d" %(ports[i]+1))
                
                if plane.lower() == 'xy' or plane.lower() == 'yx':
                    im = ax.imshow(field[i,:,:,sliceIdx].T,vmin=vmin,vmax=vmax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                elif plane.lower() == 'xz' or plane.lower() == 'xz':
                    im = ax.imshow(field[i,:,sliceIdx,:].T,vmin=vmin,vmax=vmax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")
                elif plane.lower() == 'yz' or plane.lower() == 'zy':
                    im = ax.imshow(field[i,sliceIdx,:,:].T,vmin=vmin,vmax=vmax)
                    ax.set_xlabel("y")
                    ax.set_ylabel("z")
            else:
                fig.delaxes(ax)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        
        if em_field.lower() == "b_field":
            cbar.ax.set_ylabel("B field ($\mu$T)")
        else:
            cbar.ax.set_ylabel("E field (V/m)")
        
        return fig
    
    def exportXMF(self, filename):
        
        dx=dy=dz=1
        nx = self.__nPoints[0]
        ny = self.__nPoints[1]
        nz = self.__nPoints[2]
        
        n_elem = (nx*ny*nz)
        n_nodes = (nx+1) * (ny+1) * (nz+1)
        
        nodes = np.arange(n_nodes).reshape((nx+1,ny+1,nz+1),order="F")
        
        connections = np.empty((n_elem,8))
        
        print("Arranging connection data...\n")
        
        #Fortan order
        r = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    print("\r%.2f %%" %(r/n_elem*100), end='')
                    connections[r] = nodes[i:i+2,j:j+2,k:k+2].ravel('F')
                    r += 1

        #Swap elements
        connections[:,[2,3]] = connections[:,[3,2]]
        connections[:,[6,7]] = connections[:,[7,6]]
        
        x = dx*np.tile(np.tile(np.arange(nx+1),ny+1),nz+1)
        y = dy*np.tile(np.arange(ny+1).repeat(nx+1),nz+1)
        z = dz*np.arange(nz+1).repeat((nx+1)*(ny+1))
        
        print("\n\n\nCompiling .xmf file...\n\n")
        
        try:
            if not filename:
                raise EM_FieldIOError("Please, provide a correct filename", "exportXMF")
            with open(filename+".xmf", "w") as f:
                f.write('<?xml version="1.0" ?>\n')
                f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                f.write('<Xdmf Version="2.0">\n')
                f.write('<Domain>\n')
                f.write('<Grid Name = "VoxelModel" GridType = "Uniform">\n')
                f.write('<Topology TopologyType="Hexahedron" NumberOfElements="%d" BaseOffset="0">\n'%n_elem)
                f.write('<DataStructure Format="HDF" Dimensions="%d 8" DataType="Int" Precision="8">\n'%n_elem)
                f.write('%s.h5:/Mesh/Connections\n'%(filename))
                f.write('</DataStructure>\n')
                f.write('</Topology>\n')
                f.write('<Geometry GeometryType="X_Y_Z">\n')
                f.write('<DataItem Name="X" Dimensions="%d" NumberType="Float" Precision="4" Format="HDF">\n'%n_nodes)
                f.write('%s.h5:/Mesh/Nodes_X\n'%(filename))
                f.write('</DataItem>\n')
                f.write('<DataItem Name="Y" Dimensions="%d" NumberType="Float" Precision="4" Format="HDF">\n'%n_nodes)
                f.write('%s.h5:/Mesh/Nodes_Y\n'%(filename))
                f.write('</DataItem>\n')
                f.write('<DataItem Name="Z" Dimensions="%d" NumberType="Float" Precision="4" Format="HDF">\n'%n_nodes)
                f.write('%s.h5:/Mesh/Nodes_Z\n'%(filename))
                f.write('</DataItem>\n')
                f.write('</Geometry>\n')
    
                for freq in self.__f:
                    for port in range(self.__nPorts):
                        if self.__b_field is not None:
                            f.write('<Attribute Type="Vector" Center="Cell" Name="%dMHz-p%d-B_real">\n'%(freq/1e6,port+1))
                            f.write('<DataItem Format="HDF" Dimensions="%d 3" DataType="Float" Precision="8">\n'%n_elem)
                            f.write('%s.h5:/%d_MHz/Port_%d/Breal\n'%(filename, freq/1e6, port+1))
                            f.write('</DataItem>\n')
                            f.write('</Attribute>\n')
                            f.write('<Attribute Type="Vector" Center="Cell" Name="%dMHz-p%d-B_imag">\n'%(freq/1e6,port+1))
                            f.write('<DataItem Format="HDF" Dimensions="%d 3" DataType="Float" Precision="8">\n'%n_elem)
                            f.write('%s.h5:/%d_MHz/Port_%d/Bimag\n'%(filename, freq/1e6, port+1))
                            f.write('</DataItem>\n')
                            f.write('</Attribute>\n')
                        if self.__e_field is not None:    
                            f.write('<Attribute Type="Vector" Center="Cell" Name="%dMHz-p%d-E_real">\n'%(freq/1e6,port+1))
                            f.write('<DataItem Format="HDF" Dimensions="%d 3" DataType="Float" Precision="8">\n'%n_elem)
                            f.write('%s.h5:/%d_MHz/Port_%d/Ereal\n'%(filename, freq/1e6, port+1))
                            f.write('</DataItem>\n')
                            f.write('</Attribute>\n')
                            f.write('<Attribute Type="Vector" Center="Cell" Name="%dMHz-p%d-E_imag">\n'%(freq/1e6,port+1))
                            f.write('<DataItem Format="HDF" Dimensions="%d 3" DataType="Float" Precision="8">\n'%n_elem)
                            f.write('%s.h5:/%d_MHz/Port_%d/Eimag\n'%(filename, freq/1e6, port+1))
                            f.write('</DataItem>\n')
                            f.write('</Attribute>\n')
                
                for p in self.__props:
                    if p != "idxs":
                        f.write('<Attribute Type="Scalar" Center="Cell" Name="%s">\n'%p)
                        f.write('<DataItem Format="HDF" Dimensions="%d" DataType="Float" Precision="8">\n'%n_elem)
                        f.write('%s.h5:/Properties/%s\n'%(filename, p))
                        f.write('</DataItem>\n')
                        f.write('</Attribute>\n')
    
                f.write('</Grid>\n')
                f.write('</Domain>\n')
                f.write('</Xdmf>\n')
            
            print("Compiling .h5py file...\n\n")
            
            with h5py.File(filename+".h5", "w") as f:
                f["Mesh/Connections"] = connections
                f["Mesh/Nodes_X"] = x
                f["Mesh/Nodes_Y"] = y
                f["Mesh/Nodes_Z"] = z
    
                for i,freq in enumerate(self.__f):
                    for port in range(self.__nPorts):
                        if self.__b_field is not None:
                            #Substitute nan values with zero
                            b_field = np.copy(self.__b_field)
                            b_field[np.isnan(b_field)] = 0+0j
                            f["%d_MHz/Port_%d/Breal"%(freq/1e6, port+1)] = np.real(b_field[i,port]).T
                            f["%d_MHz/Port_%d/Bimag"%(freq/1e6, port+1)] = np.imag(b_field[i,port]).T
                        if self.__e_field is not None:
                            #Substitute nan values with zero
                            e_field = np.copy(self.__e_field)
                            e_field[np.isnan(e_field)] = 0+0j
                            f["%d_MHz/Port_%d/Ereal"%(freq/1e6, port+1)] = np.real(e_field[i,port]).T
                            f["%d_MHz/Port_%d/Eimag"%(freq/1e6, port+1)] = np.imag(e_field[i,port]).T
                for p in self.__props:
                    if p != "idxs":
                        f["Properties/%s"%p] = self.getProperty(p)
        
        except Exception as e:
            if not isinstance(e, EM_FieldIOError): # I cast as EM_FieldIOError all other errors
                raise EM_FieldIOError(e.args[-1], "exportXMF")
            else:
                raise e
                
    def _newFieldComp(self, p_incM, phaseM):
        
        if p_incM.shape != phaseM.shape:
            raise EM_FieldError("p_incM and phaseM arrays are not coherent", "_newFieldComp")
        if p_incM.shape[2] != self.__nPorts or phaseM.shape[2] != self.__nPorts:
            raise EM_FieldError("The number of ports has to be equal to p_incM and phase third dimension", "_newFieldComp")
        if p_incM.shape[0] != self.__n_f or phaseM.shape[0] != self.__n_f:
            raise EM_FieldError("The number of frequencies of self has to be equal to p_incM and phaseM first dimension", "_newFieldComp")
        
        norm = np.sqrt(p_incM)*np.exp(phaseM*1j)
        #Move norm axis to obtain norm.shape = [self.n_f, self.nPorts, output.nPorts]
        norm = np.moveaxis(norm,1,-1)
        norm = np.expand_dims(norm,1)#norm.shape = [self.n_f,1, self.nPorts, output.nPorts]
        norm = np.repeat(norm,3,1)#norm.shape = [self.n_f,3, self.nPorts, output.nPorts]

        efield_new = None
        bfield_new = None
        
        if self.e_field is not None:
            
            #Temporary axis change so field.shape = [self.n_f, 3, self.nPoints, self.nPorts]
            efield_new = np.moveaxis(self.e_field,1,-1)
            efield_new = efield_new @ norm
            efield_new = np.moveaxis(efield_new,-1,1)
       
        if self.b_field is not None:
            
            #Temporary axis change so field.shape = [self.n_f, 3, self.nPoints, self.nPorts]
            bfield_new = np.moveaxis(self.b_field,1,-1)
            bfield_new = bfield_new @ norm
            bfield_new = np.moveaxis(bfield_new,-1,1)

        return EM_Field(self.__f, self.__nPoints, bfield_new, efield_new, self.__props)
            
    
    def __findFreqIndex(self, freq):
        
        if freq in self.__f:
            idx = np.where(self.__f == freq)[0][0]
        else:
            idx = np.argmin(np.abs(self.__f - freq))
            warnings.warn("%e Hz is not contained in the frequencies list. %e Hz is returned instead" %(freq, self.__f[idx]))
        
        return idx
    
    
    @classmethod
    def importFields_cst(cls, directory, freqUnit="MHz", eFieldRefString="efield_<f>_port<p>.h5", bFieldRefString="bfield_<f>_port<p>.h5", nPoints=None, Pinc_ref=1, b_multCoeff=1, pkORrms='pk', imp_efield=True, imp_bfield=True, fileType = 'ascii', col_ascii_order = 0, props={}):

        if not imp_efield and not imp_bfield:
            raise EM_FieldIOError("At least one among imp_efield and imp_bfield has to be True")
        if nPoints is not None: 
            EM_FieldPointsError.check(nPoints, "importFields_cst")
        if freqUnit.lower() not in ["hz", "khz", "mhz", "ghz"]:
            raise  EM_FieldIOError("freqUnit can only be 'Hz', 'kHz', 'MHz' or 'GHz'")
        if fileType.lower() not in ["ascii", "hdf5"]:
            raise  EM_FieldIOError("fileType can only be 'ascii' or 'hdf5'")
        if pkORrms.lower() not in ["pk", "rms"]:
            raise  EM_FieldIOError("pkORrms can only be 'pk or 'rms'")
        if col_ascii_order not in [0, 1]:
            raise  EM_FieldIOError("col_ascii_order can take 0 (Re_x, Re_y, Re_z, Im_x, Im_y, Im_z) or 1 (Re_x, Im_x, Re_y, Im_y, Re_z, Im_z) values")
        
        try:
            # Retrieving frequency and ports from files
            if imp_efield:
                efield_freqs, efield_n_ports, efield_filenames = cls.__readPortFreqsFromFilenames(directory, eFieldRefString)
            if imp_bfield:
                bfield_freqs, bfield_n_ports, bfield_filenames = cls.__readPortFreqsFromFilenames(directory, bFieldRefString)

            if imp_efield and imp_bfield:
                if not np.equal(efield_freqs, bfield_freqs).all():
                    raise EM_FieldIOError("The electric field and magnetic field have to be defined at the same frequency values", "importFields_cst")

            if imp_efield:
                nPorts = efield_n_ports
                freqs = efield_freqs
            else:
                nPorts = bfield_n_ports
                freqs = bfield_freqs

            if nPoints is None and fileType == "ascii": #I try to evaluate nPoints
                
                if imp_efield:
                    x,y,z = np.loadtxt(efield_filenames[0,0], skiprows=2, unpack=True, usecols=(0,1,2))
                else:
                    x,y,z = np.loadtxt(bfield_filenames[0,0], skiprows=2, unpack=True, usecols=(0,1,2))
                
                orig_len = len(x) #Total number of points
                
                x = np.unique(x)
                y = np.unique(y)
                z = np.unique(z)
                
                nPoints = [len(x), len(y), len(z)]
                
                if np.prod(nPoints) != orig_len:
                    raise EM_FieldIOError("nPoints evaluation failed. Please specify its value in the method argument", "importFields_cst")
    
            elif nPoints is None and fileType == "hdf5":
                if imp_efield:
                    filename = efield_filenames[0,0]
                else:
                    filename = bfield_filenames[0,0]
                    
                with h5py.File(filename, "r") as f:
                    x = np.array(f['Mesh line x'])
                    y = np.array(f['Mesh line y'])
                    z = np.array(f['Mesh line z'])
                
                nPoints = [len(x), len(y), len(z)]
                
            n = np.prod(nPoints)
            
            if imp_efield:
                e_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
            else:
                e_field = None
            if imp_bfield:
                b_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
            else:
                b_field = None
            
            if pkORrms.lower() == "pk":
                rmsCoeff = 1/np.sqrt(2)
            else:
                rmsCoeff = 1 
        
        
            if fileType.lower() == 'ascii':
                for idx_f, f in enumerate(freqs):
                    print("Importing %s MHz fields\n"%f)
        
                    for port in range(nPorts):
                        print("\r\tImporting port%d fields"%(port+1), end='', flush=True)
                        if col_ascii_order == 0:
                            re_cols = (3,4,5)
                            im_cols = (6,7,8)
                        elif col_ascii_order == 1:
                            re_cols = (3,5,7)
                            im_cols = (4,6,8)
                        if imp_efield:
                            e_real = np.loadtxt(efield_filenames[idx_f,port], skiprows=2, usecols=re_cols)
                            e_imag = np.loadtxt(efield_filenames[idx_f,port], skiprows=2, usecols=im_cols)
                            if e_real.shape[0] != n:
                                raise EM_FieldIOError("At least one of e_field files is not compatible with the evaluated or passed nPoints", "importFields_cst")
                            e_field[idx_f, port, :, :] = (e_real+1j*e_imag).T
                        if imp_bfield:
                            b_real = np.loadtxt(bfield_filenames[idx_f,port], skiprows=2, usecols=re_cols)
                            b_imag = np.loadtxt(bfield_filenames[idx_f,port], skiprows=2, usecols=im_cols)
                            if b_real.shape[0] != n:
                                raise EM_FieldIOError("At least one of b_field files is not compatible with the evaluated or passed nPoints", "importFields_cst")
                            b_field[idx_f, port, :, :] = (b_real+1j*b_imag).T
                    
                    print("\n")
            
            elif fileType.lower() == 'hdf5':
                for idx_f, f in enumerate(freqs):
                    print("Importing %s MHz fields\n"%f)
        
                    for port in range(nPorts):
                        print("\r\tImporting port%d fields"%(port+1), end='', flush=True)
                        if imp_efield:
                            
                            filename = efield_filenames[idx_f,port]
                            with h5py.File(filename, "r") as field_file:
                                e_field_raw = np.array(field_file['E-Field'])
                                x = np.array(field_file['Mesh line x'])
                                y = np.array(field_file['Mesh line y'])
                                z = np.array(field_file['Mesh line z'])
                            
                            if (len(x) * len(y) * len(z)) != n:
                                raise EM_FieldIOError("At least one of e_field files is not compatible with the evaluated or passed nPoints", "importFields_cst")
                            
                            e_flat = e_field_raw.flatten() #Flatted array (x,y,z to be reshaped as Fortran order). Each element is a np.void type made of three (x-, y-, z-component) couple of float representing real and imaginary parts of that component
                            e_flat = np.array(e_flat.tolist()) #Array n_points, 3 (components), 2 (real and imaginary)
                            e_field[idx_f, port, :, :] = (e_flat[:,:,0] + 1j*e_flat[:,:,1]).T
                        
                        if imp_bfield:
                            
                            filename = bfield_filenames[idx_f,port]
                            with h5py.File(filename, "r") as field_file:
                                b_field_raw = np.array(field_file['H-Field']) #b_field is an H field and will become  field when multiplied by b_multCoeff
                                x = np.array(field_file['Mesh line x'])
                                y = np.array(field_file['Mesh line y'])
                                z = np.array(field_file['Mesh line z'])
                            
                            if (len(x) * len(y) * len(z)) != n:
                                raise EM_FieldIOError("At least one of b_field files is not compatible with the evaluated or passed nPoints", "importFields_cst")
                            
                            b_flat = b_field_raw.flatten() #Flatted array (x,y,z to be reshaped as Fortran order). Each element is a np.void type made of three (x-, y-, z-component) couple of float representing real and imaginary parts of that component
                            b_flat = np.array(b_flat.tolist()) #Array n_points, 3 (components), 2 (real and imaginary)
                            b_field[idx_f, port, :, :] = (b_flat[:,:,0] + 1j*b_flat[:,:,1]).T
                        
                    print("\n")

            if imp_efield:
                e_field = np.sqrt(1/Pinc_ref) * rmsCoeff * e_field #cst exported field values are peak values
            if imp_bfield:
                b_field = b_multCoeff * np.sqrt(1/Pinc_ref) * rmsCoeff * b_field #cst exported field values are peak values
            
            if freqUnit.lower() == "hz":
                f_multiplier = 1
            elif freqUnit.lower() == "khz":
                f_multiplier = 1e3
            elif freqUnit.lower() == "mhz":
                f_multiplier = 1e6
            elif freqUnit.lower() == "ghz":
                f_multiplier = 1e9

            freqs = f_multiplier*np.array(freqs).astype(float)
            
            return cls(freqs, nPoints, b_field, e_field, props)
        
        except Exception as e:
            if not isinstance(e, EM_FieldIOError): # I cast as EM_FieldIOError all other errors
                raise EM_FieldIOError(e.args[-1], "importFields_cst")
            else:
                raise e

    @classmethod
    def importFields_s4l(cls, directory, freqs, nPorts, Pinc_ref=1, b_multCoeff=1, pkORrms='pk', imp_efield=True, imp_bfield=True, props={}):
        
        if not imp_efield and not imp_bfield:
            raise EM_FieldIOError("At least one among imp_efield and imp_bfield has to be True", "importFields_s4l")
        if not isinstance(nPorts, int) or nPorts < 1:
            raise  EM_FieldIOError("nPorts has to be an integer higher than zero", "importFields_s4l")
        if pkORrms.lower() not in ["pk", "rms"]:
            raise  EM_FieldIOError("pkORrms can only be 'pk or 'rms'", "importFields_s4l")
        
        if pkORrms.lower() == "pk":
            rmsCoeff = 1/np.sqrt(2)
        else:
            rmsCoeff = 1 
        
        if not imp_efield:
            e_field = None
            
        if not imp_bfield:
            b_field = None
        
        try:
            for port in range(nPorts):
                
                print("\rImporting port%d fields"%(port+1), end='', flush=True)
                
                if imp_efield:    
                    
                    data = loadmat(directory+"/efield_port%d.mat"%(port+1))
                    
                    if port == 0:
                        if data["Axis0"].shape[-1] * data["Axis1"].shape[-1] * data["Axis2"].shape[-1] == data["Snapshot0"].shape[0]:
                            n = data["Axis0"].shape[-1] * data["Axis1"].shape[-1] * data["Axis2"].shape[-1]
                            nPoints = [data["Axis0"].shape[-1], data["Axis1"].shape[-1], data["Axis2"].shape[-1]]
                        else:
                            n = (data["Axis0"].shape[-1]-1) * (data["Axis1"].shape[-1]-1) * (data["Axis2"].shape[-1]-1)
                            nPoints = [data["Axis0"].shape[-1]-1, data["Axis1"].shape[-1]-1, data["Axis2"].shape[-1]-1]
    
                        e_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
                    
                    for f in range(len(freqs)):
                        e_field[f,port,:,:] = np.moveaxis(data["Snapshot%d"%f],-1,0)
                        
                if imp_bfield:    
                    
                    data = loadmat(directory+"/bfield_port%d.mat"%(port+1))
                    
                    if port == 0:
                        if data["Axis0"].shape[-1] * data["Axis1"].shape[-1] * data["Axis2"].shape[-1] == data["Snapshot0"].shape[0]:
                            n = data["Axis0"].shape[-1] * data["Axis1"].shape[-1] * data["Axis2"].shape[-1]
                            nPoints = [data["Axis0"].shape[-1], data["Axis1"].shape[-1], data["Axis2"].shape[-1]]
                        else:
                            n = (data["Axis0"].shape[-1]-1) * (data["Axis1"].shape[-1]-1) * (data["Axis2"].shape[-1]-1)
                            nPoints = [data["Axis0"].shape[-1]-1, data["Axis1"].shape[-1]-1, data["Axis2"].shape[-1]-1]
    
                        b_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
                    
                    for f in range(len(freqs)):
                        b_field[f,port,:,:] = np.moveaxis(data["Snapshot%d"%f],-1,0)
                    
            if imp_efield:
                e_field = np.sqrt(1/Pinc_ref) * rmsCoeff * e_field
            if imp_bfield:
                b_field = b_multCoeff * np.sqrt(1/Pinc_ref) * rmsCoeff * b_field
            
            
            return cls(freqs, nPoints, b_field, e_field, props)
        
        except Exception as e:
            if not isinstance(e, EM_FieldIOError): # I cast as EM_FieldIOError all other errors
                raise EM_FieldIOError(e.args[-1], "importFields_s4l")
            else:
                raise e
            
    @classmethod
    def importFields_hfss(cls, directory, freqUnit="MHz", eFieldRefString="efield_<f>_port<p>.fld", bFieldRefString="bfield_<f>_port<p>.fld", nPoints=None, Pinc_ref=1, b_multCoeff=1, pkORrms='pk', imp_efield=True, imp_bfield=True, col_ascii_order = 1, props={}):

        if not imp_efield and not imp_bfield:
            raise EM_FieldIOError("At least one among imp_efield and imp_bfield has to be True")
        if nPoints is not None: 
            EM_FieldPointsError.check(nPoints, "importFields_hfss")
        if freqUnit.lower() not in ["hz", "khz", "mhz", "ghz"]:
            raise  EM_FieldIOError("freqUnit can only be 'Hz', 'kHz', 'MHz' or 'GHz'")
        if pkORrms.lower() not in ["pk", "rms"]:
            raise  EM_FieldIOError("pkORrms can only be 'pk' or 'rms'", "importFields_hfss")
        if col_ascii_order not in [0, 1]:
            raise  EM_FieldIOError("col_ascii_order can take 0 (Re_x, Re_y, Re_z, Im_x, Im_y, Im_z) or 1 (Re_x, Im_x, Re_y, Im_y, Re_z, Im_z) values", "importFields_hfss")
        
        try:
            # Retrieving frequency and ports from files
            if imp_efield:
                efield_freqs, efield_n_ports, efield_filenames = cls.__readPortFreqsFromFilenames(directory, eFieldRefString)
            if imp_bfield:
                bfield_freqs, bfield_n_ports, bfield_filenames = cls.__readPortFreqsFromFilenames(directory, bFieldRefString)

            if imp_efield and imp_bfield:
                if not np.equal(efield_freqs, bfield_freqs).all():
                    raise EM_FieldIOError("The electric field and magnetic field have to be defined at the same frequency values", "importFields_hfss")

            if imp_efield:
                nPorts = efield_n_ports
                freqs = efield_freqs
            else:
                nPorts = bfield_n_ports
                freqs = bfield_freqs

            if nPoints is None: #I try to evaluate nPoints
                
                if imp_efield:
                    x,y,z = np.loadtxt(efield_filenames[0,0], skiprows=2, unpack=True, usecols=(0,1,2))
                else:
                    x,y,z = np.loadtxt(bfield_filenames[0,0], skiprows=2, unpack=True, usecols=(0,1,2))
                
                orig_len = len(x) #Total number of points
                
                x = np.unique(x)
                y = np.unique(y)
                z = np.unique(z)
                
                nPoints = [len(x), len(y), len(z)]
                
                if np.prod(nPoints) != orig_len:
                    raise EM_FieldIOError("nPoints evaluation failed. Please specify its value in the method argument", "importFields_hfss")
    
            n = np.prod(nPoints)
            
            if imp_efield:
                e_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
            else:
                e_field = None
            if imp_bfield:
                b_field = np.empty((len(freqs),nPorts,3,n), dtype="complex")
            else:
                b_field = None
            
            if pkORrms.lower() == "pk":
                rmsCoeff = 1/np.sqrt(2)
            else:
                rmsCoeff = 1 
        
            for idx_f, f in enumerate(freqs):
                print(f"Importing {f} {freqUnit} fields\n")
    
                for port in range(nPorts):
                    print("\r\tImporting port%d fields"%(port+1), end='', flush=True)
                    if col_ascii_order == 0:
                        re_cols = (3,4,5)
                        im_cols = (6,7,8)
                    elif col_ascii_order == 1:
                        re_cols = (3,5,7)
                        im_cols = (4,6,8)
                    if imp_efield:
                        e_real = np.loadtxt(efield_filenames[idx_f,port], skiprows=2, usecols=re_cols)
                        e_imag = np.loadtxt(efield_filenames[idx_f,port], skiprows=2, usecols=im_cols)
                        if e_real.shape[0] != n:
                            raise EM_FieldIOError("At least one of e_field files is not compatible with the evaluated or passed nPoints", "importFields_hfss")
                        e_field[idx_f, port, :, :] = (e_real+1j*e_imag).T
                    if imp_bfield:
                        b_real = np.loadtxt(bfield_filenames[idx_f,port], skiprows=2, usecols=re_cols)
                        b_imag = np.loadtxt(bfield_filenames[idx_f,port], skiprows=2, usecols=im_cols)
                        if b_real.shape[0] != n:
                            raise EM_FieldIOError("At least one of b_field files is not compatible with the evaluated or passed nPoints", "importFields_hfss")
                        b_field[idx_f, port, :, :] = (b_real+1j*b_imag).T
                
                print("\n")
            
            if imp_efield:
                e_field = e_field.reshape(tuple(e_field.shape[:-1])+tuple(nPoints))
                e_field = e_field.swapaxes(-3,-1)
                e_field = e_field.reshape(tuple(e_field.shape[:3])+tuple([-1]), order='C')
                e_field = np.sqrt(1/Pinc_ref) * rmsCoeff * e_field
            if imp_bfield:
                b_field = b_field.reshape(tuple(b_field.shape[:-1])+tuple(nPoints), order='C')
                b_field = b_field.swapaxes(-3,-1)
                b_field = b_field.reshape(tuple(b_field.shape[:3])+tuple([-1]), order='C')
                b_field = b_multCoeff * np.sqrt(1/Pinc_ref) * rmsCoeff * b_field
            
            if freqUnit.lower() == "hz":
                f_multiplier = 1
            elif freqUnit.lower() == "khz":
                f_multiplier = 1e3
            elif freqUnit.lower() == "mhz":
                f_multiplier = 1e6
            elif freqUnit.lower() == "ghz":
                f_multiplier = 1e9

            freqs = f_multiplier*np.array(freqs).astype(float)
            
            return cls(freqs, nPoints, b_field, e_field, props)
        
        except Exception as e:
            if not isinstance(e, EM_FieldIOError): # I cast as EM_FieldIOError all other errors
                raise EM_FieldIOError(e.args[-1], "importFields_hfss")
            else:
                raise e
            
    @classmethod
    def __readPortFreqsFromFilenames(cls, directory, referenceString):
        """Reads the frequency values and number of ports from filenames following the sintax specified in referenceString

        Args:
            directory (string): The path with the files containing the EM field
            referenceString (string): The string with the sintax with which the EM field files have to be searched. <f> is for frequency value and <p> is for the port number

        Returns:
            freqs (numpy ndarray): Numpy ndarray with the frequency values
            n_ports (int): The number of ports
            filenames (numpy ndarray): nf x np numpy ndarray of strings listing the filenames to be loaded according to the information retreived. nf is the number of frequency values and np is the number of ports
        """
        # Preliminary info about the reference string

        try:
            pattern=re.compile("<p>")
            span_p = pattern.search(referenceString)
            if span_p is None:
                raise EM_FieldIOError("The <p> has to be present into the reference string", "__readPortFreqsFromFilenames")
            span_p = span_p.span()
            
            pattern=re.compile("<f>")
            span_f = pattern.search(referenceString)
            if span_f is None:
                raise EM_FieldIOError("The <f> has to be present into the reference string", "__readPortFreqsFromFilenames")
            span_f = span_f.span()

            if span_f[0] < span_p[0]:
                fp_order = 0 # First the frequency and then the port number
                if referenceString[span_f[1]:span_p[0]].isnumeric() or referenceString[span_f[1]:span_p[0]] == "":
                    raise EM_FieldIOError("<f> and <p> should be separated by at least one non numerical character", "__readPortFreqsFromFilenames")
            else:
                fp_order = 1 # First the port number and then the frequency
                if referenceString[span_p[1]:span_f[0]].isnumeric() or referenceString[span_p[1]:span_f[0]] == "":
                    raise EM_FieldIOError("<p> and <f> should be separated by at least one non numerical character", "__readPortFreqsFromFilenames")

            # Recover number of ports and frequency values

            a_candidates = []
            b_candidates = []
            decStrings = re.split("<f>|<p>", referenceString) # e.g., ["bField_," "MHz_Port", ".t"]
            reg = re.compile(r'(%s)\d+(.\d+)?(%s)\d+(.\d+)?'%(decStrings[0], decStrings[1])) # e.g., reg = re.compile(r'(bField_)\d+(.\d)*(MHz_Port)\d+(.\d)*')
                                                                                               # \d+(.\d+)? means at least one decimal and 0 or one repetition of a point with numbers (e.g., 4; 4.0; 44.00 ...)
            for fileDirName in listdir(directory):
                if isfile(join(directory,fileDirName)):
                    match = reg.match(fileDirName)
                    if  match is not None: # True if the filename has the format specified by the referenceString
                        a_candidates.append(fileDirName[match.span(1)[1]:match.span(3)[0]]) # From the end of the first group of match to the start of the third group of match 
                        b_candidates.append(fileDirName[match.span(3)[1]:match.span(0)[1]]) # From the end of the third group of match to the end of the last element of match 

            if not a_candidates or not b_candidates:
                if not a_candidates and not b_candidates:
                    raise EM_FieldIOError("None of the files in the indicated directory comply with the reference string. Please, check the latter", "__readPortFreqsFromFilenames")


            if fp_order == 0:
                freq_string_candidates = np.array(a_candidates)
                freq_candidates = np.array([float(i) for i in (freq_string_candidates)])
                port_candidates = np.array([int(i) for i in (b_candidates)])
            else:
                port_candidates = np.array([int(i) for i in (a_candidates)])
                freq_string_candidates = np.array(b_candidates)
                freq_candidates = np.array([float(i) for i in (freq_string_candidates)])

            freqs, freq_idxs = np.unique(freq_candidates, return_index=True)
            n_ports = np.max(port_candidates)


            #  Print Result

            print(f"\n{referenceString}:\nN° detected frequency values: {freqs.size}\nN° detected ports: {n_ports}\n") 


            # Create filenames to be loaded

            filenames = np.empty((freqs.size,n_ports),dtype=object)

            for i,f in enumerate(freq_string_candidates[freq_idxs]):
                for p in range(n_ports):
                    filenames[i,p] = join(directory,referenceString.replace("<f>",f).replace("<p>",str(p+1)))

            return freqs, n_ports, filenames
       
        except Exception as e:
            if not isinstance(e, EM_FieldIOError): # I cast as EM_FieldIOError all other errors
                raise EM_FieldIOError(e.args[-1], "__readPortFreqsFromFilenames")
            else:
                raise e