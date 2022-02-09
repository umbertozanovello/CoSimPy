import numpy as np
from .S_Matrix import S_Matrix
from .EM_Field import EM_Field
import struct

class RF_Coil():

    def __init__(self, s_matrix, em_field=None):

        if not isinstance(s_matrix, S_Matrix):
            raise TypeError("s_matrix must be an instance of S_Matrix class")
        
        if em_field is not None:
            if not isinstance(em_field, EM_Field):
                raise TypeError("em_field must either be None or an instance of EM_Field class")
            if em_field.nPorts != s_matrix.nPorts:
                raise ValueError("The S matrix and the em_field are not compatible. There is not a e_field and b_field distribution for each port of the S matrix")
            if not all(elem in s_matrix.frequencies for elem in em_field.frequencies):
                raise ValueError("One or more frequencies at which the em_field is computed, differ from the frequencies at which the S matrix is defined")
        
        self.__s_matrix = s_matrix
        self.__em_field = em_field
    
    
    @property
    def em_field(self):
        return self.__em_field
    
    
    @property
    def s_matrix(self):
        return self.__s_matrix
    
    
    def __repr__(self):
        string = '       """""""""""""""\n           RF COIL\n       """""""""""""""\n\n\n\n'
        string+= '"""""""""""""""\n   S MATRIX\n"""""""""""""""\n\n'
        string += "|V-| = |S||V+|\n|%d x 1| = |%d x %d||%d x 1|\n\nNumber of frequency values = %d\n\n"%(self.s_matrix.nPorts,self.s_matrix.nPorts,self.s_matrix.nPorts,self.s_matrix.nPorts, self.s_matrix.n_f)
        string+= '"""""""""""""""\n   EM FIELD\n"""""""""""""""\n\n'
        if self.__em_field is not None:
            string += "Number of frequency values = %d\n\nNumber of points = (%d, %d, %d)\n\n"%(self.__em_field.n_f,self.__em_field.nPoints[0], self.__em_field.nPoints[1], self.__em_field.nPoints[2])
            if self.__em_field.e_field is None:
                string += "E field not defined\n\n"
            elif self.__em_field.b_field is None:
                string += "B field not defined\n\n"
        else:
            string += 'Not defined\n\n'
            
        if self.__s_matrix._S0 is not None:
            string += "The RF coil is the result of previous connections and/or manipulations with a %d ports original RF coil\n\n"%self.__s_matrix._S0.nPorts
        return string
   
    
    def singlePortConnRFcoil(self, networks, comp_Pinc=False):
        
        rets = self.s_matrix._singlePortConnSMatrix(networks, comp_Pinc)
        
        if not comp_Pinc or self.em_field is None:
            #If comp_Pinc is True but the em_field property is None, 
            #the S_Matrix method returns the Pinc and phase matrices. These will be discared
            
            return RF_Coil(rets[0], None)
        
        else:
            
            idxs = np.where(self.s_matrix.frequencies[:,None]==self.em_field.frequencies)[0]
            p_inc, phase = rets[1:]
            p_inc = p_inc[idxs]
            phase = phase[idxs]
            em_field_new = self.em_field._newFieldComp(p_inc, phase)
            
            return RF_Coil(rets[0], em_field_new)
        
                
    def fullPortsConnRFcoil(self, s_matrix_other, comp_Pinc=True):
        
        rets = self.s_matrix._fullPortsConnSMatrix(s_matrix_other, comp_Pinc)
        
        if not comp_Pinc or self.em_field is None:
            #If comp_Pinc is True but the em_field property is None, 
            #the S_Matrix method returns the Pinc and phase matrices. These will be discared
            
            return RF_Coil(rets[0], None)
        
        else:
            
            idxs = np.where(self.s_matrix.frequencies[:,None]==self.em_field.frequencies)[0]
            p_inc, phase = rets[1:]
            p_inc = p_inc[idxs]
            phase = phase[idxs]
            em_field_new = self.em_field._newFieldComp(p_inc, phase)
            
            return RF_Coil(rets[0], em_field_new)


    def duplicatePortsParallel(self, ports, comp_Pinc=True):
        
        if not isinstance(ports,list) and not isinstance(ports,np.ndarray):
            raise TypeError("ports must be a list or a numpy.ndarray")
        else:
            ports = np.array(ports)
        
        if len(ports) > self.s_matrix.nPorts:
            raise ValueError("The length of ports cannot be higher than the number of ports of the RF coil")
        elif (np.logical_or(ports < 1,ports > self.s_matrix.nPorts)).any():
            raise ValueError("Each ports element identify the number of the port that has to be duplicated. It can take values from 1 to self.s_matrix.nPorts")
        elif (np.unique(ports,return_counts=True)[1] > 1).any():
            raise ValueError("At least one port number is not unique in ports")
            
        
        s_matrices = [None]*self.s_matrix.nPorts
        
        for port in ports:
            s_matrices[port-1] = S_Matrix.sMatrixDoubleY(self.s_matrix.frequencies, self.s_matrix.z0[port-1])
        
        return self.singlePortConnRFcoil(s_matrices, comp_Pinc)
    
    
    def duplicatePortsSeries(self, ports, comp_Pinc=True):
        
        if not isinstance(ports,list) and not isinstance(ports,np.ndarray):
            raise TypeError("ports must be a list or a numpy.ndarray")
        else:
            ports = np.array(ports)
        
        if len(ports) > self.s_matrix.nPorts:
            raise ValueError("The length of ports cannot be higher than the number of ports of the RF coil")
        elif (np.logical_or(ports < 1,ports > self.s_matrix.nPorts)).any():
            raise ValueError("Each ports element identify the number of the port that has to be duplicated. It can take values from 1 to self.s_matrix.nPorts")
        elif (np.unique(ports,return_counts=True)[1] > 1).any():
            raise ValueError("At least one port number is not unique in ports")
            
        
        s_matrices = [None]*self.s_matrix.nPorts
        
        for port in ports:
            s_matrices[port-1] = S_Matrix.sMatrixDoubleE(self.s_matrix.frequencies, self.s_matrix.z0[port-1])
        
        return self.singlePortConnRFcoil(s_matrices, comp_Pinc)
    
    
    def powerBalance(self, p_inc, voxVols=None, elCond=None, printReport=False):
        
        powBal = {}
        tot_p_inc = np.sum(np.abs(p_inc))
        powBal["P_inc_tot"] = tot_p_inc
        
        pows = self.s_matrix.powerBalance(p_inc)

        if self.em_field is None or self.em_field.e_field is None:
            
            frequencies = self.s_matrix.frequencies
            
            if len(pows) == 1: # No available data for external circuitries losses
                powBal["P_refl"] = tot_p_inc - pows[0]
                powBal["P_other"] = pows[0]
            elif len(pows) == 2:
                powBal["P_refl"] = tot_p_inc - pows[0]
                powBal["P_circ_loss"] = pows[0] - pows[1]
                powBal["P_other"] = tot_p_inc - powBal["P_refl"] - powBal["P_circ_loss"]
        
        else:
                
            idxs = np.where(self.s_matrix.frequencies[:,None]==self.em_field.frequencies)[0]
            
            depPow = self.em_field.compDepPow(voxVols, elCond, p_inc)
            
            if len(pows) == 1: # No available data for external circuitries losses
                powBal["P_refl"] = tot_p_inc - pows[0][idxs]
                powBal["P_dep"] = depPow
                powBal["P_other"] = pows[0][idxs] - depPow
            elif len(pows) == 2:
                powBal["P_refl"] = tot_p_inc - pows[0][idxs]
                powBal["P_dep"] = depPow
                powBal["P_circ_loss"] = pows[0] - pows[1]
                powBal["P_other"] = tot_p_inc - powBal["P_refl"] - powBal["P_circ_loss"] - powBal["P_dep"] 
            
            frequencies = self.s_matrix.frequencies[idxs]
            
        if printReport:
            print("\n\nTotal incident power: %.2e W" %(tot_p_inc))
            for i, freq in enumerate(frequencies):
                print("\n\n"+"-"*10+"\n")
                print("Frequency: %.2e Hz" %(freq))
                print("\n"+"-"*10+"\n\n")
                
                for key in powBal:
                    if key != "P_inc_tot":
                        print("%s: %.2e W\t(%.2f %%)\n"%(key, powBal[key][i], powBal[key][i]/tot_p_inc*100))
        return powBal
    
    
    def connectPorts(self, port_pairs, sMats, comp_Pinc=True):
        
        sMatrixForConnection = self.s_matrix._sMatrixConnectPorts(port_pairs, sMats)
        
        return self.fullPortsConnRFcoil(sMatrixForConnection, comp_Pinc)
    
        
    def saveRFCoil(self, filename, description=""):
        """
        !binary data in little-endian!
        All multidimensional arrays saved as flattened (C like order) list of float. If complex, real and imaginary parts are saved consecutively for each array element
        -------------------------------------
        FILE_VERSION --> short (2 bytes)
        HEADER_LINES --> short (2 bytes)
        ### HEADER BEGIN ###
        DESCRIP: Description string
        S_NFREQ: s_n_f
        N_PORTS: n1, n2, n3, ... (Number of ports starting with the number of ports of the final RF coil and ending to the number of ports of the original S matrix.
                                  For the present version, the list maximum length is 2 since the data of only the original S= matrix are saved in the S_Matrix instance)
        EMFIELD: TRUE/FALSE
        E_FIELD: TRUE/FALSE
        B_FIELD: TRUE/FALSE
        NPOINTS: nx, ny, nz
        EMNFREQ: em_n_f
        EM_PROP: property1, property2, ...
        ### HEADER END ###
        s_matrix frequencies --> float (4 bytes)
        s_matrix_S --> float (4 bytes)
        s_matrix_p_incM --> float (4 bytes)
        s_matrix_phaseM --> float (4 bytes)
        em_field frequencies --> float (4 bytes)
        em_field e_field --> float (4 bytes)
        em_field b_field --> float (4 bytes)
        em_field properties --> float (4 bytes)
        """
        
        def flattenCompArray(compArray):
            """
            Given a complex multidimensional array, it returns a flattened (C order like) array where the real and imaginary parts of its elements
            are alternated
            """
            flat_compArray = compArray.flatten()
            flat_compArray_real = np.expand_dims(flat_compArray.real,axis=1)
            flat_compArray_imag = np.expand_dims(flat_compArray.imag,axis=1)
            return np.concatenate((flat_compArray_real,flat_compArray_imag),axis=1).flatten()
        
        file_version = 1
        header_lines = 9 #Number of header lines excluding the HEADER BEGIN and HEADER END lines
        
        if filename.split(".")[-1] != "cspy":
            filename += ".cspy"
        
        if description:
            description = " ".join(description.split("\n"))
        else:
            description = "N/D"
        
        # Extraction of S_Matrix nested instances (Up to now only level 0 (final S matrix) and 1 (original S matrix) will be present)
        s_matrix = self.s_matrix
        s_matrix_instances = [s_matrix]
        n_ports_list = [s_matrix.nPorts]
        while s_matrix._S0 is not None:
            s_matrix = s_matrix._S0
            s_matrix_instances.append(s_matrix)
            n_ports_list.append(s_matrix.nPorts)
        
        with open(filename, 'wb') as f:
            
            #Saving status
            print("\nSaving: header...", end='', flush=True)
            
            #File version and header lines
            f.write(struct.pack('<h', file_version))
            f.write(struct.pack('<h', header_lines))
            
            #Header
            f.write(b"### HEADER BEGIN ###\n")
            f.write(b"DESCRIP: %s\n"%bytes(description, encoding="utf-8"))
            f.write(b"S_NFREQ: %d\n"%self.s_matrix.n_f)
            f.write(b"N_PORTS: %s\n"%(b", ".join(bytes(str(x), encoding="utf-8") for x in n_ports_list)))
            f.write(b"EMFIELD: %s\n"%(b"TRUE" if self.em_field is not None else b"FALSE"))
            f.write(b"E_FIELD: %s\n"%(b"TRUE" if (self.em_field is not None and self.em_field.e_field is not None) else b"FALSE"))
            f.write(b"B_FIELD: %s\n"%(b"TRUE" if (self.em_field is not None and self.em_field.b_field is not None) else b"FALSE"))
            f.write(b"NPOINTS: %s\n"%(b"N/D" if self.em_field is None else b", ".join(bytes(str(x), encoding="utf-8") for x in self.em_field.nPoints)))
            f.write(b"EMNFREQ: %s\n"%(b"N/D" if self.em_field is None else bytes(str(self.em_field.n_f), encoding="utf-8")))
            f.write(b"EM_PROP: %s\n"%(b"N/D" if (self.em_field is None or self.em_field.properties == {}) else b", ".join(bytes(x, encoding="utf-8") for x in self.em_field.properties.keys())))
            f.write(b"### HEADER END ###\n")
            
            #Saving status
            print("\rSaving: S_Matrix...", end='', flush=True)
            
            #s_matrix_frequencies
            for freq in self.s_matrix.frequencies:
                f.write(struct.pack('<f',freq))
            #s_matrix_S
            for s_matrix_instance in s_matrix_instances:
                s_matrix_flat = flattenCompArray(s_matrix_instance.S)
                for s in s_matrix_flat:
                    f.write(struct.pack('<f', s))
            #s_matrix_p_incM
            for s_matrix_instance in s_matrix_instances[1:]:# p_inc is stored only for S matrices with level >1
                p_inc_flat = s_matrix_instance._p_incM.flatten()
                for p in p_inc_flat:
                    f.write(struct.pack('<f', p))
            #s_matrix_phaseM
            for s_matrix_instance in s_matrix_instances[1:]:# phaseM is stored only for S matrices with level >1
                phaseM_flat = s_matrix_instance._phaseM.flatten()
                for ph in phaseM_flat:
                    f.write(struct.pack('<f', ph))
            
            #em_field
            if self.em_field is not None:
            
                em_field_prop_names = self.em_field.properties.keys()
                #EM frequencies
                for freq in self.em_field.frequencies:
                    f.write(struct.pack('<f', freq))
                #E field
                if self.em_field.e_field is not None:
                    #Saving status
                    print("\rSaving: EM_Field.e_field...", end='', flush=True)
                
                    e_field_flat = flattenCompArray(self.em_field.e_field)
                    for e in e_field_flat:
                        f.write(struct.pack('<f', e))
                #B field
                if self.em_field.b_field is not None:
                    #Saving status
                    print("\rSaving: EM_Field.b_field...", end='', flush=True)
                    
                    b_field_flat = flattenCompArray(self.em_field.b_field)
                    for b in b_field_flat:
                        f.write(struct.pack('<f', b))
                # properties
                for prop_name in em_field_prop_names:
                    #Saving status
                    print("\rSaving: EM_Field.properties...", end='', flush=True)
                    
                    for pr in self.em_field.properties[prop_name]:
                        f.write(struct.pack('<f', pr))
            
            print("\nRF coil succesfully saved as %s\n" %filename)

    @classmethod
    def loadRFCoil(cls, filename, **kwarg):
    
        with open(filename, 'rb') as f:
            
            file_version, = struct.unpack('<h', f.read(2)) #When higher version will be implemented, a check can be performed over this variable
            header_lines, = struct.unpack('<h', f.read(2))
            
            #Loading status
            print("\nLoading: header...", end='', flush=True)
            
            #Reading header
            header = {}
            f.readline() #HEADER BEGIN
            for _ in range(header_lines):
                line = f.readline().decode('utf-8')
                line = line.rstrip().split(": ")
                header[line[0]] = line[1]
            line = f.readline() #HEADER END
            
            #Loading status
            print("\rLoading: S_Matrix...", end='', flush=True)
            
            #s_matrix_frequencies
            s_frequencies = np.zeros(int(header['S_NFREQ']))
            for i in range(int(header['S_NFREQ'])):
                s_frequencies[i] = struct.unpack('<f', f.read(4))[0]
            #s_matrix
            n_ports = header['N_PORTS'].split(", ")
            n_ports = [int(x) for x in n_ports]
            s_matrices = [] #List containing the S matrices from the last to the original. Up to now, maximum 2
            for n in n_ports:
                s_matrix = np.zeros(int(header['S_NFREQ'])*n**2, dtype=np.complex)
                for i in range(int(header['S_NFREQ'])*n**2):
                    s_matrix[i] = struct.unpack('<f', f.read(4))[0]
                    s_matrix[i] += 1j*struct.unpack('<f', f.read(4))[0]
                s_matrix = s_matrix.reshape([int(header['S_NFREQ']), n, n])
                s_matrices.append(s_matrix)
            #s_matrix_p_incM
            p_incs = [] #List containing the p_incM matrices from the last to the original. Its length will be equal to the s_matrices list length minus 1. Up to now, maximum 1
            for i in range(len(n_ports)-1):
                p_inc = np.zeros(int(header['S_NFREQ'])*n_ports[i]*n_ports[i+1], dtype=np.float)
                for q in range(int(header['S_NFREQ'])*n_ports[i]*n_ports[i+1]):
                    p_inc[q] = struct.unpack('<f', f.read(4))[0]
                p_inc = p_inc.reshape([int(header['S_NFREQ']), n_ports[i], n_ports[i+1]])
                p_incs.append(p_inc)
            #s_matrix_phaseM
            phases = [] #List containing the phaseM matrices from the last to the original. Its length will be equal to the s_matrices list length minus 1. Up to now, maximum 1
            for i in range(len(n_ports)-1):
                phase = np.zeros(int(header['S_NFREQ'])*n_ports[i]*n_ports[i+1], dtype=np.float)
                for q in range(int(header['S_NFREQ'])*n_ports[i]*n_ports[i+1]):
                    phase[q] = struct.unpack('<f', f.read(4))[0]
                phase = phase.reshape([int(header['S_NFREQ']), n_ports[i], n_ports[i+1]])
                phases.append(phase)
            
            #em_field
            em_field = None
            if header["EMFIELD"] == 'TRUE':
                e_field = None
                b_field = None
                n_points = header['NPOINTS'].split(", ")
                em_points = [int(x) for x in n_points]
                n_points = np.prod(em_points)
                em_frequencies = np.zeros([int(header['EMNFREQ'])])
                for i in range(int(header['EMNFREQ'])):
                    em_frequencies[i] = struct.unpack('<f', f.read(4))[0]
                if header["E_FIELD"] == 'TRUE':
                    #Loading status
                    print("\rLoading: EM_Field.e_field...", end='', flush=True)
            
                    e_field = np.zeros(int(header['EMNFREQ'])*n_ports[0]*3*n_points, dtype=np.complex)
                    for i in range((int(header['EMNFREQ'])*n_ports[0]*3*n_points)):
                        e_field[i] = struct.unpack('<f', f.read(4))[0]
                        e_field[i] += 1j*struct.unpack('<f', f.read(4))[0]
                    e_field = e_field.reshape([int(header['EMNFREQ']),n_ports[0],3,n_points])
                if header["B_FIELD"] == 'TRUE':
                    #Loading status
                    print("\rLoading: EM_Field.b_field...", end='', flush=True)
                    
                    b_field = np.zeros(int(header['EMNFREQ'])*n_ports[0]*3*n_points, dtype=np.complex)
                    for i in range((int(header['EMNFREQ'])*n_ports[0]*3*n_points)):
                        b_field[i] = struct.unpack('<f', f.read(4))[0]
                        b_field[i] += 1j*struct.unpack('<f', f.read(4))[0]
                    b_field = b_field.reshape([int(header['EMNFREQ']),n_ports[0],3,n_points])
                em_properties = {}
                prop_names = header["EM_PROP"].split(", ")
                if prop_names[0] != 'N/D':
                    #Loading status
                    print("\rLoading: EM_Field.properties...", end='', flush=True)
                    
                    for prop_name in prop_names:
                        em_property = np.zeros([n_points], dtype=np.float)
                        for i in range(n_points):
                            em_property[i] = struct.unpack('<f', f.read(4))[0]
                        em_properties[prop_name] = em_property
                
                #EM_Field instance
                em_field = EM_Field(em_frequencies, em_points, b_field, e_field, **em_properties)
                    
            #S_Matrix instance
            s_matrix_in = None
            for i in range(len(p_incs)):
                s_matrix = S_Matrix(s_matrices[-1-i], s_frequencies)
                s_matrix._p_incM = p_incs[-1-i]
                s_matrix._phaseM = phases[-1-i]
                s_matrix._S0 = s_matrix_in
                s_matrix_in = s_matrix
            else:
                s_matrix = S_Matrix(s_matrices[0], s_frequencies, **kwarg)
                s_matrix._S0 = s_matrix_in
            
            #RF_Coil instance
            
            rf_coil = cls(s_matrix, em_field)
            
            return rf_coil
