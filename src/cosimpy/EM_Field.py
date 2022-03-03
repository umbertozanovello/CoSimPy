import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.io import loadmat

class EM_Field():
    
    def __init__(self, freqs, nPoints, b_field=None, e_field=None, **kwargs):
        
        if not isinstance(freqs, np.ndarray) and not isinstance(freqs, list): 
             raise TypeError("Frequencies can only be an Nf list")
        elif (np.unique(freqs,return_counts=True)[1] > 1).any():
            raise ValueError("At least one frequency value is not unique in freqs")
        elif not isinstance(nPoints, np.ndarray) and not isinstance(nPoints, list): 
             raise TypeError("nPoints can only be a list or numpy ndarray with length equal to 3") 
        elif len(nPoints) != 3:
             raise TypeError("nPoints can only be a list or numpy ndarray with length equal to 3")
                
        if e_field is None and b_field is None:
            raise ValueError("At least one among e_field and b_field arguments has to be different from None")
        
        if e_field is not None:
            if not isinstance(e_field, np.ndarray):
                raise TypeError("e_field can only be numpy ndarray")
            elif len(e_field.shape) != 4:
                raise ValueError("e_field can only be an Nf x Np x 3 x Nn matrix")
            elif e_field.shape[0] != len(freqs):
                raise ValueError("The frequencies list is not compatible with the e_field matrix first dimension")
            elif e_field.shape[2] != 3:
                raise ValueError("The third dimension of e_field is expected to be 3 (the number of field components)")
            elif e_field.shape[3] != np.prod(nPoints):
                raise ValueError("The fourth dimension of e_field is expected to be equal to nPoints[0]*nPoints[1]*nPoints[2]")
        if b_field is not None:
            if not isinstance(b_field, np.ndarray):
                raise TypeError("b_field can only be numpy ndarray")
            elif len(b_field.shape) != 4:
                raise ValueError("b_field can only be an Nf x Np x 3 x Nn matrix")
            elif b_field.shape[0] != len(freqs):
                raise ValueError("The frequencies list is not compatible with the b_field matrix first dimension")
            elif b_field.shape[2] != 3:
                raise ValueError("The third dimension of b_field is expected to be 3 (the number of field components)")
            elif b_field.shape[3] != np.prod(nPoints):
                raise ValueError("The fourth dimension of b_field is expected to be equal to nPoints[0]*nPoints[1]*nPoints[2]")

        for arg in kwargs.values():
            if not isinstance(arg, list) and not isinstance(arg, np.ndarray):
                raise ValueError("All the additional properties have to be list or numpy ndarray of float or int")
            elif np.array(arg).dtype != np.dtype(np.float) and np.array(arg).dtype != np.dtype(np.int):
                raise ValueError("All the additional properties have to be list or numpy ndarray of float or int")
            elif np.array(arg).size != np.prod(nPoints):
                raise ValueError("At least one of the additional properties has a length different from nPoints[0]*nPoints[1]*nPoints[2]")
            elif len(np.array(arg).shape) != 1:
                raise ValueError("All the additional properties have to be 1D-list or numpy ndarray")

        if e_field is not None:  
            self.__e_field = np.array(e_field,dtype = "complex")
            self.__nPorts = e_field.shape[1]
        else:
            self.__e_field = None
        if b_field is not None:    
            self.__b_field = np.array(b_field,dtype = "complex")
            self.__nPorts = b_field.shape[1]
        else:
            self.__b_field = None
            
        self.__freqs = np.array(freqs)
        self.__nPoints = nPoints
        self.__n_f = len(freqs)
        self.__prop = kwargs
    
    
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
        return self.__freqs
    
    
    @property
    def properties(self):
        return self.__prop
    
    
    def __repr__(self):
        string = '"""""""""""""""\n   EM FIELD\n"""""""""""""""\n\n'
        string += "Number of frequency values = %d\nNumber of ports = %d\nNumber of point (nx, ny, nz) = %d, %d, %d\n\n"%(self.__n_f,self.__nPorts, self.__nPoints[0], self.__nPoints[1],self.__nPoints[2])
        if self.e_field is None:
                string += "E field not defined\n\n"
        elif self.b_field is None:
            string += "B field not defined\n\n"
        if self.__prop != {}:
            for key in self.__prop:
                string += "'%s' additional property defined\n" %key
        return string
    
    
    def compSensitivities(self):
        
        if self.__b_field is None:
            raise ValueError("No b_field property is specifed for the EM_Field instance")
        
        sens = np.copy(self.__b_field * np.sqrt(2)) # b_field contains rms values of the B field
        sens = np.delete(sens, 2, axis = 2)
        
        b1p = 0.5*(sens[:,:,0,:] + 1j*sens[:,:,1,:])
        b1m = 0.5*np.conj(sens[:,:,0,:] - 1j*sens[:,:,1,:])
        
        sens[:,:,0,:] = b1p
        sens[:,:,1,:] = b1m
        
        return sens
    
    
    def compPowDens(self, elCond=None, p_inc=None):
        
        if elCond is None and not "elCond" in self.__prop.keys():
            raise ValueError("No 'elCond' key is found in self.properties. Please, provide the electrical conductivity as argument of the method")
        elif elCond is not None:
            if not isinstance(elCond, list) and not isinstance(elCond, np.ndarray):
                raise ValueError("elCond has to be a list or a numpy ndarray of float or int")
            elif np.array(elCond).dtype != np.dtype(np.float) and np.array(elCond).dtype != np.dtype(np.int):
                raise ValueError("elCond has to be a list or an numpy ndarray of float or int")
            elif np.array(elCond).size != np.prod(self.__nPoints):
                raise ValueError("elCond has a length different from nPoints[0]*nPoints[1]*nPoints[2]")
        else:
            elCond = self.__prop["elCond"]
            
        if self.__e_field is None:
            raise ValueError("No e field property is specifed for the EM_Field instance. Power density cannot be computed")
        
        if p_inc is not None: #Power density is computed for a defined supply configuration
            if not isinstance(p_inc, np.ndarray) and not isinstance(p_inc, list):
                raise TypeError("S matrix can only be numpy ndarray or a list")
            else:
                p_inc = np.array(p_inc)
            if p_inc.size != self.__nPorts:
                raise TypeError("p_inc has to be a self.nPorts length list or numpy ndarray")
                
            norm = np.sqrt(np.abs(p_inc))*np.exp(1j*np.angle(p_inc))
            efield_new = np.moveaxis(self.e_field,1,-1) #Temporary axis change so field.shape = [self.n_f, 3, self.nPoints, self.nPorts]
            efield_new = efield_new @ norm 
        else:
            efield_new = self.__e_field
        
        powDens = np.linalg.norm(efield_new, axis=-2)**2 * elCond
        
        return powDens
    
    
    def compDepPow(self, voxVols, elCond=None, p_inc=None):
        

        if not isinstance(voxVols, int) and not isinstance(voxVols, float):
            raise ValueError("voxVols has to be int or float")
            
        powDens = self.compPowDens(elCond, p_inc)
            
        depPow = np.nansum(powDens*voxVols, axis=-1)
        
        return depPow
        
    
    def plotB(self, comp, freq, port, plane, sliceIdx, vmin=None, vmax=None):
        
        if self.__b_field is None:
            raise ValueError("No b_field property is specifed for the EM_Field instance")
            
        f_idx = np.where(self.__freqs==freq)[0][0]
        if f_idx is None:
            raise ValueError("No B field for the specified frequency")
        
        if port not in np.arange(self.__nPorts) + 1:
            raise ValueError("No B field for the specified port")
            
        if comp not in ['b1+', 'b1-']:
            b = self.__b_field[f_idx, port-1,:,:]
            
            if comp.lower() == "x":
                b = np.abs(b[0,:])
            elif comp.lower() == 'y':
                b = np.abs(b[1,:])
            elif comp.lower() == 'z':
                b = np.abs(b[2,:])
            elif comp.lower() == 'mag':
                b = np.linalg.norm(b,axis=0)   
            else:
                raise ValueError("comp must take one of the following values: 'mag', 'x', 'y', 'z', 'b1+', 'b1-'")
        else:
            b = self.compSensitivities()
            b = b[f_idx, port-1,:,:]
            
            if comp == "b1+":
                b = np.abs(b[0,:])
            else:
                b = np.abs(b[1,:])
        
        b = np.reshape(b, self.__nPoints, order='F')
        
        plt.figure("B field, Frequency %.2f MHz, Port %d, Plane %s, Index %d" %(freq*1e-6, port, plane, sliceIdx))        
        
        if plane == 'xy':
            plt.imshow(1e6*b[:,:,sliceIdx].T,vmin=vmin,vmax=vmax)
            plt.xlabel("x")
            plt.ylabel("y")
        elif plane == 'xz':
            plt.imshow(1e6*b[:,sliceIdx,:].T,vmin=vmin,vmax=vmax)
            plt.xlabel("x")
            plt.ylabel("z")
        elif plane == 'yz':
            plt.imshow(1e6*b[sliceIdx,:,:].T,vmin=vmin,vmax=vmax)
            plt.xlabel("y")
            plt.ylabel("z")
        
        plt.title("Port %d: %s component" %(port, comp))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("B field ($\mu$T)")
        
        
    def plotE(self, comp, freq, port, plane, sliceIdx, vmin=None, vmax=None):
        
        if self.__e_field is None:
            raise ValueError("No e_field property is specifed for the EM_Field instance")
            
        f_idx = np.where(self.__freqs==freq)[0][0]
        if f_idx is None:
            raise ValueError("No E field for the specified frequency")
        
        if port not in np.arange(self.__nPorts) + 1:
            raise ValueError("No E field for the specified port")
            
        if comp != "mag":
            e = self.__e_field[f_idx, port-1,:,:]
            
            if comp == "x":
                e = np.abs(e[0,:])
            elif comp == 'y':
                e = np.abs(e[1,:])
            elif comp == 'z':
                e = np.abs(e[2,:])
            else:
                raise ValueError("comp must take one of the following values: 'mag', 'x', 'y', 'z', 'mag'")
        else:
            e = np.sqrt(np.abs(self.__e_field[f_idx, port-1,0,:])**2 + np.abs(self.__e_field[f_idx, port-1,1,:])**2 + np.abs(self.__e_field[f_idx, port-1,2,:])**2)
        
        e = np.reshape(e, self.__nPoints, order='F')
        
        plt.figure("E field, Frequency %.2f MHz, Port %d, Plane %s, Index %d" %(freq*1e-6, port, plane, sliceIdx))
        
        if plane == 'xy':
            plt.imshow(e[:,:,sliceIdx].T,vmin=vmin,vmax=vmax)
            plt.xlabel("x")
            plt.ylabel("y")
        elif plane == 'xz':
            plt.imshow(e[:,sliceIdx,:].T,vmin=vmin,vmax=vmax)
            plt.xlabel("x")
            plt.ylabel("z")
        elif plane == 'yz':
            plt.imshow(e[sliceIdx,:,:].T,vmin=vmin,vmax=vmax)
            plt.xlabel("y")
            plt.ylabel("z")
        
        plt.title("Port %d: %s component" %(port, comp))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("E field (V/m)")
        
    
    
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

            for freq in self.__freqs:
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
            
            for p in self.__prop:
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

            for i,freq in enumerate(self.__freqs):
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
            for p in self.__prop:
                f["Properties/%s"%p] = self.__prop[p]
                  
                
    def _newFieldComp(self, p_incM, phaseM):
        
        if p_incM.shape != phaseM.shape:
            raise ValueError("p_incM and phaseM arrays are not coherent")
        elif p_incM.shape[2] != self.__nPorts or phaseM.shape[2] != self.__nPorts:
            raise ValueError("The number of ports has to be equal to p_incM and phase third dimension")
        elif p_incM.shape[0] != self.__n_f or phaseM.shape[0] != self.__n_f:
            raise ValueError("The number of frequencies of self has to be equal to p_incM and phaseM first dimension")
        
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
            
        return EM_Field(self.__freqs, self.__nPoints, bfield_new, efield_new, **self.__prop)
            
    
    @classmethod
    def importFields_cst(cls, directory, freqs, nPorts, nPoints=None, Pinc_ref=1, b_multCoeff=1, pkORrms='pk', imp_efield=True, imp_bfield=True, fileType = 'ascii', col_ascii_order = 0, **kwargs):

        if not imp_efield and not imp_bfield:
            raise ValueError("At least one among imp_efield and imp_bfield has to be True")
        elif nPoints is not None and len(nPoints) != 3:
            raise  TypeError("nPoints can only be None or a list with length equal to 3")
        elif fileType.lower() not in ["ascii", "hdf5"]:
            raise  ValueError("fileType can only be 'ascii' or 'hdf5'")
        elif pkORrms.lower() not in ["pk", "rms"]:
            raise  ValueError("pkORrms can only be 'pk or 'rms'")
        elif col_ascii_order not in [0, 1]:
            raise  ValueError("col_ascii_order can take 0 (Re_x, Re_y, Re_z, Im_x, Im_y, Im_z) or 1 (Re_x, Im_x, Re_y, Im_y, Re_z, Im_z) values")
        
        if nPoints is None and fileType == "ascii": #I try to evaluate nPoints
            
            if imp_efield:
                x,y,z = np.loadtxt(directory+"/efield_%s_port1.txt"%(freqs[0]), skiprows=2, unpack=True, usecols=(0,1,2))
            else:
                x,y,z = np.loadtxt(directory+"/bfield_%s_port1.txt"%(freqs[0]), skiprows=2, unpack=True, usecols=(0,1,2))
            
            orig_len = len(x) #Total number of points
            
            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)
            
            nPoints = [len(x), len(y), len(z)]
            
            assert np.prod(nPoints) == orig_len, "nPoints evaluation failed. Please specify its value in the method argument"

        elif nPoints is None and fileType == "hdf5":
            if imp_efield:
                filename = "/efield_%s_port1.h5"%(freqs[0])
            else:
                filename = "/bfield_%s_port1.h5"%(freqs[0])
                
            with h5py.File(directory+filename, "r") as f:
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
                        e_real = np.loadtxt(directory+"/efield_%s_port%d.txt"%(f,port+1), skiprows=2, usecols=re_cols)
                        e_imag = np.loadtxt(directory+"/efield_%s_port%d.txt"%(f,port+1), skiprows=2, usecols=im_cols)
                        assert e_real.shape[0] == n, "At least one of e_field files is not compatible with the evaluated or passed nPoints"
                        e_field[idx_f, port, :, :] = (e_real+1j*e_imag).T
                    if imp_bfield:
                        b_real = np.loadtxt(directory+"/bfield_%s_port%d.txt"%(f,port+1), skiprows=2, usecols=(3,4,5))
                        b_imag = np.loadtxt(directory+"/bfield_%s_port%d.txt"%(f,port+1), skiprows=2, usecols=(6,7,8))
                        assert b_real.shape[0] == n, "At least one of b_field files is not compatible with the evaluated or passed nPoints"
                        b_field[idx_f, port, :, :] = (b_real+1j*b_imag).T
                
                print("\n")
        
        elif fileType.lower() == 'hdf5':
            for idx_f, f in enumerate(freqs):
                print("Importing %s MHz fields\n"%f)
    
                for port in range(nPorts):
                    print("\r\tImporting port%d fields"%(port+1), end='', flush=True)
                    if imp_efield:
                        
                        filename = "/efield_%s_port%d.h5"%(f,port+1)
                        with h5py.File(directory + filename, "r") as field_file:
                            e_field_raw = np.array(field_file['E-Field'])
                            x = np.array(field_file['Mesh line x'])
                            y = np.array(field_file['Mesh line y'])
                            z = np.array(field_file['Mesh line z'])
                        
                        assert len(x) * len(y) * len(z) == n, "At least one of e_field files is not compatible with the evaluated or passed nPoints"
                        
                        e_flat = e_field_raw.flatten() #Flatted array (x,y,z to be reshaped as Fortran order). Each element is a np.void type made of three (x-, y-, z-component) couple of float representing real and imaginary parts of that component
                        e_flat = np.array(e_flat.tolist()) #Array n_points, 3 (components), 2 (real and imaginary)
                        e_field[idx_f, port, :, :] = (e_flat[:,:,0] + 1j*e_flat[:,:,1]).T
                    
                    if imp_bfield:
                        
                        filename = "/bfield_%s_port%d.h5"%(f,port+1)
                        with h5py.File(directory + filename, "r") as field_file:
                            b_field_raw = np.array(field_file['H-Field']) #b_field is an H field and will become  field when multiplied by b_multCoeff
                            x = np.array(field_file['Mesh line x'])
                            y = np.array(field_file['Mesh line y'])
                            z = np.array(field_file['Mesh line z'])
                        
                        assert len(x) * len(y) * len(z) == n, "At least one of e_field files is not compatible with the evaluated or passed nPoints"
                        
                        b_flat = b_field_raw.flatten() #Flatted array (x,y,z to be reshaped as Fortran order). Each element is a np.void type made of three (x-, y-, z-component) couple of float representing real and imaginary parts of that component
                        b_flat = np.array(b_flat.tolist()) #Array n_points, 3 (components), 2 (real and imaginary)
                        b_field[idx_f, port, :, :] = (b_flat[:,:,0] + 1j*b_flat[:,:,1]).T
                    
                print("\n")
        if imp_efield:
            e_field = np.sqrt(1/Pinc_ref) * rmsCoeff * e_field #cst exported field values are peak values
        if imp_bfield:
            b_field = b_multCoeff * np.sqrt(1/Pinc_ref) * rmsCoeff * b_field #cst exported field values are peak values
        
        freqs = 1e6*np.array(freqs).astype(np.float)
        
        return cls(freqs, nPoints, b_field, e_field, **kwargs)


    @classmethod
    def importFields_s4l(cls, directory, freqs, nPorts, Pinc_ref=1, b_multCoeff=1, pkORrms='pk', imp_efield=True, imp_bfield=True, **kwargs):
        
        if not imp_efield and not imp_bfield:
            raise ValueError("At least one among imp_efield and imp_bfield has to be True")
        elif pkORrms.lower() not in ["pk", "rms"]:
            raise  ValueError("pkORrms can only be 'pk or 'rms'")
        
        if pkORrms.lower() == "pk":
            rmsCoeff = 1/np.sqrt(2)
        else:
            rmsCoeff = 1 
        
        if not imp_efield:
            e_field = None
            
        if not imp_bfield:
            b_field = None
        
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
        
        
        return cls(freqs, nPoints, b_field, e_field, **kwargs)