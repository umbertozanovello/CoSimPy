import numpy as np
from .S_Matrix import S_Matrix
from .EM_Field import EM_Field

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
                string += "E field not defined\n"
            elif self.__em_field.b_field is None:
                string += "B field not defined\n"
        else:
            string += 'Not defined\n'
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
        
                
    def fullPortsConnRFcoil(self, s_matrix_other, comp_Pinc=False):
        
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
