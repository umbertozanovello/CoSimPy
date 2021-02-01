from .S_Matrix import S_Matrix
from .EM_Field import EM_Field
import numpy as np

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
        
        if not comp_Pinc or self.em_field is None:
            
            S_l = self.s_matrix._singlePortConnSMatrix(networks, False)
            
            return RF_Coil(S_l, None)
        
        else:
            
            S_l, p_inc, phase = self.s_matrix._singlePortConnSMatrix(networks, True)
            idxs=np.where(self.s_matrix.frequencies[:,None]==self.em_field.frequencies)[0]
            p_inc = p_inc[idxs]
            phase = phase[idxs]
            em_field_new = self.em_field._newFieldComp(p_inc, phase)
            
            return RF_Coil(S_l, em_field_new)
        
                
    def fullPortsConnRFcoil(self, s_matrix_other, comp_Pinc=False):
        
        if not comp_Pinc or self.em_field is None:
            
            S_l = self.s_matrix._fullPortsConnSMatrix(s_matrix_other, False)
            
            return RF_Coil(S_l, None)
        
        else:
            
            S_l, p_inc, phase = self.s_matrix._fullPortsConnSMatrix(s_matrix_other, True)
            idxs=np.where(self.s_matrix.frequencies[:,None]==self.em_field.frequencies)[0]
            p_inc = p_inc[idxs]
            phase = phase[idxs]
            em_field_new = self.em_field._newFieldComp(p_inc, phase)
            
            return RF_Coil(S_l, em_field_new)
