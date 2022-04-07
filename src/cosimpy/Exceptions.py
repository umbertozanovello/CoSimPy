import numpy as np
import warnings

def warning_format(message, category, filename, lineno, file=None, line=None):
    return '\n%s: Line %s - WARNING - %s\n' % (filename.split("/")[-1], lineno, message)
warnings.formatwarning = warning_format


class S_MatrixError(Exception):
    """
    General error class relevant to the S_Matrix class
    """
    
    def __init__(self, message="", callingMethod=None):
        super().__init__(message)
        self.__message = message
        self.__callingMethod = callingMethod
    
    def __str__(self):
        if self.__callingMethod is None:
            return self.__message
        else:
            return "S_Matrix.%s: %s" %(self.__callingMethod, self.__message)


class S_MatrixArrayError(S_MatrixError):
    """
    Error relevant to the S_Matrix array
    """
    
    def __init__(self, message="", callingMethod=None):
        super().__init__(message, callingMethod)
        self.__message = message
    
    @classmethod
    def check(cls, S, info_warnings):
        """
        It performs some test on the S Matrix array
        """
        if not isinstance(S, np.ndarray): 
            raise cls("S can only be a numpy ndarray")
        if len(S.shape) != 3:
             raise cls("S can only be an Nf x Np x Np numpy ndarray")
        if S.shape[1] != S.shape[2]:
             raise cls("S can only be an Nf x Np x Np numpy ndarray")
        if (np.round(np.abs(S),6) > 1).any():
            warnings.warn("An S parameter higher than one has been found. Results could be unphysical at least at one frequency value. Healing the S matrix with the healSMatrix method could solve the problem")
            if info_warnings:
                print("\nMax |S_ij|:\n")
                print(np.max(np.abs(S)))
        
        # Check for positive definiteness of II - (S^H)(S)
        p = np.eye(S.shape[1]) - S @ np.conjugate(np.transpose(S,axes=[0,2,1]))
        not_nan_idxs = np.where(np.logical_not(np.isnan(p).any(axis=(1,2))))[0] #idx of new_P first dimension where no nan values along the other two dimensions are encountered
        if (np.round(np.real(np.linalg.eigvals(p[not_nan_idxs])),6) < 0).any():
            warnings.warn("The S matrix seems to be unphysical at least at one frequency value. Healing the S matrix with the healSMatrix method could solve the problem")
            if info_warnings:
                print("\nEigenvalues of II - S^H @ S:\n")
                print(np.real(np.linalg.eigvals(np.eye(S.shape[1]) - S @ np.conjugate(np.transpose(S,axes=[0,2,1])))))


class S_MatrixFrequenciesError(S_MatrixError):
    """
    Error relevant to the frequencies list
    """
    
    def __init__(self, message="", callingMethod=None):
        super().__init__(message, callingMethod)
        self.__message = message
        
    @classmethod
    def check(cls, freqs, expected_length=None):
        if not isinstance(freqs, np.ndarray) and not isinstance(freqs, list): 
             raise cls("freqs can only be a list or numpy.ndarray")
        if (np.unique(freqs,return_counts=True)[1] > 1).any():
            raise cls("At least one frequency value is not unique in freqs")
        if expected_length is not None and len(freqs) != expected_length:
             raise cls("The frequencies list has an unexpected length")
             
             
class S_MatrixPortImpedancesError(S_MatrixError):
    """
    Error relevant to the port impedances list
    """
    
    def __init__(self, message="", callingMethod=None):
        super().__init__(message, callingMethod)
        self.__message = message
        
    @classmethod
    def check(cls, z0, expected_length):
        if not isinstance(z0, np.ndarray) and not isinstance(z0, list) and not isinstance(z0, int) and not isinstance(z0, float):
            raise cls("z0 can only be a list or numpy ndarray or a scalar in case all ports share the same impedance")
        if (isinstance(z0, np.ndarray) or isinstance(z0, list)) and len(z0) != expected_length:
            raise cls("The port impedances list has an unexpected length")
        if (np.array(np.real(z0)) <= 0).any():
            raise cls("The real part of all the port impedances has to be higher than zero")
        elif (np.array(np.real(z0)) != np.array(z0)).any():
            warnings.warn("The present version of the library can only handle real port impedances. The imaginary parts will be neglected")
            
            
class S_MatrixTouchstoneFileError(S_MatrixError):
    """
    Error relevant to the Touchstone file import/export
    """
    
    def __init__(self, message="", callingMethod=None):
        super().__init__(message, callingMethod)
        self.__message = message
        
    