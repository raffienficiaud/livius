import numpy as np
import math

myfft2 = np.fft.fft2
myifft2 = np.fft.ifft2
def getAlignNdArray(image):
    return np.array(image)

if (1):
    try:
        import pyfftw
        myfft2 = pyfftw.interfaces.numpy_fft.fft2
        myifft2 = pyfftw.interfaces.numpy_fft.ifft2
        alignSize = pyfftw.simd_alignment
        
        def getAlignNdArray(image):
            return pyfftw.n_byte_align(np.array(image), alignSize)
            
    except:
        print("[templateMatching] Warning: pyfftw could not be found. Numpy fft is used instead.")

def templateMatching(image,template,Idata = None, Tdata = None, type="both"):
    '''
    templateMatching is a CPU efficient function which calculates matching score images between template
    and 2D image.
    
    It calculates:
    - The sum of squared difference (SSD Block Matching, type = 'SSD'), robust template matching
    - The normalized cross correlation (NCC, type = 'NCC'), independent of illumination, only dependent on texture
    
    Use type='SSD','NCC' or 'both' for both results. The result is: [SSD,NCC,Idata] where SSD or NCC may also be None
    where Idata is an intermediate result that can be used for faster recalculation with the same image but another template.
    
    The user can combine the two images, to get template matching which works robust with his application.
    Both matching methods are implemented using FFT based correlation.
    
    source: http://www.mathworks.com/matlabcentral/fileexchange/24925-fastrobust-template-matching
    '''
    IdataEmpty = False
    TdataEmpty = False
    
    if (Idata is None):
        IdataEmpty = True
        Idata = {}
        I = getAlignNdArray(image)
        Idata["I_shape"] = np.array(I.shape)
    
    if (Tdata is None):
        TdataEmpty = True
        Tdata = {}
        T = getAlignNdArray(template)
        Tdata["T_shape"] = np.array(T.shape)
    
    T_shape = Tdata["T_shape"]
    I_shape = Idata["I_shape"]
    outsize = T_shape + I_shape-1
    
    #calculate correlation in frequency domain
    if (TdataEmpty):
        Tdata["FT"] = myfft2( np.rot90(T,2), outsize)
        Tdata["Tsize"] = T.size
    
    if (IdataEmpty):
        Idata["FI"] = myfft2(I,outsize)
    Icorr = myifft2(Idata["FI"] * Tdata["FT"]).real
    
    #calculate local quadratic sum of image and template
    if (IdataEmpty):
        Idata["LocalQSumI"] = localSum(I*I, T_shape)
    
    I_SSD = None
    I_NCC = None
    
    if (type =="SSD" or type == "both"):
        if (TdataEmpty):
            Tdata["QSumT"] = np.sum(T*T)
    
        #SSD between template and image
        I_SSD = Idata["LocalQSumI"]+Tdata["QSumT"]-2*Icorr
        
        #Normalize to range 0..1
        I_SSD = I_SSD - np.min(I_SSD)
        I_SSD = 1 - (I_SSD / np.max(I_SSD))
        
        #remove padding
        I_SSD = unpadarray(I_SSD, I_shape)
    
    if (type == "NCC" or type == "both"):
        #normalized cross correlation
        if (IdataEmpty):
            Idata["LocalSumI"] = localSum(I, T_shape)
            
            #standard deviation
            temp = Idata["LocalQSumI"] - (Idata["LocalSumI"]*Idata["LocalSumI"])/ Tdata["Tsize"]
            temp[temp < 0] = 0
            Idata["stdI"] = np.sqrt(temp)
        
        
        if (TdataEmpty):
                Tdata["stdT"] = math.sqrt(T.size - 1) * np.std(T,ddof=1)
                Tdata["sumT"] = np.sum(T)
                
        # mean compensation
        meanIT = Idata["LocalSumI"]* Tdata["sumT"] / Tdata["Tsize"]
        temp = Idata["stdI"]
        temp[temp < (Tdata["stdT"]/1e5)] = Tdata["stdT"]/1e5
        I_NCC = 0.5+(Icorr-meanIT) / (2*Tdata["stdT"]*temp)
    
        #remove padding
        I_NCC = unpadarray(I_NCC, I_shape)
    
    return [I_SSD, I_NCC, Idata, Tdata]

def localSum(I,T_shape):
    pad_width = [ (i,i) for i in T_shape]
    B = np.pad(I,pad_width,'constant')
    #2D localsum
    s = np.cumsum(B,axis = 0)
    c = s[T_shape[0]:-1, :] - s[0:-T_shape[0]-1,:]
    s = np.cumsum(c, axis = 1)
    
    return s[:,T_shape[1]:-1] - s[:,0:-T_shape[1]-1]

def unpadarray(A,B_shape):
    B_start = np.ceil( (A.shape - B_shape) / 2 )
    B_end = B_start + B_shape
    
    return A[B_start[0]:B_end[0], B_start[1]:B_end[1]]


if(__name__ == '__main__'):
    xs=np.array([[5,1,7],[6,140,78],[0,8,1]])*1.
    ys=np.array([[75,14],[65,1]])*1.
    #I_SSD,I_NCC,Idata,Tdata = templateMatching(ys,xs)
    #I_SSD,I_NCC,Idata,Tdata = templateMatching(np.float32(I),np.float32(T))
    I_SSD,I_NCC,Idata,Tdata = templateMatching(xs,ys)
    
