import numpy as np
import abc
import time
import torch


class Kernel(metaclass=abc.ABCMeta):
    def __init__(self,gamma):
        self.gamma    = gamma
    
    def setGamma(self,gamma):
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self,r): 
        pass    
    @abc.abstractmethod
    def phiR(self,r): # phi'/r
        pass    
    @abc.abstractmethod # phiR'/r
    def phiRR(self,r): 
        pass         

    def powerFunc(self,y,x):
        if y.ndim < 2:
            if y.ndim == 1: 
                y = y[:, np.newaxis]
            else:
                y = np.reshape(y, (1, -1))
        
        Kxx = self.getGramHermite(x,x)
        Kxy = self.getGramHermite(y,x)
        Kyy = self.getGramHermite(y,y)
        return np.max(np.sqrt(np.abs(np.diag(Kyy)-np.sum(np.linalg.solve(Kxx,Kxy.T)*Kxy.T,0))))
    
    def powerFuncSingle(self,y,x):
        if y.ndim < 2:
            if y.ndim == 1: 
                y = y[:, np.newaxis]
            else:
                y = np.reshape(y, (1, -1))

        Kxx = self.getGramHermite(x,x)
        Kxy = self.getGramHermite(y,x)
        Kyy = self.getGramHermite(y,y)
        return np.sqrt(np.abs(np.diag(Kyy)-np.sum(np.linalg.solve(Kxx,Kxy.T)*Kxy.T,0)))[0]

    def getRKHSNorm(self,x,rhs):
        return np.sqrt(np.sum(np.linalg.solve(self.getGramHermite(x,x),rhs)*rhs))

    def getGramHermite(self,y,x):
        if y.ndim < 2:
            if y.ndim == 1: 
                y = y[:, np.newaxis]
            else:
                y = np.reshape(y, (1, -1))

        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        K     = self.phi(diff)
        phiR  = self.phiR(diff)       
        phiRR = self.phiRR(diff)  

        nx    = x.shape[1]
        ny    = y.shape[1]
        N     = x.shape[0]

        diff_tensor    = x[:, None, :] - y[:, :, None]  
        weighted_diff  = phiR[None, :, :] * diff_tensor   
        B1             = weighted_diff.transpose(1, 2, 0).reshape(ny, nx * N)
        diff2          = y[:, :, None] - x[:, None, :]  
        weighted_diff2 = phiR[None, :, :] * diff2   
        B2             = weighted_diff2.transpose(1, 0, 2).reshape(ny * N, nx)
        D_blocks       = diff2.transpose(1, 2, 0)  
        outer          = D_blocks[..., :, None] * D_blocks[..., None, :]  
        I              = np.eye(N)
        C_blocks       = - phiRR[:, :, None, None] * outer - phiR[:, :, None, None] * I  
        C              = C_blocks.transpose(0, 2, 1, 3).reshape(ny * N, nx * N)
        return np.r_[np.c_[K, B1], np.c_[B2, C]] 
 
    def getGramHermiteTorch(self, y, x, newGamma=None):
        oldGamma = self.gamma

        if newGamma is not None:
            self.gamma = newGamma

        # Compute pairwise Euclidean distances between columns of x and y.
        # x: (N, nx), y: (N, ny)
        diff = torch.sqrt(
            torch.abs(
                torch.sum(x**2, dim=0, keepdim=True) +
                torch.sum(y**2, dim=0, keepdim=True).t() -
                2 * torch.matmul(y.t(), x)
            )
        )

        # Compute kernel matrices using self.phi, self.phiR, self.phiRR
        K     = self.phiTorch(diff)   # shape: (ny, nx)
        phiR  = self.phiRTorch(diff)  # shape: (ny, nx)
        phiRR = self.phiRRTorch(diff) # shape: (ny, nx)

        nx = x.shape[1]
        ny = y.shape[1]
        N  = x.shape[0]

        # diff_tensor: differences between x and y for each row.
        # x.unsqueeze(1): (N, 1, nx), y.unsqueeze(2): (N, ny, 1)
        diff_tensor = x.unsqueeze(1) - y.unsqueeze(2)  # shape: (N, ny, nx)
        weighted_diff = phiR.unsqueeze(0) * diff_tensor  # broadcasting to (N, ny, nx)
        # Permute to (ny, nx, N) and reshape to (ny, nx * N)
        B1 = weighted_diff.permute(1, 2, 0).reshape(ny, nx * N)

        # Compute the differences with reversed order
        diff2 = y.unsqueeze(2) - x.unsqueeze(1)  # shape: (N, ny, nx)
        weighted_diff2 = phiR.unsqueeze(0) * diff2  # shape: (N, ny, nx)
        # Permute to (ny, N, nx) and reshape to (ny * N, nx)
        B2 = weighted_diff2.permute(1, 0, 2).reshape(ny * N, nx)

        # Build the D_blocks tensor and its outer product
        D_blocks = diff2.permute(1, 2, 0)  # shape: (ny, nx, N)
        outer = D_blocks.unsqueeze(-1) * D_blocks.unsqueeze(-2)  # shape: (ny, nx, N, N)

        # Create an identity matrix for the N dimension.
        I = torch.eye(N, device=x.device, dtype=x.dtype)

        # Compute C_blocks using broadcasting.
        # phiRR and phiR have shape (ny, nx); unsqueeze them to shape (ny, nx, 1, 1) to match outer and I.
        C_blocks = - phiRR.unsqueeze(-1).unsqueeze(-1) * outer \
                - phiR.unsqueeze(-1).unsqueeze(-1) * I.unsqueeze(0).unsqueeze(0)
        # Rearrange and reshape C_blocks to get C of shape (ny * N, nx * N)
        C = C_blocks.permute(0, 2, 1, 3).reshape(ny * N, nx * N)

        # Restore the original gamma.
        self.gamma = oldGamma

        # Concatenate the four blocks:
        # Top block: [K, B1] with shape (ny, nx*(N+1))
        top = torch.cat([K, B1], dim=1)
        # Bottom block: [B2, C] with shape (ny * N, nx*(N+1))
        bottom = torch.cat([B2, C], dim=1)

        # Final concatenation along rows.
        return torch.cat([top, bottom], dim=0)


    def evalGrad(self, y, x, alpha):
        if y.ndim < 2:
            if y.ndim == 1: 
                #y = y.reshape(-1,1)
                y = y[:, np.newaxis]
            else:
                y = np.reshape(y, (1, -1))
            
        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True) + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      

        phiR  = self.phiR(diff)       
        phiRR = self.phiRR(diff)  

        nx    = x.shape[1]
        ny    = y.shape[1]
        N     = x.shape[0]

        diff2          = y[:, :, None] - x[:, None, :]  
        weighted_diff2 = phiR[None, :, :] * diff2   
        B2             = weighted_diff2.transpose(1, 0, 2).reshape(ny * N, nx)
        D_blocks       = diff2.transpose(1, 2, 0) 
        outer          = D_blocks[..., :, None] * D_blocks[..., None, :]  
        I              = np.eye(N)
        C_blocks       = - phiRR[:, :, None, None] * outer - phiR[:, :, None, None] * I  
        C              = C_blocks.transpose(0, 2, 1, 3).reshape(ny * N, nx * N)

        return np.c_[B2, C] @ alpha
    

    def evalFunc(self,y,x,alpha):
        if y.ndim < 2:
            if y.ndim == 1: 
                y = y[:, np.newaxis]
            else:
                y = np.reshape(y, (1, -1))

        diff       = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        K          = self.phi(diff)
        phiR       = self.phiR(diff) 

        nx         = x.shape[1]
        ny         = y.shape[1]
        N          = x.shape[0]
        diff_tensor    = x[:, None, :] - y[:, :, None]  
        weighted_diff  = phiR[None, :, :] * diff_tensor   
        B1             = weighted_diff.transpose(1, 2, 0).reshape(ny, nx * N)
        return (np.c_[K, B1] @ alpha)[0,0]

    def evalFuncTorch(self, y, x, alpha, newGamma=None):
        oldGamma = self.gamma

        if newGamma is not None:
            self.gamma = newGamma

        # Compute the pairwise Euclidean distances between columns of x and y.
        # x: (N, nx), y: (N, ny)
        diff = torch.sqrt(
            torch.abs(
                torch.sum(x**2, dim=0, keepdim=True) +
                torch.sum(y**2, dim=0, keepdim=True).t() -
                2 * torch.matmul(y.t(), x)
            )
        )

        # Evaluate the kernel and its derivative
        K = self.phiTorch(diff)    # Expected shape: (ny, nx)
        phiR = self.phiRTorch(diff)  # Expected shape: (ny, nx)

        nx = x.shape[1]
        ny = y.shape[1]
        N = x.shape[0]

        # Compute the difference tensor.
        # x.unsqueeze(1): (N, 1, nx); y.unsqueeze(2): (N, ny, 1)
        diff_tensor = x.unsqueeze(1) - y.unsqueeze(2)  # shape: (N, ny, nx)
        weighted_diff = phiR.unsqueeze(0) * diff_tensor  # shape: (N, ny, nx)

        # Permute and reshape to match desired dimensions.
        B1 = weighted_diff.permute(1, 2, 0).reshape(ny, nx * N)

        # Restore gamma to its original value.
        self.gamma = oldGamma

        # Concatenate horizontally K and B1, then multiply by alpha.
        return torch.matmul(torch.cat([K, B1], dim=1), alpha)

    def testBuildHermiteGramMatrix(self,y,x,h):
        
        ker   = lambda x,y: self.phi(np.linalg.norm(x-y))
           
        nx    = x.shape[1]
        ny    = y.shape[1]
        N     = x.shape[0]

        K  = np.zeros((ny ,nx ))
        B1 = np.zeros((ny ,nx*N))
        B2 = np.zeros((ny*N,nx ))
        C  = np.zeros((ny*N,nx*N))
        
        for i in range(ny):
            for j in range(nx):
                yTemp   = y[:,i]
                xTemp   = x[:,j]
                b1      = np.zeros(N)
                b2      = np.zeros(N)
                Csmall  = np.zeros((N,N))
                for l in range(N):
                    e1     = np.zeros(N)
                    e1[l]  = 0.5*h
                    b1[l] = (ker(xTemp+e1,yTemp)-ker(xTemp-e1,yTemp))/h 
                    b2[l] = (ker(xTemp,yTemp+e1)-ker(xTemp,yTemp-e1))/h 
                    for s in range(N):
                        e2          = np.zeros(N)
                        e2[s]       = 0.5*h
                        Csmall[l,s] = ((ker(xTemp+e2,yTemp+e1)-ker(xTemp-e2,yTemp+e1))/h -(ker(xTemp+e2,yTemp-e1)-ker(xTemp-e2,yTemp-e1))/h ) /h

                K[i,j]                     = ker(xTemp,yTemp)
                B1[i,j*N:(j+1)*N]          = b1
                B2[i*N:(i+1)*N,j]          = b2
                C[i*N:(i+1)*N,j*N:(j+1)*N] = Csmall

        return np.r_[np.c_[K, B1], np.c_[B2, C]] 



class QuadWendland(Kernel):
    def __init__(self,gamma,d):
        self.l        = np.floor(d/2)+ 2 +1 
        self.gamma    = gamma
    def phi(self,r):   return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+2) * ((self.l**2+4*self.l+3)*(self.gamma*r)**2+(3*self.l+6)*self.gamma*r+3)  
    def phiR(self,r):  return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+1) * (-self.gamma**2) * (12+7*self.l+self.l**2)*(1+(1+self.l)*self.gamma*r)
    def phiRR(self,r): return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l) * self.gamma**4 * (24 + 50 * self.l + 35 *  self.l**2 + 10 *  self.l**3 + self.l**4) 


class QuadMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(3+3*self.gamma*r+self.gamma**2 *r**2)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*(1+self.gamma*r) * self.gamma**2 
    def phiRR(self,r): return self.gamma**4 * np.exp(-self.gamma*r) 

class Gauss(Kernel):
    def phi(self,r):        return np.exp(-self.gamma*(r**2)) 
    def phiR(self,r):       return (-2)*self.gamma*np.exp(-self.gamma*(r**2)) 
    def phiRR(self,r):      return 4*self.gamma**2 * np.exp(-self.gamma*(r**2)) 
    def phiTorch(self,r):   return torch.exp(-self.gamma*(r**2)) 
    def phiRTorch(self,r):  return (-2)*self.gamma*torch.exp(-self.gamma*(r**2)) 
    def phiRRTorch(self,r): return 4*self.gamma**2 * torch.exp(-self.gamma*(r**2)) 

class InvMulti(Kernel):
    def phi(self,r):   return 1/np.sqrt(1+self.gamma*(r**2)) 
    def phiR(self,r):  return (-1)*self.gamma* (1/np.sqrt((1+self.gamma*(r**2))**3)) 
    def phiRR(self,r): return 3*self.gamma**2 * (1/np.sqrt((1+self.gamma*(r**2))**5)) 

class LinMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(1+self.gamma*r)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*self.gamma**2 
    def phiRR(self,r): 
        diffMask1 = r<10**(-14)
        diffMask2 = r>10**(-14)
        return self.gamma**3 * np.exp(-self.gamma*r) * (1/(r+diffMask1)) * (diffMask2)
 