import numpy as np
import pickle
import time


class Surrogate():
    def __init__(self,kernel):
        self.trainError = []
        self.alpha      = []
        self.center     = []
        self.kernel     = kernel
        self.C          = np.array([])

    def saveSr(self,name):
        with open("data/"+name, 'wb') as outp:
             pickle.dump(self, outp, protocol=4)

    def loadSr(self,name):
        with open("data/"+name, 'rb') as inp:
             oldSelf         = pickle.load(inp)
             self.trainError = oldSelf.trainError
             self.alpha      = oldSelf.alpha    
             self.center     = oldSelf.center          
             self.kernel     = oldSelf.kernel    
             self.C          = oldSelf.C    
             self.FCenterVal = oldSelf.FCenterVal

    def doFGreedy(self,F,rhs,testPonits,nMaxGreedy,observer,eps=10**(-16),outputFlag = True):

        rhsTestPoints       = rhs(testPonits)
        fy                  = F(testPonits)
        KYX                 = np.zeros((testPonits.shape[1],nMaxGreedy))
        center              = np.zeros((testPonits.shape[0],nMaxGreedy+1))

        idx                 = np.argmax(np.abs(rhs(testPonits)))
        self.trainError     = np.atleast_1d(np.abs(rhs(testPonits))[idx])
        self.center         = np.atleast_2d(testPonits[:,idx]).T
        center[:,0]         = testPonits[:,idx]
        i                   = 0
        self.trainError[-1] = eps+1
        while i <nMaxGreedy and self.trainError[-1]>eps:
            self.fit(F,rhs,center[:,:i+1])
            if i==0:
                KYX[:,0] = self.kernel.getGramPDE(F,testPonits,center[:,:i+1])[:,0] 
            else:         
                KYX[:,i] = self.kernel.getGramPDE(F,testPonits,np.atleast_2d(center[:,i]).T,fy)[:,0] 
            preOut          = KYX[:,:i+1]@self.alpha
            res             = np.abs(preOut-rhsTestPoints)/(np.abs(rhsTestPoints)+10**(-8))
            idx             = np.argmax(res)
            self.trainError = np.r_[self.trainError,res[idx]]
            center[:,i+1]   = testPonits[:,idx]
            observer.addObjectGreedyError(self.trainError[-1])
            if outputFlag:
                print(str(i) + " Epoch: Error = " + str( self.trainError[-1] ))
            i               = i+1
        self.fit(F,rhs,center)

    def fit(self,F,rhs,center, iterativ= True):
        if iterativ:
            if self.C.shape[0]==0:
                K           = self.kernel.getGramPDE(F,center,center)
                rhsVal      = rhs(center)
                self.alpha  = np.linalg.solve(K,rhsVal)
                self.center = center
                L           = np.linalg.cholesky(K)
                self.C      = np.linalg.solve(L,np.eye(L.shape[0]))
            else:
                #nor1        = np.dot(self.alpha,rhs(center[:,:-1]))
                rhsVal      = rhs(center)
                KX          = self.kernel.getGramPDE(F,center[:,:-1],np.atleast_2d(center[:,-1]).T)    
                KXX         = self.kernel.getGramPDE(F,np.atleast_2d(center[:,-1]).T,np.atleast_2d(center[:,-1]).T)    
                dm          = self.C @ KX
                dmm         = np.sqrt(KXX- (dm.T @ dm))    
                cmm         = 1/dmm
                cm          = - self.C.T @ (dm @ cmm.T)
                self.C      = np.c_[np.r_[self.C,cm.T],np.r_[cm*0,cmm]]
                self.alpha  = self.C.T @ ( self.C @ rhsVal)
                self.center = center
                #nor2        = np.dot(self.alpha,rhsVal)
                #print((nor2-nor1)/nor1)
                
        else:        
            K           = self.kernel.getGramPDE(F,center,center)
            rhsVal      = rhs(center)
            self.alpha  = np.linalg.solve(K,rhsVal)
            self.center = center






