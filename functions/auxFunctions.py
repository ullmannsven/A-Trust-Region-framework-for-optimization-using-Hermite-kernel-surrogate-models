import numpy as np
import pickle

def kernelPI(model,surrogate,nMaxPI,nMaxGreedy,observer,testPoint,VFinput,VFoutputTrue):
    for j in range(nMaxPI):
        if  j==0:
            mu              = lambda x: model.stableControl(x)
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x))
            surrogate.doFGreedy(F,rhs,testPoint,nMaxGreedy,observer,10**(-8))
            maxFGreedyError = surrogate.trainError[-1]
            oldSRVal        = 0
        else:
            mu              = lambda x: model.getMuFromSr(x,surrogate.kernel.evalGard(FCenterVal,x,center,alpha))
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x)) 
            surrogate.fit(F,rhs,surrogate.center, iterativ= False)
            maxFGreedyError = np.max(np.abs(surrogate.kernel.getGramPDE(F,testPoint,surrogate.center)@surrogate.alpha-rhs(testPoint))/(np.abs(rhs(testPoint))+10**(-8)))
        FCenterVal = F(surrogate.center).copy()
        center     = surrogate.center.copy()
        alpha      = surrogate.alpha.copy()
        trueError  = np.sqrt( np.sum(np.abs(VFoutputTrue-surrogate.kernel.evalFunc(FCenterVal,VFinput,center,alpha))**2)/np.sum(np.abs(VFoutputTrue)**2))
        print(str(j) + " Iteration: True-Error = " + str( trueError ) + ", Residual-Error = " + str(maxFGreedyError) +" stagnation-Error = " + str( np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))) ))
        observer.addObject(trueError,0,maxFGreedyError,np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))))
        oldSRVal = surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha)    





def findBestGamma(model,surrogate,observer,points,gammaList,kfold,nMaxGreedy):
    idx     = np.arange(points.shape[1])
    np.random.shuffle(idx)
    idx     = np.reshape(idx,(kfold,-1))
    valList = np.ones(len(gammaList))
    for ol in range(len(gammaList)):
        newKernel   = surrogate.kernel 
        newKernel.setGamma(gammaList[ol])
        mu          = lambda x: model.stableControl(x)
        F           = lambda x: model.getF(x,mu(x))
        rhs         = lambda x: model.getRHS(x,mu(x))
        valList[ol] = 0
        for j in range(idx.shape[0]):      
            trainPointTemp  = points[:,np.delete(idx, j, axis=0).flatten()]
            testPointsTemp  = points[:,idx[j,:]]
            if idx.shape[0] == 1:
               trainPointTemp = testPointsTemp
            surrogate.doFGreedy(F,rhs,trainPointTemp,nMaxGreedy,observer,10**(-8),outputFlag = False)
            valList[ol]     = valList[ol] + np.max(np.abs(surrogate.kernel.getGramPDE(F,testPointsTemp,surrogate.center)@surrogate.alpha-rhs(testPointsTemp))/np.abs(rhs(testPointsTemp)+10**(-8)))
            surrogate.C         = np.array([])
        valList[ol] = valList[ol]/idx.shape[0]
        with open("data/currentFindBestGamma", 'wb') as outp:
            pickle.dump([gammaList,valList] , outp, protocol=4) 
        print(str(ol+1)+" iteration: " + str(100*(ol+1)/len(gammaList))+"% finished: MinVal is "+str(np.min(valList))+" and gamma = "+str(gammaList[np.argmin(valList)]))
    return gammaList[np.argmin(valList)]    
        