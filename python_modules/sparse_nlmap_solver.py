import time
import resource
import numpy as np
from processify import processify

#from sparse_nlmap_solver import sparse_nlmap_solver

from scipy.sparse.linalg import LinearOperator

class nlmap_operator(LinearOperator):
	def setup(self,amat,regmat,drvmat,wgtmat,reg_drvmat):
		self.amat=amat
		self.regmat = regmat
		self.drvmat = drvmat
		self.wgtmat = wgtmat
		self.reg_drvmat = reg_drvmat
		
	def _matvec(self,vec):
		chi2term = self.drvmat*(self.amat.T*(self.wgtmat*(self.amat*(self.drvmat*vec))))
		regterm = self.reg_drvmat*(self.regmat*(self.reg_drvmat*vec))
		return chi2term+regterm

	def _adjoint(self):
		return self

	def _matmat(self,mat):
		chi2term = self.drvmat*(self.amat.T*(self.wgtmat*(self.amat*(self.drvmat*mat))))
		regterm = self.reg_drvmat*(self.regmat*(self.reg_drvmat*mat))
		return chi2term+regterm
		
# Subroutine to do the inversion. Uses the log mapping and iteration from Plowman & Caspi 2020 to ensure positivity of solutions.
def sparse_nlmap_solver(data0, errors0, amat0, guess=None, reg_fac=1, func=None, dfunc=None, ifunc=None, regmat=None,
						solver=None, sqrmap=False, regfunc=None, dregfunc=None, iregfunc=None, map_reg=False, adapt_lam=True,
						solver_tol = 1.0e-3, niter=40, dtype='float32', steps=None, precompute_ata=False, flatguess=True):
	from scipy.sparse import diags
	from scipy.sparse.linalg import lgmres

	def idnfunc(s): return s
	def iidnfunc(s): return s
	def didnfunc(s): return 0*s
	def expfunc(s): return np.exp(s) #
	def dexpfunc(s): return np.exp(s) #
	def iexpfunc(c): return np.log(c) #
	def sqrfunc(s): return s*s # np.exp(s) #
	def dsqrfunc(s): return 2*s # np.exp(s) #
	def isqrfunc(c): return c**0.5 # np.log(c) #
	if(func is None or dfunc is None or ifunc is None): 
		if(sqrmap): [func,dfunc,ifunc] = [sqrfunc,dsqrfunc,isqrfunc]
		else: [func,dfunc,ifunc] = [expfunc,dexpfunc,iexpfunc]
	if(regfunc is None or dregfunc is None or iregfunc is None):
		if(map_reg): [regfunc,dregfunc,iregfunc] = [idnfunc,didnfunc,iidnfunc]
		else: [regfunc,dregfunc,iregfunc] = [func,dfunc,ifunc]
	if(solver is None): solver = lgmres

	flatdat = data0.flatten().astype(dtype)
	flaterrs = errors0.flatten().astype(dtype)
	flaterrs[flaterrs == 0] = 0.05*np.nanmean(flaterrs[flaterrs > 0])
	
	guess0 = amat0.transpose()*np.clip(flatdat,np.min(flaterrs),None)
	guess0dat = amat0*guess0
	guess0norm = np.sum(flatdat*guess0dat/flaterrs**2)/np.sum((guess0dat/flaterrs)**2)
	guess0 *= guess0norm
	guess0 = np.clip(guess0,0.005*np.mean(np.abs(guess0)),None).astype(dtype)
	if(guess is None): guess = guess0 # np.mean(guess0)+0.0*guess0
	
	[ndat, nsrc] = amat0.shape
	guess = (1+np.zeros(nsrc))*np.mean(flatdat)/np.mean(amat0.dot(1+np.zeros(nsrc)))
	if(flatguess): guess = (1+np.zeros(nsrc))*np.mean(flatdat)/np.mean(amat0.dot(1+np.zeros(nsrc)))
	svec = ifunc(guess)

	# Try these step sizes at each step of the iteration. Trial Steps are fast compared to computing 
	# the matrix inverse, so having a significant number of them is not a problem.
	# Step sizes are specified as a fraction of the full distance to the solution found by the sparse
	# matrix solver (lgmres or bicgstab).
	if(steps is None): steps = np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85],dtype=dtype)
	nsteps = len(steps)
	step_chi2 = np.zeros(nsteps,dtype=dtype)
	
	if(regmat is None): regmat = diags(1.0/iregfunc(guess0)**2)
	if(adapt_lam): 
		reglam = np.dot(dregfunc(svec)*(regmat*regfunc(svec)),dfunc(svec)*(amat0.T*(1.0/flaterrs)))/np.dot(dregfunc(svec)*(regmat*regfunc(svec)),dregfunc(svec)*(regmat*regfunc(svec)))
	else: reglam = 1
	# Still appears to be some issue with this regularization factor?
	regmat = reg_fac*regmat*reglam #*(ndat/nsrc)
	weights = 1.0/flaterrs**2 # The weights are the errors...

	print('Overall regularization factor:',reg_fac*reglam)

	if(not(precompute_ata)):
		nlmo = nlmap_operator(dtype=dtype,shape=(nsrc,nsrc))
		nlmo.setup(amat0,regmat,diags(dfunc(svec)),diags(weights),diags(dregfunc(svec)))
	
	# --------------------- Now do the iteration:
	tstart = time.time()
	for i in range(0,niter):
		# Setup intermediate matrices for solution:
		dguess = diags(dfunc(svec),dtype=dtype)
		dregguess = diags(dregfunc(svec),dtype=dtype)
		bvec = dguess*amat0.T*diags(weights)*(flatdat-amat0*(func(svec)-svec*dfunc(svec)))
		bvec -= dregguess*(regmat*(regfunc(svec)-svec*dregfunc(svec)))
				
		# Run sparse matrix solver:
		if(precompute_ata):
			chi2term = dguess*(amat0.T*(diags(weights)*(amat0*(dguess))))
			regterm = dguess*(regmat*(dguess))
			svec2 = solver(chi2term+regterm,bvec,svec,tol=solver_tol)
		else:
			print(np.min(guess),np.max(guess),np.min(svec),np.max(svec))
			[nlmo.drvmat,nlmo.reg_drvmat] = [dguess,dregguess]
			svec2 = solver(nlmo,bvec,svec,tol=solver_tol)

		# Try the step sizes:
		for j in range(0,nsteps):
			#stepguess = func(svec+steps[j]*(delt_s[0]))
			stepguess = func(svec+steps[j]*(svec2[0]-svec))
			stepguess_reg = regfunc(svec+steps[j]*(svec2[0]-svec))
			stepresid = (flatdat-amat0.dot(stepguess))*weights**0.5
			step_chi2[j] = np.dot(stepresid,stepresid)/ndat + np.sum(stepguess_reg.T*regmat*stepguess_reg)/ndat

		best_step = np.argmin(step_chi2[1:nsteps])+1 # First step is zero for comparison purposes...
		chi20 = np.sum(weights*(flatdat-amat0.dot(func(svec)))**2)/ndat # step_chi2[0]-step_chi2[best_step]
		reg0 = np.sum(regfunc(svec.T)*regmat*regfunc(svec))/ndat
		
		# Update the solution with the step size that has the best Chi squared:
		# guess = np.exp(np.log(guess)+steps[best_step]*(guess2[0]-np.log(guess)))
		svec = svec+steps[best_step]*(svec2[0]-svec)
		
		reg1 = np.sum(regfunc(svec.T)*regmat*regfunc(svec))/ndat
		chi21 = np.sum(weights*(flatdat-amat0.dot(func(svec)))**2)/ndat
				
		print(round(time.time()-tstart,2),'s i =',i,'chi2 =',round(chi21,2),'step size =',round(steps[best_step],3), 'reg. param. =', round(reg1,2), 'chi2 change =',round(chi20-chi21,5), 'reg. change =',round(reg0-reg1,5))
		print(i,np.mean((svec)),np.std((svec)),np.min((svec)),np.max((svec)))	
		if(np.abs(step_chi2[0]-step_chi2[best_step]) < 1.0e-15): break # Finish the iteration if chi squared isn't changing

	return func(svec), chi21




class nlmap_operator0(LinearOperator):
    def setup(self,amat,regmat,drvmat,wgtmat):
        self.amat=amat
        self.regmat = regmat
        self.drvmat = drvmat
        self.wgtmat = wgtmat
        
    def _matvec(self,vec):
        return self.drvmat*self.regmat*self.drvmat*vec+self.drvmat*(self.amat.transpose()*(self.wgtmat*(self.amat*self.drvmat*vec)))

# Subroutine to do the inversion. Uses the log mapping and iteration from Plowman & Caspi 2020 to ensure positivity of solutions.
def sparse_nlmap_solver0(data0, errors0, amat0, guess=None, reg_fac=1, func=None, dfunc=None, ifunc=None, solver=None, sqrmap=True, map_reg=False, regmat=None, solver_tol = 5.0e-5, niter=40, dtype='float32'):
    from scipy.sparse import diags
    from scipy.sparse.linalg import bicgstab
    from scipy.sparse.linalg import lgmres

    def expfunc(s): return np.exp(s) #
    def dexpfunc(s): return np.exp(s) #
    def iexpfunc(c): return np.log(c) #
    def sqrfunc(s): return s*s # np.exp(s) #
    def dsqrfunc(s): return 2*s # np.exp(s) #
    def isqrfunc(c): return c**0.5 # np.log(c) #
    if(func is None or dfunc is None or ifunc is None): 
        if(sqrmap): [func,dfunc,ifunc] = [sqrfunc,dsqrfunc,isqrfunc]
        else: [func,dfunc,ifunc] = [expfunc,dexpfunc,iexpfunc]
    if(solver is None): solver = bicgstab

    flatdat = data0.flatten().astype(dtype)
    flaterrs = errors0.flatten().astype(dtype)
    flaterrs[flaterrs == 0] = 0.05*np.nanmean(flaterrs[flaterrs > 0])
    guess_under_fac = 1.0
	
    
    guess0 = amat0.transpose()*np.clip(flatdat,np.min(flaterrs),None)
    guess0dat = amat0*guess0
    guess0norm = np.sum(flatdat*guess0dat/flaterrs**2)/np.sum((guess0dat/flaterrs)**2)
    guess0 *= guess0norm
    guess0 = np.clip(guess0,0.005*np.mean(np.abs(guess0)),None).astype(dtype)
    if(guess is None): guess = guess0 # np.mean(guess0)+0.0*guess0
    svec = ifunc(guess)
    
    [ndat, nsrc] = amat0.shape

    # Try these step sizes at each step of the iteration. Trial Steps are fast compared to computing 
    # the matrix inverse, so having a significant number of them is not a problem.
    # Step sizes are specified as a fraction of the full distance to the solution found by the sparse
    # matrix solver (lgmres or bicgstab).
    steps = np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85],dtype=dtype)
    nsteps = len(steps)
    step_chi2 = np.zeros(nsteps,dtype=dtype)
    
    #if(regmat is None): regmat = diags(np.ones(nsrc,dtype='float32'))
    if(map_reg):
       if(regmat is None): regmat = diags(1.0/(ifunc(guess0))**2)
       reglam = np.dot((regmat*svec),dfunc(svec)*(amat0.T*(1.0/flaterrs)))/np.dot((regmat*svec),(regmat*svec))
    else: 
        if(regmat is None): regmat = diags(1.0/(guess0)**2)
        reglam = np.dot(dfunc(svec)*(regmat*guess),dfunc(svec)*(amat0.T*(1.0/flaterrs)))/np.dot(dfunc(svec)*(regmat*guess),dfunc(svec)*(regmat*guess))
        # Still appears to be some issue with this regularization factor?
    regmat = reg_fac*regmat*reglam #*(ndat/nsrc)
    weights = 1.0/flaterrs**2 # The weights are the errors...
    guess *= guess_under_fac
    
    nlmo = nlmap_operator(dtype=dtype,shape=(nsrc,nsrc))
    nlmo.setup(amat0,regmat,diags(dfunc(svec)),diags(weights))
    
    # --------------------- Now do the iteration:
    tstart = time.time()
    for i in range(0,niter):
        # Setup intermediate matrices for solution:
        dguess = diags(dfunc(svec),dtype=dtype)
        bvec = dguess*amat0.transpose().dot((flatdat-amat0.dot(func(svec)-svec*dfunc(svec)))*weights) 
        if(map_reg == False): bvec -= dfunc(svec).T*regmat*(func(svec)-svec*dfunc(svec))
        
        nlmo.drvmat = dguess
        
        # print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Run sparse matrix solver:
        if(map_reg):
            svec2 = solver(regmat+dguess*(amat0.transpose()*diags(weights)*amat0)*dguess,bvec,svec,tol=solver_tol)
        else:
            #svec2 = solver(dguess*(regmat+amat0.transpose()*diags(weights)*amat0)*dguess,bvec,svec,tol=solver_tol)
            svec2 = solver(nlmo,bvec,svec,tol=solver_tol)

        # Try the step sizes:
        for j in range(0,nsteps):
            stepguess = func(svec+steps[j]*(svec2[0]-svec))
            stepresid = (flatdat-amat0.dot(stepguess))*weights**0.5
            step_chi2[j] = np.dot(stepresid,stepresid)/ndat + np.sum(stepguess.T*regmat*stepguess)/ndat

        best_step = np.argmin(step_chi2[1:nsteps])+1 # First step is zero for comparison purposes...
        chi20 = np.sum(weights*(flatdat-amat0.dot(func(svec)))**2)/ndat # step_chi2[0]-step_chi2[best_step]
        reg0 = np.sum(func(svec.T)*regmat*func(svec))/ndat
        
        # Update the solution with the step size that has the best Chi squared:
        # guess = np.exp(np.log(guess)+steps[best_step]*(guess2[0]-np.log(guess)))
        svec = svec+steps[best_step]*(svec2[0]-svec)
        
        reg1 = np.sum(func(svec.T)*regmat*func(svec))/ndat
        chi21 = np.sum(weights*(flatdat-amat0.dot(func(svec)))**2)/ndat
                
        print(round(time.time()-tstart,2),'s i =',i,'chi2 =',round(chi21,2),'step size =',round(steps[best_step],3), 'reg. param. =', round(reg1,2), 'chi2 change =',round(chi20-chi21,5), 'reg. change =',round(reg0-reg1,5))
        if(np.abs(step_chi2[0]-step_chi2[best_step]) < 1.0e-4): break # Finish the iteration if chi squared isn't changing
    guess = func(svec)
    #print(reglam,np.dot(regmat*guess,(amat0.T*(1.0/flaterrs)))/np.dot(regmat*guess,regmat*guess))
    

    return func(svec), chi21
