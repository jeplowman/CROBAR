import numba, time, numpy as np
from scipy.integrate import solve_ivp
from processify import processify

@numba.jit(nopython=True, fastmath=True, parallel=True)
def compute_lff_field_old(l, r, xyz, mag, alpha, tiny=1.0e-20):

	[dx, dy, dz] = [r[0]-xyz[0], r[1]-xyz[1], r[2]-xyz[2]]
	bigr2 = dx*dx+dy*dy
	lilr2i = 1/(dz*dz+bigr2)
	lilri = np.sqrt(lilr2i)
	alr = alpha/lilri
	cal = np.cos(alr)*lilri

	bz = np.sum(dz*(alpha*np.sin(alr)+cal)*lilr2i*mag)

	adz = alpha*dz
	gambardz = (cal + alpha*np.sin(adz))*lilr2i*mag
	gambar = mag*alpha*(dz*cal - np.cos(adz))/(bigr2+tiny)

	return np.sum(dx*gambardz+dy*gambar), np.sum(dy*gambardz-dx*gambar), bz

def compute_lff_field_slow(l,r,xyz,mag,alpha,tiny=1.0e-20):
	[dx, dy, dz] = [r[0]-xyz[0], r[1]-xyz[1], r[2]-xyz[2]]
	lilr = np.sqrt(dx**2+dy**2+dz**2)
	bigr = np.sqrt(dx**2+dy**2)
	gamma = dz*np.cos(alpha*lilr)/((tiny+bigr)*lilr) - np.cos(alpha*dz)/(bigr+tiny)
	car = np.cos(alpha*lilr)
	caz = np.cos(alpha*dz)
	sar = np.sin(alpha*lilr)
	saz = np.sin(alpha*dz)
	#dgdz = bigr*car/lilr**3 - alpha*(sar*dz*dz/lilr*lilr+saz)/(bigr+tiny)
	#dgdR = caz/(tiny+bigr*bigr) - dz*car/(lilr*(tiny+bigr*bigr)) - alpha*bigr*sar/(lilr*lilr) - dz*car/lilr**3
	dgdz = (alpha*saz + car/lilr - alpha*dz*dz*sar/(lilr*lilr) - dz*dz*car/lilr**3)/(bigr+tiny)
	dgdR = (caz-dz*car/lilr)/(bigr*bigr+tiny) - alpha*dz*sar/(lilr*lilr) - dz*car/lilr**3
	gx = dgdz*dx/(bigr+tiny) + alpha*gamma*dy/(bigr+tiny)
	gy = dgdz*dy/(bigr+tiny) - alpha*gamma*dx/(bigr+tiny)
	gz = -dgdR-gamma/(bigr+tiny)
	return np.sum(mag*gx), np.sum(mag*gy), np.sum(mag*gz)

def compute_lff_field00(l,r,xyz,mag,alpha,tiny=1.0e-20):
	[dx, dy, dz] = [r[0]-xyz[0], r[1]-xyz[1], r[2]-xyz[2]]
	bigr2 = dx*dx+dy*dy
	#dz2 = dz*dz
	B_bigr2i = mag/(bigr2+tiny)
	lilr = np.sqrt(bigr2+dz*dz)
	alr = alpha*lilr
	adz = alpha*dz
	#caz = np.cos(adz)
	#sar = np.sin(alr)
	#saz = np.sin(adz)
	lilri = 1/lilr
	#lilr2i = lilri*lilri
	car_lilri = np.cos(alr)*lilri

	#bigr = np.sqrt(dx**2+dy**2)
	grnz = (adz*np.sin(alr) + dz*car_lilri)*lilri*lilri
	BgammaoverR = (dz*car_lilri - np.cos(adz))*B_bigr2i
	#BdgdzoverR = (alpha*saz + car_lilri - lilr2i*dz2*(alpha*sar+car_lilri))*B_bigr2i
	BdgdzoverR = (alpha*np.sin(adz) + car_lilri - dz*grnz)*B_bigr2i
	#dgdzoverR = (alpha*saz + car*lilri - alpha*dz2*sar*lilr2i - dz2*car*lilri*lilr2i)*bigr2i
	#BdgdR = (caz-dz*car_lilri)*B_bigr2i - mag*(adz*sar + dz*car_lilri)*lilr2i
	#BdgdR = -BgammaoverR - mag*(adz*sar + dz*car_lilri)*lilr2i
	Bgx = np.sum(dx*BdgdzoverR) + alpha*np.sum(dy*BgammaoverR)
	Bgy = np.sum(dy*BdgdzoverR) - alpha*np.sum(dx*BgammaoverR)
	#Bgz = np.sum(-BdgdR-BgammaoverR)
	#Bgz = np.sum(mag*(adz*sar + dz*car_lilri)*lilr2i)
	Bgz = np.sum(mag*grnz)
	return Bgx, Bgy, Bgz


@numba.jit(nopython=True, fastmath=True, parallel=True)
def compute_lff_field(l,r,xyz,mag,alpha,tiny=1.0e-20):
    dx = r[0]-xyz[0]
    dy = r[1]-xyz[1]
    dz = r[2]-xyz[2]
    bigr2 = dx*dx+dy*dy # XY coordinate displacement vector (squared)
    B_bigr2i = mag/(bigr2+tiny)
    lilr = np.sqrt(bigr2+dz*dz) # overall displacement vector
    alr = alpha*lilr
    adz = alpha*dz
    lilri = 1/lilr
    car_lilri = np.cos(alr)*lilri

    grnz = (adz*np.sin(alr) + dz*car_lilri)*lilri*lilri # z component of Green's function
    BgammaoverR = (dz*car_lilri - np.cos(adz))*B_bigr2i
    BdgdzoverR = (alpha*np.sin(adz) + car_lilri - dz*grnz)*B_bigr2i
    return np.sum(dx*BdgdzoverR) + alpha*np.sum(dy*BgammaoverR), np.sum(dy*BdgdzoverR) - alpha*np.sum(dx*BgammaoverR), np.sum(mag*grnz)

@numba.jit(nopython=True, fastmath=True, parallel=True)
def compute_potential_field(l, r, xyz, mag, alpha, tiny=1.0e-20):

	[dx, dy, dz] = [r[0]-xyz[0], r[1]-xyz[1], r[2]-xyz[2]]
	magr3a = mag/(dx*dx+dy*dy+dz*dz)**1.5
	return np.sum(dx*magr3a), np.sum(dy*magr3a), np.sum(dz*magr3a)
	#bigr2 = dx*dx+dy*dy
	#lilr2i = 1/(dz*dz+bigr2)
	#lilri = np.sqrt(lilr2i)
	#alr = alpha/lilri
	#cal = np.cos(alr)*lilri

	#bz = np.sum(dz*(alpha*np.sin(alr)+cal)*lilr2i*mag)

	#adz = alpha*dz
	#gambardz = (cal + alpha*np.sin(adz))*lilr2i*mag
	#gambar = mag*alpha*(dz*cal - np.cos(adz))/(bigr2+tiny)

	#return np.sum(dx*gambardz+dy*gambar), np.sum(dy*gambardz-dx*gambar), bz

class interp_evaluator:
	from scipy.interpolate import RegularGridInterpolator
	def __init__(self, xvals, yvals, zvals, gridx, gridy, gridz):
		self.xinterpolator = RegularGridInterpolator((gridx,gridy,gridz),xvals)
		self.yinterpolator = RegularGridInterpolator((gridx,gridy,gridz),yvals)
		self.zinterpolator = RegularGridInterpolator((gridx,gridy,gridz),zvals)

	def eval_func(self,l, r, xyz, mag, alpha, tiny=1.0e-20):
		return self.xinterpolator(r), self.yinterpolator(r), self.zinterpolator(r)


from scipy.interpolate import RegularGridInterpolator
class rgi_evaluator:
	def __init__(self, xvals, yvals, zvals, gridx, gridy, gridz):
		self.signs = np.array([np.sign(gridx[1]-gridx[0]),np.sign(gridy[1]-gridy[0]),np.sign(gridz[1]-gridz[0])])
		grids = (self.signs[0]*gridx,self.signs[1]*gridy,self.signs[2]*gridz)
		self.xinterpolator = RegularGridInterpolator(grids,xvals,bounds_error=False, fill_value=None)
		self.yinterpolator = RegularGridInterpolator(grids,yvals,bounds_error=False, fill_value=None)
		self.zinterpolator = RegularGridInterpolator(grids,zvals,bounds_error=False, fill_value=None)

	def eval_func(self, l, r, xyz, mag, alpha, tiny=1.0e-20):
		return self.xinterpolator(self.signs*r)[0], self.yinterpolator(self.signs*r)[0], self.zinterpolator(self.signs*r)[0]

def tril_kernel(vals, ixyz0, dxyz):
    c00 = vals[ixyz0[0],ixyz0[1],ixyz0[2]]*(1-dxyz[0]) + vals[ixyz0[0]+1,ixyz0[1],ixyz0[2]]*dxyz[0]
    c01 = vals[ixyz0[0],ixyz0[1],ixyz0[2]+1]*(1-dxyz[0]) + vals[ixyz0[0]+1,ixyz0[1],ixyz0[2]+1]*dxyz[0]
    c10 = vals[ixyz0[0],ixyz0[1]+1,ixyz0[2]]*(1-dxyz[0]) + vals[ixyz0[0]+1,ixyz0[1]+1,ixyz0[2]]*dxyz[0]
    c11 = vals[ixyz0[0],ixyz0[1]+1,ixyz0[2]+1]*(1-dxyz[0]) + vals[ixyz0[0]+1,ixyz0[1]+1,ixyz0[2]+1]*dxyz[0]
    c0 = c00*(1-dxyz[1]) + c10*dxyz[1]
    c1 = c01*(1-dxyz[1]) + c11*dxyz[1]
    return c0*(1-dxyz[2]) + c1*dxyz[2]
    
class tril_evaluator:
    def __init__(self, xvals, yvals, zvals, gridx, gridy, gridz):
        self.xyz0 = np.array([gridx[0], gridy[0], gridz[0]])
        [self.xvals, self.yvals, self.zvals] = [xvals, yvals, zvals]
        self.maxinds = np.array(self.xvals.shape)-1
        self.dr = np.array([np.mean(gridx[1:]-gridx[0:-1]),
                            np.mean(gridy[1:]-gridy[0:-1]),
                            np.mean(gridz[1:]-gridz[0:-1])])
        
    def eval_func(self, l, r, xyz, mag, alpha, tiny=1.0e-20):
        ixyz0 = np.clip(np.floor((r-self.xyz0)/self.dr).astype(np.int32),0,self.maxinds-1)
        xyz0 = self.dr*ixyz0+self.xyz0
        dxyz = (r-xyz0)/self.dr
        return tril_kernel(self.xvals,ixyz0,dxyz), tril_kernel(self.yvals,ixyz0,dxyz), tril_kernel(self.zvals,ixyz0,dxyz)

#@processify
def tracer(initial_points, mag, mag_coords, alpha = 0.0, rtol=1.0e-4, atol=None, boundrad=None,
					evaluator=compute_lff_field, bounds=None, check_bounds=None, direction=1):
					
	if(bounds is None): 
		bounds = np.array([[np.min(c),np.max(c)] for c in mag_coords]).T
		bounds[1,2] = bounds[0,2] + np.min(bounds[1,0:2]-bounds[0,0:2])

	if(atol is None): atol = np.mean(bounds[1]-bounds[0])*1.0e-5
	lmax = 3.0*np.sum(bounds[1,:]-bounds[0,:])
	#if(atol is None): atol = .05*dx

	def compute_force(l,r,mag_coords,mag,direction,bounds,alpha,bvecs):
		[bx,by,bz] = evaluator(l,r,mag_coords,mag,alpha)
		bmag = np.sign(direction)*(bx**2+by**2+bz**2)**0.5
		bvecs.append([r,np.array([bx,by,bz])])
		return bx/bmag,by/bmag,bz/bmag
    
	if(check_bounds is None):
		bounds = [bounds,boundrad]
		def check_bounds(l,r_in,mag_coords,mag,direction,bounds,alpha,bvecs):
			if(boundrad is None):
				xyh=r_in
			else:
				height = np.sqrt(np.sum(r_in[0:2]**2)+(r_in[2]+boundrad)**2)-boundrad
				xyh = [r_in[0],r_in[1],height]
			#z0_offset = bounds[1]*(1.0-np.sqrt(1.0-(r_in[0]*r_in[0]+r_in[1]*r_in[1])/(bounds[1]*bounds[1])))
			#r = [r_in[0],r_in[1],r_in[2]+z0_offset]
			#print('checked', r, z0_offset)
			return 1-2*(np.sum((xyh < bounds[0][0,:]) + (bounds[0][1,:] < xyh)) > 0)
	#def check_bounds0(l,r,mag_coords,mag,direction,bounds,alpha,bvecs):
	#	return 1-2*(np.sum((r < bounds[0,:]) + (bounds[1,:] < r)) > 0)
	[check_bounds.terminal,check_bounds.direction] = [True,-1]

	ivpevts = (check_bounds)
	ivpargs_fwd = (mag_coords,mag,5*atol*direction,bounds,alpha,[])
	ivpargs_rvs = (mag_coords,mag,-5*atol*direction,bounds,alpha,[])

	#print(atol/1.0e8, 1.0/alpha/1.0e8)
	#print((bounds[1]-bounds[0])/1.0e8)

	[lines, bvecs, tstart] = [[], [], time.time()]
	for i in range(0,len(initial_points)):
		output_fwd = solve_ivp(compute_force, [0,lmax], initial_points[i], method='RK23', 
				args=ivpargs_fwd, events=ivpevts,first_step=5*atol,rtol=rtol,atol=atol)
		output_rvs = solve_ivp(compute_force, [0,lmax], initial_points[i], method='RK23', 
				args=ivpargs_rvs, events=ivpevts,first_step=5*atol,rtol=rtol,atol=atol)
		rfwd = np.vstack([arg[0] for arg in ivpargs_fwd[5]])
		bfwd = np.vstack([arg[1] for arg in ivpargs_fwd[5]])
		rrvs = np.vstack([arg[0] for arg in ivpargs_rvs[5]])
		brvs = np.vstack([arg[1] for arg in ivpargs_rvs[5]])
		bvecs_fwd = np.zeros([3,len(output_fwd.y.T)])
		bvecs_rvs = np.zeros([3,len(output_rvs.y.T)])
		for j in range(0,len(output_fwd.y.T)): bvecs_fwd[:,j] = bfwd[np.argmin(np.sum((output_fwd.y.T[j]-rfwd)**2,axis=1))]
		for j in range(0,len(output_rvs.y.T)): bvecs_rvs[:,j] = brvs[np.argmin(np.sum((output_rvs.y.T[j]-rrvs)**2,axis=1))]
		lines.append(np.vstack((np.flipud(output_rvs.y[:,1:].T),output_fwd.y.T)))
		bvecs.append(np.vstack((np.flipud(bvecs_rvs[:,1:].T),bvecs_fwd.T)))
		if((i+1)%50 == 0): print('Done with line ', i,' at ', time.time()-tstart, ' s')
		ivpargs_fwd[5].clear()
		ivpargs_rvs[5].clear()

	return lines, bvecs
