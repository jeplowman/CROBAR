import time, sunpy, resource, astropy.units as u, numpy as np, scipy.ndimage as ndimage
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from processify import processify
from get_cropmap0 import get_cropmap0
from local_cartesian_transform import transform, vox2pix, wrld2pix
from scipy.sparse import diags
from scipy.sparse.linalg import lgmres
from sparse_nlmap_solver import sparse_nlmap_solver

#def vox2pix(vox,voxmin,dvox,voxel_map,pixel_map,unit=u.cm):    
#    vox_lct = transform(voxel_map.center)    
#    vox_coord = vox_lct.coord((vox*dvox+voxmin)*u.cm)
#    pixel = pixel_map.world_to_pixel(vox_coord)    
#    return np.array([pixel.y.value, pixel.x.value],dtype=np.float32)

def get_kernel_2d(size0,dvox_in):
	dvox = dvox_in[0:2]
	ev_rad = np.array(0.5*size0/dvox)
	kern_rtot = 4*np.ceil(ev_rad).astype('int32')
	sz_voxels = 2*kern_rtot+1
	area_voxels = np.prod(sz_voxels)
	[vox_ixa,vox_iya] = np.indices(sz_voxels,dtype='int32')
	vox_ixa0 = (vox_ixa.flatten())-kern_rtot[0]
	vox_iya0 = (vox_iya.flatten())-kern_rtot[1]
	event_nvox = len(vox_ixa0)
	kernel = np.exp(-0.5*(((vox_ixa0)/(ev_rad[0]))**2+((vox_iya0)/(ev_rad[1]))**2))
	kernel = kernel/np.sum(kernel)
	kflat = kernel.flatten()
	kfsort = np.sort(kflat)
	thold = np.min(kfsort[np.cumsum(kfsort) > 5.0e-2])
	wkern = np.where(kernel > thold)
	kflat = kflat[wkern].astype('float32')
	kflat = kflat/np.sum(kflat)
	vox_ixa0 = vox_ixa0[wkern]
	vox_iya0 = vox_iya0[wkern]
	vox_ioffs = np.vstack([vox_ixa0,vox_iya0]).astype('int32')
	event_nvox = len(vox_ixa0)
	kernones = 1+np.zeros(event_nvox,dtype='int32')
	return vox_ioffs,kflat,ev_rad,area_voxels,kernones,np.reshape(kernel,sz_voxels)

def get_psfmat(outsize,bin_fac,psfsize_px,dvox,pxsz):
	from scipy.sparse import coo_matrix
	outsize_hi = (bin_fac*np.array(outsize)).astype('int64')
	nhi = outsize_hi[0]*outsize_hi[1]
	nlo = outsize[0]*outsize[1]
	kernel_diameter = ((2*psfsize_px)**2+(4.0/np.pi)*(1.0+(np.max(dvox)/pxsz)**2+(1.0/bin_fac)**2))**0.5
	[vox_ioffs,kflat,ev_rad,area_voxels,kernones,kernel] = get_kernel_2d(kernel_diameter,np.array([1,1])/bin_fac)
	kernsize = len(kflat)

	kflat *= bin_fac**2 # Needed to ensure proper normalization of the psf matrix...

	[xalo,yalo] = np.indices(outsize)
	xalo = np.array(xalo).flatten()
	yalo = np.array(yalo).flatten()
	indices_hi = np.zeros([nlo,kernsize],dtype='uint32')
	indices_lo = np.zeros([nlo,kernsize],dtype='uint32')
	kernvals = np.zeros([nlo,kernsize],dtype='float16')
	for i in range(0,nlo):
		ix = bin_fac*xalo[i]+vox_ioffs[0][:]
		iy = bin_fac*yalo[i]+vox_ioffs[1][:]
		boundflg = (ix >= 0)*(ix < outsize_hi[0])*(iy >=0)*(iy < outsize_hi[1])
		indices_hi[i,boundflg] = ix[boundflg]*outsize_hi[1]+iy[boundflg]
		indices_lo[i,:] = xalo[i]*outsize[1]+yalo[i]
		kernvals[i,boundflg] = kflat[boundflg]
	return coo_matrix((kernvals.flatten(),(indices_hi.flatten(),indices_lo.flatten())),shape=(nhi,nlo))

#@processify
def compute_fwdmat(emismap_name, errmap_name, loopids, loopid_info, vox_grid, cropr, voxmap,
					voxlengths, temp1=1.5e6, temp2=0.5e6, xpo1=2.0, xpo2=2.0, zmax = None, vox_lct=None,
					hfac=60.0e8/1.0e6, sigma_fac = 500.0, amat0=None, zmin=0.0, bin_fac=3, vox_mask=None,
					loop_weights=None, psf_size_px = 0.7, asym=False, emprofs=None, curvature=True,
					auxprof = None, temps=None, logt=None, tresp=None, map_in=None, obscenter=None, voxcenter=None, 
					obswcs=None, voxwcs=None, temp_xpo = 1.0/3.0, final_xpo=1.0, seed=None):

	print(emismap_name, errmap_name, temp1, temp2, xpo1, xpo2, zmax, vox_lct, hfac, sigma_fac, amat0, zmin, bin_fac, loop_weights, psf_size_px, asym, curvature, auxprof, temps, logt, tresp, map_in, obscenter, voxcenter, obswcs, voxwcs, temp_xpo, final_xpo)

	from scipy.sparse import diags, csr_matrix, csc_matrix

	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
	[voxmin,nvox,dvox] = [loopid_info['voxmin'], loopid_info['nvox'], loopid_info['dvox']]
	nloops = len(loopid_info['looplengths'])
	if(map_in is None): map_in = get_cropmap0(emismap_name,cropr)
	if(obscenter is None): obscenter=map_in.center
	if(voxcenter is None): voxcenter=voxmap.center
	if(obswcs is None): obswcs=map_in.wcs
	if(voxwcs is None): voxwcs=voxmap.wcs
	rsun_cm = voxmap.rsun_meters.to(u.cm).value
	asec_cm = obscenter.observer.radius.to(u.cm).value*np.pi/180/3600.0 # (rsun_cm/(map_in.rsun_obs.to(u.radian).value))*np.pi/180.0/3600.0
	pxsz = 0.5*asec_cm*(map_in.scale[0]+map_in.scale[1]).value

	loopflg = np.zeros(nloops+1,dtype='bool')+True
	loopflg[nloops]=False
	loopflg[np.where(loopid_info['looplengths'] < 3*np.max(dvox))]=False
	nl = loopid_info['lmax_scal']+1
	if(temps is None): temps = [temp1,temp2]
	if(emprofs is None):
		lbin_cents = (np.arange(nl)+0.5)/nl # np.linspace(0,1,nl)
		tprof = (((4*(lbin_cents)*(1-lbin_cents))+0.00)/1.00)**temp_xpo
		emprofs = np.zeros([2,nloops,nl])
		if(tresp is None or logt is None):
			[emprofs[0,:,:],emprofs[1,:,:]] = [tprof**(xpo1-2), tprof**(xpo2-2)]
		else:
			emprofs[0,:,:] = np.interp(np.log10(tprof*temp1),logt,tresp)/np.interp(np.log10(temp1),logt,tresp)
			emprofs[1,:,:] = np.interp(np.log10(tprof*temp1),logt,tresp)/np.interp(np.log10(temp1),logt,tresp)
		if(not(auxprof is None)): 
			for i in range(0,2): emprofs[i] *= auxprof[0]
		if(asym):
			emprofs[0,:,:] *= lbin_cents
			emprofs[1,:,:] *= (1.0-lbin_cents)
	nprofs = emprofs.shape[0]
	if(zmax is None): zmax = 6.0*hfac*np.max(temps)

	# --------------- Compute the detector matrix (PSF and pixelization):
	psfmat = get_psfmat(map_in.data.shape,bin_fac,psf_size_px,dvox,pxsz)
	outsize_hi = bin_fac*np.array(map_in.data.shape)
	nhi = outsize_hi[0]*outsize_hi[1]
	psfnorm = np.mean(np.sum(psfmat,axis=0).A1)
	psfnorm2 = np.mean(np.sum(psfmat,axis=1).A1)
			
	#---------------- Compute the Loop to pixel matrix (including PSF and pixelization):
	pxmin = vox2pix([0.0,0.0,0.0], voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_x = vox2pix(np.array([nvox[0]-1.0,0.0,0.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_y = vox2pix(np.array([0,nvox[1]-1.0,0.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_z = vox2pix(np.array([0,0,nvox[2]-1.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	xgrad = np.array([(pmax_x[0]-pxmin[0])/(nvox[0]-1.0),(pmax_y[0]-pxmin[0])/(nvox[1]-1.0),(pmax_z[0]-pxmin[0])/(nvox[2]-1.0)])
	ygrad = np.array([(pmax_x[1]-pxmin[1])/(nvox[0]-1.0),(pmax_y[1]-pxmin[1])/(nvox[1]-1.0),(pmax_z[1]-pxmin[1])/(nvox[2]-1.0)])

	[vox_ixa,vox_iya] = np.indices([nvox[0],nvox[1]],dtype='float64')
	vox_xy2 = ((vox_ixa-0.5*(nvox[0]-1))*dvox[0])**2 + ((vox_iya-0.5*(nvox[1]-1))*dvox[1])**2
	if(curvature):
		vox_iza = (rsun_cm*(1.0-np.sqrt(1.0-vox_xy2/rsun_cm**2))/dvox[2]).flatten()
	else:
		vox_iza = np.zeros(nvox[0:2],dtype='float64').flatten()
	[vox_ixa, vox_iya] = [vox_ixa.flatten(), vox_iya.flatten()]

	rgu = np.random.default_rng(seed)

	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	[imin,imax] = np.clip(np.floor(([zmin,zmax]-voxmin[2])/dvox[2]).astype(np.int32),0,nvox[2])
	t0 = time.time()
	amat0 = csc_matrix((2*nloops+1,nhi),dtype='float64')
	for i in range(imin,imax):
		loopid_slice = loopids[:,:,i].flatten()
		loopflg0 = loopflg[loopid_slice]
		segmentid_slice = voxlengths[:,:,i].flatten()
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		vox_ixa2 = vox_ixa + rgu.uniform(low=-0.5,high=0.5,size=(vox_ixa.shape))
		vox_iya2 = vox_iya + rgu.uniform(low=-0.5,high=0.5,size=(vox_iya.shape))
		vox_iza2 = vox_iza + rgu.uniform(low=-0.5,high=0.5,size=(vox_iza.shape))
		ix_out = np.rint(bin_fac*(pxmin[0]+xgrad[0]*vox_ixa2+xgrad[1]*vox_iya2+xgrad[2]*(i-vox_iza2))).astype('int32')
		iy_out = np.rint(bin_fac*(pxmin[1]+ygrad[0]*vox_ixa2+ygrad[1]*vox_iya2+ygrad[2]*(i-vox_iza2))).astype('int32')
		inbounds = (ix_out >= 0)*(iy_out >= 0)*(ix_out < outsize_hi[0])*(iy_out < outsize_hi[1])
		if(not(vox_mask is None)): inbounds *= vox_mask[:,:,i].flatten()
		loopid_out = nprofs*loopid_slice[inbounds].flatten()
		ixy = ix_out[inbounds]*outsize_hi[1]+iy_out[inbounds]
		if(curvature == False):
			height = (((i+0.5)*dvox[2]+rsun_cm+voxmin[2])**2+vox_xy2)**0.5 - rsun_cm
		else:
			height = np.zeros(nvox[0:2])+voxmin[2]+(i+0.5)*dvox[2]
		for k in range(0,nprofs):
			pscal = 1 # np.exp(-2.0*(i+0.5)*dvox[2]/(temps[k]*hfac))
			inten_slice = np.exp(-2.0*height/(temps[k]*hfac)).flatten() # np.zeros([nvox[0]*nvox[1]],dtype='float32')
			inten_slice[loopflg0] *= emprofs[k,loopid_flg,segmentid_flg]*pscal
			amat0 += csc_matrix((inten_slice[inbounds]**final_xpo,(loopid_out+k,ixy)),shape=(2*nloops+1,nhi)) # a
		amat0.sum_duplicates()            
		print('Slice ',i,' of ',imax,time.time()-t0, 'Using: %s kb' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	amat0 = (psfmat.transpose()*amat0.transpose()).transpose()
	loop_weights = np.clip(np.sum(amat0,axis=1),1,None).A1
	amat0 = diags(1.0/loop_weights)*amat0

	modelinputs = {'temp1':temp1, 'temp2':temp2, 'xpo1':xpo1, 'xpo2':xpo2, 'map':map_in, 'emprofs':emprofs, 'vox_xy2':vox_xy2,
					'loop_weights':loop_weights, 'hfac':hfac, 'psfnorm':psfnorm, 'psfnorm2':psfnorm2, 'curvature':curvature, 'rsun_cm':rsun_cm}	
	return amat0, modelinputs

def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)

# @processify
def get_3d_emission(solution_in, modelinputs, vox_grid, loopids, voxlengths, loopid_info, 
					dtype='float32', zmin=0, zmax=None, bin_facs=np.array([1,1,1])):
	# This code is not applying the pseudo-curvature, but perhaps that's as it should be.
	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	hfac = modelinputs.get('hfac',60.0e8/1.0e6)    
	solution = solution_in/modelinputs['loop_weights']
	nloops = len(loopid_info['looplengths'])
	loopflg = np.zeros(nloops+1,dtype='bool')+True
	loopflg[nloops]=False
	[voxmin,nvox,dvox] = [loopid_info['voxmin'], loopid_info['nvox'], loopid_info['dvox']]
	loopflg[np.where(loopid_info['looplengths'] < 3*np.max(dvox))]=False
	emxa = vox_grid.origin[0]+0.5+np.arange(vox_grid.dims[0])*vox_grid.fwd[0,0]
	emya = vox_grid.origin[1]+0.5+np.arange(vox_grid.dims[1])*vox_grid.fwd[1,1]
	emza = vox_grid.origin[2]+0.5+np.arange(vox_grid.dims[2])*vox_grid.fwd[2,2]
	emis3d = np.zeros(np.round(vox_grid.dims/bin_facs).astype(np.int32),dtype=dtype)
	temps = [modelinputs['temp1'],modelinputs['temp2']]
	nprofs = modelinputs['emprofs'].shape[0]
	if(zmax is None): zmax = (1+nvox[2])*dvox[2]
	[imin,imax] = np.clip(np.floor(([zmin,zmax]-voxmin[2])/dvox[2]).astype(np.int32),0,nvox[2])
	print(imin,imax,zmin,zmax)
	for i in range(imin,imax):
		loopid_slice = loopids[:,:,i]
		segmentid_slice = voxlengths[:,:,i]
		loopflg0 = loopflg[loopid_slice]
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		if(modelinputs['curvature'] == False):
			height = (((i+0.5)*dvox[2]+modelinputs['rsun_cm']+voxmin[2])**2+modelinputs['vox_xy2'])**0.5 - modelinputs['rsun_cm']
		else:
			height = np.zeros(nvox[0:2])+voxmin[2]+(i+0.5)*dvox[2]
		for k in range(0,nprofs):
			inten_slice = np.exp(-2.0*height/(temps[k]*hfac))
			inten_slice[loopflg0] *= solution[nprofs*loopid_flg+k]*modelinputs['emprofs'][k,loopid_flg,segmentid_flg]
			emis3d[:,:,np.floor(i/bin_facs[2]).astype(np.int32)] += bindown2(inten_slice,bin_facs[0:2])/dvox[2]
	return emis3d

def reconstruct(amat, modelinputs, solver=lgmres, steps=None, solver_tol=1.0e-4, reg_fac=0.01, regmat=None, dat_xpo=2, errs=None, data=None, sqrmap=False, map_reg=False):

	if(data is None): data = (modelinputs['map'].data.astype(np.float32))**dat_xpo
	if(errs is None):
		datasm = ndimage.filters.median_filter(np.abs(data),3)
		check1 = np.abs(data-datasm) > 0.75*datasm
		check2 = np.abs(data-datasm) > (np.median(datasm)+0.0*datasm)
		data[check1*check2==1]=datasm[check1*check2==1]
		datasm = ndimage.filters.median_filter(np.abs(data),3)
		errs = 1.5*ndimage.filters.gaussian_filter(np.abs(data-datasm),5)
	print(np.min(errs),np.max(errs),np.max(data))
	return sparse_nlmap_solver(np.clip(data,0.1*np.min(errs),None), errs, amat.T, solver=lgmres, steps=steps, solver_tol=solver_tol, reg_fac=reg_fac, regmat=regmat, sqrmap=sqrmap, map_reg=map_reg)

def imgfromcube(map_out, em_cube, voxmin, dvox, voxmap, zmax = None, zmin=0.0, bin_fac=3, psf_size_px = 0.7, curvature=True, obscenter=None, voxcenter=None, obswcs=None, voxwcs=None, seed=None):

	nvox = em_cube.shape
	if(obscenter is None): obscenter=map_out.center
	if(voxcenter is None): voxcenter=voxmap.center
	if(obswcs is None): obswcs=map_out.wcs
	if(voxwcs is None): voxwcs=voxmap.wcs
	rsun_cm = voxmap.rsun_meters.to(u.cm).value
	asec_cm = obscenter.observer.radius.to(u.cm).value*np.pi/180/3600.0 # 
	pxsz = 0.5*asec_cm*(map_out.scale[0]+map_out.scale[1]).value
	if(zmax is None): zmax = voxmin[2]+dvox[2]*nvox[2]

	# --------------- Compute the detector matrix (PSF and pixelization):
	psfmat = get_psfmat(map_out.data.shape,bin_fac,psf_size_px,dvox,pxsz)
	outsize_hi = bin_fac*np.array(map_out.data.shape)
	nhi = outsize_hi[0]*outsize_hi[1]
	psfnorm = np.mean(np.sum(psfmat,axis=0).A1)
	print(psfnorm)
			
	#---------------- Compute the Loop to pixel matrix (including PSF and pixelization):
	pxmin = vox2pix([0.0,0.0,0.0], voxmin, dvox, voxmap, map_out, obswcs=obswcs, voxcenter=voxcenter)
	pmax_x = vox2pix(np.array([nvox[0]-1.0,0.0,0.0]), voxmin, dvox, voxmap, map_out, obswcs=obswcs, voxcenter=voxcenter)
	pmax_y = vox2pix(np.array([0,nvox[1]-1.0,0.0]), voxmin, dvox, voxmap, map_out, obswcs=obswcs, voxcenter=voxcenter)
	pmax_z = vox2pix(np.array([0,0,nvox[2]-1.0]), voxmin, dvox, voxmap, map_out, obswcs=obswcs, voxcenter=voxcenter)
	xgrad = np.array([(pmax_x[0]-pxmin[0])/(nvox[0]-1.0),(pmax_y[0]-pxmin[0])/(nvox[1]-1.0),(pmax_z[0]-pxmin[0])/(nvox[2]-1.0)])
	ygrad = np.array([(pmax_x[1]-pxmin[1])/(nvox[0]-1.0),(pmax_y[1]-pxmin[1])/(nvox[1]-1.0),(pmax_z[1]-pxmin[1])/(nvox[2]-1.0)])

	[vox_ixa,vox_iya] = np.indices([nvox[0],nvox[1]],dtype='float32')
	vox_xy2 = ((vox_ixa-0.5*(nvox[0]-1))*dvox[0])**2 + ((vox_iya-0.5*(nvox[1]-1))*dvox[1])**2
	if(curvature):
		vox_iza = (rsun_cm*(1.0-np.sqrt(1.0-vox_xy2/rsun_cm**2))/dvox[2]).flatten()
	else:
		vox_iza = np.zeros(nvox[0:2],dtype='float32').flatten()
	[vox_ixa, vox_iya] = [vox_ixa.flatten(), vox_iya.flatten()]

	rgu = np.random.default_rng(seed)

	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	[imin,imax] = np.clip(np.floor(([zmin,zmax]-voxmin[2])/dvox[2]).astype(np.int32),0,nvox[2])
	t0 = time.time()
	img0 = np.zeros(outsize_hi)
	for i in range(imin,imax):
		vox_ixa2 = vox_ixa + rgu.uniform(low=-0.5,high=0.5,size=(vox_ixa.shape))
		vox_iya2 = vox_iya + rgu.uniform(low=-0.5,high=0.5,size=(vox_ixa.shape))
		vox_iza2 = vox_iza + rgu.uniform(low=-0.5,high=0.5,size=(vox_ixa.shape))
		ix_out = np.rint(bin_fac*(pxmin[0]+xgrad[0]*vox_ixa2+xgrad[1]*vox_iya2+xgrad[2]*(i-vox_iza2))).astype('int32')
		iy_out = np.rint(bin_fac*(pxmin[1]+ygrad[0]*vox_ixa2+ygrad[1]*vox_iya2+ygrad[2]*(i-vox_iza2))).astype('int32')
		inbounds = (ix_out >= 0)*(iy_out >= 0)*(ix_out < outsize_hi[0])*(iy_out < outsize_hi[1])
		#img0[(ix_out[inbounds],iy_out[inbounds])] += em_cube[:,:,i].flatten()[inbounds]
		np.add.at(img0, (ix_out[inbounds],iy_out[inbounds]), em_cube[:,:,i].flatten()[inbounds])
		print('Slice ',i,' of ',imax,time.time()-t0, 'Using: %s kb' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	imgout = (psfmat.T*img0.flatten()).reshape(map_out.data.shape)
	return imgout
