import numpy as np

def bindown(d,f,mask=None):
	if(mask is None):
		return bindown2(d,f)
	else:
		mask_bindown = bindown(mask,f)
		mask_bindown[mask_bindown==0]=1
		return np.prod(f)*bindown2(mask*d,f)/mask_bindown

def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)

# These are placeholder methods pending temperature response functions
# being added to AIApy. error estimates are included for the initial 
# version as well, although I believe they're already in the initial version
def estimate_aia_error(map_in,channel=None):
	if(channel is None): channel = map_in.meta['detector']+map_in.meta['wave_str']
	refchannels=np.array(['AIA94_THIN', 'AIA131_THIN', 'AIA171_THIN', 'AIA193_THIN', 'AIA211_THIN', 'AIA304_THIN', 'AIA335_THIN'])
	refg = np.array([2.128,1.523,1.168,1.024,0.946,0.658,0.596])
	refn = np.array([1.14,1.18,1.15,1.2,1.2,1.14,1.18])
	sigmas = np.zeros(map_in.data.shape)
	dnpp = refg[np.where(refchannels == channel)]
	rdn = refn[np.where(refchannels == channel)]
	return np.sqrt(np.clip(map_in.data*dnpp,0.0,None) + rdn**2)

def search_fido_response(r,s):
    hits = []
    for i in range(0,len(r)): 
        for j in range(0,r[i].file_num): 
            if(str(r[i,j]).find(s) >= 0): hits.append([i,j])
    return hits

def get_limb(vox_grid,rsun_cm):
    coorda = np.zeros([vox_grid.dims[0],vox_grid.dims[1],1,3],dtype=np.float32)
    for i in range(0,vox_grid.dims[0]): coorda[i,:,:,0] = i
    for i in range(0,vox_grid.dims[1]): coorda[:,i,:,1] = i

    limb = np.zeros([vox_grid.dims[0],vox_grid.dims[1],vox_grid.dims[2]],dtype=np.bool)
    for i in range(0,vox_grid.dims[2]):
        coorda[:,:,0,2] = i
        coords = vox_grid.coords(coorda)
        rads = np.sqrt(np.sum(coords[:,:,:,0:2]**2,axis=3)+(coords[:,:,:,2]+rsun_cm)**2)
        limb[:,:,i] = (rads > rsun_cm)[:,:,0]
    return limb

def get_limb_occlusion(vox_lct, obsmap, loopid_info):
    [voxmin,nvox,dvox] = [loopid_info['voxmin'], loopid_info['nvox'], loopid_info['dvox']]
    refvox = np.array([0.0,0.0,0.0])
    refvox_x = np.array([nvox[0]-1.0,0.0,0.0])
    refvox_y = np.array([0.0,nvox[1]-1.0,0.0])
    refvox_z = np.array([0.0,0.0,nvox[2]-1.0])
    refcoord = vox_lct.coord((refvox*dvox+voxmin)*u.cm).transform_to(obsmap.coordinate_frame)
    refcoord_x = vox_lct.coord((refvox_x*dvox+voxmin)*u.cm).transform_to(obsmap.coordinate_frame)
    refcoord_y = vox_lct.coord((refvox_y*dvox+voxmin)*u.cm).transform_to(obsmap.coordinate_frame)
    refcoord_z = vox_lct.coord((refvox_z*dvox+voxmin)*u.cm).transform_to(obsmap.coordinate_frame)
    [tx0, ty0, tz0] = [refcoord.Tx, refcoord.Ty, refcoord.distance]
    [tx_x, ty_x, tz_x] = [refcoord_x.Tx, refcoord_x.Ty, refcoord_x.distance]
    [tx_y, ty_y, tz_y] = [refcoord_y.Tx, refcoord_y.Ty, refcoord_y.distance]
    [tx_z, ty_z, tz_z] = [refcoord_z.Tx, refcoord_z.Ty, refcoord_z.distance]
    dtx_dx, dtx_dy, dtx_dz = (tx_x-tx0)/(nvox[0]-1), (tx_y-tx0)/(nvox[1]-1), (tx_z-tx0)/(nvox[2]-1)
    dty_dx, dty_dy, dty_dz = (ty_x-ty0)/(nvox[0]-1), (ty_y-ty0)/(nvox[1]-1), (ty_z-ty0)/(nvox[2]-1)
    dtz_dx, dtz_dy, dtz_dz = (tz_x-tz0)/(nvox[0]-1), (tz_y-tz0)/(nvox[1]-1), (tz_z-tz0)/(nvox[2]-1)
    limb_occlusion = np.ones(nvox,dtype=bool)
    [xa,ya] = np.indices(nvox[0:2])
    
    dsun = obsmap.dsun.to(u.cm)
    
    for i in range(0,nvox[2]):
        tx = tx0 + xa*dtx_dx + ya*dtx_dy + i*dtx_dz
        ty = ty0 + xa*dty_dx + ya*dty_dy + i*dty_dz
        tz = tz0 + xa*dtz_dx + ya*dtz_dy + i*dtz_dz
        limb_occlusion[:,:,i] = (np.sqrt(tx.value**2+ty.value**2) > obsmap.rsun_obs.to(u.arcsec).value) | (tz.value < dsun.value)

    return limb_occlusion
