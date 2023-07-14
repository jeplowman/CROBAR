import numpy as np
from numpy.random import Generator, PCG64

def volume_seed_points(number_fieldlines, bounds, rg=Generator(PCG64())):
	return bounds[0]+(bounds[1]-bounds[0])*rg.uniform(size=[number_fieldlines,len(bounds[0])])

def fluxweighted_seed_points(number_fieldlines, mag, mag_coords, bounds, rg=Generator(PCG64()), xpo = 1.0, z0 = None):
	if(z0 is None):
		# This amounts to half the mean x and y spacing between the grid points of the magnetogram.
		z0 = 0.5*(np.mean(mag_coords[0,1:,:,:]-mag_coords[0,0:-1,:,:])+np.mean(mag_coords[1,:,1:,:]-mag_coords[1,:,0:-1,:]))
	mag_flat = mag.flatten()
	mag_coords_flat = np.reshape(mag_coords,[3,mag_flat.size]).T
	domain_check = np.all((mag_coords_flat[:,0:2] >= bounds[0,0:2])*(mag_coords_flat[:,0:2] <= bounds[1,0:2]),axis=1)
	mag_flat = mag_flat[domain_check]
	mag_coords_flat = mag_coords_flat[domain_check,:]
	mag_coords_flat[:,2] += z0 # *(bounds[1,2]-bounds[0,2])
	indices_sort = np.argsort(mag_flat)
	coords_sort = list(mag_coords_flat[indices_sort,:])
	mag_sort = list(mag_flat[indices_sort])
	coords_out = []
	for i in range(0,number_fieldlines):
		weights = (np.abs(mag_sort)**xpo).cumsum()
		rand_index = np.floor(np.interp(rg.uniform(0,1),weights/np.max(weights),np.arange(0,len(weights)))).astype(np.int64)
		coords_out.append(coords_sort.pop(rand_index))
		mag_sort.pop(rand_index)
	return np.array(coords_out)
	
def fluxweighted_seed_points2(number_fieldlines, mag, mag_coords, bounds, rg=Generator(PCG64()), xpo=1.0, z0=None, clip=0.95, nplateau=0,plateau_min=100,plateau_max=200):
	if(z0 is None):
		# This amounts to half the mean x and y spacing between the grid points of the magnetogram.
		z0 = 0.5*(np.mean(mag_coords[0,1:,:,:]-mag_coords[0,0:-1,:,:])+np.mean(mag_coords[1,:,1:,:]-mag_coords[1,:,0:-1,:]))
	mag_flat = mag.flatten()
	mag_coords_flat = np.reshape(mag_coords,[3,mag_flat.size]).T
	domain_check = np.all((mag_coords_flat[:,0:2] >= bounds[0,0:2])*(mag_coords_flat[:,0:2] <= bounds[1,0:2]),axis=1)
	mag_flat = np.abs(mag_flat[domain_check])**xpo
	mag_coords_flat = mag_coords_flat[domain_check,:]
	mag_coords_flat[:,2] += z0 # *(bounds[1,2]-bounds[0,2])

	indices_sort = np.argsort(mag_flat)
	mag_sort = mag_flat[indices_sort]
	mag_sort = np.clip(mag_sort,0,mag_sort[np.round(clip*len(mag_sort)).astype(np.int64)])
	coords_sort = mag_coords_flat[indices_sort,:]
	weights = np.hstack([0,np.cumsum(mag_sort)])
	semirand_nums = (np.arange(0,number_fieldlines,dtype=np.float32)+rg.uniform(0,1,number_fieldlines))/number_fieldlines
	rand_indices = np.clip(np.floor(np.interp(semirand_nums,weights/np.max(weights),np.arange(0,len(weights)))),0,len(mag_flat)-1).astype(np.int64)
	coords_out = coords_sort[rand_indices,:]

	if(nplateau > 0):	
		mag_flat[rand_indices] = 0
		if(plateau_min != None): mag_flat[np.abs(mag_flat) < plateau_min**xpo] = 0
		if(plateau_max != None):  mag_flat[np.abs(mag_flat) > plateau_max**xpo] = plateau_max**xpo
		coords_sort = mag_coords_flat[mag_flat > 0,:]
		mag_flat = mag_flat[mag_flat > 0]
		print(mag_flat.shape)		
		indices_sort = np.argsort(mag_flat)
		mag_sort = mag_flat[indices_sort]
		#mag_sort = np.clip(mag_sort,0,mag_sort[np.round(clip*len(mag_sort)).astype(np.int64)])
		coords_sort = coords_sort[indices_sort,:]
		weights = np.hstack([0,np.cumsum(mag_sort)])
		semirand_nums = (np.arange(0,nplateau,dtype=np.float32)+rg.uniform(0,1,nplateau))/nplateau
		rand_indices = np.clip(np.floor(np.interp(semirand_nums,weights/np.max(weights),np.arange(0,len(weights)))),0,len(mag_flat)-1).astype(np.int64)
		print(coords_out.shape,coords_sort.shape,rand_indices.shape)
		coords_out = np.vstack([coords_out, coords_sort[rand_indices,:]])
	
	return np.array(coords_out)

def region_seed_points(number_fieldlines, mag, mag_coords, ivp_z0, min_flux=75, min_size=4, ivp_xpo=1):
	from skimage.measure import label
	from numpy.random import Generator, PCG64

	z0=ivp_z0
	region2label = (mag[:,:,0] >= min_flux).astype(np.float32) - (mag[:,:,0] <= -min_flux) 
	labels = label(region2label)
	rg=Generator(PCG64())

	nregion = np.max(labels)
	nper_region = np.zeros(nregion,dtype=np.int32)
	region_nums = np.arange(nregion,dtype=np.int32)+1
	for i in range(0,nregion):
		nper_region[i] = np.clip(np.round(np.sum(labels==i+1)**0.5).astype(np.int32),1,None)
	region_nums = region_nums[nper_region >= min_size]
	nper_region = nper_region[nper_region >= min_size]
	nper_region = np.round(nper_region*number_fieldlines/np.sum(nper_region)).astype(np.int32)
	ntot = np.sum(nper_region)
	nregion = len(nper_region)
	region_coords = np.zeros([ntot,3])
	mag_flat = np.abs(mag[:,:,0].flatten())**ivp_xpo
	lab_flat = labels.flatten()
	mag_coords_flat = np.reshape(mag_coords[:,:,:,0],[3,mag_flat.size]).T
	mag_coords_flat[:,2] += z0 # *(bounds[1,2]-bounds[0,2])
	count=0
	for i in range(0,nregion):
		n_current = nper_region[i]
		mags_region = mag_flat[lab_flat==region_nums[i]]
		coords_region = mag_coords_flat[lab_flat==region_nums[i],:]
		indices_sort = np.argsort(mags_region)
		mag_sort = mags_region[indices_sort]
		coords_sort = coords_region[indices_sort]
		weights = np.hstack([0,np.cumsum(mag_sort)])
		semirand_nums = (np.arange(0,n_current,dtype=np.float32)+rg.uniform(0,1,n_current))/n_current
		rand_indices = np.clip(np.floor(np.interp(semirand_nums,weights/np.max(weights),np.arange(0,len(weights)))),0,len(mag_sort)-1).astype(np.int64)
		region_coords[count:(count+nper_region[i]),:] = coords_sort[rand_indices,:]
		count += nper_region[i]
		
	return region_coords
	
	
# Code for 2D Hilbert curves, shamelessly stolen from wikipedia and translated to Python:
# convert (x,y) to d
def xy2d (n, x, y):
    d=0
    for i in range(0,np.round(np.log(n)/np.log(2)).astype(np.int32)+1): #(s=n/2; s>0; s/=2)
        s = int(n/2**(i+1))
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(n, x, y, rx, ry)
    return d

# convert d to (x,y)
def d2xy(n):
    x = np.zeros(n*n,dtype=np.int32)
    y = np.zeros(n*n,dtype=np.int32)
    t = np.arange(n*n,dtype=np.int32)
    o = np.ones(n*n,dtype=np.int32)
    for i in range(0,np.round(np.log(n)/np.log(2)).astype(np.int32)):
        s=int(2**i)
        rx = o & np.floor(t/2).astype(np.int32)
        ry = o & (t ^ rx)
        rot(s, x, y, rx, ry)
        x += s*rx
        y += s*ry
        t[::] = t[::]/4
        
    return x,y

# rotate/flip a quadrant appropriately
def rot(n, x, y, rx, ry):
    xycheck = np.where((ry==0)*(rx==1))
    x[xycheck] = n-1 - x[xycheck]
    y[xycheck] = n-1 - y[xycheck]
    ycheck = np.where(ry==0)
    t = x[ycheck]
    x[ycheck] = y[ycheck]
    y[ycheck] = t
    return x,y

# Draw random pixels from an image, weighted by its intensity, according to a Hilbert curve:
def hilbert_points(img,npts=1000):
    n = np.round(2**np.ceil(np.log(np.max(img.shape))/np.log(2))).astype(np.int32)

    xa,ya = d2xy(n)
    inb = np.where((xa < img.shape[0])*(ya < img.shape[1]))
    xa = xa[inb]
    ya = ya[inb]

    thold = np.sum(np.abs(img))/npts
    accum, count = 0, 0
    xout = np.zeros(npts,dtype=np.int32)
    yout = np.zeros(npts,dtype=np.int32)
    for i in range(0,len(xa)):
        accum += np.abs(img[xa[i],ya[i]])
        if(accum > thold):
            xout[count] = xa[i]
            yout[count] = ya[i]
            count=count+1
            accum -= thold
    return xout, yout

def hilbert_seed_points(number_fieldlines, mag, mag_coords, z0, rsun=None):
	[xout,yout] = hilbert_points(mag[:,:,0], npts=number_fieldlines)
	hilbert_coords = mag_coords[:,xout,yout,0]
	cell_sizes = np.zeros(mag_coords.shape)

	heights = (hilbert_coords[0]**2 + hilbert_coords[1]**2 + (hilbert_coords[2]+rsun)**2)**0.5 + z0 - rsun

	cell_sizes[0,1:-1,:,:] = 0.5*(mag_coords[0][2:,:,:]-mag_coords[0][0:-2,:,:])
	##cell_sizes[0,0,:,:] = 0.5*(mag_coords[0][1,:,:]-mag_coords[0][0,:,:])
	##cell_sizes[0,-1,:,:] = 0.5*(mag_coords[0][-1,:,:]-mag_coords[0][-2,:,:])

	cell_sizes[1,:,1:-1,:] = 0.5*(mag_coords[0][:,2:,:]-mag_coords[0][:,0:-2,:])
	##cell_sizes[1,:,0,:] = 0.5*(mag_coords[0][:,1,:]-mag_coords[0][:,0,:])
	##cell_sizes[1,:,-1,:] = 0.5*(mag_coords[0][:,-1,:]-mag_coords[0][:,-2,:])
	hilbert_coords[0] += cell_sizes[0,xout,yout,0]*(np.random.uniform(size=hilbert_coords[0].shape)-0.5)
	hilbert_coords[1] += cell_sizes[1,xout,yout,0]*(np.random.uniform(size=hilbert_coords[1].shape)-0.5)
	hilbert_coords[2] = ((heights-z0+rsun)**2 - hilbert_coords[0]**2 - hilbert_coords[1]**2)**0.5 - rsun + z0
	return hilbert_coords.T