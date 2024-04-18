import copy, numpy as np
import astropy.units as u
import sunpy.coordinates
from astropy.coordinates import SkyCoord

class transform(object):
	def __init__(self,center):
		self.cen = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
		self.obs = sunpy.coordinates.frames.HeliographicStonyhurst(self.cen.lon, self.cen.lat, self.cen.observer.radius, obstime=self.cen.obstime)
		self.radius = self.cen.radius
		self.frame = sunpy.coordinates.frames.Heliocentric(0.0*self.cen.radius, 0.0*self.cen.radius, self.cen.radius, observer=self.obs)
	def xyz(self, coord):
		xyz = coord.transform_to(self.frame).cartesian.xyz
		xyz[2] -= self.radius
		return xyz
	def coord(self, xyz):
		return SkyCoord(x=xyz[0],y=xyz[1],z=xyz[2]+self.radius,frame=self.frame,representation_type='cartesian')

def map_ijxy_conversion(map, wcs=None):
	if(wcs is None): wcs=map.wcs
	return list(wcs.world_to_array_index(wcs.pixel_to_world(*(np.arange(wcs.naxis)))))

def map_ind_derivs(map_in, ref_map=None, center=None, wcs=None):
	if(ref_map is None): ref_map=map_in
	if(center is None): center = ref_map.center
	if(wcs is None): wcs = map_in.wcs
	lct = transform(center)
	
	[ind1_cen, ind2_cen] = wcs.world_to_array_index(center)
	ind0_derivs = (lct.xyz(wcs.array_index_to_world(ind1_cen+0.5,ind2_cen))-lct.xyz(wcs.array_index_to_world(ind1_cen-0.5,ind2_cen))).to(u.cm)
	ind1_derivs = (lct.xyz(wcs.array_index_to_world(ind1_cen,ind2_cen+0.5))-lct.xyz(wcs.array_index_to_world(ind1_cen,ind2_cen-0.5))).to(u.cm)
	return ind0_derivs, ind1_derivs

		
def planar_map_coords(map_in, ref_map=None, center=None, wcs=None):
	ind0_derivs, ind1_derivs = map_ind_derivs(map_in, ref_map=ref_map, center=center, wcs=wcs)
	mag = copy.deepcopy(map_in.data)	
	indarr = np.indices(mag.shape)
	ndim = len(ind0_derivs)
	mag_coords = np.zeros(np.hstack([ndim,mag.shape]))*u.cm
	mag_coords[0] = ind0_derivs[0]*(indarr[0]-0.5*(mag.shape[0]-1))+ind1_derivs[0]*(indarr[1]-0.5*(mag.shape[1]-1))
	mag_coords[1] = ind0_derivs[1]*(indarr[0]-0.5*(mag.shape[0]-1))+ind1_derivs[1]*(indarr[1]-0.5*(mag.shape[1]-1))
	return np.expand_dims(mag,-1), np.expand_dims(mag_coords,-1)


class transform_curved(object):
	def __init__(self,center,earthecliptic=True):
		self.cen = center
		self.earthecliptic=earthecliptic
		if(earthecliptic): 
			self.frame = self.cen.heliocentricearthecliptic.frame
			r0 = center.heliocentricearthecliptic.cartesian
		else: 
			self.frame = self.cen.heliocentric.frame
			r0 = center.heliocentric.cartesian
		[self.r0x, self.r0y, self.r0z] = [r0.x,r0.y,r0.z]
		magr0 = np.sqrt(r0.x**2+r0.y**2+r0.z**2)
		self.radius = magr0
		magxy0 = np.sqrt(r0.x**2+r0.y**2)
		[self.rh0x,self.rh0y,self.rh0z] = [r0.x/magr0,r0.y/magr0,r0.z/magr0]
		[self.th0x,self.th0y,self.th0z] = [r0.z*r0.x/magr0/magxy0,r0.z*r0.y/magr0/magxy0,-magxy0/magr0]
		[self.ph0x,self.ph0y,self.ph0z] = [-r0.y/magxy0,r0.x/magxy0,0.0]
		
	def xyz(self, coords_in):
		coords = coords_in.transform_to(self.frame).cartesian
		xa = coords.x
		ya = coords.y
		za = coords.z
	
		xa_local = (xa-self.r0x)*self.ph0x + (ya-self.r0y)*self.ph0y + (za-self.r0z)*self.ph0z
		ya_local = (xa-self.r0x)*self.th0x + (ya-self.r0y)*self.th0y + (za-self.r0z)*self.th0z
		za_local = (xa-self.r0x)*self.rh0x + (ya-self.r0y)*self.rh0y + (za-self.r0z)*self.rh0z

		return np.stack([xa_local, -ya_local, za_local])
	
	def coord(self, xyz):
	
		xa = xyz[0]*self.ph0x - xyz[1]*self.th0x + (xyz[2])*self.rh0x + self.r0x
		ya = xyz[0]*self.ph0y - xyz[1]*self.th0y + (xyz[2])*self.rh0y + self.r0y
		za = xyz[0]*self.ph0z - xyz[1]*self.th0z + (xyz[2])*self.rh0z + self.r0z
	
		return SkyCoord(x=xa,y=ya,z=za, frame=self.frame, representation_type='cartesian')
	
	def xyz_test(self, coords_in):	
		[xa,ya,za] = coords_in
		xa_local = (xa-self.r0x)*self.ph0x + (ya-self.r0y)*self.ph0y + (za-self.r0z)*self.ph0z
		ya_local = (xa-self.r0x)*self.th0x + (ya-self.r0y)*self.th0y + (za-self.r0z)*self.th0z
		za_local = (xa-self.r0x)*self.rh0x + (ya-self.r0y)*self.rh0y + (za-self.r0z)*self.rh0z
		return xa_local, -ya_local, za_local
		
	def coord_test(self, xyz):
		xa = xyz[0]*self.ph0x - xyz[1]*self.th0x + (xyz[2])*self.rh0x + self.r0x
		ya = xyz[0]*self.ph0y - xyz[1]*self.th0y + (xyz[2])*self.rh0y + self.r0y
		za = xyz[0]*self.ph0z - xyz[1]*self.th0z + (xyz[2])*self.rh0z + self.r0z

		return xa, ya, za

	def xyz_test2(self, coords_in):
		coords = coords_in.transform_to(self.frame).cartesian
		xa = coords.x
		ya = coords.y
		za = coords.z
		return xa,ya,za	
	
def local_map_coords_curved(map_in, center=None, earthecliptic=True, lct=None):
	if(center is None): center=map_in.center
	if(lct is None): lct = transform_curved(center)
	[ia,ja] = np.indices(map_in.data.shape)
	coords = map_in.wcs.array_index_to_world(ia,ja)
	return lct.xyz(coords)

def curved_map_coords_test(map_in, center=None, earthecliptic=True):
	if(center is None): center=map_in.center
	[ia,ja] = np.indices(map_in.data.shape)
	coords = map_in.wcs.array_index_to_world(ia,ja)
	return coords

def los_field_correction(map_in):
	dsun = map_in.center.observer.radius
	rsun = map_in.center.rsun
	rsun_asec = (rsun/dsun)/(np.pi/180/3600)*u.arcsec
	[ia,ja] = np.indices(map_in.data.shape)
	rvecs = map_in.wcs.array_index_to_world(ia,ja)
	im_asecs = np.sqrt((rvecs.Tx)**2+(rvecs.Ty)**2)
	#return 1.0/np.sqrt(1.0-(im_asecs/rsun_asec)**2)
	return 1.0/(1.0-(im_asecs/rsun_asec)**2)

def curved_map_coords(map_in, ref_map=None, center=None, wcs=None, length_unit=u.cm, lct=None, thold=4):
	curved_coords = local_map_coords_curved(map_in, center=center, lct=lct)
	field_correction = los_field_correction(map_in)
	mag = copy.deepcopy(map_in.data)
	#print(curved_coords.unit)
	good_coords = np.isfinite(np.sum(curved_coords,axis=0))
	good_fields = (field_correction < 4)*(np.isfinite(field_correction))
	goods = good_coords*good_fields
	[goodi,goodj] = np.where(goods)
	curved_coords[0][goods==0]=curved_coords[0][goodi[0],goodj[0]]
	curved_coords[1][goods==0]=curved_coords[1][goodi[0],goodj[0]]
	curved_coords[2][goods==0]=curved_coords[2][goodi[0],goodj[0]]
	field_correction[goods==0]=0
	return np.expand_dims(mag*field_correction,-1), np.expand_dims(curved_coords,-1), goods

# Convert from voxel to pixel (i.e., array) indices
def vox2pix(vox, voxmin, dvox, voxel_map, pixel_map, unit=u.cm, voxcenter=None, obswcs=None, vox_lct=None):
	if(voxcenter is None): voxcenter = voxel_map.center
	if(vox_lct is None): vox_lct = transform(voxcenter)     
	vox_coord = vox_lct.coord((vox*dvox+voxmin)*u.cm)
	
	if(obswcs is None): obswcs = pixel_map.wcs
	pixel_xy = np.array(obswcs.world_to_pixel(vox_coord))
	ijxy_conversion = obswcs.world_to_array_index(obswcs.pixel_to_world(*(np.arange(obswcs.naxis))))
	return pixel_xy[list(ijxy_conversion)]

def wrld2pix(cord, voxel_map, pixel_map, voxcenter=None, obswcs=None, vox_lct=None):
	if(voxcenter is None): voxcenter = voxel_map.center
	if(vox_lct is None): vox_lct = transform(voxcenter)     
	#vox_coord = vox_lct.coord((vox*dvox+voxmin)*u.cm)
	if(obswcs is None): obswcs = pixel_map.wcs
	pixel_xy = np.array(obswcs.world_to_pixel(cord))
	ijxy_conversion = obswcs.world_to_array_index(obswcs.pixel_to_world(*(np.arange(obswcs.naxis))))
	return pixel_xy[list(ijxy_conversion)]

