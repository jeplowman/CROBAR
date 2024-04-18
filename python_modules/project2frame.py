import sunpy.map, astropy.units as u, numpy as np
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from util import get_limb_occlusion
from reconstruct_2comp_3Dloops import imgfromcube
from processify import processify
from local_cartesian_transform import transform_curved


def project(cube, loopid_info, initialpoint_dict, frame_map=None,
			wcs=None, lat=None, lon=None, rad=None, tim=None, 
			nx=None, ny=None, dx=None, dy=None, rota=None, vox_map=None,
			xrefpx=None, yrefpx=None, xrefcrd=None, yrefcrd=None,
			zmax=None, zmin=None, bin_fac=1, psf_size_px=0.5, psfmat=None,
			curvature=False, seed=None, voxcenter=None, voxwcs=None):
	
	if('data' not in dir(frame_map) or 'fits_header' not in dir(frame_map)):
		if(wcs is not None): print('frame from WCS not yet implemented, using lat/long/rad/time')
		if('unit' not in dir(lat)): lat=lat*u.deg
		if('unit' not in dir(lon)): lon=lon*u.deg
		if('unit' not in dir(rad)): rad=rad*u.m
		if('unit' in dir(dx)):
			scale = np.array([dx.value,dy.value])*dx.unit
		else:
			scale = np.array([dx,dy])*u.arcsec/u.pix
		if('unit' in dir(xrefpx)):
			refpx = np.array([xrefpx.value, yrefpx.value])*xrefpx.unit
		else:
			refpx = np.array([xrefpx,yrefpx])*u.pix
		if('unit' not in dir(xrefcrd)): xrefcrd = xrefcrd*u.arcsec
		if('unit' not in dir(yrefcrd)): yrefcrd = yrefcrd*u.arcsec
		if('unit' not in dir(rota)): rota = rota*u.deg
		
		obscoord = SkyCoord(frame=frames.HeliographicStonyhurst, lon=lon, lat=lat, 
							radius=rad, obstime=tim)
		srccoord = SkyCoord(xrefcrd,yrefcrd, obstime=tim, observer=obscoord,
							frame=frames.Helioprojective)
		dat = np.zeros([nx,ny])
		header = sunpy.map.make_fitswcs_header(dat, srccoord, scale=scale,
												reference_pixel=refpx, rotation_angle=rota)
		frame_map = sunpy.map.Map(dat, header)
	else:
		header = frame_map.fits_header

	voxmin = initialpoint_dict['vox_grid'].origin
	dvox = np.diag(initialpoint_dict['vox_grid'].fwd)
	if(vox_map is None): vox_map = initialpoint_dict['magnetogram']
	lct = transform_curved(vox_map.center)

	occlusion_mask = get_limb_occlusion(lct, frame_map, loopid_info)

	if(zmin is None): zmin = voxmin[0]
	output_img, psfmat = imgfromcube(frame_map, occlusion_mask*cube, voxmin, dvox, vox_map, 
							zmax=zmax, zmin=zmin, bin_fac=bin_fac, psf_size_px=psf_size_px,
							curvature=curvature, obscenter=None, voxcenter=voxcenter, silent=True,
							voxwcs=voxwcs, obswcs=None, seed=seed, psfmat=psfmat, return_psf=True)
							
	return sunpy.map.Map(output_img, header), psfmat
