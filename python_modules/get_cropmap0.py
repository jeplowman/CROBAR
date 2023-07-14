import copy

# Sunpy/astropy libraries:
import sunpy.coordinates
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.net import Fido,attrs as a, vso
from sunpy.time import TimeRange
from astropy.coordinates import SkyCoord
from processify import processify

from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.wcs import WCS
from reproject import reproject_interp

#@processify
def get_cropmap0(path,cropr,**kwargs):
	cropmap = sunpy.map.Map(path)#.rotate(order=3)
	blc=SkyCoord(cropr[0],cropr[2],frame=kwargs.get('frame',cropmap.coordinate_frame))
	trc=SkyCoord(cropr[1],cropr[3],frame=kwargs.get('frame',cropmap.coordinate_frame))
	cropmap = cropmap.submap(blc,top_right=trc)
	
	return cropmap
