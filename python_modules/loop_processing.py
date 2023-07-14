import copy, numpy as np
from scipy.interpolate import interp1d

def seglint(x, y, x2):
	return y[0]+(x2-x[0])*(y[1]-y[0])/(x[1]-x[0])

def resample_loops(fieldlines, bounds, spacing0, lpad=1.2, npt_min=7, lfrac=0.1, zfrac=0.075, newspacing=False, z0=None, rcurv=None, h0=None):
    
	if(not(z0 is None) and rcurv is None): h0=z0
	if(h0 is None): h0 = bounds[0,2] # +0.25*spacing0

	[fieldlines_out, lengths_out] = [[],[]]
	for i in range(0,len(fieldlines)):
		fieldline = fieldlines[i]
		if(rcurv is None): 
			heights = fieldline[:,2]
			minheight = bounds[0,2]
		else:
			heights = np.sqrt(np.sum(fieldline[:,0:2]**2,axis=1)+(rcurv+fieldline[:,2])**2)-rcurv
			minheight = bounds[0,2]
			#minheight = (np.sqrt(np.sum(bounds[0,0:2]**2)+(rcurv+bounds[0,2])**2)-rcurv +
			#			 np.sqrt(np.sum(bounds[1,0:2]**2)+(rcurv+bounds[0,2])**2)-rcurv)*0.5
		inb = np.where((heights >= minheight)*(heights <= bounds[1,2]))
		print(minheight, np.min(heights), np.sum((heights >= minheight)), np.sum(heights <= bounds[1,2]), len(inb), len(heights))
		if(len(inb[0]) >= npt_min): 
			fieldline = fieldline[np.min(inb):np.max(inb),:]
			npt = len(fieldline)
			if(heights[npt-1] < heights[0]): fieldline = np.flip(fieldline,0)
			if(heights[npt-1] == heights[npt-2]): fieldline = fieldline[0:npt-1,:]
			if(heights[0] == heights[1]): fieldline = fieldline[1:len(fieldline)]
			npt = len(fieldline)
			if(rcurv is None):
				h_in = fieldline[:,2]
			else:
				h_in = np.sqrt(np.sum(fieldline[:,0:2]**2,axis=1)+(rcurv+fieldline[:,2])**2)-rcurv
			h = copy.deepcopy(h_in)
			[x,y,z] = [fieldline[:,0], fieldline[:,1], fieldline[:,2]]
			[x0, y0, z0] = [seglint(h[0:2],x[0:2],h0), seglint(h[0:2],y[0:2],h0), seglint(h[0:2],z[0:2],h0)]
			[xn, yn, zn] = [seglint(h[(npt-2):npt],x[(npt-2):npt],h0), 
							seglint(h[(npt-2):npt],y[(npt-2):npt],h0),
							seglint(h[(npt-2):npt],z[(npt-2):npt],h0)]

			if(h_in[0] > h0 and (h_in[0] < (0.2*np.max(h))) and (h_in[0] < h_in[1])
					and (np.abs(x0-x[1]) < 0.5*np.max(h)) and (np.abs(y0-y[1]) < 0.5*np.max(h))): 
				# Left end of loop has h>h0, append h=h0 as left footpoint:
				[x, y, z] = [np.hstack((x0,x)), np.hstack((y0,y)), np.hstack((z0,z))]
			if(h_in[0] < h0): # Left end of loop has z<z0, replace with z=z0 at left footpoint:
				[x[0], y[0], z[0]] = [x0, y0, z0]
			if(h_in[npt-1] > h0 and (h_in[npt-1] < (0.2*np.max(h_in))) and (h_in[npt-1] < h_in[npt-2])
					and (np.abs(xn-x[npt-1]) < 0.5*np.max(h)) and (np.abs(yn-y[npt-1]) < 0.5*np.max(h))): 
				# Right end of loop has h>h0, append z=zn as right footpoint:
				[x, y, z] = [np.hstack((x,xn)), np.hstack((y,yn)), np.hstack((z,zn))]
			if(h_in[-1] < h0): # Right end of loop has z<z0, replace with z=zn at right footpoint:
				[x[-1], y[-1], z[-1]] = [xn, yn, zn]

			npt = len(z)
			dls = ((x[1:npt]-x[0:npt-1])**2+(y[1:npt]-y[0:npt-1])**2+(z[1:npt]-z[0:npt-1])**2)**0.5
			length0s = np.cumsum(np.hstack([0,dls]))
			looplength0 = np.max(length0s)
			
			spacing = np.clip(zfrac*z,spacing0,lfrac*looplength0)
			nofl0 = np.hstack([0,np.cumsum(dls/spacing[1:npt])])
			nofl = interp1d(length0s,nofl0,fill_value='extrapolate',kind='linear')
			npts_out = np.clip(np.ceil(nofl(looplength0)).astype('int32'),5,254)
			n_out = np.linspace(0.5,nofl(looplength0)-0.5,npts_out)
			lengths = interp1d(nofl0,length0s,fill_value='extrapolate',kind='linear')(n_out)

			coords_out = np.zeros([npts_out,3],dtype='float32')
			coords_out[:,0] = interp1d(length0s,x,fill_value='extrapolate',kind='linear')(lengths)
			coords_out[:,1] = interp1d(length0s,y,fill_value='extrapolate',kind='linear')(lengths)
			coords_out[:,2] = interp1d(length0s,z,fill_value='extrapolate',kind='linear')(lengths)
			npts_out = len(coords_out[:,0])
			dls_out = np.sum((coords_out[1:npts_out,:]-coords_out[0:npts_out-1,:])**2,axis=1)**0.5

			lengths_out.append(np.cumsum(np.hstack([0,dls_out])))
			fieldlines_out.append(coords_out[0:npts_out,:])

	return fieldlines_out,lengths_out

def filter_loops(fieldlines_in, lengths_in, flt_grid, lmin=None, zmin=None, rcurv=None):
    
	[fieldlines_out, lengths_out] = [[],[]]
	cells = np.ones(flt_grid.dims)
	if(zmin is None): zmin = flt_grid.coords(0.5*np.array([flt_grid.dims[0],flt_grid.dims[1],1]))[2]
	if(lmin is None): lmin = zmin*np.pi
	for i in range(0,len(fieldlines_in)):
		line = fieldlines_in[i]
		if(rcurv is None): 
			heights = line[:,2]
		else:
			heights = np.sqrt(np.sum(line[:,0:2]**2)+(rcurv+line[:,2])**2)-rcurv
		fp1 = np.clip(np.floor(flt_grid.inds(line[0])).astype(np.int32),0,flt_grid.dims-1)
		fp2 = np.clip(np.floor(flt_grid.inds(line[-1])).astype(np.int32),0,flt_grid.dims-1)
		length_check = np.max(lengths_in[i]) > lmin
		height_check = np.max(heights) > zmin
		print(i, zmin/1.0e8, lmin/1.0e8, np.max(heights)/1.0e8, length_check, height_check, np.max(np.abs(line[-1]-line[0]))/1.0e8)
		if(cells[tuple(fp1)]*cells[tuple(fp2)]*length_check*height_check):
			fieldlines_out.append(line)
			lengths_out.append(lengths_in[i])
			cells[tuple(fp1)] = 0
			cells[tuple(fp2)] = 0
	return fieldlines_out,lengths_out
        
   
