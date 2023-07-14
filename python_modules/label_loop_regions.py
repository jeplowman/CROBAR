import time, resource, numpy as np
#from processify import processify
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

# @processify
def label_loop_regions(fieldlines, lengths, vox_grid, loopnames=None, boundpad=0.2,
		boundpad_zl=0.0,boundpad_zh=0.2,fp_resamp_vox=1.0,cellbound_padfac=1.5):

	nloops = len(fieldlines)
	
	dvox = np.diag(vox_grid.fwd)
	nvox = vox_grid.dims
	voxmin = vox_grid.coords(0*nvox)
	
	# TO DO: fix the code so that it deals with non-cube voxels correctly. For now,
	#dvox = np.array([dvox[0],dvox[0],dvox[0]]).astype(np.float32) # Force all voxel dimensions to be the same.
	#Bd_size = Bdomain[1,:]-Bdomain[0,:]
	#nvox = np.floor((1.0-np.array([2*boundpad,2*boundpad,boundpad_zh+boundpad_zl]))*Bd_size/dvox).astype('int32')+1
	#vd_size = np.array(dvox*(nvox-1)).astype('float32') # This accounts for voxel centers offset by 0.5 from corner coordinates...
	#voxmin = np.array([Bdomain[0,0]+0.5*(Bd_size[0]-vd_size[0]),Bdomain[0,1]+0.5*(Bd_size[1]-vd_size[1]),Bdomain[0,2]+boundpad_zl*Bd_size[2]])

	# The voronoi tiling benefits from padding the domain with boundary points.
	# Generate these, placing them outside the actual domain boundary by a factor
	# of 1.5 (we don't want it assigning points inside the actual domain to these
	# extra-domain points).
	vorpad = 0.5
	nboundpts = 2
	vor_bnd = np.reshape(nvox*((1+2*vorpad)*np.indices([nboundpts]*3).transpose([1,2,3,0])-vorpad),[nboundpts**3,3])
	#vor_bnd = (vox_grid.coords[nvox]-voxmin)*((1+2*vorpad)*np.indices([nboundpts]*3).transpose([1,2,3,0])-vorpad)
	#vor_bnd = (np.reshape(Bdomain[0]+vor_bnd,[nboundpts**3,3])-voxmin)/dvox
	

	looplengths_orig = np.array([np.max(l) for l in lengths])
	looplengths = looplengths_orig/dvox[0]

	ila = np.zeros(nloops+1,dtype='int32')
	for i in range(0,nloops): ila[i+1] = ila[i]+len(fieldlines[i])#.value)
	
	loops_npts=ila[nloops]
	print("Total number of loop points = ",loops_npts)
	looppt_coords = np.zeros([loops_npts,3],dtype='float32')
	looppt_ids = np.zeros([loops_npts],dtype='uint16')
	looppt_deltas = np.zeros([loops_npts,3],dtype='float32')
	looppt_lengths = np.zeros([loops_npts],dtype='float32')
	loop_closed = np.zeros(nloops,dtype='bool')

	for i in range(0,nloops): looppt_coords[ila[i]:ila[i+1],:]=fieldlines[i]
	#looppt_coords = (looppt_coords-voxmin)/dvox
	looppt_coords = vox_grid.inds(looppt_coords)

	for i in range(0,nloops):
		#zstart = looppt_coords[ila[i],2]
		#zend = looppt_coords[ila[i+1]-1,2]
		zstart = fieldlines[i][0,2] # looppt_coords[ila[i],2]
		zend = fieldlines[i][-1,2] # zend = looppt_coords[ila[i+1]-1,2]
		zmax = np.max(looppt_coords[ila[i]:(ila[i+1]),2]) 
		zlo = np.min(np.abs([zstart,zend]))
		looppt_deltas[ila[i]:(ila[i+1]-1),:] = looppt_coords[(ila[i]+1):ila[i+1],:]-looppt_coords[ila[i]:(ila[i+1]-1),:]
		looppt_deltas[ila[i+1]-1,:] = looppt_deltas[ila[i+1]-2,:]
		looppt_lengths[ila[i]:(ila[i+1])] = np.hstack([0,np.cumsum((((looppt_deltas[ila[i]:(ila[i+1]-1),:])**2).sum(axis=1))**0.5)])
		if(np.max([zstart, zend]) < 2*dvox[2]):
			loop_closed[i] = True
		looppt_ids[ila[i]:ila[i+1]]=i

	loop_dls = ((looppt_deltas**2).sum(axis=1))**(0.5)
	loop_unitvecs = looppt_deltas/np.broadcast_to(loop_dls,[3,loops_npts]).transpose()

	print("Memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, " kb")
	vl0,vh0 = get_cell_bounds(looppt_ids, loops_npts, loop_dls, looppt_coords, vor_bnd)
	print("Memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, " kb")

	chunkpad = 1
	chunkdims = np.array([100,100,100],dtype='int32')
	nchunk_xyz = np.ceil(nvox/chunkdims).astype('int32')
	nchunk_total = np.prod(nchunk_xyz).astype('int32')
	chunk_nlooppts = np.zeros(nchunk_xyz,dtype='int32')
	chunk_bounds = np.zeros([nchunk_total,8,3],dtype='int32')
	for i in range(0,nchunk_total):
		ichunk = np.unravel_index(i,nchunk_xyz)
		chunk_voxlo = np.expand_dims(np.array(ichunk)*chunkdims,0)
		ind = np.reshape(np.array(np.indices([2,2,2])),[3,8]).transpose()
		chunk_bounds[i,:,:] = ind*np.expand_dims(chunkdims,0)+chunk_voxlo+2*chunkpad*ind-chunkpad

	cellbound_pad = np.round(cellbound_padfac*np.clip(np.ceil(0.125*loop_dls),2,20)).astype('int32') # was a constant 3...
	for i in range(0,3):
		vl0[:,i] = np.clip(np.floor(vl0[:,i]-cellbound_pad),0,nvox[i])
		vh0[:,i] = np.clip(np.ceil(vh0[:,i]+cellbound_pad),0,nvox[i])
	vl_chunk = np.floor(vl0/chunkdims).astype(np.int32)
	vh_chunk = np.ceil(vh0/chunkdims).astype(np.int32)

	for i in range(0,loops_npts-1):
		sobj = np.index_exp[vl_chunk[i,0]:vh_chunk[i,0],vl_chunk[i,1]:vh_chunk[i,1],vl_chunk[i,2]:vh_chunk[i,2]]
		np.add.at(chunk_nlooppts,sobj,1)

	chunk_nlp_cs = np.cumsum(chunk_nlooppts.flatten())
	chunk_ila_hi = np.zeros(nchunk_xyz,dtype='uint32')
	chunk_ila_lo = np.zeros(nchunk_xyz,dtype='uint32')
	chunk_ila_cur = np.zeros(nchunk_xyz,dtype='uint32')
	chunk_ila_unravel = np.unravel_index(np.arange(nchunk_total,dtype='uint32'),nchunk_xyz)
	chunk_ila_hi[chunk_ila_unravel] = chunk_nlp_cs
	chunk_ila_lo[chunk_ila_unravel] = np.hstack([0,chunk_nlp_cs[0:nchunk_total-1]])
	chunk_loop_indices = np.zeros(np.sum(chunk_nlooppts),dtype='uint32')
	for i in range(0,loops_npts-1):
		sobj = np.index_exp[vl_chunk[i,0]:vh_chunk[i,0],vl_chunk[i,1]:vh_chunk[i,1],vl_chunk[i,2]:vh_chunk[i,2]]		
		chunk_loop_indices[chunk_ila_lo[sobj]+chunk_ila_cur[sobj]] = i
		np.add.at(chunk_ila_cur,sobj,1)

	voxel_loopids = np.zeros(nvox,dtype='uint16')+nloops
	distmax = np.sum(nvox**2)
	dmax_scal = distmax
	lmax_scal = 254
	voxel_loop_lengths = np.zeros(nvox,dtype='uint8')+lmax_scal

	tstart=time.time()
	print('Number of Chunks: ',nchunk_total)
	for i in range(0,nchunk_total):
		ichunk = np.unravel_index(i,nchunk_xyz)
		chunk_voxlo = np.array(ichunk)*chunkdims
		chunk_voxhi = np.clip((np.array(ichunk)+1)*chunkdims,0,nvox)
		if(((i+1) % (np.round(nchunk_total/20)))==0): print("Checked chunk",i,", elapsed time:",time.time()-tstart, 
				"s, Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"kb")
		lids = chunk_loop_indices[chunk_ila_lo[ichunk]:chunk_ila_hi[ichunk]]
		[chunk_vlids,chunk_vlls] = label_chunk_voxels(loop_unitvecs[lids,:], loop_dls[lids], looppt_coords[lids,:], vl0[lids,:], 
				vh0[lids,:], chunk_voxlo, looppt_lengths[lids], looppt_ids[lids], looplengths[looppt_ids[lids]], lmax_scal, 
				chunk_voxhi-chunk_voxlo, nloops, distmax)
		sobj = np.index_exp[chunk_voxlo[0]:chunk_voxhi[0],chunk_voxlo[1]:chunk_voxhi[1],chunk_voxlo[2]:chunk_voxhi[2]]
		voxel_loopids[sobj]=chunk_vlids
		voxel_loop_lengths[sobj]=chunk_vlls*lmax_scal

	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	if(loopnames == None):
		loopnames = []
		for index in range(0,nloops): loopnames.append(f'loop{index:06d}')

	return voxel_loopids, {'loopcoords':looppt_coords,'loopids':looppt_ids, 'looplengths':looplengths_orig, 'loop_closed':loop_closed,
				'nvox':nvox, 'dvox':dvox, 'voxmin':voxmin,
				'ila':ila, 'looppt_lengths':looppt_lengths, 'loop_dls':loop_dls, 
				'looplengths_vox':looplengths, 'lmax_scal':lmax_scal, 'loopnames':loopnames}, voxel_loop_lengths #,'distances':voxel_loop_distances}

#@processify
def get_cell_bounds(looppt_ids, loops_npts, loop_dls, looppt_coords, vor_bnd):
	tstart=time.time()
	print("Computing Voronoi cells:")
	vor = Voronoi(np.vstack((looppt_coords, vor_bnd)))
	print("Done computing Voronoi cells, elapsed time=",time.time()-tstart)

	print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	vl0 = np.zeros([loops_npts,3],dtype='int32')
	vh0 = np.zeros([loops_npts,3],dtype='int32')
	for i in range(0,loops_npts-1):
		if(looppt_ids[i] == looppt_ids[i+1]):
			verts = np.vstack([looppt_coords[i,:],looppt_coords[i+1,:],vor.vertices[vor.regions[vor.point_region[i]],:],vor.vertices[vor.regions[vor.point_region[i+1]],:]])
		if(looppt_ids[i] != looppt_ids[i+1]):
			verts = np.vstack([looppt_coords[i,:],vor.vertices[vor.regions[vor.point_region[i]],:]])
		vl0[i,:] = np.min(verts,axis=0)
		vh0[i,:] = np.max(verts,axis=0)
	return vl0,vh0

#@processify
def label_chunk_voxels(loop_unitvecs, loop_dls, looppt_coords, vl0, vh0, chunk_voxlo, looppt_lengths, looppt_ids, looplengths, lmax_scal, chunkdims, nloops, distmax):

	dmax_scal = distmax
	lmax_scal = 1.0 # 254.0
	voxel_loopids = np.zeros(chunkdims,dtype='uint16')+nloops
	voxel_loop_distances = np.zeros(chunkdims,dtype='float32')+dmax_scal
	voxel_loop_lengths = np.zeros(chunkdims,dtype='float32')+lmax_scal
	[vox_ixa,vox_iya,vox_iza] = np.indices(chunkdims,dtype='uint16')
	r2s = np.zeros(chunkdims,dtype='float32')
	rdots = np.zeros(chunkdims,dtype='float32')
	distances = np.zeros(chunkdims,dtype='float32')
	lengths = np.zeros(chunkdims,dtype='float32')
	dflags = np.zeros(chunkdims,dtype='float32')
	ndflags = np.zeros(chunkdims,dtype='float32')
	nlooppts = len(loop_dls)
	
	for i in range(0,nlooppts):
		uvec = loop_unitvecs[i,:]
		dl = loop_dls[i]
		coords = looppt_coords[i,:]
		vl = np.clip(vl0[i,:]-chunk_voxlo,0,chunkdims)
		vh = np.clip(vh0[i,:]-chunk_voxlo,0,chunkdims)
		ludif = chunk_voxlo-coords 
		sobj = np.index_exp[vl[0]:vh[0],vl[1]:vh[1],vl[2]:vh[2]]
		r2s[sobj] = (vox_ixa[sobj]+ludif[0])**2+(vox_iya[sobj]+ludif[1])**2+(vox_iza[sobj]+ludif[2])**2
		rdots[sobj] = (vox_ixa[sobj]+ludif[0])*uvec[0]+(vox_iya[sobj]+ludif[1])*uvec[1]+(vox_iza[sobj]+ludif[2])*uvec[2]
		distances[sobj] = r2s[sobj]-(rdots[sobj] > 0)*rdots[sobj]**2+(rdots[sobj] > dl)*(rdots[sobj]-dl)**2
		lengths[sobj] = np.clip((looppt_lengths[i]+rdots[sobj])*lmax_scal/(looplengths[i]),0,lmax_scal)
		dflags[sobj] = (distances[sobj] < voxel_loop_distances[sobj])#*(lengths > 0.0)#*(lengths < 1.0)
		ndflags[sobj] = np.logical_not(dflags[sobj])
		voxel_loop_distances[sobj] = distances[sobj]*dflags[sobj]+voxel_loop_distances[sobj]*ndflags[sobj]
		voxel_loopids[sobj] = looppt_ids[i]*dflags[sobj]+voxel_loopids[sobj]*ndflags[sobj]
		voxel_loop_lengths[sobj] = lengths[sobj]*dflags[sobj]+voxel_loop_lengths[sobj]*ndflags[sobj]

	return voxel_loopids, voxel_loop_lengths
