This is a preliminary repository for the CROBAR code for 3D reconstruction of solar coronal emission measure. Needs Numpy, Numba, Scipy, Astropy, and Sunpy to run; an example notebook is included. See the following on NASA ADS for further information on the code:

https://ui.adsabs.harvard.edu/abs/2021ApJ...922..109P/abstract
https://ui.adsabs.harvard.edu/abs/2023ApJ...947....5P/abstract
https://ui.adsabs.harvard.edu/abs/2023arXiv230908053P/abstract

Troubleshooting: 

If you see an error like "TypeError: lgmres() got an unexpected keyword argument 'tol'": 
Find /python_modules/sparse_nlmap_solver.py line 110 and change "tol" to "atol". This should resolve the issue. 
