# climate_model_with_HD

We have two code files in the repository. In the first file "Helm_decom_climate_model_with_bc.py", 
we have written a code for incorporating Helmholtz decomposition into the climate mode i.e. we calculate the divergence-free and curl-free parts of atmospheric 
velocity and pass only the div-free part to the ocean model. In this file
we also solve the model on normal boundary conditions i.e. no-slip insulated boundaries.
The other file "clim_model_with_pbc_cc_1.py" is basically the same code we wrote earlier
to simulate the climate model with periodic boundary condtions for complete coupling test case.
We have made changes in this file so that we can compare the results and see if after the Helmholtz decomposition,
we still get satisfactory results.
