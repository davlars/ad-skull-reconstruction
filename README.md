# ad-skull-reconstruction
Repository for reconstruction of simulated skull CT data for AD project with files for reconstruction simulated skull CT data (KTH/SSF AD project)

All simulated data is available on the lcrnas-repo. For high dose (150 mGy) skullCT (following settings used at KI Geriatrics), data can be found in:
```
  /lcrnas/data/Simulated/120kV
```
The simulated data consists of 23 spiral turns, and is divded in 23 separate files. The data is given as:
```
  HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Turn_{0,1,2,3,...,22}.data.npy
```
with corresponding geometry pickled in
```
  HelicalSkullCT_70100644Phantom_no_bed_Turn_{0,1,2,3,...,22}.geometry.p
```  
Files to read and reconstruct these are given in this repository (the easiest example is given by a FBP reconstruction in ```HelicalSkullCT_Reconstruction_FBP.py```).

Note: Data with lower dosage is available upon request. All simulated data is given in /lcrnas/data/Simulated/120kV/raw/. DO NOT change the data in this repository. 


