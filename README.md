# ad-skull-reconstruction
Repository for reconstruction of simulated skull CT data for AD project with files for reconstruction simulated skull CT data (KTH/SSF AD project)

Usage
#####
In order to use this repo, you first need to copy the data to your local machine. This can be done by:
```
$ python -c "import adutils; adutils.load_data_from_nas('Z:\\')"
```
where `'Z:\\'` should be replaced with your local path to the data drive of the NAS. This takes quite some time (~10 minutes) to run at first, but makes subsequent reconstructions much faster. Note that it uses ~6GB of disk space in a subfolder to this project.

Files to read and reconstruct the data is given in this repository (the easiest example is given by a FBP reconstruction in ```FBP_reco_skullCT.py```). Most of the data handling is however hidden in ```adutils.py```. To make use of these, simply run the following in your script
```
import adutils
```  

Raw data
########

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
Note: Data with lower dosage is available upon request. All simulated data is given in ```/lcrnas/data/Simulated/120kV/raw/```. **DO NOT change the data in this repository**. If unsure, ask. 



