# Alzheimer's disease skull reconstruction
Repository for reconstruction of simulated skull CT data for AD project with files for reconstruction simulated skull CT data (KTH/SSF AD project)

## Usage

### Installation
To close the repository, run
```bash
$ git clone https://github.com/davlars/ad-skull-reconstruction
$ cd ad-skull-reconstruction
```
To install the required dependencies, run (in the root folder of this package)
```bash
$ pip install -r requirements.txt
```

### Load data to disc
In order to use this repo, you first need to copy the data to your local machine. This can be done by:
```bash
$ python -c "import adutils; adutils.load_data_from_nas('Z:/')"
```
where `'Z:/'` should be replaced with your local path to the "REFERENCE" data drive of the NAS (i.e 'Z:/' on windows). This takes quite some time (~10 minutes) to run at first, but makes subsequent reconstructions much faster. Note that it uses ~6GB of disk space in a sub-folder to this project.

Files to read and reconstruct the data is given in this repository (the easiest example is given by a FBP reconstruction in [`FBP_reco_skullCT.py`](FBP_reco_skullCT.py). Most of the data handling is however hidden in [`adutils.py`](adutils.py). To make use of these, simply run the following in your script
```python
import adutils
```  

### Rebin data 
To avoid heavy computations, a suggestion is to use downsampled rebinned data. To rebin data, run the following after you've downloaded data to your local drive:
```bash
$ python -c "import adutils; adutils.rebin_data(10)"
```
with desired ``` (rebin_factor)``` (default is set to 10). Once you've done this, simply load your data using the rebin data flag, as per below:
```python
rebin_factor = 10

# Discretization
reco_space = adutils.get_discretization()

#Forward operator
A = adutils.get_ray_trafo(reco_space, use_rebin=True, rebin_factor=rebin_factor)

# Data
rhs = adutils.get_data(A, use_rebin=True, rebin_factor=rebin_factor)
```

### Save data
To save data in a format that the clinical can review (typically [nifti](https://nifti.nimh.nih.gov/nifti-1)), use the ```adutils.save_data``` utility, with ```x``` being your reconstruction

```python
fileName = /my/path/myFile
adutils.save_data(x, fileName, as_nii=True, as_npy=True)
```

### Visualize data and compare to ground truth phantom
To visualize data, simply call built-in odl functionalities like ```my_reconstruction.show()```, or alternatively use ```adutils.plot_data(reco)``` to get pre-defined cuts of clinical importance. 

Also, using ```adutils.get_phantom``` the ground-truth phantom for the simulated data set can be retrieved. An example of such is given in the [2D-FBP](https://github.com/davlars/ad-skull-reconstruction/blob/master/FBP_reco_skullCT_2D.py), where the corresponding 2D-slice is loaded on the lines of:

```python
# Compare to phantom
phantom = reco_space.element(adutils.get_phantom(use_2D=True))

phantom.show()
```

Alternatively, you can load the entire 3D dataset as a label map. The phantom is given with attenuation values expected for the spectrum used for the simualted dataset (120kVp). Note that if wanted the phantom can be loaded as labelled with tissue flags such that:
```
  0 - Air/Background
  1 - CSF/Soft tissue
  2 - Grey matter
  3 - White matter
  4 - Bone
```

for that, call

```python
phantom = reco_space.element(adutils.get_phantom(use_2D=True, get_Flags=True))

phantom.show()
```

### 2D data set
To be able to work with a smaller dataset (to e.g. try out different reconstruction parameters), a 2D fanbeam dataset has been generated, consiting of a mid transversal slice of the skull. To use this dataset, simply use the flag ```use_2D = True``` in the appropriate calls to ```adutils```. This means e.g.:
``` 
# Discretization
reco_space = adutils.get_discretization(use_2D=True)

#Forward operator
A = adutils.get_ray_trafo(reco_space, use_2D=True)

# Data
rhs = adutils.get_data(A, use_2D=True)
```
Exampes of such is given for [FBP](https://github.com/davlars/ad-skull-reconstruction/blob/master/FBP_reco_skullCT_2D.py), [CGLS](https://github.com/davlars/ad-skull-reconstruction/blob/master/CGLS_reco_skullCT_2D.py), and [TV](https://github.com/davlars/ad-skull-reconstruction/blob/master/TV_reco_skullCT_2D.py).


## Raw data

All simulated data is available on the lcrnas. For high dose (150 mGy) skullCT (following settings used at KI Geriatrics), data can be found in:
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
