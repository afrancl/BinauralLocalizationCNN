# BinauralLocalizationCNN
**Code to create networks that localize sounds sources in 3D environments**


1. Main training/testing python script is `call_model_training_valid_pad_francl.py`. 
    * This script is responsible for processing the expeeriment parameters, validating the model folder, saving a copy of the experiment parameters there and ensuring the samee folder isn't used for two different training rounds.
    * An example set of parameeters can be found for testing in `slurm_CNN_spherical_testing_francl.sh` and in `slurm_CNN_spherical_training_francl.sh`.
3. Networks weights can be downloaded at: https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0

2. The model should be nervegrams with their associated metadata saved into tensorflow records. The cochlear model we use is the [PyCochleagram package ](https://github.com/mcdermottLab/pycochleagram). We have a wrapper to transform steereo wavefiles into the proper input available here: https://github.com/afrancl/BinauralDataGen



Note: Before running, please change the model save folder to point to your directory with the model architecture config file and data folder to point to your data. Both of theese are in the associated shell scripts. The code itself contains no absolute paths.

# Setup
To aid reproducability and decrease setup time we provide a [Singularity Image](https://sylabs.io/singularity/) that contains all packagees necessary to run tthe code without any further setup. The image is available on dropbox here: https://www.dropbox.com/s/ey74fiw4uquww0n/tfv1.13_tcmalloc.simg?dl=0
