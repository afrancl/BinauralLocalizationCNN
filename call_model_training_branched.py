from tf_record_CNN_spherical_gradcheckpoint_branched import tf_record_CNN_spherical
import os
import glob
import numpy as np
from layer_generator import generate
import sys

tone_version=False
itd_tones=False
ild_tones=False
manually_added=False
freq_label=False
all_positions_bkgd=True
testing=False
branched=True
zero_padded=True
stacked_channel = False

model_version=20000
num_epochs=None

#paths to stimuli and background subbands
bkgd_train_path_pattern = '/om/scratch/Tue/francl/bkgdRecords_vary_env/train*.tfrecords'
train_path_pattern = '/nobackup/scratch/Tue/francl/speechRecords_specfilt_2ord2octvfilt_upsampled_convolvedHRIRdist100/train*.tfrecords'

#arch_ID=int(sys.argv[1])
arch_ID=38

newpath='/om2/user/francl/branchpoint_search/branchpoint_1/arch_number_'+str(arch_ID)
if not os.path.exists(newpath):
    os.mkdir(newpath)

if not os.path.isfile(newpath+'/config_array.npy'):
    print("GENERATING  NEW CONFIG!")
    config_array = generate()
    np.save(newpath+'/config_array.npy',config_array)
else:
    config_array=np.load(newpath+'/config_array.npy')

files=(glob.glob(newpath+'/*'))
num_files=len(files)

if os.path.isfile(newpath+'/curve_no_resample_w_cutoff_vary_loc.json'):
    testing = True

if not testing:
    train=tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,all_positions_bkgd,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath)



