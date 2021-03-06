from tf_record_CNN_spherical_gradcheckpoint import tf_record_CNN_spherical
import os
import glob
import numpy as np
from layer_generator import generate
import sys

tone_version=False
itd_tones=False
ild_tones=False
#divide azim/elev label by 10 if false
manually_added=True
all_positions_bkgd=True
testing=True
#Sends Net builder signals to create a branched network, calculates both
#localization and recognition loss
branched=False
#Sets stim size to 30000 in length
zero_padded=True
#Parses record expecting frequency label if True
freq_label = True
#Parses record expecting [N,M,2] format instead of interleaved [2N,M] format if True
stacked_channel = False 

model_version=70000
num_epochs=None

#paths to stimuli and background subbands
bkgd_train_path_pattern = '/om/scratch/Mon/francl/bkgdRecords_pinknoise_vary_env/train*.tfrecords'
train_path_pattern = '/nobackup/scratch/Mon/francl/noiseRecords_1octv_man-added-ITD_wide_range_no_hanning/train*.tfrecords'

#arch_ID=int(sys.argv[1])
arch_ID=38

newpath='/om2/user/francl/localization_runs/old_hrirs_no_hanning_window/arch_number_'+str(arch_ID)
if not os.path.exists(newpath):
    os.mkdir(newpath)

if not os.path.isfile(newpath+'/config_array.npy'):
    config_array = generate()
    np.save(newpath+'/config_array.npy',config_array)
else:
    config_array=np.load(newpath+'/config_array.npy')

files=(glob.glob(newpath+'/*'))
num_files=len(files)

if os.path.isfile(newpath+'/curve_no_resample_w_cutoff_vary_loc.json'):
    testing = True

if testing:
    test=tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,all_positions_bkgd,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath)
