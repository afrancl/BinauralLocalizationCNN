from tf_record_CNN_spherical_gradcheckpoint_valid_pad import tf_record_CNN_spherical
import os
import glob
import numpy as np
from layer_generator_new import generate
import sys

tone_version=False
itd_tones=False
ild_tones=False
sam_tones=True
transposed_tones=False
precedence_effect=False
narrowband_noise=False
manually_added=False
freq_label=True
all_positions_bkgd=False
background_textures = True
testing=True
branched=False
zero_padded=True
stacked_channel = True

model_version=[35000,85000]
num_epochs=None

regularizer = None

#paths to stimuli and background subbands
bkgd_train_path_pattern = '/om/scratch/Wed/francl/bkgdRecords_textures_sparse_sampled_same_texture_expanded_set_44.1kHz_stackedCH_upsampled/train*.tfrecords'
train_path_pattern = '/nobackup/scratch/Wed/francl/samToneRecords_fulllengthTime_jitteredCarrier_jitterSlice_stackedCH_upsampled/train*.tfrecords'

#arch_ID=int(sys.argv[1])
arch_ID=518
init = 0

newpath='/om2/user/francl/localization_runs/arch_search_stackedCH_upsampled_sparse_textures_oldHRIR140/arch_number_'+str(arch_ID)
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
    train=tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,sam_tones,transposed_tones,precedence_effect,narrowband_noise,all_positions_bkgd,background_textures,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath,regularizer)



