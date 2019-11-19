from tf_record_CNN_spherical_gradcheckpoint_valid_pad import tf_record_CNN_spherical
import tensorflow as tf
import os
import glob
import numpy as np
from layer_generator import generate
import json
import sys
import pdb
from types import ModuleType

def allvars_filtered(offset=0):
    #Saves all variables in their current state and returns them as dict
    frame = sys._getframe(1+offset)
    d = frame.f_globals
    d.update(frame.f_locals)
    #Filters out all modules and functions so the output can be serialized to a
    #JSON
    filtered = {key:value for key, value in d.items() if not callable(value) and
               not isinstance(value, ModuleType) and '__loader__' not in key}
    return filtered

tone_version=False
itd_tones=False
ild_tones=False
#Sends Net builder signals to create a branched network, calculates both
#localization and recognition loss
branched=False
#Sets stim size to 30000 in length
zero_padded=True

#model_version=85000
num_epochs=None

#paths to stimuli and background subbands
#bkgd_train_path_pattern = '/om/scratch/Sat/francl/bkgdRecords_textures_sparse_sampled_same_texture_expanded_set_44.1kHz_stackedCH_upsampled/train*.tfrecords'
#train_path_pattern ='/nobackup/scratch/Sat/francl/stimRecords_convolved_oldHRIRdist140_no_hanning_stackedCH_upsampled/testset/train*.tfrecords'

str2bool = lambda x: True if x == "True" else False

arch_ID=int(sys.argv[1])
init = int(sys.argv[2])
regularizer=str(sys.argv[3])
exec("regularizer = "+ regularizer)
bkgd_train_path_pattern = str(sys.argv[4])
train_path_pattern = str(sys.argv[5])
model_version=[]
model_version = list(map(int,list((str(sys.argv[6]).split(',')))))
model_path = str(sys.argv[7])
SNR_max = int(sys.argv[8])
SNR_min = int(sys.argv[9])
manually_added = str2bool(sys.argv[10])
freq_label = str2bool(sys.argv[11])
sam_tones = str2bool(sys.argv[12])
transposed_tones = str2bool(sys.argv[13])
precedence_effect = str2bool(sys.argv[14])
narrowband_noise = str2bool(sys.argv[15])
stacked_channel = str2bool(sys.argv[16])
all_positions_bkgd = str2bool(sys.argv[17])
background_textures = str2bool(sys.argv[18])
testing = str2bool(sys.argv[19])

#newpath='/om2/user/francl/localization_runs/old_hrirs_no_hanning_window_valid_padding/arch_number_'+str(arch_ID)+'_init_'+str(init)
if regularizer is None:
    newpath= model_path+'/arch_number_'+str(arch_ID)+'_init_'+str(init)
else:
    newpath= model_path+'/arch_number_'+str(arch_ID)+'_init_'+str(init)+'_reg'

if not os.path.exists(newpath):
    os.mkdir(newpath)

filtered_vars = allvars_filtered()
with open(newpath+'/config.json','w') as fp:
    json.dump(filtered_vars,fp)

config_array=np.load(newpath+'/config_array.npy')

files=(glob.glob(newpath+'/*'))
files_filtered = [f for f in files if ".json" not in f]
num_files=len(files_filtered)

if os.path.isfile(newpath+'/curve_no_resample_w_cutoff_vary_loc.json'):
    print("Training may be completed. Remove training curve JSON to continue.")
    sys.exit()

test=tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,sam_tones,transposed_tones,precedence_effect,narrowband_noise,all_positions_bkgd,background_textures,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath,regularizer,SNR_max,SNR_min)
