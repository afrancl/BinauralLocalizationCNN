#!/bin/sh

#SBATCH --time=2-0:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH --qos=normal
#SBATCH -p om_all_nodes # Partition to submit to
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gahlm@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/gahlm/dataset_update/CNN_spherical_%A_%a.stdout
#SBATCH -e /home/gahlm/dataset_update/CNN_spherical_%A_%a.stder
#SBATCH --ntasks=1
#SBATCH -c 20
#SBATCH --mem=70G
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --array=0
module add openmind/singularity/2.5.1

offset=174
initialization=2
#regularizer="tf.contrib.layers.l1_regularizer(scale=0.001)"
regularizer=None
bkgd_train_path_pattern=/om/scratch/Wed/francl/bkgdRecords_textures_sparse_sampled_same_texture_expanded_set_44.1kHz_stackedCH_upsampled/train*.tfrecords
train_path_pattern=/nobackup/scratch/Wed/francl/toneRecords_fullrange_jitteredPhase_man-added-ITD_stackedCH_upsampled_interpolated_delay/train*.tfrecords
model_version=50000,100000,150000,200000

trainingID=$offset
init=$initialization
reg=$regularizer
bkgd=$bkgd_train_path_pattern
pattern=$train_path_pattern
model=$model_version

#singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.5.simg python -u call_model_training.py $trainingID
#singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.13_openmind.simg python -u call_model_testing_valid_pad.py $a
SINGULARITYENV_LD_PRELOAD=/usr/lib/libtcmalloc.so.4 singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.13_tcmalloc.simg python -u call_model_testing_valid_pad.py $trainingID $init "$reg" "$bkgd" "$pattern" "$model"
#SINGULARITYENV_LD_PRELOAD=/usr/lib/libtcmalloc.so.4 singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.13_tcmalloc.simg python -u call_model_training_valid_pad.py $trainingID $init "$reg" "$bkgd" "$pattern"