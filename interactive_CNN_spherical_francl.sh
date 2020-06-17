#!/bin/sh

#SBATCH --time=3-00:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH --partition=normal
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=francl@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/francl/dataset_update/CNN_spherical_%A_%a.stdout
#SBATCH -e /home/francl/dataset_update/CNN_spherical_%A_%a.stder
#SBATCH --ntasks=1
#SBATCH -c 20
#SBATCH --mem=70G
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --array=103,170,174,193,230,241,518
module add openmind/singularity/2.5.1

#offset=$SLURM_ARRAY_TASK_ID
offset=103
initialization=0
#regularizer="tf.contrib.layers.l1_regularizer(scale=0.001)"
regularizer=None
bkgd_train_path_pattern=/om/scratch/Sat/francl/bkgdRecords_textures_sparse_sampled_same_texture_expanded_set_44.1kHz_stackedCH_upsampled_anechoic/train*.tfrecords
train_path_pattern=/nobackup/scratch/Sat/francl/samToneRecords_fulllengthTime_jitteredCarrier_jitterSlice_stackedCH_upsampled/train*.tfrecords
model_version=50000

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
