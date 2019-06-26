#!/bin/sh

#SBATCH --time=6-0:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH --qos=mcdermott 
#SBATCH -p om_all_nodes # Partition to submit to
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gahlm@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/gahlm/5deg_arch_search/CNN_spherical_%A_%a.stdout
#SBATCH -e /home/gahlm/5deg_arch_search/CNN_spherical_%A_%a.stder
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --mem=130G
#SBATCH --gres=gpu:titan-x:2
#SBATCH --array=0-99%7
module add openmind/singularity/2.4.5

#total = 100
offset=0
trainingID=$(($SLURM_ARRAY_TASK_ID + $offset))

singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.5.simg python -u call_model_training.py $trainingID
singularity exec --nv -B /om -B /nobackup -B /om2 tfv1.5.simg python -u call_model_testing.py $trainingID
