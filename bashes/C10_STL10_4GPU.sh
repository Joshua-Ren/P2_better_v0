#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=stage2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=./logs/stage1.txt 
#SBATCH --gres=gpu:4


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/functorch-env/bin/activate

cd /home/sg955/GitWS/P2_better_v0/

srun python -m torch.distributed.launch --nproc_per_node=4 --master_port 1086 main_stage2.py \
--batch_size 128  \
--dataset stl10 --figsize 32 --loss_type ce \
--run_name C10_STL10