#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=LP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=./logs/stage1.txt 
#SBATCH --gres=gpu:1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/functorch-env/bin/activate

cd /home/sg955/GitWS/P2_better_v0/

srun python main_LP.py \
--batch_size 128  --dataset cifar10p --figsize 32 --loss_type ce \
--lr 0.001 --weight_decay 0.05 --min_lr 0.001 \
--work_dir ./results/C10_res18_PT --proj_name betterv0_Fig7LP \
--alice_name Alice_resnet18_PT.pth \
--run_name Fig7LP_C10res18_C10p