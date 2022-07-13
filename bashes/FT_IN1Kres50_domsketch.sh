#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=FT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

srun python main_FT.py \
--batch_size 128  --dataset domain_sketch --figsize 224 --loss_type ce --model resnet50 --num_workers 4 \
--lr 0.03 --epochs 100 --min_lr 0.00001 \
--work_dir ./results/IN1K_res50_PT \
--alice_name resnet50-classification.pth \
--LP_dir LP_IN1K50_domsketch_2en3wd_f224 \
--run_name FT__LP_IN1K50_domsketch_2en3wd_f224__3en2_run1