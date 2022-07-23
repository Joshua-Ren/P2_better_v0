#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=LPFT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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

srun python main_LPFT_tab1.py \
--batch_size 128  --dataset domain_sketch --figsize 224 --loss_type ce --model resnet50 --num_workers 4 \
--Bob_depth 1 --Bob_layer 1 --smoothing 0.1 \
--lr 0.01 --warmup 5 --epochs 200 \
--work_dir ./results/IN1K_res50_PT --proj_name betterv0_tab1LPFT \
--alice_name resnet50-classification.pth \
--run_name Tab1_IN1Kres50_domsketch_D1L1_FTls