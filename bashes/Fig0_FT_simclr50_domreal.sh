#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=FT
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

srun python main_FT.py \
--batch_size 128  --dataset domain_real --figsize 224 --loss_type ce --model resnet50 --num_workers 4 \
--lr 0.0001 --epochs 200 --min_lr 0.000001 \
--work_dir ./results/IN1K_simclr_PT \
--alice_name resnet50-simclr.pth \
--LP_dir LP_simlr50_domrealnew_1en2 \
--run_name FT__LP_simlr50_domrealnew_1en2