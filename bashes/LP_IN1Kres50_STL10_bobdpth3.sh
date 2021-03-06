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
--batch_size 128  --dataset stl10 --figsize 224 --loss_type ce --model resnet50 --Bob_depth 3 \
--lr 0.0005 --weight_decay 0 --optim_type sgd --scheduler_type cosine --min_lr 0.0005 \
--work_dir ./results/IN1K_res50_PT \
--alice_name resnet50-classification.pth \
--run_name LP_IN1K50_STL10_2en4nowddecay_bobdpth3