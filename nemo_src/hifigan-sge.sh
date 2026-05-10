#!/bin/bash
#
#$ -S /bin/bash
#$ -N hifigan-finetune
#$ -q long.q@@gpu
#$ -o /mnt/matylda6/xokruc00/knn/first_nemo_tts/hifigan-task.out
#$ -e /mnt/matylda6/xokruc00/knn/first_nemo_tts/hifigan-task.err
#$ -l gpu=1,gpu_ram=24G,matylda6=5,ram_free=32G,tmp_free=8G
#

ulimit -t 200000
ulimit -n 4096
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

source /mnt/matylda6/xokruc00/nemo-env/bin/activate

export HF_HOME=/mnt/matylda6/xokruc00/HF/cache/
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda6/xokruc00/training-scripts/free-gpus.sh 1)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/matylda6/xokruc00/knn/second_nemo_tts/NeMo

python /mnt/matylda6/xokruc00/knn/first_nemo_tts/NeMo/examples/tts/hifigan_finetune.py --config-path=/mnt/matylda6/xokruc00/knn/first_nemo_tts/NeMo/conf/   --config-name=hifigan_finetune