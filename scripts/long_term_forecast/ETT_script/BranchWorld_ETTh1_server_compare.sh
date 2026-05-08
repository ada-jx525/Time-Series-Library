#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
DATA_PATH="${DATA_PATH:-ETTh1.csv}"
DATA_NAME="${DATA_NAME:-ETTh1}"
FEATURES="${FEATURES:-M}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
ENC_IN="${ENC_IN:-7}"
DEC_IN="${DEC_IN:-7}"
C_OUT="${C_OUT:-7}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"

COMMON_ARGS="\
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --data ${DATA_NAME} \
  --features ${FEATURES} \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 256 \
  --dropout 0.1 \
  --factor 3 \
  --embed timeF \
  --train_epochs ${TRAIN_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --patience 3 \
  --learning_rate 0.0001 \
  --num_workers ${NUM_WORKERS} \
  --itr 1 \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${GPU}"

BRANCHWORLD_ARGS="\
  --model BranchWorldModel \
  --wm_latent_dim 128 \
  --wm_branch_num 3 \
  --wm_retrieve_k 64 \
  --wm_memory_size 4096 \
  --wm_global_proto_num 32 \
  --wm_kmeans_iters 8 \
  --wm_proto_state_alpha 1.0 \
  --wm_proto_traj_beta 2.0 \
  --wm_backbone patch_transformer \
  --wm_patch_len 16 \
  --wm_patch_stride 8 \
  --wm_memory_update_freq 1 \
  --wm_memory_warmup_epochs 0 \
  --wm_base_type dlinear \
  --wm_freeze_base 1 \
  --wm_mem_loss_type min \
  --wm_mem_weight 0.1 \
  --wm_traj_weight 0.01"

for PRED_LEN in 96 192 336 720; do
  python -u run.py ${COMMON_ARGS} ${BRANCHWORLD_ARGS} \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_BranchWorld_offline_bank \
    --wm_proto_mode offline \
    --des branchworld_offline_bank

  python -u run.py ${COMMON_ARGS} ${BRANCHWORLD_ARGS} \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_BranchWorld_local_dynamic \
    --wm_proto_mode local \
    --des branchworld_local_dynamic

  python -u run.py ${COMMON_ARGS} ${BRANCHWORLD_ARGS} \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_BranchWorld_raw_neighbor \
    --wm_proto_mode offline \
    --wm_use_branch_discovery 0 \
    --des branchworld_raw_neighbor

  python -u run.py ${COMMON_ARGS} \
    --model DLinear \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_DLinear \
    --des dlinear_baseline

  python -u run.py ${COMMON_ARGS} \
    --model PatchTST \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_PatchTST \
    --des patchtst_baseline

  python -u run.py ${COMMON_ARGS} \
    --model iTransformer \
    --pred_len ${PRED_LEN} \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN}_iTransformer \
    --des itransformer_baseline
done
