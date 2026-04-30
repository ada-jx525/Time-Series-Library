export CUDA_VISIBLE_DEVICES=0

model_name=BranchWorldModel
root_path=./dataset/ETT-small/
data_path=ETTh1.csv
data_name=ETTh1
seq_len=96
label_len=48
enc_in=7

common_args="--task_name long_term_forecast --is_training 1 --root_path ${root_path} --data_path ${data_path} --model ${model_name} --data ${data_name} --features M --seq_len ${seq_len} --label_len ${label_len} --enc_in ${enc_in} --dec_in ${enc_in} --c_out ${enc_in} --d_model 128 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 256 --wm_latent_dim 128 --wm_branch_num 4 --wm_retrieve_k 4 --wm_neighbor_k 64 --wm_memory_size 2048 --wm_backbone patch_transformer --wm_freeze_backbone 1 --wm_head_type moe --itr 1"

for pred_len in 96 192 336 720
do
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_full --des BW_full
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_nomem --des BW_nomem --wm_use_memory 0
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_rawretr --des BW_rawretr --wm_use_branch_discovery 0
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_single --des BW_single --wm_branch_num 1 --wm_retrieve_k 1
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_nogate --des BW_nogate --wm_use_gating 0
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_noaux --des BW_noaux --wm_use_aux_losses 0
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_e2e --des BW_e2e --wm_freeze_backbone 0
  python -u run.py ${common_args} --pred_len ${pred_len} --model_id ETTh1_96_${pred_len}_BW_shared --des BW_shared --wm_head_type shared
done
