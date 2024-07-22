
python submitit_pretrain.py \
    --job_dir  "/checkpoint/tomsander/experiments/" \
    --folder 240719_amp_4_prints_2 \
    --dataset_path "/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar"\
    --nb_samples 233804767 \
    --nodes 1 \
    --ngpus 4 \
    --use_volta32 \
    --num_workers 10 \
    --batch_size 1310720\
    --TAN_batch_size 512 \
    --max_physical_B 128 \
    --model mae_vit_base_patch16_autoregressive_nobias \
    --resume "/private/home/tomsander/dp_multimodal/checkpoints/DP-Cap/base-init.pth" \
    --init True\
    --DP "ghost" \
    --target_txt_len 40 \
    --mask_ratio 0 \
    --blr 1.0e-07 \
    --amp False \
    --partition devlab \
    --weight_decay 0.005 \
    --sigma 0 \
    --max_grad_norm 1 \
    --warmup_iterations 2000 \
    --overall_iterations 10000 \
    --target_step 5700\
    --save_freq 1000

# python submitit_pretrain.py \
#     --nodes 16 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 200000\
#     --max_physical_B 128 \
#     --model mae_vit_base_patch16_autoregressive_nobias \
#     --resume "/checkpoint/tomsander/experiments/M3AE/231212_Cap200k/checkpoint-1-begin.pth" \
#     --DP "ghost" \
#     --amp True \
#     --target_txt_len 40 \
#     --mask_ratio 0 \
#     --blr 6.54e-07 \
#     --partition learnlab \
#     --weight_decay 0.005 \
#     --DP "ghost" \
#     --sigma 0.513 \
#     --max_grad_norm 1 \
#     --folder 241019_Cap200k_next_2\
#     --warmup_iterations 2000 \
#     --overall_iterations 10000 \
#     --target_step 5700\
#     --save_freq 1000

# path_load_index='/checkpoint/yaodongyu/laion2b_dataset/blur_dataset_torch_format', 
# face_masks_folder = '/checkpoint/haideraltahan/laion440m_masks'):

#Without blur default:
# '/checkpoint/yaodongyu/laion2b_dedup_subset233m/dataset_torch_format'

#Without Poisson
# LAION_PATH ="/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar"


# python submitit_pretrain.py \
#     --nodes 12 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 1310720\
#     --TAN_batch_size 20480\
#     --max_physical_B 128 \
#     --model mae_vit_base_patch16_autoregressive_nobias \
#     --resume "/checkpoint/yaodongyu/experiments/M3AE-S21k-decoder/10535479/checkpoint-700.pth" \
#     --init True\
#     --DP "ghost" \
#     --amp True \
#     --target_txt_len 40 \
#     --mask_ratio 0 \
#     --blr 1e-07 \
#     --partition learnfair \
#     --weight_decay 0.005 \
#     --DP "ghost" \
#     --sigma 0 \
#     --max_grad_norm 1 \
#     --folder 231122_clip_no_noise_50k_2 \
#     --warmup_iterations 2000 \
#     --overall_iterations 50000 \
#     --target_step 5708\
#     --save_freq 2000

# python submitit_pretrain.py \
#     --nodes 4 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 13107 \
#     --max_physical_B 128 \
#     --data_ratio 0.01 \
#     --model mae_vit_small_patch16_autoregressive_nobias \
#     --resume "/checkpoint/yaodongyu/experiments/M3AE-S21k-decoder/14078798/checkpoint-700.pth"\
#     --init True \
#     --DP "ghost" \
#     --amp True \
#     --target_txt_len 40 \
#     --mask_ratio 0 \
#     --epochs 50 \
#     --warmup_epochs 5 \
#     --blr 1e-05 \
#     --TAN_batch_size 13107 \
#     --partition learnfair \
#     --weight_decay 0.005 \
#     --dataset LION \
#     --DP "ghost" \
#     --sigma 0.728 \
#     --max_grad_norm 1 \
#     --folder 230927_0.01_smallYY_3 \
#     --warmup_iterations 1000 \
#     --overall_iterations 10000 \
#     --save_freq 2000

# python submitit_pretrain.py \
#     --nodes 4 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 13107 \
#     --max_physical_B 128 \
#     --data_ratio 0.01 \
#     --model mae_vit_base_patch16_autoregressive_nobias \
#     --resume "/checkpoint/yaodongyu/experiments/M3AE-S21k-decoder/10535479/checkpoint-700.pth" \
#     --init True \
#     --DP "ghost" \
#     --amp True \
#     --target_txt_len 40 \
#     --mask_ratio 0 \
#     --epochs 50 \
#     --warmup_epochs 5 \
#     --blr 1e-05 \
#     --TAN_batch_size 13107 \
#     --partition learnfair \
#     --weight_decay 0.005 \
#     --dataset LION \
#     --DP "ghost" \
#     --sigma 0.728 \
#     --max_grad_norm 1 \
#     --folder 230927_0.01_baseYY \
#     --warmup_iterations 1000 \
#     --overall_iterations 10000 \
#     --save_freq 2000

# python submitit_pretrain.py \
#     --nodes 13 \
#     --ngpus 8 \
#     --num_workers 10\
#     --use_volta32 \
#     --batch_size 1310720 \
#     --TAN_batch_size 1310720 \
#     --max_physical_B 150 \
#     --model mae_vit_small_patch16_autoregressive_nobias \
#     --resume "/checkpoint/tomsander/experiments/M3AE/230924_eps2_small_restart2/checkpoint-10.pth"\
#     --poisson True\
#     --DP "ghost"\
#     --amp True\
#     --target_txt_len 40\
#     --mask_ratio 0 \
#     --overall_iterations 6000\
#     --warmup_iterations 600\
#     --target_step 2854\
#     --blr 1e-07\
#     --partition learnfair\
#     --weight_decay 0.005 \
#     --dataset LION \
#     --DP "ghost"\
#     --sigma 1.18\
#     --max_grad_norm 1\
#     --folder 230926_eps2_reload_5\
#     --save_freq 500


# python submitit_pretrain.py \
#     --nodes 10 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 131072 \
#     --TAN_batch_size 131072 \
#     --max_physical_B 150 \
#     --model mae_vit_small_patch16_autoregressive_nobias \
#     --poisson True\
#     --DP "ghost"\
#     --amp True\
#     --target_txt_len 40\
#     --mask_ratio 0 \
#     --overall_iterations 10000\
#     --warmup_iterations 2000\
#     --blr 1e-07\
#     --target_step 5708\
#     --partition learnfair\
#     --weight_decay 0.005 \
#     --dataset LION \
#     --data_ratio 0.1\
#     --DP "ghost"\
#     --sigma 0.728\
#     --max_grad_norm 1\
#     --folder 230920_small_random_dataratio_0.1_2\
#     --save_freq 500


# python submitit_pretrain.py \
#     --nodes 16 \
#     --ngpus 8 \
#     --num_workers 10 \
#     --use_volta32 \
#     --batch_size 1310720 \
#     --TAN_batch_size 1310720 \
#     --max_physical_B 150 \
#     --model mae_vit_small_patch16_autoregressive_nobias \
#     --resume "/checkpoint/tomsander/experiments/M3AE/230917_small_random_4/checkpoint-11.pth"\
#     --poisson True\
#     --DP "ghost"\
#     --amp True\
#     --target_txt_len 40\
#     --mask_ratio 0 \
#     --overall_iterations 10000\
#     --warmup_iterations 2000\
#     --blr 1e-07\
#     --partition learnfair\
#     --weight_decay 0.005 \
#     --dataset LION \
#     --DP "ghost"\
#     --sigma 0.728\
#     --max_grad_norm 1\
#     --folder 230921_small_random_restart\
#     --save_freq 500