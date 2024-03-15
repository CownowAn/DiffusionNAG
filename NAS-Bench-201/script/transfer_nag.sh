FOLDER_NAME='transfer_nag_nb201'

GPU=$1
DATANAME=$2

CUDA_VISIBLE_DEVICES=$GPU python main_exp/transfer_nag/main.py \
    --gpu $GPU \
    --test \
    --folder_name $FOLDER_NAME \
    --data-name $DATANAME
