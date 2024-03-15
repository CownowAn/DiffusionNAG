FOLDER_NAME='tr_meta_surrogate_nb201'

CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/tr_meta_surrogate.py \
    --mode train \
    --config.folder_name $FOLDER_NAME

