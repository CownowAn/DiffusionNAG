FOLDER_NAME='tr_meta_surrogate_ofa'

CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/tr_meta_surrogate_ofa.py \
    --mode train \
    --config.folder_name $FOLDER_NAME
