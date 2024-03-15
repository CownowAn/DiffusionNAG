FOLDER_NAME='tr_scorenet_ofa'

CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/tr_scorenet_ofa.py \
    --mode train \
    --config.folder_name $FOLDER_NAME
