FOLDER_NAME='tr_scorenet_nb201'

CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/tr_scorenet.py \
    --mode train \
    --config.folder_name $FOLDER_NAME
