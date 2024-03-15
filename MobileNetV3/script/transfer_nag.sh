FOLDER_NAME='transfer_nag_ofa'

N=30
GENSAMPLES=5000
GPU=$1
DATANAME=$2


CUDA_VISIBLE_DEVICES=$GPU python main_exp/run_transfer_nag.py \
    --test --data-name $DATANAME --gpu $GPU \
    --folder_name $FOLDER_NAME \
    --nvt 27 --search_space ofa --graph-data-name ofa \
    --epochs 500 --n_gen_samples $GENSAMPLES --classifier_scale 500 \
    --n_training_samples $N