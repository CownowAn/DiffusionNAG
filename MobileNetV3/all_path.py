RAW_DATA_PATH="./data/ofa/raw_data"
PROCESSED_DATA_PATH = "./data/ofa/data_transfer_nag"
SCORE_MODEL_DATA_PATH="./data/ofa/data_score_model/ofa_database_500000.pt"
SCORE_MODEL_DATA_IDX_PATH="./data/ofa/data_score_model/ridx-500000.pt"

NOISE_META_PREDICTOR_CKPT_PATH = "./checkpoints/ofa/noise_aware_meta_surrogate/model_best.pth.tar"
SCORE_MODEL_CKPT_PATH="./checkpoints/ofa/score_model/model_best.pth.tar"
UNNOISE_META_PREDICTOR_CKPT_PATH="./checkpoints/ofa/unnoised_meta_surrogate_from_metad2a"
CONFIG_PATH='./configs/transfer_nag_ofa.pt'