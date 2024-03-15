export LD_LIBRARY_PATH=/opt/conda/envs/gtctnz_2/lib/python3.7/site-packages/nvidia/cublas/lib/

echo '[Downloading processed]'
python main_exp/transfer_nag/get_files/get_preprocessed_data.py
