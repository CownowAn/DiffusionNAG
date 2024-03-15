export LD_LIBRARY_PATH=/opt/conda/envs/gtctnz_2/lib/python3.7/site-packages/nvidia/cublas/lib/

DATANAME=$1

if [[ $DATANAME = 'aircraft' ]]; then
    echo '[Downloading aircraft]'
    python main_exp/transfer_nag/get_files/get_aircraft.py

elif [[ $DATANAME = 'pets' ]]; then
    echo '[Downloading pets]'
    python main_exp/transfer_nag/get_files/get_pets.py

else
    echo 'Not Implemeted'
fi