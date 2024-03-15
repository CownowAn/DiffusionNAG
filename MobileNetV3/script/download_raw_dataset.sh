DATANAME=$1

if [[ $DATANAME = 'aircraft' ]]; then
    echo '[Downloading aircraft]'
    python main_exp/get_files/get_aircraft.py

elif [[ $DATANAME = 'pets' ]]; then
    echo '[Downloading pets]'
    python main_exp/get_files/get_pets.py

else
    echo 'Not Implemeted'
fi