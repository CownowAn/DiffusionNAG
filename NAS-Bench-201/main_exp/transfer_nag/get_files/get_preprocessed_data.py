###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import os
from tqdm import tqdm
import requests


DATA_PATH = "./data/transfer_nag"
dir_path = DATA_PATH
if not os.path.exists(dir_path):
	os.makedirs(dir_path)


def download_file(url, filename):
	"""
	Helper method handling downloading large files from `url`
	to `filename`. Returns a pointer to `filename`.
	"""
	chunkSize = 1024
	r = requests.get(url, stream=True)
	with open(filename, 'wb') as f:
		pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
		for chunk in r.iter_content(chunk_size=chunkSize):
			if chunk: # filter out keep-alive new chunks
				pbar.update (len(chunk))
				f.write(chunk)
	return filename


def get_preprocessed_data(file_name, url):
		print(f"Downloading {file_name} datasets\n")
		full_name = os.path.join(dir_path, file_name)
		download_file(url, full_name)
		print("Downloading done.\n")


for file_name, url in [
	('aircraftbylabel.pt', 'https://www.dropbox.com/s/mb66kitv20ykctp/aircraftbylabel.pt?dl=1'),
	('cifar100bylabel.pt', 'https://www.dropbox.com/s/y0xahxgzj29kffk/cifar100bylabel.pt?dl=1'),
	('cifar10bylabel.pt', 'https://www.dropbox.com/s/wt1pcwi991xyhwr/cifar10bylabel.pt?dl=1'),
	('imgnet32bylabel.pt', 'https://www.dropbox.com/s/7r3hpugql8qgi9d/imgnet32bylabel.pt?dl=1'),
	('petsbylabel.pt', 'https://www.dropbox.com/s/mxh6qz3grhy7wcn/petsbylabel.pt?dl=1'),
	]:

	get_preprocessed_data(file_name, url)
