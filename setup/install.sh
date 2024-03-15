conda env update --file diffusionnag_env.yaml
pip install nas-bench-201==1.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch_geometric
pip install tensorflow==1.14.0
pip install pybnn
pip install ml_collections
pip install igraph
pip install gpytorch
pip install pandas
pip uninstall protobuf
pip install protobuf==3.19.0
pip install torchdiffeq
pip install wandb
pip install einops
pip install networkx
pip install matplotlib
pip install timm
pip install ofa==0.0.4-2007200808
pip install torchprofile
pip uninstall -y nvidia_cublas_cu11
