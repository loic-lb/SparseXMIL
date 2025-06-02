FROM continuumio/miniconda3:24.1.2-0

SHELL ["/bin/bash", "--login", "-c"]

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update && \
    apt-get install -y git cmake build-essential &&\
    apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN conda create -n sparsexmil python=3.8 && echo "conda activate sparsexmil" >> ~/.bashrc

RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
RUN conda install openblas-devel==0.3.2 -c anaconda -y
RUN conda install mkl==2021.4.0 -c conda-forge -y
RUN conda install cudatoolkit-dev==11.3.1 -c conda-forge -y

RUN conda install openslide==3.4.1 -c conda-forge -y

RUN git clone --recursive https://github.com/shwoo93/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --blas=openblas --force_cuda

RUN pip install scikit-image==0.21.0 scikit-learn==1.2.2 pandas==2.0.2 h5py==3.8.0 \
        opencv-python-headless==4.7.0.72 openslide-python==1.2.0 matplotlib==3.7.1 nystrom_attention==0.0.11
RUN pip install torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
RUN pip install torch-geometric==2.4.0

WORKDIR /SparseXMIL

