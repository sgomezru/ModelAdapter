FROM nvcr.io/nvidia/pytorch:24.03-py3

# ARG user
# ARG uid
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video,utility,compute

ENV DEBIAN_FRONTEND noninteractive

# Before any libraries are installed, make sure everyone knows about cuda-aware ompi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

### update
RUN apt-get update && upgrade

### OPENGL
# RUN apt-get update && apt-get install -y \
# 	libassimp-dev \ 
# 	libjpeg-dev \
# 	libpng-dev \
# 	libglu1-mesa-dev \
# 	freeglut3-dev \
# 	mesa-common-dev \
# 	libgl1-mesa-dev \
# 	libegl1-mesa-dev

# RUN apt-get update && apt-get install -y -qq --no-install-recommends \
#     libglvnd0 \
#     libglx0 \
#     libxext6 \
#     libx11-6 \
#     libglfw3 \
#     libglfw3-dev \
#     libpcl-dev \
#     libfreetype6-dev

### utils
RUN apt-get install -y \
	htop \
	git \
	vim \
	binutils \
	tmux

### - PIP
RUN pip install \ 
	wandb \
	deep-pipe \
	scikit-learn \
	torchmetrics \
	tqdm \
	omegaconf \
	seaborn \
	monai \
	hydra-core


### - GIT
## setup
# move nvidia-stuff to separate folder
WORKDIR /workspace
RUN mv * nvidia | mkdir nvidia
## surface distances
WORKDIR /workspace/repos
RUN git clone https://github.com/deepmind/surface-distance.git
RUN pip install surface-distance/
## nnUNet
RUN git clone --branch nnunetv1 https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /workspace/repos/nnUNet
RUN pip install -e .
# remove annoying print statements from init
RUN sed -i '2,6d' nnunet/__init__.py
# set up variables for nnUnet
WORKDIR /
RUN mkdir -p /data/nnUNet_raw
ENV nnUNet_raw_data_base="/data/nnUNet_raw"
RUN mkdir -p /data/nnUNet_preprocessed
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
RUN mkdir -p /results/nnUNet_results
ENV RESULTS_FOLDER="/results/nnUNet_results"

### - CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*

### - Make user and give rights for workdir and folders we might need access to.
#RUN mkdir /out
#RUN useradd -u ${uid} -ms /bin/bash ${user}
#RUN chown -R ${user} /workspace /out /data /results

### - expose ports for various services
# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8889
### - start interactive console on attach
ENV DEBIAN_FRONTEND teletype 
