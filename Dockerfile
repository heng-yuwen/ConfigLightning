FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list

RUN pip3 install --no-cache torch==1.12.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install --no-cache lightning pytorch-lightning[extra] timm segmentation-models-pytorch
