FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list

WORKDIR /root

RUN pip3 install --no-cache torch==1.8.1+cu111 torchvision==0.9.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install --no-cache lightning pytorch-lightning[extra] timm segmentation-models-pytorch
RUN pip3 uninstall torchmetrics && pip3 install git+https://ghproxy.com/https://github.com/Lightning-AI/metrics.git
RUN pip3 install --no-cache mmcv

RUN echo "export PYTHONPATH=${PYTHONPATH}:/root/customise_pl:/root/datasets:/root/experiments:/root/models" > /root/.bahsrc

# modify the evaluation_loop, line 340 to support Chinese character.
# matrics_encoded = [metric.encode("gbk") for metric in metrics_strs]
# max_length = int(min(max(len(max(matrics_encoded, key=len)), len(max(headers, key=len)), 25), term_size / 2))