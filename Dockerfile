FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update && apt-get install -y git
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libgtk2.0-dev
