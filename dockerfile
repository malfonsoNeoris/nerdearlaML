ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.11-py3
#ARG python:3.7-buster
FROM ${BASE_IMAGE}
LABEL Manuel Alfonso <manuel.alfonso@neoris.com>


# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential ffmpeg libsm6 libxext6

# check our python environment
RUN python3 --version
RUN pip3 --version


#RUN pip install -Uqq fastbook
COPY requirements.txt /usr/src2/requirements.txt
# set the working directory for containers
WORKDIR  /usr/src2/

# Installing python dependencies
RUN pip install -r requirements.txt 
#RUN conda install --file requirements.txt
#RUN pip install --upgrade numpy
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
#RUN conda install -c conda-forge detectron2

# # THIS IS TO FIX OPENCV...check later
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y

#COPY CODE TO RUN!
COPY . /usr/src/
#COPY MODELS
#COPY models /usr/src/models/

# Running Python Application
WORKDIR  /usr/src/

EXPOSE 8501

CMD ["streamlit","run", "streamlit_app.py"]

#nvidia-docker run --ipc=host -p 8501:8501 -it -d --name streamlit_app streamlit_app