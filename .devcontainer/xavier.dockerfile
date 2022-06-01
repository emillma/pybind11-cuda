
# docker build -t devcontainer:latest -f .\.devcontainer\Dockerfile .
FROM nvcr.io/nvidia/l4t-base:r32.6.1

# install basic stuff
RUN apt-get update && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils curl git cmake sl sudo net-tools nmap 



# install python stuff
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-dev
# set py 3.8 to default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --config python3
# install pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip 
# to fix annoying pip xserver bug (https://github.com/pypa/pip/issues/8485)
RUN printf "%s\n" "alias pip3='DISPLAY= pip3'" "alias python=python3" > ~/.bash_aliases
# install packages

# install arena_sdk
# remember to apt install file to let the python lib understand that the so is a symbolic link to the ARM linked library
RUN apt-get install file
ARG arena_skd_file=ArenaSDK_v0.1.43_Linux_ARM64.tar.gz
ARG arena_whl_file=arena_api-2.1.4-py3-none-any.whl
COPY .devcontainer/files/${arena_skd_file} /home/arena/ArenaSDK_Linux.tar.gz
COPY .devcontainer/files/${arena_whl_file} /home/arena/${arena_whl_file}
RUN cd /home/arena \
    && tar -xvzf ArenaSDK_Linux.tar.gz \
    && cd ArenaSDK_Linux_ARM64 && sh Arena_SDK_ARM64.conf \
    && pip3 install ../${arena_whl_file} 

RUN pip3 install --upgrade pip && pip3 install \
    pylint numpy pandas plotly dash autopep8 dash_bootstrap_components opencv-contrib-python spidev Jetson.GPIO pylint numba pyubx2

COPY .gitconfig /home/
