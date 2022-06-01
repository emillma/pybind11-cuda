
# docker build -t devcontainer:latest -f .\.devcontainer\Dockerfile .
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# install basic stuff
RUN apt update && apt -y upgrade 

# install lates cmake
RUN apt -y install curl git wget libssl-dev cmake
RUN cd ~ \
    && mkdir cmake_tmp \
    && cd cmake_tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2.tar.gz \
    && tar -xvzf cmake-3.23.2.tar.gz \
    && cd cmake-3.23.2 \
    && ./configure \
    && make \
    && make install

# RUN apt install -y cmake gcc-arm-none-eabi libnewlib-arm-none-eabi build-essential 

# # install python stuff
RUN apt update && apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.10 python3.10-distutils python3.10-dev python3.10-dbg

# # set py 3.9 to default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --config python3

# # install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# # to fix annoying pip xserver bug (https://github.com/pypa/pip/issues/8485)
RUN printf "%s\n" "alias pip3='DISPLAY= pip3'" "alias python=python3" > ~/.bash_aliases

# # install packages
RUN pip3 install --upgrade pip
RUN pip install \
    numpy scipy matplotlib pyqt5 pandas\
    pylint autopep8 jupyter \
    sympy 

RUN pip3 install \
    pyserial \
    plotly dash\
    numba \
    opencv-python opencv-contrib-python \
    networkx \
    dash_bootstrap_components 

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN git config --global user.email "emil.martens@gmail.com" && git config --global user.name "Emil Martens"
