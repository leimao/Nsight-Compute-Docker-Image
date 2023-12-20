FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libgl1-mesa-glx \
        libxkbfile-dev \
        openssh-client \
        wget \
        xcb \
        xkb-data && \
    apt-get clean

# QT6 is required for the Nsight Compute UI.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        qt6-base-dev && \
    apt-get clean

# Nsight Compute is part of the CUDA Toolkit.

# CMD ["ncu-ui"]
