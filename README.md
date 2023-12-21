# Nsight Compute Docker Image

## Introduction

This is a portable Nsight Compute Docker image which allows the user to profile executables anywhere using the Nsight Compute inside the Docker container.

## Usages

### Build Docker Image

To build the Docker image, please run the following command.

```bash
$ docker build -f nsight-compute.Dockerfile --no-cache --tag=nsight-compute:12.0.1 .
```

### Run Docker Container

To run the Docker container, please run the following command.

```bash
$ xhost +
$ docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -v $(pwd):/mnt --network=host nsight-compute:12.0.1
$ xhost -
```

### Build Examples

```bash
$ cd /mnt/examples
$ nvcc gemm_naive.cu -o gemm_naive
```

### Run Nsight Compute

```bash
$ ncu --set full -f -o gemm_naive gemm_naive
$ ncu-ui
```
