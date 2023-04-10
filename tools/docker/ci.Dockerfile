# syntax=docker/dockerfile:1.2

# sed -i '$aRUN if [ "${$HKV_DEV_MODE}" == "true" ]; then  rm -rf /usr/local/hugectr /usr/local/hps_trt && apt update -y --fix-missing && apt install -y gdb; fi' ci.Dockerfile
# docker build --no-cache --build-arg '$HKV_DEV_MODE=true' --build-arg 'MERLIN_VERSION=main' --build-arg 'TRITON_VERSION=22.12' --build-arg 'TENSORFLOW_VERSION=22.12' --build-arg 'TORCH_VERSION=22.12' --build-arg 'BASE_IMAGE=gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:merlin_base_23.02' -t gitlab-master.nvidia.com:5005/dl/hugectr/hugectr/hkv:devel_all -f ci.Dockerfile .

ARG MERLIN_VERSION=22.12
ARG TRITON_VERSION=22.11

ARG BASE_IMAGE=nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}

FROM ${BASE_IMAGE} as base

ARG HUGECTR_VER=main
ARG HUGECTR_BACKEND_VER=main

# Envs
ENV CUDA_SHORT_VERSION=11.6
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin
ENV PATH=$PATH:/usr/lib/x86_64-linux-gnu/
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY install/install_bazel.sh /install/
RUN /install/install_bazel.sh "5.3.1"

# Install CUDA-Aware hwloc
ARG HWLOC_VER=2.4.1

RUN cd /opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201 && rm -rfv hwloc201.h hwloc/include/hwloc.h
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/hwloc/v2.4/hwloc-${HWLOC_VER}.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/hwloc-${HWLOC_VER}.tar.gz -C /var/tmp && \
    cd /var/tmp/hwloc-${HWLOC_VER} && \
    ./configure CPPFLAGS="-I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/" LDFLAGS="-L/usr/local/cuda/lib64" --enable-cuda && \
    make -j$(nproc) && make install && \
    rm -rf /var/tmp/hwloc-${HWLOC_VER} /var/tmp/hwloc-${HWLOC_VER}.tar.gz


# Arguments "_XXXX" are only valid when $HKV_DEV_MODE==false

ENV OMPI_MCA_plm_rsh_agent=ssh
ENV OMPI_MCA_opal_cuda_support=true

ENV NCCL_LAUNCH_MODE=PARALLEL
ENV NCCL_COLLNET_ENABLE=0

ENV SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
ENV SHARP_COLL_LOCK_ON_COMM_INIT=1
ENV SHARP_COLL_LOG_LEVEL=3
ENV HCOLL_ENABLE_MCAST=0

# link sub modules expected by hkv cmake
RUN ln -s /usr/lib/libcudf.so /usr/lib/libcudf_base.so
RUN ln -s /usr/lib/libcudf.so /usr/lib/libcudf_io.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN rm -rf /usr/lib/x86_64-linux-gnu/libibverbs.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.36.0 /usr/lib/x86_64-linux-gnu/libibverbs.so

# Remove fake lib
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# Clean up
RUN rm -rf /usr/local/share/jupyter/lab/staging/node_modules/marked
RUN rm -rf /usr/local/share/jupyter/lab/staging/node_modules/node-fetch