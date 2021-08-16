#!/usr/bin/env python3
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import platform
import re

FLAGS = None


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def dockerfile_common():
    df = '''
ARG BASE_IMAGE={}
ARG ONNXRUNTIME_VERSION={}
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
'''.format(FLAGS.triton_container, FLAGS.ort_version)

    if FLAGS.ort_openvino is not None:
        df += '''
ARG ONNXRUNTIME_OPENVINO_VERSION={}
'''.format(FLAGS.ort_openvino)

    df += '''
FROM ${BASE_IMAGE}
WORKDIR /workspace
'''
    return df


def dockerfile_for_linux(output_file):
    df = dockerfile_common()
    df += '''

# The Onnx Runtime dockerfile is the collection of steps in
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles

# Install dependencies from
# onnxruntime/dockerfiles/scripts/install_common_deps.sh. We don't run
# that script directly because we don't want cmake installed from that
# file.
RUN dnf groupinstall -y "Development Tools"
RUN dnf install -y \
        wget \
        zip \
        ca-certificates \
        cmake \
        curl \
        libcurl-devel \
        openssl-devel \
        python38-devel \
        python38-pip

# Is there a better way to get patchelf for ppc64le rpm?
RUN wget https://rpmfind.net/linux/epel/8/Everything/ppc64le/Packages/p/patchelf-0.12-1.el8.ppc64le.rpm && rpm -i patchelf-0.12-1.el8.ppc64le.rpm

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-ppc64le.sh \
         -O ~/miniconda.sh --no-check-certificate && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh && \
    /opt/miniconda/bin/conda clean -ya

# Allow configure to pick up cuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by base image)
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so
'''

    if FLAGS.ort_openvino is not None:
        df += '''
###
# OpenVINO is not available for ppc64le
###
'''

    df += '''
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO

RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
# Checkout onnx-tensorrt 8.0, because MYELIN was removed which is in CMakeList Version 7.2.1 which is checked out
RUN cd /workspace/onnxruntime/cmake/external/onnx-tensorrt && git checkout release/8.0
'''

    ep_flags = '--use_cuda'
    if FLAGS.cuda_version is not None:
        ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
    if FLAGS.cuda_home is not None:
        ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
    if FLAGS.cudnn_home is not None:
        ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
    if FLAGS.ort_tensorrt:
        ep_flags += ' --use_tensorrt'
        if FLAGS.tensorrt_home is not None:
            ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)
    if FLAGS.ort_openvino is not None:
        ep_flags += ' --use_openvino CPU_FP32'

    df += '''
WORKDIR /workspace/onnxruntime
ARG COMMON_BUILD_ARGS="--config Release --skip_submodule_sync --parallel --build_shared_lib --use_openmp --build_dir /workspace/build"
RUN ./build.sh ${{COMMON_BUILD_ARGS}} --update --build {}
'''.format(ep_flags)

    df += '''
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime

RUN mkdir -p /opt/onnxruntime && \
    cp /workspace/onnxruntime/LICENSE /opt/onnxruntime && \
    cat /workspace/onnxruntime/cmake/external/onnx/VERSION_NUMBER > /opt/onnxruntime/ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
RUN mkdir -p /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h \
       /opt/onnxruntime/include

RUN mkdir -p /opt/onnxruntime/lib && \
    cp /workspace/build/Release/libonnxruntime_providers_cuda.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/Release/libonnxruntime_providers_shared.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/Release/libonnxruntime.so.${ONNXRUNTIME_VERSION} \
       /opt/onnxruntime/lib && \
    (cd /opt/onnxruntime/lib && \
     ln -sf libonnxruntime.so.${ONNXRUNTIME_VERSION} libonnxruntime.so)

RUN mkdir -p /opt/onnxruntime/bin && \
    cp /workspace/build/Release/onnxruntime_perf_test \
       /opt/onnxruntime/bin && \
    cp /workspace/build/Release/onnx_test_runner \
       /opt/onnxruntime/bin && \
    (cd /opt/onnxruntime/bin && chmod a+x *)
'''

    if FLAGS.ort_tensorrt:
        df += '''
# TensorRT specific headers and libraries
RUN cp /workspace/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h \
       /opt/onnxruntime/include && \
    cp /workspace/build/Release/libonnxruntime_providers_tensorrt.so \
       /opt/onnxruntime/lib
'''

    if FLAGS.ort_openvino is not None:
        df += '''
###
# OpenVINO is not available for ppc64le
###
'''
    df += '''
RUN cd /opt/onnxruntime/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# For testing copy ONNX custom op library and model
RUN mkdir -p /opt/onnxruntime/test && \
    cp /workspace/build/Release/libcustom_op_library.so \
       /opt/onnxruntime/test && \
    cp /workspace/build/Release/testdata/custom_op_library/custom_op_test.onnx \
       /opt/onnxruntime/test
'''

    with open(output_file, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()
    df += '''
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
RUN git clone -b rel-%ONNXRUNTIME_VERSION% --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
'''

    ep_flags = '--use_cuda'
    if FLAGS.cuda_version is not None:
        ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
    if FLAGS.cuda_home is not None:
        ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
    if FLAGS.cudnn_home is not None:
        ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
    if FLAGS.ort_tensorrt:
        ep_flags += ' --use_tensorrt'
        if FLAGS.tensorrt_home is not None:
            ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)
    if FLAGS.ort_openvino is not None:
        ep_flags += ' --use_openvino CPU_FP32'

    df += '''
WORKDIR /workspace/onnxruntime
ARG VS_DEVCMD_BAT="\BuildTools\Common7\Tools\VsDevCmd.bat"
RUN powershell Set-Content 'build.bat' -value 'call %VS_DEVCMD_BAT%',(Get-Content 'build.bat')
RUN build.bat --cmake_generator "Visual Studio 16 2019" --config Release --skip_submodule_sync --build_shared_lib --use_openmp --update --build --build_dir /workspace/build {}
'''.format(ep_flags)

    df += '''
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime
RUN copy \\workspace\\onnxruntime\\LICENSE \\opt\\onnxruntime
RUN copy \\workspace\\onnxruntime\\cmake\\external\\onnx\\VERSION_NUMBER \\opt\\onnxruntime\\ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_c_api.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_session_options_config_keys.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\cpu\\cpu_provider_factory.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\cuda\\cuda_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_perf_test.exe \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnx_test_runner.exe \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.lib \\opt\\onnxruntime\\lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.lib \\opt\\onnxruntime\\lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.lib \\opt\\onnxruntime\\lib
'''

    if FLAGS.ort_tensorrt:
        df += '''
# TensorRT specific headers and libraries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\tensorrt\\tensorrt_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.dll \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.lib \\opt\\onnxruntime\\lib
'''
    with open(output_file, "w") as dfile:
        dfile.write(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--triton-container',
                        type=str,
                        required=True,
                        help='Triton base container to use for ORT build.')
    parser.add_argument('--ort-version',
                        type=str,
                        required=True,
                        help='ORT version.')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='File to write Dockerfile to.')
    parser.add_argument(
        '--target-platform',
        required=False,
        default=None,
        help=
        'Target for build, can be "ubuntu", "windows" or "jetpack". If not specified, build targets the current platform.'
    )

    parser.add_argument('--cuda-version',
                        type=str,
                        required=False,
                        help='Version for CUDA.')
    parser.add_argument('--cuda-home',
                        type=str,
                        required=False,
                        help='Home directory for CUDA.')
    parser.add_argument('--cudnn-home',
                        type=str,
                        required=False,
                        help='Home directory for CUDNN.')

    parser.add_argument(
        '--ort-openvino',
        type=str,
        required=False,
        help=
        'Enable OpenVino execution provider using specified OpenVINO version.')
    parser.add_argument('--ort-tensorrt',
                        action="store_true",
                        required=False,
                        help='Enable TensorRT execution provider.')
    parser.add_argument('--tensorrt-home',
                        type=str,
                        required=False,
                        help='Home directory for TensorRT.')

    FLAGS = parser.parse_args()

    if target_platform() == 'windows':
        # OpenVINO EP not yet supported for windows build
        if FLAGS.ort_openvino is not None:
            print("warning: OpenVINO not supported for windows, ignoring")
            FLAGS.ort_openvino = None

        # Default to CUDA based on CUDA_PATH envvar and TensorRT in
        # C:/tensorrt
        if 'CUDA_PATH'in os.environ:
            if FLAGS.cuda_home is None:
                FLAGS.cuda_home = os.environ['CUDA_PATH']
            elif FLAGS.cuda_home != os.environ['CUDA_PATH']:
                print("warning: --cuda-home does not match CUDA_PATH envvar")

        if FLAGS.cudnn_home is None:
            FLAGS.cudnn_home = FLAGS.cuda_home

        version = None
        m = re.match(r'.*v([1-9]?[0-9]+\.[0-9]+)$', FLAGS.cuda_home)
        if m:
            version = m.group(1)

        if FLAGS.cuda_version is None:
            FLAGS.cuda_version = version
        elif FLAGS.cuda_version != version:
            print("warning: --cuda-version does not match CUDA_PATH envvar")

        if (FLAGS.cuda_home is None) or (FLAGS.cuda_version is None):
            print("error: windows build requires --cuda-version and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = '/tensorrt'

        dockerfile_for_windows(FLAGS.output)

    else:
        if 'CUDNN_VERSION'in os.environ:
            version = None
            m = re.match(r'([0-9]\.[0-9])\.[0-9]\.[0-9]', os.environ['CUDNN_VERSION'])
            if m:
                version = m.group(1)
            if FLAGS.cudnn_home is None:
                FLAGS.cudnn_home = '/usr/local/cudnn-{}/cuda'.format(version)

        if FLAGS.cuda_home is None:
            FLAGS.cuda_home = '/usr/local/cuda'

        if FLAGS.cudnn_home is None:
            FLAGS.cudnn_home = '/usr/lib64'
        
        if (FLAGS.cuda_home is None) or (FLAGS.cudnn_home is None):
            print("error: linux build requires --cudnn-home and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = '/usr/src/tensorrt'

        dockerfile_for_linux(FLAGS.output)
