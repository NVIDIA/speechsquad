# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.9)
project (speechsquad)

include(CMakeDependentOption)
include(ExternalProject)
include(GNUInstallDirs)


#
# Build trtlab
#
# FIXME: Use the official repository once build is fixed
ExternalProject_Add(trtlab
  GIT_REPOSITORY "https://github.com/tanmayv25/tensorrt-laboratory.git"
  GIT_TAG "tanmayv-speech-squad"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorrt-laboratory"
  CMAKE_CACHE_ARGS
    -DBUILD_TENSORRT:BOOL=OFF
    -DBUILD_CUDA:BOOL=OFF
)

# set cmake search paths to pick up installed .cmake files
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_BINARY_DIR}/trtlab-prefix/src/trtlab-build/local/lib/cmake/")
list(INSERT CMAKE_PREFIX_PATH 0 "${CMAKE_CURRENT_BINARY_DIR}/trtlab-prefix/src/trtlab-build/local/lib/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/tensorrt-laboratory/cmake")

ExternalProject_Add(client
  PREFIX client
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/client"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
    -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
  DEPENDS trtlab
)
