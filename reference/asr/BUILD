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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cpu_asr_grpc",
    srcs = ["cpu_asr_grpc.cc", "kaldi_asr.cc", "wav_reader.cc"],
    hdrs = ["cpu_asr_grpc.h", "kaldi_asr.h", "wav_reader.h"],
    copts = ["-O0", "-g"],
    deps = [
        "//proto:jarvis_grpc_asr",
        "@glog//:glog",
        "@com_github_gflags_gflags//:gflags",
        "@kaldi//:kaldi",
        "@kaldi//:fst"
    ]
)

cc_binary(
    name = "kaldi_cpu_asr_server",
    srcs = ["server.cc"],
    deps = [
        "cpu_asr_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_gflags_gflags//:gflags",
        "@glog//:glog",
    ],
)

