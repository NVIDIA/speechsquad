/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RIVA_SRC_SERVICES_GRPC_RIVA_ASR_H_
#define RIVA_SRC_SERVICES_GRPC_RIVA_ASR_H_

#include "proto/riva_asr.grpc.pb.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using grpc::StatusCode;
using namespace std::chrono_literals;

namespace nj = nvidia::riva;
namespace nj_asr = nvidia::riva::asr;

class KaldiASRContext;

class ASRServiceImpl final : public nj_asr::RivaASR::Service {
  // Logic and data behind the server's behavior.
public:
  explicit ASRServiceImpl(const std::string& kaldi_path, const std::string& kaldi_options);
  ~ASRServiceImpl();
  Status Recognize(ServerContext *context,
                   const nj_asr::RecognizeRequest *request,
                   nj_asr::RecognizeResponse *response) override;

  Status StreamingRecognize(
      ServerContext *context,
      ServerReaderWriter<nj_asr::StreamingRecognizeResponse,
                         nj_asr::StreamingRecognizeRequest> *stream) override;

private:
  std::mutex mu_;

  bool use_kaldi_cpu_asr_;
  // optional Kaldi CPU ASR engine
  KaldiASRContext *kaldi_cpu_asr_;

  inline static const std::chrono::duration kTimeoutTime_ = 100s;
};

#endif
