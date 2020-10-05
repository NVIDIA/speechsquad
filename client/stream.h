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

#pragma once

#include <glog/logging.h>
#include <nvrpc/client/client_streaming_v3.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

namespace speech_squad {

class Stream
    : public nvrpc::client::v3::ClientStreaming<SpeechSquadInferRequest,
                                                SpeechSquadInferResponse> {
  using Client = nvrpc::client::v3::ClientStreaming<SpeechSquadInferRequest,
                                                    SpeechSquadInferResponse>;

public:
  using PrepareFn = typename Client::PrepareFn;
  using ReceiveResponseFn =
      std::function<void(SpeechSquadInferResponse &&response)>;
  using CompleteFn = std::function<void(const ::grpc::Status &status)>;

  Stream(PrepareFn prepare_fn,
         std::shared_ptr<nvrpc::client::Executor> executor,
         ReceiveResponseFn OnReceive, CompleteFn OnComplete)
      : Client(prepare_fn, executor), OnReceive_(OnReceive),
        OnComplete_(OnComplete), m_sent_count(0), m_recv_count(0) {}
  ~Stream() override {}

  void CallbackOnRequestSent(SpeechSquadInferRequest &&request) override {
    m_sent_count++;
  }

  void
  CallbackOnResponseReceived(SpeechSquadInferResponse &&response) override {
    m_recv_count++;
    OnReceive_(std::move(response));
  }

  void CallbackOnComplete(const ::grpc::Status &status) override {
    OnComplete_(status);
  }

private:
  ReceiveResponseFn OnReceive_;
  CompleteFn OnComplete_;

  std::size_t m_sent_count;
  std::size_t m_recv_count;
};

} // namespace speech_squad
