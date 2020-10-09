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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "cpu_asr_grpc.h"

using grpc::Server;
using grpc::ServerBuilder;

DEFINE_string(uri, "0.0.0.0:50051", "URI this server should bind to");
DEFINE_string(model_path, "/data/models/LibriSpeech", "Path to trained Kaldi model");
DEFINE_string(model_options, "", "Kaldi model options");

void RunServer() {
  ServerBuilder builder;
  std::unique_ptr<ASRServiceImpl> asr_service;

  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(FLAGS_uri, grpc::InsecureServerCredentials());

  asr_service = std::make_unique<ASRServiceImpl>(FLAGS_model_path, FLAGS_model_options);
  builder.RegisterService(asr_service.get());

  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());

  LOG(INFO) << "Kaldi ASR Server listening on " << FLAGS_uri;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  gflags::SetUsageMessage("kaldi-asr usage");
  gflags::SetVersionString("0.0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  RunServer();

  gflags::ShutDownCommandLineFlags();

  return 0;
}
