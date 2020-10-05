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

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

#include "audio_task.h"
#include "status.h"
#include "sync_queue.h"

#include "nvrpc/client/executor.h"

namespace speech_squad {

class SpeechSquadClient {
public:
  SpeechSquadClient(
      std::vector<std::shared_ptr<grpc::Channel>> &channels,
      int32_t num_parallel_requests, const size_t num_iterations,
      const std::string &language_code, bool print_transcripts,
      int32_t chunk_duration_ms, const int executor_count,
      const OutputFilenames &output_files,
      std::shared_ptr<speech_squad::SquadEvalDataset> &squad_eval_dataset,
      std::string &squad_questions_json, int32_t num_iteration,
      uint64_t offset_duration, int proc_index, int proc_count,
      bool true_concurrency);

  ~SpeechSquadClient();

  // This function is not thread-safe
  int Run();

private:
  double TotalAudioProcessed() { return total_audio_processed_; }
  void WaitForReaper();
  void ReaperFunction(SyncQueue<std::unique_ptr<AudioTask>> &task_queue);
  void PrintStats();
  void PrintLatencies(const std::vector<double> &raw_latencies,
                      const std::string &name);
  std::shared_ptr<SpeechSquadService::Stub> GetStub();

  std::vector<std::shared_ptr<SpeechSquadService::Stub>> stubs_;
  int num_parallel_requests_;
  bool print_results_;
  double chunk_duration_ms_;

  std::shared_ptr<speech_squad::SquadEvalDataset> squad_eval_dataset_;
  std::string squad_questions_json_;
  int32_t num_iterations_;
  std::string language_code_;
  uint64_t offset_duration_;
  int proc_index_;
  int proc_count_;
  int proc_error_;
  bool true_concurrency_;

  // std::vector<Stream::PrepareFn> infer_prepare_fns_;
  std::shared_ptr<nvrpc::client::Executor> executor_;
  std::shared_ptr<OutputFilestreams> output_filestreams_;

  std::vector<double> response_latencies_;
  std::map<std::string, std::vector<double>> component_timings_;
  double total_audio_processed_;
  std::map<std::string, double> average_latency_ms_;
  bool sending_complete_;

  std::thread reaper_thread_;
  size_t failed_tasks_count_;
};

} // namespace speech_squad
