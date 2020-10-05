
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

#include "speech_squad_client.h"

#include <glog/logging.h>
#include <trtlab/core/affinity.h>
#include <trtlab/core/thread_pool.h>

namespace speech_squad {

inline int random_range(int upper_bound) {
  int divisor = RAND_MAX / upper_bound;
  int value;

  do {
    value = rand() / divisor;
  } while (value == upper_bound);

  return value;
}

SpeechSquadClient::SpeechSquadClient(
    std::vector<std::shared_ptr<grpc::Channel>> &channels,
    int32_t num_parallel_requests, const size_t num_iterations,
    const std::string &language_code, bool print_results,
    int32_t chunk_duration_ms, const int executor_count,
    const OutputFilenames &output_files,
    std::shared_ptr<speech_squad::SquadEvalDataset> &squad_eval_dataset,
    std::string &squad_questions_json, int32_t num_iteration,
    uint64_t offset_duration, int proc_index, int proc_count,
    bool true_concurrency)
    : num_parallel_requests_(num_parallel_requests),
      print_results_(print_results), chunk_duration_ms_(chunk_duration_ms),
      squad_eval_dataset_(squad_eval_dataset),
      squad_questions_json_(squad_questions_json),
      num_iterations_(num_iterations), language_code_(language_code),
      offset_duration_(offset_duration), failed_tasks_count_(0),
      proc_index_(proc_index), proc_count_(proc_count), proc_error_(0),
      true_concurrency_(true_concurrency) {
  stubs_.reserve(channels.size());
  for (const auto &channel : channels) {
    stubs_.push_back(SpeechSquadService::NewStub(channel));
  }

  const auto processor_count = std::thread::hardware_concurrency();
  if (executor_count == 0) {
    if (processor_count == 0) {
      LOG(ERROR) << "Failed to retrieve the processor count on the machine. "
                    "Please prove the executor count explicitly.";
    }
    ::trtlab::cpu_set cpus;
    for (int i = 0; i < processor_count; i++) {
      cpus.insert(::trtlab::affinity::system::cpu_from_logical_id(i));
    }
    std::unique_ptr<::trtlab::ThreadPool> thread_pool(
        new ::trtlab::ThreadPool(cpus));
    VLOG(1) << "Created an executor thread pool of size "
            << thread_pool->Size();
    executor_ =
        std::make_shared<nvrpc::client::Executor>(std::move(thread_pool));
  } else {
    executor_ = std::make_shared<nvrpc::client::Executor>(executor_count);
  }
  output_filestreams_ =
      std::make_shared<OutputFilestreams>(output_files.root_folder_);

  if (print_results_) {
    output_filestreams_->answer_file_.open(
        GetFullpath(output_files.root_folder_, output_files.answer_json_));
    output_filestreams_->answer_file_ << "{";
    output_filestreams_->question_file_.open(
        GetFullpath(output_files.root_folder_, output_files.question_json_));
    output_filestreams_->wave_file_.open(
        GetFullpath(output_files.root_folder_, output_files.wave_json_));
  }
}

SpeechSquadClient::~SpeechSquadClient() {
  if (print_results_) {
    output_filestreams_->wave_file_.close();
    output_filestreams_->answer_file_ << "}";
    output_filestreams_->answer_file_.close();
    output_filestreams_->question_file_.close();
  }
}

int SpeechSquadClient::Run() {
  sending_complete_ = false;
  failed_tasks_count_ = 0;

  std::vector<std::shared_ptr<AudioData>> all_wav;
  LoadAudioData(all_wav, squad_questions_json_, "id", proc_index_, proc_count_);

  if (all_wav.size() == 0) {
    if (proc_count_ > 0) {
      proc_error_ = 1;
    } else {
      std::cout << "Exiting..." << std::endl;
      return 1;
    }
  }

  if (proc_count_ > 0) {
    MPI_CHECK(MPI_Allreduce(&proc_error_, &proc_error_, 1, MPI_INT, MPI_SUM,
                            MPI_COMM_WORLD));
    if (proc_error_ > 0) {
      std::cout << "Provide minimum of " << proc_count_
                << " many questions. Exiting Process " << proc_index_ << "..."
                << std::endl;
      return 1;
    }
  }

  uint32_t all_wav_max = all_wav.size() * num_iterations_;
  response_latencies_.clear();
  component_timings_.clear();

  std::vector<std::string> components;
  GetComponents(&components);
  for (const auto &component : components) {
    component_timings_[component].reserve(all_wav_max);
    average_latency_ms_[component] = 0;
  }
  average_latency_ms_["Client Latency"] = 0;
  response_latencies_.reserve(all_wav_max);

  std::vector<std::unique_ptr<AudioTask>> curr_tasks, next_tasks;
  curr_tasks.reserve(num_parallel_requests_);
  next_tasks.reserve(num_parallel_requests_);

  SyncQueue<std::unique_ptr<AudioTask>> task_queue;
  // Starts a reaper thread. It will sequentially visit all the
  // tasks.
  reaper_thread_ =
      std::thread([this, &task_queue]() { ReaperFunction(task_queue); });

  std::vector<std::shared_ptr<AudioData>> all_wav_repeated;
  all_wav_repeated.reserve(all_wav_max);
  for (uint32_t file_id = 0; file_id < all_wav.size(); file_id++) {
    for (int iter = 0; iter < num_iterations_; iter++) {
      all_wav_repeated.push_back(all_wav[file_id]);
    }
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (proc_index_ == 0) {
    std::cout << "Generating load..." << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  uint32_t all_wav_i = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  while (true) {
    int offset_index = (all_wav_i == 0) ? proc_index_ : 0;
    auto now = std::chrono::high_resolution_clock::now();
    while ((curr_tasks.size() < num_parallel_requests_) &&
           (all_wav_i < all_wav_max)) {
      DVLOG(2) << "Adding a new task with id: " << all_wav_i;
      auto scheduled_time =
          now + std::chrono::microseconds((offset_index++) * offset_duration_);
      auto prepare_fn = [this](::grpc::ClientContext * context,
                               ::grpc::CompletionQueue * cq) -> auto {
        auto stub = GetStub();
        return std::move(stub->PrepareAsyncSpeechSquadInfer(context, cq));
      };
      std::unique_ptr<AudioTask> ptr(new AudioTask(
          all_wav_repeated[all_wav_i], all_wav_i, prepare_fn, language_code_,
          chunk_duration_ms_, print_results_, squad_eval_dataset_,
          output_filestreams_, executor_, scheduled_time));
      curr_tasks.emplace_back(std::move(ptr));
      ++all_wav_i;
    }

    // If still empty, done
    if (curr_tasks.empty()) {
      break;
    }

    for (size_t itask = 0; itask < curr_tasks.size(); ++itask) {
      AudioTask &task = *(curr_tasks[itask]);

      now = std::chrono::high_resolution_clock::now();
      if (now < task.NextTimePoint()) {
        next_tasks.push_back(std::move(curr_tasks[itask]));
        continue;
      }
      if ((task.GetState() == AudioTask::SENDING) ||
          (task.GetState() == AudioTask::START)) {
        auto status = task.Step();
        if (!status.IsOk()) {
          WaitForReaper();
          std::cerr << "Failed to generate specified load. Error details: "
                    << status.AsString() << std::endl;
          return -1;
        }
      }

      if ((!true_concurrency_) &&
          task.GetState() == AudioTask::SENDING_COMPLETE) {
        task_queue.Put(std::move(curr_tasks[itask]));
      } else if ((true_concurrency_) &&
                 task.GetState() == AudioTask::RECEIVING_COMPLETE) {
        task_queue.Put(std::move(curr_tasks[itask]));
      } else {
        next_tasks.push_back(std::move(curr_tasks[itask]));
      }
    }

    curr_tasks.swap(next_tasks);
    next_tasks.clear();
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (proc_index_ == 0) {
    std::cout << "Waiting for all responses..." << std::endl;
  }
  WaitForReaper();

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  auto current_time = std::chrono::high_resolution_clock::now();

  if (proc_index_ == 0) {
    std::cout << std::endl << "Done with measurements" << std::endl;
    std::cout << "Generating Statistics Report..." << std::endl;
  }

  for (int i = 0; i < std::max(proc_count_, 1); i++) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    usleep(1000000);
    if (i == proc_index_) {
      std::cout << "\t\t================ Process " << proc_index_;
      std::cout << "================" << std::endl;
      PrintStats();
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  int success_proc_count = 0;
  if (proc_count_ > 1) {
    if (proc_index_ == 0) {
      MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &total_audio_processed_, 1, MPI_DOUBLE,
                           MPI_SUM, 0, MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Reduce(&total_audio_processed_, &total_audio_processed_, 1,
                           MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    }

    for (auto &it : average_latency_ms_) {
      if (proc_index_ == 0) {
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &it.second, 1, MPI_DOUBLE, MPI_SUM,
                             0, MPI_COMM_WORLD));
      } else {
        MPI_CHECK(MPI_Reduce(&it.second, &it.second, 1, MPI_DOUBLE, MPI_SUM, 0,
                             MPI_COMM_WORLD));
      }
    }

    if (proc_index_ == 0) {
      MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &failed_tasks_count_, 1, MPI_INT,
                           MPI_SUM, 0, MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Reduce(&failed_tasks_count_, &failed_tasks_count_, 1,
                           MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    }

    success_proc_count = (average_latency_ms_["Client Latency"] == 0) ? 0 : 1;
    if (proc_index_ == 0) {
      MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &success_proc_count, 1, MPI_INT,
                           MPI_SUM, 0, MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Reduce(&success_proc_count, &success_proc_count, 1, MPI_INT,
                           MPI_SUM, 0, MPI_COMM_WORLD));
    }
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (proc_index_ == 0) {
    double diff_time =
        std::chrono::duration<double, std::milli>(current_time - start_time)
            .count();

    std::cout << "\t\t================ Final Report ";
    std::cout << "================" << std::endl;
    std::cout << "Run time: " << diff_time / 1000. << " sec." << std::endl;
    std::cout << "Total audio processed: " << total_audio_processed_ << " sec."
              << std::endl;
    std::cout << "Throughput: " << total_audio_processed_ * 1000. / diff_time
              << " RTFX" << std::endl;
    std::cout << "Number of failed audio clips: " << failed_tasks_count_
              << std::endl;
    std::cout << "Average Latencies ====> " << std::endl;
    for (const auto it : average_latency_ms_) {
      std::cout << "\t" << it.first << ":"
                << it.second / std::max(1, success_proc_count) << " ms"
                << std::endl;
    }
  }

  return 0;
}

void SpeechSquadClient::WaitForReaper() {
  sending_complete_ = true;
  if (reaper_thread_.joinable()) {
    reaper_thread_.join();
  }
}

void SpeechSquadClient::ReaperFunction(
    SyncQueue<std::unique_ptr<AudioTask>> &task_queue) {
  while ((!sending_complete_) || (!task_queue.Empty())) {
    while (!task_queue.Empty()) {
      bool failed = false;
      auto awaited_task = std::move(task_queue.Get());
      auto grpc_status = awaited_task->WaitForCompletion();
      if (!grpc_status.IsOk()) {
        failed = true;
        failed_tasks_count_++;
      }
      total_audio_processed_ += awaited_task->AudioProcessed();
      auto task_status = awaited_task->GetStatus();
      if ((!failed) && (!task_status.IsOk())) {
        failed = true;
        failed_tasks_count_++;
      } else if (failed) {
        task_status = grpc_status;
      }

      DVLOG(2) << "Completed task with id: " << awaited_task->GetId()
               << ", Status: " << task_status.AsString();

      if (!failed) {
        auto this_result = awaited_task->GetResult();
        // WAR to capture the results only for the audio tasks that
        // received audio content
        if (!this_result->first_response) {
          response_latencies_.push_back(this_result->response_latency);
          for (const auto &itr : this_result->component_timings) {
            component_timings_[itr.first].push_back(itr.second);
          }
        }
      }
    }
    usleep(1000);
  }
}

std::shared_ptr<SpeechSquadService::Stub> SpeechSquadClient::GetStub() {
  /*
  description of the load-balancer implementaion from enovy - N = 2

  all weights equal: An O(1) algorithm which selects N random available hosts as
  specified in the configuration (2 by default) and picks the host which has the
  fewest active requests (Mitzenmacher et al. has shown that this approach is
  nearly as good as an O(N) full scan). This is also known as P2C (power of two
  choices). The P2C load balancer has the property that a host with the highest
  number of active requests in the cluster will never receive new requests. It
  will be allowed to drain until it is less than or equal to all of the other
  hosts.

  because the clients hold a shared_ptr of the stubs, we can use the stubs
  use_count as a proxy for number of active streams on a given channel
  */

  if (stubs_.size() == 1) {
    return stubs_[0];
  }

  auto n = stubs_.size();
  auto r1 = random_range(n);
  auto r2 = random_range(n);

  if (stubs_[r1].use_count() < stubs_[r2].use_count()) {
    return stubs_[r1];
  }
  return stubs_[r2];
}

void SpeechSquadClient::PrintStats() {
  for (const auto &itr : component_timings_) {
    PrintLatencies(itr.second, itr.first);
  }
  PrintLatencies(response_latencies_, "Client Latency");
}

void SpeechSquadClient::PrintLatencies(const std::vector<double> &raw_latencies,
                                       const std::string &name) {
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::vector<double> latencies = raw_latencies;

  if (latencies.size() > 0) {
    std::sort(latencies.begin(), latencies.end());
    double nresultsf = static_cast<double>(latencies.size());
    size_t per50i = static_cast<size_t>(std::floor(50. * nresultsf / 100.));
    size_t per90i = static_cast<size_t>(std::floor(90. * nresultsf / 100.));
    size_t per95i = static_cast<size_t>(std::floor(95. * nresultsf / 100.));
    size_t per99i = static_cast<size_t>(std::floor(99. * nresultsf / 100.));

    double median = latencies[per50i];
    double lat_90 = latencies[per90i];
    double lat_95 = latencies[per95i];
    double lat_99 = latencies[per99i];

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                 latencies.size();

    std::cout << std::setprecision(5);
    std::cout << " " << name << " (ms):\n";
    std::cout << "\t\tMedian\t\t90th\t\t95th\t\t99th\t\tAvg\n";
    std::cout << "\t\t" << median << "\t\t" << lat_90 << "\t\t" << lat_95
              << "\t\t" << lat_99 << "\t\t" << avg << std::endl;

    average_latency_ms_[name] = avg;
  }
}

} // namespace speech_squad
