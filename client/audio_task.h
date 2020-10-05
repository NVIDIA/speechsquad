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

#include "status.h"
#include "stream.h"
#include "utils.h"

namespace speech_squad {

inline std::string GetFullpath(const std::string &directory,
                               const std::string &filename) {
  return std::string(directory + "/" + filename);
}
struct Results {
  Results()
      : audio_content(nullptr), audio_offset(0), response_latency(0.),
        first_response(true) {}

  ~Results() {
    if (audio_content != nullptr) {
      free(audio_content);
    }
  }
  /*** SpeechSquadResponseMeta : Start ***
  // mandatory
  string squad_question = 1;
  string squad_answer = 2;

  // optional
  float squad_confidence = 10;
  string asr_transcription = 11;
  string asr_confidence = 12;
  map<string, float> component_timing = 13;
  *** SpeechSquadResponseMeta : End ***/
  std::string squad_question;
  std::string squad_answer;

  char *audio_content;
  size_t audio_offset;

  // Record of statistics
  // The latency between sending the last response and receipt of first response
  // in milliseconds.
  double response_latency;
  std::chrono::time_point<std::chrono::high_resolution_clock>
      last_response_timestamp;
  // Records the time interval between successive responses in milliseconds.
  std::vector<double> response_intervals;

  std::map<std::string, double> component_timings;

  bool first_response;
  std::mutex mtx;
};

class OutputFilestreams {
public:
  OutputFilestreams(const std::string &root_directory)
      : root_directory_(root_directory), wav_index_(0) {}

  std::ofstream question_file_;
  std::ofstream answer_file_;
  std::ofstream wave_file_;
  std::string root_directory_;

  uint64_t wav_index_;
  std::mutex mtx_;
};

class AudioTask {
public:
  // The step of processing that the AudioTask is in.
  typedef enum {
    START,
    SENDING,
    SENDING_COMPLETE,
    RECEIVING_COMPLETE,
    STREAM_CLOSED
  } State;

  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  AudioTask(const std::shared_ptr<AudioData> &audio_data,
            const uint32_t _corr_id, const Stream::PrepareFn &infer_prepare_fn,
            const std::string &language_code, const int32_t chunk_duration_ms,
            const bool print_results,
            std::shared_ptr<SquadEvalDataset> &squad_eval_dataset,
            std::shared_ptr<OutputFilestreams> &output_filestream,
            std::shared_ptr<nvrpc::client::Executor> &executor,
            const TimePoint &start_time);

  TimePoint &NextTimePoint() { return next_time_point_; }
  State GetState() { return state_; }
  int GetId() { return corr_id_; }
  double AudioProcessed() { return audio_processed_; }
  Status GetStatus() { return task_status_; }
  std::shared_ptr<Results> GetResult() { return result_; }

  Status Step();
  Status WaitForCompletion();

private:
  void ReceiveResponse(SpeechSquadInferResponse &&response);
  void FinalizeTask(const ::grpc::Status &status);

  std::string StateAsString();

  SpeechSquadInferRequest request_;

  std::shared_ptr<AudioData> audio_data_;
  size_t offset_;
  uint32_t corr_id_;
  std::string language_code_;
  int32_t chunk_duration_ms_;
  bool print_results_;
  std::shared_ptr<SquadEvalDataset> squad_eval_dataset_;

  std::shared_ptr<OutputFilestreams> output_filestreams_;

  std::unique_ptr<Stream> stream_;
  std::shared_ptr<nvrpc::client::Executor> executor_;

  Status task_status_;
  ::grpc::Status grpc_status_;

  // Marks the timepoint for the next activity
  TimePoint next_time_point_;
  // Records the timestamp of the last send activity
  TimePoint send_time_;
  // The bytes of audio data to be sent in the next step
  double bytes_to_send_;
  // The total audio processed by this task in seconds
  double audio_processed_;

  // Holds the results of the transaction
  std::shared_ptr<Results> result_;
  // Current state of the task
  std::atomic<State> state_;
};

} // namespace speech_squad
