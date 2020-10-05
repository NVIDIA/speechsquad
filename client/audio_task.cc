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

#include "audio_task.h"
#include "wave_file_writer.h"
#include <glog/logging.h>

namespace speech_squad {

std::string clean_string(std::string str) {
  const std::string s = "\"";
  const std::string t = "\\\"";

  std::string::size_type n = 0;
  while ((n = str.find(s, n)) != std::string::npos) {
    str.replace(n, s.size(), t);
    n += t.size();
  }
  return str;
}

AudioTask::AudioTask(
    const std::shared_ptr<AudioData> &audio_data, const uint32_t corr_id,
    const Stream::PrepareFn &infer_prepare_fn, const std::string &language_code,
    const int32_t chunk_duration_ms, const bool print_results,
    std::shared_ptr<speech_squad::SquadEvalDataset> &squad_eval_dataset,
    std::shared_ptr<OutputFilestreams> &output_filestream,
    std::shared_ptr<nvrpc::client::Executor> &executor,
    const TimePoint &start_time)
    : audio_data_(audio_data), offset_(0), corr_id_(corr_id),
      language_code_(language_code), chunk_duration_ms_(chunk_duration_ms),
      print_results_(print_results), squad_eval_dataset_(squad_eval_dataset),
      output_filestreams_(output_filestream), next_time_point_(start_time),
      audio_processed_(0.), state_(START) {
  // Prepare the server stream to be used with the transaction
  stream_ = std::make_unique<Stream>(
      infer_prepare_fn, executor,
      [this](SpeechSquadInferResponse &&response) {
        ReceiveResponse(std::move(response));
      },
      [this](const ::grpc::Status &status) { FinalizeTask(status); });

  result_ = std::make_shared<Results>();
  if (print_results_) {
    result_->audio_content = (char *)std::malloc(4100 * 256 * sizeof(float));
  }
}

Status AudioTask::Step() {
  if (state_ == SENDING_COMPLETE) {
    return Status(Status::Code::INTERNAL,
                  "Cannot step further from sending complete");
  }

  // Every step will overwrite this time stamp. The responses will be
  // delivered once sending is complete. At the time of the first response
  // this timestamp will carry the timestamp of the last request.
  send_time_ = std::chrono::high_resolution_clock::now();

  DVLOG(2) << "Executing step for task: " << corr_id_
           << ", state: " << StateAsString();

  // std::cerr << "step delay " << std::chrono::duration<double,
  //  std::milli>(send_time_ - next_time_point_).count() << "ms" << std::endl;

  // TODO: Can colllect the delay in scheduling to report the quality
  if (state_ == START) {
    // Send the configuration if at the first step
    auto speech_squad_config = request_.mutable_speech_squad_config();

    // Input Audio Configuration
    speech_squad_config->mutable_input_audio_config()->set_encoding(
        audio_data_->encoding);
    speech_squad_config->mutable_input_audio_config()->set_sample_rate_hertz(
        audio_data_->sample_rate);
    speech_squad_config->mutable_input_audio_config()->set_language_code(
        language_code_);
    speech_squad_config->mutable_input_audio_config()->set_audio_channel_count(
        audio_data_->channels);

    // Ouput Audio Configuration
    speech_squad_config->mutable_output_audio_config()->set_encoding(
        LINEAR_PCM);
    speech_squad_config->mutable_output_audio_config()->set_sample_rate_hertz(
        22050);
    speech_squad_config->mutable_output_audio_config()->set_language_code(
        "en-US");
    speech_squad_config->mutable_output_audio_config()->set_audio_channel_count(
        1);

    auto status = squad_eval_dataset_->GetQuestionContext(
        audio_data_->question_id, speech_squad_config->mutable_squad_context());
    if (!status.IsOk()) {
      return status;
    }

    stream_->Write(std::move(request_));
    state_ = SENDING;
  } else {
    // Send the audio content if not the first step
    request_.set_audio_content(&audio_data_->data[offset_], bytes_to_send_);
    offset_ += bytes_to_send_;
    if (!stream_->Write(std::move(request_))) {
      if (!stream_->CloseWrites()) {
        VLOG(2) << "Failed to CloseWrites for task: " << corr_id_;
      }
      state_ = SENDING_COMPLETE;
      DVLOG(2) << "Write failed for task: " << corr_id_;
    }
  }

  // Set and schedule the next chunk
  size_t chunk_size =
      (audio_data_->sample_rate * chunk_duration_ms_ / 1000) * sizeof(int16_t);
  size_t header_size = (offset_ == 0) ? sizeof(FixedWAVHeader) : 0;
  bytes_to_send_ =
      std::min(audio_data_->data.size() - offset_, chunk_size + header_size);

  // Transition to the sending completion if no more bytes to send
  if (bytes_to_send_ == 0) {
    if (!stream_->CloseWrites()) {
      VLOG(2) << "Failed to CloseWrites for task: " << corr_id_;
    }
    state_ = SENDING_COMPLETE;
    DVLOG(2) << "Sending complete for task: " << corr_id_;
  } else {
    double current_wait_time = 1000 * (bytes_to_send_ - header_size) /
                               (sizeof(int16_t) * audio_data_->sample_rate);
    // Accumulate the audio content processed
    audio_processed_ += current_wait_time / 1000.;
    next_time_point_ +=
        std::chrono::microseconds((int)current_wait_time * 1000);
  }

  return Status::Success;
}

Status AudioTask::WaitForCompletion() {
  while (!stream_->IsComplete()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  DVLOG(2) << "Completed task: " << corr_id_
           << ", status: " << grpc_status_.ok();
  if (grpc_status_.ok()) {
    return Status::Success;
  } else {
    return Status(grpc_status_.error_message());
  }
}

void AudioTask::ReceiveResponse(SpeechSquadInferResponse &&response) {
  DVLOG(2) << "Received response for task: " << corr_id_;
  auto now = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(result_->mtx);

  if (response.has_metadata()) {
    if (response.metadata().component_timing().empty()) {
      result_->squad_question = response.metadata().squad_question();
      result_->squad_answer = response.metadata().squad_answer();
    } else {
      std::vector<std::string> components;
      GetComponents(&components);
      for (const auto &component : components) {
        auto itr = response.metadata().component_timing().find(component);
        if (itr != response.metadata().component_timing().end()) {
          result_->component_timings[component] = itr->second;
        } else {
          task_status_ =
              Status(Status::Code::INTERNAL,
                     "Unable to find " + component + " in the response");
        }
      }
    }
  } else {
    if (print_results_) {
      memcpy(result_->audio_content + result_->audio_offset,
             (float *)response.audio_content().data(),
             response.audio_content().length());
      result_->audio_offset += response.audio_content().length();
    }

    if (result_->first_response) {
      result_->response_latency =
          std::chrono::duration<double, std::milli>(now - send_time_).count();
      result_->first_response = false;
    } else {
      result_->response_intervals.push_back(
          std::chrono::duration<double, std::milli>(
              now - result_->last_response_timestamp)
              .count());
    }
    result_->last_response_timestamp = now;
  }
}

void AudioTask::FinalizeTask(const ::grpc::Status &status) {
  DVLOG(2) << "Completion Callback for task: " << corr_id_
           << ", status:" << status.error_message();
  state_ = RECEIVING_COMPLETE;
  if (!status.ok()) {
    grpc_status_ = status;
    std::cout << "." << std::flush;
    return;
  }
  if (print_results_) {
    std::lock_guard<std::mutex> lock(output_filestreams_->mtx_);
    std::cout << "-----------------------------------------------------------"
              << std::endl;

    std::string filename = audio_data_->filename;
    std::cout << "File: " << filename << std::endl;
    if (result_->squad_question.size() == 0) {
      output_filestreams_->question_file_ << "{\"audio_filepath\": \""
                                          << filename << "\",";
      output_filestreams_->question_file_ << "\"question\": \"\"}" << std::endl;
    } else {
      output_filestreams_->answer_file_
          << "\"" << audio_data_->question_id << "\": \""
          << clean_string(result_->squad_answer) << "\",";

      output_filestreams_->question_file_ << "{\"audio_filepath\": \""
                                          << filename << "\",";
      output_filestreams_->question_file_
          << "\"text\": \"" << result_->squad_question << "\"}" << std::endl;

      if (result_->audio_offset == 0) {
        task_status_ =
            Status(Status::Code::INTERNAL, "No audio received in the response");
      }

      std::string output_filename = GetFullpath(
          output_filestreams_->root_directory_,
          std::string(std::to_string(output_filestreams_->wav_index_++) +
                      ".wav"));
      // WaveFileWriter::write(output_filename, 22050,
      // (float*)&result_->audio_content[0], 4100 * 256);
      WaveFileWriter::write(output_filename, 22050,
                            (float *)&result_->audio_content[0],
                            result_->audio_offset / sizeof(float));

      output_filestreams_->wave_file_
          << "{\"qid\":\"" << audio_data_->question_id << "\",\"text\":\""
          << clean_string(result_->squad_answer)
          << "\",\"synthesized_audio_path\":\"" << output_filename
          << "\",\"latencies\":[";

      bool first_latency = true;
      for (const auto lat : result_->response_intervals) {
        if (!first_latency) {
          output_filestreams_->wave_file_ << ",";
        }
        output_filestreams_->wave_file_ << "\"" << std::to_string(lat) << "\"";
        first_latency = false;
      }

      output_filestreams_->wave_file_ << "]}" << std::endl;

      std::cout << "SQUAD question: " << result_->squad_question << std::endl;
      std::cout << "SQUAD answer: " << result_->squad_answer << std::endl;
      std::cout << "Output File: " << output_filename << std::endl;
    }
  } else {
    std::cout << "." << std::flush;
  }
}

std::string AudioTask::StateAsString() {
  switch (state_) {
  case START:
    return "START";
  case SENDING:
    return "SENDING";
  case SENDING_COMPLETE:
    return "SENDING_COMPLETE";
  case RECEIVING_COMPLETE:
    return "RECEIVING_COMPLETE";
  case STREAM_CLOSED:
    return "STREAM_CLOSED";
  default:
    return "UNKNOWN";
  }
}

} // namespace speech_squad
