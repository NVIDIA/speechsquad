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
#include <grpcpp/grpcpp.h>
#include <strings.h>

#include "speech_squad_client.h"
#include "status.h"

using grpc::Status;
using grpc::StatusCode;

DEFINE_string(squad_questions_json, "questions.json",
              "Json file with location of audio files for each Squad question");
DEFINE_string(squad_dataset_json, "dev-v2.0.json",
              "Json file with Squad dataset");
DEFINE_string(speech_squad_uri, "localhost:50051",
              "URI to access speech-squad-server");
DEFINE_int32(num_iterations, 1, "Number of times to loop over audio files");
DEFINE_int32(channel_num, -1, "Number of grpc channels to create");
DEFINE_int32(
    offset_duration, -1,
    "The minimum time offset in microseconds between the launch of successive "
    "sequences");
DEFINE_bool(true_concurrency, true, "Enables the true concurrency mode ");
DEFINE_int32(num_parallel_requests, 1,
             "Number of parallel requests to keep in flight");
DEFINE_int32(chunk_duration_ms, 800, "Chunk duration in milliseconds");
DEFINE_int32(
    executor_count, 0,
    "The number of threads to perform streaming I/O. The default value is 0 "
    "which means the client will detect the hardware concurrency and create "
    "that many executor threads with each thread dedicated to one of the core");
DEFINE_bool(print_results, true, "Print final results");
DEFINE_string(output_root_folder, "./final_results",
              "Folder to hold the returned audio data along with above json");
DEFINE_string(
    question_output_filename, "squad_question.json",
    "Filename of .json file containing Squad questions. Will be stored within "
    "--output_root_folder");
DEFINE_string(
    answer_output_filename, "squad_answers.json",
    "Filename of .json file containing Squad answers. Will be stored within "
    "--output_root_folder");
DEFINE_string(
    output_wave_filename, "squad_output_wave.json",
    "Filename of .json file containing the tts output and latencies. Will be "
    "stored within --output_root_folder");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  std::stringstream str_usage;
  str_usage << "Usage: speech_squad_streaming_client " << std::endl;
  str_usage << "           --squad_questions_json=<question_json> "
            << std::endl;
  str_usage << "           --squad_dataset_json=<location_of_squad_json> "
            << std::endl;
  str_usage << "           --speech_squad_uri=<server_name:port> " << std::endl;
  str_usage << "           --chunk_duration_ms=<integer> " << std::endl;
  str_usage << "           --executor_count=<integer> " << std::endl;
  str_usage << "           --num_iterations=<integer> " << std::endl;
  str_usage << "           --offset_duration=<integer> " << std::endl;
  str_usage << "           --num_parallel_requests=<integer> " << std::endl;
  str_usage << "           --channel_num=<integer> " << std::endl;
  str_usage << "           --true_concurrency=<true|false> " << std::endl;
  str_usage << "           --print_results=<true|false> " << std::endl;
  str_usage << "           --output_root_folder=<string>" << std::endl;
  str_usage << "           --answer_output_filename=<string>" << std::endl;
  str_usage << "           --question_output_filename=<string>" << std::endl;
  str_usage << "           --output_wave_filename=<string>" << std::endl;
  ::google::SetUsageMessage(str_usage.str());
  ::google::SetVersionString("0.0.1");

  if (argc < 2) {
    std::cout << ::google::ProgramUsage();
    return 1;
  }

  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc > 1) {
    std::cout << ::google::ProgramUsage();
    return 1;
  }

  int proc_index = 0;
  int proc_count = 0;
  MPI_CHECK(MPI_Init(&argc, &argv));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &proc_index));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &proc_count));

  if (proc_count > FLAGS_num_parallel_requests) {
    std::cerr << "Specified --num_parallel_requests can not be less than "
                 "number of MPI process requested"
              << std::endl;
    return 1;
  }

  std::string suffix((proc_count > 1) ? ("/proc" + std::to_string(proc_index))
                                      : "");
  std::string output_root_folder(FLAGS_output_root_folder + suffix);
  if (FLAGS_print_results) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if ((proc_count > 1) && (proc_index == 0)) {
      if (!CreateDirectory(std::string(FLAGS_output_root_folder))) {
        return 1;
      }
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!CreateDirectory(std::string(output_root_folder))) {
      return 1;
    }
  }

  int num_parallel_requests = (FLAGS_num_parallel_requests / proc_count);
  if (proc_index < (FLAGS_num_parallel_requests % proc_count)) {
    num_parallel_requests++;
  }

  int channel_count = (FLAGS_channel_num == -1)
                          ? ((num_parallel_requests / 100) + 1)
                          : FLAGS_channel_num;
  std::vector<std::shared_ptr<grpc::Channel>> channels;
  for (int i = 0; i < channel_count; i++) {
    channels.push_back(grpc::CreateChannel(FLAGS_speech_squad_uri,
                                           grpc::InsecureChannelCredentials()));
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(10000);

    if (!WaitUntilReady(channels.back(), deadline, FLAGS_speech_squad_uri)) {
      return 1;
    }
  }

  auto squad_eval_dataset = std::make_shared<speech_squad::SquadEvalDataset>();
  speech_squad::Status status =
      squad_eval_dataset->LoadFromJson(FLAGS_squad_dataset_json);

  if (!status.IsOk()) {
    std::cerr << status.AsString() << std::endl;
    return 1;
  }

  OutputFilenames output_files(FLAGS_question_output_filename,
                               FLAGS_answer_output_filename,
                               FLAGS_output_wave_filename, output_root_folder);

  int offset_duration =
      (FLAGS_offset_duration == -1)
          ? ((FLAGS_chunk_duration_ms * 1000) / num_parallel_requests)
          : FLAGS_offset_duration;

  speech_squad::SpeechSquadClient speech_squad_client(
      channels, num_parallel_requests, FLAGS_num_iterations, "en-US",
      FLAGS_print_results, FLAGS_chunk_duration_ms, FLAGS_executor_count,
      output_files, squad_eval_dataset, FLAGS_squad_questions_json,
      FLAGS_num_iterations, FLAGS_offset_duration, proc_index, proc_count,
      FLAGS_true_concurrency);

  int ret = speech_squad_client.Run();

  MPI_CHECK(MPI_Finalize());

  return ret;
}
