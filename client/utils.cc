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

#include "utils.h"
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

bool WaitUntilReady(std::shared_ptr<grpc::Channel> channel,
                    std::chrono::system_clock::time_point &deadline,
                    std::string speech_squad_uri) {
  auto state = channel->GetState(true);
  while (state != GRPC_CHANNEL_READY) {
    if (!channel->WaitForStateChange(state, deadline)) {
      std::cout << "Cannot create GRPC channel at uri " << speech_squad_uri
                << std::endl;
      return false;
    }
    state = channel->GetState(true);
  }
  return true;
}

bool ParseAudioFileHeader(std::string file, AudioEncoding &encoding,
                          int &samplerate, int &channels) {
  std::ifstream file_stream(file);
  FixedWAVHeader header;
  std::streamsize bytes_read = file_stream.rdbuf()->sgetn(
      reinterpret_cast<char *>(&header), sizeof(header));

  if (bytes_read != sizeof(header)) {
    std::cerr << "Error reading file " << file << std::flush << std::endl;
    return false;
  }

  std::string tag(header.chunk_id, 4);
  if (tag == "RIFF") {
    if (header.audioformat == WAVE_FORMAT_PCM)
      // Only supports LINEAR_PCM
      encoding = LINEAR_PCM;
    else
      return false;
    samplerate = header.samplerate;
    channels = header.numchannels;
    return true;
  } else if (tag == "fLaC") {
    return false;
  }
  return false;
}

bool ParseQuestionsJson(
    const char *path,
    std::vector<std::pair<std::string, std::string>> &questions,
    const std::string &key) {
  questions.clear();
  std::ifstream manifest_file;
  manifest_file.open(path, std::ifstream::in);

  std::string line;
  std::string audio_filepath_key("audio_filepath");
  std::string question_id_key(key);

  while (std::getline(manifest_file, line, '\n')) {
    if (line == "") {
      continue;
    }

    rapidjson::Document doc;
    // Parse line
    doc.Parse(line.c_str());

    if (!doc.IsObject()) {
      std::cout << "Problem parsing line: " << line << std::endl;
      return false;
    }

    // Get Question ID
    if (!doc.HasMember(question_id_key.c_str())) {
      std::cout << "Line: " << line << " does not contain " << question_id_key
                << " key" << std::endl;
      return false;
    }
    std::string question_id = doc[question_id_key.c_str()].GetString();

    // Get filepath
    if (!doc.HasMember(audio_filepath_key.c_str())) {
      std::cout << "Line: " << line << " does not contain "
                << audio_filepath_key << " key" << std::endl;
      return false;
    }
    std::string filepath = doc[audio_filepath_key.c_str()].GetString();

    questions.emplace_back(std::make_pair(question_id, filepath));
  }

  manifest_file.close();
  return true;
}

int GetProcIndex(std::vector<long> &allocated_bytes) {
  int index = 0;
  if (allocated_bytes.size() > 1) {
    int smallest_value = INT_MAX;
    // Using linear search as the size of the vector
    // will be quite small.
    for (int i = 0; i < allocated_bytes.size(); i++) {
      if (smallest_value > allocated_bytes[i]) {
        smallest_value = allocated_bytes[i];
        index = i;
      }
    }
  }
  return index;
}

void LoadAudioData(std::vector<std::shared_ptr<AudioData>> &all_audio_data,
                   std::string &path, const std::string &key, int proc_index,
                   int proc_count) {
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (proc_index == 0) {
    std::cout << "Loading eval dataset..." << std::flush << std::endl;
  }

  std::vector<std::pair<std::string, std::string>> questions;

  std::vector<long> allocated_bytes_per_proc(proc_count, 0);

  ParseQuestionsJson(path.c_str(), questions, key);

  for (uint32_t i = 0; i < questions.size(); ++i) {
    std::string question_id = questions[i].first;
    std::string filename = questions[i].second;

    AudioEncoding encoding;
    int samplerate;
    int channels;
    if (!ParseAudioFileHeader(filename, encoding, samplerate, channels)) {
      std::cerr << "Cannot parse audio file header for file " << filename
                << std::endl;
      return;
    }
    std::shared_ptr<AudioData> audio_data = std::make_shared<AudioData>();

    audio_data->sample_rate = samplerate;
    audio_data->filename = filename;
    audio_data->question_id = question_id;
    audio_data->encoding = encoding;
    audio_data->channels = channels;
    audio_data->data.assign(
        std::istreambuf_iterator<char>(std::ifstream(filename).rdbuf()),
        std::istreambuf_iterator<char>());
    auto index = GetProcIndex(allocated_bytes_per_proc);
    allocated_bytes_per_proc[index] += audio_data->data.size();
    if (index == proc_index) {
      all_audio_data.push_back(std::move(audio_data));
    }
  }

  for (int i = 0; i < proc_count; i++) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (proc_index == i) {
      std::cout << "Done loading " << all_audio_data.size()
                << " files for process " << i << std::endl;
    }
  }
}

void GetComponents(std::vector<std::string> *components) {
  components->push_back("tracing.server_latency.natural_query");
  components->push_back("tracing.server_latency.speech_synthesis");
  components->push_back("tracing.server_latency.streaming_recognition");
  components->push_back("tracing.speech_squad.asr_latency");
  components->push_back("tracing.speech_squad.nlp_latency");
  components->push_back("tracing.speech_squad.tts_latency");
}

bool CreateDirectory(const std::string &directory_path) {
  // Creating a directory
  if (mkdir(directory_path.c_str(), 0777) == -1) {
    std::cerr << "Failed to create directory \"" << directory_path
              << "\" :  " << strerror(errno) << std::endl;
    return false;
  }
  return true;
}
