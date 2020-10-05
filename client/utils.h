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

#include <dirent.h>
#include <grpcpp/grpcpp.h>
#include <sys/stat.h>

#include <iostream>
#include <sstream>

#ifdef PLAYGROUND_USE_MPI
#include "mpi.h"
#define MPI_CHECK(mpicall) mpicall
#else
#define MPI_CHECK(mpicall)
#endif

#include "rapidjson/document.h"
#include "speech_squad.grpc.pb.h"
#include "squad_eval_dataset.h"

#define WAVE_FORMAT_PCM 0x0001
#define WAVE_FORMAT_ALAW 0x0006
#define WAVE_FORMAT_MULAW 0x0007

typedef struct __attribute__((packed)) {
  char chunk_id[4]; // should be "RIFF" in ASCII form
  int32_t chunk_size;
  char format[4];         // should be "WAVE"
  char subchunk1_id[4];   // should be "fmt "
  int32_t subchunk1_size; // should be 16 for PCM format
  int16_t audioformat;    // should be 1 for PCM
  int16_t numchannels;
  int32_t samplerate;
  int32_t byterate;      // == samplerate * numchannels * bitspersample/8
  int16_t blockalign;    // == numchannels * bitspersample/8
  int16_t bitspersample; //    8 bits = 8, 16 bits = 16, etc.
  int32_t subchunk2ID;
  int32_t subchunk2size;
} FixedWAVHeader;

struct OutputFilenames {
  OutputFilenames(const std::string &question_json,
                  const std::string &answer_json, const std::string &wave_json,
                  const std::string &root_folder)
      : question_json_(question_json), answer_json_(answer_json),
        wave_json_(wave_json), root_folder_(root_folder) {}

  std::string question_json_;
  std::string answer_json_;
  std::string wave_json_;
  std::string root_folder_;
};

struct AudioData {
  std::vector<char> data;
  std::string filename;
  int sample_rate;
  int channels;
  AudioEncoding encoding;
  std::string question_id;
};

bool WaitUntilReady(std::shared_ptr<grpc::Channel> channel,
                    std::chrono::system_clock::time_point &deadline,
                    std::string speech_squad_uri);

bool ParseAudioFileHeader(std::string file, AudioEncoding &encoding,
                          int &samplerate, int &channels);

bool ParseQuestionsJson(
    const char *path,
    std::vector<std::pair<std::string, std::string>> &questions,
    const std::string &key);

int GetProcIndex(std::vector<long> &allocated_bytes);

void LoadAudioData(std::vector<std::shared_ptr<AudioData>> &all_audio_data,
                   std::string &path, const std::string &key, int proc_index,
                   int proc_count);

void GetComponents(std::vector<std::string> *components);

bool CreateDirectory(const std::string &directory_path);
