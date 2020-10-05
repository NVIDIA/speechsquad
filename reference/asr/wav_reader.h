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

#ifndef WAV_READER_H_
#define WAV_READER_H_ 1

#include "cpu_asr_grpc.h"

struct AudioFormat {
  int numchannels;
  int samplerate;
  int bitspersample;
};

class WavReader {
  bool is_first_buffer = true;

  // format from header decode
  AudioFormat hdr_format;

  // final format to use
  AudioFormat format;

public:
  Status GetAudioBuffer(const std::string &raw_audio,
                        std::shared_ptr<std::vector<float>> audio_buffer,
                        bool read_header);

  Status DetectFormat(const std::string &raw_audio);

  AudioFormat GetFormat() { return format; };
};

#endif
