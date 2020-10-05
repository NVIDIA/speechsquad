
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

#include "wav_reader.h"

#include <glog/logging.h>

#include <chrono>
#include <random>
#include <thread>

// standard format codes for waveform data
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
  char subchunk2ID[4];
  int32_t subchunk2size;
} FixedWAVHeader;

static int ParseWavHeader(std::stringstream &wavfile, FixedWAVHeader &header,
                          bool read_header) {
  if (read_header) {
    bool is_header_valid = false;
    wavfile.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (strncmp(header.format, "WAVE", sizeof(header.format)) == 0) {
      if (header.audioformat == WAVE_FORMAT_PCM && header.bitspersample == 16) {
        is_header_valid = true;
      } else if ((header.audioformat == WAVE_FORMAT_MULAW ||
                  header.audioformat == WAVE_FORMAT_ALAW) &&
                 header.bitspersample == 8) {
        is_header_valid = true;
      }
    }

    if (!is_header_valid) {
      LOG(INFO) << "error: unsupported format"
                << " audioformat " << header.audioformat << " channels "
                << header.numchannels << " rate " << header.samplerate
                << " bitspersample " << header.bitspersample << std::endl;
      return -1;
    }

    // Skip to 'data' chunk
    if (strncmp(header.subchunk2ID, "data", sizeof(header.subchunk2ID))) {
      char chunk_id[4];
      while (wavfile.good()) {
        wavfile.read(reinterpret_cast<char *>(&chunk_id), sizeof(chunk_id));
        if (strncmp(chunk_id, "data", sizeof(chunk_id)) == 0) {
          // read size bytes and break
          wavfile.read(reinterpret_cast<char *>(&chunk_id), sizeof(chunk_id));
          break;
        }
        wavfile.seekg(-3, std::ios_base::cur);
      }
    }
  }

  if (wavfile) {
    int wav_size;
    // move to first sample
    // wavfile.seekg(4, std::ios_base::cur);

    // calculate size of samples
    std::streampos curr_pos = wavfile.tellg();
    wavfile.seekg(0, wavfile.end);
    wav_size = wavfile.tellg() - curr_pos;
    wavfile.seekg(curr_pos);

    return wav_size;
  }

  return -2;
}

Status
WavReader::GetAudioBuffer(const std::string &raw_audio,
                          std::shared_ptr<std::vector<float>> audio_buffer,
                          bool read_header) {
  std::stringstream wavfile(raw_audio, std::ios::in | std::ios::binary);
  FixedWAVHeader header;
  int wavsize = ParseWavHeader(wavfile, header, read_header);

  int num_samples = wavsize / (format.bitspersample / 8);

  std::vector<int16_t> buffer16(num_samples);

  wavfile.read(reinterpret_cast<char *>(buffer16.data()), wavsize);

  audio_buffer->resize(num_samples);
  for (int samp_idx = 0; samp_idx < num_samples; samp_idx++) {
    (*audio_buffer)[samp_idx] = buffer16[samp_idx] / (float)INT16_MAX;
  }
  return Status::OK;
}

Status WavReader::DetectFormat(const std::string &raw_audio) {
  char header[4];
  std::stringstream raw_stream(raw_audio, std::ios::in | std::ios::binary);

  raw_stream.read(reinterpret_cast<char *>(&header), sizeof(header));

  if (strncmp(header, "RIFF", sizeof(header)) == 0) {
    FixedWAVHeader header;
    raw_stream.seekg(std::ios_base::beg);
    int wavsize = ParseWavHeader(raw_stream, header, true /*read_header*/);
    hdr_format.numchannels = header.numchannels;
    hdr_format.samplerate = header.samplerate;
    hdr_format.bitspersample = header.bitspersample;
  } else {
    return Status(StatusCode::INVALID_ARGUMENT, "Error: invalid format");
  }

  // figure out actual stream format
  format.numchannels = hdr_format.numchannels;
  format.samplerate = hdr_format.samplerate;
  format.bitspersample = hdr_format.bitspersample;

  std::string str_is_wav;
  LOG(INFO) << "Detected format:"
            << " numchannels = " << format.numchannels
            << " samplerate = " << format.samplerate
            << " bitspersample = " << format.bitspersample;

  return Status::OK;
}
