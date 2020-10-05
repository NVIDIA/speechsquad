
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

#include "cpu_asr_grpc.h"

#include "wav_reader.h"
#include <chrono>
#include <glog/logging.h>
#include <random>
#include <thread>

#include "kaldi_asr.h"

using grpc::Status;
using grpc::StatusCode;

std::vector<std::string> ExtractWordsFromString(std::string &sentence) {
  const std::string ws = " \t\r\n";
  std::size_t pos = 0;
  std::vector<std::string> words;
  while (pos != sentence.size()) {
    std::size_t from = sentence.find_first_not_of(ws, pos);
    if (from == std::string::npos) {
      break;
    }
    std::size_t to = sentence.find_first_of(ws, from + 1);
    if (to == std::string::npos) {
      to = sentence.size();
    }
    words.push_back(sentence.substr(from, to - from));
    pos = to;
  }
  return words;
}

Status ValidateConfig(nj_asr::RecognitionConfig &config, int numchannels,
                      int bitspersample) {
  if (numchannels > 1) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  std::string("Error: channel count must be 1"));
  }

  if (bitspersample != 8 && bitspersample != 16) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  std::string("Error: bits per sample must be 8 or 16"));
  }

  if (config.enable_separate_recognition_per_channel()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        std::string(
            "Error: separate recognition per channel not yet supported"));
  }

  return Status::OK;
}

ASRServiceImpl::ASRServiceImpl() {
  LOG(INFO) << "KALDI CPU ASR coming up...";

  const char *kaldi_path = getenv("KALDI_MODEL_PATH");
  if (!kaldi_path)
    kaldi_path = "/data/models/LibriSpeech";

  use_kaldi_cpu_asr_ = true;

  kaldi_cpu_asr_ = new KaldiASRContext(kaldi_path);
  if (kaldi_cpu_asr_->Initialize() != 0) {
    std::cerr << "unable to create Kaldi object\n";
    exit(1);
  }
}

ASRServiceImpl::~ASRServiceImpl() {
  if (use_kaldi_cpu_asr_)
    delete kaldi_cpu_asr_;
}

Status ASRServiceImpl::Recognize(ServerContext *context,
                                 const nj_asr::RecognizeRequest *request,
                                 nj_asr::RecognizeResponse *response) {
  LOG(INFO) << "ASRService.Recognize called.";
  auto start = std::chrono::high_resolution_clock::now();
  Status status = Status::OK;
  LOG(ERROR) << "Not Implemented";
  return Status(StatusCode::INVALID_ARGUMENT,
                "Only StreamingRecognize is implemented for Kaldi");
}

void LatticeToString(fst::SymbolTable &word_syms,
                     const kaldi::CompactLattice &dlat, std::string *out_str) {
  kaldi::CompactLattice best_path_clat;
  kaldi::CompactLatticeShortestPath(dlat, &best_path_clat);

  kaldi::Lattice best_path_lat;
  fst::ConvertLattice(best_path_clat, &best_path_lat);

  std::vector<int32> alignment;
  std::vector<int32> words;
  kaldi::LatticeWeight weight;
  fst::GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  std::ostringstream oss;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s == "")
      std::cerr << "Word-id " << words[i] << " not in symbol table.";
    oss << s << " ";
  }
  *out_str = std::move(oss.str());
}

Status ASRServiceImpl::StreamingRecognize(
    ServerContext *context,
    ServerReaderWriter<nj_asr::StreamingRecognizeResponse,
                       nj_asr::StreamingRecognizeRequest> *stream) {
  LOG(INFO) << "ASRService.StreamingRecognize called.";
  Status status = Status::OK;
  nj_asr::StreamingRecognizeRequest streaming_request;
  nj_asr::StreamingRecognitionConfig streaming_config;
  nj_asr::RecognitionConfig config;

  WavReader s_decoder;

  std::string model_name;

  std::unique_ptr<kaldi::OnlineNnet2FeaturePipeline> feature_pipeline_;
  std::unique_ptr<kaldi::SingleUtteranceNnet3Decoder> decoder_;

  if (stream->Read(&streaming_request) == false) {
    return Status::OK;
  }

  // Check that first request has config
  if (streaming_request.has_streaming_config() &&
      streaming_request.streaming_config().has_config()) {
    streaming_config = streaming_request.streaming_config();
    config = streaming_config.config();

    feature_pipeline_.reset(
        new kaldi::OnlineNnet2FeaturePipeline(*kaldi_cpu_asr_->feature_info_));
    decoder_.reset(new kaldi::SingleUtteranceNnet3Decoder(
        kaldi_cpu_asr_->decoder_opts, kaldi_cpu_asr_->trans_model_,
        *kaldi_cpu_asr_->decodable_info_, *kaldi_cpu_asr_->decode_fst_,
        &*feature_pipeline_));

    decoder_->InitDecoding(0);
  } else {
    LOG(ERROR) << "Early return" << std::endl;
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Error: First StreamingRecognize request must contain "
                  "RecognitionConfig");
  }
  // Handle audio_content requests
  AudioFormat format;

  bool first_buffer = true;
  bool read_header = true;

  int count = 0;
  while (stream->Read(&streaming_request)) {
    const std::string &raw_audio = streaming_request.audio_content();
    if (first_buffer) {
      // Detect format & populate stream format
      status = s_decoder.DetectFormat(raw_audio);
      if (!status.ok())
        break;
      // Check that config is valid
      format = s_decoder.GetFormat();
      status = ValidateConfig(config, format.numchannels, format.bitspersample);
      if (!status.ok()) {
        LOG(ERROR) << "Invalid config " << std::endl;
        break;
      }
      first_buffer = false;
    }
    std::shared_ptr<std::vector<float>> request_buffer =
        std::make_shared<std::vector<float>>();
    status = s_decoder.GetAudioBuffer(raw_audio, request_buffer, read_header);
    read_header = false;

    int32 num_samp = request_buffer->size();
    kaldi::SubVector<BaseFloat> wave_part(request_buffer->data(), num_samp);
    wave_part.Scale(SHRT_MAX);
    feature_pipeline_->AcceptWaveform(format.samplerate, wave_part);
    count += num_samp;
    if (!status.ok()) {
      LOG(ERROR) << "Invalid buffer " << std::endl;
      break;
    }
  }

  // Decode once we're done
  feature_pipeline_->InputFinished();
  decoder_->AdvanceDecoding();
  decoder_->FinalizeDecoding();

  if (decoder_->NumFramesDecoded() > 0) {
    kaldi::CompactLattice lat;
    std::string final_transcript;
    decoder_->GetLattice(true, &lat);
    LatticeToString(*kaldi_cpu_asr_->word_syms_, lat, &final_transcript);

    float sample_freq =
        kaldi_cpu_asr_->feature_info_->mfcc_opts.frame_opts.samp_freq;
    auto audio_processed = (float)count / sample_freq;

    nj_asr::StreamingRecognizeResponse response;
    auto result = response.add_results();
    result->set_is_final(true);
    result->set_channel_tag(1);
    result->set_audio_processed(audio_processed);
    auto alternative = result->add_alternatives();
    alternative->set_transcript(final_transcript);
    alternative->set_confidence(1.0);

    float server_latency = 0;
    {
      std::unique_lock<std::mutex> lock(this->mu_);
      context->AddTrailingMetadata(
          "tracing.server_latency.streaming_recognition",
          std::to_string(server_latency));
      stream->Write(response);
    }
  }

  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    LOG(ERROR) << "Early return" << std::endl;
    return status;
  }

  LOG(INFO) << "ASRService.StreamingRecognize returning OK";
  return Status::OK;
}
