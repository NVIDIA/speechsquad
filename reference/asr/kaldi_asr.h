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

#define HAVE_CUDA 0 // Loading Kaldi headers without GPU

#include <cfloat>
#include <sstream>

#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"

#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

// #include <unordered_map>

using kaldi::BaseFloat;

class KaldiASRContext {
public:
  // init kaldi pipeline
  int Initialize();

  KaldiASRContext(const std::string &path);
  ~KaldiASRContext();

  // Models paths
  std::string model_path_;
  std::string nnet3_rxfilename_, fst_rxfilename_;
  std::string word_syms_rxfilename_;

  int max_batch_size_;
  int num_channels_;
  int num_worker_threads_;

  kaldi::OnlineNnet2FeaturePipelineConfig feature_opts;
  kaldi::nnet3::NnetSimpleLoopedComputationOptions compute_opts;
  kaldi::LatticeFasterDecoderConfig decoder_opts;

  std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info_;
  std::unique_ptr<kaldi::nnet3::DecodableNnetSimpleLoopedInfo> decodable_info_;
  std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;
  // Maintain the state of some shared objects
  kaldi::TransitionModel trans_model_;
  kaldi::nnet3::AmNnetSimple am_nnet_;
  fst::SymbolTable *word_syms_;
};
