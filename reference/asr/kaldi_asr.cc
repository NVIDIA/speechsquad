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

#include "kaldi_asr.h"

#include "feat/wave-reader.h"
#include "online2/online-timing.h"
#include "online2/onlinebin-util.h"

#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
#include <cstdlib>
#include <iostream>

using std::cerr;

using namespace kaldi;

KaldiASRContext::KaldiASRContext(const std::string &model_path)
    : //  = "/data/models/LibriSpeech"):
      model_path_(model_path) {}

KaldiASRContext::~KaldiASRContext() { delete word_syms_; }

int KaldiASRContext::Initialize() {

  // Reading config
  float beam = 12, lattice_beam = 5;
  int max_active = 10000;
  int frame_subsampling_factor = 3;
  float acoustic_scale = 1.0;
  int num_worker_threads = 40;

  num_channels_ = 64;
  max_batch_size_ = 64;

  ParseOptions po("Kaldi ASR Parameters");
  feature_opts.Register(&po);
  compute_opts.Register(&po);
  decoder_opts.Register(&po);

  feature_opts.mfcc_config = model_path_ + "/conf/mfcc.conf";
  feature_opts.ivector_extraction_config =
      model_path_ + "/conf/ivector_extractor.conf";

  nnet3_rxfilename_ = model_path_ + "/final.mdl";
  word_syms_rxfilename_ = model_path_ + "/words.txt";
  fst_rxfilename_ = model_path_ + "/HCLG.fst";

  max_batch_size_ = std::max<int>(max_batch_size_, 1);
  num_channels_ = std::max<int>(num_channels_, 1);

  compute_opts.frame_subsampling_factor = frame_subsampling_factor;
  compute_opts.acoustic_scale = acoustic_scale;
  compute_opts.frames_per_chunk = 160;

  decoder_opts.beam = beam;
  decoder_opts.lattice_beam = lattice_beam;
  decoder_opts.max_active = max_active;

  po.ReadConfigFile(model_path_ + "/conf/online.conf");

  // check command line override

  const char *optstr = getenv("KALDI_MODEL_OPTIONS");

  if (optstr) {
    try {
      char *save, *cur, *prev;
      std::vector<char *> argv;
      std::string kaldi_opts = optstr;

      for (cur = prev = kaldi_opts.data(); cur;) {
        argv.push_back(cur);
        cur = strtok_r(prev, " \t", &save);
        prev = NULL;
      }
      po.Read(argv.size(), argv.data());
    } catch (std::exception e) {
      KALDI_LOG << "Failed to read KALDI_MODEL_OPTIONS.";
      return 1;
    }
  }
  bool binary;
  kaldi::Input ki(nnet3_rxfilename_, &binary);
  trans_model_.Read(ki.Stream(), binary);
  am_nnet_.Read(ki.Stream(), binary);

  kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
  kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
  kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(),
                              &(am_nnet_.GetNnet()));

  decode_fst_.reset(fst::ReadFstKaldiGeneric(fst_rxfilename_));

  // Loading word syms for text output
  if (word_syms_rxfilename_ != "") {
    if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_))) {
      std::cerr << "Could not read symbol table from file "
                << word_syms_rxfilename_;
      return -1;
    }
  }

  feature_info_.reset(new kaldi::OnlineNnet2FeaturePipelineInfo(feature_opts));

  // this object contains precomputed stuff that is used by all
  // decodable objects.  It takes a pointer to am_nnet because if
  // it has iVectors it has to modify the nnet to accept iVectors
  // at intervals.
  decodable_info_.reset(
      new nnet3::DecodableNnetSimpleLoopedInfo(compute_opts, &am_nnet_));

  std::cerr << "Kaldi config options are below. To override, set "
               "KALDI_MODEL_OPTIONS environment variable:";

  po.PrintConfig(std::cerr);
  return 0;
}
