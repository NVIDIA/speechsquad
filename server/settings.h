#pragma once

#include <nvrpc/executor.h>
#include <trtlab/core/userspace_threads.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

#include "riva_asr.grpc.pb.h"
#include "riva_asr.pb.h"

#include "riva_nlp.grpc.pb.h"
#include "riva_nlp.pb.h"

#include "riva_tts.grpc.pb.h"
#include "riva_tts.pb.h"

namespace demo
{
    using thread_t = trtlab::userspace_threads;
    using executor_t = nvrpc::Executor;

    using asr_request_t = nvidia::riva::asr::StreamingRecognizeRequest;
    using asr_response_t = nvidia::riva::asr::StreamingRecognizeResponse;

    using nlp_request_t = nvidia::riva::nlp::NaturalQueryRequest;
    using nlp_response_t = nvidia::riva::nlp::NaturalQueryResponse;

    using tts_request_t = nvidia::riva::tts::SynthesizeSpeechRequest;
    using tts_response_t = nvidia::riva::tts::SynthesizeSpeechResponse;

    using AudioEncoding = nvidia::riva::AudioEncoding;
} // namespace demo
