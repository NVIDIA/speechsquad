#pragma once

#include <nvrpc/executor.h>
#include <trtlab/core/userspace_threads.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

#include "jarvis_asr.grpc.pb.h"
#include "jarvis_asr.pb.h"

#include "jarvis_nlp.grpc.pb.h"
#include "jarvis_nlp.pb.h"

#include "jarvis_tts.grpc.pb.h"
#include "jarvis_tts.pb.h"

namespace demo
{
    using thread_t = trtlab::userspace_threads;
    using executor_t = nvrpc::Executor;

    using asr_request_t = nvidia::jarvis::asr::StreamingRecognizeRequest;
    using asr_response_t = nvidia::jarvis::asr::StreamingRecognizeResponse;

    using nlp_request_t = nvidia::jarvis::nlp::NaturalQueryRequest;
    using nlp_response_t = nvidia::jarvis::nlp::NaturalQueryResponse;

    using tts_request_t = nvidia::jarvis::tts::SynthesizeSpeechRequest;
    using tts_response_t = nvidia::jarvis::tts::SynthesizeSpeechResponse;

    using AudioEncoding = nvidia::jarvis::asr::AudioEncoding;
} // namespace demo
