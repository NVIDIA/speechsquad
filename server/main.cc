/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <nvrpc/executor.h>
#include <nvrpc/server.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

#include "context.h"
#include "resources.h"

// old server: "misty2-speech.jarvis-ai.nvidia.com"

DEFINE_string(logging_name, "speech_squad", "possibly change this if you have multiple backends");
DEFINE_string(asr_service_url, "asr.jarvis.nvda:80", "url for jarvis asr endpoint");
DEFINE_string(nlp_service_url, "nlp.jarvis.nvda:80", "url for jarvis nlp endpoint");
DEFINE_string(tts_service_url, "tts.jarvis.nvda:80", "url for jarvis tts endpoint");
DEFINE_int32(threads, 10, "number of forward progress threads / completion queues");
DEFINE_int32(contexts_per_thread, 100, "maximum number of concurrent contexts allowed to be in flight");
DEFINE_int32(channels, 50, "number of channels");

using namespace demo;

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging(FLAGS_logging_name.c_str());
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    auto server = std::make_unique<nvrpc::Server>("0.0.0.0:1337");

    std::string asr_url = FLAGS_asr_service_url;
    std::string nlp_url = FLAGS_nlp_service_url;
    std::string tts_url = FLAGS_tts_service_url;

    auto resources     = std::make_shared<SpeechSquadResources>(asr_url, nlp_url, tts_url, FLAGS_threads, FLAGS_channels);
    auto executor      = server->RegisterExecutor(new executor_t(FLAGS_threads));
    auto service       = server->RegisterAsyncService<SpeechSquadService>();
    auto rpc_streaming = service->RegisterRPC<SpeechSquadContext>(&SpeechSquadService::AsyncService::RequestSpeechSquadInfer);
    executor->RegisterContexts(rpc_streaming, resources, FLAGS_contexts_per_thread);

    server->Run();

    return 0;
}