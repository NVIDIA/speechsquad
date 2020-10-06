

/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <memory>

#include <nvrpc/context.h>
#include <nvrpc/client/client_unary.h>
#include <nvrpc/client/client_streaming.h>
#include <nvrpc/client/client_single_up_multiple_down.h>

#include "settings.h"
#include "resources.h"

namespace demo
{
    class SpeechSquadContext final : public nvrpc::StreamingContext<SpeechSquadInferRequest, SpeechSquadInferResponse, SpeechSquadResources>
    {
        void StreamInitialized(std::shared_ptr<ServerStream>) final override;
        void RequestsFinished(std::shared_ptr<ServerStream>) final override;

        void RequestReceived(SpeechSquadInferRequest&& input, std::shared_ptr<ServerStream> stream) final override;

        enum class State
        {
            Uninitialized,
            Initialized,
            ReceivingAudio,
            AudioUploadComplete
        };

    public:
        // callbacks
        void ASRCallbackOnResponse(asr_response_t&&);
        void ASRCallbackOnFinish(const ::grpc::Status&, const meta_data_t&);
        void NLPCallbackOnResponse(const nlp_response_t&);
        void NLPCallbackOnComplete(const ::grpc::Status&, const meta_data_t&);
        void TTSCallbackOnResponse(tts_response_t&&);
        void TTSCallbackOnComplete(const ::grpc::Status&, const meta_data_t&);

    private:
        void OnContextReset() final override;

        void ExtractTimings(const meta_data_t&);

        // state variables
        State       m_state;
        std::string m_context;
        std::string m_question;
        std::string m_answer;
        float       m_nlp_score;
        AudioConfig m_tts_config;
        bool        m_first_tts_response;
        bool        m_should_cancel;
        bool        m_debug_tts;

        // timing meta data
        std::multimap<std::string, float> m_timings;

        // store access to the response stream
        std::shared_ptr<ServerStream> m_stream;

        // jarvis clients
        std::unique_ptr<asr_client_t> m_asr_client;
        std::unique_ptr<nlp_client_t> m_nlp_client;
        std::unique_ptr<tts_client_t> m_tts_client;

        // timers
        std::chrono::high_resolution_clock::time_point m_asr_writes_done;
        std::chrono::high_resolution_clock::time_point m_asr_on_complete;

        std::chrono::high_resolution_clock::time_point m_nlp_start;
        std::chrono::high_resolution_clock::time_point m_nlp_finish;

        std::chrono::high_resolution_clock::time_point m_tts_start;
        std::chrono::high_resolution_clock::time_point m_tts_first_packet;
    };

} // namespace demo