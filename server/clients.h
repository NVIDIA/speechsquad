

#pragma once

#include <glog/logging.h>

#include <nvrpc/client/executor.h>
#include <nvrpc/client/client_unary_v2.h>
#include <nvrpc/client/client_streaming_v3.h>
#include <nvrpc/client/client_single_up_multiple_down.h>

#include "settings.h"

namespace demo
{
    class SpeechSquadContext;

    class ASRClient final : public nvrpc::client::v3::ClientStreaming<asr_request_t, asr_response_t>
    {
        using Client = nvrpc::client::v3::ClientStreaming<asr_request_t, asr_response_t>;

    public:
        using PrepareFn = typename Client::PrepareFn;

        ASRClient(SpeechSquadContext* context, PrepareFn prepare_fn, std::shared_ptr<nvrpc::client::Executor> executor)
        : Client(prepare_fn, executor), m_context(context)
        {
            CHECK_NOTNULL(m_context);
        }

        void CallbackOnResponseReceived(asr_response_t&& response) final override;
        void CallbackOnComplete(const ::grpc::Status& status) final override;

    private:
        SpeechSquadContext* m_context;
    };

    class NLPClient final : public nvrpc::client::v2::ClientUnary<nlp_request_t, nlp_response_t>
    {
        using Client = nvrpc::client::v2::ClientUnary<nlp_request_t, nlp_response_t>;

    public:
        using PrepareFn = typename Client::PrepareFn;

        NLPClient(SpeechSquadContext* context, PrepareFn prepare_fn, std::shared_ptr<nvrpc::client::Executor> executor)
        : Client(prepare_fn, executor), m_context(context)
        {
            CHECK_NOTNULL(m_context);
        }

        void CallbackOnResponseReceived(nlp_response_t&&) final override;
        void CallbackOnComplete(const ::grpc::Status&) final override;
    
    private:
        SpeechSquadContext* m_context;
    };

    class TTSClient final : public nvrpc::client::ClientSingleUpMultipleDown<tts_request_t, tts_response_t>
    {
        using Client = nvrpc::client::ClientSingleUpMultipleDown<tts_request_t, tts_response_t>;

    public:
        using PrepareFn = typename Client::PrepareFn;

        TTSClient(SpeechSquadContext* context, PrepareFn prepare_fn, std::shared_ptr<nvrpc::client::Executor> executor)
        : Client(prepare_fn, executor), m_context(context)
        {
            CHECK_NOTNULL(m_context);
        }

        void CallbackOnResponseReceived(tts_response_t&& response) final override;
        void CallbackOnComplete(const ::grpc::Status& status) final override;

    private:
        SpeechSquadContext* m_context;
    };

} // namespace demo
