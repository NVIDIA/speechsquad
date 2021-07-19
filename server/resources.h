#pragma once
#include <memory>
#include <vector>

#include "settings.h"

#include <trtlab/core/resources.h>
#include <trtlab/core/pool.h>

#include <nvrpc/client/executor.h>
#include <nvrpc/client/client_unary.h>
#include <nvrpc/client/client_streaming.h>
#include <nvrpc/client/client_single_up_multiple_down.h>

#include "settings.h"
#include "clients.h"

namespace demo
{
    class SpeechSquadContext;

    using meta_data_t = std::multimap<::grpc::string_ref, ::grpc::string_ref>;

    using asr_client_t = ASRClient;
    using nlp_client_t = NLPClient;
    using tts_client_t = TTSClient;

    class SpeechSquadResources : public ::trtlab::Resources
    {
    public:
        SpeechSquadResources(std::string asr_url, std::string nlp_url, std::string tts_url, int threads, int channels, std::string asr_model_name);
        ~SpeechSquadResources() override;

        std::shared_ptr<nvrpc::client::Executor> client_executor()
        {
            return m_client_executor;
        }

        std::unique_ptr<asr_client_t> create_asr_client(SpeechSquadContext*);
        std::unique_ptr<nlp_client_t> create_nlp_client(SpeechSquadContext*);
        std::unique_ptr<tts_client_t> create_tts_client(SpeechSquadContext*);
        std::string                   get_model();
    private:
        std::string                                                        m_asr_model_name;
        std::shared_ptr<nvrpc::client::Executor>                           m_client_executor;
        std::vector<std::shared_ptr<nvidia::riva::asr::RivaSpeechRecognition::Stub>> m_asr_stubs;
        std::vector<std::shared_ptr<nvidia::riva::nlp::RivaLanguageUnderstanding::Stub>> m_nlp_stubs;
        std::vector<std::shared_ptr<nvidia::riva::tts::RivaSpeechSynthesis::Stub>> m_tts_stubs;
    };

} // namespace demo
