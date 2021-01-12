#include <chrono>
#include <grpcpp/impl/codegen/channel_interface.h>

#include "resources.h"

using namespace demo;

#include "jarvis_asr.grpc.pb.h"
#include "jarvis_asr.pb.h"

bool WaitUntilReady(std::shared_ptr<::grpc::ChannelInterface> channel)
{
    std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(10000);

    auto state = channel->GetState(true);
    while (state != GRPC_CHANNEL_READY)
    {
        if (!channel->WaitForStateChange(state, deadline))
        {
            return false;
        }
        state = channel->GetState(true);
    }
    return true;
}

int random_range(int upper_bound)
{
    int divisor = RAND_MAX / upper_bound;
    int value;

    do
    {
        value = rand() / divisor;
    } while (value == upper_bound);

    return value;
}

template <typename T>
std::shared_ptr<T> get_stub(std::vector<std::shared_ptr<T>> stubs)
{
    if (stubs.size() == 1)
    {
        return stubs[0];
    }

    auto n = stubs.size();
    auto r1 = random_range(n);
    auto r2 = random_range(n);

    if (stubs[r1].use_count() < stubs[r2].use_count())
    {
        return stubs[r1];
    }
    return stubs[r2];
}

SpeechSquadResources::SpeechSquadResources(std::string asr_url, std::string nlp_url, std::string tts_url, int threads, int channels, std::string asr_model_name)
    : m_client_executor(std::make_shared<nvrpc::client::Executor>(threads))
{
    LOG(INFO) << "jarvis asr connection established to " << asr_url;
    LOG(INFO) << "jarvis nlp connection established to " << nlp_url;
    LOG(INFO) << "jarvis tts connection established to " << tts_url;

    m_asr_stubs.reserve(channels);
    m_nlp_stubs.reserve(channels);
    m_tts_stubs.reserve(channels);
    m_asr_model_name = asr_model_name;
    for (int i = 0; i < channels; i++)
    {
        auto asr_channel = grpc::CreateChannel(asr_url, grpc::InsecureChannelCredentials());
        auto asr_stub = nvidia::jarvis::asr::JarvisASR::NewStub(asr_channel);
        m_asr_stubs.push_back(std::move(asr_stub));

        auto nlp_channel = grpc::CreateChannel(nlp_url, grpc::InsecureChannelCredentials());
        auto nlp_stub = nvidia::jarvis::nlp::JarvisNLP::NewStub(nlp_channel);
        m_nlp_stubs.push_back(std::move(nlp_stub));

        auto tts_channel = grpc::CreateChannel(tts_url, grpc::InsecureChannelCredentials());
        auto tts_stub = nvidia::jarvis::tts::JarvisTTS::NewStub(tts_channel);
        m_tts_stubs.push_back(std::move(tts_stub));

        DLOG(INFO) << "establishing connections to downstream jarvis services - " << i << " of " << channels;
        CHECK(WaitUntilReady(asr_channel)) << "failed to connect to " << asr_url;
        CHECK(WaitUntilReady(nlp_channel)) << "failed to connect to " << nlp_url;
        CHECK(WaitUntilReady(tts_channel)) << "failed to connect to " << tts_url;
    }
}

SpeechSquadResources::~SpeechSquadResources() {}

std::string SpeechSquadResources::get_model()
{

    return m_asr_model_name;

}

std::unique_ptr<asr_client_t> SpeechSquadResources::create_asr_client(SpeechSquadContext *context)
{
    auto prepare_asr_fn = [asr_stub = get_stub(m_asr_stubs)](::grpc::ClientContext * context, ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(asr_stub->PrepareAsyncStreamingRecognize(context, cq));
    };

    return std::make_unique<asr_client_t>(context, prepare_asr_fn, m_client_executor);
}

std::unique_ptr<nlp_client_t> SpeechSquadResources::create_nlp_client(SpeechSquadContext *context)
{
    auto prepare_nlp_fn = [nlp_stub = get_stub(m_nlp_stubs)](::grpc::ClientContext * context, const nlp_request_t &request,
                                                             ::grpc::CompletionQueue *cq) -> auto
    {
        return std::move(nlp_stub->PrepareAsyncNaturalQuery(context, request, cq));
    };

    return std::make_unique<nlp_client_t>(context, prepare_nlp_fn, m_client_executor);
}

std::unique_ptr<tts_client_t> SpeechSquadResources::create_tts_client(SpeechSquadContext *context)
{
    auto prepare_tts_fn = [tts_stub = get_stub(m_tts_stubs)](::grpc::ClientContext * context, const tts_request_t &request,
                                                             ::grpc::CompletionQueue *cq) -> auto
    {
        return std::move(tts_stub->PrepareAsyncSynthesizeOnline(context, request, cq));
    };

    return std::make_unique<tts_client_t>(context, prepare_tts_fn, m_client_executor);
}
