
#include "context.h"

#include <glog/logging.h>

using Input = SpeechSquadInferRequest;
using Output = SpeechSquadInferResponse;

using namespace demo;

void SpeechSquadContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    DCHECK(m_state == State::Uninitialized);
    m_state = State::Initialized;

    // if we have async clients with outstanding events registered to a cq
    // we must block stream from completing
    BlockFinish();

    // asr client
    m_asr_client = GetResources()->create_asr_client(this);

    // set stream
    m_stream = stream;

    // set initial state
    m_first_tts_response = true;

    m_should_cancel = false;
}

void SpeechSquadContext::OnContextReset()
{
    VLOG(1) << this << ": reseting context";
    m_state = State::Uninitialized;
    m_asr_client.reset();
    m_nlp_client.reset();
    m_tts_client.reset();
    m_timings.clear();
    m_stream = nullptr;
    m_first_tts_response = true;
    m_should_cancel = false;
    m_debug_tts = false;
}

void SpeechSquadContext::RequestReceived(Input &&input, std::shared_ptr<ServerStream> stream)
{
    DCHECK_NOTNULL(stream);

    if (input.has_speech_squad_config())
    {
        if (m_state != State::Initialized)
        {
            LOG(ERROR) << "squad stream received a request with an unexpected message - expected a config";
            m_should_cancel = true;
            m_asr_client->Cancel();
            return;
        }
        m_state = State::ReceivingAudio;

        // save the server stream to the context so the tts callback handler can
        // forwards tts frames back the client
        DCHECK_NOTNULL(stream);
        m_stream = stream;

        VLOG(1) << "speech squad stream initialized";

        // extract the context from the initial request
        m_context = input.speech_squad_config().squad_context();

        // initialize the jarvis async asr stream with the input audio config
        DCHECK(input.speech_squad_config().input_audio_config().encoding() == AudioEncoding::LINEAR_PCM);

        // asr configure request
        asr_request_t request;
        auto streaming_config = request.mutable_streaming_config();
        streaming_config->set_interim_results(false);
        auto config = streaming_config->mutable_config();
        config->set_encoding(AudioEncoding::LINEAR_PCM);
        config->set_sample_rate_hertz(input.speech_squad_config().input_audio_config().sample_rate_hertz());
        config->set_language_code(input.speech_squad_config().input_audio_config().language_code());
        config->set_audio_channel_count(input.speech_squad_config().input_audio_config().audio_channel_count());
        config->set_max_alternatives(1);
        config->set_enable_word_time_offsets(false);
        config->set_enable_automatic_punctuation(false);
        config->set_enable_separate_recognition_per_channel(false);
        config->set_model(GetResources()->get_model());

        DVLOG(2) << "sample rate: " << input.speech_squad_config().input_audio_config().sample_rate_hertz();
        DVLOG(2) << "channels   : " << input.speech_squad_config().input_audio_config().audio_channel_count();
        DVLOG(2) << "language   : " << input.speech_squad_config().input_audio_config().language_code();

        // save tts config for when we issue the tts request
        m_tts_config = input.speech_squad_config().output_audio_config();

        // write/send the initial request to jarvis asr
        VLOG(1) << this << ": initiating jarvis asr";
        m_asr_client->Write(std::move(request));
    }
    else
    {
        // forward audio from speech squad input to jarvis asr
        if (m_state != State::ReceivingAudio)
        {
            LOG(ERROR) << "squad stream received an unexpected request without a configuration message";
            m_should_cancel = true;
            m_asr_client->Cancel();
            return;
        }

        VLOG(2) << this << ": forwaring audio to jarvis asr; bytes=" << input.audio_content().size();
        asr_request_t request;
        request.set_audio_content(input.audio_content());
        m_asr_client->Write(std::move(request));
    }
}

void SpeechSquadContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    if (m_state != State::ReceivingAudio)
    {
        LOG(ERROR) << "received WritesDone from client before put into State::ReceivingAudio";
        m_should_cancel = true;
        m_asr_client->Cancel();
        return;
    }
    m_state = State::AudioUploadComplete;

    VLOG(1) << this << ": speech squad client closed asr upload stream; closing jarvis asr upload";

    // close upload to jarvis asr stream
    m_asr_writes_done = std::chrono::high_resolution_clock::now();
    m_asr_client->CloseWrites();
}

void SpeechSquadContext::ASRCallbackOnResponse(asr_response_t &&response)
{
    if (response.results_size() == 0)
    {
        DVLOG(2) << "no results received";
        return;
    }

    const auto &result = response.results(0);
    if (!result.is_final())
    {
        DVLOG(1) << "received non final results";
        return;
    }

    m_asr_on_complete = std::chrono::high_resolution_clock::now();

    if (result.alternatives_size() == 0)
    {
        LOG(ERROR) << "resutls final, but no transcript";
        m_asr_client->Cancel();
        return;
    }
    const auto &top_candidate = result.alternatives(0);

    m_question = top_candidate.transcript() + "?";

    VLOG(1) << this << ": jarvis asr result " << std::endl
            << "q: " << m_question << "; confidence=" << top_candidate.confidence();
}

void SpeechSquadContext::ASRCallbackOnFinish(const ::grpc::Status &status, const meta_data_t &meta_data)
{
    VLOG(1) << this << ": asr stream completed with status " << (status.ok() ? "OK" : "CANCELLED");
    if (!status.ok())
    {
        LOG(ERROR) << "asr error detected - issuing cancellation on squad stream";
        DCHECK_NOTNULL(m_stream);
        // there are no client cq events registers
        // we can now unblock and cancel the server stream
        if (!m_stream->IsConnected())
        {
            LOG(ERROR) << "SHOWSTOPPER: stream callback are disconnected from the server context";
        }
        m_stream->UnblockFinish();
        m_stream->CancelStream();
        return;
    }

    VLOG(1) << this << ": question = " << m_question;

    ExtractTimings(meta_data);

    nlp_request_t request;
    request.set_context(m_context);
    request.set_query(m_question);

    VLOG(1) << this << ": issuing nlp request";
    VLOG(3) << this << ": context = " << m_context;

    // nlp client
    m_nlp_client = GetResources()->create_nlp_client(this);

    m_nlp_start = std::chrono::high_resolution_clock::now();
    m_nlp_client->Write(std::move(request));
}

void SpeechSquadContext::NLPCallbackOnResponse(const nlp_response_t &response)
{
    if (response.results_size() == 0)
    {
        LOG(ERROR) << "nlp did not return any results";
        if (!m_stream->IsConnected())
        {
            LOG(ERROR) << "SHOWSTOPPER: stream callback are disconnected from the server context";
        }
        m_stream->UnblockFinish();
        m_stream->CancelStream();
        return;
    }

    m_nlp_finish = std::chrono::high_resolution_clock::now();

    VLOG(3) << response.DebugString();

    const auto &top_result = response.results(0);
    if (top_result.answer().size())
    {
        m_answer = top_result.answer();
        m_nlp_score = top_result.score();
    }
    else
    {
        m_answer = "";
        m_nlp_score = 0;
    }

    VLOG(1) << this << ": nlp complete." << std::endl
            << "q: " << m_question << std::endl
            << "a: " << m_answer << "; score=" << m_nlp_score;

    // write back to the squad client the initial inference meta data
    SpeechSquadInferResponse squad_response;
    auto infer_metadata = squad_response.mutable_metadata();
    infer_metadata->set_squad_question(m_question);
    infer_metadata->set_squad_answer(m_answer);
    m_stream->WriteResponse(std::move(squad_response));

    // tts client
    m_tts_client = GetResources()->create_tts_client(this);

    // setup the tts request
    tts_request_t request;
    request.set_text((m_answer.size() ? m_answer : "No answer"));
    request.set_encoding(nvidia::jarvis::tts::AudioEncoding::LINEAR_PCM);
    request.set_sample_rate_hz(22050);
    request.set_language_code(m_tts_config.language_code());
    request.set_voice_name("ljspeech");
    VLOG(1) << this << ": sending tts request";

    m_first_tts_response = true;
    m_tts_start = std::chrono::high_resolution_clock::now();
    m_tts_client->Write(std::move(request));
}

void SpeechSquadContext::NLPCallbackOnComplete(const ::grpc::Status &status, const meta_data_t &meta_data)
{
    VLOG(1) << this << ": nlp stream completed with status " << (status.ok() ? "OK" : "CANCELLED");
    if (!status.ok())
    {
        LOG(ERROR) << "nlp error detected - issuing cancellation on squad stream";
        DCHECK_NOTNULL(m_stream);
        if (!m_stream->IsConnected())
        {
            LOG(ERROR) << "SHOWSTOPPER: stream callback are disconnected from the server context";
        }
        m_stream->UnblockFinish();
        m_stream->CancelStream();
        return;
    }
    ExtractTimings(meta_data);
}

void SpeechSquadContext::TTSCallbackOnResponse(tts_response_t &&tts_response)
{
    if (m_first_tts_response)
    {
        VLOG(1) << this << ": relaying first tts response";
        m_tts_first_packet = std::chrono::high_resolution_clock::now();
        m_first_tts_response = false;
    }
    if (!tts_response.audio().size())
    {
        LOG(WARNING) << this << ": received 0 bytes of tts audio";
        m_debug_tts = true;
        return;
    }
    VLOG(3) << "forwarding tts response to client";
    SpeechSquadInferResponse squad_response;
    squad_response.set_audio_content(tts_response.audio());
    m_stream->WriteResponse(std::move(squad_response));
}

void SpeechSquadContext::TTSCallbackOnComplete(const ::grpc::Status &status, const meta_data_t &meta_data)
{
    VLOG(1) << this << ": tts stream completed with status " << (status.ok() ? "OK" : "CANCELLED");

    if (m_debug_tts)
    {
        LOG(WARNING) << this << ": tts stream completed with status " << (status.ok() ? "OK" : "CANCELLED");
    }

    // if we got here, all async clients have finished
    if (!m_stream->IsConnected())
    {
        LOG(ERROR) << "SHOWSTOPPER: stream callback are disconnected from the server context";
    }
    m_stream->UnblockFinish();

    if (!status.ok())
    {
        LOG(ERROR) << "tts error detected - issuing cancellation on squad stream";
        DCHECK_NOTNULL(m_stream);
        m_stream->CancelStream();
        return;
    }

    // get tts meta data
    ExtractTimings(meta_data);

    // send component timings
    SpeechSquadInferResponse response;

    // jarvis latencies extracted from trailing meta data
    auto timings = response.mutable_metadata()->mutable_component_timing();
    for (auto it = m_timings.cbegin(); it != m_timings.cend(); it++)
    {
        (*timings)[it->first] = it->second;
    }

    auto time_in_ms = [](std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        float ms = (float)us / 1000.;
        return ms;
    };

    // speech squad measured latencies
    (*timings)["tracing.speech_squad.asr_latency"] = time_in_ms(m_asr_writes_done, m_asr_on_complete);
    (*timings)["tracing.speech_squad.nlp_latency"] = time_in_ms(m_nlp_start, m_nlp_finish);
    (*timings)["tracing.speech_squad.tts_latency"] = time_in_ms(m_tts_start, m_tts_first_packet);

    m_stream->WriteResponse(std::move(response));
    m_stream->FinishStream();
}

void SpeechSquadContext::ExtractTimings(const meta_data_t &meta_data)
{
    for (auto it = meta_data.cbegin(); it != meta_data.cend(); it++)
    {
        VLOG(2) << this << ": meta_data - " << it->first << ": " << it->second;
        if (!it->first.starts_with("tracing."))
        {
            continue;
        }
        std::string key(it->first.cbegin(), it->first.cend());
        std::string val(it->second.cbegin(), it->second.cend());
        m_timings.insert(std::pair<std::string, float>(key, std::atof(val.c_str())));
    }
}
