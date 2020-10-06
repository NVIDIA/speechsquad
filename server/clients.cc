
#include "clients.h"
#include "context.h"

using namespace demo;

void ASRClient::CallbackOnResponseReceived(asr_response_t &&response)
{
    DCHECK_NOTNULL(m_context);
    m_context->ASRCallbackOnResponse(std::move(response));
}

void ASRClient::CallbackOnComplete(const ::grpc::Status &status)
{
    DCHECK_NOTNULL(m_context);
    auto meta_data = GetClientContext().GetServerTrailingMetadata();
    m_context->ASRCallbackOnFinish(status, meta_data);
}

void NLPClient::CallbackOnResponseReceived(nlp_response_t &&response)
{
    DCHECK_NOTNULL(m_context);
    m_context->NLPCallbackOnResponse(std::move(response));
}

void NLPClient::CallbackOnComplete(const ::grpc::Status &status)
{
    DCHECK_NOTNULL(m_context);
    auto meta_data = GetClientContext().GetServerTrailingMetadata();
    m_context->NLPCallbackOnComplete(status, meta_data);
}

void TTSClient::CallbackOnResponseReceived(tts_response_t &&response)
{
    DCHECK_NOTNULL(m_context);
    m_context->TTSCallbackOnResponse(std::move(response));
}

void TTSClient::CallbackOnComplete(const ::grpc::Status &status)
{
    DCHECK_NOTNULL(m_context);
    auto meta_data = GetClientContext().GetServerTrailingMetadata();
    m_context->TTSCallbackOnComplete(status, meta_data);
}