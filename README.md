# SpeechSquad Inference Benchmark

## Overview
The SpeechSquad benchmark is a benchmark for measuring the performance of systems hosting conversational AI services that combine automatic speech recognition, natural language processing, and speech synthesis.

A typical conversational AI interaction involves multiple clients capturing a spoken question from a microphone, sending the spoken question to the server for processing and receiving synthesized audio with the answer to the question. This benchmark generates a workload that closely matches real-world deployments of this type.

By measuring the latency, throughput, accuracy, and subjective quality of the entire system, SpeechSquad allows for complex optimizations across any part of the hardware, software, networking, or computing stack.  It also provides a tool for measuring trade-offs between optimizations that may impact performance or accuracy of only one portion of the computation.  For example, it might be worth sacrificing ASR accuracy if the NLP portion is still able to respond with the correct answer most of the time.  Real-world conversational AI systems often exhibit these types of complex trade-offs and interactions between different components.  With SpeechSquad, we aim to provide a structured way to study and compare such system and algorithm-level design choices.

## Benchmark Definition
In the SpeechSquad benchmark, questions are taken from the SQuAD 2.0 evaluation dataset. Audio recordings of those questions are provided. The load generator simulates the capture of the audio from a microphone by sending audio chunks containing chunk_size milliseconds of audio every chunk_size milliseconds (chunk_size can be varied, a typical value would be 200ms). For each question, the number of audio chunks sent by the client is simply the audio length divided by chunk_size.

The server must then process the incoming audio chunks and find the answer to the spoken question from a given text. 

That answer must then be synthesized into an audio clip, which can be broken up into multiple audio chunks. Thus for each question, there are multiple requests being sent (corresponding to multiple audio chunks for the question) and there can be multiple responses from the server (corresponding to multiple audio chunks containing the answer to the question).

### Performance metrics
Latency is defined as the time between the last audio chunk being sent to the server and the first response from the server, which contains a part (or all) of the synthesized audio with the answer to the question.

A submission should report the following results:
- Latency statistics as a function of the number of concurrent streams.  Latency statistics include average latency, P50, P90, P95, and P99.
- F1 and Accuracy scores for the SQuAD results (evaluated based on the text that is received from the NLP output).  Since we only use the public SQuAD evaluation set and not the held-out test set, these scores are approximate only.  However, this provides a measurement of the accuracy of the combined end-to-end system and is sufficient for benchmarking tradeoffs between speed and accuracy.
- Percent of audio stream responses that have missing or late packets.
- (Optional) Mean Opinion Score (MOS) for the resulting TTS audio recordings.  Since MOS requires subjective evaluation, we provide scripts and templates a submitter could use on a service like Amazon Mechanical Turk.

Details on these measurements are below.

#### Latency Statistics
Submissions should report latency statistics aggregated across all clients/streams and fill the table below, up to the desired scale.  It is reasonable to only fill in some rows of the table, based on the scale at which you wish to measure a system.


| # of concurrent streams | Avg latency | P50 latency | P90 latency | P95 latency | P99 latency |
|-------------------------|-------------|-------------|-------------|-------------|-------------|
| 1                       |             |             |             |             |             |
| 2                       |             |             |             |             |             |
| 4                       |             |             |             |             |             |
| 8                       |             |             |             |             |             |
| 16                      |             |             |             |             |             |
| 32                      |             |             |             |             |             |
| 64                      |             |             |             |             |             |
| 128                     |             |             |             |             |             |
| 256                     |             |             |             |             |             |
| 512                     |             |             |             |             |             |
| 1024                    |             |             |             |             |             |
| 2048                    |             |             |             |             |             |
| 4096                    |             |             |             |             |             |
| 8192                    |             |             |             |             |             |


#### Accuracy
Participants must report the exact and F1 scores obtained on the SQUAD benchmark using the evaluation script provided here: https://rajpurkar.github.io/SQuAD-explorer/

#### Missing or late audio packets
Synthesized audio responses should not have gaps. In other words, the client application should have received the 2nd synthesized audio chunk before it is done playing the 1st synthesized audio chunks.  Percentage of all response streams that have any gap should be reported.

#### Audio quality
Optionally, a participant can evaluate the subjective quality of the synthesized TTS response.  We are planning to provide scripts and instructions a submitter can use to evaluate Mean Opinion Score (MOS) using Amazon Mechanical Turk.

## Server specifications
Participants are free to choose the scale at which they want to run the benchmark. Server specifications should be provided when submitting results.

## Alpha Release
Please note that this is a public alpha release, so the code has undergone limited testing.  Please file issues or ask questions about the code using the github issues tracker.

In the current release of SpeechSquad, NVIDIA is providing the following:
- [Performance measurement client](https://github.com/nvidia/speechsquad/tree/master/client) that is capable of sending synthetic requests at different scales and measuring response latency, throughput, and accuracy.
- [SpeechSquad service implementation](https://github.com/nvidia/speechsquad/tree/master/server) for coordinating across microservices.
- CPU microservice reference implementations:
    - [ASR gRPC service](https://github.com/nvidia/speechsquad/tree/master/reference/asr) implemented with Kaldi running TDNN LibriSpeech model.
    - [Question Answering gRPC service](https://github.com/nvidia/speechsquad/tree/master/reference/qa) implemented with Huggingface Transformers DistilBERT trained on SQUAD 2.0.
    - For Text-to-Speech, we recommend [MaryTTS](http://mary.dfki.de/) but are not providing an implementation at this time.

Reference implementation with GPU acceleration using NVIDIA Riva will be available soon. Learn more about Riva and signup to be notified of public beta access at https://developer.nvidia.com/nvidia-riva.

Not currently provided:
- There is currently no leaderboard or other official list of scores.
- Evaluation tools to compute MOS or other quality measurements of the TTS responses.
- The SpeechSquad dataset is not currently available but we are working to make it so.  We plan to provide a small sample “toy” dataset suitable for performance analysis only soon.
