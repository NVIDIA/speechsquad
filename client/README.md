# SpeechSquad Load Generator Client

## Overview
The SpeechSquad Load Generator Client is used to generate a consistent real-time load to SpeechSquad service for benchmarking purposes.

It takes the squad context and the audio recordings of questions taken from the SQUAD 2.0 evaluation dataset to query the service. The client simulates the capture of the audio from a microphone by sending audio chunks containing X milliseconds of audio every X milliseconds. For each question, the number of audio chunks sent by the client is simply the audio length divided by the chunk size X. The user-specified `--num_parallel_requests` value is used to load the server with that many concurrent audio streams. At the end of the run client generates a report of latencies and throughput observed by client and the associated service.

## Quick Start

1. The client is built inside a docker container. Run the below command from the root of the project to start the build project.
 
`docker build -f Dockerfile.client -t SpeechSquadClient .`

2. Create and move to a work directory.

3. Download the squad dataset json file from below to the working directory.

`wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json`

4. Copy the audio recordings of the questions within the working directory.

5. Create a input json file where each entry maps a question id to the path to audio files. For example, the content of this file can look like:

```
 {"id":"56ddde6b9a695914005b9628","audio_filepath": "/work/test_files/speech_squad/speech_squad1.wav"}
 {"id":"56ddde6b9a695914005b962a","audio_filepath": "/work/test_files/speech_squad/speech_squad3.wav"}
 {"id":"56ddde6b9a695914005b962b","audio_filepath": "/work/test_files/speech_squad/speech_squad4.wav"}
```

6. Start the SpeechSquadClient container. Note, that it is not necessary to have squad dataset, audio files and input json file in same directory. As long as the appropriate locations are loaded up as volumes and accessible inside container, it should just work fine.

`docker run --rm -ti -v $PWD:/work --workdir /work --gpus all SpeechSquadClient:latest`

7. Once inside the container, client can be started as follows

`squad_perf_client --squad_questions_json=<path to input json> --squad_dataset_json=<path to squad dataset> --speech_squad_uri=<speech_squad_uri>`

### Client Command-Line Parameters

The client can be configured through a set of parameters that define its behavior. To see the full list of available options and their descriptions, invoke `squad_perf_client` without any options. Look into the `main.cc` for detailed explanation and default behavior of the client.           



## Muti-Process Support
The client supports the MPI framework to generate load using multiple processes. Users can invoke the squad_perf_client using `mpirun -n 4 squad_perf_client ...` to execute 4 processes to load the server. It is advisable to use multi-process mode when generating load with large number of concurrent streams.
