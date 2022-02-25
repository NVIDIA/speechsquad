# ASR Server using Kaldi : Open-source CPU Reference Implementation

This directory contains reference implementation of a simple GRPC Speech Recognition Server utilizing popular Kaldi C++ ASR Library on CPU.
The server is multi-threaded (each request executes on its own thread), implements API described in `riva_asr.proto`.

## Building Container Image (Ubuntu Linux)

Recommended way to build ASR server is with Dockerfile:

```
docker build -t cpu-asr-ref .
```

Docker build uses Kaldi libs built with MKL from the latest official Kaldi container (kaldiasr/kaldi:latest).
MKL boosts Kaldi CPU performance on Intel CPUs. 

Building outside of Docker and on other systems with Bazel should be possible, but not tested - check Dockerfile and BUILD for pre-requisites and proceed on your own risk.


## Running ASR server

Run ASR server from the container like this:
```
> docker run --ipc=host --rm --network host --init -it cpu-asr-ref kaldi_cpu_asr_server
```

(Use `--init` Docker option with `-it` if you want to be able to kill the server with Ctrl-C) 

### ASR server options

Use standard syntax to check server command-line options:
```
> docker run --ipc=host --rm --network host -it cpu-asr-ref kaldi_cpu_asr_server --help
...
  Flags from server.cc:
  -model_options (Kaldi model options) type: string default: ""
  -model_path (Path to trained Kaldi model) type: string default: "/data/models/LibriSpeech"
  -uri (URI this server should bind to) type: string default: "0.0.0.0:50051"		    
```

  All flags are optional. The default Kaldi model to use is pre-trained LibriSpeech model downloaded by `scripts/download_kaldi_data.sh` into `/data/models/LibriSpeech` during Docker build.

### Modifying Kaldi Model options

To modify Kaldi model options, either pass an alternative config file, or individual options:  
  ```
  > docker run --ipc=host --rm --network host -it cpu-asr-ref
  (cont)> kaldi_cpu_asr_server --model_options="--config=/work/kaldi_asr.conf"
  (cont)> kaldi_cpu_asr_server --model_options="--help"
  ```

To run server with different Kaldi model:
  ```
  (cont)> kaldi_cpu_asr_server --model_path <path_to_your_model>
  ```

