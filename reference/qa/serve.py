# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent import futures

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

import multiprocessing
import time
import math
import logging
import argparse
import grpc

import jarvis_nlp_pb2
import jarvis_nlp_pb2_grpc

def get_args():
    parser = argparse.ArgumentParser(description="Jarvis Question Answering client sample")
    parser.add_argument("--listen", default="[::]:50052", type=str, help="Address to listen to")
    parser.add_argument("--model-name", default="twmkn9/bert-base-uncased-squad2", type=str, help="pretrained HF model to use")
    parser.add_argument("--model-cache", default="/data/models", type=str, help="path to location to store downloaded checkpoints")
    return parser.parse_args()

class JarvisNLPServicer(jarvis_nlp_pb2_grpc.JarvisNLPServicer):
    def __init__(self, model_name, cache=None):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=cache)
        self.model = pipeline('question-answering', 
                              model=model, tokenizer=tokenizer)
        print(f"Model loaded, serving: {model_name}")

    def NaturalQuery(self, request, context):
        """NaturalQuery is a search function that enables querying one or more documents
        or contexts with a query that is written in natural language.
        """
        result = self.model({
            'question': str(request.query),
            'context': str(request.context)
        }, handle_impossible_answer=True)
        response = jarvis_nlp_pb2.NaturalQueryResponse()
        response.results.append(jarvis_nlp_pb2.NaturalQueryResult(answer=result['answer'], score=result['score']))
        return response

def serve(uri="[::]:50051", model="twmkn9/distilbert-base-uncased-squad2", model_cache=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    jarvis_nlp_pb2_grpc.add_JarvisNLPServicer_to_server(
        JarvisNLPServicer(model, cache=model_cache), server)
    server.add_insecure_port(uri,)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig()
    serve(uri=args.listen, model=args.model_name, model_cache=args.model_cache)
