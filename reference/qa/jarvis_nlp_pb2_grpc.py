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

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import jarvis_nlp_pb2 as jarvis__nlp__pb2


class JarvisNLPStub(object):
  """Jarvis NLP Services implement task-specific APIs for popular NLP tasks including
  intent recognition (as well as slot filling), and entity extraction.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.NaturalQuery = channel.unary_unary(
        '/nvidia.jarvis.nlp.JarvisNLP/NaturalQuery',
        request_serializer=jarvis__nlp__pb2.NaturalQueryRequest.SerializeToString,
        response_deserializer=jarvis__nlp__pb2.NaturalQueryResponse.FromString,
        )


class JarvisNLPServicer(object):
  """Jarvis NLP Services implement task-specific APIs for popular NLP tasks including
  intent recognition (as well as slot filling), and entity extraction.
  """

  def NaturalQuery(self, request, context):
    """NaturalQuery is a search function that enables querying one or more documents
    or contexts with a query that is written in natural language.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_JarvisNLPServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'NaturalQuery': grpc.unary_unary_rpc_method_handler(
          servicer.NaturalQuery,
          request_deserializer=jarvis__nlp__pb2.NaturalQueryRequest.FromString,
          response_serializer=jarvis__nlp__pb2.NaturalQueryResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'nvidia.jarvis.nlp.JarvisNLP', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))