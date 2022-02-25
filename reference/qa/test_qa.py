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

import grpc
import time
import argparse
import fastcounter

import riva_nlp_pb2 as jnlp
import riva_nlp_pb2_grpc as jnlp_srv

def get_args():
  parser = argparse.ArgumentParser(description="Riva Question Answering client sample")
  parser.add_argument("--riva-uri", default="localhost:50052", type=str, help="URI to access Riva server")
  parser.add_argument("--iterations", default=10, type=int, help="number of queries to make")


  return parser.parse_args()

parser = get_args()

grpc_server = parser.riva_uri
channel = grpc.insecure_channel(grpc_server)
riva_nlp = jnlp_srv.RivaNLPStub(channel)

ok_counter = fastcounter.Counter()
bad_counter = fastcounter.Counter()

def process_response(call_future):
  # print(call_future.exception())
  # print(call_future.result())
  if call_future.exception():
    bad_counter.increment()
  else:
    ok_counter.increment()

def run(iterations):
  req = jnlp.NaturalQueryRequest()
  req.query = "who discovered coronavirus?"
  test_context = """
  Coronaviruses were first discovered in the 1930s when an acute respiratory infection of domesticated chickens was shown
  to be caused by infectious bronchitis virus (IBV).[14] Arthur Schalk and M.C. Hawn described in 1931 a new respiratory
  infection of chickens in North Dakota. The infection of new-born chicks was characterized by gasping and listlessness.
  The chicks' mortality rate was 40–90%.[15] Fred Beaudette and Charles Hudson six years later successfully isolated and
  cultivated the infectious bronchitis virus which caused the disease.[16] In the 1940s, two more animal coronaviruses, 
  mouse hepatitis virus (MHV) and transmissible gastroenteritis virus (TGEV), were isolated.[17] It was not realized at
  the time that these three different viruses were related.[18]
  Human coronaviruses were discovered in the 1960s.[19][20] They were isolated using two different methods in the United
  Kingdom and the United States.[21] E.C. Kendall, Malcom Byone, and David Tyrrell working at the Common Cold Unit of the 
  British Medical Research Council in 1960 isolated from a boy a novel common cold virus B814.[22][23][24] The virus was
  not able to be cultivated using standard techniques which had successfully cultivated rhinoviruses, adenoviruses and
  other known common cold viruses. In 1965, Tyrrell and Byone successfully cultivated the novel virus by serially passing
   it through organ culture of human embryonic trachea.[25] The new cultivating method was introduced to the lab by Bertil
  Hoorn.[26] The isolated virus when intranasally inoculated into volunteers caused a cold and was inactivated by ether 
  which indicated it had a lipid envelope.[22][27] Around the same time, Dorothy Hamre[28] and John Procknow at the 
  University of Chicago isolated a novel cold virus 229E from medical students, which they grew in kidney tissue culture.
    The novel virus 229E, like the virus strain B814, when inoculated into volunteers caused a cold and was inactivated by
    ether.[29] """

  req.context = test_context
  for x in range(iterations):
      resp_future = riva_nlp.NaturalQuery.future(req)
      resp_future.add_done_callback(process_response)

if __name__ == '__main__':
  start_time = time.time()
  run(parser.iterations)
  while (ok_counter.value + bad_counter.value) != parser.iterations:
    time.sleep(0.01)
  print(f"total time: {time.time()-start_time}, ok: {ok_counter.value}, fail: {bad_counter.value}")
