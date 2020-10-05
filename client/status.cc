/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "status.h"

namespace speech_squad {
const Status Status::Success(Status::Code::SUCCESS);

std::string Status::AsString() const {
  std::string str(CodeString(code_));
  str += ": " + msg_;
  return str;
}

const char *Status::CodeString(const Code code) {
  switch (code) {
  case Status::Code::SUCCESS:
    return "OK";
  case Status::Code::UNKNOWN:
    return "Unknown";
  case Status::Code::INTERNAL:
    return "Internal";
  case Status::Code::NOT_FOUND:
    return "Not found";
  case Status::Code::INVALID_ARG:
    return "Invalid argument";
  case Status::Code::UNAVAILABLE:
    return "Unavailable";
  case Status::Code::UNSUPPORTED:
    return "Unsupported";
  case Status::Code::ALREADY_EXISTS:
    return "Already exists";
  default:
    break;
  }

  return "<invalid code>";
}

} // namespace speech_squad
