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

#pragma once

#include <string>

namespace speech_squad {
class Status {
public:
  // The status codes
  enum class Code {
    SUCCESS,
    UNKNOWN,
    INTERNAL,
    NOT_FOUND,
    INVALID_ARG,
    UNAVAILABLE,
    UNSUPPORTED,
    ALREADY_EXISTS
  };

public:
  // Construct a status from a code with no message.
  explicit Status(Code code = Code::SUCCESS) : code_(code) {}

  // Construct a status from a code and message.
  explicit Status(Code code, const std::string &msg) : code_(code), msg_(msg) {}
  // Construct a status from a message.
  explicit Status(const std::string &msg) : code_(Code::UNKNOWN), msg_(msg) {}

  // Convenience "success" value. Can be used as Status::Success to
  // indicate no error.
  static const Status Success;

  // Return the code for this status.
  Code StatusCode() const { return code_; }

  // Return the message for this status.
  const std::string &Message() const { return msg_; }

  // Return true if this status indicates "ok"/"success", false if
  // status indicates some kind of failure.
  bool IsOk() const { return code_ == Code::SUCCESS; }

  // Return the status as a string.
  std::string AsString() const;

  // Return the constant string name for a code.
  static const char *CodeString(const Code code);

private:
  Code code_;
  std::string msg_;
};

} // namespace speech_squad
