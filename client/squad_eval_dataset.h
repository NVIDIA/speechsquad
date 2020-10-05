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

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rapidjson/document.h"
#include "status.h"

namespace speech_squad {
class SquadEvalDataset {
public:
  Status LoadFromJson(const std::string &json_filepath);
  Status GetQuestion(const std::string &id, std::string *question);
  Status GetQuestionContext(const std::string &id, std::string *context);

private:
  std::map<std::string, std::string> questions_;
  std::vector<std::shared_ptr<std::string>> contexts_;
  std::map<std::string, std::shared_ptr<std::string>> question_contexts_;
};

} // namespace speech_squad
