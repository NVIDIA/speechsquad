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

namespace speech_squad {
//
// C++11 doesn't have a sync queue so we implement a simple one.
//
template <typename Item> class SyncQueue {
public:
  SyncQueue() {}

  bool Empty() {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.empty();
  }

  Item Get() {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      cv_.wait(lk, [this] { return !queue_.empty(); });
    }
    auto res = std::move(queue_.front());
    queue_.pop_front();
    return res;
  }

  void Put(const Item &value) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(value);
    }
    cv_.notify_all();
  }

  void Put(Item &&value) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(std::move(value));
    }
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
};

} // namespace speech_squad
