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

/**
 * @file WaveFileWriter.hpp
 * @brief The WaveFile writer.
 * @author Dominique LaSalle <dlasalle@nvidia.com>
 * Copyright 2019, NVIDIA Corporation. All rights reserved.
 * @version 1
 * @date 2019-07-01
 */

#ifndef TT2I_WAVEFILEWRITER_HPP
#define TT2I_WAVEFILEWRITER_HPP

#include <cstddef>
#include <string>

class WaveFileWriter {
public:
  /**
   * @brief Write a mono sample data to a WAV file.
   *
   * @param filename The file name.
   * @param frequency The sample frequency.
   * @param data The raw data.
   * @param numSamples The number of samples.
   */
  static void write(const std::string &filename, int frequency,
                    const float *data, size_t numSamples);
};

#endif
