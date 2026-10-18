// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>

namespace cxx {

inline constexpr std::uint32_t kDeviceAddressSpace = 1;
inline constexpr std::uint32_t kConstantAddressSpace = 2;
inline constexpr std::uint32_t kMaxBufferBinding = 31;
inline constexpr std::uint32_t kBufferDescriptorSet = 0;

struct DeviceProfileLimits {
  std::uint64_t maxWorkgroupSizeX = 0;
  std::uint64_t maxWorkgroupSizeY = 0;
  std::uint64_t maxWorkgroupSizeZ = 0;
  std::uint64_t maxWorkgroupInvocations = 0;
};

inline constexpr DeviceProfileLimits kVulkan11F32Limits{
    .maxWorkgroupSizeX = 128,
    .maxWorkgroupSizeY = 128,
    .maxWorkgroupSizeZ = 64,
    .maxWorkgroupInvocations = 128,
};

}  // namespace cxx
