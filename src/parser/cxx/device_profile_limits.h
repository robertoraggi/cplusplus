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
#include <optional>
#include <string_view>

namespace cxx {

inline constexpr std::string_view kVulkan11F32ProfileName =
    "vulkan1.1-f32/as1-device-as2-constant/index32/v2";

inline constexpr std::uint32_t kDeviceAddressSpace = 1;
inline constexpr std::uint32_t kConstantAddressSpace = 2;
inline constexpr std::uint32_t kThreadgroupAddressSpace = 3;
inline constexpr std::uint32_t kMaxBufferBinding = 31;
inline constexpr std::uint32_t kBufferDescriptorSet = 0;

struct DeviceProfileLimits {
  std::uint64_t maxWorkgroupSizeX = 0;
  std::uint64_t maxWorkgroupSizeY = 0;
  std::uint64_t maxWorkgroupSizeZ = 0;
  std::uint64_t maxWorkgroupInvocations = 0;
  std::uint64_t maxWorkgroupStorageBytes = 0;

  [[nodiscard]] constexpr auto admitsWorkgroup(std::uint64_t x, std::uint64_t y,
                                               std::uint64_t z) const -> bool {
    if (!admitsDimension(x, maxWorkgroupSizeX)) return false;
    if (!admitsDimension(y, maxWorkgroupSizeY)) return false;
    if (!admitsDimension(z, maxWorkgroupSizeZ)) return false;
    return x * y * z <= maxWorkgroupInvocations;
  }

 private:
  [[nodiscard]] static constexpr auto admitsDimension(std::uint64_t size,
                                                      std::uint64_t limit)
      -> bool {
    return size != 0 && size <= limit;
  }
};

inline constexpr DeviceProfileLimits kVulkan11F32Limits{
    .maxWorkgroupSizeX = 128,
    .maxWorkgroupSizeY = 128,
    .maxWorkgroupSizeZ = 64,
    .maxWorkgroupInvocations = 128,
    .maxWorkgroupStorageBytes = 16384,
};

[[nodiscard]] constexpr auto deviceProfileLimits(std::string_view profile)
    -> std::optional<DeviceProfileLimits> {
  if (profile == kVulkan11F32ProfileName) return kVulkan11F32Limits;
  return std::nullopt;
}

}  // namespace cxx
