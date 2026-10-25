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

#include <cxx/spirv_environment.h>

#include <algorithm>
#include <string>

namespace cxx {

namespace {

struct EnvironmentName {
  SPIRVEnvironment environment;
  std::string_view name;
};

constexpr EnvironmentName kEnvironmentNames[] = {
    {SPIRVEnvironment::kVulkan1_1, "vulkan1.1"},
    {SPIRVEnvironment::kVulkan1_2, "vulkan1.2"},
    {SPIRVEnvironment::kVulkan1_3, "vulkan1.3"},
    {SPIRVEnvironment::kVulkan1_4, "vulkan1.4"},
    {SPIRVEnvironment::kWebGPU, "webgpu"},
};

[[nodiscard]] auto isUnnamed(const std::string& component) -> bool {
  return component.empty() || component == "unknown";
}

}  // namespace

auto spirvEnvironmentOf(const Triple& triple)
    -> std::optional<SPIRVEnvironment> {
  if (!triple.isSPIRV()) return std::nullopt;
  if (!isUnnamed(triple.environmentName())) return std::nullopt;
  if (isUnnamed(triple.osName())) return SPIRVEnvironment::kVulkan1_1;
  auto entry = std::ranges::find(kEnvironmentNames, triple.osName(),
                                 &EnvironmentName::name);
  if (entry == std::ranges::end(kEnvironmentNames)) return std::nullopt;
  return entry->environment;
}

auto to_string(SPIRVEnvironment environment) -> std::string_view {
  auto entry = std::ranges::find(kEnvironmentNames, environment,
                                 &EnvironmentName::environment);
  return entry->name;
}

}  // namespace cxx
