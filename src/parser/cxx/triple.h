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

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace cxx {

enum class TripleArch {
  kUnknown,
  kAArch64,
  kX86_64,
  kWasm32,
};

enum class TripleVendor {
  kUnknown,
  kApple,
  kPC,
};

enum class TripleOS {
  kUnknown,
  kDarwin,
  kMacOSX,
  kIOS,
  kLinux,
  kWindows,
  kWasi,
  kWasiPreview1,
  kWasiPreview2,
  kEmscripten,
};

enum class TripleEnvironment {
  kUnknown,
  kGNU,
  kMSVC,
  kMinGW,
  kThreads,
};

class Triple {
 public:
  Triple() = default;
  explicit Triple(std::string_view text);

  [[nodiscard]] auto arch() const -> TripleArch { return arch_; }
  [[nodiscard]] auto vendor() const -> TripleVendor { return vendor_; }
  [[nodiscard]] auto os() const -> TripleOS { return os_; }
  [[nodiscard]] auto environment() const -> TripleEnvironment {
    return environment_;
  }

  [[nodiscard]] auto archName() const -> const std::string& {
    return names_[0];
  }
  [[nodiscard]] auto vendorName() const -> const std::string& {
    return names_[1];
  }
  [[nodiscard]] auto osName() const -> const std::string& { return names_[2]; }
  [[nodiscard]] auto environmentName() const -> const std::string& {
    return names_[3];
  }

  [[nodiscard]] auto isDarwin() const -> bool;

  [[nodiscard]] auto isWebAssembly() const -> bool;

  [[nodiscard]] auto osVersion() const -> std::optional<std::pair<int, int>>;

  [[nodiscard]] auto str() const -> std::string;

 private:
  static constexpr int kComponentCount = 4;

  TripleArch arch_ = TripleArch::kUnknown;
  TripleVendor vendor_ = TripleVendor::kUnknown;
  TripleOS os_ = TripleOS::kUnknown;
  TripleEnvironment environment_ = TripleEnvironment::kUnknown;
  std::string names_[kComponentCount];
  int printedComponents_ = 1;
};

[[nodiscard]] auto to_string(TripleArch arch) -> std::string_view;

}  // namespace cxx
