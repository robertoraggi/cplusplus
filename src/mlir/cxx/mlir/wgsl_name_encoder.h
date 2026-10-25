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

#include <llvm/ADT/StringRef.h>

#include <cstdint>
#include <string>

namespace cxx {

class WGSLNameEncoder {
 public:
  enum class Temporary {
    kValue,
    kResult,
    kSnapshot,
    kInsertedValue,
    kPrivateVariable,
    kInduction,
    kLoopValue,
    kOldValue,
    kExchanged,
    kAttempt,
  };

  enum class Builtin {
    kGlobalId,
    kLocalId,
    kWorkgroupId,
  };

  [[nodiscard]] auto encodeTemporary(Temporary kind) -> std::string;
  [[nodiscard]] auto encodeRecord() -> std::string;

  [[nodiscard]] static auto encodeField(std::uint32_t position) -> std::string;
  [[nodiscard]] static auto encodeArgument(std::uint32_t index) -> std::string;
  [[nodiscard]] static auto encodeBinding(std::uint32_t resource)
      -> std::string;
  [[nodiscard]] static auto encodeWorkgroupVariable(std::uint32_t index)
      -> std::string;
  [[nodiscard]] static auto encodeBuiltin(Builtin builtin) -> llvm::StringRef;
  [[nodiscard]] static auto encodeFunction(llvm::StringRef name) -> std::string;
  [[nodiscard]] static auto isUsableEntryName(llvm::StringRef name) -> bool;

 private:
  std::uint32_t nextTemporary_ = 0;
  std::uint32_t nextRecord_ = 0;
};

}  // namespace cxx
