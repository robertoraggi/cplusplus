// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/token.h>

#include <string>

namespace cxx {

enum class Severity { Message, Warning, Error, Fatal };

class Diagnostic {
  std::string message_;
  Token token_;
  Severity severity_ = Severity::Message;

 public:
  Diagnostic() = default;

  Diagnostic(const Diagnostic&) = default;
  auto operator=(const Diagnostic&) -> Diagnostic& = default;

  Diagnostic(Diagnostic&&) = default;
  auto operator=(Diagnostic&&) -> Diagnostic& = default;

  Diagnostic(Severity severity, const Token& token, std::string message);

  [[nodiscard]] auto severity() const -> Severity { return severity_; }

  [[nodiscard]] auto token() const -> const Token& { return token_; }

  [[nodiscard]] auto message() const -> const std::string& { return message_; }
};

}  // namespace cxx
