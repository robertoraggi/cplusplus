// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/source_location.h>

#include <string>

namespace cxx {

class TranslationUnit;

enum struct Severity { Message, Warning, Error, Fatal };

class Diagnostic {
  TranslationUnit* unit_;
  std::string message_;
  std::string fileName_;
  SourceLocation location_;
  Severity severity_ = Severity::Message;
  unsigned line_ = 0;
  unsigned column_ = 0;

 public:
  Diagnostic() = default;

  Diagnostic(const Diagnostic&) = default;
  Diagnostic& operator=(const Diagnostic&) = default;

  Diagnostic(Diagnostic&&) = default;
  Diagnostic& operator=(Diagnostic&&) = default;

  Diagnostic(TranslationUnit* unit, Severity severity,
             const SourceLocation& location, std::string message);

  Severity severity() const { return severity_; }

  const std::string& fileName() const { return fileName_; }

  const std::string& message() const { return message_; }

  unsigned line() const { return line_; }

  unsigned column() const { return column_; }

  const SourceLocation& location() const { return location_; }
};

}  // namespace cxx
