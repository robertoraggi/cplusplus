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

#include <cxx/cxx_fwd.h>

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cxx {

class Preprocessor;
class CommentHandler;

struct SystemInclude {
  std::string fileName;

  SystemInclude() = default;

  explicit SystemInclude(std::string fileName)
      : fileName(std::move(fileName)) {}
};

struct QuoteInclude {
  std::string fileName;

  QuoteInclude() = default;

  explicit QuoteInclude(std::string fileName) : fileName(std::move(fileName)) {}
};

using Include = std::variant<SystemInclude, QuoteInclude>;

struct PendingInclude {
  Preprocessor& preprocessor;
  Include include;
  bool isIncludeNext = false;
  void* loc = nullptr;
  std::function<auto()->std::vector<std::string>> candidates;

  void resolveWith(std::optional<std::string> fileName) const;
};

struct PendingHasIncludes {
  struct Request {
    Include include;
    bool isIncludeNext = false;
    bool& exists;

    void setExists(bool value) const { exists = value; }
  };

  Preprocessor& preprocessor;
  std::vector<Request> requests;
};

struct PendingFileContent {
  Preprocessor& preprocessor;
  std::string fileName;

  void setContent(std::optional<std::string> content) const;
};

struct CanContinuePreprocessing {};

struct ProcessingComplete {};

using PreprocessingState =
    std::variant<PendingInclude, PendingHasIncludes, PendingFileContent,
                 CanContinuePreprocessing, ProcessingComplete>;

}  // namespace cxx
