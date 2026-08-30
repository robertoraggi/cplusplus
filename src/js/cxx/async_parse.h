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

#include <cxx/parser_fwd.h>
#include <emscripten/val.h>

#include <functional>
#include <string>
#include <string_view>

namespace cxx {
class TranslationUnit;
}

namespace cxx::js {

struct AsyncParseRequest {
  TranslationUnit* unit = nullptr;
  std::string source;
  std::string fileName;
  emscripten::val exists = emscripten::val::undefined();
  emscripten::val readFile = emscripten::val::undefined();
  emscripten::val shouldContinue = emscripten::val::undefined();
  std::function<void(std::string_view, double, bool)> didFinishPhase;
  ParserConfiguration config;
};

[[nodiscard]] auto asyncParse(AsyncParseRequest request) -> emscripten::val;

}  // namespace cxx::js
