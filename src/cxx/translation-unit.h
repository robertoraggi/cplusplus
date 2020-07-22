// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "token.h"

namespace cxx {

class TranslationUnit {
  Control* control_;
  std::vector<Token> tokens_;
  std::vector<int> lines_;
  std::string yyfilename;
  std::string yytext;
  std::string yycode;
  const char* yyptr = nullptr;
  bool fatalErrors_ = false;

 public:
  TranslationUnit(Control* control) : control_(control) {}
  ~TranslationUnit() = default;
  Control* control() const { return control_; }

  const std::string& fileName() const { return yyfilename; }
  template <typename T>
  void setFileName(T&& fileName) {
    yyfilename = std::forward<T>(fileName);
  }

  const std::string& source() const { return yycode; }
  template <typename T>
  void setSource(T&& source) {
    yycode = std::forward<T>(source);
    yyptr = yycode.c_str();

    initializeLineMap();
  }

  bool fatalErrors() const { return fatalErrors_; }
  void setFatalErrors(bool fatalErrors) { fatalErrors_ = fatalErrors; }

  void warning(unsigned index, const char* format...);
  void error(unsigned index, const char* format...);
  void fatal(unsigned index, const char* format...);

  // tokens
  inline unsigned tokenCount() const { return unsigned(tokens_.size()); }
  inline const Token& tokenAt(unsigned index) const { return tokens_[index]; }
  inline TokenKind tokenKind(unsigned index) const {
    return tokens_[index].kind();
  }
  int tokenLength(unsigned index) const;
  std::string_view tokenText(unsigned index) const;
  void getTokenStartPosition(unsigned index, unsigned* line,
                             unsigned* column) const;

  // front end
  void tokenize();
  bool parse(const std::function<void()>& consume = nullptr);

 private:
  void initializeLineMap();
};

}  // namespace cxx
