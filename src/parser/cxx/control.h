// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <string>

namespace cxx {

class TypeEnvironment;

class Control {
 public:
  Control();
  ~Control();

  auto identifier(const std::string_view& name) -> const Identifier*;
  auto operatorNameId(TokenKind op) -> const OperatorNameId*;
  auto conversionNameId(const QualifiedType& type) -> const ConversionNameId*;

  auto integerLiteral(const std::string_view& value) -> const IntegerLiteral*;
  auto floatLiteral(const std::string_view& value) -> const FloatLiteral*;
  auto charLiteral(const std::string_view& value) -> const CharLiteral*;
  auto commentLiteral(const std::string_view& value) -> const CommentLiteral*;

  auto stringLiteral(const std::string_view& value) -> const StringLiteral*;
  auto wideStringLiteral(const std::string_view& value)
      -> const WideStringLiteral*;
  auto utf8StringLiteral(const std::string_view& value)
      -> const Utf8StringLiteral*;
  auto utf16StringLiteral(const std::string_view& value)
      -> const Utf16StringLiteral*;
  auto utf32StringLiteral(const std::string_view& value)
      -> const Utf32StringLiteral*;

  auto types() -> TypeEnvironment*;
  auto symbols() -> SymbolFactory*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
