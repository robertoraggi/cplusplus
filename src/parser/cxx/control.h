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

/**
 * @brief Control class.
 */
class Control {
 public:
  /**
   * @brief Constructs a control instance.
   */
  Control();

  /**
   * @brief Destructs a control instance.
   */
  ~Control();

  /**
   * @brief Returns the identifier for the given name.
   */
  auto identifier(const std::string_view& name) -> const Identifier*;

  /**
   * @brief Returns the operator-name-id for the given operator.
   */
  auto operatorNameId(TokenKind op) -> const OperatorNameId*;

  /**
   * @brief Returns the conversion-name-id for the given type.
   */
  auto conversionNameId(const QualifiedType& type) -> const ConversionNameId*;

  /**
   * @brief Returns the integer-literal for the given value.
   */
  auto integerLiteral(const std::string_view& value) -> const IntegerLiteral*;

  /**
   * @brief Returns the float-literal for the given value.
   */
  auto floatLiteral(const std::string_view& value) -> const FloatLiteral*;

  /**
   * @brief Returns the char-literal for the given value.
   */
  auto charLiteral(const std::string_view& value) -> const CharLiteral*;

  /**
   * @brief Returns the comment-literal for the given value.
   */
  auto commentLiteral(const std::string_view& value) -> const CommentLiteral*;

  /**
   * @brief Returns the string-literal for the given value.
   */
  auto stringLiteral(const std::string_view& value) -> const StringLiteral*;

  /**
   * @brief Returns the wide-string-literal for the given value.
   */
  auto wideStringLiteral(const std::string_view& value)
      -> const WideStringLiteral*;

  /**
   * @brief Returns the utf8-string-literal for the given value.
   */
  auto utf8StringLiteral(const std::string_view& value)
      -> const Utf8StringLiteral*;

  /**
   * @brief Returns the utf16-string-literal for the given value.
   */
  auto utf16StringLiteral(const std::string_view& value)
      -> const Utf16StringLiteral*;

  /**
   * @brief Returns the utf32-string-literal for the given value.
   */
  auto utf32StringLiteral(const std::string_view& value)
      -> const Utf32StringLiteral*;

  /**
   * @brief Returns the typing environment.
   */
  auto types() -> TypeEnvironment*;

  /**
   * @brief Returns the instance of the symbol factory.
   */
  auto symbols() -> SymbolFactory*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
