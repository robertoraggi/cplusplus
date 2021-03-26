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

  const Identifier* identifier(std::string name);
  const OperatorNameId* operatorNameId(TokenKind op);
  const ConversionNameId* conversionNameId(const QualifiedType& type);

  const NumericLiteral* numericLiteral(std::string value);
  const StringLiteral* stringLiteral(std::string value);
  const CharLiteral* charLiteral(std::string value);

  TypeEnvironment* types();
  SymbolFactory* symbols();

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
