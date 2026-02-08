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

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <string>
#include <unordered_map>

namespace cxx {

class ExternalNameEncoder {
 public:
  ExternalNameEncoder();

  [[nodiscard]] auto encode(Symbol* symbol, std::string_view suffix = "")
      -> std::string;
  [[nodiscard]] auto encode(const Type* type) -> std::string;
  [[nodiscard]] auto encodeVTable(ClassSymbol* classSymbol) -> std::string;

 private:
  [[nodiscard]] auto encodeFunction(FunctionSymbol* function) -> std::string;
  [[nodiscard]] auto encodeData(Symbol* symbol) -> std::string;

  void encodePrefix(Symbol* symbol);
  void encodeTemplatePrefix(Symbol* symbol);
  void encodeUnqualifiedName(Symbol* symbol);

  void encodeName(Symbol* symbol);
  [[nodiscard]] auto encodeLocalName(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeNestedName(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeUnscopedName(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeOperatorName(const Name* name, bool isUnary)
      -> std::string_view;

  void encodeType(const Type* type);
  void encodeConstValue(const Type* type, const ConstValue& value);
  void encodeBareFunctionType(const FunctionType* functionType,
                              bool includeReturnType = false);

  [[nodiscard]] auto encodeSubstitution(const Type* type) -> bool;
  void enterSubstitution(const Type* type);

  void out(std::string_view str) { out_.append(str); }

  struct EncodeType;
  struct EncodeUnqualifiedName;

 private:
  std::unordered_map<const Type*, int> substs_;
  std::string out_;
  int substCount_ = 0;
};

}  // namespace cxx