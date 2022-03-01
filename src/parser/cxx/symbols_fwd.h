// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <string_view>
#include <type_traits>

namespace cxx {

class SymbolVisitor;
class SymbolFactory;

class Symbol;
class Scope;

class ConceptSymbol;
class NamespaceSymbol;
class ClassSymbol;
class TypedefSymbol;
class EnumSymbol;
class EnumeratorSymbol;
class ScopedEnumSymbol;
class TemplateParameterList;
class TemplateTypeParameterSymbol;
class VariableSymbol;
class FieldSymbol;
class FunctionSymbol;
class ArgumentSymbol;
class BlockSymbol;

enum class Linkage {
  kCxx,
  kC,
};

enum class Visibility : std::uint8_t {
  kPublic,
  kProtected,
  kPrivate,
};

enum class ClassKey : std::uint8_t {
  kClass,
  kStruct,
  kUnion,
};

inline std::string_view to_string_view(ClassKey classKey) {
  switch (classKey) {
    case ClassKey::kClass:
      return "class";
    case ClassKey::kStruct:
      return "struct";
    case ClassKey::kUnion:
      return "union";
    default:
      return std::string_view();
  }  // switch
}

enum class LookupOptions {
  kDefault = 0,
  kType = 1 << 0,
  kNamespace = 1 << 1,
  kTemplate = 1 << 2,
  kTypeOrNamespace = kType | kNamespace,
};

constexpr LookupOptions operator&(LookupOptions lhs,
                                  LookupOptions rhs) noexcept {
  using U = std::underlying_type<LookupOptions>::type;
  return static_cast<LookupOptions>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

constexpr LookupOptions operator|(LookupOptions lhs,
                                  LookupOptions rhs) noexcept {
  using U = std::underlying_type<LookupOptions>::type;
  return static_cast<LookupOptions>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

}  // namespace cxx
