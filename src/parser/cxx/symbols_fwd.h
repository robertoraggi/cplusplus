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

#define CXX_FOR_EACH_SYMBOL_KIND(V) \
  V(Class)                          \
  V(Concept)                        \
  V(Dependent)                      \
  V(Enumerator)                     \
  V(Function)                       \
  V(Global)                         \
  V(InjectedClassName)              \
  V(Local)                          \
  V(Member)                         \
  V(Namespace)                      \
  V(NamespaceAlias)                 \
  V(NonTypeTemplateParameter)       \
  V(Parameter)                      \
  V(ScopedEnum)                     \
  V(TemplateParameter)              \
  V(TemplateParameterPack)          \
  V(TypeAlias)                      \
  V(Value)

namespace cxx {

enum Lang {
  LANG_CPP,
  LANG_C,
};

enum class AccessKind {
  kPublic,
  kProtected,
  kPrivate,
};

enum class TemplateParameterKind {
  kType,
  kNonType,
  kPack,
};

#define DECLARE_SYMBOL_KIND(name) k##name,

enum class SymbolKind { CXX_FOR_EACH_SYMBOL_KIND(DECLARE_SYMBOL_KIND) };

#undef DECLARE_SYMBOL_KIND

class Scope;
class Symbol;
class SymbolVisitor;
class TemplateHead;
class TemplateParameter;
class TemplateArgument;

#define DECLARE_SYMBOL(name) class name##Symbol;

CXX_FOR_EACH_SYMBOL_KIND(DECLARE_SYMBOL)

#undef DECLARE_SYMBOL

auto equal_to(const Symbol* symbol, const Symbol* other) -> bool;

auto symbol_kind_to_string(SymbolKind kind) -> const char*;

}  // namespace cxx
