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

#include <cxx/codegen/emitter_handles.h>
#include <cxx/source_location.h>

#include <span>
#include <string_view>

namespace cxx::ir {

enum class DebugEncoding {
  Unspecified,
  Boolean,
  Signed,
  Unsigned,
  Utf,
  Float,
  ComplexFloat
};
enum class DebugDerivedKind {
  Pointer,
  Reference,
  RvalueReference,
  Const,
  Volatile,
  Atomic,
  MemberPointer,
  Inheritance,
  Member
};
enum class DebugCompositeKind { Structure, Union, Enumeration };

struct DebugLocation {
  std::string_view file;
  unsigned line = 0;
  unsigned column = 0;
};

struct DebugCompileUnitInfo {
  std::string_view file;
  std::string_view directory;
  bool isCxx = true;
};

struct DebugBasicTypeInfo {
  std::string_view name;
  std::uint64_t sizeInBits = 0;
  DebugEncoding encoding = DebugEncoding::Unspecified;
};

struct DebugDerivedTypeInfo {
  DebugDerivedKind kind;
  DebugTypeRef baseType;
  std::uint64_t sizeInBits = 0;
  std::uint64_t alignInBits = 0;
  std::uint64_t offsetInBits = 0;
  std::string_view name;
  DebugTypeRef classType;
};

struct DebugCompositeTypeInfo {
  DebugCompositeKind kind;
  std::string_view name;
  DebugLocation location;
  DebugScopeRef scope;
  DebugTypeRef baseType;
  std::uint64_t sizeInBits = 0;
  std::uint64_t alignInBits = 0;
  std::span<const DebugTypeRef> elements;
  bool isScopedEnum = false;
};

struct DebugArrayTypeInfo {
  DebugTypeRef elementType;
  std::uint64_t count = 0;
  unsigned countBitWidth = 0;
  std::uint64_t sizeInBits = 0;
  std::uint64_t alignInBits = 0;
};

struct DebugFunctionInfo {
  std::string_view name;
  DebugScopeRef scope;
  DebugTypeRef type;
  DebugLocation location;
  unsigned scopeLine = 0;
};

struct DebugVariableInfo {
  std::string_view name;
  DebugScopeRef scope;
  DebugTypeRef type;
  DebugLocation location;
  unsigned argument = 0;
  bool isObjectParameter = false;
};

class DebugEmitter {
 public:
  virtual ~DebugEmitter() = default;
  [[nodiscard]] virtual auto compileUnit(const DebugCompileUnitInfo& info)
      -> DebugScopeRef = 0;
  [[nodiscard]] virtual auto fileScope(std::string_view file)
      -> DebugScopeRef = 0;
  [[nodiscard]] virtual auto lexicalBlock(DebugScopeRef parent,
                                          DebugLocation location)
      -> DebugScopeRef = 0;
  [[nodiscard]] virtual auto typeScope(DebugTypeRef type) -> DebugScopeRef = 0;
  [[nodiscard]] virtual auto basicType(const DebugBasicTypeInfo& info)
      -> DebugTypeRef = 0;
  [[nodiscard]] virtual auto derivedType(const DebugDerivedTypeInfo& info)
      -> DebugTypeRef = 0;
  [[nodiscard]] virtual auto compositeType(const DebugCompositeTypeInfo& info)
      -> DebugTypeRef = 0;
  [[nodiscard]] virtual auto arrayType(const DebugArrayTypeInfo& info)
      -> DebugTypeRef = 0;
  [[nodiscard]] virtual auto subroutineType(std::span<const DebugTypeRef> types)
      -> DebugTypeRef = 0;
  virtual auto defineFunction(FunctionRef function, SourceLocation loc,
                              const DebugFunctionInfo& info)
      -> DebugScopeRef = 0;
  virtual void localVariable(ValueRef address,
                             const DebugVariableInfo& info) = 0;
};
}  // namespace cxx::ir
