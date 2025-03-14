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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FRnewOM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/types_fwd.h>

namespace cxx {

class Control;
class TranslationUnit;
class ASTRewriter;

class DeclSpecs {
  struct Visitor;

 public:
  explicit DeclSpecs(TranslationUnit* unit = nullptr);
  explicit DeclSpecs(ASTRewriter* rewriter);

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto getType() const -> const Type*;

  [[nodiscard]] auto hasTypeSpecifier() const -> bool;
  void setTypeSpecifier(SpecifierAST* specifier);

  [[nodiscard]] auto hasClassOrEnumSpecifier() const -> bool;
  [[nodiscard]] auto hasPlaceholderTypeSpecifier() const -> bool;

  void accept(SpecifierAST* specifier);

  ASTRewriter* rewriter = nullptr;
  TranslationUnit* unit = nullptr;
  const Type* type = nullptr;
  SpecifierAST* typeSpecifier = nullptr;

  bool isTypedef = false;
  bool isFriend = false;
  bool isConstexpr = false;
  bool isConsteval = false;
  bool isConstinit = false;
  bool isInline = false;

  // cv qualifiers
  bool isConst = false;
  bool isVolatile = false;
  bool isRestrict = false;

  // storage class specifiers
  bool isStatic = false;
  bool isThreadLocal = false;
  bool isExtern = false;
  bool isMutable = false;
  bool isThread = false;

  // function specifiers
  bool isVirtual = false;
  bool isExplicit = false;

  // sign specifiers
  bool isSigned = false;
  bool isUnsigned = false;

  // sized specifiers
  bool isShort = false;
  bool isLong = false;
  bool isLongLong = false;

  bool isComplex = false;

  // placeholder type specifiers
  bool isAuto = false;
  bool isDecltypeAuto = false;

  // internal state, used during parsing
  bool no_typespecs = false;
  bool no_class_or_enum_specs = false;
};

}  // namespace cxx