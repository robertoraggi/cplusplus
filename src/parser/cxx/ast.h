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

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_kind.h>
#include <cxx/ast_visitor.h>
#include <cxx/const_value.h>
#include <cxx/source_location.h>
#include <cxx/token.h>

#include <optional>

namespace cxx {

template <typename T>
class List final : public Managed {
 public:
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

class AST : public Managed {
  ASTKind kind_;
  bool checked_ = false;

 public:
  explicit AST(ASTKind kind) : kind_(kind) {}

  virtual ~AST();

  [[nodiscard]] auto kind() const -> ASTKind { return kind_; }

  [[nodiscard]] auto checked() const -> bool { return checked_; }
  void setChecked(bool checked) { checked_ = checked; }

  virtual void accept(ASTVisitor* visitor) = 0;

  virtual auto firstSourceLocation() -> SourceLocation = 0;
  virtual auto lastSourceLocation() -> SourceLocation = 0;

  auto sourceLocationRange() -> SourceLocationRange {
    return SourceLocationRange(firstSourceLocation(), lastSourceLocation());
  }
};

inline auto firstSourceLocation(SourceLocation loc) -> SourceLocation {
  return loc;
}

template <typename T>
inline auto firstSourceLocation(T* node) -> SourceLocation {
  return node ? node->firstSourceLocation() : SourceLocation();
}

template <typename T>
inline auto firstSourceLocation(List<T>* nodes) -> SourceLocation {
  for (auto it = nodes; it; it = it->next) {
    if (auto loc = firstSourceLocation(it->value)) return loc;
  }
  return {};
}

inline auto lastSourceLocation(SourceLocation loc) -> SourceLocation {
  return loc ? loc.next() : SourceLocation();
}

template <typename T>
inline auto lastSourceLocation(T* node) -> SourceLocation {
  return node ? node->lastSourceLocation() : SourceLocation();
}

template <typename T>
inline auto lastSourceLocation(List<T>* nodes) -> SourceLocation {
  if (!nodes) return {};
  if (auto loc = lastSourceLocation(nodes->next)) return loc;
  if (auto loc = lastSourceLocation(nodes->value)) return loc;
  return {};
}

class AttributeSpecifierAST : public AST {
 public:
  using AST::AST;
};

class AttributeTokenAST : public AST {
 public:
  using AST::AST;
};

class CoreDeclaratorAST : public AST {
 public:
  using AST::AST;
};

class DeclarationAST : public AST {
 public:
  using AST::AST;
};

class DeclaratorModifierAST : public AST {
 public:
  using AST::AST;
};

class ExceptionDeclarationAST : public AST {
 public:
  using AST::AST;
};

class ExpressionAST : public AST {
 public:
  using AST::AST;
  ValueCategory valueCategory = ValueCategory::kNone;
  std::optional<ConstValue> constValue;
};

class FunctionBodyAST : public AST {
 public:
  using AST::AST;
};

class InitializerAST : public AST {
 public:
  using AST::AST;
};

class LambdaCaptureAST : public AST {
 public:
  using AST::AST;
};

class MemInitializerAST : public AST {
 public:
  using AST::AST;
};

class NameAST : public AST {
 public:
  using AST::AST;
  const Name* name = nullptr;
};

class NewInitializerAST : public AST {
 public:
  using AST::AST;
};

class PtrOperatorAST : public AST {
 public:
  using AST::AST;
};

class RequirementAST : public AST {
 public:
  using AST::AST;
};

class SpecifierAST : public AST {
 public:
  using AST::AST;
};

class StatementAST : public AST {
 public:
  using AST::AST;
};

class TemplateArgumentAST : public AST {
 public:
  using AST::AST;
};

class UnitAST : public AST {
 public:
  using AST::AST;
};

class TypeIdAST final : public AST {
 public:
  TypeIdAST() : AST(ASTKind::TypeId) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedNameSpecifierAST final : public AST {
 public:
  NestedNameSpecifierAST() : AST(ASTKind::NestedNameSpecifier) {}

  SourceLocation scopeLoc;
  List<NameAST*>* nameList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDeclaratorAST final : public AST {
 public:
  UsingDeclaratorAST() : AST(ASTKind::UsingDeclarator) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class HandlerAST final : public AST {
 public:
  HandlerAST() : AST(ASTKind::Handler) {}

  SourceLocation catchLoc;
  SourceLocation lparenLoc;
  ExceptionDeclarationAST* exceptionDeclaration = nullptr;
  SourceLocation rparenLoc;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EnumBaseAST final : public AST {
 public:
  EnumBaseAST() : AST(ASTKind::EnumBase) {}

  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EnumeratorAST final : public AST {
 public:
  EnumeratorAST() : AST(ASTKind::Enumerator) {}

  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeclaratorAST final : public AST {
 public:
  DeclaratorAST() : AST(ASTKind::Declarator) {}

  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InitDeclaratorAST final : public AST {
 public:
  InitDeclaratorAST() : AST(ASTKind::InitDeclarator) {}

  DeclaratorAST* declarator = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  InitializerAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BaseSpecifierAST final : public AST {
 public:
  BaseSpecifierAST() : AST(ASTKind::BaseSpecifier) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NameAST* name = nullptr;
  bool isVirtual = false;
  TokenKind accessSpecifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BaseClauseAST final : public AST {
 public:
  BaseClauseAST() : AST(ASTKind::BaseClause) {}

  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewTypeIdAST final : public AST {
 public:
  NewTypeIdAST() : AST(ASTKind::NewTypeId) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RequiresClauseAST final : public AST {
 public:
  RequiresClauseAST() : AST(ASTKind::RequiresClause) {}

  SourceLocation requiresLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParameterDeclarationClauseAST final : public AST {
 public:
  ParameterDeclarationClauseAST() : AST(ASTKind::ParameterDeclarationClause) {}

  List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
  SourceLocation commaLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParametersAndQualifiersAST final : public AST {
 public:
  ParametersAndQualifiersAST() : AST(ASTKind::ParametersAndQualifiers) {}

  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<SpecifierAST*>* cvQualifierList = nullptr;
  SourceLocation refLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LambdaIntroducerAST final : public AST {
 public:
  LambdaIntroducerAST() : AST(ASTKind::LambdaIntroducer) {}

  SourceLocation lbracketLoc;
  SourceLocation captureDefaultLoc;
  List<LambdaCaptureAST*>* captureList = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LambdaDeclaratorAST final : public AST {
 public:
  LambdaDeclaratorAST() : AST(ASTKind::LambdaDeclarator) {}

  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;
  RequiresClauseAST* requiresClause = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TrailingReturnTypeAST final : public AST {
 public:
  TrailingReturnTypeAST() : AST(ASTKind::TrailingReturnType) {}

  SourceLocation minusGreaterLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CtorInitializerAST final : public AST {
 public:
  CtorInitializerAST() : AST(ASTKind::CtorInitializer) {}

  SourceLocation colonLoc;
  List<MemInitializerAST*>* memInitializerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RequirementBodyAST final : public AST {
 public:
  RequirementBodyAST() : AST(ASTKind::RequirementBody) {}

  SourceLocation lbraceLoc;
  List<RequirementAST*>* requirementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeConstraintAST final : public AST {
 public:
  TypeConstraintAST() : AST(ASTKind::TypeConstraint) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GlobalModuleFragmentAST final : public AST {
 public:
  GlobalModuleFragmentAST() : AST(ASTKind::GlobalModuleFragment) {}

  SourceLocation moduleLoc;
  SourceLocation semicolonLoc;
  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PrivateModuleFragmentAST final : public AST {
 public:
  PrivateModuleFragmentAST() : AST(ASTKind::PrivateModuleFragment) {}

  SourceLocation moduleLoc;
  SourceLocation colonLoc;
  SourceLocation privateLoc;
  SourceLocation semicolonLoc;
  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleDeclarationAST final : public AST {
 public:
  ModuleDeclarationAST() : AST(ASTKind::ModuleDeclaration) {}

  SourceLocation exportLoc;
  SourceLocation moduleLoc;
  ModuleNameAST* moduleName = nullptr;
  ModulePartitionAST* modulePartition = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleNameAST final : public AST {
 public:
  ModuleNameAST() : AST(ASTKind::ModuleName) {}

  List<SourceLocation>* identifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ImportNameAST final : public AST {
 public:
  ImportNameAST() : AST(ASTKind::ImportName) {}

  SourceLocation headerLoc;
  ModulePartitionAST* modulePartition = nullptr;
  ModuleNameAST* moduleName = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModulePartitionAST final : public AST {
 public:
  ModulePartitionAST() : AST(ASTKind::ModulePartition) {}

  SourceLocation colonLoc;
  ModuleNameAST* moduleName = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeArgumentClauseAST final : public AST {
 public:
  AttributeArgumentClauseAST() : AST(ASTKind::AttributeArgumentClause) {}

  SourceLocation lparenLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeAST final : public AST {
 public:
  AttributeAST() : AST(ASTKind::Attribute) {}

  AttributeTokenAST* attributeToken = nullptr;
  AttributeArgumentClauseAST* attributeArgumentClause = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeUsingPrefixAST final : public AST {
 public:
  AttributeUsingPrefixAST() : AST(ASTKind::AttributeUsingPrefix) {}

  SourceLocation usingLoc;
  SourceLocation attributeNamespaceLoc;
  SourceLocation colonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DesignatorAST final : public AST {
 public:
  DesignatorAST() : AST(ASTKind::Designator) {}

  SourceLocation dotLoc;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DesignatedInitializerClauseAST final : public ExpressionAST {
 public:
  DesignatedInitializerClauseAST()
      : ExpressionAST(ASTKind::DesignatedInitializerClause) {}

  DesignatorAST* designator = nullptr;
  InitializerAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThisExpressionAST final : public ExpressionAST {
 public:
  ThisExpressionAST() : ExpressionAST(ASTKind::ThisExpression) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CharLiteralExpressionAST final : public ExpressionAST {
 public:
  CharLiteralExpressionAST() : ExpressionAST(ASTKind::CharLiteralExpression) {}

  SourceLocation literalLoc;
  const CharLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BoolLiteralExpressionAST final : public ExpressionAST {
 public:
  BoolLiteralExpressionAST() : ExpressionAST(ASTKind::BoolLiteralExpression) {}

  SourceLocation literalLoc;
  bool value = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IntLiteralExpressionAST final : public ExpressionAST {
 public:
  IntLiteralExpressionAST() : ExpressionAST(ASTKind::IntLiteralExpression) {}

  SourceLocation literalLoc;
  const IntegerLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FloatLiteralExpressionAST final : public ExpressionAST {
 public:
  FloatLiteralExpressionAST()
      : ExpressionAST(ASTKind::FloatLiteralExpression) {}

  SourceLocation literalLoc;
  const FloatLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NullptrLiteralExpressionAST final : public ExpressionAST {
 public:
  NullptrLiteralExpressionAST()
      : ExpressionAST(ASTKind::NullptrLiteralExpression) {}

  SourceLocation literalLoc;
  TokenKind literal = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StringLiteralExpressionAST final : public ExpressionAST {
 public:
  StringLiteralExpressionAST()
      : ExpressionAST(ASTKind::StringLiteralExpression) {}

  SourceLocation literalLoc;
  const Literal* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UserDefinedStringLiteralExpressionAST final : public ExpressionAST {
 public:
  UserDefinedStringLiteralExpressionAST()
      : ExpressionAST(ASTKind::UserDefinedStringLiteralExpression) {}

  SourceLocation literalLoc;
  const StringLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IdExpressionAST final : public ExpressionAST {
 public:
  IdExpressionAST() : ExpressionAST(ASTKind::IdExpression) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RequiresExpressionAST final : public ExpressionAST {
 public:
  RequiresExpressionAST() : ExpressionAST(ASTKind::RequiresExpression) {}

  SourceLocation requiresLoc;
  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  RequirementBodyAST* requirementBody = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedExpressionAST final : public ExpressionAST {
 public:
  NestedExpressionAST() : ExpressionAST(ASTKind::NestedExpression) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RightFoldExpressionAST final : public ExpressionAST {
 public:
  RightFoldExpressionAST() : ExpressionAST(ASTKind::RightFoldExpression) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation opLoc;
  SourceLocation ellipsisLoc;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LeftFoldExpressionAST final : public ExpressionAST {
 public:
  LeftFoldExpressionAST() : ExpressionAST(ASTKind::LeftFoldExpression) {}

  SourceLocation lparenLoc;
  SourceLocation ellipsisLoc;
  SourceLocation opLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FoldExpressionAST final : public ExpressionAST {
 public:
  FoldExpressionAST() : ExpressionAST(ASTKind::FoldExpression) {}

  SourceLocation lparenLoc;
  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  SourceLocation ellipsisLoc;
  SourceLocation foldOpLoc;
  ExpressionAST* rightExpression = nullptr;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;
  TokenKind foldOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LambdaExpressionAST final : public ExpressionAST {
 public:
  LambdaExpressionAST() : ExpressionAST(ASTKind::LambdaExpression) {}

  LambdaIntroducerAST* lambdaIntroducer = nullptr;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* requiresClause = nullptr;
  LambdaDeclaratorAST* lambdaDeclarator = nullptr;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofExpressionAST final : public ExpressionAST {
 public:
  SizeofExpressionAST() : ExpressionAST(ASTKind::SizeofExpression) {}

  SourceLocation sizeofLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofTypeExpressionAST final : public ExpressionAST {
 public:
  SizeofTypeExpressionAST() : ExpressionAST(ASTKind::SizeofTypeExpression) {}

  SourceLocation sizeofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofPackExpressionAST final : public ExpressionAST {
 public:
  SizeofPackExpressionAST() : ExpressionAST(ASTKind::SizeofPackExpression) {}

  SourceLocation sizeofLoc;
  SourceLocation ellipsisLoc;
  SourceLocation lparenLoc;
  SourceLocation identifierLoc;
  SourceLocation rparenLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeidExpressionAST final : public ExpressionAST {
 public:
  TypeidExpressionAST() : ExpressionAST(ASTKind::TypeidExpression) {}

  SourceLocation typeidLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeidOfTypeExpressionAST final : public ExpressionAST {
 public:
  TypeidOfTypeExpressionAST()
      : ExpressionAST(ASTKind::TypeidOfTypeExpression) {}

  SourceLocation typeidLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignofExpressionAST final : public ExpressionAST {
 public:
  AlignofExpressionAST() : ExpressionAST(ASTKind::AlignofExpression) {}

  SourceLocation alignofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeTraitsExpressionAST final : public ExpressionAST {
 public:
  TypeTraitsExpressionAST() : ExpressionAST(ASTKind::TypeTraitsExpression) {}

  SourceLocation typeTraitsLoc;
  SourceLocation lparenLoc;
  List<TypeIdAST*>* typeIdList = nullptr;
  SourceLocation rparenLoc;
  TokenKind typeTraits = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UnaryExpressionAST final : public ExpressionAST {
 public:
  UnaryExpressionAST() : ExpressionAST(ASTKind::UnaryExpression) {}

  SourceLocation opLoc;
  ExpressionAST* expression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BinaryExpressionAST final : public ExpressionAST {
 public:
  BinaryExpressionAST() : ExpressionAST(ASTKind::BinaryExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AssignmentExpressionAST final : public ExpressionAST {
 public:
  AssignmentExpressionAST() : ExpressionAST(ASTKind::AssignmentExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedTypeConstructionAST final : public ExpressionAST {
 public:
  BracedTypeConstructionAST()
      : ExpressionAST(ASTKind::BracedTypeConstruction) {}

  SpecifierAST* typeSpecifier = nullptr;
  BracedInitListAST* bracedInitList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeConstructionAST final : public ExpressionAST {
 public:
  TypeConstructionAST() : ExpressionAST(ASTKind::TypeConstruction) {}

  SpecifierAST* typeSpecifier = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CallExpressionAST final : public ExpressionAST {
 public:
  CallExpressionAST() : ExpressionAST(ASTKind::CallExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SubscriptExpressionAST final : public ExpressionAST {
 public:
  SubscriptExpressionAST() : ExpressionAST(ASTKind::SubscriptExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lbracketLoc;
  ExpressionAST* indexExpression = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class MemberExpressionAST final : public ExpressionAST {
 public:
  MemberExpressionAST() : ExpressionAST(ASTKind::MemberExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation accessLoc;
  SourceLocation templateLoc;
  NameAST* name = nullptr;
  TokenKind accessOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PostIncrExpressionAST final : public ExpressionAST {
 public:
  PostIncrExpressionAST() : ExpressionAST(ASTKind::PostIncrExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConditionalExpressionAST final : public ExpressionAST {
 public:
  ConditionalExpressionAST() : ExpressionAST(ASTKind::ConditionalExpression) {}

  ExpressionAST* condition = nullptr;
  SourceLocation questionLoc;
  ExpressionAST* iftrueExpression = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* iffalseExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ImplicitCastExpressionAST final : public ExpressionAST {
 public:
  ImplicitCastExpressionAST()
      : ExpressionAST(ASTKind::ImplicitCastExpression) {}

  ExpressionAST* expression = nullptr;
  ImplicitCastKind castKind = ImplicitCastKind::kIdentity;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CastExpressionAST final : public ExpressionAST {
 public:
  CastExpressionAST() : ExpressionAST(ASTKind::CastExpression) {}

  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CppCastExpressionAST final : public ExpressionAST {
 public:
  CppCastExpressionAST() : ExpressionAST(ASTKind::CppCastExpression) {}

  SourceLocation castLoc;
  SourceLocation lessLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation greaterLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewExpressionAST final : public ExpressionAST {
 public:
  NewExpressionAST() : ExpressionAST(ASTKind::NewExpression) {}

  SourceLocation scopeLoc;
  SourceLocation newLoc;
  NewTypeIdAST* typeId = nullptr;
  NewInitializerAST* newInitalizer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeleteExpressionAST final : public ExpressionAST {
 public:
  DeleteExpressionAST() : ExpressionAST(ASTKind::DeleteExpression) {}

  SourceLocation scopeLoc;
  SourceLocation deleteLoc;
  SourceLocation lbracketLoc;
  SourceLocation rbracketLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThrowExpressionAST final : public ExpressionAST {
 public:
  ThrowExpressionAST() : ExpressionAST(ASTKind::ThrowExpression) {}

  SourceLocation throwLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NoexceptExpressionAST final : public ExpressionAST {
 public:
  NoexceptExpressionAST() : ExpressionAST(ASTKind::NoexceptExpression) {}

  SourceLocation noexceptLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleRequirementAST final : public RequirementAST {
 public:
  SimpleRequirementAST() : RequirementAST(ASTKind::SimpleRequirement) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundRequirementAST final : public RequirementAST {
 public:
  CompoundRequirementAST() : RequirementAST(ASTKind::CompoundRequirement) {}

  SourceLocation lbraceLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbraceLoc;
  SourceLocation noexceptLoc;
  SourceLocation minusGreaterLoc;
  TypeConstraintAST* typeConstraint = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeRequirementAST final : public RequirementAST {
 public:
  TypeRequirementAST() : RequirementAST(ASTKind::TypeRequirement) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedRequirementAST final : public RequirementAST {
 public:
  NestedRequirementAST() : RequirementAST(ASTKind::NestedRequirement) {}

  SourceLocation requiresLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeTemplateArgumentAST final : public TemplateArgumentAST {
 public:
  TypeTemplateArgumentAST()
      : TemplateArgumentAST(ASTKind::TypeTemplateArgument) {}

  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExpressionTemplateArgumentAST final : public TemplateArgumentAST {
 public:
  ExpressionTemplateArgumentAST()
      : TemplateArgumentAST(ASTKind::ExpressionTemplateArgument) {}

  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParenMemInitializerAST final : public MemInitializerAST {
 public:
  ParenMemInitializerAST() : MemInitializerAST(ASTKind::ParenMemInitializer) {}

  NameAST* name = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedMemInitializerAST final : public MemInitializerAST {
 public:
  BracedMemInitializerAST()
      : MemInitializerAST(ASTKind::BracedMemInitializer) {}

  NameAST* name = nullptr;
  BracedInitListAST* bracedInitList = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  ThisLambdaCaptureAST() : LambdaCaptureAST(ASTKind::ThisLambdaCapture) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DerefThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  DerefThisLambdaCaptureAST()
      : LambdaCaptureAST(ASTKind::DerefThisLambdaCapture) {}

  SourceLocation starLoc;
  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  SimpleLambdaCaptureAST() : LambdaCaptureAST(ASTKind::SimpleLambdaCapture) {}

  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RefLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  RefLambdaCaptureAST() : LambdaCaptureAST(ASTKind::RefLambdaCapture) {}

  SourceLocation ampLoc;
  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RefInitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  RefInitLambdaCaptureAST() : LambdaCaptureAST(ASTKind::RefInitLambdaCapture) {}

  SourceLocation ampLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  InitializerAST* initializer = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  InitLambdaCaptureAST() : LambdaCaptureAST(ASTKind::InitLambdaCapture) {}

  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  InitializerAST* initializer = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EqualInitializerAST final : public InitializerAST {
 public:
  EqualInitializerAST() : InitializerAST(ASTKind::EqualInitializer) {}

  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedInitListAST final : public InitializerAST {
 public:
  BracedInitListAST() : InitializerAST(ASTKind::BracedInitList) {}

  SourceLocation lbraceLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParenInitializerAST final : public InitializerAST {
 public:
  ParenInitializerAST() : InitializerAST(ASTKind::ParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewParenInitializerAST final : public NewInitializerAST {
 public:
  NewParenInitializerAST() : NewInitializerAST(ASTKind::NewParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewBracedInitializerAST final : public NewInitializerAST {
 public:
  NewBracedInitializerAST()
      : NewInitializerAST(ASTKind::NewBracedInitializer) {}

  BracedInitListAST* bracedInit = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EllipsisExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  EllipsisExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::EllipsisExceptionDeclaration) {}

  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  TypeExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::TypeExceptionDeclaration) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DefaultFunctionBodyAST final : public FunctionBodyAST {
 public:
  DefaultFunctionBodyAST() : FunctionBodyAST(ASTKind::DefaultFunctionBody) {}

  SourceLocation equalLoc;
  SourceLocation defaultLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundStatementFunctionBodyAST final : public FunctionBodyAST {
 public:
  CompoundStatementFunctionBodyAST()
      : FunctionBodyAST(ASTKind::CompoundStatementFunctionBody) {}

  CtorInitializerAST* ctorInitializer = nullptr;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TryStatementFunctionBodyAST final : public FunctionBodyAST {
 public:
  TryStatementFunctionBodyAST()
      : FunctionBodyAST(ASTKind::TryStatementFunctionBody) {}

  SourceLocation tryLoc;
  CtorInitializerAST* ctorInitializer = nullptr;
  CompoundStatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeleteFunctionBodyAST final : public FunctionBodyAST {
 public:
  DeleteFunctionBodyAST() : FunctionBodyAST(ASTKind::DeleteFunctionBody) {}

  SourceLocation equalLoc;
  SourceLocation deleteLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TranslationUnitAST final : public UnitAST {
 public:
  TranslationUnitAST() : UnitAST(ASTKind::TranslationUnit) {}

  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleUnitAST final : public UnitAST {
 public:
  ModuleUnitAST() : UnitAST(ASTKind::ModuleUnit) {}

  GlobalModuleFragmentAST* globalModuleFragment = nullptr;
  ModuleDeclarationAST* moduleDeclaration = nullptr;
  List<DeclarationAST*>* declarationList = nullptr;
  PrivateModuleFragmentAST* privateModuleFragment = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LabeledStatementAST final : public StatementAST {
 public:
  LabeledStatementAST() : StatementAST(ASTKind::LabeledStatement) {}

  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CaseStatementAST final : public StatementAST {
 public:
  CaseStatementAST() : StatementAST(ASTKind::CaseStatement) {}

  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DefaultStatementAST final : public StatementAST {
 public:
  DefaultStatementAST() : StatementAST(ASTKind::DefaultStatement) {}

  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExpressionStatementAST final : public StatementAST {
 public:
  ExpressionStatementAST() : StatementAST(ASTKind::ExpressionStatement) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundStatementAST final : public StatementAST {
 public:
  CompoundStatementAST() : StatementAST(ASTKind::CompoundStatement) {}

  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IfStatementAST final : public StatementAST {
 public:
  IfStatementAST() : StatementAST(ASTKind::IfStatement) {}

  SourceLocation ifLoc;
  SourceLocation constexprLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;
  StatementAST* elseStatement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SwitchStatementAST final : public StatementAST {
 public:
  SwitchStatementAST() : StatementAST(ASTKind::SwitchStatement) {}

  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class WhileStatementAST final : public StatementAST {
 public:
  WhileStatementAST() : StatementAST(ASTKind::WhileStatement) {}

  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DoStatementAST final : public StatementAST {
 public:
  DoStatementAST() : StatementAST(ASTKind::DoStatement) {}

  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForRangeStatementAST final : public StatementAST {
 public:
  ForRangeStatementAST() : StatementAST(ASTKind::ForRangeStatement) {}

  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  DeclarationAST* rangeDeclaration = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* rangeInitializer = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForStatementAST final : public StatementAST {
 public:
  ForStatementAST() : StatementAST(ASTKind::ForStatement) {}

  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation semicolonLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BreakStatementAST final : public StatementAST {
 public:
  BreakStatementAST() : StatementAST(ASTKind::BreakStatement) {}

  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ContinueStatementAST final : public StatementAST {
 public:
  ContinueStatementAST() : StatementAST(ASTKind::ContinueStatement) {}

  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ReturnStatementAST final : public StatementAST {
 public:
  ReturnStatementAST() : StatementAST(ASTKind::ReturnStatement) {}

  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GotoStatementAST final : public StatementAST {
 public:
  GotoStatementAST() : StatementAST(ASTKind::GotoStatement) {}

  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CoroutineReturnStatementAST final : public StatementAST {
 public:
  CoroutineReturnStatementAST()
      : StatementAST(ASTKind::CoroutineReturnStatement) {}

  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeclarationStatementAST final : public StatementAST {
 public:
  DeclarationStatementAST() : StatementAST(ASTKind::DeclarationStatement) {}

  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TryBlockStatementAST final : public StatementAST {
 public:
  TryBlockStatementAST() : StatementAST(ASTKind::TryBlockStatement) {}

  SourceLocation tryLoc;
  CompoundStatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AccessDeclarationAST final : public DeclarationAST {
 public:
  AccessDeclarationAST() : DeclarationAST(ASTKind::AccessDeclaration) {}

  SourceLocation accessLoc;
  SourceLocation colonLoc;
  TokenKind accessSpecifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FunctionDefinitionAST final : public DeclarationAST {
 public:
  FunctionDefinitionAST() : DeclarationAST(ASTKind::FunctionDefinition) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  FunctionBodyAST* functionBody = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConceptDefinitionAST final : public DeclarationAST {
 public:
  ConceptDefinitionAST() : DeclarationAST(ASTKind::ConceptDefinition) {}

  SourceLocation conceptLoc;
  NameAST* name = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForRangeDeclarationAST final : public DeclarationAST {
 public:
  ForRangeDeclarationAST() : DeclarationAST(ASTKind::ForRangeDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AliasDeclarationAST final : public DeclarationAST {
 public:
  AliasDeclarationAST() : DeclarationAST(ASTKind::AliasDeclaration) {}

  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleDeclarationAST final : public DeclarationAST {
 public:
  SimpleDeclarationAST() : DeclarationAST(ASTKind::SimpleDeclaration) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StaticAssertDeclarationAST final : public DeclarationAST {
 public:
  StaticAssertDeclarationAST()
      : DeclarationAST(ASTKind::StaticAssertDeclaration) {}

  SourceLocation staticAssertLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation commaLoc;
  SourceLocation literalLoc;
  const Literal* literal = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EmptyDeclarationAST final : public DeclarationAST {
 public:
  EmptyDeclarationAST() : DeclarationAST(ASTKind::EmptyDeclaration) {}

  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeDeclarationAST final : public DeclarationAST {
 public:
  AttributeDeclarationAST() : DeclarationAST(ASTKind::AttributeDeclaration) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class OpaqueEnumDeclarationAST final : public DeclarationAST {
 public:
  OpaqueEnumDeclarationAST() : DeclarationAST(ASTKind::OpaqueEnumDeclaration) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation emicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedNamespaceSpecifierAST final : public DeclarationAST {
 public:
  NestedNamespaceSpecifierAST()
      : DeclarationAST(ASTKind::NestedNamespaceSpecifier) {}

  SourceLocation inlineLoc;
  SourceLocation identifierLoc;
  SourceLocation scopeLoc;
  const Identifier* namespaceName = nullptr;
  bool isInline = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamespaceDefinitionAST final : public DeclarationAST {
 public:
  NamespaceDefinitionAST() : DeclarationAST(ASTKind::NamespaceDefinition) {}

  SourceLocation inlineLoc;
  SourceLocation namespaceLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<NestedNamespaceSpecifierAST*>* nestedNamespaceSpecifierList = nullptr;
  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* extraAttributeList = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  const Identifier* namespaceName = nullptr;
  bool isInline = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamespaceAliasDefinitionAST final : public DeclarationAST {
 public:
  NamespaceAliasDefinitionAST()
      : DeclarationAST(ASTKind::NamespaceAliasDefinition) {}

  SourceLocation namespaceLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDirectiveAST final : public DeclarationAST {
 public:
  UsingDirectiveAST() : DeclarationAST(ASTKind::UsingDirective) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation usingLoc;
  SourceLocation namespaceLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDeclarationAST final : public DeclarationAST {
 public:
  UsingDeclarationAST() : DeclarationAST(ASTKind::UsingDeclaration) {}

  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingEnumDeclarationAST final : public DeclarationAST {
 public:
  UsingEnumDeclarationAST() : DeclarationAST(ASTKind::UsingEnumDeclaration) {}

  SourceLocation usingLoc;
  ElaboratedTypeSpecifierAST* enumTypeSpecifier = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmDeclarationAST final : public DeclarationAST {
 public:
  AsmDeclarationAST() : DeclarationAST(ASTKind::AsmDeclaration) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  SourceLocation literalLoc;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;
  const Literal* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExportDeclarationAST final : public DeclarationAST {
 public:
  ExportDeclarationAST() : DeclarationAST(ASTKind::ExportDeclaration) {}

  SourceLocation exportLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExportCompoundDeclarationAST final : public DeclarationAST {
 public:
  ExportCompoundDeclarationAST()
      : DeclarationAST(ASTKind::ExportCompoundDeclaration) {}

  SourceLocation exportLoc;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleImportDeclarationAST final : public DeclarationAST {
 public:
  ModuleImportDeclarationAST()
      : DeclarationAST(ASTKind::ModuleImportDeclaration) {}

  SourceLocation importLoc;
  ImportNameAST* importName = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateDeclarationAST final : public DeclarationAST {
 public:
  TemplateDeclarationAST() : DeclarationAST(ASTKind::TemplateDeclaration) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* requiresClause = nullptr;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypenameTypeParameterAST final : public DeclarationAST {
 public:
  TypenameTypeParameterAST() : DeclarationAST(ASTKind::TypenameTypeParameter) {}

  SourceLocation classKeyLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateTypeParameterAST final : public DeclarationAST {
 public:
  TemplateTypeParameterAST() : DeclarationAST(ASTKind::TemplateTypeParameter) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* requiresClause = nullptr;
  SourceLocation classKeyLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NameAST* name = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplatePackTypeParameterAST final : public DeclarationAST {
 public:
  TemplatePackTypeParameterAST()
      : DeclarationAST(ASTKind::TemplatePackTypeParameter) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  SourceLocation classKeyLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeductionGuideAST final : public DeclarationAST {
 public:
  DeductionGuideAST() : DeclarationAST(ASTKind::DeductionGuide) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExplicitInstantiationAST final : public DeclarationAST {
 public:
  ExplicitInstantiationAST() : DeclarationAST(ASTKind::ExplicitInstantiation) {}

  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParameterDeclarationAST final : public DeclarationAST {
 public:
  ParameterDeclarationAST() : DeclarationAST(ASTKind::ParameterDeclaration) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LinkageSpecificationAST final : public DeclarationAST {
 public:
  LinkageSpecificationAST() : DeclarationAST(ASTKind::LinkageSpecification) {}

  SourceLocation externLoc;
  SourceLocation stringliteralLoc;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  const StringLiteral* stringLiteral = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleNameAST final : public NameAST {
 public:
  SimpleNameAST() : NameAST(ASTKind::SimpleName) {}

  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DestructorNameAST final : public NameAST {
 public:
  DestructorNameAST() : NameAST(ASTKind::DestructorName) {}

  SourceLocation tildeLoc;
  NameAST* id = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeNameAST final : public NameAST {
 public:
  DecltypeNameAST() : NameAST(ASTKind::DecltypeName) {}

  SpecifierAST* decltypeSpecifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class OperatorNameAST final : public NameAST {
 public:
  OperatorNameAST() : NameAST(ASTKind::OperatorName) {}

  SourceLocation operatorLoc;
  SourceLocation opLoc;
  SourceLocation openLoc;
  SourceLocation closeLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConversionNameAST final : public NameAST {
 public:
  ConversionNameAST() : NameAST(ASTKind::ConversionName) {}

  SourceLocation operatorLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateNameAST final : public NameAST {
 public:
  TemplateNameAST() : NameAST(ASTKind::TemplateName) {}

  NameAST* id = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class QualifiedNameAST final : public NameAST {
 public:
  QualifiedNameAST() : NameAST(ASTKind::QualifiedName) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  NameAST* id = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypedefSpecifierAST final : public SpecifierAST {
 public:
  TypedefSpecifierAST() : SpecifierAST(ASTKind::TypedefSpecifier) {}

  SourceLocation typedefLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FriendSpecifierAST final : public SpecifierAST {
 public:
  FriendSpecifierAST() : SpecifierAST(ASTKind::FriendSpecifier) {}

  SourceLocation friendLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstevalSpecifierAST final : public SpecifierAST {
 public:
  ConstevalSpecifierAST() : SpecifierAST(ASTKind::ConstevalSpecifier) {}

  SourceLocation constevalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstinitSpecifierAST final : public SpecifierAST {
 public:
  ConstinitSpecifierAST() : SpecifierAST(ASTKind::ConstinitSpecifier) {}

  SourceLocation constinitLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstexprSpecifierAST final : public SpecifierAST {
 public:
  ConstexprSpecifierAST() : SpecifierAST(ASTKind::ConstexprSpecifier) {}

  SourceLocation constexprLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InlineSpecifierAST final : public SpecifierAST {
 public:
  InlineSpecifierAST() : SpecifierAST(ASTKind::InlineSpecifier) {}

  SourceLocation inlineLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StaticSpecifierAST final : public SpecifierAST {
 public:
  StaticSpecifierAST() : SpecifierAST(ASTKind::StaticSpecifier) {}

  SourceLocation staticLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExternSpecifierAST final : public SpecifierAST {
 public:
  ExternSpecifierAST() : SpecifierAST(ASTKind::ExternSpecifier) {}

  SourceLocation externLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThreadLocalSpecifierAST final : public SpecifierAST {
 public:
  ThreadLocalSpecifierAST() : SpecifierAST(ASTKind::ThreadLocalSpecifier) {}

  SourceLocation threadLocalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThreadSpecifierAST final : public SpecifierAST {
 public:
  ThreadSpecifierAST() : SpecifierAST(ASTKind::ThreadSpecifier) {}

  SourceLocation threadLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class MutableSpecifierAST final : public SpecifierAST {
 public:
  MutableSpecifierAST() : SpecifierAST(ASTKind::MutableSpecifier) {}

  SourceLocation mutableLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VirtualSpecifierAST final : public SpecifierAST {
 public:
  VirtualSpecifierAST() : SpecifierAST(ASTKind::VirtualSpecifier) {}

  SourceLocation virtualLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExplicitSpecifierAST final : public SpecifierAST {
 public:
  ExplicitSpecifierAST() : SpecifierAST(ASTKind::ExplicitSpecifier) {}

  SourceLocation explicitLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AutoTypeSpecifierAST final : public SpecifierAST {
 public:
  AutoTypeSpecifierAST() : SpecifierAST(ASTKind::AutoTypeSpecifier) {}

  SourceLocation autoLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VoidTypeSpecifierAST final : public SpecifierAST {
 public:
  VoidTypeSpecifierAST() : SpecifierAST(ASTKind::VoidTypeSpecifier) {}

  SourceLocation voidLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VaListTypeSpecifierAST final : public SpecifierAST {
 public:
  VaListTypeSpecifierAST() : SpecifierAST(ASTKind::VaListTypeSpecifier) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IntegralTypeSpecifierAST final : public SpecifierAST {
 public:
  IntegralTypeSpecifierAST() : SpecifierAST(ASTKind::IntegralTypeSpecifier) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FloatingPointTypeSpecifierAST final : public SpecifierAST {
 public:
  FloatingPointTypeSpecifierAST()
      : SpecifierAST(ASTKind::FloatingPointTypeSpecifier) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ComplexTypeSpecifierAST final : public SpecifierAST {
 public:
  ComplexTypeSpecifierAST() : SpecifierAST(ASTKind::ComplexTypeSpecifier) {}

  SourceLocation complexLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamedTypeSpecifierAST final : public SpecifierAST {
 public:
  NamedTypeSpecifierAST() : SpecifierAST(ASTKind::NamedTypeSpecifier) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AtomicTypeSpecifierAST final : public SpecifierAST {
 public:
  AtomicTypeSpecifierAST() : SpecifierAST(ASTKind::AtomicTypeSpecifier) {}

  SourceLocation atomicLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UnderlyingTypeSpecifierAST final : public SpecifierAST {
 public:
  UnderlyingTypeSpecifierAST()
      : SpecifierAST(ASTKind::UnderlyingTypeSpecifier) {}

  SourceLocation underlyingTypeLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ElaboratedTypeSpecifierAST final : public SpecifierAST {
 public:
  ElaboratedTypeSpecifierAST()
      : SpecifierAST(ASTKind::ElaboratedTypeSpecifier) {}

  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  TokenKind classKey = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeAutoSpecifierAST final : public SpecifierAST {
 public:
  DecltypeAutoSpecifierAST() : SpecifierAST(ASTKind::DecltypeAutoSpecifier) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  SourceLocation autoLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeSpecifierAST final : public SpecifierAST {
 public:
  DecltypeSpecifierAST() : SpecifierAST(ASTKind::DecltypeSpecifier) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PlaceholderTypeSpecifierAST final : public SpecifierAST {
 public:
  PlaceholderTypeSpecifierAST()
      : SpecifierAST(ASTKind::PlaceholderTypeSpecifier) {}

  TypeConstraintAST* typeConstraint = nullptr;
  SpecifierAST* specifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstQualifierAST final : public SpecifierAST {
 public:
  ConstQualifierAST() : SpecifierAST(ASTKind::ConstQualifier) {}

  SourceLocation constLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VolatileQualifierAST final : public SpecifierAST {
 public:
  VolatileQualifierAST() : SpecifierAST(ASTKind::VolatileQualifier) {}

  SourceLocation volatileLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RestrictQualifierAST final : public SpecifierAST {
 public:
  RestrictQualifierAST() : SpecifierAST(ASTKind::RestrictQualifier) {}

  SourceLocation restrictLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EnumSpecifierAST final : public SpecifierAST {
 public:
  EnumSpecifierAST() : SpecifierAST(ASTKind::EnumSpecifier) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  List<EnumeratorAST*>* enumeratorList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ClassSpecifierAST final : public SpecifierAST {
 public:
  ClassSpecifierAST() : SpecifierAST(ASTKind::ClassSpecifier) {}

  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NameAST* name = nullptr;
  SourceLocation finalLoc;
  BaseClauseAST* baseClause = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  TokenKind classKey = TokenKind::T_EOF_SYMBOL;
  bool isFinal = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypenameSpecifierAST final : public SpecifierAST {
 public:
  TypenameSpecifierAST() : SpecifierAST(ASTKind::TypenameSpecifier) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IdDeclaratorAST final : public CoreDeclaratorAST {
 public:
  IdDeclaratorAST() : CoreDeclaratorAST(ASTKind::IdDeclarator) {}

  SourceLocation ellipsisLoc;
  NameAST* name = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedDeclaratorAST final : public CoreDeclaratorAST {
 public:
  NestedDeclaratorAST() : CoreDeclaratorAST(ASTKind::NestedDeclarator) {}

  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PointerOperatorAST final : public PtrOperatorAST {
 public:
  PointerOperatorAST() : PtrOperatorAST(ASTKind::PointerOperator) {}

  SourceLocation starLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ReferenceOperatorAST final : public PtrOperatorAST {
 public:
  ReferenceOperatorAST() : PtrOperatorAST(ASTKind::ReferenceOperator) {}

  SourceLocation refLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  TokenKind refOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PtrToMemberOperatorAST final : public PtrOperatorAST {
 public:
  PtrToMemberOperatorAST() : PtrOperatorAST(ASTKind::PtrToMemberOperator) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FunctionDeclaratorAST final : public DeclaratorModifierAST {
 public:
  FunctionDeclaratorAST()
      : DeclaratorModifierAST(ASTKind::FunctionDeclarator) {}

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ArrayDeclaratorAST final : public DeclaratorModifierAST {
 public:
  ArrayDeclaratorAST() : DeclaratorModifierAST(ASTKind::ArrayDeclarator) {}

  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CxxAttributeAST final : public AttributeSpecifierAST {
 public:
  CxxAttributeAST() : AttributeSpecifierAST(ASTKind::CxxAttribute) {}

  SourceLocation lbracketLoc;
  SourceLocation lbracket2Loc;
  AttributeUsingPrefixAST* attributeUsingPrefix = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation rbracketLoc;
  SourceLocation rbracket2Loc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GCCAttributeAST final : public AttributeSpecifierAST {
 public:
  GCCAttributeAST() : AttributeSpecifierAST(ASTKind::GCCAttribute) {}

  SourceLocation attributeLoc;
  SourceLocation lparenLoc;
  SourceLocation lparen2Loc;
  SourceLocation rparenLoc;
  SourceLocation rparen2Loc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignasAttributeAST final : public AttributeSpecifierAST {
 public:
  AlignasAttributeAST() : AttributeSpecifierAST(ASTKind::AlignasAttribute) {}

  SourceLocation alignasLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation ellipsisLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmAttributeAST final : public AttributeSpecifierAST {
 public:
  AsmAttributeAST() : AttributeSpecifierAST(ASTKind::AsmAttribute) {}

  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  SourceLocation literalLoc;
  SourceLocation rparenLoc;
  const Literal* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ScopedAttributeTokenAST final : public AttributeTokenAST {
 public:
  ScopedAttributeTokenAST()
      : AttributeTokenAST(ASTKind::ScopedAttributeToken) {}

  SourceLocation attributeNamespaceLoc;
  SourceLocation scopeLoc;
  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleAttributeTokenAST final : public AttributeTokenAST {
 public:
  SimpleAttributeTokenAST()
      : AttributeTokenAST(ASTKind::SimpleAttributeToken) {}

  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

}  // namespace cxx
