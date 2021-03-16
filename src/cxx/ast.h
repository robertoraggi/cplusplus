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

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_kind.h>
#include <cxx/ast_visitor.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

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

 public:
  explicit AST(ASTKind kind) : kind_(kind) {}

  virtual ~AST();

  ASTKind kind() const { return kind_; }

  virtual void accept(ASTVisitor* visitor) = 0;

  virtual SourceLocation firstSourceLocation() = 0;
  virtual SourceLocation lastSourceLocation() = 0;

  SourceLocationRange sourceLocationRange() {
    return SourceLocationRange(firstSourceLocation(), lastSourceLocation());
  }
};

inline SourceLocation firstSourceLocation(SourceLocation loc) { return loc; }

template <typename T>
inline SourceLocation firstSourceLocation(T* node) {
  return node ? node->firstSourceLocation() : SourceLocation();
}

template <typename T>
inline SourceLocation firstSourceLocation(List<T>* nodes) {
  for (auto it = nodes; it; it = it->next) {
    if (auto loc = firstSourceLocation(it->value)) return loc;
  }
  return SourceLocation();
}

inline SourceLocation lastSourceLocation(SourceLocation loc) {
  return loc ? loc.next() : SourceLocation();
}

template <typename T>
inline SourceLocation lastSourceLocation(T* node) {
  return node ? node->lastSourceLocation() : SourceLocation();
}

template <typename T>
inline SourceLocation lastSourceLocation(List<T>* nodes) {
  if (!nodes) return SourceLocation();
  if (auto loc = lastSourceLocation(nodes->next)) return loc;
  if (auto loc = lastSourceLocation(nodes->value)) return loc;
  return SourceLocation();
}

class AttributeAST : public AST {
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
  const Type* type = nullptr;
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

class SpecifierAST : public AST {
 public:
  using AST::AST;
};

class StatementAST : public AST {
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NestedNameSpecifierAST final : public AST {
 public:
  NestedNameSpecifierAST() : AST(ASTKind::NestedNameSpecifier) {}

  SourceLocation scopeLoc;
  List<NameAST*>* nameList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UsingDeclaratorAST final : public AST {
 public:
  UsingDeclaratorAST() : AST(ASTKind::UsingDeclarator) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TemplateArgumentAST final : public AST {
 public:
  TemplateArgumentAST() : AST(ASTKind::TemplateArgument) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EnumBaseAST final : public AST {
 public:
  EnumBaseAST() : AST(ASTKind::EnumBase) {}

  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EnumeratorAST final : public AST {
 public:
  EnumeratorAST() : AST(ASTKind::Enumerator) {}

  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DeclaratorAST final : public AST {
 public:
  DeclaratorAST() : AST(ASTKind::Declarator) {}

  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class InitDeclaratorAST final : public AST {
 public:
  InitDeclaratorAST() : AST(ASTKind::InitDeclarator) {}

  DeclaratorAST* declarator = nullptr;
  InitializerAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BaseSpecifierAST final : public AST {
 public:
  BaseSpecifierAST() : AST(ASTKind::BaseSpecifier) {}

  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BaseClauseAST final : public AST {
 public:
  BaseClauseAST() : AST(ASTKind::BaseClause) {}

  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NewTypeIdAST final : public AST {
 public:
  NewTypeIdAST() : AST(ASTKind::NewTypeId) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ParameterDeclarationClauseAST final : public AST {
 public:
  ParameterDeclarationClauseAST() : AST(ASTKind::ParameterDeclarationClause) {}

  List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
  SourceLocation commaLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ParametersAndQualifiersAST final : public AST {
 public:
  ParametersAndQualifiersAST() : AST(ASTKind::ParametersAndQualifiers) {}

  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<SpecifierAST*>* cvQualifierList = nullptr;
  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class LambdaIntroducerAST final : public AST {
 public:
  LambdaIntroducerAST() : AST(ASTKind::LambdaIntroducer) {}

  SourceLocation lbracketLoc;
  SourceLocation captureDefaultLoc;
  List<LambdaCaptureAST*>* captureList = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class LambdaDeclaratorAST final : public AST {
 public:
  LambdaDeclaratorAST() : AST(ASTKind::LambdaDeclarator) {}

  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TrailingReturnTypeAST final : public AST {
 public:
  TrailingReturnTypeAST() : AST(ASTKind::TrailingReturnType) {}

  SourceLocation minusGreaterLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CtorInitializerAST final : public AST {
 public:
  CtorInitializerAST() : AST(ASTKind::CtorInitializer) {}

  SourceLocation colonLoc;
  List<MemInitializerAST*>* memInitializerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BracedMemInitializerAST final : public MemInitializerAST {
 public:
  BracedMemInitializerAST()
      : MemInitializerAST(ASTKind::BracedMemInitializer) {}

  NameAST* name = nullptr;
  BracedInitListAST* bracedInitList = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  ThisLambdaCaptureAST() : LambdaCaptureAST(ASTKind::ThisLambdaCapture) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DerefThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  DerefThisLambdaCaptureAST()
      : LambdaCaptureAST(ASTKind::DerefThisLambdaCapture) {}

  SourceLocation starLoc;
  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SimpleLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  SimpleLambdaCaptureAST() : LambdaCaptureAST(ASTKind::SimpleLambdaCapture) {}

  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class RefLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  RefLambdaCaptureAST() : LambdaCaptureAST(ASTKind::RefLambdaCapture) {}

  SourceLocation ampLoc;
  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class RefInitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  RefInitLambdaCaptureAST() : LambdaCaptureAST(ASTKind::RefInitLambdaCapture) {}

  SourceLocation ampLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  InitializerAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class InitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  InitLambdaCaptureAST() : LambdaCaptureAST(ASTKind::InitLambdaCapture) {}

  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  InitializerAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EqualInitializerAST final : public InitializerAST {
 public:
  EqualInitializerAST() : InitializerAST(ASTKind::EqualInitializer) {}

  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BracedInitListAST final : public InitializerAST {
 public:
  BracedInitListAST() : InitializerAST(ASTKind::BracedInitList) {}

  SourceLocation lbraceLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ParenInitializerAST final : public InitializerAST {
 public:
  ParenInitializerAST() : InitializerAST(ASTKind::ParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NewParenInitializerAST final : public NewInitializerAST {
 public:
  NewParenInitializerAST() : NewInitializerAST(ASTKind::NewParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NewBracedInitializerAST final : public NewInitializerAST {
 public:
  NewBracedInitializerAST()
      : NewInitializerAST(ASTKind::NewBracedInitializer) {}

  BracedInitListAST* bracedInit = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EllipsisExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  EllipsisExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::EllipsisExceptionDeclaration) {}

  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypeExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  TypeExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::TypeExceptionDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DefaultFunctionBodyAST final : public FunctionBodyAST {
 public:
  DefaultFunctionBodyAST() : FunctionBodyAST(ASTKind::DefaultFunctionBody) {}

  SourceLocation equalLoc;
  SourceLocation defaultLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CompoundStatementFunctionBodyAST final : public FunctionBodyAST {
 public:
  CompoundStatementFunctionBodyAST()
      : FunctionBodyAST(ASTKind::CompoundStatementFunctionBody) {}

  CtorInitializerAST* ctorInitializer = nullptr;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DeleteFunctionBodyAST final : public FunctionBodyAST {
 public:
  DeleteFunctionBodyAST() : FunctionBodyAST(ASTKind::DeleteFunctionBody) {}

  SourceLocation equalLoc;
  SourceLocation deleteLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TranslationUnitAST final : public UnitAST {
 public:
  TranslationUnitAST() : UnitAST(ASTKind::TranslationUnit) {}

  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ModuleUnitAST final : public UnitAST {
 public:
  ModuleUnitAST() : UnitAST(ASTKind::ModuleUnit) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ThisExpressionAST final : public ExpressionAST {
 public:
  ThisExpressionAST() : ExpressionAST(ASTKind::ThisExpression) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CharLiteralExpressionAST final : public ExpressionAST {
 public:
  CharLiteralExpressionAST() : ExpressionAST(ASTKind::CharLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BoolLiteralExpressionAST final : public ExpressionAST {
 public:
  BoolLiteralExpressionAST() : ExpressionAST(ASTKind::BoolLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class IntLiteralExpressionAST final : public ExpressionAST {
 public:
  IntLiteralExpressionAST() : ExpressionAST(ASTKind::IntLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class FloatLiteralExpressionAST final : public ExpressionAST {
 public:
  FloatLiteralExpressionAST()
      : ExpressionAST(ASTKind::FloatLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NullptrLiteralExpressionAST final : public ExpressionAST {
 public:
  NullptrLiteralExpressionAST()
      : ExpressionAST(ASTKind::NullptrLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class StringLiteralExpressionAST final : public ExpressionAST {
 public:
  StringLiteralExpressionAST()
      : ExpressionAST(ASTKind::StringLiteralExpression) {}

  List<SourceLocation>* stringLiteralList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UserDefinedStringLiteralExpressionAST final : public ExpressionAST {
 public:
  UserDefinedStringLiteralExpressionAST()
      : ExpressionAST(ASTKind::UserDefinedStringLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class IdExpressionAST final : public ExpressionAST {
 public:
  IdExpressionAST() : ExpressionAST(ASTKind::IdExpression) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NestedExpressionAST final : public ExpressionAST {
 public:
  NestedExpressionAST() : ExpressionAST(ASTKind::NestedExpression) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class LambdaExpressionAST final : public ExpressionAST {
 public:
  LambdaExpressionAST() : ExpressionAST(ASTKind::LambdaExpression) {}

  LambdaIntroducerAST* lambdaIntroducer = nullptr;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  LambdaDeclaratorAST* lambdaDeclarator = nullptr;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SizeofExpressionAST final : public ExpressionAST {
 public:
  SizeofExpressionAST() : ExpressionAST(ASTKind::SizeofExpression) {}

  SourceLocation sizeofLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SizeofTypeExpressionAST final : public ExpressionAST {
 public:
  SizeofTypeExpressionAST() : ExpressionAST(ASTKind::SizeofTypeExpression) {}

  SourceLocation sizeofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SizeofPackExpressionAST final : public ExpressionAST {
 public:
  SizeofPackExpressionAST() : ExpressionAST(ASTKind::SizeofPackExpression) {}

  SourceLocation sizeofLoc;
  SourceLocation ellipsisLoc;
  SourceLocation lparenLoc;
  SourceLocation identifierLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypeidExpressionAST final : public ExpressionAST {
 public:
  TypeidExpressionAST() : ExpressionAST(ASTKind::TypeidExpression) {}

  SourceLocation typeidLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AlignofExpressionAST final : public ExpressionAST {
 public:
  AlignofExpressionAST() : ExpressionAST(ASTKind::AlignofExpression) {}

  SourceLocation alignofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UnaryExpressionAST final : public ExpressionAST {
 public:
  UnaryExpressionAST() : ExpressionAST(ASTKind::UnaryExpression) {}

  SourceLocation opLoc;
  ExpressionAST* expression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BinaryExpressionAST final : public ExpressionAST {
 public:
  BinaryExpressionAST() : ExpressionAST(ASTKind::BinaryExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AssignmentExpressionAST final : public ExpressionAST {
 public:
  AssignmentExpressionAST() : ExpressionAST(ASTKind::AssignmentExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BracedTypeConstructionAST final : public ExpressionAST {
 public:
  BracedTypeConstructionAST()
      : ExpressionAST(ASTKind::BracedTypeConstruction) {}

  SpecifierAST* typeSpecifier = nullptr;
  BracedInitListAST* bracedInitList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypeConstructionAST final : public ExpressionAST {
 public:
  TypeConstructionAST() : ExpressionAST(ASTKind::TypeConstruction) {}

  SpecifierAST* typeSpecifier = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CallExpressionAST final : public ExpressionAST {
 public:
  CallExpressionAST() : ExpressionAST(ASTKind::CallExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SubscriptExpressionAST final : public ExpressionAST {
 public:
  SubscriptExpressionAST() : ExpressionAST(ASTKind::SubscriptExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lbracketLoc;
  ExpressionAST* indexExpression = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class MemberExpressionAST final : public ExpressionAST {
 public:
  MemberExpressionAST() : ExpressionAST(ASTKind::MemberExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation accessLoc;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CastExpressionAST final : public ExpressionAST {
 public:
  CastExpressionAST() : ExpressionAST(ASTKind::CastExpression) {}

  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NewExpressionAST final : public ExpressionAST {
 public:
  NewExpressionAST() : ExpressionAST(ASTKind::NewExpression) {}

  SourceLocation scopeLoc;
  SourceLocation newLoc;
  NewTypeIdAST* typeId = nullptr;
  NewInitializerAST* newInitalizer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ThrowExpressionAST final : public ExpressionAST {
 public:
  ThrowExpressionAST() : ExpressionAST(ASTKind::ThrowExpression) {}

  SourceLocation throwLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NoexceptExpressionAST final : public ExpressionAST {
 public:
  NoexceptExpressionAST() : ExpressionAST(ASTKind::NoexceptExpression) {}

  SourceLocation noexceptLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class LabeledStatementAST final : public StatementAST {
 public:
  LabeledStatementAST() : StatementAST(ASTKind::LabeledStatement) {}

  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CaseStatementAST final : public StatementAST {
 public:
  CaseStatementAST() : StatementAST(ASTKind::CaseStatement) {}

  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DefaultStatementAST final : public StatementAST {
 public:
  DefaultStatementAST() : StatementAST(ASTKind::DefaultStatement) {}

  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ExpressionStatementAST final : public StatementAST {
 public:
  ExpressionStatementAST() : StatementAST(ASTKind::ExpressionStatement) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CompoundStatementAST final : public StatementAST {
 public:
  CompoundStatementAST() : StatementAST(ASTKind::CompoundStatement) {}

  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class BreakStatementAST final : public StatementAST {
 public:
  BreakStatementAST() : StatementAST(ASTKind::BreakStatement) {}

  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ContinueStatementAST final : public StatementAST {
 public:
  ContinueStatementAST() : StatementAST(ASTKind::ContinueStatement) {}

  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ReturnStatementAST final : public StatementAST {
 public:
  ReturnStatementAST() : StatementAST(ASTKind::ReturnStatement) {}

  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class GotoStatementAST final : public StatementAST {
 public:
  GotoStatementAST() : StatementAST(ASTKind::GotoStatement) {}

  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class CoroutineReturnStatementAST final : public StatementAST {
 public:
  CoroutineReturnStatementAST()
      : StatementAST(ASTKind::CoroutineReturnStatement) {}

  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DeclarationStatementAST final : public StatementAST {
 public:
  DeclarationStatementAST() : StatementAST(ASTKind::DeclarationStatement) {}

  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TryBlockStatementAST final : public StatementAST {
 public:
  TryBlockStatementAST() : StatementAST(ASTKind::TryBlockStatement) {}

  SourceLocation tryLoc;
  CompoundStatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AccessDeclarationAST final : public DeclarationAST {
 public:
  AccessDeclarationAST() : DeclarationAST(ASTKind::AccessDeclaration) {}

  SourceLocation accessLoc;
  SourceLocation colonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class FunctionDefinitionAST final : public DeclarationAST {
 public:
  FunctionDefinitionAST() : DeclarationAST(ASTKind::FunctionDefinition) {}

  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  FunctionBodyAST* functionBody = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ForRangeDeclarationAST final : public DeclarationAST {
 public:
  ForRangeDeclarationAST() : DeclarationAST(ASTKind::ForRangeDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AliasDeclarationAST final : public DeclarationAST {
 public:
  AliasDeclarationAST() : DeclarationAST(ASTKind::AliasDeclaration) {}

  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SimpleDeclarationAST final : public DeclarationAST {
 public:
  SimpleDeclarationAST() : DeclarationAST(ASTKind::SimpleDeclaration) {}

  List<AttributeAST*>* attributes = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class StaticAssertDeclarationAST final : public DeclarationAST {
 public:
  StaticAssertDeclarationAST()
      : DeclarationAST(ASTKind::StaticAssertDeclaration) {}

  SourceLocation staticAssertLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation commaLoc;
  List<SourceLocation>* stringLiteralList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EmptyDeclarationAST final : public DeclarationAST {
 public:
  EmptyDeclarationAST() : DeclarationAST(ASTKind::EmptyDeclaration) {}

  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AttributeDeclarationAST final : public DeclarationAST {
 public:
  AttributeDeclarationAST() : DeclarationAST(ASTKind::AttributeDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class OpaqueEnumDeclarationAST final : public DeclarationAST {
 public:
  OpaqueEnumDeclarationAST() : DeclarationAST(ASTKind::OpaqueEnumDeclaration) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation emicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UsingEnumDeclarationAST final : public DeclarationAST {
 public:
  UsingEnumDeclarationAST() : DeclarationAST(ASTKind::UsingEnumDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NamespaceDefinitionAST final : public DeclarationAST {
 public:
  NamespaceDefinitionAST() : DeclarationAST(ASTKind::NamespaceDefinition) {}

  SourceLocation inlineLoc;
  SourceLocation namespaceLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  List<AttributeAST*>* extraAttributeList = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UsingDirectiveAST final : public DeclarationAST {
 public:
  UsingDirectiveAST() : DeclarationAST(ASTKind::UsingDirective) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UsingDeclarationAST final : public DeclarationAST {
 public:
  UsingDeclarationAST() : DeclarationAST(ASTKind::UsingDeclaration) {}

  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AsmDeclarationAST final : public DeclarationAST {
 public:
  AsmDeclarationAST() : DeclarationAST(ASTKind::AsmDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  List<SourceLocation>* stringLiteralList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ExportDeclarationAST final : public DeclarationAST {
 public:
  ExportDeclarationAST() : DeclarationAST(ASTKind::ExportDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ModuleImportDeclarationAST final : public DeclarationAST {
 public:
  ModuleImportDeclarationAST()
      : DeclarationAST(ASTKind::ModuleImportDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TemplateDeclarationAST final : public DeclarationAST {
 public:
  TemplateDeclarationAST() : DeclarationAST(ASTKind::TemplateDeclaration) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypenameTypeParameterAST final : public DeclarationAST {
 public:
  TypenameTypeParameterAST() : DeclarationAST(ASTKind::TypenameTypeParameter) {}

  SourceLocation classKeyLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypenamePackTypeParameterAST final : public DeclarationAST {
 public:
  TypenamePackTypeParameterAST()
      : DeclarationAST(ASTKind::TypenamePackTypeParameter) {}

  SourceLocation classKeyLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TemplateTypeParameterAST final : public DeclarationAST {
 public:
  TemplateTypeParameterAST() : DeclarationAST(ASTKind::TemplateTypeParameter) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  SourceLocation classKeyLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DeductionGuideAST final : public DeclarationAST {
 public:
  DeductionGuideAST() : DeclarationAST(ASTKind::DeductionGuide) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ExplicitInstantiationAST final : public DeclarationAST {
 public:
  ExplicitInstantiationAST() : DeclarationAST(ASTKind::ExplicitInstantiation) {}

  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ParameterDeclarationAST final : public DeclarationAST {
 public:
  ParameterDeclarationAST() : DeclarationAST(ASTKind::ParameterDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class LinkageSpecificationAST final : public DeclarationAST {
 public:
  LinkageSpecificationAST() : DeclarationAST(ASTKind::LinkageSpecification) {}

  SourceLocation externLoc;
  SourceLocation stringliteralLoc;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class SimpleNameAST final : public NameAST {
 public:
  SimpleNameAST() : NameAST(ASTKind::SimpleName) {}

  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DestructorNameAST final : public NameAST {
 public:
  DestructorNameAST() : NameAST(ASTKind::DestructorName) {}

  SourceLocation tildeLoc;
  NameAST* id = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DecltypeNameAST final : public NameAST {
 public:
  DecltypeNameAST() : NameAST(ASTKind::DecltypeName) {}

  SpecifierAST* decltypeSpecifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
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

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TemplateNameAST final : public NameAST {
 public:
  TemplateNameAST() : NameAST(ASTKind::TemplateName) {}

  NameAST* id = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class QualifiedNameAST final : public NameAST {
 public:
  QualifiedNameAST() : NameAST(ASTKind::QualifiedName) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  NameAST* id = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypedefSpecifierAST final : public SpecifierAST {
 public:
  TypedefSpecifierAST() : SpecifierAST(ASTKind::TypedefSpecifier) {}

  SourceLocation typedefLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class FriendSpecifierAST final : public SpecifierAST {
 public:
  FriendSpecifierAST() : SpecifierAST(ASTKind::FriendSpecifier) {}

  SourceLocation friendLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ConstevalSpecifierAST final : public SpecifierAST {
 public:
  ConstevalSpecifierAST() : SpecifierAST(ASTKind::ConstevalSpecifier) {}

  SourceLocation constevalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ConstinitSpecifierAST final : public SpecifierAST {
 public:
  ConstinitSpecifierAST() : SpecifierAST(ASTKind::ConstinitSpecifier) {}

  SourceLocation constinitLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ConstexprSpecifierAST final : public SpecifierAST {
 public:
  ConstexprSpecifierAST() : SpecifierAST(ASTKind::ConstexprSpecifier) {}

  SourceLocation constexprLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class InlineSpecifierAST final : public SpecifierAST {
 public:
  InlineSpecifierAST() : SpecifierAST(ASTKind::InlineSpecifier) {}

  SourceLocation inlineLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class StaticSpecifierAST final : public SpecifierAST {
 public:
  StaticSpecifierAST() : SpecifierAST(ASTKind::StaticSpecifier) {}

  SourceLocation staticLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ExternSpecifierAST final : public SpecifierAST {
 public:
  ExternSpecifierAST() : SpecifierAST(ASTKind::ExternSpecifier) {}

  SourceLocation externLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ThreadLocalSpecifierAST final : public SpecifierAST {
 public:
  ThreadLocalSpecifierAST() : SpecifierAST(ASTKind::ThreadLocalSpecifier) {}

  SourceLocation threadLocalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ThreadSpecifierAST final : public SpecifierAST {
 public:
  ThreadSpecifierAST() : SpecifierAST(ASTKind::ThreadSpecifier) {}

  SourceLocation threadLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class MutableSpecifierAST final : public SpecifierAST {
 public:
  MutableSpecifierAST() : SpecifierAST(ASTKind::MutableSpecifier) {}

  SourceLocation mutableLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class VirtualSpecifierAST final : public SpecifierAST {
 public:
  VirtualSpecifierAST() : SpecifierAST(ASTKind::VirtualSpecifier) {}

  SourceLocation virtualLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ExplicitSpecifierAST final : public SpecifierAST {
 public:
  ExplicitSpecifierAST() : SpecifierAST(ASTKind::ExplicitSpecifier) {}

  SourceLocation explicitLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AutoTypeSpecifierAST final : public SpecifierAST {
 public:
  AutoTypeSpecifierAST() : SpecifierAST(ASTKind::AutoTypeSpecifier) {}

  SourceLocation autoLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class VoidTypeSpecifierAST final : public SpecifierAST {
 public:
  VoidTypeSpecifierAST() : SpecifierAST(ASTKind::VoidTypeSpecifier) {}

  SourceLocation voidLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class VaListTypeSpecifierAST final : public SpecifierAST {
 public:
  VaListTypeSpecifierAST() : SpecifierAST(ASTKind::VaListTypeSpecifier) {}

  SourceLocation specifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class IntegralTypeSpecifierAST final : public SpecifierAST {
 public:
  IntegralTypeSpecifierAST() : SpecifierAST(ASTKind::IntegralTypeSpecifier) {}

  SourceLocation specifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class FloatingPointTypeSpecifierAST final : public SpecifierAST {
 public:
  FloatingPointTypeSpecifierAST()
      : SpecifierAST(ASTKind::FloatingPointTypeSpecifier) {}

  SourceLocation specifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ComplexTypeSpecifierAST final : public SpecifierAST {
 public:
  ComplexTypeSpecifierAST() : SpecifierAST(ASTKind::ComplexTypeSpecifier) {}

  SourceLocation complexLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NamedTypeSpecifierAST final : public SpecifierAST {
 public:
  NamedTypeSpecifierAST() : SpecifierAST(ASTKind::NamedTypeSpecifier) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class AtomicTypeSpecifierAST final : public SpecifierAST {
 public:
  AtomicTypeSpecifierAST() : SpecifierAST(ASTKind::AtomicTypeSpecifier) {}

  SourceLocation atomicLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class UnderlyingTypeSpecifierAST final : public SpecifierAST {
 public:
  UnderlyingTypeSpecifierAST()
      : SpecifierAST(ASTKind::UnderlyingTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ElaboratedTypeSpecifierAST final : public SpecifierAST {
 public:
  ElaboratedTypeSpecifierAST()
      : SpecifierAST(ASTKind::ElaboratedTypeSpecifier) {}

  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DecltypeAutoSpecifierAST final : public SpecifierAST {
 public:
  DecltypeAutoSpecifierAST() : SpecifierAST(ASTKind::DecltypeAutoSpecifier) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  SourceLocation autoLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class DecltypeSpecifierAST final : public SpecifierAST {
 public:
  DecltypeSpecifierAST() : SpecifierAST(ASTKind::DecltypeSpecifier) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypeofSpecifierAST final : public SpecifierAST {
 public:
  TypeofSpecifierAST() : SpecifierAST(ASTKind::TypeofSpecifier) {}

  SourceLocation typeofLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class PlaceholderTypeSpecifierAST final : public SpecifierAST {
 public:
  PlaceholderTypeSpecifierAST()
      : SpecifierAST(ASTKind::PlaceholderTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ConstQualifierAST final : public SpecifierAST {
 public:
  ConstQualifierAST() : SpecifierAST(ASTKind::ConstQualifier) {}

  SourceLocation constLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class VolatileQualifierAST final : public SpecifierAST {
 public:
  VolatileQualifierAST() : SpecifierAST(ASTKind::VolatileQualifier) {}

  SourceLocation volatileLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class RestrictQualifierAST final : public SpecifierAST {
 public:
  RestrictQualifierAST() : SpecifierAST(ASTKind::RestrictQualifier) {}

  SourceLocation restrictLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class EnumSpecifierAST final : public SpecifierAST {
 public:
  EnumSpecifierAST() : SpecifierAST(ASTKind::EnumSpecifier) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  List<EnumeratorAST*>* enumeratorList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ClassSpecifierAST final : public SpecifierAST {
 public:
  ClassSpecifierAST() : SpecifierAST(ASTKind::ClassSpecifier) {}

  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;
  BaseClauseAST* baseClause = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class TypenameSpecifierAST final : public SpecifierAST {
 public:
  TypenameSpecifierAST() : SpecifierAST(ASTKind::TypenameSpecifier) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class IdDeclaratorAST final : public CoreDeclaratorAST {
 public:
  IdDeclaratorAST() : CoreDeclaratorAST(ASTKind::IdDeclarator) {}

  SourceLocation ellipsisLoc;
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class NestedDeclaratorAST final : public CoreDeclaratorAST {
 public:
  NestedDeclaratorAST() : CoreDeclaratorAST(ASTKind::NestedDeclarator) {}

  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class PointerOperatorAST final : public PtrOperatorAST {
 public:
  PointerOperatorAST() : PtrOperatorAST(ASTKind::PointerOperator) {}

  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ReferenceOperatorAST final : public PtrOperatorAST {
 public:
  ReferenceOperatorAST() : PtrOperatorAST(ASTKind::ReferenceOperator) {}

  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class PtrToMemberOperatorAST final : public PtrOperatorAST {
 public:
  PtrToMemberOperatorAST() : PtrOperatorAST(ASTKind::PtrToMemberOperator) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class FunctionDeclaratorAST final : public DeclaratorModifierAST {
 public:
  FunctionDeclaratorAST()
      : DeclaratorModifierAST(ASTKind::FunctionDeclarator) {}

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

class ArrayDeclaratorAST final : public DeclaratorModifierAST {
 public:
  ArrayDeclaratorAST() : DeclaratorModifierAST(ASTKind::ArrayDeclarator) {}

  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

}  // namespace cxx
