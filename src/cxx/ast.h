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
#include <cxx/token.h>

namespace cxx {

template <typename T>
struct List final : Managed {
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

struct AST : Managed {
  ASTKind kind_;

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

struct AttributeAST : AST {
  using AST::AST;
};

struct CoreDeclaratorAST : AST {
  using AST::AST;
};

struct DeclarationAST : AST {
  using AST::AST;
};

struct DeclaratorModifierAST : AST {
  using AST::AST;
};

struct ExceptionDeclarationAST : AST {
  using AST::AST;
};

struct ExpressionAST : AST {
  using AST::AST;
};

struct InitializerAST : AST {
  using AST::AST;
};

struct NameAST : AST {
  using AST::AST;
};

struct NewInitializerAST : AST {
  using AST::AST;
};

struct PtrOperatorAST : AST {
  using AST::AST;
};

struct SpecifierAST : AST {
  using AST::AST;
};

struct StatementAST : AST {
  using AST::AST;
};

struct UnitAST : AST {
  using AST::AST;
};

struct TypeIdAST final : AST {
  TypeIdAST() : AST(ASTKind::TypeId) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedNameSpecifierAST final : AST {
  NestedNameSpecifierAST() : AST(ASTKind::NestedNameSpecifier) {}

  SourceLocation scopeLoc;
  List<NameAST*>* nameList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UsingDeclaratorAST final : AST {
  UsingDeclaratorAST() : AST(ASTKind::UsingDeclarator) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct HandlerAST final : AST {
  HandlerAST() : AST(ASTKind::Handler) {}

  SourceLocation catchLoc;
  SourceLocation lparenLoc;
  ExceptionDeclarationAST* exceptionDeclaration = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TemplateArgumentAST final : AST {
  TemplateArgumentAST() : AST(ASTKind::TemplateArgument) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumBaseAST final : AST {
  EnumBaseAST() : AST(ASTKind::EnumBase) {}

  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumeratorAST final : AST {
  EnumeratorAST() : AST(ASTKind::Enumerator) {}

  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DeclaratorAST final : AST {
  DeclaratorAST() : AST(ASTKind::Declarator) {}

  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BaseSpecifierAST final : AST {
  BaseSpecifierAST() : AST(ASTKind::BaseSpecifier) {}

  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BaseClauseAST final : AST {
  BaseClauseAST() : AST(ASTKind::BaseClause) {}

  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NewTypeIdAST final : AST {
  NewTypeIdAST() : AST(ASTKind::NewTypeId) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParameterDeclarationClauseAST final : AST {
  ParameterDeclarationClauseAST() : AST(ASTKind::ParameterDeclarationClause) {}

  List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
  SourceLocation commaLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParametersAndQualifiersAST final : AST {
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

struct LambdaIntroducerAST final : AST {
  LambdaIntroducerAST() : AST(ASTKind::LambdaIntroducer) {}

  SourceLocation lbracketLoc;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct LambdaDeclaratorAST final : AST {
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

struct TrailingReturnTypeAST final : AST {
  TrailingReturnTypeAST() : AST(ASTKind::TrailingReturnType) {}

  SourceLocation minusGreaterLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EqualInitializerAST final : InitializerAST {
  EqualInitializerAST() : InitializerAST(ASTKind::EqualInitializer) {}

  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BracedInitListAST final : InitializerAST {
  BracedInitListAST() : InitializerAST(ASTKind::BracedInitList) {}

  SourceLocation lbraceLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParenInitializerAST final : InitializerAST {
  ParenInitializerAST() : InitializerAST(ASTKind::ParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NewParenInitializerAST final : NewInitializerAST {
  NewParenInitializerAST() : NewInitializerAST(ASTKind::NewParenInitializer) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NewBracedInitializerAST final : NewInitializerAST {
  NewBracedInitializerAST()
      : NewInitializerAST(ASTKind::NewBracedInitializer) {}

  BracedInitListAST* bracedInit = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EllipsisExceptionDeclarationAST final : ExceptionDeclarationAST {
  EllipsisExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::EllipsisExceptionDeclaration) {}

  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TypeExceptionDeclarationAST final : ExceptionDeclarationAST {
  TypeExceptionDeclarationAST()
      : ExceptionDeclarationAST(ASTKind::TypeExceptionDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TranslationUnitAST final : UnitAST {
  TranslationUnitAST() : UnitAST(ASTKind::TranslationUnit) {}

  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ModuleUnitAST final : UnitAST {
  ModuleUnitAST() : UnitAST(ASTKind::ModuleUnit) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ThisExpressionAST final : ExpressionAST {
  ThisExpressionAST() : ExpressionAST(ASTKind::ThisExpression) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CharLiteralExpressionAST final : ExpressionAST {
  CharLiteralExpressionAST() : ExpressionAST(ASTKind::CharLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BoolLiteralExpressionAST final : ExpressionAST {
  BoolLiteralExpressionAST() : ExpressionAST(ASTKind::BoolLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IntLiteralExpressionAST final : ExpressionAST {
  IntLiteralExpressionAST() : ExpressionAST(ASTKind::IntLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FloatLiteralExpressionAST final : ExpressionAST {
  FloatLiteralExpressionAST()
      : ExpressionAST(ASTKind::FloatLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NullptrLiteralExpressionAST final : ExpressionAST {
  NullptrLiteralExpressionAST()
      : ExpressionAST(ASTKind::NullptrLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct StringLiteralExpressionAST final : ExpressionAST {
  StringLiteralExpressionAST()
      : ExpressionAST(ASTKind::StringLiteralExpression) {}

  List<SourceLocation>* stringLiteralList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UserDefinedStringLiteralExpressionAST final : ExpressionAST {
  UserDefinedStringLiteralExpressionAST()
      : ExpressionAST(ASTKind::UserDefinedStringLiteralExpression) {}

  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IdExpressionAST final : ExpressionAST {
  IdExpressionAST() : ExpressionAST(ASTKind::IdExpression) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedExpressionAST final : ExpressionAST {
  NestedExpressionAST() : ExpressionAST(ASTKind::NestedExpression) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct LambdaExpressionAST final : ExpressionAST {
  LambdaExpressionAST() : ExpressionAST(ASTKind::LambdaExpression) {}

  LambdaIntroducerAST* lambdaIntroducer = nullptr;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  LambdaDeclaratorAST* lambdaDeclarator = nullptr;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BinaryExpressionAST final : ExpressionAST {
  BinaryExpressionAST() : ExpressionAST(ASTKind::BinaryExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AssignmentExpressionAST final : ExpressionAST {
  AssignmentExpressionAST() : ExpressionAST(ASTKind::AssignmentExpression) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CallExpressionAST final : ExpressionAST {
  CallExpressionAST() : ExpressionAST(ASTKind::CallExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct SubscriptExpressionAST final : ExpressionAST {
  SubscriptExpressionAST() : ExpressionAST(ASTKind::SubscriptExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lbracketLoc;
  ExpressionAST* indexExpression = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct MemberExpressionAST final : ExpressionAST {
  MemberExpressionAST() : ExpressionAST(ASTKind::MemberExpression) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation accessLoc;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ConditionalExpressionAST final : ExpressionAST {
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

struct CppCastExpressionAST final : ExpressionAST {
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

struct NewExpressionAST final : ExpressionAST {
  NewExpressionAST() : ExpressionAST(ASTKind::NewExpression) {}

  SourceLocation scopeLoc;
  SourceLocation newLoc;
  NewTypeIdAST* typeId = nullptr;
  NewInitializerAST* newInitalizer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct LabeledStatementAST final : StatementAST {
  LabeledStatementAST() : StatementAST(ASTKind::LabeledStatement) {}

  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CaseStatementAST final : StatementAST {
  CaseStatementAST() : StatementAST(ASTKind::CaseStatement) {}

  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DefaultStatementAST final : StatementAST {
  DefaultStatementAST() : StatementAST(ASTKind::DefaultStatement) {}

  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionStatementAST() : StatementAST(ASTKind::ExpressionStatement) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CompoundStatementAST final : StatementAST {
  CompoundStatementAST() : StatementAST(ASTKind::CompoundStatement) {}

  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IfStatementAST final : StatementAST {
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

struct SwitchStatementAST final : StatementAST {
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

struct WhileStatementAST final : StatementAST {
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

struct DoStatementAST final : StatementAST {
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

struct ForRangeStatementAST final : StatementAST {
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

struct ForStatementAST final : StatementAST {
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

struct BreakStatementAST final : StatementAST {
  BreakStatementAST() : StatementAST(ASTKind::BreakStatement) {}

  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ContinueStatementAST final : StatementAST {
  ContinueStatementAST() : StatementAST(ASTKind::ContinueStatement) {}

  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ReturnStatementAST final : StatementAST {
  ReturnStatementAST() : StatementAST(ASTKind::ReturnStatement) {}

  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct GotoStatementAST final : StatementAST {
  GotoStatementAST() : StatementAST(ASTKind::GotoStatement) {}

  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CoroutineReturnStatementAST final : StatementAST {
  CoroutineReturnStatementAST()
      : StatementAST(ASTKind::CoroutineReturnStatement) {}

  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationStatementAST() : StatementAST(ASTKind::DeclarationStatement) {}

  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TryBlockStatementAST final : StatementAST {
  TryBlockStatementAST() : StatementAST(ASTKind::TryBlockStatement) {}

  SourceLocation tryLoc;
  StatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FunctionDefinitionAST final : DeclarationAST {
  FunctionDefinitionAST() : DeclarationAST(ASTKind::FunctionDefinition) {}

  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  StatementAST* functionBody = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ConceptDefinitionAST final : DeclarationAST {
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

struct ForRangeDeclarationAST final : DeclarationAST {
  ForRangeDeclarationAST() : DeclarationAST(ASTKind::ForRangeDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AliasDeclarationAST final : DeclarationAST {
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

struct SimpleDeclarationAST final : DeclarationAST {
  SimpleDeclarationAST() : DeclarationAST(ASTKind::SimpleDeclaration) {}

  List<AttributeAST*>* attributes = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<DeclaratorAST*>* declaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct StaticAssertDeclarationAST final : DeclarationAST {
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

struct EmptyDeclarationAST final : DeclarationAST {
  EmptyDeclarationAST() : DeclarationAST(ASTKind::EmptyDeclaration) {}

  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AttributeDeclarationAST final : DeclarationAST {
  AttributeDeclarationAST() : DeclarationAST(ASTKind::AttributeDeclaration) {}

  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct OpaqueEnumDeclarationAST final : DeclarationAST {
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

struct UsingEnumDeclarationAST final : DeclarationAST {
  UsingEnumDeclarationAST() : DeclarationAST(ASTKind::UsingEnumDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NamespaceDefinitionAST final : DeclarationAST {
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

struct NamespaceAliasDefinitionAST final : DeclarationAST {
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

struct UsingDirectiveAST final : DeclarationAST {
  UsingDirectiveAST() : DeclarationAST(ASTKind::UsingDirective) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UsingDeclarationAST final : DeclarationAST {
  UsingDeclarationAST() : DeclarationAST(ASTKind::UsingDeclaration) {}

  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AsmDeclarationAST final : DeclarationAST {
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

struct ExportDeclarationAST final : DeclarationAST {
  ExportDeclarationAST() : DeclarationAST(ASTKind::ExportDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ModuleImportDeclarationAST final : DeclarationAST {
  ModuleImportDeclarationAST()
      : DeclarationAST(ASTKind::ModuleImportDeclaration) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TemplateDeclarationAST final : DeclarationAST {
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

struct DeductionGuideAST final : DeclarationAST {
  DeductionGuideAST() : DeclarationAST(ASTKind::DeductionGuide) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExplicitInstantiationAST final : DeclarationAST {
  ExplicitInstantiationAST() : DeclarationAST(ASTKind::ExplicitInstantiation) {}

  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParameterDeclarationAST final : DeclarationAST {
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

struct LinkageSpecificationAST final : DeclarationAST {
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

struct SimpleNameAST final : NameAST {
  SimpleNameAST() : NameAST(ASTKind::SimpleName) {}

  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DestructorNameAST final : NameAST {
  DestructorNameAST() : NameAST(ASTKind::DestructorName) {}

  SourceLocation tildeLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeNameAST final : NameAST {
  DecltypeNameAST() : NameAST(ASTKind::DecltypeName) {}

  SpecifierAST* decltypeSpecifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct OperatorNameAST final : NameAST {
  OperatorNameAST() : NameAST(ASTKind::OperatorName) {}

  SourceLocation opLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TemplateNameAST final : NameAST {
  TemplateNameAST() : NameAST(ASTKind::TemplateName) {}

  NameAST* name = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct QualifiedNameAST final : NameAST {
  QualifiedNameAST() : NameAST(ASTKind::QualifiedName) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct SimpleSpecifierAST final : SpecifierAST {
  SimpleSpecifierAST() : SpecifierAST(ASTKind::SimpleSpecifier) {}

  SourceLocation specifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExplicitSpecifierAST final : SpecifierAST {
  ExplicitSpecifierAST() : SpecifierAST(ASTKind::ExplicitSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NamedTypeSpecifierAST final : SpecifierAST {
  NamedTypeSpecifierAST() : SpecifierAST(ASTKind::NamedTypeSpecifier) {}

  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PlaceholderTypeSpecifierHelperAST final : SpecifierAST {
  PlaceholderTypeSpecifierHelperAST()
      : SpecifierAST(ASTKind::PlaceholderTypeSpecifierHelper) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeSpecifierTypeSpecifierAST final : SpecifierAST {
  DecltypeSpecifierTypeSpecifierAST()
      : SpecifierAST(ASTKind::DecltypeSpecifierTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UnderlyingTypeSpecifierAST final : SpecifierAST {
  UnderlyingTypeSpecifierAST()
      : SpecifierAST(ASTKind::UnderlyingTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AtomicTypeSpecifierAST final : SpecifierAST {
  AtomicTypeSpecifierAST() : SpecifierAST(ASTKind::AtomicTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ElaboratedTypeSpecifierAST final : SpecifierAST {
  ElaboratedTypeSpecifierAST()
      : SpecifierAST(ASTKind::ElaboratedTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeSpecifierAST final : SpecifierAST {
  DecltypeSpecifierAST() : SpecifierAST(ASTKind::DecltypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PlaceholderTypeSpecifierAST final : SpecifierAST {
  PlaceholderTypeSpecifierAST()
      : SpecifierAST(ASTKind::PlaceholderTypeSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CvQualifierAST final : SpecifierAST {
  CvQualifierAST() : SpecifierAST(ASTKind::CvQualifier) {}

  SourceLocation qualifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumSpecifierAST final : SpecifierAST {
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

struct ClassSpecifierAST final : SpecifierAST {
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

struct TypenameSpecifierAST final : SpecifierAST {
  TypenameSpecifierAST() : SpecifierAST(ASTKind::TypenameSpecifier) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IdDeclaratorAST final : CoreDeclaratorAST {
  IdDeclaratorAST() : CoreDeclaratorAST(ASTKind::IdDeclarator) {}

  SourceLocation ellipsisLoc;
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedDeclaratorAST final : CoreDeclaratorAST {
  NestedDeclaratorAST() : CoreDeclaratorAST(ASTKind::NestedDeclarator) {}

  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PointerOperatorAST final : PtrOperatorAST {
  PointerOperatorAST() : PtrOperatorAST(ASTKind::PointerOperator) {}

  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ReferenceOperatorAST final : PtrOperatorAST {
  ReferenceOperatorAST() : PtrOperatorAST(ASTKind::ReferenceOperator) {}

  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PtrToMemberOperatorAST final : PtrOperatorAST {
  PtrToMemberOperatorAST() : PtrOperatorAST(ASTKind::PtrToMemberOperator) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FunctionDeclaratorAST final : DeclaratorModifierAST {
  FunctionDeclaratorAST()
      : DeclaratorModifierAST(ASTKind::FunctionDeclarator) {}

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ArrayDeclaratorAST final : DeclaratorModifierAST {
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
