
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
  virtual ~AST();

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

struct AttributeAST : AST {};
struct CoreDeclaratorAST : AST {};
struct DeclarationAST : AST {};
struct DeclaratorModifierAST : AST {};
struct ExceptionDeclarationAST : AST {};
struct ExpressionAST : AST {};
struct NameAST : AST {};
struct NewInitializerAST : AST {};
struct PtrOperatorAST : AST {};
struct SpecifierAST : AST {};
struct StatementAST : AST {};
struct UnitAST : AST {};

struct TypeIdAST final : AST {
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedNameSpecifierAST final : AST {
  SourceLocation scopeLoc;
  List<NameAST*>* nameList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UsingDeclaratorAST final : AST {
  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct HandlerAST final : AST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumBaseAST final : AST {
  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumeratorAST final : AST {
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DeclaratorAST final : AST {
  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BaseSpecifierAST final : AST {
  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BaseClauseAST final : AST {
  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NewTypeIdAST final : AST {
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BracedInitListAST final : AST {
  SourceLocation lbraceLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParameterDeclarationClauseAST final : AST {
  List<ParameterDeclarationAST*>* templateParameterList = nullptr;
  SourceLocation commaLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParametersAndQualifiersAST final : AST {
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

struct NewParenInitializerAST final : NewInitializerAST {
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NewBracedInitializerAST final : NewInitializerAST {
  BracedInitListAST* bracedInit = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EllipsisExceptionDeclarationAST final : ExceptionDeclarationAST {
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TypeExceptionDeclarationAST final : ExceptionDeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TranslationUnitAST final : UnitAST {
  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ModuleUnitAST final : UnitAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ThisExpressionAST final : ExpressionAST {
  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CharLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BoolLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IntLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FloatLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NullptrLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct StringLiteralExpressionAST final : ExpressionAST {
  List<SourceLocation>* stringLiteralList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UserDefinedStringLiteralExpressionAST final : ExpressionAST {
  SourceLocation literalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IdExpressionAST final : ExpressionAST {
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedExpressionAST final : ExpressionAST {
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct BinaryExpressionAST final : ExpressionAST {
  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AssignmentExpressionAST final : ExpressionAST {
  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CallExpressionAST final : ExpressionAST {
  ExpressionAST* baseExpression = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct SubscriptExpressionAST final : ExpressionAST {
  ExpressionAST* baseExpression = nullptr;
  SourceLocation lbracketLoc;
  ExpressionAST* indexExpression = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct MemberExpressionAST final : ExpressionAST {
  ExpressionAST* baseExpression = nullptr;
  SourceLocation accessLoc;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ConditionalExpressionAST final : ExpressionAST {
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
  SourceLocation scopeLoc;
  SourceLocation newLoc;
  NewTypeIdAST* typeId = nullptr;
  NewInitializerAST* newInitalizer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct LabeledStatementAST final : StatementAST {
  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CaseStatementAST final : StatementAST {
  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DefaultStatementAST final : StatementAST {
  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CompoundStatementAST final : StatementAST {
  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IfStatementAST final : StatementAST {
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
  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ContinueStatementAST final : StatementAST {
  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ReturnStatementAST final : StatementAST {
  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct GotoStatementAST final : StatementAST {
  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CoroutineReturnStatementAST final : StatementAST {
  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TryBlockStatementAST final : StatementAST {
  SourceLocation tryLoc;
  StatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FunctionDefinitionAST final : DeclarationAST {
  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  StatementAST* functionBody = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ConceptDefinitionAST final : DeclarationAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AliasDeclarationAST final : DeclarationAST {
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
  List<AttributeAST*>* attributes = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<DeclaratorAST*>* declaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct StaticAssertDeclarationAST final : DeclarationAST {
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
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AttributeDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct OpaqueEnumDeclarationAST final : DeclarationAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NamespaceDefinitionAST final : DeclarationAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UsingDeclarationAST final : DeclarationAST {
  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AsmDeclarationAST final : DeclarationAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ModuleImportDeclarationAST final : DeclarationAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TemplateDeclarationAST final : DeclarationAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExplicitInstantiationAST final : DeclarationAST {
  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ParameterDeclarationAST final : DeclarationAST {
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
  SourceLocation identifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DestructorNameAST final : NameAST {
  SourceLocation tildeLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeNameAST final : NameAST {
  SpecifierAST* decltypeSpecifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct OperatorNameAST final : NameAST {
  SourceLocation opLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct TemplateNameAST final : NameAST {
  NameAST* name = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct QualifiedNameAST final : NameAST {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct SimpleSpecifierAST final : SpecifierAST {
  SourceLocation specifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ExplicitSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NamedTypeSpecifierAST final : SpecifierAST {
  NameAST* name = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PlaceholderTypeSpecifierHelperAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeSpecifierTypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct UnderlyingTypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct AtomicTypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ElaboratedTypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct DecltypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PlaceholderTypeSpecifierAST final : SpecifierAST {
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct CvQualifierAST final : SpecifierAST {
  SourceLocation qualifierLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct EnumSpecifierAST final : SpecifierAST {
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
  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct IdDeclaratorAST final : CoreDeclaratorAST {
  SourceLocation ellipsisLoc;
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct NestedDeclaratorAST final : CoreDeclaratorAST {
  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PointerOperatorAST final : PtrOperatorAST {
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ReferenceOperatorAST final : PtrOperatorAST {
  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct PtrToMemberOperatorAST final : PtrOperatorAST {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct FunctionDeclaratorAST final : DeclaratorModifierAST {
  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

struct ArrayDeclaratorAST final : DeclaratorModifierAST {
  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  SourceLocation firstSourceLocation() override;
  SourceLocation lastSourceLocation() override;
};

}  // namespace cxx
