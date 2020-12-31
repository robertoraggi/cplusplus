// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/source_location.h>

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
};

struct UnitAST : AST {};

struct DeclarationAST : AST {
  virtual void visit(DeclarationASTVisitor*) = 0;
};

struct StatementAST : AST {
  virtual void visit(StatementASTVisitor*) = 0;
};

struct ExpressionAST : AST {};

struct SpecifierAST : AST {};

struct DeclaratorAST : AST {};

struct NameAST : AST {};

// statements

struct LabeledStatementAST final : StatementAST {
  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct CaseStatementAST final : StatementAST {
  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct DefaultStatementAST final : StatementAST {
  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct CompoundStatementAST final : StatementAST {
  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void visit(StatementASTVisitor* visitor) override;
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

  void visit(StatementASTVisitor* visitor) override;
};

struct SwitchStatementAST final : StatementAST {
  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct WhileStatementAST final : StatementAST {
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct DoStatementAST final : StatementAST {
  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
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

  void visit(StatementASTVisitor* visitor) override;
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

  void visit(StatementASTVisitor* visitor) override;
};

struct BreakStatementAST final : StatementAST {
  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct ContinueStatementAST final : StatementAST {
  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct ReturnStatementAST final : StatementAST {
  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct GotoStatementAST final : StatementAST {
  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct CoroutineReturnStatementAST final : StatementAST {
  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationAST* declaration = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

// declarations

struct ForRangeDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct AliasDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct SimpleDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct StaticAssertDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct EmptyDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct AttributeDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct OpaqueEnumDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingEnumDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct NamespaceDefinitionAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct NamespaceAliasDefinitionAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingDirectiveAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct AsmDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct LinkageSpecificationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ExportDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ModuleImportDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct MemberSpecificationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct MemberDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct TemplateDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct DeductionGuideAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ExplicitInstantiationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

}  // namespace cxx