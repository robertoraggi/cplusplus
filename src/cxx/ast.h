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
#include <cxx/source_location.h>

namespace cxx {

template <typename T>
struct List;

struct AST;
struct UnitAST;
struct DeclarationAST;
struct StatementAST;
struct ExpressionAST;
struct SpecifierAST;
struct DeclaratorAST;
struct NameAST;

template <typename T>
struct List final : Managed {
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

struct AST : Managed {
  virtual ~AST() = default;
};

struct UnitAST : AST {};

struct DeclarationAST : AST {};

struct StatementAST : AST {};

struct ExpressionAST : AST {};

struct SpecifierAST : AST {};

struct DeclaratorAST : AST {};

struct NameAST : AST {};

// statements

struct LabeledStatementAST final : StatementAST {
  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;
};

struct CaseStatementAST final : StatementAST {
  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;
};

struct DefaultStatementAST final : StatementAST {
  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;
};

struct CompoundStatementAST final : StatementAST {
  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;
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
};

struct SwitchStatementAST final : StatementAST {
  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;
};

struct WhileStatementAST final : StatementAST {
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;
};

struct DoStatementAST final : StatementAST {
  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;
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
};

struct BreakStatementAST final : StatementAST {
  SourceLocation breakLoc;
  SourceLocation semicolonLoc;
};

struct ContinueStatementAST final : StatementAST {
  SourceLocation continueLoc;
  SourceLocation semicolonLoc;
};

struct ReturnStatementAST final : StatementAST {
  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;
};

struct GotoStatementAST final : StatementAST {
  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;
};

struct CoroutineReturnStatementAST final : StatementAST {
  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationAST* declaration = nullptr;
};

}  // namespace cxx