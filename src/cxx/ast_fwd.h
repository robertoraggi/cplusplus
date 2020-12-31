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

// statements
struct LabeledStatementAST;
struct CaseStatementAST;
struct DefaultStatementAST;
struct ExpressionStatementAST;
struct CompoundStatementAST;
struct IfStatementAST;
struct SwitchStatementAST;
struct WhileStatementAST;
struct DoStatementAST;
struct ForRangeStatementAST;
struct ForStatementAST;
struct BreakStatementAST;
struct ContinueStatementAST;
struct ReturnStatementAST;
struct GotoStatementAST;
struct CoroutineReturnStatementAST;
struct DeclarationStatementAST;

// visitors
struct StatementASTVisitor;

}  // namespace cxx