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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/type_checker.h>

// cxx
#include <cxx/ast.h>

namespace cxx {

struct TypeChecker::Visitor {
  TypeChecker& check;

  void operator()(GeneratedLiteralExpressionAST* ast);
  void operator()(CharLiteralExpressionAST* ast);
  void operator()(BoolLiteralExpressionAST* ast);
  void operator()(IntLiteralExpressionAST* ast);
  void operator()(FloatLiteralExpressionAST* ast);
  void operator()(NullptrLiteralExpressionAST* ast);
  void operator()(StringLiteralExpressionAST* ast);
  void operator()(UserDefinedStringLiteralExpressionAST* ast);
  void operator()(ThisExpressionAST* ast);
  void operator()(NestedStatementExpressionAST* ast);
  void operator()(NestedExpressionAST* ast);
  void operator()(IdExpressionAST* ast);
  void operator()(LambdaExpressionAST* ast);
  void operator()(FoldExpressionAST* ast);
  void operator()(RightFoldExpressionAST* ast);
  void operator()(LeftFoldExpressionAST* ast);
  void operator()(RequiresExpressionAST* ast);
  void operator()(VaArgExpressionAST* ast);
  void operator()(SubscriptExpressionAST* ast);
  void operator()(CallExpressionAST* ast);
  void operator()(TypeConstructionAST* ast);
  void operator()(BracedTypeConstructionAST* ast);
  void operator()(SpliceMemberExpressionAST* ast);
  void operator()(MemberExpressionAST* ast);
  void operator()(PostIncrExpressionAST* ast);
  void operator()(CppCastExpressionAST* ast);
  void operator()(BuiltinBitCastExpressionAST* ast);
  void operator()(BuiltinOffsetofExpressionAST* ast);
  void operator()(TypeidExpressionAST* ast);
  void operator()(TypeidOfTypeExpressionAST* ast);
  void operator()(SpliceExpressionAST* ast);
  void operator()(GlobalScopeReflectExpressionAST* ast);
  void operator()(NamespaceReflectExpressionAST* ast);
  void operator()(TypeIdReflectExpressionAST* ast);
  void operator()(ReflectExpressionAST* ast);
  void operator()(UnaryExpressionAST* ast);
  void operator()(AwaitExpressionAST* ast);
  void operator()(SizeofExpressionAST* ast);
  void operator()(SizeofTypeExpressionAST* ast);
  void operator()(SizeofPackExpressionAST* ast);
  void operator()(AlignofTypeExpressionAST* ast);
  void operator()(AlignofExpressionAST* ast);
  void operator()(NoexceptExpressionAST* ast);
  void operator()(NewExpressionAST* ast);
  void operator()(DeleteExpressionAST* ast);
  void operator()(CastExpressionAST* ast);
  void operator()(ImplicitCastExpressionAST* ast);
  void operator()(BinaryExpressionAST* ast);
  void operator()(ConditionalExpressionAST* ast);
  void operator()(YieldExpressionAST* ast);
  void operator()(ThrowExpressionAST* ast);
  void operator()(AssignmentExpressionAST* ast);
  void operator()(PackExpansionExpressionAST* ast);
  void operator()(DesignatedInitializerClauseAST* ast);
  void operator()(TypeTraitExpressionAST* ast);
  void operator()(ConditionExpressionAST* ast);
  void operator()(EqualInitializerAST* ast);
  void operator()(BracedInitListAST* ast);
  void operator()(ParenInitializerAST* ast);
};

void TypeChecker::Visitor::operator()(GeneratedLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CharLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BoolLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(IntLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(FloatLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NullptrLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(StringLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ThisExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NestedStatementExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NestedExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(IdExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(LambdaExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(FoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(RightFoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(LeftFoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(RequiresExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(VaArgExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SubscriptExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CallExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeConstructionAST* ast) {}

void TypeChecker::Visitor::operator()(BracedTypeConstructionAST* ast) {}

void TypeChecker::Visitor::operator()(SpliceMemberExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(MemberExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(PostIncrExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CppCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BuiltinBitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BuiltinOffsetofExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeidExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeidOfTypeExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SpliceExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(GlobalScopeReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NamespaceReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeIdReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(UnaryExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(AwaitExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SizeofExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SizeofTypeExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SizeofPackExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(AlignofTypeExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(AlignofExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NoexceptExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NewExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(DeleteExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ImplicitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BinaryExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ConditionalExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(YieldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ThrowExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(AssignmentExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(PackExpansionExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(DesignatedInitializerClauseAST* ast) {}

void TypeChecker::Visitor::operator()(TypeTraitExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ConditionExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(EqualInitializerAST* ast) {}

void TypeChecker::Visitor::operator()(BracedInitListAST* ast) {}

void TypeChecker::Visitor::operator()(ParenInitializerAST* ast) {}

TypeChecker::TypeChecker(TranslationUnit* unit) : unit_(unit) {}

void TypeChecker::operator()(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

void TypeChecker::check(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

}  // namespace cxx
