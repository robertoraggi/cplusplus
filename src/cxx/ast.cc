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

#include <cxx/ast.h>

namespace cxx {

AST::~AST() = default;

SourceLocation TypeIdAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation NestedNameSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDeclaratorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation HandlerAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation EllipsisExceptionDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TypeExceptionDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TranslationUnitAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleUnitAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation ThisExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation CharLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation BoolLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation IntLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation FloatLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NullptrLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation StringLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UserDefinedStringLiteralExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation IdExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NestedExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation BinaryExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AssignmentExpressionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation LabeledStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation CaseStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DefaultStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExpressionStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation CompoundStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation IfStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation SwitchStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation WhileStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DoStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ForRangeStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ForStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation BreakStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ContinueStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ReturnStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation GotoStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation CoroutineReturnStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DeclarationStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TryBlockStatementAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation FunctionDefinitionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ConceptDefinitionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ForRangeDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AliasDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation StaticAssertDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation EmptyDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AttributeDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation OpaqueEnumDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingEnumDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NamespaceDefinitionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NamespaceAliasDefinitionAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDirectiveAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AsmDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation LinkageSpecificationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExportDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleImportDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DeductionGuideAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitInstantiationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleNameAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation DestructorNameAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeNameAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation OperatorNameAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateArgumentAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateNameAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NamedTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierHelperAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeSpecifierTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UnderlyingTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AtomicTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ElaboratedTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation CvQualifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation EnumBaseAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation EnumeratorAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation EnumSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ClassSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TypenameSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DeclaratorAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation IdDeclaratorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation NestedDeclaratorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation PointerOperatorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ReferenceOperatorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation PtrToMemberOperatorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation FunctionDeclaratorAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ArrayDeclaratorAST::firstSourceLocation() {
  return SourceLocation();
}

// last source location
SourceLocation TypeIdAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation NestedNameSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDeclaratorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation HandlerAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation EllipsisExceptionDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TypeExceptionDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TranslationUnitAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleUnitAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation ThisExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CharLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation BoolLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation IntLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation FloatLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NullptrLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation StringLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UserDefinedStringLiteralExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation IdExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NestedExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation BinaryExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AssignmentExpressionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation LabeledStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CaseStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DefaultStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ExpressionStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CompoundStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation IfStatementAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation SwitchStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation WhileStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DoStatementAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation ForRangeStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ForStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation BreakStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ContinueStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ReturnStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation GotoStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CoroutineReturnStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DeclarationStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TryBlockStatementAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation FunctionDefinitionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ConceptDefinitionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ForRangeDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AliasDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation StaticAssertDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation EmptyDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AttributeDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation OpaqueEnumDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingEnumDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NamespaceDefinitionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NamespaceAliasDefinitionAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDirectiveAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AsmDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation LinkageSpecificationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ExportDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleImportDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DeductionGuideAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitInstantiationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleNameAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation DestructorNameAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeNameAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation OperatorNameAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateArgumentAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateNameAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation SimpleSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NamedTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierHelperAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeSpecifierTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UnderlyingTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AtomicTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ElaboratedTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CvQualifierAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation EnumBaseAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation EnumeratorAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation EnumSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ClassSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TypenameSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DeclaratorAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation IdDeclaratorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NestedDeclaratorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation PointerOperatorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ReferenceOperatorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation PtrToMemberOperatorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation FunctionDeclaratorAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ArrayDeclaratorAST::lastSourceLocation() {
  return SourceLocation();
}

}  // namespace cxx