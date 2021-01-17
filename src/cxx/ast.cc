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

SourceLocationRange TypeIdAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NestedNameSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange UsingDeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange HandlerAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange EllipsisExceptionDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TypeExceptionDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TranslationUnitAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ModuleUnitAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ThisExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange CharLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange BoolLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange IntLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange FloatLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NullptrLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange StringLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange
UserDefinedStringLiteralExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange IdExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NestedExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange BinaryExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange AssignmentExpressionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange LabeledStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange CaseStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DefaultStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ExpressionStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange CompoundStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange IfStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange SwitchStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange WhileStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DoStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ForRangeStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ForStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange BreakStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ContinueStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ReturnStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange GotoStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange CoroutineReturnStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DeclarationStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TryBlockStatementAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange FunctionDefinitionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ConceptDefinitionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ForRangeDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange AliasDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange SimpleDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange StaticAssertDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange EmptyDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange AttributeDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange OpaqueEnumDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange UsingEnumDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NamespaceDefinitionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NamespaceAliasDefinitionAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange UsingDirectiveAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange UsingDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange AsmDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange LinkageSpecificationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ExportDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ModuleImportDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TemplateDeclarationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DeductionGuideAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ExplicitInstantiationAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange SimpleNameAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DestructorNameAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DecltypeNameAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange OperatorNameAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TemplateArgumentAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TemplateNameAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange SimpleSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ExplicitSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NamedTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange PlaceholderTypeSpecifierHelperAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DecltypeSpecifierTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange UnderlyingTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange AtomicTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ElaboratedTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DecltypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange PlaceholderTypeSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange CvQualifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange EnumBaseAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange EnumeratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange EnumSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ClassSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange TypenameSpecifierAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange DeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange IdDeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange NestedDeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange PointerOperatorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ReferenceOperatorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange PtrToMemberOperatorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange FunctionDeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

SourceLocationRange ArrayDeclaratorAST::sourceLocationRange() {
  return SourceLocationRange();
}

}  // namespace cxx