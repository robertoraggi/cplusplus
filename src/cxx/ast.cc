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

#include <cxx/ast.h>
#include <cxx/declaration_ast_visitor.h>
#include <cxx/specifier_ast_visitor.h>
#include <cxx/statement_ast_visitor.h>

namespace cxx {

AST::~AST() = default;

// statements

void LabeledStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CaseStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void DefaultStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ExpressionStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CompoundStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void IfStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void SwitchStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void WhileStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void DoStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ForRangeStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ForStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void BreakStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ContinueStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ReturnStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void GotoStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CoroutineReturnStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

// declarations

void DeclarationStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ForRangeDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void AliasDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void SimpleDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void StaticAssertDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void EmptyDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void AttributeDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void OpaqueEnumDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void UsingEnumDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void NamespaceDefinitionAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void NamespaceAliasDefinitionAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void UsingDirectiveAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void UsingDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void AsmDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void LinkageSpecificationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void ExportDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void ModuleImportDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void MemberSpecificationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void MemberDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void TemplateDeclarationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void DeductionGuideAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

void ExplicitInstantiationAST::visit(DeclarationASTVisitor* visitor) {
  visitor->visit(this);
}

// specifiers

void StorageClassSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void FunctionSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void ExplicitSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void SimpleTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void NamedTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void PlaceholderTypeSpecifierHelperAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void DecltypeSpecifierTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void UnderlyingTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void AtomicTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void PrimitiveTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void ElaboratedTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void DecltypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void PlaceholderTypeSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void CvQualifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void EnumSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void ClassSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

void TypenameSpecifierAST::visit(SpecifierASTVisitor* visitor) {
  visitor->visit(this);
}

}  // namespace cxx