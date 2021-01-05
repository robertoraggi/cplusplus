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
struct AttributeAST;
struct TypeIdAST;
struct PtrOperatorAST;
struct CoreDeclaratorAST;
struct DeclaratorModifierAST;
struct NestedNameSpecifierAST;
struct EnumeratorAST;
struct EnumBaseAST;

// statements
struct BreakStatementAST;
struct CaseStatementAST;
struct CompoundStatementAST;
struct ContinueStatementAST;
struct CoroutineReturnStatementAST;
struct DeclarationStatementAST;
struct DefaultStatementAST;
struct DoStatementAST;
struct ExpressionStatementAST;
struct ForRangeStatementAST;
struct ForStatementAST;
struct GotoStatementAST;
struct IfStatementAST;
struct LabeledStatementAST;
struct ReturnStatementAST;
struct SwitchStatementAST;
struct WhileStatementAST;

// declarations
struct AliasDeclarationAST;
struct AsmDeclarationAST;
struct AttributeDeclarationAST;
struct ConceptDefinitionAST;
struct DeductionGuideAST;
struct EmptyDeclarationAST;
struct ExplicitInstantiationAST;
struct ExportDeclarationAST;
struct ForRangeDeclarationAST;
struct LinkageSpecificationAST;
struct MemberDeclarationAST;
struct MemberSpecificationAST;
struct ModuleImportDeclarationAST;
struct NamespaceAliasDefinitionAST;
struct NamespaceDefinitionAST;
struct OpaqueEnumDeclarationAST;
struct SimpleDeclarationAST;
struct StaticAssertDeclarationAST;
struct TemplateDeclarationAST;
struct UsingDeclarationAST;
struct UsingDirectiveAST;
struct UsingEnumDeclarationAST;

// units
struct TranslationUnitAST;
struct ModuleUnitAST;

// names
struct SimpleNameAST;

// specifiers
struct AtomicTypeSpecifierAST;
struct ClassSpecifierAST;
struct CvQualifierAST;
struct DecltypeSpecifierAST;
struct DecltypeSpecifierTypeSpecifierAST;
struct ElaboratedTypeSpecifierAST;
struct EnumSpecifierAST;
struct ExplicitSpecifierAST;
struct FunctionSpecifierAST;
struct NamedTypeSpecifierAST;
struct PlaceholderTypeSpecifierAST;
struct PlaceholderTypeSpecifierHelperAST;
struct PrimitiveTypeSpecifierAST;
struct SimpleTypeSpecifierAST;
struct StorageClassSpecifierAST;
struct TypenameSpecifierAST;
struct UnderlyingTypeSpecifierAST;

// declarators
struct IdDeclaratorAST;
struct NestedDeclaratorAST;

struct PointerOperatorAST;
struct ReferenceOperatorAST;
struct PtrToMemberOperatorAST;
struct FunctionDeclaratorAST;
struct ArrayDeclaratorAST;

// visitors
struct StatementASTVisitor;
struct DeclarationASTVisitor;
struct SpecifierASTVisitor;

}  // namespace cxx