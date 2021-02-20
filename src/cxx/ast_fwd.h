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

namespace cxx {

template <typename T>
struct List;

struct AST;

struct AttributeAST;
struct CoreDeclaratorAST;
struct DeclarationAST;
struct DeclaratorModifierAST;
struct ExceptionDeclarationAST;
struct ExpressionAST;
struct InitializerAST;
struct NameAST;
struct NewInitializerAST;
struct PtrOperatorAST;
struct SpecifierAST;
struct StatementAST;
struct UnitAST;

// AST
struct TypeIdAST;
struct NestedNameSpecifierAST;
struct UsingDeclaratorAST;
struct HandlerAST;
struct TemplateArgumentAST;
struct EnumBaseAST;
struct EnumeratorAST;
struct DeclaratorAST;
struct BaseSpecifierAST;
struct BaseClauseAST;
struct NewTypeIdAST;
struct ParameterDeclarationClauseAST;
struct ParametersAndQualifiersAST;
struct LambdaIntroducerAST;
struct LambdaDeclaratorAST;
struct TrailingReturnTypeAST;

// InitializerAST
struct EqualInitializerAST;
struct BracedInitListAST;
struct ParenInitializerAST;

// NewInitializerAST
struct NewParenInitializerAST;
struct NewBracedInitializerAST;

// ExceptionDeclarationAST
struct EllipsisExceptionDeclarationAST;
struct TypeExceptionDeclarationAST;

// UnitAST
struct TranslationUnitAST;
struct ModuleUnitAST;

// ExpressionAST
struct ThisExpressionAST;
struct CharLiteralExpressionAST;
struct BoolLiteralExpressionAST;
struct IntLiteralExpressionAST;
struct FloatLiteralExpressionAST;
struct NullptrLiteralExpressionAST;
struct StringLiteralExpressionAST;
struct UserDefinedStringLiteralExpressionAST;
struct IdExpressionAST;
struct NestedExpressionAST;
struct LambdaExpressionAST;
struct UnaryExpressionAST;
struct BinaryExpressionAST;
struct AssignmentExpressionAST;
struct CallExpressionAST;
struct SubscriptExpressionAST;
struct MemberExpressionAST;
struct ConditionalExpressionAST;
struct CppCastExpressionAST;
struct NewExpressionAST;

// StatementAST
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
struct TryBlockStatementAST;

// DeclarationAST
struct FunctionDefinitionAST;
struct ConceptDefinitionAST;
struct ForRangeDeclarationAST;
struct AliasDeclarationAST;
struct SimpleDeclarationAST;
struct StaticAssertDeclarationAST;
struct EmptyDeclarationAST;
struct AttributeDeclarationAST;
struct OpaqueEnumDeclarationAST;
struct UsingEnumDeclarationAST;
struct NamespaceDefinitionAST;
struct NamespaceAliasDefinitionAST;
struct UsingDirectiveAST;
struct UsingDeclarationAST;
struct AsmDeclarationAST;
struct ExportDeclarationAST;
struct ModuleImportDeclarationAST;
struct TemplateDeclarationAST;
struct DeductionGuideAST;
struct ExplicitInstantiationAST;
struct ParameterDeclarationAST;
struct LinkageSpecificationAST;

// NameAST
struct SimpleNameAST;
struct DestructorNameAST;
struct DecltypeNameAST;
struct OperatorNameAST;
struct TemplateNameAST;
struct QualifiedNameAST;

// SpecifierAST
struct TypedefSpecifierAST;
struct FriendSpecifierAST;
struct ConstevalSpecifierAST;
struct ConstinitSpecifierAST;
struct ConstexprSpecifierAST;
struct InlineSpecifierAST;
struct StaticSpecifierAST;
struct ExternSpecifierAST;
struct ThreadLocalSpecifierAST;
struct ThreadSpecifierAST;
struct MutableSpecifierAST;
struct SimpleSpecifierAST;
struct ExplicitSpecifierAST;
struct AutoTypeSpecifierAST;
struct VoidTypeSpecifierAST;
struct IntegralTypeSpecifierAST;
struct FloatingPointTypeSpecifierAST;
struct ComplexTypeSpecifierAST;
struct NamedTypeSpecifierAST;
struct AtomicTypeSpecifierAST;
struct UnderlyingTypeSpecifierAST;
struct ElaboratedTypeSpecifierAST;
struct DecltypeAutoSpecifierAST;
struct DecltypeSpecifierAST;
struct TypeofSpecifierAST;
struct PlaceholderTypeSpecifierAST;
struct CvQualifierAST;
struct EnumSpecifierAST;
struct ClassSpecifierAST;
struct TypenameSpecifierAST;

// CoreDeclaratorAST
struct IdDeclaratorAST;
struct NestedDeclaratorAST;

// PtrOperatorAST
struct PointerOperatorAST;
struct ReferenceOperatorAST;
struct PtrToMemberOperatorAST;

// DeclaratorModifierAST
struct FunctionDeclaratorAST;
struct ArrayDeclaratorAST;

}  // namespace cxx
