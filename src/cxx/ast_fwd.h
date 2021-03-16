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
class List;

class AST;

class AttributeAST;
class CoreDeclaratorAST;
class DeclarationAST;
class DeclaratorModifierAST;
class ExceptionDeclarationAST;
class ExpressionAST;
class FunctionBodyAST;
class InitializerAST;
class LambdaCaptureAST;
class MemInitializerAST;
class NameAST;
class NewInitializerAST;
class PtrOperatorAST;
class SpecifierAST;
class StatementAST;
class UnitAST;

// AST
class TypeIdAST;
class NestedNameSpecifierAST;
class UsingDeclaratorAST;
class HandlerAST;
class TemplateArgumentAST;
class EnumBaseAST;
class EnumeratorAST;
class DeclaratorAST;
class InitDeclaratorAST;
class BaseSpecifierAST;
class BaseClauseAST;
class NewTypeIdAST;
class ParameterDeclarationClauseAST;
class ParametersAndQualifiersAST;
class LambdaIntroducerAST;
class LambdaDeclaratorAST;
class TrailingReturnTypeAST;
class CtorInitializerAST;

// MemInitializerAST
class ParenMemInitializerAST;
class BracedMemInitializerAST;

// LambdaCaptureAST
class ThisLambdaCaptureAST;
class DerefThisLambdaCaptureAST;
class SimpleLambdaCaptureAST;
class RefLambdaCaptureAST;
class RefInitLambdaCaptureAST;
class InitLambdaCaptureAST;

// InitializerAST
class EqualInitializerAST;
class BracedInitListAST;
class ParenInitializerAST;

// NewInitializerAST
class NewParenInitializerAST;
class NewBracedInitializerAST;

// ExceptionDeclarationAST
class EllipsisExceptionDeclarationAST;
class TypeExceptionDeclarationAST;

// FunctionBodyAST
class DefaultFunctionBodyAST;
class CompoundStatementFunctionBodyAST;
class TryStatementFunctionBodyAST;
class DeleteFunctionBodyAST;

// UnitAST
class TranslationUnitAST;
class ModuleUnitAST;

// ExpressionAST
class ThisExpressionAST;
class CharLiteralExpressionAST;
class BoolLiteralExpressionAST;
class IntLiteralExpressionAST;
class FloatLiteralExpressionAST;
class NullptrLiteralExpressionAST;
class StringLiteralExpressionAST;
class UserDefinedStringLiteralExpressionAST;
class IdExpressionAST;
class NestedExpressionAST;
class RightFoldExpressionAST;
class LeftFoldExpressionAST;
class FoldExpressionAST;
class LambdaExpressionAST;
class SizeofExpressionAST;
class SizeofTypeExpressionAST;
class SizeofPackExpressionAST;
class TypeidExpressionAST;
class TypeidOfTypeExpressionAST;
class AlignofExpressionAST;
class UnaryExpressionAST;
class BinaryExpressionAST;
class AssignmentExpressionAST;
class BracedTypeConstructionAST;
class TypeConstructionAST;
class CallExpressionAST;
class SubscriptExpressionAST;
class MemberExpressionAST;
class ConditionalExpressionAST;
class CastExpressionAST;
class CppCastExpressionAST;
class NewExpressionAST;
class DeleteExpressionAST;
class ThrowExpressionAST;
class NoexceptExpressionAST;

// StatementAST
class LabeledStatementAST;
class CaseStatementAST;
class DefaultStatementAST;
class ExpressionStatementAST;
class CompoundStatementAST;
class IfStatementAST;
class SwitchStatementAST;
class WhileStatementAST;
class DoStatementAST;
class ForRangeStatementAST;
class ForStatementAST;
class BreakStatementAST;
class ContinueStatementAST;
class ReturnStatementAST;
class GotoStatementAST;
class CoroutineReturnStatementAST;
class DeclarationStatementAST;
class TryBlockStatementAST;

// DeclarationAST
class AccessDeclarationAST;
class FunctionDefinitionAST;
class ConceptDefinitionAST;
class ForRangeDeclarationAST;
class AliasDeclarationAST;
class SimpleDeclarationAST;
class StaticAssertDeclarationAST;
class EmptyDeclarationAST;
class AttributeDeclarationAST;
class OpaqueEnumDeclarationAST;
class UsingEnumDeclarationAST;
class NamespaceDefinitionAST;
class NamespaceAliasDefinitionAST;
class UsingDirectiveAST;
class UsingDeclarationAST;
class AsmDeclarationAST;
class ExportDeclarationAST;
class ModuleImportDeclarationAST;
class TemplateDeclarationAST;
class TypenameTypeParameterAST;
class TypenamePackTypeParameterAST;
class TemplateTypeParameterAST;
class TemplatePackTypeParameterAST;
class DeductionGuideAST;
class ExplicitInstantiationAST;
class ParameterDeclarationAST;
class LinkageSpecificationAST;

// NameAST
class SimpleNameAST;
class DestructorNameAST;
class DecltypeNameAST;
class OperatorNameAST;
class TemplateNameAST;
class QualifiedNameAST;

// SpecifierAST
class TypedefSpecifierAST;
class FriendSpecifierAST;
class ConstevalSpecifierAST;
class ConstinitSpecifierAST;
class ConstexprSpecifierAST;
class InlineSpecifierAST;
class StaticSpecifierAST;
class ExternSpecifierAST;
class ThreadLocalSpecifierAST;
class ThreadSpecifierAST;
class MutableSpecifierAST;
class VirtualSpecifierAST;
class ExplicitSpecifierAST;
class AutoTypeSpecifierAST;
class VoidTypeSpecifierAST;
class VaListTypeSpecifierAST;
class IntegralTypeSpecifierAST;
class FloatingPointTypeSpecifierAST;
class ComplexTypeSpecifierAST;
class NamedTypeSpecifierAST;
class AtomicTypeSpecifierAST;
class UnderlyingTypeSpecifierAST;
class ElaboratedTypeSpecifierAST;
class DecltypeAutoSpecifierAST;
class DecltypeSpecifierAST;
class TypeofSpecifierAST;
class PlaceholderTypeSpecifierAST;
class ConstQualifierAST;
class VolatileQualifierAST;
class RestrictQualifierAST;
class EnumSpecifierAST;
class ClassSpecifierAST;
class TypenameSpecifierAST;

// CoreDeclaratorAST
class IdDeclaratorAST;
class NestedDeclaratorAST;

// PtrOperatorAST
class PointerOperatorAST;
class ReferenceOperatorAST;
class PtrToMemberOperatorAST;

// DeclaratorModifierAST
class FunctionDeclaratorAST;
class ArrayDeclaratorAST;

}  // namespace cxx
