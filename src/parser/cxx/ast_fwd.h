// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

enum class ValueCategory {
  kNone,
  kLValue,
  kXValue,
  kPrValue,
};

enum class ImplicitCastKind {
  kIdentity,
  kLValueToRValueConversion,
  kArrayToPointerConversion,
  kFunctionToPointerConversion,
  kIntegralPromotion,
  kFloatingPointPromotion,
  kIntegralConversion,
  kFloatingPointConversion,
  kFloatingIntegralConversion,
  kPointerConversion,
  kPointerToMemberConversion,
  kBooleanConversion,
  kFunctionPointerConversion,
  kQualificationConversion,
  kUserDefinedConversion,
};

class AttributeSpecifierAST;
class AttributeTokenAST;
class CoreDeclaratorAST;
class DeclarationAST;
class DeclaratorChunkAST;
class ExceptionDeclarationAST;
class ExceptionSpecifierAST;
class ExpressionAST;
class FunctionBodyAST;
class LambdaCaptureAST;
class MemInitializerAST;
class NestedNameSpecifierAST;
class NewInitializerAST;
class PtrOperatorAST;
class RequirementAST;
class SpecifierAST;
class StatementAST;
class TemplateArgumentAST;
class TemplateParameterAST;
class UnitAST;
class UnqualifiedIdAST;

// AST
class TypeIdAST;
class UsingDeclaratorAST;
class HandlerAST;
class EnumeratorAST;
class DeclaratorAST;
class InitDeclaratorAST;
class BaseSpecifierAST;
class RequiresClauseAST;
class ParameterDeclarationClauseAST;
class LambdaSpecifierAST;
class TrailingReturnTypeAST;
class TypeConstraintAST;
class GlobalModuleFragmentAST;
class PrivateModuleFragmentAST;
class ModuleQualifierAST;
class ModuleNameAST;
class ModuleDeclarationAST;
class ImportNameAST;
class ModulePartitionAST;
class AttributeArgumentClauseAST;
class AttributeAST;
class AttributeUsingPrefixAST;
class NewPlacementAST;
class NestedNamespaceSpecifierAST;

// NestedNameSpecifierAST
class GlobalNestedNameSpecifierAST;
class SimpleNestedNameSpecifierAST;
class DecltypeNestedNameSpecifierAST;
class TemplateNestedNameSpecifierAST;

// ExceptionSpecifierAST
class ThrowExceptionSpecifierAST;
class NoexceptSpecifierAST;

// ExpressionAST
class PackExpansionExpressionAST;
class DesignatedInitializerClauseAST;
class ThisExpressionAST;
class CharLiteralExpressionAST;
class BoolLiteralExpressionAST;
class IntLiteralExpressionAST;
class FloatLiteralExpressionAST;
class NullptrLiteralExpressionAST;
class StringLiteralExpressionAST;
class UserDefinedStringLiteralExpressionAST;
class IdExpressionAST;
class RequiresExpressionAST;
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
class AlignofTypeExpressionAST;
class AlignofExpressionAST;
class TypeTraitsExpressionAST;
class YieldExpressionAST;
class AwaitExpressionAST;
class UnaryExpressionAST;
class BinaryExpressionAST;
class AssignmentExpressionAST;
class ConditionExpressionAST;
class BracedTypeConstructionAST;
class TypeConstructionAST;
class CallExpressionAST;
class SubscriptExpressionAST;
class MemberExpressionAST;
class PostIncrExpressionAST;
class ConditionalExpressionAST;
class ImplicitCastExpressionAST;
class CastExpressionAST;
class CppCastExpressionAST;
class NewExpressionAST;
class DeleteExpressionAST;
class ThrowExpressionAST;
class NoexceptExpressionAST;
class EqualInitializerAST;
class BracedInitListAST;
class ParenInitializerAST;

// RequirementAST
class SimpleRequirementAST;
class CompoundRequirementAST;
class TypeRequirementAST;
class NestedRequirementAST;

// TemplateArgumentAST
class TypeTemplateArgumentAST;
class ExpressionTemplateArgumentAST;

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

// StatementAST
class LabeledStatementAST;
class CaseStatementAST;
class DefaultStatementAST;
class ExpressionStatementAST;
class CompoundStatementAST;
class IfStatementAST;
class ConstevalIfStatementAST;
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
class StructuredBindingDeclarationAST;
class StaticAssertDeclarationAST;
class EmptyDeclarationAST;
class AttributeDeclarationAST;
class OpaqueEnumDeclarationAST;
class NamespaceDefinitionAST;
class NamespaceAliasDefinitionAST;
class UsingDirectiveAST;
class UsingDeclarationAST;
class UsingEnumDeclarationAST;
class AsmOperandAST;
class AsmQualifierAST;
class AsmClobberAST;
class AsmGotoLabelAST;
class AsmDeclarationAST;
class ExportDeclarationAST;
class ExportCompoundDeclarationAST;
class ModuleImportDeclarationAST;
class TemplateDeclarationAST;
class DeductionGuideAST;
class ExplicitInstantiationAST;
class ParameterDeclarationAST;
class LinkageSpecificationAST;

// TemplateParameterAST
class TemplateTypeParameterAST;
class TemplatePackTypeParameterAST;
class NonTypeTemplateParameterAST;
class TypenameTypeParameterAST;
class ConstraintTypeParameterAST;

// UnqualifiedIdAST
class NameIdAST;
class DestructorIdAST;
class DecltypeIdAST;
class OperatorFunctionIdAST;
class LiteralOperatorIdAST;
class ConversionFunctionIdAST;
class SimpleTemplateIdAST;
class LiteralOperatorTemplateIdAST;
class OperatorFunctionTemplateIdAST;

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
class SizeTypeSpecifierAST;
class SignTypeSpecifierAST;
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
class PlaceholderTypeSpecifierAST;
class ConstQualifierAST;
class VolatileQualifierAST;
class RestrictQualifierAST;
class EnumSpecifierAST;
class ClassSpecifierAST;
class TypenameSpecifierAST;

// CoreDeclaratorAST
class BitfieldDeclaratorAST;
class ParameterPackAST;
class IdDeclaratorAST;
class NestedDeclaratorAST;

// PtrOperatorAST
class PointerOperatorAST;
class ReferenceOperatorAST;
class PtrToMemberOperatorAST;

// DeclaratorChunkAST
class FunctionDeclaratorChunkAST;
class ArrayDeclaratorChunkAST;

// AttributeSpecifierAST
class CxxAttributeAST;
class GccAttributeAST;
class AlignasAttributeAST;
class AsmAttributeAST;

// AttributeTokenAST
class ScopedAttributeTokenAST;
class SimpleAttributeTokenAST;

}  // namespace cxx
