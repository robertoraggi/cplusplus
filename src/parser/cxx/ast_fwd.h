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

#pragma once

#include <string_view>

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
  kTemporaryMaterializationConversion,
  kUserDefinedConversion,
};

class AttributeSpecifierAST;
class AttributeTokenAST;
class CoreDeclaratorAST;
class DeclarationAST;
class DeclaratorChunkAST;
class DesignatorAST;
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

// UnitAST
class TranslationUnitAST;
class ModuleUnitAST;

// DeclarationAST
class SimpleDeclarationAST;
class AsmDeclarationAST;
class NamespaceAliasDefinitionAST;
class UsingDeclarationAST;
class UsingEnumDeclarationAST;
class UsingDirectiveAST;
class StaticAssertDeclarationAST;
class AliasDeclarationAST;
class OpaqueEnumDeclarationAST;
class FunctionDefinitionAST;
class TemplateDeclarationAST;
class ConceptDefinitionAST;
class DeductionGuideAST;
class ExplicitInstantiationAST;
class ExportDeclarationAST;
class ExportCompoundDeclarationAST;
class LinkageSpecificationAST;
class NamespaceDefinitionAST;
class EmptyDeclarationAST;
class AttributeDeclarationAST;
class ModuleImportDeclarationAST;
class ParameterDeclarationAST;
class AccessDeclarationAST;
class ForRangeDeclarationAST;
class StructuredBindingDeclarationAST;
class AsmOperandAST;
class AsmQualifierAST;
class AsmClobberAST;
class AsmGotoLabelAST;

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
class CoroutineReturnStatementAST;
class GotoStatementAST;
class DeclarationStatementAST;
class TryBlockStatementAST;

// ExpressionAST
class GeneratedLiteralExpressionAST;
class CharLiteralExpressionAST;
class BoolLiteralExpressionAST;
class IntLiteralExpressionAST;
class FloatLiteralExpressionAST;
class NullptrLiteralExpressionAST;
class StringLiteralExpressionAST;
class UserDefinedStringLiteralExpressionAST;
class ThisExpressionAST;
class NestedStatementExpressionAST;
class NestedExpressionAST;
class IdExpressionAST;
class LambdaExpressionAST;
class FoldExpressionAST;
class RightFoldExpressionAST;
class LeftFoldExpressionAST;
class RequiresExpressionAST;
class VaArgExpressionAST;
class SubscriptExpressionAST;
class CallExpressionAST;
class TypeConstructionAST;
class BracedTypeConstructionAST;
class SpliceMemberExpressionAST;
class MemberExpressionAST;
class PostIncrExpressionAST;
class CppCastExpressionAST;
class BuiltinBitCastExpressionAST;
class BuiltinOffsetofExpressionAST;
class TypeidExpressionAST;
class TypeidOfTypeExpressionAST;
class SpliceExpressionAST;
class GlobalScopeReflectExpressionAST;
class NamespaceReflectExpressionAST;
class TypeIdReflectExpressionAST;
class ReflectExpressionAST;
class UnaryExpressionAST;
class AwaitExpressionAST;
class SizeofExpressionAST;
class SizeofTypeExpressionAST;
class SizeofPackExpressionAST;
class AlignofTypeExpressionAST;
class AlignofExpressionAST;
class NoexceptExpressionAST;
class NewExpressionAST;
class DeleteExpressionAST;
class CastExpressionAST;
class ImplicitCastExpressionAST;
class BinaryExpressionAST;
class ConditionalExpressionAST;
class YieldExpressionAST;
class ThrowExpressionAST;
class AssignmentExpressionAST;
class PackExpansionExpressionAST;
class DesignatedInitializerClauseAST;
class TypeTraitExpressionAST;
class ConditionExpressionAST;
class EqualInitializerAST;
class BracedInitListAST;
class ParenInitializerAST;

// DesignatorAST
class DotDesignatorAST;
class SubscriptDesignatorAST;

// AST
class SplicerAST;
class GlobalModuleFragmentAST;
class PrivateModuleFragmentAST;
class ModuleDeclarationAST;
class ModuleNameAST;
class ModuleQualifierAST;
class ModulePartitionAST;
class ImportNameAST;
class InitDeclaratorAST;
class DeclaratorAST;
class UsingDeclaratorAST;
class EnumeratorAST;
class TypeIdAST;
class HandlerAST;
class BaseSpecifierAST;
class RequiresClauseAST;
class ParameterDeclarationClauseAST;
class TrailingReturnTypeAST;
class LambdaSpecifierAST;
class TypeConstraintAST;
class AttributeArgumentClauseAST;
class AttributeAST;
class AttributeUsingPrefixAST;
class NewPlacementAST;
class NestedNamespaceSpecifierAST;

// TemplateParameterAST
class TemplateTypeParameterAST;
class NonTypeTemplateParameterAST;
class TypenameTypeParameterAST;
class ConstraintTypeParameterAST;

// SpecifierAST
class GeneratedTypeSpecifierAST;
class TypedefSpecifierAST;
class FriendSpecifierAST;
class ConstevalSpecifierAST;
class ConstinitSpecifierAST;
class ConstexprSpecifierAST;
class InlineSpecifierAST;
class NoreturnSpecifierAST;
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
class SplicerTypeSpecifierAST;

// PtrOperatorAST
class PointerOperatorAST;
class ReferenceOperatorAST;
class PtrToMemberOperatorAST;

// CoreDeclaratorAST
class BitfieldDeclaratorAST;
class ParameterPackAST;
class IdDeclaratorAST;
class NestedDeclaratorAST;

// DeclaratorChunkAST
class FunctionDeclaratorChunkAST;
class ArrayDeclaratorChunkAST;

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

// NestedNameSpecifierAST
class GlobalNestedNameSpecifierAST;
class SimpleNestedNameSpecifierAST;
class DecltypeNestedNameSpecifierAST;
class TemplateNestedNameSpecifierAST;

// FunctionBodyAST
class DefaultFunctionBodyAST;
class CompoundStatementFunctionBodyAST;
class TryStatementFunctionBodyAST;
class DeleteFunctionBodyAST;

// TemplateArgumentAST
class TypeTemplateArgumentAST;
class ExpressionTemplateArgumentAST;

// ExceptionSpecifierAST
class ThrowExceptionSpecifierAST;
class NoexceptSpecifierAST;

// RequirementAST
class SimpleRequirementAST;
class CompoundRequirementAST;
class TypeRequirementAST;
class NestedRequirementAST;

// NewInitializerAST
class NewParenInitializerAST;
class NewBracedInitializerAST;

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

// ExceptionDeclarationAST
class EllipsisExceptionDeclarationAST;
class TypeExceptionDeclarationAST;

// AttributeSpecifierAST
class CxxAttributeAST;
class GccAttributeAST;
class AlignasAttributeAST;
class AlignasTypeAttributeAST;
class AsmAttributeAST;

// AttributeTokenAST
class ScopedAttributeTokenAST;
class SimpleAttributeTokenAST;

enum class ASTKind;
auto to_string(ASTKind kind) -> std::string_view;
auto to_string(ValueCategory valueCategory) -> std::string_view;
auto to_string(ImplicitCastKind implicitCastKind) -> std::string_view;

}  // namespace cxx
