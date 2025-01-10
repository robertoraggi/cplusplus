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

#include <cxx/ast_fwd.h>

namespace cxx {

class TranslationUnit;
class Control;

class ASTInterpreter {
 public:
  explicit ASTInterpreter(TranslationUnit* unit);
  ~ASTInterpreter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  struct UnitResult {};
  struct DeclarationResult {};
  struct StatementResult {};
  struct ExpressionResult {};
  struct TemplateParameterResult {};
  struct SpecifierResult {};
  struct PtrOperatorResult {};
  struct CoreDeclaratorResult {};
  struct DeclaratorChunkResult {};
  struct UnqualifiedIdResult {};
  struct NestedNameSpecifierResult {};
  struct FunctionBodyResult {};
  struct TemplateArgumentResult {};
  struct ExceptionSpecifierResult {};
  struct RequirementResult {};
  struct NewInitializerResult {};
  struct MemInitializerResult {};
  struct LambdaCaptureResult {};
  struct ExceptionDeclarationResult {};
  struct AttributeSpecifierResult {};
  struct AttributeTokenResult {};

  struct SplicerResult {};
  struct GlobalModuleFragmentResult {};
  struct PrivateModuleFragmentResult {};
  struct ModuleDeclarationResult {};
  struct ModuleNameResult {};
  struct ModuleQualifierResult {};
  struct ModulePartitionResult {};
  struct ImportNameResult {};
  struct InitDeclaratorResult {};
  struct DeclaratorResult {};
  struct UsingDeclaratorResult {};
  struct EnumeratorResult {};
  struct TypeIdResult {};
  struct HandlerResult {};
  struct BaseSpecifierResult {};
  struct RequiresClauseResult {};
  struct ParameterDeclarationClauseResult {};
  struct TrailingReturnTypeResult {};
  struct LambdaSpecifierResult {};
  struct TypeConstraintResult {};
  struct AttributeArgumentClauseResult {};
  struct AttributeResult {};
  struct AttributeUsingPrefixResult {};
  struct NewPlacementResult {};
  struct NestedNamespaceSpecifierResult {};

  // run on the base nodes
  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitResult;
  [[nodiscard]] auto operator()(DeclarationAST* ast) -> DeclarationResult;
  [[nodiscard]] auto operator()(StatementAST* ast) -> StatementResult;
  [[nodiscard]] auto operator()(ExpressionAST* ast) -> ExpressionResult;
  [[nodiscard]] auto operator()(TemplateParameterAST* ast)
      -> TemplateParameterResult;
  [[nodiscard]] auto operator()(SpecifierAST* ast) -> SpecifierResult;
  [[nodiscard]] auto operator()(PtrOperatorAST* ast) -> PtrOperatorResult;
  [[nodiscard]] auto operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorResult;
  [[nodiscard]] auto operator()(DeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
  [[nodiscard]] auto operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdResult;
  [[nodiscard]] auto operator()(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  [[nodiscard]] auto operator()(FunctionBodyAST* ast) -> FunctionBodyResult;
  [[nodiscard]] auto operator()(TemplateArgumentAST* ast)
      -> TemplateArgumentResult;
  [[nodiscard]] auto operator()(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;
  [[nodiscard]] auto operator()(RequirementAST* ast) -> RequirementResult;
  [[nodiscard]] auto operator()(NewInitializerAST* ast) -> NewInitializerResult;
  [[nodiscard]] auto operator()(MemInitializerAST* ast) -> MemInitializerResult;
  [[nodiscard]] auto operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult;
  [[nodiscard]] auto operator()(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
  [[nodiscard]] auto operator()(AttributeSpecifierAST* ast)
      -> AttributeSpecifierResult;
  [[nodiscard]] auto operator()(AttributeTokenAST* ast) -> AttributeTokenResult;

  // run on the misc nodes
  auto operator()(SplicerAST* ast) -> SplicerResult;
  auto operator()(GlobalModuleFragmentAST* ast) -> GlobalModuleFragmentResult;
  auto operator()(PrivateModuleFragmentAST* ast) -> PrivateModuleFragmentResult;
  auto operator()(ModuleDeclarationAST* ast) -> ModuleDeclarationResult;
  auto operator()(ModuleNameAST* ast) -> ModuleNameResult;
  auto operator()(ModuleQualifierAST* ast) -> ModuleQualifierResult;
  auto operator()(ModulePartitionAST* ast) -> ModulePartitionResult;
  auto operator()(ImportNameAST* ast) -> ImportNameResult;
  auto operator()(InitDeclaratorAST* ast) -> InitDeclaratorResult;
  auto operator()(DeclaratorAST* ast) -> DeclaratorResult;
  auto operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorResult;
  auto operator()(EnumeratorAST* ast) -> EnumeratorResult;
  auto operator()(TypeIdAST* ast) -> TypeIdResult;
  auto operator()(HandlerAST* ast) -> HandlerResult;
  auto operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult;
  auto operator()(RequiresClauseAST* ast) -> RequiresClauseResult;
  auto operator()(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseResult;
  auto operator()(TrailingReturnTypeAST* ast) -> TrailingReturnTypeResult;
  auto operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierResult;
  auto operator()(TypeConstraintAST* ast) -> TypeConstraintResult;
  auto operator()(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseResult;
  auto operator()(AttributeAST* ast) -> AttributeResult;
  auto operator()(AttributeUsingPrefixAST* ast) -> AttributeUsingPrefixResult;
  auto operator()(NewPlacementAST* ast) -> NewPlacementResult;
  auto operator()(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierResult;

 private:
  struct UnitVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitResult;

    [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitResult;
  };

  struct DeclarationVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(SimpleDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(AsmDeclarationAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(NamespaceAliasDefinitionAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(UsingDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(UsingEnumDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(UsingDirectiveAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(StaticAssertDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(AliasDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(OpaqueEnumDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(FunctionDefinitionAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(TemplateDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ConceptDefinitionAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ExportDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(EmptyDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(AccessDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
        -> DeclarationResult;

    [[nodiscard]] auto operator()(AsmOperandAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(AsmQualifierAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(AsmClobberAST* ast) -> DeclarationResult;

    [[nodiscard]] auto operator()(AsmGotoLabelAST* ast) -> DeclarationResult;
  };

  struct StatementVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ExpressionStatementAST* ast)
        -> StatementResult;

    [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast)
        -> StatementResult;

    [[nodiscard]] auto operator()(SwitchStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(WhileStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(DoStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ForRangeStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ForStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(BreakStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ContinueStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(ReturnStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(CoroutineReturnStatementAST* ast)
        -> StatementResult;

    [[nodiscard]] auto operator()(GotoStatementAST* ast) -> StatementResult;

    [[nodiscard]] auto operator()(DeclarationStatementAST* ast)
        -> StatementResult;

    [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementResult;
  };

  struct ExpressionVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(GeneratedLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(UserDefinedStringLiteralExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(ThisExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(RightFoldExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(RequiresExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(VaArgExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(SubscriptExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(CallExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(TypeConstructionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(MemberExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(PostIncrExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(CppCastExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(TypeidExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(SpliceExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(ReflectExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(UnaryExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(AwaitExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(SizeofExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(AlignofExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(NoexceptExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(NewExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(DeleteExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(CastExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(ConditionalExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(YieldExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(ThrowExpressionAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(AssignmentExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(ConditionExpressionAST* ast)
        -> ExpressionResult;

    [[nodiscard]] auto operator()(EqualInitializerAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(BracedInitListAST* ast) -> ExpressionResult;

    [[nodiscard]] auto operator()(ParenInitializerAST* ast) -> ExpressionResult;
  };

  struct TemplateParameterVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
        -> TemplateParameterResult;

    [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
        -> TemplateParameterResult;

    [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
        -> TemplateParameterResult;

    [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
        -> TemplateParameterResult;
  };

  struct SpecifierVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(GeneratedTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(InlineSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(StaticSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(ExternSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(ThreadLocalSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ThreadSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(MutableSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(VirtualSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(ExplicitSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(AutoTypeSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(VoidTypeSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(SizeTypeSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(SignTypeSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(VaListTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(IntegralTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(FloatingPointTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ComplexTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(NamedTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(AtomicTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(UnderlyingTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ElaboratedTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(DecltypeAutoSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(DecltypeSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(PlaceholderTypeSpecifierAST* ast)
        -> SpecifierResult;

    [[nodiscard]] auto operator()(ConstQualifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(VolatileQualifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(RestrictQualifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierResult;

    [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast)
        -> SpecifierResult;
  };

  struct PtrOperatorVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorResult;

    [[nodiscard]] auto operator()(ReferenceOperatorAST* ast)
        -> PtrOperatorResult;

    [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
        -> PtrOperatorResult;
  };

  struct CoreDeclaratorVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
        -> CoreDeclaratorResult;

    [[nodiscard]] auto operator()(ParameterPackAST* ast)
        -> CoreDeclaratorResult;

    [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorResult;

    [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
        -> CoreDeclaratorResult;
  };

  struct DeclaratorChunkVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
        -> DeclaratorChunkResult;

    [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
        -> DeclaratorChunkResult;
  };

  struct UnqualifiedIdVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(NameIdAST* ast) -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(DestructorIdAST* ast) -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast)
        -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast)
        -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast)
        -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast)
        -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast)
        -> UnqualifiedIdResult;

    [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast)
        -> UnqualifiedIdResult;
  };

  struct NestedNameSpecifierVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierResult;

    [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierResult;

    [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierResult;

    [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierResult;
  };

  struct FunctionBodyVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
        -> FunctionBodyResult;

    [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
        -> FunctionBodyResult;

    [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
        -> FunctionBodyResult;

    [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast)
        -> FunctionBodyResult;
  };

  struct TemplateArgumentVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
        -> TemplateArgumentResult;

    [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
        -> TemplateArgumentResult;
  };

  struct ExceptionSpecifierVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
        -> ExceptionSpecifierResult;

    [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
        -> ExceptionSpecifierResult;
  };

  struct RequirementVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(SimpleRequirementAST* ast)
        -> RequirementResult;

    [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
        -> RequirementResult;

    [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementResult;

    [[nodiscard]] auto operator()(NestedRequirementAST* ast)
        -> RequirementResult;
  };

  struct NewInitializerVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
        -> NewInitializerResult;

    [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
        -> NewInitializerResult;
  };

  struct MemInitializerVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
        -> MemInitializerResult;

    [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
        -> MemInitializerResult;
  };

  struct LambdaCaptureVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(ThisLambdaCaptureAST* ast)
        -> LambdaCaptureResult;

    [[nodiscard]] auto operator()(DerefThisLambdaCaptureAST* ast)
        -> LambdaCaptureResult;

    [[nodiscard]] auto operator()(SimpleLambdaCaptureAST* ast)
        -> LambdaCaptureResult;

    [[nodiscard]] auto operator()(RefLambdaCaptureAST* ast)
        -> LambdaCaptureResult;

    [[nodiscard]] auto operator()(RefInitLambdaCaptureAST* ast)
        -> LambdaCaptureResult;

    [[nodiscard]] auto operator()(InitLambdaCaptureAST* ast)
        -> LambdaCaptureResult;
  };

  struct ExceptionDeclarationVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
        -> ExceptionDeclarationResult;

    [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
        -> ExceptionDeclarationResult;
  };

  struct AttributeSpecifierVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(CxxAttributeAST* ast)
        -> AttributeSpecifierResult;

    [[nodiscard]] auto operator()(GccAttributeAST* ast)
        -> AttributeSpecifierResult;

    [[nodiscard]] auto operator()(AlignasAttributeAST* ast)
        -> AttributeSpecifierResult;

    [[nodiscard]] auto operator()(AlignasTypeAttributeAST* ast)
        -> AttributeSpecifierResult;

    [[nodiscard]] auto operator()(AsmAttributeAST* ast)
        -> AttributeSpecifierResult;
  };

  struct AttributeTokenVisitor {
    ASTInterpreter& accept;

    [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
        -> AttributeTokenResult;

    [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
        -> AttributeTokenResult;
  };

  TranslationUnit* unit_ = nullptr;
};

}  // namespace cxx
