// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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
class Arena;

class ASTRewriter {
 public:
  explicit ASTRewriter(TranslationUnit* unit);
  ~ASTRewriter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto arena() const -> Arena*;

  // run on the base nodes
  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitAST*;
  [[nodiscard]] auto operator()(DeclarationAST* ast) -> DeclarationAST*;
  [[nodiscard]] auto operator()(StatementAST* ast) -> StatementAST*;
  [[nodiscard]] auto operator()(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto operator()(TemplateParameterAST* ast)
      -> TemplateParameterAST*;
  [[nodiscard]] auto operator()(SpecifierAST* ast) -> SpecifierAST*;
  [[nodiscard]] auto operator()(PtrOperatorAST* ast) -> PtrOperatorAST*;
  [[nodiscard]] auto operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorAST*;
  [[nodiscard]] auto operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkAST*;
  [[nodiscard]] auto operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdAST*;
  [[nodiscard]] auto operator()(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;
  [[nodiscard]] auto operator()(FunctionBodyAST* ast) -> FunctionBodyAST*;
  [[nodiscard]] auto operator()(TemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
  [[nodiscard]] auto operator()(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
  [[nodiscard]] auto operator()(RequirementAST* ast) -> RequirementAST*;
  [[nodiscard]] auto operator()(NewInitializerAST* ast) -> NewInitializerAST*;
  [[nodiscard]] auto operator()(MemInitializerAST* ast) -> MemInitializerAST*;
  [[nodiscard]] auto operator()(LambdaCaptureAST* ast) -> LambdaCaptureAST*;
  [[nodiscard]] auto operator()(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
  [[nodiscard]] auto operator()(AttributeSpecifierAST* ast)
      -> AttributeSpecifierAST*;
  [[nodiscard]] auto operator()(AttributeTokenAST* ast) -> AttributeTokenAST*;

  // run on the misc nodes
  [[nodiscard]] auto operator()(SplicerAST* ast) -> SplicerAST*;
  [[nodiscard]] auto operator()(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentAST*;
  [[nodiscard]] auto operator()(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentAST*;
  [[nodiscard]] auto operator()(ModuleDeclarationAST* ast)
      -> ModuleDeclarationAST*;
  [[nodiscard]] auto operator()(ModuleNameAST* ast) -> ModuleNameAST*;
  [[nodiscard]] auto operator()(ModuleQualifierAST* ast) -> ModuleQualifierAST*;
  [[nodiscard]] auto operator()(ModulePartitionAST* ast) -> ModulePartitionAST*;
  [[nodiscard]] auto operator()(ImportNameAST* ast) -> ImportNameAST*;
  [[nodiscard]] auto operator()(InitDeclaratorAST* ast) -> InitDeclaratorAST*;
  [[nodiscard]] auto operator()(DeclaratorAST* ast) -> DeclaratorAST*;
  [[nodiscard]] auto operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorAST*;
  [[nodiscard]] auto operator()(EnumeratorAST* ast) -> EnumeratorAST*;
  [[nodiscard]] auto operator()(TypeIdAST* ast) -> TypeIdAST*;
  [[nodiscard]] auto operator()(HandlerAST* ast) -> HandlerAST*;
  [[nodiscard]] auto operator()(BaseSpecifierAST* ast) -> BaseSpecifierAST*;
  [[nodiscard]] auto operator()(RequiresClauseAST* ast) -> RequiresClauseAST*;
  [[nodiscard]] auto operator()(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseAST*;
  [[nodiscard]] auto operator()(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeAST*;
  [[nodiscard]] auto operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierAST*;
  [[nodiscard]] auto operator()(TypeConstraintAST* ast) -> TypeConstraintAST*;
  [[nodiscard]] auto operator()(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseAST*;
  [[nodiscard]] auto operator()(AttributeAST* ast) -> AttributeAST*;
  [[nodiscard]] auto operator()(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixAST*;
  [[nodiscard]] auto operator()(NewPlacementAST* ast) -> NewPlacementAST*;
  [[nodiscard]] auto operator()(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierAST*;

 private:
  struct UnitVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitAST*;

    [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitAST*;
  };

  struct DeclarationVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(SimpleDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(AsmDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(NamespaceAliasDefinitionAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(UsingDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(UsingEnumDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(UsingDirectiveAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(StaticAssertDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(AliasDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(OpaqueEnumDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(FunctionDefinitionAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(TemplateDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(ConceptDefinitionAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(ExportDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(EmptyDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(AccessDeclarationAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
        -> DeclarationAST*;

    [[nodiscard]] auto operator()(AsmOperandAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(AsmQualifierAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(AsmClobberAST* ast) -> DeclarationAST*;

    [[nodiscard]] auto operator()(AsmGotoLabelAST* ast) -> DeclarationAST*;
  };

  struct StatementVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast)
        -> StatementAST*;

    [[nodiscard]] auto operator()(SwitchStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(WhileStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(DoStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ForRangeStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ForStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(BreakStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ContinueStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(ReturnStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(CoroutineReturnStatementAST* ast)
        -> StatementAST*;

    [[nodiscard]] auto operator()(GotoStatementAST* ast) -> StatementAST*;

    [[nodiscard]] auto operator()(DeclarationStatementAST* ast)
        -> StatementAST*;

    [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementAST*;
  };

  struct ExpressionVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(GeneratedLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(UserDefinedStringLiteralExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(ThisExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(RightFoldExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(RequiresExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(VaArgExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(SubscriptExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(CallExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(TypeConstructionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(MemberExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(CppCastExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(TypeidExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(SpliceExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(ReflectExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(UnaryExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(AwaitExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(SizeofExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(AlignofExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(NewExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(DeleteExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(CastExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(ConditionalExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(YieldExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(ThrowExpressionAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(AssignmentExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(TypeTraitsExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(ConditionExpressionAST* ast)
        -> ExpressionAST*;

    [[nodiscard]] auto operator()(EqualInitializerAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(BracedInitListAST* ast) -> ExpressionAST*;

    [[nodiscard]] auto operator()(ParenInitializerAST* ast) -> ExpressionAST*;
  };

  struct TemplateParameterVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
        -> TemplateParameterAST*;

    [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
        -> TemplateParameterAST*;

    [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
        -> TemplateParameterAST*;

    [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
        -> TemplateParameterAST*;
  };

  struct SpecifierVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(GeneratedTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(InlineSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(StaticSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ExternSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ThreadLocalSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(ThreadSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(MutableSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(VirtualSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ExplicitSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(AutoTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(VoidTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(SizeTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(SignTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(VaListTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(IntegralTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(FloatingPointTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(ComplexTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(NamedTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(AtomicTypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(UnderlyingTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(ElaboratedTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(DecltypeAutoSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(DecltypeSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(PlaceholderTypeSpecifierAST* ast)
        -> SpecifierAST*;

    [[nodiscard]] auto operator()(ConstQualifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(VolatileQualifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(RestrictQualifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierAST*;

    [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast)
        -> SpecifierAST*;
  };

  struct PtrOperatorVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorAST*;

    [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorAST*;

    [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
        -> PtrOperatorAST*;
  };

  struct CoreDeclaratorVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
        -> CoreDeclaratorAST*;

    [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorAST*;

    [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorAST*;

    [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
        -> CoreDeclaratorAST*;
  };

  struct DeclaratorChunkVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
        -> DeclaratorChunkAST*;

    [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
        -> DeclaratorChunkAST*;
  };

  struct UnqualifiedIdVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(NameIdAST* ast) -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(DestructorIdAST* ast) -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast)
        -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast)
        -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast)
        -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast)
        -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast)
        -> UnqualifiedIdAST*;

    [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast)
        -> UnqualifiedIdAST*;
  };

  struct NestedNameSpecifierVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierAST*;

    [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierAST*;

    [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierAST*;

    [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
        -> NestedNameSpecifierAST*;
  };

  struct FunctionBodyVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
        -> FunctionBodyAST*;

    [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
        -> FunctionBodyAST*;

    [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
        -> FunctionBodyAST*;

    [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast)
        -> FunctionBodyAST*;
  };

  struct TemplateArgumentVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
        -> TemplateArgumentAST*;

    [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
        -> TemplateArgumentAST*;
  };

  struct ExceptionSpecifierVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
        -> ExceptionSpecifierAST*;

    [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
        -> ExceptionSpecifierAST*;
  };

  struct RequirementVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementAST*;

    [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
        -> RequirementAST*;

    [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementAST*;

    [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementAST*;
  };

  struct NewInitializerVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
        -> NewInitializerAST*;

    [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
        -> NewInitializerAST*;
  };

  struct MemInitializerVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
        -> MemInitializerAST*;

    [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
        -> MemInitializerAST*;
  };

  struct LambdaCaptureVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(ThisLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;

    [[nodiscard]] auto operator()(DerefThisLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;

    [[nodiscard]] auto operator()(SimpleLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;

    [[nodiscard]] auto operator()(RefLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;

    [[nodiscard]] auto operator()(RefInitLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;

    [[nodiscard]] auto operator()(InitLambdaCaptureAST* ast)
        -> LambdaCaptureAST*;
  };

  struct ExceptionDeclarationVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
        -> ExceptionDeclarationAST*;

    [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
        -> ExceptionDeclarationAST*;
  };

  struct AttributeSpecifierVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(CxxAttributeAST* ast)
        -> AttributeSpecifierAST*;

    [[nodiscard]] auto operator()(GccAttributeAST* ast)
        -> AttributeSpecifierAST*;

    [[nodiscard]] auto operator()(AlignasAttributeAST* ast)
        -> AttributeSpecifierAST*;

    [[nodiscard]] auto operator()(AlignasTypeAttributeAST* ast)
        -> AttributeSpecifierAST*;

    [[nodiscard]] auto operator()(AsmAttributeAST* ast)
        -> AttributeSpecifierAST*;
  };

  struct AttributeTokenVisitor {
    ASTRewriter& rewrite;

    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

    [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
        -> AttributeTokenAST*;

    [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
        -> AttributeTokenAST*;
  };

  TranslationUnit* unit_ = nullptr;
};

}  // namespace cxx
