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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

namespace cxx {

ASTRewriter::ASTRewriter(TypeChecker* typeChcker,
                         const std::vector<TemplateArgument>& templateArguments)
    : typeChecker_(typeChcker),
      unit_(typeChcker->translationUnit()),
      templateArguments_(templateArguments) {}

ASTRewriter::~ASTRewriter() {}

auto ASTRewriter::control() const -> Control* { return unit_->control(); }

auto ASTRewriter::arena() const -> Arena* { return unit_->arena(); }

struct ASTRewriter::UnitVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitAST*;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitAST*;
};

struct ASTRewriter::DeclarationVisitor {
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

  [[nodiscard]] auto operator()(FunctionDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(TemplateDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ConceptDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExportDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(EmptyDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(AccessDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(AsmOperandAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AsmQualifierAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AsmClobberAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AsmGotoLabelAST* ast) -> DeclarationAST*;
};

struct ASTRewriter::StatementVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast) -> StatementAST*;

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

  [[nodiscard]] auto operator()(DeclarationStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementAST*;
};

struct ASTRewriter::ExpressionVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(GeneratedLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(UserDefinedStringLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(ThisExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(NestedStatementExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(VaArgExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SubscriptExpressionAST* ast) -> ExpressionAST*;

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

  [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST* ast)
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

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast) -> ExpressionAST*;

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

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ConditionExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(EqualInitializerAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(BracedInitListAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ParenInitializerAST* ast) -> ExpressionAST*;
};

struct ASTRewriter::TemplateParameterVisitor {
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

struct ASTRewriter::SpecifierVisitor {
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

  [[nodiscard]] auto operator()(ThreadLocalSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ThreadSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(MutableSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(VirtualSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ExplicitSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(AutoTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(VoidTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(SizeTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(SignTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(VaListTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(IntegralTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(FloatingPointTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(ComplexTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(NamedTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(AtomicTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(UnderlyingTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(ElaboratedTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(DecltypeAutoSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(DecltypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(PlaceholderTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(ConstQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(VolatileQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(RestrictQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast) -> SpecifierAST*;
};

struct ASTRewriter::PtrOperatorVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast) -> PtrOperatorAST*;
};

struct ASTRewriter::CoreDeclaratorVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast) -> CoreDeclaratorAST*;
};

struct ASTRewriter::DeclaratorChunkVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;
};

struct ASTRewriter::UnqualifiedIdVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(NameIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(DestructorIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast)
      -> UnqualifiedIdAST*;
};

struct ASTRewriter::NestedNameSpecifierVisitor {
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

struct ASTRewriter::FunctionBodyVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast) -> FunctionBodyAST*;
};

struct ASTRewriter::TemplateArgumentVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
};

struct ASTRewriter::ExceptionSpecifierVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
};

struct ASTRewriter::RequirementVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementAST*;
};

struct ASTRewriter::NewInitializerVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerAST*;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerAST*;
};

struct ASTRewriter::MemInitializerVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerAST*;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerAST*;
};

struct ASTRewriter::LambdaCaptureVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(ThisLambdaCaptureAST* ast) -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(DerefThisLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(SimpleLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(RefLambdaCaptureAST* ast) -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(RefInitLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(InitLambdaCaptureAST* ast) -> LambdaCaptureAST*;
};

struct ASTRewriter::ExceptionDeclarationVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
};

struct ASTRewriter::AttributeSpecifierVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(CxxAttributeAST* ast) -> AttributeSpecifierAST*;

  [[nodiscard]] auto operator()(GccAttributeAST* ast) -> AttributeSpecifierAST*;

  [[nodiscard]] auto operator()(AlignasAttributeAST* ast)
      -> AttributeSpecifierAST*;

  [[nodiscard]] auto operator()(AlignasTypeAttributeAST* ast)
      -> AttributeSpecifierAST*;

  [[nodiscard]] auto operator()(AsmAttributeAST* ast) -> AttributeSpecifierAST*;
};

struct ASTRewriter::AttributeTokenVisitor {
  ASTRewriter& rewrite;

  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }

  [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
      -> AttributeTokenAST*;

  [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
      -> AttributeTokenAST*;
};

auto ASTRewriter::operator()(UnitAST* ast) -> UnitAST* {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(DeclarationAST* ast) -> DeclarationAST* {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(StatementAST* ast) -> StatementAST* {
  if (ast) return visit(StatementVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(ExpressionAST* ast) -> ExpressionAST* {
  if (!ast) return {};
  auto expr = visit(ExpressionVisitor{*this}, ast);
  if (expr) typeChecker_->check(expr);
  return expr;
}

auto ASTRewriter::operator()(TemplateParameterAST* ast)
    -> TemplateParameterAST* {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(SpecifierAST* ast) -> SpecifierAST* {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(PtrOperatorAST* ast) -> PtrOperatorAST* {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorAST* {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdAST* {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierAST* {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(FunctionBodyAST* ast) -> FunctionBodyAST* {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(TemplateArgumentAST* ast) -> TemplateArgumentAST* {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierAST* {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(RequirementAST* ast) -> RequirementAST* {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(NewInitializerAST* ast) -> NewInitializerAST* {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(MemInitializerAST* ast) -> MemInitializerAST* {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(LambdaCaptureAST* ast) -> LambdaCaptureAST* {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationAST* {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(AttributeSpecifierAST* ast)
    -> AttributeSpecifierAST* {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(AttributeTokenAST* ast) -> AttributeTokenAST* {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto ASTRewriter::operator()(SplicerAST* ast) -> SplicerAST* {
  if (!ast) return {};

  auto copy = new (arena()) SplicerAST{};

  copy->lbracketLoc = ast->lbracketLoc;
  copy->colonLoc = ast->colonLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->expression = operator()(ast->expression);
  copy->secondColonLoc = ast->secondColonLoc;
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::operator()(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentAST* {
  if (!ast) return {};

  auto copy = new (arena()) GlobalModuleFragmentAST{};

  copy->moduleLoc = ast->moduleLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentAST* {
  if (!ast) return {};

  auto copy = new (arena()) PrivateModuleFragmentAST{};

  copy->moduleLoc = ast->moduleLoc;
  copy->colonLoc = ast->colonLoc;
  copy->privateLoc = ast->privateLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::operator()(ModuleDeclarationAST* ast)
    -> ModuleDeclarationAST* {
  if (!ast) return {};

  auto copy = new (arena()) ModuleDeclarationAST{};

  copy->exportLoc = ast->exportLoc;
  copy->moduleLoc = ast->moduleLoc;
  copy->moduleName = operator()(ast->moduleName);
  copy->modulePartition = operator()(ast->modulePartition);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::operator()(ModuleNameAST* ast) -> ModuleNameAST* {
  if (!ast) return {};

  auto copy = new (arena()) ModuleNameAST{};

  copy->moduleQualifier = operator()(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(ModuleQualifierAST* ast) -> ModuleQualifierAST* {
  if (!ast) return {};

  auto copy = new (arena()) ModuleQualifierAST{};

  copy->moduleQualifier = operator()(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->dotLoc = ast->dotLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(ModulePartitionAST* ast) -> ModulePartitionAST* {
  if (!ast) return {};

  auto copy = new (arena()) ModulePartitionAST{};

  copy->colonLoc = ast->colonLoc;
  copy->moduleName = operator()(ast->moduleName);

  return copy;
}

auto ASTRewriter::operator()(ImportNameAST* ast) -> ImportNameAST* {
  if (!ast) return {};

  auto copy = new (arena()) ImportNameAST{};

  copy->headerLoc = ast->headerLoc;
  copy->modulePartition = operator()(ast->modulePartition);
  copy->moduleName = operator()(ast->moduleName);

  return copy;
}

auto ASTRewriter::operator()(InitDeclaratorAST* ast) -> InitDeclaratorAST* {
  if (!ast) return {};

  auto copy = new (arena()) InitDeclaratorAST{};

  copy->declarator = operator()(ast->declarator);
  copy->requiresClause = operator()(ast->requiresClause);
  copy->initializer = operator()(ast->initializer);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::operator()(DeclaratorAST* ast) -> DeclaratorAST* {
  if (!ast) return {};

  auto copy = new (arena()) DeclaratorAST{};

  if (auto it = ast->ptrOpList) {
    auto out = &copy->ptrOpList;
    for (auto node : ListView{ast->ptrOpList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->coreDeclarator = operator()(ast->coreDeclarator);

  if (auto it = ast->declaratorChunkList) {
    auto out = &copy->declaratorChunkList;
    for (auto node : ListView{ast->declaratorChunkList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorAST* {
  if (!ast) return {};

  auto copy = new (arena()) UsingDeclaratorAST{};

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = operator()(ast->nestedNameSpecifier);
  copy->unqualifiedId = operator()(ast->unqualifiedId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->symbol = ast->symbol;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::operator()(EnumeratorAST* ast) -> EnumeratorAST* {
  if (!ast) return {};

  auto copy = new (arena()) EnumeratorAST{};

  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->equalLoc = ast->equalLoc;
  copy->expression = operator()(ast->expression);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(TypeIdAST* ast) -> TypeIdAST* {
  if (!ast) return {};

  auto copy = new (arena()) TypeIdAST{};

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = operator()(ast->declarator);
  copy->type = ast->type;

  return copy;
}

auto ASTRewriter::operator()(HandlerAST* ast) -> HandlerAST* {
  if (!ast) return {};

  auto copy = new (arena()) HandlerAST{};

  copy->catchLoc = ast->catchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->exceptionDeclaration = operator()(ast->exceptionDeclaration);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(operator()(ast->statement));

  return copy;
}

auto ASTRewriter::operator()(BaseSpecifierAST* ast) -> BaseSpecifierAST* {
  if (!ast) return {};

  auto copy = new (arena()) BaseSpecifierAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = operator()(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = operator()(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
  copy->isVirtual = ast->isVirtual;
  copy->accessSpecifier = ast->accessSpecifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::operator()(RequiresClauseAST* ast) -> RequiresClauseAST* {
  if (!ast) return {};

  auto copy = new (arena()) RequiresClauseAST{};

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = operator()(ast->expression);

  return copy;
}

auto ASTRewriter::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseAST* {
  if (!ast) return {};

  auto copy = new (arena()) ParameterDeclarationClauseAST{};

  if (auto it = ast->parameterDeclarationList) {
    auto out = &copy->parameterDeclarationList;
    for (auto node : ListView{ast->parameterDeclarationList}) {
      auto value = operator()(node);
      *out = new (arena()) List(ast_cast<ParameterDeclarationAST>(value));
      out = &(*out)->next;
    }
  }

  copy->commaLoc = ast->commaLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->functionParametersSymbol = ast->functionParametersSymbol;
  copy->isVariadic = ast->isVariadic;

  return copy;
}

auto ASTRewriter::operator()(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeAST* {
  if (!ast) return {};

  auto copy = new (arena()) TrailingReturnTypeAST{};

  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeId = operator()(ast->typeId);

  return copy;
}

auto ASTRewriter::operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierAST* {
  if (!ast) return {};

  auto copy = new (arena()) LambdaSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::operator()(TypeConstraintAST* ast) -> TypeConstraintAST* {
  if (!ast) return {};

  auto copy = new (arena()) TypeConstraintAST{};

  copy->nestedNameSpecifier = operator()(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;
    for (auto node : ListView{ast->templateArgumentList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseAST* {
  if (!ast) return {};

  auto copy = new (arena()) AttributeArgumentClauseAST{};

  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::operator()(AttributeAST* ast) -> AttributeAST* {
  if (!ast) return {};

  auto copy = new (arena()) AttributeAST{};

  copy->attributeToken = operator()(ast->attributeToken);
  copy->attributeArgumentClause = operator()(ast->attributeArgumentClause);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::operator()(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixAST* {
  if (!ast) return {};

  auto copy = new (arena()) AttributeUsingPrefixAST{};

  copy->usingLoc = ast->usingLoc;
  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::operator()(NewPlacementAST* ast) -> NewPlacementAST* {
  if (!ast) return {};

  auto copy = new (arena()) NewPlacementAST{};

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = operator()(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierAST* {
  if (!ast) return {};

  auto copy = new (arena()) NestedNamespaceSpecifierAST{};

  copy->inlineLoc = ast->inlineLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitAST* {
  auto copy = new (arena()) TranslationUnitAST{};

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitAST* {
  auto copy = new (arena()) ModuleUnitAST{};

  copy->globalModuleFragment = rewrite(ast->globalModuleFragment);
  copy->moduleDeclaration = rewrite(ast->moduleDeclaration);

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->privateModuleFragment = rewrite(ast->privateModuleFragment);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) SimpleDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;
    for (auto node : ListView{ast->declSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->initDeclaratorList) {
    auto out = &copy->initDeclaratorList;
    for (auto node : ListView{ast->initDeclaratorList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->requiresClause = rewrite(ast->requiresClause);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AsmDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->asmQualifierList) {
    auto out = &copy->asmQualifierList;
    for (auto node : ListView{ast->asmQualifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<AsmQualifierAST>(value));
      out = &(*out)->next;
    }
  }

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;

  if (auto it = ast->outputOperandList) {
    auto out = &copy->outputOperandList;
    for (auto node : ListView{ast->outputOperandList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<AsmOperandAST>(value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->inputOperandList) {
    auto out = &copy->inputOperandList;
    for (auto node : ListView{ast->inputOperandList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<AsmOperandAST>(value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->clobberList) {
    auto out = &copy->clobberList;
    for (auto node : ListView{ast->clobberList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<AsmClobberAST>(value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->gotoLabelList) {
    auto out = &copy->gotoLabelList;
    for (auto node : ListView{ast->gotoLabelList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<AsmGotoLabelAST>(value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationAST* {
  auto copy = new (arena()) NamespaceAliasDefinitionAST{};

  copy->namespaceLoc = ast->namespaceLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) UsingDeclarationAST{};

  copy->usingLoc = ast->usingLoc;

  if (auto it = ast->usingDeclaratorList) {
    auto out = &copy->usingDeclaratorList;
    for (auto node : ListView{ast->usingDeclaratorList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) UsingEnumDeclarationAST{};

  copy->usingLoc = ast->usingLoc;
  copy->enumTypeSpecifier =
      ast_cast<ElaboratedTypeSpecifierAST>(rewrite(ast->enumTypeSpecifier));
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) UsingDirectiveAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->usingLoc = ast->usingLoc;
  copy->namespaceLoc = ast->namespaceLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) -> DeclarationAST* {
  auto copy = new (arena()) StaticAssertDeclarationAST{};

  copy->staticAssertLoc = ast->staticAssertLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AliasDeclarationAST{};

  copy->usingLoc = ast->usingLoc;
  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->equalLoc = ast->equalLoc;

  if (auto it = ast->gnuAttributeList) {
    auto out = &copy->gnuAttributeList;
    for (auto node : ListView{ast->gnuAttributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->typeId = rewrite(ast->typeId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) OpaqueEnumDeclarationAST{};

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->emicolonLoc = ast->emicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) FunctionDefinitionAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;
    for (auto node : ListView{ast->declSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = rewrite(ast->declarator);
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->functionBody = rewrite(ast->functionBody);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) TemplateDeclarationAST{};

  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;
    for (auto node : ListView{ast->templateParameterList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->declaration = rewrite(ast->declaration);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) ConceptDefinitionAST{};

  copy->conceptLoc = ast->conceptLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) DeductionGuideAST{};

  copy->explicitSpecifier = rewrite(ast->explicitSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->arrowLoc = ast->arrowLoc;
  copy->templateId = ast_cast<SimpleTemplateIdAST>(rewrite(ast->templateId));
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) ExplicitInstantiationAST{};

  copy->externLoc = ast->externLoc;
  copy->templateLoc = ast->templateLoc;
  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) ExportDeclarationAST{};

  copy->exportLoc = ast->exportLoc;
  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationAST* {
  auto copy = new (arena()) ExportCompoundDeclarationAST{};

  copy->exportLoc = ast->exportLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) LinkageSpecificationAST{};

  copy->externLoc = ast->externLoc;
  copy->stringliteralLoc = ast->stringliteralLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->stringLiteral = ast->stringLiteral;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) NamespaceDefinitionAST{};

  copy->inlineLoc = ast->inlineLoc;
  copy->namespaceLoc = ast->namespaceLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->nestedNamespaceSpecifierList) {
    auto out = &copy->nestedNamespaceSpecifierList;
    for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->extraAttributeList) {
    auto out = &copy->extraAttributeList;
    for (auto node : ListView{ast->extraAttributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) EmptyDeclarationAST{};

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AttributeDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationAST* {
  auto copy = new (arena()) ModuleImportDeclarationAST{};

  copy->importLoc = ast->importLoc;
  copy->importName = rewrite(ast->importName);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) ParameterDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->thisLoc = ast->thisLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = rewrite(ast->declarator);
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite(ast->expression);
  copy->type = ast->type;
  copy->identifier = ast->identifier;
  copy->isThisIntroduced = ast->isThisIntroduced;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AccessDeclarationAST{};

  copy->accessLoc = ast->accessLoc;
  copy->colonLoc = ast->colonLoc;
  copy->accessSpecifier = ast->accessSpecifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) ForRangeDeclarationAST{};

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationAST* {
  auto copy = new (arena()) StructuredBindingDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;
    for (auto node : ListView{ast->declSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->refQualifierLoc = ast->refQualifierLoc;
  copy->lbracketLoc = ast->lbracketLoc;

  if (auto it = ast->bindingList) {
    auto out = &copy->bindingList;
    for (auto node : ListView{ast->bindingList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(ast_cast<NameIdAST>(value));
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AsmOperandAST{};

  copy->lbracketLoc = ast->lbracketLoc;
  copy->symbolicNameLoc = ast->symbolicNameLoc;
  copy->rbracketLoc = ast->rbracketLoc;
  copy->constraintLiteralLoc = ast->constraintLiteralLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->symbolicName = ast->symbolicName;
  copy->constraintLiteral = ast->constraintLiteral;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmQualifierAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AsmQualifierAST{};

  copy->qualifierLoc = ast->qualifierLoc;
  copy->qualifier = ast->qualifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AsmClobberAST{};

  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationAST* {
  auto copy = new (arena()) AsmGotoLabelAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) LabeledStatementAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->colonLoc = ast->colonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) CaseStatementAST{};

  copy->caseLoc = ast->caseLoc;
  copy->expression = rewrite(ast->expression);
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) DefaultStatementAST{};

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ExpressionStatementAST{};

  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) CompoundStatementAST{};

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->statementList) {
    auto out = &copy->statementList;
    for (auto node : ListView{ast->statementList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) IfStatementAST{};

  copy->ifLoc = ast->ifLoc;
  copy->constexprLoc = ast->constexprLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->condition = rewrite(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);
  copy->elseLoc = ast->elseLoc;
  copy->elseStatement = rewrite(ast->elseStatement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ConstevalIfStatementAST{};

  copy->ifLoc = ast->ifLoc;
  copy->exclaimLoc = ast->exclaimLoc;
  copy->constvalLoc = ast->constvalLoc;
  copy->statement = rewrite(ast->statement);
  copy->elseLoc = ast->elseLoc;
  copy->elseStatement = rewrite(ast->elseStatement);
  copy->isNot = ast->isNot;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) SwitchStatementAST{};

  copy->switchLoc = ast->switchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->condition = rewrite(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) WhileStatementAST{};

  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->condition = rewrite(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) DoStatementAST{};

  copy->doLoc = ast->doLoc;
  copy->statement = rewrite(ast->statement);
  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ForRangeStatementAST{};

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->rangeDeclaration = rewrite(ast->rangeDeclaration);
  copy->colonLoc = ast->colonLoc;
  copy->rangeInitializer = rewrite(ast->rangeInitializer);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ForStatementAST{};

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->condition = rewrite(ast->condition);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) BreakStatementAST{};

  copy->breakLoc = ast->breakLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ContinueStatementAST{};

  copy->continueLoc = ast->continueLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) ReturnStatementAST{};

  copy->returnLoc = ast->returnLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CoroutineReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) CoroutineReturnStatementAST{};

  copy->coreturnLoc = ast->coreturnLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) GotoStatementAST{};

  copy->gotoLoc = ast->gotoLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) DeclarationStatementAST{};

  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementAST* {
  auto copy = new (arena()) TryBlockStatementAST{};

  copy->tryLoc = ast->tryLoc;
  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  if (auto it = ast->handlerList) {
    auto out = &copy->handlerList;
    for (auto node : ListView{ast->handlerList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GeneratedLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) GeneratedLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->value = ast->value;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) CharLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) BoolLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->isTrue = ast->isTrue;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) IntLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) FloatLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) NullptrLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) StringLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) UserDefinedStringLiteralExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ThisExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) NestedStatementExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) NestedExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) IdExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->symbol = ast->symbol;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) LambdaExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->captureDefaultLoc = ast->captureDefaultLoc;

  if (auto it = ast->captureList) {
    auto out = &copy->captureList;
    for (auto node : ListView{ast->captureList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;
    for (auto node : ListView{ast->templateParameterList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->templateRequiresClause = rewrite(ast->templateRequiresClause);
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  if (auto it = ast->gnuAtributeList) {
    auto out = &copy->gnuAtributeList;
    for (auto node : ListView{ast->gnuAtributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->lambdaSpecifierList) {
    auto out = &copy->lambdaSpecifierList;
    for (auto node : ListView{ast->lambdaSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->exceptionSpecifier = rewrite(ast->exceptionSpecifier);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->trailingReturnType = rewrite(ast->trailingReturnType);
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));
  copy->captureDefault = ast->captureDefault;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) FoldExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->leftExpression = rewrite(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->foldOpLoc = ast->foldOpLoc;
  copy->rightExpression = rewrite(ast->rightExpression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;
  copy->foldOp = ast->foldOp;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) RightFoldExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) LeftFoldExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) RequiresExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->requiresLoc = ast->requiresLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->requirementList) {
    auto out = &copy->requirementList;
    for (auto node : ListView{ast->requirementList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) VaArgExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->vaArgLoc = ast->vaArgLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SubscriptExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->lbracketLoc = ast->lbracketLoc;
  copy->indexExpression = rewrite(ast->indexExpression);
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) CallExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) TypeConstructionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite(ast->typeSpecifier);
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) BracedTypeConstructionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite(ast->typeSpecifier);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SpliceMemberExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->templateLoc = ast->templateLoc;
  copy->splicer = rewrite(ast->splicer);
  copy->symbol = ast->symbol;
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) MemberExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->symbol = ast->symbol;
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) PostIncrExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->opLoc = ast->opLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) CppCastExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->castLoc = ast->castLoc;
  copy->lessLoc = ast->lessLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->greaterLoc = ast->greaterLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) BuiltinBitCastExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->castLoc = ast->castLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) BuiltinOffsetofExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->offsetofLoc = ast->offsetofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) TypeidExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) TypeidOfTypeExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SpliceExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->splicer = rewrite(ast->splicer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) GlobalScopeReflectExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) NamespaceReflectExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeIdReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) TypeIdReflectExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ReflectExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) UnaryExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite(ast->expression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) AwaitExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->awaitLoc = ast->awaitLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SizeofExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SizeofTypeExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) SizeofPackExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) AlignofTypeExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) AlignofExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) NoexceptExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) NewExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->newLoc = ast->newLoc;
  copy->newPlacement = rewrite(ast->newPlacement);
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = rewrite(ast->declarator);
  copy->rparenLoc = ast->rparenLoc;
  copy->newInitalizer = rewrite(ast->newInitalizer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) DeleteExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->rbracketLoc = ast->rbracketLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) CastExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ImplicitCastExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite(ast->expression);
  copy->castKind = ast->castKind;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) BinaryExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->leftExpression = rewrite(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ConditionalExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->condition = rewrite(ast->condition);
  copy->questionLoc = ast->questionLoc;
  copy->iftrueExpression = rewrite(ast->iftrueExpression);
  copy->colonLoc = ast->colonLoc;
  copy->iffalseExpression = rewrite(ast->iffalseExpression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) YieldExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->yieldLoc = ast->yieldLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ThrowExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->throwLoc = ast->throwLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) AssignmentExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->leftExpression = rewrite(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) PackExpansionExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionAST* {
  auto copy = new (arena()) DesignatedInitializerClauseAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->dotLoc = ast->dotLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->initializer = rewrite(ast->initializer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) TypeTraitExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeTraitLoc = ast->typeTraitLoc;
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->typeIdList) {
    auto out = &copy->typeIdList;
    for (auto node : ListView{ast->typeIdList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->typeTrait = ast->typeTrait;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ConditionExpressionAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;
    for (auto node : ListView{ast->declSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = rewrite(ast->declarator);
  copy->initializer = rewrite(ast->initializer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) EqualInitializerAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) BracedInitListAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = new (arena()) ParenInitializerAST{};

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = new (arena()) TemplateTypeParameterAST{};

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;
    for (auto node : ListView{ast->templateParameterList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->idExpression = ast_cast<IdExpressionAST>(rewrite(ast->idExpression));
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterAST* {
  auto copy = new (arena()) NonTypeTemplateParameterAST{};

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->declaration =
      ast_cast<ParameterDeclarationAST>(rewrite(ast->declaration));

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = new (arena()) TypenameTypeParameterAST{};

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = new (arena()) ConstraintTypeParameterAST{};

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->typeConstraint = rewrite(ast->typeConstraint);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(GeneratedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) GeneratedTypeSpecifierAST{};

  copy->typeLoc = ast->typeLoc;
  copy->type = ast->type;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) TypedefSpecifierAST{};

  copy->typedefLoc = ast->typedefLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) FriendSpecifierAST{};

  copy->friendLoc = ast->friendLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ConstevalSpecifierAST{};

  copy->constevalLoc = ast->constevalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ConstinitSpecifierAST{};

  copy->constinitLoc = ast->constinitLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ConstexprSpecifierAST{};

  copy->constexprLoc = ast->constexprLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) InlineSpecifierAST{};

  copy->inlineLoc = ast->inlineLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) StaticSpecifierAST{};

  copy->staticLoc = ast->staticLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ExternSpecifierAST{};

  copy->externLoc = ast->externLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ThreadLocalSpecifierAST{};

  copy->threadLocalLoc = ast->threadLocalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ThreadSpecifierAST{};

  copy->threadLoc = ast->threadLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) MutableSpecifierAST{};

  copy->mutableLoc = ast->mutableLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) VirtualSpecifierAST{};

  copy->virtualLoc = ast->virtualLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ExplicitSpecifierAST{};

  copy->explicitLoc = ast->explicitLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) AutoTypeSpecifierAST{};

  copy->autoLoc = ast->autoLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) VoidTypeSpecifierAST{};

  copy->voidLoc = ast->voidLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) SizeTypeSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) SignTypeSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) VaListTypeSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) IntegralTypeSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierAST* {
  auto copy = new (arena()) FloatingPointTypeSpecifierAST{};

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ComplexTypeSpecifierAST{};

  copy->complexLoc = ast->complexLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) NamedTypeSpecifierAST{};

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) AtomicTypeSpecifierAST{};

  copy->atomicLoc = ast->atomicLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(UnderlyingTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) UnderlyingTypeSpecifierAST{};

  copy->underlyingTypeLoc = ast->underlyingTypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ElaboratedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ElaboratedTypeSpecifierAST{};

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->classKey = ast->classKey;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) DecltypeAutoSpecifierAST{};

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->autoLoc = ast->autoLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) DecltypeSpecifierAST{};

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->type = ast->type;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(PlaceholderTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) PlaceholderTypeSpecifierAST{};

  copy->typeConstraint = rewrite(ast->typeConstraint);
  copy->specifier = rewrite(ast->specifier);

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ConstQualifierAST{};

  copy->constLoc = ast->constLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) VolatileQualifierAST{};

  copy->volatileLoc = ast->volatileLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) RestrictQualifierAST{};

  copy->restrictLoc = ast->restrictLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) EnumSpecifierAST{};

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;
  copy->commaLoc = ast->commaLoc;

  if (auto it = ast->enumeratorList) {
    auto out = &copy->enumeratorList;
    for (auto node : ListView{ast->enumeratorList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) ClassSpecifierAST{};

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->finalLoc = ast->finalLoc;
  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->baseSpecifierList) {
    auto out = &copy->baseSpecifierList;
    for (auto node : ListView{ast->baseSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;
    for (auto node : ListView{ast->declarationList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->classKey = ast->classKey;
  copy->symbol = ast->symbol;
  copy->isFinal = ast->isFinal;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) TypenameSpecifierAST{};

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = new (arena()) SplicerTypeSpecifierAST{};

  copy->typenameLoc = ast->typenameLoc;
  copy->splicer = rewrite(ast->splicer);

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = new (arena()) PointerOperatorAST{};

  copy->starLoc = ast->starLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;
    for (auto node : ListView{ast->cvQualifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = new (arena()) ReferenceOperatorAST{};

  copy->refLoc = ast->refLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->refOp = ast->refOp;

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = new (arena()) PtrToMemberOperatorAST{};

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->starLoc = ast->starLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;
    for (auto node : ListView{ast->cvQualifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = new (arena()) BitfieldDeclaratorAST{};

  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;
  copy->sizeExpression = rewrite(ast->sizeExpression);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = new (arena()) ParameterPackAST{};

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->coreDeclarator = rewrite(ast->coreDeclarator);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = new (arena()) IdDeclaratorAST{};

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = new (arena()) NestedDeclaratorAST{};

  copy->lparenLoc = ast->lparenLoc;
  copy->declarator = rewrite(ast->declarator);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = new (arena()) FunctionDeclaratorChunkAST{};

  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;
    for (auto node : ListView{ast->cvQualifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->refLoc = ast->refLoc;
  copy->exceptionSpecifier = rewrite(ast->exceptionSpecifier);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->trailingReturnType = rewrite(ast->trailingReturnType);
  copy->isFinal = ast->isFinal;
  copy->isOverride = ast->isOverride;
  copy->isPure = ast->isPure;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = new (arena()) ArrayDeclaratorChunkAST{};

  copy->lbracketLoc = ast->lbracketLoc;
  copy->expression = rewrite(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) NameIdAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) DestructorIdAST{};

  copy->tildeLoc = ast->tildeLoc;
  copy->id = rewrite(ast->id);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) DecltypeIdAST{};

  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite(ast->decltypeSpecifier));

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) OperatorFunctionIdAST{};

  copy->operatorLoc = ast->operatorLoc;
  copy->opLoc = ast->opLoc;
  copy->openLoc = ast->openLoc;
  copy->closeLoc = ast->closeLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) LiteralOperatorIdAST{};

  copy->operatorLoc = ast->operatorLoc;
  copy->literalLoc = ast->literalLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->literal = ast->literal;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) ConversionFunctionIdAST{};

  copy->operatorLoc = ast->operatorLoc;
  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = new (arena()) SimpleTemplateIdAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;
    for (auto node : ListView{ast->templateArgumentList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->primaryTemplateSymbol = ast->primaryTemplateSymbol;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = new (arena()) LiteralOperatorTemplateIdAST{};

  copy->literalOperatorId =
      ast_cast<LiteralOperatorIdAST>(rewrite(ast->literalOperatorId));
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;
    for (auto node : ListView{ast->templateArgumentList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = new (arena()) OperatorFunctionTemplateIdAST{};

  copy->operatorFunctionId =
      ast_cast<OperatorFunctionIdAST>(rewrite(ast->operatorFunctionId));
  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;
    for (auto node : ListView{ast->templateArgumentList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = new (arena()) GlobalNestedNameSpecifierAST{};

  copy->symbol = ast->symbol;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = new (arena()) SimpleNestedNameSpecifierAST{};

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = new (arena()) DecltypeNestedNameSpecifierAST{};

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = new (arena()) TemplateNestedNameSpecifierAST{};

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->templateId = ast_cast<SimpleTemplateIdAST>(rewrite(ast->templateId));
  copy->scopeLoc = ast->scopeLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = new (arena()) DefaultFunctionBodyAST{};

  copy->equalLoc = ast->equalLoc;
  copy->defaultLoc = ast->defaultLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = new (arena()) CompoundStatementFunctionBodyAST{};

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->memInitializerList) {
    auto out = &copy->memInitializerList;
    for (auto node : ListView{ast->memInitializerList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = new (arena()) TryStatementFunctionBodyAST{};

  copy->tryLoc = ast->tryLoc;
  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->memInitializerList) {
    auto out = &copy->memInitializerList;
    for (auto node : ListView{ast->memInitializerList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  if (auto it = ast->handlerList) {
    auto out = &copy->handlerList;
    for (auto node : ListView{ast->handlerList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = new (arena()) DeleteFunctionBodyAST{};

  copy->equalLoc = ast->equalLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = new (arena()) TypeTemplateArgumentAST{};

  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = new (arena()) ExpressionTemplateArgumentAST{};

  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = new (arena()) ThrowExceptionSpecifierAST{};

  copy->throwLoc = ast->throwLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = new (arena()) NoexceptSpecifierAST{};

  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementAST* {
  auto copy = new (arena()) SimpleRequirementAST{};

  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementAST* {
  auto copy = new (arena()) CompoundRequirementAST{};

  copy->lbraceLoc = ast->lbraceLoc;
  copy->expression = rewrite(ast->expression);
  copy->rbraceLoc = ast->rbraceLoc;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeConstraint = rewrite(ast->typeConstraint);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementAST* {
  auto copy = new (arena()) TypeRequirementAST{};

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementAST* {
  auto copy = new (arena()) NestedRequirementAST{};

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerAST* {
  auto copy = new (arena()) NewParenInitializerAST{};

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerAST* {
  auto copy = new (arena()) NewBracedInitializerAST{};

  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerAST* {
  auto copy = new (arena()) ParenMemInitializerAST{};

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;
    for (auto node : ListView{ast->expressionList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerAST* {
  auto copy = new (arena()) BracedMemInitializerAST{};

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = new (arena()) ThisLambdaCaptureAST{};

  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureAST* {
  auto copy = new (arena()) DerefThisLambdaCaptureAST{};

  copy->starLoc = ast->starLoc;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = new (arena()) SimpleLambdaCaptureAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = new (arena()) RefLambdaCaptureAST{};

  copy->ampLoc = ast->ampLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = new (arena()) RefInitLambdaCaptureAST{};

  copy->ampLoc = ast->ampLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = new (arena()) InitLambdaCaptureAST{};

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = new (arena()) EllipsisExceptionDeclarationAST{};

  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = new (arena()) TypeExceptionDeclarationAST{};

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;
    for (auto node : ListView{ast->typeSpecifierList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->declarator = rewrite(ast->declarator);

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = new (arena()) CxxAttributeAST{};

  copy->lbracketLoc = ast->lbracketLoc;
  copy->lbracket2Loc = ast->lbracket2Loc;
  copy->attributeUsingPrefix = rewrite(ast->attributeUsingPrefix);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;
    for (auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *out = new (arena()) List(value);
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->rbracket2Loc = ast->rbracket2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = new (arena()) GccAttributeAST{};

  copy->attributeLoc = ast->attributeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->lparen2Loc = ast->lparen2Loc;
  copy->rparenLoc = ast->rparenLoc;
  copy->rparen2Loc = ast->rparen2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierAST* {
  auto copy = new (arena()) AlignasAttributeAST{};

  copy->alignasLoc = ast->alignasLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierAST* {
  auto copy = new (arena()) AlignasTypeAttributeAST{};

  copy->alignasLoc = ast->alignasLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = new (arena()) AsmAttributeAST{};

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = new (arena()) ScopedAttributeTokenAST{};

  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->attributeNamespace = ast->attributeNamespace;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = new (arena()) SimpleAttributeTokenAST{};

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

}  // namespace cxx
