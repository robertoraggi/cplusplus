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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>

namespace cxx {

struct ASTInterpreter::UnitResult {};

struct ASTInterpreter::DeclarationResult {};

struct ASTInterpreter::StatementResult {};

struct ASTInterpreter::ExpressionResult {};

struct ASTInterpreter::TemplateParameterResult {};

struct ASTInterpreter::SpecifierResult {};

struct ASTInterpreter::PtrOperatorResult {};

struct ASTInterpreter::CoreDeclaratorResult {};

struct ASTInterpreter::DeclaratorChunkResult {};

struct ASTInterpreter::UnqualifiedIdResult {};

struct ASTInterpreter::NestedNameSpecifierResult {};

struct ASTInterpreter::FunctionBodyResult {};

struct ASTInterpreter::TemplateArgumentResult {};

struct ASTInterpreter::ExceptionSpecifierResult {};

struct ASTInterpreter::RequirementResult {};

struct ASTInterpreter::NewInitializerResult {};

struct ASTInterpreter::MemInitializerResult {};

struct ASTInterpreter::LambdaCaptureResult {};

struct ASTInterpreter::ExceptionDeclarationResult {};

struct ASTInterpreter::AttributeSpecifierResult {};

struct ASTInterpreter::AttributeTokenResult {};

struct ASTInterpreter::SplicerResult {};

struct ASTInterpreter::GlobalModuleFragmentResult {};

struct ASTInterpreter::PrivateModuleFragmentResult {};

struct ASTInterpreter::ModuleDeclarationResult {};

struct ASTInterpreter::ModuleNameResult {};

struct ASTInterpreter::ModuleQualifierResult {};

struct ASTInterpreter::ModulePartitionResult {};

struct ASTInterpreter::ImportNameResult {};

struct ASTInterpreter::InitDeclaratorResult {};

struct ASTInterpreter::DeclaratorResult {};

struct ASTInterpreter::UsingDeclaratorResult {};

struct ASTInterpreter::EnumeratorResult {};

struct ASTInterpreter::TypeIdResult {};

struct ASTInterpreter::HandlerResult {};

struct ASTInterpreter::BaseSpecifierResult {};

struct ASTInterpreter::RequiresClauseResult {};

struct ASTInterpreter::ParameterDeclarationClauseResult {};

struct ASTInterpreter::TrailingReturnTypeResult {};

struct ASTInterpreter::LambdaSpecifierResult {};

struct ASTInterpreter::TypeConstraintResult {};

struct ASTInterpreter::AttributeArgumentClauseResult {};

struct ASTInterpreter::AttributeResult {};

struct ASTInterpreter::AttributeUsingPrefixResult {};

struct ASTInterpreter::NewPlacementResult {};

struct ASTInterpreter::NestedNamespaceSpecifierResult {};

struct ASTInterpreter::UnitVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitResult;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitResult;
};

struct ASTInterpreter::DeclarationVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(SimpleDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(NamespaceAliasDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingEnumDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingDirectiveAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(StaticAssertDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(AliasDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(OpaqueEnumDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(FunctionDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(TemplateDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ConceptDefinitionAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ExportDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(EmptyDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(AccessDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmOperandAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmQualifierAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmClobberAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmGotoLabelAST* ast) -> DeclarationResult;
};

struct ASTInterpreter::StatementVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementResult;

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

struct ASTInterpreter::ExpressionVisitor {
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

  [[nodiscard]] auto operator()(NestedStatementExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) -> ExpressionResult;

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

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) -> ExpressionResult;

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

  [[nodiscard]] auto operator()(ReflectExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(AwaitExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) -> ExpressionResult;

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

struct ASTInterpreter::TemplateParameterVisitor {
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

struct ASTInterpreter::SpecifierVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(GeneratedTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast) -> SpecifierResult;

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

  [[nodiscard]] auto operator()(VaListTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(IntegralTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(FloatingPointTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(ComplexTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(NamedTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(AtomicTypeSpecifierAST* ast) -> SpecifierResult;

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

struct ASTInterpreter::PtrOperatorVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
      -> PtrOperatorResult;
};

struct ASTInterpreter::CoreDeclaratorVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
      -> CoreDeclaratorResult;
};

struct ASTInterpreter::DeclaratorChunkVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
};

struct ASTInterpreter::UnqualifiedIdVisitor {
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

struct ASTInterpreter::NestedNameSpecifierVisitor {
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

struct ASTInterpreter::FunctionBodyVisitor {
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

struct ASTInterpreter::TemplateArgumentVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentResult;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentResult;
};

struct ASTInterpreter::ExceptionSpecifierVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierResult;
};

struct ASTInterpreter::RequirementVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
      -> RequirementResult;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementResult;
};

struct ASTInterpreter::NewInitializerVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerResult;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerResult;
};

struct ASTInterpreter::MemInitializerVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerResult;
};

struct ASTInterpreter::LambdaCaptureVisitor {
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

struct ASTInterpreter::ExceptionDeclarationVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
};

struct ASTInterpreter::AttributeSpecifierVisitor {
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

struct ASTInterpreter::AttributeTokenVisitor {
  ASTInterpreter& accept;

  [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
      -> AttributeTokenResult;

  [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
      -> AttributeTokenResult;
};

auto ASTInterpreter::operator()(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(DeclarationAST* ast) -> DeclarationResult {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(StatementAST* ast) -> StatementResult {
  if (ast) return visit(StatementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExpressionAST* ast) -> ExpressionResult {
  if (ast) return visit(ExpressionVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(TemplateParameterAST* ast)
    -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(SpecifierAST* ast) -> SpecifierResult {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(CoreDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(DeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdResult {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierResult {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(TemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(NewInitializerAST* ast)
    -> NewInitializerResult {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(MemInitializerAST* ast)
    -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(AttributeSpecifierAST* ast)
    -> AttributeSpecifierResult {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(AttributeTokenAST* ast)
    -> AttributeTokenResult {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(SplicerAST* ast) -> SplicerResult {
  if (!ast) return {};

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(ModuleDeclarationAST* ast)
    -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);
  auto modulePartitionResult = operator()(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::operator()(ModuleQualifierAST* ast)
    -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::operator()(ModulePartitionAST* ast)
    -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto ASTInterpreter::operator()(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = operator()(ast->modulePartition);
  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto ASTInterpreter::operator()(InitDeclaratorAST* ast)
    -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = operator()(ast->declarator);
  auto requiresClauseResult = operator()(ast->requiresClause);
  auto initializerResult = operator()(ast->initializer);

  return {};
}

auto ASTInterpreter::operator()(DeclaratorAST* ast) -> DeclaratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->ptrOpList}) {
    auto value = operator()(node);
  }

  auto coreDeclaratorResult = operator()(ast->coreDeclarator);

  for (auto node : ListView{ast->declaratorChunkList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(UsingDeclaratorAST* ast)
    -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::operator()(EnumeratorAST* ast) -> EnumeratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(TypeIdAST* ast) -> TypeIdResult {
  if (!ast) return {};

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = operator()(node);
  }

  auto declaratorResult = operator()(ast->declarator);

  return {};
}

auto ASTInterpreter::operator()(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult = operator()(ast->exceptionDeclaration);
  auto statementResult = operator()(ast->statement);

  return {};
}

auto ASTInterpreter::operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::operator()(RequiresClauseAST* ast)
    -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = operator()(ast->typeId);

  return {};
}

auto ASTInterpreter::operator()(LambdaSpecifierAST* ast)
    -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(TypeConstraintAST* ast)
    -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(AttributeAST* ast) -> AttributeResult {
  if (!ast) return {};

  auto attributeTokenResult = operator()(ast->attributeToken);
  auto attributeArgumentClauseResult = operator()(ast->attributeArgumentClause);

  return {};
}

auto ASTInterpreter::operator()(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(NewPlacementAST* ast) -> NewPlacementResult {
  if (!ast) return {};

  for (auto node : ListView{ast->expressionList}) {
    auto value = operator()(node);
  }

  return {};
}

auto ASTInterpreter::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(TranslationUnitAST* ast)
    -> UnitResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto globalModuleFragmentResult = accept(ast->globalModuleFragment);
  auto moduleDeclarationResult = accept(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  auto privateModuleFragmentResult = accept(ast->privateModuleFragment);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = accept(node);
  }

  auto requiresClauseResult = accept(ast->requiresClause);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    UsingEnumDeclarationAST* ast) -> DeclarationResult {
  auto enumTypeSpecifierResult = accept(ast->enumTypeSpecifier);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = accept(node);
  }

  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    OpaqueEnumDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = accept(node);
  }

  auto declaratorResult = accept(ast->declarator);
  auto requiresClauseResult = accept(ast->requiresClause);
  auto functionBodyResult = accept(ast->functionBody);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = accept(node);
  }

  auto requiresClauseResult = accept(ast->requiresClause);
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  auto explicitSpecifierResult = accept(ast->explicitSpecifier);
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);
  auto templateIdResult = accept(ast->templateId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExplicitInstantiationAST* ast) -> DeclarationResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    LinkageSpecificationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    AttributeDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationResult {
  auto importNameResult = accept(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ParameterDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = accept(node);
  }

  auto declaratorResult = accept(ast->declarator);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = accept(node);
  }

  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmQualifierAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementResult {
  for (auto node : ListView{ast->statementList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);
  auto elseStatementResult = accept(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);
  auto elseStatementResult = accept(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementResult {
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto rangeDeclarationResult = accept(ast->rangeDeclaration);
  auto rangeInitializerResult = accept(ast->rangeInitializer);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto expressionResult = accept(ast->expression);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(
    CoroutineReturnStatementAST* ast) -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GeneratedLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    CharLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BoolLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    FloatLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    StringLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionResult {
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->captureList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->templateParameterList}) {
    auto value = accept(node);
  }

  auto templateRequiresClauseResult = accept(ast->templateRequiresClause);
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = accept(node);
  }

  auto exceptionSpecifierResult = accept(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto trailingReturnTypeResult = accept(ast->trailingReturnType);
  auto requiresClauseResult = accept(ast->requiresClause);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->requirementList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto indexExpressionResult = accept(ast->indexExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);

  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = accept(ast->typeSpecifier);

  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = accept(ast->typeSpecifier);
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeidOfTypeExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeIdReflectExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    AlignofTypeExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto newPlacementResult = accept(ast->newPlacement);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = accept(node);
  }

  auto declaratorResult = accept(ast->declarator);
  auto newInitalizerResult = accept(ast->newInitalizer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ConditionalExpressionAST* ast) -> ExpressionResult {
  auto conditionResult = accept(ast->condition);
  auto iftrueExpressionResult = accept(ast->iftrueExpression);
  auto iffalseExpressionResult = accept(ast->iffalseExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    PackExpansionExpressionAST* ast) -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->typeIdList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = accept(node);
  }

  auto declaratorResult = accept(ast->declarator);
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = accept(node);
  }

  auto requiresClauseResult = accept(ast->requiresClause);
  auto idExpressionResult = accept(ast->idExpression);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = accept(ast->typeConstraint);
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    GeneratedTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    UnderlyingTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    ElaboratedTypeSpecifierAST* ast) -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    PlaceholderTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeConstraintResult = accept(ast->typeConstraint);
  auto specifierResult = accept(ast->specifier);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->enumeratorList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto node : ListView{ast->baseSpecifierList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(
    BitfieldDeclaratorAST* ast) -> CoreDeclaratorResult {
  auto unqualifiedIdResult = accept(ast->unqualifiedId);
  auto sizeExpressionResult = accept(ast->sizeExpression);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = accept(ast->coreDeclarator);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = accept(ast->declarator);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = accept(node);
  }

  auto exceptionSpecifierResult = accept(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  auto trailingReturnTypeResult = accept(ast->trailingReturnType);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto expressionResult = accept(ast->expression);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdResult {
  auto idResult = accept(ast->id);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdResult {
  auto decltypeSpecifierResult = accept(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionIdAST* ast) -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    ConversionFunctionIdAST* ast) -> UnqualifiedIdResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdResult {
  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto literalOperatorIdResult = accept(ast->literalOperatorId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto operatorFunctionIdResult = accept(ast->operatorFunctionId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto decltypeSpecifierResult = accept(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto templateIdResult = accept(ast->templateId);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    DefaultFunctionBodyAST* ast) -> FunctionBodyResult {
  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = accept(node);
  }

  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = accept(node);
  }

  auto statementResult = accept(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierResult {
  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);
  auto typeConstraintResult = accept(ast->typeConstraint);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewParenInitializerAST* ast) -> NewInitializerResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerResult {
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    ParenMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto node : ListView{ast->expressionList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    SimpleLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    RefInitLambdaCaptureAST* ast) -> LambdaCaptureResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = accept(node);
  }

  auto declaratorResult = accept(ast->declarator);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto attributeUsingPrefixResult = accept(ast->attributeUsingPrefix);

  for (auto node : ListView{ast->attributeList}) {
    auto value = accept(node);
  }

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

ASTInterpreter::ASTInterpreter(TranslationUnit* unit) : unit_(unit) {}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

}  // namespace cxx
