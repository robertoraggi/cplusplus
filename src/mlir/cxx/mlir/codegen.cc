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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_cursor.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <format>

namespace cxx {

struct Codegen::UnitVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitResult;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitResult;
};

struct Codegen::DeclarationVisitor {
  Codegen& gen;

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

struct Codegen::StatementVisitor {
  Codegen& gen;

  void operator()(LabeledStatementAST* ast);
  void operator()(CaseStatementAST* ast);
  void operator()(DefaultStatementAST* ast);
  void operator()(ExpressionStatementAST* ast);
  void operator()(CompoundStatementAST* ast);
  void operator()(IfStatementAST* ast);
  void operator()(ConstevalIfStatementAST* ast);
  void operator()(SwitchStatementAST* ast);
  void operator()(WhileStatementAST* ast);
  void operator()(DoStatementAST* ast);
  void operator()(ForRangeStatementAST* ast);
  void operator()(ForStatementAST* ast);
  void operator()(BreakStatementAST* ast);
  void operator()(ContinueStatementAST* ast);
  void operator()(ReturnStatementAST* ast);
  void operator()(CoroutineReturnStatementAST* ast);
  void operator()(GotoStatementAST* ast);
  void operator()(DeclarationStatementAST* ast);
  void operator()(TryBlockStatementAST* ast);
};

struct Codegen::ExpressionVisitor {
  Codegen& gen;

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

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST* ast)
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

  [[nodiscard]] auto operator()(LabelAddressExpressionAST* ast)
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

struct Codegen::TemplateParameterVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
      -> TemplateParameterResult;
};

struct Codegen::SpecifierVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(GeneratedTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(InlineSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(NoreturnSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(StaticSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ExternSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(RegisterSpecifierAST* ast) -> SpecifierResult;

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

  [[nodiscard]] auto operator()(AtomicQualifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast)
      -> SpecifierResult;
};

struct Codegen::PtrOperatorVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
      -> PtrOperatorResult;
};

struct Codegen::CoreDeclaratorVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
      -> CoreDeclaratorResult;
};

struct Codegen::DeclaratorChunkVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
};

struct Codegen::UnqualifiedIdVisitor {
  Codegen& gen;

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

struct Codegen::NestedNameSpecifierVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
};

struct Codegen::FunctionBodyVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast)
      -> FunctionBodyResult;
};

struct Codegen::TemplateArgumentVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentResult;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentResult;
};

struct Codegen::ExceptionSpecifierVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierResult;
};

struct Codegen::RequirementVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
      -> RequirementResult;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementResult;
};

struct Codegen::NewInitializerVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerResult;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerResult;
};

struct Codegen::MemInitializerVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerResult;
};

struct Codegen::LambdaCaptureVisitor {
  Codegen& gen;

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

struct Codegen::ExceptionDeclarationVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
};

struct Codegen::AttributeSpecifierVisitor {
  Codegen& gen;

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

struct Codegen::AttributeTokenVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
      -> AttributeTokenResult;

  [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
      -> AttributeTokenResult;
};

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit)
    : builder_(&context), unit_(unit) {}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::getLocation(SourceLocation location) -> mlir::Location {
  auto [filename, line, column] = unit_->tokenStartPosition(location);

  auto loc =
      mlir::FileLineColLoc::get(builder_.getContext(), filename, line, column);

  return loc;
}

auto Codegen::emitTodoStmt(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoStmtOp {
#if true
  auto loc = getLocation(location);
#else
  auto loc = builder_.getUnknownLoc();
#endif

  auto op = builder_.create<mlir::cxx::TodoStmtOp>(loc, message);
  return op;
}

auto Codegen::emitTodoExpr(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoExprOp {
#if true
  auto loc = getLocation(location);
#else
  auto loc = builder_.getUnknownLoc();
#endif

  auto op = builder_.create<mlir::cxx::TodoExprOp>(loc, message);
  return op;
}

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(DeclarationAST* ast) -> DeclarationResult {
  // if (ast) return visit(DeclarationVisitor{*this}, ast);

  // restrict for now to declarations that are not definitions
  if (ast_cast<FunctionDefinitionAST>(ast) ||
      ast_cast<LinkageSpecificationAST>(ast) ||
      ast_cast<SimpleDeclarationAST>(ast) ||
      ast_cast<NamespaceDefinitionAST>(ast)) {
    return visit(DeclarationVisitor{*this}, ast);
  }

  return {};
}

void Codegen::statement(StatementAST* ast) {
  if (!ast) return;
  visit(StatementVisitor{*this}, ast);
}

auto Codegen::expression(ExpressionAST* ast) -> ExpressionResult {
  if (ast) return visit(ExpressionVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(TemplateParameterAST* ast) -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(SpecifierAST* ast) -> SpecifierResult {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdResult {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierResult {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(TemplateArgumentAST* ast) -> TemplateArgumentResult {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(NewInitializerAST* ast) -> NewInitializerResult {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(MemInitializerAST* ast) -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(AttributeSpecifierAST* ast)
    -> AttributeSpecifierResult {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(AttributeTokenAST* ast) -> AttributeTokenResult {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(SplicerAST* ast) -> SplicerResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::operator()(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(ModuleDeclarationAST* ast) -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);
  auto modulePartitionResult = operator()(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto Codegen::operator()(ModuleQualifierAST* ast) -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto Codegen::operator()(ModulePartitionAST* ast) -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto Codegen::operator()(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = operator()(ast->modulePartition);
  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto Codegen::operator()(InitDeclaratorAST* ast) -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = operator()(ast->declarator);
  auto requiresClauseResult = operator()(ast->requiresClause);
  auto initializerResult = expression(ast->initializer);

  return {};
}

auto Codegen::operator()(DeclaratorAST* ast) -> DeclaratorResult {
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

auto Codegen::operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto Codegen::operator()(EnumeratorAST* ast) -> EnumeratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::operator()(TypeIdAST* ast) -> TypeIdResult {
  if (!ast) return {};

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = operator()(node);
  }

  auto declaratorResult = operator()(ast->declarator);

  return {};
}

auto Codegen::operator()(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult = operator()(ast->exceptionDeclaration);
  statement(ast->statement);

  return {};
}

auto Codegen::operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto Codegen::operator()(RequiresClauseAST* ast) -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = operator()(ast->typeId);

  return {};
}

auto Codegen::operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::operator()(TypeConstraintAST* ast) -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseResult {
  if (!ast) return {};

  return {};
}

auto Codegen::operator()(AttributeAST* ast) -> AttributeResult {
  if (!ast) return {};

  auto attributeTokenResult = operator()(ast->attributeToken);
  auto attributeArgumentClauseResult = operator()(ast->attributeArgumentClause);

  return {};
}

auto Codegen::operator()(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixResult {
  if (!ast) return {};

  return {};
}

auto Codegen::operator()(NewPlacementAST* ast) -> NewPlacementResult {
  if (!ast) return {};

  for (auto node : ListView{ast->expressionList}) {
    auto value = expression(node);
  }

  return {};
}

auto Codegen::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = gen.builder_.create<mlir::ModuleOp>(loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = gen.builder_.create<mlir::ModuleOp>(loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  auto globalModuleFragmentResult = gen(ast->globalModuleFragment);
  auto moduleDeclarationResult = gen(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  auto privateModuleFragmentResult = gen(ast->privateModuleFragment);

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceAliasDefinitionAST* ast)
    -> DeclarationResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationResult {
  auto enumTypeSpecifierResult = gen(ast->enumTypeSpecifier);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(StaticAssertDeclarationAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = gen(node);
  }

  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  auto functionSymbol = ast->symbol;
  auto functionType = type_cast<FunctionType>(functionSymbol->type());

  auto exprType = gen.builder_.getType<mlir::cxx::ExprType>();

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  for (auto paramTy : functionType->parameterTypes()) {
    inputTypes.push_back(exprType);
  }

  if (!gen.control()->is_void(functionType->returnType())) {
    resultTypes.push_back(exprType);
  }

  auto funcType = gen.builder_.getFunctionType(inputTypes, resultTypes);
  auto loc = gen.builder_.getUnknownLoc();

  std::vector<std::string> path;
  for (Symbol* symbol = ast->symbol; symbol;
       symbol = symbol->enclosingSymbol()) {
    if (!symbol->name()) continue;
    path.push_back(to_string(symbol->name()));
  }

  std::string name;

  if (ast->symbol->hasCLinkage()) {
    name = to_string(ast->symbol->name());
  } else {
    // todo: external name mangling

    std::ranges::for_each(path | std::views::reverse, [&](auto& part) {
      name += "::";
      name += part;
    });

    // generate unique names until we have proper name mangling
    name += std::format("_{}", ++gen.count_);
  }

  auto savedInsertionPoint = gen.builder_.saveInsertionPoint();

  auto func = gen.builder_.create<mlir::cxx::FuncOp>(loc, name, funcType);

  auto entryBlock = &func.front();

  gen.builder_.setInsertionPointToEnd(entryBlock);

  std::swap(gen.function_, func);

#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto requiresClauseResult = gen(ast->requiresClause);
#endif

  auto functionBodyResult = gen(ast->functionBody);

  std::swap(gen.function_, func);

  auto endLoc = gen.getLocation(ast->lastSourceLocation());

  if (gen.control()->is_void(functionType->returnType())) {
    // If the function returns void, we don't need to return anything.
    gen.builder_.create<mlir::cxx::ReturnOp>(endLoc);
  } else {
    // Otherwise, we need to return a value of the correct type.
    auto r = gen.emitTodoExpr(ast->lastSourceLocation(), "result value");
    auto result =
        gen.builder_.create<mlir::cxx::ReturnOp>(endLoc, r->getResults());
  }

  gen.builder_.restoreInsertionPoint(savedInsertionPoint);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  auto explicitSpecifierResult = gen(ast->explicitSpecifier);
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);
  auto templateIdResult = gen(ast->templateId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportCompoundDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ModuleImportDeclarationAST* ast)
    -> DeclarationResult {
  auto importNameResult = gen(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = gen(node);
  }

  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmQualifierAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationResult {
  return {};
}

void Codegen::StatementVisitor::operator()(LabeledStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(CaseStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(DefaultStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ExpressionStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(CompoundStatementAST* ast) {
  for (auto node : ListView{ast->statementList}) {
    gen.statement(node);
  }
}

void Codegen::StatementVisitor::operator()(IfStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
  gen.statement(ast->elseStatement);
#endif
}

void Codegen::StatementVisitor::operator()(ConstevalIfStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
  gen.statement(ast->elseStatement);
#endif
}

void Codegen::StatementVisitor::operator()(SwitchStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(WhileStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(DoStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(ForRangeStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto rangeDeclarationResult = gen(ast->rangeDeclaration);
  auto rangeInitializerResult = gen.expression(ast->rangeInitializer);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(ForStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  auto expressionResult = gen.expression(ast->expression);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(BreakStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ContinueStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(CoroutineReturnStatementAST* ast) {
  auto op = gen.emitTodoStmt(ast->firstSourceLocation(),
                             "CoroutineReturnStatementAST");

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(GotoStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(DeclarationStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto declarationResult = gen(ast->declaration);
#endif
}

void Codegen::StatementVisitor::operator()(TryBlockStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = gen(node);
  }
#endif
}

auto Codegen::ExpressionVisitor::operator()(GeneratedLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "GeneratedLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "CharLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "BoolLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto op = gen.builder_.create<mlir::cxx::IntLiteralOp>(
      loc, ast->literal->integerValue());

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "FloatLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NullptrLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "NullptrLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "StringLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "UserDefinedStringLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "ObjectLiteralExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "ThisExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(GenericSelectionExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "GenericSelectionExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NestedStatementExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "NestedStatementExpressionAST");
  gen.statement(ast->statement);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  return gen.expression(ast->expression);
}

auto Codegen::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  if (auto id = ast_cast<NameIdAST>(ast->unqualifiedId);
      id && !ast->nestedNameSpecifier) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto name = id->identifier->name();
    auto op = gen.builder_.create<mlir::cxx::IdOp>(loc, name);
    return {op};
  }

  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "IdExpressionAST");
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "LambdaExpressionAST");

  for (auto node : ListView{ast->captureList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto templateRequiresClauseResult = gen(ast->templateRequiresClause);
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = gen(node);
  }

  auto exceptionSpecifierResult = gen(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto trailingReturnTypeResult = gen(ast->trailingReturnType);
  auto requiresClauseResult = gen(ast->requiresClause);
  gen.statement(ast->statement);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "FoldExpressionAST");
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "RightFoldExpressionAST");
  auto expressionResult = gen.expression(ast->expression);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "LeftFoldExpressionAST");
  auto expressionResult = gen.expression(ast->expression);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "RequiresExpressionAST");

  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->requirementList}) {
    auto value = gen(node);
  }

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "VaArgExpressionAST");

  auto expressionResult = gen.expression(ast->expression);
  auto typeIdResult = gen(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "SubscriptExpressionAST");

  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto indexExpressionResult = gen.expression(ast->indexExpression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = gen.expression(ast->baseExpression);

  std::vector<mlir::Value> arguments;

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  auto loc = gen.getLocation(ast->lparenLoc);

  auto op = gen.builder_.create<mlir::cxx::CallOp>(
      loc, baseExpressionResult.value, arguments);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "TypeConstructionAST");

  auto typeSpecifierResult = gen(ast->typeSpecifier);

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "BracedTypeConstructionAST");

  auto typeSpecifierResult = gen(ast->typeSpecifier);
  auto bracedInitListResult = gen.expression(ast->bracedInitList);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "SpliceMemberExpressionAST");

  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto splicerResult = gen(ast->splicer);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "MemberExpressionAST");

  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "PostIncrExpressionAST");

  auto baseExpressionResult = gen.expression(ast->baseExpression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "CppCastExpressionAST");

  auto typeIdResult = gen(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BuiltinBitCastExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "BuiltinBitCastExpressionAST");

  auto typeIdResult = gen(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BuiltinOffsetofExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "BuiltinOffsetofExpressionAST");

  auto typeIdResult = gen(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "TypeidExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "TypeidOfTypeExpressionAST");

  auto typeIdResult = gen(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "SpliceExpressionAST");

  auto splicerResult = gen(ast->splicer);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "GlobalScopeReflectExpressionAST");

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NamespaceReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "NamespaceReflectExpressionAST");

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeIdReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "TypeIdReflectExpressionAST");

  auto typeIdResult = gen(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "ReflectExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "LabelAddressExpressionAST");

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "UnaryExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "AwaitExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "SizeofExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "SizeofTypeExpressionAST");

  auto typeIdResult = gen(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "SizeofPackExpressionAST");

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "AlignofTypeExpressionAST");

  auto typeIdResult = gen(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "AlignofExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "NoexceptExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "NewExpressionAST");

  auto newPlacementResult = gen(ast->newPlacement);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto newInitalizerResult = gen(ast->newInitalizer);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "DeleteExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "CastExpressionAST");

  auto typeIdResult = gen(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto op = gen.builder_.create<mlir::cxx::ImplicitCastOp>(
      loc, to_string(ast->castKind), expressionResult.value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);

  auto loc = gen.getLocation(ast->opLoc);

  auto operation = Token::spell(ast->op);

  auto op = gen.builder_.create<mlir::cxx::BinOp>(
      loc, operation, leftExpressionResult.value, rightExpressionResult.value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "ConditionalExpressionAST");

  auto conditionResult = gen.expression(ast->condition);
  auto iftrueExpressionResult = gen.expression(ast->iftrueExpression);
  auto iffalseExpressionResult = gen.expression(ast->iffalseExpression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "YieldExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "ThrowExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "AssignmentExpressionAST");

  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "PackExpansionExpressionAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(DesignatedInitializerClauseAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                             "DesignatedInitializerClauseAST");

  auto initializerResult = gen.expression(ast->initializer);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "TypeTraitExpressionAST");

  for (auto node : ListView{ast->typeIdList}) {
    auto value = gen(node);
  }

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), "ConditionExpressionAST");

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto initializerResult = gen.expression(ast->initializer);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "EqualInitializerAST");

  auto expressionResult = gen.expression(ast->expression);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "BracedInitListAST");

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  auto op = gen.emitTodoExpr(ast->firstSourceLocation(), "ParenInitializerAST");

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {op};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
  auto idExpressionResult = gen.expression(ast->idExpression);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = gen(ast->typeConstraint);
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(GeneratedTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(NoreturnSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(RegisterSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(FloatingPointTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(UnderlyingTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ElaboratedTypeSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(PlaceholderTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeConstraintResult = gen(ast->typeConstraint);
  auto specifierResult = gen(ast->specifier);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(AtomicQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->enumeratorList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->baseSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto splicerResult = gen(ast->splicer);

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto unqualifiedIdResult = gen(ast->unqualifiedId);
  auto sizeExpressionResult = gen.expression(ast->sizeExpression);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = gen(ast->coreDeclarator);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = gen(ast->declarator);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  auto exceptionSpecifierResult = gen(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto trailingReturnTypeResult = gen(ast->trailingReturnType);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(ArrayDeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  auto expressionResult = gen.expression(ast->expression);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdResult {
  auto idResult = gen(ast->id);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdResult {
  auto decltypeSpecifierResult = gen(ast->decltypeSpecifier);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdResult {
  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto literalOperatorIdResult = gen(ast->literalOperatorId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto operatorFunctionIdResult = gen(ast->operatorFunctionId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto decltypeSpecifierResult = gen(ast->decltypeSpecifier);

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto templateIdResult = gen(ast->templateId);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
#if false
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen(node);
  }
#endif

  gen.statement(ast->statement);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(TryStatementFunctionBodyAST* ast)
    -> FunctionBodyResult {
#if false
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen(node);
  }

#endif

  gen.statement(ast->statement);

#if false
  for (auto node : ListView{ast->handlerList}) {
    auto value = gen(node);
  }
#endif

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto Codegen::TemplateArgumentVisitor::operator()(TypeTemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierResult {
  return {};
}

auto Codegen::ExceptionSpecifierVisitor::operator()(NoexceptSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);
  auto typeConstraintResult = gen(ast->typeConstraint);

  return {};
}

auto Codegen::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {};
}

auto Codegen::NewInitializerVisitor::operator()(NewBracedInitializerAST* ast)
    -> NewInitializerResult {
  auto bracedInitListResult = gen.expression(ast->bracedInitList);

  return {};
}

auto Codegen::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {};
}

auto Codegen::MemInitializerVisitor::operator()(BracedMemInitializerAST* ast)
    -> MemInitializerResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);
  auto bracedInitListResult = gen.expression(ast->bracedInitList);

  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(DerefThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto Codegen::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto attributeUsingPrefixResult = gen(ast->attributeUsingPrefix);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(AlignasAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto Codegen::AttributeTokenVisitor::operator()(ScopedAttributeTokenAST* ast)
    -> AttributeTokenResult {
  return {};
}

auto Codegen::AttributeTokenVisitor::operator()(SimpleAttributeTokenAST* ast)
    -> AttributeTokenResult {
  return {};
}

}  // namespace cxx
