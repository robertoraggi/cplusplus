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
#include <cxx/ast_cursor.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

ASTRewriter::ASTRewriter(TypeChecker* typeChcker,
                         const std::vector<TemplateArgument>& templateArguments)
    : typeChecker_(typeChcker),
      unit_(typeChcker->translationUnit()),
      templateArguments_(templateArguments),
      binder_(typeChcker->translationUnit()) {}

ASTRewriter::~ASTRewriter() {}

auto ASTRewriter::control() const -> Control* { return unit_->control(); }

auto ASTRewriter::arena() const -> Arena* { return unit_->arena(); }

auto ASTRewriter::restrictedToDeclarations() const -> bool {
  return restrictedToDeclarations_;
}

void ASTRewriter::setRestrictedToDeclarations(bool restrictedToDeclarations) {
  restrictedToDeclarations_ = restrictedToDeclarations;
}

auto ASTRewriter::getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol* {
  for (auto cursor = ASTCursor{ast, {}}; cursor; ++cursor) {
    const auto& current = *cursor;
    if (!std::holds_alternative<AST*>(current.node)) continue;

    auto id = ast_cast<IdExpressionAST>(std::get<AST*>(current.node));
    if (!id) continue;

    auto param = symbol_cast<NonTypeParameterSymbol>(id->symbol);
    if (!param) continue;

    if (param->depth() != 0) continue;

    auto arg = templateArguments_[param->index()];
    auto argSymbol = std::get<Symbol*>(arg);

    auto parameterPack = symbol_cast<ParameterPackSymbol>(argSymbol);
    if (parameterPack) return parameterPack;
  }

  return nullptr;
}

struct ASTRewriter::UnitVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitAST*;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitAST*;
};

struct ASTRewriter::DeclarationVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST* ast)
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

struct ASTRewriter::DesignatorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(DotDesignatorAST* ast) -> DesignatorAST*;

  [[nodiscard]] auto operator()(SubscriptDesignatorAST* ast) -> DesignatorAST*;
};

struct ASTRewriter::TemplateParameterVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(GeneratedTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(InlineSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(NoreturnSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(StaticSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ExternSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(RegisterSpecifierAST* ast) -> SpecifierAST*;

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

  [[nodiscard]] auto operator()(AtomicQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast) -> SpecifierAST*;
};

struct ASTRewriter::PtrOperatorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast) -> PtrOperatorAST*;
};

struct ASTRewriter::CoreDeclaratorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast) -> CoreDeclaratorAST*;
};

struct ASTRewriter::DeclaratorChunkVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;
};

struct ASTRewriter::UnqualifiedIdVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
};

struct ASTRewriter::ExceptionSpecifierVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
};

struct ASTRewriter::RequirementVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementAST*;
};

struct ASTRewriter::NewInitializerVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerAST*;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerAST*;
};

struct ASTRewriter::MemInitializerVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerAST*;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerAST*;
};

struct ASTRewriter::LambdaCaptureVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
};

struct ASTRewriter::AttributeSpecifierVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
      -> AttributeTokenAST*;

  [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
      -> AttributeTokenAST*;
};

auto ASTRewriter::operator()(UnitAST* ast) -> UnitAST* {
  if (!ast) return {};
  return visit(UnitVisitor{*this}, ast);
}

auto ASTRewriter::operator()(DeclarationAST* ast) -> DeclarationAST* {
  if (!ast) return {};
  return visit(DeclarationVisitor{*this}, ast);
}

auto ASTRewriter::operator()(StatementAST* ast) -> StatementAST* {
  if (!ast) return {};
  return visit(StatementVisitor{*this}, ast);
}

auto ASTRewriter::operator()(ExpressionAST* ast) -> ExpressionAST* {
  if (!ast) return {};
  auto expr = visit(ExpressionVisitor{*this}, ast);
  if (expr) typeChecker_->check(expr);
  return expr;
}

auto ASTRewriter::operator()(DesignatorAST* ast) -> DesignatorAST* {
  if (!ast) return {};
  return visit(DesignatorVisitor{*this}, ast);
}

auto ASTRewriter::operator()(TemplateParameterAST* ast)
    -> TemplateParameterAST* {
  if (!ast) return {};
  return visit(TemplateParameterVisitor{*this}, ast);
}

auto ASTRewriter::operator()(SpecifierAST* ast) -> SpecifierAST* {
  if (!ast) return {};
  auto specifier = visit(SpecifierVisitor{*this}, ast);
  return specifier;
}

auto ASTRewriter::operator()(PtrOperatorAST* ast) -> PtrOperatorAST* {
  if (!ast) return {};
  return visit(PtrOperatorVisitor{*this}, ast);
}

auto ASTRewriter::operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorAST* {
  if (!ast) return {};
  return visit(CoreDeclaratorVisitor{*this}, ast);
}

auto ASTRewriter::operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  if (!ast) return {};
  return visit(DeclaratorChunkVisitor{*this}, ast);
}

auto ASTRewriter::operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdAST* {
  if (!ast) return {};
  return visit(UnqualifiedIdVisitor{*this}, ast);
}

auto ASTRewriter::operator()(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierAST* {
  if (!ast) return {};
  return visit(NestedNameSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::operator()(FunctionBodyAST* ast) -> FunctionBodyAST* {
  if (!ast) return {};
  return visit(FunctionBodyVisitor{*this}, ast);
}

auto ASTRewriter::operator()(TemplateArgumentAST* ast) -> TemplateArgumentAST* {
  if (!ast) return {};
  return visit(TemplateArgumentVisitor{*this}, ast);
}

auto ASTRewriter::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierAST* {
  if (!ast) return {};
  return visit(ExceptionSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::operator()(RequirementAST* ast) -> RequirementAST* {
  if (!ast) return {};
  return visit(RequirementVisitor{*this}, ast);
}

auto ASTRewriter::operator()(NewInitializerAST* ast) -> NewInitializerAST* {
  if (!ast) return {};
  return visit(NewInitializerVisitor{*this}, ast);
}

auto ASTRewriter::operator()(MemInitializerAST* ast) -> MemInitializerAST* {
  if (!ast) return {};
  return visit(MemInitializerVisitor{*this}, ast);
}

auto ASTRewriter::operator()(LambdaCaptureAST* ast) -> LambdaCaptureAST* {
  if (!ast) return {};
  return visit(LambdaCaptureVisitor{*this}, ast);
}

auto ASTRewriter::operator()(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationAST* {
  if (!ast) return {};
  return visit(ExceptionDeclarationVisitor{*this}, ast);
}

auto ASTRewriter::operator()(AttributeSpecifierAST* ast)
    -> AttributeSpecifierAST* {
  if (!ast) return {};
  return visit(AttributeSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::operator()(AttributeTokenAST* ast) -> AttributeTokenAST* {
  if (!ast) return {};
  return visit(AttributeTokenVisitor{*this}, ast);
}

auto ASTRewriter::operator()(SplicerAST* ast) -> SplicerAST* {
  if (!ast) return {};

  auto copy = make_node<SplicerAST>(arena());

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

  auto copy = make_node<GlobalModuleFragmentAST>(arena());

  copy->moduleLoc = ast->moduleLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentAST* {
  if (!ast) return {};

  auto copy = make_node<PrivateModuleFragmentAST>(arena());

  copy->moduleLoc = ast->moduleLoc;
  copy->colonLoc = ast->colonLoc;
  copy->privateLoc = ast->privateLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::operator()(ModuleDeclarationAST* ast)
    -> ModuleDeclarationAST* {
  if (!ast) return {};

  auto copy = make_node<ModuleDeclarationAST>(arena());

  copy->exportLoc = ast->exportLoc;
  copy->moduleLoc = ast->moduleLoc;
  copy->moduleName = operator()(ast->moduleName);
  copy->modulePartition = operator()(ast->modulePartition);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::operator()(ModuleNameAST* ast) -> ModuleNameAST* {
  if (!ast) return {};

  auto copy = make_node<ModuleNameAST>(arena());

  copy->moduleQualifier = operator()(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(ModuleQualifierAST* ast) -> ModuleQualifierAST* {
  if (!ast) return {};

  auto copy = make_node<ModuleQualifierAST>(arena());

  copy->moduleQualifier = operator()(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->dotLoc = ast->dotLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(ModulePartitionAST* ast) -> ModulePartitionAST* {
  if (!ast) return {};

  auto copy = make_node<ModulePartitionAST>(arena());

  copy->colonLoc = ast->colonLoc;
  copy->moduleName = operator()(ast->moduleName);

  return copy;
}

auto ASTRewriter::operator()(ImportNameAST* ast) -> ImportNameAST* {
  if (!ast) return {};

  auto copy = make_node<ImportNameAST>(arena());

  copy->headerLoc = ast->headerLoc;
  copy->modulePartition = operator()(ast->modulePartition);
  copy->moduleName = operator()(ast->moduleName);

  return copy;
}

auto ASTRewriter::operator()(InitDeclaratorAST* ast, const DeclSpecs& declSpecs)
    -> InitDeclaratorAST* {
  if (!ast) return {};

  auto copy = make_node<InitDeclaratorAST>(arena());

  copy->declarator = operator()(ast->declarator);

  auto decl = Decl{declSpecs, copy->declarator};

  auto type =
      getDeclaratorType(translationUnit(), copy->declarator, declSpecs.type());

  // ### fix scope
  if (binder_.scope() && binder_.scope()->isClassScope()) {
    auto symbol = binder_.declareMemberSymbol(ast->declarator, decl);
    copy->symbol = symbol;
  } else {
    // ### TODO
    copy->symbol = ast->symbol;
  }

  copy->requiresClause = operator()(ast->requiresClause);
  copy->initializer = operator()(ast->initializer);
  // copy->symbol = ast->symbol; // TODO remove, done above

  return copy;
}

auto ASTRewriter::operator()(DeclaratorAST* ast) -> DeclaratorAST* {
  if (!ast) return {};

  auto copy = make_node<DeclaratorAST>(arena());

  for (auto ptrOpList = &copy->ptrOpList;
       auto node : ListView{ast->ptrOpList}) {
    auto value = operator()(node);
    *ptrOpList = make_list_node(arena(), value);
    ptrOpList = &(*ptrOpList)->next;
  }

  copy->coreDeclarator = operator()(ast->coreDeclarator);

  for (auto declaratorChunkList = &copy->declaratorChunkList;
       auto node : ListView{ast->declaratorChunkList}) {
    auto value = operator()(node);
    *declaratorChunkList = make_list_node(arena(), value);
    declaratorChunkList = &(*declaratorChunkList)->next;
  }

  return copy;
}

auto ASTRewriter::operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorAST* {
  if (!ast) return {};

  auto copy = make_node<UsingDeclaratorAST>(arena());

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

  auto copy = make_node<EnumeratorAST>(arena());

  copy->identifierLoc = ast->identifierLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->equalLoc = ast->equalLoc;
  copy->expression = operator()(ast->expression);
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::operator()(TypeIdAST* ast) -> TypeIdAST* {
  if (!ast) return {};

  auto copy = make_node<TypeIdAST>(arena());

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = operator()(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = operator()(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->type = declaratorType;

  return copy;
}

auto ASTRewriter::operator()(HandlerAST* ast) -> HandlerAST* {
  if (!ast) return {};

  auto copy = make_node<HandlerAST>(arena());

  copy->catchLoc = ast->catchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->exceptionDeclaration = operator()(ast->exceptionDeclaration);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(operator()(ast->statement));

  return copy;
}

auto ASTRewriter::operator()(BaseSpecifierAST* ast) -> BaseSpecifierAST* {
  if (!ast) return {};

  auto copy = make_node<BaseSpecifierAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->virtualOrAccessLoc = ast->virtualOrAccessLoc;
  copy->otherVirtualOrAccessLoc = ast->otherVirtualOrAccessLoc;
  copy->nestedNameSpecifier = operator()(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = operator()(ast->unqualifiedId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
  copy->isVirtual = ast->isVirtual;
  copy->isVariadic = ast->isVariadic;
  copy->accessSpecifier = ast->accessSpecifier;
  copy->symbol = ast->symbol;

  binder_.bind(ast);

  return copy;
}

auto ASTRewriter::operator()(RequiresClauseAST* ast) -> RequiresClauseAST* {
  if (!ast) return {};

  auto copy = make_node<RequiresClauseAST>(arena());

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = operator()(ast->expression);

  return copy;
}

auto ASTRewriter::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseAST* {
  if (!ast) return {};

  auto copy = make_node<ParameterDeclarationClauseAST>(arena());

  for (auto parameterDeclarationList = &copy->parameterDeclarationList;
       auto node : ListView{ast->parameterDeclarationList}) {
    auto value = operator()(node);
    *parameterDeclarationList =
        make_list_node(arena(), ast_cast<ParameterDeclarationAST>(value));
    parameterDeclarationList = &(*parameterDeclarationList)->next;
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

  auto copy = make_node<TrailingReturnTypeAST>(arena());

  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeId = operator()(ast->typeId);

  return copy;
}

auto ASTRewriter::operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierAST* {
  if (!ast) return {};

  auto copy = make_node<LambdaSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::operator()(TypeConstraintAST* ast) -> TypeConstraintAST* {
  if (!ast) return {};

  auto copy = make_node<TypeConstraintAST>(arena());

  copy->nestedNameSpecifier = operator()(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = operator()(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::operator()(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseAST* {
  if (!ast) return {};

  auto copy = make_node<AttributeArgumentClauseAST>(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::operator()(AttributeAST* ast) -> AttributeAST* {
  if (!ast) return {};

  auto copy = make_node<AttributeAST>(arena());

  copy->attributeToken = operator()(ast->attributeToken);
  copy->attributeArgumentClause = operator()(ast->attributeArgumentClause);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::operator()(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixAST* {
  if (!ast) return {};

  auto copy = make_node<AttributeUsingPrefixAST>(arena());

  copy->usingLoc = ast->usingLoc;
  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::operator()(NewPlacementAST* ast) -> NewPlacementAST* {
  if (!ast) return {};

  auto copy = make_node<NewPlacementAST>(arena());

  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = operator()(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierAST* {
  if (!ast) return {};

  auto copy = make_node<NestedNamespaceSpecifierAST>(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitAST* {
  auto copy = make_node<TranslationUnitAST>(arena());

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitAST* {
  auto copy = make_node<ModuleUnitAST>(arena());

  copy->globalModuleFragment = rewrite(ast->globalModuleFragment);
  copy->moduleDeclaration = rewrite(ast->moduleDeclaration);

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->privateModuleFragment = rewrite(ast->privateModuleFragment);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<SimpleDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  for (auto initDeclaratorList = &copy->initDeclaratorList;
       auto node : ListView{ast->initDeclaratorList}) {
    auto value = rewrite(node, declSpecifierListCtx);
    *initDeclaratorList = make_list_node(arena(), value);
    initDeclaratorList = &(*initDeclaratorList)->next;
  }

  copy->requiresClause = rewrite(ast->requiresClause);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AsmDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  for (auto asmQualifierList = &copy->asmQualifierList;
       auto node : ListView{ast->asmQualifierList}) {
    auto value = rewrite(node);
    *asmQualifierList =
        make_list_node(arena(), ast_cast<AsmQualifierAST>(value));
    asmQualifierList = &(*asmQualifierList)->next;
  }

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;

  for (auto outputOperandList = &copy->outputOperandList;
       auto node : ListView{ast->outputOperandList}) {
    auto value = rewrite(node);
    *outputOperandList =
        make_list_node(arena(), ast_cast<AsmOperandAST>(value));
    outputOperandList = &(*outputOperandList)->next;
  }

  for (auto inputOperandList = &copy->inputOperandList;
       auto node : ListView{ast->inputOperandList}) {
    auto value = rewrite(node);
    *inputOperandList = make_list_node(arena(), ast_cast<AsmOperandAST>(value));
    inputOperandList = &(*inputOperandList)->next;
  }

  for (auto clobberList = &copy->clobberList;
       auto node : ListView{ast->clobberList}) {
    auto value = rewrite(node);
    *clobberList = make_list_node(arena(), ast_cast<AsmClobberAST>(value));
    clobberList = &(*clobberList)->next;
  }

  for (auto gotoLabelList = &copy->gotoLabelList;
       auto node : ListView{ast->gotoLabelList}) {
    auto value = rewrite(node);
    *gotoLabelList = make_list_node(arena(), ast_cast<AsmGotoLabelAST>(value));
    gotoLabelList = &(*gotoLabelList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationAST* {
  auto copy = make_node<NamespaceAliasDefinitionAST>(arena());

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
  auto copy = make_node<UsingDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;

  for (auto usingDeclaratorList = &copy->usingDeclaratorList;
       auto node : ListView{ast->usingDeclaratorList}) {
    auto value = rewrite(node);
    *usingDeclaratorList = make_list_node(arena(), value);
    usingDeclaratorList = &(*usingDeclaratorList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<UsingEnumDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;
  copy->enumTypeSpecifier =
      ast_cast<ElaboratedTypeSpecifierAST>(rewrite(ast->enumTypeSpecifier));
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<UsingDirectiveAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
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
  auto copy = make_node<StaticAssertDeclarationAST>(arena());

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
  auto copy = make_node<AliasDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;
  copy->identifierLoc = ast->identifierLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->equalLoc = ast->equalLoc;

  for (auto gnuAttributeList = &copy->gnuAttributeList;
       auto node : ListView{ast->gnuAttributeList}) {
    auto value = rewrite(node);
    *gnuAttributeList = make_list_node(arena(), value);
    gnuAttributeList = &(*gnuAttributeList)->next;
  }

  copy->typeId = rewrite(ast->typeId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<OpaqueEnumDeclarationAST>(arena());

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->emicolonLoc = ast->emicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<FunctionDefinitionAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->declarator = rewrite(ast->declarator);

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->functionBody = rewrite(ast->functionBody);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<TemplateDeclarationAST>(arena());

  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->requiresClause = rewrite(ast->requiresClause);
  copy->declaration = rewrite(ast->declaration);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ConceptDefinitionAST>(arena());

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
  auto copy = make_node<DeductionGuideAST>(arena());

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
  auto copy = make_node<ExplicitInstantiationAST>(arena());

  copy->externLoc = ast->externLoc;
  copy->templateLoc = ast->templateLoc;
  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ExportDeclarationAST>(arena());

  copy->exportLoc = ast->exportLoc;
  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<ExportCompoundDeclarationAST>(arena());

  copy->exportLoc = ast->exportLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<LinkageSpecificationAST>(arena());

  copy->externLoc = ast->externLoc;
  copy->stringliteralLoc = ast->stringliteralLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->stringLiteral = ast->stringLiteral;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<NamespaceDefinitionAST>(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->namespaceLoc = ast->namespaceLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  for (auto nestedNamespaceSpecifierList = &copy->nestedNamespaceSpecifierList;
       auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = rewrite(node);
    *nestedNamespaceSpecifierList = make_list_node(arena(), value);
    nestedNamespaceSpecifierList = &(*nestedNamespaceSpecifierList)->next;
  }

  copy->identifierLoc = ast->identifierLoc;

  for (auto extraAttributeList = &copy->extraAttributeList;
       auto node : ListView{ast->extraAttributeList}) {
    auto value = rewrite(node);
    *extraAttributeList = make_list_node(arena(), value);
    extraAttributeList = &(*extraAttributeList)->next;
  }

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<EmptyDeclarationAST>(arena());

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AttributeDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<ModuleImportDeclarationAST>(arena());

  copy->importLoc = ast->importLoc;
  copy->importName = rewrite(ast->importName);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ParameterDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->thisLoc = ast->thisLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = rewrite(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->type = declaratorType;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite(ast->expression);
  copy->identifier = ast->identifier;
  copy->isThisIntroduced = ast->isThisIntroduced;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AccessDeclarationAST>(arena());

  copy->accessLoc = ast->accessLoc;
  copy->colonLoc = ast->colonLoc;
  copy->accessSpecifier = ast->accessSpecifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ForRangeDeclarationAST>(arena());

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<StructuredBindingDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->refQualifierLoc = ast->refQualifierLoc;
  copy->lbracketLoc = ast->lbracketLoc;

  for (auto bindingList = &copy->bindingList;
       auto node : ListView{ast->bindingList}) {
    auto value = rewrite(node);
    *bindingList = make_list_node(arena(), ast_cast<NameIdAST>(value));
    bindingList = &(*bindingList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AsmOperandAST>(arena());

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
  auto copy = make_node<AsmQualifierAST>(arena());

  copy->qualifierLoc = ast->qualifierLoc;
  copy->qualifier = ast->qualifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AsmClobberAST>(arena());

  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AsmGotoLabelAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<LabeledStatementAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->colonLoc = ast->colonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CaseStatementAST>(arena());

  copy->caseLoc = ast->caseLoc;
  copy->expression = rewrite(ast->expression);
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DefaultStatementAST>(arena());

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ExpressionStatementAST>(arena());

  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CompoundStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto statementList = &copy->statementList;
       auto node : ListView{ast->statementList}) {
    auto value = rewrite(node);
    *statementList = make_list_node(arena(), value);
    statementList = &(*statementList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<IfStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

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
  auto copy = make_node<ConstevalIfStatementAST>(arena());

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
  auto copy = make_node<SwitchStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

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
  auto copy = make_node<WhileStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->condition = rewrite(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DoStatementAST>(arena());

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
  auto copy = make_node<ForRangeStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

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
  auto copy = make_node<ForStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

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
  auto copy = make_node<BreakStatementAST>(arena());

  copy->breakLoc = ast->breakLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ContinueStatementAST>(arena());

  copy->continueLoc = ast->continueLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ReturnStatementAST>(arena());

  copy->returnLoc = ast->returnLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CoroutineReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CoroutineReturnStatementAST>(arena());

  copy->coreturnLoc = ast->coreturnLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<GotoStatementAST>(arena());

  copy->gotoLoc = ast->gotoLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DeclarationStatementAST>(arena());

  copy->declaration = rewrite(ast->declaration);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<TryBlockStatementAST>(arena());

  copy->tryLoc = ast->tryLoc;
  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  for (auto handlerList = &copy->handlerList;
       auto node : ListView{ast->handlerList}) {
    auto value = rewrite(node);
    *handlerList = make_list_node(arena(), value);
    handlerList = &(*handlerList)->next;
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GeneratedLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<GeneratedLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->value = ast->value;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<CharLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<BoolLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->isTrue = ast->isTrue;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<IntLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<FloatLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<NullptrLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<StringLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<UserDefinedStringLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ObjectLiteralExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ThisExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<NestedStatementExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<NestedExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionAST* {
  if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol);
      param && param->depth() == 0 &&
      param->index() < rewrite.templateArguments_.size()) {
    auto symbolPtr =
        std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

    if (!symbolPtr) {
      cxx_runtime_error("expected initializer for non-type template parameter");
    }

    auto parameterPack = symbol_cast<ParameterPackSymbol>(*symbolPtr);

    if (parameterPack && parameterPack == rewrite.parameterPack_ &&
        rewrite.elementIndex_.has_value()) {
      auto idx = rewrite.elementIndex_.value();
      auto element = parameterPack->elements()[idx];
      if (auto var = symbol_cast<VariableSymbol>(element)) {
        return rewrite(var->initializer());
      }
    }
  }

  auto copy = make_node<IdExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->symbol = ast->symbol;

  if (auto param = symbol_cast<NonTypeParameterSymbol>(copy->symbol);
      param && param->depth() == 0 &&
      param->index() < rewrite.templateArguments_.size()) {
    auto symbolPtr =
        std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

    if (!symbolPtr) {
      cxx_runtime_error("expected initializer for non-type template parameter");
    }

    copy->symbol = *symbolPtr;
    copy->type = copy->symbol->type();
  }
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<LambdaExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->captureDefaultLoc = ast->captureDefaultLoc;

  for (auto captureList = &copy->captureList;
       auto node : ListView{ast->captureList}) {
    auto value = rewrite(node);
    *captureList = make_list_node(arena(), value);
    captureList = &(*captureList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->templateRequiresClause = rewrite(ast->templateRequiresClause);
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  for (auto gnuAtributeList = &copy->gnuAtributeList;
       auto node : ListView{ast->gnuAtributeList}) {
    auto value = rewrite(node);
    *gnuAtributeList = make_list_node(arena(), value);
    gnuAtributeList = &(*gnuAtributeList)->next;
  }

  for (auto lambdaSpecifierList = &copy->lambdaSpecifierList;
       auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = rewrite(node);
    *lambdaSpecifierList = make_list_node(arena(), value);
    lambdaSpecifierList = &(*lambdaSpecifierList)->next;
  }

  {
    auto _ = Binder::ScopeGuard(binder());

    if (copy->parameterDeclarationClause) {
      binder()->setScope(
          copy->parameterDeclarationClause->functionParametersSymbol);
    }

    copy->exceptionSpecifier = rewrite(ast->exceptionSpecifier);

    for (auto attributeList = &copy->attributeList;
         auto node : ListView{ast->attributeList}) {
      auto value = rewrite(node);
      *attributeList = make_list_node(arena(), value);
      attributeList = &(*attributeList)->next;
    }

    copy->trailingReturnType = rewrite(ast->trailingReturnType);
    copy->requiresClause = rewrite(ast->requiresClause);
  }

  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));
  copy->captureDefault = ast->captureDefault;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<FoldExpressionAST>(arena());

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
  auto copy = make_node<RightFoldExpressionAST>(arena());

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
  if (auto parameterPack = rewrite.getParameterPack(ast->expression)) {
    auto savedParameterPack = rewrite.parameterPack_;
    std::swap(rewrite.parameterPack_, parameterPack);

    std::vector<ExpressionAST*> instantiations;
    ExpressionAST* current = nullptr;

    int n = 0;
    for (auto element : rewrite.parameterPack_->elements()) {
      std::optional<int> index{n};
      std::swap(rewrite.elementIndex_, index);

      auto expression = rewrite(ast->expression);
      if (!current) {
        current = expression;
      } else {
        auto binop = make_node<BinaryExpressionAST>(arena());
        binop->valueCategory = current->valueCategory;
        binop->type = current->type;
        binop->leftExpression = current;
        binop->op = ast->op;
        binop->opLoc = ast->opLoc;
        binop->rightExpression = expression;
        current = binop;
      }

      std::swap(rewrite.elementIndex_, index);
      ++n;
    }

    std::swap(rewrite.parameterPack_, parameterPack);

    return current;
  }

  auto copy = make_node<LeftFoldExpressionAST>(arena());

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
  auto copy = make_node<RequiresExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->requiresLoc = ast->requiresLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto requirementList = &copy->requirementList;
       auto node : ListView{ast->requirementList}) {
    auto value = rewrite(node);
    *requirementList = make_list_node(arena(), value);
    requirementList = &(*requirementList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<VaArgExpressionAST>(arena());

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
  auto copy = make_node<SubscriptExpressionAST>(arena());

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
  auto copy = make_node<CallExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<TypeConstructionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite(ast->typeSpecifier);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<BracedTypeConstructionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite(ast->typeSpecifier);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<SpliceMemberExpressionAST>(arena());

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
  auto copy = make_node<MemberExpressionAST>(arena());

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
  auto copy = make_node<PostIncrExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite(ast->baseExpression);
  copy->opLoc = ast->opLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<CppCastExpressionAST>(arena());

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
  auto copy = make_node<BuiltinBitCastExpressionAST>(arena());

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
  auto copy = make_node<BuiltinOffsetofExpressionAST>(arena());

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
  auto copy = make_node<TypeidExpressionAST>(arena());

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
  auto copy = make_node<TypeidOfTypeExpressionAST>(arena());

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
  auto copy = make_node<SpliceExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->splicer = rewrite(ast->splicer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<GlobalScopeReflectExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = make_node<NamespaceReflectExpressionAST>(arena());

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
  auto copy = make_node<TypeIdReflectExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ReflectExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretLoc = ast->caretLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<UnaryExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite(ast->expression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<AwaitExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->awaitLoc = ast->awaitLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<SizeofExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<SizeofTypeExpressionAST>(arena());

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
  auto copy = make_node<SizeofPackExpressionAST>(arena());

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
  auto copy = make_node<AlignofTypeExpressionAST>(arena());

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
  auto copy = make_node<AlignofExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<NoexceptExpressionAST>(arena());

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
  auto copy = make_node<NewExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->newLoc = ast->newLoc;
  copy->newPlacement = rewrite(ast->newPlacement);
  copy->lparenLoc = ast->lparenLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = rewrite(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->rparenLoc = ast->rparenLoc;
  copy->newInitalizer = rewrite(ast->newInitalizer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<DeleteExpressionAST>(arena());

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
  auto copy = make_node<CastExpressionAST>(arena());

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
  auto copy = make_node<ImplicitCastExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite(ast->expression);
  copy->castKind = ast->castKind;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<BinaryExpressionAST>(arena());

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
  auto copy = make_node<ConditionalExpressionAST>(arena());

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
  auto copy = make_node<YieldExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->yieldLoc = ast->yieldLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ThrowExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->throwLoc = ast->throwLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<AssignmentExpressionAST>(arena());

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
  auto copy = make_node<PackExpansionExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionAST* {
  auto copy = make_node<DesignatedInitializerClauseAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  for (auto designatorList = &copy->designatorList;
       auto node : ListView{ast->designatorList}) {
    auto value = rewrite(node);
    *designatorList = make_list_node(arena(), value);
    designatorList = &(*designatorList)->next;
  }

  copy->initializer = rewrite(ast->initializer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<TypeTraitExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeTraitLoc = ast->typeTraitLoc;
  copy->lparenLoc = ast->lparenLoc;

  for (auto typeIdList = &copy->typeIdList;
       auto node : ListView{ast->typeIdList}) {
    auto value = rewrite(node);
    *typeIdList = make_list_node(arena(), value);
    typeIdList = &(*typeIdList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->typeTrait = ast->typeTrait;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ConditionExpressionAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->declarator = rewrite(ast->declarator);

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());
  copy->initializer = rewrite(ast->initializer);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<EqualInitializerAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<BracedInitListAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = make_node<ParenInitializerAST>(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::DesignatorVisitor::operator()(DotDesignatorAST* ast)
    -> DesignatorAST* {
  auto copy = make_node<DotDesignatorAST>(arena());

  copy->dotLoc = ast->dotLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DesignatorVisitor::operator()(SubscriptDesignatorAST* ast)
    -> DesignatorAST* {
  auto copy = make_node<SubscriptDesignatorAST>(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->expression = rewrite(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<TemplateTypeParameterAST>(arena());

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
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
  auto copy = make_node<NonTypeTemplateParameterAST>(arena());

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->declaration =
      ast_cast<ParameterDeclarationAST>(rewrite(ast->declaration));

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<TypenameTypeParameterAST>(arena());

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
  auto copy = make_node<ConstraintTypeParameterAST>(arena());

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
  auto copy = make_node<GeneratedTypeSpecifierAST>(arena());

  copy->typeLoc = ast->typeLoc;
  copy->type = ast->type;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<TypedefSpecifierAST>(arena());

  copy->typedefLoc = ast->typedefLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<FriendSpecifierAST>(arena());

  copy->friendLoc = ast->friendLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ConstevalSpecifierAST>(arena());

  copy->constevalLoc = ast->constevalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ConstinitSpecifierAST>(arena());

  copy->constinitLoc = ast->constinitLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ConstexprSpecifierAST>(arena());

  copy->constexprLoc = ast->constexprLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<InlineSpecifierAST>(arena());

  copy->inlineLoc = ast->inlineLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(NoreturnSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<NoreturnSpecifierAST>(arena());

  copy->noreturnLoc = ast->noreturnLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<StaticSpecifierAST>(arena());

  copy->staticLoc = ast->staticLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ExternSpecifierAST>(arena());

  copy->externLoc = ast->externLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(RegisterSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<RegisterSpecifierAST>(arena());

  copy->registerLoc = ast->registerLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ThreadLocalSpecifierAST>(arena());

  copy->threadLocalLoc = ast->threadLocalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ThreadSpecifierAST>(arena());

  copy->threadLoc = ast->threadLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<MutableSpecifierAST>(arena());

  copy->mutableLoc = ast->mutableLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<VirtualSpecifierAST>(arena());

  copy->virtualLoc = ast->virtualLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ExplicitSpecifierAST>(arena());

  copy->explicitLoc = ast->explicitLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<AutoTypeSpecifierAST>(arena());

  copy->autoLoc = ast->autoLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<VoidTypeSpecifierAST>(arena());

  copy->voidLoc = ast->voidLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<SizeTypeSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<SignTypeSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<VaListTypeSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<IntegralTypeSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierAST* {
  auto copy = make_node<FloatingPointTypeSpecifierAST>(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ComplexTypeSpecifierAST>(arena());

  copy->complexLoc = ast->complexLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<NamedTypeSpecifierAST>(arena());

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
  copy->symbol = ast->symbol;

  if (auto typeParameter = symbol_cast<TypeParameterSymbol>(copy->symbol)) {
    const auto& args = rewrite.templateArguments_;
    if (typeParameter && typeParameter->depth() == 0 &&
        typeParameter->index() < args.size()) {
      auto index = typeParameter->index();

      if (auto sym = std::get_if<Symbol*>(&args[index])) {
        copy->symbol = *sym;
      }
    }
  }

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<AtomicTypeSpecifierAST>(arena());

  copy->atomicLoc = ast->atomicLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(UnderlyingTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<UnderlyingTypeSpecifierAST>(arena());

  copy->underlyingTypeLoc = ast->underlyingTypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ElaboratedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ElaboratedTypeSpecifierAST>(arena());

  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
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
  auto copy = make_node<DecltypeAutoSpecifierAST>(arena());

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->autoLoc = ast->autoLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<DecltypeSpecifierAST>(arena());

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->type = ast->type;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(PlaceholderTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<PlaceholderTypeSpecifierAST>(arena());

  copy->typeConstraint = rewrite(ast->typeConstraint);
  copy->specifier = rewrite(ast->specifier);

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ConstQualifierAST>(arena());

  copy->constLoc = ast->constLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<VolatileQualifierAST>(arena());

  copy->volatileLoc = ast->volatileLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<RestrictQualifierAST>(arena());

  copy->restrictLoc = ast->restrictLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AtomicQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<AtomicQualifierAST>(arena());

  copy->atomicLoc = ast->atomicLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<EnumSpecifierAST>(arena());

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto enumeratorList = &copy->enumeratorList;
       auto node : ListView{ast->enumeratorList}) {
    auto value = rewrite(node);
    *enumeratorList = make_list_node(arena(), value);
    enumeratorList = &(*enumeratorList)->next;
  }

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<ClassSpecifierAST>(arena());

  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->finalLoc = ast->finalLoc;
  copy->colonLoc = ast->colonLoc;

  // ### TODO: use Binder::bind()
  auto _ = Binder::ScopeGuard{binder()};
  auto location = ast->symbol->location();
  auto className = ast->symbol->name();
  auto classSymbol = control()->newClassSymbol(binder()->scope(), location);
  classSymbol->setName(className);
  classSymbol->setIsUnion(ast->symbol->isUnion());
  classSymbol->setFinal(ast->isFinal);
  binder()->setScope(classSymbol);

  copy->symbol = classSymbol;

  for (auto baseSpecifierList = &copy->baseSpecifierList;
       auto node : ListView{ast->baseSpecifierList}) {
    auto value = rewrite(node);
    *baseSpecifierList = make_list_node(arena(), value);
    baseSpecifierList = &(*baseSpecifierList)->next;

    if (value->symbol) {
      classSymbol->addBaseClass(value->symbol);
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->classKey = ast->classKey;
  // copy->symbol = ast->symbol; // TODO: remove done by the binder
  copy->isFinal = ast->isFinal;

  binder()->complete(copy);

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<TypenameSpecifierAST>(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = make_node<SplicerTypeSpecifierAST>(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->splicer = rewrite(ast->splicer);

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = make_node<PointerOperatorAST>(arena());

  copy->starLoc = ast->starLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto cvQualifierListCtx = DeclSpecs{rewriter()};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = make_node<ReferenceOperatorAST>(arena());

  copy->refLoc = ast->refLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->refOp = ast->refOp;

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = make_node<PtrToMemberOperatorAST>(arena());

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->starLoc = ast->starLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto cvQualifierListCtx = DeclSpecs{rewriter()};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = make_node<BitfieldDeclaratorAST>(arena());

  copy->unqualifiedId = ast_cast<NameIdAST>(rewrite(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;
  copy->sizeExpression = rewrite(ast->sizeExpression);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = make_node<ParameterPackAST>(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->coreDeclarator = rewrite(ast->coreDeclarator);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = make_node<IdDeclaratorAST>(arena());

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = make_node<NestedDeclaratorAST>(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->declarator = rewrite(ast->declarator);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = make_node<FunctionDeclaratorChunkAST>(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause = rewrite(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  auto _ = Binder::ScopeGuard{binder()};

  if (copy->parameterDeclarationClause) {
    binder()->setScope(
        copy->parameterDeclarationClause->functionParametersSymbol);
  }

  auto cvQualifierListCtx = DeclSpecs{rewriter()};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  copy->refLoc = ast->refLoc;
  copy->exceptionSpecifier = rewrite(ast->exceptionSpecifier);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->trailingReturnType = rewrite(ast->trailingReturnType);
  copy->isFinal = ast->isFinal;
  copy->isOverride = ast->isOverride;
  copy->isPure = ast->isPure;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = make_node<ArrayDeclaratorChunkAST>(arena());

  copy->lbracketLoc = ast->lbracketLoc;

  auto typeQualifierListCtx = DeclSpecs{rewriter()};
  for (auto typeQualifierList = &copy->typeQualifierList;
       auto node : ListView{ast->typeQualifierList}) {
    auto value = rewrite(node);
    *typeQualifierList = make_list_node(arena(), value);
    typeQualifierList = &(*typeQualifierList)->next;
    typeQualifierListCtx.accept(value);
  }

  copy->expression = rewrite(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<NameIdAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<DestructorIdAST>(arena());

  copy->tildeLoc = ast->tildeLoc;
  copy->id = rewrite(ast->id);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<DecltypeIdAST>(arena());

  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite(ast->decltypeSpecifier));

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<OperatorFunctionIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->opLoc = ast->opLoc;
  copy->openLoc = ast->openLoc;
  copy->closeLoc = ast->closeLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<LiteralOperatorIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->literalLoc = ast->literalLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->literal = ast->literal;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<ConversionFunctionIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<SimpleTemplateIdAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->primaryTemplateSymbol = ast->primaryTemplateSymbol;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = make_node<LiteralOperatorTemplateIdAST>(arena());

  copy->literalOperatorId =
      ast_cast<LiteralOperatorIdAST>(rewrite(ast->literalOperatorId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = make_node<OperatorFunctionTemplateIdAST>(arena());

  copy->operatorFunctionId =
      ast_cast<OperatorFunctionIdAST>(rewrite(ast->operatorFunctionId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<GlobalNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<SimpleNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<DecltypeNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<TemplateNestedNameSpecifierAST>(arena());

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
  auto copy = make_node<DefaultFunctionBodyAST>(arena());

  copy->equalLoc = ast->equalLoc;
  copy->defaultLoc = ast->defaultLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = make_node<CompoundStatementFunctionBodyAST>(arena());

  copy->colonLoc = ast->colonLoc;

  for (auto memInitializerList = &copy->memInitializerList;
       auto node : ListView{ast->memInitializerList}) {
    auto value = rewrite(node);
    *memInitializerList = make_list_node(arena(), value);
    memInitializerList = &(*memInitializerList)->next;
  }

  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = make_node<TryStatementFunctionBodyAST>(arena());

  copy->tryLoc = ast->tryLoc;
  copy->colonLoc = ast->colonLoc;

  for (auto memInitializerList = &copy->memInitializerList;
       auto node : ListView{ast->memInitializerList}) {
    auto value = rewrite(node);
    *memInitializerList = make_list_node(arena(), value);
    memInitializerList = &(*memInitializerList)->next;
  }

  copy->statement = ast_cast<CompoundStatementAST>(rewrite(ast->statement));

  for (auto handlerList = &copy->handlerList;
       auto node : ListView{ast->handlerList}) {
    auto value = rewrite(node);
    *handlerList = make_list_node(arena(), value);
    handlerList = &(*handlerList)->next;
  }

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = make_node<DeleteFunctionBodyAST>(arena());

  copy->equalLoc = ast->equalLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = make_node<TypeTemplateArgumentAST>(arena());

  copy->typeId = rewrite(ast->typeId);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = make_node<ExpressionTemplateArgumentAST>(arena());

  copy->expression = rewrite(ast->expression);

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = make_node<ThrowExceptionSpecifierAST>(arena());

  copy->throwLoc = ast->throwLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = make_node<NoexceptSpecifierAST>(arena());

  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<SimpleRequirementAST>(arena());

  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<CompoundRequirementAST>(arena());

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
  auto copy = make_node<TypeRequirementAST>(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<NestedRequirementAST>(arena());

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = rewrite(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerAST* {
  auto copy = make_node<NewParenInitializerAST>(arena());

  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerAST* {
  auto copy = make_node<NewBracedInitializerAST>(arena());

  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerAST* {
  auto copy = make_node<ParenMemInitializerAST>(arena());

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerAST* {
  auto copy = make_node<BracedMemInitializerAST>(arena());

  copy->nestedNameSpecifier = rewrite(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite(ast->unqualifiedId);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite(ast->bracedInitList));
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = make_node<ThisLambdaCaptureAST>(arena());

  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureAST* {
  auto copy = make_node<DerefThisLambdaCaptureAST>(arena());

  copy->starLoc = ast->starLoc;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = make_node<SimpleLambdaCaptureAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = make_node<RefLambdaCaptureAST>(arena());

  copy->ampLoc = ast->ampLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = make_node<RefInitLambdaCaptureAST>(arena());

  copy->ampLoc = ast->ampLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = make_node<InitLambdaCaptureAST>(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = make_node<EllipsisExceptionDeclarationAST>(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = make_node<TypeExceptionDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = rewrite(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = make_node<CxxAttributeAST>(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->lbracket2Loc = ast->lbracket2Loc;
  copy->attributeUsingPrefix = rewrite(ast->attributeUsingPrefix);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->rbracket2Loc = ast->rbracket2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = make_node<GccAttributeAST>(arena());

  copy->attributeLoc = ast->attributeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->lparen2Loc = ast->lparen2Loc;
  copy->rparenLoc = ast->rparenLoc;
  copy->rparen2Loc = ast->rparen2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierAST* {
  auto copy = make_node<AlignasAttributeAST>(arena());

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
  auto copy = make_node<AlignasTypeAttributeAST>(arena());

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
  auto copy = make_node<AsmAttributeAST>(arena());

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = make_node<ScopedAttributeTokenAST>(arena());

  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->attributeNamespace = ast->attributeNamespace;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = make_node<SimpleAttributeTokenAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

}  // namespace cxx
