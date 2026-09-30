// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/lambda_captures.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
namespace {

[[nodiscard]] auto memberAccessObjectType(TranslationUnit* unit,
                                          MemberExpressionAST* ast)
    -> const Type* {
  if (!ast->baseExpression || !ast->baseExpression->type) return nullptr;

  auto traits = unit->typeTraits();
  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    return traits.remove_cv(traits.get_element_type(ast->baseExpression->type));
  }
  return traits.remove_cv(ast->baseExpression->type);
}

[[nodiscard]] auto memberBelongsToObjectType(TranslationUnit* unit,
                                             const Type* objectType,
                                             Symbol* member) -> bool {
  if (!symbol_cast<ClassSymbol>(member->parent())) return true;
  if (!type_cast<ClassType>(objectType)) return true;
  if (isDependent(unit, objectType)) return true;

  return unit->typeTraits().is_member_of_object_type(objectType, member);
}

}  // namespace

struct ASTRewriter::ExpressionVisitor : VisitorBase {
  [[nodiscard]] auto typeChecker() -> TypeChecker {
    return rewrite.typeChecker();
  }

  [[nodiscard]] auto spelledIntegerLiteral(VariableSymbol* var,
                                           const Type* type) const
      -> ExpressionAST* {
    if (!var->constValue()) return nullptr;

    auto value = std::get_if<ConstInt>(&*var->constValue());
    if (!value) return nullptr;

    return TemplateArguments{translationUnit()}.integerLiteralExpression(*value,
                                                                         type);
  }

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

  [[nodiscard]] auto operator()(PackIndexExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NestedStatementExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(DefaultInitializerExpressionAST* ast)
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

  void bindDestroyedType(MemberExpressionAST* copy, const Type* objectType);

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

  [[nodiscard]] auto operator()(LabelAddressExpressionAST* ast)
      -> ExpressionAST*;

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

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ThreeWayComparisonExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(ConditionalExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(YieldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ThrowExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(TargetExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RightExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(CompoundAssignmentExpressionAST* ast)
      -> ExpressionAST*;

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

struct ASTRewriter::NewInitializerVisitor : VisitorBase {
  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerAST*;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerAST*;
};

struct ASTRewriter::GenericAssociationVisitor : VisitorBase {
  [[nodiscard]] auto operator()(DefaultGenericAssociationAST* ast)
      -> GenericAssociationAST*;

  [[nodiscard]] auto operator()(TypeGenericAssociationAST* ast)
      -> GenericAssociationAST*;
};

struct ASTRewriter::LambdaCaptureVisitor : VisitorBase {
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

auto ASTRewriter::expression(ExpressionAST* ast) -> ExpressionAST* {
  if (!ast) return {};
  auto expr = visit(ExpressionVisitor{*this}, ast);
  if (expr) {
    expr = check(expr);
    StandardConversion{unit_}.foldConstantRead(expr);
  }
  return expr;
}

auto ASTRewriter::unevaluatedExpression(ExpressionAST* ast) -> ExpressionAST* {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  ++unevaluatedOperandDepth_;
  auto result = expression(ast);
  --unevaluatedOperandDepth_;
  return result;
}

auto ASTRewriter::newInitializer(NewInitializerAST* ast) -> NewInitializerAST* {
  if (!ast) return {};
  return visit(NewInitializerVisitor{*this}, ast);
}

auto ASTRewriter::genericAssociation(GenericAssociationAST* ast)
    -> GenericAssociationAST* {
  if (!ast) return {};
  return visit(GenericAssociationVisitor{*this}, ast);
}

auto ASTRewriter::lambdaCapture(LambdaCaptureAST* ast) -> LambdaCaptureAST* {
  if (!ast) return {};
  return visit(LambdaCaptureVisitor{*this}, ast);
}

auto ASTRewriter::newPlacement(NewPlacementAST* ast) -> NewPlacementAST* {
  if (!ast) return {};

  auto copy = NewPlacementAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;

  copy->expressionList =
      rewriteList(ast->expressionList, &ASTRewriter::expression);

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::lambdaSpecifier(LambdaSpecifierAST* ast)
    -> LambdaSpecifierAST* {
  if (!ast) return {};

  auto copy = LambdaSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CharLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = BoolLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->isTrue = ast->isTrue;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = IntLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = FloatLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = NullptrLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = StringLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = UserDefinedStringLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ObjectLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ThisExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;

  auto thisType = binder()->enclosingThisType(binder()->scope());
  copy->type = thisType ? thisType : ast->type;

  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PackIndexExpressionAST* ast)
    -> ExpressionAST* {
  if (auto parameterPack =
          rewrite.findReferencedParameterPack(ast->packExpression)) {
    auto rewrittenIndex = rewrite.expression(ast->indexExpression);

    ASTInterpreter interp(rewrite.unit_);
    auto val = interp.evaluate(rewrittenIndex);
    auto idxOpt = val ? interp.toInt(*val) : std::nullopt;

    if (idxOpt.has_value()) {
      auto idx = static_cast<int>(*idxOpt);
      auto packSize = static_cast<int>(parameterPack->elements().size());

      if (idx >= 0 && idx < packSize) {
        ExpressionAST* result = nullptr;
        rewrite.expandPackElement({parameterPack}, idx, [&] {
          result = rewrite.expression(ast->packExpression);
        });

        return result;
      }
    }
  }

  auto copy = PackIndexExpressionAST::create(arena());
  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->packExpression =
      ast_cast<IdExpressionAST>(rewrite.expression(ast->packExpression));
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->indexExpression = rewrite.expression(ast->indexExpression);
  copy->rbracketLoc = ast->rbracketLoc;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GenericSelectionExpressionAST* ast) -> ExpressionAST* {
  auto copy = GenericSelectionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->genericLoc = ast->genericLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;

  copy->genericAssociationList = rewrite.rewriteList(
      ast->genericAssociationList, &ASTRewriter::genericAssociation);

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionAST* {
  auto copy = NestedStatementExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    DefaultInitializerExpressionAST* ast) -> ExpressionAST* {
  auto copy = DefaultInitializerExpressionAST::create(arena());
  copy->expression = rewrite.expression(ast->expression);
  copy->context = ast->context;
  copy->context.scope = rewrite.binder_.scope();
  copy->type = copy->expression ? copy->expression->type : ast->type;
  copy->valueCategory = ast->valueCategory;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NestedExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionAST* {
  auto call = std::exchange(rewrite.rewritingCall_, nullptr);
  const auto isCallee = call != nullptr;
  if (auto pack = rewrite.functionParameterPackFor(ast->symbol)) {
    if (auto expandedParam = rewrite.packElementAt(pack)) {
      auto copy = IdExpressionAST::create(arena());
      copy->valueCategory = ValueCategory::kLValue;
      copy->type = expandedParam->type();
      copy->nestedNameSpecifier =
          rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
      copy->templateLoc = ast->templateLoc;
      copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
      copy->isTemplateIntroduced = ast->isTemplateIntroduced;
      copy->symbol = expandedParam;
      return copy;
    }
  }

  auto copy = IdExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol);
      param && rewrite.templateArgumentFor(param)) {
    copy->symbol = param;

    if (auto substituted = rewrite.substitutedSymbol(param)) {
      copy->symbol = substituted;

      if (auto var = symbol_cast<VariableSymbol>(substituted)) {
        if (var->type()) copy->type = var->type();

        if (auto literal = spelledIntegerLiteral(var, copy->type))
          return literal;

        if (auto initializer = var->initializer()) {
          return initializer->clone(arena());
        }
      }
    }
  } else if (copy->nestedNameSpecifier && copy->nestedNameSpecifier->symbol) {
    binder()->qualifiedLookupIdExpression(copy, isCallee);
  } else if (is_function_local_predefined_variable(ast->symbol)) {
    copy->symbol = binder()->functionLocalPredefinedVariable(
        binder()->scope(), ast->symbol->name());
    if (copy->symbol) copy->type = copy->symbol->type();
  } else if (ast->symbol) {
    copy->symbol = rewrite.remapSymbol(ast->symbol);

    if (!isCallee && symbol_cast<OverloadSetSymbol>(copy->symbol) &&
        type_cast<FunctionType>(copy->type)) {
      if (auto function = designatedFunction(copy->symbol))
        copy->symbol = function;
    }

    if (auto field = rewrite.lambdaCaptureField(copy->symbol)) {
      copy->symbol = field;
      copy->type =
          translationUnit()->typeTraits().remove_reference(field->type());
      copy->valueCategory = ValueCategory::kLValue;
      return copy;
    }

    if (copy->symbol == ast->symbol) {
      auto templateId = ast_cast<SimpleTemplateIdAST>(copy->unqualifiedId);
      if (auto member = rewrite.instantiatedMemberTemplateFor(
              copy->symbol, templateId, call, ast->firstSourceLocation())) {
        copy->symbol = member;
      }
    }

    if (auto usingDecl = symbol_cast<UsingDeclarationSymbol>(copy->symbol);
        usingDecl && usingDecl->target() && !isCallee) {
      copy->symbol = usingDecl->target();
      copy->type = copy->symbol->type();
      return copy;
    }

    binder()->resolveIdExpression(copy, isCallee);

    if (copy->symbol != ast->symbol && copy->symbol) {
      copy->type = copy->symbol->type();
    }
  }

  return fold_concept_id(translationUnit(), copy);
}

auto ASTRewriter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = LambdaExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->captureDefaultLoc = ast->captureDefaultLoc;

  const bool needsFreshClosure = !type_cast<ClassType>(ast->type);

  auto scopeGuard = Binder::ScopeGuard(binder());

  if (needsFreshClosure) binder()->bind(copy);

  copy->captureList =
      rewrite.rewriteList(ast->captureList, &ASTRewriter::lambdaCapture);

  copy->rbracketLoc = ast->rbracketLoc;
  copy->lessLoc = ast->lessLoc;

  copy->templateParameterList = rewrite.rewriteList(
      ast->templateParameterList, &ASTRewriter::templateParameter);

  if (copy->templateParameterList && copy->symbol)
    copy->symbol->setTemplate(true);

  copy->greaterLoc = ast->greaterLoc;
  copy->templateRequiresClause =
      rewrite.requiresClause(ast->templateRequiresClause);

  copy->expressionAttributeList = rewrite.rewriteList(
      ast->expressionAttributeList, &ASTRewriter::attributeSpecifier);

  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  copy->gnuAtributeList = rewrite.rewriteList(ast->gnuAtributeList,
                                              &ASTRewriter::attributeSpecifier);

  copy->lambdaSpecifierList = rewrite.rewriteList(
      ast->lambdaSpecifierList, &ASTRewriter::lambdaSpecifier);

  {
    auto _ = Binder::ScopeGuard(binder());

    if (copy->parameterDeclarationClause) {
      binder()->setScope(
          copy->parameterDeclarationClause->functionParametersSymbol);
    }

    copy->exceptionSpecifier =
        rewrite.exceptionSpecifier(ast->exceptionSpecifier);

    copy->attributeList = rewrite.rewriteList(ast->attributeList,
                                              &ASTRewriter::attributeSpecifier);

    copy->trailingReturnType =
        rewrite.trailingReturnType(ast->trailingReturnType);
    copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  }

  copy->captureDefault = ast->captureDefault;

  if (needsFreshClosure) {
    binder()->complete(copy);

    rewrite.remapInitCaptures(ast->symbol, copy->symbol);

    if (auto classType = type_cast<ClassType>(copy->type)) {
      auto classSymbol = classType->symbol();

      if (!classSymbol->capturedThisField() &&
          (copy->captureDefault == TokenKind::T_AMP ||
           copy->captureDefault == TokenKind::T_EQUAL) &&
          binder()->usesImplicitThis(ast->statement)) {
        if (auto thisType = binder()->enclosingThisType(binder()->scope())) {
          auto capture = binder()->addImplicitThisCapture(classSymbol, thisType,
                                                          copy->lbracketLoc);

          auto tail = &copy->captureList;
          while (*tail) tail = &(*tail)->next;
          *tail = make_list_node<LambdaCaptureAST>(arena(), capture);

          auto status = binder()->buildRecordLayout(classSymbol);
          if (!status.has_value()) {
            rewrite.error(copy->lbracketLoc, status.error());
          }
        }
      }

      binder()->setScope(classType->symbol());

      std::unordered_map<Symbol*, FieldSymbol*> captureFields;
      for (auto captureNode : ListView{copy->captureList}) {
        auto field = capture_field(captureNode);
        if (!field) continue;

        if (ast_cast<InitLambdaCaptureAST>(captureNode) ||
            ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
          if (auto declared = Binder::declaredInitCapture(
                  copy->symbol, capture_identifier(captureNode)))
            captureFields[declared] = field;
          continue;
        }

        auto initExpr = capture_initializer(captureNode);
        if (!initExpr) continue;

        if (auto cast = ast_cast<ImplicitCastExpressionAST>(initExpr))
          initExpr = cast->expression;

        auto outerIdExpr = ast_cast<IdExpressionAST>(initExpr);
        if (!outerIdExpr || !outerIdExpr->symbol) continue;

        captureFields[outerIdExpr->symbol] = field;
      }

      rewrite.pushLambdaCaptureFields(std::move(captureFields));
      {
        auto bodyScopeGuard = Binder::ScopeGuard(binder());
        if (auto lambdaSymbol = symbol_cast<LambdaSymbol>(copy->symbol))
          binder()->setScope(lambdaSymbol);
        copy->statement = rewrite.lambdaBody(ast->statement);
      }
      rewrite.popLambdaCaptureFields();

      binder()->completeLambdaBody(copy);
    } else {
      copy->statement = rewrite.lambdaBody(ast->statement);
    }

    return copy;
  }

  copy->statement = rewrite.lambdaBody(ast->statement);
  copy->symbol = ast->symbol;
  copy->constructorSymbol = ast->constructorSymbol;

  return copy;
}

auto ASTRewriter::foldStep(ExpressionAST* left, TokenKind op,
                           SourceLocation opLoc, ExpressionAST* right)
    -> ExpressionAST* {
  auto binop = BinaryExpressionAST::create(arena());
  binop->valueCategory = left->valueCategory;
  binop->type = left->type;
  binop->leftExpression = left;
  binop->op = op;
  binop->opLoc = opLoc;
  binop->rightExpression = right;
  return check(binop);
}

auto ASTRewriter::leftFold(ExpressionAST* pattern, SourceLocation ellipsisLoc,
                           TokenKind op, SourceLocation opLoc,
                           ExpressionAST* init) -> ExpressionAST* {
  auto current = init;
  forEachPackElement(pattern, ellipsisLoc, [&] {
    auto element = expression(pattern);
    current = current ? foldStep(current, op, opLoc, element) : element;
  });
  return current;
}

auto ASTRewriter::rightFold(ExpressionAST* pattern, SourceLocation ellipsisLoc,
                            TokenKind op, SourceLocation opLoc,
                            ExpressionAST* init) -> ExpressionAST* {
  auto current = init;
  forEachPackElementReversed(pattern, ellipsisLoc, [&] {
    auto element = expression(pattern);
    current = current ? foldStep(element, op, opLoc, current) : element;
  });
  return current;
}

auto ASTRewriter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionAST* {
  if (rewrite.findReferencedParameterPack(ast->leftExpression)) {
    return rewrite.rightFold(ast->leftExpression, ast->ellipsisLoc, ast->op,
                             ast->opLoc,
                             rewrite.expression(ast->rightExpression));
  }

  if (rewrite.findReferencedParameterPack(ast->rightExpression)) {
    return rewrite.leftFold(ast->rightExpression, ast->ellipsisLoc, ast->op,
                            ast->opLoc,
                            rewrite.expression(ast->leftExpression));
  }

  auto copy = FoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->foldOpLoc = ast->foldOpLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;
  copy->foldOp = ast->foldOp;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionAST* {
  if (rewrite.findReferencedParameterPack(ast->expression)) {
    if (auto folded = rewrite.rightFold(ast->expression, ast->ellipsisLoc,
                                        ast->op, ast->opLoc, nullptr))
      return folded;
    return rewrite.emptyFoldIdentity(ast->op);
  }

  auto copy = RightFoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionAST* {
  if (rewrite.findReferencedParameterPack(ast->expression)) {
    if (auto folded = rewrite.leftFold(ast->expression, ast->ellipsisLoc,
                                       ast->op, ast->opLoc, nullptr))
      return folded;
    return rewrite.emptyFoldIdentity(ast->op);
  }

  auto copy = LeftFoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = RequiresExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->requiresLoc = ast->requiresLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  auto _ = Binder::ScopeGuard{
      binder(), copy->parameterDeclarationClause
                    ? copy->parameterDeclarationClause->functionParametersSymbol
                    : nullptr};

  bool requirementsAreInvalid = false;

  {
    ASTRewriter::ImmediateContextGuard immediateContext{rewrite};

    copy->requirementList =
        rewrite.rewriteList(ast->requirementList, &ASTRewriter::requirement);

    requirementsAreInvalid = immediateContext.substitutionFailed();
  }

  copy->rbraceLoc = ast->rbraceLoc;

  if (requirementsAreInvalid) {
    return BoolLiteralExpressionAST::create(arena(), ast->requiresLoc, false,
                                            ValueCategory::kPrValue,
                                            control()->getBoolType());
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = VaArgExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->vaArgLoc = ast->vaArgLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SubscriptExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->lbracketLoc = ast->lbracketLoc;
  copy->indexExpression = rewrite.expression(ast->indexExpression);
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::rewriteExpressionList(List<ExpressionAST*>* source)
    -> List<ExpressionAST*>* {
  List<ExpressionAST*>* result = nullptr;
  ListAppender<ExpressionAST> append{arena(), result};

  for (auto node : ListView{source}) {
    if (auto packExpansion = ast_cast<PackExpansionExpressionAST>(node);
        packExpansion && !expandsAnActivePack(packExpansion->expression)) {
      if (auto parameterPack =
              findReferencedParameterPack(packExpansion->expression)) {
        forEachPackElement(
            packExpansion->expression, packExpansion->ellipsisLoc,
            [&] { append(expression(packExpansion->expression)); },
            parameterPack);
        continue;
      }
    }

    append(expression(node));
  }

  return result;
}

auto ASTRewriter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionAST* {
  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto typeParam = symbol_cast<TypeParameterSymbol>(idExpr->symbol)) {
      if (auto substituted = rewrite.substitutedSymbol(typeParam)) {
        auto typeSpec = NamedTypeSpecifierAST::create(arena());
        typeSpec->unqualifiedId = rewrite.unqualifiedId(idExpr->unqualifiedId);
        typeSpec->symbol = substituted;

        auto tc = TypeConstructionAST::create(arena());
        tc->typeSpecifier = typeSpec;
        tc->lparenLoc = ast->lparenLoc;
        tc->rparenLoc = ast->rparenLoc;
        tc->valueCategory = ValueCategory::kPrValue;
        tc->type = substituted->type();

        tc->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

        return tc;
      }
    }
  }

  auto copy = CallExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;

  copy->lparenLoc = ast->lparenLoc;
  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);
  copy->rparenLoc = ast->rparenLoc;

  if (ast_cast<IdExpressionAST>(ast->baseExpression))
    rewrite.rewritingCall_ = copy;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  rewrite.rewritingCall_ = nullptr;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeConstructionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite.specifier(ast->typeSpecifier);
  copy->lparenLoc = ast->lparenLoc;

  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = BracedTypeConstructionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite.specifier(ast->typeSpecifier);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SpliceMemberExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->templateLoc = ast->templateLoc;
  copy->splicer = rewrite.splicer(ast->splicer);
  copy->symbol = ast->symbol;
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = MemberExpressionAST::create(arena());

  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  auto objectType = memberAccessObjectType(translationUnit(), copy);

  if (ast_cast<DestructorIdAST>(copy->unqualifiedId)) {
    bindDestroyedType(copy, objectType);
    return copy;
  }

  auto member = rewrite.remapSymbol(ast->symbol);
  if (!member) return copy;

  if (memberBelongsToObjectType(translationUnit(), objectType, member))
    copy->symbol = member;

  return copy;
}

void ASTRewriter::ExpressionVisitor::bindDestroyedType(
    MemberExpressionAST* copy, const Type* objectType) {
  if (!objectType || isDependent(translationUnit(), objectType)) return;

  auto classType = type_cast<ClassType>(objectType);
  if (!classType) {
    copy->type = control()->getPseudoDestructorType();
    return;
  }

  auto classSymbol = classType->symbol();
  translationUnit()->typeTraits().requireCompleteClass(classSymbol);
  copy->symbol = classSymbol->destructor();
}

auto ASTRewriter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = PostIncrExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->opLoc = ast->opLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CppCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->castLoc = ast->castLoc;
  copy->lessLoc = ast->lessLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->greaterLoc = ast->greaterLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->castOp = ast->castOp;

  copy->type = copy->typeId ? copy->typeId->type : ast->type;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionAST* {
  auto copy = BuiltinBitCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->castLoc = ast->castLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionAST* {
  auto copy = BuiltinOffsetofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->offsetofLoc = ast->offsetofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->identifierLoc = ast->identifierLoc;

  copy->designatorList =
      rewrite.rewriteList(ast->designatorList, &ASTRewriter::designator);

  copy->rparenLoc = ast->rparenLoc;
  copy->identifier = ast->identifier;

  return rewrite.check(copy);
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeidExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  auto traits = rewrite.unit_->typeTraits();
  if (copy->expression && copy->expression->type &&
      copy->expression->valueCategory != ValueCategory::kPrValue &&
      traits.is_polymorphic(traits.remove_cvref(copy->expression->type))) {
    copy->expression = rewrite.check(copy->expression);
  }
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeidOfTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SpliceExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->splicer = rewrite.splicer(ast->splicer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = GlobalScopeReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = NamespaceReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeIdReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeIdReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = LabelAddressExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->ampAmpLoc = ast->ampAmpLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = UnaryExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AwaitExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->awaitLoc = ast->awaitLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SizeofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);

  if (copy->expression && copy->expression->type) {
    copy->value =
        translationUnit()->typeTraits().size_of(copy->expression->type);
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SizeofTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  if (copy->typeId && copy->typeId->type) {
    copy->value = translationUnit()->typeTraits().size_of(copy->typeId->type);
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionAST* {
  auto pack = rewrite.parameterPackFor(ast->symbol);
  if (!pack) pack = rewrite.functionParameterPackFor(ast->symbol);

  if (pack) {
    auto literal =
        control()->integerLiteral(std::to_string(pack->elements().size()));
    return IntLiteralExpressionAST::create(arena(), literal,
                                           /*literalOperatorCall=*/nullptr,
                                           ValueCategory::kPrValue,
                                           control()->getSizeType());
  }

  auto copy = SizeofPackExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->identifier = ast->identifier;
  copy->symbol = rewrite.remapSymbol(ast->symbol);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AlignofTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AlignofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NoexceptExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NewExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->newLoc = ast->newLoc;
  copy->newPlacement = rewrite.newPlacement(ast->newPlacement);
  copy->lparenLoc = ast->lparenLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewrite.unit_};
  copy->typeSpecifierList = rewrite.rewriteSpecifierList(ast->typeSpecifierList,
                                                         typeSpecifierListCtx);

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());

  copy->objectType = declaratorType;

  copy->rparenLoc = ast->rparenLoc;
  copy->newInitalizer = rewrite.newInitializer(ast->newInitalizer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = DeleteExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->rbracketLoc = ast->rbracketLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionAST* {
  auto expression = rewrite.expression(ast->expression);

  if (ast->castKind == ImplicitCastKind::kUserDefinedConversion &&
      (!ast->type || isDependent(rewrite.unit_, ast->type))) {
    return expression;
  }

  auto copy = ImplicitCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = expression;
  copy->castKind = ast->castKind;
  copy->conversionFunction = ast->conversionFunction;
  copy->isVirtualDispatch = ast->isVirtualDispatch;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConstExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ConstExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite.expression(ast->expression);
  copy->constValue = ast->constValue;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = BinaryExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    ThreeWayComparisonExpressionAST* ast) -> ExpressionAST* {
  auto copy = ThreeWayComparisonExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->comparison =
      ast_cast<BinaryExpressionAST>(rewrite.expression(ast->comparison));
  copy->lessResult = ast->lessResult;
  copy->equalResult = ast->equalResult;
  copy->greaterResult = ast->greaterResult;
  copy->unorderedResult = ast->unorderedResult;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ConditionalExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->condition = rewrite.expression(ast->condition);
  copy->questionLoc = ast->questionLoc;
  copy->iftrueExpression = rewrite.expression(ast->iftrueExpression);
  copy->colonLoc = ast->colonLoc;
  copy->iffalseExpression = rewrite.expression(ast->iffalseExpression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = YieldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->yieldLoc = ast->yieldLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ThrowExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->throwLoc = ast->throwLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AssignmentExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TargetExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TargetExpressionAST::create(arena());
  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RightExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = RightExpressionAST::create(arena());
  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionAST* {
  auto copy = CompoundAssignmentExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->isVirtualDispatch = ast->isVirtualDispatch;
  copy->targetExpression = rewrite.expression(ast->targetExpression);
  copy->opLoc = ast->opLoc;
  if (ast->leftExpression) {
    copy->leftExpression = TargetExpressionAST::create(arena());
  }
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  if (ast->adjustExpression) {
    copy->adjustExpression = RightExpressionAST::create(arena());
  }
  copy->op = ast->op;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionAST* {
  if (rewrite.expandsAnActivePack(ast->expression)) {
    return rewrite.expression(ast->expression);
  }

  auto copy = PackExpansionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite.expression(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionAST* {
  auto copy = DesignatedInitializerClauseAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  copy->designatorList =
      rewrite.rewriteList(ast->designatorList, &ASTRewriter::designator);

  copy->initializer = rewrite.expression(ast->initializer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeTraitExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeTraitLoc = ast->typeTraitLoc;
  copy->lparenLoc = ast->lparenLoc;

  ListAppender<TypeIdAST> append{arena(), copy->typeIdList};
  for (auto node : ListView{ast->typeIdList}) {
    if (auto pack = rewrite.expandedParameterPack(node)) {
      rewrite.forEachPackElement(
          node, node->firstSourceLocation(),
          [&] { append(rewrite.typeId(node)); }, pack);
      continue;
    }

    append(rewrite.typeId(node));
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->typeTrait = ast->typeTrait;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ConditionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto declSpecifierListCtx = DeclSpecs{rewrite.unit_};
  copy->declSpecifierList = rewrite.rewriteSpecifierList(ast->declSpecifierList,
                                                         declSpecifierListCtx);

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());
  copy->initializer = rewrite.expression(ast->initializer);

  copy->symbol = binder()->declareVariable(copy->declarator, declaratorDecl,
                                           /*addSymbolToParentScope=*/true);

  if (ast->symbol && copy->symbol) {
    rewrite.addSymbolRemap(ast->symbol, copy->symbol);
  }

  typeChecker().check_condition_declaration(copy);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = EqualInitializerAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionAST* {
  auto copy = BracedInitListAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbraceLoc = ast->lbraceLoc;

  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;
  copy->implicitElement = ast->implicitElement;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = ParenInitializerAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;

  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerAST* {
  auto copy = NewParenInitializerAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;

  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerAST* {
  auto copy = NewBracedInitializerAST::create(arena());

  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::GenericAssociationVisitor::operator()(
    DefaultGenericAssociationAST* ast) -> GenericAssociationAST* {
  auto copy = DefaultGenericAssociationAST::create(arena());

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::GenericAssociationVisitor::operator()(
    TypeGenericAssociationAST* ast) -> GenericAssociationAST* {
  auto copy = TypeGenericAssociationAST::create(arena());

  copy->typeId = rewrite.typeId(ast->typeId);
  copy->colonLoc = ast->colonLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = ThisLambdaCaptureAST::create(arena());

  copy->thisLoc = ast->thisLoc;
  copy->initializer = rewrite.expression(ast->initializer);

  copy->symbol = ast->symbol;
  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureAST* {
  auto copy = DerefThisLambdaCaptureAST::create(arena());

  copy->starLoc = ast->starLoc;
  copy->thisLoc = ast->thisLoc;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = SimpleLambdaCaptureAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;
  copy->initializer = rewrite.expression(ast->initializer);

  copy->symbol = ast->symbol;
  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = RefLambdaCaptureAST::create(arena());

  copy->ampLoc = ast->ampLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;
  copy->initializer = rewrite.expression(ast->initializer);

  copy->symbol = ast->symbol;
  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = RefInitLambdaCaptureAST::create(arena());

  copy->ampLoc = ast->ampLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->identifier = ast->identifier;

  copy->symbol = ast->symbol;
  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = InitLambdaCaptureAST::create(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->identifier = ast->identifier;

  copy->symbol = ast->symbol;
  return copy;
}
}  // namespace cxx
