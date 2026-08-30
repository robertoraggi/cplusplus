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
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
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

 private:
  enum class PackResult { kExpanded, kEmpty, kNotPack };

  [[nodiscard]] auto expandTypePackArgument(
      TypeTemplateArgumentAST* typeArg,
      List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult;

  [[nodiscard]] auto expandExprPackArgument(
      ExpressionTemplateArgumentAST* exprArg,
      List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult;

  void substituteTemplateTemplateParameter(SimpleTemplateIdAST* copy);
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

  void resolveDependentQualifier(SimpleNestedNameSpecifierAST* copy);

  [[nodiscard]] auto lookupType(NestedNameSpecifierAST* prefix,
                                const Identifier* id) const -> Symbol*;
};

auto ASTRewriter::NestedNameSpecifierVisitor::lookupType(
    NestedNameSpecifierAST* prefix, const Identifier* id) const -> Symbol* {
  auto isType = [](Symbol* s) { return is_type(s); };
  if (prefix && prefix->symbol) {
    return qualifiedLookup(prefix->symbol, id, isType);
  }
  for (auto scope = binder()->scope(); scope; scope = scope->parent()) {
    if (auto resolved = qualifiedLookup(scope, id, isType)) return resolved;
  }
  return nullptr;
}

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

auto ASTRewriter::unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdAST* {
  if (!ast) return {};
  return visit(UnqualifiedIdVisitor{*this}, ast);
}

auto ASTRewriter::nestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierAST* {
  if (!ast) return {};
  return visit(NestedNameSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::templateArgument(TemplateArgumentAST* ast)
    -> TemplateArgumentAST* {
  if (!ast) return {};
  return visit(TemplateArgumentVisitor{*this}, ast);
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = NameIdAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = DestructorIdAST::create(arena());

  copy->tildeLoc = ast->tildeLoc;
  copy->id = rewrite.unqualifiedId(ast->id);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = DecltypeIdAST::create(arena());

  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = OperatorFunctionIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->opLoc = ast->opLoc;
  copy->openLoc = ast->openLoc;
  copy->closeLoc = ast->closeLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = LiteralOperatorIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->literalLoc = ast->literalLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->literal = ast->literal;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = ConversionFunctionIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::expandTypePackArgument(
    TypeTemplateArgumentAST* typeArg,
    List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult {
  if (!typeArg) return PackResult::kNotPack;

  auto pack = rewrite.expandedParameterPack(typeArg->typeId);
  if (!pack) return PackResult::kNotPack;
  if (pack->elements().empty()) return PackResult::kEmpty;

  rewrite.forEachPackElement(
      typeArg->typeId, typeArg->firstSourceLocation(),
      [&] {
        auto expandedArg = TypeTemplateArgumentAST::create(arena());
        expandedArg->typeId = rewrite.typeId(typeArg->typeId);

        *templateArgumentList = make_list_node(
            arena(), static_cast<TemplateArgumentAST*>(expandedArg));
        templateArgumentList = &(*templateArgumentList)->next;
      },
      pack);

  return PackResult::kExpanded;
}

auto ASTRewriter::UnqualifiedIdVisitor::expandExprPackArgument(
    ExpressionTemplateArgumentAST* exprArg,
    List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult {
  if (!exprArg) return PackResult::kNotPack;

  auto packExpr = ast_cast<PackExpansionExpressionAST>(exprArg->expression);
  if (!packExpr) return PackResult::kNotPack;

  auto parameterPack =
      rewrite.findReferencedParameterPack(packExpr->expression);
  if (!parameterPack) return PackResult::kNotPack;
  if (parameterPack->elements().empty()) return PackResult::kEmpty;

  rewrite.forEachPackElement(
      packExpr->expression, packExpr->ellipsisLoc,
      [&] {
        auto expandedArg = ExpressionTemplateArgumentAST::create(arena());
        expandedArg->expression = rewrite.expression(packExpr->expression);

        *templateArgumentList = make_list_node(
            arena(), static_cast<TemplateArgumentAST*>(expandedArg));
        templateArgumentList = &(*templateArgumentList)->next;
      },
      parameterPack);
  return PackResult::kExpanded;
}

void ASTRewriter::UnqualifiedIdVisitor::substituteTemplateTemplateParameter(
    SimpleTemplateIdAST* copy) {
  auto ttpSymbol = symbol_cast<TemplateTypeParameterSymbol>(copy->symbol);
  if (!ttpSymbol) return;

  auto substituted = rewrite.substitutedSymbol(ttpSymbol);
  if (!substituted) return;

  if (auto templateName = template_name_symbol(substituted)) {
    copy->symbol = templateName;
  }
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = SimpleTemplateIdAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto typeArg = ast_cast<TypeTemplateArgumentAST>(node);
    auto typeResult = expandTypePackArgument(typeArg, templateArgumentList);
    if (typeResult != PackResult::kNotPack) {
      if (typeResult == PackResult::kExpanded) continue;
      if (typeResult == PackResult::kEmpty) continue;
    }

    auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(node);
    auto exprResult = expandExprPackArgument(exprArg, templateArgumentList);
    if (exprResult != PackResult::kNotPack) {
      if (exprResult == PackResult::kExpanded) continue;
      if (exprResult == PackResult::kEmpty) continue;
    }

    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  if (is_member_template(ast->symbol)) {
    copy->symbol = rewrite.remapSymbol(ast->symbol);
    if (hasDependentTemplateArguments(rewrite.unit_, copy)) {
      if (auto templateSymbol = templated_symbol(copy->symbol))
        copy->symbol = templateSymbol;
    }
  }

  substituteTemplateTemplateParameter(copy);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = LiteralOperatorTemplateIdAST::create(arena());

  copy->literalOperatorId = ast_cast<LiteralOperatorIdAST>(
      rewrite.unqualifiedId(ast->literalOperatorId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = OperatorFunctionTemplateIdAST::create(arena());

  copy->operatorFunctionId = ast_cast<OperatorFunctionIdAST>(
      rewrite.unqualifiedId(ast->operatorFunctionId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = GlobalNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = SimpleNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->scopeLoc = ast->scopeLoc;

  const bool isTypeParameter =
      symbol_cast<TypeParameterSymbol>(copy->symbol) ||
      symbol_cast<ConstraintTypeParameterSymbol>(copy->symbol);

  auto needsSubstitution = !copy->symbol || isTypeParameter;
  if (needsSubstitution && copy->identifier) {
    auto emitNonScopeError = [&](SourceLocation loc, Symbol* argSym) {
      auto alias = symbol_cast<TypeAliasSymbol>(argSym);
      if (!alias || !alias->type()) return;
      if (isDependent(rewrite.unit_, alias->type())) return;
      rewrite.error(loc, std::format("type '{}' cannot be used prior to '::' "
                                     "because it has no members",
                                     to_string(alias->type())));
    };

    if (isTypeParameter) {
      if (auto substituted = rewrite.substitutedSymbol(copy->symbol)) {
        copy->symbol = binder()->resolveNestedNameSpecifier(substituted);
        if (!copy->symbol) emitNonScopeError(ast->identifierLoc, substituted);
      }
    } else if (!copy->symbol) {
      resolveDependentQualifier(copy);
    }
  } else if (copy->symbol && copy->symbol->type() &&
             isDependent(rewrite.unit_, copy->symbol->type())) {
    auto remapped = rewrite.remapSymbol(copy->symbol);
    if (remapped != copy->symbol) {
      if (auto scope = binder()->resolveNestedNameSpecifier(remapped)) {
        copy->symbol = scope;
        return copy;
      }
    }
    resolveDependentQualifier(copy);
  }

  return copy;
}

void ASTRewriter::NestedNameSpecifierVisitor::resolveDependentQualifier(
    SimpleNestedNameSpecifierAST* copy) {
  if (!copy->identifier) return;

  Symbol* resolved = lookupType(copy->nestedNameSpecifier, copy->identifier);
  if (!resolved || resolved == copy->symbol) return;

  if (resolved->type() && isDependent(rewrite.unit_, resolved->type())) return;

  if (auto scope = binder()->resolveNestedNameSpecifier(resolved)) {
    copy->symbol = scope;
  }
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = DecltypeNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  if (copy->decltypeSpecifier) {
    if (auto classType = type_cast<ClassType>(copy->decltypeSpecifier->type)) {
      copy->symbol = classType->symbol();
    } else if (auto enumType =
                   type_cast<EnumType>(copy->decltypeSpecifier->type)) {
      copy->symbol = enumType->symbol();
    } else if (auto scopedEnumType =
                   type_cast<ScopedEnumType>(copy->decltypeSpecifier->type)) {
      copy->symbol = scopedEnumType->symbol();
    }
  }

  return copy;
}

namespace {
[[nodiscard]] auto templateNameOfTypeId(TypeIdAST* typeId) -> Symbol* {
  if (!typeId) return nullptr;
  for (auto spec : ListView{typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!named) continue;
    if (!ast_cast<NameIdAST>(named->unqualifiedId)) return nullptr;
    return template_name_symbol(named->symbol);
  }
  return nullptr;
}
}  // namespace

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = TemplateNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->templateId =
      ast_cast<SimpleTemplateIdAST>(rewrite.unqualifiedId(ast->templateId));
  copy->scopeLoc = ast->scopeLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  bool hasDependentArgs = false;
  if (copy->templateId) {
    for (auto arg : ListView{copy->templateId->templateArgumentList}) {
      if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
        if (auto templateName = templateNameOfTypeId(typeArg->typeId)) {
          if (!symbol_cast<TemplateTypeParameterSymbol>(templateName)) continue;
          hasDependentArgs = true;
          break;
        }
        if (isDependent(rewrite.unit_, typeArg->typeId)) {
          hasDependentArgs = true;
          break;
        }
      }
      if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
        if (isDependent(rewrite.unit_, exprArg->expression)) {
          hasDependentArgs = true;
          break;
        }
      }
    }
  }

  if (symbol_cast<TypeAliasSymbol>(copy->templateId->symbol)) {
    auto instance =
        binder()->resolve(copy->nestedNameSpecifier, copy->templateId, true);
    if (auto alias = symbol_cast<TypeAliasSymbol>(instance)) {
      copy->symbol = alias;
      if (auto classType = unqualified_cast<ClassType>(alias->type())) {
        copy->symbol = classType->symbol();
      }
    } else {
      copy->symbol = symbol_cast<ScopeSymbol>(instance);
    }
    return copy;
  }

  if (hasDependentArgs) return copy;

  if (!copy->templateId->symbol && copy->templateId->identifier) {
    if (auto resolved =
            lookupType(copy->nestedNameSpecifier, copy->templateId->identifier))
      copy->templateId->symbol = resolved;
  }

  if (!copy->templateId->symbol && copy->templateId->identifier &&
      !isDependentTypeParameterSymbol(copy->nestedNameSpecifier
                                          ? copy->nestedNameSpecifier->symbol
                                          : nullptr)) {
    auto scopeSymbol =
        copy->nestedNameSpecifier ? copy->nestedNameSpecifier->symbol : nullptr;
    rewrite.error(copy->templateId->identifierLoc,
                  std::format("no member named '{}' in '{}'",
                              copy->templateId->identifier->value(),
                              scopeSymbol ? to_string(scopeSymbol->type())
                                          : std::string{"<error-type>"}));
  }

  if (auto primaryClass = symbol_cast<ClassSymbol>(copy->templateId->symbol)) {
    auto instance = ASTRewriter::instantiate(
        rewrite.unit_, copy->templateId->templateArgumentList, primaryClass,
        copy->templateId->identifierLoc);
    copy->symbol = symbol_cast<ClassSymbol>(instance);
  }

  if (auto cls = symbol_cast<ClassSymbol>(copy->symbol)) {
    translationUnit()->typeTraits().requireCompleteClass(cls);
  }

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = TypeTemplateArgumentAST::create(arena());

  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = ExpressionTemplateArgumentAST::create(arena());

  copy->expression = rewrite.expression(ast->expression);

  return copy;
}
}  // namespace cxx
