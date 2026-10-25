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

#include <cxx/access_control.h>
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
struct ASTRewriter::UnqualifiedIdVisitor : VisitorBase {
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
  void substituteTemplateTemplateParameter(SimpleTemplateIdAST* copy);
};

struct ASTRewriter::NestedNameSpecifierVisitor : VisitorBase {
  ClassSymbol* memberScope = nullptr;

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  void resolveDependentQualifier(SimpleNestedNameSpecifierAST* copy);

  [[nodiscard]] auto designatedScope(Symbol* entity, SourceLocation loc)
      -> Symbol*;

  void reportNonScopeQualifier(const Type* type, SourceLocation loc);

  [[nodiscard]] auto lookupType(NestedNameSpecifierAST* prefix,
                                const Identifier* id, Symbol* lexicalBinding,
                                bool& ambiguous,
                                NameUse use = NameUse::kOrdinary) const
      -> Symbol*;
};

auto ASTRewriter::NestedNameSpecifierVisitor::lookupType(
    NestedNameSpecifierAST* prefix, const Identifier* id,
    Symbol* lexicalBinding, bool& ambiguous, NameUse use) const -> Symbol* {
  ambiguous = false;
  auto isType = [](Symbol* s) { return is_type(s); };
  if (prefix) return qualifiedLookupType(prefix->symbol, id, &ambiguous, use);
  if (memberScope) {
    auto resolved = qualifiedLookupType(memberScope, id, &ambiguous, use);
    if (resolved || ambiguous) return resolved;
    if (lexicalBinding) return lexicalBinding;
  }
  for (auto scope = binder()->scope(); scope; scope = scope->parent()) {
    if (auto resolved = qualifiedLookup(scope, id, isType)) return resolved;
  }
  return nullptr;
}

struct ASTRewriter::TemplateArgumentVisitor : VisitorBase {
  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
};

auto ASTRewriter::unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdAST* {
  if (!ast) return {};
  return visit(UnqualifiedIdVisitor{*this}, ast);
}

auto ASTRewriter::nestedNameSpecifier(NestedNameSpecifierAST* ast,
                                      ClassSymbol* memberScope)
    -> NestedNameSpecifierAST* {
  if (!ast) return {};
  return visit(NestedNameSpecifierVisitor{*this, memberScope}, ast);
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

auto ASTRewriter::rewriteTemplateArgumentList(
    List<TemplateArgumentAST*>* source) -> List<TemplateArgumentAST*>* {
  List<TemplateArgumentAST*>* result = nullptr;
  ListAppender<TemplateArgumentAST> append{arena(), result};
  for (auto node : ListView{source}) {
    if (shouldStopSubstitution()) break;
    if (expandPackArgument(node, append)) continue;
    append(templateArgument(node));
  }
  return result;
}

auto ASTRewriter::expandPackArgument(TemplateArgumentAST* argument,
                                     ListAppender<TemplateArgumentAST>& append)
    -> bool {
  if (auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument)) {
    auto pattern = typeArgument->typeId;
    auto pack = expandedParameterPack(pattern);
    if (!pack) return false;
    if (pack->elements().empty()) return true;
    forEachPackElement(
        pattern, typeArgument->firstSourceLocation(),
        [&] {
          append(TypeTemplateArgumentAST::create(arena(), typeId(pattern)));
        },
        pack);
    return true;
  }

  auto expressionArgument = ast_cast<ExpressionTemplateArgumentAST>(argument);
  if (!expressionArgument) return false;

  auto expansion =
      ast_cast<PackExpansionExpressionAST>(expressionArgument->expression);
  if (!expansion) return false;

  auto pattern = expansion->expression;
  auto pack = findReferencedParameterPack(pattern);
  if (!pack) return false;
  if (pack->elements().empty()) return true;
  forEachPackElement(
      pattern, expansion->ellipsisLoc,
      [&] {
        append(ExpressionTemplateArgumentAST::create(arena(),
                                                     expression(pattern)));
      },
      pack);
  return true;
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

  copy->templateArgumentList =
      rewrite.rewriteTemplateArgumentList(ast->templateArgumentList);

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  if (is_member_template(ast->symbol)) {
    copy->symbol = rewrite.remapSymbol(ast->symbol);
    if (auto primary = primary_template_of(copy->symbol))
      copy->symbol = primary;
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

  copy->templateArgumentList =
      rewrite.rewriteTemplateArgumentList(ast->templateArgumentList);

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = OperatorFunctionTemplateIdAST::create(arena());

  copy->operatorFunctionId = ast_cast<OperatorFunctionIdAST>(
      rewrite.unqualifiedId(ast->operatorFunctionId));
  copy->lessLoc = ast->lessLoc;

  copy->templateArgumentList =
      rewrite.rewriteTemplateArgumentList(ast->templateArgumentList);

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
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier, memberScope);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->scopeLoc = ast->scopeLoc;

  if (!copy->nestedNameSpecifier && memberScope) {
    resolveDependentQualifier(copy);
    if (!copy->symbol) return copy;
  }

  const bool isTypeParameter =
      symbol_cast<TypeParameterSymbol>(copy->symbol) ||
      symbol_cast<ConstraintTypeParameterSymbol>(copy->symbol);

  auto needsSubstitution = !copy->symbol || isTypeParameter;
  if (needsSubstitution && copy->identifier) {
    if (isTypeParameter) {
      if (auto substituted = rewrite.substitutedSymbol(copy->symbol)) {
        copy->symbol = binder()->resolveNestedNameSpecifier(substituted);
        if (!copy->symbol)
          (void)designatedScope(substituted, ast->identifierLoc);
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

auto ASTRewriter::NestedNameSpecifierVisitor::designatedScope(
    Symbol* entity, SourceLocation loc) -> Symbol* {
  if (auto scope = binder()->resolveNestedNameSpecifier(entity)) return scope;
  auto type = entity ? entity->type() : nullptr;
  if (!type || isDependent(rewrite.unit_, type)) return entity;
  reportNonScopeQualifier(type, loc);
  return nullptr;
}

void ASTRewriter::NestedNameSpecifierVisitor::reportNonScopeQualifier(
    const Type* type, SourceLocation loc) {
  rewrite.error(loc, std::format("type '{}' cannot be used prior to '::' "
                                 "because it has no members",
                                 to_diag_string(type)));
}

void ASTRewriter::NestedNameSpecifierVisitor::resolveDependentQualifier(
    SimpleNestedNameSpecifierAST* copy) {
  if (!copy->identifier) return;

  bool ambiguous = false;
  auto resolved = lookupType(copy->nestedNameSpecifier, copy->identifier,
                             copy->symbol, ambiguous);
  if (ambiguous) {
    copy->symbol = nullptr;
    rewrite.error(copy->identifierLoc,
                  std::format("reference to '{}' is ambiguous",
                              copy->identifier->value()));
    return;
  }
  if (!resolved || resolved == copy->symbol) return;

  if (resolved->type() && isDependent(rewrite.unit_, resolved->type())) {
    copy->symbol = resolved;
    return;
  }

  if (!copy->nestedNameSpecifier && memberScope)
    (void)checkMemberAccess(translationUnit(), binder()->scope(), resolved,
                            memberScope, memberScope, copy->identifierLoc);
  copy->symbol = designatedScope(resolved, copy->identifierLoc);
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = DecltypeNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  if (!copy->decltypeSpecifier) return copy;

  auto type = copy->decltypeSpecifier->type;
  if (!type || isDependent(rewrite.unit_, type)) return copy;

  copy->symbol = binder()->scopeOfType(type);
  if (!copy->symbol) reportNonScopeQualifier(type, copy->firstSourceLocation());

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = TemplateNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier, memberScope);
  copy->templateLoc = ast->templateLoc;
  copy->templateId =
      ast_cast<SimpleTemplateIdAST>(rewrite.unqualifiedId(ast->templateId));
  copy->scopeLoc = ast->scopeLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  const bool memberQualifier = !copy->nestedNameSpecifier && memberScope;
  if (memberQualifier) copy->symbol = nullptr;

  if (copy->templateId->identifier &&
      (memberQualifier || !copy->templateId->symbol ||
       (copy->nestedNameSpecifier && copy->nestedNameSpecifier->symbol))) {
    bool ambiguous = false;
    auto resolved =
        lookupType(copy->nestedNameSpecifier, copy->templateId->identifier,
                   copy->templateId->symbol, ambiguous, NameUse::kTemplateName);
    if (ambiguous) {
      copy->templateId->symbol = nullptr;
      rewrite.error(copy->templateId->identifierLoc,
                    std::format("reference to '{}' is ambiguous",
                                copy->templateId->identifier->value()));
      return copy;
    }
    if (resolved) {
      if (!copy->nestedNameSpecifier && memberScope)
        (void)checkMemberAccess(translationUnit(), binder()->scope(), resolved,
                                memberScope, memberScope,
                                copy->templateId->identifierLoc);
      copy->templateId->symbol = templated_symbol(resolved);
    }
  }

  if (symbol_cast<TypeAliasSymbol>(copy->templateId->symbol)) {
    auto instance =
        binder()->resolve(copy->nestedNameSpecifier, copy->templateId, true,
                          copy->templateId->symbol);
    copy->symbol =
        designatedScope(instance, copy->templateId->firstSourceLocation());
    return copy;
  }

  if (hasDependentTemplateArguments(rewrite.unit_, copy->templateId))
    return copy;

  if (isDependent(rewrite.unit_, copy->nestedNameSpecifier)) return copy;

  if (!copy->templateId->symbol && copy->templateId->identifier &&
      !isDependentTypeParameterSymbol(copy->nestedNameSpecifier
                                          ? copy->nestedNameSpecifier->symbol
                                          : nullptr)) {
    auto scopeSymbol =
        copy->nestedNameSpecifier ? copy->nestedNameSpecifier->symbol : nullptr;
    rewrite.error(copy->templateId->identifierLoc,
                  std::format("no member named '{}' in '{}'",
                              copy->templateId->identifier->value(),
                              scopeSymbol ? to_diag_string(scopeSymbol->type())
                                          : std::string{"<error-type>"}));
  }

  if (auto primaryClass = symbol_cast<ClassSymbol>(copy->templateId->symbol)) {
    auto instance = binder()->resolve(copy->nestedNameSpecifier,
                                      copy->templateId, true, primaryClass);
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
