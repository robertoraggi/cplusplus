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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ASTRewriter::SpecifierVisitor {
  ASTRewriter& rewrite;
  TemplateDeclarationAST* templateHead = nullptr;

  SpecifierVisitor(ASTRewriter& rewrite, TemplateDeclarationAST* templateHead)
      : rewrite(rewrite), templateHead(templateHead) {}

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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

  [[nodiscard]] auto operator()(BuiltinTypeSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(UnaryBuiltinTypeSpecifierAST* ast)
      -> SpecifierAST*;

  [[nodiscard]] auto operator()(BinaryBuiltinTypeSpecifierAST* ast)
      -> SpecifierAST*;

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

  [[nodiscard]] auto operator()(AtomicQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(RestrictQualifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierAST*;

  [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast) -> SpecifierAST*;
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

auto ASTRewriter::specifier(SpecifierAST* ast,
                            TemplateDeclarationAST* templateHead)
    -> SpecifierAST* {
  if (!ast) return {};
  auto specifier = visit(SpecifierVisitor{*this, templateHead}, ast);
  return specifier;
}

auto ASTRewriter::attributeSpecifier(AttributeSpecifierAST* ast)
    -> AttributeSpecifierAST* {
  if (!ast) return {};
  return visit(AttributeSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::attributeToken(AttributeTokenAST* ast) -> AttributeTokenAST* {
  if (!ast) return {};
  return visit(AttributeTokenVisitor{*this}, ast);
}

auto ASTRewriter::baseSpecifier(BaseSpecifierAST* ast) -> BaseSpecifierAST* {
  if (!ast) return {};

  auto copy = BaseSpecifierAST::create(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->virtualOrAccessLoc = ast->virtualOrAccessLoc;
  copy->otherVirtualOrAccessLoc = ast->otherVirtualOrAccessLoc;
  copy->nestedNameSpecifier = nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = unqualifiedId(ast->unqualifiedId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
  copy->isVirtual = ast->isVirtual;
  copy->isVariadic = ast->isVariadic;
  copy->accessSpecifier = ast->accessSpecifier;
  copy->symbol = ast->symbol;

  binder_.bind(ast);

  return copy;
}

auto ASTRewriter::enumerator(EnumeratorAST* ast) -> EnumeratorAST* {
  if (!ast) return {};

  auto copy = EnumeratorAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->equalLoc = ast->equalLoc;
  copy->expression = expression(ast->expression);
  copy->identifier = ast->identifier;

  auto interp = ASTInterpreter{unit_};
  auto value = interp.evaluate(copy->expression);

  auto type = binder().scope()->type();

  binder_.bind(copy, type, std::move(value));

  return copy;
}

auto ASTRewriter::attributeArgumentClause(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseAST* {
  if (!ast) return {};

  auto copy = AttributeArgumentClauseAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::attribute(AttributeAST* ast) -> AttributeAST* {
  if (!ast) return {};

  auto copy = AttributeAST::create(arena());

  copy->attributeToken = attributeToken(ast->attributeToken);
  copy->attributeArgumentClause =
      attributeArgumentClause(ast->attributeArgumentClause);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::attributeUsingPrefix(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixAST* {
  if (!ast) return {};

  auto copy = AttributeUsingPrefixAST::create(arena());

  copy->usingLoc = ast->usingLoc;
  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::typeId(TypeIdAST* ast) -> TypeIdAST* {
  if (!ast) return {};

  auto copy = TypeIdAST::create(arena());

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = specifier(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = declarator(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->type = declaratorType;

  return copy;
}

auto ASTRewriter::splicer(SplicerAST* ast) -> SplicerAST* {
  if (!ast) return {};

  auto copy = SplicerAST::create(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->colonLoc = ast->colonLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->expression = expression(ast->expression);
  copy->secondColonLoc = ast->secondColonLoc;
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = TypedefSpecifierAST::create(arena());

  copy->typedefLoc = ast->typedefLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = FriendSpecifierAST::create(arena());

  copy->friendLoc = ast->friendLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ConstevalSpecifierAST::create(arena());

  copy->constevalLoc = ast->constevalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ConstinitSpecifierAST::create(arena());

  copy->constinitLoc = ast->constinitLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ConstexprSpecifierAST::create(arena());

  copy->constexprLoc = ast->constexprLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = InlineSpecifierAST::create(arena());

  copy->inlineLoc = ast->inlineLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(NoreturnSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = NoreturnSpecifierAST::create(arena());

  copy->noreturnLoc = ast->noreturnLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = StaticSpecifierAST::create(arena());

  copy->staticLoc = ast->staticLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ExternSpecifierAST::create(arena());

  copy->externLoc = ast->externLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(RegisterSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = RegisterSpecifierAST::create(arena());

  copy->registerLoc = ast->registerLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ThreadLocalSpecifierAST::create(arena());

  copy->threadLocalLoc = ast->threadLocalLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ThreadSpecifierAST::create(arena());

  copy->threadLoc = ast->threadLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = MutableSpecifierAST::create(arena());

  copy->mutableLoc = ast->mutableLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = VirtualSpecifierAST::create(arena());

  copy->virtualLoc = ast->virtualLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ExplicitSpecifierAST::create(arena());

  copy->explicitLoc = ast->explicitLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = AutoTypeSpecifierAST::create(arena());

  copy->autoLoc = ast->autoLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = VoidTypeSpecifierAST::create(arena());

  copy->voidLoc = ast->voidLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = SizeTypeSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = SignTypeSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(BuiltinTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = BuiltinTypeSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(
    UnaryBuiltinTypeSpecifierAST* ast) -> SpecifierAST* {
  auto copy = UnaryBuiltinTypeSpecifierAST::create(arena());

  copy->builtinLoc = ast->builtinLoc;
  copy->builtinKind = ast->builtinKind;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(
    BinaryBuiltinTypeSpecifierAST* ast) -> SpecifierAST* {
  auto copy = BinaryBuiltinTypeSpecifierAST::create(arena());

  copy->builtinLoc = ast->builtinLoc;
  copy->builtinKind = ast->builtinKind;
  copy->lparenLoc = ast->lparenLoc;
  copy->leftTypeId = rewrite.typeId(ast->leftTypeId);
  copy->commaLoc = ast->commaLoc;
  copy->rightTypeId = rewrite.typeId(ast->rightTypeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = IntegralTypeSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierAST* {
  auto copy = FloatingPointTypeSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ComplexTypeSpecifierAST::create(arena());

  copy->complexLoc = ast->complexLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = NamedTypeSpecifierAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  copy->symbol = ast->symbol;

  if (auto typeParameter = symbol_cast<TypeParameterSymbol>(ast->symbol)) {
    auto paramType = type_cast<TypeParameterType>(ast->symbol->type());
    const auto& args = rewrite.templateArguments_;
    if (paramType && paramType->depth() == rewrite.depth_ &&
        paramType->index() < args.size()) {
      auto index = paramType->index();

      if (auto sym = std::get_if<Symbol*>(&args[index])) {
        if (auto pack = symbol_cast<ParameterPackSymbol>(*sym)) {
          if (rewrite.elementIndex_.has_value()) {
            auto elemIdx = *rewrite.elementIndex_;
            if (elemIdx < static_cast<int>(pack->elements().size())) {
              copy->symbol = pack->elements()[elemIdx];
            }
          }
        } else {
          copy->symbol = *sym;
        }
      }
    }
  } else {
    // If the symbol is unresolved, we need to resolve it.
    if (auto s =
            binder()->resolve(copy->nestedNameSpecifier, copy->unqualifiedId,
                              /*checkTemplates=*/true)) {
      copy->symbol = s;
    }
  }

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = AtomicTypeSpecifierAST::create(arena());

  copy->atomicLoc = ast->atomicLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(UnderlyingTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = UnderlyingTypeSpecifierAST::create(arena());

  copy->underlyingTypeLoc = ast->underlyingTypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ElaboratedTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ElaboratedTypeSpecifierAST::create(arena());

  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->classKey = ast->classKey;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  // copy->symbol = ast->symbol;

#if false
  auto decl = symbol_cast<ClassSymbol>(ast->symbol);

  if (auto classSpec = decl->declaration()) {
    auto newClassSpec =
        ast_cast<ClassSpecifierAST>(rewrite.specifier(classSpec));

    copy->symbol = newClassSpec->symbol;
  }
#endif

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = DecltypeAutoSpecifierAST::create(arena());

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->autoLoc = ast->autoLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = DecltypeSpecifierAST::create(arena());

  copy->decltypeLoc = ast->decltypeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  if (copy->expression) {
    rewrite.check(copy->expression);
    rewrite.binder().bind(copy);
  }
  if (!copy->type) {
    copy->type = ast->type;
  }

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(PlaceholderTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = PlaceholderTypeSpecifierAST::create(arena());

  copy->typeConstraint = rewrite.typeConstraint(ast->typeConstraint);
  copy->specifier = rewrite.specifier(ast->specifier);

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = ConstQualifierAST::create(arena());

  copy->constLoc = ast->constLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = VolatileQualifierAST::create(arena());

  copy->volatileLoc = ast->volatileLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(AtomicQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = AtomicQualifierAST::create(arena());

  copy->atomicLoc = ast->atomicLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierAST* {
  auto copy = RestrictQualifierAST::create(arena());

  copy->restrictLoc = ast->restrictLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = EnumSpecifierAST::create(arena());

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite.specifier(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->lbraceLoc = ast->lbraceLoc;

  auto _ = Binder::ScopeGuard{binder()};

  binder()->bind(copy, typeSpecifierListCtx);

  for (auto enumeratorList = &copy->enumeratorList;
       auto node : ListView{ast->enumeratorList}) {
    auto value = rewrite.enumerator(node);
    *enumeratorList = make_list_node(arena(), value);
    enumeratorList = &(*enumeratorList)->next;
  }

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = ClassSpecifierAST::create(arena());

  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->finalLoc = ast->finalLoc;
  copy->colonLoc = ast->colonLoc;

  // ### TODO: use Binder::bind()
  auto _ = Binder::ScopeGuard{binder()};
  auto location = ast->symbol->location();
  auto className = ast->symbol->name();

  auto classSymbol = control()->newClassSymbol(binder()->scope(), location);
  copy->symbol = classSymbol;

  classSymbol->setName(className);
  classSymbol->setIsUnion(ast->symbol->isUnion());
  classSymbol->setFinal(ast->isFinal);
  classSymbol->setDeclaration(copy);
  classSymbol->setTemplateDeclaration(templateHead);

  if (ast->symbol == rewrite.binder().instantiatingSymbol()) {
    ast->symbol->addSpecialization(rewrite.templateArguments(), classSymbol);
  } else {
    // If we are not instantiating a template, we can add the class symbol to
    // the scope.
    binder()->declaringScope()->addSymbol(classSymbol);
  }

  // enter the class scope
  binder()->setScope(classSymbol);

  for (auto baseSpecifierList = &copy->baseSpecifierList;
       auto node : ListView{ast->baseSpecifierList}) {
    if (node->isVariadic) {
      ParameterPackSymbol* pack = nullptr;
      if (node->unqualifiedId) {
        for (const auto& arg : rewrite.templateArguments_) {
          if (auto sym = std::get_if<Symbol*>(&arg)) {
            if (auto p = symbol_cast<ParameterPackSymbol>(*sym)) {
              pack = p;
              break;
            }
          }
        }
      }

      if (pack && !pack->elements().empty()) {
        int n = static_cast<int>(pack->elements().size());
        for (int i = 0; i < n; ++i) {
          auto elem = pack->elements()[i];

          Symbol* baseResolvedSym = elem;
          if (auto typeAlias = symbol_cast<TypeAliasSymbol>(baseResolvedSym)) {
            if (auto classType = type_cast<ClassType>(
                    control()->remove_cv(typeAlias->type()))) {
              baseResolvedSym = classType->symbol();
            }
          }

          if (!baseResolvedSym || !baseResolvedSym->isClass()) continue;

          auto value = BaseSpecifierAST::create(arena());
          value->isVariadic = false;
          value->accessSpecifier = node->accessSpecifier;
          value->isVirtual = node->isVirtual;

          auto location = node->unqualifiedId
                              ? node->unqualifiedId->firstSourceLocation()
                              : SourceLocation{};
          auto baseClassSym =
              control()->newBaseClassSymbol(classSymbol, location);
          baseClassSym->setSymbol(baseResolvedSym);
          baseClassSym->setName(baseResolvedSym->name());
          value->symbol = baseClassSym;
          classSymbol->addBaseClass(baseClassSym);

          *baseSpecifierList = make_list_node(arena(), value);
          baseSpecifierList = &(*baseSpecifierList)->next;
        }

        continue;
      }

      if (pack) {
        continue;
      }
    }

    auto value = rewrite.baseSpecifier(node);
    *baseSpecifierList = make_list_node(arena(), value);
    baseSpecifierList = &(*baseSpecifierList)->next;

    if (value->symbol) {
      classSymbol->addBaseClass(value->symbol);
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  struct DelayedFunction {
    FunctionDefinitionAST* newAst = nullptr;
    FunctionDefinitionAST* oldAst = nullptr;
  };

  std::vector<DelayedFunction> delayedFunctions;

  rewrite.setRestrictedToDeclarations(true);

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;

    auto newDecl = value;
    auto oldDecl = node;

    while (auto newTempl = ast_cast<TemplateDeclarationAST>(newDecl)) {
      auto oldTempl = ast_cast<TemplateDeclarationAST>(oldDecl);
      newDecl = newTempl->declaration;
      oldDecl = oldTempl->declaration;
    }

    if (auto newFunc = ast_cast<FunctionDefinitionAST>(newDecl)) {
      delayedFunctions.push_back(
          {newFunc, ast_cast<FunctionDefinitionAST>(oldDecl)});
    }
  }

  rewrite.setRestrictedToDeclarations(false);

  for (const auto& [newAst, oldAst] : delayedFunctions) {
    auto _ = Binder::ScopeGuard{binder()};

    auto functionDeclarator = getFunctionPrototype(newAst->declarator);

    if (auto params = functionDeclarator->parameterDeclarationClause) {
      binder()->setScope(params->functionParametersSymbol);
    } else {
      binder()->setScope(newAst->symbol);
    }

    newAst->functionBody = rewrite.functionBody(oldAst->functionBody);
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
  auto copy = TypenameSpecifierAST::create(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierAST* {
  auto copy = SplicerTypeSpecifierAST::create(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->splicer = rewrite.splicer(ast->splicer);

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = CxxAttributeAST::create(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->lbracket2Loc = ast->lbracket2Loc;
  copy->attributeUsingPrefix =
      rewrite.attributeUsingPrefix(ast->attributeUsingPrefix);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attribute(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->rbracket2Loc = ast->rbracket2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = GccAttributeAST::create(arena());

  copy->attributeLoc = ast->attributeLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->lparen2Loc = ast->lparen2Loc;
  copy->rparenLoc = ast->rparenLoc;
  copy->rparen2Loc = ast->rparen2Loc;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierAST* {
  auto copy = AlignasAttributeAST::create(arena());

  copy->alignasLoc = ast->alignasLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierAST* {
  auto copy = AlignasTypeAttributeAST::create(arena());

  copy->alignasLoc = ast->alignasLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierAST* {
  auto copy = AsmAttributeAST::create(arena());

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = ScopedAttributeTokenAST::create(arena());

  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->attributeNamespace = ast->attributeNamespace;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenAST* {
  auto copy = SimpleAttributeTokenAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

}  // namespace cxx
