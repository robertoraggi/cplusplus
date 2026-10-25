// Generated file by: gen_ast_printer_cc.ts
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

#include <cxx/ast_printer.h>

// cxx
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/printing.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <format>

namespace cxx {

ASTPrinter::ASTPrinter(TranslationUnit* unit, std::format_context::iterator out)
    : unit_(unit), out_(out) {}

void ASTPrinter::operator()(AST* ast) { accept(ast); }

void ASTPrinter::accept(AST* ast, std::string_view field) {
  if (!ast) return;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  if (!field.empty()) {
    out_ = std::format_to(out_, "{}: ", field);
  }
  ast->accept(this);
  --indent_;
}

void ASTPrinter::accept(const Identifier* id, std::string_view field) {
  if (!id) return;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  if (!field.empty()) out_ = std::format_to(out_, "{}: ", field);
  out_ = std::format_to(out_, "{}\n", id->value());
  --indent_;
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  out_ = std::format_to(out_, "{}\n", "translation-unit");
  if (ast->declarationList) {
    ++indent_;
    const auto builtinsFileId = unit_->preprocessor()->builtinsFileId();
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      auto loc = firstSourceLocation(node);
      if (unit_->tokenAt(loc).fileId() == builtinsFileId) continue;
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-unit");
  accept(ast->globalModuleFragment, "global-module-fragment");
  accept(ast->moduleDeclaration, "module-declaration");
  if (ast->declarationList) {
    ++indent_;
    const auto builtinsFileId = unit_->preprocessor()->builtinsFileId();
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      auto loc = firstSourceLocation(node);
      if (unit_->tokenAt(loc).fileId() == builtinsFileId) continue;
      accept(node);
    }
    --indent_;
  }
  accept(ast->privateModuleFragment, "private-module-fragment");
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->initDeclaratorList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "init-declarator-list");
    for (auto node : ListView{ast->initDeclaratorList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-declaration");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->asmQualifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "asm-qualifier-list");
    for (auto node : ListView{ast->asmQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->outputOperandList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "output-operand-list");
    for (auto node : ListView{ast->outputOperandList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->inputOperandList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "input-operand-list");
    for (auto node : ListView{ast->inputOperandList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->clobberList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "clobber-list");
    for (auto node : ListView{ast->clobberList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->gotoLabelList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "goto-label-list");
    for (auto node : ListView{ast->gotoLabelList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  out_ = std::format_to(out_, "{}\n", "namespace-alias-definition");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "using-declaration");
  if (ast->usingDeclaratorList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "using-declarator-list");
    for (auto node : ListView{ast->usingDeclaratorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "using-enum-declaration");
  accept(ast->enumTypeSpecifier, "enum-type-specifier");
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  out_ = std::format_to(out_, "{}\n", "using-directive");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "static-assert-declaration");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "alias-declaration");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->gnuAttributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "gnu-attribute-list");
    for (auto node : ListView{ast->gnuAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "opaque-enum-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  out_ = std::format_to(out_, "{}\n", "function-definition");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->functionBody, "function-body");
}

void ASTPrinter::visit(TemplateDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "template-declaration");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "depth: {}\n", ast->depth);
  --indent_;
  if (ast->templateParameterList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  out_ = std::format_to(out_, "{}\n", "concept-definition");
  accept(ast->identifier, "identifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  out_ = std::format_to(out_, "{}\n", "deduction-guide");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->explicitSpecifier, "explicit-specifier");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "explicit-instantiation");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "export-declaration");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportCompoundDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "export-compound-declaration");
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "linkage-specification");
  if (ast->stringLiteral) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "string-literal: {}\n",
                          ast->stringLiteral->value());
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  out_ = std::format_to(out_, "{}\n", "namespace-definition");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-inline: {}\n", ast->isInline);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->nestedNamespaceSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "nested-namespace-specifier-list");
    for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->extraAttributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "extra-attribute-list");
    for (auto node : ListView{ast->extraAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "empty-declaration");
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "attribute-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-import-declaration");
  accept(ast->importName, "import-name");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "parameter-declaration");
  accept(ast->identifier, "identifier");
  if (ast->isThisIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "is-this-introduced: {}\n", ast->isThisIntroduced);
    --indent_;
  }
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AccessDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "access-declaration");
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "access-specifier: {}\n",
                          Token::spell(ast->accessSpecifier));
    --indent_;
  }
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "for-range-declaration");
}

void ASTPrinter::visit(StructuredBindingDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "structured-binding-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->bindingList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "binding-list");
    for (auto node : ListView{ast->bindingList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(AsmOperandAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-operand");
  accept(ast->symbolicName, "symbolic-name");
  if (ast->constraintLiteral) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "constraint-literal: {}\n",
                          ast->constraintLiteral->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AsmQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-qualifier");
  if (ast->qualifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "qualifier: {}\n", Token::spell(ast->qualifier));
    --indent_;
  }
}

void ASTPrinter::visit(AsmClobberAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-clobber");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(AsmGotoLabelAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-goto-label");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SplicerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "splicer");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GlobalModuleFragmentAST* ast) {
  out_ = std::format_to(out_, "{}\n", "global-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PrivateModuleFragmentAST* ast) {
  out_ = std::format_to(out_, "{}\n", "private-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-declaration");
  accept(ast->moduleName, "module-name");
  accept(ast->modulePartition, "module-partition");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleNameAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-name");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModuleQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-qualifier");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModulePartitionAST* ast) {
  out_ = std::format_to(out_, "{}\n", "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(ImportNameAST* ast) {
  out_ = std::format_to(out_, "{}\n", "import-name");
  accept(ast->modulePartition, "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(InitDeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "init-declarator");
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "declarator");
  if (ast->ptrOpList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "ptr-op-list");
    for (auto node : ListView{ast->ptrOpList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->coreDeclarator, "core-declarator");
  if (ast->declaratorChunkList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declarator-chunk-list");
    for (auto node : ListView{ast->declaratorChunkList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "using-declarator");
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "enumerator");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(HandlerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "handler");
  accept(ast->exceptionDeclaration, "exception-declaration");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "base-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->isVirtual) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual: {}\n", ast->isVirtual);
    --indent_;
  }
  if (ast->isVariadic) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-variadic: {}\n", ast->isVariadic);
    --indent_;
  }
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "access-specifier: {}\n",
                          Token::spell(ast->accessSpecifier));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(RequiresClauseAST* ast) {
  out_ = std::format_to(out_, "{}\n", "requires-clause");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  out_ = std::format_to(out_, "{}\n", "parameter-declaration-clause");
  if (ast->isVariadic) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-variadic: {}\n", ast->isVariadic);
    --indent_;
  }
  if (ast->parameterDeclarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "parameter-declaration-list");
    for (auto node : ListView{ast->parameterDeclarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "trailing-return-type");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(LambdaSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "lambda-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstraintAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-constraint");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AttributeArgumentClauseAST* ast) {
  out_ = std::format_to(out_, "{}\n", "attribute-argument-clause");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "attribute");
  accept(ast->attributeToken, "attribute-token");
  accept(ast->attributeArgumentClause, "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeUsingPrefixAST* ast) {
  out_ = std::format_to(out_, "{}\n", "attribute-using-prefix");
}

void ASTPrinter::visit(NewPlacementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "new-placement");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedNamespaceSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "nested-namespace-specifier");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-inline: {}\n", ast->isInline);
    --indent_;
  }
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "labeled-statement");
  accept(ast->identifier, "identifier");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "case-statement");
  accept(ast->expression, "expression");
  accept(ast->upperExpression, "upper-expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "default-statement");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "expression-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "compound-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->statementList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "statement-list");
    for (auto node : ListView{ast->statementList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "if-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(ConstevalIfStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "consteval-if-statement");
  if (ast->isNot) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-not: {}\n", ast->isNot);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "switch-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "while-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DoStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "do-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "for-range-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
  accept(ast->rangeDeclaration, "range-declaration");
  accept(ast->rangeInitializer, "range-initializer");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ForStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "for-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->expression, "expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "break-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "continue-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "return-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "coroutine-return-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "goto-statement");
  accept(ast->identifier, "identifier");
  if (ast->isIndirect) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-indirect: {}\n", ast->isIndirect);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "declaration-statement");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "try-block-statement");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "handler-list");
    for (auto node : ListView{ast->handlerList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "char-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->literalOperatorCall, "literal-operator-call");
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "bool-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->isTrue) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-true: {}\n", ast->isTrue);
    --indent_;
  }
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "int-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->literalOperatorCall, "literal-operator-call");
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "float-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->literalOperatorCall, "literal-operator-call");
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "nullptr-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", Token::spell(ast->literal));
    --indent_;
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "string-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->encoding != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "encoding: {}\n", Token::spell(ast->encoding));
    --indent_;
  }
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "user-defined-string-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->encoding != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "encoding: {}\n", Token::spell(ast->encoding));
    --indent_;
  }
  accept(ast->literalOperatorCall, "literal-operator-call");
}

void ASTPrinter::visit(ObjectLiteralExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "object-literal-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "this-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
}

void ASTPrinter::visit(PackIndexExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "pack-index-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->packExpression, "pack-expression");
  accept(ast->indexExpression, "index-expression");
}

void ASTPrinter::visit(GenericSelectionExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "generic-selection-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ =
      std::format_to(out_, "matched-assoc-index: {}\n", ast->matchedAssocIndex);
  --indent_;
  accept(ast->expression, "expression");
  if (ast->genericAssociationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "generic-association-list");
    for (auto node : ListView{ast->genericAssociationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedStatementExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "nested-statement-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DefaultInitializerExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "default-initializer-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "nested-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "id-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "lambda-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->captureDefault != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "capture-default: {}\n",
                          Token::spell(ast->captureDefault));
    --indent_;
  }
  if (ast->captureList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "capture-list");
    for (auto node : ListView{ast->captureList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->templateRequiresClause, "template-requires-clause");
  if (ast->expressionAttributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-attribute-list");
    for (auto node : ListView{ast->expressionAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->gnuAtributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "gnu-atribute-list");
    for (auto node : ListView{ast->gnuAtributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->lambdaSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "lambda-specifier-list");
    for (auto node : ListView{ast->lambdaSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(FoldExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "fold-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->foldOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "fold-op: {}\n", Token::spell(ast->foldOp));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(RightFoldExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "right-fold-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LeftFoldExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "left-fold-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(RequiresExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "requires-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->requirementList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "requirement-list");
    for (auto node : ListView{ast->requirementList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(VaArgExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "va-arg-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "subscript-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->indexExpression, "index-expression");
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "call-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstructionAST* ast) {
  out_ = std::format_to(out_, "{}", "type-construction");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeSpecifier, "type-specifier");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedTypeConstructionAST* ast) {
  out_ = std::format_to(out_, "{}", "braced-type-construction");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeSpecifier, "type-specifier");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(SpliceMemberExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "splice-member-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "member-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(PostIncrExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "post-incr-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "cpp-cast-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->castOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "cast-op: {}\n", Token::spell(ast->castOp));
    --indent_;
  }
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BuiltinBitCastExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "builtin-bit-cast-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BuiltinConvertVectorExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "builtin-convert-vector-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(BuiltinOffsetofExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "builtin-offsetof-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->identifier, "identifier");
  accept(ast->typeId, "type-id");
  if (ast->designatorList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "designator-list");
    for (auto node : ListView{ast->designatorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeidExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "typeid-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeidOfTypeExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "typeid-of-type-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SpliceExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "splice-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(GlobalScopeReflectExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "global-scope-reflect-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
}

void ASTPrinter::visit(NamespaceReflectExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "namespace-reflect-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(TypeIdReflectExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "type-id-reflect-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ReflectExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "reflect-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LabelAddressExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "label-address-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "unary-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AwaitExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "await-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "sizeof-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofTypeExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "sizeof-type-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SizeofPackExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "sizeof-pack-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(AlignofTypeExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "alignof-type-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AlignofExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "alignof-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NoexceptExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "noexcept-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "new-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->arrayCookieRequired) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "array-cookie-required: {}\n",
                          ast->arrayCookieRequired);
    --indent_;
  }
  if (ast->hasAlignmentArgument) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "has-alignment-argument: {}\n",
                          ast->hasAlignmentArgument);
    --indent_;
  }
  accept(ast->newPlacement, "new-placement");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->newInitalizer, "new-initalizer");
}

void ASTPrinter::visit(DeleteExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "delete-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->arrayCookieRequired) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "array-cookie-required: {}\n",
                          ast->arrayCookieRequired);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CastExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "cast-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ImplicitCastExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "implicit-cast-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "cast-kind: {}\n", to_string(ast->castKind));
  --indent_;
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ConstExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "const-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "binary-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "conditional-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->condition, "condition");
  accept(ast->iftrueExpression, "iftrue-expression");
  accept(ast->iffalseExpression, "iffalse-expression");
}

void ASTPrinter::visit(YieldExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "yield-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "throw-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "assignment-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(TargetExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "target-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
}

void ASTPrinter::visit(RightExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "right-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
}

void ASTPrinter::visit(CompoundAssignmentExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "compound-assignment-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->isVirtualDispatch) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-virtual-dispatch: {}\n",
                          ast->isVirtualDispatch);
    --indent_;
  }
  accept(ast->targetExpression, "target-expression");
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
  accept(ast->adjustExpression, "adjust-expression");
}

void ASTPrinter::visit(PackExpansionExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "pack-expansion-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DesignatedInitializerClauseAST* ast) {
  out_ = std::format_to(out_, "{}", "designated-initializer-clause");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->designatorList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "designator-list");
    for (auto node : ListView{ast->designatorList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(TypeTraitExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "type-trait-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->typeIdList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-id-list");
    for (auto node : ListView{ast->typeIdList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ConditionExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "condition-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  out_ = std::format_to(out_, "{}", "equal-initializer");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  out_ = std::format_to(out_, "{}", "braced-init-list");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  out_ = std::format_to(out_, "{}", "paren-initializer");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ThreeWayComparisonExpressionAST* ast) {
  out_ = std::format_to(out_, "{}", "three-way-comparison-expression");
  if (ast->type) {
    out_ = std::format_to(out_, " [{} {}]", to_string(ast->valueCategory),
                          formatType(ast->type));
  }
  out_ = std::format_to(out_, "{}", "\n");
  accept(ast->comparison, "comparison");
}

void ASTPrinter::visit(DefaultGenericAssociationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "default-generic-association");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeGenericAssociationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-generic-association");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DotDesignatorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "dot-designator");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SubscriptDesignatorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "subscript-designator");
  accept(ast->expression, "expression");
  accept(ast->upperExpression, "upper-expression");
}

void ASTPrinter::visit(TemplateTypeParameterAST* ast) {
  out_ = std::format_to(out_, "{}\n", "template-type-parameter");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->idExpression, "id-expression");
}

void ASTPrinter::visit(NonTypeTemplateParameterAST* ast) {
  out_ = std::format_to(out_, "{}\n", "non-type-template-parameter");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "index: {}\n", ast->index);
  --indent_;
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TypenameTypeParameterAST* ast) {
  out_ = std::format_to(out_, "{}\n", "typename-type-parameter");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->isSynthesized) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-synthesized: {}\n", ast->isSynthesized);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ConstraintTypeParameterAST* ast) {
  out_ = std::format_to(out_, "{}\n", "constraint-type-parameter");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isSynthesized) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-synthesized: {}\n", ast->isSynthesized);
    --indent_;
  }
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "typedef-specifier");
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "friend-specifier");
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "consteval-specifier");
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "constinit-specifier");
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "constexpr-specifier");
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "inline-specifier");
}

void ASTPrinter::visit(NoreturnSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "noreturn-specifier");
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "static-specifier");
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "extern-specifier");
}

void ASTPrinter::visit(RegisterSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "register-specifier");
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "thread-local-specifier");
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "thread-specifier");
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "mutable-specifier");
}

void ASTPrinter::visit(VirtualSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "virtual-specifier");
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "explicit-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AutoTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "auto-type-specifier");
}

void ASTPrinter::visit(VoidTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "void-type-specifier");
}

void ASTPrinter::visit(SizeTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "size-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(SignTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "sign-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(BuiltinTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "builtin-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(UnaryBuiltinTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "unary-builtin-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(BinaryBuiltinTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "binary-builtin-type-specifier");
  accept(ast->leftTypeId, "left-type-id");
  accept(ast->rightTypeId, "right-type-id");
}

void ASTPrinter::visit(IntegralTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "integral-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(FloatingPointTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "floating-point-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ =
        std::format_to(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(ComplexTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "complex-type-specifier");
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "named-type-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "atomic-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(BitIntTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "bit-int-type-specifier");
  ++indent_;
  out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
  out_ = std::format_to(out_, "bit-count: {}\n", ast->bitCount);
  --indent_;
  accept(ast->sizeExpression, "size-expression");
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "underlying-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "elaborated-type-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "decltype-auto-specifier");
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "decltype-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "placeholder-type-specifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->specifier, "specifier");
}

void ASTPrinter::visit(ConstQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "const-qualifier");
}

void ASTPrinter::visit(VolatileQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "volatile-qualifier");
}

void ASTPrinter::visit(AtomicQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "atomic-qualifier");
}

void ASTPrinter::visit(RestrictQualifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "restrict-qualifier");
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "enum-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->enumeratorList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "enumerator-list");
    for (auto node : ListView{ast->enumeratorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "class-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isFinal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->baseSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "base-specifier-list");
    for (auto node : ListView{ast->baseSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->trailingAttributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "trailing-attribute-list");
    for (auto node : ListView{ast->trailingAttributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "typename-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(SplicerTypeSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "splicer-type-specifier");
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "pointer-operator");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "reference-operator");
  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "ref-op: {}\n", Token::spell(ast->refOp));
    --indent_;
  }
  if (ast->isRestrict) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-restrict: {}\n", ast->isRestrict);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "ptr-to-member-operator");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BitfieldDeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "bitfield-declarator");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->sizeExpression, "size-expression");
  if (ast->trailingAttributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "trailing-attribute-list");
    for (auto node : ListView{ast->trailingAttributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParameterPackAST* ast) {
  out_ = std::format_to(out_, "{}\n", "parameter-pack");
  accept(ast->coreDeclarator, "core-declarator");
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "id-declarator");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  out_ = std::format_to(out_, "{}\n", "nested-declarator");
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(FunctionDeclaratorChunkAST* ast) {
  out_ = std::format_to(out_, "{}\n", "function-declarator-chunk");
  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "ref-op: {}\n", Token::spell(ast->refOp));
    --indent_;
  }
  if (ast->isFinal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->isOverride) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-override: {}\n", ast->isOverride);
    --indent_;
  }
  if (ast->isPure) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pure: {}\n", ast->isPure);
    --indent_;
  }
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->cvQualifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
}

void ASTPrinter::visit(ArrayDeclaratorChunkAST* ast) {
  out_ = std::format_to(out_, "{}\n", "array-declarator-chunk");
  if (ast->typeQualifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-qualifier-list");
    for (auto node : ListView{ast->typeQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NameIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "name-id");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DestructorIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "destructor-id");
  accept(ast->id, "id");
}

void ASTPrinter::visit(DecltypeIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "decltype-id");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(OperatorFunctionIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "operator-function-id");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "literal-operator-id");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(ConversionFunctionIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "conversion-function-id");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SimpleTemplateIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-template-id");
  accept(ast->identifier, "identifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorTemplateIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "literal-operator-template-id");
  accept(ast->literalOperatorId, "literal-operator-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(OperatorFunctionTemplateIdAST* ast) {
  out_ = std::format_to(out_, "{}\n", "operator-function-template-id");
  accept(ast->operatorFunctionId, "operator-function-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GlobalNestedNameSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "global-nested-name-specifier");
}

void ASTPrinter::visit(SimpleNestedNameSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-nested-name-specifier");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
}

void ASTPrinter::visit(DecltypeNestedNameSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "decltype-nested-name-specifier");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(TemplateNestedNameSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "template-nested-name-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-template-introduced: {}\n",
                          ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(DefaultFunctionBodyAST* ast) {
  out_ = std::format_to(out_, "{}\n", "default-function-body");
}

void ASTPrinter::visit(CompoundStatementFunctionBodyAST* ast) {
  out_ = std::format_to(out_, "{}\n", "compound-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "mem-initializer-list");
    for (auto node : ListView{ast->memInitializerList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(TryStatementFunctionBodyAST* ast) {
  out_ = std::format_to(out_, "{}\n", "try-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "mem-initializer-list");
    for (auto node : ListView{ast->memInitializerList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "handler-list");
    for (auto node : ListView{ast->handlerList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DeleteFunctionBodyAST* ast) {
  out_ = std::format_to(out_, "{}\n", "delete-function-body");
}

void ASTPrinter::visit(TypeTemplateArgumentAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-template-argument");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ExpressionTemplateArgumentAST* ast) {
  out_ = std::format_to(out_, "{}\n", "expression-template-argument");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExceptionSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "throw-exception-specifier");
}

void ASTPrinter::visit(NoexceptSpecifierAST* ast) {
  out_ = std::format_to(out_, "{}\n", "noexcept-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SimpleRequirementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundRequirementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "compound-requirement");
  accept(ast->expression, "expression");
  accept(ast->typeConstraint, "type-constraint");
}

void ASTPrinter::visit(TypeRequirementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-requirement");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(NestedRequirementAST* ast) {
  out_ = std::format_to(out_, "{}\n", "nested-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "new-paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "new-braced-initializer");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ParenMemInitializerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "paren-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->expressionList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedMemInitializerAST* ast) {
  out_ = std::format_to(out_, "{}\n", "braced-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "this-lambda-capture");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(DerefThisLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "deref-this-lambda-capture");
}

void ASTPrinter::visit(SimpleLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(RefLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "ref-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(RefInitLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "ref-init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(InitLambdaCaptureAST* ast) {
  out_ = std::format_to(out_, "{}\n", "init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "ellipsis-exception-declaration");
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  out_ = std::format_to(out_, "{}\n", "type-exception-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(CxxAttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "cxx-attribute");
  accept(ast->attributeUsingPrefix, "attribute-using-prefix");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GccAttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "gcc-attribute");
  if (ast->attributeList) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AlignasAttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "alignas-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AlignasTypeAttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "alignas-type-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AsmAttributeAST* ast) {
  out_ = std::format_to(out_, "{}\n", "asm-attribute");
  if (ast->literal) {
    ++indent_;
    out_ = std::format_to(out_, "{:{}}", "", indent_ * 2);
    out_ = std::format_to(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ScopedAttributeTokenAST* ast) {
  out_ = std::format_to(out_, "{}\n", "scoped-attribute-token");
  accept(ast->attributeNamespace, "attribute-namespace");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SimpleAttributeTokenAST* ast) {
  out_ = std::format_to(out_, "{}\n", "simple-attribute-token");
  accept(ast->identifier, "identifier");
}

}  // namespace cxx
