// Generated file by: gen_ast_printer_cc.ts
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

#include <cxx/ast_printer.h>

// cxx
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <format>
#include <iostream>

namespace cxx {

ASTPrinter::ASTPrinter(TranslationUnit* unit, std::ostream& out)
    : unit_(unit), out_(out) {}

void ASTPrinter::operator()(AST* ast) { accept(ast); }

void ASTPrinter::accept(AST* ast, std::string_view field) {
  if (!ast) return;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) {
    out_ << std::format("{}: ", field);
  }
  ast->accept(this);
  --indent_;
}

void ASTPrinter::accept(const Identifier* id, std::string_view field) {
  if (!id) return;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) out_ << std::format("{}: ", field);
  out_ << std::format("{}\n", id->value());
  --indent_;
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  out_ << std::format("{}\n", "translation-unit");
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  out_ << std::format("{}\n", "module-unit");
  accept(ast->globalModuleFragment, "global-module-fragment");
  accept(ast->moduleDeclaration, "module-declaration");
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->privateModuleFragment, "private-module-fragment");
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  out_ << std::format("{}\n", "simple-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->initDeclaratorList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "init-declarator-list");
    for (auto node : ListView{ast->initDeclaratorList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  out_ << std::format("{}\n", "asm-declaration");
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->asmQualifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "asm-qualifier-list");
    for (auto node : ListView{ast->asmQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->outputOperandList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "output-operand-list");
    for (auto node : ListView{ast->outputOperandList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->inputOperandList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "input-operand-list");
    for (auto node : ListView{ast->inputOperandList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->clobberList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "clobber-list");
    for (auto node : ListView{ast->clobberList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->gotoLabelList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "goto-label-list");
    for (auto node : ListView{ast->gotoLabelList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  out_ << std::format("{}\n", "namespace-alias-definition");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  out_ << std::format("{}\n", "using-declaration");
  if (ast->usingDeclaratorList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "using-declarator-list");
    for (auto node : ListView{ast->usingDeclaratorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  out_ << std::format("{}\n", "using-enum-declaration");
  accept(ast->enumTypeSpecifier, "enum-type-specifier");
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  out_ << std::format("{}\n", "using-directive");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  out_ << std::format("{}\n", "static-assert-declaration");
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  out_ << std::format("{}\n", "alias-declaration");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->gnuAttributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "gnu-attribute-list");
    for (auto node : ListView{ast->gnuAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  out_ << std::format("{}\n", "opaque-enum-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  out_ << std::format("{}\n", "function-definition");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "decl-specifier-list");
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
  out_ << std::format("{}\n", "template-declaration");
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("depth: {}\n", ast->depth);
  --indent_;
  if (ast->templateParameterList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  out_ << std::format("{}\n", "concept-definition");
  accept(ast->identifier, "identifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  out_ << std::format("{}\n", "deduction-guide");
  accept(ast->identifier, "identifier");
  accept(ast->explicitSpecifier, "explicit-specifier");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  out_ << std::format("{}\n", "explicit-instantiation");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  out_ << std::format("{}\n", "export-declaration");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportCompoundDeclarationAST* ast) {
  out_ << std::format("{}\n", "export-compound-declaration");
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  out_ << std::format("{}\n", "linkage-specification");
  if (ast->stringLiteral) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("string-literal: {}\n", ast->stringLiteral->value());
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  out_ << std::format("{}\n", "namespace-definition");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-inline: {}\n", ast->isInline);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->nestedNamespaceSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "nested-namespace-specifier-list");
    for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->extraAttributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "extra-attribute-list");
    for (auto node : ListView{ast->extraAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  out_ << std::format("{}\n", "empty-declaration");
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  out_ << std::format("{}\n", "attribute-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  out_ << std::format("{}\n", "module-import-declaration");
  accept(ast->importName, "import-name");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  out_ << std::format("{}\n", "parameter-declaration");
  accept(ast->identifier, "identifier");
  if (ast->isThisIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-this-introduced: {}\n", ast->isThisIntroduced);
    --indent_;
  }
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AccessDeclarationAST* ast) {
  out_ << std::format("{}\n", "access-declaration");
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("access-specifier: {}\n",
                        Token::spell(ast->accessSpecifier));
    --indent_;
  }
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  out_ << std::format("{}\n", "for-range-declaration");
}

void ASTPrinter::visit(StructuredBindingDeclarationAST* ast) {
  out_ << std::format("{}\n", "structured-binding-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->bindingList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "binding-list");
    for (auto node : ListView{ast->bindingList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(AsmOperandAST* ast) {
  out_ << std::format("{}\n", "asm-operand");
  accept(ast->symbolicName, "symbolic-name");
  if (ast->constraintLiteral) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("constraint-literal: {}\n",
                        ast->constraintLiteral->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AsmQualifierAST* ast) {
  out_ << std::format("{}\n", "asm-qualifier");
  if (ast->qualifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("qualifier: {}\n", Token::spell(ast->qualifier));
    --indent_;
  }
}

void ASTPrinter::visit(AsmClobberAST* ast) {
  out_ << std::format("{}\n", "asm-clobber");
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(AsmGotoLabelAST* ast) {
  out_ << std::format("{}\n", "asm-goto-label");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SplicerAST* ast) {
  out_ << std::format("{}\n", "splicer");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GlobalModuleFragmentAST* ast) {
  out_ << std::format("{}\n", "global-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PrivateModuleFragmentAST* ast) {
  out_ << std::format("{}\n", "private-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleDeclarationAST* ast) {
  out_ << std::format("{}\n", "module-declaration");
  accept(ast->moduleName, "module-name");
  accept(ast->modulePartition, "module-partition");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleNameAST* ast) {
  out_ << std::format("{}\n", "module-name");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModuleQualifierAST* ast) {
  out_ << std::format("{}\n", "module-qualifier");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModulePartitionAST* ast) {
  out_ << std::format("{}\n", "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(ImportNameAST* ast) {
  out_ << std::format("{}\n", "import-name");
  accept(ast->modulePartition, "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(InitDeclaratorAST* ast) {
  out_ << std::format("{}\n", "init-declarator");
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  out_ << std::format("{}\n", "declarator");
  if (ast->ptrOpList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "ptr-op-list");
    for (auto node : ListView{ast->ptrOpList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->coreDeclarator, "core-declarator");
  if (ast->declaratorChunkList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declarator-chunk-list");
    for (auto node : ListView{ast->declaratorChunkList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  out_ << std::format("{}\n", "using-declarator");
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  out_ << std::format("{}\n", "enumerator");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeIdAST* ast) {
  out_ << std::format("{}\n", "type-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(HandlerAST* ast) {
  out_ << std::format("{}\n", "handler");
  accept(ast->exceptionDeclaration, "exception-declaration");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  out_ << std::format("{}\n", "base-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->isVirtual) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-virtual: {}\n", ast->isVirtual);
    --indent_;
  }
  if (ast->isVariadic) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-variadic: {}\n", ast->isVariadic);
    --indent_;
  }
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("access-specifier: {}\n",
                        Token::spell(ast->accessSpecifier));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(RequiresClauseAST* ast) {
  out_ << std::format("{}\n", "requires-clause");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  out_ << std::format("{}\n", "parameter-declaration-clause");
  if (ast->isVariadic) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-variadic: {}\n", ast->isVariadic);
    --indent_;
  }
  if (ast->parameterDeclarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "parameter-declaration-list");
    for (auto node : ListView{ast->parameterDeclarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  out_ << std::format("{}\n", "trailing-return-type");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(LambdaSpecifierAST* ast) {
  out_ << std::format("{}\n", "lambda-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstraintAST* ast) {
  out_ << std::format("{}\n", "type-constraint");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AttributeArgumentClauseAST* ast) {
  out_ << std::format("{}\n", "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeAST* ast) {
  out_ << std::format("{}\n", "attribute");
  accept(ast->attributeToken, "attribute-token");
  accept(ast->attributeArgumentClause, "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeUsingPrefixAST* ast) {
  out_ << std::format("{}\n", "attribute-using-prefix");
}

void ASTPrinter::visit(NewPlacementAST* ast) {
  out_ << std::format("{}\n", "new-placement");
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedNamespaceSpecifierAST* ast) {
  out_ << std::format("{}\n", "nested-namespace-specifier");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-inline: {}\n", ast->isInline);
    --indent_;
  }
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  out_ << std::format("{}\n", "labeled-statement");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  out_ << std::format("{}\n", "case-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  out_ << std::format("{}\n", "default-statement");
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  out_ << std::format("{}\n", "expression-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  out_ << std::format("{}\n", "compound-statement");
  if (ast->statementList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "statement-list");
    for (auto node : ListView{ast->statementList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  out_ << std::format("{}\n", "if-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(ConstevalIfStatementAST* ast) {
  out_ << std::format("{}\n", "consteval-if-statement");
  if (ast->isNot) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-not: {}\n", ast->isNot);
    --indent_;
  }
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  out_ << std::format("{}\n", "switch-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  out_ << std::format("{}\n", "while-statement");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DoStatementAST* ast) {
  out_ << std::format("{}\n", "do-statement");
  accept(ast->statement, "statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  out_ << std::format("{}\n", "for-range-statement");
  accept(ast->initializer, "initializer");
  accept(ast->rangeDeclaration, "range-declaration");
  accept(ast->rangeInitializer, "range-initializer");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ForStatementAST* ast) {
  out_ << std::format("{}\n", "for-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->expression, "expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  out_ << std::format("{}\n", "break-statement");
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  out_ << std::format("{}\n", "continue-statement");
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  out_ << std::format("{}\n", "return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  out_ << std::format("{}\n", "coroutine-return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  out_ << std::format("{}\n", "goto-statement");
  accept(ast->identifier, "identifier");
  if (ast->isIndirect) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-indirect: {}\n", ast->isIndirect);
    --indent_;
  }
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  out_ << std::format("{}\n", "declaration-statement");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  out_ << std::format("{}\n", "try-block-statement");
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "handler-list");
    for (auto node : ListView{ast->handlerList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  out_ << "char-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  out_ << "bool-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->isTrue) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-true: {}\n", ast->isTrue);
    --indent_;
  }
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  out_ << "int-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  out_ << "float-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  out_ << "nullptr-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", Token::spell(ast->literal));
    --indent_;
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  out_ << "string-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  out_ << "user-defined-string-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ObjectLiteralExpressionAST* ast) {
  out_ << "object-literal-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  out_ << "this-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
}

void ASTPrinter::visit(GenericSelectionExpressionAST* ast) {
  out_ << "generic-selection-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("matched-assoc-index: {}\n", ast->matchedAssocIndex);
  --indent_;
  accept(ast->expression, "expression");
  if (ast->genericAssociationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "generic-association-list");
    for (auto node : ListView{ast->genericAssociationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedStatementExpressionAST* ast) {
  out_ << "nested-statement-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  out_ << "nested-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  out_ << "id-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  out_ << "lambda-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->captureDefault != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("capture-default: {}\n",
                        Token::spell(ast->captureDefault));
    --indent_;
  }
  if (ast->captureList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "capture-list");
    for (auto node : ListView{ast->captureList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->templateRequiresClause, "template-requires-clause");
  if (ast->expressionAttributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-attribute-list");
    for (auto node : ListView{ast->expressionAttributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->gnuAtributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "gnu-atribute-list");
    for (auto node : ListView{ast->gnuAtributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->lambdaSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "lambda-specifier-list");
    for (auto node : ListView{ast->lambdaSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
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
  out_ << "fold-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->foldOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("fold-op: {}\n", Token::spell(ast->foldOp));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(RightFoldExpressionAST* ast) {
  out_ << "right-fold-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LeftFoldExpressionAST* ast) {
  out_ << "left-fold-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(RequiresExpressionAST* ast) {
  out_ << "requires-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->requirementList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "requirement-list");
    for (auto node : ListView{ast->requirementList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(VaArgExpressionAST* ast) {
  out_ << "va-arg-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  out_ << "subscript-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->baseExpression, "base-expression");
  accept(ast->indexExpression, "index-expression");
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  out_ << "call-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->baseExpression, "base-expression");
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstructionAST* ast) {
  out_ << "type-construction";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeSpecifier, "type-specifier");
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedTypeConstructionAST* ast) {
  out_ << "braced-type-construction";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeSpecifier, "type-specifier");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(SpliceMemberExpressionAST* ast) {
  out_ << "splice-member-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  out_ << "member-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(PostIncrExpressionAST* ast) {
  out_ << "post-incr-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  out_ << "cpp-cast-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BuiltinBitCastExpressionAST* ast) {
  out_ << "builtin-bit-cast-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BuiltinOffsetofExpressionAST* ast) {
  out_ << "builtin-offsetof-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->identifier, "identifier");
  accept(ast->typeId, "type-id");
  if (ast->designatorList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "designator-list");
    for (auto node : ListView{ast->designatorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeidExpressionAST* ast) {
  out_ << "typeid-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeidOfTypeExpressionAST* ast) {
  out_ << "typeid-of-type-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SpliceExpressionAST* ast) {
  out_ << "splice-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(GlobalScopeReflectExpressionAST* ast) {
  out_ << "global-scope-reflect-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
}

void ASTPrinter::visit(NamespaceReflectExpressionAST* ast) {
  out_ << "namespace-reflect-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(TypeIdReflectExpressionAST* ast) {
  out_ << "type-id-reflect-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ReflectExpressionAST* ast) {
  out_ << "reflect-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LabelAddressExpressionAST* ast) {
  out_ << "label-address-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  out_ << "unary-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AwaitExpressionAST* ast) {
  out_ << "await-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofExpressionAST* ast) {
  out_ << "sizeof-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofTypeExpressionAST* ast) {
  out_ << "sizeof-type-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SizeofPackExpressionAST* ast) {
  out_ << "sizeof-pack-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(AlignofTypeExpressionAST* ast) {
  out_ << "alignof-type-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AlignofExpressionAST* ast) {
  out_ << "alignof-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NoexceptExpressionAST* ast) {
  out_ << "noexcept-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  out_ << "new-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->newPlacement, "new-placement");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->newInitalizer, "new-initalizer");
}

void ASTPrinter::visit(DeleteExpressionAST* ast) {
  out_ << "delete-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CastExpressionAST* ast) {
  out_ << "cast-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ImplicitCastExpressionAST* ast) {
  out_ << "implicit-cast-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("cast-kind: {}\n", to_string(ast->castKind));
  --indent_;
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  out_ << "binary-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  out_ << "conditional-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->condition, "condition");
  accept(ast->iftrueExpression, "iftrue-expression");
  accept(ast->iffalseExpression, "iffalse-expression");
}

void ASTPrinter::visit(YieldExpressionAST* ast) {
  out_ << "yield-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExpressionAST* ast) {
  out_ << "throw-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  out_ << "assignment-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(TargetExpressionAST* ast) {
  out_ << "target-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
}

void ASTPrinter::visit(RightExpressionAST* ast) {
  out_ << "right-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
}

void ASTPrinter::visit(CompoundAssignmentExpressionAST* ast) {
  out_ << "compound-assignment-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->targetExpression, "target-expression");
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
  accept(ast->adjustExpression, "adjust-expression");
}

void ASTPrinter::visit(PackExpansionExpressionAST* ast) {
  out_ << "pack-expansion-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DesignatedInitializerClauseAST* ast) {
  out_ << "designated-initializer-clause";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->designatorList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "designator-list");
    for (auto node : ListView{ast->designatorList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(TypeTraitExpressionAST* ast) {
  out_ << "type-trait-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->typeIdList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-id-list");
    for (auto node : ListView{ast->typeIdList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ConditionExpressionAST* ast) {
  out_ << "condition-expression";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "decl-specifier-list");
    for (auto node : ListView{ast->declSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  out_ << "equal-initializer";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  out_ << "braced-init-list";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  out_ << "paren-initializer";
  if (ast->type) {
    out_ << std::format(" [{} {}]", to_string(ast->valueCategory),
                        to_string(ast->type));
  }
  out_ << "\n";
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DefaultGenericAssociationAST* ast) {
  out_ << std::format("{}\n", "default-generic-association");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeGenericAssociationAST* ast) {
  out_ << std::format("{}\n", "type-generic-association");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DotDesignatorAST* ast) {
  out_ << std::format("{}\n", "dot-designator");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SubscriptDesignatorAST* ast) {
  out_ << std::format("{}\n", "subscript-designator");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TemplateTypeParameterAST* ast) {
  out_ << std::format("{}\n", "template-type-parameter");
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-parameter-list");
    for (auto node : ListView{ast->templateParameterList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->idExpression, "id-expression");
}

void ASTPrinter::visit(NonTypeTemplateParameterAST* ast) {
  out_ << std::format("{}\n", "non-type-template-parameter");
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TypenameTypeParameterAST* ast) {
  out_ << std::format("{}\n", "typename-type-parameter");
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ConstraintTypeParameterAST* ast) {
  out_ << std::format("{}\n", "constraint-type-parameter");
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << std::format("{:{}}", "", indent_ * 2);
  out_ << std::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  out_ << std::format("{}\n", "typedef-specifier");
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  out_ << std::format("{}\n", "friend-specifier");
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  out_ << std::format("{}\n", "consteval-specifier");
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  out_ << std::format("{}\n", "constinit-specifier");
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  out_ << std::format("{}\n", "constexpr-specifier");
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  out_ << std::format("{}\n", "inline-specifier");
}

void ASTPrinter::visit(NoreturnSpecifierAST* ast) {
  out_ << std::format("{}\n", "noreturn-specifier");
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  out_ << std::format("{}\n", "static-specifier");
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  out_ << std::format("{}\n", "extern-specifier");
}

void ASTPrinter::visit(RegisterSpecifierAST* ast) {
  out_ << std::format("{}\n", "register-specifier");
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  out_ << std::format("{}\n", "thread-local-specifier");
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  out_ << std::format("{}\n", "thread-specifier");
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  out_ << std::format("{}\n", "mutable-specifier");
}

void ASTPrinter::visit(VirtualSpecifierAST* ast) {
  out_ << std::format("{}\n", "virtual-specifier");
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  out_ << std::format("{}\n", "explicit-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AutoTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "auto-type-specifier");
}

void ASTPrinter::visit(VoidTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "void-type-specifier");
}

void ASTPrinter::visit(SizeTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "size-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(SignTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "sign-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(BuiltinTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "builtin-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(UnaryBuiltinTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "unary-builtin-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(BinaryBuiltinTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "binary-builtin-type-specifier");
  accept(ast->leftTypeId, "left-type-id");
  accept(ast->rightTypeId, "right-type-id");
}

void ASTPrinter::visit(IntegralTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "integral-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(FloatingPointTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "floating-point-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(ComplexTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "complex-type-specifier");
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "named-type-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "atomic-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "underlying-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "elaborated-type-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  out_ << std::format("{}\n", "decltype-auto-specifier");
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "decltype-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "placeholder-type-specifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->specifier, "specifier");
}

void ASTPrinter::visit(ConstQualifierAST* ast) {
  out_ << std::format("{}\n", "const-qualifier");
}

void ASTPrinter::visit(VolatileQualifierAST* ast) {
  out_ << std::format("{}\n", "volatile-qualifier");
}

void ASTPrinter::visit(AtomicQualifierAST* ast) {
  out_ << std::format("{}\n", "atomic-qualifier");
}

void ASTPrinter::visit(RestrictQualifierAST* ast) {
  out_ << std::format("{}\n", "restrict-qualifier");
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  out_ << std::format("{}\n", "enum-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->enumeratorList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "enumerator-list");
    for (auto node : ListView{ast->enumeratorList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  out_ << std::format("{}\n", "class-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isFinal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->baseSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "base-specifier-list");
    for (auto node : ListView{ast->baseSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "declaration-list");
    for (auto node : ListView{ast->declarationList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  out_ << std::format("{}\n", "typename-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(SplicerTypeSpecifierAST* ast) {
  out_ << std::format("{}\n", "splicer-type-specifier");
  accept(ast->splicer, "splicer");
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  out_ << std::format("{}\n", "pointer-operator");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  out_ << std::format("{}\n", "reference-operator");
  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("ref-op: {}\n", Token::spell(ast->refOp));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  out_ << std::format("{}\n", "ptr-to-member-operator");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BitfieldDeclaratorAST* ast) {
  out_ << std::format("{}\n", "bitfield-declarator");
  accept(ast->unqualifiedId, "unqualified-id");
  accept(ast->sizeExpression, "size-expression");
}

void ASTPrinter::visit(ParameterPackAST* ast) {
  out_ << std::format("{}\n", "parameter-pack");
  accept(ast->coreDeclarator, "core-declarator");
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  out_ << std::format("{}\n", "id-declarator");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  out_ << std::format("{}\n", "nested-declarator");
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(FunctionDeclaratorChunkAST* ast) {
  out_ << std::format("{}\n", "function-declarator-chunk");
  if (ast->isFinal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->isOverride) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-override: {}\n", ast->isOverride);
    --indent_;
  }
  if (ast->isPure) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pure: {}\n", ast->isPure);
    --indent_;
  }
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "cv-qualifier-list");
    for (auto node : ListView{ast->cvQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
}

void ASTPrinter::visit(ArrayDeclaratorChunkAST* ast) {
  out_ << std::format("{}\n", "array-declarator-chunk");
  if (ast->typeQualifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-qualifier-list");
    for (auto node : ListView{ast->typeQualifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NameIdAST* ast) {
  out_ << std::format("{}\n", "name-id");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DestructorIdAST* ast) {
  out_ << std::format("{}\n", "destructor-id");
  accept(ast->id, "id");
}

void ASTPrinter::visit(DecltypeIdAST* ast) {
  out_ << std::format("{}\n", "decltype-id");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(OperatorFunctionIdAST* ast) {
  out_ << std::format("{}\n", "operator-function-id");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorIdAST* ast) {
  out_ << std::format("{}\n", "literal-operator-id");
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(ConversionFunctionIdAST* ast) {
  out_ << std::format("{}\n", "conversion-function-id");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SimpleTemplateIdAST* ast) {
  out_ << std::format("{}\n", "simple-template-id");
  accept(ast->identifier, "identifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorTemplateIdAST* ast) {
  out_ << std::format("{}\n", "literal-operator-template-id");
  accept(ast->literalOperatorId, "literal-operator-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(OperatorFunctionTemplateIdAST* ast) {
  out_ << std::format("{}\n", "operator-function-template-id");
  accept(ast->operatorFunctionId, "operator-function-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "template-argument-list");
    for (auto node : ListView{ast->templateArgumentList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GlobalNestedNameSpecifierAST* ast) {
  out_ << std::format("{}\n", "global-nested-name-specifier");
}

void ASTPrinter::visit(SimpleNestedNameSpecifierAST* ast) {
  out_ << std::format("{}\n", "simple-nested-name-specifier");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
}

void ASTPrinter::visit(DecltypeNestedNameSpecifierAST* ast) {
  out_ << std::format("{}\n", "decltype-nested-name-specifier");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(TemplateNestedNameSpecifierAST* ast) {
  out_ << std::format("{}\n", "template-nested-name-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(DefaultFunctionBodyAST* ast) {
  out_ << std::format("{}\n", "default-function-body");
}

void ASTPrinter::visit(CompoundStatementFunctionBodyAST* ast) {
  out_ << std::format("{}\n", "compound-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "mem-initializer-list");
    for (auto node : ListView{ast->memInitializerList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(TryStatementFunctionBodyAST* ast) {
  out_ << std::format("{}\n", "try-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "mem-initializer-list");
    for (auto node : ListView{ast->memInitializerList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "handler-list");
    for (auto node : ListView{ast->handlerList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DeleteFunctionBodyAST* ast) {
  out_ << std::format("{}\n", "delete-function-body");
}

void ASTPrinter::visit(TypeTemplateArgumentAST* ast) {
  out_ << std::format("{}\n", "type-template-argument");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ExpressionTemplateArgumentAST* ast) {
  out_ << std::format("{}\n", "expression-template-argument");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExceptionSpecifierAST* ast) {
  out_ << std::format("{}\n", "throw-exception-specifier");
}

void ASTPrinter::visit(NoexceptSpecifierAST* ast) {
  out_ << std::format("{}\n", "noexcept-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SimpleRequirementAST* ast) {
  out_ << std::format("{}\n", "simple-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundRequirementAST* ast) {
  out_ << std::format("{}\n", "compound-requirement");
  accept(ast->expression, "expression");
  accept(ast->typeConstraint, "type-constraint");
}

void ASTPrinter::visit(TypeRequirementAST* ast) {
  out_ << std::format("{}\n", "type-requirement");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(NestedRequirementAST* ast) {
  out_ << std::format("{}\n", "nested-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  out_ << std::format("{}\n", "new-paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  out_ << std::format("{}\n", "new-braced-initializer");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ParenMemInitializerAST* ast) {
  out_ << std::format("{}\n", "paren-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->expressionList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "expression-list");
    for (auto node : ListView{ast->expressionList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedMemInitializerAST* ast) {
  out_ << std::format("{}\n", "braced-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "this-lambda-capture");
}

void ASTPrinter::visit(DerefThisLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "deref-this-lambda-capture");
}

void ASTPrinter::visit(SimpleLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "simple-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "ref-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefInitLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "ref-init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(InitLambdaCaptureAST* ast) {
  out_ << std::format("{}\n", "init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  out_ << std::format("{}\n", "ellipsis-exception-declaration");
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  out_ << std::format("{}\n", "type-exception-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "type-specifier-list");
    for (auto node : ListView{ast->typeSpecifierList}) {
      accept(node);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(CxxAttributeAST* ast) {
  out_ << std::format("{}\n", "cxx-attribute");
  accept(ast->attributeUsingPrefix, "attribute-using-prefix");
  if (ast->attributeList) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("{}\n", "attribute-list");
    for (auto node : ListView{ast->attributeList}) {
      accept(node);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GccAttributeAST* ast) {
  out_ << std::format("{}\n", "gcc-attribute");
}

void ASTPrinter::visit(AlignasAttributeAST* ast) {
  out_ << std::format("{}\n", "alignas-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AlignasTypeAttributeAST* ast) {
  out_ << std::format("{}\n", "alignas-type-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AsmAttributeAST* ast) {
  out_ << std::format("{}\n", "asm-attribute");
  if (ast->literal) {
    ++indent_;
    out_ << std::format("{:{}}", "", indent_ * 2);
    out_ << std::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ScopedAttributeTokenAST* ast) {
  out_ << std::format("{}\n", "scoped-attribute-token");
  accept(ast->attributeNamespace, "attribute-namespace");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SimpleAttributeTokenAST* ast) {
  out_ << std::format("{}\n", "simple-attribute-token");
  accept(ast->identifier, "identifier");
}

}  // namespace cxx
