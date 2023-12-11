// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "ast_printer.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <iostream>

namespace cxx {

ASTPrinter::ASTPrinter(TranslationUnit* unit, std::ostream& out)
    : unit_(unit), out_(out) {}

void ASTPrinter::operator()(AST* ast) { accept(ast); }

void ASTPrinter::accept(AST* ast, std::string_view field) {
  if (!ast) return;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) {
    out_ << cxx::format("{}: ", field);
  }
  ast->accept(this);
  --indent_;
}

void ASTPrinter::accept(const Identifier* id, std::string_view field) {
  if (!id) return;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) out_ << cxx::format("{}: ", field);
  out_ << cxx::format("{}\n", id->value());
  --indent_;
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  out_ << cxx::format("{}\n", "translation-unit");
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  out_ << cxx::format("{}\n", "module-unit");
  accept(ast->globalModuleFragment, "global-module-fragment");
  accept(ast->moduleDeclaration, "module-declaration");
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->privateModuleFragment, "private-module-fragment");
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "simple-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->initDeclaratorList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "init-declarator-list");
    for (auto it = ast->initDeclaratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "asm-declaration");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->asmQualifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "asm-qualifier-list");
    for (auto it = ast->asmQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->outputOperandList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "output-operand-list");
    for (auto it = ast->outputOperandList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->inputOperandList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "input-operand-list");
    for (auto it = ast->inputOperandList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->clobberList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "clobber-list");
    for (auto it = ast->clobberList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->gotoLabelList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "goto-label-list");
    for (auto it = ast->gotoLabelList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  out_ << cxx::format("{}\n", "namespace-alias-definition");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "using-declaration");
  if (ast->usingDeclaratorList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "using-declarator-list");
    for (auto it = ast->usingDeclaratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "using-enum-declaration");
  accept(ast->enumTypeSpecifier, "enum-type-specifier");
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  out_ << cxx::format("{}\n", "using-directive");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "static-assert-declaration");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "alias-declaration");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "opaque-enum-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  out_ << cxx::format("{}\n", "function-definition");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->functionBody, "function-body");
}

void ASTPrinter::visit(TemplateDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "template-declaration");
  if (ast->templateParameterList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  out_ << cxx::format("{}\n", "concept-definition");
  accept(ast->identifier, "identifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  out_ << cxx::format("{}\n", "deduction-guide");
  accept(ast->identifier, "identifier");
  accept(ast->explicitSpecifier, "explicit-specifier");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  out_ << cxx::format("{}\n", "explicit-instantiation");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "export-declaration");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportCompoundDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "export-compound-declaration");
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  out_ << cxx::format("{}\n", "linkage-specification");
  if (ast->stringLiteral) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("string-literal: {}\n", ast->stringLiteral->value());
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  out_ << cxx::format("{}\n", "namespace-definition");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-inline: {}\n", ast->isInline);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->nestedNamespaceSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "nested-namespace-specifier-list");
    for (auto it = ast->nestedNamespaceSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->extraAttributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "extra-attribute-list");
    for (auto it = ast->extraAttributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "empty-declaration");
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "attribute-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "module-import-declaration");
  accept(ast->importName, "import-name");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "parameter-declaration");
  accept(ast->identifier, "identifier");
  if (ast->isThisIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-this-introduced: {}\n", ast->isThisIntroduced);
    --indent_;
  }
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AccessDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "access-declaration");
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("access-specifier: {}\n",
                        Token::spell(ast->accessSpecifier));
    --indent_;
  }
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "for-range-declaration");
}

void ASTPrinter::visit(StructuredBindingDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "structured-binding-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->bindingList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "binding-list");
    for (auto it = ast->bindingList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(AsmOperandAST* ast) {
  out_ << cxx::format("{}\n", "asm-operand");
  accept(ast->symbolicName, "symbolic-name");
  if (ast->constraintLiteral) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("constraint-literal: {}\n",
                        ast->constraintLiteral->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AsmQualifierAST* ast) {
  out_ << cxx::format("{}\n", "asm-qualifier");
  if (ast->qualifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("qualifier: {}\n", Token::spell(ast->qualifier));
    --indent_;
  }
}

void ASTPrinter::visit(AsmClobberAST* ast) {
  out_ << cxx::format("{}\n", "asm-clobber");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(AsmGotoLabelAST* ast) {
  out_ << cxx::format("{}\n", "asm-goto-label");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  out_ << cxx::format("{}\n", "labeled-statement");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  out_ << cxx::format("{}\n", "case-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  out_ << cxx::format("{}\n", "default-statement");
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  out_ << cxx::format("{}\n", "expression-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  out_ << cxx::format("{}\n", "compound-statement");
  if (ast->statementList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "statement-list");
    for (auto it = ast->statementList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  out_ << cxx::format("{}\n", "if-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(ConstevalIfStatementAST* ast) {
  out_ << cxx::format("{}\n", "consteval-if-statement");
  if (ast->isNot) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-not: {}\n", ast->isNot);
    --indent_;
  }
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  out_ << cxx::format("{}\n", "switch-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  out_ << cxx::format("{}\n", "while-statement");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DoStatementAST* ast) {
  out_ << cxx::format("{}\n", "do-statement");
  accept(ast->statement, "statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  out_ << cxx::format("{}\n", "for-range-statement");
  accept(ast->initializer, "initializer");
  accept(ast->rangeDeclaration, "range-declaration");
  accept(ast->rangeInitializer, "range-initializer");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ForStatementAST* ast) {
  out_ << cxx::format("{}\n", "for-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->expression, "expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  out_ << cxx::format("{}\n", "break-statement");
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  out_ << cxx::format("{}\n", "continue-statement");
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  out_ << cxx::format("{}\n", "return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  out_ << cxx::format("{}\n", "coroutine-return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  out_ << cxx::format("{}\n", "goto-statement");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  out_ << cxx::format("{}\n", "declaration-statement");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  out_ << cxx::format("{}\n", "try-block-statement");
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "handler-list");
    for (auto it = ast->handlerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "char-literal-expression");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "bool-literal-expression");
  if (ast->isTrue) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-true: {}\n", ast->isTrue);
    --indent_;
  }
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "int-literal-expression");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "float-literal-expression");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "nullptr-literal-expression");
  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", Token::spell(ast->literal));
    --indent_;
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "string-literal-expression");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  out_ << cxx::format("{}\n", "user-defined-string-literal-expression");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  out_ << cxx::format("{}\n", "this-expression");
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  out_ << cxx::format("{}\n", "nested-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  out_ << cxx::format("{}\n", "id-expression");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  out_ << cxx::format("{}\n", "lambda-expression");
  if (ast->captureDefault != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("capture-default: {}\n",
                        Token::spell(ast->captureDefault));
    --indent_;
  }
  if (ast->captureList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "capture-list");
    for (auto it = ast->captureList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->templateRequiresClause, "template-requires-clause");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->lambdaSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "lambda-specifier-list");
    for (auto it = ast->lambdaSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(FoldExpressionAST* ast) {
  out_ << cxx::format("{}\n", "fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->foldOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("fold-op: {}\n", Token::spell(ast->foldOp));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(RightFoldExpressionAST* ast) {
  out_ << cxx::format("{}\n", "right-fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LeftFoldExpressionAST* ast) {
  out_ << cxx::format("{}\n", "left-fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(RequiresExpressionAST* ast) {
  out_ << cxx::format("{}\n", "requires-expression");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->requirementList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "requirement-list");
    for (auto it = ast->requirementList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  out_ << cxx::format("{}\n", "subscript-expression");
  accept(ast->baseExpression, "base-expression");
  accept(ast->indexExpression, "index-expression");
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  out_ << cxx::format("{}\n", "call-expression");
  accept(ast->baseExpression, "base-expression");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstructionAST* ast) {
  out_ << cxx::format("{}\n", "type-construction");
  accept(ast->typeSpecifier, "type-specifier");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedTypeConstructionAST* ast) {
  out_ << cxx::format("{}\n", "braced-type-construction");
  accept(ast->typeSpecifier, "type-specifier");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  out_ << cxx::format("{}\n", "member-expression");
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(PostIncrExpressionAST* ast) {
  out_ << cxx::format("{}\n", "post-incr-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  out_ << cxx::format("{}\n", "cpp-cast-expression");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BuiltinBitCastExpressionAST* ast) {
  out_ << cxx::format("{}\n", "builtin-bit-cast-expression");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeidExpressionAST* ast) {
  out_ << cxx::format("{}\n", "typeid-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeidOfTypeExpressionAST* ast) {
  out_ << cxx::format("{}\n", "typeid-of-type-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  out_ << cxx::format("{}\n", "unary-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AwaitExpressionAST* ast) {
  out_ << cxx::format("{}\n", "await-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofExpressionAST* ast) {
  out_ << cxx::format("{}\n", "sizeof-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofTypeExpressionAST* ast) {
  out_ << cxx::format("{}\n", "sizeof-type-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SizeofPackExpressionAST* ast) {
  out_ << cxx::format("{}\n", "sizeof-pack-expression");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(AlignofTypeExpressionAST* ast) {
  out_ << cxx::format("{}\n", "alignof-type-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AlignofExpressionAST* ast) {
  out_ << cxx::format("{}\n", "alignof-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NoexceptExpressionAST* ast) {
  out_ << cxx::format("{}\n", "noexcept-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  out_ << cxx::format("{}\n", "new-expression");
  accept(ast->newPlacement, "new-placement");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->newInitalizer, "new-initalizer");
}

void ASTPrinter::visit(DeleteExpressionAST* ast) {
  out_ << cxx::format("{}\n", "delete-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CastExpressionAST* ast) {
  out_ << cxx::format("{}\n", "cast-expression");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ImplicitCastExpressionAST* ast) {
  out_ << cxx::format("{}\n", "implicit-cast-expression");
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("cast-kind: {}\n", to_string(ast->castKind));
  --indent_;
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  out_ << cxx::format("{}\n", "binary-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  out_ << cxx::format("{}\n", "conditional-expression");
  accept(ast->condition, "condition");
  accept(ast->iftrueExpression, "iftrue-expression");
  accept(ast->iffalseExpression, "iffalse-expression");
}

void ASTPrinter::visit(YieldExpressionAST* ast) {
  out_ << cxx::format("{}\n", "yield-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExpressionAST* ast) {
  out_ << cxx::format("{}\n", "throw-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  out_ << cxx::format("{}\n", "assignment-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(PackExpansionExpressionAST* ast) {
  out_ << cxx::format("{}\n", "pack-expansion-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DesignatedInitializerClauseAST* ast) {
  out_ << cxx::format("{}\n", "designated-initializer-clause");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(TypeTraitsExpressionAST* ast) {
  out_ << cxx::format("{}\n", "type-traits-expression");
  if (ast->typeIdList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-id-list");
    for (auto it = ast->typeIdList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ConditionExpressionAST* ast) {
  out_ << cxx::format("{}\n", "condition-expression");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  out_ << cxx::format("{}\n", "equal-initializer");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  out_ << cxx::format("{}\n", "braced-init-list");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  out_ << cxx::format("{}\n", "paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TemplateTypeParameterAST* ast) {
  out_ << cxx::format("{}\n", "template-type-parameter");
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  if (ast->templateParameterList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->idExpression, "id-expression");
}

void ASTPrinter::visit(NonTypeTemplateParameterAST* ast) {
  out_ << cxx::format("{}\n", "non-type-template-parameter");
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TypenameTypeParameterAST* ast) {
  out_ << cxx::format("{}\n", "typename-type-parameter");
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ConstraintTypeParameterAST* ast) {
  out_ << cxx::format("{}\n", "constraint-type-parameter");
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("depth: {}\n", ast->depth);
  --indent_;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  out_ << cxx::format("index: {}\n", ast->index);
  --indent_;
  accept(ast->identifier, "identifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "typedef-specifier");
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "friend-specifier");
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "consteval-specifier");
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "constinit-specifier");
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "constexpr-specifier");
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "inline-specifier");
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "static-specifier");
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "extern-specifier");
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "thread-local-specifier");
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "thread-specifier");
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "mutable-specifier");
}

void ASTPrinter::visit(VirtualSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "virtual-specifier");
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "explicit-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AutoTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "auto-type-specifier");
}

void ASTPrinter::visit(VoidTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "void-type-specifier");
}

void ASTPrinter::visit(SizeTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "size-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(SignTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "sign-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(VaListTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "va-list-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(IntegralTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "integral-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(FloatingPointTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "floating-point-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(ComplexTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "complex-type-specifier");
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "named-type-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "atomic-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "underlying-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "elaborated-type-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "decltype-auto-specifier");
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "decltype-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "placeholder-type-specifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->specifier, "specifier");
}

void ASTPrinter::visit(ConstQualifierAST* ast) {
  out_ << cxx::format("{}\n", "const-qualifier");
}

void ASTPrinter::visit(VolatileQualifierAST* ast) {
  out_ << cxx::format("{}\n", "volatile-qualifier");
}

void ASTPrinter::visit(RestrictQualifierAST* ast) {
  out_ << cxx::format("{}\n", "restrict-qualifier");
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "enum-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->enumeratorList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "enumerator-list");
    for (auto it = ast->enumeratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "class-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->isFinal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->baseSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "base-specifier-list");
    for (auto it = ast->baseSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "typename-specifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  out_ << cxx::format("{}\n", "pointer-operator");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  out_ << cxx::format("{}\n", "reference-operator");
  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("ref-op: {}\n", Token::spell(ast->refOp));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  out_ << cxx::format("{}\n", "ptr-to-member-operator");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BitfieldDeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "bitfield-declarator");
  accept(ast->unqualifiedId, "unqualified-id");
  accept(ast->sizeExpression, "size-expression");
}

void ASTPrinter::visit(ParameterPackAST* ast) {
  out_ << cxx::format("{}\n", "parameter-pack");
  accept(ast->coreDeclarator, "core-declarator");
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "id-declarator");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "nested-declarator");
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(FunctionDeclaratorChunkAST* ast) {
  out_ << cxx::format("{}\n", "function-declarator-chunk");
  if (ast->isFinal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-final: {}\n", ast->isFinal);
    --indent_;
  }
  if (ast->isOverride) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-override: {}\n", ast->isOverride);
    --indent_;
  }
  if (ast->isPure) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pure: {}\n", ast->isPure);
    --indent_;
  }
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->cvQualifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->exceptionSpecifier, "exception-specifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
}

void ASTPrinter::visit(ArrayDeclaratorChunkAST* ast) {
  out_ << cxx::format("{}\n", "array-declarator-chunk");
  accept(ast->expression, "expression");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NameIdAST* ast) {
  out_ << cxx::format("{}\n", "name-id");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DestructorIdAST* ast) {
  out_ << cxx::format("{}\n", "destructor-id");
  accept(ast->id, "id");
}

void ASTPrinter::visit(DecltypeIdAST* ast) {
  out_ << cxx::format("{}\n", "decltype-id");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(OperatorFunctionIdAST* ast) {
  out_ << cxx::format("{}\n", "operator-function-id");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("op: {}\n", Token::spell(ast->op));
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorIdAST* ast) {
  out_ << cxx::format("{}\n", "literal-operator-id");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(ConversionFunctionIdAST* ast) {
  out_ << cxx::format("{}\n", "conversion-function-id");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SimpleTemplateIdAST* ast) {
  out_ << cxx::format("{}\n", "simple-template-id");
  accept(ast->identifier, "identifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-argument-list");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LiteralOperatorTemplateIdAST* ast) {
  out_ << cxx::format("{}\n", "literal-operator-template-id");
  accept(ast->literalOperatorId, "literal-operator-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-argument-list");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(OperatorFunctionTemplateIdAST* ast) {
  out_ << cxx::format("{}\n", "operator-function-template-id");
  accept(ast->operatorFunctionId, "operator-function-id");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-argument-list");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GlobalNestedNameSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "global-nested-name-specifier");
}

void ASTPrinter::visit(SimpleNestedNameSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "simple-nested-name-specifier");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
}

void ASTPrinter::visit(DecltypeNestedNameSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "decltype-nested-name-specifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(TemplateNestedNameSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "template-nested-name-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->templateId, "template-id");
}

void ASTPrinter::visit(DefaultFunctionBodyAST* ast) {
  out_ << cxx::format("{}\n", "default-function-body");
}

void ASTPrinter::visit(CompoundStatementFunctionBodyAST* ast) {
  out_ << cxx::format("{}\n", "compound-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "mem-initializer-list");
    for (auto it = ast->memInitializerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(TryStatementFunctionBodyAST* ast) {
  out_ << cxx::format("{}\n", "try-statement-function-body");
  if (ast->memInitializerList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "mem-initializer-list");
    for (auto it = ast->memInitializerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "handler-list");
    for (auto it = ast->handlerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DeleteFunctionBodyAST* ast) {
  out_ << cxx::format("{}\n", "delete-function-body");
}

void ASTPrinter::visit(TypeTemplateArgumentAST* ast) {
  out_ << cxx::format("{}\n", "type-template-argument");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ExpressionTemplateArgumentAST* ast) {
  out_ << cxx::format("{}\n", "expression-template-argument");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExceptionSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "throw-exception-specifier");
}

void ASTPrinter::visit(NoexceptSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "noexcept-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SimpleRequirementAST* ast) {
  out_ << cxx::format("{}\n", "simple-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundRequirementAST* ast) {
  out_ << cxx::format("{}\n", "compound-requirement");
  accept(ast->expression, "expression");
  accept(ast->typeConstraint, "type-constraint");
}

void ASTPrinter::visit(TypeRequirementAST* ast) {
  out_ << cxx::format("{}\n", "type-requirement");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(NestedRequirementAST* ast) {
  out_ << cxx::format("{}\n", "nested-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  out_ << cxx::format("{}\n", "new-paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  out_ << cxx::format("{}\n", "new-braced-initializer");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ParenMemInitializerAST* ast) {
  out_ << cxx::format("{}\n", "paren-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedMemInitializerAST* ast) {
  out_ << cxx::format("{}\n", "braced-mem-initializer");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "this-lambda-capture");
}

void ASTPrinter::visit(DerefThisLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "deref-this-lambda-capture");
}

void ASTPrinter::visit(SimpleLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "simple-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "ref-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefInitLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "ref-init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(InitLambdaCaptureAST* ast) {
  out_ << cxx::format("{}\n", "init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "ellipsis-exception-declaration");
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "type-exception-declaration");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(CxxAttributeAST* ast) {
  out_ << cxx::format("{}\n", "cxx-attribute");
  accept(ast->attributeUsingPrefix, "attribute-using-prefix");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GccAttributeAST* ast) {
  out_ << cxx::format("{}\n", "gcc-attribute");
}

void ASTPrinter::visit(AlignasAttributeAST* ast) {
  out_ << cxx::format("{}\n", "alignas-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AlignasTypeAttributeAST* ast) {
  out_ << cxx::format("{}\n", "alignas-type-attribute");
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AsmAttributeAST* ast) {
  out_ << cxx::format("{}\n", "asm-attribute");
  if (ast->literal) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ScopedAttributeTokenAST* ast) {
  out_ << cxx::format("{}\n", "scoped-attribute-token");
  accept(ast->attributeNamespace, "attribute-namespace");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(SimpleAttributeTokenAST* ast) {
  out_ << cxx::format("{}\n", "simple-attribute-token");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(GlobalModuleFragmentAST* ast) {
  out_ << cxx::format("{}\n", "global-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PrivateModuleFragmentAST* ast) {
  out_ << cxx::format("{}\n", "private-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleDeclarationAST* ast) {
  out_ << cxx::format("{}\n", "module-declaration");
  accept(ast->moduleName, "module-name");
  accept(ast->modulePartition, "module-partition");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleNameAST* ast) {
  out_ << cxx::format("{}\n", "module-name");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModuleQualifierAST* ast) {
  out_ << cxx::format("{}\n", "module-qualifier");
  accept(ast->identifier, "identifier");
  accept(ast->moduleQualifier, "module-qualifier");
}

void ASTPrinter::visit(ModulePartitionAST* ast) {
  out_ << cxx::format("{}\n", "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(ImportNameAST* ast) {
  out_ << cxx::format("{}\n", "import-name");
  accept(ast->modulePartition, "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(InitDeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "init-declarator");
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "declarator");
  if (ast->ptrOpList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "ptr-op-list");
    for (auto it = ast->ptrOpList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->coreDeclarator, "core-declarator");
  if (ast->declaratorChunkList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "declarator-chunk-list");
    for (auto it = ast->declaratorChunkList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  out_ << cxx::format("{}\n", "using-declarator");
  if (ast->isPack) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-pack: {}\n", ast->isPack);
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  out_ << cxx::format("{}\n", "enumerator");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeIdAST* ast) {
  out_ << cxx::format("{}\n", "type-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(HandlerAST* ast) {
  out_ << cxx::format("{}\n", "handler");
  accept(ast->exceptionDeclaration, "exception-declaration");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "base-specifier");
  if (ast->isTemplateIntroduced) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-template-introduced: {}\n",
                        ast->isTemplateIntroduced);
    --indent_;
  }
  if (ast->isVirtual) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-virtual: {}\n", ast->isVirtual);
    --indent_;
  }
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("access-specifier: {}\n",
                        Token::spell(ast->accessSpecifier));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->unqualifiedId, "unqualified-id");
}

void ASTPrinter::visit(RequiresClauseAST* ast) {
  out_ << cxx::format("{}\n", "requires-clause");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  out_ << cxx::format("{}\n", "parameter-declaration-clause");
  if (ast->isVariadic) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-variadic: {}\n", ast->isVariadic);
    --indent_;
  }
  if (ast->parameterDeclarationList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "parameter-declaration-list");
    for (auto it = ast->parameterDeclarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  out_ << cxx::format("{}\n", "trailing-return-type");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(LambdaSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "lambda-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstraintAST* ast) {
  out_ << cxx::format("{}\n", "type-constraint");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->templateArgumentList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "template-argument-list");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AttributeArgumentClauseAST* ast) {
  out_ << cxx::format("{}\n", "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeAST* ast) {
  out_ << cxx::format("{}\n", "attribute");
  accept(ast->attributeToken, "attribute-token");
  accept(ast->attributeArgumentClause, "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeUsingPrefixAST* ast) {
  out_ << cxx::format("{}\n", "attribute-using-prefix");
}

void ASTPrinter::visit(NewPlacementAST* ast) {
  out_ << cxx::format("{}\n", "new-placement");
  if (ast->expressionList) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedNamespaceSpecifierAST* ast) {
  out_ << cxx::format("{}\n", "nested-namespace-specifier");
  accept(ast->identifier, "identifier");
  if (ast->isInline) {
    ++indent_;
    out_ << cxx::format("{:{}}", "", indent_ * 2);
    out_ << cxx::format("is-inline: {}\n", ast->isInline);
    --indent_;
  }
}

}  // namespace cxx
