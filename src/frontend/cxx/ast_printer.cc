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

namespace cxx {

ASTPrinter::ASTPrinter(TranslationUnit* unit, std::ostream& out)
    : unit_(unit), out_(out) {}

void ASTPrinter::operator()(AST* ast) { accept(ast); }

void ASTPrinter::accept(AST* ast, std::string_view field) {
  if (!ast) return;
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  if (!field.empty()) {
    fmt::print(out_, "{}: ", field);
  }
  ast->accept(this);
  --indent_;
}

void ASTPrinter::accept(const Identifier* id, std::string_view field) {
  if (!id) return;
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  if (!field.empty()) fmt::print(out_, "{}: ", field);
  fmt::print(out_, "{}\n", id->value());
  --indent_;
}

void ASTPrinter::visit(TypeIdAST* ast) {
  fmt::print(out_, "{}\n", "type-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(NestedNameSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "nested-name-specifier");
  if (ast->nameList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "name-list");
    for (auto it = ast->nameList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "using-declarator");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(HandlerAST* ast) {
  fmt::print(out_, "{}\n", "handler");
  accept(ast->exceptionDeclaration, "exception-declaration");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(EnumBaseAST* ast) {
  fmt::print(out_, "{}\n", "enum-base");
  if (ast->typeSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  fmt::print(out_, "{}\n", "enumerator");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "declarator");
  if (ast->ptrOpList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "ptr-op-list");
    for (auto it = ast->ptrOpList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->coreDeclarator, "core-declarator");
  if (ast->modifiers) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "modifiers");
    for (auto it = ast->modifiers; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(InitDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "init-declarator");
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "base-specifier");
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  fmt::print(out_, "is-virtual: {}\n", ast->isVirtual);
  --indent_;
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "access-specifier: {}\n",
               Token::spell(ast->accessSpecifier));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->name, "name");
}

void ASTPrinter::visit(BaseClauseAST* ast) {
  fmt::print(out_, "{}\n", "base-clause");
  if (ast->baseSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "base-specifier-list");
    for (auto it = ast->baseSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NewTypeIdAST* ast) {
  fmt::print(out_, "{}\n", "new-type-id");
  if (ast->typeSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(RequiresClauseAST* ast) {
  fmt::print(out_, "{}\n", "requires-clause");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  fmt::print(out_, "{}\n", "parameter-declaration-clause");
  if (ast->parameterDeclarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "parameter-declaration-list");
    for (auto it = ast->parameterDeclarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParametersAndQualifiersAST* ast) {
  fmt::print(out_, "{}\n", "parameters-and-qualifiers");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->cvQualifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LambdaIntroducerAST* ast) {
  fmt::print(out_, "{}\n", "lambda-introducer");
  if (ast->captureList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "capture-list");
    for (auto it = ast->captureList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(LambdaDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "lambda-declarator");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  if (ast->declSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->trailingReturnType, "trailing-return-type");
  accept(ast->requiresClause, "requires-clause");
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  fmt::print(out_, "{}\n", "trailing-return-type");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(CtorInitializerAST* ast) {
  fmt::print(out_, "{}\n", "ctor-initializer");
  if (ast->memInitializerList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "mem-initializer-list");
    for (auto it = ast->memInitializerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(RequirementBodyAST* ast) {
  fmt::print(out_, "{}\n", "requirement-body");
  if (ast->requirementList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "requirement-list");
    for (auto it = ast->requirementList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypeConstraintAST* ast) {
  fmt::print(out_, "{}\n", "type-constraint");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(GlobalModuleFragmentAST* ast) {
  fmt::print(out_, "{}\n", "global-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PrivateModuleFragmentAST* ast) {
  fmt::print(out_, "{}\n", "private-module-fragment");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "module-declaration");
  accept(ast->moduleName, "module-name");
  accept(ast->modulePartition, "module-partition");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleNameAST* ast) {
  fmt::print(out_, "{}\n", "module-name");
}

void ASTPrinter::visit(ImportNameAST* ast) {
  fmt::print(out_, "{}\n", "import-name");
  accept(ast->modulePartition, "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(ModulePartitionAST* ast) {
  fmt::print(out_, "{}\n", "module-partition");
  accept(ast->moduleName, "module-name");
}

void ASTPrinter::visit(AttributeArgumentClauseAST* ast) {
  fmt::print(out_, "{}\n", "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeAST* ast) {
  fmt::print(out_, "{}\n", "attribute");
  accept(ast->attributeToken, "attribute-token");
  accept(ast->attributeArgumentClause, "attribute-argument-clause");
}

void ASTPrinter::visit(AttributeUsingPrefixAST* ast) {
  fmt::print(out_, "{}\n", "attribute-using-prefix");
}

void ASTPrinter::visit(DesignatorAST* ast) {
  fmt::print(out_, "{}\n", "designator");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DesignatedInitializerClauseAST* ast) {
  fmt::print(out_, "{}\n", "designated-initializer-clause");
  accept(ast->designator, "designator");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  fmt::print(out_, "{}\n", "this-expression");
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "char-literal-expression");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "bool-literal-expression");
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  fmt::print(out_, "value: {}\n", ast->value);
  --indent_;
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "int-literal-expression");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "float-literal-expression");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "nullptr-literal-expression");
  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", Token::spell(ast->literal));
    --indent_;
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "string-literal-expression");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  fmt::print(out_, "{}\n", "user-defined-string-literal-expression");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  fmt::print(out_, "{}\n", "id-expression");
  accept(ast->name, "name");
}

void ASTPrinter::visit(RequiresExpressionAST* ast) {
  fmt::print(out_, "{}\n", "requires-expression");
  accept(ast->parameterDeclarationClause, "parameter-declaration-clause");
  accept(ast->requirementBody, "requirement-body");
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  fmt::print(out_, "{}\n", "nested-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(RightFoldExpressionAST* ast) {
  fmt::print(out_, "{}\n", "right-fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LeftFoldExpressionAST* ast) {
  fmt::print(out_, "{}\n", "left-fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(FoldExpressionAST* ast) {
  fmt::print(out_, "{}\n", "fold-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  if (ast->foldOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "fold-op: {}\n", Token::spell(ast->foldOp));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  fmt::print(out_, "{}\n", "lambda-expression");
  accept(ast->lambdaIntroducer, "lambda-introducer");
  if (ast->templateParameterList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->lambdaDeclarator, "lambda-declarator");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(SizeofExpressionAST* ast) {
  fmt::print(out_, "{}\n", "sizeof-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(SizeofTypeExpressionAST* ast) {
  fmt::print(out_, "{}\n", "sizeof-type-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SizeofPackExpressionAST* ast) {
  fmt::print(out_, "{}\n", "sizeof-pack-expression");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(TypeidExpressionAST* ast) {
  fmt::print(out_, "{}\n", "typeid-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeidOfTypeExpressionAST* ast) {
  fmt::print(out_, "{}\n", "typeid-of-type-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(AlignofExpressionAST* ast) {
  fmt::print(out_, "{}\n", "alignof-expression");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TypeTraitsExpressionAST* ast) {
  fmt::print(out_, "{}\n", "type-traits-expression");
  if (ast->typeTraits != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "type-traits: {}\n", Token::spell(ast->typeTraits));
    --indent_;
  }
  if (ast->typeIdList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-id-list");
    for (auto it = ast->typeIdList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  fmt::print(out_, "{}\n", "unary-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  fmt::print(out_, "{}\n", "binary-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  fmt::print(out_, "{}\n", "assignment-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->leftExpression, "left-expression");
  accept(ast->rightExpression, "right-expression");
}

void ASTPrinter::visit(BracedTypeConstructionAST* ast) {
  fmt::print(out_, "{}\n", "braced-type-construction");
  accept(ast->typeSpecifier, "type-specifier");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(TypeConstructionAST* ast) {
  fmt::print(out_, "{}\n", "type-construction");
  accept(ast->typeSpecifier, "type-specifier");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  fmt::print(out_, "{}\n", "call-expression");
  accept(ast->baseExpression, "base-expression");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  fmt::print(out_, "{}\n", "subscript-expression");
  accept(ast->baseExpression, "base-expression");
  accept(ast->indexExpression, "index-expression");
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  fmt::print(out_, "{}\n", "member-expression");
  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "access-op: {}\n", Token::spell(ast->accessOp));
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
  accept(ast->name, "name");
}

void ASTPrinter::visit(PostIncrExpressionAST* ast) {
  fmt::print(out_, "{}\n", "post-incr-expression");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
  accept(ast->baseExpression, "base-expression");
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  fmt::print(out_, "{}\n", "conditional-expression");
  accept(ast->condition, "condition");
  accept(ast->iftrueExpression, "iftrue-expression");
  accept(ast->iffalseExpression, "iffalse-expression");
}

void ASTPrinter::visit(ImplicitCastExpressionAST* ast) {
  fmt::print(out_, "{}\n", "implicit-cast-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CastExpressionAST* ast) {
  fmt::print(out_, "{}\n", "cast-expression");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  fmt::print(out_, "{}\n", "cpp-cast-expression");
  accept(ast->typeId, "type-id");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  fmt::print(out_, "{}\n", "new-expression");
  accept(ast->typeId, "type-id");
  accept(ast->newInitalizer, "new-initalizer");
}

void ASTPrinter::visit(DeleteExpressionAST* ast) {
  fmt::print(out_, "{}\n", "delete-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ThrowExpressionAST* ast) {
  fmt::print(out_, "{}\n", "throw-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(NoexceptExpressionAST* ast) {
  fmt::print(out_, "{}\n", "noexcept-expression");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  fmt::print(out_, "{}\n", "equal-initializer");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  fmt::print(out_, "{}\n", "braced-init-list");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  fmt::print(out_, "{}\n", "paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(SimpleRequirementAST* ast) {
  fmt::print(out_, "{}\n", "simple-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundRequirementAST* ast) {
  fmt::print(out_, "{}\n", "compound-requirement");
  accept(ast->expression, "expression");
  accept(ast->typeConstraint, "type-constraint");
}

void ASTPrinter::visit(TypeRequirementAST* ast) {
  fmt::print(out_, "{}\n", "type-requirement");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(NestedRequirementAST* ast) {
  fmt::print(out_, "{}\n", "nested-requirement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(TypeTemplateArgumentAST* ast) {
  fmt::print(out_, "{}\n", "type-template-argument");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ExpressionTemplateArgumentAST* ast) {
  fmt::print(out_, "{}\n", "expression-template-argument");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ParenMemInitializerAST* ast) {
  fmt::print(out_, "{}\n", "paren-mem-initializer");
  accept(ast->name, "name");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(BracedMemInitializerAST* ast) {
  fmt::print(out_, "{}\n", "braced-mem-initializer");
  accept(ast->name, "name");
  accept(ast->bracedInitList, "braced-init-list");
}

void ASTPrinter::visit(ThisLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "this-lambda-capture");
}

void ASTPrinter::visit(DerefThisLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "deref-this-lambda-capture");
}

void ASTPrinter::visit(SimpleLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "simple-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "ref-lambda-capture");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(RefInitLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "ref-init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(InitLambdaCaptureAST* ast) {
  fmt::print(out_, "{}\n", "init-lambda-capture");
  accept(ast->identifier, "identifier");
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  fmt::print(out_, "{}\n", "new-paren-initializer");
  if (ast->expressionList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "expression-list");
    for (auto it = ast->expressionList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  fmt::print(out_, "{}\n", "new-braced-initializer");
  accept(ast->bracedInit, "braced-init");
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "ellipsis-exception-declaration");
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "type-exception-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(DefaultFunctionBodyAST* ast) {
  fmt::print(out_, "{}\n", "default-function-body");
}

void ASTPrinter::visit(CompoundStatementFunctionBodyAST* ast) {
  fmt::print(out_, "{}\n", "compound-statement-function-body");
  accept(ast->ctorInitializer, "ctor-initializer");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(TryStatementFunctionBodyAST* ast) {
  fmt::print(out_, "{}\n", "try-statement-function-body");
  accept(ast->ctorInitializer, "ctor-initializer");
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "handler-list");
    for (auto it = ast->handlerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DeleteFunctionBodyAST* ast) {
  fmt::print(out_, "{}\n", "delete-function-body");
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  fmt::print(out_, "{}\n", "translation-unit");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  fmt::print(out_, "{}\n", "module-unit");
  accept(ast->globalModuleFragment, "global-module-fragment");
  accept(ast->moduleDeclaration, "module-declaration");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->privateModuleFragment, "private-module-fragment");
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  fmt::print(out_, "{}\n", "labeled-statement");
  accept(ast->identifier, "identifier");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  fmt::print(out_, "{}\n", "case-statement");
  accept(ast->expression, "expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  fmt::print(out_, "{}\n", "default-statement");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  fmt::print(out_, "{}\n", "expression-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  fmt::print(out_, "{}\n", "compound-statement");
  if (ast->statementList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "statement-list");
    for (auto it = ast->statementList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  fmt::print(out_, "{}\n", "if-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
  accept(ast->elseStatement, "else-statement");
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  fmt::print(out_, "{}\n", "switch-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  fmt::print(out_, "{}\n", "while-statement");
  accept(ast->condition, "condition");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(DoStatementAST* ast) {
  fmt::print(out_, "{}\n", "do-statement");
  accept(ast->statement, "statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  fmt::print(out_, "{}\n", "for-range-statement");
  accept(ast->initializer, "initializer");
  accept(ast->rangeDeclaration, "range-declaration");
  accept(ast->rangeInitializer, "range-initializer");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(ForStatementAST* ast) {
  fmt::print(out_, "{}\n", "for-statement");
  accept(ast->initializer, "initializer");
  accept(ast->condition, "condition");
  accept(ast->expression, "expression");
  accept(ast->statement, "statement");
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  fmt::print(out_, "{}\n", "break-statement");
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  fmt::print(out_, "{}\n", "continue-statement");
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  fmt::print(out_, "{}\n", "return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  fmt::print(out_, "{}\n", "goto-statement");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  fmt::print(out_, "{}\n", "coroutine-return-statement");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  fmt::print(out_, "{}\n", "declaration-statement");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  fmt::print(out_, "{}\n", "try-block-statement");
  accept(ast->statement, "statement");
  if (ast->handlerList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "handler-list");
    for (auto it = ast->handlerList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(AccessDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "access-declaration");
  if (ast->accessSpecifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "access-specifier: {}\n",
               Token::spell(ast->accessSpecifier));
    --indent_;
  }
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  fmt::print(out_, "{}\n", "function-definition");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->requiresClause, "requires-clause");
  accept(ast->functionBody, "function-body");
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  fmt::print(out_, "{}\n", "concept-definition");
  accept(ast->name, "name");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "for-range-declaration");
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "alias-declaration");
  accept(ast->identifier, "identifier");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "simple-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->initDeclaratorList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "init-declarator-list");
    for (auto it = ast->initDeclaratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
}

void ASTPrinter::visit(StructuredBindingDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "structured-binding-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "decl-specifier-list");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->bindingList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "binding-list");
    for (auto it = ast->bindingList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->initializer, "initializer");
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "static-assert-declaration");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "empty-declaration");
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "attribute-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "opaque-enum-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
  accept(ast->enumBase, "enum-base");
}

void ASTPrinter::visit(NestedNamespaceSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "nested-namespace-specifier");
  accept(ast->namespaceName, "namespace-name");
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  fmt::print(out_, "is-inline: {}\n", ast->isInline);
  --indent_;
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  fmt::print(out_, "{}\n", "namespace-definition");
  accept(ast->namespaceName, "namespace-name");
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  fmt::print(out_, "is-inline: {}\n", ast->isInline);
  --indent_;
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->nestedNamespaceSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "nested-namespace-specifier-list");
    for (auto it = ast->nestedNamespaceSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->extraAttributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "extra-attribute-list");
    for (auto it = ast->extraAttributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  fmt::print(out_, "{}\n", "namespace-alias-definition");
  accept(ast->identifier, "identifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  fmt::print(out_, "{}\n", "using-directive");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "using-declaration");
  if (ast->usingDeclaratorList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "using-declarator-list");
    for (auto it = ast->usingDeclaratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "using-enum-declaration");
  accept(ast->enumTypeSpecifier, "enum-type-specifier");
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "asm-declaration");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "export-declaration");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ExportCompoundDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "export-compound-declaration");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "module-import-declaration");
  accept(ast->importName, "import-name");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TemplateDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "template-declaration");
  if (ast->templateParameterList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(TypenameTypeParameterAST* ast) {
  fmt::print(out_, "{}\n", "typename-type-parameter");
  accept(ast->identifier, "identifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TemplateTypeParameterAST* ast) {
  fmt::print(out_, "{}\n", "template-type-parameter");
  accept(ast->identifier, "identifier");
  if (ast->templateParameterList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->requiresClause, "requires-clause");
  accept(ast->name, "name");
}

void ASTPrinter::visit(TemplatePackTypeParameterAST* ast) {
  fmt::print(out_, "{}\n", "template-pack-type-parameter");
  accept(ast->identifier, "identifier");
  if (ast->templateParameterList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "template-parameter-list");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  fmt::print(out_, "{}\n", "deduction-guide");
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  fmt::print(out_, "{}\n", "explicit-instantiation");
  accept(ast->declaration, "declaration");
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  fmt::print(out_, "{}\n", "parameter-declaration");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->typeSpecifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "type-specifier-list");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->declarator, "declarator");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  fmt::print(out_, "{}\n", "linkage-specification");
  if (ast->stringLiteral) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "string-literal: {}\n", ast->stringLiteral->value());
    --indent_;
  }
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(SimpleNameAST* ast) {
  fmt::print(out_, "{}\n", "simple-name");
  accept(ast->identifier, "identifier");
}

void ASTPrinter::visit(DestructorNameAST* ast) {
  fmt::print(out_, "{}\n", "destructor-name");
  accept(ast->id, "id");
}

void ASTPrinter::visit(DecltypeNameAST* ast) {
  fmt::print(out_, "{}\n", "decltype-name");
  accept(ast->decltypeSpecifier, "decltype-specifier");
}

void ASTPrinter::visit(OperatorNameAST* ast) {
  fmt::print(out_, "{}\n", "operator-name");
  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "op: {}\n", Token::spell(ast->op));
    --indent_;
  }
}

void ASTPrinter::visit(ConversionNameAST* ast) {
  fmt::print(out_, "{}\n", "conversion-name");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(TemplateNameAST* ast) {
  fmt::print(out_, "{}\n", "template-name");
  accept(ast->id, "id");
  if (ast->templateArgumentList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "template-argument-list");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(QualifiedNameAST* ast) {
  fmt::print(out_, "{}\n", "qualified-name");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->id, "id");
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "typedef-specifier");
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "friend-specifier");
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "consteval-specifier");
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "constinit-specifier");
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "constexpr-specifier");
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "inline-specifier");
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "static-specifier");
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "extern-specifier");
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "thread-local-specifier");
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "thread-specifier");
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "mutable-specifier");
}

void ASTPrinter::visit(VirtualSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "virtual-specifier");
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "explicit-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AutoTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "auto-type-specifier");
}

void ASTPrinter::visit(VoidTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "void-type-specifier");
}

void ASTPrinter::visit(VaListTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "va-list-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(IntegralTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "integral-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(FloatingPointTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "floating-point-type-specifier");
  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "specifier: {}\n", Token::spell(ast->specifier));
    --indent_;
  }
}

void ASTPrinter::visit(ComplexTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "complex-type-specifier");
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "named-type-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "atomic-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "underlying-type-specifier");
  accept(ast->typeId, "type-id");
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "elaborated-type-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "decltype-auto-specifier");
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "decltype-specifier");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "placeholder-type-specifier");
  accept(ast->typeConstraint, "type-constraint");
  accept(ast->specifier, "specifier");
}

void ASTPrinter::visit(ConstQualifierAST* ast) {
  fmt::print(out_, "{}\n", "const-qualifier");
}

void ASTPrinter::visit(VolatileQualifierAST* ast) {
  fmt::print(out_, "{}\n", "volatile-qualifier");
}

void ASTPrinter::visit(RestrictQualifierAST* ast) {
  fmt::print(out_, "{}\n", "restrict-qualifier");
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "enum-specifier");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
  accept(ast->enumBase, "enum-base");
  if (ast->enumeratorList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "enumerator-list");
    for (auto it = ast->enumeratorList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "class-specifier");
  if (ast->classKey != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "class-key: {}\n", Token::spell(ast->classKey));
    --indent_;
  }
  ++indent_;
  fmt::print(out_, "{:{}}", "", indent_ * 2);
  fmt::print(out_, "is-final: {}\n", ast->isFinal);
  --indent_;
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  accept(ast->name, "name");
  accept(ast->baseClause, "base-clause");
  if (ast->declarationList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "declaration-list");
    for (auto it = ast->declarationList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  fmt::print(out_, "{}\n", "typename-specifier");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  accept(ast->name, "name");
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "id-declarator");
  accept(ast->name, "name");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "nested-declarator");
  accept(ast->declarator, "declarator");
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  fmt::print(out_, "{}\n", "pointer-operator");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  fmt::print(out_, "{}\n", "reference-operator");
  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "ref-op: {}\n", Token::spell(ast->refOp));
    --indent_;
  }
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  fmt::print(out_, "{}\n", "ptr-to-member-operator");
  accept(ast->nestedNameSpecifier, "nested-name-specifier");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
  if (ast->cvQualifierList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "cv-qualifier-list");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(FunctionDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "function-declarator");
  accept(ast->parametersAndQualifiers, "parameters-and-qualifiers");
  accept(ast->trailingReturnType, "trailing-return-type");
}

void ASTPrinter::visit(ArrayDeclaratorAST* ast) {
  fmt::print(out_, "{}\n", "array-declarator");
  accept(ast->expression, "expression");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(CxxAttributeAST* ast) {
  fmt::print(out_, "{}\n", "cxx-attribute");
  accept(ast->attributeUsingPrefix, "attribute-using-prefix");
  if (ast->attributeList) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "{}\n", "attribute-list");
    for (auto it = ast->attributeList; it; it = it->next) {
      accept(it->value);
    }
    --indent_;
  }
}

void ASTPrinter::visit(GCCAttributeAST* ast) {
  fmt::print(out_, "{}\n", "gccattribute");
}

void ASTPrinter::visit(AlignasAttributeAST* ast) {
  fmt::print(out_, "{}\n", "alignas-attribute");
  accept(ast->expression, "expression");
}

void ASTPrinter::visit(AsmAttributeAST* ast) {
  fmt::print(out_, "{}\n", "asm-attribute");
  if (ast->literal) {
    ++indent_;
    fmt::print(out_, "{:{}}", "", indent_ * 2);
    fmt::print(out_, "literal: {}\n", ast->literal->value());
    --indent_;
  }
}

void ASTPrinter::visit(ScopedAttributeTokenAST* ast) {
  fmt::print(out_, "{}\n", "scoped-attribute-token");
}

void ASTPrinter::visit(SimpleAttributeTokenAST* ast) {
  fmt::print(out_, "{}\n", "simple-attribute-token");
}

}  // namespace cxx
