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

#include <cxx/ast_cloner.h>

namespace cxx {

auto ASTCloner::clone(AST* ast, Arena* arena) -> AST* {
  if (!ast) return nullptr;
  std::swap(arena_, arena);
  auto copy = accept(ast);
  std::swap(arena_, arena);
  return copy;
}

void ASTCloner::visit(TypeIdAST* ast) {
  auto copy = new (arena_) TypeIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);
}

void ASTCloner::visit(UsingDeclaratorAST* ast) {
  auto copy = new (arena_) UsingDeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typenameLoc = ast->typenameLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->isPack = ast->isPack;
}

void ASTCloner::visit(HandlerAST* ast) {
  auto copy = new (arena_) HandlerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->catchLoc = ast->catchLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->exceptionDeclaration = accept(ast->exceptionDeclaration);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(EnumeratorAST* ast) {
  auto copy = new (arena_) EnumeratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(DeclaratorAST* ast) {
  auto copy = new (arena_) DeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->ptrOpList) {
    auto out = &copy->ptrOpList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->coreDeclarator = accept(ast->coreDeclarator);

  if (auto it = ast->declaratorChunkList) {
    auto out = &copy->declaratorChunkList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(InitDeclaratorAST* ast) {
  auto copy = new (arena_) InitDeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->declarator = accept(ast->declarator);

  copy->requiresClause = accept(ast->requiresClause);

  copy->initializer = accept(ast->initializer);
}

void ASTCloner::visit(BaseSpecifierAST* ast) {
  auto copy = new (arena_) BaseSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->templateLoc = ast->templateLoc;

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  copy->isVirtual = ast->isVirtual;

  copy->accessSpecifier = ast->accessSpecifier;
}

void ASTCloner::visit(RequiresClauseAST* ast) {
  auto copy = new (arena_) RequiresClauseAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->requiresLoc = ast->requiresLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ParameterDeclarationClauseAST* ast) {
  auto copy = new (arena_) ParameterDeclarationClauseAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->parameterDeclarationList) {
    auto out = &copy->parameterDeclarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->commaLoc = ast->commaLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->isVariadic = ast->isVariadic;
}

void ASTCloner::visit(LambdaSpecifierAST* ast) {
  auto copy = new (arena_) LambdaSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(TrailingReturnTypeAST* ast) {
  auto copy = new (arena_) TrailingReturnTypeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->minusGreaterLoc = ast->minusGreaterLoc;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(TypeConstraintAST* ast) {
  auto copy = new (arena_) TypeConstraintAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(GlobalModuleFragmentAST* ast) {
  auto copy = new (arena_) GlobalModuleFragmentAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->moduleLoc = ast->moduleLoc;

  copy->semicolonLoc = ast->semicolonLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(PrivateModuleFragmentAST* ast) {
  auto copy = new (arena_) PrivateModuleFragmentAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->moduleLoc = ast->moduleLoc;

  copy->colonLoc = ast->colonLoc;

  copy->privateLoc = ast->privateLoc;

  copy->semicolonLoc = ast->semicolonLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ModuleQualifierAST* ast) {
  auto copy = new (arena_) ModuleQualifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->moduleQualifier = accept(ast->moduleQualifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->dotLoc = ast->dotLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ModuleNameAST* ast) {
  auto copy = new (arena_) ModuleNameAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->moduleQualifier = accept(ast->moduleQualifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ModuleDeclarationAST* ast) {
  auto copy = new (arena_) ModuleDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->exportLoc = ast->exportLoc;

  copy->moduleLoc = ast->moduleLoc;

  copy->moduleName = accept(ast->moduleName);

  copy->modulePartition = accept(ast->modulePartition);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ImportNameAST* ast) {
  auto copy = new (arena_) ImportNameAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->headerLoc = ast->headerLoc;

  copy->modulePartition = accept(ast->modulePartition);

  copy->moduleName = accept(ast->moduleName);
}

void ASTCloner::visit(ModulePartitionAST* ast) {
  auto copy = new (arena_) ModulePartitionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->colonLoc = ast->colonLoc;

  copy->moduleName = accept(ast->moduleName);
}

void ASTCloner::visit(AttributeArgumentClauseAST* ast) {
  auto copy = new (arena_) AttributeArgumentClauseAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lparenLoc = ast->lparenLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AttributeAST* ast) {
  auto copy = new (arena_) AttributeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->attributeToken = accept(ast->attributeToken);

  copy->attributeArgumentClause = accept(ast->attributeArgumentClause);

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(AttributeUsingPrefixAST* ast) {
  auto copy = new (arena_) AttributeUsingPrefixAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->usingLoc = ast->usingLoc;

  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;

  copy->colonLoc = ast->colonLoc;
}

void ASTCloner::visit(NewPlacementAST* ast) {
  auto copy = new (arena_) NewPlacementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(NestedNamespaceSpecifierAST* ast) {
  auto copy = new (arena_) NestedNamespaceSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->inlineLoc = ast->inlineLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->scopeLoc = ast->scopeLoc;

  copy->identifier = ast->identifier;

  copy->isInline = ast->isInline;
}

void ASTCloner::visit(GlobalNestedNameSpecifierAST* ast) {
  auto copy = new (arena_) GlobalNestedNameSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->scopeLoc = ast->scopeLoc;
}

void ASTCloner::visit(SimpleNestedNameSpecifierAST* ast) {
  auto copy = new (arena_) SimpleNestedNameSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;

  copy->scopeLoc = ast->scopeLoc;
}

void ASTCloner::visit(DecltypeNestedNameSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeNestedNameSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->decltypeSpecifier = accept(ast->decltypeSpecifier);

  copy->scopeLoc = ast->scopeLoc;
}

void ASTCloner::visit(TemplateNestedNameSpecifierAST* ast) {
  auto copy = new (arena_) TemplateNestedNameSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->templateLoc = ast->templateLoc;

  copy->templateId = accept(ast->templateId);

  copy->scopeLoc = ast->scopeLoc;

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
}

void ASTCloner::visit(ThrowExceptionSpecifierAST* ast) {
  auto copy = new (arena_) ThrowExceptionSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->throwLoc = ast->throwLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(NoexceptSpecifierAST* ast) {
  auto copy = new (arena_) NoexceptSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->noexceptLoc = ast->noexceptLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(PackExpansionExpressionAST* ast) {
  auto copy = new (arena_) PackExpansionExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->expression = accept(ast->expression);

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(DesignatedInitializerClauseAST* ast) {
  auto copy = new (arena_) DesignatedInitializerClauseAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->dotLoc = ast->dotLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;

  copy->initializer = accept(ast->initializer);
}

void ASTCloner::visit(ThisExpressionAST* ast) {
  auto copy = new (arena_) ThisExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(CharLiteralExpressionAST* ast) {
  auto copy = new (arena_) CharLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(BoolLiteralExpressionAST* ast) {
  auto copy = new (arena_) BoolLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->isTrue = ast->isTrue;
}

void ASTCloner::visit(IntLiteralExpressionAST* ast) {
  auto copy = new (arena_) IntLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(FloatLiteralExpressionAST* ast) {
  auto copy = new (arena_) FloatLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(NullptrLiteralExpressionAST* ast) {
  auto copy = new (arena_) NullptrLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(StringLiteralExpressionAST* ast) {
  auto copy = new (arena_) StringLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(UserDefinedStringLiteralExpressionAST* ast) {
  auto copy = new (arena_) UserDefinedStringLiteralExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(IdExpressionAST* ast) {
  auto copy = new (arena_) IdExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->templateLoc = ast->templateLoc;

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
}

void ASTCloner::visit(RequiresExpressionAST* ast) {
  auto copy = new (arena_) RequiresExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->requiresLoc = ast->requiresLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  copy->rparenLoc = ast->rparenLoc;

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->requirementList) {
    auto out = &copy->requirementList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
}

void ASTCloner::visit(NestedExpressionAST* ast) {
  auto copy = new (arena_) NestedExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(RightFoldExpressionAST* ast) {
  auto copy = new (arena_) RightFoldExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->opLoc = ast->opLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->rparenLoc = ast->rparenLoc;

  copy->op = ast->op;
}

void ASTCloner::visit(LeftFoldExpressionAST* ast) {
  auto copy = new (arena_) LeftFoldExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->opLoc = ast->opLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;

  copy->op = ast->op;
}

void ASTCloner::visit(FoldExpressionAST* ast) {
  auto copy = new (arena_) FoldExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  copy->leftExpression = accept(ast->leftExpression);

  copy->opLoc = ast->opLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->foldOpLoc = ast->foldOpLoc;

  copy->rightExpression = accept(ast->rightExpression);

  copy->rparenLoc = ast->rparenLoc;

  copy->op = ast->op;

  copy->foldOp = ast->foldOp;
}

void ASTCloner::visit(LambdaExpressionAST* ast) {
  auto copy = new (arena_) LambdaExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lbracketLoc = ast->lbracketLoc;

  copy->captureDefaultLoc = ast->captureDefaultLoc;

  if (auto it = ast->captureList) {
    auto out = &copy->captureList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->templateRequiresClause = accept(ast->templateRequiresClause);

  copy->lparenLoc = ast->lparenLoc;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  copy->rparenLoc = ast->rparenLoc;

  if (auto it = ast->lambdaSpecifierList) {
    auto out = &copy->lambdaSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->exceptionSpecifier = accept(ast->exceptionSpecifier);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->trailingReturnType = accept(ast->trailingReturnType);

  copy->requiresClause = accept(ast->requiresClause);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(SizeofExpressionAST* ast) {
  auto copy = new (arena_) SizeofExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(SizeofTypeExpressionAST* ast) {
  auto copy = new (arena_) SizeofTypeExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(SizeofPackExpressionAST* ast) {
  auto copy = new (arena_) SizeofPackExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->rparenLoc = ast->rparenLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(TypeidExpressionAST* ast) {
  auto copy = new (arena_) TypeidExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->typeidLoc = ast->typeidLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(TypeidOfTypeExpressionAST* ast) {
  auto copy = new (arena_) TypeidOfTypeExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->typeidLoc = ast->typeidLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AlignofTypeExpressionAST* ast) {
  auto copy = new (arena_) AlignofTypeExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->alignofLoc = ast->alignofLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AlignofExpressionAST* ast) {
  auto copy = new (arena_) AlignofExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->alignofLoc = ast->alignofLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(TypeTraitsExpressionAST* ast) {
  auto copy = new (arena_) TypeTraitsExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->typeTraitsLoc = ast->typeTraitsLoc;

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->typeIdList) {
    auto out = &copy->typeIdList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  copy->typeTraits = ast->typeTraits;
}

void ASTCloner::visit(YieldExpressionAST* ast) {
  auto copy = new (arena_) YieldExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->yieldLoc = ast->yieldLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(AwaitExpressionAST* ast) {
  auto copy = new (arena_) AwaitExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->awaitLoc = ast->awaitLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(UnaryExpressionAST* ast) {
  auto copy = new (arena_) UnaryExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->opLoc = ast->opLoc;

  copy->expression = accept(ast->expression);

  copy->op = ast->op;
}

void ASTCloner::visit(BinaryExpressionAST* ast) {
  auto copy = new (arena_) BinaryExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->leftExpression = accept(ast->leftExpression);

  copy->opLoc = ast->opLoc;

  copy->rightExpression = accept(ast->rightExpression);

  copy->op = ast->op;
}

void ASTCloner::visit(AssignmentExpressionAST* ast) {
  auto copy = new (arena_) AssignmentExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->leftExpression = accept(ast->leftExpression);

  copy->opLoc = ast->opLoc;

  copy->rightExpression = accept(ast->rightExpression);

  copy->op = ast->op;
}

void ASTCloner::visit(ConditionExpressionAST* ast) {
  auto copy = new (arena_) ConditionExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->initializer = accept(ast->initializer);
}

void ASTCloner::visit(BracedTypeConstructionAST* ast) {
  auto copy = new (arena_) BracedTypeConstructionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->typeSpecifier = accept(ast->typeSpecifier);

  copy->bracedInitList = accept(ast->bracedInitList);
}

void ASTCloner::visit(TypeConstructionAST* ast) {
  auto copy = new (arena_) TypeConstructionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->typeSpecifier = accept(ast->typeSpecifier);

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(CallExpressionAST* ast) {
  auto copy = new (arena_) CallExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->baseExpression = accept(ast->baseExpression);

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(SubscriptExpressionAST* ast) {
  auto copy = new (arena_) SubscriptExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->baseExpression = accept(ast->baseExpression);

  copy->lbracketLoc = ast->lbracketLoc;

  copy->indexExpression = accept(ast->indexExpression);

  copy->rbracketLoc = ast->rbracketLoc;
}

void ASTCloner::visit(MemberExpressionAST* ast) {
  auto copy = new (arena_) MemberExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->baseExpression = accept(ast->baseExpression);

  copy->accessLoc = ast->accessLoc;

  copy->memberId = accept(ast->memberId);

  copy->accessOp = ast->accessOp;
}

void ASTCloner::visit(PostIncrExpressionAST* ast) {
  auto copy = new (arena_) PostIncrExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->baseExpression = accept(ast->baseExpression);

  copy->opLoc = ast->opLoc;

  copy->op = ast->op;
}

void ASTCloner::visit(ConditionalExpressionAST* ast) {
  auto copy = new (arena_) ConditionalExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->condition = accept(ast->condition);

  copy->questionLoc = ast->questionLoc;

  copy->iftrueExpression = accept(ast->iftrueExpression);

  copy->colonLoc = ast->colonLoc;

  copy->iffalseExpression = accept(ast->iffalseExpression);
}

void ASTCloner::visit(ImplicitCastExpressionAST* ast) {
  auto copy = new (arena_) ImplicitCastExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->expression = accept(ast->expression);

  copy->castKind = ast->castKind;
}

void ASTCloner::visit(CastExpressionAST* ast) {
  auto copy = new (arena_) CastExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(CppCastExpressionAST* ast) {
  auto copy = new (arena_) CppCastExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->castLoc = ast->castLoc;

  copy->lessLoc = ast->lessLoc;

  copy->typeId = accept(ast->typeId);

  copy->greaterLoc = ast->greaterLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(NewExpressionAST* ast) {
  auto copy = new (arena_) NewExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->scopeLoc = ast->scopeLoc;

  copy->newLoc = ast->newLoc;

  copy->newPlacement = accept(ast->newPlacement);

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->rparenLoc = ast->rparenLoc;

  copy->newInitalizer = accept(ast->newInitalizer);
}

void ASTCloner::visit(DeleteExpressionAST* ast) {
  auto copy = new (arena_) DeleteExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->scopeLoc = ast->scopeLoc;

  copy->deleteLoc = ast->deleteLoc;

  copy->lbracketLoc = ast->lbracketLoc;

  copy->rbracketLoc = ast->rbracketLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ThrowExpressionAST* ast) {
  auto copy = new (arena_) ThrowExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->throwLoc = ast->throwLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(NoexceptExpressionAST* ast) {
  auto copy = new (arena_) NoexceptExpressionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->noexceptLoc = ast->noexceptLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(EqualInitializerAST* ast) {
  auto copy = new (arena_) EqualInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(BracedInitListAST* ast) {
  auto copy = new (arena_) BracedInitListAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->commaLoc = ast->commaLoc;

  copy->rbraceLoc = ast->rbraceLoc;
}

void ASTCloner::visit(ParenInitializerAST* ast) {
  auto copy = new (arena_) ParenInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->valueCategory = ast->valueCategory;

  copy->constValue = ast->constValue;

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(SimpleRequirementAST* ast) {
  auto copy = new (arena_) SimpleRequirementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CompoundRequirementAST* ast) {
  auto copy = new (arena_) CompoundRequirementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lbraceLoc = ast->lbraceLoc;

  copy->expression = accept(ast->expression);

  copy->rbraceLoc = ast->rbraceLoc;

  copy->noexceptLoc = ast->noexceptLoc;

  copy->minusGreaterLoc = ast->minusGreaterLoc;

  copy->typeConstraint = accept(ast->typeConstraint);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(TypeRequirementAST* ast) {
  auto copy = new (arena_) TypeRequirementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typenameLoc = ast->typenameLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(NestedRequirementAST* ast) {
  auto copy = new (arena_) NestedRequirementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->requiresLoc = ast->requiresLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(TypeTemplateArgumentAST* ast) {
  auto copy = new (arena_) TypeTemplateArgumentAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(ExpressionTemplateArgumentAST* ast) {
  auto copy = new (arena_) ExpressionTemplateArgumentAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ParenMemInitializerAST* ast) {
  auto copy = new (arena_) ParenMemInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(BracedMemInitializerAST* ast) {
  auto copy = new (arena_) BracedMemInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->bracedInitList = accept(ast->bracedInitList);

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(ThisLambdaCaptureAST* ast) {
  auto copy = new (arena_) ThisLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(DerefThisLambdaCaptureAST* ast) {
  auto copy = new (arena_) DerefThisLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->starLoc = ast->starLoc;

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(SimpleLambdaCaptureAST* ast) {
  auto copy = new (arena_) SimpleLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(RefLambdaCaptureAST* ast) {
  auto copy = new (arena_) RefLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ampLoc = ast->ampLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(RefInitLambdaCaptureAST* ast) {
  auto copy = new (arena_) RefInitLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ampLoc = ast->ampLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->initializer = accept(ast->initializer);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(InitLambdaCaptureAST* ast) {
  auto copy = new (arena_) InitLambdaCaptureAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->initializer = accept(ast->initializer);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(NewParenInitializerAST* ast) {
  auto copy = new (arena_) NewParenInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(NewBracedInitializerAST* ast) {
  auto copy = new (arena_) NewBracedInitializerAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->bracedInitList = accept(ast->bracedInitList);
}

void ASTCloner::visit(EllipsisExceptionDeclarationAST* ast) {
  auto copy = new (arena_) EllipsisExceptionDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(TypeExceptionDeclarationAST* ast) {
  auto copy = new (arena_) TypeExceptionDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);
}

void ASTCloner::visit(DefaultFunctionBodyAST* ast) {
  auto copy = new (arena_) DefaultFunctionBodyAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->equalLoc = ast->equalLoc;

  copy->defaultLoc = ast->defaultLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CompoundStatementFunctionBodyAST* ast) {
  auto copy = new (arena_) CompoundStatementFunctionBodyAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->memInitializerList) {
    auto out = &copy->memInitializerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(TryStatementFunctionBodyAST* ast) {
  auto copy = new (arena_) TryStatementFunctionBodyAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->tryLoc = ast->tryLoc;

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->memInitializerList) {
    auto out = &copy->memInitializerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->statement = accept(ast->statement);

  if (auto it = ast->handlerList) {
    auto out = &copy->handlerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(DeleteFunctionBodyAST* ast) {
  auto copy = new (arena_) DeleteFunctionBodyAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->equalLoc = ast->equalLoc;

  copy->deleteLoc = ast->deleteLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(TranslationUnitAST* ast) {
  auto copy = new (arena_) TranslationUnitAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ModuleUnitAST* ast) {
  auto copy = new (arena_) ModuleUnitAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->globalModuleFragment = accept(ast->globalModuleFragment);

  copy->moduleDeclaration = accept(ast->moduleDeclaration);

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->privateModuleFragment = accept(ast->privateModuleFragment);
}

void ASTCloner::visit(LabeledStatementAST* ast) {
  auto copy = new (arena_) LabeledStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->colonLoc = ast->colonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(CaseStatementAST* ast) {
  auto copy = new (arena_) CaseStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->caseLoc = ast->caseLoc;

  copy->expression = accept(ast->expression);

  copy->colonLoc = ast->colonLoc;
}

void ASTCloner::visit(DefaultStatementAST* ast) {
  auto copy = new (arena_) DefaultStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->defaultLoc = ast->defaultLoc;

  copy->colonLoc = ast->colonLoc;
}

void ASTCloner::visit(ExpressionStatementAST* ast) {
  auto copy = new (arena_) ExpressionStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CompoundStatementAST* ast) {
  auto copy = new (arena_) CompoundStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->statementList) {
    auto out = &copy->statementList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
}

void ASTCloner::visit(IfStatementAST* ast) {
  auto copy = new (arena_) IfStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ifLoc = ast->ifLoc;

  copy->constexprLoc = ast->constexprLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);

  copy->elseLoc = ast->elseLoc;

  copy->elseStatement = accept(ast->elseStatement);
}

void ASTCloner::visit(ConstevalIfStatementAST* ast) {
  auto copy = new (arena_) ConstevalIfStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ifLoc = ast->ifLoc;

  copy->exclaimLoc = ast->exclaimLoc;

  copy->constvalLoc = ast->constvalLoc;

  copy->statement = accept(ast->statement);

  copy->elseLoc = ast->elseLoc;

  copy->elseStatement = accept(ast->elseStatement);

  copy->isNot = ast->isNot;
}

void ASTCloner::visit(SwitchStatementAST* ast) {
  auto copy = new (arena_) SwitchStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->switchLoc = ast->switchLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(WhileStatementAST* ast) {
  auto copy = new (arena_) WhileStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->whileLoc = ast->whileLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->condition = accept(ast->condition);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(DoStatementAST* ast) {
  auto copy = new (arena_) DoStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->doLoc = ast->doLoc;

  copy->statement = accept(ast->statement);

  copy->whileLoc = ast->whileLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ForRangeStatementAST* ast) {
  auto copy = new (arena_) ForRangeStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->forLoc = ast->forLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->initializer = accept(ast->initializer);

  copy->rangeDeclaration = accept(ast->rangeDeclaration);

  copy->colonLoc = ast->colonLoc;

  copy->rangeInitializer = accept(ast->rangeInitializer);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(ForStatementAST* ast) {
  auto copy = new (arena_) ForStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->forLoc = ast->forLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->semicolonLoc = ast->semicolonLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(BreakStatementAST* ast) {
  auto copy = new (arena_) BreakStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->breakLoc = ast->breakLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ContinueStatementAST* ast) {
  auto copy = new (arena_) ContinueStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->continueLoc = ast->continueLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ReturnStatementAST* ast) {
  auto copy = new (arena_) ReturnStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->returnLoc = ast->returnLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(GotoStatementAST* ast) {
  auto copy = new (arena_) GotoStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->gotoLoc = ast->gotoLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->semicolonLoc = ast->semicolonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(CoroutineReturnStatementAST* ast) {
  auto copy = new (arena_) CoroutineReturnStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->coreturnLoc = ast->coreturnLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(DeclarationStatementAST* ast) {
  auto copy = new (arena_) DeclarationStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(TryBlockStatementAST* ast) {
  auto copy = new (arena_) TryBlockStatementAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->tryLoc = ast->tryLoc;

  copy->statement = accept(ast->statement);

  if (auto it = ast->handlerList) {
    auto out = &copy->handlerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(AccessDeclarationAST* ast) {
  auto copy = new (arena_) AccessDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->accessLoc = ast->accessLoc;

  copy->colonLoc = ast->colonLoc;

  copy->accessSpecifier = ast->accessSpecifier;
}

void ASTCloner::visit(FunctionDefinitionAST* ast) {
  auto copy = new (arena_) FunctionDefinitionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->requiresClause = accept(ast->requiresClause);

  copy->functionBody = accept(ast->functionBody);
}

void ASTCloner::visit(ConceptDefinitionAST* ast) {
  auto copy = new (arena_) ConceptDefinitionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->conceptLoc = ast->conceptLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ForRangeDeclarationAST* ast) {
  auto copy = new (arena_) ForRangeDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());
}

void ASTCloner::visit(AliasDeclarationAST* ast) {
  auto copy = new (arena_) AliasDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->usingLoc = ast->usingLoc;

  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->equalLoc = ast->equalLoc;

  copy->typeId = accept(ast->typeId);

  copy->semicolonLoc = ast->semicolonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(SimpleDeclarationAST* ast) {
  auto copy = new (arena_) SimpleDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->initDeclaratorList) {
    auto out = &copy->initDeclaratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->requiresClause = accept(ast->requiresClause);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(StructuredBindingDeclarationAST* ast) {
  auto copy = new (arena_) StructuredBindingDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->refQualifierLoc = ast->refQualifierLoc;

  copy->lbracketLoc = ast->lbracketLoc;

  if (auto it = ast->bindingList) {
    auto out = &copy->bindingList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;

  copy->initializer = accept(ast->initializer);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(StaticAssertDeclarationAST* ast) {
  auto copy = new (arena_) StaticAssertDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->staticAssertLoc = ast->staticAssertLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->commaLoc = ast->commaLoc;

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;

  copy->rparenLoc = ast->rparenLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(EmptyDeclarationAST* ast) {
  auto copy = new (arena_) EmptyDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(AttributeDeclarationAST* ast) {
  auto copy = new (arena_) AttributeDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(OpaqueEnumDeclarationAST* ast) {
  auto copy = new (arena_) OpaqueEnumDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->enumLoc = ast->enumLoc;

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->emicolonLoc = ast->emicolonLoc;
}

void ASTCloner::visit(NamespaceDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceDefinitionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->inlineLoc = ast->inlineLoc;

  copy->namespaceLoc = ast->namespaceLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->nestedNamespaceSpecifierList) {
    auto out = &copy->nestedNamespaceSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->identifierLoc = ast->identifierLoc;

  if (auto it = ast->extraAttributeList) {
    auto out = &copy->extraAttributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  copy->identifier = ast->identifier;

  copy->isInline = ast->isInline;
}

void ASTCloner::visit(NamespaceAliasDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceAliasDefinitionAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->namespaceLoc = ast->namespaceLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->semicolonLoc = ast->semicolonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(UsingDirectiveAST* ast) {
  auto copy = new (arena_) UsingDirectiveAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->usingLoc = ast->usingLoc;

  copy->namespaceLoc = ast->namespaceLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(UsingDeclarationAST* ast) {
  auto copy = new (arena_) UsingDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->usingLoc = ast->usingLoc;

  if (auto it = ast->usingDeclaratorList) {
    auto out = &copy->usingDeclaratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(UsingEnumDeclarationAST* ast) {
  auto copy = new (arena_) UsingEnumDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->usingLoc = ast->usingLoc;

  copy->enumTypeSpecifier = accept(ast->enumTypeSpecifier);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(AsmOperandAST* ast) {
  auto copy = new (arena_) AsmOperandAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lbracketLoc = ast->lbracketLoc;

  copy->symbolicNameLoc = ast->symbolicNameLoc;

  copy->rbracketLoc = ast->rbracketLoc;

  copy->constraintLiteralLoc = ast->constraintLiteralLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;

  copy->symbolicName = ast->symbolicName;

  copy->constraintLiteral = ast->constraintLiteral;
}

void ASTCloner::visit(AsmQualifierAST* ast) {
  auto copy = new (arena_) AsmQualifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->qualifierLoc = ast->qualifierLoc;

  copy->qualifier = ast->qualifier;
}

void ASTCloner::visit(AsmClobberAST* ast) {
  auto copy = new (arena_) AsmClobberAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->literalLoc = ast->literalLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(AsmGotoLabelAST* ast) {
  auto copy = new (arena_) AsmGotoLabelAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(AsmDeclarationAST* ast) {
  auto copy = new (arena_) AsmDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->asmQualifierList) {
    auto out = &copy->asmQualifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->asmLoc = ast->asmLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->literalLoc = ast->literalLoc;

  if (auto it = ast->outputOperandList) {
    auto out = &copy->outputOperandList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->inputOperandList) {
    auto out = &copy->inputOperandList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->clobberList) {
    auto out = &copy->clobberList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->gotoLabelList) {
    auto out = &copy->gotoLabelList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  copy->semicolonLoc = ast->semicolonLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(ExportDeclarationAST* ast) {
  auto copy = new (arena_) ExportDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->exportLoc = ast->exportLoc;

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(ExportCompoundDeclarationAST* ast) {
  auto copy = new (arena_) ExportCompoundDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->exportLoc = ast->exportLoc;

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
}

void ASTCloner::visit(ModuleImportDeclarationAST* ast) {
  auto copy = new (arena_) ModuleImportDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->importLoc = ast->importLoc;

  copy->importName = accept(ast->importName);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(TemplateDeclarationAST* ast) {
  auto copy = new (arena_) TemplateDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->templateLoc = ast->templateLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->requiresClause = accept(ast->requiresClause);

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(DeductionGuideAST* ast) {
  auto copy = new (arena_) DeductionGuideAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->explicitSpecifier = accept(ast->explicitSpecifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  copy->rparenLoc = ast->rparenLoc;

  copy->arrowLoc = ast->arrowLoc;

  copy->templateId = accept(ast->templateId);

  copy->semicolonLoc = ast->semicolonLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ExplicitInstantiationAST* ast) {
  auto copy = new (arena_) ExplicitInstantiationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->externLoc = ast->externLoc;

  copy->templateLoc = ast->templateLoc;

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(ParameterDeclarationAST* ast) {
  auto copy = new (arena_) ParameterDeclarationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->thisLoc = ast->thisLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);

  copy->isThisIntroduced = ast->isThisIntroduced;
}

void ASTCloner::visit(LinkageSpecificationAST* ast) {
  auto copy = new (arena_) LinkageSpecificationAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->externLoc = ast->externLoc;

  copy->stringliteralLoc = ast->stringliteralLoc;

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  copy->stringLiteral = ast->stringLiteral;
}

void ASTCloner::visit(TemplateTypeParameterAST* ast) {
  auto copy = new (arena_) TemplateTypeParameterAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->templateLoc = ast->templateLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->requiresClause = accept(ast->requiresClause);

  copy->classKeyLoc = ast->classKeyLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->idExpression = accept(ast->idExpression);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(TemplatePackTypeParameterAST* ast) {
  auto copy = new (arena_) TemplatePackTypeParameterAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->templateLoc = ast->templateLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->classKeyLoc = ast->classKeyLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(NonTypeTemplateParameterAST* ast) {
  auto copy = new (arena_) NonTypeTemplateParameterAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(TypenameTypeParameterAST* ast) {
  auto copy = new (arena_) TypenameTypeParameterAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->classKeyLoc = ast->classKeyLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->typeId = accept(ast->typeId);

  copy->identifier = ast->identifier;

  copy->isPack = ast->isPack;
}

void ASTCloner::visit(ConstraintTypeParameterAST* ast) {
  auto copy = new (arena_) ConstraintTypeParameterAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typeConstraint = accept(ast->typeConstraint);

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->typeId = accept(ast->typeId);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(NameIdAST* ast) {
  auto copy = new (arena_) NameIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(DestructorIdAST* ast) {
  auto copy = new (arena_) DestructorIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->tildeLoc = ast->tildeLoc;

  copy->id = accept(ast->id);
}

void ASTCloner::visit(DecltypeIdAST* ast) {
  auto copy = new (arena_) DecltypeIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->decltypeSpecifier = accept(ast->decltypeSpecifier);
}

void ASTCloner::visit(OperatorFunctionIdAST* ast) {
  auto copy = new (arena_) OperatorFunctionIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->operatorLoc = ast->operatorLoc;

  copy->opLoc = ast->opLoc;

  copy->openLoc = ast->openLoc;

  copy->closeLoc = ast->closeLoc;

  copy->op = ast->op;
}

void ASTCloner::visit(LiteralOperatorIdAST* ast) {
  auto copy = new (arena_) LiteralOperatorIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->operatorLoc = ast->operatorLoc;

  copy->literalLoc = ast->literalLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->literal = ast->literal;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ConversionFunctionIdAST* ast) {
  auto copy = new (arena_) ConversionFunctionIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->operatorLoc = ast->operatorLoc;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(SimpleTemplateIdAST* ast) {
  auto copy = new (arena_) SimpleTemplateIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(LiteralOperatorTemplateIdAST* ast) {
  auto copy = new (arena_) LiteralOperatorTemplateIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->literalOperatorId = accept(ast->literalOperatorId);

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
}

void ASTCloner::visit(OperatorFunctionTemplateIdAST* ast) {
  auto copy = new (arena_) OperatorFunctionTemplateIdAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->operatorFunctionId = accept(ast->operatorFunctionId);

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;
}

void ASTCloner::visit(TypedefSpecifierAST* ast) {
  auto copy = new (arena_) TypedefSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typedefLoc = ast->typedefLoc;
}

void ASTCloner::visit(FriendSpecifierAST* ast) {
  auto copy = new (arena_) FriendSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->friendLoc = ast->friendLoc;
}

void ASTCloner::visit(ConstevalSpecifierAST* ast) {
  auto copy = new (arena_) ConstevalSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->constevalLoc = ast->constevalLoc;
}

void ASTCloner::visit(ConstinitSpecifierAST* ast) {
  auto copy = new (arena_) ConstinitSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->constinitLoc = ast->constinitLoc;
}

void ASTCloner::visit(ConstexprSpecifierAST* ast) {
  auto copy = new (arena_) ConstexprSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->constexprLoc = ast->constexprLoc;
}

void ASTCloner::visit(InlineSpecifierAST* ast) {
  auto copy = new (arena_) InlineSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->inlineLoc = ast->inlineLoc;
}

void ASTCloner::visit(StaticSpecifierAST* ast) {
  auto copy = new (arena_) StaticSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->staticLoc = ast->staticLoc;
}

void ASTCloner::visit(ExternSpecifierAST* ast) {
  auto copy = new (arena_) ExternSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->externLoc = ast->externLoc;
}

void ASTCloner::visit(ThreadLocalSpecifierAST* ast) {
  auto copy = new (arena_) ThreadLocalSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->threadLocalLoc = ast->threadLocalLoc;
}

void ASTCloner::visit(ThreadSpecifierAST* ast) {
  auto copy = new (arena_) ThreadSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->threadLoc = ast->threadLoc;
}

void ASTCloner::visit(MutableSpecifierAST* ast) {
  auto copy = new (arena_) MutableSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->mutableLoc = ast->mutableLoc;
}

void ASTCloner::visit(VirtualSpecifierAST* ast) {
  auto copy = new (arena_) VirtualSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->virtualLoc = ast->virtualLoc;
}

void ASTCloner::visit(ExplicitSpecifierAST* ast) {
  auto copy = new (arena_) ExplicitSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->explicitLoc = ast->explicitLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AutoTypeSpecifierAST* ast) {
  auto copy = new (arena_) AutoTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->autoLoc = ast->autoLoc;
}

void ASTCloner::visit(VoidTypeSpecifierAST* ast) {
  auto copy = new (arena_) VoidTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->voidLoc = ast->voidLoc;
}

void ASTCloner::visit(SizeTypeSpecifierAST* ast) {
  auto copy = new (arena_) SizeTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(SignTypeSpecifierAST* ast) {
  auto copy = new (arena_) SignTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(VaListTypeSpecifierAST* ast) {
  auto copy = new (arena_) VaListTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(IntegralTypeSpecifierAST* ast) {
  auto copy = new (arena_) IntegralTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(FloatingPointTypeSpecifierAST* ast) {
  auto copy = new (arena_) FloatingPointTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->specifierLoc = ast->specifierLoc;

  copy->specifier = ast->specifier;
}

void ASTCloner::visit(ComplexTypeSpecifierAST* ast) {
  auto copy = new (arena_) ComplexTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->complexLoc = ast->complexLoc;
}

void ASTCloner::visit(NamedTypeSpecifierAST* ast) {
  auto copy = new (arena_) NamedTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->templateLoc = ast->templateLoc;

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;
}

void ASTCloner::visit(AtomicTypeSpecifierAST* ast) {
  auto copy = new (arena_) AtomicTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->atomicLoc = ast->atomicLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(UnderlyingTypeSpecifierAST* ast) {
  auto copy = new (arena_) UnderlyingTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->underlyingTypeLoc = ast->underlyingTypeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(ElaboratedTypeSpecifierAST* ast) {
  auto copy = new (arena_) ElaboratedTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->classKey = ast->classKey;
}

void ASTCloner::visit(DecltypeAutoSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeAutoSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->decltypeLoc = ast->decltypeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->autoLoc = ast->autoLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(DecltypeSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->decltypeLoc = ast->decltypeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(PlaceholderTypeSpecifierAST* ast) {
  auto copy = new (arena_) PlaceholderTypeSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typeConstraint = accept(ast->typeConstraint);

  copy->specifier = accept(ast->specifier);
}

void ASTCloner::visit(ConstQualifierAST* ast) {
  auto copy = new (arena_) ConstQualifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->constLoc = ast->constLoc;
}

void ASTCloner::visit(VolatileQualifierAST* ast) {
  auto copy = new (arena_) VolatileQualifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->volatileLoc = ast->volatileLoc;
}

void ASTCloner::visit(RestrictQualifierAST* ast) {
  auto copy = new (arena_) RestrictQualifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->restrictLoc = ast->restrictLoc;
}

void ASTCloner::visit(EnumSpecifierAST* ast) {
  auto copy = new (arena_) EnumSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->enumLoc = ast->enumLoc;

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  copy->commaLoc = ast->commaLoc;

  if (auto it = ast->enumeratorList) {
    auto out = &copy->enumeratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;
}

void ASTCloner::visit(ClassSpecifierAST* ast) {
  auto copy = new (arena_) ClassSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);

  copy->finalLoc = ast->finalLoc;

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->baseSpecifierList) {
    auto out = &copy->baseSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  copy->classKey = ast->classKey;

  copy->isFinal = ast->isFinal;
}

void ASTCloner::visit(TypenameSpecifierAST* ast) {
  auto copy = new (arena_) TypenameSpecifierAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->typenameLoc = ast->typenameLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->unqualifiedId = accept(ast->unqualifiedId);
}

void ASTCloner::visit(BitfieldDeclaratorAST* ast) {
  auto copy = new (arena_) BitfieldDeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->colonLoc = ast->colonLoc;

  copy->sizeExpression = accept(ast->sizeExpression);

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(ParameterPackAST* ast) {
  auto copy = new (arena_) ParameterPackAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->coreDeclarator = accept(ast->coreDeclarator);
}

void ASTCloner::visit(IdDeclaratorAST* ast) {
  auto copy = new (arena_) IdDeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->declaratorId = accept(ast->declaratorId);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(NestedDeclaratorAST* ast) {
  auto copy = new (arena_) NestedDeclaratorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lparenLoc = ast->lparenLoc;

  copy->declarator = accept(ast->declarator);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(PointerOperatorAST* ast) {
  auto copy = new (arena_) PointerOperatorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->starLoc = ast->starLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ReferenceOperatorAST* ast) {
  auto copy = new (arena_) ReferenceOperatorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->refLoc = ast->refLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->refOp = ast->refOp;
}

void ASTCloner::visit(PtrToMemberOperatorAST* ast) {
  auto copy = new (arena_) PtrToMemberOperatorAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->starLoc = ast->starLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(FunctionDeclaratorChunkAST* ast) {
  auto copy = new (arena_) FunctionDeclaratorChunkAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lparenLoc = ast->lparenLoc;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  copy->rparenLoc = ast->rparenLoc;

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->refLoc = ast->refLoc;

  copy->exceptionSpecifier = accept(ast->exceptionSpecifier);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->trailingReturnType = accept(ast->trailingReturnType);

  copy->isFinal = ast->isFinal;

  copy->isOverride = ast->isOverride;

  copy->isPure = ast->isPure;
}

void ASTCloner::visit(ArrayDeclaratorChunkAST* ast) {
  auto copy = new (arena_) ArrayDeclaratorChunkAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lbracketLoc = ast->lbracketLoc;

  copy->expression = accept(ast->expression);

  copy->rbracketLoc = ast->rbracketLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(CxxAttributeAST* ast) {
  auto copy = new (arena_) CxxAttributeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->lbracketLoc = ast->lbracketLoc;

  copy->lbracket2Loc = ast->lbracket2Loc;

  copy->attributeUsingPrefix = accept(ast->attributeUsingPrefix);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbracketLoc = ast->rbracketLoc;

  copy->rbracket2Loc = ast->rbracket2Loc;
}

void ASTCloner::visit(GccAttributeAST* ast) {
  auto copy = new (arena_) GccAttributeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->attributeLoc = ast->attributeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->lparen2Loc = ast->lparen2Loc;

  copy->rparenLoc = ast->rparenLoc;

  copy->rparen2Loc = ast->rparen2Loc;
}

void ASTCloner::visit(AlignasAttributeAST* ast) {
  auto copy = new (arena_) AlignasAttributeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->alignasLoc = ast->alignasLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AsmAttributeAST* ast) {
  auto copy = new (arena_) AsmAttributeAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->asmLoc = ast->asmLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->literalLoc = ast->literalLoc;

  copy->rparenLoc = ast->rparenLoc;

  copy->literal = ast->literal;
}

void ASTCloner::visit(ScopedAttributeTokenAST* ast) {
  auto copy = new (arena_) ScopedAttributeTokenAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->attributeNamespaceLoc = ast->attributeNamespaceLoc;

  copy->scopeLoc = ast->scopeLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->attributeNamespace = ast->attributeNamespace;

  copy->identifier = ast->identifier;
}

void ASTCloner::visit(SimpleAttributeTokenAST* ast) {
  auto copy = new (arena_) SimpleAttributeTokenAST();
  copy_ = copy;

  copy->setChecked(ast->checked());

  copy->identifierLoc = ast->identifierLoc;

  copy->identifier = ast->identifier;
}

}  // namespace cxx
