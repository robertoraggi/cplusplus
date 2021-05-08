// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

AST* ASTCloner::clone(AST* ast, Arena* arena) {
  if (!ast) return nullptr;
  std::swap(arena_, arena);
  auto copy = accept(ast);
  std::swap(arena_, arena);
  return copy;
}

void ASTCloner::visit(TypeIdAST* ast) {
  auto copy = new (arena_) TypeIdAST();
  copy_ = copy;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->type = ast->type;
}

void ASTCloner::visit(NestedNameSpecifierAST* ast) {
  auto copy = new (arena_) NestedNameSpecifierAST();
  copy_ = copy;

  copy->scopeLoc = ast->scopeLoc;

  if (auto it = ast->nameList) {
    auto out = &copy->nameList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(UsingDeclaratorAST* ast) {
  auto copy = new (arena_) UsingDeclaratorAST();
  copy_ = copy;

  copy->typenameLoc = ast->typenameLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(HandlerAST* ast) {
  auto copy = new (arena_) HandlerAST();
  copy_ = copy;

  copy->catchLoc = ast->catchLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->exceptionDeclaration = accept(ast->exceptionDeclaration);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(EnumBaseAST* ast) {
  auto copy = new (arena_) EnumBaseAST();
  copy_ = copy;

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(EnumeratorAST* ast) {
  auto copy = new (arena_) EnumeratorAST();
  copy_ = copy;

  copy->name = accept(ast->name);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(DeclaratorAST* ast) {
  auto copy = new (arena_) DeclaratorAST();
  copy_ = copy;

  if (auto it = ast->ptrOpList) {
    auto out = &copy->ptrOpList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->coreDeclarator = accept(ast->coreDeclarator);

  if (auto it = ast->modifiers) {
    auto out = &copy->modifiers;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(InitDeclaratorAST* ast) {
  auto copy = new (arena_) InitDeclaratorAST();
  copy_ = copy;

  copy->declarator = accept(ast->declarator);

  copy->initializer = accept(ast->initializer);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(BaseSpecifierAST* ast) {
  auto copy = new (arena_) BaseSpecifierAST();
  copy_ = copy;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->name = accept(ast->name);
}

void ASTCloner::visit(BaseClauseAST* ast) {
  auto copy = new (arena_) BaseClauseAST();
  copy_ = copy;

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->baseSpecifierList) {
    auto out = &copy->baseSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(NewTypeIdAST* ast) {
  auto copy = new (arena_) NewTypeIdAST();
  copy_ = copy;

  if (auto it = ast->typeSpecifierList) {
    auto out = &copy->typeSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ParameterDeclarationClauseAST* ast) {
  auto copy = new (arena_) ParameterDeclarationClauseAST();
  copy_ = copy;

  if (auto it = ast->parameterDeclarationList) {
    auto out = &copy->parameterDeclarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->commaLoc = ast->commaLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(ParametersAndQualifiersAST* ast) {
  auto copy = new (arena_) ParametersAndQualifiersAST();
  copy_ = copy;

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

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(LambdaIntroducerAST* ast) {
  auto copy = new (arena_) LambdaIntroducerAST();
  copy_ = copy;

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
}

void ASTCloner::visit(LambdaDeclaratorAST* ast) {
  auto copy = new (arena_) LambdaDeclaratorAST();
  copy_ = copy;

  copy->lparenLoc = ast->lparenLoc;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  copy->rparenLoc = ast->rparenLoc;

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->trailingReturnType = accept(ast->trailingReturnType);
}

void ASTCloner::visit(TrailingReturnTypeAST* ast) {
  auto copy = new (arena_) TrailingReturnTypeAST();
  copy_ = copy;

  copy->minusGreaterLoc = ast->minusGreaterLoc;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(CtorInitializerAST* ast) {
  auto copy = new (arena_) CtorInitializerAST();
  copy_ = copy;

  copy->colonLoc = ast->colonLoc;

  if (auto it = ast->memInitializerList) {
    auto out = &copy->memInitializerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(TypeTemplateArgumentAST* ast) {
  auto copy = new (arena_) TypeTemplateArgumentAST();
  copy_ = copy;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(ExpressionTemplateArgumentAST* ast) {
  auto copy = new (arena_) ExpressionTemplateArgumentAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ParenMemInitializerAST* ast) {
  auto copy = new (arena_) ParenMemInitializerAST();
  copy_ = copy;

  copy->name = accept(ast->name);

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

  copy->name = accept(ast->name);

  copy->bracedInitList = accept(ast->bracedInitList);

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(ThisLambdaCaptureAST* ast) {
  auto copy = new (arena_) ThisLambdaCaptureAST();
  copy_ = copy;

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(DerefThisLambdaCaptureAST* ast) {
  auto copy = new (arena_) DerefThisLambdaCaptureAST();
  copy_ = copy;

  copy->starLoc = ast->starLoc;

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(SimpleLambdaCaptureAST* ast) {
  auto copy = new (arena_) SimpleLambdaCaptureAST();
  copy_ = copy;

  copy->identifierLoc = ast->identifierLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(RefLambdaCaptureAST* ast) {
  auto copy = new (arena_) RefLambdaCaptureAST();
  copy_ = copy;

  copy->ampLoc = ast->ampLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(RefInitLambdaCaptureAST* ast) {
  auto copy = new (arena_) RefInitLambdaCaptureAST();
  copy_ = copy;

  copy->ampLoc = ast->ampLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->initializer = accept(ast->initializer);
}

void ASTCloner::visit(InitLambdaCaptureAST* ast) {
  auto copy = new (arena_) InitLambdaCaptureAST();
  copy_ = copy;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->initializer = accept(ast->initializer);
}

void ASTCloner::visit(EqualInitializerAST* ast) {
  auto copy = new (arena_) EqualInitializerAST();
  copy_ = copy;

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(BracedInitListAST* ast) {
  auto copy = new (arena_) BracedInitListAST();
  copy_ = copy;

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

void ASTCloner::visit(NewParenInitializerAST* ast) {
  auto copy = new (arena_) NewParenInitializerAST();
  copy_ = copy;

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

  copy->bracedInit = accept(ast->bracedInit);
}

void ASTCloner::visit(EllipsisExceptionDeclarationAST* ast) {
  auto copy = new (arena_) EllipsisExceptionDeclarationAST();
  copy_ = copy;

  copy->ellipsisLoc = ast->ellipsisLoc;
}

void ASTCloner::visit(TypeExceptionDeclarationAST* ast) {
  auto copy = new (arena_) TypeExceptionDeclarationAST();
  copy_ = copy;

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

  copy->equalLoc = ast->equalLoc;

  copy->defaultLoc = ast->defaultLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CompoundStatementFunctionBodyAST* ast) {
  auto copy = new (arena_) CompoundStatementFunctionBodyAST();
  copy_ = copy;

  copy->ctorInitializer = accept(ast->ctorInitializer);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(TryStatementFunctionBodyAST* ast) {
  auto copy = new (arena_) TryStatementFunctionBodyAST();
  copy_ = copy;

  copy->tryLoc = ast->tryLoc;

  copy->ctorInitializer = accept(ast->ctorInitializer);

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

  copy->equalLoc = ast->equalLoc;

  copy->deleteLoc = ast->deleteLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(TranslationUnitAST* ast) {
  auto copy = new (arena_) TranslationUnitAST();
  copy_ = copy;

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
}

void ASTCloner::visit(ThisExpressionAST* ast) {
  auto copy = new (arena_) ThisExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->thisLoc = ast->thisLoc;
}

void ASTCloner::visit(CharLiteralExpressionAST* ast) {
  auto copy = new (arena_) CharLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(BoolLiteralExpressionAST* ast) {
  auto copy = new (arena_) BoolLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(IntLiteralExpressionAST* ast) {
  auto copy = new (arena_) IntLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(FloatLiteralExpressionAST* ast) {
  auto copy = new (arena_) FloatLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(NullptrLiteralExpressionAST* ast) {
  auto copy = new (arena_) NullptrLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(StringLiteralExpressionAST* ast) {
  auto copy = new (arena_) StringLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  if (auto it = ast->stringLiteralList) {
    auto out = &copy->stringLiteralList;

    for (; it; it = it->next) {
      *out = new (arena_) List(it->value);
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(UserDefinedStringLiteralExpressionAST* ast) {
  auto copy = new (arena_) UserDefinedStringLiteralExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->literalLoc = ast->literalLoc;
}

void ASTCloner::visit(IdExpressionAST* ast) {
  auto copy = new (arena_) IdExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->name = accept(ast->name);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(NestedExpressionAST* ast) {
  auto copy = new (arena_) NestedExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(RightFoldExpressionAST* ast) {
  auto copy = new (arena_) RightFoldExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

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

  copy->type = ast->type;

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

  copy->type = ast->type;

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

  copy->type = ast->type;

  copy->lambdaIntroducer = accept(ast->lambdaIntroducer);

  copy->lessLoc = ast->lessLoc;

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->lambdaDeclarator = accept(ast->lambdaDeclarator);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(SizeofExpressionAST* ast) {
  auto copy = new (arena_) SizeofExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(SizeofTypeExpressionAST* ast) {
  auto copy = new (arena_) SizeofTypeExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(SizeofPackExpressionAST* ast) {
  auto copy = new (arena_) SizeofPackExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->sizeofLoc = ast->sizeofLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(TypeidExpressionAST* ast) {
  auto copy = new (arena_) TypeidExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->typeidLoc = ast->typeidLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(TypeidOfTypeExpressionAST* ast) {
  auto copy = new (arena_) TypeidOfTypeExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->typeidLoc = ast->typeidLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AlignofExpressionAST* ast) {
  auto copy = new (arena_) AlignofExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->alignofLoc = ast->alignofLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(UnaryExpressionAST* ast) {
  auto copy = new (arena_) UnaryExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->opLoc = ast->opLoc;

  copy->expression = accept(ast->expression);

  copy->op = ast->op;
}

void ASTCloner::visit(BinaryExpressionAST* ast) {
  auto copy = new (arena_) BinaryExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->leftExpression = accept(ast->leftExpression);

  copy->opLoc = ast->opLoc;

  copy->rightExpression = accept(ast->rightExpression);

  copy->op = ast->op;
}

void ASTCloner::visit(AssignmentExpressionAST* ast) {
  auto copy = new (arena_) AssignmentExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->leftExpression = accept(ast->leftExpression);

  copy->opLoc = ast->opLoc;

  copy->rightExpression = accept(ast->rightExpression);

  copy->op = ast->op;
}

void ASTCloner::visit(BracedTypeConstructionAST* ast) {
  auto copy = new (arena_) BracedTypeConstructionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->typeSpecifier = accept(ast->typeSpecifier);

  copy->bracedInitList = accept(ast->bracedInitList);
}

void ASTCloner::visit(TypeConstructionAST* ast) {
  auto copy = new (arena_) TypeConstructionAST();
  copy_ = copy;

  copy->type = ast->type;

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

  copy->type = ast->type;

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

  copy->type = ast->type;

  copy->baseExpression = accept(ast->baseExpression);

  copy->lbracketLoc = ast->lbracketLoc;

  copy->indexExpression = accept(ast->indexExpression);

  copy->rbracketLoc = ast->rbracketLoc;
}

void ASTCloner::visit(MemberExpressionAST* ast) {
  auto copy = new (arena_) MemberExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->baseExpression = accept(ast->baseExpression);

  copy->accessLoc = ast->accessLoc;

  copy->templateLoc = ast->templateLoc;

  copy->name = accept(ast->name);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(PostIncrExpressionAST* ast) {
  auto copy = new (arena_) PostIncrExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->baseExpression = accept(ast->baseExpression);

  copy->opLoc = ast->opLoc;
}

void ASTCloner::visit(ConditionalExpressionAST* ast) {
  auto copy = new (arena_) ConditionalExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->condition = accept(ast->condition);

  copy->questionLoc = ast->questionLoc;

  copy->iftrueExpression = accept(ast->iftrueExpression);

  copy->colonLoc = ast->colonLoc;

  copy->iffalseExpression = accept(ast->iffalseExpression);
}

void ASTCloner::visit(CastExpressionAST* ast) {
  auto copy = new (arena_) CastExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(CppCastExpressionAST* ast) {
  auto copy = new (arena_) CppCastExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

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

  copy->type = ast->type;

  copy->scopeLoc = ast->scopeLoc;

  copy->newLoc = ast->newLoc;

  copy->typeId = accept(ast->typeId);

  copy->newInitalizer = accept(ast->newInitalizer);
}

void ASTCloner::visit(DeleteExpressionAST* ast) {
  auto copy = new (arena_) DeleteExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->scopeLoc = ast->scopeLoc;

  copy->deleteLoc = ast->deleteLoc;

  copy->lbracketLoc = ast->lbracketLoc;

  copy->rbracketLoc = ast->rbracketLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ThrowExpressionAST* ast) {
  auto copy = new (arena_) ThrowExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->throwLoc = ast->throwLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(NoexceptExpressionAST* ast) {
  auto copy = new (arena_) NoexceptExpressionAST();
  copy_ = copy;

  copy->type = ast->type;

  copy->noexceptLoc = ast->noexceptLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(LabeledStatementAST* ast) {
  auto copy = new (arena_) LabeledStatementAST();
  copy_ = copy;

  copy->identifierLoc = ast->identifierLoc;

  copy->colonLoc = ast->colonLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(CaseStatementAST* ast) {
  auto copy = new (arena_) CaseStatementAST();
  copy_ = copy;

  copy->caseLoc = ast->caseLoc;

  copy->expression = accept(ast->expression);

  copy->colonLoc = ast->colonLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(DefaultStatementAST* ast) {
  auto copy = new (arena_) DefaultStatementAST();
  copy_ = copy;

  copy->defaultLoc = ast->defaultLoc;

  copy->colonLoc = ast->colonLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(ExpressionStatementAST* ast) {
  auto copy = new (arena_) ExpressionStatementAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CompoundStatementAST* ast) {
  auto copy = new (arena_) CompoundStatementAST();
  copy_ = copy;

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

  copy->ifLoc = ast->ifLoc;

  copy->constexprLoc = ast->constexprLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);

  copy->elseStatement = accept(ast->elseStatement);
}

void ASTCloner::visit(SwitchStatementAST* ast) {
  auto copy = new (arena_) SwitchStatementAST();
  copy_ = copy;

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

  copy->whileLoc = ast->whileLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->condition = accept(ast->condition);

  copy->rparenLoc = ast->rparenLoc;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(DoStatementAST* ast) {
  auto copy = new (arena_) DoStatementAST();
  copy_ = copy;

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

  copy->breakLoc = ast->breakLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ContinueStatementAST* ast) {
  auto copy = new (arena_) ContinueStatementAST();
  copy_ = copy;

  copy->continueLoc = ast->continueLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ReturnStatementAST* ast) {
  auto copy = new (arena_) ReturnStatementAST();
  copy_ = copy;

  copy->returnLoc = ast->returnLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(GotoStatementAST* ast) {
  auto copy = new (arena_) GotoStatementAST();
  copy_ = copy;

  copy->gotoLoc = ast->gotoLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(CoroutineReturnStatementAST* ast) {
  auto copy = new (arena_) CoroutineReturnStatementAST();
  copy_ = copy;

  copy->coreturnLoc = ast->coreturnLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(DeclarationStatementAST* ast) {
  auto copy = new (arena_) DeclarationStatementAST();
  copy_ = copy;

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(TryBlockStatementAST* ast) {
  auto copy = new (arena_) TryBlockStatementAST();
  copy_ = copy;

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

  copy->accessLoc = ast->accessLoc;

  copy->colonLoc = ast->colonLoc;
}

void ASTCloner::visit(FunctionDefinitionAST* ast) {
  auto copy = new (arena_) FunctionDefinitionAST();
  copy_ = copy;

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

  copy->functionBody = accept(ast->functionBody);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(ConceptDefinitionAST* ast) {
  auto copy = new (arena_) ConceptDefinitionAST();
  copy_ = copy;

  copy->conceptLoc = ast->conceptLoc;

  copy->name = accept(ast->name);

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ForRangeDeclarationAST* ast) {
  auto copy = new (arena_) ForRangeDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(AliasDeclarationAST* ast) {
  auto copy = new (arena_) AliasDeclarationAST();
  copy_ = copy;

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
}

void ASTCloner::visit(SimpleDeclarationAST* ast) {
  auto copy = new (arena_) SimpleDeclarationAST();
  copy_ = copy;

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

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(StaticAssertDeclarationAST* ast) {
  auto copy = new (arena_) StaticAssertDeclarationAST();
  copy_ = copy;

  copy->staticAssertLoc = ast->staticAssertLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->commaLoc = ast->commaLoc;

  if (auto it = ast->stringLiteralList) {
    auto out = &copy->stringLiteralList;

    for (; it; it = it->next) {
      *out = new (arena_) List(it->value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(EmptyDeclarationAST* ast) {
  auto copy = new (arena_) EmptyDeclarationAST();
  copy_ = copy;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(AttributeDeclarationAST* ast) {
  auto copy = new (arena_) AttributeDeclarationAST();
  copy_ = copy;

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

  copy->name = accept(ast->name);

  copy->enumBase = accept(ast->enumBase);

  copy->emicolonLoc = ast->emicolonLoc;
}

void ASTCloner::visit(UsingEnumDeclarationAST* ast) {
  auto copy = new (arena_) UsingEnumDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(NamespaceDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceDefinitionAST();
  copy_ = copy;

  copy->inlineLoc = ast->inlineLoc;

  copy->namespaceLoc = ast->namespaceLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);

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
}

void ASTCloner::visit(NamespaceAliasDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceAliasDefinitionAST();
  copy_ = copy;

  copy->namespaceLoc = ast->namespaceLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(UsingDirectiveAST* ast) {
  auto copy = new (arena_) UsingDirectiveAST();
  copy_ = copy;
}

void ASTCloner::visit(UsingDeclarationAST* ast) {
  auto copy = new (arena_) UsingDeclarationAST();
  copy_ = copy;

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

void ASTCloner::visit(AsmDeclarationAST* ast) {
  auto copy = new (arena_) AsmDeclarationAST();
  copy_ = copy;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->asmLoc = ast->asmLoc;

  copy->lparenLoc = ast->lparenLoc;

  if (auto it = ast->stringLiteralList) {
    auto out = &copy->stringLiteralList;

    for (; it; it = it->next) {
      *out = new (arena_) List(it->value);
      out = &(*out)->next;
    }
  }

  copy->rparenLoc = ast->rparenLoc;

  copy->semicolonLoc = ast->semicolonLoc;
}

void ASTCloner::visit(ExportDeclarationAST* ast) {
  auto copy = new (arena_) ExportDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(ModuleImportDeclarationAST* ast) {
  auto copy = new (arena_) ModuleImportDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(TemplateDeclarationAST* ast) {
  auto copy = new (arena_) TemplateDeclarationAST();
  copy_ = copy;

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

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(TypenameTypeParameterAST* ast) {
  auto copy = new (arena_) TypenameTypeParameterAST();
  copy_ = copy;

  copy->classKeyLoc = ast->classKeyLoc;

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(TypenamePackTypeParameterAST* ast) {
  auto copy = new (arena_) TypenamePackTypeParameterAST();
  copy_ = copy;

  copy->classKeyLoc = ast->classKeyLoc;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->identifierLoc = ast->identifierLoc;
}

void ASTCloner::visit(TemplateTypeParameterAST* ast) {
  auto copy = new (arena_) TemplateTypeParameterAST();
  copy_ = copy;

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

  copy->identifierLoc = ast->identifierLoc;

  copy->equalLoc = ast->equalLoc;

  copy->name = accept(ast->name);
}

void ASTCloner::visit(TemplatePackTypeParameterAST* ast) {
  auto copy = new (arena_) TemplatePackTypeParameterAST();
  copy_ = copy;

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
}

void ASTCloner::visit(DeductionGuideAST* ast) {
  auto copy = new (arena_) DeductionGuideAST();
  copy_ = copy;
}

void ASTCloner::visit(ExplicitInstantiationAST* ast) {
  auto copy = new (arena_) ExplicitInstantiationAST();
  copy_ = copy;

  copy->externLoc = ast->externLoc;

  copy->templateLoc = ast->templateLoc;

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(ParameterDeclarationAST* ast) {
  auto copy = new (arena_) ParameterDeclarationAST();
  copy_ = copy;

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

  copy->equalLoc = ast->equalLoc;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(LinkageSpecificationAST* ast) {
  auto copy = new (arena_) LinkageSpecificationAST();
  copy_ = copy;

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
}

void ASTCloner::visit(SimpleNameAST* ast) {
  auto copy = new (arena_) SimpleNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->identifierLoc = ast->identifierLoc;
}

void ASTCloner::visit(DestructorNameAST* ast) {
  auto copy = new (arena_) DestructorNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->tildeLoc = ast->tildeLoc;

  copy->id = accept(ast->id);
}

void ASTCloner::visit(DecltypeNameAST* ast) {
  auto copy = new (arena_) DecltypeNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->decltypeSpecifier = accept(ast->decltypeSpecifier);
}

void ASTCloner::visit(OperatorNameAST* ast) {
  auto copy = new (arena_) OperatorNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->operatorLoc = ast->operatorLoc;

  copy->opLoc = ast->opLoc;

  copy->openLoc = ast->openLoc;

  copy->closeLoc = ast->closeLoc;

  copy->op = ast->op;
}

void ASTCloner::visit(ConversionNameAST* ast) {
  auto copy = new (arena_) ConversionNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->operatorLoc = ast->operatorLoc;

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(TemplateNameAST* ast) {
  auto copy = new (arena_) TemplateNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->id = accept(ast->id);

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

void ASTCloner::visit(QualifiedNameAST* ast) {
  auto copy = new (arena_) QualifiedNameAST();
  copy_ = copy;

  copy->name = ast->name;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->templateLoc = ast->templateLoc;

  copy->id = accept(ast->id);
}

void ASTCloner::visit(TypedefSpecifierAST* ast) {
  auto copy = new (arena_) TypedefSpecifierAST();
  copy_ = copy;

  copy->typedefLoc = ast->typedefLoc;
}

void ASTCloner::visit(FriendSpecifierAST* ast) {
  auto copy = new (arena_) FriendSpecifierAST();
  copy_ = copy;

  copy->friendLoc = ast->friendLoc;
}

void ASTCloner::visit(ConstevalSpecifierAST* ast) {
  auto copy = new (arena_) ConstevalSpecifierAST();
  copy_ = copy;

  copy->constevalLoc = ast->constevalLoc;
}

void ASTCloner::visit(ConstinitSpecifierAST* ast) {
  auto copy = new (arena_) ConstinitSpecifierAST();
  copy_ = copy;

  copy->constinitLoc = ast->constinitLoc;
}

void ASTCloner::visit(ConstexprSpecifierAST* ast) {
  auto copy = new (arena_) ConstexprSpecifierAST();
  copy_ = copy;

  copy->constexprLoc = ast->constexprLoc;
}

void ASTCloner::visit(InlineSpecifierAST* ast) {
  auto copy = new (arena_) InlineSpecifierAST();
  copy_ = copy;

  copy->inlineLoc = ast->inlineLoc;
}

void ASTCloner::visit(StaticSpecifierAST* ast) {
  auto copy = new (arena_) StaticSpecifierAST();
  copy_ = copy;

  copy->staticLoc = ast->staticLoc;
}

void ASTCloner::visit(ExternSpecifierAST* ast) {
  auto copy = new (arena_) ExternSpecifierAST();
  copy_ = copy;

  copy->externLoc = ast->externLoc;
}

void ASTCloner::visit(ThreadLocalSpecifierAST* ast) {
  auto copy = new (arena_) ThreadLocalSpecifierAST();
  copy_ = copy;

  copy->threadLocalLoc = ast->threadLocalLoc;
}

void ASTCloner::visit(ThreadSpecifierAST* ast) {
  auto copy = new (arena_) ThreadSpecifierAST();
  copy_ = copy;

  copy->threadLoc = ast->threadLoc;
}

void ASTCloner::visit(MutableSpecifierAST* ast) {
  auto copy = new (arena_) MutableSpecifierAST();
  copy_ = copy;

  copy->mutableLoc = ast->mutableLoc;
}

void ASTCloner::visit(VirtualSpecifierAST* ast) {
  auto copy = new (arena_) VirtualSpecifierAST();
  copy_ = copy;

  copy->virtualLoc = ast->virtualLoc;
}

void ASTCloner::visit(ExplicitSpecifierAST* ast) {
  auto copy = new (arena_) ExplicitSpecifierAST();
  copy_ = copy;

  copy->explicitLoc = ast->explicitLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(AutoTypeSpecifierAST* ast) {
  auto copy = new (arena_) AutoTypeSpecifierAST();
  copy_ = copy;

  copy->autoLoc = ast->autoLoc;
}

void ASTCloner::visit(VoidTypeSpecifierAST* ast) {
  auto copy = new (arena_) VoidTypeSpecifierAST();
  copy_ = copy;

  copy->voidLoc = ast->voidLoc;
}

void ASTCloner::visit(VaListTypeSpecifierAST* ast) {
  auto copy = new (arena_) VaListTypeSpecifierAST();
  copy_ = copy;

  copy->specifierLoc = ast->specifierLoc;
}

void ASTCloner::visit(IntegralTypeSpecifierAST* ast) {
  auto copy = new (arena_) IntegralTypeSpecifierAST();
  copy_ = copy;

  copy->specifierLoc = ast->specifierLoc;
}

void ASTCloner::visit(FloatingPointTypeSpecifierAST* ast) {
  auto copy = new (arena_) FloatingPointTypeSpecifierAST();
  copy_ = copy;

  copy->specifierLoc = ast->specifierLoc;
}

void ASTCloner::visit(ComplexTypeSpecifierAST* ast) {
  auto copy = new (arena_) ComplexTypeSpecifierAST();
  copy_ = copy;

  copy->complexLoc = ast->complexLoc;
}

void ASTCloner::visit(NamedTypeSpecifierAST* ast) {
  auto copy = new (arena_) NamedTypeSpecifierAST();
  copy_ = copy;

  copy->name = accept(ast->name);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(AtomicTypeSpecifierAST* ast) {
  auto copy = new (arena_) AtomicTypeSpecifierAST();
  copy_ = copy;

  copy->atomicLoc = ast->atomicLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->typeId = accept(ast->typeId);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(UnderlyingTypeSpecifierAST* ast) {
  auto copy = new (arena_) UnderlyingTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(ElaboratedTypeSpecifierAST* ast) {
  auto copy = new (arena_) ElaboratedTypeSpecifierAST();
  copy_ = copy;

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(DecltypeAutoSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeAutoSpecifierAST();
  copy_ = copy;

  copy->decltypeLoc = ast->decltypeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->autoLoc = ast->autoLoc;

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(DecltypeSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeSpecifierAST();
  copy_ = copy;

  copy->decltypeLoc = ast->decltypeLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(TypeofSpecifierAST* ast) {
  auto copy = new (arena_) TypeofSpecifierAST();
  copy_ = copy;

  copy->typeofLoc = ast->typeofLoc;

  copy->lparenLoc = ast->lparenLoc;

  copy->expression = accept(ast->expression);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(PlaceholderTypeSpecifierAST* ast) {
  auto copy = new (arena_) PlaceholderTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(ConstQualifierAST* ast) {
  auto copy = new (arena_) ConstQualifierAST();
  copy_ = copy;

  copy->constLoc = ast->constLoc;
}

void ASTCloner::visit(VolatileQualifierAST* ast) {
  auto copy = new (arena_) VolatileQualifierAST();
  copy_ = copy;

  copy->volatileLoc = ast->volatileLoc;
}

void ASTCloner::visit(RestrictQualifierAST* ast) {
  auto copy = new (arena_) RestrictQualifierAST();
  copy_ = copy;

  copy->restrictLoc = ast->restrictLoc;
}

void ASTCloner::visit(EnumSpecifierAST* ast) {
  auto copy = new (arena_) EnumSpecifierAST();
  copy_ = copy;

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

  copy->name = accept(ast->name);

  copy->enumBase = accept(ast->enumBase);

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

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(ClassSpecifierAST* ast) {
  auto copy = new (arena_) ClassSpecifierAST();
  copy_ = copy;

  copy->classLoc = ast->classLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->name = accept(ast->name);

  copy->baseClause = accept(ast->baseClause);

  copy->lbraceLoc = ast->lbraceLoc;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->rbraceLoc = ast->rbraceLoc;

  copy->symbol = ast->symbol;
}

void ASTCloner::visit(TypenameSpecifierAST* ast) {
  auto copy = new (arena_) TypenameSpecifierAST();
  copy_ = copy;

  copy->typenameLoc = ast->typenameLoc;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(IdDeclaratorAST* ast) {
  auto copy = new (arena_) IdDeclaratorAST();
  copy_ = copy;

  copy->ellipsisLoc = ast->ellipsisLoc;

  copy->name = accept(ast->name);

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

  copy->lparenLoc = ast->lparenLoc;

  copy->declarator = accept(ast->declarator);

  copy->rparenLoc = ast->rparenLoc;
}

void ASTCloner::visit(PointerOperatorAST* ast) {
  auto copy = new (arena_) PointerOperatorAST();
  copy_ = copy;

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

  copy->refLoc = ast->refLoc;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(PtrToMemberOperatorAST* ast) {
  auto copy = new (arena_) PtrToMemberOperatorAST();
  copy_ = copy;

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

void ASTCloner::visit(FunctionDeclaratorAST* ast) {
  auto copy = new (arena_) FunctionDeclaratorAST();
  copy_ = copy;

  copy->parametersAndQualifiers = accept(ast->parametersAndQualifiers);

  copy->trailingReturnType = accept(ast->trailingReturnType);
}

void ASTCloner::visit(ArrayDeclaratorAST* ast) {
  auto copy = new (arena_) ArrayDeclaratorAST();
  copy_ = copy;

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

}  // namespace cxx
