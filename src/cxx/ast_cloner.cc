
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
}

void ASTCloner::visit(NestedNameSpecifierAST* ast) {
  auto copy = new (arena_) NestedNameSpecifierAST();
  copy_ = copy;

  if (auto it = ast->nameList) {
    auto out = &copy->nameList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(UsingDeclaratorAST* ast) {
  auto copy = new (arena_) UsingDeclaratorAST();
  copy_ = copy;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(HandlerAST* ast) {
  auto copy = new (arena_) HandlerAST();
  copy_ = copy;

  copy->exceptionDeclaration = accept(ast->exceptionDeclaration);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(TemplateArgumentAST* ast) {
  auto copy = new (arena_) TemplateArgumentAST();
  copy_ = copy;
}

void ASTCloner::visit(EnumBaseAST* ast) {
  auto copy = new (arena_) EnumBaseAST();
  copy_ = copy;

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

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ParametersAndQualifiersAST* ast) {
  auto copy = new (arena_) ParametersAndQualifiersAST();
  copy_ = copy;

  copy->parameterDeclarationClause = accept(ast->parameterDeclarationClause);

  if (auto it = ast->cvQualifierList) {
    auto out = &copy->cvQualifierList;

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
}

void ASTCloner::visit(EqualInitializerAST* ast) {
  auto copy = new (arena_) EqualInitializerAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(BracedInitListAST* ast) {
  auto copy = new (arena_) BracedInitListAST();
  copy_ = copy;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ParenInitializerAST* ast) {
  auto copy = new (arena_) ParenInitializerAST();
  copy_ = copy;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(NewParenInitializerAST* ast) {
  auto copy = new (arena_) NewParenInitializerAST();
  copy_ = copy;

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(NewBracedInitializerAST* ast) {
  auto copy = new (arena_) NewBracedInitializerAST();
  copy_ = copy;

  copy->bracedInit = accept(ast->bracedInit);
}

void ASTCloner::visit(EllipsisExceptionDeclarationAST* ast) {
  auto copy = new (arena_) EllipsisExceptionDeclarationAST();
  copy_ = copy;
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
}

void ASTCloner::visit(CharLiteralExpressionAST* ast) {
  auto copy = new (arena_) CharLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(BoolLiteralExpressionAST* ast) {
  auto copy = new (arena_) BoolLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(IntLiteralExpressionAST* ast) {
  auto copy = new (arena_) IntLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(FloatLiteralExpressionAST* ast) {
  auto copy = new (arena_) FloatLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(NullptrLiteralExpressionAST* ast) {
  auto copy = new (arena_) NullptrLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(StringLiteralExpressionAST* ast) {
  auto copy = new (arena_) StringLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(UserDefinedStringLiteralExpressionAST* ast) {
  auto copy = new (arena_) UserDefinedStringLiteralExpressionAST();
  copy_ = copy;
}

void ASTCloner::visit(IdExpressionAST* ast) {
  auto copy = new (arena_) IdExpressionAST();
  copy_ = copy;

  copy->name = accept(ast->name);
}

void ASTCloner::visit(NestedExpressionAST* ast) {
  auto copy = new (arena_) NestedExpressionAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(BinaryExpressionAST* ast) {
  auto copy = new (arena_) BinaryExpressionAST();
  copy_ = copy;

  copy->leftExpression = accept(ast->leftExpression);

  copy->rightExpression = accept(ast->rightExpression);
}

void ASTCloner::visit(AssignmentExpressionAST* ast) {
  auto copy = new (arena_) AssignmentExpressionAST();
  copy_ = copy;

  copy->leftExpression = accept(ast->leftExpression);

  copy->rightExpression = accept(ast->rightExpression);
}

void ASTCloner::visit(CallExpressionAST* ast) {
  auto copy = new (arena_) CallExpressionAST();
  copy_ = copy;

  copy->baseExpression = accept(ast->baseExpression);

  if (auto it = ast->expressionList) {
    auto out = &copy->expressionList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(SubscriptExpressionAST* ast) {
  auto copy = new (arena_) SubscriptExpressionAST();
  copy_ = copy;

  copy->baseExpression = accept(ast->baseExpression);

  copy->indexExpression = accept(ast->indexExpression);
}

void ASTCloner::visit(MemberExpressionAST* ast) {
  auto copy = new (arena_) MemberExpressionAST();
  copy_ = copy;

  copy->baseExpression = accept(ast->baseExpression);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(ConditionalExpressionAST* ast) {
  auto copy = new (arena_) ConditionalExpressionAST();
  copy_ = copy;

  copy->condition = accept(ast->condition);

  copy->iftrueExpression = accept(ast->iftrueExpression);

  copy->iffalseExpression = accept(ast->iffalseExpression);
}

void ASTCloner::visit(CppCastExpressionAST* ast) {
  auto copy = new (arena_) CppCastExpressionAST();
  copy_ = copy;

  copy->typeId = accept(ast->typeId);

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(NewExpressionAST* ast) {
  auto copy = new (arena_) NewExpressionAST();
  copy_ = copy;

  copy->typeId = accept(ast->typeId);

  copy->newInitalizer = accept(ast->newInitalizer);
}

void ASTCloner::visit(LabeledStatementAST* ast) {
  auto copy = new (arena_) LabeledStatementAST();
  copy_ = copy;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(CaseStatementAST* ast) {
  auto copy = new (arena_) CaseStatementAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(DefaultStatementAST* ast) {
  auto copy = new (arena_) DefaultStatementAST();
  copy_ = copy;

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(ExpressionStatementAST* ast) {
  auto copy = new (arena_) ExpressionStatementAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(CompoundStatementAST* ast) {
  auto copy = new (arena_) CompoundStatementAST();
  copy_ = copy;

  if (auto it = ast->statementList) {
    auto out = &copy->statementList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(IfStatementAST* ast) {
  auto copy = new (arena_) IfStatementAST();
  copy_ = copy;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->statement = accept(ast->statement);

  copy->elseStatement = accept(ast->elseStatement);
}

void ASTCloner::visit(SwitchStatementAST* ast) {
  auto copy = new (arena_) SwitchStatementAST();
  copy_ = copy;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(WhileStatementAST* ast) {
  auto copy = new (arena_) WhileStatementAST();
  copy_ = copy;

  copy->condition = accept(ast->condition);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(DoStatementAST* ast) {
  auto copy = new (arena_) DoStatementAST();
  copy_ = copy;

  copy->statement = accept(ast->statement);

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ForRangeStatementAST* ast) {
  auto copy = new (arena_) ForRangeStatementAST();
  copy_ = copy;

  copy->initializer = accept(ast->initializer);

  copy->rangeDeclaration = accept(ast->rangeDeclaration);

  copy->rangeInitializer = accept(ast->rangeInitializer);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(ForStatementAST* ast) {
  auto copy = new (arena_) ForStatementAST();
  copy_ = copy;

  copy->initializer = accept(ast->initializer);

  copy->condition = accept(ast->condition);

  copy->expression = accept(ast->expression);

  copy->statement = accept(ast->statement);
}

void ASTCloner::visit(BreakStatementAST* ast) {
  auto copy = new (arena_) BreakStatementAST();
  copy_ = copy;
}

void ASTCloner::visit(ContinueStatementAST* ast) {
  auto copy = new (arena_) ContinueStatementAST();
  copy_ = copy;
}

void ASTCloner::visit(ReturnStatementAST* ast) {
  auto copy = new (arena_) ReturnStatementAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(GotoStatementAST* ast) {
  auto copy = new (arena_) GotoStatementAST();
  copy_ = copy;
}

void ASTCloner::visit(CoroutineReturnStatementAST* ast) {
  auto copy = new (arena_) CoroutineReturnStatementAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(DeclarationStatementAST* ast) {
  auto copy = new (arena_) DeclarationStatementAST();
  copy_ = copy;

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(TryBlockStatementAST* ast) {
  auto copy = new (arena_) TryBlockStatementAST();
  copy_ = copy;

  copy->statement = accept(ast->statement);

  if (auto it = ast->handlerList) {
    auto out = &copy->handlerList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(FunctionDefinitionAST* ast) {
  auto copy = new (arena_) FunctionDefinitionAST();
  copy_ = copy;

  if (auto it = ast->declSpecifierList) {
    auto out = &copy->declSpecifierList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declarator = accept(ast->declarator);

  copy->functionBody = accept(ast->functionBody);
}

void ASTCloner::visit(ConceptDefinitionAST* ast) {
  auto copy = new (arena_) ConceptDefinitionAST();
  copy_ = copy;

  copy->name = accept(ast->name);

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(ForRangeDeclarationAST* ast) {
  auto copy = new (arena_) ForRangeDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(AliasDeclarationAST* ast) {
  auto copy = new (arena_) AliasDeclarationAST();
  copy_ = copy;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->typeId = accept(ast->typeId);
}

void ASTCloner::visit(SimpleDeclarationAST* ast) {
  auto copy = new (arena_) SimpleDeclarationAST();
  copy_ = copy;

  if (auto it = ast->attributes) {
    auto out = &copy->attributes;

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

  if (auto it = ast->declaratorList) {
    auto out = &copy->declaratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(StaticAssertDeclarationAST* ast) {
  auto copy = new (arena_) StaticAssertDeclarationAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(EmptyDeclarationAST* ast) {
  auto copy = new (arena_) EmptyDeclarationAST();
  copy_ = copy;
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
}

void ASTCloner::visit(OpaqueEnumDeclarationAST* ast) {
  auto copy = new (arena_) OpaqueEnumDeclarationAST();
  copy_ = copy;

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
}

void ASTCloner::visit(UsingEnumDeclarationAST* ast) {
  auto copy = new (arena_) UsingEnumDeclarationAST();
  copy_ = copy;
}

void ASTCloner::visit(NamespaceDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceDefinitionAST();
  copy_ = copy;

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

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(NamespaceAliasDefinitionAST* ast) {
  auto copy = new (arena_) NamespaceAliasDefinitionAST();
  copy_ = copy;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(UsingDirectiveAST* ast) {
  auto copy = new (arena_) UsingDirectiveAST();
  copy_ = copy;
}

void ASTCloner::visit(UsingDeclarationAST* ast) {
  auto copy = new (arena_) UsingDeclarationAST();
  copy_ = copy;

  if (auto it = ast->usingDeclaratorList) {
    auto out = &copy->usingDeclaratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
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

  if (auto it = ast->templateParameterList) {
    auto out = &copy->templateParameterList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->declaration = accept(ast->declaration);
}

void ASTCloner::visit(DeductionGuideAST* ast) {
  auto copy = new (arena_) DeductionGuideAST();
  copy_ = copy;
}

void ASTCloner::visit(ExplicitInstantiationAST* ast) {
  auto copy = new (arena_) ExplicitInstantiationAST();
  copy_ = copy;

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

  copy->expression = accept(ast->expression);
}

void ASTCloner::visit(LinkageSpecificationAST* ast) {
  auto copy = new (arena_) LinkageSpecificationAST();
  copy_ = copy;

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(SimpleNameAST* ast) {
  auto copy = new (arena_) SimpleNameAST();
  copy_ = copy;
}

void ASTCloner::visit(DestructorNameAST* ast) {
  auto copy = new (arena_) DestructorNameAST();
  copy_ = copy;

  copy->name = accept(ast->name);
}

void ASTCloner::visit(DecltypeNameAST* ast) {
  auto copy = new (arena_) DecltypeNameAST();
  copy_ = copy;

  copy->decltypeSpecifier = accept(ast->decltypeSpecifier);
}

void ASTCloner::visit(OperatorNameAST* ast) {
  auto copy = new (arena_) OperatorNameAST();
  copy_ = copy;
}

void ASTCloner::visit(TemplateNameAST* ast) {
  auto copy = new (arena_) TemplateNameAST();
  copy_ = copy;

  copy->name = accept(ast->name);

  if (auto it = ast->templateArgumentList) {
    auto out = &copy->templateArgumentList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(QualifiedNameAST* ast) {
  auto copy = new (arena_) QualifiedNameAST();
  copy_ = copy;

  copy->nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  copy->name = accept(ast->name);
}

void ASTCloner::visit(SimpleSpecifierAST* ast) {
  auto copy = new (arena_) SimpleSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(ExplicitSpecifierAST* ast) {
  auto copy = new (arena_) ExplicitSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(NamedTypeSpecifierAST* ast) {
  auto copy = new (arena_) NamedTypeSpecifierAST();
  copy_ = copy;

  copy->name = accept(ast->name);
}

void ASTCloner::visit(PlaceholderTypeSpecifierHelperAST* ast) {
  auto copy = new (arena_) PlaceholderTypeSpecifierHelperAST();
  copy_ = copy;
}

void ASTCloner::visit(DecltypeSpecifierTypeSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeSpecifierTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(UnderlyingTypeSpecifierAST* ast) {
  auto copy = new (arena_) UnderlyingTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(AtomicTypeSpecifierAST* ast) {
  auto copy = new (arena_) AtomicTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(ElaboratedTypeSpecifierAST* ast) {
  auto copy = new (arena_) ElaboratedTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(DecltypeSpecifierAST* ast) {
  auto copy = new (arena_) DecltypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(PlaceholderTypeSpecifierAST* ast) {
  auto copy = new (arena_) PlaceholderTypeSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(CvQualifierAST* ast) {
  auto copy = new (arena_) CvQualifierAST();
  copy_ = copy;
}

void ASTCloner::visit(EnumSpecifierAST* ast) {
  auto copy = new (arena_) EnumSpecifierAST();
  copy_ = copy;

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

  if (auto it = ast->enumeratorList) {
    auto out = &copy->enumeratorList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(ClassSpecifierAST* ast) {
  auto copy = new (arena_) ClassSpecifierAST();
  copy_ = copy;

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }

  copy->name = accept(ast->name);

  copy->baseClause = accept(ast->baseClause);

  if (auto it = ast->declarationList) {
    auto out = &copy->declarationList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

void ASTCloner::visit(TypenameSpecifierAST* ast) {
  auto copy = new (arena_) TypenameSpecifierAST();
  copy_ = copy;
}

void ASTCloner::visit(IdDeclaratorAST* ast) {
  auto copy = new (arena_) IdDeclaratorAST();
  copy_ = copy;

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

  copy->declarator = accept(ast->declarator);
}

void ASTCloner::visit(PointerOperatorAST* ast) {
  auto copy = new (arena_) PointerOperatorAST();
  copy_ = copy;

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
}

void ASTCloner::visit(ArrayDeclaratorAST* ast) {
  auto copy = new (arena_) ArrayDeclaratorAST();
  copy_ = copy;

  copy->expression = accept(ast->expression);

  if (auto it = ast->attributeList) {
    auto out = &copy->attributeList;

    for (; it; it = it->next) {
      *out = new (arena_) List(accept(it->value));
      out = &(*out)->next;
    }
  }
}

}  // namespace cxx
