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

#include <cxx/ast.h>
#include <cxx/ast_slot.h>

#include <algorithm>
#include <stdexcept>

namespace cxx {

std::intptr_t ASTSlot::operator()(AST* ast, int slot) {
  if (!ast) return 0;
  std::intptr_t value = 0;
  std::swap(slot_, slot);
  std::swap(value_, value);
  ast->accept(this);
  std::swap(value_, value);
  std::swap(slot_, slot);
  return value;
}

void ASTSlot::visit(TypeIdAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
  }  // switch
}

void ASTSlot::visit(NestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nameList);
      break;
  }  // switch
}

void ASTSlot::visit(UsingDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typenameLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(HandlerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->catchLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionDeclaration);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(TemplateArgumentAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(EnumBaseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      break;
  }  // switch
}

void ASTSlot::visit(EnumeratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 2:
      value_ = ast->equalLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(DeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->ptrOpList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->modifiers);
      break;
  }  // switch
}

void ASTSlot::visit(InitDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
  }  // switch
}

void ASTSlot::visit(BaseSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(BaseClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseSpecifierList);
      break;
  }  // switch
}

void ASTSlot::visit(NewTypeIdAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      break;
  }  // switch
}

void ASTSlot::visit(ParameterDeclarationClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationList);
      break;
    case 1:
      value_ = ast->commaLoc.index();
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ParametersAndQualifiersAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      break;
    case 4:
      value_ = ast->refLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
  }  // switch
}

void ASTSlot::visit(LambdaIntroducerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbracketLoc.index();
      break;
    case 1:
      value_ = ast->captureDefaultLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->captureList);
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(LambdaDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      break;
  }  // switch
}

void ASTSlot::visit(TrailingReturnTypeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->minusGreaterLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
  }  // switch
}

void ASTSlot::visit(CtorInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->memInitializerList);
      break;
  }  // switch
}

void ASTSlot::visit(ParenMemInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
    case 4:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(BracedMemInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->thisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(DerefThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->starLoc.index();
      break;
    case 1:
      value_ = ast->thisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(SimpleLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(RefLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ampLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(RefInitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ampLoc.index();
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      break;
    case 2:
      value_ = ast->identifierLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
  }  // switch
}

void ASTSlot::visit(InitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
  }  // switch
}

void ASTSlot::visit(EqualInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(BracedInitListAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 2:
      value_ = ast->commaLoc.index();
      break;
    case 3:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ParenInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NewParenInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NewBracedInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInit);
      break;
  }  // switch
}

void ASTSlot::visit(EllipsisExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TypeExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
  }  // switch
}

void ASTSlot::visit(DefaultFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      break;
    case 1:
      value_ = ast->defaultLoc.index();
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(CompoundStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(TryStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tryLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      break;
  }  // switch
}

void ASTSlot::visit(DeleteFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      break;
    case 1:
      value_ = ast->deleteLoc.index();
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TranslationUnitAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      break;
  }  // switch
}

void ASTSlot::visit(ModuleUnitAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(ThisExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->thisLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(CharLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(BoolLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(IntLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(FloatLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NullptrLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(StringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      throw std::runtime_error("not implemented yet");
      break;
  }  // switch
}

void ASTSlot::visit(UserDefinedStringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(IdExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(NestedExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(RightFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->opLoc.index();
      break;
    case 3:
      value_ = ast->ellipsisLoc.index();
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(LeftFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      break;
    case 2:
      value_ = ast->opLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(FoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      break;
    case 2:
      value_ = ast->opLoc.index();
      break;
    case 3:
      value_ = ast->ellipsisLoc.index();
      break;
    case 4:
      value_ = ast->foldOpLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(LambdaExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaIntroducer);
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaDeclarator);
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(SizeofExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(SizeofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(SizeofPackExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      break;
    case 3:
      value_ = ast->identifierLoc.index();
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TypeidExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeidLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TypeidOfTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeidLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(AlignofExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->alignofLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(UnaryExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->opLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(BinaryExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      break;
    case 1:
      value_ = ast->opLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      break;
  }  // switch
}

void ASTSlot::visit(AssignmentExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      break;
    case 1:
      value_ = ast->opLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      break;
  }  // switch
}

void ASTSlot::visit(BracedTypeConstructionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      break;
  }  // switch
}

void ASTSlot::visit(TypeConstructionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(CallExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(SubscriptExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      break;
    case 1:
      value_ = ast->lbracketLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->indexExpression);
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(MemberExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      break;
    case 1:
      value_ = ast->accessLoc.index();
      break;
    case 2:
      value_ = ast->templateLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(ConditionalExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      break;
    case 1:
      value_ = ast->questionLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->iftrueExpression);
      break;
    case 3:
      value_ = ast->colonLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->iffalseExpression);
      break;
  }  // switch
}

void ASTSlot::visit(CastExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(CppCastExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->castLoc.index();
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
    case 4:
      value_ = ast->lparenLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NewExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      break;
    case 1:
      value_ = ast->newLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->newInitalizer);
      break;
  }  // switch
}

void ASTSlot::visit(DeleteExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      break;
    case 1:
      value_ = ast->deleteLoc.index();
      break;
    case 2:
      value_ = ast->lbracketLoc.index();
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(ThrowExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->throwLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(NoexceptExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->noexceptLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(LabeledStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      break;
    case 1:
      value_ = ast->colonLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(CaseStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->caseLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->colonLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(DefaultStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->defaultLoc.index();
      break;
    case 1:
      value_ = ast->colonLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(ExpressionStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(CompoundStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statementList);
      break;
    case 2:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(IfStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ifLoc.index();
      break;
    case 1:
      value_ = ast->constexprLoc.index();
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->elseStatement);
      break;
  }  // switch
}

void ASTSlot::visit(SwitchStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->switchLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(WhileStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->whileLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(DoStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->doLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
    case 2:
      value_ = ast->whileLoc.index();
      break;
    case 3:
      value_ = ast->lparenLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      break;
    case 6:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ForRangeStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->forLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeDeclaration);
      break;
    case 4:
      value_ = ast->colonLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeInitializer);
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(ForStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->forLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      break;
    case 4:
      value_ = ast->semicolonLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
  }  // switch
}

void ASTSlot::visit(BreakStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->breakLoc.index();
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ContinueStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->continueLoc.index();
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ReturnStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->returnLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(GotoStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->gotoLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(CoroutineReturnStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->coreturnLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(DeclarationStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      break;
  }  // switch
}

void ASTSlot::visit(TryBlockStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tryLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      break;
  }  // switch
}

void ASTSlot::visit(AccessDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->accessLoc.index();
      break;
    case 1:
      value_ = ast->colonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(FunctionDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->functionBody);
      break;
  }  // switch
}

void ASTSlot::visit(ConceptDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->conceptLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 2:
      value_ = ast->equalLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 4:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ForRangeDeclarationAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(AliasDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 3:
      value_ = ast->equalLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(SimpleDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributes);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initDeclaratorList);
      break;
    case 3:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(StaticAssertDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->staticAssertLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->commaLoc.index();
      break;
    case 4:
      throw std::runtime_error("not implemented yet");
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      break;
    case 6:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(EmptyDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(AttributeDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(OpaqueEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->enumLoc.index();
      break;
    case 1:
      value_ = ast->classLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      break;
    case 6:
      value_ = ast->emicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(UsingEnumDeclarationAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(NamespaceDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->inlineLoc.index();
      break;
    case 1:
      value_ = ast->namespaceLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->extraAttributeList);
      break;
    case 6:
      value_ = ast->lbraceLoc.index();
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      break;
    case 8:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NamespaceAliasDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->namespaceLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = ast->equalLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(UsingDirectiveAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(UsingDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->usingDeclaratorList);
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(AsmDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 1:
      value_ = ast->asmLoc.index();
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      break;
    case 3:
      throw std::runtime_error("not implemented yet");
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ExportDeclarationAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(ModuleImportDeclarationAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(TemplateDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      break;
  }  // switch
}

void ASTSlot::visit(TypenameTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classKeyLoc.index();
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      break;
    case 2:
      value_ = ast->equalLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
  }  // switch
}

void ASTSlot::visit(TypenamePackTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classKeyLoc.index();
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      break;
    case 2:
      value_ = ast->identifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TemplateTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
    case 4:
      value_ = ast->classKeyLoc.index();
      break;
    case 5:
      value_ = ast->identifierLoc.index();
      break;
    case 6:
      value_ = ast->equalLoc.index();
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(TemplatePackTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
    case 4:
      value_ = ast->classKeyLoc.index();
      break;
    case 5:
      value_ = ast->ellipsisLoc.index();
      break;
    case 6:
      value_ = ast->identifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(DeductionGuideAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(ExplicitInstantiationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      break;
    case 1:
      value_ = ast->templateLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      break;
  }  // switch
}

void ASTSlot::visit(ParameterDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
    case 3:
      value_ = ast->equalLoc.index();
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
  }  // switch
}

void ASTSlot::visit(LinkageSpecificationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      break;
    case 1:
      value_ = ast->stringliteralLoc.index();
      break;
    case 2:
      value_ = ast->lbraceLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      break;
    case 4:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(SimpleNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(DestructorNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tildeLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      break;
  }  // switch
}

void ASTSlot::visit(DecltypeNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      break;
  }  // switch
}

void ASTSlot::visit(OperatorNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->operatorLoc.index();
      break;
    case 1:
      value_ = ast->opLoc.index();
      break;
    case 2:
      value_ = ast->openLoc.index();
      break;
    case 3:
      value_ = ast->closeLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TemplateNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      break;
    case 1:
      value_ = ast->lessLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(QualifiedNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 1:
      value_ = ast->templateLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      break;
  }  // switch
}

void ASTSlot::visit(TypedefSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typedefLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(FriendSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->friendLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ConstevalSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constevalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ConstinitSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constinitLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ConstexprSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constexprLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(InlineSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->inlineLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(StaticSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->staticLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ExternSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ThreadLocalSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->threadLocalLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ThreadSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->threadLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(MutableSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->mutableLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(VirtualSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->virtualLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ExplicitSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->explicitLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(AutoTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->autoLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(VoidTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->voidLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(VaListTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(IntegralTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ComplexTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->complexLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(NamedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(AtomicTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->atomicLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(UnderlyingTypeSpecifierAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(ElaboratedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(DecltypeAutoSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->decltypeLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = ast->autoLoc.index();
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(DecltypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->decltypeLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TypeofSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeofLoc.index();
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(PlaceholderTypeSpecifierAST* ast) {
  switch (slot_) {}  // switch
}

void ASTSlot::visit(ConstQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(VolatileQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->volatileLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(RestrictQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->restrictLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(EnumSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->enumLoc.index();
      break;
    case 1:
      value_ = ast->classLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      break;
    case 6:
      value_ = ast->lbraceLoc.index();
      break;
    case 7:
      value_ = ast->commaLoc.index();
      break;
    case 8:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumeratorList);
      break;
    case 9:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(ClassSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseClause);
      break;
    case 4:
      value_ = ast->lbraceLoc.index();
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      break;
    case 6:
      value_ = ast->rbraceLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(TypenameSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typenameLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
  }  // switch
}

void ASTSlot::visit(IdDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
  }  // switch
}

void ASTSlot::visit(NestedDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      break;
  }  // switch
}

void ASTSlot::visit(PointerOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->starLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      break;
  }  // switch
}

void ASTSlot::visit(ReferenceOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->refLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
  }  // switch
}

void ASTSlot::visit(PtrToMemberOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      break;
    case 1:
      value_ = ast->starLoc.index();
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      break;
  }  // switch
}

void ASTSlot::visit(FunctionDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->parametersAndQualifiers);
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      break;
  }  // switch
}

void ASTSlot::visit(ArrayDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbracketLoc.index();
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      break;
    case 2:
      value_ = ast->rbracketLoc.index();
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      break;
  }  // switch
}

}  // namespace cxx
