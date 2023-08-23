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

#include <cxx/ast.h>
#include <cxx/ast_slot.h>

#include <algorithm>
#include <stdexcept>

namespace cxx {

auto ASTSlot::operator()(AST* ast, int slot)
    -> std::tuple<std::intptr_t, ASTSlotKind, int> {
  std::intptr_t value = 0;
  ASTSlotKind slotKind = ASTSlotKind::kInvalid;
  int slotCount = 0;
  if (ast) {
    std::swap(slot_, slot);
    std::swap(value_, value);
    std::swap(slotKind_, slotKind);
    std::swap(slotCount_, slotCount);
    ast->accept(this);
    std::swap(slotCount_, slotCount);
    std::swap(slotKind_, slotKind);
    std::swap(value_, value);
    std::swap(slot_, slot);
  }
  return std::tuple(value, slotKind, slotCount);
}

void ASTSlot::visit(TypeIdAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nameList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UsingDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(HandlerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->catchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(EnumBaseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(EnumeratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->ptrOpList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->modifiers);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(InitDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BaseSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BaseClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NewTypeIdAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(RequiresClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ParameterDeclarationClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ParametersAndQualifiersAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(LambdaIntroducerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->captureDefaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->captureList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LambdaDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(TrailingReturnTypeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CtorInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->memInitializerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(RequirementBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->requirementList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeConstraintAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(GlobalModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PrivateModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->privateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ModuleDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ModuleNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = 0;  // not implemented yet
      slotKind_ = ASTSlotKind::kTokenList;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ImportNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->headerLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ModulePartitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AttributeArgumentClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AttributeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeToken);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeArgumentClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AttributeUsingPrefixAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DesignatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->dotLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DesignatedInitializerClauseAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->designator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ThisExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(CharLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(BoolLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(IntLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(FloatLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(NullptrLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(StringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(UserDefinedStringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(IdExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(RequiresExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->requirementBody);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(NestedExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(RightFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LeftFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(FoldExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->foldOpLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(LambdaExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaIntroducer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(SizeofExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SizeofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SizeofPackExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(TypeidExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeidOfTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AlignofExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->alignofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeTraitsExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typeTraitsLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeIdList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(UnaryExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BinaryExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AssignmentExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BracedTypeConstructionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TypeConstructionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CallExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SubscriptExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->indexExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(MemberExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PostIncrExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ConditionalExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->questionLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->iftrueExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->iffalseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ImplicitCastExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(CastExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CppCastExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->castLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(NewExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->newLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->newInitalizer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DeleteExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ThrowExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->throwLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NoexceptExpressionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(EqualInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BracedInitListAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParenInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SimpleRequirementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundRequirementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(TypeRequirementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NestedRequirementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExpressionTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ParenMemInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(BracedMemInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(DerefThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SimpleLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(RefLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(RefInitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(InitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewParenInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewBracedInitializerAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInit);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(EllipsisExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TypeExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DefaultFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CompoundStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TryStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DeleteFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TranslationUnitAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ModuleUnitAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->globalModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->privateModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LabeledStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CaseStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->caseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DefaultStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ExpressionStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statementList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(IfStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ifLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->elseStatement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(SwitchStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->switchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(WhileStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DoStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->doLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(ForRangeStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(ForStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(BreakStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->breakLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ContinueStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->continueLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ReturnStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->returnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(GotoStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->gotoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CoroutineReturnStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->coreturnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DeclarationStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TryBlockStatementAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AccessDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(FunctionDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->functionBody);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ConceptDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->conceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ForRangeDeclarationAST* ast) {
  switch (slot_) {}  // switch

  slotCount_ = 0;
}

void ASTSlot::visit(AliasDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(SimpleDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->initDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(StructuredBindingDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->refQualifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->bindingList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 5:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(StaticAssertDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->staticAssertLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(EmptyDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(AttributeDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(OpaqueEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->emicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(NestedNamespaceSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NamespaceDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ =
          reinterpret_cast<std::intptr_t>(ast->nestedNamespaceSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->extraAttributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 6:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 8:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(NamespaceAliasDefinitionAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(UsingDirectiveAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(UsingDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->usingDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(UsingEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumTypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AsmDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ExportDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ExportCompoundDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ModuleImportDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->importLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->importName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TemplateDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(TypenameTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(TemplateTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(TemplatePackTypeParameterAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(DeductionGuideAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->explicitSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = ast->arrowLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(ExplicitInstantiationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ParameterDeclarationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LinkageSpecificationAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->stringliteralLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(SimpleNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(DestructorNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->tildeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DecltypeNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(OperatorNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->openLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->closeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ConversionNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TemplateNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(QualifiedNameAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypedefSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typedefLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(FriendSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->friendLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstevalSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constevalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstinitSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constinitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstexprSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(InlineSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(StaticSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->staticLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExternSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadLocalSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->threadLocalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->threadLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(MutableSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->mutableLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VirtualSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->virtualLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExplicitSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->explicitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AutoTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VoidTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->voidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VaListTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(IntegralTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ComplexTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->complexLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(NamedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(AtomicTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->atomicLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(UnderlyingTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->underlyingTypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ElaboratedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeAutoSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PlaceholderTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->specifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ConstQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->constLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VolatileQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->volatileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(RestrictQualifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->restrictLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(EnumSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:
      value_ = reinterpret_cast<std::intptr_t>(ast->enumeratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 9:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 10;
}

void ASTSlot::visit(ClassSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->finalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = reinterpret_cast<std::intptr_t>(ast->baseClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 7:
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(TypenameSpecifierAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BitfieldDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->sizeExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(IdDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->name);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NestedDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PointerOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ReferenceOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(PtrToMemberOperatorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(FunctionDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = reinterpret_cast<std::intptr_t>(ast->parametersAndQualifiers);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ArrayDeclaratorAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CxxAttributeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeUsingPrefix);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:
      value_ = ast->rbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GCCAttributeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->attributeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->lparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->rparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AlignasAttributeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->alignasLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AsmAttributeAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ScopedAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SimpleAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

}  // namespace cxx
