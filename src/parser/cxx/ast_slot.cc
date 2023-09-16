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
    case 0:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UsingDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(HandlerAST* ast) {
  switch (slot_) {
    case 0:  // catchLoc
      value_ = ast->catchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // exceptionDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(EnumBaseAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(EnumeratorAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // ptrOpList
      value_ = reinterpret_cast<std::intptr_t>(ast->ptrOpList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // coreDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // modifiers
      value_ = reinterpret_cast<std::intptr_t>(ast->modifiers);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(InitDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BaseSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // isTemplateIntroduced
      value_ = intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
    case 5:  // isVirtual
      value_ = intptr_t(ast->isVirtual != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
    case 6:  // accessSpecifier
      value_ = intptr_t(ast->accessSpecifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(BaseClauseAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // baseSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->baseSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NewDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // ptrOpList
      value_ = reinterpret_cast<std::intptr_t>(ast->ptrOpList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // modifiers
      value_ = reinterpret_cast<std::intptr_t>(ast->modifiers);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NewTypeIdAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // newDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->newDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(RequiresClauseAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ParameterDeclarationClauseAST* ast) {
  switch (slot_) {
    case 0:  // parameterDeclarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // isVariadic
      value_ = intptr_t(ast->isVariadic != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParametersAndQualifiersAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // refLoc
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // exceptionSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(LambdaIntroducerAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // captureDefaultLoc
      value_ = ast->captureDefaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // captureList
      value_ = reinterpret_cast<std::intptr_t>(ast->captureList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LambdaDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // exceptionSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 6:  // trailingReturnType
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(TrailingReturnTypeAST* ast) {
  switch (slot_) {
    case 0:  // minusGreaterLoc
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CtorInitializerAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // memInitializerList
      value_ = reinterpret_cast<std::intptr_t>(ast->memInitializerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(RequirementBodyAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // requirementList
      value_ = reinterpret_cast<std::intptr_t>(ast->requirementList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeConstraintAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GlobalModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PrivateModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // privateLoc
      value_ = ast->privateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ModuleQualifierAST* ast) {
  switch (slot_) {
    case 0:  // moduleQualifier
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleQualifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // dotLoc
      value_ = ast->dotLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ModuleNameAST* ast) {
  switch (slot_) {
    case 0:  // moduleQualifier
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleQualifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ModuleDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // modulePartition
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ImportNameAST* ast) {
  switch (slot_) {
    case 0:  // headerLoc
      value_ = ast->headerLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // modulePartition
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ModulePartitionAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AttributeArgumentClauseAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AttributeAST* ast) {
  switch (slot_) {
    case 0:  // attributeToken
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeToken);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // attributeArgumentClause
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeArgumentClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AttributeUsingPrefixAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeNamespaceLoc
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DesignatorAST* ast) {
  switch (slot_) {
    case 0:  // dotLoc
      value_ = ast->dotLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewPlacementAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(GlobalNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(SimpleNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
    case 3:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // decltypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TemplateNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateName
      value_ = reinterpret_cast<std::intptr_t>(ast->templateName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // isTemplateIntroduced
      value_ = intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ThrowExceptionSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // throwLoc
      value_ = ast->throwLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NoexceptSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PackExpansionExpressionAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DesignatedInitializerClauseAST* ast) {
  switch (slot_) {
    case 0:  // designator
      value_ = reinterpret_cast<std::intptr_t>(ast->designator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ThisExpressionAST* ast) {
  switch (slot_) {
    case 0:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(CharLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BoolLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // isTrue
      value_ = intptr_t(ast->isTrue != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IntLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(FloatLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NullptrLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = intptr_t(ast->literal);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(StringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UserDefinedStringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IdExpressionAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // isTemplateIntroduced
      value_ = intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(RequiresExpressionAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // requirementBody
      value_ = reinterpret_cast<std::intptr_t>(ast->requirementBody);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(NestedExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(RightFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(LeftFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(FoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // foldOpLoc
      value_ = ast->foldOpLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
    case 8:  // foldOp
      value_ = intptr_t(ast->foldOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(LambdaExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lambdaIntroducer
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaIntroducer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // lambdaDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(SizeofExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SizeofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SizeofPackExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(TypeidExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeidLoc
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeidOfTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeidLoc
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AlignofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // alignofLoc
      value_ = ast->alignofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AlignofExpressionAST* ast) {
  switch (slot_) {
    case 0:  // alignofLoc
      value_ = ast->alignofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TypeTraitsExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeTraitsLoc
      value_ = ast->typeTraitsLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeIdList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeIdList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // typeTraits
      value_ = intptr_t(ast->typeTraits);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(YieldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // yieldLoc
      value_ = ast->yieldLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AwaitExpressionAST* ast) {
  switch (slot_) {
    case 0:  // awaitLoc
      value_ = ast->awaitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UnaryExpressionAST* ast) {
  switch (slot_) {
    case 0:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BinaryExpressionAST* ast) {
  switch (slot_) {
    case 0:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AssignmentExpressionAST* ast) {
  switch (slot_) {
    case 0:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ConditionExpressionAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(BracedTypeConstructionAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TypeConstructionAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CallExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SubscriptExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // indexExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->indexExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(MemberExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // accessLoc
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // idExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->idExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // accessOp
      value_ = intptr_t(ast->accessOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(PostIncrExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ConditionalExpressionAST* ast) {
  switch (slot_) {
    case 0:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // questionLoc
      value_ = ast->questionLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // iftrueExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->iftrueExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // iffalseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->iffalseExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ImplicitCastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(CastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CppCastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // castLoc
      value_ = ast->castLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(NewExpressionAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // newLoc
      value_ = ast->newLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // newPlacement
      value_ = reinterpret_cast<std::intptr_t>(ast->newPlacement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // newInitalizer
      value_ = reinterpret_cast<std::intptr_t>(ast->newInitalizer);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DeleteExpressionAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // deleteLoc
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ThrowExpressionAST* ast) {
  switch (slot_) {
    case 0:  // throwLoc
      value_ = ast->throwLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NoexceptExpressionAST* ast) {
  switch (slot_) {
    case 0:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(EqualInitializerAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BracedInitListAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParenInitializerAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SimpleRequirementAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundRequirementAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // minusGreaterLoc
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // typeConstraint
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(TypeRequirementAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NestedRequirementAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExpressionTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ParenMemInitializerAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(BracedMemInitializerAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(DerefThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SimpleLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(RefLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ampLoc
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(RefInitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ampLoc
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(InitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NewParenInitializerAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewBracedInitializerAST* ast) {
  switch (slot_) {
    case 0:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(EllipsisExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TypeExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DefaultFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // defaultLoc
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CompoundStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // ctorInitializer
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TryStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // tryLoc
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ctorInitializer
      value_ = reinterpret_cast<std::intptr_t>(ast->ctorInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // handlerList
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DeleteFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // deleteLoc
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TranslationUnitAST* ast) {
  switch (slot_) {
    case 0:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ModuleUnitAST* ast) {
  switch (slot_) {
    case 0:  // globalModuleFragment
      value_ = reinterpret_cast<std::intptr_t>(ast->globalModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // moduleDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // privateModuleFragment
      value_ = reinterpret_cast<std::intptr_t>(ast->privateModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LabeledStatementAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CaseStatementAST* ast) {
  switch (slot_) {
    case 0:  // caseLoc
      value_ = ast->caseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DefaultStatementAST* ast) {
  switch (slot_) {
    case 0:  // defaultLoc
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ExpressionStatementAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundStatementAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // statementList
      value_ = reinterpret_cast<std::intptr_t>(ast->statementList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(IfStatementAST* ast) {
  switch (slot_) {
    case 0:  // ifLoc
      value_ = ast->ifLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // constexprLoc
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:  // elseLoc
      value_ = ast->elseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:  // elseStatement
      value_ = reinterpret_cast<std::intptr_t>(ast->elseStatement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(SwitchStatementAST* ast) {
  switch (slot_) {
    case 0:  // switchLoc
      value_ = ast->switchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(WhileStatementAST* ast) {
  switch (slot_) {
    case 0:  // whileLoc
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DoStatementAST* ast) {
  switch (slot_) {
    case 0:  // doLoc
      value_ = ast->doLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // whileLoc
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(ForRangeStatementAST* ast) {
  switch (slot_) {
    case 0:  // forLoc
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rangeDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // rangeInitializer
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeInitializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(ForStatementAST* ast) {
  switch (slot_) {
    case 0:  // forLoc
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(BreakStatementAST* ast) {
  switch (slot_) {
    case 0:  // breakLoc
      value_ = ast->breakLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ContinueStatementAST* ast) {
  switch (slot_) {
    case 0:  // continueLoc
      value_ = ast->continueLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ReturnStatementAST* ast) {
  switch (slot_) {
    case 0:  // returnLoc
      value_ = ast->returnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(GotoStatementAST* ast) {
  switch (slot_) {
    case 0:  // gotoLoc
      value_ = ast->gotoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CoroutineReturnStatementAST* ast) {
  switch (slot_) {
    case 0:  // coreturnLoc
      value_ = ast->coreturnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DeclarationStatementAST* ast) {
  switch (slot_) {
    case 0:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TryBlockStatementAST* ast) {
  switch (slot_) {
    case 0:  // tryLoc
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // handlerList
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AccessDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // accessLoc
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // accessSpecifier
      value_ = intptr_t(ast->accessSpecifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(FunctionDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // functionBody
      value_ = reinterpret_cast<std::intptr_t>(ast->functionBody);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ConceptDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // conceptLoc
      value_ = ast->conceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ForRangeDeclarationAST* ast) {
  switch (slot_) {}  // switch

  slotCount_ = 0;
}

void ASTSlot::visit(AliasDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(SimpleDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // initDeclaratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->initDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(StructuredBindingDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // refQualifierLoc
      value_ = ast->refQualifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // bindingList
      value_ = reinterpret_cast<std::intptr_t>(ast->bindingList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 5:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(StaticAssertDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // staticAssertLoc
      value_ = ast->staticAssertLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(EmptyDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(AttributeDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(OpaqueEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // enumLoc
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // enumBase
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // emicolonLoc
      value_ = ast->emicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(NestedNamespaceSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // namespaceName
      value_ = reinterpret_cast<std::intptr_t>(ast->namespaceName);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
    case 4:  // isInline
      value_ = intptr_t(ast->isInline != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(NamespaceDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // nestedNamespaceSpecifierList
      value_ =
          reinterpret_cast<std::intptr_t>(ast->nestedNamespaceSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // extraAttributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->extraAttributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 6:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 8:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 9:  // namespaceName
      value_ = reinterpret_cast<std::intptr_t>(ast->namespaceName);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
    case 10:  // isInline
      value_ = intptr_t(ast->isInline != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 11;
}

void ASTSlot::visit(NamespaceAliasDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(UsingDirectiveAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(UsingDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // usingDeclaratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->usingDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(UsingEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // enumTypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->enumTypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AsmDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // asmLoc
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(ExportDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ExportCompoundDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ModuleImportDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // importLoc
      value_ = ast->importLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // importName
      value_ = reinterpret_cast<std::intptr_t>(ast->importName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TemplateDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(TypenameTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // classKeyLoc
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(TemplateTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // classKeyLoc
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:  // idExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->idExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 9:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 10;
}

void ASTSlot::visit(TemplatePackTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // classKeyLoc
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(DeductionGuideAST* ast) {
  switch (slot_) {
    case 0:  // explicitSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->explicitSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // arrowLoc
      value_ = ast->arrowLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 6:  // templateId
      value_ = reinterpret_cast<std::intptr_t>(ast->templateId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(ExplicitInstantiationAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ParameterDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 1:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LinkageSpecificationAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // stringliteralLoc
      value_ = ast->stringliteralLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // stringLiteral
      value_ = reinterpret_cast<std::intptr_t>(ast->stringLiteral);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(SimpleNameAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DestructorNameAST* ast) {
  switch (slot_) {
    case 0:  // tildeLoc
      value_ = ast->tildeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // id
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DecltypeNameAST* ast) {
  switch (slot_) {
    case 0:  // decltypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(OperatorFunctionNameAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // openLoc
      value_ = ast->openLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // closeLoc
      value_ = ast->closeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // op
      value_ = intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LiteralOperatorNameAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ConversionFunctionNameAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SimpleTemplateNameAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LiteralOperatorTemplateNameAST* ast) {
  switch (slot_) {
    case 0:  // literalOperatorName
      value_ = reinterpret_cast<std::intptr_t>(ast->literalOperatorName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(OperatorFunctionTemplateNameAST* ast) {
  switch (slot_) {
    case 0:  // operatorFunctionName
      value_ = reinterpret_cast<std::intptr_t>(ast->operatorFunctionName);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypedefSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typedefLoc
      value_ = ast->typedefLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(FriendSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // friendLoc
      value_ = ast->friendLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstevalSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constevalLoc
      value_ = ast->constevalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstinitSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constinitLoc
      value_ = ast->constinitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstexprSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constexprLoc
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(InlineSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(StaticSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // staticLoc
      value_ = ast->staticLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExternSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadLocalSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // threadLocalLoc
      value_ = ast->threadLocalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // threadLoc
      value_ = ast->threadLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(MutableSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // mutableLoc
      value_ = ast->mutableLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VirtualSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // virtualLoc
      value_ = ast->virtualLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExplicitSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // explicitLoc
      value_ = ast->explicitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AutoTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // autoLoc
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VoidTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // voidLoc
      value_ = ast->voidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VaListTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // specifier
      value_ = intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IntegralTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // specifier
      value_ = intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // specifier
      value_ = intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ComplexTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // complexLoc
      value_ = ast->complexLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(NamedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // isTemplateIntroduced
      value_ = intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AtomicTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // atomicLoc
      value_ = ast->atomicLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(UnderlyingTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // underlyingTypeLoc
      value_ = ast->underlyingTypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ElaboratedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // classKey
      value_ = intptr_t(ast->classKey);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DecltypeAutoSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // decltypeLoc
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // autoLoc
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // decltypeLoc
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PlaceholderTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typeConstraint
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // specifier
      value_ = reinterpret_cast<std::intptr_t>(ast->specifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ConstQualifierAST* ast) {
  switch (slot_) {
    case 0:  // constLoc
      value_ = ast->constLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VolatileQualifierAST* ast) {
  switch (slot_) {
    case 0:  // volatileLoc
      value_ = ast->volatileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(RestrictQualifierAST* ast) {
  switch (slot_) {
    case 0:  // restrictLoc
      value_ = ast->restrictLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(EnumSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // enumLoc
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 5:  // enumBase
      value_ = reinterpret_cast<std::intptr_t>(ast->enumBase);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 8:  // enumeratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->enumeratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 9:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 10;
}

void ASTSlot::visit(ClassSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 4:  // finalLoc
      value_ = ast->finalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // baseClause
      value_ = reinterpret_cast<std::intptr_t>(ast->baseClause);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 6:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 7:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 8:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 9:  // classKey
      value_ = intptr_t(ast->classKey);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
    case 10:  // isFinal
      value_ = intptr_t(ast->isFinal != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 11;
}

void ASTSlot::visit(TypenameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(BitfieldDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // sizeExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->sizeExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParameterPackAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // coreDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IdDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // idExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->idExpression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NestedDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PointerOperatorAST* ast) {
  switch (slot_) {
    case 0:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ReferenceOperatorAST* ast) {
  switch (slot_) {
    case 0:  // refLoc
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 2:  // refOp
      value_ = intptr_t(ast->refOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PtrToMemberOperatorAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 3:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(FunctionDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // parametersAndQualifiers
      value_ = reinterpret_cast<std::intptr_t>(ast->parametersAndQualifiers);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 1:  // trailingReturnType
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // isFinal
      value_ = intptr_t(ast->isFinal != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
    case 3:  // isOverride
      value_ = intptr_t(ast->isOverride != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
    case 4:  // isPure
      value_ = intptr_t(ast->isPure != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ArrayDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 2:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CxxAttributeAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lbracket2Loc
      value_ = ast->lbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // attributeUsingPrefix
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeUsingPrefix);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      break;
    case 4:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 5:  // rbracket2Loc
      value_ = ast->rbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GccAttributeAST* ast) {
  switch (slot_) {
    case 0:  // attributeLoc
      value_ = ast->attributeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // lparen2Loc
      value_ = ast->lparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // rparen2Loc
      value_ = ast->rparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AlignasAttributeAST* ast) {
  switch (slot_) {
    case 0:  // alignasLoc
      value_ = ast->alignasLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AsmAttributeAST* ast) {
  switch (slot_) {
    case 0:  // asmLoc
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 4:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ScopedAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:  // attributeNamespaceLoc
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 1:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SimpleAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      break;
  }  // switch

  slotCount_ = 1;
}

}  // namespace cxx
