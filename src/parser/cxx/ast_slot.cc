// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

auto ASTSlot::operator()(AST* ast, int slot) -> SlotInfo {
  std::intptr_t value = 0;
  ASTSlotKind slotKind = ASTSlotKind::kInvalid;
  SlotNameIndex slotNameIndex{};
  int slotCount = 0;
  if (ast) {
    std::swap(slot_, slot);
    std::swap(value_, value);
    std::swap(slotKind_, slotKind);
    std::swap(slotNameIndex_, slotNameIndex);
    std::swap(slotCount_, slotCount);
    ast->accept(this);
    std::swap(slot_, slot);
    std::swap(value_, value);
    std::swap(slotKind_, slotKind);
    std::swap(slotNameIndex_, slotNameIndex);
    std::swap(slotCount_, slotCount);
  }
  return {value, slotKind, SlotNameIndex{slotNameIndex}, slotCount};
}

namespace {
std::string_view kMemberSlotNames[] = {
    "accessLoc",
    "accessOp",
    "accessSpecifier",
    "alignasLoc",
    "alignofLoc",
    "ampLoc",
    "arrowLoc",
    "asmLoc",
    "asmQualifierList",
    "atomicLoc",
    "attributeArgumentClause",
    "attributeList",
    "attributeLoc",
    "attributeNamespace",
    "attributeNamespaceLoc",
    "attributeToken",
    "attributeUsingPrefix",
    "autoLoc",
    "awaitLoc",
    "baseExpression",
    "baseSpecifierList",
    "bindingList",
    "bracedInitList",
    "breakLoc",
    "captureDefault",
    "captureDefaultLoc",
    "captureList",
    "caretLoc",
    "caseLoc",
    "castLoc",
    "catchLoc",
    "classKey",
    "classKeyLoc",
    "classLoc",
    "clobberList",
    "closeLoc",
    "colonLoc",
    "commaLoc",
    "complexLoc",
    "conceptLoc",
    "condition",
    "constLoc",
    "constevalLoc",
    "constexprLoc",
    "constinitLoc",
    "constraintLiteral",
    "constraintLiteralLoc",
    "constvalLoc",
    "continueLoc",
    "coreDeclarator",
    "coreturnLoc",
    "cvQualifierList",
    "declSpecifierList",
    "declaration",
    "declarationList",
    "declarator",
    "declaratorChunkList",
    "decltypeLoc",
    "decltypeSpecifier",
    "defaultLoc",
    "deleteLoc",
    "doLoc",
    "dotLoc",
    "ellipsisLoc",
    "elseLoc",
    "elseStatement",
    "emicolonLoc",
    "enumLoc",
    "enumTypeSpecifier",
    "enumeratorList",
    "equalLoc",
    "exceptionDeclaration",
    "exceptionSpecifier",
    "exclaimLoc",
    "explicitLoc",
    "explicitSpecifier",
    "exportLoc",
    "expression",
    "expressionList",
    "externLoc",
    "extraAttributeList",
    "finalLoc",
    "foldOp",
    "foldOpLoc",
    "forLoc",
    "friendLoc",
    "functionBody",
    "globalModuleFragment",
    "gnuAtributeList",
    "gotoLabelList",
    "gotoLoc",
    "greaterLoc",
    "handlerList",
    "headerLoc",
    "id",
    "idExpression",
    "identifier",
    "identifierLoc",
    "ifLoc",
    "iffalseExpression",
    "iftrueExpression",
    "importLoc",
    "importName",
    "indexExpression",
    "initDeclaratorList",
    "initializer",
    "inlineLoc",
    "inputOperandList",
    "isFinal",
    "isInline",
    "isNot",
    "isOverride",
    "isPack",
    "isPure",
    "isTemplateIntroduced",
    "isThisIntroduced",
    "isTrue",
    "isVariadic",
    "isVirtual",
    "lambdaSpecifierList",
    "lbraceLoc",
    "lbracket2Loc",
    "lbracketLoc",
    "leftExpression",
    "lessLoc",
    "literal",
    "literalLoc",
    "literalOperatorId",
    "lparen2Loc",
    "lparenLoc",
    "memInitializerList",
    "minusGreaterLoc",
    "moduleDeclaration",
    "moduleLoc",
    "moduleName",
    "modulePartition",
    "moduleQualifier",
    "mutableLoc",
    "namespaceLoc",
    "nestedNameSpecifier",
    "nestedNamespaceSpecifierList",
    "newInitalizer",
    "newLoc",
    "newPlacement",
    "noexceptLoc",
    "op",
    "opLoc",
    "openLoc",
    "operatorFunctionId",
    "operatorLoc",
    "outputOperandList",
    "parameterDeclarationClause",
    "parameterDeclarationList",
    "privateLoc",
    "privateModuleFragment",
    "ptrOpList",
    "qualifier",
    "qualifierLoc",
    "questionLoc",
    "rangeDeclaration",
    "rangeInitializer",
    "rbraceLoc",
    "rbracket2Loc",
    "rbracketLoc",
    "refLoc",
    "refOp",
    "refQualifierLoc",
    "requirementList",
    "requiresClause",
    "requiresLoc",
    "restrictLoc",
    "returnLoc",
    "rightExpression",
    "rparen2Loc",
    "rparenLoc",
    "scopeLoc",
    "secondColonLoc",
    "semicolonLoc",
    "sizeExpression",
    "sizeofLoc",
    "specifier",
    "specifierLoc",
    "splicer",
    "starLoc",
    "statement",
    "statementList",
    "staticAssertLoc",
    "staticLoc",
    "stringLiteral",
    "stringliteralLoc",
    "switchLoc",
    "symbolicName",
    "symbolicNameLoc",
    "templateArgumentList",
    "templateId",
    "templateLoc",
    "templateParameterList",
    "templateRequiresClause",
    "thisLoc",
    "threadLoc",
    "threadLocalLoc",
    "throwLoc",
    "tildeLoc",
    "trailingReturnType",
    "tryLoc",
    "typeConstraint",
    "typeId",
    "typeIdList",
    "typeLoc",
    "typeSpecifier",
    "typeSpecifierList",
    "typeTraitLoc",
    "typedefLoc",
    "typeidLoc",
    "typenameLoc",
    "underlyingTypeLoc",
    "unqualifiedId",
    "usingDeclaratorList",
    "usingLoc",
    "vaArgLoc",
    "virtualLoc",
    "voidLoc",
    "volatileLoc",
    "whileLoc",
    "yieldLoc",
};
}  // namespace
std::string_view to_string(SlotNameIndex index) {
  return kMemberSlotNames[int(index)];
}

void ASTSlot::visit(TranslationUnitAST* ast) {
  switch (slot_) {
    case 0:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ModuleUnitAST* ast) {
  switch (slot_) {
    case 0:  // globalModuleFragment
      value_ = reinterpret_cast<std::intptr_t>(ast->globalModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{87};
      break;
    case 1:  // moduleDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{132};
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
    case 3:  // privateModuleFragment
      value_ = reinterpret_cast<std::intptr_t>(ast->privateModuleFragment);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{154};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SimpleDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{52};
      break;
    case 2:  // initDeclaratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->initDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{104};
      break;
    case 3:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AsmDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // asmQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->asmQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{8};
      break;
    case 2:  // asmLoc
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{7};
      break;
    case 3:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 4:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 5:  // outputOperandList
      value_ = reinterpret_cast<std::intptr_t>(ast->outputOperandList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{150};
      break;
    case 6:  // inputOperandList
      value_ = reinterpret_cast<std::intptr_t>(ast->inputOperandList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{107};
      break;
    case 7:  // clobberList
      value_ = reinterpret_cast<std::intptr_t>(ast->clobberList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{34};
      break;
    case 8:  // gotoLabelList
      value_ = reinterpret_cast<std::intptr_t>(ast->gotoLabelList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{89};
      break;
    case 9:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 10:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 11:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 12;
}

void ASTSlot::visit(NamespaceAliasDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{138};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 6:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(UsingDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{218};
      break;
    case 1:  // usingDeclaratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->usingDeclaratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{217};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(UsingEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{218};
      break;
    case 1:  // enumTypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->enumTypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{68};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(UsingDirectiveAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{218};
      break;
    case 2:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{138};
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(StaticAssertDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // staticAssertLoc
      value_ = ast->staticAssertLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{186};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 4:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 5:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(AliasDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{218};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 6:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(OpaqueEnumDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // enumLoc
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{67};
      break;
    case 1:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{33};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 6:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 7:  // emicolonLoc
      value_ = ast->emicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{66};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(FunctionDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{52};
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 3:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 4:  // functionBody
      value_ = reinterpret_cast<std::intptr_t>(ast->functionBody);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{86};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(TemplateDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{196};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 4:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 5:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{53};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ConceptDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // conceptLoc
      value_ = ast->conceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{39};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(DeductionGuideAST* ast) {
  switch (slot_) {
    case 0:  // explicitSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->explicitSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{75};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 3:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{151};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // arrowLoc
      value_ = ast->arrowLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{6};
      break;
    case 6:  // templateId
      value_ = reinterpret_cast<std::intptr_t>(ast->templateId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{194};
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 8:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(ExplicitInstantiationAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{79};
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 2:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{53};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ExportDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{76};
      break;
    case 1:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{53};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ExportCompoundDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{76};
      break;
    case 1:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
    case 3:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LinkageSpecificationAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{79};
      break;
    case 1:  // stringliteralLoc
      value_ = ast->stringliteralLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{189};
      break;
    case 2:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 3:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
    case 4:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
    case 5:  // stringLiteral
      value_ = reinterpret_cast<std::intptr_t>(ast->stringLiteral);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{188};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(NamespaceDefinitionAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{106};
      break;
    case 1:  // namespaceLoc
      value_ = ast->namespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{138};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // nestedNamespaceSpecifierList
      value_ =
          reinterpret_cast<std::intptr_t>(ast->nestedNamespaceSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{140};
      break;
    case 4:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 5:  // extraAttributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->extraAttributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{80};
      break;
    case 6:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 7:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
    case 8:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
    case 9:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 10:  // isInline
      value_ = std::intptr_t(ast->isInline != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{109};
      break;
  }  // switch

  slotCount_ = 11;
}

void ASTSlot::visit(EmptyDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(AttributeDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ModuleImportDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // importLoc
      value_ = ast->importLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{101};
      break;
    case 1:  // importName
      value_ = reinterpret_cast<std::intptr_t>(ast->importName);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{102};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParameterDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{198};
      break;
    case 2:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 3:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 4:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 6:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 7:  // isThisIntroduced
      value_ = std::intptr_t(ast->isThisIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{115};
      break;
    case 8:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(AccessDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // accessLoc
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{0};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // accessSpecifier
      value_ = std::intptr_t(ast->accessSpecifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{2};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ForRangeDeclarationAST* ast) {}

void ASTSlot::visit(StructuredBindingDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{52};
      break;
    case 2:  // refQualifierLoc
      value_ = ast->refQualifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{166};
      break;
    case 3:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 4:  // bindingList
      value_ = reinterpret_cast<std::intptr_t>(ast->bindingList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{21};
      break;
    case 5:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 6:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 7:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(AsmOperandAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 1:  // symbolicNameLoc
      value_ = ast->symbolicNameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{192};
      break;
    case 2:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 3:  // constraintLiteralLoc
      value_ = ast->constraintLiteralLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{46};
      break;
    case 4:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // symbolicName
      value_ = reinterpret_cast<std::intptr_t>(ast->symbolicName);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{191};
      break;
    case 8:  // constraintLiteral
      value_ = reinterpret_cast<std::intptr_t>(ast->constraintLiteral);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{45};
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(AsmQualifierAST* ast) {
  switch (slot_) {
    case 0:  // qualifierLoc
      value_ = ast->qualifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{157};
      break;
    case 1:  // qualifier
      value_ = std::intptr_t(ast->qualifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{156};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AsmClobberAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AsmGotoLabelAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(LabeledStatementAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CaseStatementAST* ast) {
  switch (slot_) {
    case 0:  // caseLoc
      value_ = ast->caseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{28};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DefaultStatementAST* ast) {
  switch (slot_) {
    case 0:  // defaultLoc
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{59};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ExpressionStatementAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundStatementAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 1:  // statementList
      value_ = reinterpret_cast<std::intptr_t>(ast->statementList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{185};
      break;
    case 2:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(IfStatementAST* ast) {
  switch (slot_) {
    case 0:  // ifLoc
      value_ = ast->ifLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{98};
      break;
    case 1:  // constexprLoc
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{43};
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 4:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{40};
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 6:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 7:  // elseLoc
      value_ = ast->elseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{64};
      break;
    case 8:  // elseStatement
      value_ = reinterpret_cast<std::intptr_t>(ast->elseStatement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{65};
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(ConstevalIfStatementAST* ast) {
  switch (slot_) {
    case 0:  // ifLoc
      value_ = ast->ifLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{98};
      break;
    case 1:  // exclaimLoc
      value_ = ast->exclaimLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{73};
      break;
    case 2:  // constvalLoc
      value_ = ast->constvalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{47};
      break;
    case 3:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 4:  // elseLoc
      value_ = ast->elseLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{64};
      break;
    case 5:  // elseStatement
      value_ = reinterpret_cast<std::intptr_t>(ast->elseStatement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{65};
      break;
    case 6:  // isNot
      value_ = std::intptr_t(ast->isNot != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{110};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(SwitchStatementAST* ast) {
  switch (slot_) {
    case 0:  // switchLoc
      value_ = ast->switchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{190};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 3:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{40};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(WhileStatementAST* ast) {
  switch (slot_) {
    case 0:  // whileLoc
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{223};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{40};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 4:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DoStatementAST* ast) {
  switch (slot_) {
    case 0:  // doLoc
      value_ = ast->doLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{61};
      break;
    case 1:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 2:  // whileLoc
      value_ = ast->whileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{223};
      break;
    case 3:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 6:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(ForRangeStatementAST* ast) {
  switch (slot_) {
    case 0:  // forLoc
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{84};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 3:  // rangeDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{159};
      break;
    case 4:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 5:  // rangeInitializer
      value_ = reinterpret_cast<std::intptr_t>(ast->rangeInitializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{160};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(ForStatementAST* ast) {
  switch (slot_) {
    case 0:  // forLoc
      value_ = ast->forLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{84};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 3:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{40};
      break;
    case 4:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(BreakStatementAST* ast) {
  switch (slot_) {
    case 0:  // breakLoc
      value_ = ast->breakLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{23};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ContinueStatementAST* ast) {
  switch (slot_) {
    case 0:  // continueLoc
      value_ = ast->continueLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{48};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ReturnStatementAST* ast) {
  switch (slot_) {
    case 0:  // returnLoc
      value_ = ast->returnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{171};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CoroutineReturnStatementAST* ast) {
  switch (slot_) {
    case 0:  // coreturnLoc
      value_ = ast->coreturnLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{50};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(GotoStatementAST* ast) {
  switch (slot_) {
    case 0:  // gotoLoc
      value_ = ast->gotoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{90};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DeclarationStatementAST* ast) {
  switch (slot_) {
    case 0:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{53};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TryBlockStatementAST* ast) {
  switch (slot_) {
    case 0:  // tryLoc
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{204};
      break;
    case 1:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 2:  // handlerList
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{92};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(GeneratedLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(CharLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BoolLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // isTrue
      value_ = std::intptr_t(ast->isTrue != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{116};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IntLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(FloatLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NullptrLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = std::intptr_t(ast->literal);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(StringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UserDefinedStringLiteralExpressionAST* ast) {
  switch (slot_) {
    case 0:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 1:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ThisExpressionAST* ast) {
  switch (slot_) {
    case 0:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{198};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(NestedExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(IdExpressionAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 3:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(LambdaExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 1:  // captureDefaultLoc
      value_ = ast->captureDefaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{25};
      break;
    case 2:  // captureList
      value_ = reinterpret_cast<std::intptr_t>(ast->captureList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{26};
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 4:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 5:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{196};
      break;
    case 6:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 7:  // templateRequiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->templateRequiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{197};
      break;
    case 8:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 9:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{151};
      break;
    case 10:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 11:  // gnuAtributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->gnuAtributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{88};
      break;
    case 12:  // lambdaSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->lambdaSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{119};
      break;
    case 13:  // exceptionSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{72};
      break;
    case 14:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 15:  // trailingReturnType
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{203};
      break;
    case 16:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 17:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 18:  // captureDefault
      value_ = std::intptr_t(ast->captureDefault);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{24};
      break;
  }  // switch

  slotCount_ = 19;
}

void ASTSlot::visit(FoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{123};
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 4:  // foldOpLoc
      value_ = ast->foldOpLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{83};
      break;
    case 5:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{172};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
    case 8:  // foldOp
      value_ = std::intptr_t(ast->foldOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{82};
      break;
  }  // switch

  slotCount_ = 9;
}

void ASTSlot::visit(RightFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(LeftFoldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(RequiresExpressionAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{169};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{151};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 4:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 5:  // requirementList
      value_ = reinterpret_cast<std::intptr_t>(ast->requirementList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{167};
      break;
    case 6:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(VaArgExpressionAST* ast) {
  switch (slot_) {
    case 0:  // vaArgLoc
      value_ = ast->vaArgLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{219};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(SubscriptExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{19};
      break;
    case 1:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 2:  // indexExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->indexExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{103};
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(CallExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{19};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeConstructionAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{209};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(BracedTypeConstructionAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{209};
      break;
    case 1:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{22};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SpliceMemberExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{19};
      break;
    case 1:  // accessLoc
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{0};
      break;
    case 2:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 3:  // splicer
      value_ = reinterpret_cast<std::intptr_t>(ast->splicer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{182};
      break;
    case 4:  // accessOp
      value_ = std::intptr_t(ast->accessOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{1};
      break;
    case 5:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(MemberExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{19};
      break;
    case 1:  // accessLoc
      value_ = ast->accessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{0};
      break;
    case 2:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 3:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // accessOp
      value_ = std::intptr_t(ast->accessOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{1};
      break;
    case 6:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(PostIncrExpressionAST* ast) {
  switch (slot_) {
    case 0:  // baseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->baseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{19};
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 2:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CppCastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // castLoc
      value_ = ast->castLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{29};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 4:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 5:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(BuiltinBitCastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // castLoc
      value_ = ast->castLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{29};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 5:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(TypeidExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeidLoc
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{213};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeidOfTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeidLoc
      value_ = ast->typeidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{213};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SpliceExpressionAST* ast) {
  switch (slot_) {
    case 0:  // splicer
      value_ = reinterpret_cast<std::intptr_t>(ast->splicer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{182};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(GlobalScopeReflectExpressionAST* ast) {
  switch (slot_) {
    case 0:  // caretLoc
      value_ = ast->caretLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{27};
      break;
    case 1:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NamespaceReflectExpressionAST* ast) {
  switch (slot_) {
    case 0:  // caretLoc
      value_ = ast->caretLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{27};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeIdReflectExpressionAST* ast) {
  switch (slot_) {
    case 0:  // caretLoc
      value_ = ast->caretLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{27};
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ReflectExpressionAST* ast) {
  switch (slot_) {
    case 0:  // caretLoc
      value_ = ast->caretLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{27};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(UnaryExpressionAST* ast) {
  switch (slot_) {
    case 0:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AwaitExpressionAST* ast) {
  switch (slot_) {
    case 0:  // awaitLoc
      value_ = ast->awaitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{18};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SizeofExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{179};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SizeofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{179};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SizeofPackExpressionAST* ast) {
  switch (slot_) {
    case 0:  // sizeofLoc
      value_ = ast->sizeofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{179};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 3:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(AlignofTypeExpressionAST* ast) {
  switch (slot_) {
    case 0:  // alignofLoc
      value_ = ast->alignofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{4};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AlignofExpressionAST* ast) {
  switch (slot_) {
    case 0:  // alignofLoc
      value_ = ast->alignofLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{4};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(NoexceptExpressionAST* ast) {
  switch (slot_) {
    case 0:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{144};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NewExpressionAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
    case 1:  // newLoc
      value_ = ast->newLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{142};
      break;
    case 2:  // newPlacement
      value_ = reinterpret_cast<std::intptr_t>(ast->newPlacement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{143};
      break;
    case 3:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 4:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 5:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 6:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 7:  // newInitalizer
      value_ = reinterpret_cast<std::intptr_t>(ast->newInitalizer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{141};
      break;
  }  // switch

  slotCount_ = 8;
}

void ASTSlot::visit(DeleteExpressionAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
    case 1:  // deleteLoc
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{60};
      break;
    case 2:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 3:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 4:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(CastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ImplicitCastExpressionAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(BinaryExpressionAST* ast) {
  switch (slot_) {
    case 0:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{123};
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 2:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{172};
      break;
    case 3:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ConditionalExpressionAST* ast) {
  switch (slot_) {
    case 0:  // condition
      value_ = reinterpret_cast<std::intptr_t>(ast->condition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{40};
      break;
    case 1:  // questionLoc
      value_ = ast->questionLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{158};
      break;
    case 2:  // iftrueExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->iftrueExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{100};
      break;
    case 3:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 4:  // iffalseExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->iffalseExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{99};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(YieldExpressionAST* ast) {
  switch (slot_) {
    case 0:  // yieldLoc
      value_ = ast->yieldLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{224};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ThrowExpressionAST* ast) {
  switch (slot_) {
    case 0:  // throwLoc
      value_ = ast->throwLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{201};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AssignmentExpressionAST* ast) {
  switch (slot_) {
    case 0:  // leftExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->leftExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{123};
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 2:  // rightExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->rightExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{172};
      break;
    case 3:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PackExpansionExpressionAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DesignatedInitializerClauseAST* ast) {
  switch (slot_) {
    case 0:  // dotLoc
      value_ = ast->dotLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{62};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TypeTraitExpressionAST* ast) {
  switch (slot_) {
    case 0:  // typeTraitLoc
      value_ = ast->typeTraitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{211};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeIdList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeIdList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{207};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ConditionExpressionAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // declSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->declSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{52};
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(EqualInitializerAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(BracedInitListAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 2:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 3:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ParenInitializerAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SplicerAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 4:  // secondColonLoc
      value_ = ast->secondColonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{176};
      break;
    case 5:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GlobalModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{133};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 2:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PrivateModuleFragmentAST* ast) {
  switch (slot_) {
    case 0:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{133};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // privateLoc
      value_ = ast->privateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{153};
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
    case 4:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ModuleDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // exportLoc
      value_ = ast->exportLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{76};
      break;
    case 1:  // moduleLoc
      value_ = ast->moduleLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{133};
      break;
    case 2:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{134};
      break;
    case 3:  // modulePartition
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{135};
      break;
    case 4:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 5:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(ModuleNameAST* ast) {
  switch (slot_) {
    case 0:  // moduleQualifier
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleQualifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{136};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ModuleQualifierAST* ast) {
  switch (slot_) {
    case 0:  // moduleQualifier
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleQualifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{136};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // dotLoc
      value_ = ast->dotLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{62};
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ModulePartitionAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 1:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{134};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ImportNameAST* ast) {
  switch (slot_) {
    case 0:  // headerLoc
      value_ = ast->headerLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{93};
      break;
    case 1:  // modulePartition
      value_ = reinterpret_cast<std::intptr_t>(ast->modulePartition);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{135};
      break;
    case 2:  // moduleName
      value_ = reinterpret_cast<std::intptr_t>(ast->moduleName);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{134};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(InitDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 1:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(DeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // ptrOpList
      value_ = reinterpret_cast<std::intptr_t>(ast->ptrOpList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{155};
      break;
    case 1:  // coreDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{49};
      break;
    case 2:  // declaratorChunkList
      value_ = reinterpret_cast<std::intptr_t>(ast->declaratorChunkList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{56};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(UsingDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{214};
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 4:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(EnumeratorAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 2:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 3:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(TypeIdAST* ast) {
  switch (slot_) {
    case 0:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 1:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(HandlerAST* ast) {
  switch (slot_) {
    case 0:  // catchLoc
      value_ = ast->catchLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{30};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // exceptionDeclaration
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionDeclaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{71};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 4:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(BaseSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 2:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 3:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 4:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
    case 5:  // isVirtual
      value_ = std::intptr_t(ast->isVirtual != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{118};
      break;
    case 6:  // accessSpecifier
      value_ = std::intptr_t(ast->accessSpecifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{2};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(RequiresClauseAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{169};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ParameterDeclarationClauseAST* ast) {
  switch (slot_) {
    case 0:  // parameterDeclarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{152};
      break;
    case 1:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 3:  // isVariadic
      value_ = std::intptr_t(ast->isVariadic != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{117};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(TrailingReturnTypeAST* ast) {
  switch (slot_) {
    case 0:  // minusGreaterLoc
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{131};
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(LambdaSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(TypeConstraintAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 3:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{193};
      break;
    case 4:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(AttributeArgumentClauseAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(AttributeAST* ast) {
  switch (slot_) {
    case 0:  // attributeToken
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeToken);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{15};
      break;
    case 1:  // attributeArgumentClause
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeArgumentClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{10};
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(AttributeUsingPrefixAST* ast) {
  switch (slot_) {
    case 0:  // usingLoc
      value_ = ast->usingLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{218};
      break;
    case 1:  // attributeNamespaceLoc
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{14};
      break;
    case 2:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewPlacementAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NestedNamespaceSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{106};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 4:  // isInline
      value_ = std::intptr_t(ast->isInline != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{109};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(TemplateTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // templateParameterList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateParameterList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{196};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 4:  // requiresClause
      value_ = reinterpret_cast<std::intptr_t>(ast->requiresClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{168};
      break;
    case 5:  // classKeyLoc
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{32};
      break;
    case 6:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 7:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 8:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 9:  // idExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->idExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{95};
      break;
    case 10:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 11:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 12;
}

void ASTSlot::visit(NonTypeTemplateParameterAST* ast) {
  switch (slot_) {
    case 0:  // declaration
      value_ = reinterpret_cast<std::intptr_t>(ast->declaration);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{53};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TypenameTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // classKeyLoc
      value_ = ast->classKeyLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{32};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 6:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(ConstraintTypeParameterAST* ast) {
  switch (slot_) {
    case 0:  // typeConstraint
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{205};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 3:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 4:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 5:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GeneratedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typeLoc
      value_ = ast->typeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{208};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TypedefSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typedefLoc
      value_ = ast->typedefLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{212};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(FriendSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // friendLoc
      value_ = ast->friendLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{85};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstevalSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constevalLoc
      value_ = ast->constevalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{42};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstinitSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constinitLoc
      value_ = ast->constinitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{44};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ConstexprSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // constexprLoc
      value_ = ast->constexprLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{43};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(InlineSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // inlineLoc
      value_ = ast->inlineLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{106};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(StaticSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // staticLoc
      value_ = ast->staticLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{187};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExternSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // externLoc
      value_ = ast->externLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{79};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadLocalSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // threadLocalLoc
      value_ = ast->threadLocalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{200};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThreadSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // threadLoc
      value_ = ast->threadLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{199};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(MutableSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // mutableLoc
      value_ = ast->mutableLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{137};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VirtualSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // virtualLoc
      value_ = ast->virtualLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{220};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExplicitSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // explicitLoc
      value_ = ast->explicitLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{74};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AutoTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // autoLoc
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{17};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VoidTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // voidLoc
      value_ = ast->voidLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{221};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(SizeTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SignTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(VaListTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IntegralTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // specifierLoc
      value_ = ast->specifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{181};
      break;
    case 1:  // specifier
      value_ = std::intptr_t(ast->specifier);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ComplexTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // complexLoc
      value_ = ast->complexLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{38};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(NamedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 3:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(AtomicTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // atomicLoc
      value_ = ast->atomicLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{9};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(UnderlyingTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // underlyingTypeLoc
      value_ = ast->underlyingTypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{215};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ElaboratedTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{33};
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 2:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 3:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // classKey
      value_ = std::intptr_t(ast->classKey);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{31};
      break;
    case 6:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(DecltypeAutoSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // decltypeLoc
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{57};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // autoLoc
      value_ = ast->autoLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{17};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // decltypeLoc
      value_ = ast->decltypeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{57};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(PlaceholderTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typeConstraint
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{205};
      break;
    case 1:  // specifier
      value_ = reinterpret_cast<std::intptr_t>(ast->specifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{180};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(ConstQualifierAST* ast) {
  switch (slot_) {
    case 0:  // constLoc
      value_ = ast->constLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{41};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(VolatileQualifierAST* ast) {
  switch (slot_) {
    case 0:  // volatileLoc
      value_ = ast->volatileLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{222};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(RestrictQualifierAST* ast) {
  switch (slot_) {
    case 0:  // restrictLoc
      value_ = ast->restrictLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{170};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(EnumSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // enumLoc
      value_ = ast->enumLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{67};
      break;
    case 1:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{33};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 4:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 5:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 6:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 7:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 8:  // commaLoc
      value_ = ast->commaLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{37};
      break;
    case 9:  // enumeratorList
      value_ = reinterpret_cast<std::intptr_t>(ast->enumeratorList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{69};
      break;
    case 10:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
  }  // switch

  slotCount_ = 11;
}

void ASTSlot::visit(ClassSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // classLoc
      value_ = ast->classLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{33};
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 2:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 3:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 4:  // finalLoc
      value_ = ast->finalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{81};
      break;
    case 5:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 6:  // baseSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->baseSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{20};
      break;
    case 7:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 8:  // declarationList
      value_ = reinterpret_cast<std::intptr_t>(ast->declarationList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{54};
      break;
    case 9:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
    case 10:  // classKey
      value_ = std::intptr_t(ast->classKey);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{31};
      break;
    case 11:  // isFinal
      value_ = std::intptr_t(ast->isFinal != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{108};
      break;
  }  // switch

  slotCount_ = 12;
}

void ASTSlot::visit(TypenameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{214};
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(SplicerTypeSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{214};
      break;
    case 1:  // splicer
      value_ = reinterpret_cast<std::intptr_t>(ast->splicer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{182};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(PointerOperatorAST* ast) {
  switch (slot_) {
    case 0:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{183};
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 2:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{51};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ReferenceOperatorAST* ast) {
  switch (slot_) {
    case 0:  // refLoc
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{164};
      break;
    case 1:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 2:  // refOp
      value_ = std::intptr_t(ast->refOp);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{165};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(PtrToMemberOperatorAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{183};
      break;
    case 2:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 3:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{51};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(BitfieldDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // sizeExpression
      value_ = reinterpret_cast<std::intptr_t>(ast->sizeExpression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{178};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(ParameterPackAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 1:  // coreDeclarator
      value_ = reinterpret_cast<std::intptr_t>(ast->coreDeclarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{49};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(IdDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 3:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 4:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(NestedDeclaratorAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(FunctionDeclaratorChunkAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // parameterDeclarationClause
      value_ = reinterpret_cast<std::intptr_t>(ast->parameterDeclarationClause);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{151};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 3:  // cvQualifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->cvQualifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{51};
      break;
    case 4:  // refLoc
      value_ = ast->refLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{164};
      break;
    case 5:  // exceptionSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->exceptionSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{72};
      break;
    case 6:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 7:  // trailingReturnType
      value_ = reinterpret_cast<std::intptr_t>(ast->trailingReturnType);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{203};
      break;
    case 8:  // isFinal
      value_ = std::intptr_t(ast->isFinal != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{108};
      break;
    case 9:  // isOverride
      value_ = std::intptr_t(ast->isOverride != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{111};
      break;
    case 10:  // isPure
      value_ = std::intptr_t(ast->isPure != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{113};
      break;
  }  // switch

  slotCount_ = 11;
}

void ASTSlot::visit(ArrayDeclaratorChunkAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 3:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NameIdAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DestructorIdAST* ast) {
  switch (slot_) {
    case 0:  // tildeLoc
      value_ = ast->tildeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{202};
      break;
    case 1:  // id
      value_ = reinterpret_cast<std::intptr_t>(ast->id);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{94};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(DecltypeIdAST* ast) {
  switch (slot_) {
    case 0:  // decltypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{58};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(OperatorFunctionIdAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{149};
      break;
    case 1:  // opLoc
      value_ = ast->opLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{146};
      break;
    case 2:  // openLoc
      value_ = ast->openLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{147};
      break;
    case 3:  // closeLoc
      value_ = ast->closeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{35};
      break;
    case 4:  // op
      value_ = std::intptr_t(ast->op);
      slotKind_ = ASTSlotKind::kIntAttribute;
      slotNameIndex_ = SlotNameIndex{145};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LiteralOperatorIdAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{149};
      break;
    case 1:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 3:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ConversionFunctionIdAST* ast) {
  switch (slot_) {
    case 0:  // operatorLoc
      value_ = ast->operatorLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{149};
      break;
    case 1:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SimpleTemplateIdAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{193};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(LiteralOperatorTemplateIdAST* ast) {
  switch (slot_) {
    case 0:  // literalOperatorId
      value_ = reinterpret_cast<std::intptr_t>(ast->literalOperatorId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{127};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{193};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(OperatorFunctionTemplateIdAST* ast) {
  switch (slot_) {
    case 0:  // operatorFunctionId
      value_ = reinterpret_cast<std::intptr_t>(ast->operatorFunctionId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{148};
      break;
    case 1:  // lessLoc
      value_ = ast->lessLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{124};
      break;
    case 2:  // templateArgumentList
      value_ = reinterpret_cast<std::intptr_t>(ast->templateArgumentList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{193};
      break;
    case 3:  // greaterLoc
      value_ = ast->greaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{91};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(GlobalNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(SimpleNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
    case 3:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(DecltypeNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // decltypeSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->decltypeSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{58};
      break;
    case 2:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TemplateNestedNameSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // templateLoc
      value_ = ast->templateLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{195};
      break;
    case 2:  // templateId
      value_ = reinterpret_cast<std::intptr_t>(ast->templateId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{194};
      break;
    case 3:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
    case 4:  // isTemplateIntroduced
      value_ = std::intptr_t(ast->isTemplateIntroduced != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{114};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DefaultFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 1:  // defaultLoc
      value_ = ast->defaultLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{59};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CompoundStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 1:  // memInitializerList
      value_ = reinterpret_cast<std::intptr_t>(ast->memInitializerList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{130};
      break;
    case 2:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TryStatementFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // tryLoc
      value_ = ast->tryLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{204};
      break;
    case 1:  // colonLoc
      value_ = ast->colonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{36};
      break;
    case 2:  // memInitializerList
      value_ = reinterpret_cast<std::intptr_t>(ast->memInitializerList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{130};
      break;
    case 3:  // statement
      value_ = reinterpret_cast<std::intptr_t>(ast->statement);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{184};
      break;
    case 4:  // handlerList
      value_ = reinterpret_cast<std::intptr_t>(ast->handlerList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{92};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(DeleteFunctionBodyAST* ast) {
  switch (slot_) {
    case 0:  // equalLoc
      value_ = ast->equalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{70};
      break;
    case 1:  // deleteLoc
      value_ = ast->deleteLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{60};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(TypeTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ExpressionTemplateArgumentAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ThrowExceptionSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // throwLoc
      value_ = ast->throwLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{201};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NoexceptSpecifierAST* ast) {
  switch (slot_) {
    case 0:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{144};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(SimpleRequirementAST* ast) {
  switch (slot_) {
    case 0:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 1:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(CompoundRequirementAST* ast) {
  switch (slot_) {
    case 0:  // lbraceLoc
      value_ = ast->lbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{120};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // rbraceLoc
      value_ = ast->rbraceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{161};
      break;
    case 3:  // noexceptLoc
      value_ = ast->noexceptLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{144};
      break;
    case 4:  // minusGreaterLoc
      value_ = ast->minusGreaterLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{131};
      break;
    case 5:  // typeConstraint
      value_ = reinterpret_cast<std::intptr_t>(ast->typeConstraint);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{205};
      break;
    case 6:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 7;
}

void ASTSlot::visit(TypeRequirementAST* ast) {
  switch (slot_) {
    case 0:  // typenameLoc
      value_ = ast->typenameLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{214};
      break;
    case 1:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 2:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 3:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(NestedRequirementAST* ast) {
  switch (slot_) {
    case 0:  // requiresLoc
      value_ = ast->requiresLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{169};
      break;
    case 1:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 2:  // semicolonLoc
      value_ = ast->semicolonLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{177};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewParenInitializerAST* ast) {
  switch (slot_) {
    case 0:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 1:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 2:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(NewBracedInitializerAST* ast) {
  switch (slot_) {
    case 0:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{22};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(ParenMemInitializerAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 2:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 3:  // expressionList
      value_ = reinterpret_cast<std::intptr_t>(ast->expressionList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{78};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(BracedMemInitializerAST* ast) {
  switch (slot_) {
    case 0:  // nestedNameSpecifier
      value_ = reinterpret_cast<std::intptr_t>(ast->nestedNameSpecifier);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{139};
      break;
    case 1:  // unqualifiedId
      value_ = reinterpret_cast<std::intptr_t>(ast->unqualifiedId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{216};
      break;
    case 2:  // bracedInitList
      value_ = reinterpret_cast<std::intptr_t>(ast->bracedInitList);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{22};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(ThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{198};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(DerefThisLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // starLoc
      value_ = ast->starLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{183};
      break;
    case 1:  // thisLoc
      value_ = ast->thisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{198};
      break;
  }  // switch

  slotCount_ = 2;
}

void ASTSlot::visit(SimpleLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(RefLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ampLoc
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{5};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(RefInitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ampLoc
      value_ = ast->ampLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{5};
      break;
    case 1:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 3:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(InitLambdaCaptureAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 1:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 2:  // initializer
      value_ = reinterpret_cast<std::intptr_t>(ast->initializer);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{105};
      break;
    case 3:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 4;
}

void ASTSlot::visit(EllipsisExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
  }  // switch

  slotCount_ = 1;
}

void ASTSlot::visit(TypeExceptionDeclarationAST* ast) {
  switch (slot_) {
    case 0:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 1:  // typeSpecifierList
      value_ = reinterpret_cast<std::intptr_t>(ast->typeSpecifierList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{210};
      break;
    case 2:  // declarator
      value_ = reinterpret_cast<std::intptr_t>(ast->declarator);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{55};
      break;
  }  // switch

  slotCount_ = 3;
}

void ASTSlot::visit(CxxAttributeAST* ast) {
  switch (slot_) {
    case 0:  // lbracketLoc
      value_ = ast->lbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{122};
      break;
    case 1:  // lbracket2Loc
      value_ = ast->lbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{121};
      break;
    case 2:  // attributeUsingPrefix
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeUsingPrefix);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{16};
      break;
    case 3:  // attributeList
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeList);
      slotKind_ = ASTSlotKind::kNodeList;
      slotNameIndex_ = SlotNameIndex{11};
      break;
    case 4:  // rbracketLoc
      value_ = ast->rbracketLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{163};
      break;
    case 5:  // rbracket2Loc
      value_ = ast->rbracket2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{162};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(GccAttributeAST* ast) {
  switch (slot_) {
    case 0:  // attributeLoc
      value_ = ast->attributeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{12};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // lparen2Loc
      value_ = ast->lparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{128};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 4:  // rparen2Loc
      value_ = ast->rparen2Loc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{173};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(AlignasAttributeAST* ast) {
  switch (slot_) {
    case 0:  // alignasLoc
      value_ = ast->alignasLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{3};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // expression
      value_ = reinterpret_cast<std::intptr_t>(ast->expression);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{77};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(AlignasTypeAttributeAST* ast) {
  switch (slot_) {
    case 0:  // alignasLoc
      value_ = ast->alignasLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{3};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // typeId
      value_ = reinterpret_cast<std::intptr_t>(ast->typeId);
      slotKind_ = ASTSlotKind::kNode;
      slotNameIndex_ = SlotNameIndex{206};
      break;
    case 3:  // ellipsisLoc
      value_ = ast->ellipsisLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{63};
      break;
    case 4:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 5:  // isPack
      value_ = std::intptr_t(ast->isPack != 0);
      slotKind_ = ASTSlotKind::kBoolAttribute;
      slotNameIndex_ = SlotNameIndex{112};
      break;
  }  // switch

  slotCount_ = 6;
}

void ASTSlot::visit(AsmAttributeAST* ast) {
  switch (slot_) {
    case 0:  // asmLoc
      value_ = ast->asmLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{7};
      break;
    case 1:  // lparenLoc
      value_ = ast->lparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{129};
      break;
    case 2:  // literalLoc
      value_ = ast->literalLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{126};
      break;
    case 3:  // rparenLoc
      value_ = ast->rparenLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{174};
      break;
    case 4:  // literal
      value_ = reinterpret_cast<std::intptr_t>(ast->literal);
      slotKind_ = ASTSlotKind::kLiteralAttribute;
      slotNameIndex_ = SlotNameIndex{125};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(ScopedAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:  // attributeNamespaceLoc
      value_ = ast->attributeNamespaceLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{14};
      break;
    case 1:  // scopeLoc
      value_ = ast->scopeLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{175};
      break;
    case 2:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 3:  // attributeNamespace
      value_ = reinterpret_cast<std::intptr_t>(ast->attributeNamespace);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{13};
      break;
    case 4:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 5;
}

void ASTSlot::visit(SimpleAttributeTokenAST* ast) {
  switch (slot_) {
    case 0:  // identifierLoc
      value_ = ast->identifierLoc.index();
      slotKind_ = ASTSlotKind::kToken;
      slotNameIndex_ = SlotNameIndex{97};
      break;
    case 1:  // identifier
      value_ = reinterpret_cast<std::intptr_t>(ast->identifier);
      slotKind_ = ASTSlotKind::kIdentifierAttribute;
      slotNameIndex_ = SlotNameIndex{96};
      break;
  }  // switch

  slotCount_ = 2;
}

}  // namespace cxx
