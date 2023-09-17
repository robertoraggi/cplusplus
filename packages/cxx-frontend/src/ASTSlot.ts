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

export enum ASTSlot {
  accessLoc = 0,
  accessOp = 1,
  accessSpecifier = 2,
  alignasLoc = 3,
  alignofLoc = 4,
  ampLoc = 5,
  arrowLoc = 6,
  asmLoc = 7,
  atomicLoc = 8,
  attributeArgumentClause = 9,
  attributeList = 10,
  attributeLoc = 11,
  attributeNamespace = 12,
  attributeNamespaceLoc = 13,
  attributeToken = 14,
  attributeUsingPrefix = 15,
  autoLoc = 16,
  awaitLoc = 17,
  baseClause = 18,
  baseExpression = 19,
  baseSpecifierList = 20,
  bindingList = 21,
  bracedInitList = 22,
  breakLoc = 23,
  captureDefaultLoc = 24,
  captureList = 25,
  caseLoc = 26,
  castLoc = 27,
  catchLoc = 28,
  classKey = 29,
  classKeyLoc = 30,
  classLoc = 31,
  closeLoc = 32,
  colonLoc = 33,
  commaLoc = 34,
  complexLoc = 35,
  conceptLoc = 36,
  condition = 37,
  constLoc = 38,
  constevalLoc = 39,
  constexprLoc = 40,
  constinitLoc = 41,
  continueLoc = 42,
  coreDeclarator = 43,
  coreturnLoc = 44,
  ctorInitializer = 45,
  cvQualifierList = 46,
  declSpecifierList = 47,
  declaration = 48,
  declarationList = 49,
  declarator = 50,
  declaratorChunkList = 51,
  declaratorId = 52,
  decltypeLoc = 53,
  decltypeSpecifier = 54,
  defaultLoc = 55,
  deleteLoc = 56,
  designator = 57,
  doLoc = 58,
  dotLoc = 59,
  ellipsisLoc = 60,
  elseLoc = 61,
  elseStatement = 62,
  emicolonLoc = 63,
  enumBase = 64,
  enumLoc = 65,
  enumTypeSpecifier = 66,
  enumeratorList = 67,
  equalLoc = 68,
  exceptionDeclaration = 69,
  exceptionSpecifier = 70,
  explicitLoc = 71,
  explicitSpecifier = 72,
  exportLoc = 73,
  expression = 74,
  expressionList = 75,
  externLoc = 76,
  extraAttributeList = 77,
  finalLoc = 78,
  foldOp = 79,
  foldOpLoc = 80,
  forLoc = 81,
  friendLoc = 82,
  functionBody = 83,
  globalModuleFragment = 84,
  gotoLoc = 85,
  greaterLoc = 86,
  handlerList = 87,
  headerLoc = 88,
  id = 89,
  idExpression = 90,
  identifier = 91,
  identifierLoc = 92,
  ifLoc = 93,
  iffalseExpression = 94,
  iftrueExpression = 95,
  importLoc = 96,
  importName = 97,
  indexExpression = 98,
  initDeclaratorList = 99,
  initializer = 100,
  inlineLoc = 101,
  isFinal = 102,
  isInline = 103,
  isOverride = 104,
  isPack = 105,
  isPure = 106,
  isTemplateIntroduced = 107,
  isTrue = 108,
  isVariadic = 109,
  isVirtual = 110,
  lambdaDeclarator = 111,
  lambdaIntroducer = 112,
  lbraceLoc = 113,
  lbracket2Loc = 114,
  lbracketLoc = 115,
  leftExpression = 116,
  lessLoc = 117,
  literal = 118,
  literalLoc = 119,
  literalOperatorId = 120,
  lparen2Loc = 121,
  lparenLoc = 122,
  memInitializerList = 123,
  memberId = 124,
  minusGreaterLoc = 125,
  moduleDeclaration = 126,
  moduleLoc = 127,
  moduleName = 128,
  modulePartition = 129,
  moduleQualifier = 130,
  mutableLoc = 131,
  namespaceLoc = 132,
  nestedNameSpecifier = 133,
  nestedNamespaceSpecifierList = 134,
  newDeclarator = 135,
  newInitalizer = 136,
  newLoc = 137,
  newPlacement = 138,
  noexceptLoc = 139,
  op = 140,
  opLoc = 141,
  openLoc = 142,
  operatorFunctionId = 143,
  operatorLoc = 144,
  parameterDeclarationClause = 145,
  parameterDeclarationList = 146,
  parametersAndQualifiers = 147,
  privateLoc = 148,
  privateModuleFragment = 149,
  ptrOpList = 150,
  questionLoc = 151,
  rangeDeclaration = 152,
  rangeInitializer = 153,
  rbraceLoc = 154,
  rbracket2Loc = 155,
  rbracketLoc = 156,
  refLoc = 157,
  refOp = 158,
  refQualifierLoc = 159,
  requirementBody = 160,
  requirementList = 161,
  requiresClause = 162,
  requiresLoc = 163,
  restrictLoc = 164,
  returnLoc = 165,
  rightExpression = 166,
  rparen2Loc = 167,
  rparenLoc = 168,
  scopeLoc = 169,
  semicolonLoc = 170,
  sizeExpression = 171,
  sizeofLoc = 172,
  specifier = 173,
  specifierLoc = 174,
  starLoc = 175,
  statement = 176,
  statementList = 177,
  staticAssertLoc = 178,
  staticLoc = 179,
  stringLiteral = 180,
  stringliteralLoc = 181,
  switchLoc = 182,
  templateArgumentList = 183,
  templateId = 184,
  templateLoc = 185,
  templateParameterList = 186,
  thisLoc = 187,
  threadLoc = 188,
  threadLocalLoc = 189,
  throwLoc = 190,
  tildeLoc = 191,
  trailingReturnType = 192,
  tryLoc = 193,
  typeConstraint = 194,
  typeId = 195,
  typeIdList = 196,
  typeSpecifier = 197,
  typeSpecifierList = 198,
  typeTraits = 199,
  typeTraitsLoc = 200,
  typedefLoc = 201,
  typeidLoc = 202,
  typenameLoc = 203,
  underlyingTypeLoc = 204,
  unqualifiedId = 205,
  usingDeclaratorList = 206,
  usingLoc = 207,
  virtualLoc = 208,
  voidLoc = 209,
  volatileLoc = 210,
  whileLoc = 211,
  yieldLoc = 212,
}
