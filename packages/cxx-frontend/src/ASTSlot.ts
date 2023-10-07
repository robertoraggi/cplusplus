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
  asmQualifierList = 8,
  atomicLoc = 9,
  attributeArgumentClause = 10,
  attributeList = 11,
  attributeLoc = 12,
  attributeNamespace = 13,
  attributeNamespaceLoc = 14,
  attributeToken = 15,
  attributeUsingPrefix = 16,
  autoLoc = 17,
  awaitLoc = 18,
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
  clobberList = 32,
  closeLoc = 33,
  colonLoc = 34,
  commaLoc = 35,
  complexLoc = 36,
  conceptLoc = 37,
  condition = 38,
  constLoc = 39,
  constevalLoc = 40,
  constexprLoc = 41,
  constinitLoc = 42,
  constraintLiteral = 43,
  constraintLiteralLoc = 44,
  constvalLoc = 45,
  continueLoc = 46,
  coreDeclarator = 47,
  coreturnLoc = 48,
  cvQualifierList = 49,
  declSpecifierList = 50,
  declaration = 51,
  declarationList = 52,
  declarator = 53,
  declaratorChunkList = 54,
  declaratorId = 55,
  decltypeLoc = 56,
  decltypeSpecifier = 57,
  defaultLoc = 58,
  deleteLoc = 59,
  doLoc = 60,
  dotLoc = 61,
  ellipsisLoc = 62,
  elseLoc = 63,
  elseStatement = 64,
  emicolonLoc = 65,
  enumBase = 66,
  enumLoc = 67,
  enumTypeSpecifier = 68,
  enumeratorList = 69,
  equalLoc = 70,
  exceptionDeclaration = 71,
  exceptionSpecifier = 72,
  exclaimLoc = 73,
  explicitLoc = 74,
  explicitSpecifier = 75,
  exportLoc = 76,
  expression = 77,
  expressionList = 78,
  externLoc = 79,
  extraAttributeList = 80,
  finalLoc = 81,
  foldOp = 82,
  foldOpLoc = 83,
  forLoc = 84,
  friendLoc = 85,
  functionBody = 86,
  globalModuleFragment = 87,
  gotoLabelList = 88,
  gotoLoc = 89,
  greaterLoc = 90,
  handlerList = 91,
  headerLoc = 92,
  id = 93,
  idExpression = 94,
  identifier = 95,
  identifierLoc = 96,
  ifLoc = 97,
  iffalseExpression = 98,
  iftrueExpression = 99,
  importLoc = 100,
  importName = 101,
  indexExpression = 102,
  initDeclaratorList = 103,
  initializer = 104,
  inlineLoc = 105,
  inputOperandList = 106,
  isFinal = 107,
  isInline = 108,
  isNot = 109,
  isOverride = 110,
  isPack = 111,
  isPure = 112,
  isTemplateIntroduced = 113,
  isThisIntroduced = 114,
  isTrue = 115,
  isVariadic = 116,
  isVirtual = 117,
  lambdaSpecifierList = 118,
  lbraceLoc = 119,
  lbracket2Loc = 120,
  lbracketLoc = 121,
  leftExpression = 122,
  lessLoc = 123,
  literal = 124,
  literalLoc = 125,
  literalOperatorId = 126,
  lparen2Loc = 127,
  lparenLoc = 128,
  memInitializerList = 129,
  memberId = 130,
  minusGreaterLoc = 131,
  moduleDeclaration = 132,
  moduleLoc = 133,
  moduleName = 134,
  modulePartition = 135,
  moduleQualifier = 136,
  mutableLoc = 137,
  namespaceLoc = 138,
  nestedNameSpecifier = 139,
  nestedNamespaceSpecifierList = 140,
  newInitalizer = 141,
  newLoc = 142,
  newPlacement = 143,
  noexceptLoc = 144,
  op = 145,
  opLoc = 146,
  openLoc = 147,
  operatorFunctionId = 148,
  operatorLoc = 149,
  outputOperandList = 150,
  parameterDeclarationClause = 151,
  parameterDeclarationList = 152,
  privateLoc = 153,
  privateModuleFragment = 154,
  ptrOpList = 155,
  qualifier = 156,
  qualifierLoc = 157,
  questionLoc = 158,
  rangeDeclaration = 159,
  rangeInitializer = 160,
  rbraceLoc = 161,
  rbracket2Loc = 162,
  rbracketLoc = 163,
  refLoc = 164,
  refOp = 165,
  refQualifierLoc = 166,
  requirementList = 167,
  requiresClause = 168,
  requiresLoc = 169,
  restrictLoc = 170,
  returnLoc = 171,
  rightExpression = 172,
  rparen2Loc = 173,
  rparenLoc = 174,
  scopeLoc = 175,
  semicolonLoc = 176,
  sizeExpression = 177,
  sizeofLoc = 178,
  specifier = 179,
  specifierLoc = 180,
  starLoc = 181,
  statement = 182,
  statementList = 183,
  staticAssertLoc = 184,
  staticLoc = 185,
  stringLiteral = 186,
  stringliteralLoc = 187,
  switchLoc = 188,
  symbolicName = 189,
  symbolicNameLoc = 190,
  templateArgumentList = 191,
  templateId = 192,
  templateLoc = 193,
  templateParameterList = 194,
  templateRequiresClause = 195,
  thisLoc = 196,
  threadLoc = 197,
  threadLocalLoc = 198,
  throwLoc = 199,
  tildeLoc = 200,
  trailingReturnType = 201,
  tryLoc = 202,
  typeConstraint = 203,
  typeId = 204,
  typeIdList = 205,
  typeSpecifier = 206,
  typeSpecifierList = 207,
  typeTraits = 208,
  typeTraitsLoc = 209,
  typedefLoc = 210,
  typeidLoc = 211,
  typenameLoc = 212,
  underlyingTypeLoc = 213,
  unqualifiedId = 214,
  usingDeclaratorList = 215,
  usingLoc = 216,
  virtualLoc = 217,
  voidLoc = 218,
  volatileLoc = 219,
  whileLoc = 220,
  yieldLoc = 221,
}
