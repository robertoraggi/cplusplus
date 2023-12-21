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
  captureDefault = 24,
  captureDefaultLoc = 25,
  captureList = 26,
  caseLoc = 27,
  castLoc = 28,
  catchLoc = 29,
  classKey = 30,
  classKeyLoc = 31,
  classLoc = 32,
  clobberList = 33,
  closeLoc = 34,
  colonLoc = 35,
  commaLoc = 36,
  complexLoc = 37,
  conceptLoc = 38,
  condition = 39,
  constLoc = 40,
  constevalLoc = 41,
  constexprLoc = 42,
  constinitLoc = 43,
  constraintLiteral = 44,
  constraintLiteralLoc = 45,
  constvalLoc = 46,
  continueLoc = 47,
  coreDeclarator = 48,
  coreturnLoc = 49,
  cvQualifierList = 50,
  declSpecifierList = 51,
  declaration = 52,
  declarationList = 53,
  declarator = 54,
  declaratorChunkList = 55,
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
  enumLoc = 66,
  enumTypeSpecifier = 67,
  enumeratorList = 68,
  equalLoc = 69,
  exceptionDeclaration = 70,
  exceptionSpecifier = 71,
  exclaimLoc = 72,
  explicitLoc = 73,
  explicitSpecifier = 74,
  exportLoc = 75,
  expression = 76,
  expressionList = 77,
  externLoc = 78,
  extraAttributeList = 79,
  finalLoc = 80,
  foldOp = 81,
  foldOpLoc = 82,
  forLoc = 83,
  friendLoc = 84,
  functionBody = 85,
  globalModuleFragment = 86,
  gotoLabelList = 87,
  gotoLoc = 88,
  greaterLoc = 89,
  handlerList = 90,
  headerLoc = 91,
  id = 92,
  idExpression = 93,
  identifier = 94,
  identifierLoc = 95,
  ifLoc = 96,
  iffalseExpression = 97,
  iftrueExpression = 98,
  importLoc = 99,
  importName = 100,
  indexExpression = 101,
  initDeclaratorList = 102,
  initializer = 103,
  inlineLoc = 104,
  inputOperandList = 105,
  isFinal = 106,
  isInline = 107,
  isNot = 108,
  isOverride = 109,
  isPack = 110,
  isPure = 111,
  isTemplateIntroduced = 112,
  isThisIntroduced = 113,
  isTrue = 114,
  isVariadic = 115,
  isVirtual = 116,
  lambdaSpecifierList = 117,
  lbraceLoc = 118,
  lbracket2Loc = 119,
  lbracketLoc = 120,
  leftExpression = 121,
  lessLoc = 122,
  literal = 123,
  literalLoc = 124,
  literalOperatorId = 125,
  lparen2Loc = 126,
  lparenLoc = 127,
  memInitializerList = 128,
  minusGreaterLoc = 129,
  moduleDeclaration = 130,
  moduleLoc = 131,
  moduleName = 132,
  modulePartition = 133,
  moduleQualifier = 134,
  mutableLoc = 135,
  namespaceLoc = 136,
  nestedNameSpecifier = 137,
  nestedNamespaceSpecifierList = 138,
  newInitalizer = 139,
  newLoc = 140,
  newPlacement = 141,
  noexceptLoc = 142,
  op = 143,
  opLoc = 144,
  openLoc = 145,
  operatorFunctionId = 146,
  operatorLoc = 147,
  outputOperandList = 148,
  parameterDeclarationClause = 149,
  parameterDeclarationList = 150,
  privateLoc = 151,
  privateModuleFragment = 152,
  ptrOpList = 153,
  qualifier = 154,
  qualifierLoc = 155,
  questionLoc = 156,
  rangeDeclaration = 157,
  rangeInitializer = 158,
  rbraceLoc = 159,
  rbracket2Loc = 160,
  rbracketLoc = 161,
  refLoc = 162,
  refOp = 163,
  refQualifierLoc = 164,
  requirementList = 165,
  requiresClause = 166,
  requiresLoc = 167,
  restrictLoc = 168,
  returnLoc = 169,
  rightExpression = 170,
  rparen2Loc = 171,
  rparenLoc = 172,
  scopeLoc = 173,
  semicolonLoc = 174,
  sizeExpression = 175,
  sizeofLoc = 176,
  specifier = 177,
  specifierLoc = 178,
  starLoc = 179,
  statement = 180,
  statementList = 181,
  staticAssertLoc = 182,
  staticLoc = 183,
  stringLiteral = 184,
  stringliteralLoc = 185,
  switchLoc = 186,
  symbolicName = 187,
  symbolicNameLoc = 188,
  templateArgumentList = 189,
  templateId = 190,
  templateLoc = 191,
  templateParameterList = 192,
  templateRequiresClause = 193,
  thisLoc = 194,
  threadLoc = 195,
  threadLocalLoc = 196,
  throwLoc = 197,
  tildeLoc = 198,
  trailingReturnType = 199,
  tryLoc = 200,
  typeConstraint = 201,
  typeId = 202,
  typeIdList = 203,
  typeSpecifier = 204,
  typeSpecifierList = 205,
  typeTraitsLoc = 206,
  typedefLoc = 207,
  typeidLoc = 208,
  typenameLoc = 209,
  underlyingTypeLoc = 210,
  unqualifiedId = 211,
  usingDeclaratorList = 212,
  usingLoc = 213,
  vaArgLoc = 214,
  virtualLoc = 215,
  voidLoc = 216,
  volatileLoc = 217,
  whileLoc = 218,
  yieldLoc = 219,
}
