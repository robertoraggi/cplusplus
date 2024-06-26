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
  caretLoc = 27,
  caseLoc = 28,
  castLoc = 29,
  catchLoc = 30,
  classKey = 31,
  classKeyLoc = 32,
  classLoc = 33,
  clobberList = 34,
  closeLoc = 35,
  colonLoc = 36,
  commaLoc = 37,
  complexLoc = 38,
  conceptLoc = 39,
  condition = 40,
  constLoc = 41,
  constevalLoc = 42,
  constexprLoc = 43,
  constinitLoc = 44,
  constraintLiteral = 45,
  constraintLiteralLoc = 46,
  constvalLoc = 47,
  continueLoc = 48,
  coreDeclarator = 49,
  coreturnLoc = 50,
  cvQualifierList = 51,
  declSpecifierList = 52,
  declaration = 53,
  declarationList = 54,
  declarator = 55,
  declaratorChunkList = 56,
  decltypeLoc = 57,
  decltypeSpecifier = 58,
  defaultLoc = 59,
  deleteLoc = 60,
  doLoc = 61,
  dotLoc = 62,
  ellipsisLoc = 63,
  elseLoc = 64,
  elseStatement = 65,
  emicolonLoc = 66,
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
  gnuAtributeList = 88,
  gnuAttributeList = 89,
  gotoLabelList = 90,
  gotoLoc = 91,
  greaterLoc = 92,
  handlerList = 93,
  headerLoc = 94,
  id = 95,
  idExpression = 96,
  identifier = 97,
  identifierLoc = 98,
  ifLoc = 99,
  iffalseExpression = 100,
  iftrueExpression = 101,
  importLoc = 102,
  importName = 103,
  indexExpression = 104,
  initDeclaratorList = 105,
  initializer = 106,
  inlineLoc = 107,
  inputOperandList = 108,
  isFinal = 109,
  isInline = 110,
  isNot = 111,
  isOverride = 112,
  isPack = 113,
  isPure = 114,
  isTemplateIntroduced = 115,
  isThisIntroduced = 116,
  isTrue = 117,
  isVariadic = 118,
  isVirtual = 119,
  lambdaSpecifierList = 120,
  lbraceLoc = 121,
  lbracket2Loc = 122,
  lbracketLoc = 123,
  leftExpression = 124,
  lessLoc = 125,
  literal = 126,
  literalLoc = 127,
  literalOperatorId = 128,
  lparen2Loc = 129,
  lparenLoc = 130,
  memInitializerList = 131,
  minusGreaterLoc = 132,
  moduleDeclaration = 133,
  moduleLoc = 134,
  moduleName = 135,
  modulePartition = 136,
  moduleQualifier = 137,
  mutableLoc = 138,
  namespaceLoc = 139,
  nestedNameSpecifier = 140,
  nestedNamespaceSpecifierList = 141,
  newInitalizer = 142,
  newLoc = 143,
  newPlacement = 144,
  noexceptLoc = 145,
  op = 146,
  opLoc = 147,
  openLoc = 148,
  operatorFunctionId = 149,
  operatorLoc = 150,
  outputOperandList = 151,
  parameterDeclarationClause = 152,
  parameterDeclarationList = 153,
  privateLoc = 154,
  privateModuleFragment = 155,
  ptrOpList = 156,
  qualifier = 157,
  qualifierLoc = 158,
  questionLoc = 159,
  rangeDeclaration = 160,
  rangeInitializer = 161,
  rbraceLoc = 162,
  rbracket2Loc = 163,
  rbracketLoc = 164,
  refLoc = 165,
  refOp = 166,
  refQualifierLoc = 167,
  requirementList = 168,
  requiresClause = 169,
  requiresLoc = 170,
  restrictLoc = 171,
  returnLoc = 172,
  rightExpression = 173,
  rparen2Loc = 174,
  rparenLoc = 175,
  scopeLoc = 176,
  secondColonLoc = 177,
  semicolonLoc = 178,
  sizeExpression = 179,
  sizeofLoc = 180,
  specifier = 181,
  specifierLoc = 182,
  splicer = 183,
  starLoc = 184,
  statement = 185,
  statementList = 186,
  staticAssertLoc = 187,
  staticLoc = 188,
  stringLiteral = 189,
  stringliteralLoc = 190,
  switchLoc = 191,
  symbolicName = 192,
  symbolicNameLoc = 193,
  templateArgumentList = 194,
  templateId = 195,
  templateLoc = 196,
  templateParameterList = 197,
  templateRequiresClause = 198,
  thisLoc = 199,
  threadLoc = 200,
  threadLocalLoc = 201,
  throwLoc = 202,
  tildeLoc = 203,
  trailingReturnType = 204,
  tryLoc = 205,
  typeConstraint = 206,
  typeId = 207,
  typeIdList = 208,
  typeLoc = 209,
  typeSpecifier = 210,
  typeSpecifierList = 211,
  typeTraitLoc = 212,
  typedefLoc = 213,
  typeidLoc = 214,
  typenameLoc = 215,
  underlyingTypeLoc = 216,
  unqualifiedId = 217,
  usingDeclaratorList = 218,
  usingLoc = 219,
  vaArgLoc = 220,
  virtualLoc = 221,
  voidLoc = 222,
  volatileLoc = 223,
  whileLoc = 224,
  yieldLoc = 225,
}
