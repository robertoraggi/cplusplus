// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
  offsetofLoc = 146,
  op = 147,
  opLoc = 148,
  openLoc = 149,
  operatorFunctionId = 150,
  operatorLoc = 151,
  outputOperandList = 152,
  parameterDeclarationClause = 153,
  parameterDeclarationList = 154,
  privateLoc = 155,
  privateModuleFragment = 156,
  ptrOpList = 157,
  qualifier = 158,
  qualifierLoc = 159,
  questionLoc = 160,
  rangeDeclaration = 161,
  rangeInitializer = 162,
  rbraceLoc = 163,
  rbracket2Loc = 164,
  rbracketLoc = 165,
  refLoc = 166,
  refOp = 167,
  refQualifierLoc = 168,
  requirementList = 169,
  requiresClause = 170,
  requiresLoc = 171,
  restrictLoc = 172,
  returnLoc = 173,
  rightExpression = 174,
  rparen2Loc = 175,
  rparenLoc = 176,
  scopeLoc = 177,
  secondColonLoc = 178,
  semicolonLoc = 179,
  sizeExpression = 180,
  sizeofLoc = 181,
  specifier = 182,
  specifierLoc = 183,
  splicer = 184,
  starLoc = 185,
  statement = 186,
  statementList = 187,
  staticAssertLoc = 188,
  staticLoc = 189,
  stringLiteral = 190,
  stringliteralLoc = 191,
  switchLoc = 192,
  symbolicName = 193,
  symbolicNameLoc = 194,
  templateArgumentList = 195,
  templateId = 196,
  templateLoc = 197,
  templateParameterList = 198,
  templateRequiresClause = 199,
  thisLoc = 200,
  threadLoc = 201,
  threadLocalLoc = 202,
  throwLoc = 203,
  tildeLoc = 204,
  trailingReturnType = 205,
  tryLoc = 206,
  typeConstraint = 207,
  typeId = 208,
  typeIdList = 209,
  typeLoc = 210,
  typeSpecifier = 211,
  typeSpecifierList = 212,
  typeTraitLoc = 213,
  typedefLoc = 214,
  typeidLoc = 215,
  typenameLoc = 216,
  underlyingTypeLoc = 217,
  unqualifiedId = 218,
  usingDeclaratorList = 219,
  usingLoc = 220,
  vaArgLoc = 221,
  virtualLoc = 222,
  voidLoc = 223,
  volatileLoc = 224,
  whileLoc = 225,
  yieldLoc = 226,
}
