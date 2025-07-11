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
  designatorList = 61,
  doLoc = 62,
  dotLoc = 63,
  ellipsisLoc = 64,
  elseLoc = 65,
  elseStatement = 66,
  emicolonLoc = 67,
  enumLoc = 68,
  enumTypeSpecifier = 69,
  enumeratorList = 70,
  equalLoc = 71,
  exceptionDeclaration = 72,
  exceptionSpecifier = 73,
  exclaimLoc = 74,
  explicitLoc = 75,
  explicitSpecifier = 76,
  exportLoc = 77,
  expression = 78,
  expressionList = 79,
  externLoc = 80,
  extraAttributeList = 81,
  finalLoc = 82,
  foldOp = 83,
  foldOpLoc = 84,
  forLoc = 85,
  friendLoc = 86,
  functionBody = 87,
  globalModuleFragment = 88,
  gnuAtributeList = 89,
  gnuAttributeList = 90,
  gotoLabelList = 91,
  gotoLoc = 92,
  greaterLoc = 93,
  handlerList = 94,
  headerLoc = 95,
  id = 96,
  idExpression = 97,
  identifier = 98,
  identifierLoc = 99,
  ifLoc = 100,
  iffalseExpression = 101,
  iftrueExpression = 102,
  importLoc = 103,
  importName = 104,
  indexExpression = 105,
  initDeclaratorList = 106,
  initializer = 107,
  inlineLoc = 108,
  inputOperandList = 109,
  isFinal = 110,
  isInline = 111,
  isNot = 112,
  isOverride = 113,
  isPack = 114,
  isPure = 115,
  isTemplateIntroduced = 116,
  isThisIntroduced = 117,
  isTrue = 118,
  isVariadic = 119,
  isVirtual = 120,
  lambdaSpecifierList = 121,
  lbraceLoc = 122,
  lbracket2Loc = 123,
  lbracketLoc = 124,
  leftExpression = 125,
  lessLoc = 126,
  literal = 127,
  literalLoc = 128,
  literalOperatorId = 129,
  lparen2Loc = 130,
  lparenLoc = 131,
  memInitializerList = 132,
  minusGreaterLoc = 133,
  moduleDeclaration = 134,
  moduleLoc = 135,
  moduleName = 136,
  modulePartition = 137,
  moduleQualifier = 138,
  mutableLoc = 139,
  namespaceLoc = 140,
  nestedNameSpecifier = 141,
  nestedNamespaceSpecifierList = 142,
  newInitalizer = 143,
  newLoc = 144,
  newPlacement = 145,
  noexceptLoc = 146,
  noreturnLoc = 147,
  offsetofLoc = 148,
  op = 149,
  opLoc = 150,
  openLoc = 151,
  operatorFunctionId = 152,
  operatorLoc = 153,
  otherVirtualOrAccessLoc = 154,
  outputOperandList = 155,
  parameterDeclarationClause = 156,
  parameterDeclarationList = 157,
  privateLoc = 158,
  privateModuleFragment = 159,
  ptrOpList = 160,
  qualifier = 161,
  qualifierLoc = 162,
  questionLoc = 163,
  rangeDeclaration = 164,
  rangeInitializer = 165,
  rbraceLoc = 166,
  rbracket2Loc = 167,
  rbracketLoc = 168,
  refLoc = 169,
  refOp = 170,
  refQualifierLoc = 171,
  registerLoc = 172,
  requirementList = 173,
  requiresClause = 174,
  requiresLoc = 175,
  restrictLoc = 176,
  returnLoc = 177,
  rightExpression = 178,
  rparen2Loc = 179,
  rparenLoc = 180,
  scopeLoc = 181,
  secondColonLoc = 182,
  semicolonLoc = 183,
  sizeExpression = 184,
  sizeofLoc = 185,
  specifier = 186,
  specifierLoc = 187,
  splicer = 188,
  starLoc = 189,
  statement = 190,
  statementList = 191,
  staticAssertLoc = 192,
  staticLoc = 193,
  stringLiteral = 194,
  stringliteralLoc = 195,
  switchLoc = 196,
  symbolicName = 197,
  symbolicNameLoc = 198,
  templateArgumentList = 199,
  templateId = 200,
  templateLoc = 201,
  templateParameterList = 202,
  templateRequiresClause = 203,
  thisLoc = 204,
  threadLoc = 205,
  threadLocalLoc = 206,
  throwLoc = 207,
  tildeLoc = 208,
  trailingReturnType = 209,
  tryLoc = 210,
  typeConstraint = 211,
  typeId = 212,
  typeIdList = 213,
  typeLoc = 214,
  typeSpecifier = 215,
  typeSpecifierList = 216,
  typeTraitLoc = 217,
  typedefLoc = 218,
  typeidLoc = 219,
  typenameLoc = 220,
  underlyingTypeLoc = 221,
  unqualifiedId = 222,
  usingDeclaratorList = 223,
  usingLoc = 224,
  vaArgLoc = 225,
  virtualLoc = 226,
  virtualOrAccessLoc = 227,
  voidLoc = 228,
  volatileLoc = 229,
  whileLoc = 230,
  yieldLoc = 231,
}
