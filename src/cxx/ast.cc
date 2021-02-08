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

namespace cxx {

AST::~AST() {}

SourceLocation TypeIdAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  return SourceLocation();
}

SourceLocation TypeIdAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation NestedNameSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nameList)) return loc;
  return SourceLocation();
}

SourceLocation NestedNameSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(nameList)) return loc;
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  return SourceLocation();
}

SourceLocation UsingDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(typenameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation UsingDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
  return SourceLocation();
}

SourceLocation HandlerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(catchLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionDeclaration)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation HandlerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(exceptionDeclaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(catchLoc)) return loc;
  return SourceLocation();
}

SourceLocation TemplateArgumentAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateArgumentAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation EnumBaseAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation EnumBaseAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return SourceLocation();
}

SourceLocation EnumeratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return SourceLocation();
}

SourceLocation EnumeratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation DeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(ptrOpList)) return loc;
  if (auto loc = cxx::firstSourceLocation(coreDeclarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(modifiers)) return loc;
  return SourceLocation();
}

SourceLocation DeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(modifiers)) return loc;
  if (auto loc = cxx::lastSourceLocation(coreDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(ptrOpList)) return loc;
  return SourceLocation();
}

SourceLocation BaseSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation BaseSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation BaseClauseAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(baseSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation BaseClauseAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(baseSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return SourceLocation();
}

SourceLocation NewTypeIdAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation NewTypeIdAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation ParameterDeclarationClauseAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return SourceLocation();
}

SourceLocation ParameterDeclarationClauseAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationList)) return loc;
  return SourceLocation();
}

SourceLocation ParametersAndQualifiersAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation ParametersAndQualifiersAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation LambdaIntroducerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return SourceLocation();
}

SourceLocation LambdaIntroducerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return SourceLocation();
}

SourceLocation LambdaDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  return SourceLocation();
}

SourceLocation LambdaDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation TrailingReturnTypeAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(minusGreaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return SourceLocation();
}

SourceLocation TrailingReturnTypeAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(minusGreaterLoc)) return loc;
  return SourceLocation();
}

SourceLocation EqualInitializerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return SourceLocation();
}

SourceLocation EqualInitializerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  return SourceLocation();
}

SourceLocation BracedInitListAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation BracedInitListAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation ParenInitializerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation ParenInitializerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation NewParenInitializerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation NewParenInitializerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation NewBracedInitializerAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(bracedInit)) return loc;
  return SourceLocation();
}

SourceLocation NewBracedInitializerAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(bracedInit)) return loc;
  return SourceLocation();
}

SourceLocation EllipsisExceptionDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return SourceLocation();
}

SourceLocation EllipsisExceptionDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  return SourceLocation();
}

SourceLocation TypeExceptionDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  return SourceLocation();
}

SourceLocation TypeExceptionDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation TranslationUnitAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  return SourceLocation();
}

SourceLocation TranslationUnitAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  return SourceLocation();
}

SourceLocation ModuleUnitAST::firstSourceLocation() { return SourceLocation(); }

SourceLocation ModuleUnitAST::lastSourceLocation() { return SourceLocation(); }

SourceLocation ThisExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(thisLoc)) return loc;
  return SourceLocation();
}

SourceLocation ThisExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(thisLoc)) return loc;
  return SourceLocation();
}

SourceLocation CharLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation CharLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation BoolLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation BoolLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation IntLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation IntLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation FloatLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation FloatLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation NullptrLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation NullptrLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation StringLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(stringLiteralList)) return loc;
  return SourceLocation();
}

SourceLocation StringLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(stringLiteralList)) return loc;
  return SourceLocation();
}

SourceLocation UserDefinedStringLiteralExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation UserDefinedStringLiteralExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return SourceLocation();
}

SourceLocation IdExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation IdExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation NestedExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation NestedExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation LambdaExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lambdaIntroducer)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lambdaDeclarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation LambdaExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaIntroducer)) return loc;
  return SourceLocation();
}

SourceLocation UnaryExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return SourceLocation();
}

SourceLocation UnaryExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  return SourceLocation();
}

SourceLocation BinaryExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  return SourceLocation();
}

SourceLocation BinaryExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  return SourceLocation();
}

SourceLocation AssignmentExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  return SourceLocation();
}

SourceLocation AssignmentExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  return SourceLocation();
}

SourceLocation CallExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation CallExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return SourceLocation();
}

SourceLocation SubscriptExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(indexExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return SourceLocation();
}

SourceLocation SubscriptExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(indexExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return SourceLocation();
}

SourceLocation MemberExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation MemberExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return SourceLocation();
}

SourceLocation ConditionalExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(questionLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(iftrueExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(iffalseExpression)) return loc;
  return SourceLocation();
}

SourceLocation ConditionalExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(iffalseExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(iftrueExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(questionLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  return SourceLocation();
}

SourceLocation CppCastExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(castLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation CppCastExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(castLoc)) return loc;
  return SourceLocation();
}

SourceLocation NewExpressionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(newInitalizer)) return loc;
  return SourceLocation();
}

SourceLocation NewExpressionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(newInitalizer)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(newLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  return SourceLocation();
}

SourceLocation LabeledStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation LabeledStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation CaseStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(caseLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation CaseStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(caseLoc)) return loc;
  return SourceLocation();
}

SourceLocation DefaultStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(defaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation DefaultStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(defaultLoc)) return loc;
  return SourceLocation();
}

SourceLocation ExpressionStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation ExpressionStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return SourceLocation();
}

SourceLocation CompoundStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statementList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation CompoundStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statementList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation IfStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(ifLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(constexprLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(elseStatement)) return loc;
  return SourceLocation();
}

SourceLocation IfStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(elseStatement)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(constexprLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ifLoc)) return loc;
  return SourceLocation();
}

SourceLocation SwitchStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(switchLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation SwitchStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(switchLoc)) return loc;
  return SourceLocation();
}

SourceLocation WhileStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation WhileStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(whileLoc)) return loc;
  return SourceLocation();
}

SourceLocation DoStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(doLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation DoStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(doLoc)) return loc;
  return SourceLocation();
}

SourceLocation ForRangeStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(forLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(rangeDeclaration)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rangeInitializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation ForRangeStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rangeInitializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rangeDeclaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(forLoc)) return loc;
  return SourceLocation();
}

SourceLocation ForStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(forLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return SourceLocation();
}

SourceLocation ForStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(forLoc)) return loc;
  return SourceLocation();
}

SourceLocation BreakStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(breakLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation BreakStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(breakLoc)) return loc;
  return SourceLocation();
}

SourceLocation ContinueStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(continueLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation ContinueStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(continueLoc)) return loc;
  return SourceLocation();
}

SourceLocation ReturnStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(returnLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation ReturnStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(returnLoc)) return loc;
  return SourceLocation();
}

SourceLocation GotoStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(gotoLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation GotoStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(gotoLoc)) return loc;
  return SourceLocation();
}

SourceLocation CoroutineReturnStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(coreturnLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation CoroutineReturnStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(coreturnLoc)) return loc;
  return SourceLocation();
}

SourceLocation DeclarationStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return SourceLocation();
}

SourceLocation DeclarationStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  return SourceLocation();
}

SourceLocation TryBlockStatementAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(tryLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(handlerList)) return loc;
  return SourceLocation();
}

SourceLocation TryBlockStatementAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(handlerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(tryLoc)) return loc;
  return SourceLocation();
}

SourceLocation FunctionDefinitionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(functionBody)) return loc;
  return SourceLocation();
}

SourceLocation FunctionDefinitionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(functionBody)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  return SourceLocation();
}

SourceLocation ConceptDefinitionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(conceptLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation ConceptDefinitionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(conceptLoc)) return loc;
  return SourceLocation();
}

SourceLocation ForRangeDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ForRangeDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AliasDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation AliasDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return SourceLocation();
}

SourceLocation SimpleDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributes)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation SimpleDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declaratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributes)) return loc;
  return SourceLocation();
}

SourceLocation StaticAssertDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(staticAssertLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(stringLiteralList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation StaticAssertDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(stringLiteralList)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(staticAssertLoc)) return loc;
  return SourceLocation();
}

SourceLocation EmptyDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation EmptyDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation AttributeDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation AttributeDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation OpaqueEnumDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(enumLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::firstSourceLocation(emicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation OpaqueEnumDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(emicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumLoc)) return loc;
  return SourceLocation();
}

SourceLocation UsingEnumDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingEnumDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NamespaceDefinitionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(inlineLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(extraAttributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation NamespaceDefinitionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(extraAttributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(inlineLoc)) return loc;
  return SourceLocation();
}

SourceLocation NamespaceAliasDefinitionAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation NamespaceAliasDefinitionAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(namespaceLoc)) return loc;
  return SourceLocation();
}

SourceLocation UsingDirectiveAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDirectiveAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation UsingDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(usingDeclaratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation UsingDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingDeclaratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return SourceLocation();
}

SourceLocation AsmDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(asmLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(stringLiteralList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return SourceLocation();
}

SourceLocation AsmDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(stringLiteralList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(asmLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation ExportDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExportDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleImportDeclarationAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ModuleImportDeclarationAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation TemplateDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return SourceLocation();
}

SourceLocation TemplateDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  return SourceLocation();
}

SourceLocation DeductionGuideAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation DeductionGuideAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitInstantiationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(externLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return SourceLocation();
}

SourceLocation ExplicitInstantiationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(externLoc)) return loc;
  return SourceLocation();
}

SourceLocation ParameterDeclarationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return SourceLocation();
}

SourceLocation ParameterDeclarationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation LinkageSpecificationAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(externLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(stringliteralLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation LinkageSpecificationAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(stringliteralLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(externLoc)) return loc;
  return SourceLocation();
}

SourceLocation SimpleNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation SimpleNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation DestructorNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(tildeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation DestructorNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(tildeLoc)) return loc;
  return SourceLocation();
}

SourceLocation DecltypeNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(decltypeSpecifier)) return loc;
  return SourceLocation();
}

SourceLocation DecltypeNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(decltypeSpecifier)) return loc;
  return SourceLocation();
}

SourceLocation OperatorNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  return SourceLocation();
}

SourceLocation OperatorNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  return SourceLocation();
}

SourceLocation TemplateNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  return SourceLocation();
}

SourceLocation TemplateNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation QualifiedNameAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation QualifiedNameAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return SourceLocation();
}

SourceLocation SimpleSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation SimpleSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation ExplicitSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ExplicitSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation NamedTypeSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation NamedTypeSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  return SourceLocation();
}

SourceLocation UnderlyingTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation UnderlyingTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation AtomicTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation AtomicTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation ElaboratedTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation ElaboratedTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation DecltypeAutoSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(decltypeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(autoLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation DecltypeAutoSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(autoLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeLoc)) return loc;
  return SourceLocation();
}

SourceLocation DecltypeSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(decltypeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation DecltypeSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeLoc)) return loc;
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation PlaceholderTypeSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation CvQualifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(qualifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation CvQualifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(qualifierLoc)) return loc;
  return SourceLocation();
}

SourceLocation EnumSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(enumLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation EnumSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumLoc)) return loc;
  return SourceLocation();
}

SourceLocation ClassSpecifierAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(baseClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return SourceLocation();
}

SourceLocation ClassSpecifierAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  return SourceLocation();
}

SourceLocation TypenameSpecifierAST::firstSourceLocation() {
  return SourceLocation();
}

SourceLocation TypenameSpecifierAST::lastSourceLocation() {
  return SourceLocation();
}

SourceLocation IdDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(name)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation IdDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(name)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  return SourceLocation();
}

SourceLocation NestedDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation NestedDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return SourceLocation();
}

SourceLocation PointerOperatorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  return SourceLocation();
}

SourceLocation PointerOperatorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  return SourceLocation();
}

SourceLocation ReferenceOperatorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation ReferenceOperatorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(refLoc)) return loc;
  return SourceLocation();
}

SourceLocation PtrToMemberOperatorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  return SourceLocation();
}

SourceLocation PtrToMemberOperatorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return SourceLocation();
}

SourceLocation FunctionDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(parametersAndQualifiers)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  return SourceLocation();
}

SourceLocation FunctionDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::lastSourceLocation(parametersAndQualifiers)) return loc;
  return SourceLocation();
}

SourceLocation ArrayDeclaratorAST::firstSourceLocation() {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return SourceLocation();
}

SourceLocation ArrayDeclaratorAST::lastSourceLocation() {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return SourceLocation();
}

}  // namespace cxx
