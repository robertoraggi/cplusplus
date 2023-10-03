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

namespace cxx {

AST::~AST() = default;

auto TypeIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  return {};
}

auto TypeIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  return {};
}

auto UsingDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typenameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto UsingDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
  return {};
}

auto HandlerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(catchLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionDeclaration)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto HandlerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(exceptionDeclaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(catchLoc)) return loc;
  return {};
}

auto EnumBaseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  return {};
}

auto EnumBaseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return {};
}

auto EnumeratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto EnumeratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto DeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ptrOpList)) return loc;
  if (auto loc = cxx::firstSourceLocation(coreDeclarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaratorChunkList)) return loc;
  return {};
}

auto DeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaratorChunkList)) return loc;
  if (auto loc = cxx::lastSourceLocation(coreDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(ptrOpList)) return loc;
  return {};
}

auto InitDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto InitDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  return {};
}

auto BaseSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto BaseSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto BaseClauseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(baseSpecifierList)) return loc;
  return {};
}

auto BaseClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(baseSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return {};
}

auto NewDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ptrOpList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaratorChunkList)) return loc;
  return {};
}

auto NewDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaratorChunkList)) return loc;
  if (auto loc = cxx::lastSourceLocation(ptrOpList)) return loc;
  return {};
}

auto NewTypeIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(newDeclarator)) return loc;
  return {};
}

auto NewTypeIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(newDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  return {};
}

auto RequiresClauseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(requiresLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto RequiresClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresLoc)) return loc;
  return {};
}

auto ParameterDeclarationClauseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto ParameterDeclarationClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationList)) return loc;
  return {};
}

auto ParametersAndQualifiersAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto ParametersAndQualifiersAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto LambdaIntroducerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(captureDefaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(captureList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return {};
}

auto LambdaIntroducerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(captureList)) return loc;
  if (auto loc = cxx::lastSourceLocation(captureDefaultLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return {};
}

auto LambdaSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto LambdaSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto LambdaDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lambdaSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  return {};
}

auto LambdaDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto TrailingReturnTypeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(minusGreaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto TrailingReturnTypeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(minusGreaterLoc)) return loc;
  return {};
}

auto CtorInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(memInitializerList)) return loc;
  return {};
}

auto CtorInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(memInitializerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return {};
}

auto RequirementBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requirementList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto RequirementBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requirementList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return {};
}

auto TypeConstraintAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  return {};
}

auto TypeConstraintAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto GlobalModuleFragmentAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(moduleLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  return {};
}

auto GlobalModuleFragmentAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleLoc)) return loc;
  return {};
}

auto PrivateModuleFragmentAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(moduleLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(privateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  return {};
}

auto PrivateModuleFragmentAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(privateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleLoc)) return loc;
  return {};
}

auto ModuleQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(moduleQualifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(dotLoc)) return loc;
  return {};
}

auto ModuleQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(dotLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleQualifier)) return loc;
  return {};
}

auto ModuleNameAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(moduleQualifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto ModuleNameAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleQualifier)) return loc;
  return {};
}

auto ModuleDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(exportLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(moduleLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(moduleName)) return loc;
  if (auto loc = cxx::firstSourceLocation(modulePartition)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ModuleDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(modulePartition)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleName)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(exportLoc)) return loc;
  return {};
}

auto ImportNameAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(headerLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(modulePartition)) return loc;
  if (auto loc = cxx::firstSourceLocation(moduleName)) return loc;
  return {};
}

auto ImportNameAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(moduleName)) return loc;
  if (auto loc = cxx::lastSourceLocation(modulePartition)) return loc;
  if (auto loc = cxx::lastSourceLocation(headerLoc)) return loc;
  return {};
}

auto ModulePartitionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(moduleName)) return loc;
  return {};
}

auto ModulePartitionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(moduleName)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return {};
}

auto AttributeArgumentClauseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AttributeArgumentClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto AttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeToken)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeArgumentClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto AttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeArgumentClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeToken)) return loc;
  return {};
}

auto AttributeUsingPrefixAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeNamespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  return {};
}

auto AttributeUsingPrefixAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeNamespaceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return {};
}

auto DesignatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(dotLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto DesignatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(dotLoc)) return loc;
  return {};
}

auto NewPlacementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NewPlacementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto NestedNamespaceSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(inlineLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto NestedNamespaceSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(inlineLoc)) return loc;
  return {};
}

auto GlobalNestedNameSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto GlobalNestedNameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  return {};
}

auto SimpleNestedNameSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto SimpleNestedNameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto DecltypeNestedNameSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(decltypeSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto DecltypeNestedNameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto TemplateNestedNameSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateId)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto TemplateNestedNameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto ThrowExceptionSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(throwLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto ThrowExceptionSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(throwLoc)) return loc;
  return {};
}

auto NoexceptSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(noexceptLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NoexceptSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(noexceptLoc)) return loc;
  return {};
}

auto PackExpansionExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto PackExpansionExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return {};
}

auto DesignatedInitializerClauseAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(designator)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto DesignatedInitializerClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(designator)) return loc;
  return {};
}

auto ThisExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(thisLoc)) return loc;
  return {};
}

auto ThisExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(thisLoc)) return loc;
  return {};
}

auto CharLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto CharLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto BoolLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto BoolLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto IntLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto IntLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto FloatLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto FloatLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto NullptrLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto NullptrLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto StringLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto StringLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto UserDefinedStringLiteralExpressionAST::firstSourceLocation()
    -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto UserDefinedStringLiteralExpressionAST::lastSourceLocation()
    -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto IdExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto IdExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto RequiresExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(requiresLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requirementBody)) return loc;
  return {};
}

auto RequiresExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(requirementBody)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresLoc)) return loc;
  return {};
}

auto NestedExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NestedExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto RightFoldExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto RightFoldExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto LeftFoldExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto LeftFoldExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto FoldExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(foldOpLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto FoldExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(foldOpLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto LambdaExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lambdaIntroducer)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(lambdaDeclarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto LambdaExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaIntroducer)) return loc;
  return {};
}

auto SizeofExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(sizeofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto SizeofExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(sizeofLoc)) return loc;
  return {};
}

auto SizeofTypeExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(sizeofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto SizeofTypeExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(sizeofLoc)) return loc;
  return {};
}

auto SizeofPackExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(sizeofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto SizeofPackExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(sizeofLoc)) return loc;
  return {};
}

auto TypeidExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeidLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto TypeidExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeidLoc)) return loc;
  return {};
}

auto TypeidOfTypeExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeidLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto TypeidOfTypeExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeidLoc)) return loc;
  return {};
}

auto AlignofTypeExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(alignofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AlignofTypeExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(alignofLoc)) return loc;
  return {};
}

auto AlignofExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(alignofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto AlignofExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(alignofLoc)) return loc;
  return {};
}

auto TypeTraitsExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeTraitsLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeIdList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto TypeTraitsExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeIdList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeTraitsLoc)) return loc;
  return {};
}

auto YieldExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(yieldLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto YieldExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(yieldLoc)) return loc;
  return {};
}

auto AwaitExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(awaitLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto AwaitExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(awaitLoc)) return loc;
  return {};
}

auto UnaryExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto UnaryExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  return {};
}

auto BinaryExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  return {};
}

auto BinaryExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  return {};
}

auto AssignmentExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  return {};
}

auto AssignmentExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  return {};
}

auto ConditionExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto ConditionExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto BracedTypeConstructionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(bracedInitList)) return loc;
  return {};
}

auto BracedTypeConstructionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(bracedInitList)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifier)) return loc;
  return {};
}

auto TypeConstructionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto TypeConstructionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifier)) return loc;
  return {};
}

auto CallExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto CallExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return {};
}

auto SubscriptExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(indexExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return {};
}

auto SubscriptExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(indexExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return {};
}

auto MemberExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(memberId)) return loc;
  return {};
}

auto MemberExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(memberId)) return loc;
  if (auto loc = cxx::lastSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return {};
}

auto PostIncrExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  return {};
}

auto PostIncrExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return {};
}

auto ConditionalExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(questionLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(iftrueExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(iffalseExpression)) return loc;
  return {};
}

auto ConditionalExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(iffalseExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(iftrueExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(questionLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  return {};
}

auto ImplicitCastExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ImplicitCastExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return {};
}

auto CastExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto CastExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto CppCastExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(castLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto CppCastExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(castLoc)) return loc;
  return {};
}

auto NewExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newPlacement)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(newInitalizer)) return loc;
  return {};
}

auto NewExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(newInitalizer)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(newPlacement)) return loc;
  if (auto loc = cxx::lastSourceLocation(newLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  return {};
}

auto DeleteExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(deleteLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto DeleteExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(deleteLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  return {};
}

auto ThrowExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(throwLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ThrowExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(throwLoc)) return loc;
  return {};
}

auto NoexceptExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(noexceptLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NoexceptExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(noexceptLoc)) return loc;
  return {};
}

auto EqualInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto EqualInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  return {};
}

auto BracedInitListAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto BracedInitListAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return {};
}

auto ParenInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto ParenInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto SimpleRequirementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto SimpleRequirementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return {};
}

auto CompoundRequirementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(noexceptLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(minusGreaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeConstraint)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto CompoundRequirementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeConstraint)) return loc;
  if (auto loc = cxx::lastSourceLocation(minusGreaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(noexceptLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return {};
}

auto TypeRequirementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typenameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto TypeRequirementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
  return {};
}

auto NestedRequirementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(requiresLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto NestedRequirementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresLoc)) return loc;
  return {};
}

auto TypeTemplateArgumentAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto TypeTemplateArgumentAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  return {};
}

auto ExpressionTemplateArgumentAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ExpressionTemplateArgumentAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return {};
}

auto ParenMemInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto ParenMemInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto BracedMemInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(bracedInitList)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto BracedMemInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(bracedInitList)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto ThisLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(thisLoc)) return loc;
  return {};
}

auto ThisLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(thisLoc)) return loc;
  return {};
}

auto DerefThisLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(thisLoc)) return loc;
  return {};
}

auto DerefThisLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(thisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  return {};
}

auto SimpleLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto SimpleLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto RefLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ampLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto RefLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ampLoc)) return loc;
  return {};
}

auto RefInitLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ampLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto RefInitLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ampLoc)) return loc;
  return {};
}

auto InitLambdaCaptureAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto InitLambdaCaptureAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto NewParenInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NewParenInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto NewBracedInitializerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(bracedInitList)) return loc;
  return {};
}

auto NewBracedInitializerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(bracedInitList)) return loc;
  return {};
}

auto EllipsisExceptionDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto EllipsisExceptionDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto TypeExceptionDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  return {};
}

auto TypeExceptionDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto DefaultFunctionBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(defaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto DefaultFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(defaultLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  return {};
}

auto CompoundStatementFunctionBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ctorInitializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto CompoundStatementFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(ctorInitializer)) return loc;
  return {};
}

auto TryStatementFunctionBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(tryLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ctorInitializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(handlerList)) return loc;
  return {};
}

auto TryStatementFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(handlerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(ctorInitializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(tryLoc)) return loc;
  return {};
}

auto DeleteFunctionBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(deleteLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto DeleteFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(deleteLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  return {};
}

auto TranslationUnitAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  return {};
}

auto TranslationUnitAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  return {};
}

auto ModuleUnitAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(globalModuleFragment)) return loc;
  if (auto loc = cxx::firstSourceLocation(moduleDeclaration)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(privateModuleFragment)) return loc;
  return {};
}

auto ModuleUnitAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(privateModuleFragment)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(moduleDeclaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(globalModuleFragment)) return loc;
  return {};
}

auto LabeledStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  return {};
}

auto LabeledStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto CaseStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(caseLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  return {};
}

auto CaseStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(caseLoc)) return loc;
  return {};
}

auto DefaultStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(defaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  return {};
}

auto DefaultStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(defaultLoc)) return loc;
  return {};
}

auto ExpressionStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ExpressionStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  return {};
}

auto CompoundStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statementList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto CompoundStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statementList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  return {};
}

auto IfStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ifLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(constexprLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(elseLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(elseStatement)) return loc;
  return {};
}

auto IfStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(elseStatement)) return loc;
  if (auto loc = cxx::lastSourceLocation(elseLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(constexprLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ifLoc)) return loc;
  return {};
}

auto ConstevalIfStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ifLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(exclaimLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(constvalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(elseLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(elseStatement)) return loc;
  return {};
}

auto ConstevalIfStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(elseStatement)) return loc;
  if (auto loc = cxx::lastSourceLocation(elseLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(constvalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(exclaimLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ifLoc)) return loc;
  return {};
}

auto SwitchStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(switchLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto SwitchStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(switchLoc)) return loc;
  return {};
}

auto WhileStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto WhileStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(whileLoc)) return loc;
  return {};
}

auto DoStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(doLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto DoStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(whileLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(doLoc)) return loc;
  return {};
}

auto ForRangeStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(forLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(rangeDeclaration)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rangeInitializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto ForRangeStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rangeInitializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rangeDeclaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(forLoc)) return loc;
  return {};
}

auto ForStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(forLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(condition)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto ForStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(condition)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(forLoc)) return loc;
  return {};
}

auto BreakStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(breakLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto BreakStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(breakLoc)) return loc;
  return {};
}

auto ContinueStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(continueLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ContinueStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(continueLoc)) return loc;
  return {};
}

auto ReturnStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(returnLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ReturnStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(returnLoc)) return loc;
  return {};
}

auto GotoStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(gotoLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto GotoStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(gotoLoc)) return loc;
  return {};
}

auto CoroutineReturnStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(coreturnLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto CoroutineReturnStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(coreturnLoc)) return loc;
  return {};
}

auto DeclarationStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return {};
}

auto DeclarationStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  return {};
}

auto TryBlockStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(tryLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(handlerList)) return loc;
  return {};
}

auto TryBlockStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(handlerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(tryLoc)) return loc;
  return {};
}

auto AccessDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  return {};
}

auto AccessDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(accessLoc)) return loc;
  return {};
}

auto FunctionDefinitionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(functionBody)) return loc;
  return {};
}

auto FunctionDefinitionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(functionBody)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto ConceptDefinitionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(conceptLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ConceptDefinitionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(conceptLoc)) return loc;
  return {};
}

auto ForRangeDeclarationAST::firstSourceLocation() -> SourceLocation {
  return {};
}

auto ForRangeDeclarationAST::lastSourceLocation() -> SourceLocation {
  return {};
}

auto AliasDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto AliasDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return {};
}

auto SimpleDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(initDeclaratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto SimpleDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(initDeclaratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto StructuredBindingDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(refQualifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(bindingList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto StructuredBindingDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(bindingList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(refQualifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto StaticAssertDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(staticAssertLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto StaticAssertDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(staticAssertLoc)) return loc;
  return {};
}

auto EmptyDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto EmptyDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto AttributeDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto AttributeDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto OpaqueEnumDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(enumLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::firstSourceLocation(emicolonLoc)) return loc;
  return {};
}

auto OpaqueEnumDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(emicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumLoc)) return loc;
  return {};
}

auto NamespaceDefinitionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(inlineLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNamespaceSpecifierList))
    return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(extraAttributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto NamespaceDefinitionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(extraAttributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNamespaceSpecifierList))
    return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(inlineLoc)) return loc;
  return {};
}

auto NamespaceAliasDefinitionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto NamespaceAliasDefinitionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(namespaceLoc)) return loc;
  return {};
}

auto UsingDirectiveAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto UsingDirectiveAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(namespaceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto UsingDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(usingDeclaratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto UsingDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingDeclaratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return {};
}

auto UsingEnumDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumTypeSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto UsingEnumDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumTypeSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return {};
}

auto AsmOperandAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(symbolicNameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(constraintLiteralLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AsmOperandAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(constraintLiteralLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(symbolicNameLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return {};
}

auto AsmQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(qualifierLoc)) return loc;
  return {};
}

auto AsmQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(qualifierLoc)) return loc;
  return {};
}

auto AsmClobberAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  return {};
}

auto AsmClobberAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  return {};
}

auto AsmGotoLabelAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto AsmGotoLabelAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto AsmDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(asmQualifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(asmLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(outputOperandList)) return loc;
  if (auto loc = cxx::firstSourceLocation(inputOperandList)) return loc;
  if (auto loc = cxx::firstSourceLocation(clobberList)) return loc;
  if (auto loc = cxx::firstSourceLocation(gotoLabelList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto AsmDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(gotoLabelList)) return loc;
  if (auto loc = cxx::lastSourceLocation(clobberList)) return loc;
  if (auto loc = cxx::lastSourceLocation(inputOperandList)) return loc;
  if (auto loc = cxx::lastSourceLocation(outputOperandList)) return loc;
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(asmLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(asmQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto ExportDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(exportLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return {};
}

auto ExportDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(exportLoc)) return loc;
  return {};
}

auto ExportCompoundDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(exportLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto ExportCompoundDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(exportLoc)) return loc;
  return {};
}

auto ModuleImportDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(importLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(importName)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto ModuleImportDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(importName)) return loc;
  if (auto loc = cxx::lastSourceLocation(importLoc)) return loc;
  return {};
}

auto TemplateDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return {};
}

auto TemplateDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  return {};
}

auto TypenameTypeParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto TypenameTypeParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(classKeyLoc)) return loc;
  return {};
}

auto TemplateTypeParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(idExpression)) return loc;
  return {};
}

auto TemplateTypeParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(idExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  return {};
}

auto TemplatePackTypeParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto TemplatePackTypeParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  return {};
}

auto DeductionGuideAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(explicitSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(arrowLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto DeductionGuideAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateId)) return loc;
  if (auto loc = cxx::lastSourceLocation(arrowLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(explicitSpecifier)) return loc;
  return {};
}

auto ExplicitInstantiationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(externLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return {};
}

auto ExplicitInstantiationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(externLoc)) return loc;
  return {};
}

auto ParameterDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(thisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ParameterDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(thisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  return {};
}

auto LinkageSpecificationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(externLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(stringliteralLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto LinkageSpecificationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(stringliteralLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(externLoc)) return loc;
  return {};
}

auto NameIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto NameIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto DestructorIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(tildeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(id)) return loc;
  return {};
}

auto DestructorIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(id)) return loc;
  if (auto loc = cxx::lastSourceLocation(tildeLoc)) return loc;
  return {};
}

auto DecltypeIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(decltypeSpecifier)) return loc;
  return {};
}

auto DecltypeIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(decltypeSpecifier)) return loc;
  return {};
}

auto OperatorFunctionIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(operatorLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(openLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(closeLoc)) return loc;
  return {};
}

auto OperatorFunctionIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(closeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(openLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(operatorLoc)) return loc;
  return {};
}

auto LiteralOperatorIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(operatorLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto LiteralOperatorIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(operatorLoc)) return loc;
  return {};
}

auto ConversionFunctionIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(operatorLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto ConversionFunctionIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(operatorLoc)) return loc;
  return {};
}

auto SimpleTemplateIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  return {};
}

auto SimpleTemplateIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto LiteralOperatorTemplateIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(literalOperatorId)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  return {};
}

auto LiteralOperatorTemplateIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(literalOperatorId)) return loc;
  return {};
}

auto OperatorFunctionTemplateIdAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(operatorFunctionId)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  return {};
}

auto OperatorFunctionTemplateIdAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateArgumentList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(operatorFunctionId)) return loc;
  return {};
}

auto TypedefSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typedefLoc)) return loc;
  return {};
}

auto TypedefSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typedefLoc)) return loc;
  return {};
}

auto FriendSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(friendLoc)) return loc;
  return {};
}

auto FriendSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(friendLoc)) return loc;
  return {};
}

auto ConstevalSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(constevalLoc)) return loc;
  return {};
}

auto ConstevalSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(constevalLoc)) return loc;
  return {};
}

auto ConstinitSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(constinitLoc)) return loc;
  return {};
}

auto ConstinitSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(constinitLoc)) return loc;
  return {};
}

auto ConstexprSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(constexprLoc)) return loc;
  return {};
}

auto ConstexprSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(constexprLoc)) return loc;
  return {};
}

auto InlineSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(inlineLoc)) return loc;
  return {};
}

auto InlineSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(inlineLoc)) return loc;
  return {};
}

auto StaticSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(staticLoc)) return loc;
  return {};
}

auto StaticSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(staticLoc)) return loc;
  return {};
}

auto ExternSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(externLoc)) return loc;
  return {};
}

auto ExternSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(externLoc)) return loc;
  return {};
}

auto ThreadLocalSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(threadLocalLoc)) return loc;
  return {};
}

auto ThreadLocalSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(threadLocalLoc)) return loc;
  return {};
}

auto ThreadSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(threadLoc)) return loc;
  return {};
}

auto ThreadSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(threadLoc)) return loc;
  return {};
}

auto MutableSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(mutableLoc)) return loc;
  return {};
}

auto MutableSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(mutableLoc)) return loc;
  return {};
}

auto VirtualSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(virtualLoc)) return loc;
  return {};
}

auto VirtualSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(virtualLoc)) return loc;
  return {};
}

auto ExplicitSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(explicitLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto ExplicitSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(explicitLoc)) return loc;
  return {};
}

auto AutoTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(autoLoc)) return loc;
  return {};
}

auto AutoTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(autoLoc)) return loc;
  return {};
}

auto VoidTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(voidLoc)) return loc;
  return {};
}

auto VoidTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(voidLoc)) return loc;
  return {};
}

auto SizeTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto SizeTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto SignTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto SignTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto VaListTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto VaListTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto IntegralTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto IntegralTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto FloatingPointTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto FloatingPointTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto ComplexTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(complexLoc)) return loc;
  return {};
}

auto ComplexTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(complexLoc)) return loc;
  return {};
}

auto NamedTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto NamedTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto AtomicTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(atomicLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AtomicTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(atomicLoc)) return loc;
  return {};
}

auto UnderlyingTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(underlyingTypeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto UnderlyingTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(underlyingTypeLoc)) return loc;
  return {};
}

auto ElaboratedTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto ElaboratedTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  return {};
}

auto DecltypeAutoSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(decltypeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(autoLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto DecltypeAutoSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(autoLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeLoc)) return loc;
  return {};
}

auto DecltypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(decltypeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto DecltypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeLoc)) return loc;
  return {};
}

auto PlaceholderTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeConstraint)) return loc;
  if (auto loc = cxx::firstSourceLocation(specifier)) return loc;
  return {};
}

auto PlaceholderTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeConstraint)) return loc;
  return {};
}

auto ConstQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(constLoc)) return loc;
  return {};
}

auto ConstQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(constLoc)) return loc;
  return {};
}

auto VolatileQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(volatileLoc)) return loc;
  return {};
}

auto VolatileQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(volatileLoc)) return loc;
  return {};
}

auto RestrictQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(restrictLoc)) return loc;
  return {};
}

auto RestrictQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(restrictLoc)) return loc;
  return {};
}

auto EnumSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(enumLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto EnumSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumBase)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumLoc)) return loc;
  return {};
}

auto ClassSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(finalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(baseClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto ClassSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(finalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  return {};
}

auto TypenameSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typenameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto TypenameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
  return {};
}

auto BitfieldDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(sizeExpression)) return loc;
  return {};
}

auto BitfieldDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(sizeExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

auto ParameterPackAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(coreDeclarator)) return loc;
  return {};
}

auto ParameterPackAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(coreDeclarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto IdDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(declaratorId)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto IdDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(declaratorId)) return loc;
  return {};
}

auto NestedDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NestedDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  return {};
}

auto PointerOperatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  return {};
}

auto PointerOperatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  return {};
}

auto ReferenceOperatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto ReferenceOperatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(refLoc)) return loc;
  return {};
}

auto PtrToMemberOperatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  return {};
}

auto PtrToMemberOperatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  return {};
}

auto FunctionDeclaratorChunkAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(parametersAndQualifiers)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  return {};
}

auto FunctionDeclaratorChunkAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::lastSourceLocation(parametersAndQualifiers)) return loc;
  return {};
}

auto ArrayDeclaratorChunkAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto ArrayDeclaratorChunkAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return {};
}

auto CxxAttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbracket2Loc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeUsingPrefix)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracket2Loc)) return loc;
  return {};
}

auto CxxAttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbracket2Loc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeUsingPrefix)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracket2Loc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return {};
}

auto GccAttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparen2Loc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparen2Loc)) return loc;
  return {};
}

auto GccAttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparen2Loc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparen2Loc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeLoc)) return loc;
  return {};
}

auto AlignasAttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(alignasLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AlignasAttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(alignasLoc)) return loc;
  return {};
}

auto AsmAttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(asmLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AsmAttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(literalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(asmLoc)) return loc;
  return {};
}

auto ScopedAttributeTokenAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeNamespaceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto ScopedAttributeTokenAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeNamespaceLoc)) return loc;
  return {};
}

auto SimpleAttributeTokenAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto SimpleAttributeTokenAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  return {};
}

}  // namespace cxx
