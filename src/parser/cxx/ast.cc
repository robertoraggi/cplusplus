// Generated file by: gen_ast_cc.ts
// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

auto AliasDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(usingLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(gnuAttributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto AliasDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(gnuAttributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(usingLoc)) return loc;
  return {};
}

auto OpaqueEnumDeclarationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(enumLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(emicolonLoc)) return loc;
  return {};
}

auto OpaqueEnumDeclarationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(emicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(classLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumLoc)) return loc;
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

auto ForRangeDeclarationAST::firstSourceLocation() -> SourceLocation {
  return {};
}

auto ForRangeDeclarationAST::lastSourceLocation() -> SourceLocation {
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

auto SplicerAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(secondColonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return {};
}

auto SplicerAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(secondColonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
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

auto BaseSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(virtualOrAccessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(otherVirtualOrAccessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  return {};
}

auto BaseSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(otherVirtualOrAccessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(virtualOrAccessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
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

auto LambdaSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto LambdaSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
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

auto GotoStatementAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(gotoLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto GotoStatementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(starLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(gotoLoc)) return loc;
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

auto ObjectLiteralExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(bracedInitList)) return loc;
  return {};
}

auto ObjectLiteralExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(bracedInitList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
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

auto GenericSelectionExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(genericLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(genericAssociationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto GenericSelectionExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(genericAssociationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(genericLoc)) return loc;
  return {};
}

auto NestedStatementExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto NestedStatementExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
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

auto LambdaExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(captureDefaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(captureList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateRequiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(expressionAttributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(gnuAtributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lambdaSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto LambdaExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(lambdaSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(gnuAtributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expressionAttributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateRequiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(captureList)) return loc;
  if (auto loc = cxx::lastSourceLocation(captureDefaultLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
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

auto RequiresExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(requiresLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requirementList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto RequiresExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requirementList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresLoc)) return loc;
  return {};
}

auto VaArgExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(vaArgLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto VaArgExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(vaArgLoc)) return loc;
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

auto SpliceMemberExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(splicer)) return loc;
  return {};
}

auto SpliceMemberExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(splicer)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseExpression)) return loc;
  return {};
}

auto MemberExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(baseExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(accessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto MemberExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
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

auto BuiltinBitCastExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(castLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto BuiltinBitCastExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(castLoc)) return loc;
  return {};
}

auto BuiltinOffsetofExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(offsetofLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(designatorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto BuiltinOffsetofExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(designatorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(offsetofLoc)) return loc;
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

auto SpliceExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(splicer)) return loc;
  return {};
}

auto SpliceExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(splicer)) return loc;
  return {};
}

auto GlobalScopeReflectExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(caretCaretLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto GlobalScopeReflectExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(caretCaretLoc)) return loc;
  return {};
}

auto NamespaceReflectExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(caretCaretLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto NamespaceReflectExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(caretCaretLoc)) return loc;
  return {};
}

auto TypeIdReflectExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(caretCaretLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto TypeIdReflectExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(caretCaretLoc)) return loc;
  return {};
}

auto ReflectExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(caretCaretLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ReflectExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(caretCaretLoc)) return loc;
  return {};
}

auto LabelAddressExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(ampAmpLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto LabelAddressExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ampAmpLoc)) return loc;
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

auto NewExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newPlacement)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarator)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(newInitalizer)) return loc;
  return {};
}

auto NewExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(newInitalizer)) return loc;
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarator)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
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

auto ImplicitCastExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto ImplicitCastExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
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

auto TargetExpressionAST::firstSourceLocation() -> SourceLocation { return {}; }

auto TargetExpressionAST::lastSourceLocation() -> SourceLocation { return {}; }

auto RightExpressionAST::firstSourceLocation() -> SourceLocation { return {}; }

auto RightExpressionAST::lastSourceLocation() -> SourceLocation { return {}; }

auto CompoundAssignmentExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(targetExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::firstSourceLocation(adjustExpression)) return loc;
  return {};
}

auto CompoundAssignmentExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(adjustExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(rightExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(opLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(targetExpression)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(designatorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(initializer)) return loc;
  return {};
}

auto DesignatedInitializerClauseAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(initializer)) return loc;
  if (auto loc = cxx::lastSourceLocation(designatorList)) return loc;
  return {};
}

auto TypeTraitExpressionAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeTraitLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeIdList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto TypeTraitExpressionAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeIdList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeTraitLoc)) return loc;
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

auto DefaultGenericAssociationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(defaultLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto DefaultGenericAssociationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(defaultLoc)) return loc;
  return {};
}

auto TypeGenericAssociationAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  return {};
}

auto TypeGenericAssociationAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  return {};
}

auto DotDesignatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(dotLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  return {};
}

auto DotDesignatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(dotLoc)) return loc;
  return {};
}

auto SubscriptDesignatorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  return {};
}

auto SubscriptDesignatorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
  return {};
}

auto TemplateTypeParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::firstSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::firstSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(idExpression)) return loc;
  return {};
}

auto TemplateTypeParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(idExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(classKeyLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(requiresClause)) return loc;
  if (auto loc = cxx::lastSourceLocation(greaterLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateParameterList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lessLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  return {};
}

auto NonTypeTemplateParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(declaration)) return loc;
  return {};
}

auto NonTypeTemplateParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(declaration)) return loc;
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

auto ConstraintTypeParameterAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typeConstraint)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  return {};
}

auto ConstraintTypeParameterAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(equalLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(identifierLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeConstraint)) return loc;
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

auto NoreturnSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(noreturnLoc)) return loc;
  return {};
}

auto NoreturnSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(noreturnLoc)) return loc;
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

auto RegisterSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(registerLoc)) return loc;
  return {};
}

auto RegisterSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(registerLoc)) return loc;
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

auto BuiltinTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(specifierLoc)) return loc;
  return {};
}

auto BuiltinTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(specifierLoc)) return loc;
  return {};
}

auto UnaryBuiltinTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(builtinLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto UnaryBuiltinTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(builtinLoc)) return loc;
  return {};
}

auto BinaryBuiltinTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(builtinLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(leftTypeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rightTypeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto BinaryBuiltinTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(rightTypeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(leftTypeId)) return loc;
  if (auto loc = cxx::lastSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(builtinLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto ElaboratedTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
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

auto AtomicQualifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(atomicLoc)) return loc;
  return {};
}

auto AtomicQualifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(atomicLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::firstSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto EnumSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(commaLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(enumeratorList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(baseSpecifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbraceLoc)) return loc;
  return {};
}

auto ClassSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(declarationList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbraceLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(baseSpecifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  return {};
}

auto TypenameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
  return {};
}

auto SplicerTypeSpecifierAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(typenameLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(splicer)) return loc;
  return {};
}

auto SplicerTypeSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(splicer)) return loc;
  if (auto loc = cxx::lastSourceLocation(typenameLoc)) return loc;
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

auto BitfieldDeclaratorAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(sizeExpression)) return loc;
  return {};
}

auto BitfieldDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(sizeExpression)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(nestedNameSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto IdDeclaratorAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(nestedNameSpecifier)) return loc;
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

auto FunctionDeclaratorChunkAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(parameterDeclarationClause))
    return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(cvQualifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(refLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(exceptionSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::firstSourceLocation(trailingReturnType)) return loc;
  return {};
}

auto FunctionDeclaratorChunkAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(trailingReturnType)) return loc;
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

auto ArrayDeclaratorChunkAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(lbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeQualifierList)) return loc;
  if (auto loc = cxx::firstSourceLocation(expression)) return loc;
  if (auto loc = cxx::firstSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(attributeList)) return loc;
  return {};
}

auto ArrayDeclaratorChunkAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(attributeList)) return loc;
  if (auto loc = cxx::lastSourceLocation(rbracketLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(expression)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeQualifierList)) return loc;
  if (auto loc = cxx::lastSourceLocation(lbracketLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(decltypeSpecifier)) return loc;
  if (auto loc = cxx::firstSourceLocation(scopeLoc)) return loc;
  return {};
}

auto DecltypeNestedNameSpecifierAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(scopeLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(decltypeSpecifier)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(memInitializerList)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  return {};
}

auto CompoundStatementFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(memInitializerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
  return {};
}

auto TryStatementFunctionBodyAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(tryLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(colonLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(memInitializerList)) return loc;
  if (auto loc = cxx::firstSourceLocation(statement)) return loc;
  if (auto loc = cxx::firstSourceLocation(handlerList)) return loc;
  return {};
}

auto TryStatementFunctionBodyAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(handlerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(statement)) return loc;
  if (auto loc = cxx::lastSourceLocation(memInitializerList)) return loc;
  if (auto loc = cxx::lastSourceLocation(colonLoc)) return loc;
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
  if (auto loc = cxx::firstSourceLocation(templateLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::firstSourceLocation(semicolonLoc)) return loc;
  return {};
}

auto TypeRequirementAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(semicolonLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(unqualifiedId)) return loc;
  if (auto loc = cxx::lastSourceLocation(templateLoc)) return loc;
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

auto AlignasTypeAttributeAST::firstSourceLocation() -> SourceLocation {
  if (auto loc = cxx::firstSourceLocation(alignasLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(lparenLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(typeId)) return loc;
  if (auto loc = cxx::firstSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::firstSourceLocation(rparenLoc)) return loc;
  return {};
}

auto AlignasTypeAttributeAST::lastSourceLocation() -> SourceLocation {
  if (auto loc = cxx::lastSourceLocation(rparenLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(ellipsisLoc)) return loc;
  if (auto loc = cxx::lastSourceLocation(typeId)) return loc;
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

namespace {
std::string_view kASTKindNames[] = {

    // UnitAST
    "translation-unit",
    "module-unit",

    // DeclarationAST
    "simple-declaration",
    "asm-declaration",
    "namespace-alias-definition",
    "using-declaration",
    "using-enum-declaration",
    "using-directive",
    "static-assert-declaration",
    "alias-declaration",
    "opaque-enum-declaration",
    "function-definition",
    "template-declaration",
    "concept-definition",
    "deduction-guide",
    "explicit-instantiation",
    "export-declaration",
    "export-compound-declaration",
    "linkage-specification",
    "namespace-definition",
    "empty-declaration",
    "attribute-declaration",
    "module-import-declaration",
    "parameter-declaration",
    "access-declaration",
    "for-range-declaration",
    "structured-binding-declaration",

    // AST
    "asm-operand",
    "asm-qualifier",
    "asm-clobber",
    "asm-goto-label",
    "splicer",
    "global-module-fragment",
    "private-module-fragment",
    "module-declaration",
    "module-name",
    "module-qualifier",
    "module-partition",
    "import-name",
    "init-declarator",
    "declarator",
    "using-declarator",
    "enumerator",
    "type-id",
    "handler",
    "base-specifier",
    "requires-clause",
    "parameter-declaration-clause",
    "trailing-return-type",
    "lambda-specifier",
    "type-constraint",
    "attribute-argument-clause",
    "attribute",
    "attribute-using-prefix",
    "new-placement",
    "nested-namespace-specifier",

    // StatementAST
    "labeled-statement",
    "case-statement",
    "default-statement",
    "expression-statement",
    "compound-statement",
    "if-statement",
    "consteval-if-statement",
    "switch-statement",
    "while-statement",
    "do-statement",
    "for-range-statement",
    "for-statement",
    "break-statement",
    "continue-statement",
    "return-statement",
    "coroutine-return-statement",
    "goto-statement",
    "declaration-statement",
    "try-block-statement",

    // ExpressionAST
    "char-literal-expression",
    "bool-literal-expression",
    "int-literal-expression",
    "float-literal-expression",
    "nullptr-literal-expression",
    "string-literal-expression",
    "user-defined-string-literal-expression",
    "object-literal-expression",
    "this-expression",
    "generic-selection-expression",
    "nested-statement-expression",
    "nested-expression",
    "id-expression",
    "lambda-expression",
    "fold-expression",
    "right-fold-expression",
    "left-fold-expression",
    "requires-expression",
    "va-arg-expression",
    "subscript-expression",
    "call-expression",
    "type-construction",
    "braced-type-construction",
    "splice-member-expression",
    "member-expression",
    "post-incr-expression",
    "cpp-cast-expression",
    "builtin-bit-cast-expression",
    "builtin-offsetof-expression",
    "typeid-expression",
    "typeid-of-type-expression",
    "splice-expression",
    "global-scope-reflect-expression",
    "namespace-reflect-expression",
    "type-id-reflect-expression",
    "reflect-expression",
    "label-address-expression",
    "unary-expression",
    "await-expression",
    "sizeof-expression",
    "sizeof-type-expression",
    "sizeof-pack-expression",
    "alignof-type-expression",
    "alignof-expression",
    "noexcept-expression",
    "new-expression",
    "delete-expression",
    "cast-expression",
    "implicit-cast-expression",
    "binary-expression",
    "conditional-expression",
    "yield-expression",
    "throw-expression",
    "assignment-expression",
    "target-expression",
    "right-expression",
    "compound-assignment-expression",
    "pack-expansion-expression",
    "designated-initializer-clause",
    "type-trait-expression",
    "condition-expression",
    "equal-initializer",
    "braced-init-list",
    "paren-initializer",

    // GenericAssociationAST
    "default-generic-association",
    "type-generic-association",

    // DesignatorAST
    "dot-designator",
    "subscript-designator",

    // TemplateParameterAST
    "template-type-parameter",
    "non-type-template-parameter",
    "typename-type-parameter",
    "constraint-type-parameter",

    // SpecifierAST
    "typedef-specifier",
    "friend-specifier",
    "consteval-specifier",
    "constinit-specifier",
    "constexpr-specifier",
    "inline-specifier",
    "noreturn-specifier",
    "static-specifier",
    "extern-specifier",
    "register-specifier",
    "thread-local-specifier",
    "thread-specifier",
    "mutable-specifier",
    "virtual-specifier",
    "explicit-specifier",
    "auto-type-specifier",
    "void-type-specifier",
    "size-type-specifier",
    "sign-type-specifier",
    "builtin-type-specifier",
    "unary-builtin-type-specifier",
    "binary-builtin-type-specifier",
    "integral-type-specifier",
    "floating-point-type-specifier",
    "complex-type-specifier",
    "named-type-specifier",
    "atomic-type-specifier",
    "underlying-type-specifier",
    "elaborated-type-specifier",
    "decltype-auto-specifier",
    "decltype-specifier",
    "placeholder-type-specifier",
    "const-qualifier",
    "volatile-qualifier",
    "atomic-qualifier",
    "restrict-qualifier",
    "enum-specifier",
    "class-specifier",
    "typename-specifier",
    "splicer-type-specifier",

    // PtrOperatorAST
    "pointer-operator",
    "reference-operator",
    "ptr-to-member-operator",

    // CoreDeclaratorAST
    "bitfield-declarator",
    "parameter-pack",
    "id-declarator",
    "nested-declarator",

    // DeclaratorChunkAST
    "function-declarator-chunk",
    "array-declarator-chunk",

    // UnqualifiedIdAST
    "name-id",
    "destructor-id",
    "decltype-id",
    "operator-function-id",
    "literal-operator-id",
    "conversion-function-id",
    "simple-template-id",
    "literal-operator-template-id",
    "operator-function-template-id",

    // NestedNameSpecifierAST
    "global-nested-name-specifier",
    "simple-nested-name-specifier",
    "decltype-nested-name-specifier",
    "template-nested-name-specifier",

    // FunctionBodyAST
    "default-function-body",
    "compound-statement-function-body",
    "try-statement-function-body",
    "delete-function-body",

    // TemplateArgumentAST
    "type-template-argument",
    "expression-template-argument",

    // ExceptionSpecifierAST
    "throw-exception-specifier",
    "noexcept-specifier",

    // RequirementAST
    "simple-requirement",
    "compound-requirement",
    "type-requirement",
    "nested-requirement",

    // NewInitializerAST
    "new-paren-initializer",
    "new-braced-initializer",

    // MemInitializerAST
    "paren-mem-initializer",
    "braced-mem-initializer",

    // LambdaCaptureAST
    "this-lambda-capture",
    "deref-this-lambda-capture",
    "simple-lambda-capture",
    "ref-lambda-capture",
    "ref-init-lambda-capture",
    "init-lambda-capture",

    // ExceptionDeclarationAST
    "ellipsis-exception-declaration",
    "type-exception-declaration",

    // AttributeSpecifierAST
    "cxx-attribute",
    "gcc-attribute",
    "alignas-attribute",
    "alignas-type-attribute",
    "asm-attribute",

    // AttributeTokenAST
    "scoped-attribute-token",
    "simple-attribute-token",
};
}  // namespace
auto to_string(ASTKind kind) -> std::string_view {
  return kASTKindNames[int(kind)];
}

auto TranslationUnitAST::clone(Arena* arena) -> TranslationUnitAST* {
  auto node = create(arena);

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto TranslationUnitAST::create(Arena* arena) -> TranslationUnitAST* {
  auto node = new (arena) TranslationUnitAST();
  return node;
}

auto TranslationUnitAST::create(Arena* arena,
                                List<DeclarationAST*>* declarationList)
    -> TranslationUnitAST* {
  auto node = new (arena) TranslationUnitAST();
  node->declarationList = declarationList;
  return node;
}

auto ModuleUnitAST::clone(Arena* arena) -> ModuleUnitAST* {
  auto node = create(arena);

  if (globalModuleFragment)
    node->globalModuleFragment = globalModuleFragment->clone(arena);

  if (moduleDeclaration)
    node->moduleDeclaration = moduleDeclaration->clone(arena);

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (privateModuleFragment)
    node->privateModuleFragment = privateModuleFragment->clone(arena);

  return node;
}

auto ModuleUnitAST::create(Arena* arena) -> ModuleUnitAST* {
  auto node = new (arena) ModuleUnitAST();
  return node;
}

auto ModuleUnitAST::create(Arena* arena,
                           GlobalModuleFragmentAST* globalModuleFragment,
                           ModuleDeclarationAST* moduleDeclaration,
                           List<DeclarationAST*>* declarationList,
                           PrivateModuleFragmentAST* privateModuleFragment)
    -> ModuleUnitAST* {
  auto node = new (arena) ModuleUnitAST();
  node->globalModuleFragment = globalModuleFragment;
  node->moduleDeclaration = moduleDeclaration;
  node->declarationList = declarationList;
  node->privateModuleFragment = privateModuleFragment;
  return node;
}

auto SimpleDeclarationAST::clone(Arena* arena) -> SimpleDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declSpecifierList) {
    auto it = &node->declSpecifierList;
    for (auto node : ListView{declSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (initDeclaratorList) {
    auto it = &node->initDeclaratorList;
    for (auto node : ListView{initDeclaratorList}) {
      *it = make_list_node<InitDeclaratorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto SimpleDeclarationAST::create(Arena* arena) -> SimpleDeclarationAST* {
  auto node = new (arena) SimpleDeclarationAST();
  return node;
}

auto SimpleDeclarationAST::create(Arena* arena,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  List<SpecifierAST*>* declSpecifierList,
                                  List<InitDeclaratorAST*>* initDeclaratorList,
                                  RequiresClauseAST* requiresClause,
                                  SourceLocation semicolonLoc)
    -> SimpleDeclarationAST* {
  auto node = new (arena) SimpleDeclarationAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->initDeclaratorList = initDeclaratorList;
  node->requiresClause = requiresClause;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto SimpleDeclarationAST::create(Arena* arena,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  List<SpecifierAST*>* declSpecifierList,
                                  List<InitDeclaratorAST*>* initDeclaratorList,
                                  RequiresClauseAST* requiresClause)
    -> SimpleDeclarationAST* {
  auto node = new (arena) SimpleDeclarationAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->initDeclaratorList = initDeclaratorList;
  node->requiresClause = requiresClause;
  return node;
}

auto AsmDeclarationAST::clone(Arena* arena) -> AsmDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (asmQualifierList) {
    auto it = &node->asmQualifierList;
    for (auto node : ListView{asmQualifierList}) {
      *it = make_list_node<AsmQualifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->asmLoc = asmLoc;
  node->lparenLoc = lparenLoc;
  node->literalLoc = literalLoc;

  if (outputOperandList) {
    auto it = &node->outputOperandList;
    for (auto node : ListView{outputOperandList}) {
      *it = make_list_node<AsmOperandAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (inputOperandList) {
    auto it = &node->inputOperandList;
    for (auto node : ListView{inputOperandList}) {
      *it = make_list_node<AsmOperandAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (clobberList) {
    auto it = &node->clobberList;
    for (auto node : ListView{clobberList}) {
      *it = make_list_node<AsmClobberAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (gotoLabelList) {
    auto it = &node->gotoLabelList;
    for (auto node : ListView{gotoLabelList}) {
      *it = make_list_node<AsmGotoLabelAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;
  node->literal = literal;

  return node;
}

auto AsmDeclarationAST::create(Arena* arena) -> AsmDeclarationAST* {
  auto node = new (arena) AsmDeclarationAST();
  return node;
}

auto AsmDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<AsmQualifierAST*>* asmQualifierList, SourceLocation asmLoc,
    SourceLocation lparenLoc, SourceLocation literalLoc,
    List<AsmOperandAST*>* outputOperandList,
    List<AsmOperandAST*>* inputOperandList, List<AsmClobberAST*>* clobberList,
    List<AsmGotoLabelAST*>* gotoLabelList, SourceLocation rparenLoc,
    SourceLocation semicolonLoc, const Literal* literal) -> AsmDeclarationAST* {
  auto node = new (arena) AsmDeclarationAST();
  node->attributeList = attributeList;
  node->asmQualifierList = asmQualifierList;
  node->asmLoc = asmLoc;
  node->lparenLoc = lparenLoc;
  node->literalLoc = literalLoc;
  node->outputOperandList = outputOperandList;
  node->inputOperandList = inputOperandList;
  node->clobberList = clobberList;
  node->gotoLabelList = gotoLabelList;
  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;
  node->literal = literal;
  return node;
}

auto AsmDeclarationAST::create(Arena* arena,
                               List<AttributeSpecifierAST*>* attributeList,
                               List<AsmQualifierAST*>* asmQualifierList,
                               List<AsmOperandAST*>* outputOperandList,
                               List<AsmOperandAST*>* inputOperandList,
                               List<AsmClobberAST*>* clobberList,
                               List<AsmGotoLabelAST*>* gotoLabelList,
                               const Literal* literal) -> AsmDeclarationAST* {
  auto node = new (arena) AsmDeclarationAST();
  node->attributeList = attributeList;
  node->asmQualifierList = asmQualifierList;
  node->outputOperandList = outputOperandList;
  node->inputOperandList = inputOperandList;
  node->clobberList = clobberList;
  node->gotoLabelList = gotoLabelList;
  node->literal = literal;
  return node;
}

auto NamespaceAliasDefinitionAST::clone(Arena* arena)
    -> NamespaceAliasDefinitionAST* {
  auto node = create(arena);

  node->namespaceLoc = namespaceLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;

  return node;
}

auto NamespaceAliasDefinitionAST::create(Arena* arena)
    -> NamespaceAliasDefinitionAST* {
  auto node = new (arena) NamespaceAliasDefinitionAST();
  return node;
}

auto NamespaceAliasDefinitionAST::create(
    Arena* arena, SourceLocation namespaceLoc, SourceLocation identifierLoc,
    SourceLocation equalLoc, NestedNameSpecifierAST* nestedNameSpecifier,
    NameIdAST* unqualifiedId, SourceLocation semicolonLoc,
    const Identifier* identifier) -> NamespaceAliasDefinitionAST* {
  auto node = new (arena) NamespaceAliasDefinitionAST();
  node->namespaceLoc = namespaceLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  return node;
}

auto NamespaceAliasDefinitionAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    NameIdAST* unqualifiedId, const Identifier* identifier)
    -> NamespaceAliasDefinitionAST* {
  auto node = new (arena) NamespaceAliasDefinitionAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->identifier = identifier;
  return node;
}

auto UsingDeclarationAST::clone(Arena* arena) -> UsingDeclarationAST* {
  auto node = create(arena);

  node->usingLoc = usingLoc;

  if (usingDeclaratorList) {
    auto it = &node->usingDeclaratorList;
    for (auto node : ListView{usingDeclaratorList}) {
      *it = make_list_node<UsingDeclaratorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto UsingDeclarationAST::create(Arena* arena) -> UsingDeclarationAST* {
  auto node = new (arena) UsingDeclarationAST();
  return node;
}

auto UsingDeclarationAST::create(Arena* arena, SourceLocation usingLoc,
                                 List<UsingDeclaratorAST*>* usingDeclaratorList,
                                 SourceLocation semicolonLoc)
    -> UsingDeclarationAST* {
  auto node = new (arena) UsingDeclarationAST();
  node->usingLoc = usingLoc;
  node->usingDeclaratorList = usingDeclaratorList;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto UsingDeclarationAST::create(Arena* arena,
                                 List<UsingDeclaratorAST*>* usingDeclaratorList)
    -> UsingDeclarationAST* {
  auto node = new (arena) UsingDeclarationAST();
  node->usingDeclaratorList = usingDeclaratorList;
  return node;
}

auto UsingEnumDeclarationAST::clone(Arena* arena) -> UsingEnumDeclarationAST* {
  auto node = create(arena);

  node->usingLoc = usingLoc;

  if (enumTypeSpecifier)
    node->enumTypeSpecifier = enumTypeSpecifier->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto UsingEnumDeclarationAST::create(Arena* arena) -> UsingEnumDeclarationAST* {
  auto node = new (arena) UsingEnumDeclarationAST();
  return node;
}

auto UsingEnumDeclarationAST::create(
    Arena* arena, SourceLocation usingLoc,
    ElaboratedTypeSpecifierAST* enumTypeSpecifier, SourceLocation semicolonLoc)
    -> UsingEnumDeclarationAST* {
  auto node = new (arena) UsingEnumDeclarationAST();
  node->usingLoc = usingLoc;
  node->enumTypeSpecifier = enumTypeSpecifier;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto UsingEnumDeclarationAST::create(
    Arena* arena, ElaboratedTypeSpecifierAST* enumTypeSpecifier)
    -> UsingEnumDeclarationAST* {
  auto node = new (arena) UsingEnumDeclarationAST();
  node->enumTypeSpecifier = enumTypeSpecifier;
  return node;
}

auto UsingDirectiveAST::clone(Arena* arena) -> UsingDirectiveAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->usingLoc = usingLoc;
  node->namespaceLoc = namespaceLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto UsingDirectiveAST::create(Arena* arena) -> UsingDirectiveAST* {
  auto node = new (arena) UsingDirectiveAST();
  return node;
}

auto UsingDirectiveAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    SourceLocation usingLoc, SourceLocation namespaceLoc,
    NestedNameSpecifierAST* nestedNameSpecifier, NameIdAST* unqualifiedId,
    SourceLocation semicolonLoc) -> UsingDirectiveAST* {
  auto node = new (arena) UsingDirectiveAST();
  node->attributeList = attributeList;
  node->usingLoc = usingLoc;
  node->namespaceLoc = namespaceLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto UsingDirectiveAST::create(Arena* arena,
                               List<AttributeSpecifierAST*>* attributeList,
                               NestedNameSpecifierAST* nestedNameSpecifier,
                               NameIdAST* unqualifiedId) -> UsingDirectiveAST* {
  auto node = new (arena) UsingDirectiveAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  return node;
}

auto StaticAssertDeclarationAST::clone(Arena* arena)
    -> StaticAssertDeclarationAST* {
  auto node = create(arena);

  node->staticAssertLoc = staticAssertLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->commaLoc = commaLoc;
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;
  node->value = value;

  return node;
}

auto StaticAssertDeclarationAST::create(Arena* arena)
    -> StaticAssertDeclarationAST* {
  auto node = new (arena) StaticAssertDeclarationAST();
  return node;
}

auto StaticAssertDeclarationAST::create(
    Arena* arena, SourceLocation staticAssertLoc, SourceLocation lparenLoc,
    ExpressionAST* expression, SourceLocation commaLoc,
    SourceLocation literalLoc, const Literal* literal, SourceLocation rparenLoc,
    SourceLocation semicolonLoc, std::optional<bool> value)
    -> StaticAssertDeclarationAST* {
  auto node = new (arena) StaticAssertDeclarationAST();
  node->staticAssertLoc = staticAssertLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->commaLoc = commaLoc;
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;
  node->value = value;
  return node;
}

auto StaticAssertDeclarationAST::create(Arena* arena, ExpressionAST* expression,
                                        const Literal* literal,
                                        std::optional<bool> value)
    -> StaticAssertDeclarationAST* {
  auto node = new (arena) StaticAssertDeclarationAST();
  node->expression = expression;
  node->literal = literal;
  node->value = value;
  return node;
}

auto AliasDeclarationAST::clone(Arena* arena) -> AliasDeclarationAST* {
  auto node = create(arena);

  node->usingLoc = usingLoc;
  node->identifierLoc = identifierLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->equalLoc = equalLoc;

  if (gnuAttributeList) {
    auto it = &node->gnuAttributeList;
    for (auto node : ListView{gnuAttributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (typeId) node->typeId = typeId->clone(arena);

  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->symbol = symbol;

  return node;
}

auto AliasDeclarationAST::create(Arena* arena) -> AliasDeclarationAST* {
  auto node = new (arena) AliasDeclarationAST();
  return node;
}

auto AliasDeclarationAST::create(
    Arena* arena, SourceLocation usingLoc, SourceLocation identifierLoc,
    List<AttributeSpecifierAST*>* attributeList, SourceLocation equalLoc,
    List<AttributeSpecifierAST*>* gnuAttributeList, TypeIdAST* typeId,
    SourceLocation semicolonLoc, const Identifier* identifier,
    TypeAliasSymbol* symbol) -> AliasDeclarationAST* {
  auto node = new (arena) AliasDeclarationAST();
  node->usingLoc = usingLoc;
  node->identifierLoc = identifierLoc;
  node->attributeList = attributeList;
  node->equalLoc = equalLoc;
  node->gnuAttributeList = gnuAttributeList;
  node->typeId = typeId;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto AliasDeclarationAST::create(Arena* arena,
                                 List<AttributeSpecifierAST*>* attributeList,
                                 List<AttributeSpecifierAST*>* gnuAttributeList,
                                 TypeIdAST* typeId,
                                 const Identifier* identifier,
                                 TypeAliasSymbol* symbol)
    -> AliasDeclarationAST* {
  auto node = new (arena) AliasDeclarationAST();
  node->attributeList = attributeList;
  node->gnuAttributeList = gnuAttributeList;
  node->typeId = typeId;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto OpaqueEnumDeclarationAST::clone(Arena* arena)
    -> OpaqueEnumDeclarationAST* {
  auto node = create(arena);

  node->enumLoc = enumLoc;
  node->classLoc = classLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->colonLoc = colonLoc;

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->emicolonLoc = emicolonLoc;

  return node;
}

auto OpaqueEnumDeclarationAST::create(Arena* arena)
    -> OpaqueEnumDeclarationAST* {
  auto node = new (arena) OpaqueEnumDeclarationAST();
  return node;
}

auto OpaqueEnumDeclarationAST::create(
    Arena* arena, SourceLocation enumLoc, SourceLocation classLoc,
    List<AttributeSpecifierAST*>* attributeList,
    NestedNameSpecifierAST* nestedNameSpecifier, NameIdAST* unqualifiedId,
    SourceLocation colonLoc, List<SpecifierAST*>* typeSpecifierList,
    SourceLocation emicolonLoc) -> OpaqueEnumDeclarationAST* {
  auto node = new (arena) OpaqueEnumDeclarationAST();
  node->enumLoc = enumLoc;
  node->classLoc = classLoc;
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->colonLoc = colonLoc;
  node->typeSpecifierList = typeSpecifierList;
  node->emicolonLoc = emicolonLoc;
  return node;
}

auto OpaqueEnumDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    NestedNameSpecifierAST* nestedNameSpecifier, NameIdAST* unqualifiedId,
    List<SpecifierAST*>* typeSpecifierList) -> OpaqueEnumDeclarationAST* {
  auto node = new (arena) OpaqueEnumDeclarationAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->typeSpecifierList = typeSpecifierList;
  return node;
}

auto FunctionDefinitionAST::clone(Arena* arena) -> FunctionDefinitionAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declSpecifierList) {
    auto it = &node->declSpecifierList;
    for (auto node : ListView{declSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  if (functionBody) node->functionBody = functionBody->clone(arena);

  node->symbol = symbol;

  return node;
}

auto FunctionDefinitionAST::create(Arena* arena) -> FunctionDefinitionAST* {
  auto node = new (arena) FunctionDefinitionAST();
  return node;
}

auto FunctionDefinitionAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* declSpecifierList, DeclaratorAST* declarator,
    RequiresClauseAST* requiresClause, FunctionBodyAST* functionBody,
    FunctionSymbol* symbol) -> FunctionDefinitionAST* {
  auto node = new (arena) FunctionDefinitionAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->declarator = declarator;
  node->requiresClause = requiresClause;
  node->functionBody = functionBody;
  node->symbol = symbol;
  return node;
}

auto TemplateDeclarationAST::clone(Arena* arena) -> TemplateDeclarationAST* {
  auto node = create(arena);

  node->templateLoc = templateLoc;
  node->lessLoc = lessLoc;

  if (templateParameterList) {
    auto it = &node->templateParameterList;
    for (auto node : ListView{templateParameterList}) {
      *it = make_list_node<TemplateParameterAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  if (declaration) node->declaration = declaration->clone(arena);

  node->symbol = symbol;
  node->depth = depth;

  return node;
}

auto TemplateDeclarationAST::create(Arena* arena) -> TemplateDeclarationAST* {
  auto node = new (arena) TemplateDeclarationAST();
  return node;
}

auto TemplateDeclarationAST::create(
    Arena* arena, SourceLocation templateLoc, SourceLocation lessLoc,
    List<TemplateParameterAST*>* templateParameterList,
    SourceLocation greaterLoc, RequiresClauseAST* requiresClause,
    DeclarationAST* declaration, TemplateParametersSymbol* symbol, int depth)
    -> TemplateDeclarationAST* {
  auto node = new (arena) TemplateDeclarationAST();
  node->templateLoc = templateLoc;
  node->lessLoc = lessLoc;
  node->templateParameterList = templateParameterList;
  node->greaterLoc = greaterLoc;
  node->requiresClause = requiresClause;
  node->declaration = declaration;
  node->symbol = symbol;
  node->depth = depth;
  return node;
}

auto TemplateDeclarationAST::create(
    Arena* arena, List<TemplateParameterAST*>* templateParameterList,
    RequiresClauseAST* requiresClause, DeclarationAST* declaration,
    TemplateParametersSymbol* symbol, int depth) -> TemplateDeclarationAST* {
  auto node = new (arena) TemplateDeclarationAST();
  node->templateParameterList = templateParameterList;
  node->requiresClause = requiresClause;
  node->declaration = declaration;
  node->symbol = symbol;
  node->depth = depth;
  return node;
}

auto ConceptDefinitionAST::clone(Arena* arena) -> ConceptDefinitionAST* {
  auto node = create(arena);

  node->conceptLoc = conceptLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->symbol = symbol;

  return node;
}

auto ConceptDefinitionAST::create(Arena* arena) -> ConceptDefinitionAST* {
  auto node = new (arena) ConceptDefinitionAST();
  return node;
}

auto ConceptDefinitionAST::create(
    Arena* arena, SourceLocation conceptLoc, SourceLocation identifierLoc,
    SourceLocation equalLoc, ExpressionAST* expression,
    SourceLocation semicolonLoc, const Identifier* identifier,
    ConceptSymbol* symbol) -> ConceptDefinitionAST* {
  auto node = new (arena) ConceptDefinitionAST();
  node->conceptLoc = conceptLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto ConceptDefinitionAST::create(Arena* arena, ExpressionAST* expression,
                                  const Identifier* identifier,
                                  ConceptSymbol* symbol)
    -> ConceptDefinitionAST* {
  auto node = new (arena) ConceptDefinitionAST();
  node->expression = expression;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto DeductionGuideAST::clone(Arena* arena) -> DeductionGuideAST* {
  auto node = create(arena);

  if (explicitSpecifier)
    node->explicitSpecifier = explicitSpecifier->clone(arena);

  node->identifierLoc = identifierLoc;
  node->lparenLoc = lparenLoc;

  if (parameterDeclarationClause)
    node->parameterDeclarationClause = parameterDeclarationClause->clone(arena);

  node->rparenLoc = rparenLoc;
  node->arrowLoc = arrowLoc;

  if (templateId) node->templateId = templateId->clone(arena);

  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;

  return node;
}

auto DeductionGuideAST::create(Arena* arena) -> DeductionGuideAST* {
  auto node = new (arena) DeductionGuideAST();
  return node;
}

auto DeductionGuideAST::create(
    Arena* arena, SpecifierAST* explicitSpecifier, SourceLocation identifierLoc,
    SourceLocation lparenLoc,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    SourceLocation rparenLoc, SourceLocation arrowLoc,
    SimpleTemplateIdAST* templateId, SourceLocation semicolonLoc,
    const Identifier* identifier) -> DeductionGuideAST* {
  auto node = new (arena) DeductionGuideAST();
  node->explicitSpecifier = explicitSpecifier;
  node->identifierLoc = identifierLoc;
  node->lparenLoc = lparenLoc;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->rparenLoc = rparenLoc;
  node->arrowLoc = arrowLoc;
  node->templateId = templateId;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  return node;
}

auto DeductionGuideAST::create(
    Arena* arena, SpecifierAST* explicitSpecifier,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    SimpleTemplateIdAST* templateId, const Identifier* identifier)
    -> DeductionGuideAST* {
  auto node = new (arena) DeductionGuideAST();
  node->explicitSpecifier = explicitSpecifier;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->templateId = templateId;
  node->identifier = identifier;
  return node;
}

auto ExplicitInstantiationAST::clone(Arena* arena)
    -> ExplicitInstantiationAST* {
  auto node = create(arena);

  node->externLoc = externLoc;
  node->templateLoc = templateLoc;

  if (declaration) node->declaration = declaration->clone(arena);

  return node;
}

auto ExplicitInstantiationAST::create(Arena* arena)
    -> ExplicitInstantiationAST* {
  auto node = new (arena) ExplicitInstantiationAST();
  return node;
}

auto ExplicitInstantiationAST::create(Arena* arena, SourceLocation externLoc,
                                      SourceLocation templateLoc,
                                      DeclarationAST* declaration)
    -> ExplicitInstantiationAST* {
  auto node = new (arena) ExplicitInstantiationAST();
  node->externLoc = externLoc;
  node->templateLoc = templateLoc;
  node->declaration = declaration;
  return node;
}

auto ExplicitInstantiationAST::create(Arena* arena, DeclarationAST* declaration)
    -> ExplicitInstantiationAST* {
  auto node = new (arena) ExplicitInstantiationAST();
  node->declaration = declaration;
  return node;
}

auto ExportDeclarationAST::clone(Arena* arena) -> ExportDeclarationAST* {
  auto node = create(arena);

  node->exportLoc = exportLoc;

  if (declaration) node->declaration = declaration->clone(arena);

  return node;
}

auto ExportDeclarationAST::create(Arena* arena) -> ExportDeclarationAST* {
  auto node = new (arena) ExportDeclarationAST();
  return node;
}

auto ExportDeclarationAST::create(Arena* arena, SourceLocation exportLoc,
                                  DeclarationAST* declaration)
    -> ExportDeclarationAST* {
  auto node = new (arena) ExportDeclarationAST();
  node->exportLoc = exportLoc;
  node->declaration = declaration;
  return node;
}

auto ExportDeclarationAST::create(Arena* arena, DeclarationAST* declaration)
    -> ExportDeclarationAST* {
  auto node = new (arena) ExportDeclarationAST();
  node->declaration = declaration;
  return node;
}

auto ExportCompoundDeclarationAST::clone(Arena* arena)
    -> ExportCompoundDeclarationAST* {
  auto node = create(arena);

  node->exportLoc = exportLoc;
  node->lbraceLoc = lbraceLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;

  return node;
}

auto ExportCompoundDeclarationAST::create(Arena* arena)
    -> ExportCompoundDeclarationAST* {
  auto node = new (arena) ExportCompoundDeclarationAST();
  return node;
}

auto ExportCompoundDeclarationAST::create(
    Arena* arena, SourceLocation exportLoc, SourceLocation lbraceLoc,
    List<DeclarationAST*>* declarationList, SourceLocation rbraceLoc)
    -> ExportCompoundDeclarationAST* {
  auto node = new (arena) ExportCompoundDeclarationAST();
  node->exportLoc = exportLoc;
  node->lbraceLoc = lbraceLoc;
  node->declarationList = declarationList;
  node->rbraceLoc = rbraceLoc;
  return node;
}

auto ExportCompoundDeclarationAST::create(
    Arena* arena, List<DeclarationAST*>* declarationList)
    -> ExportCompoundDeclarationAST* {
  auto node = new (arena) ExportCompoundDeclarationAST();
  node->declarationList = declarationList;
  return node;
}

auto LinkageSpecificationAST::clone(Arena* arena) -> LinkageSpecificationAST* {
  auto node = create(arena);

  node->externLoc = externLoc;
  node->stringliteralLoc = stringliteralLoc;
  node->lbraceLoc = lbraceLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;
  node->stringLiteral = stringLiteral;

  return node;
}

auto LinkageSpecificationAST::create(Arena* arena) -> LinkageSpecificationAST* {
  auto node = new (arena) LinkageSpecificationAST();
  return node;
}

auto LinkageSpecificationAST::create(Arena* arena, SourceLocation externLoc,
                                     SourceLocation stringliteralLoc,
                                     SourceLocation lbraceLoc,
                                     List<DeclarationAST*>* declarationList,
                                     SourceLocation rbraceLoc,
                                     const StringLiteral* stringLiteral)
    -> LinkageSpecificationAST* {
  auto node = new (arena) LinkageSpecificationAST();
  node->externLoc = externLoc;
  node->stringliteralLoc = stringliteralLoc;
  node->lbraceLoc = lbraceLoc;
  node->declarationList = declarationList;
  node->rbraceLoc = rbraceLoc;
  node->stringLiteral = stringLiteral;
  return node;
}

auto LinkageSpecificationAST::create(Arena* arena,
                                     List<DeclarationAST*>* declarationList,
                                     const StringLiteral* stringLiteral)
    -> LinkageSpecificationAST* {
  auto node = new (arena) LinkageSpecificationAST();
  node->declarationList = declarationList;
  node->stringLiteral = stringLiteral;
  return node;
}

auto NamespaceDefinitionAST::clone(Arena* arena) -> NamespaceDefinitionAST* {
  auto node = create(arena);

  node->inlineLoc = inlineLoc;
  node->namespaceLoc = namespaceLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (nestedNamespaceSpecifierList) {
    auto it = &node->nestedNamespaceSpecifierList;
    for (auto node : ListView{nestedNamespaceSpecifierList}) {
      *it = make_list_node<NestedNamespaceSpecifierAST>(arena,
                                                        node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->identifierLoc = identifierLoc;

  if (extraAttributeList) {
    auto it = &node->extraAttributeList;
    for (auto node : ListView{extraAttributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->lbraceLoc = lbraceLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;
  node->identifier = identifier;
  node->isInline = isInline;

  return node;
}

auto NamespaceDefinitionAST::create(Arena* arena) -> NamespaceDefinitionAST* {
  auto node = new (arena) NamespaceDefinitionAST();
  return node;
}

auto NamespaceDefinitionAST::create(
    Arena* arena, SourceLocation inlineLoc, SourceLocation namespaceLoc,
    List<AttributeSpecifierAST*>* attributeList,
    List<NestedNamespaceSpecifierAST*>* nestedNamespaceSpecifierList,
    SourceLocation identifierLoc,
    List<AttributeSpecifierAST*>* extraAttributeList, SourceLocation lbraceLoc,
    List<DeclarationAST*>* declarationList, SourceLocation rbraceLoc,
    const Identifier* identifier, bool isInline) -> NamespaceDefinitionAST* {
  auto node = new (arena) NamespaceDefinitionAST();
  node->inlineLoc = inlineLoc;
  node->namespaceLoc = namespaceLoc;
  node->attributeList = attributeList;
  node->nestedNamespaceSpecifierList = nestedNamespaceSpecifierList;
  node->identifierLoc = identifierLoc;
  node->extraAttributeList = extraAttributeList;
  node->lbraceLoc = lbraceLoc;
  node->declarationList = declarationList;
  node->rbraceLoc = rbraceLoc;
  node->identifier = identifier;
  node->isInline = isInline;
  return node;
}

auto NamespaceDefinitionAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<NestedNamespaceSpecifierAST*>* nestedNamespaceSpecifierList,
    List<AttributeSpecifierAST*>* extraAttributeList,
    List<DeclarationAST*>* declarationList, const Identifier* identifier,
    bool isInline) -> NamespaceDefinitionAST* {
  auto node = new (arena) NamespaceDefinitionAST();
  node->attributeList = attributeList;
  node->nestedNamespaceSpecifierList = nestedNamespaceSpecifierList;
  node->extraAttributeList = extraAttributeList;
  node->declarationList = declarationList;
  node->identifier = identifier;
  node->isInline = isInline;
  return node;
}

auto EmptyDeclarationAST::clone(Arena* arena) -> EmptyDeclarationAST* {
  auto node = create(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto EmptyDeclarationAST::create(Arena* arena) -> EmptyDeclarationAST* {
  auto node = new (arena) EmptyDeclarationAST();
  return node;
}

auto EmptyDeclarationAST::create(Arena* arena, SourceLocation semicolonLoc)
    -> EmptyDeclarationAST* {
  auto node = new (arena) EmptyDeclarationAST();
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto AttributeDeclarationAST::clone(Arena* arena) -> AttributeDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto AttributeDeclarationAST::create(Arena* arena) -> AttributeDeclarationAST* {
  auto node = new (arena) AttributeDeclarationAST();
  return node;
}

auto AttributeDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    SourceLocation semicolonLoc) -> AttributeDeclarationAST* {
  auto node = new (arena) AttributeDeclarationAST();
  node->attributeList = attributeList;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto AttributeDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList)
    -> AttributeDeclarationAST* {
  auto node = new (arena) AttributeDeclarationAST();
  node->attributeList = attributeList;
  return node;
}

auto ModuleImportDeclarationAST::clone(Arena* arena)
    -> ModuleImportDeclarationAST* {
  auto node = create(arena);

  node->importLoc = importLoc;

  if (importName) node->importName = importName->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto ModuleImportDeclarationAST::create(Arena* arena)
    -> ModuleImportDeclarationAST* {
  auto node = new (arena) ModuleImportDeclarationAST();
  return node;
}

auto ModuleImportDeclarationAST::create(
    Arena* arena, SourceLocation importLoc, ImportNameAST* importName,
    List<AttributeSpecifierAST*>* attributeList, SourceLocation semicolonLoc)
    -> ModuleImportDeclarationAST* {
  auto node = new (arena) ModuleImportDeclarationAST();
  node->importLoc = importLoc;
  node->importName = importName;
  node->attributeList = attributeList;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ModuleImportDeclarationAST::create(
    Arena* arena, ImportNameAST* importName,
    List<AttributeSpecifierAST*>* attributeList)
    -> ModuleImportDeclarationAST* {
  auto node = new (arena) ModuleImportDeclarationAST();
  node->importName = importName;
  node->attributeList = attributeList;
  return node;
}

auto ParameterDeclarationAST::clone(Arena* arena) -> ParameterDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->thisLoc = thisLoc;

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  node->equalLoc = equalLoc;

  if (expression) node->expression = expression->clone(arena);

  node->type = type;
  node->identifier = identifier;
  node->isThisIntroduced = isThisIntroduced;
  node->isPack = isPack;

  return node;
}

auto ParameterDeclarationAST::create(Arena* arena) -> ParameterDeclarationAST* {
  auto node = new (arena) ParameterDeclarationAST();
  return node;
}

auto ParameterDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    SourceLocation thisLoc, List<SpecifierAST*>* typeSpecifierList,
    DeclaratorAST* declarator, SourceLocation equalLoc,
    ExpressionAST* expression, const Type* type, const Identifier* identifier,
    bool isThisIntroduced, bool isPack) -> ParameterDeclarationAST* {
  auto node = new (arena) ParameterDeclarationAST();
  node->attributeList = attributeList;
  node->thisLoc = thisLoc;
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  node->equalLoc = equalLoc;
  node->expression = expression;
  node->type = type;
  node->identifier = identifier;
  node->isThisIntroduced = isThisIntroduced;
  node->isPack = isPack;
  return node;
}

auto ParameterDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* typeSpecifierList, DeclaratorAST* declarator,
    ExpressionAST* expression, const Type* type, const Identifier* identifier,
    bool isThisIntroduced, bool isPack) -> ParameterDeclarationAST* {
  auto node = new (arena) ParameterDeclarationAST();
  node->attributeList = attributeList;
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  node->expression = expression;
  node->type = type;
  node->identifier = identifier;
  node->isThisIntroduced = isThisIntroduced;
  node->isPack = isPack;
  return node;
}

auto AccessDeclarationAST::clone(Arena* arena) -> AccessDeclarationAST* {
  auto node = create(arena);

  node->accessLoc = accessLoc;
  node->colonLoc = colonLoc;
  node->accessSpecifier = accessSpecifier;

  return node;
}

auto AccessDeclarationAST::create(Arena* arena) -> AccessDeclarationAST* {
  auto node = new (arena) AccessDeclarationAST();
  return node;
}

auto AccessDeclarationAST::create(Arena* arena, SourceLocation accessLoc,
                                  SourceLocation colonLoc,
                                  TokenKind accessSpecifier)
    -> AccessDeclarationAST* {
  auto node = new (arena) AccessDeclarationAST();
  node->accessLoc = accessLoc;
  node->colonLoc = colonLoc;
  node->accessSpecifier = accessSpecifier;
  return node;
}

auto AccessDeclarationAST::create(Arena* arena, TokenKind accessSpecifier)
    -> AccessDeclarationAST* {
  auto node = new (arena) AccessDeclarationAST();
  node->accessSpecifier = accessSpecifier;
  return node;
}

auto ForRangeDeclarationAST::clone(Arena* arena) -> ForRangeDeclarationAST* {
  auto node = create(arena);

  return node;
}

auto ForRangeDeclarationAST::create(Arena* arena) -> ForRangeDeclarationAST* {
  auto node = new (arena) ForRangeDeclarationAST();
  return node;
}

auto StructuredBindingDeclarationAST::clone(Arena* arena)
    -> StructuredBindingDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declSpecifierList) {
    auto it = &node->declSpecifierList;
    for (auto node : ListView{declSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->refQualifierLoc = refQualifierLoc;
  node->lbracketLoc = lbracketLoc;

  if (bindingList) {
    auto it = &node->bindingList;
    for (auto node : ListView{bindingList}) {
      *it = make_list_node<NameIdAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbracketLoc = rbracketLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto StructuredBindingDeclarationAST::create(Arena* arena)
    -> StructuredBindingDeclarationAST* {
  auto node = new (arena) StructuredBindingDeclarationAST();
  return node;
}

auto StructuredBindingDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* declSpecifierList, SourceLocation refQualifierLoc,
    SourceLocation lbracketLoc, List<NameIdAST*>* bindingList,
    SourceLocation rbracketLoc, ExpressionAST* initializer,
    SourceLocation semicolonLoc) -> StructuredBindingDeclarationAST* {
  auto node = new (arena) StructuredBindingDeclarationAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->refQualifierLoc = refQualifierLoc;
  node->lbracketLoc = lbracketLoc;
  node->bindingList = bindingList;
  node->rbracketLoc = rbracketLoc;
  node->initializer = initializer;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto StructuredBindingDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* declSpecifierList, List<NameIdAST*>* bindingList,
    ExpressionAST* initializer) -> StructuredBindingDeclarationAST* {
  auto node = new (arena) StructuredBindingDeclarationAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->bindingList = bindingList;
  node->initializer = initializer;
  return node;
}

auto AsmOperandAST::clone(Arena* arena) -> AsmOperandAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;
  node->symbolicNameLoc = symbolicNameLoc;
  node->rbracketLoc = rbracketLoc;
  node->constraintLiteralLoc = constraintLiteralLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->symbolicName = symbolicName;
  node->constraintLiteral = constraintLiteral;

  return node;
}

auto AsmOperandAST::create(Arena* arena) -> AsmOperandAST* {
  auto node = new (arena) AsmOperandAST();
  return node;
}

auto AsmOperandAST::create(Arena* arena, SourceLocation lbracketLoc,
                           SourceLocation symbolicNameLoc,
                           SourceLocation rbracketLoc,
                           SourceLocation constraintLiteralLoc,
                           SourceLocation lparenLoc, ExpressionAST* expression,
                           SourceLocation rparenLoc,
                           const Identifier* symbolicName,
                           const Literal* constraintLiteral) -> AsmOperandAST* {
  auto node = new (arena) AsmOperandAST();
  node->lbracketLoc = lbracketLoc;
  node->symbolicNameLoc = symbolicNameLoc;
  node->rbracketLoc = rbracketLoc;
  node->constraintLiteralLoc = constraintLiteralLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->symbolicName = symbolicName;
  node->constraintLiteral = constraintLiteral;
  return node;
}

auto AsmOperandAST::create(Arena* arena, ExpressionAST* expression,
                           const Identifier* symbolicName,
                           const Literal* constraintLiteral) -> AsmOperandAST* {
  auto node = new (arena) AsmOperandAST();
  node->expression = expression;
  node->symbolicName = symbolicName;
  node->constraintLiteral = constraintLiteral;
  return node;
}

auto AsmQualifierAST::clone(Arena* arena) -> AsmQualifierAST* {
  auto node = create(arena);

  node->qualifierLoc = qualifierLoc;
  node->qualifier = qualifier;

  return node;
}

auto AsmQualifierAST::create(Arena* arena) -> AsmQualifierAST* {
  auto node = new (arena) AsmQualifierAST();
  return node;
}

auto AsmQualifierAST::create(Arena* arena, SourceLocation qualifierLoc,
                             TokenKind qualifier) -> AsmQualifierAST* {
  auto node = new (arena) AsmQualifierAST();
  node->qualifierLoc = qualifierLoc;
  node->qualifier = qualifier;
  return node;
}

auto AsmQualifierAST::create(Arena* arena, TokenKind qualifier)
    -> AsmQualifierAST* {
  auto node = new (arena) AsmQualifierAST();
  node->qualifier = qualifier;
  return node;
}

auto AsmClobberAST::clone(Arena* arena) -> AsmClobberAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;

  return node;
}

auto AsmClobberAST::create(Arena* arena) -> AsmClobberAST* {
  auto node = new (arena) AsmClobberAST();
  return node;
}

auto AsmClobberAST::create(Arena* arena, SourceLocation literalLoc,
                           const StringLiteral* literal) -> AsmClobberAST* {
  auto node = new (arena) AsmClobberAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  return node;
}

auto AsmClobberAST::create(Arena* arena, const StringLiteral* literal)
    -> AsmClobberAST* {
  auto node = new (arena) AsmClobberAST();
  node->literal = literal;
  return node;
}

auto AsmGotoLabelAST::clone(Arena* arena) -> AsmGotoLabelAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->identifier = identifier;

  return node;
}

auto AsmGotoLabelAST::create(Arena* arena) -> AsmGotoLabelAST* {
  auto node = new (arena) AsmGotoLabelAST();
  return node;
}

auto AsmGotoLabelAST::create(Arena* arena, SourceLocation identifierLoc,
                             const Identifier* identifier) -> AsmGotoLabelAST* {
  auto node = new (arena) AsmGotoLabelAST();
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  return node;
}

auto AsmGotoLabelAST::create(Arena* arena, const Identifier* identifier)
    -> AsmGotoLabelAST* {
  auto node = new (arena) AsmGotoLabelAST();
  node->identifier = identifier;
  return node;
}

auto SplicerAST::clone(Arena* arena) -> SplicerAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;
  node->colonLoc = colonLoc;
  node->ellipsisLoc = ellipsisLoc;

  if (expression) node->expression = expression->clone(arena);

  node->secondColonLoc = secondColonLoc;
  node->rbracketLoc = rbracketLoc;

  return node;
}

auto SplicerAST::create(Arena* arena) -> SplicerAST* {
  auto node = new (arena) SplicerAST();
  return node;
}

auto SplicerAST::create(Arena* arena, SourceLocation lbracketLoc,
                        SourceLocation colonLoc, SourceLocation ellipsisLoc,
                        ExpressionAST* expression,
                        SourceLocation secondColonLoc,
                        SourceLocation rbracketLoc) -> SplicerAST* {
  auto node = new (arena) SplicerAST();
  node->lbracketLoc = lbracketLoc;
  node->colonLoc = colonLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->expression = expression;
  node->secondColonLoc = secondColonLoc;
  node->rbracketLoc = rbracketLoc;
  return node;
}

auto SplicerAST::create(Arena* arena, ExpressionAST* expression)
    -> SplicerAST* {
  auto node = new (arena) SplicerAST();
  node->expression = expression;
  return node;
}

auto GlobalModuleFragmentAST::clone(Arena* arena) -> GlobalModuleFragmentAST* {
  auto node = create(arena);

  node->moduleLoc = moduleLoc;
  node->semicolonLoc = semicolonLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto GlobalModuleFragmentAST::create(Arena* arena) -> GlobalModuleFragmentAST* {
  auto node = new (arena) GlobalModuleFragmentAST();
  return node;
}

auto GlobalModuleFragmentAST::create(Arena* arena, SourceLocation moduleLoc,
                                     SourceLocation semicolonLoc,
                                     List<DeclarationAST*>* declarationList)
    -> GlobalModuleFragmentAST* {
  auto node = new (arena) GlobalModuleFragmentAST();
  node->moduleLoc = moduleLoc;
  node->semicolonLoc = semicolonLoc;
  node->declarationList = declarationList;
  return node;
}

auto GlobalModuleFragmentAST::create(Arena* arena,
                                     List<DeclarationAST*>* declarationList)
    -> GlobalModuleFragmentAST* {
  auto node = new (arena) GlobalModuleFragmentAST();
  node->declarationList = declarationList;
  return node;
}

auto PrivateModuleFragmentAST::clone(Arena* arena)
    -> PrivateModuleFragmentAST* {
  auto node = create(arena);

  node->moduleLoc = moduleLoc;
  node->colonLoc = colonLoc;
  node->privateLoc = privateLoc;
  node->semicolonLoc = semicolonLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto PrivateModuleFragmentAST::create(Arena* arena)
    -> PrivateModuleFragmentAST* {
  auto node = new (arena) PrivateModuleFragmentAST();
  return node;
}

auto PrivateModuleFragmentAST::create(Arena* arena, SourceLocation moduleLoc,
                                      SourceLocation colonLoc,
                                      SourceLocation privateLoc,
                                      SourceLocation semicolonLoc,
                                      List<DeclarationAST*>* declarationList)
    -> PrivateModuleFragmentAST* {
  auto node = new (arena) PrivateModuleFragmentAST();
  node->moduleLoc = moduleLoc;
  node->colonLoc = colonLoc;
  node->privateLoc = privateLoc;
  node->semicolonLoc = semicolonLoc;
  node->declarationList = declarationList;
  return node;
}

auto PrivateModuleFragmentAST::create(Arena* arena,
                                      List<DeclarationAST*>* declarationList)
    -> PrivateModuleFragmentAST* {
  auto node = new (arena) PrivateModuleFragmentAST();
  node->declarationList = declarationList;
  return node;
}

auto ModuleDeclarationAST::clone(Arena* arena) -> ModuleDeclarationAST* {
  auto node = create(arena);

  node->exportLoc = exportLoc;
  node->moduleLoc = moduleLoc;

  if (moduleName) node->moduleName = moduleName->clone(arena);

  if (modulePartition) node->modulePartition = modulePartition->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto ModuleDeclarationAST::create(Arena* arena) -> ModuleDeclarationAST* {
  auto node = new (arena) ModuleDeclarationAST();
  return node;
}

auto ModuleDeclarationAST::create(Arena* arena, SourceLocation exportLoc,
                                  SourceLocation moduleLoc,
                                  ModuleNameAST* moduleName,
                                  ModulePartitionAST* modulePartition,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  SourceLocation semicolonLoc)
    -> ModuleDeclarationAST* {
  auto node = new (arena) ModuleDeclarationAST();
  node->exportLoc = exportLoc;
  node->moduleLoc = moduleLoc;
  node->moduleName = moduleName;
  node->modulePartition = modulePartition;
  node->attributeList = attributeList;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ModuleDeclarationAST::create(Arena* arena, ModuleNameAST* moduleName,
                                  ModulePartitionAST* modulePartition,
                                  List<AttributeSpecifierAST*>* attributeList)
    -> ModuleDeclarationAST* {
  auto node = new (arena) ModuleDeclarationAST();
  node->moduleName = moduleName;
  node->modulePartition = modulePartition;
  node->attributeList = attributeList;
  return node;
}

auto ModuleNameAST::clone(Arena* arena) -> ModuleNameAST* {
  auto node = create(arena);

  if (moduleQualifier) node->moduleQualifier = moduleQualifier->clone(arena);

  node->identifierLoc = identifierLoc;
  node->identifier = identifier;

  return node;
}

auto ModuleNameAST::create(Arena* arena) -> ModuleNameAST* {
  auto node = new (arena) ModuleNameAST();
  return node;
}

auto ModuleNameAST::create(Arena* arena, ModuleQualifierAST* moduleQualifier,
                           SourceLocation identifierLoc,
                           const Identifier* identifier) -> ModuleNameAST* {
  auto node = new (arena) ModuleNameAST();
  node->moduleQualifier = moduleQualifier;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  return node;
}

auto ModuleNameAST::create(Arena* arena, ModuleQualifierAST* moduleQualifier,
                           const Identifier* identifier) -> ModuleNameAST* {
  auto node = new (arena) ModuleNameAST();
  node->moduleQualifier = moduleQualifier;
  node->identifier = identifier;
  return node;
}

auto ModuleQualifierAST::clone(Arena* arena) -> ModuleQualifierAST* {
  auto node = create(arena);

  if (moduleQualifier) node->moduleQualifier = moduleQualifier->clone(arena);

  node->identifierLoc = identifierLoc;
  node->dotLoc = dotLoc;
  node->identifier = identifier;

  return node;
}

auto ModuleQualifierAST::create(Arena* arena) -> ModuleQualifierAST* {
  auto node = new (arena) ModuleQualifierAST();
  return node;
}

auto ModuleQualifierAST::create(Arena* arena,
                                ModuleQualifierAST* moduleQualifier,
                                SourceLocation identifierLoc,
                                SourceLocation dotLoc,
                                const Identifier* identifier)
    -> ModuleQualifierAST* {
  auto node = new (arena) ModuleQualifierAST();
  node->moduleQualifier = moduleQualifier;
  node->identifierLoc = identifierLoc;
  node->dotLoc = dotLoc;
  node->identifier = identifier;
  return node;
}

auto ModuleQualifierAST::create(Arena* arena,
                                ModuleQualifierAST* moduleQualifier,
                                const Identifier* identifier)
    -> ModuleQualifierAST* {
  auto node = new (arena) ModuleQualifierAST();
  node->moduleQualifier = moduleQualifier;
  node->identifier = identifier;
  return node;
}

auto ModulePartitionAST::clone(Arena* arena) -> ModulePartitionAST* {
  auto node = create(arena);

  node->colonLoc = colonLoc;

  if (moduleName) node->moduleName = moduleName->clone(arena);

  return node;
}

auto ModulePartitionAST::create(Arena* arena) -> ModulePartitionAST* {
  auto node = new (arena) ModulePartitionAST();
  return node;
}

auto ModulePartitionAST::create(Arena* arena, SourceLocation colonLoc,
                                ModuleNameAST* moduleName)
    -> ModulePartitionAST* {
  auto node = new (arena) ModulePartitionAST();
  node->colonLoc = colonLoc;
  node->moduleName = moduleName;
  return node;
}

auto ModulePartitionAST::create(Arena* arena, ModuleNameAST* moduleName)
    -> ModulePartitionAST* {
  auto node = new (arena) ModulePartitionAST();
  node->moduleName = moduleName;
  return node;
}

auto ImportNameAST::clone(Arena* arena) -> ImportNameAST* {
  auto node = create(arena);

  node->headerLoc = headerLoc;

  if (modulePartition) node->modulePartition = modulePartition->clone(arena);

  if (moduleName) node->moduleName = moduleName->clone(arena);

  return node;
}

auto ImportNameAST::create(Arena* arena) -> ImportNameAST* {
  auto node = new (arena) ImportNameAST();
  return node;
}

auto ImportNameAST::create(Arena* arena, SourceLocation headerLoc,
                           ModulePartitionAST* modulePartition,
                           ModuleNameAST* moduleName) -> ImportNameAST* {
  auto node = new (arena) ImportNameAST();
  node->headerLoc = headerLoc;
  node->modulePartition = modulePartition;
  node->moduleName = moduleName;
  return node;
}

auto ImportNameAST::create(Arena* arena, ModulePartitionAST* modulePartition,
                           ModuleNameAST* moduleName) -> ImportNameAST* {
  auto node = new (arena) ImportNameAST();
  node->modulePartition = modulePartition;
  node->moduleName = moduleName;
  return node;
}

auto InitDeclaratorAST::clone(Arena* arena) -> InitDeclaratorAST* {
  auto node = create(arena);

  if (declarator) node->declarator = declarator->clone(arena);

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  if (initializer) node->initializer = initializer->clone(arena);

  node->symbol = symbol;

  return node;
}

auto InitDeclaratorAST::create(Arena* arena) -> InitDeclaratorAST* {
  auto node = new (arena) InitDeclaratorAST();
  return node;
}

auto InitDeclaratorAST::create(Arena* arena, DeclaratorAST* declarator,
                               RequiresClauseAST* requiresClause,
                               ExpressionAST* initializer, Symbol* symbol)
    -> InitDeclaratorAST* {
  auto node = new (arena) InitDeclaratorAST();
  node->declarator = declarator;
  node->requiresClause = requiresClause;
  node->initializer = initializer;
  node->symbol = symbol;
  return node;
}

auto DeclaratorAST::clone(Arena* arena) -> DeclaratorAST* {
  auto node = create(arena);

  if (ptrOpList) {
    auto it = &node->ptrOpList;
    for (auto node : ListView{ptrOpList}) {
      *it = make_list_node<PtrOperatorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (coreDeclarator) node->coreDeclarator = coreDeclarator->clone(arena);

  if (declaratorChunkList) {
    auto it = &node->declaratorChunkList;
    for (auto node : ListView{declaratorChunkList}) {
      *it = make_list_node<DeclaratorChunkAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto DeclaratorAST::create(Arena* arena) -> DeclaratorAST* {
  auto node = new (arena) DeclaratorAST();
  return node;
}

auto DeclaratorAST::create(Arena* arena, List<PtrOperatorAST*>* ptrOpList,
                           CoreDeclaratorAST* coreDeclarator,
                           List<DeclaratorChunkAST*>* declaratorChunkList)
    -> DeclaratorAST* {
  auto node = new (arena) DeclaratorAST();
  node->ptrOpList = ptrOpList;
  node->coreDeclarator = coreDeclarator;
  node->declaratorChunkList = declaratorChunkList;
  return node;
}

auto UsingDeclaratorAST::clone(Arena* arena) -> UsingDeclaratorAST* {
  auto node = create(arena);

  node->typenameLoc = typenameLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->symbol = symbol;
  node->isPack = isPack;

  return node;
}

auto UsingDeclaratorAST::create(Arena* arena) -> UsingDeclaratorAST* {
  auto node = new (arena) UsingDeclaratorAST();
  return node;
}

auto UsingDeclaratorAST::create(Arena* arena, SourceLocation typenameLoc,
                                NestedNameSpecifierAST* nestedNameSpecifier,
                                UnqualifiedIdAST* unqualifiedId,
                                SourceLocation ellipsisLoc,
                                UsingDeclarationSymbol* symbol, bool isPack)
    -> UsingDeclaratorAST* {
  auto node = new (arena) UsingDeclaratorAST();
  node->typenameLoc = typenameLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->ellipsisLoc = ellipsisLoc;
  node->symbol = symbol;
  node->isPack = isPack;
  return node;
}

auto UsingDeclaratorAST::create(Arena* arena,
                                NestedNameSpecifierAST* nestedNameSpecifier,
                                UnqualifiedIdAST* unqualifiedId,
                                UsingDeclarationSymbol* symbol, bool isPack)
    -> UsingDeclaratorAST* {
  auto node = new (arena) UsingDeclaratorAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->symbol = symbol;
  node->isPack = isPack;
  return node;
}

auto EnumeratorAST::clone(Arena* arena) -> EnumeratorAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->equalLoc = equalLoc;

  if (expression) node->expression = expression->clone(arena);

  node->identifier = identifier;
  node->symbol = symbol;

  return node;
}

auto EnumeratorAST::create(Arena* arena) -> EnumeratorAST* {
  auto node = new (arena) EnumeratorAST();
  return node;
}

auto EnumeratorAST::create(Arena* arena, SourceLocation identifierLoc,
                           List<AttributeSpecifierAST*>* attributeList,
                           SourceLocation equalLoc, ExpressionAST* expression,
                           const Identifier* identifier,
                           EnumeratorSymbol* symbol) -> EnumeratorAST* {
  auto node = new (arena) EnumeratorAST();
  node->identifierLoc = identifierLoc;
  node->attributeList = attributeList;
  node->equalLoc = equalLoc;
  node->expression = expression;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto EnumeratorAST::create(Arena* arena,
                           List<AttributeSpecifierAST*>* attributeList,
                           ExpressionAST* expression,
                           const Identifier* identifier,
                           EnumeratorSymbol* symbol) -> EnumeratorAST* {
  auto node = new (arena) EnumeratorAST();
  node->attributeList = attributeList;
  node->expression = expression;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto TypeIdAST::clone(Arena* arena) -> TypeIdAST* {
  auto node = create(arena);

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  node->type = type;

  return node;
}

auto TypeIdAST::create(Arena* arena) -> TypeIdAST* {
  auto node = new (arena) TypeIdAST();
  return node;
}

auto TypeIdAST::create(Arena* arena, List<SpecifierAST*>* typeSpecifierList,
                       DeclaratorAST* declarator, const Type* type)
    -> TypeIdAST* {
  auto node = new (arena) TypeIdAST();
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  node->type = type;
  return node;
}

auto HandlerAST::clone(Arena* arena) -> HandlerAST* {
  auto node = create(arena);

  node->catchLoc = catchLoc;
  node->lparenLoc = lparenLoc;

  if (exceptionDeclaration)
    node->exceptionDeclaration = exceptionDeclaration->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  return node;
}

auto HandlerAST::create(Arena* arena) -> HandlerAST* {
  auto node = new (arena) HandlerAST();
  return node;
}

auto HandlerAST::create(Arena* arena, SourceLocation catchLoc,
                        SourceLocation lparenLoc,
                        ExceptionDeclarationAST* exceptionDeclaration,
                        SourceLocation rparenLoc,
                        CompoundStatementAST* statement) -> HandlerAST* {
  auto node = new (arena) HandlerAST();
  node->catchLoc = catchLoc;
  node->lparenLoc = lparenLoc;
  node->exceptionDeclaration = exceptionDeclaration;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  return node;
}

auto HandlerAST::create(Arena* arena,
                        ExceptionDeclarationAST* exceptionDeclaration,
                        CompoundStatementAST* statement) -> HandlerAST* {
  auto node = new (arena) HandlerAST();
  node->exceptionDeclaration = exceptionDeclaration;
  node->statement = statement;
  return node;
}

auto BaseSpecifierAST::clone(Arena* arena) -> BaseSpecifierAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->virtualOrAccessLoc = virtualOrAccessLoc;
  node->otherVirtualOrAccessLoc = otherVirtualOrAccessLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->isVirtual = isVirtual;
  node->isVariadic = isVariadic;
  node->accessSpecifier = accessSpecifier;
  node->symbol = symbol;

  return node;
}

auto BaseSpecifierAST::create(Arena* arena) -> BaseSpecifierAST* {
  auto node = new (arena) BaseSpecifierAST();
  return node;
}

auto BaseSpecifierAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    SourceLocation virtualOrAccessLoc, SourceLocation otherVirtualOrAccessLoc,
    NestedNameSpecifierAST* nestedNameSpecifier, SourceLocation templateLoc,
    UnqualifiedIdAST* unqualifiedId, SourceLocation ellipsisLoc,
    bool isTemplateIntroduced, bool isVirtual, bool isVariadic,
    TokenKind accessSpecifier, BaseClassSymbol* symbol) -> BaseSpecifierAST* {
  auto node = new (arena) BaseSpecifierAST();
  node->attributeList = attributeList;
  node->virtualOrAccessLoc = virtualOrAccessLoc;
  node->otherVirtualOrAccessLoc = otherVirtualOrAccessLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->ellipsisLoc = ellipsisLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->isVirtual = isVirtual;
  node->isVariadic = isVariadic;
  node->accessSpecifier = accessSpecifier;
  node->symbol = symbol;
  return node;
}

auto BaseSpecifierAST::create(Arena* arena,
                              List<AttributeSpecifierAST*>* attributeList,
                              NestedNameSpecifierAST* nestedNameSpecifier,
                              UnqualifiedIdAST* unqualifiedId,
                              bool isTemplateIntroduced, bool isVirtual,
                              bool isVariadic, TokenKind accessSpecifier,
                              BaseClassSymbol* symbol) -> BaseSpecifierAST* {
  auto node = new (arena) BaseSpecifierAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->isVirtual = isVirtual;
  node->isVariadic = isVariadic;
  node->accessSpecifier = accessSpecifier;
  node->symbol = symbol;
  return node;
}

auto RequiresClauseAST::clone(Arena* arena) -> RequiresClauseAST* {
  auto node = create(arena);

  node->requiresLoc = requiresLoc;

  if (expression) node->expression = expression->clone(arena);

  return node;
}

auto RequiresClauseAST::create(Arena* arena) -> RequiresClauseAST* {
  auto node = new (arena) RequiresClauseAST();
  return node;
}

auto RequiresClauseAST::create(Arena* arena, SourceLocation requiresLoc,
                               ExpressionAST* expression)
    -> RequiresClauseAST* {
  auto node = new (arena) RequiresClauseAST();
  node->requiresLoc = requiresLoc;
  node->expression = expression;
  return node;
}

auto RequiresClauseAST::create(Arena* arena, ExpressionAST* expression)
    -> RequiresClauseAST* {
  auto node = new (arena) RequiresClauseAST();
  node->expression = expression;
  return node;
}

auto ParameterDeclarationClauseAST::clone(Arena* arena)
    -> ParameterDeclarationClauseAST* {
  auto node = create(arena);

  if (parameterDeclarationList) {
    auto it = &node->parameterDeclarationList;
    for (auto node : ListView{parameterDeclarationList}) {
      *it = make_list_node<ParameterDeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->commaLoc = commaLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->functionParametersSymbol = functionParametersSymbol;
  node->isVariadic = isVariadic;

  return node;
}

auto ParameterDeclarationClauseAST::create(Arena* arena)
    -> ParameterDeclarationClauseAST* {
  auto node = new (arena) ParameterDeclarationClauseAST();
  return node;
}

auto ParameterDeclarationClauseAST::create(
    Arena* arena, List<ParameterDeclarationAST*>* parameterDeclarationList,
    SourceLocation commaLoc, SourceLocation ellipsisLoc,
    FunctionParametersSymbol* functionParametersSymbol, bool isVariadic)
    -> ParameterDeclarationClauseAST* {
  auto node = new (arena) ParameterDeclarationClauseAST();
  node->parameterDeclarationList = parameterDeclarationList;
  node->commaLoc = commaLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->functionParametersSymbol = functionParametersSymbol;
  node->isVariadic = isVariadic;
  return node;
}

auto ParameterDeclarationClauseAST::create(
    Arena* arena, List<ParameterDeclarationAST*>* parameterDeclarationList,
    FunctionParametersSymbol* functionParametersSymbol, bool isVariadic)
    -> ParameterDeclarationClauseAST* {
  auto node = new (arena) ParameterDeclarationClauseAST();
  node->parameterDeclarationList = parameterDeclarationList;
  node->functionParametersSymbol = functionParametersSymbol;
  node->isVariadic = isVariadic;
  return node;
}

auto TrailingReturnTypeAST::clone(Arena* arena) -> TrailingReturnTypeAST* {
  auto node = create(arena);

  node->minusGreaterLoc = minusGreaterLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  return node;
}

auto TrailingReturnTypeAST::create(Arena* arena) -> TrailingReturnTypeAST* {
  auto node = new (arena) TrailingReturnTypeAST();
  return node;
}

auto TrailingReturnTypeAST::create(Arena* arena, SourceLocation minusGreaterLoc,
                                   TypeIdAST* typeId)
    -> TrailingReturnTypeAST* {
  auto node = new (arena) TrailingReturnTypeAST();
  node->minusGreaterLoc = minusGreaterLoc;
  node->typeId = typeId;
  return node;
}

auto TrailingReturnTypeAST::create(Arena* arena, TypeIdAST* typeId)
    -> TrailingReturnTypeAST* {
  auto node = new (arena) TrailingReturnTypeAST();
  node->typeId = typeId;
  return node;
}

auto LambdaSpecifierAST::clone(Arena* arena) -> LambdaSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto LambdaSpecifierAST::create(Arena* arena) -> LambdaSpecifierAST* {
  auto node = new (arena) LambdaSpecifierAST();
  return node;
}

auto LambdaSpecifierAST::create(Arena* arena, SourceLocation specifierLoc,
                                TokenKind specifier) -> LambdaSpecifierAST* {
  auto node = new (arena) LambdaSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto LambdaSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> LambdaSpecifierAST* {
  auto node = new (arena) LambdaSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto TypeConstraintAST::clone(Arena* arena) -> TypeConstraintAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->identifierLoc = identifierLoc;
  node->lessLoc = lessLoc;

  if (templateArgumentList) {
    auto it = &node->templateArgumentList;
    for (auto node : ListView{templateArgumentList}) {
      *it = make_list_node<TemplateArgumentAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;
  node->identifier = identifier;

  return node;
}

auto TypeConstraintAST::create(Arena* arena) -> TypeConstraintAST* {
  auto node = new (arena) TypeConstraintAST();
  return node;
}

auto TypeConstraintAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    SourceLocation identifierLoc, SourceLocation lessLoc,
    List<TemplateArgumentAST*>* templateArgumentList, SourceLocation greaterLoc,
    const Identifier* identifier) -> TypeConstraintAST* {
  auto node = new (arena) TypeConstraintAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->identifierLoc = identifierLoc;
  node->lessLoc = lessLoc;
  node->templateArgumentList = templateArgumentList;
  node->greaterLoc = greaterLoc;
  node->identifier = identifier;
  return node;
}

auto TypeConstraintAST::create(Arena* arena,
                               NestedNameSpecifierAST* nestedNameSpecifier,
                               List<TemplateArgumentAST*>* templateArgumentList,
                               const Identifier* identifier)
    -> TypeConstraintAST* {
  auto node = new (arena) TypeConstraintAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateArgumentList = templateArgumentList;
  node->identifier = identifier;
  return node;
}

auto AttributeArgumentClauseAST::clone(Arena* arena)
    -> AttributeArgumentClauseAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;
  node->rparenLoc = rparenLoc;

  return node;
}

auto AttributeArgumentClauseAST::create(Arena* arena)
    -> AttributeArgumentClauseAST* {
  auto node = new (arena) AttributeArgumentClauseAST();
  return node;
}

auto AttributeArgumentClauseAST::create(Arena* arena, SourceLocation lparenLoc,
                                        SourceLocation rparenLoc)
    -> AttributeArgumentClauseAST* {
  auto node = new (arena) AttributeArgumentClauseAST();
  node->lparenLoc = lparenLoc;
  node->rparenLoc = rparenLoc;
  return node;
}

auto AttributeAST::clone(Arena* arena) -> AttributeAST* {
  auto node = create(arena);

  if (attributeToken) node->attributeToken = attributeToken->clone(arena);

  if (attributeArgumentClause)
    node->attributeArgumentClause = attributeArgumentClause->clone(arena);

  node->ellipsisLoc = ellipsisLoc;

  return node;
}

auto AttributeAST::create(Arena* arena) -> AttributeAST* {
  auto node = new (arena) AttributeAST();
  return node;
}

auto AttributeAST::create(Arena* arena, AttributeTokenAST* attributeToken,
                          AttributeArgumentClauseAST* attributeArgumentClause,
                          SourceLocation ellipsisLoc) -> AttributeAST* {
  auto node = new (arena) AttributeAST();
  node->attributeToken = attributeToken;
  node->attributeArgumentClause = attributeArgumentClause;
  node->ellipsisLoc = ellipsisLoc;
  return node;
}

auto AttributeAST::create(Arena* arena, AttributeTokenAST* attributeToken,
                          AttributeArgumentClauseAST* attributeArgumentClause)
    -> AttributeAST* {
  auto node = new (arena) AttributeAST();
  node->attributeToken = attributeToken;
  node->attributeArgumentClause = attributeArgumentClause;
  return node;
}

auto AttributeUsingPrefixAST::clone(Arena* arena) -> AttributeUsingPrefixAST* {
  auto node = create(arena);

  node->usingLoc = usingLoc;
  node->attributeNamespaceLoc = attributeNamespaceLoc;
  node->colonLoc = colonLoc;

  return node;
}

auto AttributeUsingPrefixAST::create(Arena* arena) -> AttributeUsingPrefixAST* {
  auto node = new (arena) AttributeUsingPrefixAST();
  return node;
}

auto AttributeUsingPrefixAST::create(Arena* arena, SourceLocation usingLoc,
                                     SourceLocation attributeNamespaceLoc,
                                     SourceLocation colonLoc)
    -> AttributeUsingPrefixAST* {
  auto node = new (arena) AttributeUsingPrefixAST();
  node->usingLoc = usingLoc;
  node->attributeNamespaceLoc = attributeNamespaceLoc;
  node->colonLoc = colonLoc;
  return node;
}

auto NewPlacementAST::clone(Arena* arena) -> NewPlacementAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;

  return node;
}

auto NewPlacementAST::create(Arena* arena) -> NewPlacementAST* {
  auto node = new (arena) NewPlacementAST();
  return node;
}

auto NewPlacementAST::create(Arena* arena, SourceLocation lparenLoc,
                             List<ExpressionAST*>* expressionList,
                             SourceLocation rparenLoc) -> NewPlacementAST* {
  auto node = new (arena) NewPlacementAST();
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  return node;
}

auto NewPlacementAST::create(Arena* arena, List<ExpressionAST*>* expressionList)
    -> NewPlacementAST* {
  auto node = new (arena) NewPlacementAST();
  node->expressionList = expressionList;
  return node;
}

auto NestedNamespaceSpecifierAST::clone(Arena* arena)
    -> NestedNamespaceSpecifierAST* {
  auto node = create(arena);

  node->inlineLoc = inlineLoc;
  node->identifierLoc = identifierLoc;
  node->scopeLoc = scopeLoc;
  node->identifier = identifier;
  node->isInline = isInline;

  return node;
}

auto NestedNamespaceSpecifierAST::create(Arena* arena)
    -> NestedNamespaceSpecifierAST* {
  auto node = new (arena) NestedNamespaceSpecifierAST();
  return node;
}

auto NestedNamespaceSpecifierAST::create(Arena* arena, SourceLocation inlineLoc,
                                         SourceLocation identifierLoc,
                                         SourceLocation scopeLoc,
                                         const Identifier* identifier,
                                         bool isInline)
    -> NestedNamespaceSpecifierAST* {
  auto node = new (arena) NestedNamespaceSpecifierAST();
  node->inlineLoc = inlineLoc;
  node->identifierLoc = identifierLoc;
  node->scopeLoc = scopeLoc;
  node->identifier = identifier;
  node->isInline = isInline;
  return node;
}

auto NestedNamespaceSpecifierAST::create(Arena* arena,
                                         const Identifier* identifier,
                                         bool isInline)
    -> NestedNamespaceSpecifierAST* {
  auto node = new (arena) NestedNamespaceSpecifierAST();
  node->identifier = identifier;
  node->isInline = isInline;
  return node;
}

auto LabeledStatementAST::clone(Arena* arena) -> LabeledStatementAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->colonLoc = colonLoc;
  node->identifier = identifier;

  return node;
}

auto LabeledStatementAST::create(Arena* arena) -> LabeledStatementAST* {
  auto node = new (arena) LabeledStatementAST();
  return node;
}

auto LabeledStatementAST::create(Arena* arena, SourceLocation identifierLoc,
                                 SourceLocation colonLoc,
                                 const Identifier* identifier)
    -> LabeledStatementAST* {
  auto node = new (arena) LabeledStatementAST();
  node->identifierLoc = identifierLoc;
  node->colonLoc = colonLoc;
  node->identifier = identifier;
  return node;
}

auto LabeledStatementAST::create(Arena* arena, const Identifier* identifier)
    -> LabeledStatementAST* {
  auto node = new (arena) LabeledStatementAST();
  node->identifier = identifier;
  return node;
}

auto CaseStatementAST::clone(Arena* arena) -> CaseStatementAST* {
  auto node = create(arena);

  node->caseLoc = caseLoc;

  if (expression) node->expression = expression->clone(arena);

  node->colonLoc = colonLoc;
  node->caseValue = caseValue;

  return node;
}

auto CaseStatementAST::create(Arena* arena) -> CaseStatementAST* {
  auto node = new (arena) CaseStatementAST();
  return node;
}

auto CaseStatementAST::create(Arena* arena, SourceLocation caseLoc,
                              ExpressionAST* expression,
                              SourceLocation colonLoc, std::int64_t caseValue)
    -> CaseStatementAST* {
  auto node = new (arena) CaseStatementAST();
  node->caseLoc = caseLoc;
  node->expression = expression;
  node->colonLoc = colonLoc;
  node->caseValue = caseValue;
  return node;
}

auto CaseStatementAST::create(Arena* arena, ExpressionAST* expression,
                              std::int64_t caseValue) -> CaseStatementAST* {
  auto node = new (arena) CaseStatementAST();
  node->expression = expression;
  node->caseValue = caseValue;
  return node;
}

auto DefaultStatementAST::clone(Arena* arena) -> DefaultStatementAST* {
  auto node = create(arena);

  node->defaultLoc = defaultLoc;
  node->colonLoc = colonLoc;

  return node;
}

auto DefaultStatementAST::create(Arena* arena) -> DefaultStatementAST* {
  auto node = new (arena) DefaultStatementAST();
  return node;
}

auto DefaultStatementAST::create(Arena* arena, SourceLocation defaultLoc,
                                 SourceLocation colonLoc)
    -> DefaultStatementAST* {
  auto node = new (arena) DefaultStatementAST();
  node->defaultLoc = defaultLoc;
  node->colonLoc = colonLoc;
  return node;
}

auto ExpressionStatementAST::clone(Arena* arena) -> ExpressionStatementAST* {
  auto node = create(arena);

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto ExpressionStatementAST::create(Arena* arena) -> ExpressionStatementAST* {
  auto node = new (arena) ExpressionStatementAST();
  return node;
}

auto ExpressionStatementAST::create(Arena* arena, ExpressionAST* expression,
                                    SourceLocation semicolonLoc)
    -> ExpressionStatementAST* {
  auto node = new (arena) ExpressionStatementAST();
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ExpressionStatementAST::create(Arena* arena, ExpressionAST* expression)
    -> ExpressionStatementAST* {
  auto node = new (arena) ExpressionStatementAST();
  node->expression = expression;
  return node;
}

auto CompoundStatementAST::clone(Arena* arena) -> CompoundStatementAST* {
  auto node = create(arena);

  node->lbraceLoc = lbraceLoc;

  if (statementList) {
    auto it = &node->statementList;
    for (auto node : ListView{statementList}) {
      *it = make_list_node<StatementAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;
  node->symbol = symbol;

  return node;
}

auto CompoundStatementAST::create(Arena* arena) -> CompoundStatementAST* {
  auto node = new (arena) CompoundStatementAST();
  return node;
}

auto CompoundStatementAST::create(Arena* arena, SourceLocation lbraceLoc,
                                  List<StatementAST*>* statementList,
                                  SourceLocation rbraceLoc, BlockSymbol* symbol)
    -> CompoundStatementAST* {
  auto node = new (arena) CompoundStatementAST();
  node->lbraceLoc = lbraceLoc;
  node->statementList = statementList;
  node->rbraceLoc = rbraceLoc;
  node->symbol = symbol;
  return node;
}

auto CompoundStatementAST::create(Arena* arena,
                                  List<StatementAST*>* statementList,
                                  BlockSymbol* symbol)
    -> CompoundStatementAST* {
  auto node = new (arena) CompoundStatementAST();
  node->statementList = statementList;
  node->symbol = symbol;
  return node;
}

auto IfStatementAST::clone(Arena* arena) -> IfStatementAST* {
  auto node = create(arena);

  node->ifLoc = ifLoc;
  node->constexprLoc = constexprLoc;
  node->lparenLoc = lparenLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  if (condition) node->condition = condition->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->elseLoc = elseLoc;

  if (elseStatement) node->elseStatement = elseStatement->clone(arena);

  node->symbol = symbol;

  return node;
}

auto IfStatementAST::create(Arena* arena) -> IfStatementAST* {
  auto node = new (arena) IfStatementAST();
  return node;
}

auto IfStatementAST::create(Arena* arena, SourceLocation ifLoc,
                            SourceLocation constexprLoc,
                            SourceLocation lparenLoc, StatementAST* initializer,
                            ExpressionAST* condition, SourceLocation rparenLoc,
                            StatementAST* statement, SourceLocation elseLoc,
                            StatementAST* elseStatement, BlockSymbol* symbol)
    -> IfStatementAST* {
  auto node = new (arena) IfStatementAST();
  node->ifLoc = ifLoc;
  node->constexprLoc = constexprLoc;
  node->lparenLoc = lparenLoc;
  node->initializer = initializer;
  node->condition = condition;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  node->elseLoc = elseLoc;
  node->elseStatement = elseStatement;
  node->symbol = symbol;
  return node;
}

auto IfStatementAST::create(Arena* arena, StatementAST* initializer,
                            ExpressionAST* condition, StatementAST* statement,
                            StatementAST* elseStatement, BlockSymbol* symbol)
    -> IfStatementAST* {
  auto node = new (arena) IfStatementAST();
  node->initializer = initializer;
  node->condition = condition;
  node->statement = statement;
  node->elseStatement = elseStatement;
  node->symbol = symbol;
  return node;
}

auto ConstevalIfStatementAST::clone(Arena* arena) -> ConstevalIfStatementAST* {
  auto node = create(arena);

  node->ifLoc = ifLoc;
  node->exclaimLoc = exclaimLoc;
  node->constvalLoc = constvalLoc;

  if (statement) node->statement = statement->clone(arena);

  node->elseLoc = elseLoc;

  if (elseStatement) node->elseStatement = elseStatement->clone(arena);

  node->isNot = isNot;

  return node;
}

auto ConstevalIfStatementAST::create(Arena* arena) -> ConstevalIfStatementAST* {
  auto node = new (arena) ConstevalIfStatementAST();
  return node;
}

auto ConstevalIfStatementAST::create(
    Arena* arena, SourceLocation ifLoc, SourceLocation exclaimLoc,
    SourceLocation constvalLoc, StatementAST* statement, SourceLocation elseLoc,
    StatementAST* elseStatement, bool isNot) -> ConstevalIfStatementAST* {
  auto node = new (arena) ConstevalIfStatementAST();
  node->ifLoc = ifLoc;
  node->exclaimLoc = exclaimLoc;
  node->constvalLoc = constvalLoc;
  node->statement = statement;
  node->elseLoc = elseLoc;
  node->elseStatement = elseStatement;
  node->isNot = isNot;
  return node;
}

auto ConstevalIfStatementAST::create(Arena* arena, StatementAST* statement,
                                     StatementAST* elseStatement, bool isNot)
    -> ConstevalIfStatementAST* {
  auto node = new (arena) ConstevalIfStatementAST();
  node->statement = statement;
  node->elseStatement = elseStatement;
  node->isNot = isNot;
  return node;
}

auto SwitchStatementAST::clone(Arena* arena) -> SwitchStatementAST* {
  auto node = create(arena);

  node->switchLoc = switchLoc;
  node->lparenLoc = lparenLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  if (condition) node->condition = condition->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->symbol = symbol;

  return node;
}

auto SwitchStatementAST::create(Arena* arena) -> SwitchStatementAST* {
  auto node = new (arena) SwitchStatementAST();
  return node;
}

auto SwitchStatementAST::create(Arena* arena, SourceLocation switchLoc,
                                SourceLocation lparenLoc,
                                StatementAST* initializer,
                                ExpressionAST* condition,
                                SourceLocation rparenLoc,
                                StatementAST* statement, BlockSymbol* symbol)
    -> SwitchStatementAST* {
  auto node = new (arena) SwitchStatementAST();
  node->switchLoc = switchLoc;
  node->lparenLoc = lparenLoc;
  node->initializer = initializer;
  node->condition = condition;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto SwitchStatementAST::create(Arena* arena, StatementAST* initializer,
                                ExpressionAST* condition,
                                StatementAST* statement, BlockSymbol* symbol)
    -> SwitchStatementAST* {
  auto node = new (arena) SwitchStatementAST();
  node->initializer = initializer;
  node->condition = condition;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto WhileStatementAST::clone(Arena* arena) -> WhileStatementAST* {
  auto node = create(arena);

  node->whileLoc = whileLoc;
  node->lparenLoc = lparenLoc;

  if (condition) node->condition = condition->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->symbol = symbol;

  return node;
}

auto WhileStatementAST::create(Arena* arena) -> WhileStatementAST* {
  auto node = new (arena) WhileStatementAST();
  return node;
}

auto WhileStatementAST::create(Arena* arena, SourceLocation whileLoc,
                               SourceLocation lparenLoc,
                               ExpressionAST* condition,
                               SourceLocation rparenLoc,
                               StatementAST* statement, BlockSymbol* symbol)
    -> WhileStatementAST* {
  auto node = new (arena) WhileStatementAST();
  node->whileLoc = whileLoc;
  node->lparenLoc = lparenLoc;
  node->condition = condition;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto WhileStatementAST::create(Arena* arena, ExpressionAST* condition,
                               StatementAST* statement, BlockSymbol* symbol)
    -> WhileStatementAST* {
  auto node = new (arena) WhileStatementAST();
  node->condition = condition;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto DoStatementAST::clone(Arena* arena) -> DoStatementAST* {
  auto node = create(arena);

  node->doLoc = doLoc;

  if (statement) node->statement = statement->clone(arena);

  node->whileLoc = whileLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;

  return node;
}

auto DoStatementAST::create(Arena* arena) -> DoStatementAST* {
  auto node = new (arena) DoStatementAST();
  return node;
}

auto DoStatementAST::create(Arena* arena, SourceLocation doLoc,
                            StatementAST* statement, SourceLocation whileLoc,
                            SourceLocation lparenLoc, ExpressionAST* expression,
                            SourceLocation rparenLoc,
                            SourceLocation semicolonLoc) -> DoStatementAST* {
  auto node = new (arena) DoStatementAST();
  node->doLoc = doLoc;
  node->statement = statement;
  node->whileLoc = whileLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto DoStatementAST::create(Arena* arena, StatementAST* statement,
                            ExpressionAST* expression) -> DoStatementAST* {
  auto node = new (arena) DoStatementAST();
  node->statement = statement;
  node->expression = expression;
  return node;
}

auto ForRangeStatementAST::clone(Arena* arena) -> ForRangeStatementAST* {
  auto node = create(arena);

  node->forLoc = forLoc;
  node->lparenLoc = lparenLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  if (rangeDeclaration) node->rangeDeclaration = rangeDeclaration->clone(arena);

  node->colonLoc = colonLoc;

  if (rangeInitializer) node->rangeInitializer = rangeInitializer->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->symbol = symbol;

  return node;
}

auto ForRangeStatementAST::create(Arena* arena) -> ForRangeStatementAST* {
  auto node = new (arena) ForRangeStatementAST();
  return node;
}

auto ForRangeStatementAST::create(
    Arena* arena, SourceLocation forLoc, SourceLocation lparenLoc,
    StatementAST* initializer, DeclarationAST* rangeDeclaration,
    SourceLocation colonLoc, ExpressionAST* rangeInitializer,
    SourceLocation rparenLoc, StatementAST* statement, BlockSymbol* symbol)
    -> ForRangeStatementAST* {
  auto node = new (arena) ForRangeStatementAST();
  node->forLoc = forLoc;
  node->lparenLoc = lparenLoc;
  node->initializer = initializer;
  node->rangeDeclaration = rangeDeclaration;
  node->colonLoc = colonLoc;
  node->rangeInitializer = rangeInitializer;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto ForRangeStatementAST::create(Arena* arena, StatementAST* initializer,
                                  DeclarationAST* rangeDeclaration,
                                  ExpressionAST* rangeInitializer,
                                  StatementAST* statement, BlockSymbol* symbol)
    -> ForRangeStatementAST* {
  auto node = new (arena) ForRangeStatementAST();
  node->initializer = initializer;
  node->rangeDeclaration = rangeDeclaration;
  node->rangeInitializer = rangeInitializer;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto ForStatementAST::clone(Arena* arena) -> ForStatementAST* {
  auto node = create(arena);

  node->forLoc = forLoc;
  node->lparenLoc = lparenLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  if (condition) node->condition = condition->clone(arena);

  node->semicolonLoc = semicolonLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->symbol = symbol;

  return node;
}

auto ForStatementAST::create(Arena* arena) -> ForStatementAST* {
  auto node = new (arena) ForStatementAST();
  return node;
}

auto ForStatementAST::create(Arena* arena, SourceLocation forLoc,
                             SourceLocation lparenLoc,
                             StatementAST* initializer,
                             ExpressionAST* condition,
                             SourceLocation semicolonLoc,
                             ExpressionAST* expression,
                             SourceLocation rparenLoc, StatementAST* statement,
                             BlockSymbol* symbol) -> ForStatementAST* {
  auto node = new (arena) ForStatementAST();
  node->forLoc = forLoc;
  node->lparenLoc = lparenLoc;
  node->initializer = initializer;
  node->condition = condition;
  node->semicolonLoc = semicolonLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto ForStatementAST::create(Arena* arena, StatementAST* initializer,
                             ExpressionAST* condition,
                             ExpressionAST* expression, StatementAST* statement,
                             BlockSymbol* symbol) -> ForStatementAST* {
  auto node = new (arena) ForStatementAST();
  node->initializer = initializer;
  node->condition = condition;
  node->expression = expression;
  node->statement = statement;
  node->symbol = symbol;
  return node;
}

auto BreakStatementAST::clone(Arena* arena) -> BreakStatementAST* {
  auto node = create(arena);

  node->breakLoc = breakLoc;
  node->semicolonLoc = semicolonLoc;

  return node;
}

auto BreakStatementAST::create(Arena* arena) -> BreakStatementAST* {
  auto node = new (arena) BreakStatementAST();
  return node;
}

auto BreakStatementAST::create(Arena* arena, SourceLocation breakLoc,
                               SourceLocation semicolonLoc)
    -> BreakStatementAST* {
  auto node = new (arena) BreakStatementAST();
  node->breakLoc = breakLoc;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ContinueStatementAST::clone(Arena* arena) -> ContinueStatementAST* {
  auto node = create(arena);

  node->continueLoc = continueLoc;
  node->semicolonLoc = semicolonLoc;

  return node;
}

auto ContinueStatementAST::create(Arena* arena) -> ContinueStatementAST* {
  auto node = new (arena) ContinueStatementAST();
  return node;
}

auto ContinueStatementAST::create(Arena* arena, SourceLocation continueLoc,
                                  SourceLocation semicolonLoc)
    -> ContinueStatementAST* {
  auto node = new (arena) ContinueStatementAST();
  node->continueLoc = continueLoc;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ReturnStatementAST::clone(Arena* arena) -> ReturnStatementAST* {
  auto node = create(arena);

  node->returnLoc = returnLoc;

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto ReturnStatementAST::create(Arena* arena) -> ReturnStatementAST* {
  auto node = new (arena) ReturnStatementAST();
  return node;
}

auto ReturnStatementAST::create(Arena* arena, SourceLocation returnLoc,
                                ExpressionAST* expression,
                                SourceLocation semicolonLoc)
    -> ReturnStatementAST* {
  auto node = new (arena) ReturnStatementAST();
  node->returnLoc = returnLoc;
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto ReturnStatementAST::create(Arena* arena, ExpressionAST* expression)
    -> ReturnStatementAST* {
  auto node = new (arena) ReturnStatementAST();
  node->expression = expression;
  return node;
}

auto CoroutineReturnStatementAST::clone(Arena* arena)
    -> CoroutineReturnStatementAST* {
  auto node = create(arena);

  node->coreturnLoc = coreturnLoc;

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto CoroutineReturnStatementAST::create(Arena* arena)
    -> CoroutineReturnStatementAST* {
  auto node = new (arena) CoroutineReturnStatementAST();
  return node;
}

auto CoroutineReturnStatementAST::create(Arena* arena,
                                         SourceLocation coreturnLoc,
                                         ExpressionAST* expression,
                                         SourceLocation semicolonLoc)
    -> CoroutineReturnStatementAST* {
  auto node = new (arena) CoroutineReturnStatementAST();
  node->coreturnLoc = coreturnLoc;
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto CoroutineReturnStatementAST::create(Arena* arena,
                                         ExpressionAST* expression)
    -> CoroutineReturnStatementAST* {
  auto node = new (arena) CoroutineReturnStatementAST();
  node->expression = expression;
  return node;
}

auto GotoStatementAST::clone(Arena* arena) -> GotoStatementAST* {
  auto node = create(arena);

  node->gotoLoc = gotoLoc;
  node->starLoc = starLoc;
  node->identifierLoc = identifierLoc;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->isIndirect = isIndirect;

  return node;
}

auto GotoStatementAST::create(Arena* arena) -> GotoStatementAST* {
  auto node = new (arena) GotoStatementAST();
  return node;
}

auto GotoStatementAST::create(Arena* arena, SourceLocation gotoLoc,
                              SourceLocation starLoc,
                              SourceLocation identifierLoc,
                              SourceLocation semicolonLoc,
                              const Identifier* identifier, bool isIndirect)
    -> GotoStatementAST* {
  auto node = new (arena) GotoStatementAST();
  node->gotoLoc = gotoLoc;
  node->starLoc = starLoc;
  node->identifierLoc = identifierLoc;
  node->semicolonLoc = semicolonLoc;
  node->identifier = identifier;
  node->isIndirect = isIndirect;
  return node;
}

auto GotoStatementAST::create(Arena* arena, const Identifier* identifier,
                              bool isIndirect) -> GotoStatementAST* {
  auto node = new (arena) GotoStatementAST();
  node->identifier = identifier;
  node->isIndirect = isIndirect;
  return node;
}

auto DeclarationStatementAST::clone(Arena* arena) -> DeclarationStatementAST* {
  auto node = create(arena);

  if (declaration) node->declaration = declaration->clone(arena);

  return node;
}

auto DeclarationStatementAST::create(Arena* arena) -> DeclarationStatementAST* {
  auto node = new (arena) DeclarationStatementAST();
  return node;
}

auto DeclarationStatementAST::create(Arena* arena, DeclarationAST* declaration)
    -> DeclarationStatementAST* {
  auto node = new (arena) DeclarationStatementAST();
  node->declaration = declaration;
  return node;
}

auto TryBlockStatementAST::clone(Arena* arena) -> TryBlockStatementAST* {
  auto node = create(arena);

  node->tryLoc = tryLoc;

  if (statement) node->statement = statement->clone(arena);

  if (handlerList) {
    auto it = &node->handlerList;
    for (auto node : ListView{handlerList}) {
      *it = make_list_node<HandlerAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto TryBlockStatementAST::create(Arena* arena) -> TryBlockStatementAST* {
  auto node = new (arena) TryBlockStatementAST();
  return node;
}

auto TryBlockStatementAST::create(Arena* arena, SourceLocation tryLoc,
                                  CompoundStatementAST* statement,
                                  List<HandlerAST*>* handlerList)
    -> TryBlockStatementAST* {
  auto node = new (arena) TryBlockStatementAST();
  node->tryLoc = tryLoc;
  node->statement = statement;
  node->handlerList = handlerList;
  return node;
}

auto TryBlockStatementAST::create(Arena* arena, CompoundStatementAST* statement,
                                  List<HandlerAST*>* handlerList)
    -> TryBlockStatementAST* {
  auto node = new (arena) TryBlockStatementAST();
  node->statement = statement;
  node->handlerList = handlerList;
  return node;
}

auto CharLiteralExpressionAST::clone(Arena* arena)
    -> CharLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto CharLiteralExpressionAST::create(Arena* arena)
    -> CharLiteralExpressionAST* {
  auto node = new (arena) CharLiteralExpressionAST();
  return node;
}

auto CharLiteralExpressionAST::create(Arena* arena, SourceLocation literalLoc,
                                      const CharLiteral* literal,
                                      ValueCategory valueCategory,
                                      const Type* type)
    -> CharLiteralExpressionAST* {
  auto node = new (arena) CharLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CharLiteralExpressionAST::create(Arena* arena, const CharLiteral* literal,
                                      ValueCategory valueCategory,
                                      const Type* type)
    -> CharLiteralExpressionAST* {
  auto node = new (arena) CharLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BoolLiteralExpressionAST::clone(Arena* arena)
    -> BoolLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->isTrue = isTrue;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BoolLiteralExpressionAST::create(Arena* arena)
    -> BoolLiteralExpressionAST* {
  auto node = new (arena) BoolLiteralExpressionAST();
  return node;
}

auto BoolLiteralExpressionAST::create(Arena* arena, SourceLocation literalLoc,
                                      bool isTrue, ValueCategory valueCategory,
                                      const Type* type)
    -> BoolLiteralExpressionAST* {
  auto node = new (arena) BoolLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->isTrue = isTrue;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BoolLiteralExpressionAST::create(Arena* arena, bool isTrue,
                                      ValueCategory valueCategory,
                                      const Type* type)
    -> BoolLiteralExpressionAST* {
  auto node = new (arena) BoolLiteralExpressionAST();
  node->isTrue = isTrue;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto IntLiteralExpressionAST::clone(Arena* arena) -> IntLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto IntLiteralExpressionAST::create(Arena* arena) -> IntLiteralExpressionAST* {
  auto node = new (arena) IntLiteralExpressionAST();
  return node;
}

auto IntLiteralExpressionAST::create(Arena* arena, SourceLocation literalLoc,
                                     const IntegerLiteral* literal,
                                     ValueCategory valueCategory,
                                     const Type* type)
    -> IntLiteralExpressionAST* {
  auto node = new (arena) IntLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto IntLiteralExpressionAST::create(Arena* arena,
                                     const IntegerLiteral* literal,
                                     ValueCategory valueCategory,
                                     const Type* type)
    -> IntLiteralExpressionAST* {
  auto node = new (arena) IntLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto FloatLiteralExpressionAST::clone(Arena* arena)
    -> FloatLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto FloatLiteralExpressionAST::create(Arena* arena)
    -> FloatLiteralExpressionAST* {
  auto node = new (arena) FloatLiteralExpressionAST();
  return node;
}

auto FloatLiteralExpressionAST::create(Arena* arena, SourceLocation literalLoc,
                                       const FloatLiteral* literal,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> FloatLiteralExpressionAST* {
  auto node = new (arena) FloatLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto FloatLiteralExpressionAST::create(Arena* arena,
                                       const FloatLiteral* literal,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> FloatLiteralExpressionAST* {
  auto node = new (arena) FloatLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NullptrLiteralExpressionAST::clone(Arena* arena)
    -> NullptrLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NullptrLiteralExpressionAST::create(Arena* arena)
    -> NullptrLiteralExpressionAST* {
  auto node = new (arena) NullptrLiteralExpressionAST();
  return node;
}

auto NullptrLiteralExpressionAST::create(Arena* arena,
                                         SourceLocation literalLoc,
                                         TokenKind literal,
                                         ValueCategory valueCategory,
                                         const Type* type)
    -> NullptrLiteralExpressionAST* {
  auto node = new (arena) NullptrLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NullptrLiteralExpressionAST::create(Arena* arena, TokenKind literal,
                                         ValueCategory valueCategory,
                                         const Type* type)
    -> NullptrLiteralExpressionAST* {
  auto node = new (arena) NullptrLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto StringLiteralExpressionAST::clone(Arena* arena)
    -> StringLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto StringLiteralExpressionAST::create(Arena* arena)
    -> StringLiteralExpressionAST* {
  auto node = new (arena) StringLiteralExpressionAST();
  return node;
}

auto StringLiteralExpressionAST::create(Arena* arena, SourceLocation literalLoc,
                                        const StringLiteral* literal,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> StringLiteralExpressionAST* {
  auto node = new (arena) StringLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto StringLiteralExpressionAST::create(Arena* arena,
                                        const StringLiteral* literal,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> StringLiteralExpressionAST* {
  auto node = new (arena) StringLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto UserDefinedStringLiteralExpressionAST::clone(Arena* arena)
    -> UserDefinedStringLiteralExpressionAST* {
  auto node = create(arena);

  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto UserDefinedStringLiteralExpressionAST::create(Arena* arena)
    -> UserDefinedStringLiteralExpressionAST* {
  auto node = new (arena) UserDefinedStringLiteralExpressionAST();
  return node;
}

auto UserDefinedStringLiteralExpressionAST::create(Arena* arena,
                                                   SourceLocation literalLoc,
                                                   const StringLiteral* literal,
                                                   ValueCategory valueCategory,
                                                   const Type* type)
    -> UserDefinedStringLiteralExpressionAST* {
  auto node = new (arena) UserDefinedStringLiteralExpressionAST();
  node->literalLoc = literalLoc;
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto UserDefinedStringLiteralExpressionAST::create(Arena* arena,
                                                   const StringLiteral* literal,
                                                   ValueCategory valueCategory,
                                                   const Type* type)
    -> UserDefinedStringLiteralExpressionAST* {
  auto node = new (arena) UserDefinedStringLiteralExpressionAST();
  node->literal = literal;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ObjectLiteralExpressionAST::clone(Arena* arena)
    -> ObjectLiteralExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;

  if (bracedInitList) node->bracedInitList = bracedInitList->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ObjectLiteralExpressionAST::create(Arena* arena)
    -> ObjectLiteralExpressionAST* {
  auto node = new (arena) ObjectLiteralExpressionAST();
  return node;
}

auto ObjectLiteralExpressionAST::create(Arena* arena, SourceLocation lparenLoc,
                                        TypeIdAST* typeId,
                                        SourceLocation rparenLoc,
                                        BracedInitListAST* bracedInitList,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> ObjectLiteralExpressionAST* {
  auto node = new (arena) ObjectLiteralExpressionAST();
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->bracedInitList = bracedInitList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ObjectLiteralExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                        BracedInitListAST* bracedInitList,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> ObjectLiteralExpressionAST* {
  auto node = new (arena) ObjectLiteralExpressionAST();
  node->typeId = typeId;
  node->bracedInitList = bracedInitList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ThisExpressionAST::clone(Arena* arena) -> ThisExpressionAST* {
  auto node = create(arena);

  node->thisLoc = thisLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ThisExpressionAST::create(Arena* arena) -> ThisExpressionAST* {
  auto node = new (arena) ThisExpressionAST();
  return node;
}

auto ThisExpressionAST::create(Arena* arena, SourceLocation thisLoc,
                               ValueCategory valueCategory, const Type* type)
    -> ThisExpressionAST* {
  auto node = new (arena) ThisExpressionAST();
  node->thisLoc = thisLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ThisExpressionAST::create(Arena* arena, ValueCategory valueCategory,
                               const Type* type) -> ThisExpressionAST* {
  auto node = new (arena) ThisExpressionAST();
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto GenericSelectionExpressionAST::clone(Arena* arena)
    -> GenericSelectionExpressionAST* {
  auto node = create(arena);

  node->genericLoc = genericLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->commaLoc = commaLoc;

  if (genericAssociationList) {
    auto it = &node->genericAssociationList;
    for (auto node : ListView{genericAssociationList}) {
      *it = make_list_node<GenericAssociationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->matchedAssocIndex = matchedAssocIndex;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto GenericSelectionExpressionAST::create(Arena* arena)
    -> GenericSelectionExpressionAST* {
  auto node = new (arena) GenericSelectionExpressionAST();
  return node;
}

auto GenericSelectionExpressionAST::create(
    Arena* arena, SourceLocation genericLoc, SourceLocation lparenLoc,
    ExpressionAST* expression, SourceLocation commaLoc,
    List<GenericAssociationAST*>* genericAssociationList,
    SourceLocation rparenLoc, int matchedAssocIndex,
    ValueCategory valueCategory, const Type* type)
    -> GenericSelectionExpressionAST* {
  auto node = new (arena) GenericSelectionExpressionAST();
  node->genericLoc = genericLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->commaLoc = commaLoc;
  node->genericAssociationList = genericAssociationList;
  node->rparenLoc = rparenLoc;
  node->matchedAssocIndex = matchedAssocIndex;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto GenericSelectionExpressionAST::create(
    Arena* arena, ExpressionAST* expression,
    List<GenericAssociationAST*>* genericAssociationList, int matchedAssocIndex,
    ValueCategory valueCategory, const Type* type)
    -> GenericSelectionExpressionAST* {
  auto node = new (arena) GenericSelectionExpressionAST();
  node->expression = expression;
  node->genericAssociationList = genericAssociationList;
  node->matchedAssocIndex = matchedAssocIndex;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NestedStatementExpressionAST::clone(Arena* arena)
    -> NestedStatementExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (statement) node->statement = statement->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NestedStatementExpressionAST::create(Arena* arena)
    -> NestedStatementExpressionAST* {
  auto node = new (arena) NestedStatementExpressionAST();
  return node;
}

auto NestedStatementExpressionAST::create(
    Arena* arena, SourceLocation lparenLoc, CompoundStatementAST* statement,
    SourceLocation rparenLoc, ValueCategory valueCategory, const Type* type)
    -> NestedStatementExpressionAST* {
  auto node = new (arena) NestedStatementExpressionAST();
  node->lparenLoc = lparenLoc;
  node->statement = statement;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NestedStatementExpressionAST::create(Arena* arena,
                                          CompoundStatementAST* statement,
                                          ValueCategory valueCategory,
                                          const Type* type)
    -> NestedStatementExpressionAST* {
  auto node = new (arena) NestedStatementExpressionAST();
  node->statement = statement;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NestedExpressionAST::clone(Arena* arena) -> NestedExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NestedExpressionAST::create(Arena* arena) -> NestedExpressionAST* {
  auto node = new (arena) NestedExpressionAST();
  return node;
}

auto NestedExpressionAST::create(Arena* arena, SourceLocation lparenLoc,
                                 ExpressionAST* expression,
                                 SourceLocation rparenLoc,
                                 ValueCategory valueCategory, const Type* type)
    -> NestedExpressionAST* {
  auto node = new (arena) NestedExpressionAST();
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NestedExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> NestedExpressionAST* {
  auto node = new (arena) NestedExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto IdExpressionAST::clone(Arena* arena) -> IdExpressionAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->symbol = symbol;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto IdExpressionAST::create(Arena* arena) -> IdExpressionAST* {
  auto node = new (arena) IdExpressionAST();
  return node;
}

auto IdExpressionAST::create(Arena* arena,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             SourceLocation templateLoc,
                             UnqualifiedIdAST* unqualifiedId, Symbol* symbol,
                             bool isTemplateIntroduced,
                             ValueCategory valueCategory, const Type* type)
    -> IdExpressionAST* {
  auto node = new (arena) IdExpressionAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->symbol = symbol;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto IdExpressionAST::create(Arena* arena,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId, Symbol* symbol,
                             bool isTemplateIntroduced,
                             ValueCategory valueCategory, const Type* type)
    -> IdExpressionAST* {
  auto node = new (arena) IdExpressionAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->symbol = symbol;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LambdaExpressionAST::clone(Arena* arena) -> LambdaExpressionAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;
  node->captureDefaultLoc = captureDefaultLoc;

  if (captureList) {
    auto it = &node->captureList;
    for (auto node : ListView{captureList}) {
      *it = make_list_node<LambdaCaptureAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbracketLoc = rbracketLoc;
  node->lessLoc = lessLoc;

  if (templateParameterList) {
    auto it = &node->templateParameterList;
    for (auto node : ListView{templateParameterList}) {
      *it = make_list_node<TemplateParameterAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;

  if (templateRequiresClause)
    node->templateRequiresClause = templateRequiresClause->clone(arena);

  if (expressionAttributeList) {
    auto it = &node->expressionAttributeList;
    for (auto node : ListView{expressionAttributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->lparenLoc = lparenLoc;

  if (parameterDeclarationClause)
    node->parameterDeclarationClause = parameterDeclarationClause->clone(arena);

  node->rparenLoc = rparenLoc;

  if (gnuAtributeList) {
    auto it = &node->gnuAtributeList;
    for (auto node : ListView{gnuAtributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (lambdaSpecifierList) {
    auto it = &node->lambdaSpecifierList;
    for (auto node : ListView{lambdaSpecifierList}) {
      *it = make_list_node<LambdaSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (exceptionSpecifier)
    node->exceptionSpecifier = exceptionSpecifier->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (trailingReturnType)
    node->trailingReturnType = trailingReturnType->clone(arena);

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  if (statement) node->statement = statement->clone(arena);

  node->captureDefault = captureDefault;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto LambdaExpressionAST::create(Arena* arena) -> LambdaExpressionAST* {
  auto node = new (arena) LambdaExpressionAST();
  return node;
}

auto LambdaExpressionAST::create(
    Arena* arena, SourceLocation lbracketLoc, SourceLocation captureDefaultLoc,
    List<LambdaCaptureAST*>* captureList, SourceLocation rbracketLoc,
    SourceLocation lessLoc, List<TemplateParameterAST*>* templateParameterList,
    SourceLocation greaterLoc, RequiresClauseAST* templateRequiresClause,
    List<AttributeSpecifierAST*>* expressionAttributeList,
    SourceLocation lparenLoc,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    SourceLocation rparenLoc, List<AttributeSpecifierAST*>* gnuAtributeList,
    List<LambdaSpecifierAST*>* lambdaSpecifierList,
    ExceptionSpecifierAST* exceptionSpecifier,
    List<AttributeSpecifierAST*>* attributeList,
    TrailingReturnTypeAST* trailingReturnType,
    RequiresClauseAST* requiresClause, CompoundStatementAST* statement,
    TokenKind captureDefault, LambdaSymbol* symbol, ValueCategory valueCategory,
    const Type* type) -> LambdaExpressionAST* {
  auto node = new (arena) LambdaExpressionAST();
  node->lbracketLoc = lbracketLoc;
  node->captureDefaultLoc = captureDefaultLoc;
  node->captureList = captureList;
  node->rbracketLoc = rbracketLoc;
  node->lessLoc = lessLoc;
  node->templateParameterList = templateParameterList;
  node->greaterLoc = greaterLoc;
  node->templateRequiresClause = templateRequiresClause;
  node->expressionAttributeList = expressionAttributeList;
  node->lparenLoc = lparenLoc;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->rparenLoc = rparenLoc;
  node->gnuAtributeList = gnuAtributeList;
  node->lambdaSpecifierList = lambdaSpecifierList;
  node->exceptionSpecifier = exceptionSpecifier;
  node->attributeList = attributeList;
  node->trailingReturnType = trailingReturnType;
  node->requiresClause = requiresClause;
  node->statement = statement;
  node->captureDefault = captureDefault;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LambdaExpressionAST::create(
    Arena* arena, List<LambdaCaptureAST*>* captureList,
    List<TemplateParameterAST*>* templateParameterList,
    RequiresClauseAST* templateRequiresClause,
    List<AttributeSpecifierAST*>* expressionAttributeList,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    List<AttributeSpecifierAST*>* gnuAtributeList,
    List<LambdaSpecifierAST*>* lambdaSpecifierList,
    ExceptionSpecifierAST* exceptionSpecifier,
    List<AttributeSpecifierAST*>* attributeList,
    TrailingReturnTypeAST* trailingReturnType,
    RequiresClauseAST* requiresClause, CompoundStatementAST* statement,
    TokenKind captureDefault, LambdaSymbol* symbol, ValueCategory valueCategory,
    const Type* type) -> LambdaExpressionAST* {
  auto node = new (arena) LambdaExpressionAST();
  node->captureList = captureList;
  node->templateParameterList = templateParameterList;
  node->templateRequiresClause = templateRequiresClause;
  node->expressionAttributeList = expressionAttributeList;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->gnuAtributeList = gnuAtributeList;
  node->lambdaSpecifierList = lambdaSpecifierList;
  node->exceptionSpecifier = exceptionSpecifier;
  node->attributeList = attributeList;
  node->trailingReturnType = trailingReturnType;
  node->requiresClause = requiresClause;
  node->statement = statement;
  node->captureDefault = captureDefault;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto FoldExpressionAST::clone(Arena* arena) -> FoldExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (leftExpression) node->leftExpression = leftExpression->clone(arena);

  node->opLoc = opLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->foldOpLoc = foldOpLoc;

  if (rightExpression) node->rightExpression = rightExpression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->op = op;
  node->foldOp = foldOp;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto FoldExpressionAST::create(Arena* arena) -> FoldExpressionAST* {
  auto node = new (arena) FoldExpressionAST();
  return node;
}

auto FoldExpressionAST::create(Arena* arena, SourceLocation lparenLoc,
                               ExpressionAST* leftExpression,
                               SourceLocation opLoc, SourceLocation ellipsisLoc,
                               SourceLocation foldOpLoc,
                               ExpressionAST* rightExpression,
                               SourceLocation rparenLoc, TokenKind op,
                               TokenKind foldOp, ValueCategory valueCategory,
                               const Type* type) -> FoldExpressionAST* {
  auto node = new (arena) FoldExpressionAST();
  node->lparenLoc = lparenLoc;
  node->leftExpression = leftExpression;
  node->opLoc = opLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->foldOpLoc = foldOpLoc;
  node->rightExpression = rightExpression;
  node->rparenLoc = rparenLoc;
  node->op = op;
  node->foldOp = foldOp;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto FoldExpressionAST::create(Arena* arena, ExpressionAST* leftExpression,
                               ExpressionAST* rightExpression, TokenKind op,
                               TokenKind foldOp, ValueCategory valueCategory,
                               const Type* type) -> FoldExpressionAST* {
  auto node = new (arena) FoldExpressionAST();
  node->leftExpression = leftExpression;
  node->rightExpression = rightExpression;
  node->op = op;
  node->foldOp = foldOp;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto RightFoldExpressionAST::clone(Arena* arena) -> RightFoldExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->opLoc = opLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto RightFoldExpressionAST::create(Arena* arena) -> RightFoldExpressionAST* {
  auto node = new (arena) RightFoldExpressionAST();
  return node;
}

auto RightFoldExpressionAST::create(
    Arena* arena, SourceLocation lparenLoc, ExpressionAST* expression,
    SourceLocation opLoc, SourceLocation ellipsisLoc, SourceLocation rparenLoc,
    TokenKind op, ValueCategory valueCategory, const Type* type)
    -> RightFoldExpressionAST* {
  auto node = new (arena) RightFoldExpressionAST();
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->opLoc = opLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto RightFoldExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                    TokenKind op, ValueCategory valueCategory,
                                    const Type* type)
    -> RightFoldExpressionAST* {
  auto node = new (arena) RightFoldExpressionAST();
  node->expression = expression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LeftFoldExpressionAST::clone(Arena* arena) -> LeftFoldExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->opLoc = opLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto LeftFoldExpressionAST::create(Arena* arena) -> LeftFoldExpressionAST* {
  auto node = new (arena) LeftFoldExpressionAST();
  return node;
}

auto LeftFoldExpressionAST::create(Arena* arena, SourceLocation lparenLoc,
                                   SourceLocation ellipsisLoc,
                                   SourceLocation opLoc,
                                   ExpressionAST* expression,
                                   SourceLocation rparenLoc, TokenKind op,
                                   ValueCategory valueCategory,
                                   const Type* type) -> LeftFoldExpressionAST* {
  auto node = new (arena) LeftFoldExpressionAST();
  node->lparenLoc = lparenLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->opLoc = opLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LeftFoldExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                   TokenKind op, ValueCategory valueCategory,
                                   const Type* type) -> LeftFoldExpressionAST* {
  auto node = new (arena) LeftFoldExpressionAST();
  node->expression = expression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto RequiresExpressionAST::clone(Arena* arena) -> RequiresExpressionAST* {
  auto node = create(arena);

  node->requiresLoc = requiresLoc;
  node->lparenLoc = lparenLoc;

  if (parameterDeclarationClause)
    node->parameterDeclarationClause = parameterDeclarationClause->clone(arena);

  node->rparenLoc = rparenLoc;
  node->lbraceLoc = lbraceLoc;

  if (requirementList) {
    auto it = &node->requirementList;
    for (auto node : ListView{requirementList}) {
      *it = make_list_node<RequirementAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto RequiresExpressionAST::create(Arena* arena) -> RequiresExpressionAST* {
  auto node = new (arena) RequiresExpressionAST();
  return node;
}

auto RequiresExpressionAST::create(
    Arena* arena, SourceLocation requiresLoc, SourceLocation lparenLoc,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    SourceLocation rparenLoc, SourceLocation lbraceLoc,
    List<RequirementAST*>* requirementList, SourceLocation rbraceLoc,
    ValueCategory valueCategory, const Type* type) -> RequiresExpressionAST* {
  auto node = new (arena) RequiresExpressionAST();
  node->requiresLoc = requiresLoc;
  node->lparenLoc = lparenLoc;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->rparenLoc = rparenLoc;
  node->lbraceLoc = lbraceLoc;
  node->requirementList = requirementList;
  node->rbraceLoc = rbraceLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto RequiresExpressionAST::create(
    Arena* arena, ParameterDeclarationClauseAST* parameterDeclarationClause,
    List<RequirementAST*>* requirementList, ValueCategory valueCategory,
    const Type* type) -> RequiresExpressionAST* {
  auto node = new (arena) RequiresExpressionAST();
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->requirementList = requirementList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto VaArgExpressionAST::clone(Arena* arena) -> VaArgExpressionAST* {
  auto node = create(arena);

  node->vaArgLoc = vaArgLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->commaLoc = commaLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto VaArgExpressionAST::create(Arena* arena) -> VaArgExpressionAST* {
  auto node = new (arena) VaArgExpressionAST();
  return node;
}

auto VaArgExpressionAST::create(Arena* arena, SourceLocation vaArgLoc,
                                SourceLocation lparenLoc,
                                ExpressionAST* expression,
                                SourceLocation commaLoc, TypeIdAST* typeId,
                                SourceLocation rparenLoc,
                                ValueCategory valueCategory, const Type* type)
    -> VaArgExpressionAST* {
  auto node = new (arena) VaArgExpressionAST();
  node->vaArgLoc = vaArgLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->commaLoc = commaLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto VaArgExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                TypeIdAST* typeId, ValueCategory valueCategory,
                                const Type* type) -> VaArgExpressionAST* {
  auto node = new (arena) VaArgExpressionAST();
  node->expression = expression;
  node->typeId = typeId;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SubscriptExpressionAST::clone(Arena* arena) -> SubscriptExpressionAST* {
  auto node = create(arena);

  if (baseExpression) node->baseExpression = baseExpression->clone(arena);

  node->lbracketLoc = lbracketLoc;

  if (indexExpression) node->indexExpression = indexExpression->clone(arena);

  node->rbracketLoc = rbracketLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SubscriptExpressionAST::create(Arena* arena) -> SubscriptExpressionAST* {
  auto node = new (arena) SubscriptExpressionAST();
  return node;
}

auto SubscriptExpressionAST::create(
    Arena* arena, ExpressionAST* baseExpression, SourceLocation lbracketLoc,
    ExpressionAST* indexExpression, SourceLocation rbracketLoc,
    ValueCategory valueCategory, const Type* type) -> SubscriptExpressionAST* {
  auto node = new (arena) SubscriptExpressionAST();
  node->baseExpression = baseExpression;
  node->lbracketLoc = lbracketLoc;
  node->indexExpression = indexExpression;
  node->rbracketLoc = rbracketLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SubscriptExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                                    ExpressionAST* indexExpression,
                                    ValueCategory valueCategory,
                                    const Type* type)
    -> SubscriptExpressionAST* {
  auto node = new (arena) SubscriptExpressionAST();
  node->baseExpression = baseExpression;
  node->indexExpression = indexExpression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CallExpressionAST::clone(Arena* arena) -> CallExpressionAST* {
  auto node = create(arena);

  if (baseExpression) node->baseExpression = baseExpression->clone(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto CallExpressionAST::create(Arena* arena) -> CallExpressionAST* {
  auto node = new (arena) CallExpressionAST();
  return node;
}

auto CallExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                               SourceLocation lparenLoc,
                               List<ExpressionAST*>* expressionList,
                               SourceLocation rparenLoc,
                               ValueCategory valueCategory, const Type* type)
    -> CallExpressionAST* {
  auto node = new (arena) CallExpressionAST();
  node->baseExpression = baseExpression;
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CallExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                               List<ExpressionAST*>* expressionList,
                               ValueCategory valueCategory, const Type* type)
    -> CallExpressionAST* {
  auto node = new (arena) CallExpressionAST();
  node->baseExpression = baseExpression;
  node->expressionList = expressionList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeConstructionAST::clone(Arena* arena) -> TypeConstructionAST* {
  auto node = create(arena);

  if (typeSpecifier) node->typeSpecifier = typeSpecifier->clone(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TypeConstructionAST::create(Arena* arena) -> TypeConstructionAST* {
  auto node = new (arena) TypeConstructionAST();
  return node;
}

auto TypeConstructionAST::create(Arena* arena, SpecifierAST* typeSpecifier,
                                 SourceLocation lparenLoc,
                                 List<ExpressionAST*>* expressionList,
                                 SourceLocation rparenLoc,
                                 ValueCategory valueCategory, const Type* type)
    -> TypeConstructionAST* {
  auto node = new (arena) TypeConstructionAST();
  node->typeSpecifier = typeSpecifier;
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeConstructionAST::create(Arena* arena, SpecifierAST* typeSpecifier,
                                 List<ExpressionAST*>* expressionList,
                                 ValueCategory valueCategory, const Type* type)
    -> TypeConstructionAST* {
  auto node = new (arena) TypeConstructionAST();
  node->typeSpecifier = typeSpecifier;
  node->expressionList = expressionList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BracedTypeConstructionAST::clone(Arena* arena)
    -> BracedTypeConstructionAST* {
  auto node = create(arena);

  if (typeSpecifier) node->typeSpecifier = typeSpecifier->clone(arena);

  if (bracedInitList) node->bracedInitList = bracedInitList->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BracedTypeConstructionAST::create(Arena* arena)
    -> BracedTypeConstructionAST* {
  auto node = new (arena) BracedTypeConstructionAST();
  return node;
}

auto BracedTypeConstructionAST::create(Arena* arena,
                                       SpecifierAST* typeSpecifier,
                                       BracedInitListAST* bracedInitList,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> BracedTypeConstructionAST* {
  auto node = new (arena) BracedTypeConstructionAST();
  node->typeSpecifier = typeSpecifier;
  node->bracedInitList = bracedInitList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SpliceMemberExpressionAST::clone(Arena* arena)
    -> SpliceMemberExpressionAST* {
  auto node = create(arena);

  if (baseExpression) node->baseExpression = baseExpression->clone(arena);

  node->accessLoc = accessLoc;
  node->templateLoc = templateLoc;

  if (splicer) node->splicer = splicer->clone(arena);

  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SpliceMemberExpressionAST::create(Arena* arena)
    -> SpliceMemberExpressionAST* {
  auto node = new (arena) SpliceMemberExpressionAST();
  return node;
}

auto SpliceMemberExpressionAST::create(
    Arena* arena, ExpressionAST* baseExpression, SourceLocation accessLoc,
    SourceLocation templateLoc, SplicerAST* splicer, Symbol* symbol,
    TokenKind accessOp, bool isTemplateIntroduced, ValueCategory valueCategory,
    const Type* type) -> SpliceMemberExpressionAST* {
  auto node = new (arena) SpliceMemberExpressionAST();
  node->baseExpression = baseExpression;
  node->accessLoc = accessLoc;
  node->templateLoc = templateLoc;
  node->splicer = splicer;
  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SpliceMemberExpressionAST::create(
    Arena* arena, ExpressionAST* baseExpression, SplicerAST* splicer,
    Symbol* symbol, TokenKind accessOp, bool isTemplateIntroduced,
    ValueCategory valueCategory, const Type* type)
    -> SpliceMemberExpressionAST* {
  auto node = new (arena) SpliceMemberExpressionAST();
  node->baseExpression = baseExpression;
  node->splicer = splicer;
  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto MemberExpressionAST::clone(Arena* arena) -> MemberExpressionAST* {
  auto node = create(arena);

  if (baseExpression) node->baseExpression = baseExpression->clone(arena);

  node->accessLoc = accessLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto MemberExpressionAST::create(Arena* arena) -> MemberExpressionAST* {
  auto node = new (arena) MemberExpressionAST();
  return node;
}

auto MemberExpressionAST::create(
    Arena* arena, ExpressionAST* baseExpression, SourceLocation accessLoc,
    NestedNameSpecifierAST* nestedNameSpecifier, SourceLocation templateLoc,
    UnqualifiedIdAST* unqualifiedId, Symbol* symbol, TokenKind accessOp,
    bool isTemplateIntroduced, ValueCategory valueCategory, const Type* type)
    -> MemberExpressionAST* {
  auto node = new (arena) MemberExpressionAST();
  node->baseExpression = baseExpression;
  node->accessLoc = accessLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto MemberExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                                 NestedNameSpecifierAST* nestedNameSpecifier,
                                 UnqualifiedIdAST* unqualifiedId,
                                 Symbol* symbol, TokenKind accessOp,
                                 bool isTemplateIntroduced,
                                 ValueCategory valueCategory, const Type* type)
    -> MemberExpressionAST* {
  auto node = new (arena) MemberExpressionAST();
  node->baseExpression = baseExpression;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->symbol = symbol;
  node->accessOp = accessOp;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto PostIncrExpressionAST::clone(Arena* arena) -> PostIncrExpressionAST* {
  auto node = create(arena);

  if (baseExpression) node->baseExpression = baseExpression->clone(arena);

  node->opLoc = opLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto PostIncrExpressionAST::create(Arena* arena) -> PostIncrExpressionAST* {
  auto node = new (arena) PostIncrExpressionAST();
  return node;
}

auto PostIncrExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                                   SourceLocation opLoc, TokenKind op,
                                   ValueCategory valueCategory,
                                   const Type* type) -> PostIncrExpressionAST* {
  auto node = new (arena) PostIncrExpressionAST();
  node->baseExpression = baseExpression;
  node->opLoc = opLoc;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto PostIncrExpressionAST::create(Arena* arena, ExpressionAST* baseExpression,
                                   TokenKind op, ValueCategory valueCategory,
                                   const Type* type) -> PostIncrExpressionAST* {
  auto node = new (arena) PostIncrExpressionAST();
  node->baseExpression = baseExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CppCastExpressionAST::clone(Arena* arena) -> CppCastExpressionAST* {
  auto node = create(arena);

  node->castLoc = castLoc;
  node->lessLoc = lessLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->greaterLoc = greaterLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto CppCastExpressionAST::create(Arena* arena) -> CppCastExpressionAST* {
  auto node = new (arena) CppCastExpressionAST();
  return node;
}

auto CppCastExpressionAST::create(
    Arena* arena, SourceLocation castLoc, SourceLocation lessLoc,
    TypeIdAST* typeId, SourceLocation greaterLoc, SourceLocation lparenLoc,
    ExpressionAST* expression, SourceLocation rparenLoc,
    ValueCategory valueCategory, const Type* type) -> CppCastExpressionAST* {
  auto node = new (arena) CppCastExpressionAST();
  node->castLoc = castLoc;
  node->lessLoc = lessLoc;
  node->typeId = typeId;
  node->greaterLoc = greaterLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CppCastExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                  ExpressionAST* expression,
                                  ValueCategory valueCategory, const Type* type)
    -> CppCastExpressionAST* {
  auto node = new (arena) CppCastExpressionAST();
  node->typeId = typeId;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BuiltinBitCastExpressionAST::clone(Arena* arena)
    -> BuiltinBitCastExpressionAST* {
  auto node = create(arena);

  node->castLoc = castLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->commaLoc = commaLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BuiltinBitCastExpressionAST::create(Arena* arena)
    -> BuiltinBitCastExpressionAST* {
  auto node = new (arena) BuiltinBitCastExpressionAST();
  return node;
}

auto BuiltinBitCastExpressionAST::create(
    Arena* arena, SourceLocation castLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation commaLoc, ExpressionAST* expression,
    SourceLocation rparenLoc, ValueCategory valueCategory, const Type* type)
    -> BuiltinBitCastExpressionAST* {
  auto node = new (arena) BuiltinBitCastExpressionAST();
  node->castLoc = castLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->commaLoc = commaLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BuiltinBitCastExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                         ExpressionAST* expression,
                                         ValueCategory valueCategory,
                                         const Type* type)
    -> BuiltinBitCastExpressionAST* {
  auto node = new (arena) BuiltinBitCastExpressionAST();
  node->typeId = typeId;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BuiltinOffsetofExpressionAST::clone(Arena* arena)
    -> BuiltinOffsetofExpressionAST* {
  auto node = create(arena);

  node->offsetofLoc = offsetofLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->commaLoc = commaLoc;
  node->identifierLoc = identifierLoc;

  if (designatorList) {
    auto it = &node->designatorList;
    for (auto node : ListView{designatorList}) {
      *it = make_list_node<DesignatorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BuiltinOffsetofExpressionAST::create(Arena* arena)
    -> BuiltinOffsetofExpressionAST* {
  auto node = new (arena) BuiltinOffsetofExpressionAST();
  return node;
}

auto BuiltinOffsetofExpressionAST::create(
    Arena* arena, SourceLocation offsetofLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation commaLoc, SourceLocation identifierLoc,
    List<DesignatorAST*>* designatorList, SourceLocation rparenLoc,
    const Identifier* identifier, FieldSymbol* symbol,
    ValueCategory valueCategory, const Type* type)
    -> BuiltinOffsetofExpressionAST* {
  auto node = new (arena) BuiltinOffsetofExpressionAST();
  node->offsetofLoc = offsetofLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->commaLoc = commaLoc;
  node->identifierLoc = identifierLoc;
  node->designatorList = designatorList;
  node->rparenLoc = rparenLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BuiltinOffsetofExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                          List<DesignatorAST*>* designatorList,
                                          const Identifier* identifier,
                                          FieldSymbol* symbol,
                                          ValueCategory valueCategory,
                                          const Type* type)
    -> BuiltinOffsetofExpressionAST* {
  auto node = new (arena) BuiltinOffsetofExpressionAST();
  node->typeId = typeId;
  node->designatorList = designatorList;
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeidExpressionAST::clone(Arena* arena) -> TypeidExpressionAST* {
  auto node = create(arena);

  node->typeidLoc = typeidLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TypeidExpressionAST::create(Arena* arena) -> TypeidExpressionAST* {
  auto node = new (arena) TypeidExpressionAST();
  return node;
}

auto TypeidExpressionAST::create(Arena* arena, SourceLocation typeidLoc,
                                 SourceLocation lparenLoc,
                                 ExpressionAST* expression,
                                 SourceLocation rparenLoc,
                                 ValueCategory valueCategory, const Type* type)
    -> TypeidExpressionAST* {
  auto node = new (arena) TypeidExpressionAST();
  node->typeidLoc = typeidLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeidExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> TypeidExpressionAST* {
  auto node = new (arena) TypeidExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeidOfTypeExpressionAST::clone(Arena* arena)
    -> TypeidOfTypeExpressionAST* {
  auto node = create(arena);

  node->typeidLoc = typeidLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TypeidOfTypeExpressionAST::create(Arena* arena)
    -> TypeidOfTypeExpressionAST* {
  auto node = new (arena) TypeidOfTypeExpressionAST();
  return node;
}

auto TypeidOfTypeExpressionAST::create(
    Arena* arena, SourceLocation typeidLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation rparenLoc, ValueCategory valueCategory,
    const Type* type) -> TypeidOfTypeExpressionAST* {
  auto node = new (arena) TypeidOfTypeExpressionAST();
  node->typeidLoc = typeidLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeidOfTypeExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> TypeidOfTypeExpressionAST* {
  auto node = new (arena) TypeidOfTypeExpressionAST();
  node->typeId = typeId;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SpliceExpressionAST::clone(Arena* arena) -> SpliceExpressionAST* {
  auto node = create(arena);

  if (splicer) node->splicer = splicer->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SpliceExpressionAST::create(Arena* arena) -> SpliceExpressionAST* {
  auto node = new (arena) SpliceExpressionAST();
  return node;
}

auto SpliceExpressionAST::create(Arena* arena, SplicerAST* splicer,
                                 ValueCategory valueCategory, const Type* type)
    -> SpliceExpressionAST* {
  auto node = new (arena) SpliceExpressionAST();
  node->splicer = splicer;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto GlobalScopeReflectExpressionAST::clone(Arena* arena)
    -> GlobalScopeReflectExpressionAST* {
  auto node = create(arena);

  node->caretCaretLoc = caretCaretLoc;
  node->scopeLoc = scopeLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto GlobalScopeReflectExpressionAST::create(Arena* arena)
    -> GlobalScopeReflectExpressionAST* {
  auto node = new (arena) GlobalScopeReflectExpressionAST();
  return node;
}

auto GlobalScopeReflectExpressionAST::create(Arena* arena,
                                             SourceLocation caretCaretLoc,
                                             SourceLocation scopeLoc,
                                             ValueCategory valueCategory,
                                             const Type* type)
    -> GlobalScopeReflectExpressionAST* {
  auto node = new (arena) GlobalScopeReflectExpressionAST();
  node->caretCaretLoc = caretCaretLoc;
  node->scopeLoc = scopeLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto GlobalScopeReflectExpressionAST::create(Arena* arena,
                                             ValueCategory valueCategory,
                                             const Type* type)
    -> GlobalScopeReflectExpressionAST* {
  auto node = new (arena) GlobalScopeReflectExpressionAST();
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NamespaceReflectExpressionAST::clone(Arena* arena)
    -> NamespaceReflectExpressionAST* {
  auto node = create(arena);

  node->caretCaretLoc = caretCaretLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NamespaceReflectExpressionAST::create(Arena* arena)
    -> NamespaceReflectExpressionAST* {
  auto node = new (arena) NamespaceReflectExpressionAST();
  return node;
}

auto NamespaceReflectExpressionAST::create(
    Arena* arena, SourceLocation caretCaretLoc, SourceLocation identifierLoc,
    const Identifier* identifier, NamespaceSymbol* symbol,
    ValueCategory valueCategory, const Type* type)
    -> NamespaceReflectExpressionAST* {
  auto node = new (arena) NamespaceReflectExpressionAST();
  node->caretCaretLoc = caretCaretLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NamespaceReflectExpressionAST::create(Arena* arena,
                                           const Identifier* identifier,
                                           NamespaceSymbol* symbol,
                                           ValueCategory valueCategory,
                                           const Type* type)
    -> NamespaceReflectExpressionAST* {
  auto node = new (arena) NamespaceReflectExpressionAST();
  node->identifier = identifier;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeIdReflectExpressionAST::clone(Arena* arena)
    -> TypeIdReflectExpressionAST* {
  auto node = create(arena);

  node->caretCaretLoc = caretCaretLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TypeIdReflectExpressionAST::create(Arena* arena)
    -> TypeIdReflectExpressionAST* {
  auto node = new (arena) TypeIdReflectExpressionAST();
  return node;
}

auto TypeIdReflectExpressionAST::create(Arena* arena,
                                        SourceLocation caretCaretLoc,
                                        TypeIdAST* typeId,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> TypeIdReflectExpressionAST* {
  auto node = new (arena) TypeIdReflectExpressionAST();
  node->caretCaretLoc = caretCaretLoc;
  node->typeId = typeId;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeIdReflectExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> TypeIdReflectExpressionAST* {
  auto node = new (arena) TypeIdReflectExpressionAST();
  node->typeId = typeId;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ReflectExpressionAST::clone(Arena* arena) -> ReflectExpressionAST* {
  auto node = create(arena);

  node->caretCaretLoc = caretCaretLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ReflectExpressionAST::create(Arena* arena) -> ReflectExpressionAST* {
  auto node = new (arena) ReflectExpressionAST();
  return node;
}

auto ReflectExpressionAST::create(Arena* arena, SourceLocation caretCaretLoc,
                                  ExpressionAST* expression,
                                  ValueCategory valueCategory, const Type* type)
    -> ReflectExpressionAST* {
  auto node = new (arena) ReflectExpressionAST();
  node->caretCaretLoc = caretCaretLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ReflectExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                  ValueCategory valueCategory, const Type* type)
    -> ReflectExpressionAST* {
  auto node = new (arena) ReflectExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LabelAddressExpressionAST::clone(Arena* arena)
    -> LabelAddressExpressionAST* {
  auto node = create(arena);

  node->ampAmpLoc = ampAmpLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto LabelAddressExpressionAST::create(Arena* arena)
    -> LabelAddressExpressionAST* {
  auto node = new (arena) LabelAddressExpressionAST();
  return node;
}

auto LabelAddressExpressionAST::create(Arena* arena, SourceLocation ampAmpLoc,
                                       SourceLocation identifierLoc,
                                       const Identifier* identifier,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> LabelAddressExpressionAST* {
  auto node = new (arena) LabelAddressExpressionAST();
  node->ampAmpLoc = ampAmpLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto LabelAddressExpressionAST::create(Arena* arena,
                                       const Identifier* identifier,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> LabelAddressExpressionAST* {
  auto node = new (arena) LabelAddressExpressionAST();
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto UnaryExpressionAST::clone(Arena* arena) -> UnaryExpressionAST* {
  auto node = create(arena);

  node->opLoc = opLoc;

  if (expression) node->expression = expression->clone(arena);

  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto UnaryExpressionAST::create(Arena* arena) -> UnaryExpressionAST* {
  auto node = new (arena) UnaryExpressionAST();
  return node;
}

auto UnaryExpressionAST::create(Arena* arena, SourceLocation opLoc,
                                ExpressionAST* expression, TokenKind op,
                                ValueCategory valueCategory, const Type* type)
    -> UnaryExpressionAST* {
  auto node = new (arena) UnaryExpressionAST();
  node->opLoc = opLoc;
  node->expression = expression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto UnaryExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                TokenKind op, ValueCategory valueCategory,
                                const Type* type) -> UnaryExpressionAST* {
  auto node = new (arena) UnaryExpressionAST();
  node->expression = expression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AwaitExpressionAST::clone(Arena* arena) -> AwaitExpressionAST* {
  auto node = create(arena);

  node->awaitLoc = awaitLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto AwaitExpressionAST::create(Arena* arena) -> AwaitExpressionAST* {
  auto node = new (arena) AwaitExpressionAST();
  return node;
}

auto AwaitExpressionAST::create(Arena* arena, SourceLocation awaitLoc,
                                ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> AwaitExpressionAST* {
  auto node = new (arena) AwaitExpressionAST();
  node->awaitLoc = awaitLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AwaitExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> AwaitExpressionAST* {
  auto node = new (arena) AwaitExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofExpressionAST::clone(Arena* arena) -> SizeofExpressionAST* {
  auto node = create(arena);

  node->sizeofLoc = sizeofLoc;

  if (expression) node->expression = expression->clone(arena);

  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SizeofExpressionAST::create(Arena* arena) -> SizeofExpressionAST* {
  auto node = new (arena) SizeofExpressionAST();
  return node;
}

auto SizeofExpressionAST::create(Arena* arena, SourceLocation sizeofLoc,
                                 ExpressionAST* expression,
                                 std::optional<std::int64_t> value,
                                 ValueCategory valueCategory, const Type* type)
    -> SizeofExpressionAST* {
  auto node = new (arena) SizeofExpressionAST();
  node->sizeofLoc = sizeofLoc;
  node->expression = expression;
  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                 std::optional<std::int64_t> value,
                                 ValueCategory valueCategory, const Type* type)
    -> SizeofExpressionAST* {
  auto node = new (arena) SizeofExpressionAST();
  node->expression = expression;
  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofTypeExpressionAST::clone(Arena* arena) -> SizeofTypeExpressionAST* {
  auto node = create(arena);

  node->sizeofLoc = sizeofLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SizeofTypeExpressionAST::create(Arena* arena) -> SizeofTypeExpressionAST* {
  auto node = new (arena) SizeofTypeExpressionAST();
  return node;
}

auto SizeofTypeExpressionAST::create(
    Arena* arena, SourceLocation sizeofLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation rparenLoc,
    std::optional<std::int64_t> value, ValueCategory valueCategory,
    const Type* type) -> SizeofTypeExpressionAST* {
  auto node = new (arena) SizeofTypeExpressionAST();
  node->sizeofLoc = sizeofLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofTypeExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                     std::optional<std::int64_t> value,
                                     ValueCategory valueCategory,
                                     const Type* type)
    -> SizeofTypeExpressionAST* {
  auto node = new (arena) SizeofTypeExpressionAST();
  node->typeId = typeId;
  node->value = value;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofPackExpressionAST::clone(Arena* arena) -> SizeofPackExpressionAST* {
  auto node = create(arena);

  node->sizeofLoc = sizeofLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->lparenLoc = lparenLoc;
  node->identifierLoc = identifierLoc;
  node->rparenLoc = rparenLoc;
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto SizeofPackExpressionAST::create(Arena* arena) -> SizeofPackExpressionAST* {
  auto node = new (arena) SizeofPackExpressionAST();
  return node;
}

auto SizeofPackExpressionAST::create(
    Arena* arena, SourceLocation sizeofLoc, SourceLocation ellipsisLoc,
    SourceLocation lparenLoc, SourceLocation identifierLoc,
    SourceLocation rparenLoc, const Identifier* identifier,
    ValueCategory valueCategory, const Type* type) -> SizeofPackExpressionAST* {
  auto node = new (arena) SizeofPackExpressionAST();
  node->sizeofLoc = sizeofLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->lparenLoc = lparenLoc;
  node->identifierLoc = identifierLoc;
  node->rparenLoc = rparenLoc;
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto SizeofPackExpressionAST::create(Arena* arena, const Identifier* identifier,
                                     ValueCategory valueCategory,
                                     const Type* type)
    -> SizeofPackExpressionAST* {
  auto node = new (arena) SizeofPackExpressionAST();
  node->identifier = identifier;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AlignofTypeExpressionAST::clone(Arena* arena)
    -> AlignofTypeExpressionAST* {
  auto node = create(arena);

  node->alignofLoc = alignofLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto AlignofTypeExpressionAST::create(Arena* arena)
    -> AlignofTypeExpressionAST* {
  auto node = new (arena) AlignofTypeExpressionAST();
  return node;
}

auto AlignofTypeExpressionAST::create(
    Arena* arena, SourceLocation alignofLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation rparenLoc, ValueCategory valueCategory,
    const Type* type) -> AlignofTypeExpressionAST* {
  auto node = new (arena) AlignofTypeExpressionAST();
  node->alignofLoc = alignofLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AlignofTypeExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                                      ValueCategory valueCategory,
                                      const Type* type)
    -> AlignofTypeExpressionAST* {
  auto node = new (arena) AlignofTypeExpressionAST();
  node->typeId = typeId;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AlignofExpressionAST::clone(Arena* arena) -> AlignofExpressionAST* {
  auto node = create(arena);

  node->alignofLoc = alignofLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto AlignofExpressionAST::create(Arena* arena) -> AlignofExpressionAST* {
  auto node = new (arena) AlignofExpressionAST();
  return node;
}

auto AlignofExpressionAST::create(Arena* arena, SourceLocation alignofLoc,
                                  ExpressionAST* expression,
                                  ValueCategory valueCategory, const Type* type)
    -> AlignofExpressionAST* {
  auto node = new (arena) AlignofExpressionAST();
  node->alignofLoc = alignofLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AlignofExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                  ValueCategory valueCategory, const Type* type)
    -> AlignofExpressionAST* {
  auto node = new (arena) AlignofExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NoexceptExpressionAST::clone(Arena* arena) -> NoexceptExpressionAST* {
  auto node = create(arena);

  node->noexceptLoc = noexceptLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NoexceptExpressionAST::create(Arena* arena) -> NoexceptExpressionAST* {
  auto node = new (arena) NoexceptExpressionAST();
  return node;
}

auto NoexceptExpressionAST::create(Arena* arena, SourceLocation noexceptLoc,
                                   SourceLocation lparenLoc,
                                   ExpressionAST* expression,
                                   SourceLocation rparenLoc,
                                   ValueCategory valueCategory,
                                   const Type* type) -> NoexceptExpressionAST* {
  auto node = new (arena) NoexceptExpressionAST();
  node->noexceptLoc = noexceptLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NoexceptExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                   ValueCategory valueCategory,
                                   const Type* type) -> NoexceptExpressionAST* {
  auto node = new (arena) NoexceptExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NewExpressionAST::clone(Arena* arena) -> NewExpressionAST* {
  auto node = create(arena);

  node->scopeLoc = scopeLoc;
  node->newLoc = newLoc;

  if (newPlacement) node->newPlacement = newPlacement->clone(arena);

  node->lparenLoc = lparenLoc;

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  node->rparenLoc = rparenLoc;

  if (newInitalizer) node->newInitalizer = newInitalizer->clone(arena);

  node->objectType = objectType;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto NewExpressionAST::create(Arena* arena) -> NewExpressionAST* {
  auto node = new (arena) NewExpressionAST();
  return node;
}

auto NewExpressionAST::create(
    Arena* arena, SourceLocation scopeLoc, SourceLocation newLoc,
    NewPlacementAST* newPlacement, SourceLocation lparenLoc,
    List<SpecifierAST*>* typeSpecifierList, DeclaratorAST* declarator,
    SourceLocation rparenLoc, NewInitializerAST* newInitalizer,
    const Type* objectType, ValueCategory valueCategory, const Type* type)
    -> NewExpressionAST* {
  auto node = new (arena) NewExpressionAST();
  node->scopeLoc = scopeLoc;
  node->newLoc = newLoc;
  node->newPlacement = newPlacement;
  node->lparenLoc = lparenLoc;
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  node->rparenLoc = rparenLoc;
  node->newInitalizer = newInitalizer;
  node->objectType = objectType;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto NewExpressionAST::create(Arena* arena, NewPlacementAST* newPlacement,
                              List<SpecifierAST*>* typeSpecifierList,
                              DeclaratorAST* declarator,
                              NewInitializerAST* newInitalizer,
                              const Type* objectType,
                              ValueCategory valueCategory, const Type* type)
    -> NewExpressionAST* {
  auto node = new (arena) NewExpressionAST();
  node->newPlacement = newPlacement;
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  node->newInitalizer = newInitalizer;
  node->objectType = objectType;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto DeleteExpressionAST::clone(Arena* arena) -> DeleteExpressionAST* {
  auto node = create(arena);

  node->scopeLoc = scopeLoc;
  node->deleteLoc = deleteLoc;
  node->lbracketLoc = lbracketLoc;
  node->rbracketLoc = rbracketLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto DeleteExpressionAST::create(Arena* arena) -> DeleteExpressionAST* {
  auto node = new (arena) DeleteExpressionAST();
  return node;
}

auto DeleteExpressionAST::create(Arena* arena, SourceLocation scopeLoc,
                                 SourceLocation deleteLoc,
                                 SourceLocation lbracketLoc,
                                 SourceLocation rbracketLoc,
                                 ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> DeleteExpressionAST* {
  auto node = new (arena) DeleteExpressionAST();
  node->scopeLoc = scopeLoc;
  node->deleteLoc = deleteLoc;
  node->lbracketLoc = lbracketLoc;
  node->rbracketLoc = rbracketLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto DeleteExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> DeleteExpressionAST* {
  auto node = new (arena) DeleteExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CastExpressionAST::clone(Arena* arena) -> CastExpressionAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto CastExpressionAST::create(Arena* arena) -> CastExpressionAST* {
  auto node = new (arena) CastExpressionAST();
  return node;
}

auto CastExpressionAST::create(Arena* arena, SourceLocation lparenLoc,
                               TypeIdAST* typeId, SourceLocation rparenLoc,
                               ExpressionAST* expression,
                               ValueCategory valueCategory, const Type* type)
    -> CastExpressionAST* {
  auto node = new (arena) CastExpressionAST();
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CastExpressionAST::create(Arena* arena, TypeIdAST* typeId,
                               ExpressionAST* expression,
                               ValueCategory valueCategory, const Type* type)
    -> CastExpressionAST* {
  auto node = new (arena) CastExpressionAST();
  node->typeId = typeId;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ImplicitCastExpressionAST::clone(Arena* arena)
    -> ImplicitCastExpressionAST* {
  auto node = create(arena);

  if (expression) node->expression = expression->clone(arena);

  node->castKind = castKind;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ImplicitCastExpressionAST::create(Arena* arena)
    -> ImplicitCastExpressionAST* {
  auto node = new (arena) ImplicitCastExpressionAST();
  return node;
}

auto ImplicitCastExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                       ImplicitCastKind castKind,
                                       ValueCategory valueCategory,
                                       const Type* type)
    -> ImplicitCastExpressionAST* {
  auto node = new (arena) ImplicitCastExpressionAST();
  node->expression = expression;
  node->castKind = castKind;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BinaryExpressionAST::clone(Arena* arena) -> BinaryExpressionAST* {
  auto node = create(arena);

  if (leftExpression) node->leftExpression = leftExpression->clone(arena);

  node->opLoc = opLoc;

  if (rightExpression) node->rightExpression = rightExpression->clone(arena);

  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BinaryExpressionAST::create(Arena* arena) -> BinaryExpressionAST* {
  auto node = new (arena) BinaryExpressionAST();
  return node;
}

auto BinaryExpressionAST::create(Arena* arena, ExpressionAST* leftExpression,
                                 SourceLocation opLoc,
                                 ExpressionAST* rightExpression, TokenKind op,
                                 ValueCategory valueCategory, const Type* type)
    -> BinaryExpressionAST* {
  auto node = new (arena) BinaryExpressionAST();
  node->leftExpression = leftExpression;
  node->opLoc = opLoc;
  node->rightExpression = rightExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BinaryExpressionAST::create(Arena* arena, ExpressionAST* leftExpression,
                                 ExpressionAST* rightExpression, TokenKind op,
                                 ValueCategory valueCategory, const Type* type)
    -> BinaryExpressionAST* {
  auto node = new (arena) BinaryExpressionAST();
  node->leftExpression = leftExpression;
  node->rightExpression = rightExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ConditionalExpressionAST::clone(Arena* arena)
    -> ConditionalExpressionAST* {
  auto node = create(arena);

  if (condition) node->condition = condition->clone(arena);

  node->questionLoc = questionLoc;

  if (iftrueExpression) node->iftrueExpression = iftrueExpression->clone(arena);

  node->colonLoc = colonLoc;

  if (iffalseExpression)
    node->iffalseExpression = iffalseExpression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ConditionalExpressionAST::create(Arena* arena)
    -> ConditionalExpressionAST* {
  auto node = new (arena) ConditionalExpressionAST();
  return node;
}

auto ConditionalExpressionAST::create(
    Arena* arena, ExpressionAST* condition, SourceLocation questionLoc,
    ExpressionAST* iftrueExpression, SourceLocation colonLoc,
    ExpressionAST* iffalseExpression, ValueCategory valueCategory,
    const Type* type) -> ConditionalExpressionAST* {
  auto node = new (arena) ConditionalExpressionAST();
  node->condition = condition;
  node->questionLoc = questionLoc;
  node->iftrueExpression = iftrueExpression;
  node->colonLoc = colonLoc;
  node->iffalseExpression = iffalseExpression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ConditionalExpressionAST::create(Arena* arena, ExpressionAST* condition,
                                      ExpressionAST* iftrueExpression,
                                      ExpressionAST* iffalseExpression,
                                      ValueCategory valueCategory,
                                      const Type* type)
    -> ConditionalExpressionAST* {
  auto node = new (arena) ConditionalExpressionAST();
  node->condition = condition;
  node->iftrueExpression = iftrueExpression;
  node->iffalseExpression = iffalseExpression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto YieldExpressionAST::clone(Arena* arena) -> YieldExpressionAST* {
  auto node = create(arena);

  node->yieldLoc = yieldLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto YieldExpressionAST::create(Arena* arena) -> YieldExpressionAST* {
  auto node = new (arena) YieldExpressionAST();
  return node;
}

auto YieldExpressionAST::create(Arena* arena, SourceLocation yieldLoc,
                                ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> YieldExpressionAST* {
  auto node = new (arena) YieldExpressionAST();
  node->yieldLoc = yieldLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto YieldExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> YieldExpressionAST* {
  auto node = new (arena) YieldExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ThrowExpressionAST::clone(Arena* arena) -> ThrowExpressionAST* {
  auto node = create(arena);

  node->throwLoc = throwLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ThrowExpressionAST::create(Arena* arena) -> ThrowExpressionAST* {
  auto node = new (arena) ThrowExpressionAST();
  return node;
}

auto ThrowExpressionAST::create(Arena* arena, SourceLocation throwLoc,
                                ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> ThrowExpressionAST* {
  auto node = new (arena) ThrowExpressionAST();
  node->throwLoc = throwLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ThrowExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                ValueCategory valueCategory, const Type* type)
    -> ThrowExpressionAST* {
  auto node = new (arena) ThrowExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AssignmentExpressionAST::clone(Arena* arena) -> AssignmentExpressionAST* {
  auto node = create(arena);

  if (leftExpression) node->leftExpression = leftExpression->clone(arena);

  node->opLoc = opLoc;

  if (rightExpression) node->rightExpression = rightExpression->clone(arena);

  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto AssignmentExpressionAST::create(Arena* arena) -> AssignmentExpressionAST* {
  auto node = new (arena) AssignmentExpressionAST();
  return node;
}

auto AssignmentExpressionAST::create(
    Arena* arena, ExpressionAST* leftExpression, SourceLocation opLoc,
    ExpressionAST* rightExpression, TokenKind op, ValueCategory valueCategory,
    const Type* type) -> AssignmentExpressionAST* {
  auto node = new (arena) AssignmentExpressionAST();
  node->leftExpression = leftExpression;
  node->opLoc = opLoc;
  node->rightExpression = rightExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto AssignmentExpressionAST::create(Arena* arena,
                                     ExpressionAST* leftExpression,
                                     ExpressionAST* rightExpression,
                                     TokenKind op, ValueCategory valueCategory,
                                     const Type* type)
    -> AssignmentExpressionAST* {
  auto node = new (arena) AssignmentExpressionAST();
  node->leftExpression = leftExpression;
  node->rightExpression = rightExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TargetExpressionAST::clone(Arena* arena) -> TargetExpressionAST* {
  auto node = create(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TargetExpressionAST::create(Arena* arena) -> TargetExpressionAST* {
  auto node = new (arena) TargetExpressionAST();
  return node;
}

auto TargetExpressionAST::create(Arena* arena, ValueCategory valueCategory,
                                 const Type* type) -> TargetExpressionAST* {
  auto node = new (arena) TargetExpressionAST();
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto RightExpressionAST::clone(Arena* arena) -> RightExpressionAST* {
  auto node = create(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto RightExpressionAST::create(Arena* arena) -> RightExpressionAST* {
  auto node = new (arena) RightExpressionAST();
  return node;
}

auto RightExpressionAST::create(Arena* arena, ValueCategory valueCategory,
                                const Type* type) -> RightExpressionAST* {
  auto node = new (arena) RightExpressionAST();
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CompoundAssignmentExpressionAST::clone(Arena* arena)
    -> CompoundAssignmentExpressionAST* {
  auto node = create(arena);

  if (targetExpression) node->targetExpression = targetExpression->clone(arena);

  node->opLoc = opLoc;

  if (leftExpression) node->leftExpression = leftExpression->clone(arena);

  if (rightExpression) node->rightExpression = rightExpression->clone(arena);

  if (adjustExpression) node->adjustExpression = adjustExpression->clone(arena);

  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto CompoundAssignmentExpressionAST::create(Arena* arena)
    -> CompoundAssignmentExpressionAST* {
  auto node = new (arena) CompoundAssignmentExpressionAST();
  return node;
}

auto CompoundAssignmentExpressionAST::create(
    Arena* arena, ExpressionAST* targetExpression, SourceLocation opLoc,
    ExpressionAST* leftExpression, ExpressionAST* rightExpression,
    ExpressionAST* adjustExpression, TokenKind op, ValueCategory valueCategory,
    const Type* type) -> CompoundAssignmentExpressionAST* {
  auto node = new (arena) CompoundAssignmentExpressionAST();
  node->targetExpression = targetExpression;
  node->opLoc = opLoc;
  node->leftExpression = leftExpression;
  node->rightExpression = rightExpression;
  node->adjustExpression = adjustExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto CompoundAssignmentExpressionAST::create(
    Arena* arena, ExpressionAST* targetExpression,
    ExpressionAST* leftExpression, ExpressionAST* rightExpression,
    ExpressionAST* adjustExpression, TokenKind op, ValueCategory valueCategory,
    const Type* type) -> CompoundAssignmentExpressionAST* {
  auto node = new (arena) CompoundAssignmentExpressionAST();
  node->targetExpression = targetExpression;
  node->leftExpression = leftExpression;
  node->rightExpression = rightExpression;
  node->adjustExpression = adjustExpression;
  node->op = op;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto PackExpansionExpressionAST::clone(Arena* arena)
    -> PackExpansionExpressionAST* {
  auto node = create(arena);

  if (expression) node->expression = expression->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto PackExpansionExpressionAST::create(Arena* arena)
    -> PackExpansionExpressionAST* {
  auto node = new (arena) PackExpansionExpressionAST();
  return node;
}

auto PackExpansionExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                        SourceLocation ellipsisLoc,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> PackExpansionExpressionAST* {
  auto node = new (arena) PackExpansionExpressionAST();
  node->expression = expression;
  node->ellipsisLoc = ellipsisLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto PackExpansionExpressionAST::create(Arena* arena, ExpressionAST* expression,
                                        ValueCategory valueCategory,
                                        const Type* type)
    -> PackExpansionExpressionAST* {
  auto node = new (arena) PackExpansionExpressionAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto DesignatedInitializerClauseAST::clone(Arena* arena)
    -> DesignatedInitializerClauseAST* {
  auto node = create(arena);

  if (designatorList) {
    auto it = &node->designatorList;
    for (auto node : ListView{designatorList}) {
      *it = make_list_node<DesignatorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (initializer) node->initializer = initializer->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto DesignatedInitializerClauseAST::create(Arena* arena)
    -> DesignatedInitializerClauseAST* {
  auto node = new (arena) DesignatedInitializerClauseAST();
  return node;
}

auto DesignatedInitializerClauseAST::create(
    Arena* arena, List<DesignatorAST*>* designatorList,
    ExpressionAST* initializer, ValueCategory valueCategory, const Type* type)
    -> DesignatedInitializerClauseAST* {
  auto node = new (arena) DesignatedInitializerClauseAST();
  node->designatorList = designatorList;
  node->initializer = initializer;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeTraitExpressionAST::clone(Arena* arena) -> TypeTraitExpressionAST* {
  auto node = create(arena);

  node->typeTraitLoc = typeTraitLoc;
  node->lparenLoc = lparenLoc;

  if (typeIdList) {
    auto it = &node->typeIdList;
    for (auto node : ListView{typeIdList}) {
      *it = make_list_node<TypeIdAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->typeTrait = typeTrait;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto TypeTraitExpressionAST::create(Arena* arena) -> TypeTraitExpressionAST* {
  auto node = new (arena) TypeTraitExpressionAST();
  return node;
}

auto TypeTraitExpressionAST::create(
    Arena* arena, SourceLocation typeTraitLoc, SourceLocation lparenLoc,
    List<TypeIdAST*>* typeIdList, SourceLocation rparenLoc,
    BuiltinTypeTraitKind typeTrait, ValueCategory valueCategory,
    const Type* type) -> TypeTraitExpressionAST* {
  auto node = new (arena) TypeTraitExpressionAST();
  node->typeTraitLoc = typeTraitLoc;
  node->lparenLoc = lparenLoc;
  node->typeIdList = typeIdList;
  node->rparenLoc = rparenLoc;
  node->typeTrait = typeTrait;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto TypeTraitExpressionAST::create(Arena* arena, List<TypeIdAST*>* typeIdList,
                                    BuiltinTypeTraitKind typeTrait,
                                    ValueCategory valueCategory,
                                    const Type* type)
    -> TypeTraitExpressionAST* {
  auto node = new (arena) TypeTraitExpressionAST();
  node->typeIdList = typeIdList;
  node->typeTrait = typeTrait;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ConditionExpressionAST::clone(Arena* arena) -> ConditionExpressionAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declSpecifierList) {
    auto it = &node->declSpecifierList;
    for (auto node : ListView{declSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  if (initializer) node->initializer = initializer->clone(arena);

  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ConditionExpressionAST::create(Arena* arena) -> ConditionExpressionAST* {
  auto node = new (arena) ConditionExpressionAST();
  return node;
}

auto ConditionExpressionAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* declSpecifierList, DeclaratorAST* declarator,
    ExpressionAST* initializer, VariableSymbol* symbol,
    ValueCategory valueCategory, const Type* type) -> ConditionExpressionAST* {
  auto node = new (arena) ConditionExpressionAST();
  node->attributeList = attributeList;
  node->declSpecifierList = declSpecifierList;
  node->declarator = declarator;
  node->initializer = initializer;
  node->symbol = symbol;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto EqualInitializerAST::clone(Arena* arena) -> EqualInitializerAST* {
  auto node = create(arena);

  node->equalLoc = equalLoc;

  if (expression) node->expression = expression->clone(arena);

  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto EqualInitializerAST::create(Arena* arena) -> EqualInitializerAST* {
  auto node = new (arena) EqualInitializerAST();
  return node;
}

auto EqualInitializerAST::create(Arena* arena, SourceLocation equalLoc,
                                 ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> EqualInitializerAST* {
  auto node = new (arena) EqualInitializerAST();
  node->equalLoc = equalLoc;
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto EqualInitializerAST::create(Arena* arena, ExpressionAST* expression,
                                 ValueCategory valueCategory, const Type* type)
    -> EqualInitializerAST* {
  auto node = new (arena) EqualInitializerAST();
  node->expression = expression;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BracedInitListAST::clone(Arena* arena) -> BracedInitListAST* {
  auto node = create(arena);

  node->lbraceLoc = lbraceLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->commaLoc = commaLoc;
  node->rbraceLoc = rbraceLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto BracedInitListAST::create(Arena* arena) -> BracedInitListAST* {
  auto node = new (arena) BracedInitListAST();
  return node;
}

auto BracedInitListAST::create(Arena* arena, SourceLocation lbraceLoc,
                               List<ExpressionAST*>* expressionList,
                               SourceLocation commaLoc,
                               SourceLocation rbraceLoc,
                               ValueCategory valueCategory, const Type* type)
    -> BracedInitListAST* {
  auto node = new (arena) BracedInitListAST();
  node->lbraceLoc = lbraceLoc;
  node->expressionList = expressionList;
  node->commaLoc = commaLoc;
  node->rbraceLoc = rbraceLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto BracedInitListAST::create(Arena* arena,
                               List<ExpressionAST*>* expressionList,
                               ValueCategory valueCategory, const Type* type)
    -> BracedInitListAST* {
  auto node = new (arena) BracedInitListAST();
  node->expressionList = expressionList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ParenInitializerAST::clone(Arena* arena) -> ParenInitializerAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;

  return node;
}

auto ParenInitializerAST::create(Arena* arena) -> ParenInitializerAST* {
  auto node = new (arena) ParenInitializerAST();
  return node;
}

auto ParenInitializerAST::create(Arena* arena, SourceLocation lparenLoc,
                                 List<ExpressionAST*>* expressionList,
                                 SourceLocation rparenLoc,
                                 ValueCategory valueCategory, const Type* type)
    -> ParenInitializerAST* {
  auto node = new (arena) ParenInitializerAST();
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto ParenInitializerAST::create(Arena* arena,
                                 List<ExpressionAST*>* expressionList,
                                 ValueCategory valueCategory, const Type* type)
    -> ParenInitializerAST* {
  auto node = new (arena) ParenInitializerAST();
  node->expressionList = expressionList;
  node->valueCategory = valueCategory;
  node->type = type;
  return node;
}

auto DefaultGenericAssociationAST::clone(Arena* arena)
    -> DefaultGenericAssociationAST* {
  auto node = create(arena);

  node->defaultLoc = defaultLoc;
  node->colonLoc = colonLoc;

  if (expression) node->expression = expression->clone(arena);

  return node;
}

auto DefaultGenericAssociationAST::create(Arena* arena)
    -> DefaultGenericAssociationAST* {
  auto node = new (arena) DefaultGenericAssociationAST();
  return node;
}

auto DefaultGenericAssociationAST::create(Arena* arena,
                                          SourceLocation defaultLoc,
                                          SourceLocation colonLoc,
                                          ExpressionAST* expression)
    -> DefaultGenericAssociationAST* {
  auto node = new (arena) DefaultGenericAssociationAST();
  node->defaultLoc = defaultLoc;
  node->colonLoc = colonLoc;
  node->expression = expression;
  return node;
}

auto DefaultGenericAssociationAST::create(Arena* arena,
                                          ExpressionAST* expression)
    -> DefaultGenericAssociationAST* {
  auto node = new (arena) DefaultGenericAssociationAST();
  node->expression = expression;
  return node;
}

auto TypeGenericAssociationAST::clone(Arena* arena)
    -> TypeGenericAssociationAST* {
  auto node = create(arena);

  if (typeId) node->typeId = typeId->clone(arena);

  node->colonLoc = colonLoc;

  if (expression) node->expression = expression->clone(arena);

  return node;
}

auto TypeGenericAssociationAST::create(Arena* arena)
    -> TypeGenericAssociationAST* {
  auto node = new (arena) TypeGenericAssociationAST();
  return node;
}

auto TypeGenericAssociationAST::create(Arena* arena, TypeIdAST* typeId,
                                       SourceLocation colonLoc,
                                       ExpressionAST* expression)
    -> TypeGenericAssociationAST* {
  auto node = new (arena) TypeGenericAssociationAST();
  node->typeId = typeId;
  node->colonLoc = colonLoc;
  node->expression = expression;
  return node;
}

auto TypeGenericAssociationAST::create(Arena* arena, TypeIdAST* typeId,
                                       ExpressionAST* expression)
    -> TypeGenericAssociationAST* {
  auto node = new (arena) TypeGenericAssociationAST();
  node->typeId = typeId;
  node->expression = expression;
  return node;
}

auto DotDesignatorAST::clone(Arena* arena) -> DotDesignatorAST* {
  auto node = create(arena);

  node->dotLoc = dotLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;

  return node;
}

auto DotDesignatorAST::create(Arena* arena) -> DotDesignatorAST* {
  auto node = new (arena) DotDesignatorAST();
  return node;
}

auto DotDesignatorAST::create(Arena* arena, SourceLocation dotLoc,
                              SourceLocation identifierLoc,
                              const Identifier* identifier)
    -> DotDesignatorAST* {
  auto node = new (arena) DotDesignatorAST();
  node->dotLoc = dotLoc;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  return node;
}

auto DotDesignatorAST::create(Arena* arena, const Identifier* identifier)
    -> DotDesignatorAST* {
  auto node = new (arena) DotDesignatorAST();
  node->identifier = identifier;
  return node;
}

auto SubscriptDesignatorAST::clone(Arena* arena) -> SubscriptDesignatorAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rbracketLoc = rbracketLoc;

  return node;
}

auto SubscriptDesignatorAST::create(Arena* arena) -> SubscriptDesignatorAST* {
  auto node = new (arena) SubscriptDesignatorAST();
  return node;
}

auto SubscriptDesignatorAST::create(Arena* arena, SourceLocation lbracketLoc,
                                    ExpressionAST* expression,
                                    SourceLocation rbracketLoc)
    -> SubscriptDesignatorAST* {
  auto node = new (arena) SubscriptDesignatorAST();
  node->lbracketLoc = lbracketLoc;
  node->expression = expression;
  node->rbracketLoc = rbracketLoc;
  return node;
}

auto SubscriptDesignatorAST::create(Arena* arena, ExpressionAST* expression)
    -> SubscriptDesignatorAST* {
  auto node = new (arena) SubscriptDesignatorAST();
  node->expression = expression;
  return node;
}

auto TemplateTypeParameterAST::clone(Arena* arena)
    -> TemplateTypeParameterAST* {
  auto node = create(arena);

  node->templateLoc = templateLoc;
  node->lessLoc = lessLoc;

  if (templateParameterList) {
    auto it = &node->templateParameterList;
    for (auto node : ListView{templateParameterList}) {
      *it = make_list_node<TemplateParameterAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;

  if (requiresClause) node->requiresClause = requiresClause->clone(arena);

  node->classKeyLoc = classKeyLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;

  if (idExpression) node->idExpression = idExpression->clone(arena);

  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;

  return node;
}

auto TemplateTypeParameterAST::create(Arena* arena)
    -> TemplateTypeParameterAST* {
  auto node = new (arena) TemplateTypeParameterAST();
  return node;
}

auto TemplateTypeParameterAST::create(
    Arena* arena, SourceLocation templateLoc, SourceLocation lessLoc,
    List<TemplateParameterAST*>* templateParameterList,
    SourceLocation greaterLoc, RequiresClauseAST* requiresClause,
    SourceLocation classKeyLoc, SourceLocation ellipsisLoc,
    SourceLocation identifierLoc, SourceLocation equalLoc,
    IdExpressionAST* idExpression, const Identifier* identifier, bool isPack,
    Symbol* symbol, int depth, int index) -> TemplateTypeParameterAST* {
  auto node = new (arena) TemplateTypeParameterAST();
  node->templateLoc = templateLoc;
  node->lessLoc = lessLoc;
  node->templateParameterList = templateParameterList;
  node->greaterLoc = greaterLoc;
  node->requiresClause = requiresClause;
  node->classKeyLoc = classKeyLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;
  node->idExpression = idExpression;
  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto TemplateTypeParameterAST::create(
    Arena* arena, List<TemplateParameterAST*>* templateParameterList,
    RequiresClauseAST* requiresClause, IdExpressionAST* idExpression,
    const Identifier* identifier, bool isPack, Symbol* symbol, int depth,
    int index) -> TemplateTypeParameterAST* {
  auto node = new (arena) TemplateTypeParameterAST();
  node->templateParameterList = templateParameterList;
  node->requiresClause = requiresClause;
  node->idExpression = idExpression;
  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto NonTypeTemplateParameterAST::clone(Arena* arena)
    -> NonTypeTemplateParameterAST* {
  auto node = create(arena);

  if (declaration) node->declaration = declaration->clone(arena);

  node->symbol = symbol;
  node->depth = depth;
  node->index = index;

  return node;
}

auto NonTypeTemplateParameterAST::create(Arena* arena)
    -> NonTypeTemplateParameterAST* {
  auto node = new (arena) NonTypeTemplateParameterAST();
  return node;
}

auto NonTypeTemplateParameterAST::create(Arena* arena,
                                         ParameterDeclarationAST* declaration,
                                         Symbol* symbol, int depth, int index)
    -> NonTypeTemplateParameterAST* {
  auto node = new (arena) NonTypeTemplateParameterAST();
  node->declaration = declaration;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto TypenameTypeParameterAST::clone(Arena* arena)
    -> TypenameTypeParameterAST* {
  auto node = create(arena);

  node->classKeyLoc = classKeyLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;

  return node;
}

auto TypenameTypeParameterAST::create(Arena* arena)
    -> TypenameTypeParameterAST* {
  auto node = new (arena) TypenameTypeParameterAST();
  return node;
}

auto TypenameTypeParameterAST::create(
    Arena* arena, SourceLocation classKeyLoc, SourceLocation ellipsisLoc,
    SourceLocation identifierLoc, SourceLocation equalLoc, TypeIdAST* typeId,
    const Identifier* identifier, bool isPack, Symbol* symbol, int depth,
    int index) -> TypenameTypeParameterAST* {
  auto node = new (arena) TypenameTypeParameterAST();
  node->classKeyLoc = classKeyLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;
  node->typeId = typeId;
  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto TypenameTypeParameterAST::create(Arena* arena, TypeIdAST* typeId,
                                      const Identifier* identifier, bool isPack,
                                      Symbol* symbol, int depth, int index)
    -> TypenameTypeParameterAST* {
  auto node = new (arena) TypenameTypeParameterAST();
  node->typeId = typeId;
  node->identifier = identifier;
  node->isPack = isPack;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto ConstraintTypeParameterAST::clone(Arena* arena)
    -> ConstraintTypeParameterAST* {
  auto node = create(arena);

  if (typeConstraint) node->typeConstraint = typeConstraint->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->identifier = identifier;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;

  return node;
}

auto ConstraintTypeParameterAST::create(Arena* arena)
    -> ConstraintTypeParameterAST* {
  auto node = new (arena) ConstraintTypeParameterAST();
  return node;
}

auto ConstraintTypeParameterAST::create(
    Arena* arena, TypeConstraintAST* typeConstraint, SourceLocation ellipsisLoc,
    SourceLocation identifierLoc, SourceLocation equalLoc, TypeIdAST* typeId,
    const Identifier* identifier, Symbol* symbol, int depth, int index)
    -> ConstraintTypeParameterAST* {
  auto node = new (arena) ConstraintTypeParameterAST();
  node->typeConstraint = typeConstraint;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->equalLoc = equalLoc;
  node->typeId = typeId;
  node->identifier = identifier;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto ConstraintTypeParameterAST::create(Arena* arena,
                                        TypeConstraintAST* typeConstraint,
                                        TypeIdAST* typeId,
                                        const Identifier* identifier,
                                        Symbol* symbol, int depth, int index)
    -> ConstraintTypeParameterAST* {
  auto node = new (arena) ConstraintTypeParameterAST();
  node->typeConstraint = typeConstraint;
  node->typeId = typeId;
  node->identifier = identifier;
  node->symbol = symbol;
  node->depth = depth;
  node->index = index;
  return node;
}

auto TypedefSpecifierAST::clone(Arena* arena) -> TypedefSpecifierAST* {
  auto node = create(arena);

  node->typedefLoc = typedefLoc;

  return node;
}

auto TypedefSpecifierAST::create(Arena* arena) -> TypedefSpecifierAST* {
  auto node = new (arena) TypedefSpecifierAST();
  return node;
}

auto TypedefSpecifierAST::create(Arena* arena, SourceLocation typedefLoc)
    -> TypedefSpecifierAST* {
  auto node = new (arena) TypedefSpecifierAST();
  node->typedefLoc = typedefLoc;
  return node;
}

auto FriendSpecifierAST::clone(Arena* arena) -> FriendSpecifierAST* {
  auto node = create(arena);

  node->friendLoc = friendLoc;

  return node;
}

auto FriendSpecifierAST::create(Arena* arena) -> FriendSpecifierAST* {
  auto node = new (arena) FriendSpecifierAST();
  return node;
}

auto FriendSpecifierAST::create(Arena* arena, SourceLocation friendLoc)
    -> FriendSpecifierAST* {
  auto node = new (arena) FriendSpecifierAST();
  node->friendLoc = friendLoc;
  return node;
}

auto ConstevalSpecifierAST::clone(Arena* arena) -> ConstevalSpecifierAST* {
  auto node = create(arena);

  node->constevalLoc = constevalLoc;

  return node;
}

auto ConstevalSpecifierAST::create(Arena* arena) -> ConstevalSpecifierAST* {
  auto node = new (arena) ConstevalSpecifierAST();
  return node;
}

auto ConstevalSpecifierAST::create(Arena* arena, SourceLocation constevalLoc)
    -> ConstevalSpecifierAST* {
  auto node = new (arena) ConstevalSpecifierAST();
  node->constevalLoc = constevalLoc;
  return node;
}

auto ConstinitSpecifierAST::clone(Arena* arena) -> ConstinitSpecifierAST* {
  auto node = create(arena);

  node->constinitLoc = constinitLoc;

  return node;
}

auto ConstinitSpecifierAST::create(Arena* arena) -> ConstinitSpecifierAST* {
  auto node = new (arena) ConstinitSpecifierAST();
  return node;
}

auto ConstinitSpecifierAST::create(Arena* arena, SourceLocation constinitLoc)
    -> ConstinitSpecifierAST* {
  auto node = new (arena) ConstinitSpecifierAST();
  node->constinitLoc = constinitLoc;
  return node;
}

auto ConstexprSpecifierAST::clone(Arena* arena) -> ConstexprSpecifierAST* {
  auto node = create(arena);

  node->constexprLoc = constexprLoc;

  return node;
}

auto ConstexprSpecifierAST::create(Arena* arena) -> ConstexprSpecifierAST* {
  auto node = new (arena) ConstexprSpecifierAST();
  return node;
}

auto ConstexprSpecifierAST::create(Arena* arena, SourceLocation constexprLoc)
    -> ConstexprSpecifierAST* {
  auto node = new (arena) ConstexprSpecifierAST();
  node->constexprLoc = constexprLoc;
  return node;
}

auto InlineSpecifierAST::clone(Arena* arena) -> InlineSpecifierAST* {
  auto node = create(arena);

  node->inlineLoc = inlineLoc;

  return node;
}

auto InlineSpecifierAST::create(Arena* arena) -> InlineSpecifierAST* {
  auto node = new (arena) InlineSpecifierAST();
  return node;
}

auto InlineSpecifierAST::create(Arena* arena, SourceLocation inlineLoc)
    -> InlineSpecifierAST* {
  auto node = new (arena) InlineSpecifierAST();
  node->inlineLoc = inlineLoc;
  return node;
}

auto NoreturnSpecifierAST::clone(Arena* arena) -> NoreturnSpecifierAST* {
  auto node = create(arena);

  node->noreturnLoc = noreturnLoc;

  return node;
}

auto NoreturnSpecifierAST::create(Arena* arena) -> NoreturnSpecifierAST* {
  auto node = new (arena) NoreturnSpecifierAST();
  return node;
}

auto NoreturnSpecifierAST::create(Arena* arena, SourceLocation noreturnLoc)
    -> NoreturnSpecifierAST* {
  auto node = new (arena) NoreturnSpecifierAST();
  node->noreturnLoc = noreturnLoc;
  return node;
}

auto StaticSpecifierAST::clone(Arena* arena) -> StaticSpecifierAST* {
  auto node = create(arena);

  node->staticLoc = staticLoc;

  return node;
}

auto StaticSpecifierAST::create(Arena* arena) -> StaticSpecifierAST* {
  auto node = new (arena) StaticSpecifierAST();
  return node;
}

auto StaticSpecifierAST::create(Arena* arena, SourceLocation staticLoc)
    -> StaticSpecifierAST* {
  auto node = new (arena) StaticSpecifierAST();
  node->staticLoc = staticLoc;
  return node;
}

auto ExternSpecifierAST::clone(Arena* arena) -> ExternSpecifierAST* {
  auto node = create(arena);

  node->externLoc = externLoc;

  return node;
}

auto ExternSpecifierAST::create(Arena* arena) -> ExternSpecifierAST* {
  auto node = new (arena) ExternSpecifierAST();
  return node;
}

auto ExternSpecifierAST::create(Arena* arena, SourceLocation externLoc)
    -> ExternSpecifierAST* {
  auto node = new (arena) ExternSpecifierAST();
  node->externLoc = externLoc;
  return node;
}

auto RegisterSpecifierAST::clone(Arena* arena) -> RegisterSpecifierAST* {
  auto node = create(arena);

  node->registerLoc = registerLoc;

  return node;
}

auto RegisterSpecifierAST::create(Arena* arena) -> RegisterSpecifierAST* {
  auto node = new (arena) RegisterSpecifierAST();
  return node;
}

auto RegisterSpecifierAST::create(Arena* arena, SourceLocation registerLoc)
    -> RegisterSpecifierAST* {
  auto node = new (arena) RegisterSpecifierAST();
  node->registerLoc = registerLoc;
  return node;
}

auto ThreadLocalSpecifierAST::clone(Arena* arena) -> ThreadLocalSpecifierAST* {
  auto node = create(arena);

  node->threadLocalLoc = threadLocalLoc;

  return node;
}

auto ThreadLocalSpecifierAST::create(Arena* arena) -> ThreadLocalSpecifierAST* {
  auto node = new (arena) ThreadLocalSpecifierAST();
  return node;
}

auto ThreadLocalSpecifierAST::create(Arena* arena,
                                     SourceLocation threadLocalLoc)
    -> ThreadLocalSpecifierAST* {
  auto node = new (arena) ThreadLocalSpecifierAST();
  node->threadLocalLoc = threadLocalLoc;
  return node;
}

auto ThreadSpecifierAST::clone(Arena* arena) -> ThreadSpecifierAST* {
  auto node = create(arena);

  node->threadLoc = threadLoc;

  return node;
}

auto ThreadSpecifierAST::create(Arena* arena) -> ThreadSpecifierAST* {
  auto node = new (arena) ThreadSpecifierAST();
  return node;
}

auto ThreadSpecifierAST::create(Arena* arena, SourceLocation threadLoc)
    -> ThreadSpecifierAST* {
  auto node = new (arena) ThreadSpecifierAST();
  node->threadLoc = threadLoc;
  return node;
}

auto MutableSpecifierAST::clone(Arena* arena) -> MutableSpecifierAST* {
  auto node = create(arena);

  node->mutableLoc = mutableLoc;

  return node;
}

auto MutableSpecifierAST::create(Arena* arena) -> MutableSpecifierAST* {
  auto node = new (arena) MutableSpecifierAST();
  return node;
}

auto MutableSpecifierAST::create(Arena* arena, SourceLocation mutableLoc)
    -> MutableSpecifierAST* {
  auto node = new (arena) MutableSpecifierAST();
  node->mutableLoc = mutableLoc;
  return node;
}

auto VirtualSpecifierAST::clone(Arena* arena) -> VirtualSpecifierAST* {
  auto node = create(arena);

  node->virtualLoc = virtualLoc;

  return node;
}

auto VirtualSpecifierAST::create(Arena* arena) -> VirtualSpecifierAST* {
  auto node = new (arena) VirtualSpecifierAST();
  return node;
}

auto VirtualSpecifierAST::create(Arena* arena, SourceLocation virtualLoc)
    -> VirtualSpecifierAST* {
  auto node = new (arena) VirtualSpecifierAST();
  node->virtualLoc = virtualLoc;
  return node;
}

auto ExplicitSpecifierAST::clone(Arena* arena) -> ExplicitSpecifierAST* {
  auto node = create(arena);

  node->explicitLoc = explicitLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;

  return node;
}

auto ExplicitSpecifierAST::create(Arena* arena) -> ExplicitSpecifierAST* {
  auto node = new (arena) ExplicitSpecifierAST();
  return node;
}

auto ExplicitSpecifierAST::create(Arena* arena, SourceLocation explicitLoc,
                                  SourceLocation lparenLoc,
                                  ExpressionAST* expression,
                                  SourceLocation rparenLoc)
    -> ExplicitSpecifierAST* {
  auto node = new (arena) ExplicitSpecifierAST();
  node->explicitLoc = explicitLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  return node;
}

auto ExplicitSpecifierAST::create(Arena* arena, ExpressionAST* expression)
    -> ExplicitSpecifierAST* {
  auto node = new (arena) ExplicitSpecifierAST();
  node->expression = expression;
  return node;
}

auto AutoTypeSpecifierAST::clone(Arena* arena) -> AutoTypeSpecifierAST* {
  auto node = create(arena);

  node->autoLoc = autoLoc;

  return node;
}

auto AutoTypeSpecifierAST::create(Arena* arena) -> AutoTypeSpecifierAST* {
  auto node = new (arena) AutoTypeSpecifierAST();
  return node;
}

auto AutoTypeSpecifierAST::create(Arena* arena, SourceLocation autoLoc)
    -> AutoTypeSpecifierAST* {
  auto node = new (arena) AutoTypeSpecifierAST();
  node->autoLoc = autoLoc;
  return node;
}

auto VoidTypeSpecifierAST::clone(Arena* arena) -> VoidTypeSpecifierAST* {
  auto node = create(arena);

  node->voidLoc = voidLoc;

  return node;
}

auto VoidTypeSpecifierAST::create(Arena* arena) -> VoidTypeSpecifierAST* {
  auto node = new (arena) VoidTypeSpecifierAST();
  return node;
}

auto VoidTypeSpecifierAST::create(Arena* arena, SourceLocation voidLoc)
    -> VoidTypeSpecifierAST* {
  auto node = new (arena) VoidTypeSpecifierAST();
  node->voidLoc = voidLoc;
  return node;
}

auto SizeTypeSpecifierAST::clone(Arena* arena) -> SizeTypeSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto SizeTypeSpecifierAST::create(Arena* arena) -> SizeTypeSpecifierAST* {
  auto node = new (arena) SizeTypeSpecifierAST();
  return node;
}

auto SizeTypeSpecifierAST::create(Arena* arena, SourceLocation specifierLoc,
                                  TokenKind specifier)
    -> SizeTypeSpecifierAST* {
  auto node = new (arena) SizeTypeSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto SizeTypeSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> SizeTypeSpecifierAST* {
  auto node = new (arena) SizeTypeSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto SignTypeSpecifierAST::clone(Arena* arena) -> SignTypeSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto SignTypeSpecifierAST::create(Arena* arena) -> SignTypeSpecifierAST* {
  auto node = new (arena) SignTypeSpecifierAST();
  return node;
}

auto SignTypeSpecifierAST::create(Arena* arena, SourceLocation specifierLoc,
                                  TokenKind specifier)
    -> SignTypeSpecifierAST* {
  auto node = new (arena) SignTypeSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto SignTypeSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> SignTypeSpecifierAST* {
  auto node = new (arena) SignTypeSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto BuiltinTypeSpecifierAST::clone(Arena* arena) -> BuiltinTypeSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto BuiltinTypeSpecifierAST::create(Arena* arena) -> BuiltinTypeSpecifierAST* {
  auto node = new (arena) BuiltinTypeSpecifierAST();
  return node;
}

auto BuiltinTypeSpecifierAST::create(Arena* arena, SourceLocation specifierLoc,
                                     TokenKind specifier)
    -> BuiltinTypeSpecifierAST* {
  auto node = new (arena) BuiltinTypeSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto BuiltinTypeSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> BuiltinTypeSpecifierAST* {
  auto node = new (arena) BuiltinTypeSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto UnaryBuiltinTypeSpecifierAST::clone(Arena* arena)
    -> UnaryBuiltinTypeSpecifierAST* {
  auto node = create(arena);

  node->builtinLoc = builtinLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->builtinKind = builtinKind;

  return node;
}

auto UnaryBuiltinTypeSpecifierAST::create(Arena* arena)
    -> UnaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) UnaryBuiltinTypeSpecifierAST();
  return node;
}

auto UnaryBuiltinTypeSpecifierAST::create(
    Arena* arena, SourceLocation builtinLoc, SourceLocation lparenLoc,
    TypeIdAST* typeId, SourceLocation rparenLoc,
    UnaryBuiltinTypeKind builtinKind) -> UnaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) UnaryBuiltinTypeSpecifierAST();
  node->builtinLoc = builtinLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  node->builtinKind = builtinKind;
  return node;
}

auto UnaryBuiltinTypeSpecifierAST::create(Arena* arena, TypeIdAST* typeId,
                                          UnaryBuiltinTypeKind builtinKind)
    -> UnaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) UnaryBuiltinTypeSpecifierAST();
  node->typeId = typeId;
  node->builtinKind = builtinKind;
  return node;
}

auto BinaryBuiltinTypeSpecifierAST::clone(Arena* arena)
    -> BinaryBuiltinTypeSpecifierAST* {
  auto node = create(arena);

  node->builtinLoc = builtinLoc;
  node->lparenLoc = lparenLoc;

  if (leftTypeId) node->leftTypeId = leftTypeId->clone(arena);

  node->commaLoc = commaLoc;

  if (rightTypeId) node->rightTypeId = rightTypeId->clone(arena);

  node->rparenLoc = rparenLoc;
  node->builtinKind = builtinKind;

  return node;
}

auto BinaryBuiltinTypeSpecifierAST::create(Arena* arena)
    -> BinaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) BinaryBuiltinTypeSpecifierAST();
  return node;
}

auto BinaryBuiltinTypeSpecifierAST::create(
    Arena* arena, SourceLocation builtinLoc, SourceLocation lparenLoc,
    TypeIdAST* leftTypeId, SourceLocation commaLoc, TypeIdAST* rightTypeId,
    SourceLocation rparenLoc, BinaryBuiltinTypeKind builtinKind)
    -> BinaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) BinaryBuiltinTypeSpecifierAST();
  node->builtinLoc = builtinLoc;
  node->lparenLoc = lparenLoc;
  node->leftTypeId = leftTypeId;
  node->commaLoc = commaLoc;
  node->rightTypeId = rightTypeId;
  node->rparenLoc = rparenLoc;
  node->builtinKind = builtinKind;
  return node;
}

auto BinaryBuiltinTypeSpecifierAST::create(Arena* arena, TypeIdAST* leftTypeId,
                                           TypeIdAST* rightTypeId,
                                           BinaryBuiltinTypeKind builtinKind)
    -> BinaryBuiltinTypeSpecifierAST* {
  auto node = new (arena) BinaryBuiltinTypeSpecifierAST();
  node->leftTypeId = leftTypeId;
  node->rightTypeId = rightTypeId;
  node->builtinKind = builtinKind;
  return node;
}

auto IntegralTypeSpecifierAST::clone(Arena* arena)
    -> IntegralTypeSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto IntegralTypeSpecifierAST::create(Arena* arena)
    -> IntegralTypeSpecifierAST* {
  auto node = new (arena) IntegralTypeSpecifierAST();
  return node;
}

auto IntegralTypeSpecifierAST::create(Arena* arena, SourceLocation specifierLoc,
                                      TokenKind specifier)
    -> IntegralTypeSpecifierAST* {
  auto node = new (arena) IntegralTypeSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto IntegralTypeSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> IntegralTypeSpecifierAST* {
  auto node = new (arena) IntegralTypeSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto FloatingPointTypeSpecifierAST::clone(Arena* arena)
    -> FloatingPointTypeSpecifierAST* {
  auto node = create(arena);

  node->specifierLoc = specifierLoc;
  node->specifier = specifier;

  return node;
}

auto FloatingPointTypeSpecifierAST::create(Arena* arena)
    -> FloatingPointTypeSpecifierAST* {
  auto node = new (arena) FloatingPointTypeSpecifierAST();
  return node;
}

auto FloatingPointTypeSpecifierAST::create(Arena* arena,
                                           SourceLocation specifierLoc,
                                           TokenKind specifier)
    -> FloatingPointTypeSpecifierAST* {
  auto node = new (arena) FloatingPointTypeSpecifierAST();
  node->specifierLoc = specifierLoc;
  node->specifier = specifier;
  return node;
}

auto FloatingPointTypeSpecifierAST::create(Arena* arena, TokenKind specifier)
    -> FloatingPointTypeSpecifierAST* {
  auto node = new (arena) FloatingPointTypeSpecifierAST();
  node->specifier = specifier;
  return node;
}

auto ComplexTypeSpecifierAST::clone(Arena* arena) -> ComplexTypeSpecifierAST* {
  auto node = create(arena);

  node->complexLoc = complexLoc;

  return node;
}

auto ComplexTypeSpecifierAST::create(Arena* arena) -> ComplexTypeSpecifierAST* {
  auto node = new (arena) ComplexTypeSpecifierAST();
  return node;
}

auto ComplexTypeSpecifierAST::create(Arena* arena, SourceLocation complexLoc)
    -> ComplexTypeSpecifierAST* {
  auto node = new (arena) ComplexTypeSpecifierAST();
  node->complexLoc = complexLoc;
  return node;
}

auto NamedTypeSpecifierAST::clone(Arena* arena) -> NamedTypeSpecifierAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;

  return node;
}

auto NamedTypeSpecifierAST::create(Arena* arena) -> NamedTypeSpecifierAST* {
  auto node = new (arena) NamedTypeSpecifierAST();
  return node;
}

auto NamedTypeSpecifierAST::create(Arena* arena,
                                   NestedNameSpecifierAST* nestedNameSpecifier,
                                   SourceLocation templateLoc,
                                   UnqualifiedIdAST* unqualifiedId,
                                   bool isTemplateIntroduced, Symbol* symbol)
    -> NamedTypeSpecifierAST* {
  auto node = new (arena) NamedTypeSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto NamedTypeSpecifierAST::create(Arena* arena,
                                   NestedNameSpecifierAST* nestedNameSpecifier,
                                   UnqualifiedIdAST* unqualifiedId,
                                   bool isTemplateIntroduced, Symbol* symbol)
    -> NamedTypeSpecifierAST* {
  auto node = new (arena) NamedTypeSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto AtomicTypeSpecifierAST::clone(Arena* arena) -> AtomicTypeSpecifierAST* {
  auto node = create(arena);

  node->atomicLoc = atomicLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;

  return node;
}

auto AtomicTypeSpecifierAST::create(Arena* arena) -> AtomicTypeSpecifierAST* {
  auto node = new (arena) AtomicTypeSpecifierAST();
  return node;
}

auto AtomicTypeSpecifierAST::create(Arena* arena, SourceLocation atomicLoc,
                                    SourceLocation lparenLoc, TypeIdAST* typeId,
                                    SourceLocation rparenLoc)
    -> AtomicTypeSpecifierAST* {
  auto node = new (arena) AtomicTypeSpecifierAST();
  node->atomicLoc = atomicLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  return node;
}

auto AtomicTypeSpecifierAST::create(Arena* arena, TypeIdAST* typeId)
    -> AtomicTypeSpecifierAST* {
  auto node = new (arena) AtomicTypeSpecifierAST();
  node->typeId = typeId;
  return node;
}

auto UnderlyingTypeSpecifierAST::clone(Arena* arena)
    -> UnderlyingTypeSpecifierAST* {
  auto node = create(arena);

  node->underlyingTypeLoc = underlyingTypeLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->rparenLoc = rparenLoc;

  return node;
}

auto UnderlyingTypeSpecifierAST::create(Arena* arena)
    -> UnderlyingTypeSpecifierAST* {
  auto node = new (arena) UnderlyingTypeSpecifierAST();
  return node;
}

auto UnderlyingTypeSpecifierAST::create(Arena* arena,
                                        SourceLocation underlyingTypeLoc,
                                        SourceLocation lparenLoc,
                                        TypeIdAST* typeId,
                                        SourceLocation rparenLoc)
    -> UnderlyingTypeSpecifierAST* {
  auto node = new (arena) UnderlyingTypeSpecifierAST();
  node->underlyingTypeLoc = underlyingTypeLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->rparenLoc = rparenLoc;
  return node;
}

auto UnderlyingTypeSpecifierAST::create(Arena* arena, TypeIdAST* typeId)
    -> UnderlyingTypeSpecifierAST* {
  auto node = new (arena) UnderlyingTypeSpecifierAST();
  node->typeId = typeId;
  return node;
}

auto ElaboratedTypeSpecifierAST::clone(Arena* arena)
    -> ElaboratedTypeSpecifierAST* {
  auto node = create(arena);

  node->classLoc = classLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->classKey = classKey;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;

  return node;
}

auto ElaboratedTypeSpecifierAST::create(Arena* arena)
    -> ElaboratedTypeSpecifierAST* {
  auto node = new (arena) ElaboratedTypeSpecifierAST();
  return node;
}

auto ElaboratedTypeSpecifierAST::create(
    Arena* arena, SourceLocation classLoc,
    List<AttributeSpecifierAST*>* attributeList,
    NestedNameSpecifierAST* nestedNameSpecifier, SourceLocation templateLoc,
    UnqualifiedIdAST* unqualifiedId, TokenKind classKey,
    bool isTemplateIntroduced, Symbol* symbol) -> ElaboratedTypeSpecifierAST* {
  auto node = new (arena) ElaboratedTypeSpecifierAST();
  node->classLoc = classLoc;
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->classKey = classKey;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto ElaboratedTypeSpecifierAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, TokenKind classKey,
    bool isTemplateIntroduced, Symbol* symbol) -> ElaboratedTypeSpecifierAST* {
  auto node = new (arena) ElaboratedTypeSpecifierAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->classKey = classKey;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto DecltypeAutoSpecifierAST::clone(Arena* arena)
    -> DecltypeAutoSpecifierAST* {
  auto node = create(arena);

  node->decltypeLoc = decltypeLoc;
  node->lparenLoc = lparenLoc;
  node->autoLoc = autoLoc;
  node->rparenLoc = rparenLoc;

  return node;
}

auto DecltypeAutoSpecifierAST::create(Arena* arena)
    -> DecltypeAutoSpecifierAST* {
  auto node = new (arena) DecltypeAutoSpecifierAST();
  return node;
}

auto DecltypeAutoSpecifierAST::create(Arena* arena, SourceLocation decltypeLoc,
                                      SourceLocation lparenLoc,
                                      SourceLocation autoLoc,
                                      SourceLocation rparenLoc)
    -> DecltypeAutoSpecifierAST* {
  auto node = new (arena) DecltypeAutoSpecifierAST();
  node->decltypeLoc = decltypeLoc;
  node->lparenLoc = lparenLoc;
  node->autoLoc = autoLoc;
  node->rparenLoc = rparenLoc;
  return node;
}

auto DecltypeSpecifierAST::clone(Arena* arena) -> DecltypeSpecifierAST* {
  auto node = create(arena);

  node->decltypeLoc = decltypeLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;
  node->type = type;

  return node;
}

auto DecltypeSpecifierAST::create(Arena* arena) -> DecltypeSpecifierAST* {
  auto node = new (arena) DecltypeSpecifierAST();
  return node;
}

auto DecltypeSpecifierAST::create(Arena* arena, SourceLocation decltypeLoc,
                                  SourceLocation lparenLoc,
                                  ExpressionAST* expression,
                                  SourceLocation rparenLoc, const Type* type)
    -> DecltypeSpecifierAST* {
  auto node = new (arena) DecltypeSpecifierAST();
  node->decltypeLoc = decltypeLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  node->type = type;
  return node;
}

auto DecltypeSpecifierAST::create(Arena* arena, ExpressionAST* expression,
                                  const Type* type) -> DecltypeSpecifierAST* {
  auto node = new (arena) DecltypeSpecifierAST();
  node->expression = expression;
  node->type = type;
  return node;
}

auto PlaceholderTypeSpecifierAST::clone(Arena* arena)
    -> PlaceholderTypeSpecifierAST* {
  auto node = create(arena);

  if (typeConstraint) node->typeConstraint = typeConstraint->clone(arena);

  if (specifier) node->specifier = specifier->clone(arena);

  return node;
}

auto PlaceholderTypeSpecifierAST::create(Arena* arena)
    -> PlaceholderTypeSpecifierAST* {
  auto node = new (arena) PlaceholderTypeSpecifierAST();
  return node;
}

auto PlaceholderTypeSpecifierAST::create(Arena* arena,
                                         TypeConstraintAST* typeConstraint,
                                         SpecifierAST* specifier)
    -> PlaceholderTypeSpecifierAST* {
  auto node = new (arena) PlaceholderTypeSpecifierAST();
  node->typeConstraint = typeConstraint;
  node->specifier = specifier;
  return node;
}

auto ConstQualifierAST::clone(Arena* arena) -> ConstQualifierAST* {
  auto node = create(arena);

  node->constLoc = constLoc;

  return node;
}

auto ConstQualifierAST::create(Arena* arena) -> ConstQualifierAST* {
  auto node = new (arena) ConstQualifierAST();
  return node;
}

auto ConstQualifierAST::create(Arena* arena, SourceLocation constLoc)
    -> ConstQualifierAST* {
  auto node = new (arena) ConstQualifierAST();
  node->constLoc = constLoc;
  return node;
}

auto VolatileQualifierAST::clone(Arena* arena) -> VolatileQualifierAST* {
  auto node = create(arena);

  node->volatileLoc = volatileLoc;

  return node;
}

auto VolatileQualifierAST::create(Arena* arena) -> VolatileQualifierAST* {
  auto node = new (arena) VolatileQualifierAST();
  return node;
}

auto VolatileQualifierAST::create(Arena* arena, SourceLocation volatileLoc)
    -> VolatileQualifierAST* {
  auto node = new (arena) VolatileQualifierAST();
  node->volatileLoc = volatileLoc;
  return node;
}

auto AtomicQualifierAST::clone(Arena* arena) -> AtomicQualifierAST* {
  auto node = create(arena);

  node->atomicLoc = atomicLoc;

  return node;
}

auto AtomicQualifierAST::create(Arena* arena) -> AtomicQualifierAST* {
  auto node = new (arena) AtomicQualifierAST();
  return node;
}

auto AtomicQualifierAST::create(Arena* arena, SourceLocation atomicLoc)
    -> AtomicQualifierAST* {
  auto node = new (arena) AtomicQualifierAST();
  node->atomicLoc = atomicLoc;
  return node;
}

auto RestrictQualifierAST::clone(Arena* arena) -> RestrictQualifierAST* {
  auto node = create(arena);

  node->restrictLoc = restrictLoc;

  return node;
}

auto RestrictQualifierAST::create(Arena* arena) -> RestrictQualifierAST* {
  auto node = new (arena) RestrictQualifierAST();
  return node;
}

auto RestrictQualifierAST::create(Arena* arena, SourceLocation restrictLoc)
    -> RestrictQualifierAST* {
  auto node = new (arena) RestrictQualifierAST();
  node->restrictLoc = restrictLoc;
  return node;
}

auto EnumSpecifierAST::clone(Arena* arena) -> EnumSpecifierAST* {
  auto node = create(arena);

  node->enumLoc = enumLoc;
  node->classLoc = classLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->colonLoc = colonLoc;

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->lbraceLoc = lbraceLoc;

  if (enumeratorList) {
    auto it = &node->enumeratorList;
    for (auto node : ListView{enumeratorList}) {
      *it = make_list_node<EnumeratorAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->commaLoc = commaLoc;
  node->rbraceLoc = rbraceLoc;
  node->symbol = symbol;

  return node;
}

auto EnumSpecifierAST::create(Arena* arena) -> EnumSpecifierAST* {
  auto node = new (arena) EnumSpecifierAST();
  return node;
}

auto EnumSpecifierAST::create(Arena* arena, SourceLocation enumLoc,
                              SourceLocation classLoc,
                              List<AttributeSpecifierAST*>* attributeList,
                              NestedNameSpecifierAST* nestedNameSpecifier,
                              NameIdAST* unqualifiedId, SourceLocation colonLoc,
                              List<SpecifierAST*>* typeSpecifierList,
                              SourceLocation lbraceLoc,
                              List<EnumeratorAST*>* enumeratorList,
                              SourceLocation commaLoc, SourceLocation rbraceLoc,
                              Symbol* symbol) -> EnumSpecifierAST* {
  auto node = new (arena) EnumSpecifierAST();
  node->enumLoc = enumLoc;
  node->classLoc = classLoc;
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->colonLoc = colonLoc;
  node->typeSpecifierList = typeSpecifierList;
  node->lbraceLoc = lbraceLoc;
  node->enumeratorList = enumeratorList;
  node->commaLoc = commaLoc;
  node->rbraceLoc = rbraceLoc;
  node->symbol = symbol;
  return node;
}

auto EnumSpecifierAST::create(Arena* arena,
                              List<AttributeSpecifierAST*>* attributeList,
                              NestedNameSpecifierAST* nestedNameSpecifier,
                              NameIdAST* unqualifiedId,
                              List<SpecifierAST*>* typeSpecifierList,
                              List<EnumeratorAST*>* enumeratorList,
                              Symbol* symbol) -> EnumSpecifierAST* {
  auto node = new (arena) EnumSpecifierAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->typeSpecifierList = typeSpecifierList;
  node->enumeratorList = enumeratorList;
  node->symbol = symbol;
  return node;
}

auto ClassSpecifierAST::clone(Arena* arena) -> ClassSpecifierAST* {
  auto node = create(arena);

  node->classLoc = classLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->finalLoc = finalLoc;
  node->colonLoc = colonLoc;

  if (baseSpecifierList) {
    auto it = &node->baseSpecifierList;
    for (auto node : ListView{baseSpecifierList}) {
      *it = make_list_node<BaseSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->lbraceLoc = lbraceLoc;

  if (declarationList) {
    auto it = &node->declarationList;
    for (auto node : ListView{declarationList}) {
      *it = make_list_node<DeclarationAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbraceLoc = rbraceLoc;
  node->classKey = classKey;
  node->symbol = symbol;
  node->isFinal = isFinal;

  return node;
}

auto ClassSpecifierAST::create(Arena* arena) -> ClassSpecifierAST* {
  auto node = new (arena) ClassSpecifierAST();
  return node;
}

auto ClassSpecifierAST::create(Arena* arena, SourceLocation classLoc,
                               List<AttributeSpecifierAST*>* attributeList,
                               NestedNameSpecifierAST* nestedNameSpecifier,
                               UnqualifiedIdAST* unqualifiedId,
                               SourceLocation finalLoc, SourceLocation colonLoc,
                               List<BaseSpecifierAST*>* baseSpecifierList,
                               SourceLocation lbraceLoc,
                               List<DeclarationAST*>* declarationList,
                               SourceLocation rbraceLoc, TokenKind classKey,
                               ClassSymbol* symbol, bool isFinal)
    -> ClassSpecifierAST* {
  auto node = new (arena) ClassSpecifierAST();
  node->classLoc = classLoc;
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->finalLoc = finalLoc;
  node->colonLoc = colonLoc;
  node->baseSpecifierList = baseSpecifierList;
  node->lbraceLoc = lbraceLoc;
  node->declarationList = declarationList;
  node->rbraceLoc = rbraceLoc;
  node->classKey = classKey;
  node->symbol = symbol;
  node->isFinal = isFinal;
  return node;
}

auto ClassSpecifierAST::create(Arena* arena,
                               List<AttributeSpecifierAST*>* attributeList,
                               NestedNameSpecifierAST* nestedNameSpecifier,
                               UnqualifiedIdAST* unqualifiedId,
                               List<BaseSpecifierAST*>* baseSpecifierList,
                               List<DeclarationAST*>* declarationList,
                               TokenKind classKey, ClassSymbol* symbol,
                               bool isFinal) -> ClassSpecifierAST* {
  auto node = new (arena) ClassSpecifierAST();
  node->attributeList = attributeList;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->baseSpecifierList = baseSpecifierList;
  node->declarationList = declarationList;
  node->classKey = classKey;
  node->symbol = symbol;
  node->isFinal = isFinal;
  return node;
}

auto TypenameSpecifierAST::clone(Arena* arena) -> TypenameSpecifierAST* {
  auto node = create(arena);

  node->typenameLoc = typenameLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->isTemplateIntroduced = isTemplateIntroduced;

  return node;
}

auto TypenameSpecifierAST::create(Arena* arena) -> TypenameSpecifierAST* {
  auto node = new (arena) TypenameSpecifierAST();
  return node;
}

auto TypenameSpecifierAST::create(Arena* arena, SourceLocation typenameLoc,
                                  NestedNameSpecifierAST* nestedNameSpecifier,
                                  SourceLocation templateLoc,
                                  UnqualifiedIdAST* unqualifiedId,
                                  bool isTemplateIntroduced)
    -> TypenameSpecifierAST* {
  auto node = new (arena) TypenameSpecifierAST();
  node->typenameLoc = typenameLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto TypenameSpecifierAST::create(Arena* arena,
                                  NestedNameSpecifierAST* nestedNameSpecifier,
                                  UnqualifiedIdAST* unqualifiedId,
                                  bool isTemplateIntroduced)
    -> TypenameSpecifierAST* {
  auto node = new (arena) TypenameSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto SplicerTypeSpecifierAST::clone(Arena* arena) -> SplicerTypeSpecifierAST* {
  auto node = create(arena);

  node->typenameLoc = typenameLoc;

  if (splicer) node->splicer = splicer->clone(arena);

  return node;
}

auto SplicerTypeSpecifierAST::create(Arena* arena) -> SplicerTypeSpecifierAST* {
  auto node = new (arena) SplicerTypeSpecifierAST();
  return node;
}

auto SplicerTypeSpecifierAST::create(Arena* arena, SourceLocation typenameLoc,
                                     SplicerAST* splicer)
    -> SplicerTypeSpecifierAST* {
  auto node = new (arena) SplicerTypeSpecifierAST();
  node->typenameLoc = typenameLoc;
  node->splicer = splicer;
  return node;
}

auto SplicerTypeSpecifierAST::create(Arena* arena, SplicerAST* splicer)
    -> SplicerTypeSpecifierAST* {
  auto node = new (arena) SplicerTypeSpecifierAST();
  node->splicer = splicer;
  return node;
}

auto PointerOperatorAST::clone(Arena* arena) -> PointerOperatorAST* {
  auto node = create(arena);

  node->starLoc = starLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (cvQualifierList) {
    auto it = &node->cvQualifierList;
    for (auto node : ListView{cvQualifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto PointerOperatorAST::create(Arena* arena) -> PointerOperatorAST* {
  auto node = new (arena) PointerOperatorAST();
  return node;
}

auto PointerOperatorAST::create(Arena* arena, SourceLocation starLoc,
                                List<AttributeSpecifierAST*>* attributeList,
                                List<SpecifierAST*>* cvQualifierList)
    -> PointerOperatorAST* {
  auto node = new (arena) PointerOperatorAST();
  node->starLoc = starLoc;
  node->attributeList = attributeList;
  node->cvQualifierList = cvQualifierList;
  return node;
}

auto PointerOperatorAST::create(Arena* arena,
                                List<AttributeSpecifierAST*>* attributeList,
                                List<SpecifierAST*>* cvQualifierList)
    -> PointerOperatorAST* {
  auto node = new (arena) PointerOperatorAST();
  node->attributeList = attributeList;
  node->cvQualifierList = cvQualifierList;
  return node;
}

auto ReferenceOperatorAST::clone(Arena* arena) -> ReferenceOperatorAST* {
  auto node = create(arena);

  node->refLoc = refLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->refOp = refOp;

  return node;
}

auto ReferenceOperatorAST::create(Arena* arena) -> ReferenceOperatorAST* {
  auto node = new (arena) ReferenceOperatorAST();
  return node;
}

auto ReferenceOperatorAST::create(Arena* arena, SourceLocation refLoc,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  TokenKind refOp) -> ReferenceOperatorAST* {
  auto node = new (arena) ReferenceOperatorAST();
  node->refLoc = refLoc;
  node->attributeList = attributeList;
  node->refOp = refOp;
  return node;
}

auto ReferenceOperatorAST::create(Arena* arena,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  TokenKind refOp) -> ReferenceOperatorAST* {
  auto node = new (arena) ReferenceOperatorAST();
  node->attributeList = attributeList;
  node->refOp = refOp;
  return node;
}

auto PtrToMemberOperatorAST::clone(Arena* arena) -> PtrToMemberOperatorAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->starLoc = starLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (cvQualifierList) {
    auto it = &node->cvQualifierList;
    for (auto node : ListView{cvQualifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto PtrToMemberOperatorAST::create(Arena* arena) -> PtrToMemberOperatorAST* {
  auto node = new (arena) PtrToMemberOperatorAST();
  return node;
}

auto PtrToMemberOperatorAST::create(Arena* arena,
                                    NestedNameSpecifierAST* nestedNameSpecifier,
                                    SourceLocation starLoc,
                                    List<AttributeSpecifierAST*>* attributeList,
                                    List<SpecifierAST*>* cvQualifierList)
    -> PtrToMemberOperatorAST* {
  auto node = new (arena) PtrToMemberOperatorAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->starLoc = starLoc;
  node->attributeList = attributeList;
  node->cvQualifierList = cvQualifierList;
  return node;
}

auto PtrToMemberOperatorAST::create(Arena* arena,
                                    NestedNameSpecifierAST* nestedNameSpecifier,
                                    List<AttributeSpecifierAST*>* attributeList,
                                    List<SpecifierAST*>* cvQualifierList)
    -> PtrToMemberOperatorAST* {
  auto node = new (arena) PtrToMemberOperatorAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->attributeList = attributeList;
  node->cvQualifierList = cvQualifierList;
  return node;
}

auto BitfieldDeclaratorAST::clone(Arena* arena) -> BitfieldDeclaratorAST* {
  auto node = create(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->colonLoc = colonLoc;

  if (sizeExpression) node->sizeExpression = sizeExpression->clone(arena);

  return node;
}

auto BitfieldDeclaratorAST::create(Arena* arena) -> BitfieldDeclaratorAST* {
  auto node = new (arena) BitfieldDeclaratorAST();
  return node;
}

auto BitfieldDeclaratorAST::create(Arena* arena, NameIdAST* unqualifiedId,
                                   SourceLocation colonLoc,
                                   ExpressionAST* sizeExpression)
    -> BitfieldDeclaratorAST* {
  auto node = new (arena) BitfieldDeclaratorAST();
  node->unqualifiedId = unqualifiedId;
  node->colonLoc = colonLoc;
  node->sizeExpression = sizeExpression;
  return node;
}

auto BitfieldDeclaratorAST::create(Arena* arena, NameIdAST* unqualifiedId,
                                   ExpressionAST* sizeExpression)
    -> BitfieldDeclaratorAST* {
  auto node = new (arena) BitfieldDeclaratorAST();
  node->unqualifiedId = unqualifiedId;
  node->sizeExpression = sizeExpression;
  return node;
}

auto ParameterPackAST::clone(Arena* arena) -> ParameterPackAST* {
  auto node = create(arena);

  node->ellipsisLoc = ellipsisLoc;

  if (coreDeclarator) node->coreDeclarator = coreDeclarator->clone(arena);

  return node;
}

auto ParameterPackAST::create(Arena* arena) -> ParameterPackAST* {
  auto node = new (arena) ParameterPackAST();
  return node;
}

auto ParameterPackAST::create(Arena* arena, SourceLocation ellipsisLoc,
                              CoreDeclaratorAST* coreDeclarator)
    -> ParameterPackAST* {
  auto node = new (arena) ParameterPackAST();
  node->ellipsisLoc = ellipsisLoc;
  node->coreDeclarator = coreDeclarator;
  return node;
}

auto ParameterPackAST::create(Arena* arena, CoreDeclaratorAST* coreDeclarator)
    -> ParameterPackAST* {
  auto node = new (arena) ParameterPackAST();
  node->coreDeclarator = coreDeclarator;
  return node;
}

auto IdDeclaratorAST::clone(Arena* arena) -> IdDeclaratorAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->isTemplateIntroduced = isTemplateIntroduced;

  return node;
}

auto IdDeclaratorAST::create(Arena* arena) -> IdDeclaratorAST* {
  auto node = new (arena) IdDeclaratorAST();
  return node;
}

auto IdDeclaratorAST::create(Arena* arena,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             SourceLocation templateLoc,
                             UnqualifiedIdAST* unqualifiedId,
                             List<AttributeSpecifierAST*>* attributeList,
                             bool isTemplateIntroduced) -> IdDeclaratorAST* {
  auto node = new (arena) IdDeclaratorAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->attributeList = attributeList;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto IdDeclaratorAST::create(Arena* arena,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId,
                             List<AttributeSpecifierAST*>* attributeList,
                             bool isTemplateIntroduced) -> IdDeclaratorAST* {
  auto node = new (arena) IdDeclaratorAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->attributeList = attributeList;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto NestedDeclaratorAST::clone(Arena* arena) -> NestedDeclaratorAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (declarator) node->declarator = declarator->clone(arena);

  node->rparenLoc = rparenLoc;

  return node;
}

auto NestedDeclaratorAST::create(Arena* arena) -> NestedDeclaratorAST* {
  auto node = new (arena) NestedDeclaratorAST();
  return node;
}

auto NestedDeclaratorAST::create(Arena* arena, SourceLocation lparenLoc,
                                 DeclaratorAST* declarator,
                                 SourceLocation rparenLoc)
    -> NestedDeclaratorAST* {
  auto node = new (arena) NestedDeclaratorAST();
  node->lparenLoc = lparenLoc;
  node->declarator = declarator;
  node->rparenLoc = rparenLoc;
  return node;
}

auto NestedDeclaratorAST::create(Arena* arena, DeclaratorAST* declarator)
    -> NestedDeclaratorAST* {
  auto node = new (arena) NestedDeclaratorAST();
  node->declarator = declarator;
  return node;
}

auto FunctionDeclaratorChunkAST::clone(Arena* arena)
    -> FunctionDeclaratorChunkAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (parameterDeclarationClause)
    node->parameterDeclarationClause = parameterDeclarationClause->clone(arena);

  node->rparenLoc = rparenLoc;

  if (cvQualifierList) {
    auto it = &node->cvQualifierList;
    for (auto node : ListView{cvQualifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->refLoc = refLoc;

  if (exceptionSpecifier)
    node->exceptionSpecifier = exceptionSpecifier->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (trailingReturnType)
    node->trailingReturnType = trailingReturnType->clone(arena);

  node->isFinal = isFinal;
  node->isOverride = isOverride;
  node->isPure = isPure;

  return node;
}

auto FunctionDeclaratorChunkAST::create(Arena* arena)
    -> FunctionDeclaratorChunkAST* {
  auto node = new (arena) FunctionDeclaratorChunkAST();
  return node;
}

auto FunctionDeclaratorChunkAST::create(
    Arena* arena, SourceLocation lparenLoc,
    ParameterDeclarationClauseAST* parameterDeclarationClause,
    SourceLocation rparenLoc, List<SpecifierAST*>* cvQualifierList,
    SourceLocation refLoc, ExceptionSpecifierAST* exceptionSpecifier,
    List<AttributeSpecifierAST*>* attributeList,
    TrailingReturnTypeAST* trailingReturnType, bool isFinal, bool isOverride,
    bool isPure) -> FunctionDeclaratorChunkAST* {
  auto node = new (arena) FunctionDeclaratorChunkAST();
  node->lparenLoc = lparenLoc;
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->rparenLoc = rparenLoc;
  node->cvQualifierList = cvQualifierList;
  node->refLoc = refLoc;
  node->exceptionSpecifier = exceptionSpecifier;
  node->attributeList = attributeList;
  node->trailingReturnType = trailingReturnType;
  node->isFinal = isFinal;
  node->isOverride = isOverride;
  node->isPure = isPure;
  return node;
}

auto FunctionDeclaratorChunkAST::create(
    Arena* arena, ParameterDeclarationClauseAST* parameterDeclarationClause,
    List<SpecifierAST*>* cvQualifierList,
    ExceptionSpecifierAST* exceptionSpecifier,
    List<AttributeSpecifierAST*>* attributeList,
    TrailingReturnTypeAST* trailingReturnType, bool isFinal, bool isOverride,
    bool isPure) -> FunctionDeclaratorChunkAST* {
  auto node = new (arena) FunctionDeclaratorChunkAST();
  node->parameterDeclarationClause = parameterDeclarationClause;
  node->cvQualifierList = cvQualifierList;
  node->exceptionSpecifier = exceptionSpecifier;
  node->attributeList = attributeList;
  node->trailingReturnType = trailingReturnType;
  node->isFinal = isFinal;
  node->isOverride = isOverride;
  node->isPure = isPure;
  return node;
}

auto ArrayDeclaratorChunkAST::clone(Arena* arena) -> ArrayDeclaratorChunkAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;

  if (typeQualifierList) {
    auto it = &node->typeQualifierList;
    for (auto node : ListView{typeQualifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (expression) node->expression = expression->clone(arena);

  node->rbracketLoc = rbracketLoc;

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto ArrayDeclaratorChunkAST::create(Arena* arena) -> ArrayDeclaratorChunkAST* {
  auto node = new (arena) ArrayDeclaratorChunkAST();
  return node;
}

auto ArrayDeclaratorChunkAST::create(
    Arena* arena, SourceLocation lbracketLoc,
    List<SpecifierAST*>* typeQualifierList, ExpressionAST* expression,
    SourceLocation rbracketLoc, List<AttributeSpecifierAST*>* attributeList)
    -> ArrayDeclaratorChunkAST* {
  auto node = new (arena) ArrayDeclaratorChunkAST();
  node->lbracketLoc = lbracketLoc;
  node->typeQualifierList = typeQualifierList;
  node->expression = expression;
  node->rbracketLoc = rbracketLoc;
  node->attributeList = attributeList;
  return node;
}

auto ArrayDeclaratorChunkAST::create(
    Arena* arena, List<SpecifierAST*>* typeQualifierList,
    ExpressionAST* expression, List<AttributeSpecifierAST*>* attributeList)
    -> ArrayDeclaratorChunkAST* {
  auto node = new (arena) ArrayDeclaratorChunkAST();
  node->typeQualifierList = typeQualifierList;
  node->expression = expression;
  node->attributeList = attributeList;
  return node;
}

auto NameIdAST::clone(Arena* arena) -> NameIdAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->identifier = identifier;

  return node;
}

auto NameIdAST::create(Arena* arena) -> NameIdAST* {
  auto node = new (arena) NameIdAST();
  return node;
}

auto NameIdAST::create(Arena* arena, SourceLocation identifierLoc,
                       const Identifier* identifier) -> NameIdAST* {
  auto node = new (arena) NameIdAST();
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  return node;
}

auto NameIdAST::create(Arena* arena, const Identifier* identifier)
    -> NameIdAST* {
  auto node = new (arena) NameIdAST();
  node->identifier = identifier;
  return node;
}

auto DestructorIdAST::clone(Arena* arena) -> DestructorIdAST* {
  auto node = create(arena);

  node->tildeLoc = tildeLoc;

  if (id) node->id = id->clone(arena);

  return node;
}

auto DestructorIdAST::create(Arena* arena) -> DestructorIdAST* {
  auto node = new (arena) DestructorIdAST();
  return node;
}

auto DestructorIdAST::create(Arena* arena, SourceLocation tildeLoc,
                             UnqualifiedIdAST* id) -> DestructorIdAST* {
  auto node = new (arena) DestructorIdAST();
  node->tildeLoc = tildeLoc;
  node->id = id;
  return node;
}

auto DestructorIdAST::create(Arena* arena, UnqualifiedIdAST* id)
    -> DestructorIdAST* {
  auto node = new (arena) DestructorIdAST();
  node->id = id;
  return node;
}

auto DecltypeIdAST::clone(Arena* arena) -> DecltypeIdAST* {
  auto node = create(arena);

  if (decltypeSpecifier)
    node->decltypeSpecifier = decltypeSpecifier->clone(arena);

  return node;
}

auto DecltypeIdAST::create(Arena* arena) -> DecltypeIdAST* {
  auto node = new (arena) DecltypeIdAST();
  return node;
}

auto DecltypeIdAST::create(Arena* arena,
                           DecltypeSpecifierAST* decltypeSpecifier)
    -> DecltypeIdAST* {
  auto node = new (arena) DecltypeIdAST();
  node->decltypeSpecifier = decltypeSpecifier;
  return node;
}

auto OperatorFunctionIdAST::clone(Arena* arena) -> OperatorFunctionIdAST* {
  auto node = create(arena);

  node->operatorLoc = operatorLoc;
  node->opLoc = opLoc;
  node->openLoc = openLoc;
  node->closeLoc = closeLoc;
  node->op = op;

  return node;
}

auto OperatorFunctionIdAST::create(Arena* arena) -> OperatorFunctionIdAST* {
  auto node = new (arena) OperatorFunctionIdAST();
  return node;
}

auto OperatorFunctionIdAST::create(Arena* arena, SourceLocation operatorLoc,
                                   SourceLocation opLoc, SourceLocation openLoc,
                                   SourceLocation closeLoc, TokenKind op)
    -> OperatorFunctionIdAST* {
  auto node = new (arena) OperatorFunctionIdAST();
  node->operatorLoc = operatorLoc;
  node->opLoc = opLoc;
  node->openLoc = openLoc;
  node->closeLoc = closeLoc;
  node->op = op;
  return node;
}

auto OperatorFunctionIdAST::create(Arena* arena, TokenKind op)
    -> OperatorFunctionIdAST* {
  auto node = new (arena) OperatorFunctionIdAST();
  node->op = op;
  return node;
}

auto LiteralOperatorIdAST::clone(Arena* arena) -> LiteralOperatorIdAST* {
  auto node = create(arena);

  node->operatorLoc = operatorLoc;
  node->literalLoc = literalLoc;
  node->identifierLoc = identifierLoc;
  node->literal = literal;
  node->identifier = identifier;

  return node;
}

auto LiteralOperatorIdAST::create(Arena* arena) -> LiteralOperatorIdAST* {
  auto node = new (arena) LiteralOperatorIdAST();
  return node;
}

auto LiteralOperatorIdAST::create(Arena* arena, SourceLocation operatorLoc,
                                  SourceLocation literalLoc,
                                  SourceLocation identifierLoc,
                                  const Literal* literal,
                                  const Identifier* identifier)
    -> LiteralOperatorIdAST* {
  auto node = new (arena) LiteralOperatorIdAST();
  node->operatorLoc = operatorLoc;
  node->literalLoc = literalLoc;
  node->identifierLoc = identifierLoc;
  node->literal = literal;
  node->identifier = identifier;
  return node;
}

auto LiteralOperatorIdAST::create(Arena* arena, const Literal* literal,
                                  const Identifier* identifier)
    -> LiteralOperatorIdAST* {
  auto node = new (arena) LiteralOperatorIdAST();
  node->literal = literal;
  node->identifier = identifier;
  return node;
}

auto ConversionFunctionIdAST::clone(Arena* arena) -> ConversionFunctionIdAST* {
  auto node = create(arena);

  node->operatorLoc = operatorLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  return node;
}

auto ConversionFunctionIdAST::create(Arena* arena) -> ConversionFunctionIdAST* {
  auto node = new (arena) ConversionFunctionIdAST();
  return node;
}

auto ConversionFunctionIdAST::create(Arena* arena, SourceLocation operatorLoc,
                                     TypeIdAST* typeId)
    -> ConversionFunctionIdAST* {
  auto node = new (arena) ConversionFunctionIdAST();
  node->operatorLoc = operatorLoc;
  node->typeId = typeId;
  return node;
}

auto ConversionFunctionIdAST::create(Arena* arena, TypeIdAST* typeId)
    -> ConversionFunctionIdAST* {
  auto node = new (arena) ConversionFunctionIdAST();
  node->typeId = typeId;
  return node;
}

auto SimpleTemplateIdAST::clone(Arena* arena) -> SimpleTemplateIdAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->lessLoc = lessLoc;

  if (templateArgumentList) {
    auto it = &node->templateArgumentList;
    for (auto node : ListView{templateArgumentList}) {
      *it = make_list_node<TemplateArgumentAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;
  node->identifier = identifier;
  node->symbol = symbol;

  return node;
}

auto SimpleTemplateIdAST::create(Arena* arena) -> SimpleTemplateIdAST* {
  auto node = new (arena) SimpleTemplateIdAST();
  return node;
}

auto SimpleTemplateIdAST::create(
    Arena* arena, SourceLocation identifierLoc, SourceLocation lessLoc,
    List<TemplateArgumentAST*>* templateArgumentList, SourceLocation greaterLoc,
    const Identifier* identifier, Symbol* symbol) -> SimpleTemplateIdAST* {
  auto node = new (arena) SimpleTemplateIdAST();
  node->identifierLoc = identifierLoc;
  node->lessLoc = lessLoc;
  node->templateArgumentList = templateArgumentList;
  node->greaterLoc = greaterLoc;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto SimpleTemplateIdAST::create(
    Arena* arena, List<TemplateArgumentAST*>* templateArgumentList,
    const Identifier* identifier, Symbol* symbol) -> SimpleTemplateIdAST* {
  auto node = new (arena) SimpleTemplateIdAST();
  node->templateArgumentList = templateArgumentList;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto LiteralOperatorTemplateIdAST::clone(Arena* arena)
    -> LiteralOperatorTemplateIdAST* {
  auto node = create(arena);

  if (literalOperatorId)
    node->literalOperatorId = literalOperatorId->clone(arena);

  node->lessLoc = lessLoc;

  if (templateArgumentList) {
    auto it = &node->templateArgumentList;
    for (auto node : ListView{templateArgumentList}) {
      *it = make_list_node<TemplateArgumentAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;

  return node;
}

auto LiteralOperatorTemplateIdAST::create(Arena* arena)
    -> LiteralOperatorTemplateIdAST* {
  auto node = new (arena) LiteralOperatorTemplateIdAST();
  return node;
}

auto LiteralOperatorTemplateIdAST::create(
    Arena* arena, LiteralOperatorIdAST* literalOperatorId,
    SourceLocation lessLoc, List<TemplateArgumentAST*>* templateArgumentList,
    SourceLocation greaterLoc) -> LiteralOperatorTemplateIdAST* {
  auto node = new (arena) LiteralOperatorTemplateIdAST();
  node->literalOperatorId = literalOperatorId;
  node->lessLoc = lessLoc;
  node->templateArgumentList = templateArgumentList;
  node->greaterLoc = greaterLoc;
  return node;
}

auto LiteralOperatorTemplateIdAST::create(
    Arena* arena, LiteralOperatorIdAST* literalOperatorId,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> LiteralOperatorTemplateIdAST* {
  auto node = new (arena) LiteralOperatorTemplateIdAST();
  node->literalOperatorId = literalOperatorId;
  node->templateArgumentList = templateArgumentList;
  return node;
}

auto OperatorFunctionTemplateIdAST::clone(Arena* arena)
    -> OperatorFunctionTemplateIdAST* {
  auto node = create(arena);

  if (operatorFunctionId)
    node->operatorFunctionId = operatorFunctionId->clone(arena);

  node->lessLoc = lessLoc;

  if (templateArgumentList) {
    auto it = &node->templateArgumentList;
    for (auto node : ListView{templateArgumentList}) {
      *it = make_list_node<TemplateArgumentAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->greaterLoc = greaterLoc;

  return node;
}

auto OperatorFunctionTemplateIdAST::create(Arena* arena)
    -> OperatorFunctionTemplateIdAST* {
  auto node = new (arena) OperatorFunctionTemplateIdAST();
  return node;
}

auto OperatorFunctionTemplateIdAST::create(
    Arena* arena, OperatorFunctionIdAST* operatorFunctionId,
    SourceLocation lessLoc, List<TemplateArgumentAST*>* templateArgumentList,
    SourceLocation greaterLoc) -> OperatorFunctionTemplateIdAST* {
  auto node = new (arena) OperatorFunctionTemplateIdAST();
  node->operatorFunctionId = operatorFunctionId;
  node->lessLoc = lessLoc;
  node->templateArgumentList = templateArgumentList;
  node->greaterLoc = greaterLoc;
  return node;
}

auto OperatorFunctionTemplateIdAST::create(
    Arena* arena, OperatorFunctionIdAST* operatorFunctionId,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> OperatorFunctionTemplateIdAST* {
  auto node = new (arena) OperatorFunctionTemplateIdAST();
  node->operatorFunctionId = operatorFunctionId;
  node->templateArgumentList = templateArgumentList;
  return node;
}

auto GlobalNestedNameSpecifierAST::clone(Arena* arena)
    -> GlobalNestedNameSpecifierAST* {
  auto node = create(arena);

  node->scopeLoc = scopeLoc;
  node->symbol = symbol;

  return node;
}

auto GlobalNestedNameSpecifierAST::create(Arena* arena)
    -> GlobalNestedNameSpecifierAST* {
  auto node = new (arena) GlobalNestedNameSpecifierAST();
  return node;
}

auto GlobalNestedNameSpecifierAST::create(Arena* arena, SourceLocation scopeLoc,
                                          ScopeSymbol* symbol)
    -> GlobalNestedNameSpecifierAST* {
  auto node = new (arena) GlobalNestedNameSpecifierAST();
  node->scopeLoc = scopeLoc;
  node->symbol = symbol;
  return node;
}

auto GlobalNestedNameSpecifierAST::create(Arena* arena, ScopeSymbol* symbol)
    -> GlobalNestedNameSpecifierAST* {
  auto node = new (arena) GlobalNestedNameSpecifierAST();
  node->symbol = symbol;
  return node;
}

auto SimpleNestedNameSpecifierAST::clone(Arena* arena)
    -> SimpleNestedNameSpecifierAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->scopeLoc = scopeLoc;
  node->symbol = symbol;

  return node;
}

auto SimpleNestedNameSpecifierAST::create(Arena* arena)
    -> SimpleNestedNameSpecifierAST* {
  auto node = new (arena) SimpleNestedNameSpecifierAST();
  return node;
}

auto SimpleNestedNameSpecifierAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    SourceLocation identifierLoc, const Identifier* identifier,
    SourceLocation scopeLoc, ScopeSymbol* symbol)
    -> SimpleNestedNameSpecifierAST* {
  auto node = new (arena) SimpleNestedNameSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  node->scopeLoc = scopeLoc;
  node->symbol = symbol;
  return node;
}

auto SimpleNestedNameSpecifierAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    const Identifier* identifier, ScopeSymbol* symbol)
    -> SimpleNestedNameSpecifierAST* {
  auto node = new (arena) SimpleNestedNameSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->identifier = identifier;
  node->symbol = symbol;
  return node;
}

auto DecltypeNestedNameSpecifierAST::clone(Arena* arena)
    -> DecltypeNestedNameSpecifierAST* {
  auto node = create(arena);

  if (decltypeSpecifier)
    node->decltypeSpecifier = decltypeSpecifier->clone(arena);

  node->scopeLoc = scopeLoc;
  node->symbol = symbol;

  return node;
}

auto DecltypeNestedNameSpecifierAST::create(Arena* arena)
    -> DecltypeNestedNameSpecifierAST* {
  auto node = new (arena) DecltypeNestedNameSpecifierAST();
  return node;
}

auto DecltypeNestedNameSpecifierAST::create(
    Arena* arena, DecltypeSpecifierAST* decltypeSpecifier,
    SourceLocation scopeLoc, ScopeSymbol* symbol)
    -> DecltypeNestedNameSpecifierAST* {
  auto node = new (arena) DecltypeNestedNameSpecifierAST();
  node->decltypeSpecifier = decltypeSpecifier;
  node->scopeLoc = scopeLoc;
  node->symbol = symbol;
  return node;
}

auto DecltypeNestedNameSpecifierAST::create(
    Arena* arena, DecltypeSpecifierAST* decltypeSpecifier, ScopeSymbol* symbol)
    -> DecltypeNestedNameSpecifierAST* {
  auto node = new (arena) DecltypeNestedNameSpecifierAST();
  node->decltypeSpecifier = decltypeSpecifier;
  node->symbol = symbol;
  return node;
}

auto TemplateNestedNameSpecifierAST::clone(Arena* arena)
    -> TemplateNestedNameSpecifierAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (templateId) node->templateId = templateId->clone(arena);

  node->scopeLoc = scopeLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;

  return node;
}

auto TemplateNestedNameSpecifierAST::create(Arena* arena)
    -> TemplateNestedNameSpecifierAST* {
  auto node = new (arena) TemplateNestedNameSpecifierAST();
  return node;
}

auto TemplateNestedNameSpecifierAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    SourceLocation templateLoc, SimpleTemplateIdAST* templateId,
    SourceLocation scopeLoc, bool isTemplateIntroduced, ScopeSymbol* symbol)
    -> TemplateNestedNameSpecifierAST* {
  auto node = new (arena) TemplateNestedNameSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->templateId = templateId;
  node->scopeLoc = scopeLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto TemplateNestedNameSpecifierAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    SimpleTemplateIdAST* templateId, bool isTemplateIntroduced,
    ScopeSymbol* symbol) -> TemplateNestedNameSpecifierAST* {
  auto node = new (arena) TemplateNestedNameSpecifierAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateId = templateId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  node->symbol = symbol;
  return node;
}

auto DefaultFunctionBodyAST::clone(Arena* arena) -> DefaultFunctionBodyAST* {
  auto node = create(arena);

  node->equalLoc = equalLoc;
  node->defaultLoc = defaultLoc;
  node->semicolonLoc = semicolonLoc;

  return node;
}

auto DefaultFunctionBodyAST::create(Arena* arena) -> DefaultFunctionBodyAST* {
  auto node = new (arena) DefaultFunctionBodyAST();
  return node;
}

auto DefaultFunctionBodyAST::create(Arena* arena, SourceLocation equalLoc,
                                    SourceLocation defaultLoc,
                                    SourceLocation semicolonLoc)
    -> DefaultFunctionBodyAST* {
  auto node = new (arena) DefaultFunctionBodyAST();
  node->equalLoc = equalLoc;
  node->defaultLoc = defaultLoc;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto CompoundStatementFunctionBodyAST::clone(Arena* arena)
    -> CompoundStatementFunctionBodyAST* {
  auto node = create(arena);

  node->colonLoc = colonLoc;

  if (memInitializerList) {
    auto it = &node->memInitializerList;
    for (auto node : ListView{memInitializerList}) {
      *it = make_list_node<MemInitializerAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (statement) node->statement = statement->clone(arena);

  return node;
}

auto CompoundStatementFunctionBodyAST::create(Arena* arena)
    -> CompoundStatementFunctionBodyAST* {
  auto node = new (arena) CompoundStatementFunctionBodyAST();
  return node;
}

auto CompoundStatementFunctionBodyAST::create(
    Arena* arena, SourceLocation colonLoc,
    List<MemInitializerAST*>* memInitializerList,
    CompoundStatementAST* statement) -> CompoundStatementFunctionBodyAST* {
  auto node = new (arena) CompoundStatementFunctionBodyAST();
  node->colonLoc = colonLoc;
  node->memInitializerList = memInitializerList;
  node->statement = statement;
  return node;
}

auto CompoundStatementFunctionBodyAST::create(
    Arena* arena, List<MemInitializerAST*>* memInitializerList,
    CompoundStatementAST* statement) -> CompoundStatementFunctionBodyAST* {
  auto node = new (arena) CompoundStatementFunctionBodyAST();
  node->memInitializerList = memInitializerList;
  node->statement = statement;
  return node;
}

auto TryStatementFunctionBodyAST::clone(Arena* arena)
    -> TryStatementFunctionBodyAST* {
  auto node = create(arena);

  node->tryLoc = tryLoc;
  node->colonLoc = colonLoc;

  if (memInitializerList) {
    auto it = &node->memInitializerList;
    for (auto node : ListView{memInitializerList}) {
      *it = make_list_node<MemInitializerAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (statement) node->statement = statement->clone(arena);

  if (handlerList) {
    auto it = &node->handlerList;
    for (auto node : ListView{handlerList}) {
      *it = make_list_node<HandlerAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  return node;
}

auto TryStatementFunctionBodyAST::create(Arena* arena)
    -> TryStatementFunctionBodyAST* {
  auto node = new (arena) TryStatementFunctionBodyAST();
  return node;
}

auto TryStatementFunctionBodyAST::create(
    Arena* arena, SourceLocation tryLoc, SourceLocation colonLoc,
    List<MemInitializerAST*>* memInitializerList,
    CompoundStatementAST* statement, List<HandlerAST*>* handlerList)
    -> TryStatementFunctionBodyAST* {
  auto node = new (arena) TryStatementFunctionBodyAST();
  node->tryLoc = tryLoc;
  node->colonLoc = colonLoc;
  node->memInitializerList = memInitializerList;
  node->statement = statement;
  node->handlerList = handlerList;
  return node;
}

auto TryStatementFunctionBodyAST::create(
    Arena* arena, List<MemInitializerAST*>* memInitializerList,
    CompoundStatementAST* statement, List<HandlerAST*>* handlerList)
    -> TryStatementFunctionBodyAST* {
  auto node = new (arena) TryStatementFunctionBodyAST();
  node->memInitializerList = memInitializerList;
  node->statement = statement;
  node->handlerList = handlerList;
  return node;
}

auto DeleteFunctionBodyAST::clone(Arena* arena) -> DeleteFunctionBodyAST* {
  auto node = create(arena);

  node->equalLoc = equalLoc;
  node->deleteLoc = deleteLoc;
  node->semicolonLoc = semicolonLoc;

  return node;
}

auto DeleteFunctionBodyAST::create(Arena* arena) -> DeleteFunctionBodyAST* {
  auto node = new (arena) DeleteFunctionBodyAST();
  return node;
}

auto DeleteFunctionBodyAST::create(Arena* arena, SourceLocation equalLoc,
                                   SourceLocation deleteLoc,
                                   SourceLocation semicolonLoc)
    -> DeleteFunctionBodyAST* {
  auto node = new (arena) DeleteFunctionBodyAST();
  node->equalLoc = equalLoc;
  node->deleteLoc = deleteLoc;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto TypeTemplateArgumentAST::clone(Arena* arena) -> TypeTemplateArgumentAST* {
  auto node = create(arena);

  if (typeId) node->typeId = typeId->clone(arena);

  return node;
}

auto TypeTemplateArgumentAST::create(Arena* arena) -> TypeTemplateArgumentAST* {
  auto node = new (arena) TypeTemplateArgumentAST();
  return node;
}

auto TypeTemplateArgumentAST::create(Arena* arena, TypeIdAST* typeId)
    -> TypeTemplateArgumentAST* {
  auto node = new (arena) TypeTemplateArgumentAST();
  node->typeId = typeId;
  return node;
}

auto ExpressionTemplateArgumentAST::clone(Arena* arena)
    -> ExpressionTemplateArgumentAST* {
  auto node = create(arena);

  if (expression) node->expression = expression->clone(arena);

  return node;
}

auto ExpressionTemplateArgumentAST::create(Arena* arena)
    -> ExpressionTemplateArgumentAST* {
  auto node = new (arena) ExpressionTemplateArgumentAST();
  return node;
}

auto ExpressionTemplateArgumentAST::create(Arena* arena,
                                           ExpressionAST* expression)
    -> ExpressionTemplateArgumentAST* {
  auto node = new (arena) ExpressionTemplateArgumentAST();
  node->expression = expression;
  return node;
}

auto ThrowExceptionSpecifierAST::clone(Arena* arena)
    -> ThrowExceptionSpecifierAST* {
  auto node = create(arena);

  node->throwLoc = throwLoc;
  node->lparenLoc = lparenLoc;
  node->rparenLoc = rparenLoc;

  return node;
}

auto ThrowExceptionSpecifierAST::create(Arena* arena)
    -> ThrowExceptionSpecifierAST* {
  auto node = new (arena) ThrowExceptionSpecifierAST();
  return node;
}

auto ThrowExceptionSpecifierAST::create(Arena* arena, SourceLocation throwLoc,
                                        SourceLocation lparenLoc,
                                        SourceLocation rparenLoc)
    -> ThrowExceptionSpecifierAST* {
  auto node = new (arena) ThrowExceptionSpecifierAST();
  node->throwLoc = throwLoc;
  node->lparenLoc = lparenLoc;
  node->rparenLoc = rparenLoc;
  return node;
}

auto NoexceptSpecifierAST::clone(Arena* arena) -> NoexceptSpecifierAST* {
  auto node = create(arena);

  node->noexceptLoc = noexceptLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rparenLoc = rparenLoc;

  return node;
}

auto NoexceptSpecifierAST::create(Arena* arena) -> NoexceptSpecifierAST* {
  auto node = new (arena) NoexceptSpecifierAST();
  return node;
}

auto NoexceptSpecifierAST::create(Arena* arena, SourceLocation noexceptLoc,
                                  SourceLocation lparenLoc,
                                  ExpressionAST* expression,
                                  SourceLocation rparenLoc)
    -> NoexceptSpecifierAST* {
  auto node = new (arena) NoexceptSpecifierAST();
  node->noexceptLoc = noexceptLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->rparenLoc = rparenLoc;
  return node;
}

auto NoexceptSpecifierAST::create(Arena* arena, ExpressionAST* expression)
    -> NoexceptSpecifierAST* {
  auto node = new (arena) NoexceptSpecifierAST();
  node->expression = expression;
  return node;
}

auto SimpleRequirementAST::clone(Arena* arena) -> SimpleRequirementAST* {
  auto node = create(arena);

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto SimpleRequirementAST::create(Arena* arena) -> SimpleRequirementAST* {
  auto node = new (arena) SimpleRequirementAST();
  return node;
}

auto SimpleRequirementAST::create(Arena* arena, ExpressionAST* expression,
                                  SourceLocation semicolonLoc)
    -> SimpleRequirementAST* {
  auto node = new (arena) SimpleRequirementAST();
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto SimpleRequirementAST::create(Arena* arena, ExpressionAST* expression)
    -> SimpleRequirementAST* {
  auto node = new (arena) SimpleRequirementAST();
  node->expression = expression;
  return node;
}

auto CompoundRequirementAST::clone(Arena* arena) -> CompoundRequirementAST* {
  auto node = create(arena);

  node->lbraceLoc = lbraceLoc;

  if (expression) node->expression = expression->clone(arena);

  node->rbraceLoc = rbraceLoc;
  node->noexceptLoc = noexceptLoc;
  node->minusGreaterLoc = minusGreaterLoc;

  if (typeConstraint) node->typeConstraint = typeConstraint->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto CompoundRequirementAST::create(Arena* arena) -> CompoundRequirementAST* {
  auto node = new (arena) CompoundRequirementAST();
  return node;
}

auto CompoundRequirementAST::create(
    Arena* arena, SourceLocation lbraceLoc, ExpressionAST* expression,
    SourceLocation rbraceLoc, SourceLocation noexceptLoc,
    SourceLocation minusGreaterLoc, TypeConstraintAST* typeConstraint,
    SourceLocation semicolonLoc) -> CompoundRequirementAST* {
  auto node = new (arena) CompoundRequirementAST();
  node->lbraceLoc = lbraceLoc;
  node->expression = expression;
  node->rbraceLoc = rbraceLoc;
  node->noexceptLoc = noexceptLoc;
  node->minusGreaterLoc = minusGreaterLoc;
  node->typeConstraint = typeConstraint;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto CompoundRequirementAST::create(Arena* arena, ExpressionAST* expression,
                                    TypeConstraintAST* typeConstraint)
    -> CompoundRequirementAST* {
  auto node = new (arena) CompoundRequirementAST();
  node->expression = expression;
  node->typeConstraint = typeConstraint;
  return node;
}

auto TypeRequirementAST::clone(Arena* arena) -> TypeRequirementAST* {
  auto node = create(arena);

  node->typenameLoc = typenameLoc;

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  node->templateLoc = templateLoc;

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->semicolonLoc = semicolonLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;

  return node;
}

auto TypeRequirementAST::create(Arena* arena) -> TypeRequirementAST* {
  auto node = new (arena) TypeRequirementAST();
  return node;
}

auto TypeRequirementAST::create(Arena* arena, SourceLocation typenameLoc,
                                NestedNameSpecifierAST* nestedNameSpecifier,
                                SourceLocation templateLoc,
                                UnqualifiedIdAST* unqualifiedId,
                                SourceLocation semicolonLoc,
                                bool isTemplateIntroduced)
    -> TypeRequirementAST* {
  auto node = new (arena) TypeRequirementAST();
  node->typenameLoc = typenameLoc;
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->templateLoc = templateLoc;
  node->unqualifiedId = unqualifiedId;
  node->semicolonLoc = semicolonLoc;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto TypeRequirementAST::create(Arena* arena,
                                NestedNameSpecifierAST* nestedNameSpecifier,
                                UnqualifiedIdAST* unqualifiedId,
                                bool isTemplateIntroduced)
    -> TypeRequirementAST* {
  auto node = new (arena) TypeRequirementAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->isTemplateIntroduced = isTemplateIntroduced;
  return node;
}

auto NestedRequirementAST::clone(Arena* arena) -> NestedRequirementAST* {
  auto node = create(arena);

  node->requiresLoc = requiresLoc;

  if (expression) node->expression = expression->clone(arena);

  node->semicolonLoc = semicolonLoc;

  return node;
}

auto NestedRequirementAST::create(Arena* arena) -> NestedRequirementAST* {
  auto node = new (arena) NestedRequirementAST();
  return node;
}

auto NestedRequirementAST::create(Arena* arena, SourceLocation requiresLoc,
                                  ExpressionAST* expression,
                                  SourceLocation semicolonLoc)
    -> NestedRequirementAST* {
  auto node = new (arena) NestedRequirementAST();
  node->requiresLoc = requiresLoc;
  node->expression = expression;
  node->semicolonLoc = semicolonLoc;
  return node;
}

auto NestedRequirementAST::create(Arena* arena, ExpressionAST* expression)
    -> NestedRequirementAST* {
  auto node = new (arena) NestedRequirementAST();
  node->expression = expression;
  return node;
}

auto NewParenInitializerAST::clone(Arena* arena) -> NewParenInitializerAST* {
  auto node = create(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;

  return node;
}

auto NewParenInitializerAST::create(Arena* arena) -> NewParenInitializerAST* {
  auto node = new (arena) NewParenInitializerAST();
  return node;
}

auto NewParenInitializerAST::create(Arena* arena, SourceLocation lparenLoc,
                                    List<ExpressionAST*>* expressionList,
                                    SourceLocation rparenLoc)
    -> NewParenInitializerAST* {
  auto node = new (arena) NewParenInitializerAST();
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  return node;
}

auto NewParenInitializerAST::create(Arena* arena,
                                    List<ExpressionAST*>* expressionList)
    -> NewParenInitializerAST* {
  auto node = new (arena) NewParenInitializerAST();
  node->expressionList = expressionList;
  return node;
}

auto NewBracedInitializerAST::clone(Arena* arena) -> NewBracedInitializerAST* {
  auto node = create(arena);

  if (bracedInitList) node->bracedInitList = bracedInitList->clone(arena);

  return node;
}

auto NewBracedInitializerAST::create(Arena* arena) -> NewBracedInitializerAST* {
  auto node = new (arena) NewBracedInitializerAST();
  return node;
}

auto NewBracedInitializerAST::create(Arena* arena,
                                     BracedInitListAST* bracedInitList)
    -> NewBracedInitializerAST* {
  auto node = new (arena) NewBracedInitializerAST();
  node->bracedInitList = bracedInitList;
  return node;
}

auto ParenMemInitializerAST::clone(Arena* arena) -> ParenMemInitializerAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  node->lparenLoc = lparenLoc;

  if (expressionList) {
    auto it = &node->expressionList;
    for (auto node : ListView{expressionList}) {
      *it = make_list_node<ExpressionAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rparenLoc = rparenLoc;
  node->ellipsisLoc = ellipsisLoc;

  return node;
}

auto ParenMemInitializerAST::create(Arena* arena) -> ParenMemInitializerAST* {
  auto node = new (arena) ParenMemInitializerAST();
  return node;
}

auto ParenMemInitializerAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, SourceLocation lparenLoc,
    List<ExpressionAST*>* expressionList, SourceLocation rparenLoc,
    SourceLocation ellipsisLoc) -> ParenMemInitializerAST* {
  auto node = new (arena) ParenMemInitializerAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->lparenLoc = lparenLoc;
  node->expressionList = expressionList;
  node->rparenLoc = rparenLoc;
  node->ellipsisLoc = ellipsisLoc;
  return node;
}

auto ParenMemInitializerAST::create(Arena* arena,
                                    NestedNameSpecifierAST* nestedNameSpecifier,
                                    UnqualifiedIdAST* unqualifiedId,
                                    List<ExpressionAST*>* expressionList)
    -> ParenMemInitializerAST* {
  auto node = new (arena) ParenMemInitializerAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->expressionList = expressionList;
  return node;
}

auto BracedMemInitializerAST::clone(Arena* arena) -> BracedMemInitializerAST* {
  auto node = create(arena);

  if (nestedNameSpecifier)
    node->nestedNameSpecifier = nestedNameSpecifier->clone(arena);

  if (unqualifiedId) node->unqualifiedId = unqualifiedId->clone(arena);

  if (bracedInitList) node->bracedInitList = bracedInitList->clone(arena);

  node->ellipsisLoc = ellipsisLoc;

  return node;
}

auto BracedMemInitializerAST::create(Arena* arena) -> BracedMemInitializerAST* {
  auto node = new (arena) BracedMemInitializerAST();
  return node;
}

auto BracedMemInitializerAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, BracedInitListAST* bracedInitList,
    SourceLocation ellipsisLoc) -> BracedMemInitializerAST* {
  auto node = new (arena) BracedMemInitializerAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->bracedInitList = bracedInitList;
  node->ellipsisLoc = ellipsisLoc;
  return node;
}

auto BracedMemInitializerAST::create(
    Arena* arena, NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, BracedInitListAST* bracedInitList)
    -> BracedMemInitializerAST* {
  auto node = new (arena) BracedMemInitializerAST();
  node->nestedNameSpecifier = nestedNameSpecifier;
  node->unqualifiedId = unqualifiedId;
  node->bracedInitList = bracedInitList;
  return node;
}

auto ThisLambdaCaptureAST::clone(Arena* arena) -> ThisLambdaCaptureAST* {
  auto node = create(arena);

  node->thisLoc = thisLoc;

  return node;
}

auto ThisLambdaCaptureAST::create(Arena* arena) -> ThisLambdaCaptureAST* {
  auto node = new (arena) ThisLambdaCaptureAST();
  return node;
}

auto ThisLambdaCaptureAST::create(Arena* arena, SourceLocation thisLoc)
    -> ThisLambdaCaptureAST* {
  auto node = new (arena) ThisLambdaCaptureAST();
  node->thisLoc = thisLoc;
  return node;
}

auto DerefThisLambdaCaptureAST::clone(Arena* arena)
    -> DerefThisLambdaCaptureAST* {
  auto node = create(arena);

  node->starLoc = starLoc;
  node->thisLoc = thisLoc;

  return node;
}

auto DerefThisLambdaCaptureAST::create(Arena* arena)
    -> DerefThisLambdaCaptureAST* {
  auto node = new (arena) DerefThisLambdaCaptureAST();
  return node;
}

auto DerefThisLambdaCaptureAST::create(Arena* arena, SourceLocation starLoc,
                                       SourceLocation thisLoc)
    -> DerefThisLambdaCaptureAST* {
  auto node = new (arena) DerefThisLambdaCaptureAST();
  node->starLoc = starLoc;
  node->thisLoc = thisLoc;
  return node;
}

auto SimpleLambdaCaptureAST::clone(Arena* arena) -> SimpleLambdaCaptureAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifier = identifier;

  return node;
}

auto SimpleLambdaCaptureAST::create(Arena* arena) -> SimpleLambdaCaptureAST* {
  auto node = new (arena) SimpleLambdaCaptureAST();
  return node;
}

auto SimpleLambdaCaptureAST::create(Arena* arena, SourceLocation identifierLoc,
                                    SourceLocation ellipsisLoc,
                                    const Identifier* identifier)
    -> SimpleLambdaCaptureAST* {
  auto node = new (arena) SimpleLambdaCaptureAST();
  node->identifierLoc = identifierLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifier = identifier;
  return node;
}

auto SimpleLambdaCaptureAST::create(Arena* arena, const Identifier* identifier)
    -> SimpleLambdaCaptureAST* {
  auto node = new (arena) SimpleLambdaCaptureAST();
  node->identifier = identifier;
  return node;
}

auto RefLambdaCaptureAST::clone(Arena* arena) -> RefLambdaCaptureAST* {
  auto node = create(arena);

  node->ampLoc = ampLoc;
  node->identifierLoc = identifierLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifier = identifier;

  return node;
}

auto RefLambdaCaptureAST::create(Arena* arena) -> RefLambdaCaptureAST* {
  auto node = new (arena) RefLambdaCaptureAST();
  return node;
}

auto RefLambdaCaptureAST::create(Arena* arena, SourceLocation ampLoc,
                                 SourceLocation identifierLoc,
                                 SourceLocation ellipsisLoc,
                                 const Identifier* identifier)
    -> RefLambdaCaptureAST* {
  auto node = new (arena) RefLambdaCaptureAST();
  node->ampLoc = ampLoc;
  node->identifierLoc = identifierLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifier = identifier;
  return node;
}

auto RefLambdaCaptureAST::create(Arena* arena, const Identifier* identifier)
    -> RefLambdaCaptureAST* {
  auto node = new (arena) RefLambdaCaptureAST();
  node->identifier = identifier;
  return node;
}

auto RefInitLambdaCaptureAST::clone(Arena* arena) -> RefInitLambdaCaptureAST* {
  auto node = create(arena);

  node->ampLoc = ampLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  node->identifier = identifier;

  return node;
}

auto RefInitLambdaCaptureAST::create(Arena* arena) -> RefInitLambdaCaptureAST* {
  auto node = new (arena) RefInitLambdaCaptureAST();
  return node;
}

auto RefInitLambdaCaptureAST::create(Arena* arena, SourceLocation ampLoc,
                                     SourceLocation ellipsisLoc,
                                     SourceLocation identifierLoc,
                                     ExpressionAST* initializer,
                                     const Identifier* identifier)
    -> RefInitLambdaCaptureAST* {
  auto node = new (arena) RefInitLambdaCaptureAST();
  node->ampLoc = ampLoc;
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->initializer = initializer;
  node->identifier = identifier;
  return node;
}

auto RefInitLambdaCaptureAST::create(Arena* arena, ExpressionAST* initializer,
                                     const Identifier* identifier)
    -> RefInitLambdaCaptureAST* {
  auto node = new (arena) RefInitLambdaCaptureAST();
  node->initializer = initializer;
  node->identifier = identifier;
  return node;
}

auto InitLambdaCaptureAST::clone(Arena* arena) -> InitLambdaCaptureAST* {
  auto node = create(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;

  if (initializer) node->initializer = initializer->clone(arena);

  node->identifier = identifier;

  return node;
}

auto InitLambdaCaptureAST::create(Arena* arena) -> InitLambdaCaptureAST* {
  auto node = new (arena) InitLambdaCaptureAST();
  return node;
}

auto InitLambdaCaptureAST::create(Arena* arena, SourceLocation ellipsisLoc,
                                  SourceLocation identifierLoc,
                                  ExpressionAST* initializer,
                                  const Identifier* identifier)
    -> InitLambdaCaptureAST* {
  auto node = new (arena) InitLambdaCaptureAST();
  node->ellipsisLoc = ellipsisLoc;
  node->identifierLoc = identifierLoc;
  node->initializer = initializer;
  node->identifier = identifier;
  return node;
}

auto InitLambdaCaptureAST::create(Arena* arena, ExpressionAST* initializer,
                                  const Identifier* identifier)
    -> InitLambdaCaptureAST* {
  auto node = new (arena) InitLambdaCaptureAST();
  node->initializer = initializer;
  node->identifier = identifier;
  return node;
}

auto EllipsisExceptionDeclarationAST::clone(Arena* arena)
    -> EllipsisExceptionDeclarationAST* {
  auto node = create(arena);

  node->ellipsisLoc = ellipsisLoc;

  return node;
}

auto EllipsisExceptionDeclarationAST::create(Arena* arena)
    -> EllipsisExceptionDeclarationAST* {
  auto node = new (arena) EllipsisExceptionDeclarationAST();
  return node;
}

auto EllipsisExceptionDeclarationAST::create(Arena* arena,
                                             SourceLocation ellipsisLoc)
    -> EllipsisExceptionDeclarationAST* {
  auto node = new (arena) EllipsisExceptionDeclarationAST();
  node->ellipsisLoc = ellipsisLoc;
  return node;
}

auto TypeExceptionDeclarationAST::clone(Arena* arena)
    -> TypeExceptionDeclarationAST* {
  auto node = create(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeSpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (typeSpecifierList) {
    auto it = &node->typeSpecifierList;
    for (auto node : ListView{typeSpecifierList}) {
      *it = make_list_node<SpecifierAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  if (declarator) node->declarator = declarator->clone(arena);

  return node;
}

auto TypeExceptionDeclarationAST::create(Arena* arena)
    -> TypeExceptionDeclarationAST* {
  auto node = new (arena) TypeExceptionDeclarationAST();
  return node;
}

auto TypeExceptionDeclarationAST::create(
    Arena* arena, List<AttributeSpecifierAST*>* attributeList,
    List<SpecifierAST*>* typeSpecifierList, DeclaratorAST* declarator)
    -> TypeExceptionDeclarationAST* {
  auto node = new (arena) TypeExceptionDeclarationAST();
  node->attributeList = attributeList;
  node->typeSpecifierList = typeSpecifierList;
  node->declarator = declarator;
  return node;
}

auto CxxAttributeAST::clone(Arena* arena) -> CxxAttributeAST* {
  auto node = create(arena);

  node->lbracketLoc = lbracketLoc;
  node->lbracket2Loc = lbracket2Loc;

  if (attributeUsingPrefix)
    node->attributeUsingPrefix = attributeUsingPrefix->clone(arena);

  if (attributeList) {
    auto it = &node->attributeList;
    for (auto node : ListView{attributeList}) {
      *it = make_list_node<AttributeAST>(arena, node->clone(arena));
      it = &(*it)->next;
    }
  }

  node->rbracketLoc = rbracketLoc;
  node->rbracket2Loc = rbracket2Loc;

  return node;
}

auto CxxAttributeAST::create(Arena* arena) -> CxxAttributeAST* {
  auto node = new (arena) CxxAttributeAST();
  return node;
}

auto CxxAttributeAST::create(Arena* arena, SourceLocation lbracketLoc,
                             SourceLocation lbracket2Loc,
                             AttributeUsingPrefixAST* attributeUsingPrefix,
                             List<AttributeAST*>* attributeList,
                             SourceLocation rbracketLoc,
                             SourceLocation rbracket2Loc) -> CxxAttributeAST* {
  auto node = new (arena) CxxAttributeAST();
  node->lbracketLoc = lbracketLoc;
  node->lbracket2Loc = lbracket2Loc;
  node->attributeUsingPrefix = attributeUsingPrefix;
  node->attributeList = attributeList;
  node->rbracketLoc = rbracketLoc;
  node->rbracket2Loc = rbracket2Loc;
  return node;
}

auto CxxAttributeAST::create(Arena* arena,
                             AttributeUsingPrefixAST* attributeUsingPrefix,
                             List<AttributeAST*>* attributeList)
    -> CxxAttributeAST* {
  auto node = new (arena) CxxAttributeAST();
  node->attributeUsingPrefix = attributeUsingPrefix;
  node->attributeList = attributeList;
  return node;
}

auto GccAttributeAST::clone(Arena* arena) -> GccAttributeAST* {
  auto node = create(arena);

  node->attributeLoc = attributeLoc;
  node->lparenLoc = lparenLoc;
  node->lparen2Loc = lparen2Loc;
  node->rparenLoc = rparenLoc;
  node->rparen2Loc = rparen2Loc;

  return node;
}

auto GccAttributeAST::create(Arena* arena) -> GccAttributeAST* {
  auto node = new (arena) GccAttributeAST();
  return node;
}

auto GccAttributeAST::create(Arena* arena, SourceLocation attributeLoc,
                             SourceLocation lparenLoc,
                             SourceLocation lparen2Loc,
                             SourceLocation rparenLoc,
                             SourceLocation rparen2Loc) -> GccAttributeAST* {
  auto node = new (arena) GccAttributeAST();
  node->attributeLoc = attributeLoc;
  node->lparenLoc = lparenLoc;
  node->lparen2Loc = lparen2Loc;
  node->rparenLoc = rparenLoc;
  node->rparen2Loc = rparen2Loc;
  return node;
}

auto AlignasAttributeAST::clone(Arena* arena) -> AlignasAttributeAST* {
  auto node = create(arena);

  node->alignasLoc = alignasLoc;
  node->lparenLoc = lparenLoc;

  if (expression) node->expression = expression->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->isPack = isPack;

  return node;
}

auto AlignasAttributeAST::create(Arena* arena) -> AlignasAttributeAST* {
  auto node = new (arena) AlignasAttributeAST();
  return node;
}

auto AlignasAttributeAST::create(Arena* arena, SourceLocation alignasLoc,
                                 SourceLocation lparenLoc,
                                 ExpressionAST* expression,
                                 SourceLocation ellipsisLoc,
                                 SourceLocation rparenLoc, bool isPack)
    -> AlignasAttributeAST* {
  auto node = new (arena) AlignasAttributeAST();
  node->alignasLoc = alignasLoc;
  node->lparenLoc = lparenLoc;
  node->expression = expression;
  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->isPack = isPack;
  return node;
}

auto AlignasAttributeAST::create(Arena* arena, ExpressionAST* expression,
                                 bool isPack) -> AlignasAttributeAST* {
  auto node = new (arena) AlignasAttributeAST();
  node->expression = expression;
  node->isPack = isPack;
  return node;
}

auto AlignasTypeAttributeAST::clone(Arena* arena) -> AlignasTypeAttributeAST* {
  auto node = create(arena);

  node->alignasLoc = alignasLoc;
  node->lparenLoc = lparenLoc;

  if (typeId) node->typeId = typeId->clone(arena);

  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->isPack = isPack;

  return node;
}

auto AlignasTypeAttributeAST::create(Arena* arena) -> AlignasTypeAttributeAST* {
  auto node = new (arena) AlignasTypeAttributeAST();
  return node;
}

auto AlignasTypeAttributeAST::create(Arena* arena, SourceLocation alignasLoc,
                                     SourceLocation lparenLoc,
                                     TypeIdAST* typeId,
                                     SourceLocation ellipsisLoc,
                                     SourceLocation rparenLoc, bool isPack)
    -> AlignasTypeAttributeAST* {
  auto node = new (arena) AlignasTypeAttributeAST();
  node->alignasLoc = alignasLoc;
  node->lparenLoc = lparenLoc;
  node->typeId = typeId;
  node->ellipsisLoc = ellipsisLoc;
  node->rparenLoc = rparenLoc;
  node->isPack = isPack;
  return node;
}

auto AlignasTypeAttributeAST::create(Arena* arena, TypeIdAST* typeId,
                                     bool isPack) -> AlignasTypeAttributeAST* {
  auto node = new (arena) AlignasTypeAttributeAST();
  node->typeId = typeId;
  node->isPack = isPack;
  return node;
}

auto AsmAttributeAST::clone(Arena* arena) -> AsmAttributeAST* {
  auto node = create(arena);

  node->asmLoc = asmLoc;
  node->lparenLoc = lparenLoc;
  node->literalLoc = literalLoc;
  node->rparenLoc = rparenLoc;
  node->literal = literal;

  return node;
}

auto AsmAttributeAST::create(Arena* arena) -> AsmAttributeAST* {
  auto node = new (arena) AsmAttributeAST();
  return node;
}

auto AsmAttributeAST::create(Arena* arena, SourceLocation asmLoc,
                             SourceLocation lparenLoc,
                             SourceLocation literalLoc,
                             SourceLocation rparenLoc, const Literal* literal)
    -> AsmAttributeAST* {
  auto node = new (arena) AsmAttributeAST();
  node->asmLoc = asmLoc;
  node->lparenLoc = lparenLoc;
  node->literalLoc = literalLoc;
  node->rparenLoc = rparenLoc;
  node->literal = literal;
  return node;
}

auto AsmAttributeAST::create(Arena* arena, const Literal* literal)
    -> AsmAttributeAST* {
  auto node = new (arena) AsmAttributeAST();
  node->literal = literal;
  return node;
}

auto ScopedAttributeTokenAST::clone(Arena* arena) -> ScopedAttributeTokenAST* {
  auto node = create(arena);

  node->attributeNamespaceLoc = attributeNamespaceLoc;
  node->scopeLoc = scopeLoc;
  node->identifierLoc = identifierLoc;
  node->attributeNamespace = attributeNamespace;
  node->identifier = identifier;

  return node;
}

auto ScopedAttributeTokenAST::create(Arena* arena) -> ScopedAttributeTokenAST* {
  auto node = new (arena) ScopedAttributeTokenAST();
  return node;
}

auto ScopedAttributeTokenAST::create(
    Arena* arena, SourceLocation attributeNamespaceLoc, SourceLocation scopeLoc,
    SourceLocation identifierLoc, const Identifier* attributeNamespace,
    const Identifier* identifier) -> ScopedAttributeTokenAST* {
  auto node = new (arena) ScopedAttributeTokenAST();
  node->attributeNamespaceLoc = attributeNamespaceLoc;
  node->scopeLoc = scopeLoc;
  node->identifierLoc = identifierLoc;
  node->attributeNamespace = attributeNamespace;
  node->identifier = identifier;
  return node;
}

auto ScopedAttributeTokenAST::create(Arena* arena,
                                     const Identifier* attributeNamespace,
                                     const Identifier* identifier)
    -> ScopedAttributeTokenAST* {
  auto node = new (arena) ScopedAttributeTokenAST();
  node->attributeNamespace = attributeNamespace;
  node->identifier = identifier;
  return node;
}

auto SimpleAttributeTokenAST::clone(Arena* arena) -> SimpleAttributeTokenAST* {
  auto node = create(arena);

  node->identifierLoc = identifierLoc;
  node->identifier = identifier;

  return node;
}

auto SimpleAttributeTokenAST::create(Arena* arena) -> SimpleAttributeTokenAST* {
  auto node = new (arena) SimpleAttributeTokenAST();
  return node;
}

auto SimpleAttributeTokenAST::create(Arena* arena, SourceLocation identifierLoc,
                                     const Identifier* identifier)
    -> SimpleAttributeTokenAST* {
  auto node = new (arena) SimpleAttributeTokenAST();
  node->identifierLoc = identifierLoc;
  node->identifier = identifier;
  return node;
}

auto SimpleAttributeTokenAST::create(Arena* arena, const Identifier* identifier)
    -> SimpleAttributeTokenAST* {
  auto node = new (arena) SimpleAttributeTokenAST();
  node->identifier = identifier;
  return node;
}

auto to_string(ValueCategory valueCategory) -> std::string_view {
  switch (valueCategory) {
    case ValueCategory::kNone:
      return "none";
    case ValueCategory::kLValue:
      return "lvalue";
    case ValueCategory::kXValue:
      return "xvalue";
    case ValueCategory::kPrValue:
      return "prvalue";
    default:
      cxx_runtime_error("Invalid value category");
  }  // switch
}

auto to_string(ImplicitCastKind implicitCastKind) -> std::string_view {
  switch (implicitCastKind) {
    case ImplicitCastKind::kIdentity:
      return "identity";
    case ImplicitCastKind::kLValueToRValueConversion:
      return "lvalue-to-rvalue-conversion";
    case ImplicitCastKind::kArrayToPointerConversion:
      return "array-to-pointer-conversion";
    case ImplicitCastKind::kFunctionToPointerConversion:
      return "function-to-pointer-conversion";
    case ImplicitCastKind::kIntegralPromotion:
      return "integral-promotion";
    case ImplicitCastKind::kFloatingPointPromotion:
      return "floating-point-promotion";
    case ImplicitCastKind::kIntegralConversion:
      return "integral-conversion";
    case ImplicitCastKind::kFloatingPointConversion:
      return "floating-point-conversion";
    case ImplicitCastKind::kFloatingIntegralConversion:
      return "floating-integral-conversion";
    case ImplicitCastKind::kPointerConversion:
      return "pointer-conversion";
    case ImplicitCastKind::kPointerToMemberConversion:
      return "pointer-to-member-conversion";
    case ImplicitCastKind::kBooleanConversion:
      return "boolean-conversion";
    case ImplicitCastKind::kFunctionPointerConversion:
      return "function-pointer-conversion";
    case ImplicitCastKind::kQualificationConversion:
      return "qualification-conversion";
    case ImplicitCastKind::kTemporaryMaterializationConversion:
      return "temporary-materialization-conversion";
    case ImplicitCastKind::kUserDefinedConversion:
      return "user-defined-conversion";
    default:
      cxx_runtime_error("Invalid implicit cast kind");
  }  // switch
}

}  // namespace cxx
