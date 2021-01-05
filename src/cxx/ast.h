// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#pragma once

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/source_location.h>

namespace cxx {

template <typename T>
struct List final : Managed {
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

struct AST : Managed {
  virtual ~AST();
};

struct UnitAST : AST {};

struct DeclarationAST : AST {
  virtual void visit(DeclarationASTVisitor*) = 0;
};

struct StatementAST : AST {
  virtual void visit(StatementASTVisitor*) = 0;
};

struct ExpressionAST : AST {};

struct SpecifierAST : AST {
  virtual void visit(SpecifierASTVisitor*) = 0;
};

struct NameAST : AST {};

struct AttributeAST : AST {};

struct TypeIdAST : AST {
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
};

struct PtrOperatorAST : AST {};

struct CoreDeclaratorAST : AST {};

struct DeclaratorModifierAST : AST {};

struct NestedNameSpecifierAST : AST {};

struct UsingDeclaratorAST : AST {
  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
};

// units

struct TranslationUnitAST final : UnitAST {
  List<DeclarationAST*>* declarationList = nullptr;
};

struct ModuleUnitAST final : UnitAST {};

// statements

struct LabeledStatementAST final : StatementAST {
  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct CaseStatementAST final : StatementAST {
  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct DefaultStatementAST final : StatementAST {
  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct CompoundStatementAST final : StatementAST {
  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct IfStatementAST final : StatementAST {
  SourceLocation ifLoc;
  SourceLocation constexprLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;
  StatementAST* elseStatement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct SwitchStatementAST final : StatementAST {
  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct WhileStatementAST final : StatementAST {
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct DoStatementAST final : StatementAST {
  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct ForRangeStatementAST final : StatementAST {
  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  DeclarationAST* rangeDeclaration = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* rangeInitializer = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct ForStatementAST final : StatementAST {
  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation semicolonLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

struct BreakStatementAST final : StatementAST {
  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct ContinueStatementAST final : StatementAST {
  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct ReturnStatementAST final : StatementAST {
  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct GotoStatementAST final : StatementAST {
  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct CoroutineReturnStatementAST final : StatementAST {
  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(StatementASTVisitor* visitor) override;
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationAST* declaration = nullptr;

  void visit(StatementASTVisitor* visitor) override;
};

// declarations

struct ConceptDefinitionAST final : DeclarationAST {
  SourceLocation conceptLoc;
  NameAST* name = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct ForRangeDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct AliasDeclarationAST final : DeclarationAST {
  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeAST*>* attributeList;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct SimpleDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributes = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<DeclaratorAST*>* declaratorList = nullptr;
  SourceLocation semicolonToken;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct StaticAssertDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct EmptyDeclarationAST final : DeclarationAST {
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct AttributeDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct OpaqueEnumDeclarationAST final : DeclarationAST {
  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation emicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingEnumDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct NamespaceDefinitionAST final : DeclarationAST {
  SourceLocation inlineLoc;
  SourceLocation namespaceLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  List<AttributeAST*>* extraAttributeList = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct NamespaceAliasDefinitionAST final : DeclarationAST {
  SourceLocation namespaceLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingDirectiveAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct UsingDeclarationAST final : DeclarationAST {
  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct AsmDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct LinkageSpecificationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ExportDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ModuleImportDeclarationAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct TemplateDeclarationAST final : DeclarationAST {
  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  DeclarationAST* declaration = nullptr;

  void visit(DeclarationASTVisitor* visitor) override;
};

struct DeductionGuideAST final : DeclarationAST {
  void visit(DeclarationASTVisitor* visitor) override;
};

struct ExplicitInstantiationAST final : DeclarationAST {
  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void visit(DeclarationASTVisitor* visitor) override;
};

// names

struct SimpleNameAST final : NameAST {
  SourceLocation identifierLoc;
};

// specifiers

struct StorageClassSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct FunctionSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct ExplicitSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct SimpleTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct NamedTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct PlaceholderTypeSpecifierHelperAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct DecltypeSpecifierTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct UnderlyingTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct AtomicTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct PrimitiveTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct ElaboratedTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct DecltypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct PlaceholderTypeSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

struct CvQualifierAST final : SpecifierAST {
  SourceLocation qualifierLoc;

  void visit(SpecifierASTVisitor* visitor) override;
};

struct EnumBaseAST final : AST {
  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
};

struct EnumeratorAST final : AST {
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
};

struct EnumSpecifierAST final : SpecifierAST {
  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  List<EnumeratorAST*>* enumeratorList = nullptr;
  SourceLocation rbraceLoc;

  void visit(SpecifierASTVisitor* visitor) override;
};

struct ClassSpecifierAST final : SpecifierAST {
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void visit(SpecifierASTVisitor* visitor) override;
};

struct TypenameSpecifierAST final : SpecifierAST {
  void visit(SpecifierASTVisitor* visitor) override;
};

// declarators

struct DeclaratorAST final : AST {
  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;
};

struct IdDeclaratorAST final : CoreDeclaratorAST {
  List<AttributeAST*>* attributeList = nullptr;
};

struct NestedDeclaratorAST final : CoreDeclaratorAST {
  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;
};

struct PointerOperatorAST final : PtrOperatorAST {
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;
};

struct ReferenceOperatorAST final : PtrOperatorAST {
  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;
};

struct PtrToMemberOperatorAST final : PtrOperatorAST {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;
};

struct FunctionDeclaratorAST final : DeclaratorModifierAST {};

struct ArrayDeclaratorAST final : DeclaratorModifierAST {
  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeAST*>* attributeList = nullptr;
};

}  // namespace cxx