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

#pragma once

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_visitor.h>
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

  virtual void visit(ASTVisitor* visitor) = 0;
};

struct UnitAST : AST {};

struct DeclarationAST : AST {};

struct StatementAST : AST {};

struct ExpressionAST : AST {};

struct SpecifierAST : AST {};

struct NameAST : AST {};

struct AttributeAST : AST {};

struct PtrOperatorAST : AST {};

struct CoreDeclaratorAST : AST {};

struct DeclaratorModifierAST : AST {};

struct ExceptionDeclarationAST : AST {};

// misc

struct TypeIdAST final : AST {
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct NestedNameSpecifierAST final : AST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct UsingDeclaratorAST final : AST {
  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct HandlerAST final : AST {
  SourceLocation catchLoc;
  SourceLocation lparenLoc;
  ExceptionDeclarationAST* exceptionDeclaration = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// exception declarations

struct EllipsisExceptionDeclarationAST final : ExceptionDeclarationAST {
  SourceLocation ellipsisLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TypeExceptionDeclarationAST final : ExceptionDeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// units

struct TranslationUnitAST final : UnitAST {
  List<DeclarationAST*>* declarationList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ModuleUnitAST final : UnitAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// statements

struct LabeledStatementAST final : StatementAST {
  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct CaseStatementAST final : StatementAST {
  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DefaultStatementAST final : StatementAST {
  SourceLocation defaultLoc;
  SourceLocation colonLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ExpressionStatementAST final : StatementAST {
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct CompoundStatementAST final : StatementAST {
  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
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

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct SwitchStatementAST final : StatementAST {
  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct WhileStatementAST final : StatementAST {
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DoStatementAST final : StatementAST {
  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
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

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
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

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct BreakStatementAST final : StatementAST {
  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ContinueStatementAST final : StatementAST {
  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ReturnStatementAST final : StatementAST {
  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct GotoStatementAST final : StatementAST {
  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct CoroutineReturnStatementAST final : StatementAST {
  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DeclarationStatementAST final : StatementAST {
  DeclarationAST* declaration = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TryBlockStatementAST final : StatementAST {
  SourceLocation tryLoc;
  StatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// declarations

struct ConceptDefinitionAST final : DeclarationAST {
  SourceLocation conceptLoc;
  NameAST* name = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ForRangeDeclarationAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct AliasDeclarationAST final : DeclarationAST {
  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct SimpleDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributes = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<DeclaratorAST*>* declaratorList = nullptr;
  SourceLocation semicolonToken;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct StaticAssertDeclarationAST final : DeclarationAST {
  SourceLocation staticAssertLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation commaLoc;
  List<SourceLocation>* stringLiteralList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct EmptyDeclarationAST final : DeclarationAST {
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct AttributeDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct OpaqueEnumDeclarationAST final : DeclarationAST {
  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation emicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct UsingEnumDeclarationAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
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

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct NamespaceAliasDefinitionAST final : DeclarationAST {
  SourceLocation namespaceLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct UsingDirectiveAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct UsingDeclarationAST final : DeclarationAST {
  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct AsmDeclarationAST final : DeclarationAST {
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  List<SourceLocation>* stringLiteralList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct LinkageSpecificationAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ExportDeclarationAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ModuleImportDeclarationAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TemplateDeclarationAST final : DeclarationAST {
  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  DeclarationAST* declaration = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DeductionGuideAST final : DeclarationAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ExplicitInstantiationAST final : DeclarationAST {
  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// names

struct SimpleNameAST final : NameAST {
  SourceLocation identifierLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct OperatorNameAST final : NameAST {
  SourceLocation loc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TemplateArgumentAST final : AST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TemplateNameAST final : NameAST {
  NameAST* name = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// specifiers

struct StorageClassSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct FunctionSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ExplicitSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct SimpleTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct NamedTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct PlaceholderTypeSpecifierHelperAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DecltypeSpecifierTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct UnderlyingTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct AtomicTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct PrimitiveTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ElaboratedTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct DecltypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct PlaceholderTypeSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct CvQualifierAST final : SpecifierAST {
  SourceLocation qualifierLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct EnumBaseAST final : AST {
  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct EnumeratorAST final : AST {
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
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

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ClassSpecifierAST final : SpecifierAST {
  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NameAST* name = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct TypenameSpecifierAST final : SpecifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

// declarators

struct DeclaratorAST final : AST {
  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorModifierAST*>* modifiers = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct IdDeclaratorAST final : CoreDeclaratorAST {
  SourceLocation ellipsisLoc;
  NameAST* name = nullptr;
  List<AttributeAST*>* attributeList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct NestedDeclaratorAST final : CoreDeclaratorAST {
  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct PointerOperatorAST final : PtrOperatorAST {
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ReferenceOperatorAST final : PtrOperatorAST {
  SourceLocation refLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct PtrToMemberOperatorAST final : PtrOperatorAST {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct FunctionDeclaratorAST final : DeclaratorModifierAST {
  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

struct ArrayDeclaratorAST final : DeclaratorModifierAST {
  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeAST*>* attributeList = nullptr;

  void visit(ASTVisitor* visitor) override { visitor->visit(this); }
};

}  // namespace cxx