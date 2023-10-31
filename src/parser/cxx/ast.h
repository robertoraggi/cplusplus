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

#pragma once

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_kind.h>
#include <cxx/ast_visitor.h>
#include <cxx/const_value.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>

#include <optional>

namespace cxx {

template <typename T>
class List final : public Managed {
 public:
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

class AST : public Managed {
 public:
  explicit AST(ASTKind kind) : kind_(kind) {}

  virtual ~AST();

  [[nodiscard]] auto kind() const -> ASTKind { return kind_; }

  virtual void accept(ASTVisitor* visitor) = 0;

  virtual auto firstSourceLocation() -> SourceLocation = 0;
  virtual auto lastSourceLocation() -> SourceLocation = 0;

  [[nodiscard]] auto sourceLocationRange() -> SourceLocationRange {
    return SourceLocationRange(firstSourceLocation(), lastSourceLocation());
  }

 private:
  ASTKind kind_;
};

[[nodiscard]] inline auto firstSourceLocation(SourceLocation loc)
    -> SourceLocation {
  return loc;
}

template <typename T>
[[nodiscard]] inline auto firstSourceLocation(T* node) -> SourceLocation {
  return node ? node->firstSourceLocation() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto firstSourceLocation(List<T>* nodes)
    -> SourceLocation {
  for (auto it = nodes; it; it = it->next) {
    if (auto loc = firstSourceLocation(it->value)) return loc;
  }
  return {};
}

[[nodiscard]] inline auto lastSourceLocation(SourceLocation loc)
    -> SourceLocation {
  return loc ? loc.next() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto lastSourceLocation(T* node) -> SourceLocation {
  return node ? node->lastSourceLocation() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto lastSourceLocation(List<T>* nodes) -> SourceLocation {
  if (!nodes) return {};
  if (auto loc = lastSourceLocation(nodes->next)) return loc;
  if (auto loc = lastSourceLocation(nodes->value)) return loc;
  return {};
}

class AttributeSpecifierAST : public AST {
 public:
  using AST::AST;
};

class AttributeTokenAST : public AST {
 public:
  using AST::AST;
};

class CoreDeclaratorAST : public AST {
 public:
  using AST::AST;
};

class DeclarationAST : public AST {
 public:
  using AST::AST;
};

class DeclaratorChunkAST : public AST {
 public:
  using AST::AST;
};

class ExceptionDeclarationAST : public AST {
 public:
  using AST::AST;
};

class ExceptionSpecifierAST : public AST {
 public:
  using AST::AST;
};

class ExpressionAST : public AST {
 public:
  using AST::AST;
  ValueCategory valueCategory = ValueCategory::kPrValue;
  std::optional<ConstValue> constValue;
  const Type* type = nullptr;
};

class FunctionBodyAST : public AST {
 public:
  using AST::AST;
};

class LambdaCaptureAST : public AST {
 public:
  using AST::AST;
};

class MemInitializerAST : public AST {
 public:
  using AST::AST;
};

class NestedNameSpecifierAST : public AST {
 public:
  using AST::AST;
};

class NewInitializerAST : public AST {
 public:
  using AST::AST;
};

class PtrOperatorAST : public AST {
 public:
  using AST::AST;
};

class RequirementAST : public AST {
 public:
  using AST::AST;
};

class SpecifierAST : public AST {
 public:
  using AST::AST;
};

class StatementAST : public AST {
 public:
  using AST::AST;
};

class TemplateArgumentAST : public AST {
 public:
  using AST::AST;
};

class TemplateParameterAST : public AST {
 public:
  using AST::AST;
  Symbol* symbol = nullptr;
  int depth = 0;
  int index = 0;
};

class UnitAST : public AST {
 public:
  using AST::AST;
};

class UnqualifiedIdAST : public AST {
 public:
  using AST::AST;
};

class TranslationUnitAST final : public UnitAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TranslationUnit;

  TranslationUnitAST() : UnitAST(Kind) {}

  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleUnitAST final : public UnitAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModuleUnit;

  ModuleUnitAST() : UnitAST(Kind) {}

  GlobalModuleFragmentAST* globalModuleFragment = nullptr;
  ModuleDeclarationAST* moduleDeclaration = nullptr;
  List<DeclarationAST*>* declarationList = nullptr;
  PrivateModuleFragmentAST* privateModuleFragment = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleDeclaration;

  SimpleDeclarationAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmDeclaration;

  AsmDeclarationAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<AsmQualifierAST*>* asmQualifierList = nullptr;
  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  SourceLocation literalLoc;
  List<AsmOperandAST*>* outputOperandList = nullptr;
  List<AsmOperandAST*>* inputOperandList = nullptr;
  List<AsmClobberAST*>* clobberList = nullptr;
  List<AsmGotoLabelAST*>* gotoLabelList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;
  const Literal* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamespaceAliasDefinitionAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NamespaceAliasDefinition;

  NamespaceAliasDefinitionAST() : DeclarationAST(Kind) {}

  SourceLocation namespaceLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameIdAST* unqualifiedId = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UsingDeclaration;

  UsingDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation usingLoc;
  List<UsingDeclaratorAST*>* usingDeclaratorList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingEnumDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UsingEnumDeclaration;

  UsingEnumDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation usingLoc;
  ElaboratedTypeSpecifierAST* enumTypeSpecifier = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDirectiveAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UsingDirective;

  UsingDirectiveAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation usingLoc;
  SourceLocation namespaceLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameIdAST* unqualifiedId = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StaticAssertDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::StaticAssertDeclaration;

  StaticAssertDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation staticAssertLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation commaLoc;
  SourceLocation literalLoc;
  const Literal* literal = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AliasDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AliasDeclaration;

  AliasDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class OpaqueEnumDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::OpaqueEnumDeclaration;

  OpaqueEnumDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameIdAST* unqualifiedId = nullptr;
  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  SourceLocation emicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FunctionDefinitionAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FunctionDefinition;

  FunctionDefinitionAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  FunctionBodyAST* functionBody = nullptr;
  FunctionSymbol* symbol = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TemplateDeclaration;

  TemplateDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<TemplateParameterAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* requiresClause = nullptr;
  DeclarationAST* declaration = nullptr;
  TemplateParametersSymbol* symbol = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConceptDefinitionAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConceptDefinition;

  ConceptDefinitionAST() : DeclarationAST(Kind) {}

  SourceLocation conceptLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;
  ConceptSymbol* symbol = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeductionGuideAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DeductionGuide;

  DeductionGuideAST() : DeclarationAST(Kind) {}

  SpecifierAST* explicitSpecifier = nullptr;
  SourceLocation identifierLoc;
  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  SourceLocation arrowLoc;
  SimpleTemplateIdAST* templateId = nullptr;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExplicitInstantiationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExplicitInstantiation;

  ExplicitInstantiationAST() : DeclarationAST(Kind) {}

  SourceLocation externLoc;
  SourceLocation templateLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExportDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExportDeclaration;

  ExportDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation exportLoc;
  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExportCompoundDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExportCompoundDeclaration;

  ExportCompoundDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation exportLoc;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LinkageSpecificationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LinkageSpecification;

  LinkageSpecificationAST() : DeclarationAST(Kind) {}

  SourceLocation externLoc;
  SourceLocation stringliteralLoc;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  const StringLiteral* stringLiteral = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamespaceDefinitionAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NamespaceDefinition;

  NamespaceDefinitionAST() : DeclarationAST(Kind) {}

  SourceLocation inlineLoc;
  SourceLocation namespaceLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<NestedNamespaceSpecifierAST*>* nestedNamespaceSpecifierList = nullptr;
  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* extraAttributeList = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  const Identifier* identifier = nullptr;
  bool isInline = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EmptyDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::EmptyDeclaration;

  EmptyDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AttributeDeclaration;

  AttributeDeclarationAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleImportDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModuleImportDeclaration;

  ModuleImportDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation importLoc;
  ImportNameAST* importName = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParameterDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ParameterDeclaration;

  ParameterDeclarationAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation thisLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  const Type* type = nullptr;
  const Identifier* identifier = nullptr;
  bool isThisIntroduced = false;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AccessDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AccessDeclaration;

  AccessDeclarationAST() : DeclarationAST(Kind) {}

  SourceLocation accessLoc;
  SourceLocation colonLoc;
  TokenKind accessSpecifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForRangeDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ForRangeDeclaration;

  ForRangeDeclarationAST() : DeclarationAST(Kind) {}

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StructuredBindingDeclarationAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::StructuredBindingDeclaration;

  StructuredBindingDeclarationAST() : DeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  SourceLocation refQualifierLoc;
  SourceLocation lbracketLoc;
  List<NameIdAST*>* bindingList = nullptr;
  SourceLocation rbracketLoc;
  ExpressionAST* initializer = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmOperandAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmOperand;

  AsmOperandAST() : DeclarationAST(Kind) {}

  SourceLocation lbracketLoc;
  SourceLocation symbolicNameLoc;
  SourceLocation rbracketLoc;
  SourceLocation constraintLiteralLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  const Identifier* symbolicName = nullptr;
  const Literal* constraintLiteral = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmQualifierAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmQualifier;

  AsmQualifierAST() : DeclarationAST(Kind) {}

  SourceLocation qualifierLoc;
  TokenKind qualifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmClobberAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmClobber;

  AsmClobberAST() : DeclarationAST(Kind) {}

  SourceLocation literalLoc;
  const StringLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmGotoLabelAST final : public DeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmGotoLabel;

  AsmGotoLabelAST() : DeclarationAST(Kind) {}

  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LabeledStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LabeledStatement;

  LabeledStatementAST() : StatementAST(Kind) {}

  SourceLocation identifierLoc;
  SourceLocation colonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CaseStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CaseStatement;

  CaseStatementAST() : StatementAST(Kind) {}

  SourceLocation caseLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation colonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DefaultStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DefaultStatement;

  DefaultStatementAST() : StatementAST(Kind) {}

  SourceLocation defaultLoc;
  SourceLocation colonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExpressionStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExpressionStatement;

  ExpressionStatementAST() : StatementAST(Kind) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CompoundStatement;

  CompoundStatementAST() : StatementAST(Kind) {}

  SourceLocation lbraceLoc;
  List<StatementAST*>* statementList = nullptr;
  SourceLocation rbraceLoc;
  BlockSymbol* symbol = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IfStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::IfStatement;

  IfStatementAST() : StatementAST(Kind) {}

  SourceLocation ifLoc;
  SourceLocation constexprLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;
  SourceLocation elseLoc;
  StatementAST* elseStatement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstevalIfStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstevalIfStatement;

  ConstevalIfStatementAST() : StatementAST(Kind) {}

  SourceLocation ifLoc;
  SourceLocation exclaimLoc;
  SourceLocation constvalLoc;
  StatementAST* statement = nullptr;
  SourceLocation elseLoc;
  StatementAST* elseStatement = nullptr;
  bool isNot = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SwitchStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SwitchStatement;

  SwitchStatementAST() : StatementAST(Kind) {}

  SourceLocation switchLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class WhileStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::WhileStatement;

  WhileStatementAST() : StatementAST(Kind) {}

  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* condition = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DoStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DoStatement;

  DoStatementAST() : StatementAST(Kind) {}

  SourceLocation doLoc;
  StatementAST* statement = nullptr;
  SourceLocation whileLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForRangeStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ForRangeStatement;

  ForRangeStatementAST() : StatementAST(Kind) {}

  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  DeclarationAST* rangeDeclaration = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* rangeInitializer = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ForStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ForStatement;

  ForStatementAST() : StatementAST(Kind) {}

  SourceLocation forLoc;
  SourceLocation lparenLoc;
  StatementAST* initializer = nullptr;
  ExpressionAST* condition = nullptr;
  SourceLocation semicolonLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  StatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BreakStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BreakStatement;

  BreakStatementAST() : StatementAST(Kind) {}

  SourceLocation breakLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ContinueStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ContinueStatement;

  ContinueStatementAST() : StatementAST(Kind) {}

  SourceLocation continueLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ReturnStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ReturnStatement;

  ReturnStatementAST() : StatementAST(Kind) {}

  SourceLocation returnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CoroutineReturnStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CoroutineReturnStatement;

  CoroutineReturnStatementAST() : StatementAST(Kind) {}

  SourceLocation coreturnLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GotoStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::GotoStatement;

  GotoStatementAST() : StatementAST(Kind) {}

  SourceLocation gotoLoc;
  SourceLocation identifierLoc;
  SourceLocation semicolonLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeclarationStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DeclarationStatement;

  DeclarationStatementAST() : StatementAST(Kind) {}

  DeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TryBlockStatementAST final : public StatementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TryBlockStatement;

  TryBlockStatementAST() : StatementAST(Kind) {}

  SourceLocation tryLoc;
  CompoundStatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CharLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CharLiteralExpression;

  CharLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  const CharLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BoolLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BoolLiteralExpression;

  BoolLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  bool isTrue = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IntLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::IntLiteralExpression;

  IntLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  const IntegerLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FloatLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FloatLiteralExpression;

  FloatLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  const FloatLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NullptrLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NullptrLiteralExpression;

  NullptrLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  TokenKind literal = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StringLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::StringLiteralExpression;

  StringLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  const StringLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UserDefinedStringLiteralExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UserDefinedStringLiteralExpression;

  UserDefinedStringLiteralExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation literalLoc;
  const StringLiteral* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThisExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThisExpression;

  ThisExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NestedExpression;

  NestedExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IdExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::IdExpression;

  IdExpressionAST() : ExpressionAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  bool isTemplateIntroduced = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LambdaExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LambdaExpression;

  LambdaExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lbracketLoc;
  SourceLocation captureDefaultLoc;
  List<LambdaCaptureAST*>* captureList = nullptr;
  SourceLocation rbracketLoc;
  SourceLocation lessLoc;
  List<TemplateParameterAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* templateRequiresClause = nullptr;
  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<LambdaSpecifierAST*>* lambdaSpecifierList = nullptr;
  ExceptionSpecifierAST* exceptionSpecifier = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  CompoundStatementAST* statement = nullptr;
  TokenKind captureDefault = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FoldExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FoldExpression;

  FoldExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  SourceLocation ellipsisLoc;
  SourceLocation foldOpLoc;
  ExpressionAST* rightExpression = nullptr;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;
  TokenKind foldOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RightFoldExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::RightFoldExpression;

  RightFoldExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation opLoc;
  SourceLocation ellipsisLoc;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LeftFoldExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LeftFoldExpression;

  LeftFoldExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  SourceLocation ellipsisLoc;
  SourceLocation opLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RequiresExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::RequiresExpression;

  RequiresExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation requiresLoc;
  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  SourceLocation lbraceLoc;
  List<RequirementAST*>* requirementList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SubscriptExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SubscriptExpression;

  SubscriptExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lbracketLoc;
  ExpressionAST* indexExpression = nullptr;
  SourceLocation rbracketLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CallExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CallExpression;

  CallExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeConstructionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeConstruction;

  TypeConstructionAST() : ExpressionAST(Kind) {}

  SpecifierAST* typeSpecifier = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedTypeConstructionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BracedTypeConstruction;

  BracedTypeConstructionAST() : ExpressionAST(Kind) {}

  SpecifierAST* typeSpecifier = nullptr;
  BracedInitListAST* bracedInitList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class MemberExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::MemberExpression;

  MemberExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation accessLoc;
  IdExpressionAST* memberId = nullptr;
  TokenKind accessOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PostIncrExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::PostIncrExpression;

  PostIncrExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* baseExpression = nullptr;
  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CppCastExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CppCastExpression;

  CppCastExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation castLoc;
  SourceLocation lessLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation greaterLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeidExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeidExpression;

  TypeidExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation typeidLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeidOfTypeExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeidOfTypeExpression;

  TypeidOfTypeExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation typeidLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UnaryExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UnaryExpression;

  UnaryExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation opLoc;
  ExpressionAST* expression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AwaitExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AwaitExpression;

  AwaitExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation awaitLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SizeofExpression;

  SizeofExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation sizeofLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofTypeExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SizeofTypeExpression;

  SizeofTypeExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation sizeofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeofPackExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SizeofPackExpression;

  SizeofPackExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation sizeofLoc;
  SourceLocation ellipsisLoc;
  SourceLocation lparenLoc;
  SourceLocation identifierLoc;
  SourceLocation rparenLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignofTypeExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AlignofTypeExpression;

  AlignofTypeExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation alignofLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignofExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AlignofExpression;

  AlignofExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation alignofLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NoexceptExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NoexceptExpression;

  NoexceptExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation noexceptLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NewExpression;

  NewExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation scopeLoc;
  SourceLocation newLoc;
  NewPlacementAST* newPlacement = nullptr;
  SourceLocation lparenLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;
  NewInitializerAST* newInitalizer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeleteExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DeleteExpression;

  DeleteExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation scopeLoc;
  SourceLocation deleteLoc;
  SourceLocation lbracketLoc;
  SourceLocation rbracketLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CastExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CastExpression;

  CastExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ImplicitCastExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ImplicitCastExpression;

  ImplicitCastExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* expression = nullptr;
  ImplicitCastKind castKind = ImplicitCastKind::kIdentity;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BinaryExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BinaryExpression;

  BinaryExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConditionalExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConditionalExpression;

  ConditionalExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* condition = nullptr;
  SourceLocation questionLoc;
  ExpressionAST* iftrueExpression = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* iffalseExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class YieldExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::YieldExpression;

  YieldExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation yieldLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThrowExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThrowExpression;

  ThrowExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation throwLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AssignmentExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AssignmentExpression;

  AssignmentExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* leftExpression = nullptr;
  SourceLocation opLoc;
  ExpressionAST* rightExpression = nullptr;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PackExpansionExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::PackExpansionExpression;

  PackExpansionExpressionAST() : ExpressionAST(Kind) {}

  ExpressionAST* expression = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DesignatedInitializerClauseAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DesignatedInitializerClause;

  DesignatedInitializerClauseAST() : ExpressionAST(Kind) {}

  SourceLocation dotLoc;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;
  ExpressionAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeTraitsExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeTraitsExpression;

  TypeTraitsExpressionAST() : ExpressionAST(Kind) {}

  SourceLocation typeTraitsLoc;
  SourceLocation lparenLoc;
  List<TypeIdAST*>* typeIdList = nullptr;
  SourceLocation rparenLoc;
  TokenKind typeTraits = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConditionExpressionAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConditionExpression;

  ConditionExpressionAST() : ExpressionAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  ExpressionAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EqualInitializerAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::EqualInitializer;

  EqualInitializerAST() : ExpressionAST(Kind) {}

  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedInitListAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BracedInitList;

  BracedInitListAST() : ExpressionAST(Kind) {}

  SourceLocation lbraceLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParenInitializerAST final : public ExpressionAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ParenInitializer;

  ParenInitializerAST() : ExpressionAST(Kind) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateTypeParameterAST final : public TemplateParameterAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TemplateTypeParameter;

  TemplateTypeParameterAST() : TemplateParameterAST(Kind) {}

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<TemplateParameterAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;
  RequiresClauseAST* requiresClause = nullptr;
  SourceLocation classKeyLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  IdExpressionAST* idExpression = nullptr;
  const Identifier* identifier = nullptr;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NonTypeTemplateParameterAST final : public TemplateParameterAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NonTypeTemplateParameter;

  NonTypeTemplateParameterAST() : TemplateParameterAST(Kind) {}

  ParameterDeclarationAST* declaration = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypenameTypeParameterAST final : public TemplateParameterAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypenameTypeParameter;

  TypenameTypeParameterAST() : TemplateParameterAST(Kind) {}

  SourceLocation classKeyLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  const Identifier* identifier = nullptr;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstraintTypeParameterAST final : public TemplateParameterAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstraintTypeParameter;

  ConstraintTypeParameterAST() : TemplateParameterAST(Kind) {}

  TypeConstraintAST* typeConstraint = nullptr;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypedefSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypedefSpecifier;

  TypedefSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation typedefLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FriendSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FriendSpecifier;

  FriendSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation friendLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstevalSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstevalSpecifier;

  ConstevalSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation constevalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstinitSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstinitSpecifier;

  ConstinitSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation constinitLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstexprSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstexprSpecifier;

  ConstexprSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation constexprLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InlineSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::InlineSpecifier;

  InlineSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation inlineLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class StaticSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::StaticSpecifier;

  StaticSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation staticLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExternSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExternSpecifier;

  ExternSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation externLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThreadLocalSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThreadLocalSpecifier;

  ThreadLocalSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation threadLocalLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThreadSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThreadSpecifier;

  ThreadSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation threadLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class MutableSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::MutableSpecifier;

  MutableSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation mutableLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VirtualSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::VirtualSpecifier;

  VirtualSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation virtualLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExplicitSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExplicitSpecifier;

  ExplicitSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation explicitLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AutoTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AutoTypeSpecifier;

  AutoTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation autoLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VoidTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::VoidTypeSpecifier;

  VoidTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation voidLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SizeTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SizeTypeSpecifier;

  SizeTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SignTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SignTypeSpecifier;

  SignTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VaListTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::VaListTypeSpecifier;

  VaListTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IntegralTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::IntegralTypeSpecifier;

  IntegralTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FloatingPointTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FloatingPointTypeSpecifier;

  FloatingPointTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ComplexTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ComplexTypeSpecifier;

  ComplexTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation complexLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NamedTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NamedTypeSpecifier;

  NamedTypeSpecifierAST() : SpecifierAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  bool isTemplateIntroduced = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AtomicTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AtomicTypeSpecifier;

  AtomicTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation atomicLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UnderlyingTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::UnderlyingTypeSpecifier;

  UnderlyingTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation underlyingTypeLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ElaboratedTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ElaboratedTypeSpecifier;

  ElaboratedTypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  TokenKind classKey = TokenKind::T_EOF_SYMBOL;
  bool isTemplateIntroduced = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeAutoSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DecltypeAutoSpecifier;

  DecltypeAutoSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  SourceLocation autoLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DecltypeSpecifier;

  DecltypeSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation decltypeLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;
  const Type* type = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PlaceholderTypeSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::PlaceholderTypeSpecifier;

  PlaceholderTypeSpecifierAST() : SpecifierAST(Kind) {}

  TypeConstraintAST* typeConstraint = nullptr;
  SpecifierAST* specifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConstQualifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConstQualifier;

  ConstQualifierAST() : SpecifierAST(Kind) {}

  SourceLocation constLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class VolatileQualifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::VolatileQualifier;

  VolatileQualifierAST() : SpecifierAST(Kind) {}

  SourceLocation volatileLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RestrictQualifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::RestrictQualifier;

  RestrictQualifierAST() : SpecifierAST(Kind) {}

  SourceLocation restrictLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EnumSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::EnumSpecifier;

  EnumSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation enumLoc;
  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameIdAST* unqualifiedId = nullptr;
  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  List<EnumeratorAST*>* enumeratorList = nullptr;
  SourceLocation rbraceLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ClassSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ClassSpecifier;

  ClassSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  SourceLocation finalLoc;
  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;
  SourceLocation lbraceLoc;
  List<DeclarationAST*>* declarationList = nullptr;
  SourceLocation rbraceLoc;
  TokenKind classKey = TokenKind::T_EOF_SYMBOL;
  bool isFinal = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypenameSpecifierAST final : public SpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypenameSpecifier;

  TypenameSpecifierAST() : SpecifierAST(Kind) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PointerOperatorAST final : public PtrOperatorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::PointerOperator;

  PointerOperatorAST() : PtrOperatorAST(Kind) {}

  SourceLocation starLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ReferenceOperatorAST final : public PtrOperatorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ReferenceOperator;

  ReferenceOperatorAST() : PtrOperatorAST(Kind) {}

  SourceLocation refLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  TokenKind refOp = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PtrToMemberOperatorAST final : public PtrOperatorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::PtrToMemberOperator;

  PtrToMemberOperatorAST() : PtrOperatorAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation starLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* cvQualifierList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BitfieldDeclaratorAST final : public CoreDeclaratorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BitfieldDeclarator;

  BitfieldDeclaratorAST() : CoreDeclaratorAST(Kind) {}

  NameIdAST* unqualifiedId = nullptr;
  SourceLocation colonLoc;
  ExpressionAST* sizeExpression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParameterPackAST final : public CoreDeclaratorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ParameterPack;

  ParameterPackAST() : CoreDeclaratorAST(Kind) {}

  SourceLocation ellipsisLoc;
  CoreDeclaratorAST* coreDeclarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class IdDeclaratorAST final : public CoreDeclaratorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::IdDeclarator;

  IdDeclaratorAST() : CoreDeclaratorAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  bool isTemplateIntroduced = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedDeclaratorAST final : public CoreDeclaratorAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NestedDeclarator;

  NestedDeclaratorAST() : CoreDeclaratorAST(Kind) {}

  SourceLocation lparenLoc;
  DeclaratorAST* declarator = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class FunctionDeclaratorChunkAST final : public DeclaratorChunkAST {
 public:
  static constexpr ASTKind Kind = ASTKind::FunctionDeclaratorChunk;

  FunctionDeclaratorChunkAST() : DeclaratorChunkAST(Kind) {}

  SourceLocation lparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;
  SourceLocation rparenLoc;
  List<SpecifierAST*>* cvQualifierList = nullptr;
  SourceLocation refLoc;
  ExceptionSpecifierAST* exceptionSpecifier = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;
  bool isFinal = false;
  bool isOverride = false;
  bool isPure = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ArrayDeclaratorChunkAST final : public DeclaratorChunkAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ArrayDeclaratorChunk;

  ArrayDeclaratorChunkAST() : DeclaratorChunkAST(Kind) {}

  SourceLocation lbracketLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbracketLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NameIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NameId;

  NameIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DestructorIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DestructorId;

  DestructorIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation tildeLoc;
  UnqualifiedIdAST* id = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DecltypeId;

  DecltypeIdAST() : UnqualifiedIdAST(Kind) {}

  DecltypeSpecifierAST* decltypeSpecifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class OperatorFunctionIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::OperatorFunctionId;

  OperatorFunctionIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation operatorLoc;
  SourceLocation opLoc;
  SourceLocation openLoc;
  SourceLocation closeLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LiteralOperatorIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LiteralOperatorId;

  LiteralOperatorIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation operatorLoc;
  SourceLocation literalLoc;
  SourceLocation identifierLoc;
  const Literal* literal = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ConversionFunctionIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ConversionFunctionId;

  ConversionFunctionIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation operatorLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleTemplateIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleTemplateId;

  SimpleTemplateIdAST() : UnqualifiedIdAST(Kind) {}

  SourceLocation identifierLoc;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LiteralOperatorTemplateIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::LiteralOperatorTemplateId;

  LiteralOperatorTemplateIdAST() : UnqualifiedIdAST(Kind) {}

  LiteralOperatorIdAST* literalOperatorId = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class OperatorFunctionTemplateIdAST final : public UnqualifiedIdAST {
 public:
  static constexpr ASTKind Kind = ASTKind::OperatorFunctionTemplateId;

  OperatorFunctionTemplateIdAST() : UnqualifiedIdAST(Kind) {}

  OperatorFunctionIdAST* operatorFunctionId = nullptr;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GlobalNestedNameSpecifierAST final : public NestedNameSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::GlobalNestedNameSpecifier;

  GlobalNestedNameSpecifierAST() : NestedNameSpecifierAST(Kind) {}

  SourceLocation scopeLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleNestedNameSpecifierAST final : public NestedNameSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleNestedNameSpecifier;

  SimpleNestedNameSpecifierAST() : NestedNameSpecifierAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;
  SourceLocation scopeLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DecltypeNestedNameSpecifierAST final : public NestedNameSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DecltypeNestedNameSpecifier;

  DecltypeNestedNameSpecifierAST() : NestedNameSpecifierAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  DecltypeSpecifierAST* decltypeSpecifier = nullptr;
  SourceLocation scopeLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TemplateNestedNameSpecifierAST final : public NestedNameSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TemplateNestedNameSpecifier;

  TemplateNestedNameSpecifierAST() : NestedNameSpecifierAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  SimpleTemplateIdAST* templateId = nullptr;
  SourceLocation scopeLoc;
  bool isTemplateIntroduced = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DefaultFunctionBodyAST final : public FunctionBodyAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DefaultFunctionBody;

  DefaultFunctionBodyAST() : FunctionBodyAST(Kind) {}

  SourceLocation equalLoc;
  SourceLocation defaultLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundStatementFunctionBodyAST final : public FunctionBodyAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CompoundStatementFunctionBody;

  CompoundStatementFunctionBodyAST() : FunctionBodyAST(Kind) {}

  SourceLocation colonLoc;
  List<MemInitializerAST*>* memInitializerList = nullptr;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TryStatementFunctionBodyAST final : public FunctionBodyAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TryStatementFunctionBody;

  TryStatementFunctionBodyAST() : FunctionBodyAST(Kind) {}

  SourceLocation tryLoc;
  SourceLocation colonLoc;
  List<MemInitializerAST*>* memInitializerList = nullptr;
  CompoundStatementAST* statement = nullptr;
  List<HandlerAST*>* handlerList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeleteFunctionBodyAST final : public FunctionBodyAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DeleteFunctionBody;

  DeleteFunctionBodyAST() : FunctionBodyAST(Kind) {}

  SourceLocation equalLoc;
  SourceLocation deleteLoc;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeTemplateArgumentAST final : public TemplateArgumentAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeTemplateArgument;

  TypeTemplateArgumentAST() : TemplateArgumentAST(Kind) {}

  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ExpressionTemplateArgumentAST final : public TemplateArgumentAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ExpressionTemplateArgument;

  ExpressionTemplateArgumentAST() : TemplateArgumentAST(Kind) {}

  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThrowExceptionSpecifierAST final : public ExceptionSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThrowExceptionSpecifier;

  ThrowExceptionSpecifierAST() : ExceptionSpecifierAST(Kind) {}

  SourceLocation throwLoc;
  SourceLocation lparenLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NoexceptSpecifierAST final : public ExceptionSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NoexceptSpecifier;

  NoexceptSpecifierAST() : ExceptionSpecifierAST(Kind) {}

  SourceLocation noexceptLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleRequirementAST final : public RequirementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleRequirement;

  SimpleRequirementAST() : RequirementAST(Kind) {}

  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CompoundRequirementAST final : public RequirementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CompoundRequirement;

  CompoundRequirementAST() : RequirementAST(Kind) {}

  SourceLocation lbraceLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation rbraceLoc;
  SourceLocation noexceptLoc;
  SourceLocation minusGreaterLoc;
  TypeConstraintAST* typeConstraint = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeRequirementAST final : public RequirementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeRequirement;

  TypeRequirementAST() : RequirementAST(Kind) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedRequirementAST final : public RequirementAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NestedRequirement;

  NestedRequirementAST() : RequirementAST(Kind) {}

  SourceLocation requiresLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewParenInitializerAST final : public NewInitializerAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NewParenInitializer;

  NewParenInitializerAST() : NewInitializerAST(Kind) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewBracedInitializerAST final : public NewInitializerAST {
 public:
  static constexpr ASTKind Kind = ASTKind::NewBracedInitializer;

  NewBracedInitializerAST() : NewInitializerAST(Kind) {}

  BracedInitListAST* bracedInitList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParenMemInitializerAST final : public MemInitializerAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ParenMemInitializer;

  ParenMemInitializerAST() : MemInitializerAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BracedMemInitializerAST final : public MemInitializerAST {
 public:
  static constexpr ASTKind Kind = ASTKind::BracedMemInitializer;

  BracedMemInitializerAST() : MemInitializerAST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  BracedInitListAST* bracedInitList = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ThisLambdaCapture;

  ThisLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DerefThisLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::DerefThisLambdaCapture;

  DerefThisLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation starLoc;
  SourceLocation thisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleLambdaCapture;

  SimpleLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RefLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::RefLambdaCapture;

  RefLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation ampLoc;
  SourceLocation identifierLoc;
  SourceLocation ellipsisLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RefInitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::RefInitLambdaCapture;

  RefInitLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation ampLoc;
  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  ExpressionAST* initializer = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InitLambdaCaptureAST final : public LambdaCaptureAST {
 public:
  static constexpr ASTKind Kind = ASTKind::InitLambdaCapture;

  InitLambdaCaptureAST() : LambdaCaptureAST(Kind) {}

  SourceLocation ellipsisLoc;
  SourceLocation identifierLoc;
  ExpressionAST* initializer = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EllipsisExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::EllipsisExceptionDeclaration;

  EllipsisExceptionDeclarationAST() : ExceptionDeclarationAST(Kind) {}

  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeExceptionDeclarationAST final : public ExceptionDeclarationAST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeExceptionDeclaration;

  TypeExceptionDeclarationAST() : ExceptionDeclarationAST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class CxxAttributeAST final : public AttributeSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::CxxAttribute;

  CxxAttributeAST() : AttributeSpecifierAST(Kind) {}

  SourceLocation lbracketLoc;
  SourceLocation lbracket2Loc;
  AttributeUsingPrefixAST* attributeUsingPrefix = nullptr;
  List<AttributeAST*>* attributeList = nullptr;
  SourceLocation rbracketLoc;
  SourceLocation rbracket2Loc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GccAttributeAST final : public AttributeSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::GccAttribute;

  GccAttributeAST() : AttributeSpecifierAST(Kind) {}

  SourceLocation attributeLoc;
  SourceLocation lparenLoc;
  SourceLocation lparen2Loc;
  SourceLocation rparenLoc;
  SourceLocation rparen2Loc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignasAttributeAST final : public AttributeSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AlignasAttribute;

  AlignasAttributeAST() : AttributeSpecifierAST(Kind) {}

  SourceLocation alignasLoc;
  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation ellipsisLoc;
  SourceLocation rparenLoc;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AlignasTypeAttributeAST final : public AttributeSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AlignasTypeAttribute;

  AlignasTypeAttributeAST() : AttributeSpecifierAST(Kind) {}

  SourceLocation alignasLoc;
  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation ellipsisLoc;
  SourceLocation rparenLoc;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AsmAttributeAST final : public AttributeSpecifierAST {
 public:
  static constexpr ASTKind Kind = ASTKind::AsmAttribute;

  AsmAttributeAST() : AttributeSpecifierAST(Kind) {}

  SourceLocation asmLoc;
  SourceLocation lparenLoc;
  SourceLocation literalLoc;
  SourceLocation rparenLoc;
  const Literal* literal = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ScopedAttributeTokenAST final : public AttributeTokenAST {
 public:
  static constexpr ASTKind Kind = ASTKind::ScopedAttributeToken;

  ScopedAttributeTokenAST() : AttributeTokenAST(Kind) {}

  SourceLocation attributeNamespaceLoc;
  SourceLocation scopeLoc;
  SourceLocation identifierLoc;
  const Identifier* attributeNamespace = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class SimpleAttributeTokenAST final : public AttributeTokenAST {
 public:
  static constexpr ASTKind Kind = ASTKind::SimpleAttributeToken;

  SimpleAttributeTokenAST() : AttributeTokenAST(Kind) {}

  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class GlobalModuleFragmentAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::GlobalModuleFragment;

  GlobalModuleFragmentAST() : AST(Kind) {}

  SourceLocation moduleLoc;
  SourceLocation semicolonLoc;
  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class PrivateModuleFragmentAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::PrivateModuleFragment;

  PrivateModuleFragmentAST() : AST(Kind) {}

  SourceLocation moduleLoc;
  SourceLocation colonLoc;
  SourceLocation privateLoc;
  SourceLocation semicolonLoc;
  List<DeclarationAST*>* declarationList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleDeclarationAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModuleDeclaration;

  ModuleDeclarationAST() : AST(Kind) {}

  SourceLocation exportLoc;
  SourceLocation moduleLoc;
  ModuleNameAST* moduleName = nullptr;
  ModulePartitionAST* modulePartition = nullptr;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation semicolonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleNameAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModuleName;

  ModuleNameAST() : AST(Kind) {}

  ModuleQualifierAST* moduleQualifier = nullptr;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModuleQualifierAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModuleQualifier;

  ModuleQualifierAST() : AST(Kind) {}

  ModuleQualifierAST* moduleQualifier = nullptr;
  SourceLocation identifierLoc;
  SourceLocation dotLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ModulePartitionAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ModulePartition;

  ModulePartitionAST() : AST(Kind) {}

  SourceLocation colonLoc;
  ModuleNameAST* moduleName = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ImportNameAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ImportName;

  ImportNameAST() : AST(Kind) {}

  SourceLocation headerLoc;
  ModulePartitionAST* modulePartition = nullptr;
  ModuleNameAST* moduleName = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class InitDeclaratorAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::InitDeclarator;

  InitDeclaratorAST() : AST(Kind) {}

  DeclaratorAST* declarator = nullptr;
  RequiresClauseAST* requiresClause = nullptr;
  ExpressionAST* initializer = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class DeclaratorAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::Declarator;

  DeclaratorAST() : AST(Kind) {}

  List<PtrOperatorAST*>* ptrOpList = nullptr;
  CoreDeclaratorAST* coreDeclarator = nullptr;
  List<DeclaratorChunkAST*>* declaratorChunkList = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class UsingDeclaratorAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::UsingDeclarator;

  UsingDeclaratorAST() : AST(Kind) {}

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  SourceLocation ellipsisLoc;
  bool isPack = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class EnumeratorAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::Enumerator;

  EnumeratorAST() : AST(Kind) {}

  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  SourceLocation equalLoc;
  ExpressionAST* expression = nullptr;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeIdAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeId;

  TypeIdAST() : AST(Kind) {}

  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclaratorAST* declarator = nullptr;
  const Type* type = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class HandlerAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::Handler;

  HandlerAST() : AST(Kind) {}

  SourceLocation catchLoc;
  SourceLocation lparenLoc;
  ExceptionDeclarationAST* exceptionDeclaration = nullptr;
  SourceLocation rparenLoc;
  CompoundStatementAST* statement = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class BaseSpecifierAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::BaseSpecifier;

  BaseSpecifierAST() : AST(Kind) {}

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  bool isTemplateIntroduced = false;
  bool isVirtual = false;
  TokenKind accessSpecifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class RequiresClauseAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::RequiresClause;

  RequiresClauseAST() : AST(Kind) {}

  SourceLocation requiresLoc;
  ExpressionAST* expression = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class ParameterDeclarationClauseAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::ParameterDeclarationClause;

  ParameterDeclarationClauseAST() : AST(Kind) {}

  List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
  SourceLocation commaLoc;
  SourceLocation ellipsisLoc;
  FunctionParametersSymbol* functionParametersSymbol = nullptr;
  bool isVariadic = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TrailingReturnTypeAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::TrailingReturnType;

  TrailingReturnTypeAST() : AST(Kind) {}

  SourceLocation minusGreaterLoc;
  TypeIdAST* typeId = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class LambdaSpecifierAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::LambdaSpecifier;

  LambdaSpecifierAST() : AST(Kind) {}

  SourceLocation specifierLoc;
  TokenKind specifier = TokenKind::T_EOF_SYMBOL;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class TypeConstraintAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::TypeConstraint;

  TypeConstraintAST() : AST(Kind) {}

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation identifierLoc;
  SourceLocation lessLoc;
  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  SourceLocation greaterLoc;
  const Identifier* identifier = nullptr;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeArgumentClauseAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::AttributeArgumentClause;

  AttributeArgumentClauseAST() : AST(Kind) {}

  SourceLocation lparenLoc;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::Attribute;

  AttributeAST() : AST(Kind) {}

  AttributeTokenAST* attributeToken = nullptr;
  AttributeArgumentClauseAST* attributeArgumentClause = nullptr;
  SourceLocation ellipsisLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class AttributeUsingPrefixAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::AttributeUsingPrefix;

  AttributeUsingPrefixAST() : AST(Kind) {}

  SourceLocation usingLoc;
  SourceLocation attributeNamespaceLoc;
  SourceLocation colonLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NewPlacementAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::NewPlacement;

  NewPlacementAST() : AST(Kind) {}

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

class NestedNamespaceSpecifierAST final : public AST {
 public:
  static constexpr ASTKind Kind = ASTKind::NestedNamespaceSpecifier;

  NestedNamespaceSpecifierAST() : AST(Kind) {}

  SourceLocation inlineLoc;
  SourceLocation identifierLoc;
  SourceLocation scopeLoc;
  const Identifier* identifier = nullptr;
  bool isInline = false;

  void accept(ASTVisitor* visitor) override { visitor->visit(this); }

  auto firstSourceLocation() -> SourceLocation override;
  auto lastSourceLocation() -> SourceLocation override;
};

template <typename Visitor>
auto visit(Visitor&& visitor, UnitAST* ast) {
  switch (ast->kind()) {
    case TranslationUnitAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TranslationUnitAST*>(ast));
    case ModuleUnitAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ModuleUnitAST*>(ast));
    default:
      cxx_runtime_error("unexpected Unit");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, DeclarationAST* ast) {
  switch (ast->kind()) {
    case SimpleDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleDeclarationAST*>(ast));
    case AsmDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmDeclarationAST*>(ast));
    case NamespaceAliasDefinitionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NamespaceAliasDefinitionAST*>(ast));
    case UsingDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<UsingDeclarationAST*>(ast));
    case UsingEnumDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<UsingEnumDeclarationAST*>(ast));
    case UsingDirectiveAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<UsingDirectiveAST*>(ast));
    case StaticAssertDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<StaticAssertDeclarationAST*>(ast));
    case AliasDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AliasDeclarationAST*>(ast));
    case OpaqueEnumDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<OpaqueEnumDeclarationAST*>(ast));
    case FunctionDefinitionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FunctionDefinitionAST*>(ast));
    case TemplateDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TemplateDeclarationAST*>(ast));
    case ConceptDefinitionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConceptDefinitionAST*>(ast));
    case DeductionGuideAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DeductionGuideAST*>(ast));
    case ExplicitInstantiationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExplicitInstantiationAST*>(ast));
    case ExportDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExportDeclarationAST*>(ast));
    case ExportCompoundDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExportCompoundDeclarationAST*>(ast));
    case LinkageSpecificationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LinkageSpecificationAST*>(ast));
    case NamespaceDefinitionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NamespaceDefinitionAST*>(ast));
    case EmptyDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<EmptyDeclarationAST*>(ast));
    case AttributeDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AttributeDeclarationAST*>(ast));
    case ModuleImportDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ModuleImportDeclarationAST*>(ast));
    case ParameterDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ParameterDeclarationAST*>(ast));
    case AccessDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AccessDeclarationAST*>(ast));
    case ForRangeDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ForRangeDeclarationAST*>(ast));
    case StructuredBindingDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<StructuredBindingDeclarationAST*>(ast));
    case AsmOperandAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmOperandAST*>(ast));
    case AsmQualifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmQualifierAST*>(ast));
    case AsmClobberAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmClobberAST*>(ast));
    case AsmGotoLabelAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmGotoLabelAST*>(ast));
    default:
      cxx_runtime_error("unexpected Declaration");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, StatementAST* ast) {
  switch (ast->kind()) {
    case LabeledStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LabeledStatementAST*>(ast));
    case CaseStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CaseStatementAST*>(ast));
    case DefaultStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DefaultStatementAST*>(ast));
    case ExpressionStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExpressionStatementAST*>(ast));
    case CompoundStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CompoundStatementAST*>(ast));
    case IfStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<IfStatementAST*>(ast));
    case ConstevalIfStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstevalIfStatementAST*>(ast));
    case SwitchStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SwitchStatementAST*>(ast));
    case WhileStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<WhileStatementAST*>(ast));
    case DoStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DoStatementAST*>(ast));
    case ForRangeStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ForRangeStatementAST*>(ast));
    case ForStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ForStatementAST*>(ast));
    case BreakStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BreakStatementAST*>(ast));
    case ContinueStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ContinueStatementAST*>(ast));
    case ReturnStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ReturnStatementAST*>(ast));
    case CoroutineReturnStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CoroutineReturnStatementAST*>(ast));
    case GotoStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<GotoStatementAST*>(ast));
    case DeclarationStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DeclarationStatementAST*>(ast));
    case TryBlockStatementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TryBlockStatementAST*>(ast));
    default:
      cxx_runtime_error("unexpected Statement");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, ExpressionAST* ast) {
  switch (ast->kind()) {
    case CharLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CharLiteralExpressionAST*>(ast));
    case BoolLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BoolLiteralExpressionAST*>(ast));
    case IntLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<IntLiteralExpressionAST*>(ast));
    case FloatLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FloatLiteralExpressionAST*>(ast));
    case NullptrLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NullptrLiteralExpressionAST*>(ast));
    case StringLiteralExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<StringLiteralExpressionAST*>(ast));
    case UserDefinedStringLiteralExpressionAST::Kind:
      return std::invoke(
          std::forward<Visitor>(visitor),
          static_cast<UserDefinedStringLiteralExpressionAST*>(ast));
    case ThisExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThisExpressionAST*>(ast));
    case NestedExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NestedExpressionAST*>(ast));
    case IdExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<IdExpressionAST*>(ast));
    case LambdaExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LambdaExpressionAST*>(ast));
    case FoldExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FoldExpressionAST*>(ast));
    case RightFoldExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<RightFoldExpressionAST*>(ast));
    case LeftFoldExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LeftFoldExpressionAST*>(ast));
    case RequiresExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<RequiresExpressionAST*>(ast));
    case SubscriptExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SubscriptExpressionAST*>(ast));
    case CallExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CallExpressionAST*>(ast));
    case TypeConstructionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeConstructionAST*>(ast));
    case BracedTypeConstructionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BracedTypeConstructionAST*>(ast));
    case MemberExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<MemberExpressionAST*>(ast));
    case PostIncrExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<PostIncrExpressionAST*>(ast));
    case CppCastExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CppCastExpressionAST*>(ast));
    case TypeidExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeidExpressionAST*>(ast));
    case TypeidOfTypeExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeidOfTypeExpressionAST*>(ast));
    case UnaryExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<UnaryExpressionAST*>(ast));
    case AwaitExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AwaitExpressionAST*>(ast));
    case SizeofExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SizeofExpressionAST*>(ast));
    case SizeofTypeExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SizeofTypeExpressionAST*>(ast));
    case SizeofPackExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SizeofPackExpressionAST*>(ast));
    case AlignofTypeExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AlignofTypeExpressionAST*>(ast));
    case AlignofExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AlignofExpressionAST*>(ast));
    case NoexceptExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NoexceptExpressionAST*>(ast));
    case NewExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NewExpressionAST*>(ast));
    case DeleteExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DeleteExpressionAST*>(ast));
    case CastExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CastExpressionAST*>(ast));
    case ImplicitCastExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ImplicitCastExpressionAST*>(ast));
    case BinaryExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BinaryExpressionAST*>(ast));
    case ConditionalExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConditionalExpressionAST*>(ast));
    case YieldExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<YieldExpressionAST*>(ast));
    case ThrowExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThrowExpressionAST*>(ast));
    case AssignmentExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AssignmentExpressionAST*>(ast));
    case PackExpansionExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<PackExpansionExpressionAST*>(ast));
    case DesignatedInitializerClauseAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DesignatedInitializerClauseAST*>(ast));
    case TypeTraitsExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeTraitsExpressionAST*>(ast));
    case ConditionExpressionAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConditionExpressionAST*>(ast));
    case EqualInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<EqualInitializerAST*>(ast));
    case BracedInitListAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BracedInitListAST*>(ast));
    case ParenInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ParenInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected Expression");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, TemplateParameterAST* ast) {
  switch (ast->kind()) {
    case TemplateTypeParameterAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TemplateTypeParameterAST*>(ast));
    case NonTypeTemplateParameterAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NonTypeTemplateParameterAST*>(ast));
    case TypenameTypeParameterAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypenameTypeParameterAST*>(ast));
    case ConstraintTypeParameterAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstraintTypeParameterAST*>(ast));
    default:
      cxx_runtime_error("unexpected TemplateParameter");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, SpecifierAST* ast) {
  switch (ast->kind()) {
    case TypedefSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypedefSpecifierAST*>(ast));
    case FriendSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FriendSpecifierAST*>(ast));
    case ConstevalSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstevalSpecifierAST*>(ast));
    case ConstinitSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstinitSpecifierAST*>(ast));
    case ConstexprSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstexprSpecifierAST*>(ast));
    case InlineSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<InlineSpecifierAST*>(ast));
    case StaticSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<StaticSpecifierAST*>(ast));
    case ExternSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExternSpecifierAST*>(ast));
    case ThreadLocalSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThreadLocalSpecifierAST*>(ast));
    case ThreadSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThreadSpecifierAST*>(ast));
    case MutableSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<MutableSpecifierAST*>(ast));
    case VirtualSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<VirtualSpecifierAST*>(ast));
    case ExplicitSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExplicitSpecifierAST*>(ast));
    case AutoTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AutoTypeSpecifierAST*>(ast));
    case VoidTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<VoidTypeSpecifierAST*>(ast));
    case SizeTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SizeTypeSpecifierAST*>(ast));
    case SignTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SignTypeSpecifierAST*>(ast));
    case VaListTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<VaListTypeSpecifierAST*>(ast));
    case IntegralTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<IntegralTypeSpecifierAST*>(ast));
    case FloatingPointTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FloatingPointTypeSpecifierAST*>(ast));
    case ComplexTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ComplexTypeSpecifierAST*>(ast));
    case NamedTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NamedTypeSpecifierAST*>(ast));
    case AtomicTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AtomicTypeSpecifierAST*>(ast));
    case UnderlyingTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<UnderlyingTypeSpecifierAST*>(ast));
    case ElaboratedTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ElaboratedTypeSpecifierAST*>(ast));
    case DecltypeAutoSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DecltypeAutoSpecifierAST*>(ast));
    case DecltypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DecltypeSpecifierAST*>(ast));
    case PlaceholderTypeSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<PlaceholderTypeSpecifierAST*>(ast));
    case ConstQualifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConstQualifierAST*>(ast));
    case VolatileQualifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<VolatileQualifierAST*>(ast));
    case RestrictQualifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<RestrictQualifierAST*>(ast));
    case EnumSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<EnumSpecifierAST*>(ast));
    case ClassSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ClassSpecifierAST*>(ast));
    case TypenameSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypenameSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected Specifier");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, PtrOperatorAST* ast) {
  switch (ast->kind()) {
    case PointerOperatorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<PointerOperatorAST*>(ast));
    case ReferenceOperatorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ReferenceOperatorAST*>(ast));
    case PtrToMemberOperatorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<PtrToMemberOperatorAST*>(ast));
    default:
      cxx_runtime_error("unexpected PtrOperator");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, CoreDeclaratorAST* ast) {
  switch (ast->kind()) {
    case BitfieldDeclaratorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BitfieldDeclaratorAST*>(ast));
    case ParameterPackAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ParameterPackAST*>(ast));
    case IdDeclaratorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<IdDeclaratorAST*>(ast));
    case NestedDeclaratorAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NestedDeclaratorAST*>(ast));
    default:
      cxx_runtime_error("unexpected CoreDeclarator");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, DeclaratorChunkAST* ast) {
  switch (ast->kind()) {
    case FunctionDeclaratorChunkAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<FunctionDeclaratorChunkAST*>(ast));
    case ArrayDeclaratorChunkAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ArrayDeclaratorChunkAST*>(ast));
    default:
      cxx_runtime_error("unexpected DeclaratorChunk");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, UnqualifiedIdAST* ast) {
  switch (ast->kind()) {
    case NameIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NameIdAST*>(ast));
    case DestructorIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DestructorIdAST*>(ast));
    case DecltypeIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DecltypeIdAST*>(ast));
    case OperatorFunctionIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<OperatorFunctionIdAST*>(ast));
    case LiteralOperatorIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LiteralOperatorIdAST*>(ast));
    case ConversionFunctionIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ConversionFunctionIdAST*>(ast));
    case SimpleTemplateIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleTemplateIdAST*>(ast));
    case LiteralOperatorTemplateIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<LiteralOperatorTemplateIdAST*>(ast));
    case OperatorFunctionTemplateIdAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<OperatorFunctionTemplateIdAST*>(ast));
    default:
      cxx_runtime_error("unexpected UnqualifiedId");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, NestedNameSpecifierAST* ast) {
  switch (ast->kind()) {
    case GlobalNestedNameSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<GlobalNestedNameSpecifierAST*>(ast));
    case SimpleNestedNameSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleNestedNameSpecifierAST*>(ast));
    case DecltypeNestedNameSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DecltypeNestedNameSpecifierAST*>(ast));
    case TemplateNestedNameSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TemplateNestedNameSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected NestedNameSpecifier");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, FunctionBodyAST* ast) {
  switch (ast->kind()) {
    case DefaultFunctionBodyAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DefaultFunctionBodyAST*>(ast));
    case CompoundStatementFunctionBodyAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CompoundStatementFunctionBodyAST*>(ast));
    case TryStatementFunctionBodyAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TryStatementFunctionBodyAST*>(ast));
    case DeleteFunctionBodyAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DeleteFunctionBodyAST*>(ast));
    default:
      cxx_runtime_error("unexpected FunctionBody");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, TemplateArgumentAST* ast) {
  switch (ast->kind()) {
    case TypeTemplateArgumentAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeTemplateArgumentAST*>(ast));
    case ExpressionTemplateArgumentAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ExpressionTemplateArgumentAST*>(ast));
    default:
      cxx_runtime_error("unexpected TemplateArgument");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, ExceptionSpecifierAST* ast) {
  switch (ast->kind()) {
    case ThrowExceptionSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThrowExceptionSpecifierAST*>(ast));
    case NoexceptSpecifierAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NoexceptSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected ExceptionSpecifier");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, RequirementAST* ast) {
  switch (ast->kind()) {
    case SimpleRequirementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleRequirementAST*>(ast));
    case CompoundRequirementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CompoundRequirementAST*>(ast));
    case TypeRequirementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeRequirementAST*>(ast));
    case NestedRequirementAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NestedRequirementAST*>(ast));
    default:
      cxx_runtime_error("unexpected Requirement");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, NewInitializerAST* ast) {
  switch (ast->kind()) {
    case NewParenInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NewParenInitializerAST*>(ast));
    case NewBracedInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<NewBracedInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected NewInitializer");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, MemInitializerAST* ast) {
  switch (ast->kind()) {
    case ParenMemInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ParenMemInitializerAST*>(ast));
    case BracedMemInitializerAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<BracedMemInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected MemInitializer");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, LambdaCaptureAST* ast) {
  switch (ast->kind()) {
    case ThisLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ThisLambdaCaptureAST*>(ast));
    case DerefThisLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<DerefThisLambdaCaptureAST*>(ast));
    case SimpleLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleLambdaCaptureAST*>(ast));
    case RefLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<RefLambdaCaptureAST*>(ast));
    case RefInitLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<RefInitLambdaCaptureAST*>(ast));
    case InitLambdaCaptureAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<InitLambdaCaptureAST*>(ast));
    default:
      cxx_runtime_error("unexpected LambdaCapture");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, ExceptionDeclarationAST* ast) {
  switch (ast->kind()) {
    case EllipsisExceptionDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<EllipsisExceptionDeclarationAST*>(ast));
    case TypeExceptionDeclarationAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<TypeExceptionDeclarationAST*>(ast));
    default:
      cxx_runtime_error("unexpected ExceptionDeclaration");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, AttributeSpecifierAST* ast) {
  switch (ast->kind()) {
    case CxxAttributeAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<CxxAttributeAST*>(ast));
    case GccAttributeAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<GccAttributeAST*>(ast));
    case AlignasAttributeAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AlignasAttributeAST*>(ast));
    case AlignasTypeAttributeAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AlignasTypeAttributeAST*>(ast));
    case AsmAttributeAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<AsmAttributeAST*>(ast));
    default:
      cxx_runtime_error("unexpected AttributeSpecifier");
  }  // switch
}

template <typename Visitor>
auto visit(Visitor&& visitor, AttributeTokenAST* ast) {
  switch (ast->kind()) {
    case ScopedAttributeTokenAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<ScopedAttributeTokenAST*>(ast));
    case SimpleAttributeTokenAST::Kind:
      return std::invoke(std::forward<Visitor>(visitor),
                         static_cast<SimpleAttributeTokenAST*>(ast));
    default:
      cxx_runtime_error("unexpected AttributeToken");
  }  // switch
}

template <typename T>
[[nodiscard]] auto ast_cast(AST* ast) -> T* {
  return ast && ast->kind() == T::Kind ? static_cast<T*>(ast) : nullptr;
}

}  // namespace cxx
