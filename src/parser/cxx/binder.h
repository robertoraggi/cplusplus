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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace cxx {

class DeclSpecs;
class Decl;

class TranslationUnit;

class Binder {
 public:
  struct DefaultArgumentInfo {
    ExpressionAST* expression = nullptr;
    SourceLocation location = {};
  };

  explicit Binder(TranslationUnit* unit = nullptr);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  void setTranslationUnit(TranslationUnit* unit);

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto reportErrors() const -> bool;
  void setReportErrors(bool reportErrors);

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  [[nodiscard]] auto scope() const -> ScopeSymbol*;
  void setScope(ScopeSymbol* scope);

  [[nodiscard]] auto languageLinkage() const -> LanguageKind;
  void setLanguageLinkage(LanguageKind linkage);

  [[nodiscard]] auto changeLanguageLinkage(LanguageKind linkage)
      -> LanguageKind;

  [[nodiscard]] auto isInstantiating() const -> bool;
  [[nodiscard]] auto instantiatingSymbol() const -> Symbol*;
  void setInstantiatingSymbol(Symbol* symbol);

  [[nodiscard]] auto declaringScope() const -> ScopeSymbol*;

  [[nodiscard]] auto currentTemplateParameters() const
      -> TemplateParametersSymbol*;

  [[nodiscard]] auto inTemplate() const -> bool;

  [[nodiscard]] auto enterBlock(SourceLocation loc) -> BlockSymbol*;

  [[nodiscard]] auto declareTypeAlias(SourceLocation identifierLoc,
                                      TypeIdAST* typeId,
                                      bool addSymbolToParentScope = true)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareTypedef(DeclaratorAST* declarator, const Decl& decl)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareFunction(DeclaratorAST* declarator,
                                     const Decl& decl) -> FunctionSymbol*;

  [[nodiscard]] auto declareField(DeclaratorAST* declarator, const Decl& decl)
      -> FieldSymbol*;

  [[nodiscard]] auto declareVariable(DeclaratorAST* declarator,
                                     const Decl& decl,
                                     bool addSymbolToParentScope)
      -> VariableSymbol*;

  [[nodiscard]] auto declareMemberSymbol(DeclaratorAST* declarator,
                                         const Decl& decl) -> Symbol*;

  void applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs);

  void bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpec);

  void bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs,
            bool isDeclaration);

  void bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs);

  void complete(ClassSpecifierAST* ast);

  void bind(DecltypeSpecifierAST* ast);

  void bind(EnumeratorAST* ast, const Type* type,
            std::optional<ConstValue> value);

  void bind(ParameterDeclarationAST* ast, const Decl& decl,
            bool inTemplateParameters);

  void bind(UsingDeclaratorAST* ast, Symbol* target);

  void bind(BaseSpecifierAST* ast);

  void bind(NonTypeTemplateParameterAST* ast, int index, int depth);

  void bind(TypenameTypeParameterAST* ast, int index, int depth);

  void bind(ConstraintTypeParameterAST* ast, int index, int depth);

  void bind(TemplateTypeParameterAST* ast, int index, int depth);

  void bind(ConceptDefinitionAST* ast);

  void bind(LambdaExpressionAST* ast);

  void complete(LambdaExpressionAST* ast);

  void completeLambdaBody(LambdaExpressionAST* ast);

  void bind(ParameterDeclarationClauseAST* ast);

  void bind(UsingDirectiveAST* ast);

  void bind(TypeIdAST* ast, const Decl& decl);

  void bind(IdExpressionAST* ast);

  [[nodiscard]] auto resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId,
                             bool checkTemplates) -> Symbol*;

  [[nodiscard]] auto resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol*;

  [[nodiscard]] auto getFunction(ScopeSymbol* scope, const Name* name,
                                 const Type* type) -> FunctionSymbol*;

  class ScopeGuard {
   public:
    Binder* p = nullptr;
    ScopeSymbol* savedScope = nullptr;

    ScopeGuard(const ScopeGuard&) = delete;
    auto operator=(const ScopeGuard&) -> ScopeGuard& = delete;

    ScopeGuard() = default;

    explicit ScopeGuard(Binder* p, ScopeSymbol* scope = nullptr)
        : p(p), savedScope(p->scope_) {
      if (scope) p->setScope(scope);
    }

    ~ScopeGuard() { p->setScope(savedScope); }
  };

  [[nodiscard]] auto is_parsing_c() const -> bool;
  [[nodiscard]] auto is_parsing_cxx() const -> bool;

  void mergeDefaultArguments(FunctionSymbol* functionSymbol,
                             DeclaratorAST* declarator);

  void computeClassFlags(ClassSymbol* classSymbol);

 private:
  struct BindClass;
  struct CompleteClass;
  struct DeclareFunction;

 private:
  TranslationUnit* unit_ = nullptr;
  ScopeSymbol* scope_ = nullptr;
  Symbol* instantiatingSymbol_ = nullptr;
  LanguageKind languageLinkage_ = LanguageKind::kCXX;
  int lambdaCount_ = 0;
  bool inTemplate_ = false;
  bool reportErrors_ = true;
  std::unordered_map<FunctionSymbol*, std::vector<DefaultArgumentInfo>>
      defaultArguments_;
};

}  // namespace cxx
