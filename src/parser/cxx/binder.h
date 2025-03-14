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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <optional>

namespace cxx {

class DeclSpecs;
class Decl;

class TranslationUnit;

class Binder {
 public:
  explicit Binder(TranslationUnit* unit = nullptr);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  void setTranslationUnit(TranslationUnit* unit);

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto reportErrors() const -> bool;
  void setReportErrors(bool reportErrors);

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  [[nodiscard]] auto scope() const -> Scope*;
  void setScope(Scope* scope);
  void setScope(ScopedSymbol* symbol);

  [[nodiscard]] auto declaringScope() const -> Scope*;

  [[nodiscard]] auto currentTemplateParameters() const
      -> TemplateParametersSymbol*;

  [[nodiscard]] auto inTemplate() const -> bool;

  [[nodiscard]] auto enterBlock(SourceLocation loc) -> BlockSymbol*;

  [[nodiscard]] auto declareTypeAlias(SourceLocation identifierLoc,
                                      TypeIdAST* typeId) -> TypeAliasSymbol*;

  [[nodiscard]] auto declareTypedef(DeclaratorAST* declarator, const Decl& decl)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareFunction(DeclaratorAST* declarator,
                                     const Decl& decl) -> FunctionSymbol*;

  [[nodiscard]] auto declareField(DeclaratorAST* declarator, const Decl& decl)
      -> FieldSymbol*;

  [[nodiscard]] auto declareVariable(DeclaratorAST* declarator,
                                     const Decl& decl) -> VariableSymbol*;

  [[nodiscard]] auto declareMemberSymbol(DeclaratorAST* declarator,
                                         const Decl& decl) -> Symbol*;

  void applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs);

  [[nodiscard]] auto isConstructor(Symbol* symbol) const -> bool;

  void bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpec);

  void bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs);

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

  void bind(ParameterDeclarationClauseAST* ast);

  void bind(UsingDirectiveAST* ast);

  void bind(TypeIdAST* ast, const Decl& decl);

  void bind(IdExpressionAST* ast);

  [[nodiscard]] auto instantiate(SimpleTemplateIdAST* templateId) -> Symbol*;

  [[nodiscard]] auto resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId,
                             bool canInstantiate) -> Symbol*;

  class ScopeGuard {
   public:
    Binder* p = nullptr;
    Scope* savedScope = nullptr;

    ScopeGuard(const ScopeGuard&) = delete;
    auto operator=(const ScopeGuard&) -> ScopeGuard& = delete;

    ScopeGuard() = default;

    explicit ScopeGuard(Binder* p, Scope* scope = nullptr)
        : p(p), savedScope(p->scope_) {
      if (scope) p->setScope(scope);
    }

    ~ScopeGuard() { p->setScope(savedScope); }
  };

 private:
  TranslationUnit* unit_ = nullptr;
  Scope* scope_ = nullptr;
  bool inTemplate_ = false;
  bool reportErrors_ = false;
};

}  // namespace cxx
