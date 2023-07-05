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

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <vector>

namespace cxx {

class Control;

class TemplateHead {
 public:
  auto templateParameters() const -> const std::vector<TemplateParameter*>& {
    return templateParameters_;
  }

  void addTemplateParameter(TemplateParameter* templateParameter);

  auto isTemplateSpecialization() const -> bool {
    return isTemplateSpecialization_;
  }

 private:
  std::vector<TemplateParameter*> templateParameters_;
  bool isTemplateSpecialization_ = false;
};

class TemplateParameter {
 public:
  TemplateParameterKind kind;
  const Name* name = nullptr;
  const Type* type = nullptr;

  static auto makeTypeParameter(const Name* name) -> TemplateParameter*;
  static auto makeTypeParameterPack(const Name* name) -> TemplateParameter*;
  static auto makeNonTypeParameter(const Type* type, const Name* name)
      -> TemplateParameter*;
};

class Symbol {
 public:
  Symbol* next = nullptr;  // internal

  explicit Symbol(SymbolKind kind, const Name* name, const Type* type)
      : kind_(kind), name_(name), type_(type) {}

  virtual ~Symbol();

  virtual void accept(SymbolVisitor* visitor) = 0;

  auto kind() const { return kind_; }

  auto name() const { return name_; }

  auto mangledName() const -> const std::string& { return mangledName_; }

  void setMangledName(std::string mangledName) {
    mangledName_ = std::move(mangledName);
  }

  auto type() const { return type_; }
  void setType(const Type* type) { type_ = type; }

  auto is(SymbolKind kind) const -> bool { return kind_ == kind; }
  auto isNot(SymbolKind kind) const -> bool { return kind_ != kind; }

  auto accessKind() const -> AccessKind { return accessKind_; }
  void setAccessKind(AccessKind accessKind) { accessKind_ = accessKind; }

  auto lang() const -> Lang { return lang_; }
  void setLang(Lang lang) { lang_ = lang; }

  auto size() const -> int { return size_; }
  void setSize(int size) { size_ = size; }

  auto alignment() const -> int { return alignment_; }
  void setAlignment(int alignment) { alignment_ = alignment; }

  auto isType() const -> bool;
  auto isMemberFunction() const -> bool;
  auto isGlobalNamespace() const -> bool;
  auto isClassOrNamespace() const -> bool;

  auto templateParameters() const -> const std::vector<TemplateParameter*>&;

  auto findTemplateInstance(TemplateArgumentList* templ_arguments,
                            Symbol** sym) const -> bool;

  auto isAnonymous() const -> bool;

  auto enclosingScope() const -> Scope* { return enclosingScope_; }
  void setEnclosingScope(Scope* scope) { enclosingScope_ = scope; }

  auto enclosingClassOrNamespace() const -> Symbol*;

  void addTemplateInstance(Symbol* instantiatedSymbol);

  auto definition() const -> SourceLocationRange { return definition_; }
  void setDefinition(SourceLocationRange definition) {
    definition_ = definition;
  }

  auto isExtern() const -> bool { return isExtern_; }
  void setExtern(bool value) { isExtern_ = value; }

  auto isStatic() const -> bool { return isStatic_; }
  void setStatic(bool value) { isStatic_ = value; }

  auto isInline() const -> bool { return isInline_; }
  void setInline(bool value) { isInline_ = value; }

  auto isConstexpr() const -> bool { return isConstexpr_; }
  void setConstexpr(bool value) { isConstexpr_ = value; }

  auto isTemplate() const -> bool { return isTemplate_; }
  void setTemplate(bool value) { isTemplate_ = value; }

  auto templateHead() -> TemplateHead* { return templateHead_; }

  auto templateArguments() -> TemplateArgumentList* {
    return templateArguments_;
  }

  auto templateInstances() -> const std::vector<Symbol*>& {
    return templateInstances_;
  }

  auto primaryTemplate() -> Symbol* { return primaryTemplate_; }

  auto members() -> Scope* { return members_; }

 private:
  SymbolKind kind_;
  const Name* name_ = nullptr;
  const Type* type_ = nullptr;
  Scope* enclosingScope_ = nullptr;
  std::string mangledName_;
  AccessKind accessKind_ = AccessKind::kPublic;
  Lang lang_ = LANG_CPP;
  int size_ = 0;
  int alignment_ = 0;
  SourceLocationRange definition_;
  bool isExtern_ = false;
  bool isStatic_ = false;
  bool isInline_ = false;
  bool isConstexpr_ = false;
  bool isTemplate_ = false;
  TemplateHead* templateHead_ = nullptr;
  TemplateArgumentList* templateArguments_ = nullptr;
  std::vector<Symbol*> templateInstances_;
  Symbol* primaryTemplate_ = nullptr;
  Scope* members_ = nullptr;
};

template <enum SymbolKind K>
class SymbolMaker : public Symbol {
 public:
  static constexpr SymbolKind Kind = K;

  explicit SymbolMaker(const Name* name, const Type* type = nullptr)
      : Symbol(K, name, type) {}
};

class ParameterSymbol final : public SymbolMaker<SymbolKind::kParameter> {
 public:
  ParameterSymbol(Control* control, const Name* name, const Type* type,
                  int index);

  void accept(SymbolVisitor* visitor) override;

 private:
  int index_ = 0;
};

class ClassSymbol final : public SymbolMaker<SymbolKind::kClass> {
 public:
  ClassSymbol(Control* control, const Name* name);

  void accept(SymbolVisitor* visitor) override;

  auto isDerivedFrom(ClassSymbol* symbol) -> bool;

  auto baseClasses() const -> const std::vector<Symbol*>& {
    return baseClasses_;
  }

  auto constructors() const -> FunctionSymbol* { return constructors_; }
  auto destructor() const -> FunctionSymbol* { return destructor_; }

  auto isUnion() const -> bool { return isUnion_; }
  auto isPolymorphic() const -> bool { return isPolymorphic_; }
  auto isComplete() const -> bool { return isComplete_; }
  auto isEmpty() const -> bool { return isEmpty_; }

 private:
  std::vector<Symbol*> baseClasses_;
  FunctionSymbol* constructors_ = nullptr;
  FunctionSymbol* defaultConstructor_ = nullptr;
  FunctionSymbol* destructor_ = nullptr;
  bool isUnion_ = false;
  bool isPolymorphic_ = false;
  bool isComplete_ = false;
  bool isEmpty_ = false;
};

class EnumeratorSymbol final : public SymbolMaker<SymbolKind::kEnumerator> {
 public:
  EnumeratorSymbol(Control* control, const Name* name, const Type* type,
                   long value);

  void accept(SymbolVisitor* visitor) override;

  auto value() const -> long { return value_; }

 private:
  long value_ = 0;
};

class FunctionSymbol final : public SymbolMaker<SymbolKind::kFunction> {
 public:
  FunctionSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;

  auto allocateStack(int size, int alignment) -> int;

 private:
  DeclarationAST* ast_ = nullptr;
  int stackSize_ = 0;
  bool isVirtual_ = false;
  bool isLambda_ = false;
};

class GlobalSymbol final : public SymbolMaker<SymbolKind::kGlobal> {
 public:
  GlobalSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;

 private:
  ConstValue constValue_;
};

class InjectedClassNameSymbol final
    : public SymbolMaker<SymbolKind::kInjectedClassName> {
 public:
  InjectedClassNameSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;
};

class DependentSymbol final : public SymbolMaker<SymbolKind::kDependent> {
 public:
  explicit DependentSymbol(Control* control);

  void accept(SymbolVisitor* visitor) override;
};

class LocalSymbol final : public SymbolMaker<SymbolKind::kLocal> {
 public:
  LocalSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;

 private:
  ConstValue constValue_;
};

class MemberSymbol final : public SymbolMaker<SymbolKind::kMember> {
 public:
  MemberSymbol(Control* control, const Name* name, const Type* type,
               int offset);

  void accept(SymbolVisitor* visitor) override;

 private:
  int offset_ = 0;
};

class NamespaceSymbol final : public SymbolMaker<SymbolKind::kNamespace> {
 public:
  NamespaceSymbol(Control* control, const Name* name);

  void accept(SymbolVisitor* visitor) override;
};

class NamespaceAliasSymbol final
    : public SymbolMaker<SymbolKind::kNamespaceAlias> {
 public:
  NamespaceAliasSymbol(Control* control, const Name* name, Symbol* ns);

  void accept(SymbolVisitor* visitor) override;
};

class NonTypeTemplateParameterSymbol final
    : public SymbolMaker<SymbolKind::kNonTypeTemplateParameter> {
 public:
  NonTypeTemplateParameterSymbol(Control* control, const Name* name,
                                 const Type* type, int index);

  void accept(SymbolVisitor* visitor) override;

 private:
  int index_ = 0;
};

class ScopedEnumSymbol final : public SymbolMaker<SymbolKind::kScopedEnum> {
 public:
  ScopedEnumSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;
};

class TemplateParameterPackSymbol final
    : public SymbolMaker<SymbolKind::kTemplateParameterPack> {
 public:
  TemplateParameterPackSymbol(Control* control, const Name* name, int index);

  void accept(SymbolVisitor* visitor) override;

 private:
  int index_ = 0;
};

class TemplateParameterSymbol final
    : public SymbolMaker<SymbolKind::kTemplateParameter> {
 public:
  TemplateParameterSymbol(Control* control, const Name* name, int index);

  void accept(SymbolVisitor* visitor) override;

 private:
  int index_ = 0;
};

class ConceptSymbol final : public SymbolMaker<SymbolKind::kConcept> {
 public:
  ConceptSymbol(Control* control, const Name* name);

  void accept(SymbolVisitor* visitor) override;
};

class TypeAliasSymbol final : public SymbolMaker<SymbolKind::kTypeAlias> {
 public:
  TypeAliasSymbol(Control* control, const Name* name, const Type* type);

  void accept(SymbolVisitor* visitor) override;
};

class ValueSymbol final : public SymbolMaker<SymbolKind::kValue> {
 public:
  ValueSymbol(Control* control, const Name* name, const Type* type, long value);

  void accept(SymbolVisitor* visitor) override;

  auto value() const -> long { return value_; }

 private:
  long value_ = 0;
};

template <typename T>
auto symbol_cast(Symbol* symbol) -> T* {
  if (symbol && symbol->is(T::Kind)) return static_cast<T*>(symbol);
  return nullptr;
}

}  // namespace cxx
