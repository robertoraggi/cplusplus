// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names_fwd.h>
#include <cxx/qualified_type.h>
#include <cxx/symbol_visitor.h>
#include <cxx/symbols_fwd.h>

#include <memory>
#include <string>
#include <vector>

namespace cxx {

class Symbol {
 public:
  Symbol(const Symbol& other) = delete;
  auto operator=(const Symbol& other) -> Symbol& = delete;

  Symbol(Scope* enclosingScope, const Name* name);
  virtual ~Symbol();

  [[nodiscard]] auto unqualifiedId() const -> std::string;
  [[nodiscard]] auto qualifiedId() const -> std::string;

  [[nodiscard]] auto name() const -> const Name* { return name_; }
  void setName(const Name* name) { name_ = name; }

  [[nodiscard]] auto next() const -> Symbol* { return next_; }
  void setNext(Symbol* next) { next_ = next; }

  [[nodiscard]] auto enclosingScope() const -> Scope* {
    return enclosingScope_;
  }
  void setEnclosingScope(Scope* enclosingScope) {
    enclosingScope_ = enclosingScope;
  }

  [[nodiscard]] auto enclosingClassOrNamespace() const -> Symbol*;
  [[nodiscard]] auto enclosingNamespace() const -> NamespaceSymbol*;
  [[nodiscard]] auto enclosingClass() const -> ClassSymbol*;
  [[nodiscard]] auto enclosingFunction() const -> FunctionSymbol*;
  [[nodiscard]] auto enclosingBlock() const -> BlockSymbol*;

  [[nodiscard]] auto type() const -> const QualifiedType&;
  void setType(const QualifiedType& type);

  [[nodiscard]] auto linkage() const -> Linkage;
  void setLinkage(Linkage linkage);

  [[nodiscard]] auto visibility() const -> Visibility;
  void setVisibility(Visibility visibility);

  [[nodiscard]] auto templateParameterList() const -> TemplateParameterList*;
  void setTemplateParameterList(TemplateParameterList* templateParameterList);

  [[nodiscard]] virtual auto isTypeSymbol() const -> bool;

  [[nodiscard]] virtual auto scope() const -> Scope* { return nullptr; }

  void addToEnclosingScope();

  [[nodiscard]] auto index() const -> int;

  virtual void accept(SymbolVisitor* visitor) = 0;

 private:
  Scope* enclosingScope_ = nullptr;
  const Name* name_ = nullptr;
  Symbol* next_ = nullptr;
  TemplateParameterList* templateParameterList_ = nullptr;
  QualifiedType type_;
  Linkage linkage_ = Linkage::kCxx;
  Visibility visibility_ = Visibility::kPublic;
};

class TypeSymbol : public Symbol {
 public:
  explicit TypeSymbol(Scope* enclosingScope, const Name* name);

  [[nodiscard]] auto isTypeSymbol() const -> bool override;
};

class ConceptSymbol final : public TypeSymbol {
 public:
  explicit ConceptSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class NamespaceSymbol final : public Symbol {
 public:
  explicit NamespaceSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~NamespaceSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

  [[nodiscard]] auto isInline() const -> bool { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

  [[nodiscard]] auto usingNamespaces() const
      -> const std::vector<NamespaceSymbol*>& {
    return usingNamespaces_;
  }

  void addUsingNamespace(NamespaceSymbol* symbol);

 private:
  std::unique_ptr<Scope> scope_;
  std::vector<NamespaceSymbol*> usingNamespaces_;
  bool isInline_ = false;
};

class ClassSymbol final : public TypeSymbol {
 public:
  explicit ClassSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~ClassSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

  [[nodiscard]] auto classKey() const -> ClassKey { return classKey_; }
  void setClassKey(ClassKey classKey) { classKey_ = classKey; }

  [[nodiscard]] auto baseClasses() const -> const std::vector<ClassSymbol*>& {
    return baseClasses_;
  }
  void addBaseClass(ClassSymbol* baseClass);

  [[nodiscard]] auto isDefined() const -> bool { return isDefined_; }
  void setDefined(bool isDefined) { isDefined_ = isDefined; }

 private:
  std::unique_ptr<Scope> scope_;
  std::vector<ClassSymbol*> baseClasses_;
  ClassKey classKey_ = ClassKey::kClass;
  bool isDefined_ = false;
};

class TypedefSymbol final : public TypeSymbol {
 public:
  explicit TypedefSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class EnumSymbol final : public TypeSymbol {
 public:
  explicit EnumSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~EnumSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class ScopedEnumSymbol final : public TypeSymbol {
 public:
  explicit ScopedEnumSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~ScopedEnumSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

  [[nodiscard]] auto underlyingType() const -> QualifiedType {
    return underlyingType_;
  }

  void setUnderlyingType(const QualifiedType& underlyingType) {
    underlyingType_ = underlyingType;
  }

 private:
  std::unique_ptr<Scope> scope_;
  QualifiedType underlyingType_;
};

class EnumeratorSymbol final : public Symbol {
 public:
  explicit EnumeratorSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class TemplateParameterList final : public Symbol {
 public:
  explicit TemplateParameterList(Scope* enclosingScope);
  ~TemplateParameterList() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class TemplateTypeParameterSymbol final : public TypeSymbol {
 public:
  explicit TemplateTypeParameterSymbol(Scope* enclosingScope,
                                       const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto defaultType() const -> QualifiedType {
    return defaultType_;
  }

  void setDefaultType(const QualifiedType& defaultType) {
    defaultType_ = defaultType;
  }

  [[nodiscard]] auto isParameterPack() const -> bool { return parameterPack_; }

  void setParameterPack(bool parameterPack) { parameterPack_ = parameterPack; }

 private:
  QualifiedType defaultType_;
  bool parameterPack_ = false;
};

class VariableSymbol final : public Symbol {
 public:
  explicit VariableSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class FieldSymbol final : public Symbol {
 public:
  explicit FieldSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class FunctionSymbol final : public Symbol {
 public:
  explicit FunctionSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~FunctionSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

  [[nodiscard]] auto block() const -> BlockSymbol*;
  void setBlock(BlockSymbol* block);

 private:
  std::unique_ptr<Scope> scope_;
  BlockSymbol* block_ = nullptr;
};

class ArgumentSymbol final : public Symbol {
 public:
  explicit ArgumentSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class BlockSymbol final : public Symbol {
 public:
  explicit BlockSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~BlockSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  [[nodiscard]] auto scope() const -> Scope* override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

}  // namespace cxx
