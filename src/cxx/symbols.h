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

#include <cxx/fully_specified_type.h>
#include <cxx/names_fwd.h>
#include <cxx/symbol_visitor.h>
#include <cxx/symbols_fwd.h>

#include <memory>

namespace cxx {

class Symbol {
 public:
  Symbol(const Symbol& other) = delete;
  Symbol& operator=(const Symbol& other) = delete;

  Symbol(Scope* enclosingScope, const Name* name);
  virtual ~Symbol();

  const Name* name() const;
  void setName(const Name* name);

  Scope* enclosingScope() const;
  void setEnclosingScope(Scope* enclosingScope);

  NamespaceSymbol* enclosingNamespace() const;
  ClassSymbol* enclosingClass() const;
  FunctionSymbol* enclosingFunction() const;
  BlockSymbol* enclosingBlock() const;

  const FullySpecifiedType& type() const;
  void setType(const FullySpecifiedType& type);

  virtual Scope* scope() const { return nullptr; }

  virtual void accept(SymbolVisitor* visitor) = 0;

 private:
  const Name* name_ = nullptr;
  Scope* enclosingScope_ = nullptr;
  FullySpecifiedType type_;
};

class ConceptSymbol final : public Symbol {
 public:
  explicit ConceptSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class NamespaceSymbol final : public Symbol {
 public:
  explicit NamespaceSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~NamespaceSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  Scope* scope() const override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class ClassSymbol final : public Symbol {
 public:
  explicit ClassSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~ClassSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  Scope* scope() const override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class TypedefSymbol final : public Symbol {
 public:
  explicit TypedefSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class EnumSymbol final : public Symbol {
 public:
  explicit EnumSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class EnumeratorSymbol final : public Symbol {
 public:
  explicit EnumeratorSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class ScopedEnumSymbol final : public Symbol {
 public:
  explicit ScopedEnumSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class TemplateSymbol final : public Symbol {
 public:
  explicit TemplateSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class TemplateArgumentSymbol final : public Symbol {
 public:
  explicit TemplateArgumentSymbol(Scope* enclosingScope,
                                  const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class VariableSymbol final : public Symbol {
 public:
  explicit VariableSymbol(Scope* enclosingScope, const Name* name = nullptr);

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }
};

class FunctionSymbol final : public Symbol {
 public:
  explicit FunctionSymbol(Scope* enclosingScope, const Name* name = nullptr);
  ~FunctionSymbol() override;

  void accept(SymbolVisitor* visitor) override { visitor->visit(this); }

  Scope* scope() const override { return scope_.get(); }

  BlockSymbol* block() const;
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

  Scope* scope() const override { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

}  // namespace cxx
