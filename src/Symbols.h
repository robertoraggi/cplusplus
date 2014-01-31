// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SYMBOLS_H
#define SYMBOLS_H

#include "Globals.h"
#include "Types.h"
#include <vector>

class Symbol {
  SymbolKind _kind;
  const Name* _name{nullptr};
  Scope* _enclosingScope{nullptr};
  QualType _type;
public:
  explicit Symbol(SymbolKind kind): _kind(kind) {}
  virtual ~Symbol() = default;

  virtual void dump(std::ostream& out, int depth) = 0;

  inline SymbolKind kind() const { return _kind; }

  virtual Scope* asScope() { return 0; }

  const Name* unqualifiedName() const;

  const Name* name() const;
  void setName(const Name* name);

  QualType type() const;
  void setType(const QualType& type);

  Scope* enclosingScope() const;
  void setEnclosingScope(Scope* enclosingScope);

#define VISIT_SYMBOL(T) \
  inline bool is##T##Symbol() const { \
    return _kind == SymbolKind::k##T; \
  } \
  inline const T##Symbol* as##T##Symbol() const { \
    return const_cast<Symbol*>(this)->as##T##Symbol(); \
  } \
  inline T##Symbol* as##T##Symbol() { \
    return is##T##Symbol() ? reinterpret_cast<T##Symbol*>(this) : nullptr; \
  }
  FOR_EACH_SYMBOL(VISIT_SYMBOL)
#undef VISIT_SYMBOL
};

template <SymbolKind K, typename Base = Symbol>
struct ExtendsSymbol: Base {
  inline ExtendsSymbol(): Base(K) {}
};

class Scope: public Symbol {
public:
  using Symbol::Symbol;

  Scope* asScope() override { return this; }

  unsigned symbolCount() const { return _symbols.size(); }
  Symbol* symbolAt(unsigned index) const { return _symbols[index]; }
  virtual void addSymbol(Symbol* symbol) { _symbols.push_back(symbol); }
  const std::vector<Symbol*>& symbols() const { return _symbols; }

  virtual NamespaceSymbol* findNamespace(const Name* name) const;

private:
  std::vector<Symbol*> _symbols; // ### TODO: index by name and types.
};

class NamespaceSymbol final: public ExtendsSymbol<SymbolKind::kNamespace, Scope> {
public:
  void dump(std::ostream& out, int depth) override;
};

class BaseClassSymbol final: public ExtendsSymbol<SymbolKind::kBaseClass, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

class ClassSymbol final: public ExtendsSymbol<SymbolKind::kClass, Scope> {
public:
  TokenKind classKey() const;
  void setClassKey(TokenKind classKey);
  void dump(std::ostream& out, int depth) override;
  const std::vector<BaseClassSymbol*>& baseClasses() const { return _baseClasses; }
  void addBaseClass(BaseClassSymbol* baseClass) { _baseClasses.push_back(baseClass); }
private:
  TokenKind _classKey;
  std::vector<BaseClassSymbol*> _baseClasses;
};

class TemplateSymbol final: public ExtendsSymbol<SymbolKind::kTemplate, Scope> {
public:
  const std::vector<Symbol*>& parameters() const { return _parameters; }
  void addParameter(Symbol* param) { this->Scope::addSymbol(param); }
  Symbol* symbol() const { return _symbol; }
  void setSymbol(Symbol* symbol) { _symbol = symbol; }
  void dump(std::ostream& out, int depth) override;
  void addSymbol(Symbol *symbol) override;
private:
  Symbol* _symbol;
  std::vector<Symbol*> _parameters;
};

class FunctionSymbol final: public ExtendsSymbol<SymbolKind::kFunction, Scope> {
public:
  // ### FIXME
  unsigned argumentCount() const { return _arguments.size(); }
  ArgumentSymbol* argumentAt(unsigned index) const { return _arguments[index]; }
  void addArgument(ArgumentSymbol* arg) { _arguments.push_back(arg); }

  void dump(std::ostream& out, int depth) override;

  // ### internal
  unsigned sourceLocation() const;
  void setSourceLocation(unsigned sourceLocation);

  StatementAST** internalNode() const;
  void setInternalNode(StatementAST** internalNode);

private:
  QualType _returnType;
  std::vector<ArgumentSymbol*> _arguments;
  StatementAST** _internalNode{nullptr};
  unsigned _sourceLocation{0};
  bool _isVariadic{false};
  bool _isConst{false};
};

class BlockSymbol final: public ExtendsSymbol<SymbolKind::kBlock, Scope> {
public:
  void dump(std::ostream& out, int depth) override;
};

class ArgumentSymbol final: public ExtendsSymbol<SymbolKind::kArgument, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

class DeclarationSymbol final: public ExtendsSymbol<SymbolKind::kDeclaration, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

class TypedefSymbol final: public ExtendsSymbol<SymbolKind::kTypedef, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

class TypeParameterSymbol final: public ExtendsSymbol<SymbolKind::kTypeParameter, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

class TemplateTypeParameterSymbol final: public ExtendsSymbol<SymbolKind::kTemplateTypeParameter, Symbol> {
public:
  void dump(std::ostream& out, int depth) override;
};

#endif // SYMBOLS_H
