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

class SymbolTable;

class Symbol {
  friend class SymbolTable;
  SymbolKind _kind;
  const Name* _name{nullptr};
  Scope* _enclosingScope{nullptr};
  Symbol* _next{nullptr};
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

  Symbol* next() const;

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
  ~Scope() override;

  Scope* asScope() override { return this; }

  unsigned symbolCount() const;
  Symbol* symbolAt(unsigned index) const;
  Symbol* findSymbol(const Name* name) const;
  virtual void addSymbol(Symbol* symbol);

  using iterator = Symbol**;
  iterator begin() const;
  iterator end() const;

  NamespaceSymbol* currentNamespace();
  ClassSymbol* currentClass();
  FunctionSymbol* currentFunction();

private:
  SymbolTable* _symbols{nullptr};
};

class NamespaceSymbol final: public ExtendsSymbol<SymbolKind::kNamespace, Scope> {
public:
  const std::vector<NamespaceSymbol*>& usings() const { return _usings; }
  void addUsing(NamespaceSymbol* u) { _usings.push_back(u); }

  void dump(std::ostream& out, int depth) override;

private:
  std::vector<NamespaceSymbol*> _usings;
};

class BaseClassSymbol final: public ExtendsSymbol<SymbolKind::kBaseClass, Symbol> {
public:
  ClassSymbol* symbol() const { return _symbol; }
  void setSymbol(ClassSymbol* symbol) { _symbol = symbol; }

  void dump(std::ostream& out, int depth) override;

private:
  ClassSymbol* _symbol{nullptr};
};

class ClassSymbol final: public ExtendsSymbol<SymbolKind::kClass, Scope> {
public:
  TokenKind classKey() const;
  void setClassKey(TokenKind classKey);
  void dump(std::ostream& out, int depth) override;
  const std::vector<BaseClassSymbol*>& baseClasses() const { return _baseClasses; }
  void addBaseClass(BaseClassSymbol* baseClass) { _baseClasses.push_back(baseClass); }
  bool isCompleted() const { return _isCompleted; }
  void setCompleted(bool isCompleted) { _isCompleted = isCompleted; }
private:
  TokenKind _classKey{T_EOF_SYMBOL};
  std::vector<BaseClassSymbol*> _baseClasses;
  bool _isCompleted{false};
};

class EnumSymbol final: public ExtendsSymbol<SymbolKind::kEnum, Scope> {
public:
  void dump(std::ostream& out, int depth) override;
};

class TemplateSymbol final: public ExtendsSymbol<SymbolKind::kTemplate, Scope> {
public:
  void addParameter(Symbol* param);
  Symbol* symbol() const { return _symbol; }
  void setSymbol(Symbol* symbol) { _symbol = symbol; }
  void dump(std::ostream& out, int depth) override;
  void addSymbol(Symbol *symbol) override;
private:
  Symbol* _symbol;
};

class FunctionSymbol final: public ExtendsSymbol<SymbolKind::kFunction, Scope> {
public:
  TokenKind storageClassSpecifier() const;
  void setStorageClassSpecifier(TokenKind storageClassSpecifier);

  // ### FIXME
  unsigned argumentCount() const { return symbolCount(); }
  ArgumentSymbol* argumentAt(unsigned index) const;
  void addArgument(ArgumentSymbol* arg);
  void addSymbol(Symbol* symbol) override;

  void dump(std::ostream& out, int depth) override;

  // ### internal
  unsigned sourceLocation() const;
  void setSourceLocation(unsigned sourceLocation);

  FunctionDefinitionAST* internalNode() const;
  void setInternalNode(FunctionDefinitionAST* internalNode);

  IR::Function* code() const;
  void setCode(IR::Function* code);

  BlockSymbol* block() const;
  void setBlock(BlockSymbol* block);

private:
  QualType _returnType;
  BlockSymbol* _block{nullptr};
  FunctionDefinitionAST* _internalNode{nullptr};
  unsigned _sourceLocation{0};
  TokenKind _storageClassSpecifier{T_EOF_SYMBOL};
  IR::Function* _code{nullptr};
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
  TokenKind storageClassSpecifier() const;
  void setStorageClassSpecifier(TokenKind storageClassSpecifier);

  void dump(std::ostream& out, int depth) override;

private:
  TokenKind _storageClassSpecifier{T_EOF_SYMBOL};
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
