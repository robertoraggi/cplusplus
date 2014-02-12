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

#include "Symbols.h"
#include "Names.h"
#include "Token.h"
#include <functional>
#include <algorithm>
#include <iostream>
#include <cassert>

class SymbolTable {
public:
  SymbolTable(const SymbolTable& other) = delete;
  SymbolTable& operator=(const SymbolTable& other) = delete;

  SymbolTable(): _buckets(4) {}
  ~SymbolTable() = default;

  using iterator = Symbol**;

  inline iterator begin() {
    return ! _symbols.empty() ? &_symbols[0] : nullptr;
  }

  inline iterator end() {
    return begin() + _symbols.size();
  }

  inline unsigned symbolCount() const {
    return _symbols.size();
  }

  inline Symbol* symbolAt(unsigned index) const {
    return _symbols[index];
  }

  void addSymbol(Symbol* symbol) {
    _symbols.push_back(symbol);
    if (_symbols.size() * 3 >= _buckets.size() * 2) {
      rehash();
    } else {
      auto h = hashValue(symbol->name()) % _buckets.size();
      symbol->_next = _buckets[h];
      _buckets[h] = symbol;
    }
  }

  Symbol* findSymbol(const Name* name) const {
    if (! _symbols.empty()) {
      auto h = hashValue(name) % _buckets.size();
      for (auto symbol = _buckets[h]; symbol; symbol = symbol->_next) {
        if (symbol->name() == name)
          return symbol;
      }
    }
    return nullptr;
  }

  void rehash() {
    _buckets.resize(_buckets.size() * 2);
    std::fill(_buckets.begin(), _buckets.end(), nullptr);
    for (auto symbol: _symbols) {
      auto h = hashValue(symbol->name()) % _buckets.size();
      symbol->_next = _buckets[h];
      _buckets[h] = symbol;
    }
  }

private:
  std::vector<Symbol*> _symbols;
  std::vector<Symbol*> _buckets;
  std::hash<const Name*> hashValue;
};

namespace {
static std::string indent(int depth) {
  return std::string(4 * depth, ' ');
}
TypeToString typeToString;
} // anonymous namespace

const Name* Symbol::unqualifiedName() const {
  auto q = _name ? _name->asQualifiedName() : nullptr;
  return q ? q->name() : _name;
}

const Name* Symbol::name() const {
  return _name;
}

void Symbol::setName(const Name* name) {
  _name = name;
}

QualType Symbol::type() const {
  return _type;
}

void Symbol::setType(const QualType& type) {
  _type = type;
}

Scope* Symbol::enclosingScope() const {
  return _enclosingScope;
}

void Symbol::setEnclosingScope(Scope* enclosingScope) {
  assert(enclosingScope != this);
  _enclosingScope = enclosingScope;
}

Symbol* Symbol::next() const {
  return _next;
}

void NamespaceSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "namespace";
  if (auto n = unqualifiedName())
    out << " " << n->toString();
  out << " {";
  out << std::endl;
  for (auto sym: *this) {
    sym->dump(out, depth + 1);
  }
  out << indent(depth) << "}";
  out << std::endl;
}

void BaseClassSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

void ClassSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << token_spell[_classKey];
  if (auto n = name())
    out << " " << n->toString();
  bool first = true;
  for (auto&& bc: _baseClasses) {
    if (first) {
      out << ": ";
      first = false;
    } else {
      out << ", ";
    }
    out << bc->name()->toString();
  }
  out << " {";
  out << std::endl;
  for (auto sym: *this) {
    sym->dump(out, depth + 1);
  }
  out << indent(depth) << "}";
  out << ';' << std::endl;
}

TokenKind ClassSymbol::classKey() const {
  return _classKey;
}

void ClassSymbol::setClassKey(TokenKind classKey) {
  _classKey = classKey;
}

void TemplateSymbol::addParameter(Symbol* param) {
  this->Scope::addSymbol(param);
}

void TemplateSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "template <";
  bool first = true;
  for (auto sym: *this) {
    if (first)
      first = false;
    else
      out << ", ";
    sym->dump(out, depth + 1);
  }
  out << ">" << std::endl;
  if (auto decl = symbol())
    decl->dump(out, depth + 1);
  else
    out << indent(depth + 1) << "@template-declaration" << std::endl;
}

void TemplateSymbol::addSymbol(Symbol* symbol) {
  assert(! _symbol);
  _symbol = symbol;
  auto scope = enclosingScope();
  assert(scope);
  scope->addSymbol(symbol);
}

TokenKind FunctionSymbol::storageClassSpecifier() const {
  return _storageClassSpecifier;
}

void FunctionSymbol::setStorageClassSpecifier(TokenKind storageClassSpecifier) {
  _storageClassSpecifier = storageClassSpecifier;
}

ArgumentSymbol* FunctionSymbol::argumentAt(unsigned index) const {
  auto arg = symbolAt(index)->asArgumentSymbol();
  assert(arg);
  return arg;
}

void FunctionSymbol::addArgument(ArgumentSymbol* arg) {
  addSymbol(arg);
}

void FunctionSymbol::addSymbol(Symbol* symbol) {
  if (symbol->isArgumentSymbol()) {
    Scope::addSymbol(symbol);
  } else if (auto block = symbol->asBlockSymbol()) {
    setBlock(block);
  } else {
    assert(!"unreachable");
  }
}

void FunctionSymbol::dump(std::ostream& out, int depth) {
  auto funTy = type()->asFunctionType();
  assert(funTy);
  std::vector<const Name*> actuals;
  for (auto&& arg: *this)
    actuals.push_back(arg->name()); // ### this is a bit slow.
  out << indent(depth);
  if (_storageClassSpecifier)
    out << token_spell[_storageClassSpecifier] << ' ';
  out << typeToString(funTy->returnType(), name())
      << typeToString.prototype(funTy, actuals);
  out << " {}" << std::endl;
}

unsigned FunctionSymbol::sourceLocation() const {
  return _sourceLocation;
}

void FunctionSymbol::setSourceLocation(unsigned sourceLocation) {
  _sourceLocation = sourceLocation;
}

FunctionDefinitionAST* FunctionSymbol::internalNode() const {
  return _internalNode;
}

void FunctionSymbol::setInternalNode(FunctionDefinitionAST* internalNode) {
  _internalNode = internalNode;
}

IR::Function* FunctionSymbol::code() const {
  return _code;
}

void FunctionSymbol::setCode(IR::Function* code) {
  _code = code;
}

BlockSymbol* FunctionSymbol::block() const {
  return _block;
}

void FunctionSymbol::setBlock(BlockSymbol* block) {
  assert(! _block);
  _block = block;
}

void BlockSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

void ArgumentSymbol::dump(std::ostream& out, int depth) {
  out << typeToString(type(), name());
}

TokenKind DeclarationSymbol::storageClassSpecifier() const {
  return _storageClassSpecifier;
}

void DeclarationSymbol::setStorageClassSpecifier(TokenKind storageClassSpecifier) {
  _storageClassSpecifier = storageClassSpecifier;
}

void DeclarationSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth);
  if (_storageClassSpecifier)
    out << token_spell[_storageClassSpecifier] << ' ';
  out << typeToString(type(), name());
  out << ';' << std::endl;
}

void TypedefSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "typedef " << typeToString(type(), name());
  out << ';' << std::endl;
}

void TypeParameterSymbol::dump(std::ostream& out, int) {
  out << "typename";
  if (auto id = name())
    out << " " << id->toString();
}

void TemplateTypeParameterSymbol::dump(std::ostream& out, int) {
  out << "template <@...@> class";
  if (auto id = name())
    out << " " << id->toString();
}

Scope::~Scope() {
  delete _symbols;
}

unsigned Scope::symbolCount() const {
  return _symbols ? _symbols->symbolCount() : 0;
}

Symbol* Scope::symbolAt(unsigned index) const {
  return _symbols ? _symbols->symbolAt(index) : nullptr;
}

Symbol* Scope::findSymbol(const Name* name) const {
  return _symbols ? _symbols->findSymbol(name) : nullptr;
}

void Scope::addSymbol(Symbol* symbol) {
  if (! _symbols)
    _symbols = new SymbolTable();
  _symbols->addSymbol(symbol);
}

Scope::iterator Scope::begin() const {
  return _symbols ? _symbols->begin() : nullptr;
}

Scope::iterator Scope::end() const {
  return _symbols ? _symbols->end() : nullptr;
}

NamespaceSymbol* Scope::findNamespace(const Name* name) const {
  if (name) {
    auto id = name->asIdentifier();
    assert(id);
    for (auto sym = findSymbol(name); sym; sym = sym->next()) {
      if (sym->name() != name)
        continue;
      if (auto ns = sym->asNamespaceSymbol())
        return ns;
    }
  }
  return nullptr;
}
