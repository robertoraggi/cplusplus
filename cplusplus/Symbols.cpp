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
#include "Types.h"
#include <iostream>
#include <cassert>

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
  _enclosingScope = enclosingScope;
}

void NamespaceSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "namespace";
  if (auto n = unqualifiedName())
    out << " " << n->toString();
  out << " {";
  out << std::endl;
  for (auto sym: symbols()) {
    sym->dump(out, depth + 1);
  }
  out << indent(depth) << "}";
  out << std::endl;
}

void BaseClassSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

void ClassSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "class";
  if (auto n = unqualifiedName())
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
  for (auto sym: symbols()) {
    sym->dump(out, depth + 1);
  }
  out << indent(depth) << "}";
  out << std::endl;
}

void TemplateSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << "template <";
  bool first = true;
  for (auto sym: symbols()) {
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

void FunctionSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << typeToString(type(), name());
  out << std::endl;
}

void BlockSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

void ArgumentSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

void DeclarationSymbol::dump(std::ostream& out, int depth) {
  out << indent(depth) << typeToString(type(), name());
  out << std::endl;
}

void TypeParameterSymbol::dump(std::ostream& out, int depth) {
  assert(!"todo");
}

NamespaceSymbol* Scope::findNamespace(const Name* name) const {
  auto q = name ? name->asQualifiedName() : nullptr;
  auto u = q ? q->name() : name;
  for (auto sym: _symbols) {
    if (sym->unqualifiedName() == u && sym->isNamespaceSymbol())
      return sym->asNamespaceSymbol();
  }
  return nullptr;
}

void TemplateSymbol::addSymbol(Symbol *symbol) {
  // nothing to do (for now).
  if (_symbol) {
    std::cout << "*** there is already a symbol in this template" << std::endl;
    _symbol->dump(std::cout, 0);
  }
  //assert(! _symbol);
  _symbol = symbol;
}
