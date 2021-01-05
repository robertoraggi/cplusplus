// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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

namespace cxx {

struct DeclarationASTVisitor {
  DeclarationASTVisitor();
  virtual ~DeclarationASTVisitor();

  virtual void visit(AliasDeclarationAST*) = 0;
  virtual void visit(AsmDeclarationAST*) = 0;
  virtual void visit(AttributeDeclarationAST*) = 0;
  virtual void visit(ConceptDefinitionAST*) = 0;
  virtual void visit(DeductionGuideAST*) = 0;
  virtual void visit(EmptyDeclarationAST*) = 0;
  virtual void visit(ExplicitInstantiationAST*) = 0;
  virtual void visit(ExportDeclarationAST*) = 0;
  virtual void visit(ForRangeDeclarationAST*) = 0;
  virtual void visit(LinkageSpecificationAST*) = 0;
  virtual void visit(ModuleImportDeclarationAST*) = 0;
  virtual void visit(NamespaceAliasDefinitionAST*) = 0;
  virtual void visit(NamespaceDefinitionAST*) = 0;
  virtual void visit(OpaqueEnumDeclarationAST*) = 0;
  virtual void visit(SimpleDeclarationAST*) = 0;
  virtual void visit(StaticAssertDeclarationAST*) = 0;
  virtual void visit(TemplateDeclarationAST*) = 0;
  virtual void visit(UsingDeclarationAST*) = 0;
  virtual void visit(UsingDirectiveAST*) = 0;
  virtual void visit(UsingEnumDeclarationAST*) = 0;
};

}  // namespace cxx
