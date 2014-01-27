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

#include "Codegen.h"
#include "AST.h"
#include "TranslationUnit.h"
#include "IR.h"
#include <cassert>

Codegen::Codegen(TranslationUnit* unit)
  : unit(unit) {
}

Codegen::~Codegen() {
}

Control* Codegen::control() const {
  return unit->control();
}

void Codegen::operator()(TranslationUnitAST* ast) {
  accept(ast);
}

Codegen::Result Codegen::reduce(const Result& expr) {
  return expr;
}

Codegen::Result Codegen::expression(ExpressionAST* ast) {
  Result r{ex};
  if (ast) {
    std::swap(result, r);
    accept(ast);
    std::swap(result, r);
  }
  assert(r.is(ex));
  return r;
}

void Codegen::condition(ExpressionAST* ast,
                        IR::BasicBlock* iftrue,
                        IR::BasicBlock* iffalse) {
  Result r{iftrue, iffalse};
  if (ast) {
    std::swap(result, r);
    accept(ast);
    std::swap(result, r);
  }
  assert(r.is(cx));
}

void Codegen::statement(ExpressionAST* ast) {
  Result r{nx};
  if (ast) {
    std::swap(result, r);
    accept(ast);
    std::swap(result, r);
  }
  assert(r.is(nx));
}

void Codegen::statement(StatementAST* ast) {
  accept(ast);
}

void Codegen::declaration(DeclarationAST* ast) {
  accept(ast);
}

void Codegen::accept(AST* ast) {
  if (! ast)
    return;
  switch (ast->kind()) {
#define VISIT_AST(x) case ASTKind::k##x: visit(reinterpret_cast<x##AST*>(ast)); break;
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
  } // switch
}

// ASTs
void Codegen::visit(AttributeAST* ast) {
}

void Codegen::visit(BaseClassAST* ast) {
}

void Codegen::visit(DeclaratorAST* ast) {
}

void Codegen::visit(EnumeratorAST* ast) {
}

void Codegen::visit(LambdaCaptureAST* ast) {
}

void Codegen::visit(LambdaDeclaratorAST* ast) {
}

void Codegen::visit(MemInitializerAST* ast) {
}

void Codegen::visit(ParametersAndQualifiersAST* ast) {
}

void Codegen::visit(PtrOperatorAST* ast) {
}

void Codegen::visit(TranslationUnitAST* ast) {
}

// core declarators
void Codegen::visit(DeclaratorIdAST* ast) {
}

void Codegen::visit(NestedDeclaratorAST* ast) {
}

// declarations
void Codegen::visit(AccessDeclarationAST* ast) {
}

void Codegen::visit(AliasDeclarationAST* ast) {
}

void Codegen::visit(AsmDefinitionAST* ast) {
}

void Codegen::visit(FunctionDefinitionAST* ast) {
}

void Codegen::visit(LinkageSpecificationAST* ast) {
}

void Codegen::visit(NamespaceAliasDefinitionAST* ast) {
}

void Codegen::visit(NamespaceDefinitionAST* ast) {
}

void Codegen::visit(OpaqueEnumDeclarationAST* ast) {
}

void Codegen::visit(ParameterDeclarationAST *ast) {
}

void Codegen::visit(SimpleDeclarationAST* ast) {
}

void Codegen::visit(StaticAssertDeclarationAST* ast) {
}

void Codegen::visit(TemplateDeclarationAST* ast) {
}

void Codegen::visit(TemplateTypeParameterAST* ast) {
}

void Codegen::visit(TypeParameterAST* ast) {
}

void Codegen::visit(UsingDeclarationAST* ast) {
}

void Codegen::visit(UsingDirectiveAST* ast) {
}

// expressions
void Codegen::visit(AlignofExpressionAST* ast) {
}

void Codegen::visit(BinaryExpressionAST* ast) {
}

void Codegen::visit(BracedInitializerAST* ast) {
}

void Codegen::visit(BracedTypeCallExpressionAST* ast) {
}

void Codegen::visit(CallExpressionAST* ast) {
}

void Codegen::visit(CastExpressionAST* ast) {
}

void Codegen::visit(ConditionAST* ast) {
}

void Codegen::visit(ConditionalExpressionAST* ast) {
}

void Codegen::visit(CppCastExpressionAST* ast) {
}

void Codegen::visit(DeleteExpressionAST* ast) {
}

void Codegen::visit(IdExpressionAST* ast) {
}

void Codegen::visit(IncrExpressionAST* ast) {
}

void Codegen::visit(LambdaExpressionAST* ast) {
}

void Codegen::visit(LiteralExpressionAST* ast) {
}

void Codegen::visit(MemberExpressionAST* ast) {
}

void Codegen::visit(NestedExpressionAST* ast) {
}

void Codegen::visit(NewExpressionAST* ast) {
}

void Codegen::visit(NoexceptExpressionAST* ast) {
}

void Codegen::visit(PackedExpressionAST* ast) {
}

void Codegen::visit(SimpleInitializerAST* ast) {
}

void Codegen::visit(SizeofExpressionAST* ast) {
}

void Codegen::visit(SizeofPackedArgsExpressionAST* ast) {
}

void Codegen::visit(SizeofTypeExpressionAST* ast) {
}

void Codegen::visit(SubscriptExpressionAST* ast) {
}

void Codegen::visit(TemplateArgumentAST* ast) {
}

void Codegen::visit(ThisExpressionAST* ast) {
}

void Codegen::visit(TypeCallExpressionAST* ast) {
}

void Codegen::visit(TypeIdAST* ast) {
}

void Codegen::visit(TypeidExpressionAST* ast) {
}

void Codegen::visit(UnaryExpressionAST* ast) {
}

// names
void Codegen::visit(DecltypeAutoNameAST* ast) {
}

void Codegen::visit(DecltypeNameAST* ast) {
}

void Codegen::visit(DestructorNameAST* ast) {
}

void Codegen::visit(OperatorNameAST* ast) {
}

void Codegen::visit(PackedNameAST* ast) {
}

void Codegen::visit(QualifiedNameAST* ast) {
}

void Codegen::visit(SimpleNameAST* ast) {
}

void Codegen::visit(TemplateIdAST* ast) {
}

// postfix declarations
void Codegen::visit(ArrayDeclaratorAST* ast) {
}

void Codegen::visit(FunctionDeclaratorAST* ast) {
}


// specifiers
void Codegen::visit(AlignasAttributeSpecifierAST* ast) {
}

void Codegen::visit(AlignasTypeAttributeSpecifierAST* ast) {
}

void Codegen::visit(AttributeSpecifierAST* ast) {
}

void Codegen::visit(ClassSpecifierAST* ast) {
}

void Codegen::visit(ElaboratedTypeSpecifierAST* ast) {
}

void Codegen::visit(EnumSpecifierAST* ast) {
}

void Codegen::visit(ExceptionSpecificationAST* ast) {
}

void Codegen::visit(NamedSpecifierAST* ast) {
}

void Codegen::visit(SimpleSpecifierAST* ast) {
}

void Codegen::visit(TypenameSpecifierAST* ast) {
}


// statements
void Codegen::visit(BreakStatementAST* ast) {
}

void Codegen::visit(CaseStatementAST* ast) {
}

void Codegen::visit(CompoundStatementAST* ast) {
}

void Codegen::visit(ContinueStatementAST* ast) {
}

void Codegen::visit(DeclarationStatementAST* ast) {
}

void Codegen::visit(DefaultStatementAST* ast) {
}

void Codegen::visit(DoStatementAST* ast) {
}

void Codegen::visit(ExpressionStatementAST* ast) {
}

void Codegen::visit(ForRangeStatementAST* ast) {
}

void Codegen::visit(ForStatementAST* ast) {
}

void Codegen::visit(GotoStatementAST* ast) {
}

void Codegen::visit(IfStatementAST* ast) {
}

void Codegen::visit(LabeledStatementAST* ast) {
}

void Codegen::visit(ReturnStatementAST* ast) {
}

void Codegen::visit(SwitchStatementAST* ast) {
}

void Codegen::visit(TryBlockStatementAST* ast) {
}

void Codegen::visit(WhileStatementAST* ast) {
}
