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

#include "ASTVisitor.h"
#include "AST.h"
#include "TranslationUnit.h"
#include <string>

static const char* const ast_name[] = {
#define VISIT_AST(x) #x,
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
};

Control* ASTVisitor::control() const {
  return unit->control();
}

RecursiveASTVisitor::~RecursiveASTVisitor() {
}

void RecursiveASTVisitor::operator()(TranslationUnitAST* ast) {
  if (preVisit(ast))
    accept(ast);
  postVisit(ast);
}

void RecursiveASTVisitor::accept(AST* ast) {
  if (! ast)
    return;

  static int depth = -1;

  ++depth;

  std::string ind(2 * depth, ' ');
  printf("%s%s\n", ind.c_str(), ast_name[(int)ast->kind()]);

  switch (ast->kind()) {
#define VISIT_AST(x) case ASTKind::k##x: visit(reinterpret_cast<x##AST*>(ast)); break;
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
  } // switch

  --depth;
}

void RecursiveASTVisitor::visit(TypeIdAST* ast) {
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->declarator);
}

void RecursiveASTVisitor::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declaration_list; it; it = it->next) {
    accept(it->value);
  }
}

void RecursiveASTVisitor::visit(ExceptionSpecificationAST* ast) {
}

void RecursiveASTVisitor::visit(AttributeAST* ast) {
}

void RecursiveASTVisitor::visit(AttributeSpecifierAST* ast) {
  for (auto it = ast->attribute_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(AlignasTypeAttributeSpecifierAST* ast) {
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(AlignasAttributeSpecifierAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(SimpleSpecifierAST* ast) {
}

void RecursiveASTVisitor::visit(NamedSpecifierAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(TypenameSpecifierAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(EnumeratorAST* ast) {
  accept(ast->name);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->enumerator_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(BaseClassAST* ast) {
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->base_class_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->declaration_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(QualifiedNameAST* ast) {
  accept(ast->base);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(PackedNameAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(SimpleNameAST* ast) {
}

void RecursiveASTVisitor::visit(DestructorNameAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(OperatorNameAST* ast) {
}

void RecursiveASTVisitor::visit(TemplateArgumentAST* ast) {
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(TemplateIdAST* ast) {
  accept(ast->name);
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(DecltypeNameAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(DecltypeAutoNameAST* ast) {
}

void RecursiveASTVisitor::visit(PackedExpressionAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(LiteralExpressionAST* ast) {
}

void RecursiveASTVisitor::visit(ThisExpressionAST* ast) {
}

void RecursiveASTVisitor::visit(IdExpressionAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(NestedExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(LambdaCaptureAST* ast) {
}

void RecursiveASTVisitor::visit(LambdaDeclaratorAST* ast) {
  accept(ast->exception_specification);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(LambdaExpressionAST* ast) {
  accept(ast->lambda_declarator);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(SubscriptExpressionAST* ast) {
  accept(ast->base_expression);
  accept(ast->index_expression);
}

void RecursiveASTVisitor::visit(CallExpressionAST* ast) {
  accept(ast->base_expression);
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(TypeCallExpressionAST* ast) {
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(BracedTypeCallExpressionAST* ast) {
  for (auto it = ast->type_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(MemberExpressionAST* ast) {
  accept(ast->base_expression);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(IncrExpressionAST* ast) {
  accept(ast->base_expression);
}

void RecursiveASTVisitor::visit(CppCastExpressionAST* ast) {
  accept(ast->type_id);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(TypeidExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(UnaryExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(SizeofExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(SizeofTypeExpressionAST* ast) {
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(SizeofPackedArgsExpressionAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(AlignofExpressionAST* ast) {
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(NoexceptExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(NewExpressionAST* ast) {
  for (auto it = ast->placement_expression_list; it; it = it->next)
    accept(it->value);
  accept(ast->type_id);
  accept(ast->initializer);
}

void RecursiveASTVisitor::visit(DeleteExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(CastExpressionAST* ast) {
  accept(ast->type_id);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(BinaryExpressionAST* ast) {
  accept(ast->left_expression);
  accept(ast->right_expression);
}

void RecursiveASTVisitor::visit(ConditionalExpressionAST* ast) {
  accept(ast->expression);
  accept(ast->iftrue_expression);
  accept(ast->iffalse_expression);
}

void RecursiveASTVisitor::visit(BracedInitializerAST* ast) {
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(SimpleInitializerAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(ConditionAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->declarator);
  accept(ast->initializer);
}

void RecursiveASTVisitor::visit(LabeledStatementAST* ast) {
  accept(ast->name);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(CaseStatementAST* ast) {
  accept(ast->expression);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(DefaultStatementAST* ast) {
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(ExpressionStatementAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statement_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(TryBlockStatementAST* ast) {
  accept(ast->statemebt);
}

void RecursiveASTVisitor::visit(DeclarationStatementAST* ast) {
  accept(ast->declaration);
}

void RecursiveASTVisitor::visit(IfStatementAST* ast) {
  accept(ast->condition);
  accept(ast->statement);
  accept(ast->else_statement);
}

void RecursiveASTVisitor::visit(SwitchStatementAST* ast) {
  accept(ast->condition);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(WhileStatementAST* ast) {
  accept(ast->condition);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(DoStatementAST* ast) {
  accept(ast->statement);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(ForStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->expression);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(ForRangeStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->expression);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(BreakStatementAST* ast) {
}

void RecursiveASTVisitor::visit(ContinueStatementAST* ast) {
}

void RecursiveASTVisitor::visit(ReturnStatementAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(GotoStatementAST* ast) {
  accept(ast->name);
}

void RecursiveASTVisitor::visit(AccessDeclarationAST* ast) {
}

void RecursiveASTVisitor::visit(MemInitializerAST* ast) {
  accept(ast->name);
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(FunctionDefinitionAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->declarator);
  for (auto it = ast->mem_initializer_list; it; it = it->next)
    accept(it->value);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(TypeParameterAST* ast) {
  accept(ast->name);
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(TemplateTypeParameterAST* ast) {
  for (auto it = ast->template_parameter_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(ParameterDeclarationAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->declarator);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->template_parameter_list; it; it = it->next)
    accept(it->value);
  accept(ast->declaration);
}

void RecursiveASTVisitor::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declaration_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(NamespaceDefinitionAST* ast) {
  const char* name = "";
//  if (auto id = dynamic_cast<SimpleNameAST*>(ast->name))
//    name = translationUnit()->tokenText(id->identifier_token);
  //translationUnit()->warning(ast->namespace_token, "enter namespace `%s'", name);
  accept(ast->name);
  for (auto it = ast->declaration_list; it; it = it->next)
    accept(it->value);
  //translationUnit()->warning(ast->namespace_token, "leave namespace `%s'", name);
}

void RecursiveASTVisitor::visit(AsmDefinitionAST* ast) {
  for (auto it = ast->expression_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  accept(ast->alias_name);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(UsingDirectiveAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->name);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(AliasDeclarationAST* ast) {
  accept(ast->alias_name);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->type_id);
}

void RecursiveASTVisitor::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->declarator_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptr_op_list; it; it = it->next)
    accept(it->value);
  accept(ast->core_declarator);
  for (auto it = ast->postfix_declarator_list; it; it = it->next)
    accept(it->value);
  accept(ast->initializer);
}

void RecursiveASTVisitor::visit(NestedDeclaratorAST* ast) {
  accept(ast->declarator);
}

void RecursiveASTVisitor::visit(DeclaratorIdAST* ast) {
//  if (auto id = dynamic_cast<SimpleNameAST*>(ast->name))
//    translationUnit()->warning(id->identifier_token, "declared `%s'",
//                               translationUnit()->tokenText(id->identifier_token));
  accept(ast->name);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(PtrOperatorAST* ast) {
  for (auto it = ast->nested_name_specifier; it; it = it->next)
    accept(it->value);
  for (auto it = ast->cv_qualifier_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(ArrayDeclaratorAST* ast) {
  accept(ast->size_expression);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(FunctionDeclaratorAST* ast) {
  accept(ast->parameters_and_qualifiers);
  for (auto it = ast->trailing_type_specifier_list; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(ParametersAndQualifiersAST* ast) {
  for (auto it = ast->parameter_list; it; it = it->next)
    accept(it->value);
  for (auto it = ast->specifier_list; it; it = it->next)
    accept(it->value);
  accept(ast->ref_qualifier);
  accept(ast->exception_specification);
  for (auto it = ast->attribute_specifier_list; it; it = it->next)
    accept(it->value);
}
