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

#ifndef ASTFWD_H
#define ASTFWD_H

#define FOR_EACH_AST(V) \
  V(TypeId) \
  V(TranslationUnit) \
  V(ExceptionSpecification) \
  V(Attribute) \
  V(AttributeSpecifier) \
  V(AlignasTypeAttributeSpecifier) \
  V(AlignasAttributeSpecifier) \
  V(SimpleSpecifier) \
  V(NamedSpecifier) \
  V(TypenameSpecifier) \
  V(ElaboratedTypeSpecifier) \
  V(Enumerator) \
  V(EnumSpecifier) \
  V(BaseClass) \
  V(ClassSpecifier) \
  V(QualifiedName) \
  V(PackedName) \
  V(SimpleName) \
  V(DestructorName) \
  V(OperatorName) \
  V(TemplateArgument) \
  V(TemplateId) \
  V(DecltypeName) \
  V(DecltypeAutoName) \
  V(PackedExpression) \
  V(LiteralExpression) \
  V(ThisExpression) \
  V(IdExpression) \
  V(NestedExpression) \
  V(LambdaCapture) \
  V(LambdaDeclarator) \
  V(LambdaExpression) \
  V(SubscriptExpression) \
  V(CallExpression) \
  V(TypeCallExpression) \
  V(BracedTypeCallExpression) \
  V(MemberExpression) \
  V(IncrExpression) \
  V(CppCastExpression) \
  V(TypeidExpression) \
  V(UnaryExpression) \
  V(SizeofExpression) \
  V(SizeofTypeExpression) \
  V(SizeofPackedArgsExpression) \
  V(AlignofExpression) \
  V(NoexceptExpression) \
  V(NewExpression) \
  V(DeleteExpression) \
  V(CastExpression) \
  V(BinaryExpression) \
  V(ConditionalExpression) \
  V(BracedInitializer) \
  V(SimpleInitializer) \
  V(Condition) \
  V(LabeledStatement) \
  V(CaseStatement) \
  V(DefaultStatement) \
  V(ExpressionStatement) \
  V(CompoundStatement) \
  V(TryBlockStatement) \
  V(DeclarationStatement) \
  V(IfStatement) \
  V(SwitchStatement) \
  V(WhileStatement) \
  V(DoStatement) \
  V(ForStatement) \
  V(ForRangeStatement) \
  V(BreakStatement) \
  V(ContinueStatement) \
  V(ReturnStatement) \
  V(GotoStatement) \
  V(AccessDeclaration) \
  V(MemInitializer) \
  V(FunctionDefinition) \
  V(TypeParameter) \
  V(TemplateTypeParameter) \
  V(ParameterDeclaration) \
  V(TemplateDeclaration) \
  V(LinkageSpecification) \
  V(NamespaceDefinition) \
  V(AsmDefinition) \
  V(NamespaceAliasDefinition) \
  V(UsingDeclaration) \
  V(UsingDirective) \
  V(OpaqueEnumDeclaration) \
  V(AliasDeclaration) \
  V(SimpleDeclaration) \
  V(StaticAssertDeclaration) \
  V(Declarator) \
  V(NestedDeclarator) \
  V(DeclaratorId) \
  V(PtrOperator) \
  V(ArrayDeclarator) \
  V(ParametersAndQualifiers) \
  V(FunctionDeclarator)

template <typename T> struct List;
struct AST;
struct DeclarationAST;
struct CoreDeclaratorAST;
struct PostfixDeclaratorAST;
struct SpecifierAST;
struct NameAST;
struct ExpressionAST;
struct StatementAST;

#define VISIT_AST(x) struct x##AST;
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST

#endif // ASTFWD_H
