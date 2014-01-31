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

#ifndef GLOBALS_H
#define GLOBALS_H

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

#define FOR_EACH_NAME(V) \
  V(Identifier) \
  V(DestructorName) \
  V(OperatorName) \
  V(QualifiedName) \
  V(TemplateName)


#define FOR_EACH_SYMBOL(V) \
  V(Namespace) \
  V(Class) \
  V(BaseClass) \
  V(Template) \
  V(Function) \
  V(Block) \
  V(Argument) \
  V(Declaration) \
  V(Typedef) \
  V(TypeParameter) \
  V(TemplateTypeParameter)

#define FOR_EACH_TOKEN(V) \
  V(EOF_SYMBOL, "<eof symbol>") \
  V(ERROR, "<error symbol>") \
  V(INT_LITERAL, "<int literal>") \
  V(CHAR_LITERAL, "<char literal>") \
  V(STRING_LITERAL, "<string literal>") \
  V(IDENTIFIER, "<identifier>") \
  V(AMP, "&") \
  V(AMP_AMP, "&&") \
  V(AMP_EQUAL, "&=") \
  V(BAR, "|") \
  V(BAR_BAR, "||") \
  V(BAR_EQUAL, "|=") \
  V(CARET, "^") \
  V(CARET_EQUAL, "^=") \
  V(COLON, ":") \
  V(COLON_COLON, "::") \
  V(COMMA, ",") \
  V(DOT, ".") \
  V(DOT_DOT_DOT, "...") \
  V(DOT_STAR, ".*") \
  V(EQUAL, "=") \
  V(EQUAL_EQUAL, "==") \
  V(EXCLAIM, "!") \
  V(EXCLAIM_EQUAL, "!=") \
  V(GREATER, ">") \
  V(GREATER_EQUAL, ">=") \
  V(GREATER_GREATER, ">>") \
  V(GREATER_GREATER_EQUAL, ">>=") \
  V(LBRACE, "{") \
  V(LBRACKET, "[") \
  V(LESS, "<") \
  V(LESS_EQUAL, "<=") \
  V(LESS_LESS, "<<") \
  V(LESS_LESS_EQUAL, "<<=") \
  V(LPAREN, "(") \
  V(MINUS, "-") \
  V(MINUS_EQUAL, "-=") \
  V(MINUS_GREATER, "->") \
  V(MINUS_GREATER_STAR, "->*") \
  V(MINUS_MINUS, "--") \
  V(PERCENT, "%") \
  V(PERCENT_EQUAL, "%=") \
  V(PLUS, "+") \
  V(PLUS_EQUAL, "+=") \
  V(PLUS_PLUS, "++") \
  V(POUND, "#") \
  V(POUND_POUND, "##") \
  V(QUESTION, "?") \
  V(RBRACE, "}") \
  V(RBRACKET, "]") \
  V(RPAREN, ")") \
  V(SEMICOLON, ";") \
  V(SLASH, "/") \
  V(SLASH_EQUAL, "/=") \
  V(STAR, "*") \
  V(STAR_EQUAL, "*=") \
  V(TILDE, "~") \
  V(TILDE_EQUAL, "~=") \
  V(ALIGNAS, "alignas") \
  V(ALIGNOF, "alignof") \
  V(ASM, "asm") \
  V(AUTO, "auto") \
  V(BOOL, "bool") \
  V(BREAK, "break") \
  V(CASE, "case") \
  V(CATCH, "catch") \
  V(CHAR, "char") \
  V(CHAR16_T, "char16_t") \
  V(CHAR32_T, "char32_t") \
  V(CLASS, "class") \
  V(CONST, "const") \
  V(CONST_CAST, "const_cast") \
  V(CONSTEXPR, "constexpr") \
  V(CONTINUE, "continue") \
  V(DECLTYPE, "decltype") \
  V(DEFAULT, "default") \
  V(DELETE, "delete") \
  V(DELETE_ARRAY, "delete[]") \
  V(DO, "do") \
  V(DOUBLE, "double") \
  V(DYNAMIC_CAST, "dynamic_cast") \
  V(ELSE, "else") \
  V(ENUM, "enum") \
  V(EXPLICIT, "explicit") \
  V(EXPORT, "export") \
  V(EXTERN, "extern") \
  V(FALSE, "false") \
  V(FLOAT, "float") \
  V(FOR, "for") \
  V(FRIEND, "friend") \
  V(GOTO, "goto") \
  V(IF, "if") \
  V(INLINE, "inline") \
  V(INT, "int") \
  V(LONG, "long") \
  V(MUTABLE, "mutable") \
  V(NAMESPACE, "namespace") \
  V(NEW, "new") \
  V(NEW_ARRAY, "new[]") \
  V(NOEXCEPT, "noexcept") \
  V(NULLPTR, "nullptr") \
  V(OPERATOR, "operator") \
  V(PRIVATE, "private") \
  V(PROTECTED, "protected") \
  V(PUBLIC, "public") \
  V(REGISTER, "register") \
  V(REINTERPRET_CAST, "reintepret_cast") \
  V(RETURN, "return") \
  V(SHORT, "short") \
  V(SIGNED, "signed") \
  V(SIZEOF, "sizeof") \
  V(STATIC, "static") \
  V(STATIC_ASSERT, "static_assert") \
  V(STATIC_CAST, "static_cast") \
  V(STRUCT, "struct") \
  V(SWITCH, "switch") \
  V(TEMPLATE, "template") \
  V(THIS, "this") \
  V(THREAD_LOCAL, "thread_local") \
  V(THROW, "throw") \
  V(TRUE, "true") \
  V(TRY, "try") \
  V(TYPEDEF, "typedef") \
  V(TYPEID, "typeid") \
  V(TYPENAME, "typename") \
  V(UNION, "union") \
  V(UNSIGNED, "unsigned") \
  V(USING, "using") \
  V(VIRTUAL, "virtual") \
  V(VOID, "void") \
  V(VOLATILE, "volatile") \
  V(WCHAR_T, "wchar_t") \
  V(WHILE, "while")

#define FOR_EACH_SINGLETON_TYPE(V) \
  V(Undefined) \
  V(Auto) \
  V(Void) \
  V(Nullptr)

#define FOR_EACH_OTHER_TYPE(V) \
  V(Integer) \
  V(Float) \
  V(Pointer) \
  V(LValueReference) \
  V(RValueReference) \
  V(BoundedArray) \
  V(UnboundedArray) \
  V(Function) \
  V(Class) \
  V(Named) \
  V(Elaborated)

#define FOR_EACH_TYPE(V) \
  FOR_EACH_SINGLETON_TYPE(V) \
  FOR_EACH_OTHER_TYPE(V)

#define FOR_EACH_INTEGER_TYPE(V) \
  V(SignedChar, "signed char") \
  V(ShortInt, "short int") \
  V(Int, "int") \
  V(LongInt, "long int") \
  V(LongLongInt, "long long int") \
  V(UnsignedChar, "unsigned char") \
  V(UnsignedShortInt, "unsigned short int") \
  V(UnsignedInt, "unsigned int") \
  V(UnsignedLongInt, "unsigned long int") \
  V(UnsignedLongLongInt, "unsigned long long int") \
  V(WCharT, "wchar_t") \
  V(Char, "char") \
  V(Char16T, "char16_t") \
  V(Char32T, "char32_t") \
  V(Bool, "bool")

#define FOR_EACH_FLOAT_TYPE(V) \
  V(Float, "float") \
  V(Double, "double") \
  V(LongDouble, "long double")


template <typename T> struct List;
struct AST;
struct DeclarationAST;
struct CoreDeclaratorAST;
struct PostfixDeclaratorAST;
struct SpecifierAST;
struct NameAST;
struct ExpressionAST;
struct StatementAST;

class ASTVisitor;
class RecursiveASTVisitor;

class Arena;
class Control;
class Name;
class QualType;
class Scope;
class Symbol;
class TranslationUnit;
class Token;
class Type;
class ArrayType;
class ReferenceType;

namespace IR {
struct Module;
struct Function;
struct BasicBlock;
struct Terminator;
struct Stmt;
struct Expr;
struct Temp;
} // end of namespace IR

//
// forward classes
//
#define VISIT_AST(x) struct x##AST;
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST

#define VISIT_NAME(T) class T;
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME

#define VISIT_SYMBOL(T) class T##Symbol;
  FOR_EACH_SYMBOL(VISIT_SYMBOL)
#undef VISIT_SYMBOL

#define VISIT_TYPE(T) class T##Type;
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

//
// ids
//
enum struct ASTKind {
#define VISIT_AST(x) k##x,
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
};

enum struct NameKind {
#define VISIT_NAME(T) k##T,
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME
};

enum struct SymbolKind {
#define VISIT_SYMBOL(T) k##T,
  FOR_EACH_SYMBOL(VISIT_SYMBOL)
#undef VISIT_SYMBOL
};

enum TokenKind {
#define TOKEN_ENUM(tk, _) T_##tk,
  FOR_EACH_TOKEN(TOKEN_ENUM)
};

enum struct TypeKind {
#define VISIT_TYPE(T) k##T,
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
};

enum struct IntegerKind {
#define VISIT_INTEGER_TYPE(T,N) k##T,
  FOR_EACH_INTEGER_TYPE(VISIT_INTEGER_TYPE)
#undef VISIT_INTEGER_TYPE
};

enum struct FloatKind {
#define VISIT_FLOAT_TYPE(T,N) k##T,
  FOR_EACH_FLOAT_TYPE(VISIT_FLOAT_TYPE)
#undef VISIT_FLOAT_TYPE
};

#endif // GLOBALS_H
