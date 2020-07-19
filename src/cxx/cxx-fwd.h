// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cstdint>

#include "ir-fwd.h"

#define FOR_EACH_AST(V)            \
  V(TypeId)                        \
  V(TranslationUnit)               \
  V(ExceptionSpecification)        \
  V(Attribute)                     \
  V(AttributeSpecifier)            \
  V(AlignasTypeAttributeSpecifier) \
  V(AlignasAttributeSpecifier)     \
  V(SimpleSpecifier)               \
  V(NamedSpecifier)                \
  V(TypenameSpecifier)             \
  V(ElaboratedTypeSpecifier)       \
  V(Enumerator)                    \
  V(EnumSpecifier)                 \
  V(BaseClass)                     \
  V(ClassSpecifier)                \
  V(QualifiedName)                 \
  V(PackedName)                    \
  V(SimpleName)                    \
  V(DestructorName)                \
  V(OperatorName)                  \
  V(ConversionFunctionId)          \
  V(TemplateArgument)              \
  V(TemplateId)                    \
  V(DecltypeName)                  \
  V(DecltypeAutoName)              \
  V(PackedExpression)              \
  V(LiteralExpression)             \
  V(ThisExpression)                \
  V(IdExpression)                  \
  V(NestedExpression)              \
  V(LambdaCapture)                 \
  V(LambdaDeclarator)              \
  V(LambdaExpression)              \
  V(SubscriptExpression)           \
  V(CallExpression)                \
  V(TypeCallExpression)            \
  V(BracedTypeCallExpression)      \
  V(MemberExpression)              \
  V(IncrExpression)                \
  V(CppCastExpression)             \
  V(TypeidExpression)              \
  V(UnaryExpression)               \
  V(SizeofExpression)              \
  V(SizeofTypeExpression)          \
  V(SizeofPackedArgsExpression)    \
  V(AlignofExpression)             \
  V(NoexceptExpression)            \
  V(NewExpression)                 \
  V(DeleteExpression)              \
  V(CastExpression)                \
  V(BinaryExpression)              \
  V(ConditionalExpression)         \
  V(BracedInitializer)             \
  V(SimpleInitializer)             \
  V(Condition)                     \
  V(LabeledStatement)              \
  V(CaseStatement)                 \
  V(DefaultStatement)              \
  V(ExpressionStatement)           \
  V(CompoundStatement)             \
  V(TryBlockStatement)             \
  V(DeclarationStatement)          \
  V(IfStatement)                   \
  V(SwitchStatement)               \
  V(WhileStatement)                \
  V(DoStatement)                   \
  V(ForStatement)                  \
  V(ForRangeStatement)             \
  V(BreakStatement)                \
  V(ContinueStatement)             \
  V(ReturnStatement)               \
  V(GotoStatement)                 \
  V(AccessDeclaration)             \
  V(MemInitializer)                \
  V(FunctionDefinition)            \
  V(TypeParameter)                 \
  V(TemplateTypeParameter)         \
  V(ParameterDeclaration)          \
  V(TemplateDeclaration)           \
  V(LinkageSpecification)          \
  V(NamespaceDefinition)           \
  V(AsmDefinition)                 \
  V(NamespaceAliasDefinition)      \
  V(UsingDeclaration)              \
  V(UsingDirective)                \
  V(OpaqueEnumDeclaration)         \
  V(AliasDeclaration)              \
  V(SimpleDeclaration)             \
  V(StaticAssertDeclaration)       \
  V(Declarator)                    \
  V(NestedDeclarator)              \
  V(DeclaratorId)                  \
  V(PtrOperator)                   \
  V(ArrayDeclarator)               \
  V(ParametersAndQualifiers)       \
  V(FunctionDeclarator)

#define FOR_EACH_NAME(V) \
  V(Identifier)          \
  V(DestructorName)      \
  V(OperatorName)        \
  V(ConversionName)      \
  V(QualifiedName)       \
  V(TemplateName)        \
  V(DecltypeName)

#define FOR_EACH_SYMBOL(V) \
  V(Namespace)             \
  V(Enum)                  \
  V(Class)                 \
  V(BaseClass)             \
  V(Template)              \
  V(Function)              \
  V(Block)                 \
  V(Argument)              \
  V(Declaration)           \
  V(Typedef)               \
  V(TypeParameter)         \
  V(TemplateTypeParameter)

#define FOR_EACH_TOKEN(V)                                             \
  V(EOF_SYMBOL, "eof")                                                \
  V(ERROR, "error")                                                   \
  V(IDENTIFIER, "identifier")                                         \
  V(CHARACTER_LITERAL, "character_literal")                           \
  V(FLOATING_POINT_LITERAL, "floating_point_literal")                 \
  V(INTEGER_LITERAL, "integer_literal")                               \
  V(STRING_LITERAL, "string_literal")                                 \
  V(USER_DEFINED_LITERAL, "user_defined_literal")                     \
  V(USER_DEFINED_STRING_LITERAL, "user_defined_string_literal")       \
  V(EXCLAIM, "!")                                                     \
  V(EXCLAIM_EQUAL, "!=")                                              \
  V(PERCENT, "%")                                                     \
  V(PERCENT_EQUAL, "%=")                                              \
  V(AMP, "&")                                                         \
  V(AMP_AMP, "&&")                                                    \
  V(AMP_EQUAL, "&=")                                                  \
  V(LPAREN, "(")                                                      \
  V(RPAREN, ")")                                                      \
  V(STAR, "*")                                                        \
  V(STAR_EQUAL, "*=")                                                 \
  V(PLUS, "+")                                                        \
  V(PLUS_PLUS, "++")                                                  \
  V(PLUS_EQUAL, "+=")                                                 \
  V(COMMA, ",")                                                       \
  V(MINUS, "-")                                                       \
  V(MINUS_MINUS, "--")                                                \
  V(MINUS_EQUAL, "-=")                                                \
  V(MINUS_GREATER, "->")                                              \
  V(MINUS_GREATER_STAR, "->*")                                        \
  V(DOT, ".")                                                         \
  V(DOT_STAR, ".*")                                                   \
  V(DOT_DOT_DOT, "...")                                               \
  V(SLASH, "/")                                                       \
  V(SLASH_EQUAL, "/=")                                                \
  V(COLON, ":")                                                       \
  V(COLON_COLON, "::")                                                \
  V(SEMICOLON, ";")                                                   \
  V(LESS, "<")                                                        \
  V(LESS_LESS, "<<")                                                  \
  V(LESS_LESS_EQUAL, "<<=")                                           \
  V(LESS_EQUAL, "<=")                                                 \
  V(LESS_EQUAL_GREATER, "<=>")                                        \
  V(EQUAL, "=")                                                       \
  V(EQUAL_EQUAL, "==")                                                \
  V(GREATER, ">")                                                     \
  V(GREATER_EQUAL, ">=")                                              \
  V(GREATER_GREATER, ">>")                                            \
  V(GREATER_GREATER_EQUAL, ">>=")                                     \
  V(QUESTION, "?")                                                    \
  V(LBRACKET, "[")                                                    \
  V(RBRACKET, "]")                                                    \
  V(CARET, "^")                                                       \
  V(CARET_EQUAL, "^=")                                                \
  V(LBRACE, "{")                                                      \
  V(BAR, "|")                                                         \
  V(BAR_EQUAL, "|=")                                                  \
  V(BAR_BAR, "||")                                                    \
  V(RBRACE, "}")                                                      \
  V(TILDE, "~")                                                       \
  V(NEW_ARRAY, "new[]")                                               \
  V(DELETE_ARRAY, "delete[]")                                         \
  V(__INT64, "__int64")                                               \
  V(__INT128, "__int128")                                             \
  V(__FLOAT80, "__float80")                                           \
  V(__FLOAT128, "__float128")                                         \
  V(__ALIGNOF, "__alignof")                                           \
  V(__ATTRIBUTE__, "__attribute__")                                   \
  V(__ATTRIBUTE, "__attribute")                                       \
  V(__EXTENSION__, "__extension__")                                   \
  V(__HAS_UNIQUE_OBJECT_REPRESENTATIONS,                              \
    "__has_unique_object_representations")                            \
  V(__HAS_VIRTUAL_DESTRUCTOR, "__has_virtual_destructor")             \
  V(__IS_ABSTRACT, "__is_abstract")                                   \
  V(__IS_AGGREGATE, "__is_aggregate")                                 \
  V(__IS_BASE_OF, "__is_base_of")                                     \
  V(__IS_CLASS, "__is_class")                                         \
  V(__IS_CONSTRUCTIBLE, "__is_constructible")                         \
  V(__IS_CONVERTIBLE_TO, "__is_convertible_to")                       \
  V(__IS_EMPTY, "__is_empty")                                         \
  V(__IS_ENUM, "__is_enum")                                           \
  V(__IS_FINAL, "__is_final")                                         \
  V(__IS_FUNCTION, "__is_function")                                   \
  V(__IS_LITERAL, "__is_literal")                                     \
  V(__IS_NOTHROW_ASSIGNABLE, "__is_nothrow_assignable")               \
  V(__IS_NOTHROW_CONSTRUCTIBLE, "__is_nothrow_constructible")         \
  V(__IS_POD, "__is_pod")                                             \
  V(__IS_POLYMORPHIC, "__is_polymorphic")                             \
  V(__IS_SAME, "__is_same")                                           \
  V(__IS_STANDARD_LAYOUT, "__is_standard_layout")                     \
  V(__IS_TRIVIAL, "__is_trivial")                                     \
  V(__IS_TRIVIALLY_ASSIGNABLE, "__is_trivially_assignable")           \
  V(__IS_TRIVIALLY_CONSTRUCTIBLE, "__is_trivially_constructible")     \
  V(__IS_TRIVIALLY_COPYABLE, "__is_trivially_copyable")               \
  V(__IS_TRIVIALLY_DESTRUCTIBLE, "__is_trivially_destructible")       \
  V(__IS_UNION, "__is_union")                                         \
  V(__REFERENCE_BINDS_TO_TEMPORARY, "__reference_binds_to_temporary") \
  V(__RESTRICT, "__restrict")                                         \
  V(__UNDERLYING_TYPE, "__underlying_type")                           \
  V(_ATOMIC, "_Atomic")                                               \
  V(ALIGNAS, "alignas")                                               \
  V(ALIGNOF, "alignof")                                               \
  V(ASM, "asm")                                                       \
  V(AUTO, "auto")                                                     \
  V(BOOL, "bool")                                                     \
  V(BREAK, "break")                                                   \
  V(CASE, "case")                                                     \
  V(CATCH, "catch")                                                   \
  V(CHAR, "char")                                                     \
  V(CHAR16_T, "char16_t")                                             \
  V(CHAR32_T, "char32_t")                                             \
  V(CHAR8_T, "char8_t")                                               \
  V(CLASS, "class")                                                   \
  V(CO_AWAIT, "co_await")                                             \
  V(CO_RETURN, "co_return")                                           \
  V(CO_YIELD, "co_yield")                                             \
  V(CONCEPT, "concept")                                               \
  V(CONST, "const")                                                   \
  V(CONST_CAST, "const_cast")                                         \
  V(CONSTEVAL, "consteval")                                           \
  V(CONSTEXPR, "constexpr")                                           \
  V(CONSTINIT, "constinit")                                           \
  V(CONTINUE, "continue")                                             \
  V(DECLTYPE, "decltype")                                             \
  V(DEFAULT, "default")                                               \
  V(DELETE, "delete")                                                 \
  V(DO, "do")                                                         \
  V(DOUBLE, "double")                                                 \
  V(DYNAMIC_CAST, "dynamic_cast")                                     \
  V(ELSE, "else")                                                     \
  V(ENUM, "enum")                                                     \
  V(EXPLICIT, "explicit")                                             \
  V(EXPORT, "export")                                                 \
  V(EXTERN, "extern")                                                 \
  V(FALSE, "false")                                                   \
  V(FLOAT, "float")                                                   \
  V(FOR, "for")                                                       \
  V(FRIEND, "friend")                                                 \
  V(GOTO, "goto")                                                     \
  V(IF, "if")                                                         \
  V(INLINE, "inline")                                                 \
  V(INT, "int")                                                       \
  V(LONG, "long")                                                     \
  V(MUTABLE, "mutable")                                               \
  V(NAMESPACE, "namespace")                                           \
  V(NEW, "new")                                                       \
  V(NOEXCEPT, "noexcept")                                             \
  V(NULLPTR, "nullptr")                                               \
  V(OPERATOR, "operator")                                             \
  V(PRIVATE, "private")                                               \
  V(PROTECTED, "protected")                                           \
  V(PUBLIC, "public")                                                 \
  V(REINTERPRET_CAST, "reinterpret_cast")                             \
  V(REQUIRES, "requires")                                             \
  V(RETURN, "return")                                                 \
  V(SHORT, "short")                                                   \
  V(SIGNED, "signed")                                                 \
  V(SIZEOF, "sizeof")                                                 \
  V(STATIC, "static")                                                 \
  V(STATIC_ASSERT, "static_assert")                                   \
  V(STATIC_CAST, "static_cast")                                       \
  V(STRUCT, "struct")                                                 \
  V(SWITCH, "switch")                                                 \
  V(TEMPLATE, "template")                                             \
  V(THIS, "this")                                                     \
  V(THREAD_LOCAL, "thread_local")                                     \
  V(THROW, "throw")                                                   \
  V(TRUE, "true")                                                     \
  V(TRY, "try")                                                       \
  V(TYPEDEF, "typedef")                                               \
  V(TYPEID, "typeid")                                                 \
  V(TYPENAME, "typename")                                             \
  V(UNION, "union")                                                   \
  V(UNSIGNED, "unsigned")                                             \
  V(USING, "using")                                                   \
  V(VIRTUAL, "virtual")                                               \
  V(VOID, "void")                                                     \
  V(VOLATILE, "volatile")                                             \
  V(WCHAR_T, "wchar_t")                                               \
  V(WHILE, "while")

#define FOR_EACH_SINGLETON_TYPE(V) \
  V(Undefined)                     \
  V(Auto)                          \
  V(Void)                          \
  V(Nullptr)

#define FOR_EACH_OTHER_TYPE(V) \
  V(Integer)                   \
  V(Float)                     \
  V(Pointer)                   \
  V(LValueReference)           \
  V(RValueReference)           \
  V(Array)                     \
  V(Function)                  \
  V(OverloadSet)               \
  V(Class)                     \
  V(Enum)                      \
  V(Named)                     \
  V(Elaborated)

#define FOR_EACH_TYPE(V)     \
  FOR_EACH_SINGLETON_TYPE(V) \
  FOR_EACH_OTHER_TYPE(V)

#define FOR_EACH_INTEGER_TYPE(V)                   \
  V(SignedChar, "signed char")                     \
  V(ShortInt, "short int")                         \
  V(Int, "int")                                    \
  V(LongInt, "long int")                           \
  V(LongLongInt, "long long int")                  \
  V(UnsignedChar, "unsigned char")                 \
  V(UnsignedShortInt, "unsigned short int")        \
  V(UnsignedInt, "unsigned int")                   \
  V(UnsignedLongInt, "unsigned long int")          \
  V(UnsignedLongLongInt, "unsigned long long int") \
  V(WCharT, "wchar_t")                             \
  V(Char, "char")                                  \
  V(Char16T, "char16_t")                           \
  V(Char32T, "char32_t")                           \
  V(Bool, "bool")                                  \
  V(Int128, "__int128")                            \
  V(UnsignedInt128, "unsigned __int128")

#define FOR_EACH_FLOAT_TYPE(V) \
  V(Float, "float")            \
  V(Double, "double")          \
  V(LongDouble, "long double") \
  V(Float128, "__float128")

namespace cxx {

template <typename T>
struct List;
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
class ReferenceType;

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

enum struct TokenKind : uint16_t {
#define TOKEN_ENUM(tk, _) T_##tk,
  FOR_EACH_TOKEN(TOKEN_ENUM)
};

enum struct TypeKind {
#define VISIT_TYPE(T) k##T,
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
};

enum struct IntegerKind {
#define VISIT_INTEGER_TYPE(T, N) k##T,
  FOR_EACH_INTEGER_TYPE(VISIT_INTEGER_TYPE)
#undef VISIT_INTEGER_TYPE
};

enum struct FloatKind {
#define VISIT_FLOAT_TYPE(T, N) k##T,
  FOR_EACH_FLOAT_TYPE(VISIT_FLOAT_TYPE)
#undef VISIT_FLOAT_TYPE
};

}  // namespace cxx
