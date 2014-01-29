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

#ifndef AST_H
#define AST_H

#include "ASTVisitor.h"
#include "Types.h"
#include "Arena.h"

template <typename T>
struct List final: Managed {
  T value;
  List* next;
  explicit List(const T& value, List* next = nullptr)
    : value(value), next(next) {}
};

struct AST: Managed {
  AST(ASTKind kind): _kind(kind) {}
  inline ASTKind kind() const { return _kind; }
#define VISIT_AST(x) \
  inline bool is##x() const { return _kind == ASTKind::k##x; } \
  inline x##AST* as##x() { return is##x() ? reinterpret_cast<x##AST*>(this) : nullptr; }
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
private:
  ASTKind _kind;
};

template <ASTKind K, typename Base>
struct ExtendsAST: Base {
  ExtendsAST(): Base(K) {}
};

struct DeclarationAST: AST {
  using AST::AST;
};

struct CoreDeclaratorAST: AST {
  using AST::AST;
};

struct PostfixDeclaratorAST: AST {
  using AST::AST;
};

struct SpecifierAST: AST {
  using AST::AST;
};

struct NameAST: AST {
  using AST::AST;

private:
  const Name* _name{nullptr};
  friend class ParseContext;
};

struct ExpressionAST: AST {
  using AST::AST;
};

struct StatementAST: AST {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  using AST::AST;
};

struct TypeIdAST final: ExtendsAST<ASTKind::kTypeId, ExpressionAST> {
  List<SpecifierAST*>* specifier_list{nullptr};
  DeclaratorAST* declarator{nullptr};
};

//
// unit
//
struct TranslationUnitAST final: ExtendsAST<ASTKind::kTranslationUnit, AST> {
  List<DeclarationAST*>* declaration_list{nullptr};
// attributes
  NamespaceSymbol* globalScope{nullptr};
};

//
// specifiers
//
struct ExceptionSpecificationAST final: ExtendsAST<ASTKind::kExceptionSpecification, SpecifierAST> {
  // ### todo
};

struct AttributeAST final: ExtendsAST<ASTKind::kAttribute, AST> {
};

struct AttributeSpecifierAST final: ExtendsAST<ASTKind::kAttributeSpecifier, SpecifierAST> {
  List<AttributeAST*>* attribute_list{nullptr};
  unsigned lbracket_token{0};
  unsigned nested_lbracket_token{0};
  unsigned nested_rbracket_token{0};
  unsigned rbracket_token{0};
};

struct AlignasTypeAttributeSpecifierAST final: ExtendsAST<ASTKind::kAlignasTypeAttributeSpecifier, SpecifierAST> {
  TypeIdAST* type_id{nullptr};
  unsigned alignas_token{0};
  unsigned lparen_token{0};
  unsigned dot_dot_dot_token{0};
  unsigned rparen_token{0};
};

struct AlignasAttributeSpecifierAST final: ExtendsAST<ASTKind::kAlignasAttributeSpecifier, SpecifierAST> {
  ExpressionAST* expression{nullptr};
  unsigned alignas_token{0};
  unsigned lparen_token{0};
  unsigned dot_dot_dot_token{0};
  unsigned rparen_token{0};
};

struct SimpleSpecifierAST final: ExtendsAST<ASTKind::kSimpleSpecifier, SpecifierAST> {
  unsigned specifier_token{0};
};

struct NamedSpecifierAST final: ExtendsAST<ASTKind::kNamedSpecifier, SpecifierAST> {
  NameAST* name{nullptr};
};

struct TypenameSpecifierAST final: ExtendsAST<ASTKind::kTypenameSpecifier, SpecifierAST> {
  NameAST* name{nullptr};
};

struct ElaboratedTypeSpecifierAST final: ExtendsAST<ASTKind::kElaboratedTypeSpecifier, SpecifierAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  unsigned class_key_token{0};
};

struct EnumeratorAST final: ExtendsAST<ASTKind::kEnumerator, AST> {
  NameAST* name{nullptr};
  ExpressionAST* expression{nullptr};
  unsigned equal_token{0};
};

struct EnumSpecifierAST final: ExtendsAST<ASTKind::kEnumSpecifier, SpecifierAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  List<EnumeratorAST*>* enumerator_list{nullptr};
  unsigned enum_token{0};
  unsigned enum_key_token{0};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct BaseClassAST final: ExtendsAST<ASTKind::kBaseClass, AST> {
  List<SpecifierAST*>* specifier_list{nullptr};
  NameAST* name{nullptr};
};

struct ClassSpecifierAST final: ExtendsAST<ASTKind::kClassSpecifier, SpecifierAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  List<BaseClassAST*>* base_class_list{nullptr};
  List<DeclarationAST*>* declaration_list{nullptr};
  unsigned class_key_token{0};
  unsigned colon_token{0};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
// attributes
  ClassSymbol* symbol{nullptr};
};

//
// names
//
struct QualifiedNameAST final: ExtendsAST<ASTKind::kQualifiedName, NameAST> {
  NameAST* base{nullptr};
  NameAST* name{nullptr};
  unsigned scope_token{0};
};

struct PackedNameAST final: ExtendsAST<ASTKind::kPackedName, NameAST> {
  NameAST* name{nullptr};
  unsigned dot_dot_dot_token{0};
};

struct SimpleNameAST final: ExtendsAST<ASTKind::kSimpleName, NameAST> {
  unsigned identifier_token{0};
};

struct DestructorNameAST final: ExtendsAST<ASTKind::kDestructorName, NameAST> {
  NameAST* name{nullptr};
  unsigned tilde_token{0};
};

struct OperatorNameAST final: ExtendsAST<ASTKind::kOperatorName, NameAST> {
  unsigned operator_token{0};
  unsigned op_token{0};
};

struct TemplateArgumentAST final: ExtendsAST<ASTKind::kTemplateArgument, ExpressionAST> {
  TypeIdAST* type_id{nullptr};
};

struct TemplateIdAST final: ExtendsAST<ASTKind::kTemplateId, NameAST> {
  NameAST* name{nullptr};
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned template_token{0};
  unsigned less_token{0};
  unsigned greater_token{0};
};

struct DecltypeNameAST final: ExtendsAST<ASTKind::kDecltypeName, NameAST> {
  ExpressionAST* expression{nullptr};
  unsigned decltype_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct DecltypeAutoNameAST final: ExtendsAST<ASTKind::kDecltypeAutoName, NameAST> {
  unsigned decltype_token{0};
  unsigned lparen_token{0};
  unsigned auto_token{0};
  unsigned rparen_token{0};
};

//
// expressions
//
struct PackedExpressionAST final: ExtendsAST<ASTKind::kPackedExpression, ExpressionAST> {
  ExpressionAST* name{nullptr};
  unsigned dot_dot_dot_token{0};
};

struct LiteralExpressionAST final: ExtendsAST<ASTKind::kLiteralExpression, ExpressionAST> {
  unsigned literal_token{0};
};

struct ThisExpressionAST final: ExtendsAST<ASTKind::kThisExpression, ExpressionAST> {
  unsigned this_token{0};
};

struct IdExpressionAST final: ExtendsAST<ASTKind::kIdExpression, ExpressionAST> {
  NameAST* name{nullptr};
// attributes
  const Name* id{nullptr}; // ### rename
};

struct NestedExpressionAST final: ExtendsAST<ASTKind::kNestedExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct LambdaCaptureAST final: ExtendsAST<ASTKind::kLambdaCapture, AST> {
  // ### todo
};

struct LambdaDeclaratorAST final: ExtendsAST<ASTKind::kLambdaDeclarator, AST> { // ### ?
  ParametersAndQualifiersAST* parameters_and_qualifiers{nullptr};
  ExceptionSpecificationAST* exception_specification{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
};

struct LambdaExpressionAST final: ExtendsAST<ASTKind::kLambdaExpression, ExpressionAST> {
  LambdaDeclaratorAST* lambda_declarator{nullptr};
  StatementAST* statement{nullptr};
  unsigned lbracket_token{0};
  unsigned rbracket_token{0};
};

struct SubscriptExpressionAST final: ExtendsAST<ASTKind::kSubscriptExpression, ExpressionAST> {
  ExpressionAST* base_expression{nullptr};
  ExpressionAST* index_expression{nullptr};
  unsigned lbracket_token{0};
  unsigned rbracket_token{0};
};

struct CallExpressionAST final: ExtendsAST<ASTKind::kCallExpression, ExpressionAST> {
  ExpressionAST* base_expression{nullptr};
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct TypeCallExpressionAST final: ExtendsAST<ASTKind::kTypeCallExpression, ExpressionAST> {
  List<SpecifierAST*>* specifier_list{nullptr};
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct BracedTypeCallExpressionAST final: ExtendsAST<ASTKind::kBracedTypeCallExpression, ExpressionAST> {
  List<SpecifierAST*>* type_specifier_list{nullptr};
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct MemberExpressionAST final: ExtendsAST<ASTKind::kMemberExpression, ExpressionAST> {
  ExpressionAST* base_expression{nullptr};
  NameAST* name{nullptr};
  unsigned access_token{0};
  unsigned template_token{0};
// attributes
  const Name* id{nullptr};
};

struct IncrExpressionAST final: ExtendsAST<ASTKind::kIncrExpression, ExpressionAST> {
  ExpressionAST* base_expression{nullptr};
  unsigned incr_token{0};
};

struct CppCastExpressionAST final: ExtendsAST<ASTKind::kCppCastExpression, ExpressionAST> {
  TypeIdAST* type_id{nullptr};
  ExpressionAST* expression{nullptr};
  unsigned cast_token{0};
  unsigned less_token{0};
  unsigned greater_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
// attributes
  QualType targetTy;
};

struct TypeidExpressionAST final: ExtendsAST<ASTKind::kTypeidExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned typeid_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct UnaryExpressionAST final: ExtendsAST<ASTKind::kUnaryExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  TokenKind op{T_EOF_SYMBOL};
};

struct SizeofExpressionAST final: ExtendsAST<ASTKind::kSizeofExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned sizeof_token{0};
};

struct SizeofTypeExpressionAST final: ExtendsAST<ASTKind::kSizeofTypeExpression, ExpressionAST> {
  TypeIdAST* type_id{nullptr};
  unsigned sizeof_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct SizeofPackedArgsExpressionAST final: ExtendsAST<ASTKind::kSizeofPackedArgsExpression, ExpressionAST> {
  NameAST* name{nullptr};
  unsigned sizeof_token{0};
  unsigned dot_dot_dot_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct AlignofExpressionAST final: ExtendsAST<ASTKind::kAlignofExpression, ExpressionAST> {
  TypeIdAST* type_id{nullptr};
  unsigned alignof_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct NoexceptExpressionAST final: ExtendsAST<ASTKind::kNoexceptExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned noexcept_token{0};
};

struct NewExpressionAST final: ExtendsAST<ASTKind::kNewExpression, ExpressionAST> {
  List<ExpressionAST*>* placement_expression_list{nullptr};
  TypeIdAST* type_id{nullptr};
  ExpressionAST* initializer{nullptr};
  unsigned scope_token{0};
  unsigned new_token{0};
  unsigned placement_lparen_token{0};
  unsigned placement_rparen_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct DeleteExpressionAST final: ExtendsAST<ASTKind::kDeleteExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned scope_token{0};
  unsigned delete_token{0};
  unsigned lbracket_token{0};
  unsigned rbracket_token{0};
};

struct CastExpressionAST final: ExtendsAST<ASTKind::kCastExpression, ExpressionAST> {
  TypeIdAST* type_id{nullptr};
  ExpressionAST* expression{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
// attributes
  QualType targetTy;
};

struct BinaryExpressionAST final: ExtendsAST<ASTKind::kBinaryExpression, ExpressionAST> {
  ExpressionAST* left_expression{nullptr};
  ExpressionAST* right_expression{nullptr};
  unsigned op_token{0};
  TokenKind op{T_EOF_SYMBOL};
};

struct ConditionalExpressionAST final: ExtendsAST<ASTKind::kConditionalExpression, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  ExpressionAST* iftrue_expression{nullptr};
  ExpressionAST* iffalse_expression{nullptr};
  unsigned question_token{0};
  unsigned colon_token{0};
};

struct BracedInitializerAST final: ExtendsAST<ASTKind::kBracedInitializer, ExpressionAST> {
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct SimpleInitializerAST final: ExtendsAST<ASTKind::kSimpleInitializer, ExpressionAST> {
  ExpressionAST* expression{nullptr};
  unsigned equal_token{0};
};


struct ConditionAST final: ExtendsAST<ASTKind::kCondition, ExpressionAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  DeclaratorAST* declarator{nullptr};
  ExpressionAST* initializer{nullptr};
};


//
// statements
//
struct LabeledStatementAST final: ExtendsAST<ASTKind::kLabeledStatement, StatementAST> {
  NameAST* name{nullptr};
  StatementAST* statement{nullptr};
  unsigned colon_token{0};
// attributes
  const Name* id{nullptr};
};

struct CaseStatementAST final: ExtendsAST<ASTKind::kCaseStatement, StatementAST> {
  ExpressionAST* expression{nullptr};
  StatementAST* statement{nullptr};
  unsigned case_token{0};
  unsigned colon_token{0};
};

struct DefaultStatementAST final: ExtendsAST<ASTKind::kDefaultStatement, StatementAST> {
  StatementAST* statement{nullptr};
  unsigned default_token{0};
  unsigned colon_token{0};
};

struct ExpressionStatementAST final: ExtendsAST<ASTKind::kExpressionStatement, StatementAST> {
  ExpressionAST* expression{nullptr};
  unsigned semicolon_token{0};
};

struct CompoundStatementAST final: ExtendsAST<ASTKind::kCompoundStatement, StatementAST> {
  List<StatementAST*>* statement_list{nullptr};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct TryBlockStatementAST final: ExtendsAST<ASTKind::kTryBlockStatement, StatementAST> {
  StatementAST* statemebt{nullptr};
  unsigned try_token{0};
  unsigned catch_token{0};
  unsigned lparen_token{0};
  // ### catch clauses
  unsigned rparen_token{0};
};

struct DeclarationStatementAST final: ExtendsAST<ASTKind::kDeclarationStatement, StatementAST> {
  DeclarationAST* declaration{nullptr};
};

struct IfStatementAST final: ExtendsAST<ASTKind::kIfStatement, StatementAST> {
  ExpressionAST* condition{nullptr};
  StatementAST* statement{nullptr};
  StatementAST* else_statement{nullptr};
  unsigned if_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
  unsigned else_token{0};
};

struct SwitchStatementAST final: ExtendsAST<ASTKind::kSwitchStatement, StatementAST> {
  ExpressionAST* condition{nullptr};
  StatementAST* statement{nullptr};
  unsigned switch_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct WhileStatementAST final: ExtendsAST<ASTKind::kWhileStatement, StatementAST> {
  ExpressionAST* condition{nullptr};
  StatementAST* statement{nullptr};
  unsigned while_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct DoStatementAST final: ExtendsAST<ASTKind::kDoStatement, StatementAST> {
  StatementAST* statement{nullptr};
  ExpressionAST* expression{nullptr};
  unsigned do_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
  unsigned semicolon_token{0};
};

struct ForStatementAST final: ExtendsAST<ASTKind::kForStatement, StatementAST> {
  StatementAST* initializer{nullptr};
  ExpressionAST* condition{nullptr};
  ExpressionAST* expression{nullptr};
  StatementAST* statement{nullptr};
  unsigned for_token{0};
  unsigned lparen_token{0};
  unsigned semicolon_token{0};
  unsigned rparen_token{0};
};

struct ForRangeStatementAST final: ExtendsAST<ASTKind::kForRangeStatement, StatementAST> {
  DeclarationAST* initializer{nullptr};
  ExpressionAST* expression{nullptr};
  StatementAST* statement{nullptr};
  unsigned for_token{0};
  unsigned lparen_token{0};
  unsigned colon_token{0};
  unsigned rparen_token{0};
};

struct BreakStatementAST final: ExtendsAST<ASTKind::kBreakStatement, StatementAST> {
  unsigned break_token{0};
  unsigned semicolon_token{0};
};

struct ContinueStatementAST final: ExtendsAST<ASTKind::kContinueStatement, StatementAST> {
  unsigned continue_token{0};
  unsigned semicolon_token{0};
};

struct ReturnStatementAST final: ExtendsAST<ASTKind::kReturnStatement, StatementAST> {
  ExpressionAST* expression{nullptr};
  unsigned return_token{0};
  unsigned semicolon_token{0};
};

struct GotoStatementAST final: ExtendsAST<ASTKind::kGotoStatement, StatementAST> {
  NameAST* name{nullptr};
  unsigned goto_token{0};
  unsigned semicolon_token{0};
// attributes
  const Name* id{nullptr};
};

//
// declarations
//
struct AccessDeclarationAST final: ExtendsAST<ASTKind::kAccessDeclaration, DeclarationAST> {
  unsigned access_token{0};
  unsigned colon_token{0};
};

struct MemInitializerAST final: ExtendsAST<ASTKind::kMemInitializer, AST> {
  NameAST* name{nullptr};
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct FunctionDefinitionAST final: ExtendsAST<ASTKind::kFunctionDefinition, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  DeclaratorAST* declarator{nullptr};
  List<MemInitializerAST*>* mem_initializer_list{nullptr};
  StatementAST* statement{nullptr};
  unsigned colon_token{0};
  unsigned equal_token{0};
  unsigned special_token{0};
  unsigned semicolon_token{0};
// attributes
  FunctionSymbol* symbol{0};
};

struct TypeParameterAST final: ExtendsAST<ASTKind::kTypeParameter, DeclarationAST> {
  NameAST* name{nullptr};
  TypeIdAST* type_id{nullptr};
  unsigned typename_token{0};
  unsigned dot_dot_dot_token{0};
  unsigned equal_token{0};
};

struct TemplateTypeParameterAST final: ExtendsAST<ASTKind::kTemplateTypeParameter, DeclarationAST> {
  List<DeclarationAST*>* template_parameter_list{nullptr};
  NameAST* name{nullptr};
  TypeIdAST* type_id{nullptr};
  unsigned template_token{0};
  unsigned less_token{0};
  unsigned greater_token{0};
  unsigned class_token{0};
  unsigned dot_dot_dot_token{0};
  unsigned equal_token{0};
};

struct ParameterDeclarationAST final: ExtendsAST<ASTKind::kParameterDeclaration, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  DeclaratorAST* declarator{nullptr};
  ExpressionAST* expression{nullptr};
};

struct TemplateDeclarationAST final: ExtendsAST<ASTKind::kTemplateDeclaration, DeclarationAST> {
  List<DeclarationAST*>* template_parameter_list{nullptr};
  DeclarationAST* declaration{nullptr};
  unsigned extern_token{0};
  unsigned template_token{0};
  unsigned less_token{0};
  unsigned greater_token{0};
};

struct LinkageSpecificationAST final: ExtendsAST<ASTKind::kLinkageSpecification, DeclarationAST> {
  List<DeclarationAST*>* declaration_list{nullptr};
  unsigned extern_token{0};
  unsigned string_literal_token{0};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct NamespaceDefinitionAST final: ExtendsAST<ASTKind::kNamespaceDefinition, DeclarationAST> {
  NameAST* name{nullptr};
  List<DeclarationAST*>* declaration_list{nullptr};
  unsigned inline_token{0};
  unsigned namespace_token{0};
  unsigned lbrace_token{0};
  unsigned rbrace_token{0};
};

struct AsmDefinitionAST final: ExtendsAST<ASTKind::kAsmDefinition, DeclarationAST> {
  List<ExpressionAST*>* expression_list{nullptr};
  unsigned asm_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
  unsigned semicolon_token{0};
};

struct NamespaceAliasDefinitionAST final: ExtendsAST<ASTKind::kNamespaceAliasDefinition, DeclarationAST> {
  NameAST* alias_name{nullptr};
  NameAST* name{nullptr};
  unsigned namespace_token{0};
  unsigned equal_token{0};
  unsigned semicolon_token{0};
};

struct UsingDeclarationAST final: ExtendsAST<ASTKind::kUsingDeclaration, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  unsigned using_token{0};
  unsigned typename_token{0};
  unsigned semicolon_token{0};
};

struct UsingDirectiveAST final: ExtendsAST<ASTKind::kUsingDirective, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  unsigned using_token{0};
  unsigned namespace_token{0};
  unsigned semicolon_token{0};
};

struct OpaqueEnumDeclarationAST final: ExtendsAST<ASTKind::kOpaqueEnumDeclaration, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  NameAST* name{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  unsigned enum_token{0};
  unsigned enum_key_token{0};
};

struct AliasDeclarationAST final: ExtendsAST<ASTKind::kAliasDeclaration, DeclarationAST> {
  NameAST* alias_name{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  TypeIdAST* type_id{nullptr};
  unsigned using_token{0};
  unsigned equal_token{0};
  unsigned semicolon_token{0};
};

struct SimpleDeclarationAST final: ExtendsAST<ASTKind::kSimpleDeclaration, DeclarationAST> {
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  List<DeclaratorAST*>* declarator_list{nullptr};
  unsigned semicolon_token{0};
};

struct StaticAssertDeclarationAST final: ExtendsAST<ASTKind::kStaticAssertDeclaration, DeclarationAST> {
  ExpressionAST* expression{nullptr};
  unsigned static_assert_token{0};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
  unsigned semicolon_token{0};
};

//
// declarators
//
struct DeclaratorAST final: ExtendsAST<ASTKind::kDeclarator, AST> {
  List<PtrOperatorAST*>* ptr_op_list{nullptr};
  CoreDeclaratorAST* core_declarator{nullptr};
  List<PostfixDeclaratorAST*>* postfix_declarator_list{nullptr};
  ExpressionAST* initializer{nullptr};

private:
  QualType _type;
  const Name* _name{nullptr};
  friend class ParseContext;
};

struct NestedDeclaratorAST final: ExtendsAST<ASTKind::kNestedDeclarator, CoreDeclaratorAST> {
  DeclaratorAST* declarator{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct DeclaratorIdAST final: ExtendsAST<ASTKind::kDeclaratorId, CoreDeclaratorAST> {
  NameAST* name{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
};

struct PtrOperatorAST final: ExtendsAST<ASTKind::kPtrOperator, AST> {
  List<NameAST*>* nested_name_specifier{nullptr};
  List<SpecifierAST*>* cv_qualifier_list{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  TokenKind op{T_EOF_SYMBOL};
};

struct ArrayDeclaratorAST final: ExtendsAST<ASTKind::kArrayDeclarator, PostfixDeclaratorAST> {
  ExpressionAST* size_expression{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  unsigned lbracket_token{0};
  unsigned rbracket_token{0};
};

struct ParametersAndQualifiersAST final: ExtendsAST<ASTKind::kParametersAndQualifiers, AST> {
  List<DeclarationAST*>* parameter_list{nullptr};
  List<SpecifierAST*>* specifier_list{nullptr};
  SpecifierAST* ref_qualifier{nullptr};
  ExceptionSpecificationAST* exception_specification{nullptr};
  List<SpecifierAST*>* attribute_specifier_list{nullptr};
  unsigned lparen_token{0};
  unsigned rparen_token{0};
};

struct FunctionDeclaratorAST final: ExtendsAST<ASTKind::kFunctionDeclarator, PostfixDeclaratorAST> {
  ParametersAndQualifiersAST* parameters_and_qualifiers{nullptr};
  List<SpecifierAST*>* trailing_type_specifier_list{nullptr};
  unsigned arrow_token{0};
};

#endif // AST_H
