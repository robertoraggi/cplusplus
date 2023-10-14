// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <optional>
#include <variant>

namespace cxx::syntax {

using Unit = std::variant<TranslationUnitAST*, ModuleUnitAST*>;

using Declaration = std::variant<
    SimpleDeclarationAST*, AsmDeclarationAST*, NamespaceAliasDefinitionAST*,
    UsingDeclarationAST*, UsingEnumDeclarationAST*, UsingDirectiveAST*,
    StaticAssertDeclarationAST*, AliasDeclarationAST*,
    OpaqueEnumDeclarationAST*, FunctionDefinitionAST*, TemplateDeclarationAST*,
    ConceptDefinitionAST*, DeductionGuideAST*, ExplicitInstantiationAST*,
    ExportDeclarationAST*, ExportCompoundDeclarationAST*,
    LinkageSpecificationAST*, NamespaceDefinitionAST*, EmptyDeclarationAST*,
    AttributeDeclarationAST*, ModuleImportDeclarationAST*,
    ParameterDeclarationAST*, AccessDeclarationAST*, ForRangeDeclarationAST*,
    StructuredBindingDeclarationAST*, AsmOperandAST*, AsmQualifierAST*,
    AsmClobberAST*, AsmGotoLabelAST*>;

using Statement =
    std::variant<LabeledStatementAST*, CaseStatementAST*, DefaultStatementAST*,
                 ExpressionStatementAST*, CompoundStatementAST*,
                 IfStatementAST*, ConstevalIfStatementAST*, SwitchStatementAST*,
                 WhileStatementAST*, DoStatementAST*, ForRangeStatementAST*,
                 ForStatementAST*, BreakStatementAST*, ContinueStatementAST*,
                 ReturnStatementAST*, CoroutineReturnStatementAST*,
                 GotoStatementAST*, DeclarationStatementAST*,
                 TryBlockStatementAST*>;

using Expression = std::variant<
    CharLiteralExpressionAST*, BoolLiteralExpressionAST*,
    IntLiteralExpressionAST*, FloatLiteralExpressionAST*,
    NullptrLiteralExpressionAST*, StringLiteralExpressionAST*,
    UserDefinedStringLiteralExpressionAST*, ThisExpressionAST*,
    NestedExpressionAST*, IdExpressionAST*, LambdaExpressionAST*,
    FoldExpressionAST*, RightFoldExpressionAST*, LeftFoldExpressionAST*,
    RequiresExpressionAST*, SubscriptExpressionAST*, CallExpressionAST*,
    TypeConstructionAST*, BracedTypeConstructionAST*, MemberExpressionAST*,
    PostIncrExpressionAST*, CppCastExpressionAST*, TypeidExpressionAST*,
    TypeidOfTypeExpressionAST*, UnaryExpressionAST*, AwaitExpressionAST*,
    SizeofExpressionAST*, SizeofTypeExpressionAST*, SizeofPackExpressionAST*,
    AlignofTypeExpressionAST*, AlignofExpressionAST*, NoexceptExpressionAST*,
    NewExpressionAST*, DeleteExpressionAST*, CastExpressionAST*,
    ImplicitCastExpressionAST*, BinaryExpressionAST*, ConditionalExpressionAST*,
    YieldExpressionAST*, ThrowExpressionAST*, AssignmentExpressionAST*,
    PackExpansionExpressionAST*, DesignatedInitializerClauseAST*,
    TypeTraitsExpressionAST*, ConditionExpressionAST*, EqualInitializerAST*,
    BracedInitListAST*, ParenInitializerAST*>;

using TemplateParameter =
    std::variant<TemplateTypeParameterAST*, TemplatePackTypeParameterAST*,
                 NonTypeTemplateParameterAST*, TypenameTypeParameterAST*,
                 ConstraintTypeParameterAST*>;

using Specifier = std::variant<
    TypedefSpecifierAST*, FriendSpecifierAST*, ConstevalSpecifierAST*,
    ConstinitSpecifierAST*, ConstexprSpecifierAST*, InlineSpecifierAST*,
    StaticSpecifierAST*, ExternSpecifierAST*, ThreadLocalSpecifierAST*,
    ThreadSpecifierAST*, MutableSpecifierAST*, VirtualSpecifierAST*,
    ExplicitSpecifierAST*, AutoTypeSpecifierAST*, VoidTypeSpecifierAST*,
    SizeTypeSpecifierAST*, SignTypeSpecifierAST*, VaListTypeSpecifierAST*,
    IntegralTypeSpecifierAST*, FloatingPointTypeSpecifierAST*,
    ComplexTypeSpecifierAST*, NamedTypeSpecifierAST*, AtomicTypeSpecifierAST*,
    UnderlyingTypeSpecifierAST*, ElaboratedTypeSpecifierAST*,
    DecltypeAutoSpecifierAST*, DecltypeSpecifierAST*,
    PlaceholderTypeSpecifierAST*, ConstQualifierAST*, VolatileQualifierAST*,
    RestrictQualifierAST*, EnumSpecifierAST*, ClassSpecifierAST*,
    TypenameSpecifierAST*>;

using PtrOperator = std::variant<PointerOperatorAST*, ReferenceOperatorAST*,
                                 PtrToMemberOperatorAST*>;

using CoreDeclarator = std::variant<BitfieldDeclaratorAST*, ParameterPackAST*,
                                    IdDeclaratorAST*, NestedDeclaratorAST*>;

using DeclaratorChunk =
    std::variant<FunctionDeclaratorChunkAST*, ArrayDeclaratorChunkAST*>;

using UnqualifiedId =
    std::variant<NameIdAST*, DestructorIdAST*, DecltypeIdAST*,
                 OperatorFunctionIdAST*, LiteralOperatorIdAST*,
                 ConversionFunctionIdAST*, SimpleTemplateIdAST*,
                 LiteralOperatorTemplateIdAST*, OperatorFunctionTemplateIdAST*>;

using NestedNameSpecifier =
    std::variant<GlobalNestedNameSpecifierAST*, SimpleNestedNameSpecifierAST*,
                 DecltypeNestedNameSpecifierAST*,
                 TemplateNestedNameSpecifierAST*>;

using FunctionBody =
    std::variant<DefaultFunctionBodyAST*, CompoundStatementFunctionBodyAST*,
                 TryStatementFunctionBodyAST*, DeleteFunctionBodyAST*>;

using TemplateArgument =
    std::variant<TypeTemplateArgumentAST*, ExpressionTemplateArgumentAST*>;

using ExceptionSpecifier =
    std::variant<ThrowExceptionSpecifierAST*, NoexceptSpecifierAST*>;

using Requirement = std::variant<SimpleRequirementAST*, CompoundRequirementAST*,
                                 TypeRequirementAST*, NestedRequirementAST*>;

using NewInitializer =
    std::variant<NewParenInitializerAST*, NewBracedInitializerAST*>;

using MemInitializer =
    std::variant<ParenMemInitializerAST*, BracedMemInitializerAST*>;

using LambdaCapture =
    std::variant<ThisLambdaCaptureAST*, DerefThisLambdaCaptureAST*,
                 SimpleLambdaCaptureAST*, RefLambdaCaptureAST*,
                 RefInitLambdaCaptureAST*, InitLambdaCaptureAST*>;

using ExceptionDeclaration = std::variant<EllipsisExceptionDeclarationAST*,
                                          TypeExceptionDeclarationAST*>;

using AttributeSpecifier =
    std::variant<CxxAttributeAST*, GccAttributeAST*, AlignasAttributeAST*,
                 AlignasTypeAttributeAST*, AsmAttributeAST*>;

using AttributeToken =
    std::variant<ScopedAttributeTokenAST*, SimpleAttributeTokenAST*>;

auto from(UnitAST* ast) -> std::optional<Unit>;
auto from(DeclarationAST* ast) -> std::optional<Declaration>;
auto from(StatementAST* ast) -> std::optional<Statement>;
auto from(ExpressionAST* ast) -> std::optional<Expression>;
auto from(TemplateParameterAST* ast) -> std::optional<TemplateParameter>;
auto from(SpecifierAST* ast) -> std::optional<Specifier>;
auto from(PtrOperatorAST* ast) -> std::optional<PtrOperator>;
auto from(CoreDeclaratorAST* ast) -> std::optional<CoreDeclarator>;
auto from(DeclaratorChunkAST* ast) -> std::optional<DeclaratorChunk>;
auto from(UnqualifiedIdAST* ast) -> std::optional<UnqualifiedId>;
auto from(NestedNameSpecifierAST* ast) -> std::optional<NestedNameSpecifier>;
auto from(FunctionBodyAST* ast) -> std::optional<FunctionBody>;
auto from(TemplateArgumentAST* ast) -> std::optional<TemplateArgument>;
auto from(ExceptionSpecifierAST* ast) -> std::optional<ExceptionSpecifier>;
auto from(RequirementAST* ast) -> std::optional<Requirement>;
auto from(NewInitializerAST* ast) -> std::optional<NewInitializer>;
auto from(MemInitializerAST* ast) -> std::optional<MemInitializer>;
auto from(LambdaCaptureAST* ast) -> std::optional<LambdaCapture>;
auto from(ExceptionDeclarationAST* ast) -> std::optional<ExceptionDeclaration>;
auto from(AttributeSpecifierAST* ast) -> std::optional<AttributeSpecifier>;
auto from(AttributeTokenAST* ast) -> std::optional<AttributeToken>;

}  // namespace cxx::syntax
