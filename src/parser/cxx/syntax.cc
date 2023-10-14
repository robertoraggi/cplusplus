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

#include <cxx/ast.h>
#include <cxx/syntax.h>

namespace cxx::syntax {

auto from(UnitAST* ast) -> std::optional<Unit> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case TranslationUnitAST::Kind:
      return Unit(static_cast<TranslationUnitAST*>(ast));
    case ModuleUnitAST::Kind:
      return Unit(static_cast<ModuleUnitAST*>(ast));
    default:
      cxx_runtime_error("unexpected Unit");
  }  // switch
}

auto from(DeclarationAST* ast) -> std::optional<Declaration> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case SimpleDeclarationAST::Kind:
      return Declaration(static_cast<SimpleDeclarationAST*>(ast));
    case AsmDeclarationAST::Kind:
      return Declaration(static_cast<AsmDeclarationAST*>(ast));
    case NamespaceAliasDefinitionAST::Kind:
      return Declaration(static_cast<NamespaceAliasDefinitionAST*>(ast));
    case UsingDeclarationAST::Kind:
      return Declaration(static_cast<UsingDeclarationAST*>(ast));
    case UsingEnumDeclarationAST::Kind:
      return Declaration(static_cast<UsingEnumDeclarationAST*>(ast));
    case UsingDirectiveAST::Kind:
      return Declaration(static_cast<UsingDirectiveAST*>(ast));
    case StaticAssertDeclarationAST::Kind:
      return Declaration(static_cast<StaticAssertDeclarationAST*>(ast));
    case AliasDeclarationAST::Kind:
      return Declaration(static_cast<AliasDeclarationAST*>(ast));
    case OpaqueEnumDeclarationAST::Kind:
      return Declaration(static_cast<OpaqueEnumDeclarationAST*>(ast));
    case FunctionDefinitionAST::Kind:
      return Declaration(static_cast<FunctionDefinitionAST*>(ast));
    case TemplateDeclarationAST::Kind:
      return Declaration(static_cast<TemplateDeclarationAST*>(ast));
    case ConceptDefinitionAST::Kind:
      return Declaration(static_cast<ConceptDefinitionAST*>(ast));
    case DeductionGuideAST::Kind:
      return Declaration(static_cast<DeductionGuideAST*>(ast));
    case ExplicitInstantiationAST::Kind:
      return Declaration(static_cast<ExplicitInstantiationAST*>(ast));
    case ExportDeclarationAST::Kind:
      return Declaration(static_cast<ExportDeclarationAST*>(ast));
    case ExportCompoundDeclarationAST::Kind:
      return Declaration(static_cast<ExportCompoundDeclarationAST*>(ast));
    case LinkageSpecificationAST::Kind:
      return Declaration(static_cast<LinkageSpecificationAST*>(ast));
    case NamespaceDefinitionAST::Kind:
      return Declaration(static_cast<NamespaceDefinitionAST*>(ast));
    case EmptyDeclarationAST::Kind:
      return Declaration(static_cast<EmptyDeclarationAST*>(ast));
    case AttributeDeclarationAST::Kind:
      return Declaration(static_cast<AttributeDeclarationAST*>(ast));
    case ModuleImportDeclarationAST::Kind:
      return Declaration(static_cast<ModuleImportDeclarationAST*>(ast));
    case ParameterDeclarationAST::Kind:
      return Declaration(static_cast<ParameterDeclarationAST*>(ast));
    case AccessDeclarationAST::Kind:
      return Declaration(static_cast<AccessDeclarationAST*>(ast));
    case ForRangeDeclarationAST::Kind:
      return Declaration(static_cast<ForRangeDeclarationAST*>(ast));
    case StructuredBindingDeclarationAST::Kind:
      return Declaration(static_cast<StructuredBindingDeclarationAST*>(ast));
    case AsmOperandAST::Kind:
      return Declaration(static_cast<AsmOperandAST*>(ast));
    case AsmQualifierAST::Kind:
      return Declaration(static_cast<AsmQualifierAST*>(ast));
    case AsmClobberAST::Kind:
      return Declaration(static_cast<AsmClobberAST*>(ast));
    case AsmGotoLabelAST::Kind:
      return Declaration(static_cast<AsmGotoLabelAST*>(ast));
    default:
      cxx_runtime_error("unexpected Declaration");
  }  // switch
}

auto from(StatementAST* ast) -> std::optional<Statement> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case LabeledStatementAST::Kind:
      return Statement(static_cast<LabeledStatementAST*>(ast));
    case CaseStatementAST::Kind:
      return Statement(static_cast<CaseStatementAST*>(ast));
    case DefaultStatementAST::Kind:
      return Statement(static_cast<DefaultStatementAST*>(ast));
    case ExpressionStatementAST::Kind:
      return Statement(static_cast<ExpressionStatementAST*>(ast));
    case CompoundStatementAST::Kind:
      return Statement(static_cast<CompoundStatementAST*>(ast));
    case IfStatementAST::Kind:
      return Statement(static_cast<IfStatementAST*>(ast));
    case ConstevalIfStatementAST::Kind:
      return Statement(static_cast<ConstevalIfStatementAST*>(ast));
    case SwitchStatementAST::Kind:
      return Statement(static_cast<SwitchStatementAST*>(ast));
    case WhileStatementAST::Kind:
      return Statement(static_cast<WhileStatementAST*>(ast));
    case DoStatementAST::Kind:
      return Statement(static_cast<DoStatementAST*>(ast));
    case ForRangeStatementAST::Kind:
      return Statement(static_cast<ForRangeStatementAST*>(ast));
    case ForStatementAST::Kind:
      return Statement(static_cast<ForStatementAST*>(ast));
    case BreakStatementAST::Kind:
      return Statement(static_cast<BreakStatementAST*>(ast));
    case ContinueStatementAST::Kind:
      return Statement(static_cast<ContinueStatementAST*>(ast));
    case ReturnStatementAST::Kind:
      return Statement(static_cast<ReturnStatementAST*>(ast));
    case CoroutineReturnStatementAST::Kind:
      return Statement(static_cast<CoroutineReturnStatementAST*>(ast));
    case GotoStatementAST::Kind:
      return Statement(static_cast<GotoStatementAST*>(ast));
    case DeclarationStatementAST::Kind:
      return Statement(static_cast<DeclarationStatementAST*>(ast));
    case TryBlockStatementAST::Kind:
      return Statement(static_cast<TryBlockStatementAST*>(ast));
    default:
      cxx_runtime_error("unexpected Statement");
  }  // switch
}

auto from(ExpressionAST* ast) -> std::optional<Expression> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case CharLiteralExpressionAST::Kind:
      return Expression(static_cast<CharLiteralExpressionAST*>(ast));
    case BoolLiteralExpressionAST::Kind:
      return Expression(static_cast<BoolLiteralExpressionAST*>(ast));
    case IntLiteralExpressionAST::Kind:
      return Expression(static_cast<IntLiteralExpressionAST*>(ast));
    case FloatLiteralExpressionAST::Kind:
      return Expression(static_cast<FloatLiteralExpressionAST*>(ast));
    case NullptrLiteralExpressionAST::Kind:
      return Expression(static_cast<NullptrLiteralExpressionAST*>(ast));
    case StringLiteralExpressionAST::Kind:
      return Expression(static_cast<StringLiteralExpressionAST*>(ast));
    case UserDefinedStringLiteralExpressionAST::Kind:
      return Expression(
          static_cast<UserDefinedStringLiteralExpressionAST*>(ast));
    case ThisExpressionAST::Kind:
      return Expression(static_cast<ThisExpressionAST*>(ast));
    case NestedExpressionAST::Kind:
      return Expression(static_cast<NestedExpressionAST*>(ast));
    case IdExpressionAST::Kind:
      return Expression(static_cast<IdExpressionAST*>(ast));
    case LambdaExpressionAST::Kind:
      return Expression(static_cast<LambdaExpressionAST*>(ast));
    case FoldExpressionAST::Kind:
      return Expression(static_cast<FoldExpressionAST*>(ast));
    case RightFoldExpressionAST::Kind:
      return Expression(static_cast<RightFoldExpressionAST*>(ast));
    case LeftFoldExpressionAST::Kind:
      return Expression(static_cast<LeftFoldExpressionAST*>(ast));
    case RequiresExpressionAST::Kind:
      return Expression(static_cast<RequiresExpressionAST*>(ast));
    case SubscriptExpressionAST::Kind:
      return Expression(static_cast<SubscriptExpressionAST*>(ast));
    case CallExpressionAST::Kind:
      return Expression(static_cast<CallExpressionAST*>(ast));
    case TypeConstructionAST::Kind:
      return Expression(static_cast<TypeConstructionAST*>(ast));
    case BracedTypeConstructionAST::Kind:
      return Expression(static_cast<BracedTypeConstructionAST*>(ast));
    case MemberExpressionAST::Kind:
      return Expression(static_cast<MemberExpressionAST*>(ast));
    case PostIncrExpressionAST::Kind:
      return Expression(static_cast<PostIncrExpressionAST*>(ast));
    case CppCastExpressionAST::Kind:
      return Expression(static_cast<CppCastExpressionAST*>(ast));
    case TypeidExpressionAST::Kind:
      return Expression(static_cast<TypeidExpressionAST*>(ast));
    case TypeidOfTypeExpressionAST::Kind:
      return Expression(static_cast<TypeidOfTypeExpressionAST*>(ast));
    case UnaryExpressionAST::Kind:
      return Expression(static_cast<UnaryExpressionAST*>(ast));
    case AwaitExpressionAST::Kind:
      return Expression(static_cast<AwaitExpressionAST*>(ast));
    case SizeofExpressionAST::Kind:
      return Expression(static_cast<SizeofExpressionAST*>(ast));
    case SizeofTypeExpressionAST::Kind:
      return Expression(static_cast<SizeofTypeExpressionAST*>(ast));
    case SizeofPackExpressionAST::Kind:
      return Expression(static_cast<SizeofPackExpressionAST*>(ast));
    case AlignofTypeExpressionAST::Kind:
      return Expression(static_cast<AlignofTypeExpressionAST*>(ast));
    case AlignofExpressionAST::Kind:
      return Expression(static_cast<AlignofExpressionAST*>(ast));
    case NoexceptExpressionAST::Kind:
      return Expression(static_cast<NoexceptExpressionAST*>(ast));
    case NewExpressionAST::Kind:
      return Expression(static_cast<NewExpressionAST*>(ast));
    case DeleteExpressionAST::Kind:
      return Expression(static_cast<DeleteExpressionAST*>(ast));
    case CastExpressionAST::Kind:
      return Expression(static_cast<CastExpressionAST*>(ast));
    case ImplicitCastExpressionAST::Kind:
      return Expression(static_cast<ImplicitCastExpressionAST*>(ast));
    case BinaryExpressionAST::Kind:
      return Expression(static_cast<BinaryExpressionAST*>(ast));
    case ConditionalExpressionAST::Kind:
      return Expression(static_cast<ConditionalExpressionAST*>(ast));
    case YieldExpressionAST::Kind:
      return Expression(static_cast<YieldExpressionAST*>(ast));
    case ThrowExpressionAST::Kind:
      return Expression(static_cast<ThrowExpressionAST*>(ast));
    case AssignmentExpressionAST::Kind:
      return Expression(static_cast<AssignmentExpressionAST*>(ast));
    case PackExpansionExpressionAST::Kind:
      return Expression(static_cast<PackExpansionExpressionAST*>(ast));
    case DesignatedInitializerClauseAST::Kind:
      return Expression(static_cast<DesignatedInitializerClauseAST*>(ast));
    case TypeTraitsExpressionAST::Kind:
      return Expression(static_cast<TypeTraitsExpressionAST*>(ast));
    case ConditionExpressionAST::Kind:
      return Expression(static_cast<ConditionExpressionAST*>(ast));
    case EqualInitializerAST::Kind:
      return Expression(static_cast<EqualInitializerAST*>(ast));
    case BracedInitListAST::Kind:
      return Expression(static_cast<BracedInitListAST*>(ast));
    case ParenInitializerAST::Kind:
      return Expression(static_cast<ParenInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected Expression");
  }  // switch
}

auto from(TemplateParameterAST* ast) -> std::optional<TemplateParameter> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case TemplateTypeParameterAST::Kind:
      return TemplateParameter(static_cast<TemplateTypeParameterAST*>(ast));
    case TemplatePackTypeParameterAST::Kind:
      return TemplateParameter(static_cast<TemplatePackTypeParameterAST*>(ast));
    case NonTypeTemplateParameterAST::Kind:
      return TemplateParameter(static_cast<NonTypeTemplateParameterAST*>(ast));
    case TypenameTypeParameterAST::Kind:
      return TemplateParameter(static_cast<TypenameTypeParameterAST*>(ast));
    case ConstraintTypeParameterAST::Kind:
      return TemplateParameter(static_cast<ConstraintTypeParameterAST*>(ast));
    default:
      cxx_runtime_error("unexpected TemplateParameter");
  }  // switch
}

auto from(SpecifierAST* ast) -> std::optional<Specifier> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case TypedefSpecifierAST::Kind:
      return Specifier(static_cast<TypedefSpecifierAST*>(ast));
    case FriendSpecifierAST::Kind:
      return Specifier(static_cast<FriendSpecifierAST*>(ast));
    case ConstevalSpecifierAST::Kind:
      return Specifier(static_cast<ConstevalSpecifierAST*>(ast));
    case ConstinitSpecifierAST::Kind:
      return Specifier(static_cast<ConstinitSpecifierAST*>(ast));
    case ConstexprSpecifierAST::Kind:
      return Specifier(static_cast<ConstexprSpecifierAST*>(ast));
    case InlineSpecifierAST::Kind:
      return Specifier(static_cast<InlineSpecifierAST*>(ast));
    case StaticSpecifierAST::Kind:
      return Specifier(static_cast<StaticSpecifierAST*>(ast));
    case ExternSpecifierAST::Kind:
      return Specifier(static_cast<ExternSpecifierAST*>(ast));
    case ThreadLocalSpecifierAST::Kind:
      return Specifier(static_cast<ThreadLocalSpecifierAST*>(ast));
    case ThreadSpecifierAST::Kind:
      return Specifier(static_cast<ThreadSpecifierAST*>(ast));
    case MutableSpecifierAST::Kind:
      return Specifier(static_cast<MutableSpecifierAST*>(ast));
    case VirtualSpecifierAST::Kind:
      return Specifier(static_cast<VirtualSpecifierAST*>(ast));
    case ExplicitSpecifierAST::Kind:
      return Specifier(static_cast<ExplicitSpecifierAST*>(ast));
    case AutoTypeSpecifierAST::Kind:
      return Specifier(static_cast<AutoTypeSpecifierAST*>(ast));
    case VoidTypeSpecifierAST::Kind:
      return Specifier(static_cast<VoidTypeSpecifierAST*>(ast));
    case SizeTypeSpecifierAST::Kind:
      return Specifier(static_cast<SizeTypeSpecifierAST*>(ast));
    case SignTypeSpecifierAST::Kind:
      return Specifier(static_cast<SignTypeSpecifierAST*>(ast));
    case VaListTypeSpecifierAST::Kind:
      return Specifier(static_cast<VaListTypeSpecifierAST*>(ast));
    case IntegralTypeSpecifierAST::Kind:
      return Specifier(static_cast<IntegralTypeSpecifierAST*>(ast));
    case FloatingPointTypeSpecifierAST::Kind:
      return Specifier(static_cast<FloatingPointTypeSpecifierAST*>(ast));
    case ComplexTypeSpecifierAST::Kind:
      return Specifier(static_cast<ComplexTypeSpecifierAST*>(ast));
    case NamedTypeSpecifierAST::Kind:
      return Specifier(static_cast<NamedTypeSpecifierAST*>(ast));
    case AtomicTypeSpecifierAST::Kind:
      return Specifier(static_cast<AtomicTypeSpecifierAST*>(ast));
    case UnderlyingTypeSpecifierAST::Kind:
      return Specifier(static_cast<UnderlyingTypeSpecifierAST*>(ast));
    case ElaboratedTypeSpecifierAST::Kind:
      return Specifier(static_cast<ElaboratedTypeSpecifierAST*>(ast));
    case DecltypeAutoSpecifierAST::Kind:
      return Specifier(static_cast<DecltypeAutoSpecifierAST*>(ast));
    case DecltypeSpecifierAST::Kind:
      return Specifier(static_cast<DecltypeSpecifierAST*>(ast));
    case PlaceholderTypeSpecifierAST::Kind:
      return Specifier(static_cast<PlaceholderTypeSpecifierAST*>(ast));
    case ConstQualifierAST::Kind:
      return Specifier(static_cast<ConstQualifierAST*>(ast));
    case VolatileQualifierAST::Kind:
      return Specifier(static_cast<VolatileQualifierAST*>(ast));
    case RestrictQualifierAST::Kind:
      return Specifier(static_cast<RestrictQualifierAST*>(ast));
    case EnumSpecifierAST::Kind:
      return Specifier(static_cast<EnumSpecifierAST*>(ast));
    case ClassSpecifierAST::Kind:
      return Specifier(static_cast<ClassSpecifierAST*>(ast));
    case TypenameSpecifierAST::Kind:
      return Specifier(static_cast<TypenameSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected Specifier");
  }  // switch
}

auto from(PtrOperatorAST* ast) -> std::optional<PtrOperator> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case PointerOperatorAST::Kind:
      return PtrOperator(static_cast<PointerOperatorAST*>(ast));
    case ReferenceOperatorAST::Kind:
      return PtrOperator(static_cast<ReferenceOperatorAST*>(ast));
    case PtrToMemberOperatorAST::Kind:
      return PtrOperator(static_cast<PtrToMemberOperatorAST*>(ast));
    default:
      cxx_runtime_error("unexpected PtrOperator");
  }  // switch
}

auto from(CoreDeclaratorAST* ast) -> std::optional<CoreDeclarator> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case BitfieldDeclaratorAST::Kind:
      return CoreDeclarator(static_cast<BitfieldDeclaratorAST*>(ast));
    case ParameterPackAST::Kind:
      return CoreDeclarator(static_cast<ParameterPackAST*>(ast));
    case IdDeclaratorAST::Kind:
      return CoreDeclarator(static_cast<IdDeclaratorAST*>(ast));
    case NestedDeclaratorAST::Kind:
      return CoreDeclarator(static_cast<NestedDeclaratorAST*>(ast));
    default:
      cxx_runtime_error("unexpected CoreDeclarator");
  }  // switch
}

auto from(DeclaratorChunkAST* ast) -> std::optional<DeclaratorChunk> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case FunctionDeclaratorChunkAST::Kind:
      return DeclaratorChunk(static_cast<FunctionDeclaratorChunkAST*>(ast));
    case ArrayDeclaratorChunkAST::Kind:
      return DeclaratorChunk(static_cast<ArrayDeclaratorChunkAST*>(ast));
    default:
      cxx_runtime_error("unexpected DeclaratorChunk");
  }  // switch
}

auto from(UnqualifiedIdAST* ast) -> std::optional<UnqualifiedId> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case NameIdAST::Kind:
      return UnqualifiedId(static_cast<NameIdAST*>(ast));
    case DestructorIdAST::Kind:
      return UnqualifiedId(static_cast<DestructorIdAST*>(ast));
    case DecltypeIdAST::Kind:
      return UnqualifiedId(static_cast<DecltypeIdAST*>(ast));
    case OperatorFunctionIdAST::Kind:
      return UnqualifiedId(static_cast<OperatorFunctionIdAST*>(ast));
    case LiteralOperatorIdAST::Kind:
      return UnqualifiedId(static_cast<LiteralOperatorIdAST*>(ast));
    case ConversionFunctionIdAST::Kind:
      return UnqualifiedId(static_cast<ConversionFunctionIdAST*>(ast));
    case SimpleTemplateIdAST::Kind:
      return UnqualifiedId(static_cast<SimpleTemplateIdAST*>(ast));
    case LiteralOperatorTemplateIdAST::Kind:
      return UnqualifiedId(static_cast<LiteralOperatorTemplateIdAST*>(ast));
    case OperatorFunctionTemplateIdAST::Kind:
      return UnqualifiedId(static_cast<OperatorFunctionTemplateIdAST*>(ast));
    default:
      cxx_runtime_error("unexpected UnqualifiedId");
  }  // switch
}

auto from(NestedNameSpecifierAST* ast) -> std::optional<NestedNameSpecifier> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case GlobalNestedNameSpecifierAST::Kind:
      return NestedNameSpecifier(
          static_cast<GlobalNestedNameSpecifierAST*>(ast));
    case SimpleNestedNameSpecifierAST::Kind:
      return NestedNameSpecifier(
          static_cast<SimpleNestedNameSpecifierAST*>(ast));
    case DecltypeNestedNameSpecifierAST::Kind:
      return NestedNameSpecifier(
          static_cast<DecltypeNestedNameSpecifierAST*>(ast));
    case TemplateNestedNameSpecifierAST::Kind:
      return NestedNameSpecifier(
          static_cast<TemplateNestedNameSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected NestedNameSpecifier");
  }  // switch
}

auto from(FunctionBodyAST* ast) -> std::optional<FunctionBody> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case DefaultFunctionBodyAST::Kind:
      return FunctionBody(static_cast<DefaultFunctionBodyAST*>(ast));
    case CompoundStatementFunctionBodyAST::Kind:
      return FunctionBody(static_cast<CompoundStatementFunctionBodyAST*>(ast));
    case TryStatementFunctionBodyAST::Kind:
      return FunctionBody(static_cast<TryStatementFunctionBodyAST*>(ast));
    case DeleteFunctionBodyAST::Kind:
      return FunctionBody(static_cast<DeleteFunctionBodyAST*>(ast));
    default:
      cxx_runtime_error("unexpected FunctionBody");
  }  // switch
}

auto from(TemplateArgumentAST* ast) -> std::optional<TemplateArgument> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case TypeTemplateArgumentAST::Kind:
      return TemplateArgument(static_cast<TypeTemplateArgumentAST*>(ast));
    case ExpressionTemplateArgumentAST::Kind:
      return TemplateArgument(static_cast<ExpressionTemplateArgumentAST*>(ast));
    default:
      cxx_runtime_error("unexpected TemplateArgument");
  }  // switch
}

auto from(ExceptionSpecifierAST* ast) -> std::optional<ExceptionSpecifier> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case ThrowExceptionSpecifierAST::Kind:
      return ExceptionSpecifier(static_cast<ThrowExceptionSpecifierAST*>(ast));
    case NoexceptSpecifierAST::Kind:
      return ExceptionSpecifier(static_cast<NoexceptSpecifierAST*>(ast));
    default:
      cxx_runtime_error("unexpected ExceptionSpecifier");
  }  // switch
}

auto from(RequirementAST* ast) -> std::optional<Requirement> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case SimpleRequirementAST::Kind:
      return Requirement(static_cast<SimpleRequirementAST*>(ast));
    case CompoundRequirementAST::Kind:
      return Requirement(static_cast<CompoundRequirementAST*>(ast));
    case TypeRequirementAST::Kind:
      return Requirement(static_cast<TypeRequirementAST*>(ast));
    case NestedRequirementAST::Kind:
      return Requirement(static_cast<NestedRequirementAST*>(ast));
    default:
      cxx_runtime_error("unexpected Requirement");
  }  // switch
}

auto from(NewInitializerAST* ast) -> std::optional<NewInitializer> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case NewParenInitializerAST::Kind:
      return NewInitializer(static_cast<NewParenInitializerAST*>(ast));
    case NewBracedInitializerAST::Kind:
      return NewInitializer(static_cast<NewBracedInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected NewInitializer");
  }  // switch
}

auto from(MemInitializerAST* ast) -> std::optional<MemInitializer> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case ParenMemInitializerAST::Kind:
      return MemInitializer(static_cast<ParenMemInitializerAST*>(ast));
    case BracedMemInitializerAST::Kind:
      return MemInitializer(static_cast<BracedMemInitializerAST*>(ast));
    default:
      cxx_runtime_error("unexpected MemInitializer");
  }  // switch
}

auto from(LambdaCaptureAST* ast) -> std::optional<LambdaCapture> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case ThisLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<ThisLambdaCaptureAST*>(ast));
    case DerefThisLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<DerefThisLambdaCaptureAST*>(ast));
    case SimpleLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<SimpleLambdaCaptureAST*>(ast));
    case RefLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<RefLambdaCaptureAST*>(ast));
    case RefInitLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<RefInitLambdaCaptureAST*>(ast));
    case InitLambdaCaptureAST::Kind:
      return LambdaCapture(static_cast<InitLambdaCaptureAST*>(ast));
    default:
      cxx_runtime_error("unexpected LambdaCapture");
  }  // switch
}

auto from(ExceptionDeclarationAST* ast) -> std::optional<ExceptionDeclaration> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case EllipsisExceptionDeclarationAST::Kind:
      return ExceptionDeclaration(
          static_cast<EllipsisExceptionDeclarationAST*>(ast));
    case TypeExceptionDeclarationAST::Kind:
      return ExceptionDeclaration(
          static_cast<TypeExceptionDeclarationAST*>(ast));
    default:
      cxx_runtime_error("unexpected ExceptionDeclaration");
  }  // switch
}

auto from(AttributeSpecifierAST* ast) -> std::optional<AttributeSpecifier> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case CxxAttributeAST::Kind:
      return AttributeSpecifier(static_cast<CxxAttributeAST*>(ast));
    case GccAttributeAST::Kind:
      return AttributeSpecifier(static_cast<GccAttributeAST*>(ast));
    case AlignasAttributeAST::Kind:
      return AttributeSpecifier(static_cast<AlignasAttributeAST*>(ast));
    case AlignasTypeAttributeAST::Kind:
      return AttributeSpecifier(static_cast<AlignasTypeAttributeAST*>(ast));
    case AsmAttributeAST::Kind:
      return AttributeSpecifier(static_cast<AsmAttributeAST*>(ast));
    default:
      cxx_runtime_error("unexpected AttributeSpecifier");
  }  // switch
}

auto from(AttributeTokenAST* ast) -> std::optional<AttributeToken> {
  if (!ast) return std::nullopt;
  switch (ast->kind()) {
    case ScopedAttributeTokenAST::Kind:
      return AttributeToken(static_cast<ScopedAttributeTokenAST*>(ast));
    case SimpleAttributeTokenAST::Kind:
      return AttributeToken(static_cast<SimpleAttributeTokenAST*>(ast));
    default:
      cxx_runtime_error("unexpected AttributeToken");
  }  // switch
}

}  // namespace cxx::syntax
