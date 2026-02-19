// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/dependent_types.h>

// cxx
#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

namespace {

struct IsDependent {
  TranslationUnit* unit = nullptr;

  [[nodiscard]] auto isDependent(ExpressionAST* ast) -> bool {
    if (!ast) return false;
    return visit(*this, ast);
  }

  [[nodiscard]] auto isDependent(TypeIdAST* ast) -> bool {
    if (isDependent(ast->type)) return true;
    for (auto typeSpecifier : ListView{ast->typeSpecifierList}) {
      if (isDependent(typeSpecifier)) return true;
    }
    return false;
  }

  [[nodiscard]] auto isInTemplateScope(Symbol* symbol) -> bool {
    for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
      if (scope->isTemplateParameters()) return true;
    }
    return false;
  }

  [[nodiscard]] auto isDependent(const Type* type) -> bool {
    if (!type) return false;
    return visit(*this, type);
  }

  auto operator()(const VoidType* type) -> bool { return false; }
  auto operator()(const NullptrType* type) -> bool { return false; }
  auto operator()(const DecltypeAutoType* type) -> bool { return false; }
  auto operator()(const AutoType* type) -> bool { return false; }
  auto operator()(const BoolType* type) -> bool { return false; }
  auto operator()(const SignedCharType* type) -> bool { return false; }
  auto operator()(const ShortIntType* type) -> bool { return false; }
  auto operator()(const IntType* type) -> bool { return false; }
  auto operator()(const LongIntType* type) -> bool { return false; }
  auto operator()(const LongLongIntType* type) -> bool { return false; }
  auto operator()(const Int128Type* type) -> bool { return false; }
  auto operator()(const UnsignedCharType* type) -> bool { return false; }
  auto operator()(const UnsignedShortIntType* type) -> bool { return false; }
  auto operator()(const UnsignedIntType* type) -> bool { return false; }
  auto operator()(const UnsignedLongIntType* type) -> bool { return false; }
  auto operator()(const UnsignedLongLongIntType* type) -> bool { return false; }
  auto operator()(const UnsignedInt128Type* type) -> bool { return false; }
  auto operator()(const CharType* type) -> bool { return false; }
  auto operator()(const Char8Type* type) -> bool { return false; }
  auto operator()(const Char16Type* type) -> bool { return false; }
  auto operator()(const Char32Type* type) -> bool { return false; }
  auto operator()(const WideCharType* type) -> bool { return false; }
  auto operator()(const FloatType* type) -> bool { return false; }
  auto operator()(const DoubleType* type) -> bool { return false; }
  auto operator()(const LongDoubleType* type) -> bool { return false; }
  auto operator()(const Float16Type* type) -> bool { return false; }

  auto operator()(const QualType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const BoundedArrayType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const PointerType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const LvalueReferenceType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const RvalueReferenceType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const FunctionType* type) -> bool {
    if (isDependent(type->returnType())) return true;
    for (const auto* param : type->parameterTypes()) {
      if (isDependent(param)) return true;
    }
    return false;
  }

  auto operator()(const ClassType* type) -> bool { return false; }

  auto operator()(const EnumType* type) -> bool { return false; }

  auto operator()(const ScopedEnumType* type) -> bool { return false; }

  auto operator()(const MemberObjectPointerType* type) -> bool {
    if (isDependent(type->classType())) return true;
    if (isDependent(type->elementType())) return true;
    return false;
  }

  auto operator()(const MemberFunctionPointerType* type) -> bool {
    if (isDependent(type->classType())) return true;
    if (isDependent(type->functionType())) return true;
    return false;
  }

  auto operator()(const NamespaceType* type) -> bool { return false; }

  auto operator()(const TypeParameterType* type) -> bool { return true; }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    return true;
  }

  auto operator()(const UnresolvedNameType* type) -> bool { return true; }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    return true;
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool { return true; }

  auto operator()(const OverloadSetType* type) -> bool { return false; }

  auto operator()(const BuiltinVaListType* type) -> bool { return false; }

  auto operator()(const BuiltinMetaInfoType* type) -> bool { return false; }

  // clang-format off
  [[nodiscard]] auto isDependent(NestedNameSpecifierAST* ast) -> bool;
  auto operator()(GlobalNestedNameSpecifierAST* ast) -> bool;
  auto operator()(SimpleNestedNameSpecifierAST* ast) -> bool;
  auto operator()(DecltypeNestedNameSpecifierAST* ast) -> bool;
  auto operator()(TemplateNestedNameSpecifierAST* ast) -> bool;


  [[nodiscard]] auto isDependent(StatementAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(UnqualifiedIdAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(LambdaCaptureAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(TemplateParameterAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(AttributeSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(RequiresClauseAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(ParameterDeclarationClauseAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(LambdaSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(ExceptionSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(TrailingReturnTypeAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(RequirementAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(SplicerAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(DesignatorAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(NewPlacementAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(DeclaratorAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(NewInitializerAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(GenericAssociationAST* ast) -> bool { return false; }
  // clang-format on

  auto operator()(CharLiteralExpressionAST* ast) -> bool;
  auto operator()(BoolLiteralExpressionAST* ast) -> bool;
  auto operator()(IntLiteralExpressionAST* ast) -> bool;
  auto operator()(FloatLiteralExpressionAST* ast) -> bool;
  auto operator()(NullptrLiteralExpressionAST* ast) -> bool;
  auto operator()(StringLiteralExpressionAST* ast) -> bool;
  auto operator()(UserDefinedStringLiteralExpressionAST* ast) -> bool;
  auto operator()(ObjectLiteralExpressionAST* ast) -> bool;
  auto operator()(ThisExpressionAST* ast) -> bool;
  auto operator()(GenericSelectionExpressionAST* ast) -> bool;
  auto operator()(NestedStatementExpressionAST* ast) -> bool;
  auto operator()(NestedExpressionAST* ast) -> bool;
  auto operator()(IdExpressionAST* ast) -> bool;
  auto operator()(LambdaExpressionAST* ast) -> bool;
  auto operator()(FoldExpressionAST* ast) -> bool;
  auto operator()(RightFoldExpressionAST* ast) -> bool;
  auto operator()(LeftFoldExpressionAST* ast) -> bool;
  auto operator()(RequiresExpressionAST* ast) -> bool;
  auto operator()(VaArgExpressionAST* ast) -> bool;
  auto operator()(SubscriptExpressionAST* ast) -> bool;
  auto operator()(CallExpressionAST* ast) -> bool;
  auto operator()(TypeConstructionAST* ast) -> bool;
  auto operator()(BracedTypeConstructionAST* ast) -> bool;
  auto operator()(SpliceMemberExpressionAST* ast) -> bool;
  auto operator()(MemberExpressionAST* ast) -> bool;
  auto operator()(PostIncrExpressionAST* ast) -> bool;
  auto operator()(CppCastExpressionAST* ast) -> bool;
  auto operator()(BuiltinBitCastExpressionAST* ast) -> bool;
  auto operator()(BuiltinOffsetofExpressionAST* ast) -> bool;
  auto operator()(TypeidExpressionAST* ast) -> bool;
  auto operator()(TypeidOfTypeExpressionAST* ast) -> bool;
  auto operator()(SpliceExpressionAST* ast) -> bool;
  auto operator()(GlobalScopeReflectExpressionAST* ast) -> bool;
  auto operator()(NamespaceReflectExpressionAST* ast) -> bool;
  auto operator()(TypeIdReflectExpressionAST* ast) -> bool;
  auto operator()(ReflectExpressionAST* ast) -> bool;
  auto operator()(LabelAddressExpressionAST* ast) -> bool;
  auto operator()(UnaryExpressionAST* ast) -> bool;
  auto operator()(AwaitExpressionAST* ast) -> bool;
  auto operator()(SizeofExpressionAST* ast) -> bool;
  auto operator()(SizeofTypeExpressionAST* ast) -> bool;
  auto operator()(SizeofPackExpressionAST* ast) -> bool;
  auto operator()(AlignofTypeExpressionAST* ast) -> bool;
  auto operator()(AlignofExpressionAST* ast) -> bool;
  auto operator()(NoexceptExpressionAST* ast) -> bool;
  auto operator()(NewExpressionAST* ast) -> bool;
  auto operator()(DeleteExpressionAST* ast) -> bool;
  auto operator()(CastExpressionAST* ast) -> bool;
  auto operator()(ImplicitCastExpressionAST* ast) -> bool;
  auto operator()(BinaryExpressionAST* ast) -> bool;
  auto operator()(ConditionalExpressionAST* ast) -> bool;
  auto operator()(YieldExpressionAST* ast) -> bool;
  auto operator()(ThrowExpressionAST* ast) -> bool;
  auto operator()(AssignmentExpressionAST* ast) -> bool;
  auto operator()(TargetExpressionAST* ast) -> bool;
  auto operator()(RightExpressionAST* ast) -> bool;
  auto operator()(CompoundAssignmentExpressionAST* ast) -> bool;
  auto operator()(PackExpansionExpressionAST* ast) -> bool;
  auto operator()(DesignatedInitializerClauseAST* ast) -> bool;
  auto operator()(TypeTraitExpressionAST* ast) -> bool;
  auto operator()(ConditionExpressionAST* ast) -> bool;
  auto operator()(EqualInitializerAST* ast) -> bool;
  auto operator()(BracedInitListAST* ast) -> bool;
  auto operator()(ParenInitializerAST* ast) -> bool;

  [[nodiscard]] auto isDependent(SpecifierAST* ast) -> bool {
    if (!ast) return false;
    return visit(*this, ast);
  }

  // clang-format off
  auto operator()(TypedefSpecifierAST* ast) -> bool { return false; }
  auto operator()(FriendSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstevalSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstinitSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstexprSpecifierAST* ast) -> bool { return false; }
  auto operator()(InlineSpecifierAST* ast) -> bool { return false; }
  auto operator()(NoreturnSpecifierAST* ast) -> bool { return false; }
  auto operator()(StaticSpecifierAST* ast) -> bool { return false; }
  auto operator()(ExternSpecifierAST* ast) -> bool { return false; }
  auto operator()(RegisterSpecifierAST* ast) -> bool { return false; }
  auto operator()(ThreadLocalSpecifierAST* ast) -> bool { return false; }
  auto operator()(ThreadSpecifierAST* ast) -> bool { return false; }
  auto operator()(MutableSpecifierAST* ast) -> bool { return false; }
  auto operator()(VirtualSpecifierAST* ast) -> bool { return false; }
  auto operator()(ExplicitSpecifierAST* ast) -> bool { return false; }
  auto operator()(AutoTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(VoidTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(SizeTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(SignTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(BuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(UnaryBuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(BinaryBuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(IntegralTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(FloatingPointTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ComplexTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(NamedTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(AtomicTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(UnderlyingTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ElaboratedTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(DecltypeAutoSpecifierAST* ast) -> bool { return false; }

  auto operator()(DecltypeSpecifierAST* ast) -> bool { 
    if (ast->type && isDependent(ast->type)) return true;
    if (ast->expression && !ast->type) return true;
    if (ast->expression && isDependent(ast->expression)) return true;
    return false;
  }

  auto operator()(PlaceholderTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstQualifierAST* ast) -> bool { return false; }
  auto operator()(VolatileQualifierAST* ast) -> bool { return false; }
  auto operator()(RestrictQualifierAST* ast) -> bool { return false; }
  auto operator()(AtomicQualifierAST* ast) -> bool { return false; }
  auto operator()(EnumSpecifierAST* ast) -> bool { return false; }
  auto operator()(ClassSpecifierAST* ast) -> bool { return false; }
  auto operator()(TypenameSpecifierAST* ast) -> bool { return false; }
  auto operator()(SplicerTypeSpecifierAST* ast) -> bool { return false; }
  // clang-format on
};

}  // namespace

auto IsDependent::isDependent(NestedNameSpecifierAST* ast) -> bool {
  if (!ast) return false;
  if (!ast->symbol) return true;
  if (visit(*this, ast)) return true;
  return false;
}

auto IsDependent::operator()(GlobalNestedNameSpecifierAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(SimpleNestedNameSpecifierAST* ast) -> bool {
  if (isDependent(ast->nestedNameSpecifier)) return true;
  return false;
}

auto IsDependent::operator()(DecltypeNestedNameSpecifierAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(TemplateNestedNameSpecifierAST* ast) -> bool {
  if (ast->templateId) {
    for (auto arg : ListView{ast->templateId->templateArgumentList}) {
      if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
        if (isDependent(typeArg->typeId)) return true;
      }
      if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
        if (isDependent(exprArg->expression)) return true;
      }
    }
  }

  if (isDependent(ast->nestedNameSpecifier)) return true;

  return false;
}

auto IsDependent::operator()(CharLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(BoolLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(IntLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(FloatLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(NullptrLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(StringLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(UserDefinedStringLiteralExpressionAST* ast)
    -> bool {
  return false;
}

auto IsDependent::operator()(ObjectLiteralExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->bracedInitList)) return true;

  return false;
}

auto IsDependent::operator()(ThisExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(GenericSelectionExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  for (auto node : ListView{ast->genericAssociationList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(NestedStatementExpressionAST* ast) -> bool {
  if (isDependent(ast->statement)) return true;

  return false;
}

auto IsDependent::operator()(NestedExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(IdExpressionAST* ast) -> bool {
  if (isDependent(ast->nestedNameSpecifier)) return true;
  if (isDependent(ast->unqualifiedId)) return true;

  if (symbol_cast<NonTypeParameterSymbol>(ast->symbol)) return true;
  if (symbol_cast<TypeParameterSymbol>(ast->symbol)) return true;
  if (symbol_cast<TemplateTypeParameterSymbol>(ast->symbol)) return true;

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (field->isStatic() && field->initializer()) {
      if (isInTemplateScope(field)) return true;
    }
  }
  if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
    if (var->initializer() && isInTemplateScope(var)) return true;
  }

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
    for (auto arg : ListView{templateId->templateArgumentList}) {
      if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
        if (isDependent(typeArg->typeId)) return true;
      }
      if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
        if (isDependent(exprArg->expression)) return true;
      }
    }
  }

  return false;
}

auto IsDependent::operator()(LambdaExpressionAST* ast) -> bool {
  for (auto node : ListView{ast->captureList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->templateParameterList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->templateRequiresClause)) return true;

  for (auto node : ListView{ast->expressionAttributeList}) {
    if (isDependent(node)) return true;
  }

  auto parameterDeclarationClauseResult =
      isDependent(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->exceptionSpecifier)) return true;

  for (auto node : ListView{ast->attributeList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->trailingReturnType)) return true;
  if (isDependent(ast->requiresClause)) return true;
  if (isDependent(ast->statement)) return true;

  return false;
}

auto IsDependent::operator()(FoldExpressionAST* ast) -> bool { return true; }

auto IsDependent::operator()(RightFoldExpressionAST* ast) -> bool {
  return true;
}

auto IsDependent::operator()(LeftFoldExpressionAST* ast) -> bool {
  return true;
}

auto IsDependent::operator()(RequiresExpressionAST* ast) -> bool {
  auto parameterDeclarationClauseResult =
      isDependent(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->requirementList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(VaArgExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SubscriptExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->indexExpression)) return true;

  return false;
}

auto IsDependent::operator()(CallExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;

  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(TypeConstructionAST* ast) -> bool {
  if (isDependent(ast->typeSpecifier)) return true;

  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(BracedTypeConstructionAST* ast) -> bool {
  if (isDependent(ast->typeSpecifier)) return true;
  if (isDependent(ast->bracedInitList)) return true;

  return false;
}

auto IsDependent::operator()(SpliceMemberExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->splicer)) return true;

  return false;
}

auto IsDependent::operator()(MemberExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->nestedNameSpecifier)) return true;
  if (isDependent(ast->unqualifiedId)) return true;

  return false;
}

auto IsDependent::operator()(PostIncrExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;

  return false;
}

auto IsDependent::operator()(CppCastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BuiltinBitCastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BuiltinOffsetofExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  for (auto node : ListView{ast->designatorList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(TypeidExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(TypeidOfTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SpliceExpressionAST* ast) -> bool {
  if (isDependent(ast->splicer)) return true;

  return false;
}

auto IsDependent::operator()(GlobalScopeReflectExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(NamespaceReflectExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(TypeIdReflectExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(ReflectExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(LabelAddressExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(UnaryExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(AwaitExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(SizeofExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(SizeofTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SizeofPackExpressionAST* ast) -> bool {
  return true;
}

auto IsDependent::operator()(AlignofTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(AlignofExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(NoexceptExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(NewExpressionAST* ast) -> bool {
  if (isDependent(ast->newPlacement)) return true;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->declarator)) return true;
  if (isDependent(ast->newInitalizer)) return true;

  return false;
}

auto IsDependent::operator()(DeleteExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(CastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(ImplicitCastExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BinaryExpressionAST* ast) -> bool {
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;

  return false;
}

auto IsDependent::operator()(ConditionalExpressionAST* ast) -> bool {
  if (isDependent(ast->condition)) return true;
  if (isDependent(ast->iftrueExpression)) return true;
  if (isDependent(ast->iffalseExpression)) return true;

  return false;
}

auto IsDependent::operator()(YieldExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(ThrowExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(AssignmentExpressionAST* ast) -> bool {
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;

  return false;
}

auto IsDependent::operator()(TargetExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(RightExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(CompoundAssignmentExpressionAST* ast) -> bool {
  if (isDependent(ast->targetExpression)) return true;
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;
  if (isDependent(ast->adjustExpression)) return true;

  return false;
}

auto IsDependent::operator()(PackExpansionExpressionAST* ast) -> bool {
  return true;
}

auto IsDependent::operator()(DesignatedInitializerClauseAST* ast) -> bool {
  for (auto node : ListView{ast->designatorList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->initializer)) return true;

  return false;
}

auto IsDependent::operator()(TypeTraitExpressionAST* ast) -> bool {
  if (isDependent(ast->type)) return true;

  for (auto node : ListView{ast->typeIdList}) {
    if (!node->type) return true;
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(ConditionExpressionAST* ast) -> bool {
  for (auto node : ListView{ast->attributeList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->declarator)) return true;
  if (isDependent(ast->initializer)) return true;

  return false;
}

auto IsDependent::operator()(EqualInitializerAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BracedInitListAST* ast) -> bool {
  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }
  return false;
}

auto IsDependent::operator()(ParenInitializerAST* ast) -> bool {
  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }
  return false;
}

auto isDependent(TranslationUnit* unit, ExpressionAST* ast) -> bool {
  return IsDependent{unit}.isDependent(ast);
}

auto isDependent(TranslationUnit* unit, TypeIdAST* ast) -> bool {
  return IsDependent{unit}.isDependent(ast);
}

auto isDependent(TranslationUnit* unit, SpecifierAST* spec) -> bool {
  return IsDependent{unit}.isDependent(spec);
}

auto isDependent(TranslationUnit* unit, const Type* type) -> bool {
  return IsDependent{unit}.isDependent(type);
}

}  // namespace cxx
