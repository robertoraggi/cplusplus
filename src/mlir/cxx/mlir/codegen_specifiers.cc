// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>

namespace cxx {

struct Codegen::SpecifierVisitor {
  Codegen& gen;

  auto operator()(GeneratedTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(TypedefSpecifierAST* ast) -> SpecifierResult;
  auto operator()(FriendSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ConstevalSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ConstinitSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ConstexprSpecifierAST* ast) -> SpecifierResult;
  auto operator()(InlineSpecifierAST* ast) -> SpecifierResult;
  auto operator()(NoreturnSpecifierAST* ast) -> SpecifierResult;
  auto operator()(StaticSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ExternSpecifierAST* ast) -> SpecifierResult;
  auto operator()(RegisterSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ThreadLocalSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ThreadSpecifierAST* ast) -> SpecifierResult;
  auto operator()(MutableSpecifierAST* ast) -> SpecifierResult;
  auto operator()(VirtualSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ExplicitSpecifierAST* ast) -> SpecifierResult;
  auto operator()(AutoTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(VoidTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(SizeTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(SignTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(VaListTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(IntegralTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(FloatingPointTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ComplexTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(NamedTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(AtomicTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(UnderlyingTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ElaboratedTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(DecltypeAutoSpecifierAST* ast) -> SpecifierResult;
  auto operator()(DecltypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(PlaceholderTypeSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ConstQualifierAST* ast) -> SpecifierResult;
  auto operator()(VolatileQualifierAST* ast) -> SpecifierResult;
  auto operator()(RestrictQualifierAST* ast) -> SpecifierResult;
  auto operator()(AtomicQualifierAST* ast) -> SpecifierResult;
  auto operator()(EnumSpecifierAST* ast) -> SpecifierResult;
  auto operator()(ClassSpecifierAST* ast) -> SpecifierResult;
  auto operator()(TypenameSpecifierAST* ast) -> SpecifierResult;
  auto operator()(SplicerTypeSpecifierAST* ast) -> SpecifierResult;
};

struct Codegen::AttributeSpecifierVisitor {
  Codegen& gen;

  auto operator()(CxxAttributeAST* ast) -> AttributeSpecifierResult;
  auto operator()(GccAttributeAST* ast) -> AttributeSpecifierResult;
  auto operator()(AlignasAttributeAST* ast) -> AttributeSpecifierResult;
  auto operator()(AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult;
  auto operator()(AsmAttributeAST* ast) -> AttributeSpecifierResult;
};

struct Codegen::AttributeTokenVisitor {
  Codegen& gen;

  auto operator()(ScopedAttributeTokenAST* ast) -> AttributeTokenResult;
  auto operator()(SimpleAttributeTokenAST* ast) -> AttributeTokenResult;
};

auto Codegen::enumerator(EnumeratorAST* ast) -> EnumeratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::baseSpecifier(BaseSpecifierAST* ast) -> BaseSpecifierResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::specifier(SpecifierAST* ast) -> SpecifierResult {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::attributeSpecifier(AttributeSpecifierAST* ast)
    -> AttributeSpecifierResult {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::attributeToken(AttributeTokenAST* ast) -> AttributeTokenResult {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto Codegen::attributeArgumentClause(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseResult {
  if (!ast) return {};

  return {};
}

auto Codegen::attribute(AttributeAST* ast) -> AttributeResult {
  if (!ast) return {};

  auto attributeTokenResult = attributeToken(ast->attributeToken);

  auto attributeArgumentClauseResult =
      attributeArgumentClause(ast->attributeArgumentClause);

  return {};
}

auto Codegen::attributeUsingPrefix(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixResult {
  if (!ast) return {};

  return {};
}

auto Codegen::splicer(SplicerAST* ast) -> SplicerResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::typeId(TypeIdAST* ast) -> TypeIdResult {
  if (!ast) return {};

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = specifier(node);
  }

  auto declaratorResult = declarator(ast->declarator);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(GeneratedTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(NoreturnSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(RegisterSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(FloatingPointTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(UnderlyingTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ElaboratedTypeSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(PlaceholderTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeConstraintResult = gen.typeConstraint(ast->typeConstraint);

  auto specifierResult = gen.specifier(ast->specifier);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(AtomicQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto Codegen::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }

  for (auto node : ListView{ast->enumeratorList}) {
    auto value = gen.enumerator(node);
  }

  return {};
}

auto Codegen::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->baseSpecifierList}) {
    auto value = gen.baseSpecifier(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto splicerResult = gen.splicer(ast->splicer);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto attributeUsingPrefixResult =
      gen.attributeUsingPrefix(ast->attributeUsingPrefix);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attribute(node);
  }

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(AlignasAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto Codegen::AttributeTokenVisitor::operator()(ScopedAttributeTokenAST* ast)
    -> AttributeTokenResult {
  return {};
}

auto Codegen::AttributeTokenVisitor::operator()(SimpleAttributeTokenAST* ast)
    -> AttributeTokenResult {
  return {};
}

}  // namespace cxx