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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ASTInterpreter::SpecifierVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(TypedefSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(FriendSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstevalSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstinitSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstexprSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(InlineSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(NoreturnSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(StaticSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ExternSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(RegisterSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ThreadLocalSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(ThreadSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(MutableSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(VirtualSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ExplicitSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(AutoTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(VoidTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(SizeTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(SignTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(BuiltinTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(UnaryBuiltinTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(BinaryBuiltinTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(IntegralTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(FloatingPointTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(ComplexTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(NamedTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(AtomicTypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(UnderlyingTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(ElaboratedTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(DecltypeAutoSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(DecltypeSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(PlaceholderTypeSpecifierAST* ast)
      -> SpecifierResult;

  [[nodiscard]] auto operator()(ConstQualifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(VolatileQualifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(RestrictQualifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(AtomicQualifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(EnumSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(ClassSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(TypenameSpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto operator()(SplicerTypeSpecifierAST* ast)
      -> SpecifierResult;
};

struct ASTInterpreter::AttributeSpecifierVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(CxxAttributeAST* ast)
      -> AttributeSpecifierResult;

  [[nodiscard]] auto operator()(GccAttributeAST* ast)
      -> AttributeSpecifierResult;

  [[nodiscard]] auto operator()(AlignasAttributeAST* ast)
      -> AttributeSpecifierResult;

  [[nodiscard]] auto operator()(AlignasTypeAttributeAST* ast)
      -> AttributeSpecifierResult;

  [[nodiscard]] auto operator()(AsmAttributeAST* ast)
      -> AttributeSpecifierResult;
};

struct ASTInterpreter::AttributeTokenVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(ScopedAttributeTokenAST* ast)
      -> AttributeTokenResult;

  [[nodiscard]] auto operator()(SimpleAttributeTokenAST* ast)
      -> AttributeTokenResult;
};

auto ASTInterpreter::specifier(SpecifierAST* ast) -> SpecifierResult {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::attributeSpecifier(AttributeSpecifierAST* ast)
    -> AttributeSpecifierResult {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::attributeToken(AttributeTokenAST* ast)
    -> AttributeTokenResult {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::baseSpecifier(BaseSpecifierAST* ast)
    -> BaseSpecifierResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::enumerator(EnumeratorAST* ast) -> EnumeratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  auto expressionResult = expression(ast->expression);

  return {};
}

auto ASTInterpreter::typeId(TypeIdAST* ast) -> TypeIdResult {
  if (!ast) return {};

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = specifier(node);
  }

  auto declaratorResult = declarator(ast->declarator);

  return {};
}

auto ASTInterpreter::attributeArgumentClause(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::attribute(AttributeAST* ast) -> AttributeResult {
  if (!ast) return {};

  auto attributeTokenResult = attributeToken(ast->attributeToken);
  auto attributeArgumentClauseResult =
      attributeArgumentClause(ast->attributeArgumentClause);

  return {};
}

auto ASTInterpreter::attributeUsingPrefix(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::splicer(SplicerAST* ast) -> ExpressionResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return expressionResult;
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(NoreturnSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(RegisterSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(BuiltinTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    UnaryBuiltinTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    BinaryBuiltinTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    UnderlyingTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    ElaboratedTypeSpecifierAST* ast) -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    PlaceholderTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeConstraintResult = interp.typeConstraint(ast->typeConstraint);
  auto specifierResult = interp.specifier(ast->specifier);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AtomicQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  for (auto node : ListView{ast->enumeratorList}) {
    auto value = interp.enumerator(node);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->baseSpecifierList}) {
    auto value = interp.baseSpecifier(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto splicerResult = interp.splicer(ast->splicer);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto attributeUsingPrefixResult =
      interp.attributeUsingPrefix(ast->attributeUsingPrefix);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attribute(node);
  }

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

}  // namespace cxx
