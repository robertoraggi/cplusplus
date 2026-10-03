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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(Control, integer_literals) {
  Control control;

  auto literal = control.integerLiteral("42");
  EXPECT_EQ(literal->value(), "42");

  EXPECT_EQ(control.integerLiteral("42"), literal);
}

TEST(Control, float_literals) {
  Control control;

  auto literal = control.floatLiteral("42.0");
  EXPECT_EQ(literal->value(), "42.0");

  EXPECT_EQ(control.floatLiteral("42.0"), literal);
}

TEST(Control, string_literals) {
  Control control;

  auto literal = control.stringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.stringLiteral("foo"), literal);
}

TEST(Control, char_literals) {
  Control control;

  auto literal = control.charLiteral("a");
  EXPECT_EQ(literal->value(), "a");

  EXPECT_EQ(control.charLiteral("a"), literal);
}

TEST(Control, wide_string_literals) {
  Control control;

  auto literal = control.wideStringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.wideStringLiteral("foo"), literal);
}

TEST(Control, utf8_string_literals) {
  Control control;

  auto literal = control.utf8StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf8StringLiteral("foo"), literal);
}

TEST(Control, utf16_string_literals) {
  Control control;

  auto literal = control.utf16StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf16StringLiteral("foo"), literal);
}

TEST(Control, utf32_string_literals) {
  Control control;

  auto literal = control.utf32StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf32StringLiteral("foo"), literal);
}

TEST(Control, comment_literals) {
  Control control;

  auto literal = control.commentLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.commentLiteral("foo"), literal);
}

TEST(Control, make_anonymous_id) {
  Control control;

  auto id = control.newAnonymousId("foo");
  EXPECT_TRUE(id->isAnonymous());
  EXPECT_TRUE(id->name().starts_with("$foo"));

  auto otherId = control.newAnonymousId("foo");
  EXPECT_NE(id, otherId);
  EXPECT_NE(id->name(), otherId->name());
}

TEST(Control, get_identifier) {
  Control control;

  auto id = control.getIdentifier("foo");
  EXPECT_EQ(id->name(), "foo");

  EXPECT_EQ(control.getIdentifier("foo"), id);
  EXPECT_NE(name_cast<Identifier>(id), nullptr);
}

TEST(Control, get_operator_id) {
  Control control;

  auto id = control.getOperatorId(TokenKind::T_PLUS);
  EXPECT_EQ(id->op(), TokenKind::T_PLUS);

  EXPECT_EQ(to_string(id), "operator +");

  EXPECT_EQ(control.getOperatorId(TokenKind::T_PLUS), id);
  EXPECT_NE(name_cast<OperatorId>(id), nullptr);
}

TEST(Control, get_destructor_id) {
  Control control;

  auto id = control.getDestructorId(control.getIdentifier("foo"));
  EXPECT_EQ(to_string(id), "~foo");

  EXPECT_EQ(control.getDestructorId(control.getIdentifier("foo")), id);
  EXPECT_NE(name_cast<DestructorId>(id), nullptr);
}

TEST(Control, overload_set_dedup_by_canonical) {
  Control control;

  auto global = control.newNamespaceSymbol(nullptr, {});
  auto overloadSet = control.newOverloadSetSymbol(global, {});

  auto f1 = control.newFunctionSymbol(global, {});
  f1->setName(control.getIdentifier("f"));
  f1->setType(control.getFunctionType(control.getVoidType(), {}));

  auto f2 = control.newFunctionSymbol(global, {});
  f2->setName(control.getIdentifier("f"));
  f2->setType(control.getFunctionType(control.getVoidType(), {}));
  f2->setCanonical(f1);

  overloadSet->addFunction(f1);
  overloadSet->addFunction(f2);

  EXPECT_EQ(overloadSet->functions().size(), 1);
}

TEST(Control, compare_args_with_type_arguments) {
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  auto& control = *unit.control();

  std::vector<TemplateArgument> lhs{control.getIntType()};
  std::vector<TemplateArgument> rhs{control.getIntType()};

  EXPECT_TRUE(compare_args(&unit, lhs, rhs));
}

TEST(Control, compare_args_symbol_and_type_equivalent) {
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  auto& control = *unit.control();

  auto global = control.newNamespaceSymbol(nullptr, {});
  auto typeAlias = control.newTypeAliasSymbol(global, {});
  typeAlias->setName(control.getIdentifier("AliasInt"));
  typeAlias->setType(control.getIntType());

  std::vector<TemplateArgument> lhs{static_cast<Symbol*>(typeAlias)};
  std::vector<TemplateArgument> rhs{control.getIntType()};

  EXPECT_TRUE(compare_args(&unit, lhs, rhs));
}

TEST(Control, compare_args_parameter_pack_equivalent) {
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  auto& control = *unit.control();

  auto global = control.newNamespaceSymbol(nullptr, {});

  auto packL = control.newParameterPackSymbol(global, {});
  auto l0 = control.newTypeAliasSymbol(global, {});
  l0->setType(control.getIntType());
  auto l1 = control.newTypeAliasSymbol(global, {});
  l1->setType(control.getCharType());
  packL->addElement(l0);
  packL->addElement(l1);

  auto packR = control.newParameterPackSymbol(global, {});
  auto r0 = control.newTypeAliasSymbol(global, {});
  r0->setType(control.getIntType());
  auto r1 = control.newTypeAliasSymbol(global, {});
  r1->setType(control.getCharType());
  packR->addElement(r0);
  packR->addElement(r1);

  std::vector<TemplateArgument> lhs{static_cast<Symbol*>(packL)};
  std::vector<TemplateArgument> rhs{static_cast<Symbol*>(packR)};

  EXPECT_TRUE(compare_args(&unit, lhs, rhs));
}

TEST(Control, compare_args_parameter_pack_order_matters) {
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  auto& control = *unit.control();

  auto global = control.newNamespaceSymbol(nullptr, {});

  auto packL = control.newParameterPackSymbol(global, {});
  auto l0 = control.newTypeAliasSymbol(global, {});
  l0->setType(control.getIntType());
  auto l1 = control.newTypeAliasSymbol(global, {});
  l1->setType(control.getCharType());
  packL->addElement(l0);
  packL->addElement(l1);

  auto packR = control.newParameterPackSymbol(global, {});
  auto r0 = control.newTypeAliasSymbol(global, {});
  r0->setType(control.getCharType());
  auto r1 = control.newTypeAliasSymbol(global, {});
  r1->setType(control.getIntType());
  packR->addElement(r0);
  packR->addElement(r1);

  std::vector<TemplateArgument> lhs{static_cast<Symbol*>(packL)};
  std::vector<TemplateArgument> rhs{static_cast<Symbol*>(packR)};

  EXPECT_FALSE(compare_args(&unit, lhs, rhs));
}

TEST(Control, interned_function_types) {
  Control control;
  auto integer = control.getIntType();
  auto floating = control.getFloatType();
  auto function = control.getFunctionType(integer, {floating, integer});
  EXPECT_EQ(function, control.getFunctionType(integer, {floating, integer}));
  EXPECT_NE(function, control.getFunctionType(integer, {integer, floating}));
  EXPECT_NE(function, control.getFunctionType(floating, {floating, integer}));
  EXPECT_NE(function,
            control.getFunctionType(integer, {floating, integer}, true));
  EXPECT_EQ(control.getPointerType(function), control.getPointerType(function));
}

TEST(Control, type_dependence_cycles) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto primary = control->newClassSymbol(nullptr, {});
  auto first = control->newClassSymbol(nullptr, {});
  auto second = control->newClassSymbol(nullptr, {});
  auto dependent = control->getTypeParameterType(0, 0, false);
  primary->addSpecialization(&unit, {second->type(), dependent}, first);
  primary->addSpecialization(&unit, {first->type()}, second);
  auto function = control->getFunctionType(control->getVoidType(),
                                           {first->type(), second->type()});
  EXPECT_TRUE(isDependent(&unit, function));
  EXPECT_TRUE(isDependent(&unit, second->type()));
}

TEST(Control, type_dependence_observes_symbol_changes) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto symbol = control->newClassSymbol(nullptr, {});
  EXPECT_FALSE(isDependent(&unit, symbol->type()));
  auto parameters = control->newTemplateParametersSymbol(nullptr, {});
  auto parameter = control->newTypeParameterSymbol(parameters, {}, 0, 0, false);
  parameters->addSymbol(parameter);
  symbol->setTemplateParameters(parameters);
  EXPECT_TRUE(isDependent(&unit, symbol->type()));
}

TEST(Control, constant_template_argument_keys) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto variable = control->newVariableSymbol(nullptr, {});
  variable->setType(control->getIntType());
  variable->setConstValue(ConstInt{42});
  auto alias = control->newTypeAliasSymbol(nullptr, {});
  alias->setType(control->getIntType());

  std::vector<TemplateArgument> symbolArguments{static_cast<Symbol*>(variable)};
  std::vector<TemplateArgument> valueArguments{ConstValue{ConstInt{42}}};
  std::vector<TemplateArgument> typeArguments{control->getIntType()};
  std::vector<TemplateArgument> aliasArguments{static_cast<Symbol*>(alias)};
  EXPECT_TRUE(compare_args(&unit, symbolArguments, valueArguments));
  EXPECT_FALSE(compare_args(&unit, symbolArguments, typeArguments));
  EXPECT_FALSE(compare_args(&unit, typeArguments, symbolArguments));
  EXPECT_FALSE(compare_args(&unit, symbolArguments, aliasArguments));
  EXPECT_FALSE(compare_args(&unit, aliasArguments, symbolArguments));
  ASSERT_TRUE(hash_template_arguments(symbolArguments).has_value());
  EXPECT_EQ(hash_template_arguments(symbolArguments),
            hash_template_arguments(valueArguments));

  auto primary = control->newClassSymbol(nullptr, {});
  auto instance = control->newClassSymbol(nullptr, {});
  primary->addSpecialization(&unit, symbolArguments, instance);
  EXPECT_EQ(primary->findSpecialization(&unit, valueArguments), instance);
  EXPECT_EQ(primary->findSpecialization(&unit, typeArguments), nullptr);

  auto otherPrimary = control->newClassSymbol(nullptr, {});
  otherPrimary->addSpecialization(&unit, valueArguments, instance);
  EXPECT_EQ(otherPrimary->findSpecialization(&unit, symbolArguments), instance);
}

TEST(Control, integer_template_argument_key_distribution) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  std::vector<std::size_t> hashes;
  for (int i = 0; i < 1024; ++i) {
    auto variable = control->newVariableSymbol(nullptr, {});
    variable->setType(control->getIntType());
    variable->setConstValue(ConstInt{i});
    std::vector<TemplateArgument> arguments{static_cast<Symbol*>(variable)};
    auto hash = hash_template_arguments(arguments);
    ASSERT_TRUE(hash.has_value());
    hashes.push_back(*hash);
  }
  std::ranges::sort(hashes);
  EXPECT_EQ(std::ranges::unique(hashes).size(), 0);
}

TEST(Control, dependent_template_argument_keys) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto identifier = control->getIdentifier("member_type");
  auto firstName = NameIdAST::create(unit.arena(), identifier);
  auto secondName = NameIdAST::create(unit.arena(), identifier);
  auto firstType = control->getUnresolvedNameType(&unit, nullptr, firstName);
  auto secondType = control->getUnresolvedNameType(&unit, nullptr, secondName);
  std::vector<TemplateArgument> first{ConstValue{ConstInt{1}}, firstType};
  std::vector<TemplateArgument> second{ConstValue{ConstInt{1}}, secondType};
  std::vector<TemplateArgument> third{ConstValue{ConstInt{2}}, secondType};
  EXPECT_TRUE(compare_args(&unit, first, second));
  ASSERT_TRUE(hash_template_arguments(first).has_value());
  EXPECT_EQ(hash_template_arguments(first), hash_template_arguments(second));
  EXPECT_NE(hash_template_arguments(first), hash_template_arguments(third));

  auto primary = control->newClassSymbol(nullptr, {});
  auto instance = control->newClassSymbol(nullptr, {});
  primary->addSpecialization(&unit, first, instance);
  EXPECT_EQ(primary->findSpecialization(&unit, second), instance);
  EXPECT_EQ(primary->findSpecialization(&unit, third), nullptr);

  auto otherName =
      NameIdAST::create(unit.arena(), control->getIdentifier("other"));
  auto otherType = control->getUnresolvedNameType(&unit, nullptr, otherName);
  std::vector<TemplateArgument> collision{ConstValue{ConstInt{1}}, otherType};
  EXPECT_EQ(hash_template_arguments(first), hash_template_arguments(collision));
  EXPECT_FALSE(compare_args(&unit, first, collision));
  EXPECT_EQ(primary->findSpecialization(&unit, collision), nullptr);
}

TEST(Control, shared_function_type_argument_graph) {
  Control control;
  const Type* type = control.getIntType();
  for (int i = 0; i < 40; ++i) {
    type = control.getPointerType(control.getFunctionType(type, {type, type}));
  }
  std::vector<TemplateArgument> arguments{type};
  EXPECT_TRUE(hash_template_arguments(arguments).has_value());
}

TEST(Control, constant_template_parameter_keys) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto first = control->newNonTypeParameterSymbol(nullptr, {});
  auto second = control->newNonTypeParameterSymbol(nullptr, {});
  first->setObjectType(control->getIntType());
  second->setObjectType(
      control->getQualType(control->getIntType(), CvQualifiers::kConst));
  std::vector<TemplateArgument> lhs{static_cast<Symbol*>(first)};
  std::vector<TemplateArgument> rhs{static_cast<Symbol*>(second)};
  EXPECT_TRUE(compare_args(&unit, lhs, rhs));
  ASSERT_TRUE(hash_template_arguments(lhs).has_value());
  EXPECT_EQ(hash_template_arguments(lhs), hash_template_arguments(rhs));
  second->setIndex(1);
  EXPECT_FALSE(compare_args(&unit, lhs, rhs));
  EXPECT_NE(hash_template_arguments(lhs), hash_template_arguments(rhs));
}

TEST(Control, shared_constant_initializer_dependence) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto arena = unit.arena();
  auto base = control->newVariableSymbol(nullptr, {});
  base->setType(control->getIntType());
  base->setConstexpr(true);
  base->setInitializer(IntLiteralExpressionAST::create(arena));
  auto variable = base;
  for (int i = 0; i < 40; ++i) {
    auto reference = IdExpressionAST::create(arena);
    reference->symbol = variable;
    reference->type = variable->type();
    auto initializer = BinaryExpressionAST::create(arena);
    initializer->leftExpression = reference;
    initializer->rightExpression = reference;
    initializer->op = TokenKind::T_PLUS;
    variable = control->newVariableSymbol(nullptr, {});
    variable->setType(control->getIntType());
    variable->setConstexpr(true);
    variable->setInitializer(initializer);
  }
  auto reference = IdExpressionAST::create(arena);
  reference->symbol = variable;
  reference->type = variable->type();
  EXPECT_FALSE(isDependent(&unit, reference));
  base->setType(control->getTypeParameterType(0, 0, false));
  EXPECT_TRUE(isDependent(&unit, reference));
  base->setType(control->getIntType());
  EXPECT_FALSE(isDependent(&unit, reference));
}

TEST(Control, specialization_symbol_index) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  SpecializationTable table;
  auto first = control->newClassSymbol(nullptr, {});
  EXPECT_EQ(table.entryOf(first), nullptr);
  EXPECT_EQ(table.add(&unit, {ConstValue{ConstInt{0}}}, first), 0);
  for (int i = 1; i < 1024; ++i) {
    auto instance = control->newClassSymbol(nullptr, {});
    EXPECT_EQ(table.add(&unit, {ConstValue{ConstInt{i}}}, instance), i);
    EXPECT_EQ(table.entryOf(instance), &table.entries().back());
  }
  EXPECT_EQ(table.add(&unit, {ConstValue{ConstInt{1024}}}, first), 1024);
  EXPECT_EQ(table.entryOf(first), &table.entries().front());

  SpecializationTable restored;
  for (const auto& entry : table.entries()) restored.restore(entry);
  restored.rebuildIndex();
  for (const auto& entry : restored.entries().first(1024))
    EXPECT_EQ(restored.entryOf(entry.symbol), &entry);
  EXPECT_EQ(restored.entryOf(nullptr), nullptr);
}

TEST(Control, restored_specialization_indexed_after_fixup) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto control = unit.control();
  auto zero = control->getConstantArgumentSymbol(control->getIntType(),
                                                 ConstValue{ConstInt{0}});
  auto specialization = control->newClassSymbol(nullptr, {});

  SpecializationTable restored;
  auto decodedPack = control->newParameterPackSymbol(nullptr, {});
  restored.restore({.arguments = {decodedPack}, .symbol = specialization});
  decodedPack->addElement(zero);
  restored.rebuildIndex();

  std::vector<Symbol*> elements{zero};
  std::vector<TemplateArgument> arguments{
      control->getPackArgumentSymbol(elements)};
  EXPECT_EQ(restored.find(&unit, arguments), specialization);
}
