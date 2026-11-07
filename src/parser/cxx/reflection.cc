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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/reflection.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <array>
#include <limits>

namespace cxx {
namespace {

[[nodiscard]] auto reflectedValue(const ConstValue& argument) -> const Meta* {
  auto value = std::get_if<std::shared_ptr<Meta>>(&argument);
  return value && *value ? value->get() : nullptr;
}

[[nodiscard]] auto isReflectionVector(const TypeTraits& traits,
                                      const Type* type) -> bool {
  auto vector = type_cast<ClassType>(type);
  if (!vector) return false;
  auto cls = vector->symbol();
  if (!traits.is_std_namespace(cls->parent()) || !cls->isSpecialization() ||
      cls->name() != traits.control()->getIdentifier("vector"))
    return false;
  auto arguments = template_arguments_of(cls);
  if (arguments.size() != 2 || template_argument_as_type(arguments.front()) !=
                                   traits.control()->getBuiltinMetaInfoType())
    return false;
  auto allocator =
      type_cast<ClassType>(template_argument_as_type(arguments.back()));
  if (!allocator) return false;
  auto symbol = allocator->symbol();
  if (!traits.is_std_namespace(symbol->parent()) ||
      !symbol->isSpecialization() ||
      symbol->name() != traits.control()->getIdentifier("allocator"))
    return false;
  auto allocatorArguments = template_arguments_of(symbol);
  return allocatorArguments.size() == 1 &&
         template_argument_as_type(allocatorArguments.front()) ==
             traits.control()->getBuiltinMetaInfoType();
}

[[nodiscard]] auto completeTraitType(TypeTraits& traits, const Type* type)
    -> bool {
  if (traits.is_unbounded_array(type)) return true;
  type = traits.remove_all_extents(type);
  return !traits.is_class_or_union(type) || traits.require_complete_class(type);
}

[[nodiscard]] auto reflectedName(TypeIdAST* typeId) -> NamedTypeSpecifierAST* {
  if (typeId->declarator || typeId->attributeList) return nullptr;
  auto specifiers = typeId->typeSpecifierList;
  if (!specifiers || specifiers->next) return nullptr;
  return ast_cast<NamedTypeSpecifierAST>(specifiers->value);
}

struct EntityType {
  [[nodiscard]] auto operator()(BaseClassSymbol* symbol) const -> const Type* {
    auto base = resolved_base_class(symbol);
    return base ? base->type() : nullptr;
  }

  [[nodiscard]] auto operator()(VariableSymbol* symbol) const -> const Type* {
    if (symbol->isTemplatePattern() || symbol->isStructuredBinding())
      return nullptr;
    return symbol->type();
  }

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> const Type* {
    if (symbol->isTemplatePattern() || symbol->isConstructor() ||
        symbol->isDestructor() || containsPlaceholderType(symbol->type()))
      return nullptr;
    return symbol->type();
  }

  [[nodiscard]] auto operator()(ParameterSymbol* symbol) const -> const Type* {
    return symbol->type();
  }

  [[nodiscard]] auto operator()(FieldSymbol* symbol) const -> const Type* {
    return symbol->type();
  }

  [[nodiscard]] auto operator()(EnumeratorSymbol* symbol) const -> const Type* {
    return symbol->type();
  }

  [[nodiscard]] auto operator()(Symbol*) const -> const Type* {
    return nullptr;
  }
};

[[nodiscard]] auto reflectedEntityType(const Meta& reflection,
                                       const TypeTraits& traits)
    -> const Type* {
  if (auto value = std::get_if<Meta::ConstExpr>(&reflection.value))
    return value->type;

  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value))
    return *object ? (*object)->objectType(traits) : nullptr;

  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value)) {
    auto type = type_cast<FunctionType>(parameter->function->type());
    if (type && parameter->index < type->parameterTypes().size())
      return type->parameterTypes()[parameter->index];
    return nullptr;
  }
  auto symbol = std::get_if<const Symbol*>(&reflection.value);
  if (!symbol || !*symbol) return nullptr;
  return visit(EntityType{}, const_cast<Symbol*>(*symbol));
}

[[nodiscard]] auto standardMetaType(TranslationUnit* unit,
                                    std::string_view name) -> const Type* {
  auto control = unit->control();
  auto standard = qualifiedLookupNamespace(
      unit->globalScope(), control->getIdentifier(WellKnownName::T_STD));
  if (!standard) return nullptr;
  auto meta = qualifiedLookupNamespace(
      standard, control->getIdentifier(WellKnownName::T_META));
  if (!meta) return nullptr;
  auto symbol = qualifiedTypeOnlyLookup(meta, control->getIdentifier(name));
  return symbol ? symbol->type() : nullptr;
}

[[nodiscard]] auto isTextView(const TypeTraits& traits, const Type* type,
                              bool utf8) -> bool {
  auto view = type_cast<ClassType>(type);
  if (!view) return false;
  auto symbol = view->symbol();
  if (!traits.is_std_namespace(symbol->parent()) ||
      !symbol->isSpecialization() ||
      symbol->name() != traits.control()->getIdentifier("basic_string_view"))
    return false;
  auto arguments = template_arguments_of(symbol);
  const Type* character = traits.control()->getCharType();
  if (utf8) character = traits.control()->getChar8Type();
  if (arguments.size() != 2 ||
      template_argument_as_type(arguments[0]) != character)
    return false;
  auto traitsType =
      type_cast<ClassType>(template_argument_as_type(arguments[1]));
  if (!traitsType) return false;
  auto characterTraits = traitsType->symbol();
  if (!traits.is_std_namespace(characterTraits->parent()) ||
      !characterTraits->isSpecialization() ||
      characterTraits->name() != traits.control()->getIdentifier("char_traits"))
    return false;
  auto parameters = template_arguments_of(characterTraits);
  return parameters.size() == 1 &&
         template_argument_as_type(parameters[0]) == character;
}

[[nodiscard]] auto accessContextInfo(ASTInterpreter* interpreter,
                                     const std::shared_ptr<ConstObject>& object,
                                     std::string_view name)
    -> std::optional<Meta> {
  if (!interpreter || !object) return std::nullopt;
  auto type = unqualified_cast<ClassType>(object->type());
  if (!type) return std::nullopt;
  auto control = interpreter->control();
  auto function = designatedFunction(
      qualifiedLookup(type->symbol(), control->getIdentifier(name)));
  if (!function) return std::nullopt;
  auto result = interpreter->evaluateCall(function, {}, object);
  auto value = result ? reflectedValue(*result) : nullptr;
  return value ? std::optional{*value} : std::nullopt;
}

[[nodiscard]] auto scopeEntity(const TypeTraits& traits, const Meta& reflection)
    -> std::optional<ScopeSymbol*> {
  Symbol* symbol = nullptr;
  if (auto type = std::get_if<const Type*>(&reflection.value)) {
    if (!*type) return static_cast<ScopeSymbol*>(nullptr);
    symbol = traits.entity_of(*type);
  } else if (auto value = std::get_if<const Symbol*>(&reflection.value)) {
    if (!*value) return static_cast<ScopeSymbol*>(nullptr);
    symbol = const_cast<Symbol*>(*value);
  }
  auto scope = symbol_cast<ScopeSymbol>(symbol);
  if (!scope ||
      (!scope->isClassOrNamespace() && !symbol_cast<FunctionSymbol>(scope)))
    return std::nullopt;
  return scope;
}

[[nodiscard]] auto contextResult(Control* control, Meta scope, Meta designating)
    -> ConstValue {
  auto result = std::make_shared<InitializerList>();
  auto type = control->getBuiltinMetaInfoType();
  result->append(std::make_shared<Meta>(std::move(scope)), type);
  result->append(std::make_shared<Meta>(std::move(designating)), type);
  return result;
}

struct OperatorDescription {
  TokenKind token;
  std::string_view identifier;
  std::string_view spelling;
};

constexpr OperatorDescription operators[] = {
    {TokenKind::T_NEW, "op_new", "new"},
    {TokenKind::T_DELETE, "op_delete", "delete"},
    {TokenKind::T_NEW_ARRAY, "op_array_new", "new[]"},
    {TokenKind::T_DELETE_ARRAY, "op_array_delete", "delete[]"},
    {TokenKind::T_CO_AWAIT, "op_co_await", "co_await"},
    {TokenKind::T_LPAREN, "op_parentheses", "()"},
    {TokenKind::T_LBRACKET, "op_square_brackets", "[]"},
    {TokenKind::T_MINUS_GREATER, "op_arrow", "->"},
    {TokenKind::T_MINUS_GREATER_STAR, "op_arrow_star", "->*"},
    {TokenKind::T_TILDE, "op_tilde", "~"},
    {TokenKind::T_EXCLAIM, "op_exclamation", "!"},
    {TokenKind::T_PLUS, "op_plus", "+"},
    {TokenKind::T_MINUS, "op_minus", "-"},
    {TokenKind::T_STAR, "op_star", "*"},
    {TokenKind::T_SLASH, "op_slash", "/"},
    {TokenKind::T_PERCENT, "op_percent", "%"},
    {TokenKind::T_CARET, "op_caret", "^"},
    {TokenKind::T_AMP, "op_ampersand", "&"},
    {TokenKind::T_EQUAL, "op_equals", "="},
    {TokenKind::T_BAR, "op_pipe", "|"},
    {TokenKind::T_COMMA, "op_comma", ","},
    {TokenKind::T_LESS, "op_less", "<"},
    {TokenKind::T_GREATER, "op_greater", ">"},
    {TokenKind::T_PLUS_EQUAL, "op_plus_equals", "+="},
    {TokenKind::T_MINUS_EQUAL, "op_minus_equals", "-="},
    {TokenKind::T_STAR_EQUAL, "op_star_equals", "*="},
    {TokenKind::T_SLASH_EQUAL, "op_slash_equals", "/="},
    {TokenKind::T_PERCENT_EQUAL, "op_percent_equals", "%="},
    {TokenKind::T_CARET_EQUAL, "op_caret_equals", "^="},
    {TokenKind::T_AMP_EQUAL, "op_ampersand_equals", "&="},
    {TokenKind::T_BAR_EQUAL, "op_pipe_equals", "|="},
    {TokenKind::T_LESS_LESS, "op_less_less", "<<"},
    {TokenKind::T_GREATER_GREATER, "op_greater_greater", ">>"},
    {TokenKind::T_LESS_LESS_EQUAL, "op_less_less_equals", "<<="},
    {TokenKind::T_GREATER_GREATER_EQUAL, "op_greater_greater_equals", ">>="},
    {TokenKind::T_EQUAL_EQUAL, "op_equals_equals", "=="},
    {TokenKind::T_EXCLAIM_EQUAL, "op_exclamation_equals", "!="},
    {TokenKind::T_LESS_EQUAL, "op_less_equals", "<="},
    {TokenKind::T_GREATER_EQUAL, "op_greater_equals", ">="},
    {TokenKind::T_LESS_EQUAL_GREATER, "op_spaceship", "<=>"},
    {TokenKind::T_AMP_AMP, "op_ampersand_ampersand", "&&"},
    {TokenKind::T_BAR_BAR, "op_pipe_pipe", "||"},
    {TokenKind::T_PLUS_PLUS, "op_plus_plus", "++"},
    {TokenKind::T_MINUS_MINUS, "op_minus_minus", "--"},
};

[[nodiscard]] auto operatorValue(Control* control, const Type* type,
                                 std::string_view name)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  auto scope = symbol_cast<ScopeSymbol>(TypeTraits::entity_of(type));
  if (!scope) return std::nullopt;
  auto symbol = qualifiedLookup(scope, control->getIdentifier(name));
  auto enumerator = symbol_cast<EnumeratorSymbol>(symbol);
  return enumerator ? enumerator->value() : std::nullopt;
}

[[nodiscard]] auto textResult(Control* control, std::string_view text,
                              bool utf8) -> ConstValue {
  auto spelling = quoteStringLiteral(text);
  auto literal = utf8 ? control->utf8StringLiteral("u8" + spelling)
                      : control->stringLiteral(spelling);
  const Type* character = control->getCharType();
  if (utf8) character = control->getChar8Type();
  auto result = std::make_shared<InitializerList>();
  result->append(ConstValue{literal},
                 control->getPointerType(
                     control->getQualType(character, CvQualifiers::kConst)));
  result->append(ConstValue{ConstInt{literal->charCount()}},
                 control->getSizeType());
  return result;
}

[[nodiscard]] auto displayString(const Meta& reflection) -> std::string {
  if (auto type = std::get_if<const Type*>(&reflection.value))
    return *type ? to_string(*type) : "null reflection";
  if (auto symbol = std::get_if<const Symbol*>(&reflection.value)) {
    if (!*symbol) return "null reflection";
    return to_string(const_cast<Symbol*>(*symbol));
  }
  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value))
    return std::format("parameter {} of {}", parameter->index,
                       to_string(parameter->function->name()));
  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value))
    return to_string(TemplateArgument{ConstValue{*object}});
  return to_string(
      TemplateArgument{std::get<Meta::ConstExpr>(reflection.value).value});
}

struct ReflectEntity {
  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> Meta {
    if (!symbol->name()) return Meta{symbol->type()};
    return Meta{static_cast<const Symbol*>(symbol->canonical())};
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> Meta {
    if (symbol->isTemplatePattern())
      return Meta{static_cast<const Symbol*>(symbol->canonical())};
    return Meta{symbol->type()};
  }

  template <class T>
  [[nodiscard]] auto operator()(T* symbol) const -> Meta {
    if (is_type(symbol)) return Meta{symbol->type()};
    return Meta{static_cast<const Symbol*>(symbol->canonical())};
  }
};

struct Reflect {
  TranslationUnit* unit;

  [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST*) const
      -> std::optional<ConstValue> {
    return std::make_shared<Meta>(
        static_cast<const Symbol*>(unit->globalScope()));
  }

  [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast) const
      -> std::optional<ConstValue> {
    if (!ast->symbol) return std::nullopt;
    return std::make_shared<Meta>(static_cast<const Symbol*>(ast->symbol));
  }

  [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast) const
      -> std::optional<ConstValue> {
    if (!ast->typeId || !ast->typeId->type) return std::nullopt;
    if (auto named = reflectedName(ast->typeId); named && named->symbol)
      return std::make_shared<Meta>(visit(ReflectEntity{}, named->symbol));
    return std::make_shared<Meta>(ast->typeId->type);
  }

  [[nodiscard]] auto operator()(ReflectExpressionAST* ast) const
      -> std::optional<ConstValue> {
    auto id = ast_cast<IdExpressionAST>(ast->expression);
    if (!id || !id->symbol) return std::nullopt;
    auto symbol = id->symbol;
    if (auto function = designatedFunction(symbol)) symbol = function;
    return std::make_shared<Meta>(visit(ReflectEntity{}, symbol));
  }

  template <class T>
  [[nodiscard]] auto operator()(T*) const -> std::optional<ConstValue> {
    return std::nullopt;
  }
};

[[nodiscard]] auto isResultProducer(WellKnownName name) -> bool {
  return name == WellKnownName::T_REFLECT_CONSTANT ||
         name == WellKnownName::T_REFLECT_OBJECT ||
         name == WellKnownName::T_REFLECT_FUNCTION;
}

[[nodiscard]] auto hasResultProducerSignature(FunctionSymbol* function,
                                              WellKnownName name,
                                              Control* control) -> bool {
  auto type = type_cast<FunctionType>(function->type());
  if (type->parameterTypes().size() != 1) return false;
  if (type->returnType() != control->getBuiltinMetaInfoType()) return false;
  auto primary = symbol_cast<FunctionSymbol>(primary_template_of(function));
  auto head = primary ? primary->templateParameters() : nullptr;
  if (!head || head->members().size() != 1) return false;
  auto parameter = symbol_cast<TypeParameterSymbol>(head->members().front());
  if (!parameter) return false;
  auto pattern = type_cast<FunctionType>(primary->type());
  if (!pattern || pattern->isVariadic()) return false;
  if (pattern->parameterTypes().size() != 1) return false;
  auto parameterType = pattern->parameterTypes().front();
  if (name == WellKnownName::T_REFLECT_CONSTANT)
    return parameterType == parameter->type();
  auto reference = type_cast<LvalueReferenceType>(parameterType);
  return reference && reference->elementType() == parameter->type();
}

[[nodiscard]] auto producerArgumentType(FunctionSymbol* function)
    -> const Type* {
  auto arguments = template_arguments_of(function);
  if (arguments.size() != 1) return nullptr;
  return template_argument_as_type(arguments.front());
}

[[nodiscard]] auto reflectConstantValue(TranslationUnit* unit, const Type* type,
                                        const ConstValue& argument,
                                        SourceLocation location,
                                        ScopeSymbol* scope)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  if (!type || traits.is_reference(type) || traits.remove_cv(type) != type)
    return std::nullopt;
  if (!traits.is_structural(type)) return std::nullopt;
  if (!traits.is_copy_constructible(type)) return std::nullopt;
  if (!traits.is_constant_template_argument(argument, type))
    return std::nullopt;
  if (!traits.is_class_or_union(type))
    return std::make_shared<Meta>(Meta::ConstExpr{type, argument});
  auto parameter = Substitution::initializeTemplateParameterObject(
      unit, type, argument, location, scope);
  if (!parameter) return std::nullopt;
  return std::make_shared<Meta>(std::make_shared<ConstAddress>(parameter));
}

[[nodiscard]] auto constantValueOf(const Meta& reflection)
    -> std::optional<ConstValue> {
  if (auto value = std::get_if<Meta::ConstExpr>(&reflection.value))
    return value->value;
  auto object = std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value);
  if (!object || !*object) return std::nullopt;
  auto variable = symbol_cast<VariableSymbol>((*object)->symbol());
  if (!variable || !variable->isTemplateParameterObject() ||
      !variable->constValue())
    return std::nullopt;
  return *variable->constValue();
}

[[nodiscard]] auto referredFunction(const ConstAddress& address)
    -> FunctionSymbol* {
  return symbol_cast<FunctionSymbol>(address.symbol());
}

[[nodiscard]] auto reflectReferent(TranslationUnit* unit,
                                   FunctionSymbol* function,
                                   const ConstValue& argument)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto signature = type_cast<FunctionType>(function->type());
  auto referenceType = signature->parameterTypes().front();
  if (!traits.is_lvalue_reference(referenceType)) return std::nullopt;
  if (!traits.is_constant_template_argument(argument, referenceType))
    return std::nullopt;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&argument);
  if (!address || !*address) return std::nullopt;
  auto referred = traits.remove_reference(referenceType);
  if (well_known_name(function->name()) == WellKnownName::T_REFLECT_OBJECT) {
    if (!traits.is_object(referred)) return std::nullopt;
    return std::make_shared<Meta>(*address);
  }
  if (!traits.is_function(referred)) return std::nullopt;
  auto target = referredFunction(**address);
  if (!target) return std::nullopt;
  return std::make_shared<Meta>(visit(ReflectEntity{}, target));
}

[[nodiscard]] auto boundReferenceAddress(TranslationUnit* unit, Symbol* symbol,
                                         ScopeSymbol* scope,
                                         ASTInterpreter* interpreter)
    -> std::shared_ptr<ConstAddress> {
  auto traits = unit->typeTraits();
  const auto activeInEvaluation =
      interpreter && interpreter->hasActiveReference(symbol);
  if (!activeInEvaluation &&
      !traits.is_usable_in_constant_expressions(symbol, scope))
    return nullptr;
  ASTInterpreter local{unit, scope};
  auto bound = (interpreter ? interpreter : &local)->variableAddress(symbol);
  if (!bound) return nullptr;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*bound);
  return address ? *address : nullptr;
}

[[nodiscard]] auto objectOf(TranslationUnit* unit, const Meta& reflection,
                            Symbol* symbol, ScopeSymbol* scope,
                            ASTInterpreter* interpreter)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  std::shared_ptr<ConstAddress> address;
  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value)) {
    address = *object;
  } else if (!traits.is_variable(symbol)) {
    return std::nullopt;
  } else if (traits.is_reference(symbol->type())) {
    address = boundReferenceAddress(unit, symbol, scope, interpreter);
  } else {
    address = std::make_shared<ConstAddress>(symbol);
  }
  if (!address || address->isNull() || address->lifetimeEnded())
    return std::nullopt;
  auto root = address->root() ? address->root() : address->symbol();
  if (!has_static_storage_duration(root)) return std::nullopt;
  auto type = address->objectType(traits);
  if (!type || !traits.is_object(type)) return std::nullopt;
  return std::make_shared<Meta>(address);
}

[[nodiscard]] auto constantArgumentReflection(TranslationUnit* unit,
                                              const TemplateArgument& argument)
    -> std::optional<Meta> {
  auto traits = unit->typeTraits();
  auto type = template_argument_type(argument);
  auto constant = template_argument_value(argument);
  if (!type || !constant) return std::nullopt;
  if (traits.is_reference(type)) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*constant);
    if (!address || !*address) return std::nullopt;
    if (!traits.is_function(traits.remove_reference(type)))
      return Meta{*address};
    auto target = referredFunction(**address);
    if (!target) return std::nullopt;
    return visit(ReflectEntity{}, target);
  }
  if (!traits.is_class_or_union(type))
    return Meta{Meta::ConstExpr{type, *constant}};
  auto entity = std::get_if<Symbol*>(&argument);
  auto object = entity ? symbol_cast<VariableSymbol>(*entity) : nullptr;
  if (!object || !object->isTemplateParameterObject()) return std::nullopt;
  return Meta{std::make_shared<ConstAddress>(object)};
}

[[nodiscard]] auto templateArgumentReflection(TranslationUnit* unit,
                                              const TemplateArgument& argument)
    -> std::optional<Meta> {
  if (auto entity = std::get_if<Symbol*>(&argument);
      entity && unit->typeTraits().is_template(*entity))
    return visit(ReflectEntity{}, *entity);
  if (auto type = template_argument_as_type(argument)) return Meta{type};
  return constantArgumentReflection(unit, argument);
}

[[nodiscard]] auto templateArgumentReflections(TranslationUnit* unit,
                                               Symbol* specialization)
    -> std::optional<ConstValue> {
  if (!unit->typeTraits().has_template_arguments(specialization))
    return std::nullopt;
  auto result = std::make_shared<InitializerList>();
  for (const auto& argument :
       expand_template_arguments(template_arguments_of(specialization))) {
    auto reflected = templateArgumentReflection(unit, argument);
    if (!reflected) return std::nullopt;
    result->append(std::make_shared<Meta>(std::move(*reflected)),
                   unit->control()->getBuiltinMetaInfoType());
  }
  return result;
}

[[nodiscard]] auto designatedDeclaration(const ConstAddress& address)
    -> Symbol* {
  auto symbol = address.symbol();
  if (!symbol || address.arrayType()) return nullptr;
  auto type = address.elementType();
  if (type && unqualified_type(type) != unqualified_type(symbol->type()))
    return nullptr;
  return symbol;
}

[[nodiscard]] auto reflectionList(TranslationUnit* unit,
                                  std::vector<Meta> reflections) -> ConstValue {
  auto result = std::make_shared<InitializerList>();
  for (auto& reflection : reflections)
    result->append(std::make_shared<Meta>(std::move(reflection)),
                   unit->control()->getBuiltinMetaInfoType());
  return result;
}

[[nodiscard]] auto entityList(TranslationUnit* unit,
                              const std::vector<Symbol*>& entities)
    -> ConstValue {
  std::vector<Meta> reflections;
  reflections.reserve(entities.size());
  for (auto entity : entities)
    reflections.push_back(visit(ReflectEntity{}, entity));
  return reflectionList(unit, std::move(reflections));
}

[[nodiscard]] auto scopeQuery(TranslationUnit* unit, WellKnownName builtin,
                              std::span<const ConstValue> arguments,
                              ScopeSymbol* evaluationScope)
    -> std::optional<ConstValue> {
  auto scope = Reflection::scope(evaluationScope);
  if (!scope || !arguments.empty()) return std::nullopt;
  if (builtin == WellKnownName::T_CURRENT)
    return contextResult(unit->control(), visit(ReflectEntity{}, scope),
                         Meta{static_cast<const Type*>(nullptr)});
  if (builtin == WellKnownName::T_CURRENT_FUNCTION &&
      !symbol_cast<FunctionSymbol>(scope))
    return std::nullopt;
  if (builtin == WellKnownName::T_CURRENT_CLASS &&
      !symbol_cast<ClassSymbol>(scope)) {
    auto function = symbol_cast<FunctionSymbol>(scope);
    if (!function || namespace_scope_of(function)) return std::nullopt;
    scope = function->enclosingClass();
  }
  if (builtin == WellKnownName::T_CURRENT_NAMESPACE &&
      !symbol_cast<NamespaceSymbol>(scope))
    scope = scope->enclosingNamespace();
  if (!scope) return std::nullopt;
  return std::make_shared<Meta>(visit(ReflectEntity{}, scope));
}

[[nodiscard]] auto accessContextVia(TranslationUnit* unit,
                                    const ConstValue& argument,
                                    ASTInterpreter* interpreter,
                                    const std::shared_ptr<ConstObject>& object)
    -> std::optional<ConstValue> {
  auto value = reflectedValue(argument);
  if (!value) return std::nullopt;
  auto traits = unit->typeTraits();
  auto designating = scopeEntity(traits, *value);
  if (!designating) return std::nullopt;
  if (*designating && (!symbol_cast<ClassSymbol>(*designating) ||
                       !traits.is_complete((*designating)->type())))
    return std::nullopt;
  auto scope = accessContextInfo(interpreter, object, "scope");
  if (!scope) return std::nullopt;
  return contextResult(unit->control(), *scope, *value);
}

[[nodiscard]] auto symbolOf(TranslationUnit* unit, const ConstValue& argument,
                            bool utf8) -> std::optional<ConstValue> {
  for (const auto& description : operators) {
    auto value =
        operatorValue(unit->control(), standardMetaType(unit, "operators"),
                      description.identifier);
    if (!value || !equivalent_values(*value, argument)) continue;
    return textResult(unit->control(), description.spelling, utf8);
  }
  return std::nullopt;
}

[[nodiscard]] auto operatorOf(TranslationUnit* unit, Symbol* symbol)
    -> std::optional<ConstValue> {
  auto function = symbol_cast<FunctionSymbol>(symbol);
  auto name = function ? name_cast<OperatorId>(function->name()) : nullptr;
  if (!name) return std::nullopt;
  for (const auto& description : operators)
    if (description.token == name->op())
      return operatorValue(unit->control(), standardMetaType(unit, "operators"),
                           description.identifier);
  return std::nullopt;
}

[[nodiscard]] auto isStringLiteral(const ConstValue& argument) -> ConstValue {
  if (auto literal = std::get_if<const StringLiteral*>(&argument))
    return *literal != nullptr;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&argument);
  if (!address || !*address || !(*address)->stringLiteral()) return false;
  auto offset = (*address)->offset();
  return offset >= 0 && static_cast<std::uintmax_t>(offset) <=
                            (*address)->stringLiteral()->charCount();
}

[[nodiscard]] auto isAccessContextQuery(WellKnownName builtin) -> bool {
  switch (builtin) {
    case WellKnownName::T_IS_ACCESSIBLE:
    case WellKnownName::T_HAS_INACCESSIBLE_NONSTATIC_DATA_MEMBERS:
    case WellKnownName::T_HAS_INACCESSIBLE_BASES:
    case WellKnownName::T_HAS_INACCESSIBLE_SUBOBJECTS:
    case WellKnownName::T_MEMBERS_OF:
    case WellKnownName::T_BASES_OF:
    case WellKnownName::T_STATIC_DATA_MEMBERS_OF:
    case WellKnownName::T_NONSTATIC_DATA_MEMBERS_OF:
    case WellKnownName::T_SUBOBJECTS_OF:
      return true;
    default:
      return false;
  }
}

struct DesignatedAccess {
  TypeTraits traits;
  ScopeSymbol* scope;
  ClassSymbol* designating;

  [[nodiscard]] auto operator()(Symbol* member) const -> std::optional<bool> {
    return traits.is_accessible_member(member, scope, designating);
  }
};

[[nodiscard]] auto membersOfQuery(TranslationUnit* unit, const Meta& reflection,
                                  Symbol* symbol,
                                  const DesignatedAccess& access,
                                  ScopeSymbol* evaluationScope,
                                  ASTInterpreter* interpreter)
    -> std::optional<ConstValue> {
  if (!interpreter || !interpreter->evaluationPoint()) return std::nullopt;
  auto traits = unit->typeTraits();
  auto point = interpreter->evaluationPoint();
  ScopeSymbol* owner = symbol_cast<NamespaceSymbol>(symbol);
  if (auto alias = symbol_cast<NamespaceAliasSymbol>(symbol))
    owner = alias->namespaceSymbol();
  if (auto type = unqualified_cast<ClassType>(Reflection::type(reflection))) {
    auto cls = type->symbol()->resolvedDefinition();
    if (!traits.require_complete_class(cls)) return std::nullopt;
    cls = cls->resolvedDefinition();
    if (!traits.class_is_complete_at(cls, evaluationScope, point))
      return std::nullopt;
    owner = cls;
  }
  if (!owner) return std::nullopt;
  std::vector<Symbol*> accessible;
  for (auto member : traits.members_of(owner, evaluationScope, point)) {
    auto allowed = access(member);
    if (!allowed) return std::nullopt;
    if (*allowed) accessible.push_back(member);
  }
  return entityList(unit, accessible);
}

[[nodiscard]] auto subobjectsQuery(TranslationUnit* unit, WellKnownName builtin,
                                   ClassSymbol* cls,
                                   const DesignatedAccess& access)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  const bool bases = builtin == WellKnownName::T_BASES_OF ||
                     builtin == WellKnownName::T_SUBOBJECTS_OF ||
                     builtin == WellKnownName::T_HAS_INACCESSIBLE_BASES ||
                     builtin == WellKnownName::T_HAS_INACCESSIBLE_SUBOBJECTS;
  const bool members = builtin != WellKnownName::T_BASES_OF &&
                       builtin != WellKnownName::T_HAS_INACCESSIBLE_BASES;
  const bool predicate =
      builtin == WellKnownName::T_HAS_INACCESSIBLE_BASES ||
      builtin == WellKnownName::T_HAS_INACCESSIBLE_SUBOBJECTS ||
      builtin == WellKnownName::T_HAS_INACCESSIBLE_NONSTATIC_DATA_MEMBERS;
  if (predicate && members && cls->isClosureType()) return std::nullopt;
  std::vector<Symbol*> candidates;
  if (bases)
    for (auto base : cls->baseClasses()) candidates.push_back(base);
  if (members) {
    auto fields = traits.data_members_of(
        cls, builtin == WellKnownName::T_STATIC_DATA_MEMBERS_OF);
    candidates.insert(candidates.end(), fields.begin(), fields.end());
  }
  std::vector<Meta> accessible;
  for (auto candidate : candidates) {
    auto allowed = access(candidate);
    if (!allowed) return std::nullopt;
    if (predicate && !*allowed) return true;
    if (!*allowed) continue;
    if (symbol_cast<BaseClassSymbol>(candidate))
      accessible.push_back(Meta{static_cast<const Symbol*>(candidate)});
    else
      accessible.push_back(visit(ReflectEntity{}, candidate));
  }
  if (predicate) return false;
  return reflectionList(unit, std::move(accessible));
}

[[nodiscard]] auto accessContextQuery(
    TranslationUnit* unit, WellKnownName builtin, const Meta& reflection,
    Symbol* symbol, const ConstValue& argument, ScopeSymbol* evaluationScope,
    ASTInterpreter* interpreter) -> std::optional<ConstValue> {
  auto context = std::get_if<std::shared_ptr<ConstObject>>(&argument);
  if (!context || !*context) return std::nullopt;
  auto scopeInfo = accessContextInfo(interpreter, *context, "scope");
  auto classInfo =
      accessContextInfo(interpreter, *context, "designating_class");
  if (!scopeInfo || !classInfo) return std::nullopt;
  auto traits = unit->typeTraits();
  auto scope = scopeEntity(traits, *scopeInfo);
  auto designating = scopeEntity(traits, *classInfo);
  if (!scope || !designating) return std::nullopt;
  auto naming = symbol_cast<ClassSymbol>(*designating);
  if (*designating && !naming) return std::nullopt;
  DesignatedAccess access{traits, *scope, naming};
  if (builtin == WellKnownName::T_IS_ACCESSIBLE) {
    auto result = access(symbol);
    return result ? std::optional<ConstValue>{*result} : std::nullopt;
  }
  if (builtin == WellKnownName::T_MEMBERS_OF)
    return membersOfQuery(unit, reflection, symbol, access, evaluationScope,
                          interpreter);
  auto type = unqualified_cast<ClassType>(Reflection::type(reflection));
  if (!type || !type->symbol()->resolvedDefinition()->isComplete())
    return std::nullopt;
  return subobjectsQuery(unit, builtin, type->symbol()->resolvedDefinition(),
                         access);
}

[[nodiscard]] auto entityPredicate(const TypeTraits& traits,
                                   WellKnownName builtin, Symbol* symbol)
    -> std::optional<bool> {
  switch (builtin) {
    case WellKnownName::T_IS_NAMESPACE:
      return traits.is_namespace(symbol);
    case WellKnownName::T_IS_NAMESPACE_ALIAS:
      return traits.is_namespace_alias(symbol);
    case WellKnownName::T_IS_VARIABLE:
      return traits.is_variable(symbol);
    case WellKnownName::T_IS_STRUCTURED_BINDING:
      return traits.is_structured_binding(symbol);
    case WellKnownName::T_IS_FUNCTION:
      return traits.is_function(symbol);
    case WellKnownName::T_IS_ENUMERATOR:
      return traits.is_enumerator(symbol);
    case WellKnownName::T_IS_TEMPLATE:
      return traits.is_template(symbol);
    case WellKnownName::T_IS_FUNCTION_TEMPLATE:
      return traits.is_function_template(symbol);
    case WellKnownName::T_IS_VARIABLE_TEMPLATE:
      return traits.is_variable_template(symbol);
    case WellKnownName::T_IS_CLASS_TEMPLATE:
      return traits.is_class_template(symbol);
    case WellKnownName::T_IS_ALIAS_TEMPLATE:
      return traits.is_alias_template(symbol);
    case WellKnownName::T_IS_CONCEPT:
      return traits.is_concept(symbol);
    case WellKnownName::T_IS_NONSTATIC_DATA_MEMBER:
      return traits.is_nonstatic_data_member(symbol);
    case WellKnownName::T_IS_BIT_FIELD:
      return traits.is_bit_field(symbol);
    case WellKnownName::T_IS_MUTABLE_MEMBER:
      return traits.is_mutable_member(symbol);
    case WellKnownName::T_HAS_DEFAULT_MEMBER_INITIALIZER:
      return traits.has_default_member_initializer(symbol);
    case WellKnownName::T_IS_VIRTUAL:
      return traits.is_virtual(symbol);
    case WellKnownName::T_IS_PURE_VIRTUAL:
      return traits.is_pure_virtual(symbol);
    case WellKnownName::T_IS_OVERRIDE:
      return traits.is_override(symbol);
    case WellKnownName::T_IS_FINAL:
      return traits.is_final(symbol);
    case WellKnownName::T_IS_DELETED:
      return traits.is_deleted(symbol);
    case WellKnownName::T_IS_DEFAULTED:
      return traits.is_defaulted(symbol);
    case WellKnownName::T_IS_USER_PROVIDED:
      return traits.is_user_provided(symbol);
    case WellKnownName::T_IS_USER_DECLARED:
      return traits.is_user_declared(symbol);
    case WellKnownName::T_IS_EXPLICIT:
      return traits.is_explicit(symbol);
    case WellKnownName::T_HAS_STATIC_STORAGE_DURATION:
      return traits.has_static_storage_duration(symbol);
    case WellKnownName::T_HAS_THREAD_STORAGE_DURATION:
      return traits.has_thread_storage_duration(symbol);
    case WellKnownName::T_HAS_AUTOMATIC_STORAGE_DURATION:
      return traits.has_automatic_storage_duration(symbol);
    case WellKnownName::T_IS_CONVERSION_FUNCTION:
      return traits.is_conversion_function(symbol);
    case WellKnownName::T_IS_OPERATOR_FUNCTION:
      return traits.is_operator_function(symbol);
    case WellKnownName::T_IS_LITERAL_OPERATOR:
      return traits.is_literal_operator(symbol);
    case WellKnownName::T_IS_SPECIAL_MEMBER_FUNCTION:
      return traits.is_special_member_function(symbol);
    case WellKnownName::T_IS_CONSTRUCTOR:
      return traits.is_constructor(symbol);
    case WellKnownName::T_IS_DEFAULT_CONSTRUCTOR:
      return traits.is_default_constructor(symbol);
    case WellKnownName::T_IS_COPY_CONSTRUCTOR:
      return traits.is_copy_constructor(symbol);
    case WellKnownName::T_IS_MOVE_CONSTRUCTOR:
      return traits.is_move_constructor(symbol);
    case WellKnownName::T_IS_ASSIGNMENT:
      return traits.is_assignment(symbol);
    case WellKnownName::T_IS_COPY_ASSIGNMENT:
      return traits.is_copy_assignment(symbol);
    case WellKnownName::T_IS_MOVE_ASSIGNMENT:
      return traits.is_move_assignment(symbol);
    case WellKnownName::T_IS_DESTRUCTOR:
      return traits.is_destructor(symbol);
    case WellKnownName::T_IS_CONVERSION_FUNCTION_TEMPLATE:
      return traits.is_conversion_function_template(symbol);
    case WellKnownName::T_IS_OPERATOR_FUNCTION_TEMPLATE:
      return traits.is_operator_function_template(symbol);
    case WellKnownName::T_IS_LITERAL_OPERATOR_TEMPLATE:
      return traits.is_literal_operator_template(symbol);
    case WellKnownName::T_IS_CONSTRUCTOR_TEMPLATE:
      return traits.is_constructor_template(symbol);
    case WellKnownName::T_IS_STATIC_MEMBER:
      return traits.is_static_member(symbol);
    case WellKnownName::T_IS_PUBLIC:
      return traits.is_public(symbol);
    case WellKnownName::T_IS_PROTECTED:
      return traits.is_protected(symbol);
    case WellKnownName::T_IS_PRIVATE:
      return traits.is_private(symbol);
    case WellKnownName::T_IS_CLASS_MEMBER:
      return is_class_member(symbol);
    case WellKnownName::T_IS_NAMESPACE_MEMBER:
      return traits.is_namespace_member(symbol);
    case WellKnownName::T_IS_BASE:
      return traits.is_base(symbol);
    case WellKnownName::T_HAS_INTERNAL_LINKAGE:
      return traits.has_internal_linkage(symbol);
    case WellKnownName::T_HAS_MODULE_LINKAGE:
      return traits.has_module_linkage(symbol);
    case WellKnownName::T_HAS_EXTERNAL_LINKAGE:
      return traits.has_external_linkage(symbol);
    case WellKnownName::T_HAS_C_LANGUAGE_LINKAGE:
      return traits.has_c_language_linkage(symbol);
    case WellKnownName::T_HAS_LINKAGE:
      return traits.has_linkage(symbol);
    default:
      return std::nullopt;
  }
}

[[nodiscard]] auto parameterIdentifier(const Meta::Parameter& parameter,
                                       SourceLocation point)
    -> std::optional<const Name*> {
  std::vector<FunctionSymbol*> declarations;
  auto function = const_cast<FunctionSymbol*>(parameter.function);
  for (auto declaration : function->declarations())
    declarations.push_back(declaration);
  if (auto primary =
          symbol_cast<FunctionSymbol>(primary_template_of(function))) {
    std::uint32_t position = 0;
    for (auto candidate : primary->parameters()) {
      if (position++ > parameter.index) break;
      if (candidate->isParameterPack()) return std::nullopt;
    }
    for (auto declaration : primary->declarations())
      declarations.push_back(declaration);
  }
  const Name* name = nullptr;
  for (auto declaration : declarations) {
    if (point && !(declaration->location() < point)) continue;
    auto parameters = declaration->parameters();
    if (parameter.index >= parameters.size()) continue;
    auto introduced = parameters[parameter.index]->name();
    if (!introduced) continue;
    if (name && name != introduced) return std::nullopt;
    name = introduced;
  }
  if (!name) return std::nullopt;
  return name;
}

[[nodiscard]] auto identifierOf(const TypeTraits& traits,
                                const Meta& reflection, Symbol* symbol,
                                SourceLocation point)
    -> std::optional<std::string> {
  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value)) {
    auto name = parameterIdentifier(*parameter, point);
    return name ? std::optional{to_string(*name)} : std::nullopt;
  }
  if (auto type = std::get_if<const Type*>(&reflection.value);
      type && *type && unqualified_type(*type) != *type)
    return std::nullopt;
  if (std::holds_alternative<Meta::ConstExpr>(reflection.value))
    return std::nullopt;
  if (std::holds_alternative<std::shared_ptr<ConstAddress>>(reflection.value))
    return std::nullopt;
  if (!traits.has_identifier(symbol)) return std::nullopt;
  return traits.identifier_of(symbol);
}

[[nodiscard]] auto sourceLocationOf(TranslationUnit* unit,
                                    ASTInterpreter* interpreter,
                                    FunctionSymbol* function,
                                    const Meta& reflection, Symbol* symbol)
    -> std::optional<ConstValue> {
  auto signature = type_cast<FunctionType>(function->type());
  auto result = type_cast<ClassType>(signature->returnType());
  if (!interpreter || !result) return std::nullopt;
  SourceLocation location;
  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value)) {
    location = parameter->function->location();
  } else if (!std::holds_alternative<Meta::ConstExpr>(reflection.value) &&
             symbol && symbol != unit->globalScope()) {
    location = symbol->location();
  }
  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value);
      object && *object && (*object)->root())
    location = (*object)->root()->location();
  if (!location) return std::make_shared<InitializerList>();
  auto current = designatedFunction(qualifiedLookup(
      result->symbol(), unit->control()->getIdentifier("current")));
  auto currentType =
      current ? type_cast<FunctionType>(current->type()) : nullptr;
  if (!currentType || currentType->parameterTypes().size() != 1)
    return std::nullopt;
  auto pointer = interpreter->sourceLocationAddress(
      currentType->parameterTypes().front(), location, nullptr);
  if (!pointer) return std::nullopt;
  return interpreter->evaluateCall(current, {*pointer}, nullptr);
}

[[nodiscard]] auto parentOf(const TypeTraits& traits, const Meta& reflection,
                            Symbol* symbol) -> std::optional<ConstValue> {
  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value))
    return std::make_shared<Meta>(visit(
        ReflectEntity{}, const_cast<FunctionSymbol*>(parameter->function)));
  auto parent = traits.parent_of(symbol);
  if (!parent) return std::nullopt;
  return std::make_shared<Meta>(visit(ReflectEntity{}, parent));
}

[[nodiscard]] auto offsetOf(TranslationUnit* unit, Symbol* symbol)
    -> std::optional<ConstValue> {
  if (!symbol) return std::nullopt;
  auto traits = unit->typeTraits();
  auto bits = traits.subobject_offset(symbol);
  if (!bits ||
      *bits / 8 > std::uint64_t(std::numeric_limits<std::intmax_t>::max()))
    return std::nullopt;
  auto cls = type_cast<ClassType>(standardMetaType(unit, "member_offset"));
  if (!cls) return std::nullopt;
  auto members = traits.aggregate_elements(cls->symbol());
  if (members.size() != 2) return std::nullopt;
  auto result = std::make_shared<InitializerList>();
  result->append(ConstValue{ConstInt{*bits / 8}}, members[0]->type());
  result->append(ConstValue{ConstInt{*bits % 8}}, members[1]->type());
  return result;
}

[[nodiscard]] auto bitFieldLayout(const FieldSymbol* field,
                                  WellKnownName builtin)
    -> std::optional<ConstValue> {
  if (builtin != WellKnownName::T_BIT_SIZE_OF || !field->bitFieldWidth())
    return std::nullopt;
  auto width = std::get_if<ConstInt>(&*field->bitFieldWidth());
  if (!width) return std::nullopt;
  return ConstInt{width->toUIntMax()};
}

[[nodiscard]] auto alignmentOfEntity(TranslationUnit* unit, const Type* type,
                                     Symbol* symbol)
    -> std::optional<ConstValue> {
  if (auto field = symbol_cast<FieldSymbol>(symbol))
    return ConstInt{std::uintmax_t(
        field->isStatic() ? field->alignment() : field->effectiveAlignment())};
  auto alignment = unit->control()->memoryLayout()->alignmentOf(type);
  if (!alignment) return std::nullopt;
  if (auto variable = symbol_cast<VariableSymbol>(symbol))
    *alignment =
        std::max(*alignment, std::size_t(variable->explicitAlignment()));
  return ConstInt{std::uintmax_t(*alignment)};
}

[[nodiscard]] auto layoutQuery(TranslationUnit* unit, WellKnownName builtin,
                               const Meta& reflection, Symbol* symbol)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto type = Reflection::type(reflection);
  const bool representsType = type != nullptr;
  auto object = std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value);
  if (object && *object) symbol = designatedDeclaration(**object);
  if (!type) type = traits.layout_type(symbol);
  const bool isValue =
      std::holds_alternative<Meta::ConstExpr>(reflection.value);
  const bool hasValueLayout =
      builtin != WellKnownName::T_ALIGNMENT_OF && isValue;
  if (!type && (object || hasValueLayout))
    type = reflectedEntityType(reflection, traits);
  if (!type) return std::nullopt;
  if (traits.is_reference(type)) {
    if (!representsType && !symbol_cast<FieldSymbol>(symbol))
      return std::nullopt;
    type = traits.add_pointer(traits.remove_reference(type));
  }
  if (traits.is_function(type) || !traits.is_complete(type))
    return std::nullopt;
  if (auto field = symbol_cast<FieldSymbol>(symbol);
      field && field->isBitField())
    return bitFieldLayout(field, builtin);
  if (builtin == WellKnownName::T_ALIGNMENT_OF)
    return alignmentOfEntity(unit, type, symbol);
  auto size = unit->control()->memoryLayout()->sizeOf(type);
  if (!size) return std::nullopt;
  auto sizeType = unit->control()->getSizeType();
  auto bytes = traits.integral_constant(sizeType, *size);
  if (!bytes) return std::nullopt;
  if (builtin != WellKnownName::T_BIT_SIZE_OF) return *bytes;
  auto bitsPerByte = traits.integral_constant(sizeType, 8);
  if (!bitsPerByte) return std::nullopt;
  return *bytes * *bitsPerByte;
}

[[nodiscard]] auto variableOf(const Meta& reflection,
                              ScopeSymbol* evaluationScope)
    -> std::optional<ConstValue> {
  auto parameter = std::get_if<Meta::Parameter>(&reflection.value);
  if (!parameter) return std::nullopt;
  while (symbol_cast<BlockSymbol>(evaluationScope))
    evaluationScope = evaluationScope->parent();
  auto parameters = symbol_cast<FunctionParametersSymbol>(evaluationScope);
  if (!parameters) return std::nullopt;
  auto function = symbol_cast<FunctionSymbol>(parameters->parent());
  if (!function || function->canonical() != parameter->function)
    return std::nullopt;
  std::uint32_t index = 0;
  for (auto variable : views::members(parameters) | views::parameters) {
    if (index++ == parameter->index) return std::make_shared<Meta>(variable);
  }
  return std::nullopt;
}

[[nodiscard]] auto hasDefaultArgument(const Meta& reflection) -> bool {
  auto parameter = std::get_if<Meta::Parameter>(&reflection.value);
  if (!parameter) return false;
  for (auto declaration : parameter->function->declarations()) {
    auto parameters = declaration->parameters();
    if (parameter->index >= parameters.size()) continue;
    if (parameters[parameter->index]->hasDefaultArgument()) return true;
  }
  return false;
}

[[nodiscard]] auto isExplicitObjectParameter(const Meta& reflection) -> bool {
  auto parameter = std::get_if<Meta::Parameter>(&reflection.value);
  return parameter && parameter->index == 0 &&
         parameter->function->hasExplicitObjectParameter();
}

[[nodiscard]] auto qualifierQuery(const TypeTraits& traits,
                                  WellKnownName builtin, const Meta& reflection)
    -> bool {
  auto type = reflectedEntityType(reflection, traits);
  if (!type) type = Reflection::type(reflection);
  if (!type) return false;
  if (auto function = type_cast<FunctionType>(type)) {
    switch (builtin) {
      case WellKnownName::T_IS_CONST:
        return has_const(function->cvQualifiers());
      case WellKnownName::T_IS_VOLATILE:
        return has_volatile(function->cvQualifiers());
      case WellKnownName::T_IS_LVALUE_REFERENCE_QUALIFIED:
        return function->refQualifier() == RefQualifier::kLvalue;
      default:
        return function->refQualifier() == RefQualifier::kRvalue;
    }
  }
  if (builtin == WellKnownName::T_IS_CONST) return traits.is_const(type);
  if (builtin == WellKnownName::T_IS_VOLATILE) return traits.is_volatile(type);
  return false;
}

[[nodiscard]] auto functionTypeQuery(TranslationUnit* unit,
                                     WellKnownName builtin,
                                     const Meta& reflection, Symbol* symbol)
    -> bool {
  auto reflectedType = std::get_if<const Type*>(&reflection.value);
  const Type* type = reflectedType ? *reflectedType : nullptr;
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (function->isTemplatePattern()) return false;
    if (builtin == WellKnownName::T_IS_NOEXCEPT) {
      ASTRewriter::completePendingExceptionSpecification(unit, function);
      Binder{unit}.completeDeferredImplicitExceptionSpecification(function);
    }
    type = function->type();
  }
  auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return false;
  if (builtin == WellKnownName::T_IS_NOEXCEPT)
    return functionType->isNoexcept();
  return functionType->isVariadic();
}

[[nodiscard]] auto typeOf(const TypeTraits& traits, WellKnownName builtin,
                          const Meta& reflection) -> std::optional<ConstValue> {
  auto type = reflectedEntityType(reflection, traits);
  if (builtin == WellKnownName::T_RETURN_TYPE_OF) {
    if (!type) type = Reflection::type(reflection);
    auto functionType = type_cast<FunctionType>(type);
    if (!functionType) return std::nullopt;
    type = functionType->returnType();
  }
  if (!type) return std::nullopt;
  return std::make_shared<Meta>(type);
}

[[nodiscard]] auto baseRelation(TypeTraits& traits, WellKnownName builtin,
                                const Type* base, const Type* derived)
    -> std::optional<ConstValue> {
  if (!traits.is_class(base) || !traits.is_class(derived)) return false;
  if (builtin != WellKnownName::T_IS_VIRTUAL_BASE_OF_TYPE &&
      traits.is_same(traits.remove_cv(base), traits.remove_cv(derived)))
    return true;
  if (!traits.require_complete_class(derived)) return std::nullopt;
  if (builtin == WellKnownName::T_IS_POINTER_INTERCONVERTIBLE_BASE_OF_TYPE)
    return traits.is_pointer_interconvertible_base_of(base, derived);
  if (builtin == WellKnownName::T_IS_VIRTUAL_BASE_OF_TYPE)
    return traits.is_virtual_base_of(base, derived);
  return traits.is_base_of(base, derived);
}

[[nodiscard]] auto typeRelation(TypeTraits& traits, WellKnownName builtin,
                                const Type* type, const ConstValue& argument,
                                ASTInterpreter* interpreter)
    -> std::optional<ConstValue> {
  if (builtin == WellKnownName::T_EXTENT) {
    auto index = std::get_if<ConstInt>(&argument);
    if (!index) return std::nullopt;
    return ConstInt{traits.extent(type, index->toUIntMax())};
  }
  auto otherValue = reflectedValue(argument);
  auto other = otherValue ? Reflection::type(*otherValue) : nullptr;
  if (!other) return std::nullopt;
  switch (builtin) {
    case WellKnownName::T_IS_SAME_TYPE:
      return traits.is_same(type, other);
    case WellKnownName::T_IS_BASE_OF_TYPE:
    case WellKnownName::T_IS_VIRTUAL_BASE_OF_TYPE:
    case WellKnownName::T_IS_POINTER_INTERCONVERTIBLE_BASE_OF_TYPE:
      return baseRelation(traits, builtin, type, other);
    default:
      break;
  }
  if (!completeTraitType(traits, type) || !completeTraitType(traits, other)) {
    if (interpreter) interpreter->abortEvaluation();
    return std::nullopt;
  }
  switch (builtin) {
    case WellKnownName::T_IS_LAYOUT_COMPATIBLE_TYPE:
      return traits.is_layout_compatible(type, other);
    case WellKnownName::T_IS_SWAPPABLE_WITH_TYPE:
      return traits.is_swappable_with(type, other);
    case WellKnownName::T_IS_NOTHROW_SWAPPABLE_WITH_TYPE:
      return traits.is_nothrow_swappable_with(type, other);
    case WellKnownName::T_IS_CONVERTIBLE_TYPE:
      return traits.is_convertible(type, other);
    case WellKnownName::T_IS_NOTHROW_CONVERTIBLE_TYPE:
      return traits.is_nothrow_convertible(type, other);
    case WellKnownName::T_IS_ASSIGNABLE_TYPE:
      return traits.is_assignable(type, other);
    case WellKnownName::T_IS_TRIVIALLY_ASSIGNABLE_TYPE:
      return traits.is_trivially_assignable(type, other);
    case WellKnownName::T_IS_NOTHROW_ASSIGNABLE_TYPE:
      return traits.is_nothrow_assignable(type, other);
    case WellKnownName::T_REFERENCE_CONSTRUCTS_FROM_TEMPORARY:
      return traits.reference_constructs_from_temporary(type, other);
    case WellKnownName::T_REFERENCE_CONVERTS_FROM_TEMPORARY:
      return traits.reference_converts_from_temporary(type, other);
    default:
      return std::nullopt;
  }
}

[[nodiscard]] auto hasCompleteOperand(TypeTraits& traits, WellKnownName builtin,
                                      const Type* type) -> bool {
  switch (builtin) {
    case WellKnownName::T_IS_TRIVIALLY_COPYABLE_TYPE:
    case WellKnownName::T_IS_STANDARD_LAYOUT_TYPE:
    case WellKnownName::T_HAS_UNIQUE_OBJECT_REPRESENTATIONS:
    case WellKnownName::T_IS_STRUCTURAL_TYPE:
      return completeTraitType(traits, traits.remove_all_extents(type));
    case WellKnownName::T_IS_EMPTY_TYPE:
    case WellKnownName::T_IS_POLYMORPHIC_TYPE:
    case WellKnownName::T_IS_ABSTRACT_TYPE:
    case WellKnownName::T_HAS_VIRTUAL_DESTRUCTOR:
      if (traits.is_union(type)) return true;
      [[fallthrough]];
    case WellKnownName::T_IS_FINAL_TYPE:
    case WellKnownName::T_IS_AGGREGATE_TYPE:
    case WellKnownName::T_IS_IMPLICIT_LIFETIME_TYPE:
      return !traits.is_class_or_union(type) ||
             traits.require_complete_class(type);
    case WellKnownName::T_IS_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_SWAPPABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_SWAPPABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_MOVE_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_MOVE_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_MOVE_ASSIGNABLE_TYPE:
      return completeTraitType(traits, type);
    default:
      return true;
  }
}

[[nodiscard]] auto typeProperty(TypeTraits& traits, WellKnownName builtin,
                                const Type* type) -> std::optional<bool> {
  switch (builtin) {
    case WellKnownName::T_IS_EMPTY_TYPE:
      return !traits.is_union(type) && traits.is_empty(type);
    case WellKnownName::T_IS_POLYMORPHIC_TYPE:
      return !traits.is_union(type) && traits.is_polymorphic(type);
    case WellKnownName::T_IS_ABSTRACT_TYPE:
      return !traits.is_union(type) && traits.is_abstract(type);
    case WellKnownName::T_HAS_VIRTUAL_DESTRUCTOR:
      return !traits.is_union(type) && traits.has_virtual_destructor(type);
    case WellKnownName::T_IS_FINAL_TYPE:
      return traits.is_final(type);
    case WellKnownName::T_IS_SWAPPABLE_TYPE:
      return traits.is_swappable(type);
    case WellKnownName::T_IS_NOTHROW_SWAPPABLE_TYPE:
      return traits.is_nothrow_swappable(type);
    case WellKnownName::T_IS_DEFAULT_CONSTRUCTIBLE_TYPE:
      return traits.is_default_constructible(type);
    case WellKnownName::T_IS_COPY_CONSTRUCTIBLE_TYPE:
      return traits.is_copy_constructible(type);
    case WellKnownName::T_IS_MOVE_CONSTRUCTIBLE_TYPE:
      return traits.is_move_constructible(type);
    case WellKnownName::T_IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE_TYPE:
      return traits.is_trivially_default_constructible(type);
    case WellKnownName::T_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_TYPE:
      return traits.is_trivially_copy_constructible(type);
    case WellKnownName::T_IS_TRIVIALLY_MOVE_CONSTRUCTIBLE_TYPE:
      return traits.is_trivially_move_constructible(type);
    case WellKnownName::T_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE:
      return traits.is_nothrow_default_constructible(type);
    case WellKnownName::T_IS_NOTHROW_COPY_CONSTRUCTIBLE_TYPE:
      return traits.is_nothrow_copy_constructible(type);
    case WellKnownName::T_IS_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE:
      return traits.is_nothrow_move_constructible(type);
    case WellKnownName::T_IS_COPY_ASSIGNABLE_TYPE:
      return traits.is_copy_assignable(type);
    case WellKnownName::T_IS_MOVE_ASSIGNABLE_TYPE:
      return traits.is_move_assignable(type);
    case WellKnownName::T_IS_TRIVIALLY_COPY_ASSIGNABLE_TYPE:
      return traits.is_trivially_copy_assignable(type);
    case WellKnownName::T_IS_TRIVIALLY_MOVE_ASSIGNABLE_TYPE:
      return traits.is_trivially_move_assignable(type);
    case WellKnownName::T_IS_NOTHROW_COPY_ASSIGNABLE_TYPE:
      return traits.is_nothrow_copy_assignable(type);
    case WellKnownName::T_IS_NOTHROW_MOVE_ASSIGNABLE_TYPE:
      return traits.is_nothrow_move_assignable(type);
    case WellKnownName::T_IS_TRIVIALLY_COPYABLE_TYPE:
      return traits.is_trivially_copyable(type);
    case WellKnownName::T_HAS_UNIQUE_OBJECT_REPRESENTATIONS:
      return traits.has_unique_object_representations(type);
    case WellKnownName::T_IS_STANDARD_LAYOUT_TYPE:
      return traits.is_standard_layout(type);
    case WellKnownName::T_IS_AGGREGATE_TYPE:
      return traits.is_aggregate(type);
    case WellKnownName::T_IS_STRUCTURAL_TYPE:
      return traits.is_structural(type);
    case WellKnownName::T_IS_IMPLICIT_LIFETIME_TYPE:
      return traits.is_implicit_lifetime(type);
    case WellKnownName::T_IS_SIGNED_TYPE:
      return traits.is_signed(type);
    case WellKnownName::T_IS_UNSIGNED_TYPE:
      return traits.is_unsigned(type);
    case WellKnownName::T_IS_BOUNDED_ARRAY_TYPE:
      return traits.is_bounded_array(type);
    case WellKnownName::T_IS_UNBOUNDED_ARRAY_TYPE:
      return traits.is_unbounded_array(type);
    case WellKnownName::T_IS_SCOPED_ENUM_TYPE:
      return traits.is_scoped_enum(type);
    case WellKnownName::T_IS_DESTRUCTIBLE_TYPE:
      return traits.is_destructible(type);
    case WellKnownName::T_IS_TRIVIALLY_DESTRUCTIBLE_TYPE:
      return traits.is_trivially_destructible(type);
    case WellKnownName::T_IS_NOTHROW_DESTRUCTIBLE_TYPE:
      return traits.is_nothrow_destructible(type);
    case WellKnownName::T_IS_VOID_TYPE:
      return traits.is_void(type);
    case WellKnownName::T_IS_NULL_POINTER_TYPE:
      return traits.is_null_pointer(type);
    case WellKnownName::T_IS_INTEGRAL_TYPE:
      return traits.is_integral(type);
    case WellKnownName::T_IS_FLOATING_POINT_TYPE:
      return traits.is_floating_point(type);
    case WellKnownName::T_IS_ARRAY_TYPE:
      return traits.is_array(type);
    case WellKnownName::T_IS_POINTER_TYPE:
      return traits.is_pointer(type);
    case WellKnownName::T_IS_LVALUE_REFERENCE_TYPE:
      return traits.is_lvalue_reference(type);
    case WellKnownName::T_IS_RVALUE_REFERENCE_TYPE:
      return traits.is_rvalue_reference(type);
    case WellKnownName::T_IS_MEMBER_OBJECT_POINTER_TYPE:
      return traits.is_member_object_pointer(type);
    case WellKnownName::T_IS_MEMBER_FUNCTION_POINTER_TYPE:
      return traits.is_member_function_pointer(type);
    case WellKnownName::T_IS_ENUM_TYPE:
      return traits.is_enum(type);
    case WellKnownName::T_IS_UNION_TYPE:
      return traits.is_union(type);
    case WellKnownName::T_IS_CLASS_TYPE:
      return traits.is_class(type);
    case WellKnownName::T_IS_FUNCTION_TYPE:
      return traits.is_function(type);
    case WellKnownName::T_IS_REFLECTION_TYPE:
      return traits.is_reflection(type);
    case WellKnownName::T_IS_REFERENCE_TYPE:
      return traits.is_reference(type);
    case WellKnownName::T_IS_ARITHMETIC_TYPE:
      return traits.is_arithmetic(type);
    case WellKnownName::T_IS_FUNDAMENTAL_TYPE:
      return traits.is_fundamental(type);
    case WellKnownName::T_IS_OBJECT_TYPE:
      return traits.is_object(type);
    case WellKnownName::T_IS_SCALAR_TYPE:
      return traits.is_scalar(type);
    case WellKnownName::T_IS_COMPOUND_TYPE:
      return traits.is_compound(type);
    case WellKnownName::T_IS_MEMBER_POINTER_TYPE:
      return traits.is_member_pointer(type);
    case WellKnownName::T_IS_CONST_TYPE:
      return traits.is_const(type);
    case WellKnownName::T_IS_VOLATILE_TYPE:
      return traits.is_volatile(type);
    default:
      return std::nullopt;
  }
}

[[nodiscard]] auto typeTransformation(const TypeTraits& traits,
                                      WellKnownName builtin, const Type* type)
    -> const Type* {
  switch (builtin) {
    case WellKnownName::T_ADD_CV:
      return traits.add_cv(type, CvQualifiers::kConstVolatile);
    case WellKnownName::T_MAKE_SIGNED:
    case WellKnownName::T_MAKE_UNSIGNED: {
      if (!traits.is_integral_or_enum(type) || traits.is_bool(type))
        return nullptr;
      auto result = builtin == WellKnownName::T_MAKE_SIGNED
                        ? traits.make_signed(type)
                        : traits.make_unsigned(type);
      return traits.is_integral(result) ? result : nullptr;
    }
    case WellKnownName::T_UNDERLYING_TYPE: {
      if (!traits.is_enum(type)) return nullptr;
      auto underlying = traits.underlying_type(type);
      return traits.is_enum(underlying) ? nullptr : underlying;
    }
    case WellKnownName::T_REMOVE_CONST:
      return traits.remove_const(type);
    case WellKnownName::T_REMOVE_VOLATILE:
      return traits.remove_volatile(type);
    case WellKnownName::T_REMOVE_CV:
      return traits.remove_cv(type);
    case WellKnownName::T_ADD_CONST:
      return traits.add_const(type);
    case WellKnownName::T_ADD_VOLATILE:
      return traits.add_volatile(type);
    case WellKnownName::T_REMOVE_REFERENCE:
      return traits.remove_reference(type);
    case WellKnownName::T_ADD_LVALUE_REFERENCE:
      return traits.add_lvalue_reference(type);
    case WellKnownName::T_ADD_RVALUE_REFERENCE:
      return traits.add_rvalue_reference(type);
    case WellKnownName::T_REMOVE_EXTENT:
      return traits.remove_extent(type);
    case WellKnownName::T_REMOVE_ALL_EXTENTS:
      return traits.remove_all_extents(type);
    case WellKnownName::T_REMOVE_POINTER:
      return traits.remove_pointer(type);
    case WellKnownName::T_ADD_POINTER:
      return traits.add_pointer(type);
    case WellKnownName::T_REMOVE_CVREF:
      return traits.remove_cvref(type);
    case WellKnownName::T_DECAY:
      return traits.decay(type);
    case WellKnownName::T_UNWRAP_REFERENCE:
      return traits.unwrap_reference(type);
    case WellKnownName::T_UNWRAP_REF_DECAY:
      return traits.unwrap_ref_decay(type);
    default:
      return nullptr;
  }
}

[[nodiscard]] auto reflectionConstant(TranslationUnit* unit,
                                      std::shared_ptr<Meta> reflection)
    -> ExpressionAST* {
  auto arena = unit->arena();
  auto constant = ConstExpressionAST::create(arena);
  constant->type = unit->control()->getBuiltinMetaInfoType();
  constant->valueCategory = ValueCategory::kPrValue;
  constant->constValue = arena->make<ConstValue>(std::move(reflection));
  return constant;
}

[[nodiscard]] auto checkedExpression(TranslationUnit* unit, ScopeSymbol* scope,
                                     ExpressionAST* expression)
    -> ExpressionAST* {
  SilentDiagnosticsScope silent{unit};
  TypeChecker check{unit};
  check.setScope(scope ? scope : unit->globalScope());
  check.check(&expression);
  if (silent.hadError() || !expression || !expression->type) return nullptr;
  return expression;
}

[[nodiscard]] auto checkedSplice(TranslationUnit* unit, ScopeSymbol* scope,
                                 std::shared_ptr<Meta> reflection)
    -> ExpressionAST* {
  auto arena = unit->arena();
  auto splicer = SplicerAST::create(arena);
  splicer->expression = reflectionConstant(unit, std::move(reflection));
  auto splice = SpliceExpressionAST::create(arena);
  splice->splicer = splicer;
  return checkedExpression(unit, scope, splice);
}

[[nodiscard]] auto checkedAddressOf(TranslationUnit* unit, ScopeSymbol* scope,
                                    ExpressionAST* operand) -> ExpressionAST* {
  auto address = UnaryExpressionAST::create(unit->arena());
  address->op = TokenKind::T_AMP;
  address->expression = operand;
  return checkedExpression(unit, scope, address);
}

[[nodiscard]] auto templateTemplateArgument(TranslationUnit* unit,
                                            Symbol* symbol)
    -> TemplateArgumentAST* {
  auto arena = unit->arena();
  auto named = NamedTypeSpecifierAST::create(arena);
  named->symbol = symbol;
  named->unqualifiedId =
      NameIdAST::create(arena, name_cast<Identifier>(symbol->name()));
  auto typeId = TypeIdAST::create(arena);
  typeId->typeSpecifierList = make_list_node<SpecifierAST>(arena, named);
  return TypeTemplateArgumentAST::create(arena, typeId);
}

[[nodiscard]] auto typeTemplateArgument(TranslationUnit* unit, const Type* type)
    -> TemplateArgumentAST* {
  auto arena = unit->arena();
  auto typeId = TypeIdAST::create(arena);
  typeId->type = type;
  return TypeTemplateArgumentAST::create(arena, typeId);
}

[[nodiscard]] auto templateArgumentOf(TranslationUnit* unit, ScopeSymbol* scope,
                                      const std::shared_ptr<Meta>& reflection)
    -> TemplateArgumentAST* {
  auto entity = std::get_if<const Symbol*>(&reflection->value);
  auto symbol = entity ? const_cast<Symbol*>(*entity) : nullptr;
  if (symbol && unit->typeTraits().is_template(symbol))
    return templateTemplateArgument(unit, symbol);
  if (auto type = Reflection::type(*reflection))
    return typeTemplateArgument(unit, type);
  auto splice = checkedSplice(unit, scope, reflection);
  if (!splice) return nullptr;
  auto arena = unit->arena();
  auto nested = NestedExpressionAST::create(arena);
  nested->expression = splice;
  nested->type = splice->type;
  nested->valueCategory = splice->valueCategory;
  auto argument = ExpressionTemplateArgumentAST::create(arena);
  argument->expression = nested;
  argument->evaluationScope = scope;
  return argument;
}

[[nodiscard]] auto templateArgumentListOf(
    TranslationUnit* unit, ScopeSymbol* scope,
    const std::vector<std::shared_ptr<Meta>>& arguments)
    -> std::optional<List<TemplateArgumentAST*>*> {
  List<TemplateArgumentAST*>* list = nullptr;
  auto tail = &list;
  for (const auto& argument : arguments) {
    auto node = templateArgumentOf(unit, scope, argument);
    if (!node) return std::nullopt;
    *tail = make_list_node(unit->arena(), node);
    tail = &(*tail)->next;
  }
  return list;
}

[[nodiscard]] auto substituteTemplate(TranslationUnit* unit, Symbol* primary,
                                      List<TemplateArgumentAST*>* arguments,
                                      SourceLocation location) -> Symbol* {
  const auto isFunction = symbol_cast<FunctionSymbol>(primary) != nullptr;
  if (!TemplateArity::matches(template_declaration_of(primary), arguments,
                              isFunction))
    return nullptr;
  auto specialization =
      ASTRewriter::instantiate(unit, arguments, primary, location,
                               /*sfinaeContext=*/true, /*argsComplete=*/false,
                               /*declarationOnly=*/isFunction);
  auto function = symbol_cast<FunctionSymbol>(specialization);
  if (!function) return specialization;
  if (containsPlaceholderType(function->type()))
    (void)ASTRewriter::completePendingBodyFor(unit, function);
  return containsPlaceholderType(function->type()) ? nullptr : specialization;
}

class RangeWalk {
 public:
  RangeWalk(TranslationUnit* unit, ScopeSymbol* scope,
            ASTInterpreter* interpreter)
      : unit_(unit), scope_(scope), interpreter_(interpreter) {}

  [[nodiscard]] auto begin(const Type* rangeType,
                           std::shared_ptr<ConstAddress> range) -> bool {
    auto object = constant(rangeType, std::move(range), ValueCategory::kLValue);
    auto first = rangesCall("begin", object);
    auto last = rangesCall("end", object);
    if (!first || !last) return false;
    auto firstValue = interpreter_->evaluate(first);
    auto lastValue = interpreter_->evaluate(last);
    if (!firstValue || !lastValue) {
      interpreter_->abortEvaluation();
      return false;
    }
    iterator_ = unit_->arena()->make<ConstValue>(std::move(*firstValue));
    auto current = ConstExpressionAST::create(unit_->arena());
    current->type = first->type;
    current->valueCategory = ValueCategory::kPrValue;
    current->constValue = iterator_;
    auto sentinel =
        constant(last->type, std::move(*lastValue), ValueCategory::kPrValue);
    auto comparison = BinaryExpressionAST::create(unit_->arena());
    comparison->op = TokenKind::T_EQUAL_EQUAL;
    comparison->leftExpression = current;
    comparison->rightExpression = sentinel;
    atEnd_ = checkedExpression(unit_, scope_, comparison);
    auto dereference = UnaryExpressionAST::create(unit_->arena());
    dereference->op = TokenKind::T_STAR;
    dereference->expression = current;
    element_ = checkedExpression(unit_, scope_, dereference);
    next_ = rangesCall("next", current);
    return atEnd_ && element_ && next_;
  }

  [[nodiscard]] auto element() const -> ExpressionAST* { return element_; }

  [[nodiscard]] auto atEnd() -> std::optional<bool> {
    auto value = interpreter_->evaluate(atEnd_);
    auto result = value ? std::get_if<ConstInt>(&*value) : nullptr;
    if (!result) {
      interpreter_->abortEvaluation();
      return std::nullopt;
    }
    return result->toUIntMax() != 0;
  }

  [[nodiscard]] auto advance() -> bool {
    auto value = interpreter_->evaluate(next_);
    if (!value) {
      interpreter_->abortEvaluation();
      return false;
    }
    *iterator_ = std::move(*value);
    return true;
  }

 private:
  [[nodiscard]] auto constant(const Type* type, ConstValue value,
                              ValueCategory category) -> ConstExpressionAST* {
    auto arena = unit_->arena();
    auto expression = ConstExpressionAST::create(arena);
    expression->type = type;
    expression->valueCategory = category;
    expression->constValue = arena->make<ConstValue>(std::move(value));
    return expression;
  }

  [[nodiscard]] auto rangesCall(std::string_view name, ExpressionAST* argument)
      -> ExpressionAST* {
    auto function = unit_->typeTraits().standard_ranges_member(name);
    if (!function || !function->type()) return nullptr;
    auto arena = unit_->arena();
    auto callee = IdExpressionAST::create(arena);
    callee->symbol = function;
    callee->type = function->type();
    callee->valueCategory = ValueCategory::kLValue;
    auto call = CallExpressionAST::create(arena);
    call->baseExpression = callee;
    call->expressionList = make_list_node<ExpressionAST>(arena, argument);
    return checkedExpression(unit_, scope_, call);
  }

  TranslationUnit* unit_;
  ScopeSymbol* scope_;
  ASTInterpreter* interpreter_;
  ConstValue* iterator_ = nullptr;
  ExpressionAST* atEnd_ = nullptr;
  ExpressionAST* element_ = nullptr;
  ExpressionAST* next_ = nullptr;
};

[[nodiscard]] auto reflectionsOf(TranslationUnit* unit, ScopeSymbol* scope,
                                 ASTInterpreter* interpreter,
                                 const Type* rangeType,
                                 std::shared_ptr<ConstAddress> range)
    -> std::optional<std::vector<std::shared_ptr<Meta>>> {
  RangeWalk walk{unit, scope, interpreter};
  if (!walk.begin(rangeType, std::move(range))) return std::nullopt;
  PreparedInitialization element{
      unit, scope, unit->control()->getBuiltinMetaInfoType(), walk.element()};
  std::vector<std::shared_ptr<Meta>> result;
  while (true) {
    auto done = walk.atEnd();
    if (!done) return std::nullopt;
    if (*done) return result;
    auto value = element.value(interpreter);
    if (!value || !reflectedValue(*value)) return std::nullopt;
    result.push_back(std::get<std::shared_ptr<Meta>>(*value));
    if (!walk.advance()) return std::nullopt;
  }
}

[[nodiscard]] auto rangeValueType(TranslationUnit* unit, const Type* rangeType)
    -> const Type* {
  auto alias = unit->typeTraits().standard_ranges_member("range_value_t");
  if (!alias) return nullptr;
  auto arguments =
      make_list_node(unit->arena(), typeTemplateArgument(unit, rangeType));
  auto specialization = ASTRewriter::instantiate(unit, arguments, alias);
  return specialization ? specialization->type() : nullptr;
}

[[nodiscard]] auto parameterObjectReflection(VariableSymbol* parameter)
    -> std::shared_ptr<Meta> {
  if (!parameter) return nullptr;
  return std::make_shared<Meta>(std::make_shared<ConstAddress>(parameter));
}

class ConstantArrayBuilder {
 public:
  ConstantArrayBuilder(TranslationUnit* unit, ScopeSymbol* scope,
                       ASTInterpreter* interpreter)
      : unit_(unit), scope_(scope), interpreter_(interpreter) {}

  [[nodiscard]] auto string(const Type* rangeType,
                            std::shared_ptr<ConstAddress> range)
      -> std::shared_ptr<Meta> {
    auto traits = unit_->typeTraits();
    auto character = rangeValueType(unit_, rangeType);
    if (!character || !traits.is_char_type(character)) return nullptr;
    const auto isLiteral = range->stringLiteral() != nullptr;
    auto values = elements(character, rangeType, std::move(range));
    if (!values) return nullptr;
    if (isLiteral && !values->empty()) values->pop_back();
    for (auto& [value, type] : *values) type = traits.add_const(character);
    auto representation = traits.integral_representation(character);
    auto terminator = representation ? ConstInt::make(0, representation->bits,
                                                      representation->isSigned)
                                     : std::nullopt;
    if (!terminator) return nullptr;
    values->emplace_back(*terminator, traits.add_const(character));
    return object(character, *values);
  }

  [[nodiscard]] auto array(const Type* rangeType,
                           std::shared_ptr<ConstAddress> range)
      -> std::shared_ptr<Meta> {
    auto element = rangeValueType(unit_, rangeType);
    if (!element) return nullptr;
    auto values = elements(element, rangeType, std::move(range));
    if (!values) return nullptr;
    if (values->empty()) return emptyArray(element);
    return object(element, *values);
  }

 private:
  using Elements = std::vector<InitializerList::Element>;

  [[nodiscard]] auto scalarElement(const Type* type,
                                   PreparedInitialization& initialization)
      -> std::optional<ConstValue> {
    auto value = initialization.value(interpreter_);
    if (!value) return std::nullopt;
    auto reflection = reflectConstantValue(
        unit_, type, *value, interpreter_->evaluationPoint(), scope_);
    if (!reflection) return std::nullopt;
    auto constant = std::get_if<std::shared_ptr<Meta>>(&*reflection);
    return constantValueOf(**constant);
  }

  [[nodiscard]] auto arrayElement(const Type* type, ExpressionAST* address)
      -> std::optional<ConstValue> {
    auto value = interpreter_->evaluate(address);
    auto subarray =
        value ? std::get_if<std::shared_ptr<ConstAddress>>(&*value) : nullptr;
    if (!subarray || !*subarray) return std::nullopt;
    auto traits = unit_->typeTraits();
    auto rangeType = traits.add_lvalue_reference(traits.add_const(type));
    auto nested = array(rangeType, *subarray);
    return nested ? constantValueOf(*nested) : std::nullopt;
  }

  [[nodiscard]] auto elements(const Type* element, const Type* rangeType,
                              std::shared_ptr<ConstAddress> range)
      -> std::optional<Elements> {
    auto traits = unit_->typeTraits();
    auto objectType = traits.remove_reference(rangeType);
    RangeWalk walk{unit_, scope_, interpreter_};
    if (!walk.begin(objectType, std::move(range))) return std::nullopt;
    const auto isArray = traits.is_array(element);
    std::optional<PreparedInitialization> initialization;
    ExpressionAST* address = nullptr;
    if (isArray) {
      address = checkedAddressOf(unit_, scope_, walk.element());
      if (!address) return std::nullopt;
    } else {
      initialization.emplace(unit_, scope_, element, walk.element(),
                             InitializationKind::kDirectInitialization);
    }
    Elements result;
    while (true) {
      auto done = walk.atEnd();
      if (!done) return std::nullopt;
      if (*done) return result;
      auto value = isArray ? arrayElement(element, address)
                           : scalarElement(element, *initialization);
      if (!value) return std::nullopt;
      result.emplace_back(std::move(*value),
                          isArray ? traits.add_const(element) : element);
      if (!walk.advance()) return std::nullopt;
    }
  }

  [[nodiscard]] auto object(const Type* element, const Elements& values)
      -> std::shared_ptr<Meta> {
    auto control = unit_->control();
    auto traits = unit_->typeTraits();
    auto type =
        control->getBoundedArrayType(traits.add_const(element), values.size());
    return parameterObjectReflection(control->getTemplateParameterObject(
        type, std::make_shared<InitializerList>(values)));
  }

  [[nodiscard]] auto emptyArray(const Type* element) -> std::shared_ptr<Meta> {
    auto traits = unit_->typeTraits();
    auto base = traits.remove_all_extents(element);
    auto arena = unit_->arena();
    auto size = ExpressionTemplateArgumentAST::create(arena);
    auto zero = ConstExpressionAST::create(arena);
    zero->type = unit_->control()->getSizeType();
    zero->valueCategory = ValueCategory::kPrValue;
    zero->constValue = arena->make<ConstValue>(ConstInt{std::size_t{0}});
    size->expression = zero;
    auto arguments = make_list_node(arena, typeTemplateArgument(unit_, base));
    arguments->next = make_list_node<TemplateArgumentAST>(arena, size);
    auto array = traits.standard_class_specialization(WellKnownName::T_ARRAY,
                                                      arguments, {});
    if (!array) return nullptr;
    auto braced = BracedInitListAST::create(arena);
    PreparedInitialization initialization{
        unit_, scope_, array->type(), braced,
        InitializationKind::kCopyListInitialization};
    auto value = initialization.value(interpreter_);
    if (!value) return nullptr;
    return parameterObjectReflection(
        Substitution::initializeTemplateParameterObject(
            unit_, array->type(), *value, interpreter_->evaluationPoint(),
            scope_));
  }

  TranslationUnit* unit_;
  ScopeSymbol* scope_;
  ASTInterpreter* interpreter_;
};

[[nodiscard]] auto constantOf(TranslationUnit* unit, ScopeSymbol* scope,
                              ASTInterpreter* interpreter,
                              const std::shared_ptr<Meta>& reflection)
    -> std::shared_ptr<Meta> {
  auto traits = unit->typeTraits();
  auto splice = checkedSplice(unit, scope, reflection);
  if (!splice) return nullptr;
  auto type = traits.remove_cv(splice->type);
  if (traits.is_function(type)) return reflection;
  if (traits.is_array(splice->type)) {
    auto address = checkedAddressOf(unit, scope, splice);
    auto value = address ? interpreter->evaluate(address) : std::nullopt;
    auto range =
        value ? std::get_if<std::shared_ptr<ConstAddress>>(&*value) : nullptr;
    if (!range || !*range) return nullptr;
    return ConstantArrayBuilder{unit, scope, interpreter}.array(
        traits.add_lvalue_reference(splice->type), *range);
  }
  PreparedInitialization initialization{unit, scope, type, splice};
  auto value = initialization.value(interpreter);
  if (!value) return nullptr;
  auto constant = reflectConstantValue(unit, type, *value,
                                       interpreter->evaluationPoint(), scope);
  if (!constant) return nullptr;
  return std::get<std::shared_ptr<Meta>>(*constant);
}

struct RangeQuerySignature {
  const Type* result = nullptr;
  std::size_t leadingReflections = 0;
};

[[nodiscard]] auto rangeQuerySignature(WellKnownName name, Control* control)
    -> std::optional<RangeQuerySignature> {
  auto info = control->getBuiltinMetaInfoType();
  auto boolean = control->getBoolType();
  switch (name) {
    case WellKnownName::T_CAN_SUBSTITUTE:
    case WellKnownName::T_IS_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_INVOCABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_INVOCABLE_TYPE:
      return RangeQuerySignature{boolean, 1};
    case WellKnownName::T_IS_INVOCABLE_R_TYPE:
    case WellKnownName::T_IS_NOTHROW_INVOCABLE_R_TYPE:
      return RangeQuerySignature{boolean, 2};
    case WellKnownName::T_SUBSTITUTE:
    case WellKnownName::T_INVOKE_RESULT:
      return RangeQuerySignature{info, 1};
    case WellKnownName::T_COMMON_TYPE:
    case WellKnownName::T_COMMON_REFERENCE:
    case WellKnownName::T_REFLECT_CONSTANT_STRING:
    case WellKnownName::T_REFLECT_CONSTANT_ARRAY:
      return RangeQuerySignature{info, 0};
    default:
      return std::nullopt;
  }
}

[[nodiscard]] auto hasRangeQuerySignature(FunctionSymbol* function,
                                          const RangeQuerySignature& signature,
                                          Control* control) -> bool {
  auto primary = symbol_cast<FunctionSymbol>(primary_template_of(function));
  auto head = primary ? primary->templateParameters() : nullptr;
  if (!head || head->members().size() != 1) return false;
  auto parameter = head->members().front();
  auto pattern = type_cast<FunctionType>(primary->type());
  if (!pattern || pattern->isVariadic() ||
      pattern->returnType() != signature.result)
    return false;
  const auto& parameters = pattern->parameterTypes();
  if (parameters.size() != signature.leadingReflections + 1) return false;
  for (std::size_t index = 0; index < signature.leadingReflections; ++index) {
    if (parameters[index] != control->getBuiltinMetaInfoType()) return false;
  }
  auto range = type_cast<RvalueReferenceType>(parameters.back());
  return range && range->elementType() == parameter->type();
}

[[nodiscard]] auto hasExtractSignature(FunctionSymbol* function,
                                       Control* control) -> bool {
  auto type = type_cast<FunctionType>(function->type());
  if (type->parameterTypes().size() != 1 ||
      type->parameterTypes().front() != control->getBuiltinMetaInfoType())
    return false;
  auto primary = symbol_cast<FunctionSymbol>(primary_template_of(function));
  auto head = primary ? primary->templateParameters() : nullptr;
  if (!head || head->members().size() != 1) return false;
  auto parameter = symbol_cast<TypeParameterSymbol>(head->members().front());
  if (!parameter) return false;
  auto pattern = type_cast<FunctionType>(primary->type());
  return pattern && pattern->returnType() == parameter->type();
}

[[nodiscard]] auto hasIndexedTraitSignature(const FunctionType* type,
                                            Control* control) -> bool {
  const std::vector<const Type*> parameters{control->getSizeType(),
                                            control->getBuiltinMetaInfoType()};
  return type->returnType() == control->getBuiltinMetaInfoType() &&
         type->parameterTypes() == parameters;
}

[[nodiscard]] auto reflectedTypes(
    const std::vector<std::shared_ptr<Meta>>& range)
    -> std::optional<std::vector<const Type*>> {
  std::vector<const Type*> types;
  types.reserve(range.size());
  for (const auto& reflection : range) {
    auto type = Reflection::type(*reflection);
    if (!type) return std::nullopt;
    types.push_back(type);
  }
  return types;
}

[[nodiscard]] auto associatedTraitClass(TranslationUnit* unit,
                                        WellKnownName trait,
                                        std::span<const Type* const> types)
    -> ClassSymbol* {
  List<TemplateArgumentAST*>* arguments = nullptr;
  auto tail = &arguments;
  for (auto type : types) {
    *tail = make_list_node(unit->arena(), typeTemplateArgument(unit, type));
    tail = &(*tail)->next;
  }
  return unit->typeTraits().standard_class_specialization(trait, arguments, {});
}

[[nodiscard]] auto associatedTraitValue(TranslationUnit* unit,
                                        WellKnownName trait,
                                        std::span<const Type* const> types)
    -> std::optional<ConstValue> {
  auto traitClass = associatedTraitClass(unit, trait, types);
  if (!traitClass) return std::nullopt;
  return unit->typeTraits().member_value(traitClass);
}

[[nodiscard]] auto associatedTraitType(TranslationUnit* unit,
                                       WellKnownName trait,
                                       std::span<const Type* const> types)
    -> std::optional<ConstValue> {
  auto traitClass = associatedTraitClass(unit, trait, types);
  auto type = traitClass ? unit->typeTraits().member_type(traitClass) : nullptr;
  if (!type) return std::nullopt;
  return std::make_shared<Meta>(type);
}

[[nodiscard]] auto constructibleQuery(TypeTraits& traits, WellKnownName builtin,
                                      std::span<const Type* const> types,
                                      ASTInterpreter* interpreter)
    -> std::optional<ConstValue> {
  for (auto type : types) {
    if (completeTraitType(traits, type)) continue;
    interpreter->abortEvaluation();
    return std::nullopt;
  }
  auto arguments = types.subspan(1);
  switch (builtin) {
    case WellKnownName::T_IS_CONSTRUCTIBLE_TYPE:
      return traits.is_constructible(types.front(), arguments);
    case WellKnownName::T_IS_TRIVIALLY_CONSTRUCTIBLE_TYPE:
      return traits.is_trivially_constructible(types.front(), arguments);
    default:
      return traits.is_nothrow_constructible(types.front(), arguments);
  }
}

[[nodiscard]] auto typeRangeQuery(TranslationUnit* unit, WellKnownName builtin,
                                  std::span<const Type* const> types,
                                  ASTInterpreter* interpreter)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  switch (builtin) {
    case WellKnownName::T_IS_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_CONSTRUCTIBLE_TYPE:
      return constructibleQuery(traits, builtin, types, interpreter);
    case WellKnownName::T_IS_INVOCABLE_TYPE:
      return associatedTraitValue(unit, WellKnownName::T_IS_INVOCABLE, types);
    case WellKnownName::T_IS_INVOCABLE_R_TYPE:
      return associatedTraitValue(unit, WellKnownName::T_IS_INVOCABLE_R, types);
    case WellKnownName::T_IS_NOTHROW_INVOCABLE_TYPE:
      return associatedTraitValue(unit, WellKnownName::T_IS_NOTHROW_INVOCABLE,
                                  types);
    case WellKnownName::T_IS_NOTHROW_INVOCABLE_R_TYPE:
      return associatedTraitValue(unit, WellKnownName::T_IS_NOTHROW_INVOCABLE_R,
                                  types);
    case WellKnownName::T_COMMON_TYPE:
    case WellKnownName::T_COMMON_REFERENCE:
    case WellKnownName::T_INVOKE_RESULT:
      return associatedTraitType(unit, builtin, types);
    default:
      return std::nullopt;
  }
}

[[nodiscard]] auto rangeQuery(TranslationUnit* unit, WellKnownName builtin,
                              std::span<const ConstValue> arguments,
                              ScopeSymbol* scope, ASTInterpreter* interpreter,
                              FunctionSymbol* function)
    -> std::optional<ConstValue> {
  if (!function || !interpreter || arguments.empty()) return std::nullopt;
  auto range = std::get_if<std::shared_ptr<ConstAddress>>(&arguments.back());
  if (!range || !*range) return std::nullopt;
  auto traits = unit->typeTraits();
  auto parameterType =
      type_cast<FunctionType>(function->type())->parameterTypes().back();
  auto rangeType = traits.remove_reference(parameterType);
  if (builtin == WellKnownName::T_REFLECT_CONSTANT_STRING ||
      builtin == WellKnownName::T_REFLECT_CONSTANT_ARRAY) {
    ConstantArrayBuilder builder{unit, scope, interpreter};
    auto result = builtin == WellKnownName::T_REFLECT_CONSTANT_STRING
                      ? builder.string(parameterType, *range)
                      : builder.array(parameterType, *range);
    if (!result) return std::nullopt;
    return result;
  }
  auto elements = reflectionsOf(unit, scope, interpreter, rangeType, *range);
  if (!elements) return std::nullopt;
  if (builtin == WellKnownName::T_CAN_SUBSTITUTE ||
      builtin == WellKnownName::T_SUBSTITUTE) {
    auto subject = reflectedValue(arguments.front());
    auto entity =
        subject ? std::get_if<const Symbol*>(&subject->value) : nullptr;
    auto primary = entity ? const_cast<Symbol*>(*entity) : nullptr;
    if (!primary || !traits.is_template(primary)) return std::nullopt;
    auto list = templateArgumentListOf(unit, scope, *elements);
    if (!list) return std::nullopt;
    auto specialization = substituteTemplate(unit, primary, *list,
                                             interpreter->evaluationPoint());
    if (builtin == WellKnownName::T_CAN_SUBSTITUTE)
      return specialization != nullptr;
    if (!specialization) return std::nullopt;
    return std::make_shared<Meta>(visit(ReflectEntity{}, specialization));
  }
  auto ranged = reflectedTypes(*elements);
  if (!ranged) return std::nullopt;
  std::vector<const Type*> types;
  for (const auto& leading : arguments.first(arguments.size() - 1)) {
    auto reflection = reflectedValue(leading);
    auto type = reflection ? Reflection::type(*reflection) : nullptr;
    if (!type) return std::nullopt;
    types.push_back(type);
  }
  types.insert(types.end(), ranged->begin(), ranged->end());
  return typeRangeQuery(unit, builtin, types, interpreter);
}

[[nodiscard]] auto transcodedText(TranslationUnit* unit, WellKnownName builtin,
                                  std::span<const ConstValue> arguments,
                                  ScopeSymbol* scope,
                                  ASTInterpreter* interpreter,
                                  FunctionSymbol* function)
    -> std::optional<ConstValue> {
  if (!function || !interpreter || arguments.size() != 1) return std::nullopt;
  auto range = std::get_if<std::shared_ptr<ConstAddress>>(&arguments.front());
  if (!range || !*range) return std::nullopt;
  auto traits = unit->typeTraits();
  auto parameterType =
      type_cast<FunctionType>(function->type())->parameterTypes().back();
  RangeWalk walk{unit, scope, interpreter};
  if (!walk.begin(traits.remove_reference(parameterType), *range))
    return std::nullopt;
  PreparedInitialization element{unit, scope, unit->control()->getIntType(),
                                 walk.element()};
  std::string text;
  while (true) {
    auto done = walk.atEnd();
    if (!done) return std::nullopt;
    if (*done) break;
    auto value = element.value(interpreter);
    auto character = value ? std::get_if<ConstInt>(&*value) : nullptr;
    if (!character) return std::nullopt;
    text.push_back(static_cast<char>(character->toIntMax()));
    if (!walk.advance()) return std::nullopt;
  }
  return textResult(unit->control(), text,
                    builtin == WellKnownName::T__S_EXCEPTION_CVT_TO_UTF8);
}

[[nodiscard]] auto indexedTrait(TranslationUnit* unit, WellKnownName builtin,
                                std::span<const ConstValue> arguments)
    -> std::optional<ConstValue> {
  auto index = std::get_if<ConstInt>(&arguments[0]);
  auto reflection = reflectedValue(arguments[1]);
  auto type = reflection ? Reflection::type(*reflection) : nullptr;
  if (!index || !type) return std::nullopt;
  auto arena = unit->arena();
  auto constant = ConstExpressionAST::create(arena);
  constant->type = unit->control()->getSizeType();
  constant->valueCategory = ValueCategory::kPrValue;
  constant->constValue = arena->make<ConstValue>(*index);
  auto value = ExpressionTemplateArgumentAST::create(arena);
  value->expression = constant;
  auto list = make_list_node<TemplateArgumentAST>(arena, value);
  list->next = make_list_node(arena, typeTemplateArgument(unit, type));
  auto traits = unit->typeTraits();
  auto traitClass = traits.standard_class_specialization(builtin, list, {});
  auto member = traitClass ? traits.member_type(traitClass) : nullptr;
  if (!member) return std::nullopt;
  return std::make_shared<Meta>(member);
}

[[nodiscard]] auto typeOrder(TranslationUnit* unit,
                             std::span<const ConstValue> arguments)
    -> std::optional<ConstValue> {
  auto first = reflectedValue(arguments[0]);
  auto second = reflectedValue(arguments[1]);
  auto firstType = first ? Reflection::type(*first) : nullptr;
  auto secondType = second ? Reflection::type(*second) : nullptr;
  auto ordering = symbol_cast<ClassSymbol>(
      lookupStandardLibraryType(unit, WellKnownName::T_STRONG_ORDERING));
  if (!firstType || !secondType || !ordering) return std::nullopt;
  auto traits = unit->typeTraits();
  const auto order = traits.type_order(firstType, secondType);
  if (order == std::strong_ordering::less)
    return traits.member_value(ordering, "less");
  if (order == std::strong_ordering::greater)
    return traits.member_value(ordering, "greater");
  return traits.member_value(ordering, "equal");
}

[[nodiscard]] auto extractReference(TranslationUnit* unit, ScopeSymbol* scope,
                                    ASTInterpreter* interpreter,
                                    const Type* type, const Meta& reflection)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto control = unit->control();
  std::shared_ptr<ConstAddress> address;
  const Type* objectType = nullptr;
  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value)) {
    address = *object;
    objectType = address ? address->objectType(traits) : nullptr;
  } else {
    auto entity = std::get_if<const Symbol*>(&reflection.value);
    auto symbol = entity ? const_cast<Symbol*>(*entity) : nullptr;
    if (!traits.is_variable(symbol)) return std::nullopt;
    objectType = traits.remove_reference(symbol->type());
    if (traits.is_reference(symbol->type())) {
      address = boundReferenceAddress(unit, symbol, scope, interpreter);
    } else if (traits.is_usable_in_constant_expressions(symbol, scope)) {
      address = std::make_shared<ConstAddress>(symbol);
    } else {
      auto bound = interpreter->variableAddress(symbol);
      auto active =
          bound ? std::get_if<std::shared_ptr<ConstAddress>>(&*bound) : nullptr;
      if (active) address = *active;
    }
  }
  if (!address || !objectType) return std::nullopt;
  if (!traits.is_convertible(
          control->getPointerType(control->getUnboundedArrayType(objectType)),
          control->getPointerType(
              control->getUnboundedArrayType(traits.remove_reference(type)))))
    return std::nullopt;
  return address;
}

[[nodiscard]] auto isFunctionPointer(const TypeTraits& traits, const Type* type)
    -> bool {
  return traits.is_pointer(type) &&
         traits.is_function(traits.remove_pointer(type));
}

[[nodiscard]] auto extractMemberOrFunction(TranslationUnit* unit,
                                           ScopeSymbol* scope,
                                           ASTInterpreter* interpreter,
                                           const Type* type, Symbol* symbol,
                                           std::shared_ptr<Meta> reflection)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto control = unit->control();
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    auto owner = symbol_cast<ClassSymbol>(field->parent());
    if (field->isBitField() || !owner) return std::nullopt;
    auto member =
        control->getMemberObjectPointerType(owner->type(), field->type());
    if (!traits.is_similar(type, member)) return std::nullopt;
  } else if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    auto owner = symbol_cast<ClassSymbol>(function->parent());
    if (owner && !function->isStatic() &&
        !function->hasExplicitObjectParameter()) {
      auto member = type_cast<MemberFunctionPointerType>(type);
      if (!member || member->classType() != owner->type()) return std::nullopt;
    } else if (!isFunctionPointer(traits, type)) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  auto splice = checkedSplice(unit, scope, std::move(reflection));
  auto address = splice ? checkedAddressOf(unit, scope, splice) : nullptr;
  if (!address) return std::nullopt;
  PreparedInitialization initialization{unit, scope, type, address};
  return initialization.value(interpreter);
}

[[nodiscard]] auto canExtractPointer(const TypeTraits& traits, const Type* type,
                                     const Type* source) -> bool {
  if (traits.is_similar(type, source))
    return traits.is_convertible(source, type);
  if (!isFunctionPointer(traits, type) || !isFunctionPointer(traits, source))
    return false;
  return traits.is_convertible(source, type);
}

[[nodiscard]] auto canExtractValue(const TypeTraits& traits, const Type* type,
                                   const Type* source) -> bool {
  if (traits.is_pointer(source)) return canExtractPointer(traits, type, source);
  if (source == traits.remove_cv(type)) return true;
  if (traits.is_array(source) && traits.is_pointer(type)) {
    auto element = traits.add_pointer(traits.remove_extent(source));
    return traits.is_similar(element, type);
  }
  auto closure = type_cast<ClassType>(source);
  return closure && closure->symbol()->isClosureType() &&
         isFunctionPointer(traits, type);
}

[[nodiscard]] auto extractValue(TranslationUnit* unit, ScopeSymbol* scope,
                                ASTInterpreter* interpreter, const Type* type,
                                const std::shared_ptr<Meta>& reflection)
    -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto constant = constantOf(unit, scope, interpreter, reflection);
  auto splice = constant ? checkedSplice(unit, scope, constant) : nullptr;
  if (!splice) return std::nullopt;
  if (!canExtractValue(traits, type, traits.remove_cv(splice->type)))
    return std::nullopt;
  PreparedInitialization initialization{unit, scope, traits.remove_cv(type),
                                        splice};
  return initialization.value(interpreter);
}

[[nodiscard]] auto extract(TranslationUnit* unit, ScopeSymbol* scope,
                           ASTInterpreter* interpreter,
                           FunctionSymbol* function,
                           const std::shared_ptr<Meta>& reflection,
                           Symbol* symbol) -> std::optional<ConstValue> {
  auto traits = unit->typeTraits();
  auto type = producerArgumentType(function);
  if (!type || !interpreter) return std::nullopt;
  if (traits.is_reference(type))
    return extractReference(unit, scope, interpreter, type, *reflection);
  auto unqualified = traits.remove_cv(type);
  if (traits.is_nonstatic_data_member(symbol) || traits.is_function(symbol))
    return extractMemberOrFunction(unit, scope, interpreter, unqualified,
                                   symbol, reflection);
  return extractValue(unit, scope, interpreter, unqualified, reflection);
}

}  // namespace

void Reflection::check(ExpressionAST* expression) const {
  expression->type = unit_->control()->getBuiltinMetaInfoType();
  expression->valueCategory = ValueCategory::kPrValue;
  if (auto reflected = ast_cast<TypeIdReflectExpressionAST>(expression)) {
    auto typeId = reflected->typeId;
    if (!typeId || !typeId->type) {
      unit_->error(expression->firstSourceLocation(), "expected a type");
      return;
    }
    if (containsPlaceholderType(typeId->type))
      unit_->error(expression->firstSourceLocation(),
                   "reflection cannot designate a placeholder type");
    return;
  }
  auto reflected = ast_cast<ReflectExpressionAST>(expression);
  if (!reflected) return;
  auto id = ast_cast<IdExpressionAST>(reflected->expression);
  if (!id) {
    unit_->error(expression->firstSourceLocation(),
                 "reflection requires an id-expression");
    return;
  }
  auto symbol = id->symbol;
  if (symbol_cast<NonTypeParameterSymbol>(symbol) ||
      is_function_local_predefined_variable(symbol)) {
    unit_->error(id->firstSourceLocation(), "invalid entity in reflection");
    return;
  }
  if (symbol && (symbol->isNamespaceName() || is_type(symbol) ||
                 symbol_cast<ConceptSymbol>(symbol)))
    return;
  if (auto overloads = symbol_cast<OverloadSetSymbol>(symbol)) {
    auto functions = views::each_function(overloads);
    FunctionSymbol* selected = nullptr;
    for (auto function : functions) {
      if (selected && selected->canonical() != function->canonical()) {
        unit_->error(id->firstSourceLocation(),
                     "reflection names an ambiguous overload set");
        return;
      }
      selected = function;
    }
    if (selected) id->symbol = selected;
  }
}

auto Reflection::isScopeQuery(WellKnownName builtin) -> bool {
  return builtin == WellKnownName::T_CURRENT_FUNCTION ||
         builtin == WellKnownName::T_CURRENT_CLASS ||
         builtin == WellKnownName::T_CURRENT_NAMESPACE ||
         builtin == WellKnownName::T_CURRENT;
}

auto Reflection::scope(ScopeSymbol* lexicalScope) -> ScopeSymbol* {
  for (auto scope = lexicalScope; scope; scope = scope->parent()) {
    auto function = Binder::functionOfBody(scope);
    if (is_consteval_block_call_operator(function)) {
      scope = function->parent();
      continue;
    }
    if (scope->isClassOrNamespace()) return scope;
    if (function) return function;
  }
  return nullptr;
}

auto Reflection::builtin(FunctionSymbol* function) const -> WellKnownName {
  if (!function || !function->isConsteval()) return WellKnownName::T_NONE;
  auto memberClass = symbol_cast<ClassSymbol>(function->parent());
  if (memberClass && unit_->typeTraits().is_std_meta_type(memberClass->type(),
                                                          "access_context")) {
    auto type = type_cast<FunctionType>(function->type());
    if (!type || type->isVariadic() ||
        type->returnType() != memberClass->type())
      return WellKnownName::T_NONE;
    auto name = well_known_name(function->name());
    if (name == WellKnownName::T_CURRENT && function->isStatic() &&
        type->isNoexcept() && type->parameterTypes().empty())
      return name;
    if (name == WellKnownName::T_VIA && !function->isStatic() &&
        type->cvQualifiers() == CvQualifiers::kConst &&
        type->parameterTypes() ==
            std::vector<const Type*>{
                unit_->control()->getBuiltinMetaInfoType()})
      return name;
    return WellKnownName::T_NONE;
  }
  if (memberClass &&
      unit_->typeTraits().is_std_meta_type(memberClass->type(), "exception")) {
    auto name = well_known_name(function->name());
    if (name != WellKnownName::T__S_EXCEPTION_CVT_TO_UTF8 &&
        name != WellKnownName::T__S_EXCEPTION_CVT_FROM_UTF8)
      return WellKnownName::T_NONE;
    if (!function->isStatic() || function->parameters().size() != 1)
      return WellKnownName::T_NONE;
    return name;
  }
  auto scope = symbol_cast<NamespaceSymbol>(function->parent());
  if (!scope) return WellKnownName::T_NONE;
  auto control = unit_->control();
  auto type = type_cast<FunctionType>(function->type());
  if (!type || type->isVariadic()) return WellKnownName::T_NONE;
  auto name = well_known_name(function->name());
  if (name == WellKnownName::T_IS_STRING_LITERAL) {
    if (!unit_->typeTraits().is_std_namespace(scope))
      return WellKnownName::T_NONE;
    if (type->returnType() != control->getBoolType() ||
        type->parameterTypes().size() != 1)
      return WellKnownName::T_NONE;
    const std::array<const Type*, 5> characters{
        control->getCharType(), control->getWideCharType(),
        control->getChar8Type(), control->getChar16Type(),
        control->getChar32Type()};
    for (const Type* character : characters) {
      auto qualified = control->getQualType(character, CvQualifiers::kConst);
      if (type->parameterTypes().front() == control->getPointerType(qualified))
        return name;
    }
    return WellKnownName::T_NONE;
  }
  if (well_known_name(scope->name()) != WellKnownName::T_META ||
      !unit_->typeTraits().is_std_namespace(scope->parent()))
    return WellKnownName::T_NONE;
  if (name == WellKnownName::T_EXTRACT)
    return hasExtractSignature(function, control) ? name
                                                  : WellKnownName::T_NONE;
  if (auto signature = rangeQuerySignature(name, control))
    return hasRangeQuerySignature(function, *signature, control)
               ? name
               : WellKnownName::T_NONE;
  if (name == WellKnownName::T_TUPLE_ELEMENT ||
      name == WellKnownName::T_VARIANT_ALTERNATIVE)
    return hasIndexedTraitSignature(type, control) ? name
                                                   : WellKnownName::T_NONE;
  if (name == WellKnownName::T_TYPE_ORDER) {
    const std::vector<const Type*> parameters{
        control->getBuiltinMetaInfoType(), control->getBuiltinMetaInfoType()};
    if (type->parameterTypes() != parameters ||
        !unit_->typeTraits().is_std_type(type->returnType(),
                                         WellKnownName::T_STRONG_ORDERING))
      return WellKnownName::T_NONE;
    return name;
  }
  if (isResultProducer(name))
    return hasResultProducerSignature(function, name, control)
               ? name
               : WellKnownName::T_NONE;
  const Type* result = nullptr;
  std::size_t parameterCount = 1;
  switch (name) {
    case WellKnownName::T_MEMBERS_OF:
    case WellKnownName::T_BASES_OF:
    case WellKnownName::T_STATIC_DATA_MEMBERS_OF:
    case WellKnownName::T_NONSTATIC_DATA_MEMBERS_OF:
    case WellKnownName::T_SUBOBJECTS_OF:
      parameterCount = 2;
      [[fallthrough]];
    case WellKnownName::T_TEMPLATE_ARGUMENTS_OF:
    case WellKnownName::T_PARAMETERS_OF:
    case WellKnownName::T_ENUMERATORS_OF:
      if (!isReflectionVector(unit_->typeTraits(), type->returnType()))
        return WellKnownName::T_NONE;
      result = type->returnType();
      break;
    case WellKnownName::T_OPERATOR_OF:
      if (!unit_->typeTraits().is_enum(type->returnType()) ||
          !unit_->typeTraits().is_std_meta_type(type->returnType(),
                                                "operators"))
        return WellKnownName::T_NONE;
      result = type->returnType();
      break;
    case WellKnownName::T_OFFSET_OF:
      if (!type_cast<ClassType>(type->returnType()) ||
          !unit_->typeTraits().is_std_meta_type(type->returnType(),
                                                "member_offset"))
        return WellKnownName::T_NONE;
      result = type->returnType();
      break;
    case WellKnownName::T_SYMBOL_OF:
    case WellKnownName::T_U8SYMBOL_OF:
    case WellKnownName::T_DISPLAY_STRING_OF:
    case WellKnownName::T_U8DISPLAY_STRING_OF:
    case WellKnownName::T_IDENTIFIER_OF:
    case WellKnownName::T_U8IDENTIFIER_OF:
      if (!isTextView(unit_->typeTraits(), type->returnType(),
                      name == WellKnownName::T_U8SYMBOL_OF ||
                          name == WellKnownName::T_U8DISPLAY_STRING_OF ||
                          name == WellKnownName::T_U8IDENTIFIER_OF))
        return WellKnownName::T_NONE;
      result = type->returnType();
      break;
    case WellKnownName::T_SOURCE_LOCATION_OF:
      if (!unit_->typeTraits().is_std_type(type->returnType(),
                                           WellKnownName::T_SOURCE_LOCATION))
        return WellKnownName::T_NONE;
      result = type->returnType();
      break;
    case WellKnownName::T_CURRENT_FUNCTION:
    case WellKnownName::T_CURRENT_CLASS:
    case WellKnownName::T_CURRENT_NAMESPACE:
      parameterCount = 0;
      result = control->getBuiltinMetaInfoType();
      break;
    case WellKnownName::T_OBJECT_OF:
    case WellKnownName::T_TYPE_OF:
    case WellKnownName::T_VARIABLE_OF:
    case WellKnownName::T_RETURN_TYPE_OF:
    case WellKnownName::T_TEMPLATE_OF:
    case WellKnownName::T_PARENT_OF:
    case WellKnownName::T_MAKE_SIGNED:
    case WellKnownName::T_MAKE_UNSIGNED:
    case WellKnownName::T_REMOVE_EXTENT:
    case WellKnownName::T_REMOVE_ALL_EXTENTS:
    case WellKnownName::T_REMOVE_POINTER:
    case WellKnownName::T_ADD_POINTER:
    case WellKnownName::T_REMOVE_CVREF:
    case WellKnownName::T_DECAY:
    case WellKnownName::T_UNWRAP_REFERENCE:
    case WellKnownName::T_UNWRAP_REF_DECAY:
    case WellKnownName::T_UNDERLYING_TYPE:
    case WellKnownName::T_DEALIAS:
    case WellKnownName::T_CONSTANT_OF:
    case WellKnownName::T_REMOVE_CONST:
    case WellKnownName::T_REMOVE_VOLATILE:
    case WellKnownName::T_REMOVE_CV:
    case WellKnownName::T_ADD_CONST:
    case WellKnownName::T_ADD_VOLATILE:
    case WellKnownName::T_ADD_CV:
    case WellKnownName::T_REMOVE_REFERENCE:
    case WellKnownName::T_ADD_LVALUE_REFERENCE:
    case WellKnownName::T_ADD_RVALUE_REFERENCE:
      result = control->getBuiltinMetaInfoType();
      break;
    case WellKnownName::T_IS_ACCESSIBLE:
    case WellKnownName::T_HAS_INACCESSIBLE_NONSTATIC_DATA_MEMBERS:
    case WellKnownName::T_HAS_INACCESSIBLE_BASES:
    case WellKnownName::T_HAS_INACCESSIBLE_SUBOBJECTS:
    case WellKnownName::T_IS_SAME_TYPE:
    case WellKnownName::T_IS_BASE_OF_TYPE:
    case WellKnownName::T_IS_VIRTUAL_BASE_OF_TYPE:
    case WellKnownName::T_IS_CONVERTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_CONVERTIBLE_TYPE:
    case WellKnownName::T_IS_LAYOUT_COMPATIBLE_TYPE:
    case WellKnownName::T_IS_POINTER_INTERCONVERTIBLE_BASE_OF_TYPE:
    case WellKnownName::T_IS_SWAPPABLE_WITH_TYPE:
    case WellKnownName::T_IS_NOTHROW_SWAPPABLE_WITH_TYPE:
    case WellKnownName::T_IS_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_ASSIGNABLE_TYPE:
    case WellKnownName::T_REFERENCE_CONSTRUCTS_FROM_TEMPORARY:
    case WellKnownName::T_REFERENCE_CONVERTS_FROM_TEMPORARY:
      parameterCount = 2;
      [[fallthrough]];
    case WellKnownName::T_IS_NAMESPACE:
    case WellKnownName::T_IS_FUNCTION_PARAMETER:
    case WellKnownName::T_IS_EXPLICIT_OBJECT_PARAMETER:
    case WellKnownName::T_HAS_DEFAULT_ARGUMENT:
    case WellKnownName::T_IS_PUBLIC:
    case WellKnownName::T_IS_PROTECTED:
    case WellKnownName::T_IS_PRIVATE:
    case WellKnownName::T_IS_VIRTUAL:
    case WellKnownName::T_IS_PURE_VIRTUAL:
    case WellKnownName::T_IS_OVERRIDE:
    case WellKnownName::T_IS_FINAL:
    case WellKnownName::T_IS_DELETED:
    case WellKnownName::T_IS_DEFAULTED:
    case WellKnownName::T_IS_USER_PROVIDED:
    case WellKnownName::T_IS_USER_DECLARED:
    case WellKnownName::T_IS_EXPLICIT:
    case WellKnownName::T_IS_NOEXCEPT:
    case WellKnownName::T_IS_CONST:
    case WellKnownName::T_IS_VOLATILE:
    case WellKnownName::T_IS_LVALUE_REFERENCE_QUALIFIED:
    case WellKnownName::T_IS_RVALUE_REFERENCE_QUALIFIED:
    case WellKnownName::T_HAS_STATIC_STORAGE_DURATION:
    case WellKnownName::T_HAS_THREAD_STORAGE_DURATION:
    case WellKnownName::T_HAS_AUTOMATIC_STORAGE_DURATION:
    case WellKnownName::T_IS_CONVERSION_FUNCTION:
    case WellKnownName::T_IS_OPERATOR_FUNCTION:
    case WellKnownName::T_IS_LITERAL_OPERATOR:
    case WellKnownName::T_IS_SPECIAL_MEMBER_FUNCTION:
    case WellKnownName::T_IS_CONSTRUCTOR:
    case WellKnownName::T_IS_DEFAULT_CONSTRUCTOR:
    case WellKnownName::T_IS_COPY_CONSTRUCTOR:
    case WellKnownName::T_IS_MOVE_CONSTRUCTOR:
    case WellKnownName::T_IS_ASSIGNMENT:
    case WellKnownName::T_IS_COPY_ASSIGNMENT:
    case WellKnownName::T_IS_MOVE_ASSIGNMENT:
    case WellKnownName::T_IS_DESTRUCTOR:
    case WellKnownName::T_IS_VARARG_FUNCTION:
    case WellKnownName::T_IS_CONVERSION_FUNCTION_TEMPLATE:
    case WellKnownName::T_IS_OPERATOR_FUNCTION_TEMPLATE:
    case WellKnownName::T_IS_LITERAL_OPERATOR_TEMPLATE:
    case WellKnownName::T_IS_CONSTRUCTOR_TEMPLATE:
    case WellKnownName::T_IS_CLASS_MEMBER:
    case WellKnownName::T_IS_NAMESPACE_MEMBER:
    case WellKnownName::T_IS_STATIC_MEMBER:
    case WellKnownName::T_IS_BASE:
    case WellKnownName::T_HAS_PARENT:
    case WellKnownName::T_IS_COMPLETE_TYPE:
    case WellKnownName::T_IS_ENUMERABLE_TYPE:
    case WellKnownName::T_IS_VALUE:
    case WellKnownName::T_IS_OBJECT:
    case WellKnownName::T_HAS_IDENTIFIER:
    case WellKnownName::T_HAS_INTERNAL_LINKAGE:
    case WellKnownName::T_HAS_MODULE_LINKAGE:
    case WellKnownName::T_HAS_EXTERNAL_LINKAGE:
    case WellKnownName::T_HAS_C_LANGUAGE_LINKAGE:
    case WellKnownName::T_HAS_LINKAGE:
    case WellKnownName::T_HAS_TEMPLATE_ARGUMENTS:
    case WellKnownName::T_IS_NAMESPACE_ALIAS:
    case WellKnownName::T_IS_VARIABLE:
    case WellKnownName::T_IS_STRUCTURED_BINDING:
    case WellKnownName::T_IS_FUNCTION:
    case WellKnownName::T_IS_ENUMERATOR:
    case WellKnownName::T_IS_TEMPLATE:
    case WellKnownName::T_IS_FUNCTION_TEMPLATE:
    case WellKnownName::T_IS_VARIABLE_TEMPLATE:
    case WellKnownName::T_IS_CLASS_TEMPLATE:
    case WellKnownName::T_IS_ALIAS_TEMPLATE:
    case WellKnownName::T_IS_CONCEPT:
    case WellKnownName::T_IS_NONSTATIC_DATA_MEMBER:
    case WellKnownName::T_IS_BIT_FIELD:
    case WellKnownName::T_IS_MUTABLE_MEMBER:
    case WellKnownName::T_HAS_DEFAULT_MEMBER_INITIALIZER:
    case WellKnownName::T_IS_TRIVIALLY_COPYABLE_TYPE:
    case WellKnownName::T_IS_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_COPY_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_MOVE_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_SWAPPABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_SWAPPABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_MOVE_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_COPY_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_MOVE_ASSIGNABLE_TYPE:
    case WellKnownName::T_IS_STANDARD_LAYOUT_TYPE:
    case WellKnownName::T_IS_EMPTY_TYPE:
    case WellKnownName::T_IS_POLYMORPHIC_TYPE:
    case WellKnownName::T_IS_ABSTRACT_TYPE:
    case WellKnownName::T_IS_FINAL_TYPE:
    case WellKnownName::T_IS_AGGREGATE_TYPE:
    case WellKnownName::T_IS_STRUCTURAL_TYPE:
    case WellKnownName::T_IS_IMPLICIT_LIFETIME_TYPE:
    case WellKnownName::T_IS_SIGNED_TYPE:
    case WellKnownName::T_IS_UNSIGNED_TYPE:
    case WellKnownName::T_IS_BOUNDED_ARRAY_TYPE:
    case WellKnownName::T_IS_UNBOUNDED_ARRAY_TYPE:
    case WellKnownName::T_IS_SCOPED_ENUM_TYPE:
    case WellKnownName::T_IS_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_TRIVIALLY_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_IS_NOTHROW_DESTRUCTIBLE_TYPE:
    case WellKnownName::T_HAS_VIRTUAL_DESTRUCTOR:
    case WellKnownName::T_HAS_UNIQUE_OBJECT_REPRESENTATIONS:
    case WellKnownName::T_IS_TYPE:
    case WellKnownName::T_IS_TYPE_ALIAS:
    case WellKnownName::T_IS_VOID_TYPE:
    case WellKnownName::T_IS_NULL_POINTER_TYPE:
    case WellKnownName::T_IS_INTEGRAL_TYPE:
    case WellKnownName::T_IS_FLOATING_POINT_TYPE:
    case WellKnownName::T_IS_ARRAY_TYPE:
    case WellKnownName::T_IS_POINTER_TYPE:
    case WellKnownName::T_IS_LVALUE_REFERENCE_TYPE:
    case WellKnownName::T_IS_RVALUE_REFERENCE_TYPE:
    case WellKnownName::T_IS_MEMBER_OBJECT_POINTER_TYPE:
    case WellKnownName::T_IS_MEMBER_FUNCTION_POINTER_TYPE:
    case WellKnownName::T_IS_ENUM_TYPE:
    case WellKnownName::T_IS_UNION_TYPE:
    case WellKnownName::T_IS_CLASS_TYPE:
    case WellKnownName::T_IS_FUNCTION_TYPE:
    case WellKnownName::T_IS_REFLECTION_TYPE:
    case WellKnownName::T_IS_REFERENCE_TYPE:
    case WellKnownName::T_IS_ARITHMETIC_TYPE:
    case WellKnownName::T_IS_FUNDAMENTAL_TYPE:
    case WellKnownName::T_IS_OBJECT_TYPE:
    case WellKnownName::T_IS_SCALAR_TYPE:
    case WellKnownName::T_IS_COMPOUND_TYPE:
    case WellKnownName::T_IS_MEMBER_POINTER_TYPE:
    case WellKnownName::T_IS_CONST_TYPE:
    case WellKnownName::T_IS_VOLATILE_TYPE:
      result = control->getBoolType();
      break;
    case WellKnownName::T_EXTENT:
      parameterCount = 2;
      [[fallthrough]];
    case WellKnownName::T_SIZE_OF:
    case WellKnownName::T_ALIGNMENT_OF:
    case WellKnownName::T_BIT_SIZE_OF:
    case WellKnownName::T_RANK:
    case WellKnownName::T_TUPLE_SIZE:
    case WellKnownName::T_VARIANT_SIZE:
      result = control->getSizeType();
      break;
    default:
      return WellKnownName::T_NONE;
  }
  if (type->returnType() != result ||
      type->parameterTypes().size() != parameterCount)
    return WellKnownName::T_NONE;
  for (std::size_t i = 0; i < parameterCount; ++i) {
    const Type* expected = control->getBuiltinMetaInfoType();
    if (name == WellKnownName::T_EXTENT && i == 1)
      expected = control->getUnsignedIntType();
    if (name == WellKnownName::T_SYMBOL_OF ||
        name == WellKnownName::T_U8SYMBOL_OF) {
      if (!unit_->typeTraits().is_enum(type->parameterTypes()[i]) ||
          !unit_->typeTraits().is_std_meta_type(type->parameterTypes()[i],
                                                "operators"))
        return WellKnownName::T_NONE;
      continue;
    }
    if (i == 1 && name != WellKnownName::T_EXTENT &&
        (name == WellKnownName::T_IS_ACCESSIBLE ||
         name == WellKnownName::T_HAS_INACCESSIBLE_NONSTATIC_DATA_MEMBERS ||
         name == WellKnownName::T_HAS_INACCESSIBLE_BASES ||
         name == WellKnownName::T_HAS_INACCESSIBLE_SUBOBJECTS ||
         name == WellKnownName::T_MEMBERS_OF ||
         name == WellKnownName::T_BASES_OF ||
         name == WellKnownName::T_STATIC_DATA_MEMBERS_OF ||
         name == WellKnownName::T_NONSTATIC_DATA_MEMBERS_OF ||
         name == WellKnownName::T_SUBOBJECTS_OF)) {
      if (!type_cast<ClassType>(type->parameterTypes()[i]) ||
          !unit_->typeTraits().is_std_meta_type(type->parameterTypes()[i],
                                                "access_context"))
        return WellKnownName::T_NONE;
      continue;
    }
    if (type->parameterTypes()[i] != expected) return WellKnownName::T_NONE;
  }
  return name;
}

auto Reflection::type(const Meta& reflection) -> const Type* {
  if (auto type = std::get_if<const Type*>(&reflection.value)) return *type;
  auto symbol = std::get_if<const Symbol*>(&reflection.value);
  if (!symbol || !*symbol) return nullptr;
  auto alias = symbol_cast<TypeAliasSymbol>(const_cast<Symbol*>(*symbol));
  return alias && !alias->isTemplatePattern() ? alias->type() : nullptr;
}

void Reflection::throwFailure(FunctionSymbol* function,
                              ScopeSymbol* evaluationScope,
                              ASTInterpreter* interpreter) const {
  auto type = standardMetaType(unit_, "exception");
  if (!type || !unit_->typeTraits().has_constexpr_standard_exception()) {
    interpreter->abortEvaluation();
    return;
  }
  auto control = unit_->control();
  auto arena = unit_->arena();
  auto text = std::format("invalid reflection in call to '{}'",
                          to_diag_string(function->name()));
  auto literal = control->utf8StringLiteral("u8" + quoteStringLiteral(text));
  auto what = ConstExpressionAST::create(arena);
  what->type = control->getPointerType(
      control->getQualType(control->getChar8Type(), CvQualifiers::kConst));
  what->valueCategory = ValueCategory::kPrValue;
  what->constValue =
      arena->make<ConstValue>(std::make_shared<ConstAddress>(literal));
  auto from = reflectionConstant(
      unit_, std::make_shared<Meta>(visit(ReflectEntity{}, function)));
  auto arguments = ParenInitializerAST::create(arena);
  arguments->expressionList = make_list_node<ExpressionAST>(arena, what);
  arguments->expressionList->next = make_list_node<ExpressionAST>(arena, from);
  PreparedInitialization initialization{
      unit_, evaluationScope, type, arguments,
      InitializationKind::kDirectInitialization};
  auto value = initialization.value(interpreter);
  if (!value) {
    interpreter->abortEvaluation();
    return;
  }
  interpreter->throwException(
      std::make_shared<ConstStorage>(type, std::move(*value)), type);
}

auto Reflection::parameters(const ConstValue& argument) const
    -> std::optional<std::vector<Meta>> {
  auto reflection = reflectedValue(argument);
  if (!reflection) return std::nullopt;
  FunctionSymbol* function = nullptr;
  if (auto symbol = std::get_if<const Symbol*>(&reflection->value))
    function = symbol_cast<FunctionSymbol>(const_cast<Symbol*>(*symbol));
  const Type* type = Reflection::type(*reflection);
  if (function) {
    if (function->isTemplatePattern()) return std::nullopt;
    type = function->type();
  }
  auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return std::nullopt;
  std::vector<Meta> result;
  const auto& types = functionType->parameterTypes();
  result.reserve(types.size());
  for (std::uint32_t index = 0; index < types.size(); ++index) {
    if (function) {
      auto canonical = symbol_cast<FunctionSymbol>(function->canonical());
      result.push_back(Meta{Meta::Parameter{canonical, index}});
    } else {
      result.push_back(Meta{types[index]});
    }
  }
  return result;
}

auto Reflection::call(WellKnownName builtin,
                      std::span<const ConstValue> arguments,
                      ScopeSymbol* evaluationScope, ASTInterpreter* interpreter,
                      std::shared_ptr<ConstObject> object,
                      FunctionSymbol* function) const
    -> std::optional<ConstValue> {
  if (isScopeQuery(builtin))
    return scopeQuery(unit_, builtin, arguments, evaluationScope);
  if (builtin == WellKnownName::T__S_EXCEPTION_CVT_TO_UTF8 ||
      builtin == WellKnownName::T__S_EXCEPTION_CVT_FROM_UTF8)
    return transcodedText(unit_, builtin, arguments, evaluationScope,
                          interpreter, function);
  if (rangeQuerySignature(builtin, unit_->control()))
    return rangeQuery(unit_, builtin, arguments, evaluationScope, interpreter,
                      function);
  if (builtin == WellKnownName::T_TUPLE_ELEMENT ||
      builtin == WellKnownName::T_VARIANT_ALTERNATIVE)
    return indexedTrait(unit_, builtin, arguments);
  if (builtin == WellKnownName::T_TYPE_ORDER)
    return typeOrder(unit_, arguments);
  if (arguments.empty() || arguments.size() > 2) return std::nullopt;
  if (isResultProducer(builtin)) {
    if (!function || arguments.size() != 1) return std::nullopt;
    if (builtin != WellKnownName::T_REFLECT_CONSTANT)
      return reflectReferent(unit_, function, arguments.front());
    return reflectConstantValue(
        unit_, producerArgumentType(function), arguments.front(),
        interpreter ? interpreter->evaluationPoint() : SourceLocation{},
        evaluationScope);
  }
  if (builtin == WellKnownName::T_VIA)
    return accessContextVia(unit_, arguments.front(), interpreter, object);
  if (builtin == WellKnownName::T_SYMBOL_OF ||
      builtin == WellKnownName::T_U8SYMBOL_OF)
    return symbolOf(unit_, arguments.front(),
                    builtin == WellKnownName::T_U8SYMBOL_OF);
  if (builtin == WellKnownName::T_IS_STRING_LITERAL) {
    if (arguments.size() != 1) return std::nullopt;
    return isStringLiteral(arguments.front());
  }
  auto value = reflectedValue(arguments.front());
  if (!value) return std::nullopt;
  const auto& reflection = *value;
  auto traits = unit_->typeTraits();
  auto symbolValue = std::get_if<const Symbol*>(&reflection.value);
  auto symbol = symbolValue ? const_cast<Symbol*>(*symbolValue) : nullptr;
  if (!symbol) {
    if (auto type = std::get_if<const Type*>(&reflection.value); type && *type)
      symbol = traits.entity_of(*type);
  }
  if (arguments.size() == 2 && isAccessContextQuery(builtin))
    return accessContextQuery(unit_, builtin, reflection, symbol, arguments[1],
                              evaluationScope, interpreter);
  if (auto predicate = entityPredicate(traits, builtin, symbol))
    return *predicate;
  switch (builtin) {
    case WellKnownName::T_DISPLAY_STRING_OF:
    case WellKnownName::T_U8DISPLAY_STRING_OF:
      return textResult(unit_->control(), displayString(reflection),
                        builtin == WellKnownName::T_U8DISPLAY_STRING_OF);
    case WellKnownName::T_OPERATOR_OF:
      return operatorOf(unit_, symbol);
    case WellKnownName::T_HAS_IDENTIFIER:
      return identifierOf(traits, reflection, symbol,
                          interpreter ? interpreter->evaluationPoint()
                                      : SourceLocation{})
          .has_value();
    case WellKnownName::T_IDENTIFIER_OF:
    case WellKnownName::T_U8IDENTIFIER_OF: {
      auto text = identifierOf(
          traits, reflection, symbol,
          interpreter ? interpreter->evaluationPoint() : SourceLocation{});
      if (!text) return std::nullopt;
      return textResult(unit_->control(), *text,
                        builtin == WellKnownName::T_U8IDENTIFIER_OF);
    }
    case WellKnownName::T_SOURCE_LOCATION_OF:
      if (!function) return std::nullopt;
      return sourceLocationOf(unit_, interpreter, function, reflection, symbol);
    case WellKnownName::T_IS_COMPLETE_TYPE: {
      auto type = Reflection::type(reflection);
      return type && traits.is_complete(type);
    }
    case WellKnownName::T_IS_ENUMERABLE_TYPE:
      return traits.is_enumerable(Reflection::type(reflection));
    case WellKnownName::T_PARAMETERS_OF: {
      auto values = parameters(arguments.front());
      if (!values) return std::nullopt;
      return reflectionList(unit_, std::move(*values));
    }
    case WellKnownName::T_ENUMERATORS_OF: {
      auto enumerators = traits.enumerators_of(Reflection::type(reflection));
      if (!enumerators || !symbol) return std::nullopt;
      return entityList(unit_, *enumerators);
    }
    case WellKnownName::T_PARENT_OF:
      return parentOf(traits, reflection, symbol);
    case WellKnownName::T_OFFSET_OF:
      return offsetOf(unit_, symbol);
    case WellKnownName::T_SIZE_OF:
    case WellKnownName::T_ALIGNMENT_OF:
    case WellKnownName::T_BIT_SIZE_OF:
      return layoutQuery(unit_, builtin, reflection, symbol);
    case WellKnownName::T_VARIABLE_OF:
      return variableOf(reflection, evaluationScope);
    case WellKnownName::T_IS_FUNCTION_PARAMETER:
      return std::holds_alternative<Meta::Parameter>(reflection.value);
    case WellKnownName::T_HAS_DEFAULT_ARGUMENT:
      return hasDefaultArgument(reflection);
    case WellKnownName::T_IS_EXPLICIT_OBJECT_PARAMETER:
      return isExplicitObjectParameter(reflection);
    case WellKnownName::T_HAS_PARENT:
      return std::holds_alternative<Meta::Parameter>(reflection.value) ||
             traits.has_parent(symbol);
    case WellKnownName::T_IS_VALUE:
      return std::holds_alternative<Meta::ConstExpr>(reflection.value);
    case WellKnownName::T_IS_OBJECT:
      return std::holds_alternative<std::shared_ptr<ConstAddress>>(
          reflection.value);
    case WellKnownName::T_OBJECT_OF:
      return objectOf(unit_, reflection, symbol, evaluationScope, interpreter);
    case WellKnownName::T_TEMPLATE_ARGUMENTS_OF:
      return templateArgumentReflections(unit_, symbol);
    case WellKnownName::T_HAS_TEMPLATE_ARGUMENTS:
      return traits.has_template_arguments(symbol);
    case WellKnownName::T_TEMPLATE_OF:
      if (!traits.has_template_arguments(symbol)) return std::nullopt;
      return std::make_shared<Meta>(
          visit(ReflectEntity{}, primary_template_of(symbol)));
    case WellKnownName::T_IS_CONST:
    case WellKnownName::T_IS_VOLATILE:
    case WellKnownName::T_IS_LVALUE_REFERENCE_QUALIFIED:
    case WellKnownName::T_IS_RVALUE_REFERENCE_QUALIFIED:
      return qualifierQuery(traits, builtin, reflection);
    case WellKnownName::T_IS_NOEXCEPT:
    case WellKnownName::T_IS_VARARG_FUNCTION:
      return functionTypeQuery(unit_, builtin, reflection, symbol);
    case WellKnownName::T_TYPE_OF:
    case WellKnownName::T_RETURN_TYPE_OF:
      return typeOf(traits, builtin, reflection);
    case WellKnownName::T_DEALIAS:
      if (traits.is_type_alias(symbol))
        return std::make_shared<Meta>(symbol->type());
      if (auto alias = symbol_cast<NamespaceAliasSymbol>(symbol))
        return std::make_shared<Meta>(
            static_cast<const Symbol*>(alias->namespaceSymbol()));
      return arguments.front();
    case WellKnownName::T_CONSTANT_OF: {
      if (!interpreter) return std::nullopt;
      auto constant = constantOf(unit_, evaluationScope, interpreter,
                                 std::get<std::shared_ptr<Meta>>(arguments[0]));
      if (!constant) return std::nullopt;
      return constant;
    }
    case WellKnownName::T_EXTRACT:
      if (!function) return std::nullopt;
      return extract(unit_, evaluationScope, interpreter, function,
                     std::get<std::shared_ptr<Meta>>(arguments[0]), symbol);
    case WellKnownName::T_TUPLE_SIZE:
    case WellKnownName::T_VARIANT_SIZE: {
      auto reflected = type(reflection);
      if (!reflected) return std::nullopt;
      const std::array<const Type*, 1> types{reflected};
      return associatedTraitValue(unit_, builtin, types);
    }
    case WellKnownName::T_IS_TYPE:
      return type(reflection) != nullptr;
    case WellKnownName::T_IS_TYPE_ALIAS:
      return traits.is_type_alias(symbol);
    default:
      break;
  }
  auto reflected = type(reflection);
  if (!reflected) return std::nullopt;
  if (arguments.size() == 2)
    return typeRelation(traits, builtin, reflected, arguments[1], interpreter);
  if (!hasCompleteOperand(traits, builtin, reflected)) {
    if (interpreter) interpreter->abortEvaluation();
    return std::nullopt;
  }
  if (auto property = typeProperty(traits, builtin, reflected))
    return *property;
  if (builtin == WellKnownName::T_RANK) return ConstInt{traits.rank(reflected)};
  auto transformed = typeTransformation(traits, builtin, reflected);
  if (!transformed) return std::nullopt;
  return std::make_shared<Meta>(transformed);
}

auto Reflection::evaluate(ExpressionAST* expression) const
    -> std::optional<ConstValue> {
  return expression ? visit(Reflect{unit_}, expression) : std::nullopt;
}

auto Reflection::compare(TokenKind op, const ConstValue& left,
                         const ConstValue& right) -> std::optional<ConstValue> {
  auto a = std::get_if<std::shared_ptr<Meta>>(&left);
  auto b = std::get_if<std::shared_ptr<Meta>>(&right);
  if (!a || !b || !*a || !*b) return std::nullopt;
  if (op == TokenKind::T_EQUAL_EQUAL) return equal(**a, **b);
  if (op == TokenKind::T_EXCLAIM_EQUAL) return !equal(**a, **b);
  return std::nullopt;
}

auto Reflection::hash(const Meta& reflection) -> std::size_t {
  if (auto type = std::get_if<const Type*>(&reflection.value))
    return std::hash<const Type*>{}(*type);
  if (auto symbol = std::get_if<const Symbol*>(&reflection.value))
    return std::hash<const Symbol*>{}(*symbol);
  if (auto parameter = std::get_if<Meta::Parameter>(&reflection.value))
    return std::hash<const FunctionSymbol*>{}(parameter->function) ^
           std::hash<std::uint32_t>{}(parameter->index);
  if (auto object =
          std::get_if<std::shared_ptr<ConstAddress>>(&reflection.value))
    return *object ? (*object)->objectHash() : 0;
  const auto& value = std::get<Meta::ConstExpr>(reflection.value);
  auto seed = std::hash<const Type*>{}(value.type);
  hash_combine(seed, hash_value(value.value).value_or(0));
  return seed;
}

auto Reflection::equal(const Meta& left, const Meta& right) -> bool {
  if (left.value.index() != right.value.index()) return false;
  if (auto type = std::get_if<const Type*>(&left.value))
    return *type == std::get<const Type*>(right.value);
  if (auto symbol = std::get_if<const Symbol*>(&left.value))
    return *symbol == std::get<const Symbol*>(right.value);
  if (auto parameter = std::get_if<Meta::Parameter>(&left.value))
    return *parameter == std::get<Meta::Parameter>(right.value);
  if (auto object = std::get_if<std::shared_ptr<ConstAddress>>(&left.value)) {
    const auto& other = std::get<std::shared_ptr<ConstAddress>>(right.value);
    return *object && other && (*object)->sameObject(*other);
  }
  const auto& a = std::get<Meta::ConstExpr>(left.value);
  const auto& b = std::get<Meta::ConstExpr>(right.value);
  return a.type == b.type && equivalent_values(a.value, b.value);
}

}  // namespace cxx
