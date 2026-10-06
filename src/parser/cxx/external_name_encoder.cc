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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <set>
#include <span>

namespace cxx {
namespace {
[[nodiscard]] auto enclosing_class_or_namespace(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  auto parent = symbol->parent();
  if (!parent || !parent->isClassOrNamespace()) return nullptr;
  return parent;
}

[[nodiscard]] auto is_unmangled_main(FunctionSymbol* function) -> bool {
  auto id = name_cast<Identifier>(function->name());
  return id && id->name() == "main" && is_global_namespace(function->parent());
}

[[nodiscard]] auto has_global_qualifier(NestedNameSpecifierAST* nns) -> bool {
  while (nns) {
    if (ast_cast<GlobalNestedNameSpecifierAST>(nns)) return true;
    if (auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns)) {
      nns = simple->nestedNameSpecifier;
      continue;
    }
    if (auto templ = ast_cast<TemplateNestedNameSpecifierAST>(nns)) {
      nns = templ->nestedNameSpecifier;
      continue;
    }
    return false;
  }
  return false;
}

[[nodiscard]] auto encodes_return_type(FunctionSymbol* function) -> bool {
  if (!function->isSpecialization()) return false;
  if (function->templateArguments().empty()) return false;
  if (function->isConstructor()) return false;
  if (name_cast<DestructorId>(function->name())) return false;
  if (name_cast<ConversionFunctionId>(function->name())) return false;
  return true;
}

[[nodiscard]] auto written_declaration(Symbol* symbol) -> Symbol* {
  while (symbol && symbol->instantiationPattern())
    symbol = symbol->instantiationPattern();
  return symbol;
}

[[nodiscard]] auto is_overloadable_template(Symbol* templateSymbol) -> bool {
  if (!symbol_cast<FunctionSymbol>(templateSymbol)) return false;
  auto classSymbol =
      symbol_cast<ClassSymbol>(enclosing_class_or_namespace(templateSymbol));
  return !classSymbol || !classSymbol->isClosureType();
}

[[nodiscard]] auto written_trailing_requires_function(FunctionSymbol* function)
    -> FunctionSymbol* {
  Symbol* declared = function;
  if (auto primary = function->primaryTemplateSymbol();
      primary && function->isSpecialization()) {
    declared = primary;
  }
  return symbol_cast<FunctionSymbol>(written_declaration(declared));
}

[[nodiscard]] auto is_synthesized_template_parameter(
    TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter))
    return typeParameter->isSynthesized;
  if (auto constrained = ast_cast<ConstraintTypeParameterAST>(parameter))
    return constrained->isSynthesized;
  return false;
}

[[nodiscard]] auto placeholder_type_constraint(TemplateParameterAST* parameter)
    -> TypeConstraintAST* {
  auto nonType = ast_cast<NonTypeTemplateParameterAST>(parameter);
  if (!nonType || !nonType->declaration) return nullptr;
  for (auto specifier : ListView{nonType->declaration->typeSpecifierList}) {
    if (auto placeholder = ast_cast<PlaceholderTypeSpecifierAST>(specifier))
      return placeholder->typeConstraint;
  }
  return nullptr;
}

[[nodiscard]] auto designates_subobject(const TypeTraits& traits,
                                        const Type* pointee,
                                        const ConstAddress& address) -> bool {
  if (address.offset() != 0) return true;
  auto entityType = address.symbol()->type();
  if (traits.is_array(entityType)) return true;
  return !traits.is_same(traits.remove_cv(pointee),
                         traits.remove_cv(entityType));
}

[[nodiscard]] auto mangling_parent(Symbol* symbol) -> Symbol* {
  auto parent = enclosing_class_or_namespace(symbol);

  if (auto function = symbol_cast<FunctionSymbol>(symbol);
      function && function->isFriend()) {
    while (parent && parent->isClass()) {
      parent = enclosing_class_or_namespace(parent);
    }
  }

  return parent;
}

[[nodiscard]] auto unary_builtin_name(UnaryBuiltinTypeKind kind)
    -> std::string_view {
  switch (kind) {
#define PROCESS_UNARY_BUILTIN(id, name) \
  case UnaryBuiltinTypeKind::T_##id:    \
    return name;
    FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(PROCESS_UNARY_BUILTIN)
#undef PROCESS_UNARY_BUILTIN
    default:
      return {};
  }
}

[[nodiscard]] auto has_complete_signature(const FunctionType* type) -> bool {
  if (!type) return false;
  if (!type->returnType()) return false;
  for (auto param : type->parameterTypes()) {
    if (!param) return false;
  }
  return true;
}

[[nodiscard]] auto has_complete_signature(Symbol* function) -> bool {
  return function &&
         has_complete_signature(type_cast<FunctionType>(function->type()));
}

[[nodiscard]] auto signature_function(FunctionSymbol* function)
    -> FunctionSymbol* {
  if (auto inherited = function->inheritedConstructorOrigin()) {
    if (auto primary =
            symbol_cast<FunctionSymbol>(inherited->primaryTemplateSymbol());
        has_complete_signature(primary)) {
      return primary;
    }
    if (has_complete_signature(inherited)) return inherited;
  }
  if (function->isSpecialization()) {
    if (auto primary =
            symbol_cast<FunctionSymbol>(function->primaryTemplateSymbol());
        has_complete_signature(primary)) {
      return primary;
    }
  }
  return function;
}

[[nodiscard]] auto template_name(Symbol* symbol) -> Symbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (classSymbol->isSpecialization())
      return classSymbol->primaryTemplateSymbol();
    if (classSymbol->templateParameters()) return classSymbol;
  } else if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    if (functionSymbol->isSpecialization())
      return functionSymbol->primaryTemplateSymbol();
  } else if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    if (variableSymbol->isSpecialization())
      return variableSymbol->primaryTemplateSymbol();
  }
  return nullptr;
}

[[nodiscard]] auto as_written_conversion_type(Symbol* symbol,
                                              const ConversionFunctionId* name)
    -> const Type* {
  auto pattern = template_name(symbol);
  if (!pattern) return name->type();
  auto patternName = name_cast<ConversionFunctionId>(pattern->name());
  if (!patternName) return name->type();
  return patternName->type();
}

[[nodiscard]] auto dependent_prefix_type_param(NestedNameSpecifierAST* nns)
    -> Symbol* {
  auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns);
  if (!simple) return nullptr;
  if (simple->nestedNameSpecifier) return nullptr;
  if (symbol_cast<TypeParameterSymbol>(simple->symbol) ||
      symbol_cast<TemplateTypeParameterSymbol>(simple->symbol)) {
    return simple->symbol;
  }
  return nullptr;
}

struct EnclosingQualifier {
  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast) const
      -> NestedNameSpecifierAST* {
    return ast->nestedNameSpecifier;
  }

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast) const
      -> NestedNameSpecifierAST* {
    return ast->nestedNameSpecifier;
  }

  [[nodiscard]] auto operator()(NestedNameSpecifierAST*) const
      -> NestedNameSpecifierAST* {
    return nullptr;
  }
};

[[nodiscard]] auto outermost_qualifier(NestedNameSpecifierAST* nns)
    -> NestedNameSpecifierAST* {
  while (auto enclosing = visit(EnclosingQualifier{}, nns)) nns = enclosing;
  return nns;
}

[[nodiscard]] auto parameter_pack_arguments(Symbol* templateName,
                                            std::size_t argumentCount)
    -> std::optional<std::pair<std::size_t, std::size_t>> {
  auto parameters = template_parameters_of(templateName);
  if (!parameters) return std::nullopt;

  const auto& members = parameters->members();
  auto pack = std::ranges::find_if(members, is_template_parameter_pack);
  if (pack == members.end()) return std::nullopt;

  const auto first = static_cast<std::size_t>(pack - members.begin());
  const auto trailing = static_cast<std::size_t>(members.end() - pack - 1);
  if (argumentCount < first + trailing) return std::nullopt;
  return std::pair{first, argumentCount - trailing};
}

[[nodiscard]] auto resolved_qualifier_scope(Symbol* symbol) -> Symbol* {
  if (symbol_cast<NamespaceSymbol>(symbol)) return symbol;
  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol || classSymbol->templateParameters()) return nullptr;
  return classSymbol;
}

[[nodiscard]] auto type_template_name(Symbol* symbol) -> Symbol* {
  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol || !classSymbol->templateParameters()) return nullptr;
  return classSymbol;
}

[[nodiscard]] auto designates_declared_entity(Symbol* symbol) -> bool {
  if (template_parameters_of(symbol)) return false;
  if (symbol_cast<VariableSymbol>(symbol)) return true;
  if (symbol_cast<FunctionSymbol>(symbol)) return true;
  auto field = symbol_cast<FieldSymbol>(symbol);
  return field && field->isStatic();
}

[[nodiscard]] auto designates_entity_or_member(ExpressionAST* expression)
    -> bool {
  if (ast_cast<IdExpressionAST>(expression)) return true;
  return ast_cast<MemberExpressionAST>(expression) != nullptr;
}

struct IsExpressionPrimary {
  [[nodiscard]] auto operator()(NestedExpressionAST* ast) const -> bool {
    return visit(*this, ast->expression);
  }

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast) const -> bool {
    return visit(*this, ast->expression);
  }

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) const -> bool {
    return visit(*this, ast->expression);
  }

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(IntLiteralExpressionAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(IdExpressionAST* ast) const -> bool {
    auto symbol = resolve_using_declaration(ast->symbol);
    if (auto enumerator = symbol_cast<EnumeratorSymbol>(symbol))
      return enumerator->value().has_value();
    return designates_declared_entity(symbol);
  }

  [[nodiscard]] auto operator()(ExpressionAST*) const -> bool { return false; }
};

[[nodiscard]] auto is_increment_or_decrement(TokenKind op) -> bool {
  return op == TokenKind::T_PLUS_PLUS || op == TokenKind::T_MINUS_MINUS;
}

[[nodiscard]] auto floating_digits(double value,
                                   const FloatingPointFormat& format)
    -> std::string {
  const int bitCount = 1 + format.exponentBits + format.fractionBits();
  const auto bits = format.representation(value);
  std::string digits;
  for (int shift = bitCount - 4; shift >= 0; shift -= 4) {
    digits += "0123456789abcdef"[static_cast<int>((bits >> shift) & 0xf)];
  }
  return digits;
}

[[nodiscard]] auto cpp_cast_operator_name(TokenKind castOp)
    -> std::string_view {
  switch (castOp) {
    case TokenKind::T_STATIC_CAST:
      return "sc";
    case TokenKind::T_DYNAMIC_CAST:
      return "dc";
    case TokenKind::T_CONST_CAST:
      return "cc";
    case TokenKind::T_REINTERPRET_CAST:
      return "rc";
    default:
      return {};
  }
}

[[nodiscard]] auto names_unresolved_type(NestedNameSpecifierAST* nns) -> bool {
  if (dependent_prefix_type_param(nns)) return true;
  return ast_cast<DecltypeNestedNameSpecifierAST>(nns) != nullptr;
}

[[nodiscard]] auto is_abi_std_namespace(Symbol* symbol) -> bool {
  if (!symbol_cast<NamespaceSymbol>(symbol)) return false;

  auto parent = enclosing_class_or_namespace(symbol);
  if (!parent) return false;

  if (!is_global_namespace(parent)) return false;

  return well_known_name(symbol->name()) == WellKnownName::T_STD;
}

[[nodiscard]] auto std_class_identifier(Symbol* symbol) -> const Identifier* {
  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol) return nullptr;
  if (!is_abi_std_namespace(mangling_parent(classSymbol))) return nullptr;
  return name_cast<Identifier>(classSymbol->name());
}

[[nodiscard]] auto std_template_abbreviation(Symbol* symbol)
    -> std::string_view {
  auto identifier = std_class_identifier(symbol);
  if (!identifier) return {};
  if (identifier->name() == "allocator") return "Sa";
  if (identifier->name() == "basic_string") return "Sb";
  return {};
}

[[nodiscard]] auto std_specialization_arguments(const Type* type,
                                                std::string_view name)
    -> std::span<const TemplateArgument> {
  auto classType = type_cast<ClassType>(type);
  if (!classType) return {};
  auto classSymbol = classType->symbol();
  if (!classSymbol->isSpecialization()) return {};
  auto identifier = std_class_identifier(template_name(classSymbol));
  if (!identifier) return {};
  if (identifier->name() != name) return {};
  return classSymbol->templateArguments();
}

[[nodiscard]] auto is_type_argument(const TemplateArgument& argument,
                                    const Type* type) -> bool {
  return template_argument_type(argument) == type;
}

[[nodiscard]] auto is_std_specialization_of_char(
    const TemplateArgument& argument, std::string_view name,
    const Type* charType) -> bool {
  auto type = template_argument_type(argument);
  if (!type) return false;
  auto args = std_specialization_arguments(type, name);
  if (args.size() != 1) return false;
  return is_type_argument(args[0], charType);
}

[[nodiscard]] auto has_char_stream_arguments(
    std::span<const TemplateArgument> args, const Type* charType) -> bool {
  if (args.size() != 2) return false;
  if (!is_type_argument(args[0], charType)) return false;
  return is_std_specialization_of_char(args[1], "char_traits", charType);
}

[[nodiscard]] auto std_type_abbreviation(Control* control, const Type* type)
    -> std::string_view {
  auto charType = static_cast<const Type*>(control->getCharType());

  if (auto args = std_specialization_arguments(type, "basic_string");
      !args.empty()) {
    if (args.size() != 3) return {};
    if (!is_type_argument(args[0], charType)) return {};
    if (!is_std_specialization_of_char(args[1], "char_traits", charType)) {
      return {};
    }
    if (!is_std_specialization_of_char(args[2], "allocator", charType)) {
      return {};
    }
    return "Ss";
  }

  if (auto args = std_specialization_arguments(type, "basic_istream");
      has_char_stream_arguments(args, charType)) {
    return "Si";
  }

  if (auto args = std_specialization_arguments(type, "basic_ostream");
      has_char_stream_arguments(args, charType)) {
    return "So";
  }

  if (auto args = std_specialization_arguments(type, "basic_iostream");
      has_char_stream_arguments(args, charType)) {
    return "Sd";
  }

  return {};
}

[[nodiscard]] auto needsInternalLinkageMarker(Symbol* symbol) -> bool {
  if (is_in_unnamed_namespace(symbol)) return false;
  return has_internal_linkage(symbol);
}

}  // namespace

struct ExternalNameEncoder::EncodeType {
  ExternalNameEncoder& encoder;

  auto operator()(const VoidType* type) -> bool {
    encoder.out("v");
    return false;
  }

  auto operator()(const NullptrType* type) -> bool {
    encoder.out("Dn");
    return false;
  }

  auto operator()(const DecltypeAutoType* type) -> bool {
    encoder.out("Dc");
    return false;
  }

  auto operator()(const AutoType* type) -> bool {
    if (auto constraint =
            std::exchange(encoder.placeholderConstraint_, nullptr)) {
      encoder.out("Dk");
      encoder.encodeTypeConstraint(constraint);
      return false;
    }
    encoder.out("Da");
    return false;
  }

  auto operator()(const BoolType* type) -> bool {
    encoder.out("b");
    return false;
  }

  auto operator()(const SignedCharType* type) -> bool {
    encoder.out("a");
    return false;
  }

  auto operator()(const ShortIntType* type) -> bool {
    encoder.out("s");
    return false;
  }

  auto operator()(const IntType* type) -> bool {
    encoder.out("i");
    return false;
  }

  auto operator()(const LongIntType* type) -> bool {
    encoder.out("l");
    return false;
  }

  auto operator()(const LongLongIntType* type) -> bool {
    encoder.out("x");
    return false;
  }

  auto operator()(const Int128Type* type) -> bool {
    encoder.out("n");
    return false;
  }

  auto operator()(const UnsignedCharType* type) -> bool {
    encoder.out("h");
    return false;
  }

  auto operator()(const UnsignedShortIntType* type) -> bool {
    encoder.out("t");
    return false;
  }

  auto operator()(const UnsignedIntType* type) -> bool {
    encoder.out("j");
    return false;
  }

  auto operator()(const UnsignedLongIntType* type) -> bool {
    encoder.out("m");
    return false;
  }

  auto operator()(const UnsignedLongLongIntType* type) -> bool {
    encoder.out("y");
    return false;
  }

  auto operator()(const UnsignedInt128Type* type) -> bool {
    encoder.out("o");
    return false;
  }

  auto operator()(const CharType* type) -> bool {
    encoder.out("c");
    return false;
  }

  auto operator()(const Char8Type* type) -> bool {
    encoder.out("Du");
    return false;
  }

  auto operator()(const Char16Type* type) -> bool {
    encoder.out("Ds");
    return false;
  }

  auto operator()(const Char32Type* type) -> bool {
    encoder.out("Di");
    return false;
  }

  auto operator()(const WideCharType* type) -> bool {
    encoder.out("w");
    return false;
  }

  auto operator()(const FloatType* type) -> bool {
    encoder.out("f");
    return false;
  }

  auto operator()(const DoubleType* type) -> bool {
    encoder.out("d");
    return false;
  }

  auto operator()(const LongDoubleType* type) -> bool {
    encoder.out("e");
    return false;
  }

  auto operator()(const Float16Type* type) -> bool {
    encoder.out("DF16_");
    return false;
  }

  auto operator()(const Float128Type* type) -> bool {
    encoder.out("g");
    return false;
  }

  auto operator()(const QualType* type) -> bool {
    encoder.encodeCvQualifiers(type->cvQualifiers());
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const BoundedArrayType* type) -> bool {
    encoder.out(std::format("A{}_", type->size()));
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const UnboundedArrayType* type) -> bool {
    encoder.out("A_");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const DecltypeType* type) -> bool {
    auto expression = type->expression();
    encoder.out(designates_entity_or_member(expression) ? "Dt" : "DT");
    encoder.encodeRequiredExpression(expression, "decltype operand");
    encoder.out("E");
    return true;
  }

  auto operator()(const PackExpansionType* type) -> bool {
    encoder.out("Dp");
    encoder.encodeType(type->pattern());
    return true;
  }

  auto operator()(const PointerType* type) -> bool {
    encoder.out("P");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const LvalueReferenceType* type) -> bool {
    encoder.out("R");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const RvalueReferenceType* type) -> bool {
    encoder.out("O");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const FunctionType* type) -> bool {
    encoder.encodeCvQualifiers(type->cvQualifiers());

    if (type->noexceptExpression()) {
      encoder.out("DO");
      if (!encoder.encodeExpression(type->noexceptExpression())) return false;
      encoder.out("E");
    } else if (type->isNoexcept())
      encoder.out("Do");

    encoder.out("F");

    encoder.encodeBareFunctionType(type, /*includeReturnType=*/true);

    if (type->refQualifier() == RefQualifier::kLvalue)
      encoder.out("R");
    else if (type->refQualifier() == RefQualifier::kRvalue)
      encoder.out("O");

    encoder.out("E");
    return true;
  }

  auto operator()(const ClassType* type) -> bool {
    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const EnumType* type) -> bool {
    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const ScopedEnumType* type) -> bool {
    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const MemberObjectPointerType* type) -> bool {
    encoder.out("M");
    encoder.encodeType(type->classType());
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const MemberFunctionPointerType* type) -> bool {
    encoder.out("M");
    encoder.encodeType(type->classType());
    encoder.encodeType(type->functionType());
    return true;
  }

  auto operator()(const NamespaceType* type) -> bool { return false; }

  auto operator()(const TypeParameterType* type) -> bool {
    encoder.encodeTemplateParamValue(type->depth(), type->index());
    return true;
  }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    encoder.encodeTemplateParamValue(type->depth(), type->index());
    return true;
  }

  auto operator()(const TemplateTypeParameterSpecializationType* type) -> bool;

  auto operator()(const UnresolvedNameType* type) -> bool {
    if (encoder.encodeDependentName(type->nestedNameSpecifier(),
                                    type->unqualifiedId())) {
      return true;
    }
    cxx_runtime_error(std::format("cannot mangle unresolved name type '{}'",
                                  to_string(type)));
  }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    encoder.out("A");

    if (isDependent(type->translationUnit(), type->size())) {
      if (!encoder.encodeExpression(type->size())) {
        cxx_runtime_error(std::format(
            "cannot mangle dependent array bound of '{}'", to_string(type)));
      }
    }

    encoder.out("_");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool {
    constexpr std::string_view name = "__underlying_type";
    auto typeId = type->typeId();
    if (!typeId || !typeId->type) {
      cxx_runtime_error("cannot mangle unresolved underlying type");
    }
    encoder.out(std::format("u{}{}I", name.size(), name));
    encoder.encodeType(typeId->type);
    encoder.out("E");
    return true;
  }

  auto operator()(const UnresolvedBuiltinType* type) -> bool {
    auto name = unary_builtin_name(type->builtinKind());
    auto typeId = type->typeId();
    if (name.empty() || !typeId || !typeId->type) {
      cxx_runtime_error("cannot mangle unresolved builtin type");
    }
    encoder.out(std::format("u{}{}I", name.size(), name));
    encoder.encodeType(typeId->type);
    encoder.out("E");
    return true;
  }

  auto operator()(const OverloadSetType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const BuiltinVaListType* type) -> bool {
    encoder.out("Pc");
    return true;
  }

  auto operator()(const BuiltinMetaInfoType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return true;
  }

  auto operator()(const SveType* type) -> bool {
    const auto& name = Token::spell(type->sveKind());
    encoder.out(std::format("u{}{}", name.size(), name));
    return true;
  }

  auto operator()(const BitIntType* type) -> bool {
    encoder.out(std::format("DB{}_", type->numBits()));
    return false;
  }

  auto operator()(const UnsignedBitIntType* type) -> bool {
    encoder.out(std::format("DU{}_", type->numBits()));
    return false;
  }

  auto operator()(const UnresolvedBitIntType* type) -> bool {
    cxx_runtime_error(std::format("cannot mangle unresolved bit-int type '{}'",
                                  to_string(type)));
  }

  auto operator()(const VectorType* type) -> bool {
    encoder.out(std::format("Dv{}_", type->elementCount()));
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const UnresolvedVectorType* type) -> bool {
    cxx_runtime_error(std::format("cannot mangle unresolved vector type '{}'",
                                  to_string(type)));
  }

  auto operator()(const ComplexType* type) -> bool {
    encoder.out("C");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const AtomicType* type) -> bool {
    encoder.out("U7_Atomic");
    encoder.encodeType(type->elementType());
    return true;
  }
};

struct ExternalNameEncoder::EncodeTemplateParameterDeclaration {
  ExternalNameEncoder& encoder;
  TemplateParameterAST* written = nullptr;

  void operator()(TypenameTypeParameterAST* ast) const {
    if (ast->isPack) encoder.out("Tp");
    encoder.out("Ty");
  }

  void operator()(ConstraintTypeParameterAST* ast) const {
    if (ast->ellipsisLoc) encoder.out("Tp");
    encoder.out("Tk");
    auto writtenParameter = ast_cast<ConstraintTypeParameterAST>(written);
    encoder.encodeTypeConstraint(writtenParameter
                                     ? writtenParameter->typeConstraint
                                     : ast->typeConstraint);
  }

  void operator()(NonTypeTemplateParameterAST* ast) const {
    auto declaration = ast->declaration;
    if (!declaration) return;
    if (declaration->isPack) encoder.out("Tp");
    encoder.out("Tn");
    auto saved = std::exchange(encoder.placeholderConstraint_,
                               placeholder_type_constraint(written));
    encoder.encodeType(declaration->type);
    encoder.placeholderConstraint_ = saved;
  }

  void operator()(TemplateTypeParameterAST* ast) const {
    if (ast->isPack) encoder.out("Tp");
    encoder.out("Tt");
    auto writtenParameter = ast_cast<TemplateTypeParameterAST>(written);
    if (!writtenParameter) writtenParameter = ast;
    for (auto [inner, writtenInner] :
         std::views::zip(ListView{ast->templateParameterList},
                         ListView{writtenParameter->templateParameterList})) {
      encoder.encodeTemplateParameterDeclaration(inner, writtenInner);
    }
    encoder.out("E");
  }
};

struct ExternalNameEncoder::EncodeUnqualifiedName {
  ExternalNameEncoder& encoder;
  Symbol* symbol = nullptr;

  [[nodiscard]] auto needsTemplateParameterDeclaration(
      TemplateParameterAST* parameter, const TemplateArgument& argument) const
      -> bool {
    if (ast_cast<ConstraintTypeParameterAST>(parameter)) return true;

    if (auto nonType = ast_cast<NonTypeTemplateParameterAST>(parameter)) {
      return nonType->declaration &&
             hasDeducedOrDependentType(nonType->declaration->type);
    }

    auto templateParameter = ast_cast<TemplateTypeParameterAST>(parameter);
    if (!templateParameter) return false;

    auto sym = std::get_if<Symbol*>(&argument);
    if (!sym) return false;

    auto argumentTemplate = template_name_symbol(*sym);
    if (!argumentTemplate) return false;

    auto declaration = template_declaration_of(argumentTemplate);
    if (!declaration) return false;

    return !TemplateEquivalence{encoder.unit_}.same(
        templateParameter->templateParameterList,
        declaration->templateParameterList);
  }

  [[nodiscard]] auto hasDeducedOrDependentType(const Type* type) const -> bool {
    if (containsPlaceholderType(type)) return true;
    return encoder.unit_ && isDependent(encoder.unit_, type);
  }

  void encodeTemplateArguments(Symbol* symbol) {
    if (!symbol) return;
    if (symbol == encoder.templateNameOnly_) return;

    std::span<const TemplateArgument> args;
    Symbol* templateName = nullptr;

    if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
      args = classSymbol->templateArguments();
      if (classSymbol->isSpecialization()) {
        templateName = classSymbol->primaryTemplateSymbol();
      } else if (classSymbol->templateParameters()) {
        encodeTemplateParameters(classSymbol);
        return;
      }
    } else if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      args = functionSymbol->templateArguments();
      if (functionSymbol->isSpecialization())
        templateName = functionSymbol->primaryTemplateSymbol();
    } else if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
      args = variableSymbol->templateArguments();
      if (variableSymbol->isSpecialization())
        templateName = variableSymbol->primaryTemplateSymbol();
    }

    if (args.empty()) return;

    if (templateName && std_template_abbreviation(templateName).empty())
      encoder.enterSubstitution(templateName);

    encoder.out("I");

    auto templateSymbol = templateName ? templateName : symbol;
    auto declaration = template_declaration_of(templateSymbol);
    auto writtenDeclaration =
        template_declaration_of(written_declaration(templateSymbol));
    if (!writtenDeclaration) writtenDeclaration = declaration;

    std::vector<TemplateParameterAST*> parameters;
    if (declaration) {
      for (auto parameter : ListView{declaration->templateParameterList})
        parameters.push_back(parameter);
    }

    std::vector<TemplateParameterAST*> writtenParameters;
    if (writtenDeclaration) {
      for (auto parameter : ListView{writtenDeclaration->templateParameterList})
        writtenParameters.push_back(parameter);
    }

    const bool isOverloadable = is_overloadable_template(templateSymbol);

    for (std::size_t index = 0; index < args.size(); ++index) {
      const auto& arg = args[index];

      const Type* declaredType = nullptr;
      if (index < parameters.size()) {
        auto parameter = parameters[index];
        if (isOverloadable &&
            needsTemplateParameterDeclaration(parameter, arg)) {
          auto written = index < writtenParameters.size()
                             ? writtenParameters[index]
                             : parameter;
          encoder.encodeTemplateParameterDeclaration(parameter, written);
        }
        if (auto nonType = ast_cast<NonTypeTemplateParameterAST>(parameter);
            nonType && nonType->declaration) {
          declaredType = nonType->declaration->type;
        }
      }

      if (auto sym = std::get_if<Symbol*>(&arg)) {
        encodeTemplateArgumentSymbol(*sym);
      } else if (auto type = std::get_if<const Type*>(&arg)) {
        if (!*type) continue;
        encoder.encodeType(*type);
      } else if (auto val = std::get_if<ConstValue>(&arg)) {
        if (!declaredType) continue;
        encoder.encodeTemplateArgumentValue(declaredType, *val);
      } else if (auto exprArg = std::get_if<ExpressionAST*>(&arg)) {
        encodeDependentExpressionArgument(*exprArg);
      }
    }

    if (isOverloadable && writtenDeclaration)
      encoder.encodeRequiresClause(writtenDeclaration->requiresClause);

    encoder.out("E");
  }

  void encodeAbiTagsAndTemplateArguments(Symbol* symbol) {
    if (!symbol) return;
    encoder.encodeAbiTags(symbol);
    encodeTemplateArguments(symbol);
  }

  void encodeTemplateArgumentSymbol(Symbol* sym) {
    if (!sym) return;

    if (encoder.encodeTemplateTemplateArgument(sym)) return;

    if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
      encoder.out("J");
      for (auto element : pack->elements()) {
        encodeTemplateArgumentSymbol(element);
      }
      encoder.out("E");
      return;
    }

    if (auto parameter = symbol_cast<NonTypeParameterSymbol>(sym)) {
      encoder.out("X");
      encoder.encodeTemplateParamValue(parameter->depth(), parameter->index());
      encoder.out("E");
      return;
    }

    auto type = sym->type();

    if (auto var = symbol_cast<VariableSymbol>(sym)) {
      if (var->constValue().has_value() && type) {
        encoder.encodeTemplateArgumentValue(type, var->constValue().value());
        return;
      }
      if (!var->constValue().has_value() && var->initializer()) {
        encodeDependentExpressionArgument(var->initializer());
        return;
      }
    }

    if (!type) return;
    encoder.encodeType(type);
  }

  void encodeDependentExpressionArgument(ExpressionAST* expression) {
    encoder.out("X");
    encoder.encodeRequiredExpression(expression,
                                     "dependent template argument expression");
    encoder.out("E");
  }

  void encodeTemplateParameters(ClassSymbol* classSymbol) {
    encoder.enterSubstitution(classSymbol);

    encoder.out("I");

    for (auto member : classSymbol->templateParameters()->members()) {
      if (auto nonTypeParameter = symbol_cast<NonTypeParameterSymbol>(member)) {
        if (nonTypeParameter->isParameterPack()) {
          encoder.out("JXsp");
          encoder.encodeTemplateParamValue(nonTypeParameter->depth(),
                                           nonTypeParameter->index());
          encoder.out("EE");
          continue;
        }
        encoder.out("X");
        encoder.encodeTemplateParamValue(nonTypeParameter->depth(),
                                         nonTypeParameter->index());
        encoder.out("E");
        continue;
      }

      auto parameterType = member->type();
      if (!parameterType) continue;

      if (is_template_parameter_pack(member)) {
        encoder.out("JDp");
        encoder.encodeType(parameterType);
        encoder.out("E");
        continue;
      }

      encoder.encodeType(parameterType);
    }

    encoder.out("E");
  }

  auto encodeConstructor() -> bool {
    auto function = symbol_cast<FunctionSymbol>(symbol);
    if (!function || !function->isConstructor()) return false;

    if (auto inherited = function->inheritedConstructorOrigin()) {
      if (encoder.structorVariant_ == StructorVariant::Base)
        out("CI2");
      else
        out("CI1");
      auto base = enclosing_class_or_namespace(inherited);
      if (!base || !base->type()) {
        cxx_runtime_error("cannot mangle inherited constructor");
      }
      encoder.encodeType(base->type());
      encodeTemplateArguments(inherited);
      return true;
    }

    if (encoder.structorVariant_ == StructorVariant::Base)
      out("C2");
    else
      out("C1");
    encodeAbiTagsAndTemplateArguments(symbol);
    return true;
  }

  void operator()(const Identifier* id) {
    if (encodeConstructor()) return;

    if (needsInternalLinkageMarker(symbol)) out("L");

    out(std::format("{}{}", id->name().length(), id->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const OperatorId* name) {
    auto is_unary = [&] {
      auto function = symbol_cast<FunctionSymbol>(symbol);
      if (!function) {
        cxx_runtime_error(
            std::format("cannot encode operator '{}' for non-function symbol",
                        to_string(name)));
      }

      auto functionType = type_cast<FunctionType>(function->type());
      if (!functionType) {
        cxx_runtime_error(
            std::format("cannot encode operator '{}' for non-function type",
                        to_string(name)));
      }

      bool unary = false;
      switch (name->op()) {
        case TokenKind::T_PLUS:
        case TokenKind::T_MINUS:
        case TokenKind::T_AMP:
        case TokenKind::T_STAR: {
          auto argc = functionType->parameterTypes().size();
          if (argc == 0)
            unary = true;
          else if (argc == 1 &&
                   (!function->parent()->isClass() || function->isFriend()))
            unary = true;
          break;
        }

        default:
          break;
      }

      return unary;
    };

    const auto unary = is_unary();

    out(encoder.encodeOperatorName(name->op(), unary));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const DestructorId* name) {
    switch (encoder.structorVariant_) {
      case StructorVariant::Complete:
        out("D1");
        break;
      case StructorVariant::Base:
        out("D2");
        break;
      case StructorVariant::Deleting:
        out("D0");
        break;
    }
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const LiteralOperatorId* name) {
    out("li");
    encoder.out(std::format("{}{}", name->name().length(), name->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const ConversionFunctionId* name) {
    out("cv");
    encoder.encodeType(as_written_conversion_type(symbol, name));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const TemplateId* name) {
    auto baseId = name_cast<Identifier>(name->name());
    if (!baseId) {
      cxx_runtime_error(
          std::format("cannot encode template-id '{}'", to_string(name)));
    }

    out(std::format("{}{}", baseId->name().length(), baseId->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void out(std::string_view str) { encoder.out(str); }
};

ExternalNameEncoder::ExternalNameEncoder(TranslationUnit* unit) : unit_(unit) {}

auto ExternalNameEncoder::encode(Symbol* symbol, std::string_view suffix)
    -> std::string {
  encodingSymbol_ = symbol;
  std::string result;
  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    if (!hasExplicitStructorVariant_) {
      if (auto principal = functionSymbol->structorPrincipal()) {
        structorVariant_ = principal->deletingDtorVariant() == functionSymbol
                               ? StructorVariant::Deleting
                               : StructorVariant::Complete;
        functionSymbol = principal;
      } else if (functionSymbol->completeObjectVariant()) {
        structorVariant_ = StructorVariant::Base;
      }
    }
    result = encodeFunction(functionSymbol);
  } else {
    result = encodeData(symbol);
  }
  result.append(suffix);
  return result;
}

auto ExternalNameEncoder::encode(const Type* type) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  encodeType(type);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeVTable(ClassSymbol* classSymbol)
    -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZTV");
  encodeName(classSymbol);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeConstructionVTable(ClassSymbol* completeClass,
                                                   std::int64_t offset,
                                                   ClassSymbol* baseClass)
    -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZTC");
  encodeName(completeClass);
  if (offset < 0) {
    out("n");
    out(std::to_string(-offset));
  } else {
    out(std::to_string(offset));
  }
  out("_");
  encodeName(baseClass);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeVTT(ClassSymbol* classSymbol) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZTT");
  encodeName(classSymbol);

  std::swap(externalName, out_);
  return externalName;
}

namespace {
void appendCallOffsetNumber(std::string& name, std::int64_t value) {
  if (value < 0) name += 'n';
  name += std::to_string(value < 0 ? -value : value);
}

void appendCallOffset(std::string& name,
                      const VTableLayout::CallOffset& callOffset) {
  if (!callOffset.virtualOffset) {
    name += 'h';
    appendCallOffsetNumber(name, callOffset.nonVirtual);
    name += '_';
    return;
  }
  name += 'v';
  appendCallOffsetNumber(name, callOffset.nonVirtual);
  name += '_';
  appendCallOffsetNumber(name, callOffset.virtualOffset);
  name += '_';
}
}  // namespace

auto ExternalNameEncoder::encodeThunk(
    FunctionSymbol* target, const VTableLayout::CallOffset& thisAdjustment,
    const VTableLayout::CallOffset& returnAdjustment) -> std::string {
  auto encoding = encode(target);
  std::string name = returnAdjustment.isEmpty() ? "_ZT" : "_ZTc";
  appendCallOffset(name, thisAdjustment);
  if (!returnAdjustment.isEmpty()) appendCallOffset(name, returnAdjustment);
  name += std::string_view{encoding}.substr(2);
  return name;
}

auto ExternalNameEncoder::encodeGuardVariable(Symbol* symbol) -> std::string {
  encodingSymbol_ = symbol;
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZGV");
  encodeName(symbol);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeTypeInfo(const Type* type) -> std::string {
  return std::format("_ZTI{}", encode(type));
}

auto ExternalNameEncoder::encodeTypeInfoName(const Type* type) -> std::string {
  return std::format("_ZTS{}", encode(type));
}

auto ExternalNameEncoder::isUnmangledData(Symbol* symbol) -> bool {
  if (!is_global_namespace(enclosing_class_or_namespace(symbol))) return false;
  if (template_name(symbol)) return false;
  if (needsInternalLinkageMarker(symbol)) return false;
  return mangledAbiTags(symbol).empty();
}

auto ExternalNameEncoder::encodeData(Symbol* symbol) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);
  if (isUnmangledData(symbol)) {
    auto id = name_cast<Identifier>(symbol->name());
    out(id->name());
  } else {
    out("_Z");
    encodeName(symbol);
  }
  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeFunction(FunctionSymbol* function)
    -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  const auto id = name_cast<Identifier>(function->name());

  if (id &&
      (function->hasCLinkage() ||
       (id->name() == "main" && is_global_namespace(function->parent())))) {
    out(id->name());
  } else {
    out("_Z");
    encodeName(function);
    encodeFunctionSignature(function);
  }

  std::swap(externalName, out_);

  return externalName;
}

void ExternalNameEncoder::encodeName(Symbol* symbol) {
  if (encodeLocalName(symbol)) return;
  if (encodeNestedName(symbol)) return;
  if (encodeUnscopedName(symbol)) return;

  cxx_runtime_error(std::format("cannot encode name for symbol \'{}\'",
                                to_string(symbol->type(), symbol->name())));
}

auto ExternalNameEncoder::encodeTemplateTemplateArgument(Symbol* symbol)
    -> bool {
  auto templateName = template_name_symbol(symbol);
  if (!templateName || symbol_cast<TemplateTypeParameterSymbol>(templateName)) {
    return false;
  }
  encodeTemplateName(templateName);
  return true;
}

void ExternalNameEncoder::encodeTemplateName(Symbol* symbol) {
  auto templateName = template_name(symbol);
  if (templateName && encodeSubstitution(templateName)) return;
  auto saved = std::exchange(templateNameOnly_, symbol);
  encodeName(symbol);
  templateNameOnly_ = saved;
  if (templateName && std_template_abbreviation(templateName).empty())
    enterSubstitution(templateName);
}

void ExternalNameEncoder::encodeClosureTypeName(ClassSymbol* closure) {
  if (!closure_mangling_context(closure)) {
    auto name = std::format("$_{}", closure->closureDiscriminator());
    out(std::format("{}{}", name.length(), name));
    return;
  }

  out("Ul");
  if (auto callOperator = closure->functionCallOperator())
    encodeLambdaSig(callOperator);
  out("E");
  if (auto number = closure->closureDiscriminator())
    out(std::to_string(number - 1));
  out("_");
}

namespace {

[[nodiscard]] auto isUnnamedLocalType(Symbol* symbol) -> bool {
  if (symbol->name()) return false;
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
    return !classSymbol->isClosureType();
  return symbol->isEnumOrScopedEnum();
}

}  // namespace

auto ExternalNameEncoder::unnamedTypeIndex(Symbol* symbol) const -> int {
  if (!symbol) return -1;

  std::vector<Symbol*> unnamedTypes;
  std::set<ScopeSymbol*> visited;

  if (auto function = symbol->enclosingFunction()) {
    std::function<void(ScopeSymbol*)> collect;
    collect = [&](ScopeSymbol* scope) {
      if (!scope || !visited.insert(scope).second) return;
      for (auto member : scope->members()) {
        if (member->enclosingFunction() != function) continue;
        if (isUnnamedLocalType(member)) unnamedTypes.push_back(member);

        auto childScope = member->asScopeSymbol();
        if (!childScope || member->isFunction()) continue;
        collect(childScope);
      }
    };

    collect(function);
  } else if (auto context = symbol_cast<ScopeSymbol>(symbol->parent())) {
    for (auto member : context->members()) {
      if (isUnnamedLocalType(member)) unnamedTypes.push_back(member);
    }
  }

  std::ranges::sort(unnamedTypes, {}, &Symbol::location);

  auto it = std::ranges::find(unnamedTypes, symbol);
  if (it == unnamedTypes.end()) return -1;
  return static_cast<int>(it - unnamedTypes.begin());
}

void ExternalNameEncoder::encodeUnnamedTypeName(Symbol* symbol) {
  auto index = unnamedTypeIndex(symbol);
  if (index < 0) {
    cxx_runtime_error("cannot determine unnamed local type index");
  }

  out("Ut");
  if (index > 0) out(std::to_string(index - 1));
  out("_");
}

auto ExternalNameEncoder::encodeLocalName(Symbol* symbol) -> bool {
  auto function = symbol->enclosingFunction();
  if (!function) return false;

  out("Z");
  encodeName(function);
  if (!is_unmangled_main(function)) encodeFunctionSignature(function);
  out("E");

  if (isUnnamedLocalType(symbol)) {
    encodeUnnamedTypeName(symbol);
    return true;
  }

  if (encodeNestedName(symbol)) return true;

  encodeUnqualifiedName(symbol);
  return true;
}

void ExternalNameEncoder::encodeFunctionSignature(FunctionSymbol* function) {
  auto signature = signature_function(function);
  auto parameters = signature->functionParameters();
  if (parameters) parameterScopes_.push_back(parameters);
  encodeBareFunctionType(type_cast<FunctionType>(signature->type()),
                         encodes_return_type(function));
  if (parameters) parameterScopes_.pop_back();

  if (auto written = written_trailing_requires_function(function)) {
    encodeRequiresClause(written->trailingRequiresClause(), written);
  }
}

auto ExternalNameEncoder::encodeLambdaSignature(FunctionSymbol* callOperator)
    -> std::string {
  std::string signature;
  std::swap(signature, out_);
  encodeLambdaSig(callOperator);
  std::swap(signature, out_);
  return signature;
}

void ExternalNameEncoder::encodeLambdaSig(FunctionSymbol* callOperator) {
  if (auto declaration = callOperator->templateDeclaration()) {
    for (auto parameter : ListView{declaration->templateParameterList}) {
      if (is_synthesized_template_parameter(parameter)) continue;
      encodeTemplateParameterDeclaration(parameter, parameter);
    }
    encodeRequiresClause(declaration->requiresClause);
  }

  if (auto callOperatorType = type_cast<FunctionType>(callOperator->type()))
    encodeBareFunctionType(callOperatorType, false);
}

void ExternalNameEncoder::encodeTemplateParameterDeclaration(
    TemplateParameterAST* parameter, TemplateParameterAST* written) {
  visit(EncodeTemplateParameterDeclaration{*this, written}, parameter);
}

void ExternalNameEncoder::encodeTypeConstraint(
    TypeConstraintAST* typeConstraint) {
  auto conceptSymbol = typeConstraint->symbol;
  if (!conceptSymbol) {
    reportUnencodable(typeConstraint->firstSourceLocation(),
                      "type-constraint without a concept");
  }

  auto depth = std::exchange(encodesTemplateParameterDepth_, true);

  if (!typeConstraint->templateArgumentList) {
    encodeName(conceptSymbol);
  } else {
    auto parent = mangling_parent(conceptSymbol);
    const bool isNested =
        parent && !is_global_namespace(parent) && !is_abi_std_namespace(parent);
    if (isNested) out("N");
    if (!encodeWrittenTemplateId(conceptSymbol,
                                 typeConstraint->templateArgumentList,
                                 /*isPrefix=*/false)) {
      reportUnencodable(typeConstraint->firstSourceLocation(),
                        "type-constraint template arguments");
    }
    if (isNested) out("E");
  }

  encodesTemplateParameterDepth_ = depth;
}

void ExternalNameEncoder::encodeRequiresClause(
    RequiresClauseAST* requiresClause, FunctionSymbol* function) {
  if (!requiresClause || !requiresClause->expression) return;

  auto depth = std::exchange(encodesTemplateParameterDepth_, true);
  auto parameters = function ? function->functionParameters() : nullptr;
  if (parameters) parameterScopes_.push_back(parameters);
  out("Q");
  encodeRequiredExpression(requiresClause->expression, "constraint expression");
  if (parameters) parameterScopes_.pop_back();
  encodesTemplateParameterDepth_ = depth;
}

auto ExternalNameEncoder::encodeFunctionParameter(ParameterSymbol* parameter)
    -> bool {
  auto scope = symbol_cast<FunctionParametersSymbol>(parameter->parent());
  auto innermostFirst = parameterScopes_ | std::views::reverse;
  auto it = std::ranges::find(innermostFirst, scope);
  if (it == innermostFirst.end()) return false;

  const auto level = std::ranges::distance(innermostFirst.begin(), it);
  if (level == 0) {
    out("fp");
  } else {
    out(std::format("fL{}p", level - 1));
  }

  encodeCvQualifiers(cv_qualifiers(parameter->type()));

  auto parameters = views::members(scope) | views::parameters;
  auto position = std::ranges::find(parameters, parameter);
  const auto index = std::ranges::distance(parameters.begin(), position);
  if (index > 0) out(std::to_string(index - 1));
  out("_");
  return true;
}

auto ExternalNameEncoder::encodeUnresolvedName(NestedNameSpecifierAST* nns,
                                               UnqualifiedIdAST* id) -> bool {
  if (!nns) return encodeBaseUnresolvedName(id);

  if (ast_cast<GlobalNestedNameSpecifierAST>(nns)) {
    out("gs");
    return encodeBaseUnresolvedName(id);
  }

  auto root = outermost_qualifier(nns);
  if (names_unresolved_type(root)) {
    if (root == nns) {
      out("sr");
      if (!encodeUnresolvedType(root)) return false;
      return encodeBaseUnresolvedName(id);
    }
    out("srN");
    if (!encodeUnresolvedType(root)) return false;
    if (!encodeUnresolvedQualifierLevels(nns, root)) return false;
    out("E");
    return encodeBaseUnresolvedName(id);
  }

  if (has_global_qualifier(nns)) out("gs");
  out("sr");
  if (!encodeUnresolvedQualifierLevels(nns, nullptr)) return false;
  out("E");
  return encodeBaseUnresolvedName(id);
}

auto ExternalNameEncoder::encodeUnresolvedType(NestedNameSpecifierAST* nns)
    -> bool {
  if (auto parameter = dependent_prefix_type_param(nns)) {
    encodeType(parameter->type());
    return true;
  }
  if (auto decltypeQualifier = ast_cast<DecltypeNestedNameSpecifierAST>(nns)) {
    if (!decltypeQualifier->decltypeSpecifier) return false;
    auto type = decltypeQualifier->decltypeSpecifier->type;
    if (!type) return false;
    encodeType(type);
    return true;
  }
  return false;
}

void ExternalNameEncoder::encodeCvQualifiers(CvQualifiers cvQualifiers) {
  if (has_volatile(cvQualifiers)) out("V");
  if (has_const(cvQualifiers)) out("K");
}

void ExternalNameEncoder::encodeObjectParameterQualifiers(
    FunctionSymbol* function) {
  if (function->hasExplicitObjectParameter()) {
    out("H");
    return;
  }

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return;

  encodeCvQualifiers(functionType->cvQualifiers());

  if (functionType->refQualifier() == RefQualifier::kLvalue)
    out("R");
  else if (functionType->refQualifier() == RefQualifier::kRvalue)
    out("O");
}

auto ExternalNameEncoder::encodeNestedName(Symbol* symbol) -> bool {
  auto parent = mangling_parent(symbol);
  if (!parent) return false;
  if (is_global_namespace(parent)) return false;
  if (is_abi_std_namespace(parent)) return false;

  out("N");

  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    encodeObjectParameterQualifiers(functionSymbol);
  }

  if (encodeTemplateNameSubstitution(symbol)) {
    out("E");
    return true;
  }

  encodePrefix(parent);
  encodeUnqualifiedName(symbol);
  out("E");
  return true;
}

auto ExternalNameEncoder::encodeStdTypeAbbreviation(Symbol* symbol) -> bool {
  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol) return false;
  auto abbreviation =
      std_type_abbreviation(unit_->control(), classSymbol->type());
  if (abbreviation.empty()) return false;
  out(abbreviation);
  return true;
}

auto ExternalNameEncoder::encodeStdTemplateAbbreviation(Symbol* symbol)
    -> bool {
  auto abbreviation = std_template_abbreviation(template_name(symbol));
  if (abbreviation.empty()) return false;

  out(abbreviation);
  EncodeUnqualifiedName{*this, symbol}.encodeTemplateArguments(symbol);
  return true;
}

auto ExternalNameEncoder::encodeUnscopedName(Symbol* symbol) -> bool {
  if (encodeStdTypeAbbreviation(symbol)) return true;
  if (encodeStdTemplateAbbreviation(symbol)) return true;
  if (encodeTemplateNameSubstitution(symbol)) return true;

  if (is_abi_std_namespace(mangling_parent(symbol))) {
    out("St");
  }

  encodeUnqualifiedName(symbol);
  return true;
}

void ExternalNameEncoder::encodePrefix(Symbol* symbol) {
  if (is_abi_std_namespace(symbol)) {
    out("St");
    return;
  }

  if (encodeSubstitution(symbol->type())) return;

  if (encodeStdTypeAbbreviation(symbol)) return;

  if (encodeStdTemplateAbbreviation(symbol) ||
      encodeTemplateNameSubstitution(symbol)) {
    enterSubstitution(symbol->type());
    return;
  }

  if (auto parent = enclosing_class_or_namespace(symbol);
      parent && !is_global_namespace(parent)) {
    encodePrefix(parent);
  }

  encodeUnqualifiedName(symbol);
  enterSubstitution(symbol->type());
}

void ExternalNameEncoder::encodeTemplatePrefix(Symbol* symbol) {}

void ExternalNameEncoder::encodeUnqualifiedName(Symbol* symbol) {
  if (auto closure = symbol_cast<ClassSymbol>(symbol);
      closure && closure->isClosureType()) {
    encodeClosureTypeName(closure);
    return;
  }

  if (auto ns = symbol_cast<NamespaceSymbol>(symbol); ns && !ns->name()) {
    auto index = ns->anonNamespaceIndex().value();
    std::string name = std::format("_GLOBAL__N_{}", index + 1);
    out(std::format("{}{}", name.length(), name));
    return;
  }

  if (!symbol->name()) {
    EncodeUnqualifiedName encode{*this, symbol};
    if (encode.encodeConstructor()) return;
    encodeUnnamedTypeName(symbol);
    return;
  }

  visit(EncodeUnqualifiedName{*this, symbol}, symbol->name());
}

void ExternalNameEncoder::encodeBareFunctionType(
    const FunctionType* functionType, bool includeReturnType) {
  if (includeReturnType) {
    encodeType(functionType->returnType());
  }

  for (auto param : functionType->parameterTypes()) {
    encodeType(param);
  }

  if (functionType->isVariadic()) {
    out("z");
  } else if (functionType->parameterTypes().empty()) {
    out("v");
  }
}

void ExternalNameEncoder::encodeType(const Type* type) {
  if (auto abbreviation = std_type_abbreviation(unit_->control(), type);
      !abbreviation.empty()) {
    out(abbreviation);
    return;
  }
  if (auto parameter = templateParameterSubstitution(type)) {
    if (encodeSubstitution(*parameter)) return;
    (void)visit(EncodeType{*this}, type);
    enterSubstitution(*parameter);
    return;
  }
  if (encodeSubstitution(type)) return;
  if (!visit(EncodeType{*this}, type)) return;
  enterSubstitution(type);
}

struct ExternalNameEncoder::EncodeSimpleId {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto operator()(NameIdAST* ast) const -> bool {
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return true;
  }

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) const -> bool {
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return encoder.encodeWrittenTemplateArguments(nullptr,
                                                  ast->templateArgumentList);
  }

  [[nodiscard]] auto operator()(UnqualifiedIdAST*) const -> bool {
    return false;
  }
};

struct ExternalNameEncoder::EncodeDependentQualifier {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto enclosing(NestedNameSpecifierAST* nns) const -> bool {
    if (!nns) return true;
    return visit(*this, nns);
  }

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast) const
      -> bool {
    if (auto parameter = dependent_prefix_type_param(ast)) {
      encoder.encodeType(parameter->type());
      return true;
    }
    if (auto scope = resolved_qualifier_scope(ast->symbol)) {
      encoder.encodePrefix(scope);
      return true;
    }
    if (!enclosing(ast->nestedNameSpecifier)) return false;
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return true;
  }

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast) const
      -> bool {
    auto templateId = ast->templateId;
    if (!templateId) return false;
    if (auto templateName = type_template_name(templateId->symbol)) {
      return encoder.encodeWrittenTemplateId(
          templateName, templateId->templateArgumentList, /*isPrefix=*/true);
    }
    if (!enclosing(ast->nestedNameSpecifier)) return false;
    return EncodeSimpleId{encoder}(templateId);
  }

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast) const
      -> bool {
    if (!ast->decltypeSpecifier || !ast->decltypeSpecifier->type) return false;
    encoder.encodeType(ast->decltypeSpecifier->type);
    return true;
  }
};

auto ExternalNameEncoder::encodeDependentName(NestedNameSpecifierAST* nns,
                                              UnqualifiedIdAST* id) -> bool {
  out("N");
  if (!nns || !visit(EncodeDependentQualifier{*this}, nns)) return false;
  if (!visit(EncodeSimpleId{*this}, id)) return false;
  out("E");
  return true;
}

struct ExternalNameEncoder::EncodeUnresolvedQualifierLevel {
  ExternalNameEncoder& encoder;
  NestedNameSpecifierAST* root = nullptr;

  [[nodiscard]] auto enclosing(NestedNameSpecifierAST* nns) const -> bool {
    if (!nns || nns == root) return true;
    return visit(*this, nns);
  }

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast) const
      -> bool {
    if (!enclosing(ast->nestedNameSpecifier)) return false;
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return true;
  }

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast) const
      -> bool {
    if (!enclosing(ast->nestedNameSpecifier)) return false;
    if (!ast->templateId || !ast->templateId->identifier) return false;
    encoder.encodeSourceName(ast->templateId->identifier);
    return encoder.encodeWrittenTemplateArguments(
        nullptr, ast->templateId->templateArgumentList);
  }

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST*) const -> bool {
    return false;
  }
};

auto ExternalNameEncoder::encodeUnresolvedQualifierLevels(
    NestedNameSpecifierAST* nns, NestedNameSpecifierAST* root) -> bool {
  return EncodeUnresolvedQualifierLevel{*this, root}.enclosing(nns);
}

void ExternalNameEncoder::encodeSourceName(const Identifier* identifier) {
  const auto name = identifier->name();
  out(std::format("{}{}", name.size(), name));
}

auto ExternalNameEncoder::encodeOperatorName(TokenKind op, bool isUnary)
    -> std::string_view {
  switch (op) {
    case TokenKind::T_NEW:
      return "nw";
    case TokenKind::T_NEW_ARRAY:
      return "na";
    case TokenKind::T_DELETE:
      return "dl";
    case TokenKind::T_DELETE_ARRAY:
      return "da";
    case TokenKind::T_CO_AWAIT:
      return "aw";
    case TokenKind::T_PLUS:
      return isUnary ? "ps" : "pl";
    case TokenKind::T_MINUS:
      return isUnary ? "ng" : "mi";
    case TokenKind::T_AMP:
      return isUnary ? "ad" : "an";
    case TokenKind::T_STAR:
      return isUnary ? "de" : "ml";
    case TokenKind::T_TILDE:
      return "co";
    case TokenKind::T_SLASH:
      return "dv";
    case TokenKind::T_PERCENT:
      return "rm";
    case TokenKind::T_BAR:
      return "or";
    case TokenKind::T_CARET:
      return "eo";
    case TokenKind::T_EQUAL:
      return "aS";
    case TokenKind::T_PLUS_EQUAL:
      return "pL";
    case TokenKind::T_MINUS_EQUAL:
      return "mI";
    case TokenKind::T_STAR_EQUAL:
      return "mL";
    case TokenKind::T_SLASH_EQUAL:
      return "dV";
    case TokenKind::T_PERCENT_EQUAL:
      return "rM";
    case TokenKind::T_AMP_EQUAL:
      return "aN";
    case TokenKind::T_BAR_EQUAL:
      return "oR";
    case TokenKind::T_CARET_EQUAL:
      return "eO";
    case TokenKind::T_LESS_LESS:
      return "ls";
    case TokenKind::T_GREATER_GREATER:
      return "rs";
    case TokenKind::T_LESS_LESS_EQUAL:
      return "lS";
    case TokenKind::T_GREATER_GREATER_EQUAL:
      return "rS";
    case TokenKind::T_EQUAL_EQUAL:
      return "eq";
    case TokenKind::T_EXCLAIM_EQUAL:
      return "ne";
    case TokenKind::T_LESS:
      return "lt";
    case TokenKind::T_GREATER:
      return "gt";
    case TokenKind::T_LESS_EQUAL:
      return "le";
    case TokenKind::T_GREATER_EQUAL:
      return "ge";
    case TokenKind::T_LESS_EQUAL_GREATER:
      return "ss";
    case TokenKind::T_EXCLAIM:
      return "nt";
    case TokenKind::T_AMP_AMP:
      return "aa";
    case TokenKind::T_BAR_BAR:
      return "oo";
    case TokenKind::T_PLUS_PLUS:
      return "pp";
    case TokenKind::T_MINUS_MINUS:
      return "mm";
    case TokenKind::T_COMMA:
      return "cm";
    case TokenKind::T_MINUS_GREATER_STAR:
      return "pm";
    case TokenKind::T_MINUS_GREATER:
      return "pt";
    case TokenKind::T_LPAREN:
      return "cl";
    case TokenKind::T_LBRACKET:
      return "ix";
    case TokenKind::T_QUESTION:
      return "qu";
    default:
      cxx_runtime_error(
          std::format("cannot encode operator '{}'", Token::spell(op)));
  }
}

void ExternalNameEncoder::encodeTemplateParamValue(int depth, int index) {
  out("T");
  if (encodesTemplateParameterDepth_ && depth > 0)
    out(std::format("L{}_", depth - 1));
  if (index == 0) {
    out("_");
  } else {
    out(std::format("{}_", index - 1));
  }
}

auto ExternalNameEncoder::EncodeType::operator()(
    const TemplateTypeParameterSpecializationType* type) -> bool {
  encoder.encodeType(type->templateParameter());
  EncodeUnqualifiedName arguments{encoder};
  encoder.out("I");
  for (const auto& argument : type->templateArguments()) {
    if (auto symbol = std::get_if<Symbol*>(&argument)) {
      arguments.encodeTemplateArgumentSymbol(*symbol);
    } else if (auto argumentType = std::get_if<const Type*>(&argument)) {
      encoder.encodeType(*argumentType);
    }
  }
  encoder.out("E");
  return true;
}

struct ExternalNameEncoder::EncodeBaseUnresolvedName {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto operator()(NameIdAST* ast) const -> bool {
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return true;
  }

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) const -> bool {
    if (!ast->identifier) return false;
    encoder.encodeSourceName(ast->identifier);
    return encoder.encodeWrittenTemplateArguments(nullptr,
                                                  ast->templateArgumentList);
  }

  [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast) const -> bool {
    encoder.out("on");
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return true;
  }

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast) const
      -> bool {
    if (!ast->operatorFunctionId) return false;
    if (!operator()(ast->operatorFunctionId)) return false;
    return encoder.encodeWrittenTemplateArguments(nullptr,
                                                  ast->templateArgumentList);
  }

  [[nodiscard]] auto operator()(UnqualifiedIdAST*) const -> bool {
    return false;
  }
};

struct ExternalNameEncoder::EncodeRequirement {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) const -> bool {
    encoder.out("X");
    return encoder.encodeExpression(ast->expression);
  }

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) const -> bool {
    encoder.out("X");
    if (!encoder.encodeExpression(ast->expression)) return false;
    if (ast->noexceptLoc) encoder.out("N");
    if (!ast->typeConstraint) return true;
    encoder.out("R");
    encoder.encodeTypeConstraint(ast->typeConstraint);
    return true;
  }

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("T");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) const -> bool {
    encoder.out("Q");
    return encoder.encodeExpression(ast->expression);
  }
};

struct ExternalNameEncoder::EncodeExpression {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto encode(ExpressionAST* expr) const -> bool {
    return encoder.encodeExpression(expr);
  }

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast) const -> bool {
    encoder.out("sp");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.encodeConstValue(ast->type,
                             ConstValue{std::intmax_t(ast->isTrue ? 1 : 0)});
    return true;
  }

  [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast) const -> bool {
    if (ast->literalOperatorCall) return encode(ast->literalOperatorCall);
    if (!ast->literal || !ast->type) return false;
    encoder.encodeConstValue(
        ast->type,
        ConstValue{static_cast<std::intmax_t>(ast->literal->charValue())});
    return true;
  }

  [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast) const -> bool {
    if (ast->literalOperatorCall) return encode(ast->literalOperatorCall);
    if (!ast->literal || !ast->type) return false;
    encoder.encodeConstValue(ast->type, ConstValue{ast->literal->floatValue()});
    return true;
  }

  [[nodiscard]] auto operator()(NullptrLiteralExpressionAST*) const -> bool {
    encoder.out("LDnE");
    return true;
  }

  [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.out("L");
    encoder.encodeType(ast->type);
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(
      UserDefinedStringLiteralExpressionAST* ast) const -> bool {
    return encode(ast->literalOperatorCall);
  }

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast) const -> bool {
    if (ast->literalOperatorCall) return encode(ast->literalOperatorCall);
    if (!ast->literal || !ast->type) return false;
    encoder.encodeConstValue(
        ast->type,
        ConstValue{static_cast<std::intmax_t>(ast->literal->integerValue())});
    return true;
  }

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast) const -> bool {
    auto parameter = template_parameter_info(ast->symbol);
    if (!parameter) return false;
    encoder.out("sZ");
    encoder.encodeTemplateParamValue(parameter->depth, parameter->index);
    return true;
  }

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast) const -> bool {
    const auto name = encoder.unit_->tokenText(ast->typeTraitLoc);
    encoder.out(std::format("u{}{}", name.length(), name));
    for (auto typeId : ListView{ast->typeIdList}) {
      if (!typeId || !typeId->type) return false;
      encoder.encodeType(typeId->type);
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("st");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) const -> bool {
    encoder.out("sz");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("at");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) const -> bool {
    encoder.out("az");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) const -> bool {
    encoder.out("nx");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/true));
    if (is_increment_or_decrement(ast->op)) encoder.out("_");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/true));
    return encode(ast->baseExpression);
  }

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    if (!encode(ast->leftExpression)) return false;
    return encode(ast->rightExpression);
  }

  [[nodiscard]] auto operator()(CallExpressionAST* ast) const -> bool {
    encoder.out("cl");
    if (!encodeCallee(ast)) return false;
    for (auto argument : ListView{ast->expressionList}) {
      if (!encode(argument)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(MemberExpressionAST* ast) const -> bool {
    encoder.out(ast->accessOp == TokenKind::T_MINUS_GREATER ? "pt" : "dt");
    if (!encode(ast->baseExpression)) return false;
    return encoder.encodeUnresolvedName(ast->nestedNameSpecifier,
                                        ast->unqualifiedId);
  }

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) const -> bool {
    auto operatorName = cpp_cast_operator_name(ast->castOp);
    if (operatorName.empty() || !ast->type) return false;
    encoder.out(operatorName);
    encoder.encodeType(ast->type);
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(CastExpressionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.out("cv");
    encoder.encodeType(ast->type);
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(TypeConstructionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.out("cv");
    encoder.encodeType(ast->type);
    if (ast->expressionList && !ast->expressionList->next)
      return encode(ast->expressionList->value);
    encoder.out("_");
    for (auto argument : ListView{ast->expressionList}) {
      if (!encode(argument)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.out("tl");
    encoder.encodeType(ast->type);
    if (ast->bracedInitList) {
      for (auto element : ListView{ast->bracedInitList->expressionList}) {
        if (!encode(element)) return false;
      }
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(BracedInitListAST* ast) const -> bool {
    encoder.out("il");
    for (auto element : ListView{ast->expressionList}) {
      if (!encode(element)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(ThisExpressionAST*) const -> bool {
    encoder.out("fpT");
    return true;
  }

  [[nodiscard]] auto operator()(NewExpressionAST* ast) const -> bool {
    if (!ast->objectType) return false;
    const auto& traits = encoder.unit_->typeTraits();
    const bool isArrayNew = traits.is_array(ast->objectType);
    if (ast->scopeLoc) encoder.out("gs");
    encoder.out(isArrayNew ? "na" : "nw");
    if (ast->newPlacement) {
      for (auto argument : ListView{ast->newPlacement->expressionList}) {
        if (!encode(argument)) return false;
      }
    }
    encoder.out("_");
    encoder.encodeType(isArrayNew ? traits.remove_extent(ast->objectType)
                                  : ast->objectType);
    return encodeNewInitializer(ast->newInitalizer);
  }

  [[nodiscard]] auto operator()(DeleteExpressionAST* ast) const -> bool {
    if (ast->scopeLoc) encoder.out("gs");
    encoder.out(ast->lbracketLoc ? "da" : "dl");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) const -> bool {
    if (!ast->lparenLoc) {
      encoder.out("rq");
      return encodeRequirements(ast->requirementList);
    }

    encoder.out("rQ");
    auto parameterTypes =
        getParameterTypes(encoder.unit_, ast->parameterDeclarationClause);
    if (parameterTypes.empty()) encoder.out("v");
    for (auto parameterType : parameterTypes) encoder.encodeType(parameterType);
    encoder.out("_");

    auto parameters =
        ast->parameterDeclarationClause
            ? ast->parameterDeclarationClause->functionParametersSymbol
            : nullptr;
    encoder.parameterScopes_.push_back(parameters);
    const auto encoded = encodeRequirements(ast->requirementList);
    encoder.parameterScopes_.pop_back();
    return encoded;
  }

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) const -> bool {
    encoder.out("fl");
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast) const -> bool {
    encoder.out("fr");
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) const -> bool {
    encoder.out(containsUnexpandedParameterPack(ast->leftExpression) ? "fR"
                                                                     : "fL");
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return encode(ast->leftExpression) && encode(ast->rightExpression);
  }

  [[nodiscard]] auto operator()(ConditionalExpressionAST* ast) const -> bool {
    encoder.out("qu");
    return encode(ast->condition) && encode(ast->iftrueExpression) &&
           encode(ast->iffalseExpression);
  }

  [[nodiscard]] auto operator()(IdExpressionAST* ast) const -> bool {
    auto symbol = resolve_using_declaration(ast->symbol);
    if (auto enumerator = symbol_cast<EnumeratorSymbol>(symbol);
        enumerator && enumerator->value() && ast->type) {
      encoder.encodeConstValue(ast->type, *enumerator->value());
      return true;
    }
    if (auto param = symbol_cast<NonTypeParameterSymbol>(symbol)) {
      encoder.encodeTemplateParamValue(param->depth(), param->index());
      return true;
    }
    if (auto param = symbol_cast<ParameterSymbol>(symbol))
      return encoder.encodeFunctionParameter(param);
    if (designates_declared_entity(symbol)) {
      encoder.encodeExternalName(symbol);
      return true;
    }
    return encoder.encodeUnresolvedName(ast->nestedNameSpecifier,
                                        ast->unqualifiedId);
  }

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(PackIndexExpressionAST* ast) const -> bool {
    encoder.out("sy");
    return encode(ast->packExpression) && encode(ast->indexExpression);
  }

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(NestedStatementExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(DefaultInitializerExpressionAST* ast) const
      -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(LambdaExpressionAST* ast) const -> bool {
    if (!ast->symbol || !ast->symbol->closureType()) return false;
    encoder.out("L");
    encoder.encodeType(ast->symbol->closureType()->type());
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(VaArgExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(SubscriptExpressionAST* ast) const -> bool {
    encoder.out("ix");
    return encode(ast->baseExpression) && encode(ast->indexExpression);
  }

  [[nodiscard]] auto operator()(SpliceMemberExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(BuiltinConvertVectorExpressionAST*) const
      -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(TypeidExpressionAST* ast) const -> bool {
    encoder.out("te");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("ti");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(SpliceExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST*) const
      -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(NamespaceReflectExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(TypeIdReflectExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(ReflectExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(LabelAddressExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(AwaitExpressionAST* ast) const -> bool {
    encoder.out("aw");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(YieldExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(ThrowExpressionAST* ast) const -> bool {
    if (!ast->expression) {
      encoder.out("tr");
      return true;
    }
    encoder.out("tw");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return encode(ast->leftExpression) && encode(ast->rightExpression);
  }

  [[nodiscard]] auto operator()(TargetExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(RightExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(CompoundAssignmentExpressionAST* ast) const
      -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    return encode(ast->targetExpression) && encode(ast->rightExpression);
  }

  [[nodiscard]] auto operator()(ConditionExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(EqualInitializerAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(ParenInitializerAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(ThreeWayComparisonExpressionAST* ast) const
      -> bool {
    return encode(ast->comparison);
  }

  [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast) const
      -> bool {
    for (auto designator : ListView{ast->designatorList}) {
      if (!encodeDesignator(designator)) return false;
    }
    return encode(ast->initializer);
  }

 private:
  [[nodiscard]] auto encodeDesignator(DesignatorAST* designator) const -> bool {
    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      if (!dot->identifier) return false;
      encoder.out("di");
      encoder.encodeSourceName(dot->identifier);
      return true;
    }
    encoder.out("dx");
    return encode(ast_cast<SubscriptDesignatorAST>(designator)->expression);
  }

  [[nodiscard]] auto encodeNewInitializer(NewInitializerAST* initializer) const
      -> bool {
    if (!initializer) {
      encoder.out("E");
      return true;
    }
    if (auto braced = ast_cast<NewBracedInitializerAST>(initializer))
      return encode(braced->bracedInitList);
    encoder.out("pi");
    auto paren = ast_cast<NewParenInitializerAST>(initializer);
    for (auto argument : ListView{paren->expressionList}) {
      if (!encode(argument)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto encodeRequirements(
      List<RequirementAST*>* requirements) const -> bool {
    for (auto requirement : ListView{requirements}) {
      if (!encoder.encodeRequirement(requirement)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto encodeCallee(CallExpressionAST* ast) const -> bool {
    auto callee = ast_cast<IdExpressionAST>(ast->baseExpression);
    if (!callee || !isDependent(encoder.unit_, ast) ||
        !names_functions(callee->symbol)) {
      return encode(ast->baseExpression);
    }
    return encoder.encodeUnresolvedName(callee->nestedNameSpecifier,
                                        callee->unqualifiedId);
  }
};

auto ExternalNameEncoder::encodeRequirement(RequirementAST* requirement)
    -> bool {
  return visit(EncodeRequirement{*this}, requirement);
}

auto ExternalNameEncoder::encodeBaseUnresolvedName(UnqualifiedIdAST* id)
    -> bool {
  return visit(EncodeBaseUnresolvedName{*this}, id);
}

auto ExternalNameEncoder::encodeExpression(ExpressionAST* expr) -> bool {
  if (!expr) return false;

  const auto outMark = out_.size();
  const auto substsSnapshot = substs_;

  if (visit(EncodeExpression{*this}, expr)) return true;

  out_.resize(outMark);
  substs_ = substsSnapshot;
  return false;
}

void ExternalNameEncoder::encodeRequiredExpression(ExpressionAST* expr,
                                                   std::string_view what) {
  if (expr && encodeExpression(expr)) return;
  reportUnencodable(expr ? expr->firstSourceLocation() : SourceLocation{},
                    what);
}

void ExternalNameEncoder::reportUnencodable(SourceLocation location,
                                            std::string_view what) {
  if (unit_ && location) {
    unit_->error(location,
                 std::format("cannot mangle {} while encoding '{}'", what,
                             encodingSymbol_
                                 ? to_string(encodingSymbol_->type(),
                                             to_string(encodingSymbol_->name()))
                                 : std::string{}));
  }
  cxx_runtime_error(std::format("cannot mangle {}", what));
}

struct ExternalNameEncoder::EncodeWrittenTemplateArgument {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast) const -> bool {
    auto typeId = ast->typeId;
    if (!typeId || !typeId->type) return false;
    if (isPackExpansion(typeId)) {
      encoder.encodeType(
          encoder.unit_->control()->getPackExpansionType(typeId->type));
      return true;
    }
    encoder.encodeType(typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast) const
      -> bool {
    if (visit(IsExpressionPrimary{}, ast->expression))
      return encoder.encodeExpression(ast->expression);
    encoder.out("X");
    if (!encoder.encodeExpression(ast->expression)) return false;
    encoder.out("E");
    return true;
  }
};

auto ExternalNameEncoder::encodeWrittenTemplateArguments(
    Symbol* templateName, List<TemplateArgumentAST*>* arguments) -> bool {
  std::vector<TemplateArgumentAST*> written;
  for (auto argument : ListView{arguments}) written.push_back(argument);

  auto pack = parameter_pack_arguments(templateName, written.size());

  out("I");
  for (std::size_t index = 0; index <= written.size(); ++index) {
    if (pack && index == pack->first) out("J");
    if (pack && index == pack->second) out("E");
    if (index == written.size()) break;
    if (!visit(EncodeWrittenTemplateArgument{*this}, written[index]))
      return false;
  }
  out("E");
  return true;
}

auto ExternalNameEncoder::encodeWrittenTemplateId(
    Symbol* templateName, List<TemplateArgumentAST*>* arguments, bool isPrefix)
    -> bool {
  if (isPrefix && encodeTemplatePrefixSubstitution(templateName, arguments))
    return true;

  if (!encodeSubstitution(templateName)) {
    if (auto parent = mangling_parent(templateName);
        parent && !is_global_namespace(parent)) {
      encodePrefix(parent);
    }
    auto saved = std::exchange(templateNameOnly_, templateName);
    encodeUnqualifiedName(templateName);
    templateNameOnly_ = saved;
    enterSubstitution(templateName);
  }

  if (!encodeWrittenTemplateArguments(templateName, arguments)) return false;
  if (isPrefix) enterTemplatePrefixSubstitution(templateName, arguments);
  return true;
}

auto ExternalNameEncoder::normalizeConstInt(const Type* type,
                                            const ConstInt& value) const
    -> ConstInt {
  if (!unit_) return value;

  auto normalized =
      TypeTraits{unit_}.integral_constant(type, value.toWideValue());
  if (!normalized) return value;

  return *normalized;
}

void ExternalNameEncoder::encodeTemplateArgumentValue(const Type* type,
                                                      const ConstValue& value) {
  if (isExpressionPrimary(type, value)) {
    encodeValueExpression(type, value);
    return;
  }
  out("X");
  encodeValueExpression(type, value);
  out("E");
}

auto ExternalNameEncoder::isExpressionPrimary(const Type* type,
                                              const ConstValue& value) const
    -> bool {
  if (std::holds_alternative<std::shared_ptr<ConstObject>>(value)) return false;
  if (std::holds_alternative<std::shared_ptr<InitializerList>>(value))
    return false;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  if (!address || !*address || !(*address)->symbol()) return true;
  auto traits = TypeTraits{unit_};
  if (!traits.is_reference(type)) return false;
  if ((*address)->offset() != 0) return false;
  return traits.is_same(traits.remove_reference(type),
                        (*address)->symbol()->type());
}

void ExternalNameEncoder::encodeValueExpression(const Type* type,
                                                const ConstValue& value) {
  if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
      address && *address) {
    encodeAddressValue(type, **address);
    return;
  }
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
      object && *object) {
    encodeObjectValue(type, **object);
    return;
  }
  if (auto elements = std::get_if<std::shared_ptr<InitializerList>>(&value);
      elements && *elements) {
    encodeArrayValue(type, **elements);
    return;
  }
  encodeConstValue(type, value);
}

void ExternalNameEncoder::encodeAddressValue(const Type* type,
                                             const ConstAddress& address) {
  auto entity = address.symbol();
  if (!entity) {
    out("L");
    encodeType(type);
    out("0E");
    return;
  }

  auto traits = TypeTraits{unit_};

  if (traits.is_member_pointer(type)) {
    out("ad");
    encodeExternalName(entity);
    return;
  }

  if (traits.is_reference(type)) {
    if (!traits.is_same(traits.remove_reference(type), entity->type())) {
      out("cv");
      encodeType(type);
    }
    encodeExternalName(entity);
    return;
  }

  auto pointee = traits.get_element_type(type);
  if (designates_subobject(traits, pointee, address)) {
    out("adso");
    encodeType(pointee);
    encodeExternalName(entity);
    if (address.offset() != 0) {
      auto elementSize = unit_->control()->memoryLayout()->sizeOf(pointee);
      out(std::to_string(address.offset() *
                         static_cast<std::intmax_t>(elementSize.value_or(1))));
    }
    out("E");
    return;
  }

  if (!traits.is_same(pointee, entity->type())) {
    out("cv");
    encodeType(type);
  }
  out("ad");
  encodeExternalName(entity);
}

void ExternalNameEncoder::encodeExternalName(Symbol* symbol) {
  out("L_Z");
  encodeName(symbol);
  if (auto function = symbol_cast<FunctionSymbol>(symbol);
      function && !function->hasCLinkage()) {
    encodeFunctionSignature(function);
  }
  out("E");
}

void ExternalNameEncoder::encodeObjectValue(const Type* type,
                                            const ConstObject& object) {
  out("tl");
  encodeType(type);

  const auto& members = object.members();
  auto count = members.size();
  while (count > 0 && isZeroValue(members[count - 1].value)) --count;

  for (const auto& member : members | std::views::take(count)) {
    auto symbol = member.symbol;
    if (!symbol || !symbol->type()) {
      reportUnencodable({}, "class template argument member");
    }
    if (object.isUnion()) {
      out("di");
      encodeSourceName(name_cast<Identifier>(symbol->name()));
    }
    encodeValueExpression(symbol->type(), member.value);
  }
  out("E");
}

void ExternalNameEncoder::encodeArrayValue(const Type* type,
                                           const InitializerList& list) {
  out("tl");
  encodeType(type);

  const auto& elements = list.elements;
  auto count = elements.size();
  while (count > 0 && isZeroValue(std::get<0>(elements[count - 1]))) --count;

  for (const auto& [value, elementType] : elements | std::views::take(count))
    encodeValueExpression(elementType, value);
  out("E");
}

auto ExternalNameEncoder::isZeroValue(const ConstValue& value) const -> bool {
  if (auto integer = std::get_if<ConstInt>(&value)) return integer->isZero();
  if (auto real = std::get_if<double>(&value))
    return std::bit_cast<std::uint64_t>(*real) == 0;
  if (auto real = std::get_if<float>(&value))
    return std::bit_cast<std::uint32_t>(*real) == 0;
  if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value))
    return *address && !(*address)->symbol() && !(*address)->stringLiteral();
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    return *object && std::ranges::all_of((*object)->members(),
                                          [&](const ConstObject::Member& m) {
                                            return isZeroValue(m.value);
                                          });
  }
  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    return *list && std::ranges::all_of((*list)->elements, [&](const auto& e) {
      return isZeroValue(std::get<0>(e));
    });
  }
  return false;
}

void ExternalNameEncoder::encodeFloatingValue(const Type* type, double value) {
  auto format = unit_->control()->memoryLayout()->floatingPointFormat(type);
  if (!format) {
    reportUnencodable(SourceLocation{}, "floating-point value");
  }
  out(floating_digits(value, *format));
}

void ExternalNameEncoder::encodeConstValue(const Type* type,
                                           const ConstValue& value) {
  out("L");
  encodeType(type);
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, ConstInt>) {
          auto normalized = normalizeConstInt(type, v);
          if (normalized.isNegative()) out("n");
          out(normalized.toDecimalString());
        } else if constexpr (std::is_same_v<T, bool>) {
          out(v ? "1" : "0");
        } else if constexpr (std::is_same_v<T, double>) {
          encodeFloatingValue(type, v);
        }
      },
      value);
  out("E");
}

namespace {

struct CollectAbiTags {
  std::set<const Identifier*>& tags;
  std::set<const Type*> visited;

  void collect(const Type* type) {
    if (!type) return;
    if (!visited.insert(type).second) return;
    visit(*this, type);
  }

  void collect(std::span<const TemplateArgument> args) {
    for (const auto& arg : args) {
      if (auto sym = std::get_if<Symbol*>(&arg)) collect(*sym);
      if (auto type = std::get_if<const Type*>(&arg)) collect(*type);
    }
  }

  void collect(Symbol* symbol) {
    if (!symbol) return;
    if (auto pack = symbol_cast<ParameterPackSymbol>(symbol)) {
      for (auto element : pack->elements()) collect(element);
      return;
    }
    collect(symbol->type());
  }

  void operator()(const QualType* type) { collect(type->elementType()); }
  void operator()(const PackExpansionType* type) { collect(type->pattern()); }
  void operator()(const PointerType* type) { collect(type->elementType()); }

  void operator()(const LvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const ClassType* type) {
    auto classSymbol = type->symbol();
    if (!classSymbol) return;
    addTags(classSymbol);
    collect(classSymbol->templateArguments());
  }

  void operator()(const EnumType* type) { addTags(type->symbol()); }
  void operator()(const ScopedEnumType* type) { addTags(type->symbol()); }

  void operator()(const FunctionType* type) {
    collect(type->returnType());
    for (auto parameterType : type->parameterTypes()) collect(parameterType);
  }

  void operator()(const Type*) {}

  void addTags(Symbol* symbol) {
    if (!symbol) return;
    for (auto tag : symbol->abiTags()) tags.insert(tag);
    for (auto enclosing : symbol->enclosingSymbols()) {
      auto ns = symbol_cast<NamespaceSymbol>(enclosing);
      if (!ns || !ns->isInline()) continue;
      for (auto tag : ns->abiTags()) tags.insert(tag);
    }
  }
};

}  // namespace

auto ExternalNameEncoder::mangledAbiTags(Symbol* symbol)
    -> std::vector<const Identifier*> {
  if (!symbol) return {};
  if (symbol_cast<NamespaceSymbol>(symbol)) return {};

  std::set<const Identifier*> declaredTags;
  auto addDeclaredTags = [&](Symbol* declaration) {
    for (auto tag : declaration->abiTags()) declaredTags.insert(tag);
  };

  addDeclaredTags(symbol);
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    for (auto declaration : function->declarations())
      addDeclaredTags(declaration);
  }

  std::vector<const Identifier*> tags{declaredTags.begin(), declaredTags.end()};
  std::ranges::sort(tags, {}, [](const Identifier* id) { return id->name(); });

  std::set<const Identifier*> mangled{tags.begin(), tags.end()};
  CollectAbiTags mangledCollector{mangled};
  const Type* unmangledType = nullptr;

  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (function->isConstructor() || function->isDestructor()) return tags;

    auto functionType = type_cast<FunctionType>(function->type());
    if (!functionType) return tags;

    unmangledType = functionType->returnType();

    for (auto parameterType : functionType->parameterTypes()) {
      mangledCollector.collect(parameterType);
    }
    mangledCollector.collect(function->templateArguments());
    if (encodes_return_type(function)) mangledCollector.collect(unmangledType);
  } else if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
    unmangledType = variable->type();
    mangledCollector.collect(variable->templateArguments());
  } else if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    unmangledType = field->type();
  } else {
    return tags;
  }

  for (auto enclosing : symbol->enclosingSymbols()) {
    for (auto tag : enclosing->abiTags()) mangled.insert(tag);
  }

  std::set<const Identifier*> unmangled;
  CollectAbiTags{unmangled}.collect(unmangledType);

  for (auto tag : unmangled) {
    if (mangled.contains(tag)) continue;
    tags.push_back(tag);
  }

  std::ranges::sort(tags, {}, [](const Identifier* id) { return id->name(); });
  tags.erase(std::ranges::unique(tags).begin(), tags.end());

  return tags;
}

void ExternalNameEncoder::encodeAbiTags(Symbol* symbol) {
  for (auto tag : mangledAbiTags(symbol)) {
    out(std::format("B{}{}", tag->name().length(), tag->name()));
  }
}

auto ExternalNameEncoder::encodeTemplateNameSubstitution(Symbol* symbol)
    -> bool {
  auto templateName = template_name(symbol);
  if (!templateName) return false;
  if (!encodeSubstitution(templateName)) return false;
  EncodeUnqualifiedName{*this, symbol}.encodeAbiTagsAndTemplateArguments(
      symbol);
  return true;
}

auto ExternalNameEncoder::encodeSubstitution(const Type* type) -> bool {
  auto sameType = [&](const Substitution& substitution) {
    auto candidate = std::get_if<const Type*>(&substitution);
    if (!candidate) return false;
    if (*candidate == type) return true;
    return unit_ && TypeTraits{unit_}.is_same(*candidate, type);
  };
  return encodeSubstitutionAt(static_cast<std::size_t>(
      std::ranges::find_if(substs_, sameType) - substs_.begin()));
}

auto ExternalNameEncoder::encodeSubstitution(Symbol* symbol) -> bool {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<Symbol*>(&substitution);
    return candidate && *candidate == symbol;
  };
  return encodeSubstitutionAt(static_cast<std::size_t>(
      std::ranges::find_if(substs_, matches) - substs_.begin()));
}

auto ExternalNameEncoder::encodeSubstitution(
    const TemplateParameterSubstitution& parameter) -> bool {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplateParameterSubstitution>(&substitution);
    return candidate && *candidate == parameter;
  };
  return encodeSubstitutionAt(static_cast<std::size_t>(
      std::ranges::find_if(substs_, matches) - substs_.begin()));
}

auto ExternalNameEncoder::encodeSubstitutionAt(std::size_t index) -> bool {
  if (index >= substs_.size()) return false;

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", encodeSeqId(static_cast<int>(index) - 1)));
  return true;
}

auto ExternalNameEncoder::templateParameterSubstitution(const Type* type) const
    -> std::optional<TemplateParameterSubstitution> {
  auto position = [&](int depth, int index) {
    return TemplateParameterSubstitution{
        .depth = encodesTemplateParameterDepth_ ? depth : 0, .index = index};
  };
  if (auto parameter = type_cast<TypeParameterType>(type))
    return position(parameter->depth(), parameter->index());
  if (auto parameter = type_cast<TemplateTypeParameterType>(type))
    return position(parameter->depth(), parameter->index());
  return std::nullopt;
}

void ExternalNameEncoder::enterSubstitution(
    const TemplateParameterSubstitution& parameter) {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplateParameterSubstitution>(&substitution);
    return candidate && *candidate == parameter;
  };
  if (std::ranges::any_of(substs_, matches)) return;
  substs_.emplace_back(parameter);
}

auto ExternalNameEncoder::encodeTemplatePrefixSubstitution(
    Symbol* templateSymbol, List<TemplateArgumentAST*>* arguments) -> bool {
  auto sameTemplateId = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplatePrefixSubstitution>(&substitution);
    if (!candidate || candidate->templateSymbol != templateSymbol) return false;
    return unit_ && TemplateEquivalence{unit_}.sameWritten(candidate->arguments,
                                                           arguments);
  };
  return encodeSubstitutionAt(static_cast<std::size_t>(
      std::ranges::find_if(substs_, sameTemplateId) - substs_.begin()));
}

auto ExternalNameEncoder::encodeSeqId(int id) -> std::string {
  static constexpr char digits[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string result;
  do {
    result.insert(result.begin(), digits[id % 36]);
    id /= 36;
  } while (id != 0);
  return result;
}

void ExternalNameEncoder::enterSubstitution(const Type* type) {
  auto sameType = [&](const Substitution& substitution) {
    auto candidate = std::get_if<const Type*>(&substitution);
    if (!candidate) return false;
    if (*candidate == type) return true;
    return unit_ && TypeTraits{unit_}.is_same(*candidate, type);
  };
  if (std::ranges::any_of(substs_, sameType)) return;
  substs_.emplace_back(type);
}

void ExternalNameEncoder::enterSubstitution(Symbol* symbol) {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<Symbol*>(&substitution);
    return candidate && *candidate == symbol;
  };
  if (std::ranges::any_of(substs_, matches)) return;
  substs_.emplace_back(symbol);
}

void ExternalNameEncoder::enterTemplatePrefixSubstitution(
    Symbol* templateSymbol, List<TemplateArgumentAST*>* arguments) {
  auto sameTemplateId = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplatePrefixSubstitution>(&substitution);
    if (!candidate || candidate->templateSymbol != templateSymbol) return false;
    return unit_ && TemplateEquivalence{unit_}.sameWritten(candidate->arguments,
                                                           arguments);
  };
  if (std::ranges::any_of(substs_, sameTemplateId)) return;
  substs_.push_back(TemplatePrefixSubstitution{templateSymbol, arguments});
}

}  // namespace cxx
