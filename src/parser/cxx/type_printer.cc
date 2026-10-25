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
#include <cxx/ast_pretty_printer.h>
#include <cxx/names.h>
#include <cxx/printing.h>
#include <cxx/private/printing-priv.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
namespace {

constexpr detail::PrintOptions kSourceSpelling{{}, detail::PrintMode::Source};

[[nodiscard]] auto sourceScopePrefix(Symbol* scope,
                                     const detail::PrintOptions& options)
    -> std::string;

[[nodiscard]] auto sourceQualifiedName(Symbol* symbol,
                                       const detail::PrintOptions& options)
    -> std::string;

[[nodiscard]] auto isVisibleFromDeclarationScope(
    Symbol* scope, const detail::PrintOptions& options) -> bool {
  auto declarationScope = options.declarationScope;
  if (!declarationScope) return false;
  if (declarationScope == scope) return true;
  return declarationScope->hasEnclosingSymbol(scope);
}

struct SourceScopePrefix {
  const detail::PrintOptions& options;

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) const -> std::string {
    if (!symbol->parent()) return {};
    if (isNamedThroughEnclosingNamespace(symbol))
      return sourceScopePrefix(symbol->parent(), options);
    return sourceQualifiedName(symbol, options) + "::";
  }

  [[nodiscard]] auto isNamedThroughEnclosingNamespace(
      NamespaceSymbol* symbol) const -> bool {
    if (!options.declarationScope) return false;
    return options.mode == detail::PrintMode::Source && symbol->isInline();
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> std::string {
    std::string prefix;
    auto parent = symbol->parent();
    if (parent && parent->isBlock())
      prefix = sourceScopePrefix(parent, options);
    return prefix + detail::renderType(symbol->type(), "", options) + "::";
  }

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> std::string {
    auto functionOptions = options;
    functionOptions.omitFunctionReturnType = true;
    return detail::renderType(symbol->type(),
                              sourceQualifiedName(symbol, options),
                              functionOptions) +
           "::";
  }

  [[nodiscard]] auto operator()(Symbol* symbol) const -> std::string {
    return sourceScopePrefix(symbol->parent(), options);
  }
};

auto sourceScopePrefix(Symbol* scope, const detail::PrintOptions& options)
    -> std::string {
  if (!scope) return {};
  if (options.mode == detail::PrintMode::Source &&
      isVisibleFromDeclarationScope(scope, options))
    return {};
  return visit(SourceScopePrefix{options}, scope);
}

[[nodiscard]] auto sourceUnqualifiedName(Symbol* symbol,
                                         const detail::PrintOptions& options)
    -> std::string {
  if (symbol->isNamespace()) {
    auto identifier = name_cast<Identifier>(symbol->name());
    if (!identifier || identifier->value().empty() || identifier->isAnonymous())
      return "(anonymous namespace)";
  }
  return detail::renderName(symbol->name(), options);
}

auto sourceQualifiedName(Symbol* symbol, const detail::PrintOptions& options)
    -> std::string {
  if (!symbol) return {};
  return sourceScopePrefix(symbol->parent(), options) +
         sourceUnqualifiedName(symbol, options);
}

[[nodiscard]] auto templateParametersDeclaredBy(Symbol* scope)
    -> TemplateParametersSymbol* {
  if (auto parameters = symbol_cast<TemplateParametersSymbol>(scope))
    return parameters;
  return template_parameters_of(scope);
}

[[nodiscard]] auto declaredTemplateParameter(const Type* type,
                                             Symbol* declarationScope)
    -> Symbol* {
  for (auto scope = declarationScope; scope; scope = scope->parent()) {
    auto parameters = templateParametersDeclaredBy(scope);
    if (!parameters) continue;
    for (auto parameter : views::members(parameters)) {
      if (parameter->type() == type) return parameter;
    }
  }
  return nullptr;
}

[[nodiscard]] auto templateArgumentBindings(Symbol* specialization)
    -> std::vector<std::string> {
  std::vector<std::string> bindings;
  auto parameters = template_parameters_of(primary_template_of(specialization));
  if (!parameters) return bindings;
  auto arguments = template_arguments_of(specialization);
  std::size_t index = 0;
  for (auto parameter : views::members(parameters)) {
    if (index == arguments.size()) break;
    const auto& argument = arguments[index++];
    if (!parameter->name()) continue;
    bindings.push_back(std::format(
        "{} = {}", detail::renderName(parameter->name(), kSourceSpelling),
        detail::renderTemplateArgument(argument, kSourceSpelling)));
  }
  return bindings;
}

[[nodiscard]] auto enclosingTemplateArgumentBindings(Symbol* symbol)
    -> std::string {
  std::vector<std::string> bindings;
  for (auto scope = symbol; scope; scope = scope->parent()) {
    auto scopeBindings = templateArgumentBindings(scope);
    bindings.insert(bindings.begin(), scopeBindings.begin(),
                    scopeBindings.end());
  }
  if (bindings.empty()) return {};
  std::string text = " [";
  std::string_view sep = "";
  for (const auto& binding : bindings) {
    text += sep;
    text += binding;
    sep = ", ";
  }
  text += ']';
  return text;
}

[[nodiscard]] auto isStaticMemberFunction(FunctionSymbol* function) -> bool {
  return function->isStatic() && function->enclosingClass();
}

class TypePrinter {
 public:
  explicit TypePrinter(detail::PrintOptions options)
      : options_(options), nestedOptions_(options) {
    nestedOptions_.omitFunctionReturnType = false;
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
  }

  auto operator()(const Type* type, const std::string& id) -> std::string {
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
    declarator_.append(id);

    accept(type);

    std::string buffer;

    buffer.append(specifiers_);
    buffer.append(ptrOps_);
    if (!declarator_.empty()) {
      if (!buffer.empty()) buffer.append(" ");
      buffer.append(declarator_);
    }

    return buffer;
  }

  void accept(const Type* type) {
    if (type) visit(*this, type);
  }

  void operator()(const NullptrType* type) {
    specifiers_.append("decltype(nullptr)");
  }

  void operator()(const DecltypeAutoType* type) {
    specifiers_.append("decltype(auto)");
  }

  void operator()(const AutoType* type) { specifiers_.append("auto"); }

  void operator()(const BuiltinVaListType* type) {
    specifiers_.append("__builtin_va_list");
  }

  void operator()(const BuiltinMetaInfoType* type) {
    specifiers_.append("__builtin_meta_info");
  }

  void operator()(const SveType* type) {
    specifiers_.append(Token::spell(type->sveKind()));
  }

  void operator()(const VoidType* type) { specifiers_.append("void"); }

  void operator()(const BoolType* type) { specifiers_.append("bool"); }

  void operator()(const CharType* type) { specifiers_.append("char"); }

  void operator()(const SignedCharType* type) {
    specifiers_.append("signed char");
  }

  void operator()(const UnsignedCharType* type) {
    specifiers_.append("unsigned char");
  }

  void operator()(const Char8Type* type) { specifiers_.append("char8_t"); }

  void operator()(const Char16Type* type) { specifiers_.append("char16_t"); }

  void operator()(const Char32Type* type) { specifiers_.append("char32_t"); }

  void operator()(const WideCharType* type) { specifiers_.append("wchar_t"); }

  void operator()(const ShortIntType* type) { specifiers_.append("short"); }

  void operator()(const UnsignedShortIntType* type) {
    specifiers_.append("unsigned short");
  }

  void operator()(const IntType* type) { specifiers_.append("int"); }

  void operator()(const UnsignedIntType* type) {
    specifiers_.append("unsigned int");
  }

  void operator()(const LongIntType* type) { specifiers_.append("long"); }

  void operator()(const UnsignedLongIntType* type) {
    specifiers_.append("unsigned long");
  }

  void operator()(const LongLongIntType* type) {
    specifiers_.append("long long");
  }

  void operator()(const UnsignedLongLongIntType* type) {
    specifiers_.append("unsigned long long");
  }

  void operator()(const Int128Type* type) { specifiers_.append("__int128_t"); }

  void operator()(const UnsignedInt128Type* type) {
    specifiers_.append("__uint128_t");
  }

  void operator()(const FloatType* type) { specifiers_.append("float"); }

  void operator()(const DoubleType* type) { specifiers_.append("double"); }

  void operator()(const LongDoubleType* type) {
    specifiers_.append("long double");
  }

  void operator()(const HalfType* type) { specifiers_.append("__fp16"); }

  void operator()(const Float16Type* type) { specifiers_.append("_Float16"); }

  void operator()(const Float128Type* type) {
    specifiers_.append("__float128");
  }

  void operator()(const QualType* type) {
    if (auto ptrTy = type_cast<PointerType>(type->elementType())) {
      accept(ptrTy->elementType());

      std::string op = "*";

      if (type->isConst()) {
        op += " const";
      }

      if (type->isVolatile()) {
        op += " volatile";
      }

      if (auto space = type->addressSpace()) {
        op += std::format(" __attribute__((address_space({})))", *space);
      }

      ptrOps_ = op + ptrOps_;

      return;
    }

    if (type->isConst()) {
      specifiers_.append("const ");
    }

    if (type->isVolatile()) {
      specifiers_.append("volatile ");
    }

    if (auto space = type->addressSpace()) {
      specifiers_ += std::format("__attribute__((address_space({}))) ", *space);
    }

    accept(type->elementType());
  }

  void operator()(const PointerType* type) {
    ptrOps_ = "*" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const LvalueReferenceType* type) {
    ptrOps_ = "&" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    ptrOps_ = "&&" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    auto buf = "[" + std::format("{}", type->size()) + "]";

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    std::string buf = "[]";

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const OverloadSetType* type) {
    if (options_.mode == detail::PrintMode::Internal) {
      specifiers_.append("$overload-set");
      return;
    }
    specifiers_.append("overloaded function");
    if (auto symbol = type->symbol(); symbol && symbol->name()) {
      specifiers_ += " " + sourceQualifiedName(symbol, nestedOptions_);
    }
  }

  void operator()(const FunctionType* type) {
    std::string signature;

    signature.append("(");

    const auto& params = type->parameterTypes();

    for (std::size_t i = 0; i < params.size(); ++i) {
      const auto& param = params[i];
      signature.append(detail::renderType(param, "", nestedOptions_));

      if (i != params.size() - 1) {
        signature.append(", ");
      }
    }

    if (type->isVariadic()) {
      if (!params.empty() && options_.mode != detail::PrintMode::Internal)
        signature.append(", ");
      signature.append("...");
    }

    signature.append(")");

    switch (type->cvQualifiers()) {
      case CvQualifiers::kConst:
        signature.append(" const");
        break;
      case CvQualifiers::kVolatile:
        signature.append(" volatile");
        break;
      case CvQualifiers::kConstVolatile:
        signature.append(" const volatile");
        break;
      default:
        break;
    }

    switch (type->refQualifier()) {
      case RefQualifier::kLvalue:
        signature.append(" &");
        break;
      case RefQualifier::kRvalue:
        signature.append(" &&");
        break;
      default:
        break;
    }

    if (type->isNoexcept()) {
      signature.append(" noexcept");
    }

    if (!ptrOps_.empty()) {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      ptrOps_.clear();
    }

    declarator_.append(signature);

    if (!options_.omitFunctionReturnType) accept(type->returnType());
  }

  void appendEnclosingScope(Symbol* symbol) {
    auto parent = symbol->parent();
    if (options_.mode != detail::PrintMode::Internal) {
      if (parent && parent->isBlock()) return;
      specifiers_.append(sourceScopePrefix(parent, nestedOptions_));
      return;
    }
    if (!parent) return;
    while (symbol_cast<TemplateParametersSymbol>(parent)) {
      parent = parent->parent();
    }
    accept(parent->type());
    specifiers_.append("::");
  }

  void appendClosureType(ClassSymbol* closure) {
    if (options_.mode != detail::PrintMode::Internal) {
      if (options_.mode == detail::PrintMode::Diagnostic && options_.unit &&
          closure->location()) {
        auto [file, line, column] =
            options_.unit->tokenStartPosition(closure->location());
        specifiers_ += std::format("(lambda at {}:{}:{})", file, line, column);
        return;
      }
      if (options_.mode == detail::PrintMode::Source)
        appendEnclosingScope(closure);
      specifiers_.append("(lambda)");
      return;
    }
    auto callOperator = closure->functionCallOperator();
    auto signature = callOperator ? detail::renderType(callOperator->type(), "",
                                                       nestedOptions_)
                                  : "";
    specifiers_.append(std::format("(lambda {})", signature));
  }

  void operator()(const ClassType* type) {
    if (type->symbol()->isClosureType()) {
      appendClosureType(type->symbol());
      return;
    }

    appendEnclosingScope(type->symbol());

    auto symbol = type->symbol();
    if (appendAnonymousType(
            symbol, symbol->isUnion() ? "anonymous union" : "anonymous class"))
      return;
    std::string out = detail::renderName(symbol->name(), nestedOptions_);

    if (namesInjectedClassName(type->symbol())) {
      specifiers_.append(out);
      return;
    }

    auto templateDeclaration = symbol->primaryTemplateSymbol();
    if (!templateDeclaration && options_.mode != detail::PrintMode::Internal &&
        symbol->templateParameters())
      templateDeclaration = symbol;

    if (type->symbol()->isSpecialization()) {
      out += '<';
      std::string_view sep = "";
      for (const auto& arg :
           expand_template_arguments(type->symbol()->templateArguments())) {
        out += std::format("{}{}", sep,
                           detail::renderTemplateArgument(arg, nestedOptions_));
        sep = ", ";
      }
      out += '>';
    } else if (templateDeclaration) {
      out += '<';
      std::string_view sep = "";
      for (const auto& param :
           views::members(templateDeclaration->templateParameters())) {
        out += std::format("{}{}", sep, templateParameterSpelling(param));
        sep = ", ";
      }
      out += '>';
    }

    specifiers_.append(out);
  }

  [[nodiscard]] auto namesInjectedClassName(ClassSymbol* classSymbol) const
      -> bool {
    if (options_.mode != detail::PrintMode::Source) return false;
    return isVisibleFromDeclarationScope(classSymbol, options_);
  }

  [[nodiscard]] auto appendDeclaredTemplateParameter(const Type* type) -> bool {
    if (options_.mode == detail::PrintMode::Internal) return false;
    auto parameter = declaredTemplateParameter(type, options_.declarationScope);
    if (!parameter || !parameter->name()) return false;
    specifiers_.append(detail::renderName(parameter->name(), options_));
    return true;
  }

  [[nodiscard]] auto templateParameterSpelling(Symbol* parameter) const
      -> std::string {
    if (options_.mode != detail::PrintMode::Internal && parameter->name())
      return detail::renderName(parameter->name(), options_);
    return detail::renderType(parameter->type(), "", nestedOptions_);
  }

  void operator()(const NamespaceType* type) {
    if (options_.mode != detail::PrintMode::Internal) {
      specifiers_.append(sourceQualifiedName(type->symbol(), nestedOptions_));
      return;
    }
    appendEnclosingScope(type->symbol());
    specifiers_.append(
        detail::renderName(type->symbol()->name(), nestedOptions_));
  }

  void operator()(const MemberObjectPointerType* type) {
    ptrOps_ = std::format(" {}::*", detail::renderType(type->classType(), "",
                                                       nestedOptions_)) +
              ptrOps_;
    accept(type->elementType());
  }

  void operator()(const MemberFunctionPointerType* type) {
    ptrOps_ = std::format("{}::*", detail::renderType(type->classType(), "",
                                                      nestedOptions_)) +
              ptrOps_;
    accept(type->functionType());
  }

  void operator()(const EnumType* type) {
    appendEnclosingScope(type->symbol());
    if (appendAnonymousType(type->symbol(), "anonymous enum")) return;
    specifiers_.append(
        detail::renderName(type->symbol()->name(), nestedOptions_));
  }

  void operator()(const ScopedEnumType* type) {
    appendEnclosingScope(type->symbol());
    specifiers_.append(
        detail::renderName(type->symbol()->name(), nestedOptions_));
  }

  [[nodiscard]] auto appendAnonymousType(Symbol* symbol, std::string_view kind)
      -> bool {
    if (options_.mode == detail::PrintMode::Internal) return false;
    auto identifier = name_cast<Identifier>(symbol->name());
    if (identifier && !identifier->value().empty() &&
        !identifier->isAnonymous())
      return false;
    if (options_.mode == detail::PrintMode::Diagnostic && options_.unit &&
        symbol->location()) {
      auto [file, line, column] =
          options_.unit->tokenStartPosition(symbol->location());
      specifiers_ += std::format("({} at {}:{}:{})", kind, file, line, column);
      return true;
    }
    specifiers_ += std::format("({})", kind);
    return true;
  }

  void operator()(const TypeParameterType* type) {
    if (appendDeclaredTemplateParameter(type)) return;
    if (options_.mode != detail::PrintMode::Internal) {
      specifiers_ += type->depth() < 0 || type->index() < 0
                         ? "dependent type"
                         : "type template parameter";
      return;
    }
    if (type->depth() < 0 || type->index() < 0) {
      specifiers_.append("<dependent-type>");
      return;
    }
    specifiers_.append(
        std::format("type-param<{}, {}>", type->index(), type->depth()));
  }

  void operator()(const TemplateTypeParameterType* type) {
    if (appendDeclaredTemplateParameter(type)) return;
    if (options_.mode != detail::PrintMode::Internal) {
      specifiers_ += "template template parameter";
      return;
    }
    specifiers_.append(std::format("template-type-param<{}, {}>", type->index(),
                                   type->depth()));
  }

  void operator()(const PackExpansionType* type) {
    ptrOps_ = "..." + ptrOps_;
    accept(type->pattern());
  }

  void operator()(const TemplateTypeParameterSpecializationType* type) {
    (*this)(type->templateParameter());
    specifiers_.append("<");
    auto first = true;
    for (const auto& argument : type->templateArguments()) {
      if (!first) specifiers_.append(", ");
      first = false;
      specifiers_.append(
          detail::renderTemplateArgument(argument, nestedOptions_));
    }
    specifiers_.append(">");
  }

  void operator()(const DecltypeType* type) {
    specifiers_ += std::format(
        "decltype({})", formatAST(type->translationUnit(), type->expression()));
  }

  void operator()(const UnresolvedNameType* type) {
    auto unit = type->translationUnit();

    // The snippet is the only spelling available once the tokens the name was
    // written with are gone, as they are for an adopted prefix (6.8).
    auto snippet = unit->snippetText(type->sourceLocationRange());
    if (!snippet.empty()) {
      specifiers_ += snippet;
      return;
    }

    specifiers_ +=
        std::format("{}{}", formatAST(unit, type->nestedNameSpecifier()),
                    formatAST(unit, type->unqualifiedId()));
  }

  [[nodiscard]] auto textOf(TranslationUnit* unit,
                            SourceLocationRange range) const
      -> std::string_view {
    return unit->snippetText(range);
  }

  void operator()(const UnresolvedBoundedArrayType* type) {
    std::string buf;
    buf += '[';
    buf += textOf(type->translationUnit(), type->size()->sourceLocationRange());
    buf += ']';

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const UnresolvedUnderlyingType* type) {
    specifiers_ += "__underlying_type(";
    specifiers_ +=
        textOf(type->translationUnit(), type->typeId()->sourceLocationRange());
    specifiers_ += ")";
  }

  void operator()(const UnresolvedBuiltinType* type) {
    switch (type->builtinKind()) {
#define PROCESS_UNARY_BUILTIN(id, name) \
  case UnaryBuiltinTypeKind::T_##id:    \
    specifiers_ += name;                \
    break;
      FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(PROCESS_UNARY_BUILTIN)
#undef PROCESS_UNARY_BUILTIN
      default:
        specifiers_ += "__builtin";
        break;
    }
    specifiers_ += "(";
    specifiers_ +=
        textOf(type->translationUnit(), type->typeId()->sourceLocationRange());
    specifiers_ += ")";
  }

  void operator()(const BitIntType* type) {
    specifiers_ += std::format("_BitInt({})", type->numBits());
  }

  void operator()(const UnsignedBitIntType* type) {
    specifiers_ += std::format("unsigned _BitInt({})", type->numBits());
  }

  void operator()(const UnresolvedBitIntType* type) {
    if (type->isUnsigned()) specifiers_ += "unsigned ";
    specifiers_ += "_BitInt(";
    specifiers_ += textOf(type->translationUnit(),
                          type->sizeExpression()->sourceLocationRange());
    specifiers_ += ")";
  }

  void operator()(const VectorType* type) {
    const auto elementType =
        detail::renderType(type->elementType(), "", nestedOptions_);
    specifiers_ += elementType;
    if (type->vectorKind() == VectorKind::kExt) {
      specifiers_ += std::format(" __attribute__((ext_vector_type({})))",
                                 type->elementCount());
    } else {
      specifiers_ +=
          std::format(" __attribute__((vector_size({} * sizeof({}))))",
                      type->elementCount(), elementType);
    }
  }

  void operator()(const ComplexType* type) {
    specifiers_ += "_Complex ";
    accept(type->elementType());
  }

  void operator()(const AtomicType* type) {
    specifiers_ += "_Atomic(";
    specifiers_ += detail::renderType(type->elementType(), "", nestedOptions_);
    specifiers_ += ")";
  }

  void operator()(const UnresolvedVectorType* type) {
    specifiers_ += detail::renderType(type->elementType(), "", nestedOptions_);
    specifiers_ += type->vectorKind() == VectorKind::kExt
                       ? " __attribute__((ext_vector_type("
                       : " __attribute__((vector_size(";
    specifiers_ += textOf(type->translationUnit(),
                          type->sizeExpression()->sourceLocationRange());
    specifiers_ += ")))";
  }

 private:
  detail::PrintOptions options_;
  detail::PrintOptions nestedOptions_;
  std::string specifiers_;
  std::string ptrOps_;
  std::string declarator_;
};
}  // namespace

auto detail::renderType(const Type* type, const std::string& identifier,
                        PrintOptions options) -> std::string {
  if (!type) return {};
  return TypePrinter{options}(type, identifier);
}

auto detail::renderType(const Type* type, const Name* identifier,
                        PrintOptions options) -> std::string {
  return TypePrinter{options}(type, renderName(identifier, options));
}

auto TypeFormat::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  detail::PrintOptions printOptions{options, mode};
  if (auto id = std::get_if<std::string_view>(&identifier))
    return std::format_to(
        out, "{}", detail::renderType(type, std::string(*id), printOptions));
  return std::format_to(
      out, "{}",
      detail::renderType(type, std::get<const Name*>(identifier),
                         printOptions));
}

auto SymbolFormat::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  if (!symbol) return out;
  detail::PrintOptions printOptions{options, mode};
  auto name = sourceQualifiedName(symbol, printOptions);
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (!printOptions.declarationScope)
      printOptions.declarationScope = function;
    printOptions.omitFunctionReturnType |=
        function->isConstructor() || function->isDestructor();
    return std::format_to(
        out, "{}", detail::renderType(function->type(), name, printOptions));
  }
  if (symbol->isClass() || symbol->isEnum() || symbol->isScopedEnum())
    return std::format_to(out, "{}",
                          detail::renderType(symbol->type(), "", printOptions));
  return std::format_to(out, "{}", name);
}

auto to_string(const Type* type, const std::string& id,
               TypePrintOptions options) -> std::string {
  return std::format("{}", formatType(type, id, options));
}

auto to_string(const Type* type, const Name* name, TypePrintOptions options)
    -> std::string {
  return std::format("{}", formatType(type, name, options));
}

auto to_string(Symbol* symbol, TypePrintOptions options) -> std::string {
  return std::format("{}", formatSymbol(symbol, options));
}

auto to_pretty_string(const Type* type, const std::string& id,
                      TypePrintOptions options) -> std::string {
  return std::format("{}", formatPrettyType(type, id, options));
}

auto to_pretty_string(const Type* type, const Name* name,
                      TypePrintOptions options) -> std::string {
  return std::format("{}", formatPrettyType(type, name, options));
}

auto to_pretty_string(Symbol* symbol, TypePrintOptions options) -> std::string {
  return std::format("{}", formatPrettySymbol(symbol, options));
}

auto to_diag_string(const Type* type, const std::string& id,
                    TypePrintOptions options) -> std::string {
  return std::format("{}", formatDiagType(type, id, options));
}

auto to_diag_string(const Type* type, const Name* name,
                    TypePrintOptions options) -> std::string {
  return std::format("{}", formatDiagType(type, name, options));
}

auto to_diag_string(Symbol* symbol, TypePrintOptions options) -> std::string {
  return std::format("{}", formatDiagSymbol(symbol, options));
}

auto pretty_function_name(FunctionSymbol* function) -> std::string {
  auto options = kSourceSpelling;
  options.omitFunctionReturnType =
      function->isConstructor() || function->isDestructor();

  std::string text;
  if (isStaticMemberFunction(function)) text += "static ";
  if (function->isVirtual()) text += "virtual ";
  text += to_pretty_string(function->type(),
                           sourceQualifiedName(function, options), options);
  text += enclosingTemplateArgumentBindings(function);
  return text;
}
}  // namespace cxx
