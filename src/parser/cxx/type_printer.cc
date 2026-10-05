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
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>
#include <sstream>

namespace cxx {
namespace {

constexpr TypePrintOptions kSourceSpelling{.sourceSpelling = true};

[[nodiscard]] auto sourceScopePrefix(Symbol* scope,
                                     const TypePrintOptions& options)
    -> std::string;

[[nodiscard]] auto sourceQualifiedName(Symbol* symbol,
                                       const TypePrintOptions& options)
    -> std::string;

[[nodiscard]] auto isVisibleFromDeclarationScope(
    Symbol* scope, const TypePrintOptions& options) -> bool {
  auto declarationScope = options.declarationScope;
  if (!declarationScope) return false;
  if (declarationScope == scope) return true;
  return declarationScope->hasEnclosingSymbol(scope);
}

struct SourceScopePrefix {
  const TypePrintOptions& options;

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) const -> std::string {
    if (!symbol->parent()) return {};
    if (isNamedThroughEnclosingNamespace(symbol))
      return sourceScopePrefix(symbol->parent(), options);
    return sourceQualifiedName(symbol, options) + "::";
  }

  [[nodiscard]] auto isNamedThroughEnclosingNamespace(
      NamespaceSymbol* symbol) const -> bool {
    if (!options.declarationScope) return false;
    return symbol->isInline();
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> std::string {
    std::string prefix;
    auto parent = symbol->parent();
    if (parent && parent->isBlock())
      prefix = sourceScopePrefix(parent, options);
    return prefix + to_string(symbol->type(), "", options) + "::";
  }

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> std::string {
    auto functionOptions = options;
    functionOptions.omitFunctionReturnType = true;
    return to_string(symbol->type(), sourceQualifiedName(symbol, options),
                     functionOptions) +
           "::";
  }

  [[nodiscard]] auto operator()(Symbol* symbol) const -> std::string {
    return sourceScopePrefix(symbol->parent(), options);
  }
};

auto sourceScopePrefix(Symbol* scope, const TypePrintOptions& options)
    -> std::string {
  if (!scope) return {};
  if (isVisibleFromDeclarationScope(scope, options)) return {};
  return visit(SourceScopePrefix{options}, scope);
}

[[nodiscard]] auto sourceUnqualifiedName(Symbol* symbol) -> std::string {
  if (symbol->isNamespace() && !symbol->name()) return "(anonymous namespace)";
  return to_string(symbol->name(), kSourceSpelling);
}

auto sourceQualifiedName(Symbol* symbol, const TypePrintOptions& options)
    -> std::string {
  if (!symbol) return {};
  return sourceScopePrefix(symbol->parent(), options) +
         sourceUnqualifiedName(symbol);
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
    bindings.push_back(std::format("{} = {}", to_string(parameter->name()),
                                   to_string(argument, kSourceSpelling)));
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
  explicit TypePrinter(TypePrintOptions options)
      : options_(options), nestedOptions_(options) {
    nestedOptions_.omitFunctionReturnType = false;
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
    addFormals_ = true;
  }

  ~TypePrinter() {
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

      ptrOps_ = op + ptrOps_;

      return;
    }

    if (type->isConst()) {
      specifiers_.append("const ");
    }

    if (type->isVolatile()) {
      specifiers_.append("volatile ");
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
    auto buf = "[" + std::to_string(type->size()) + "]";

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
    specifiers_.append("$overload-set");
  }

  void operator()(const FunctionType* type) {
    std::string signature;

    signature.append("(");

    const auto& params = type->parameterTypes();

    for (std::size_t i = 0; i < params.size(); ++i) {
      const auto& param = params[i];
      signature.append(to_string(param, "", nestedOptions_));

      if (i != params.size() - 1) {
        signature.append(", ");
      }
    }

    if (type->isVariadic()) {
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
    if (options_.sourceSpelling) {
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
    if (options_.sourceSpelling) {
      appendEnclosingScope(closure);
      specifiers_.append("(lambda)");
      return;
    }
    auto callOperator = closure->functionCallOperator();
    auto signature = callOperator ? to_string(callOperator->type()) : "";
    specifiers_.append(std::format("(lambda {})", signature));
  }

  void operator()(const ClassType* type) {
    if (type->symbol()->isClosureType()) {
      appendClosureType(type->symbol());
      return;
    }

    appendEnclosingScope(type->symbol());

    std::string out = to_string(type->symbol()->name(), nestedOptions_);

    if (namesInjectedClassName(type->symbol())) {
      specifiers_.append(out);
      return;
    }

    if (type->symbol()->isSpecialization()) {
      out += '<';
      std::string_view sep = "";
      for (const auto& arg :
           expand_template_arguments(type->symbol()->templateArguments())) {
        out += std::format("{}{}", sep, to_string(arg, nestedOptions_));
        sep = ", ";
      }
      out += '>';
    } else if (auto templDecl = type->symbol()->primaryTemplateSymbol()) {
      out += '<';
      std::string_view sep = "";
      for (const auto& param :
           views::members(templDecl->templateParameters())) {
        out += std::format("{}{}", sep, templateParameterSpelling(param));
        sep = ", ";
      }
      out += '>';
    }

    specifiers_.append(out);
  }

  [[nodiscard]] auto namesInjectedClassName(ClassSymbol* classSymbol) const
      -> bool {
    if (!options_.sourceSpelling) return false;
    return isVisibleFromDeclarationScope(classSymbol, options_);
  }

  [[nodiscard]] auto appendDeclaredTemplateParameter(const Type* type) -> bool {
    if (!options_.sourceSpelling) return false;
    auto parameter = declaredTemplateParameter(type, options_.declarationScope);
    if (!parameter || !parameter->name()) return false;
    specifiers_.append(to_string(parameter->name()));
    return true;
  }

  [[nodiscard]] auto templateParameterSpelling(Symbol* parameter) const
      -> std::string {
    if (options_.sourceSpelling && parameter->name())
      return to_string(parameter->name());
    return to_string(parameter->type(), "", nestedOptions_);
  }

  void operator()(const NamespaceType* type) {
    if (options_.sourceSpelling) {
      specifiers_.append(sourceQualifiedName(type->symbol(), nestedOptions_));
      return;
    }
    appendEnclosingScope(type->symbol());
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const MemberObjectPointerType* type) {
    ptrOps_ = std::format(" {}::*",
                          to_string(type->classType(), "", nestedOptions_)) +
              ptrOps_;
    accept(type->elementType());
  }

  void operator()(const MemberFunctionPointerType* type) {
    ptrOps_ =
        std::format("{}::*", to_string(type->classType(), "", nestedOptions_)) +
        ptrOps_;
    accept(type->functionType());
  }

  void operator()(const EnumType* type) {
    appendEnclosingScope(type->symbol());
    specifiers_.append(to_string(type->symbol()->name(), nestedOptions_));
  }

  void operator()(const ScopedEnumType* type) {
    appendEnclosingScope(type->symbol());
    specifiers_.append(to_string(type->symbol()->name(), nestedOptions_));
  }

  void operator()(const TypeParameterType* type) {
    if (appendDeclaredTemplateParameter(type)) return;
    if (type->depth() < 0 || type->index() < 0) {
      specifiers_.append("<dependent-type>");
      return;
    }
    specifiers_.append(
        std::format("type-param<{}, {}>", type->index(), type->depth()));
  }

  void operator()(const TemplateTypeParameterType* type) {
    if (appendDeclaredTemplateParameter(type)) return;
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
      specifiers_.append(to_string(argument, nestedOptions_));
    }
    specifiers_.append(">");
  }

  void operator()(const DecltypeType* type) {
    std::ostringstream os;
    ASTPrettyPrinter pp(type->translationUnit(), os);
    pp(type->expression());
    specifiers_ += std::format("decltype({})", os.str());
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

    std::ostringstream os;
    ASTPrettyPrinter pp(unit, os);
    if (type->nestedNameSpecifier()) {
      pp(type->nestedNameSpecifier());
    }
    if (type->unqualifiedId()) pp(type->unqualifiedId());
    specifiers_ += os.str();
  }

  auto textOf(TranslationUnit* unit, SourceLocationRange range) const
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
    const auto elementType = to_string(type->elementType());
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
    specifiers_ += to_string(type->elementType());
    specifiers_ += ")";
  }

  void operator()(const UnresolvedVectorType* type) {
    specifiers_ += to_string(type->elementType());
    specifiers_ += type->vectorKind() == VectorKind::kExt
                       ? " __attribute__((ext_vector_type("
                       : " __attribute__((vector_size(";
    specifiers_ += textOf(type->translationUnit(),
                          type->sizeExpression()->sourceLocationRange());
    specifiers_ += ")))";
  }

 private:
  TypePrintOptions options_;
  TypePrintOptions nestedOptions_;
  std::string specifiers_;
  std::string ptrOps_;
  std::string declarator_;
  bool addFormals_ = false;
};
}  // namespace

auto to_string(const Type* type, const std::string& id,
               TypePrintOptions options) -> std::string {
  if (!type) return {};
  return TypePrinter{options}(type, id);
}

auto to_string(const Type* type, const Name* name, TypePrintOptions options)
    -> std::string {
  return TypePrinter{options}(type, to_string(name, options));
}

auto pretty_function_name(FunctionSymbol* function) -> std::string {
  auto options = kSourceSpelling;
  options.omitFunctionReturnType =
      function->isConstructor() || function->isDestructor();

  std::string text;
  if (isStaticMemberFunction(function)) text += "static ";
  if (function->isVirtual()) text += "virtual ";
  text += to_string(function->type(), sourceQualifiedName(function, options),
                    options);
  text += enclosingTemplateArgumentBindings(function);
  return text;
}
}  // namespace cxx
