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
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <bit>
#include <format>

namespace cxx {
namespace {
struct ToInt {
  auto operator()(bool v) const -> std::optional<std::intmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(std::intmax_t v) const -> std::optional<std::intmax_t> {
    return v;
  }

  auto operator()(float v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(const std::shared_ptr<ConstComplex>& v) const
      -> std::optional<std::intmax_t> {
    if (!v) return std::nullopt;
    return std::visit(*this, v->real());
  }

  auto operator()(auto x) const -> std::optional<std::intmax_t> {
    return std::nullopt;
  }
};

struct ToUInt {
  auto operator()(bool v) const -> std::optional<std::uintmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(std::intmax_t v) const -> std::optional<std::uintmax_t> {
    return std::bit_cast<std::uintmax_t>(v);
  }

  auto operator()(float v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
  }

  auto operator()(const std::shared_ptr<ConstComplex>& v) const
      -> std::optional<std::uintmax_t> {
    if (!v) return std::nullopt;
    return std::visit(*this, v->real());
  }

  auto operator()(auto x) const -> std::optional<std::uintmax_t> {
    return std::nullopt;
  }
};

template <typename T>
struct ArithmeticCast {
  auto operator()(const StringLiteral*) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<Meta>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<InitializerList>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstObject>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstAddress>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstLabelAddress>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(IndeterminateValue) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstComplex>& value) const -> T {
    if (!value) return T{};
    return std::visit(*this, value->real());
  }

  auto operator()(auto value) const -> T { return static_cast<T>(value); }
};
}  // namespace

struct ASTInterpreter::ToBool {
  ASTInterpreter& interp;

  auto operator()(const StringLiteral*) const -> std::optional<bool> {
    return true;
  }

  auto operator()(const Meta&) const -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const std::shared_ptr<ConstObject>&) const
      -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const std::shared_ptr<ConstAddress>&) const
      -> std::optional<bool> {
    return true;
  }

  auto operator()(IndeterminateValue) const -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const std::shared_ptr<ConstComplex>& value) const
      -> std::optional<bool> {
    if (!value) return std::nullopt;
    auto real = std::visit(*this, value->real());
    auto imag = std::visit(*this, value->imag());
    if (!real || !imag) return std::nullopt;
    return *real || *imag;
  }

  auto operator()(const auto& value) const -> std::optional<bool> {
    return bool(value);
  }
};

ASTInterpreter::ASTInterpreter(TranslationUnit* unit, ScopeSymbol* scope)
    : unit_(unit), traits(unit) {
  if (scope) currentFunction_ = scope->enclosingFunctionOrSelf();
}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::cloneValue(const ConstValue& value) -> ConstValue {
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return value;
    auto copy = std::make_shared<ConstObject>((*object)->type());
    for (const auto& member : (*object)->members())
      copy->addMember(member.symbol, cloneValue(member.value));
    return ConstValue{std::move(copy)};
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return value;
    auto copy = std::make_shared<InitializerList>();
    for (const auto& [element, type] : (*list)->elements)
      copy->elements.emplace_back(cloneValue(element), type);
    return ConstValue{std::move(copy)};
  }

  if (auto complexValue = std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
    if (!*complexValue) return value;
    return ConstValue{
        std::make_shared<ConstComplex>(cloneValue((*complexValue)->real()),
                                       cloneValue((*complexValue)->imag()))};
  }

  return value;
}

auto isFullyInitialized(const ConstValue& value) -> bool {
  if (std::holds_alternative<IndeterminateValue>(value)) return false;

  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return false;
    for (const auto& member : (*object)->members()) {
      if (!isFullyInitialized(member.value)) return false;
    }
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return false;
    for (const auto& [element, type] : (*list)->elements) {
      if (!isFullyInitialized(element)) return false;
    }
  }

  if (auto complexValue = std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
    if (!*complexValue) return false;
    if (!isFullyInitialized((*complexValue)->real())) return false;
    if (!isFullyInitialized((*complexValue)->imag())) return false;
  }

  return true;
}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::evaluate(ExpressionAST* ast) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto result = expression(ast);
  return result;
}

auto ASTInterpreter::evaluateAddress(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  return addressOfLvalue(ast);
}

auto ASTInterpreter::toBool(const ConstValue& value) -> std::optional<bool> {
  return std::visit(ToBool{*this}, value);
}

auto ASTInterpreter::toInt(const ConstValue& value)
    -> std::optional<std::intmax_t> {
  return std::visit(ToInt{}, value);
}

auto ASTInterpreter::toUInt(const ConstValue& value)
    -> std::optional<std::uintmax_t> {
  return std::visit(ToUInt{}, value);
}

auto ASTInterpreter::toIntegralType(const ConstValue& value, const Type* type)
    -> std::optional<ConstValue> {
  auto representation =
      translationUnit()->typeTraits().integral_representation(type);
  if (!representation) return std::nullopt;

  constexpr auto storageBits = static_cast<int>(sizeof(std::uintmax_t) * 8);
  const auto bits = std::min(representation->bits, storageBits);

  if (!representation->isSigned) {
    auto result = toUInt(value);
    if (!result.has_value()) return std::nullopt;
    if (bits < storageBits) *result &= (std::uintmax_t{1} << bits) - 1;
    return ConstValue{std::bit_cast<std::intmax_t>(*result)};
  }

  auto result = toInt(value);
  if (!result.has_value()) return std::nullopt;
  if (bits >= storageBits) return ConstValue{*result};

  const auto mask = (std::uintmax_t{1} << bits) - 1;
  auto bitPattern = std::bit_cast<std::uintmax_t>(*result) & mask;
  if (bitPattern & (std::uintmax_t{1} << (bits - 1))) bitPattern |= ~mask;
  return ConstValue{std::bit_cast<std::intmax_t>(bitPattern)};
}

auto ASTInterpreter::toArithmeticType(const ConstValue& value, const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;

  const auto holdsArithmetic =
      std::holds_alternative<std::intmax_t>(value) ||
      std::holds_alternative<float>(value) ||
      std::holds_alternative<double>(value) ||
      std::holds_alternative<long double>(value) ||
      std::holds_alternative<std::shared_ptr<ConstComplex>>(value);

  if (!holdsArithmetic) return std::nullopt;

  type = traits.remove_cv(type);

  if (auto complexType = type_cast<ComplexType>(type)) {
    auto elementType = complexType->elementType();
    if (auto complexValue =
            std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
      if (!*complexValue) return std::nullopt;
      auto real = toArithmeticType((*complexValue)->real(), elementType);
      auto imag = toArithmeticType((*complexValue)->imag(), elementType);
      if (!real || !imag) return std::nullopt;
      return ConstValue{std::make_shared<ConstComplex>(*real, *imag)};
    }

    auto real = toArithmeticType(value, elementType);
    auto zero = zeroInitialize(elementType);
    if (!real || !zero) return std::nullopt;
    return ConstValue{std::make_shared<ConstComplex>(*real, *zero)};
  }

  switch (type->kind()) {
    case TypeKind::kFloat: {
      auto result = toFloat(value);
      if (!result) return std::nullopt;
      return ConstValue{*result};
    }

    case TypeKind::kDouble: {
      auto result = toDouble(value);
      if (!result) return std::nullopt;
      return ConstValue{*result};
    }

    case TypeKind::kLongDouble: {
      auto result = toLongDouble(value);
      if (!result) return std::nullopt;
      return ConstValue{*result};
    }

    default:
      break;
  }

  return toIntegralType(value, type);
}

auto ASTInterpreter::toFloat(const ConstValue& value) -> std::optional<float> {
  return std::visit(ArithmeticCast<float>{}, value);
}

auto ASTInterpreter::toDouble(const ConstValue& value)
    -> std::optional<double> {
  return std::visit(ArithmeticCast<double>{}, value);
}

auto ASTInterpreter::toLongDouble(const ConstValue& value)
    -> std::optional<long double> {
  return std::visit(ArithmeticCast<long double>{}, value);
}

auto ASTInterpreter::lookupLocal(const Symbol* sym) const
    -> std::optional<ConstValue> {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto ref = it->refs.find(sym);
    if (ref != it->refs.end()) return *ref->second;
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return found->second;
  }
  return std::nullopt;
}

auto ASTInterpreter::lookupLocalSlot(const Symbol* sym) -> ConstValue* {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto ref = it->refs.find(sym);
    if (ref != it->refs.end()) return ref->second;
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return &found->second;
  }
  return nullptr;
}

void ASTInterpreter::bindReference(const Symbol* sym, ConstValue* target) {
  if (frames_.empty()) frames_.push_back({});
  frames_.back().refs.insert_or_assign(sym, target);
}

void ASTInterpreter::setLocal(const Symbol* sym, ConstValue value) {
  if (frames_.empty()) frames_.push_back({});
  frames_.back().locals.insert_or_assign(sym, std::move(value));
}

auto ASTInterpreter::bindParameters(Frame& frame, FunctionSymbol* func,
                                    std::vector<ConstValue>& args) -> bool {
  auto params = func->parameters();
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (i < args.size()) {
      auto value = traits.is_reference(params[i]->type()) ? args[i]
                                                          : cloneValue(args[i]);
      frame.locals.insert_or_assign(params[i], std::move(value));
    } else {
      if (!params[i]->defaultArgument()) return false;
      if (!bindOneParameter(frame, params[i], params[i]->defaultArgument()))
        return false;
    }
  }
  return true;
}

auto ASTInterpreter::bindOneParameter(Frame& frame, Symbol* paramSymbol,
                                      ExpressionAST* argExpr) -> bool {
  auto param = symbol_cast<ParameterSymbol>(paramSymbol);
  if (param && traits.is_reference(param->type())) {
    if (auto value = addressOfLvalue(argExpr)) {
      auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
      if (!address || !*address) return false;
      frame.referenceAddresses.insert_or_assign(paramSymbol, *value);
      if (auto slot = addressSlot(**address, 0, param->type())) {
        frame.refs.insert_or_assign(paramSymbol, slot);
        return true;
      }
      auto referent = loadAddress(**address, 0, param->type());
      if (!referent) return false;
      frame.locals.insert_or_assign(paramSymbol, std::move(*referent));
      return true;
    }
    if (auto slot = lvalue(argExpr)) {
      frame.refs.insert_or_assign(paramSymbol, slot);
      return true;
    }
  }
  auto value = evaluate(argExpr);
  if (!value) return false;
  frame.locals.insert_or_assign(paramSymbol, cloneValue(*value));
  return true;
}

auto ASTInterpreter::bindParametersFromExprs(
    Frame& frame, FunctionSymbol* function,
    std::span<ExpressionAST* const> arguments) -> bool {
  auto parameters = function->parameters();
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    auto argument =
        i < arguments.size() ? arguments[i] : parameters[i]->defaultArgument();
    if (!argument || !bindOneParameter(frame, parameters[i], argument))
      return false;
  }
  for (std::size_t i = parameters.size(); i < arguments.size(); ++i)
    if (!expression(arguments[i])) return false;
  return true;
}

void ASTInterpreter::applyNsdmis(const std::shared_ptr<ConstObject>& obj) {
  auto classType = unqualified_cast<ClassType>(obj->type());
  auto classSymbol = classType ? classType->symbol() : nullptr;
  if (!classSymbol) return;
  auto savedThis = std::exchange(thisObject_, obj);
  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic() || !field->initializer()) continue;
    auto value = evaluate(field->initializer());
    if (!value) continue;
    obj->setMember(field, std::move(*value));
    if (classSymbol->isUnion()) break;
  }
  thisObject_ = std::move(savedThis);
}

auto ASTInterpreter::initializeDefaultedObject(
    const std::shared_ptr<ConstObject>& obj, ClassSymbol* classSymbol) -> bool {
  if (!obj || !classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();
  auto savedThis = std::exchange(thisObject_, obj);

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    auto value = defaultConstruct(baseClass->type());
    if (!value) {
      thisObject_ = std::move(savedThis);
      return false;
    }
    obj->addMember(base, std::move(*value));
  }

  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;

    if (field->initializer()) {
      auto value = evaluate(field->initializer());
      if (!value) {
        thisObject_ = std::move(savedThis);
        return false;
      }
      obj->setMember(field, std::move(*value));
      if (classSymbol->isUnion()) break;
      continue;
    }

    if (!obj->subobject(field)) {
      auto value = defaultConstruct(field->type());
      if (!value) {
        thisObject_ = std::move(savedThis);
        return false;
      }
      obj->setMember(field, std::move(*value));
    }
    if (classSymbol->isUnion()) break;
  }

  thisObject_ = std::move(savedThis);
  return true;
}

auto ASTInterpreter::subobjectSlot(const std::shared_ptr<ConstObject>& object,
                                   const Symbol* symbol) -> ConstValue* {
  if (!object || !symbol) return nullptr;

  if (auto slot = object->mutableSubobject(symbol)) return slot;

  auto classType = unqualified_cast<ClassType>(object->type());
  auto classSymbol = classType ? classType->symbol() : nullptr;
  if (classSymbol) classSymbol = classSymbol->resolvedDefinition();

  auto owner = symbol_cast<ClassSymbol>(symbol->parent());

  if (classSymbol && owner &&
      owner->resolvedDefinition() != classSymbol->resolvedDefinition()) {
    for (auto base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (!traits.is_member_of_object_type(baseClass->type(),
                                           const_cast<Symbol*>(symbol)))
        continue;

      auto baseSlot = object->mutableSubobject(base);
      if (!baseSlot) {
        baseSlot = object->addMember(
            base, ConstValue{std::make_shared<ConstObject>(baseClass->type())});
      }

      auto nested = std::get_if<std::shared_ptr<ConstObject>>(baseSlot);
      if (!nested) continue;
      if (!*nested) *nested = std::make_shared<ConstObject>(baseClass->type());
      return subobjectSlot(*nested, symbol);
    }
  }

  return object->addMember(symbol, ConstValue{IndeterminateValue{}});
}

auto ASTInterpreter::constructSubobject(MemInitializerAST* ast,
                                        const Type* type,
                                        std::vector<ConstValue> args)
    -> std::optional<ConstValue> {
  if (auto array = type_cast<BoundedArrayType>(traits.remove_cv(type))) {
    auto elements = std::make_shared<InitializerList>();
    for (std::size_t i = 0; i < array->size(); ++i) {
      auto element = constructSubobject(ast, array->elementType(), args);
      if (!element) return std::nullopt;
      elements->elements.emplace_back(std::move(*element),
                                      array->elementType());
    }
    return elements;
  }
  auto paren = ast_cast<ParenMemInitializerAST>(ast);
  const auto valueInitialized =
      paren && paren->lparenLoc && !paren->expressionList;

  if (valueInitialized &&
      traits.requires_zero_initialization(type, ast->constructor)) {
    auto zero = zeroInitialize(type);
    if (!zero) return std::nullopt;
    if (traits.is_trivially_constructible(type, {})) return zero;
    auto object = std::get_if<std::shared_ptr<ConstObject>>(&*zero);
    if (!object || !*object) return std::nullopt;
    return evaluateConstructor(ast->constructor, type, std::move(args),
                               *object);
  }

  return evaluateConstructor(ast->constructor, type, std::move(args));
}

void ASTInterpreter::applyMemInitializer(MemInitializerAST* ast,
                                         std::vector<ConstValue> args) {
  if (!ast->symbol || !thisObject_) return;

  if (auto cls = symbol_cast<ClassSymbol>(ast->symbol)) {
    if (cls != currentConstructorClass_) return;
    if (!ast->constructor) return;
    auto result = evaluateConstructor(ast->constructor, thisObject_->type(),
                                      std::move(args));
    if (result) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&*result)) {
        if (*obj) *thisObject_ = **obj;
      }
    }
    return;
  }

  if (auto base = symbol_cast<BaseClassSymbol>(ast->symbol)) {
    if (base->isVirtual()) return;
    auto baseClassSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSym) return;
    if (ast->constructor) {
      auto result =
          constructSubobject(ast, baseClassSym->type(), std::move(args));
      if (result)
        thisObject_->setMember(base, std::move(*result));
      else
        aborted_ = true;
    } else if (!args.empty()) {
      thisObject_->setMember(base, std::move(args.front()));
    }
    return;
  }

  auto field = symbol_cast<FieldSymbol>(ast->symbol);
  if (!field) return;

  if (ast->constructor) {
    auto result = constructSubobject(ast, field->type(), std::move(args));
    if (result)
      thisObject_->setMember(field, std::move(*result));
    else
      aborted_ = true;
    return;
  }

  if (!args.empty()) thisObject_->setMember(field, std::move(args.front()));
}

auto ASTInterpreter::defaultConstruct(const Type* type)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto unqualified = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqualified)) {
    auto elements = std::make_shared<InitializerList>();
    elements->elements.reserve(arrayType->size());
    for (std::size_t index = 0; index < arrayType->size(); ++index) {
      auto value = defaultConstruct(arrayType->elementType());
      if (!value) return std::nullopt;
      elements->elements.emplace_back(std::move(*value),
                                      arrayType->elementType());
    }
    return ConstValue{std::move(elements)};
  }

  if (!traits.is_class(unqualified)) return ConstValue{IndeterminateValue{}};

  auto classType = type_cast<ClassType>(unqualified);
  if (!classType || !classType->symbol()) return std::nullopt;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  auto constructor = classSymbol->defaultConstructor();
  if (!constructor) return std::nullopt;
  if (!constructor->isConstexpr()) return std::nullopt;
  return evaluateConstructor(constructor, type, {});
}

void ASTInterpreter::pushFrame() { frames_.push_back({}); }

void ASTInterpreter::popFrame() {
  if (!frames_.empty()) frames_.pop_back();
}

void ASTInterpreter::retireFrame() {
  if (frames_.empty()) return;
  retiredFrames_.push_back(std::move(frames_.back()));
  frames_.pop_back();
}

auto ASTInterpreter::beginAutomaticScope() const -> std::size_t {
  if (frames_.empty()) return 0;
  return frames_.back().automaticObjects.size();
}

void ASTInterpreter::registerAutomaticObject(VariableSymbol* variable) {
  if (!variable || variable->isStatic()) return;
  if (frames_.empty()) return;
  auto type = traits.remove_cv(variable->type());
  if (!traits.is_class(type) && !traits.is_array(type)) return;
  frames_.back().automaticObjects.push_back(variable);
}

auto ASTInterpreter::endAutomaticScope(std::size_t mark) -> bool {
  if (frames_.empty()) return true;
  auto& objects = frames_.back().automaticObjects;
  if (mark > objects.size()) return false;
  while (objects.size() > mark) {
    auto variable = objects.back();
    objects.pop_back();
    auto value = lookupLocalSlot(variable);
    if (!value) continue;
    if (destroyValue(variable->type(), *value)) continue;
    aborted_ = true;
    return false;
  }
  return true;
}

auto ASTInterpreter::destroyValue(const Type* type, ConstValue& value) -> bool {
  auto unqual = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqual)) {
    auto elements = std::get_if<std::shared_ptr<InitializerList>>(&value);
    if (!elements || !*elements) return false;
    for (auto it = (*elements)->elements.rbegin();
         it != (*elements)->elements.rend(); ++it) {
      auto& [element, elementType] = *it;
      auto typeToDestroy = elementType;
      if (!typeToDestroy) typeToDestroy = arrayType->elementType();
      if (!destroyValue(typeToDestroy, element)) return false;
    }
    return true;
  }

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return true;
  if (traits.has_trivial_destructor(unqual)) return true;

  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!object || !*object) return false;
  auto classSymbol = classType->definition();
  traits.requireCompleteClass(classSymbol);
  if (!classSymbol || !classSymbol->isComplete()) return false;
  auto destructor = classSymbol->destructor();
  if (!destructor || destructor->isDeleted()) return false;

  if (!destructor->isDefaulted()) {
    if (!destructor->isConstexpr()) return false;
    auto savedReturnValue = std::move(returnValue_);
    auto savedCaptureReturnLValue = std::exchange(captureReturnLValue_, false);
    auto savedReturnLValue = std::exchange(returnLValue_, nullptr);
    auto savedCaptureReturnAddress =
        std::exchange(captureReturnAddress_, false);
    auto savedReturnAddress = std::move(returnAddress_);
    (void)evaluateCall(destructor, {}, *object);
    returnValue_ = std::move(savedReturnValue);
    captureReturnLValue_ = savedCaptureReturnLValue;
    returnLValue_ = savedReturnLValue;
    captureReturnAddress_ = savedCaptureReturnAddress;
    returnAddress_ = std::move(savedReturnAddress);
    if (aborted_) return false;
  }

  auto& members = (*object)->mutableMembers();
  for (auto it = members.rbegin(); it != members.rend(); ++it) {
    auto memberType =
        traits.aggregate_element_type(const_cast<Symbol*>(it->symbol));
    if (!memberType) continue;
    if (!destroyValue(memberType, it->value)) return false;
  }

  return true;
}

auto ASTInterpreter::executeFunction(FunctionSymbol* function, Frame frame,
                                     CallResultKind kind,
                                     std::shared_ptr<ConstObject> object,
                                     bool constructor) -> CallResult {
  if (!function || !function->isConstexpr() || depth_ >= kMaxDepth) return {};
  auto definition = function->definition();
  if (!definition) definition = function;
  if (definition->hasPendingBody())
    ASTRewriter::completePendingBodyFor(unit_, definition);
  auto declaration = definition->declaration();
  auto body = declaration ? ast_cast<CompoundStatementFunctionBodyAST>(
                                declaration->functionBody)
                          : nullptr;
  if (!body) return {};

  auto savedValue = std::exchange(returnValue_, std::nullopt);
  auto savedLValue = std::exchange(returnLValue_, nullptr);
  auto savedAddress = std::exchange(returnAddress_, std::nullopt);
  auto savedCaptureLValue =
      std::exchange(captureReturnLValue_, kind == CallResultKind::kLValue);
  auto savedCaptureAddress =
      std::exchange(captureReturnAddress_, kind == CallResultKind::kAddress);
  auto savedFunction = std::exchange(currentFunction_, function);
  auto savedThis = thisObject_;
  if (object) thisObject_ = std::move(object);
  auto savedConstructor = std::exchange(
      currentConstructorClass_,
      constructor ? symbol_cast<ClassSymbol>(function->parent()) : nullptr);
  auto savedContext =
      std::exchange(defaultInitializerContext_, DefaultInitializerContext{});
  ++depth_;
  frames_.push_back(std::move(frame));
  if (constructor) {
    for (auto initializer : ListView{body->memInitializerList}) {
      (void)memInitializer(initializer);
      if (aborted_) break;
    }
    if (!aborted_ && currentConstructorClass_ &&
        !currentConstructorClass_->isUnion()) {
      for (auto field :
           views::members(currentConstructorClass_->resolvedDefinition()) |
               views::non_static_fields) {
        if (thisObject_->subobject(field)) continue;
        auto value = defaultConstruct(field->type());
        if (!value) {
          aborted_ = true;
          break;
        }
        thisObject_->setMember(field, std::move(*value));
      }
    }
  }
  if (!aborted_ && body->statement) (void)statement(body->statement);
  if (auto type = type_cast<FunctionType>(function->type());
      type && traits.is_void(type->returnType()) && !returnValue_)
    returnValue_ = ConstValue{std::intmax_t{0}};
  CallResult result;
  if (constructor)
    result.value = thisObject_;
  else if (kind == CallResultKind::kAddress)
    result.value = returnAddress_;
  else if (kind == CallResultKind::kLValue)
    result.lvalue = returnLValue_;
  else
    result.value = returnValue_;
  if (result.lvalue)
    retireFrame();
  else
    popFrame();
  --depth_;
  returnValue_ = std::move(savedValue);
  returnLValue_ = savedLValue;
  returnAddress_ = std::move(savedAddress);
  captureReturnLValue_ = savedCaptureLValue;
  captureReturnAddress_ = savedCaptureAddress;
  currentFunction_ = savedFunction;
  thisObject_ = std::move(savedThis);
  currentConstructorClass_ = savedConstructor;
  defaultInitializerContext_ = savedContext;
  if (aborted_) return {};
  return result;
}

auto ASTInterpreter::evaluateCall(FunctionSymbol* func,
                                  std::vector<ConstValue> args,
                                  std::shared_ptr<ConstObject> thisObject)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  Frame frame;
  if (!func || !func->isConstexpr() || !bindParameters(frame, func, args))
    return std::nullopt;
  return executeFunction(func, std::move(frame), CallResultKind::kValue,
                         std::move(thisObject))
      .value;
}

auto ASTInterpreter::evaluateCallLValue(FunctionSymbol* func,
                                        std::vector<ConstValue> args)
    -> ConstValue* {
  Frame frame;
  if (!func || !func->isConstexpr() || !bindParameters(frame, func, args))
    return nullptr;
  return executeFunction(func, std::move(frame), CallResultKind::kLValue)
      .lvalue;
}

auto ASTInterpreter::evaluateConstructorFromExprs(
    FunctionSymbol* constructor, const Type* type,
    const std::vector<ExpressionAST*>& arguments) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (!constructor || !constructor->isConstexpr()) return std::nullopt;
  if (constructor->isDefaulted()) {
    std::vector<ConstValue> values;
    for (auto argument : arguments) {
      auto value = expression(argument);
      if (!value) return std::nullopt;
      values.push_back(std::move(*value));
    }
    return evaluateConstructor(constructor, type, std::move(values));
  }
  Frame frame;
  if (!bindParametersFromExprs(frame, constructor, arguments))
    return std::nullopt;
  return executeFunction(constructor, std::move(frame), CallResultKind::kValue,
                         std::make_shared<ConstObject>(type), true)
      .value;
}

auto ASTInterpreter::evaluateConstructor(FunctionSymbol* ctor,
                                         const Type* classType,
                                         std::vector<ConstValue> args,
                                         std::shared_ptr<ConstObject> object)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (!ctor) return std::nullopt;
  if (!ctor->isConstexpr()) return std::nullopt;

  auto defn = ctor->definition();
  if (!defn) defn = ctor;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto body = funcDef->functionBody;
  if (!body) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  auto defaultedClass = symbol_cast<ClassSymbol>(ctor->parent());
  if (defaultedClass) defaultedClass = defaultedClass->resolvedDefinition();

  if (ctor->isDefaulted() && defaultedClass &&
      ast_cast<DefaultFunctionBodyAST>(body)) {
    auto defaultConstructor = defaultedClass->defaultConstructor();
    if (defaultConstructor &&
        ctor->canonical() == defaultConstructor->canonical()) {
      auto obj = object ? object : std::make_shared<ConstObject>(classType);
      ++depth_;
      auto initialized = initializeDefaultedObject(obj, defaultedClass);
      --depth_;
      if (!initialized) return std::nullopt;
      return ConstValue{std::move(obj)};
    }
  }

  if (ast_cast<DefaultFunctionBodyAST>(body)) {
    auto classSymbol = defaultedClass;
    if (!classSymbol) return std::nullopt;

    auto copy = classSymbol->copyConstructor();
    auto move = classSymbol->moveConstructor();
    auto canonical = ctor->canonical();
    auto copiesObject = copy && canonical == copy->canonical();
    if (move && canonical == move->canonical()) copiesObject = true;
    if (!copiesObject || args.size() != 1) return std::nullopt;
    auto source = std::get_if<std::shared_ptr<ConstObject>>(&args.front());
    if (!source || !*source) return std::nullopt;
    return cloneValue(args.front());
  }

  Frame frame;
  if (!bindParameters(frame, ctor, args)) return std::nullopt;
  auto obj = object ? object : std::make_shared<ConstObject>(classType);
  return executeFunction(ctor, std::move(frame), CallResultKind::kValue,
                         std::move(obj), true)
      .value;
}
}  // namespace cxx
