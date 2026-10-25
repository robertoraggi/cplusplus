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
#include <cxx/function_body.h>
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
#include <cmath>
#include <format>
#include <limits>
#include <type_traits>

namespace cxx {
namespace {
template <typename Integer>
auto floatingToInteger(long double value,
                       int bits = std::numeric_limits<Integer>::digits +
                                  std::numeric_limits<Integer>::is_signed)
    -> std::optional<Integer> {
  const auto bound =
      std::ldexp(1.0L, bits - std::numeric_limits<Integer>::is_signed);
  const auto lower = std::numeric_limits<Integer>::is_signed ? -bound : 0.0L;
  const auto truncated = std::trunc(value);
  if (!std::isfinite(truncated) || truncated < lower || truncated >= bound)
    return std::nullopt;
  return static_cast<Integer>(truncated);
}

struct ToInt {
  auto operator()(bool v) const -> std::optional<std::intmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(ConstInt v) const -> std::optional<std::intmax_t> {
    return v.toIntMax();
  }

  auto operator()(ConstFloat16 v) const -> std::optional<std::intmax_t> {
    return floatingToInteger<std::intmax_t>(v.value());
  }

  auto operator()(float v) const -> std::optional<std::intmax_t> {
    return floatingToInteger<std::intmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::intmax_t> {
    return floatingToInteger<std::intmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::intmax_t> {
    return floatingToInteger<std::intmax_t>(v);
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

  auto operator()(ConstInt v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v.toWideValue());
  }

  auto operator()(ConstFloat16 v) const -> std::optional<std::uintmax_t> {
    return floatingToInteger<std::uintmax_t>(v.value());
  }

  auto operator()(float v) const -> std::optional<std::uintmax_t> {
    return floatingToInteger<std::uintmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::uintmax_t> {
    return floatingToInteger<std::uintmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::uintmax_t> {
    return floatingToInteger<std::uintmax_t>(v);
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
  [[nodiscard]] auto operator()(
      const std::shared_ptr<ConstComplex>& value) const -> std::optional<T> {
    if (!value) return std::nullopt;
    return std::visit(*this, value->real());
  }

  [[nodiscard]] auto operator()(ConstFloat16 value) const -> std::optional<T> {
    return static_cast<T>(value.value());
  }

  [[nodiscard]] auto operator()(ConstInt value) const -> std::optional<T> {
    if (value.isSigned()) return static_cast<T>(value.toIntMax());
    return static_cast<T>(value.toUIntMax());
  }

  [[nodiscard]] auto operator()(auto value) const -> std::optional<T> {
    if constexpr (std::is_arithmetic_v<decltype(value)>)
      return static_cast<T>(value);
    return std::nullopt;
  }
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

  auto operator()(const std::shared_ptr<ConstAddress>& address) const
      -> std::optional<bool> {
    if (!address || address->lifetimeEnded()) return std::nullopt;
    return !address->isNull();
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

  auto operator()(ConstFloat16 value) const -> std::optional<bool> {
    return (value.bits & 0x7fff) != 0;
  }

  auto operator()(ConstInt value) const -> std::optional<bool> {
    return !value.isZero();
  }

  auto operator()(const std::shared_ptr<InitializerList>&) const
      -> std::optional<bool> {
    return std::nullopt;
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

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::evaluate(ExpressionAST* ast) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto result = expression(ast);
  if (activeEvaluations_ == 1) endTemporaries(0);
  if (aborted_) return std::nullopt;
  return result;
}

auto ASTInterpreter::evaluateAddress(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  return addressOfLvalue(ast);
}

auto ASTInterpreter::evaluateStaticDataMember(FieldSymbol* field)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  return evaluateStaticField(field);
}

auto ASTInterpreter::toBool(const ConstValue& value) -> std::optional<bool> {
  return std::visit(ToBool{*this}, value);
}

auto ASTInterpreter::toInt(const ConstValue& value)
    -> std::optional<std::intmax_t> {
  return std::visit(ToInt{}, value);
}

auto ASTInterpreter::toOffset(const ConstValue& value) const
    -> std::optional<std::intmax_t> {
  auto integer = std::get_if<ConstInt>(&value);
  if (!integer) return std::nullopt;
  auto limit =
      static_cast<ConstInt::UWide>(std::numeric_limits<std::intmax_t>::max());
  if (integer->isNegative()) ++limit;
  if (integer->magnitude() > limit) return std::nullopt;
  return integer->toIntMax();
}

auto ASTInterpreter::memberObjectPointerOffset(const ConstValue& value) const
    -> std::optional<std::intmax_t> {
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  if (!address || !*address) return std::nullopt;
  if (!(*address)->symbol())
    return control()->memoryLayout()->nullMemberObjectPointer();
  return (*address)->memberAdjustment();
}

auto ASTInterpreter::toUInt(const ConstValue& value)
    -> std::optional<std::uintmax_t> {
  return std::visit(ToUInt{}, value);
}

auto ASTInterpreter::toIntegralType(const ConstValue& value, const Type* type)
    -> std::optional<ConstValue> {
  auto traits = translationUnit()->typeTraits();

  auto representation = traits.integral_representation(type);
  if (!representation) return std::nullopt;

  auto floating = std::visit(
      [](const auto& stored) -> std::optional<long double> {
        using T = std::decay_t<decltype(stored)>;
        if constexpr (std::is_same_v<T, ConstFloat16>)
          return stored.value();
        else if constexpr (std::is_floating_point_v<T>)
          return stored;
        else
          return std::nullopt;
      },
      value);

  ConstInt::Wide raw = 0;
  if (floating) {
    if (representation->isSigned) {
      auto result =
          floatingToInteger<ConstInt::Wide>(*floating, representation->bits);
      if (!result) return std::nullopt;
      raw = *result;
    } else {
      auto result =
          floatingToInteger<ConstInt::UWide>(*floating, representation->bits);
      if (!result) return std::nullopt;
      raw = static_cast<ConstInt::Wide>(*result);
    }
  } else if (auto stored = std::get_if<ConstInt>(&value)) {
    raw = stored->toWideValue();
  } else if (representation->isSigned) {
    auto result = toInt(value);
    if (!result.has_value()) return std::nullopt;
    raw = static_cast<ConstInt::Wide>(*result);
  } else {
    auto result = toUInt(value);
    if (!result.has_value()) return std::nullopt;
    raw = static_cast<ConstInt::Wide>(static_cast<ConstInt::UWide>(*result));
  }

  auto converted = traits.integral_constant(type, raw);
  if (!converted) return std::nullopt;

  return ConstValue{*converted};
}

auto ASTInterpreter::toArithmeticType(const ConstValue& value, const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;

  const auto holdsArithmetic =
      std::holds_alternative<ConstInt>(value) ||
      std::holds_alternative<ConstFloat16>(value) ||
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

    case TypeKind::kHalf:
    case TypeKind::kFloat16: {
      if (std::holds_alternative<ConstFloat16>(value)) return value;
      auto result = toDouble(value);
      auto format = control()->memoryLayout()->floatingPointFormat(type);
      if (!result || !format) return std::nullopt;
      return ConstValue{ConstFloat16{
          static_cast<std::uint16_t>(format->representation(*result))}};
    }

    default:
      break;
  }

  if (traits.is_floating_point(type)) return std::nullopt;

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

auto ASTInterpreter::lookupLocal(const Symbol* sym)
    -> std::optional<ConstValue> {
  auto slot = lookupLocalSlot(sym);
  if (!slot) return std::nullopt;
  if (asConstexprUnknownObject(*slot)) return std::nullopt;
  return *slot;
}

auto ASTInterpreter::lookupLocalStorage(const Symbol* sym)
    -> std::shared_ptr<ConstStorage> {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return found->second;
  }
  return nullptr;
}

auto ASTInterpreter::lookupLocalSlot(const Symbol* sym) -> ConstValue* {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto address = it->referenceAddresses.find(sym);
    if (address != it->referenceAddresses.end()) {
      auto pointer =
          std::get_if<std::shared_ptr<ConstAddress>>(&address->second);
      if (pointer && *pointer && (*pointer)->lifetimeEnded()) return nullptr;
    }
    auto ref = it->refs.find(sym);
    if (ref != it->refs.end()) return ref->second;
    auto found = it->locals.find(sym);
    if (found == it->locals.end()) continue;
    if (found->second->lifetimeEnded()) return nullptr;
    return &found->second->value();
  }
  return nullptr;
}

auto ASTInterpreter::materializedTemporary(ExpressionAST* ast)
    -> ExpressionAST* {
  while (ast) {
    if (auto nested = ast_cast<NestedExpressionAST>(ast)) {
      ast = nested->expression;
      continue;
    }
    if (!is_glvalue(ast)) return ast;
    auto cast = ast_cast<ImplicitCastExpressionAST>(ast);
    if (!cast) return nullptr;
    if (cast->castKind == ImplicitCastKind::kTemporaryMaterializationConversion)
      return cast->expression;
    if (cast->castKind != ImplicitCastKind::kQualificationConversion &&
        cast->castKind != ImplicitCastKind::kIdentity)
      return nullptr;
    ast = cast->expression;
  }
  return nullptr;
}

auto ASTInterpreter::materialize(ExpressionAST* ast)
    -> std::shared_ptr<ConstAddress> {
  auto storage =
      std::make_shared<ConstStorage>(ast->type, IndeterminateValue{});
  auto address = std::make_shared<ConstAddress>();
  address->setStorage(storage);
  temporaries_.push_back({ast, storage});
  auto value = expression(ast, address);
  if (!value) return nullptr;
  storage->setValue(std::move(*value));
  return address;
}

auto ASTInterpreter::bindReferenceTo(Frame& frame, Symbol* reference,
                                     ExpressionAST* initializer) -> bool {
  auto temporaryMark = temporaries_.size();
  auto value = is_prvalue(initializer)
                   ? std::optional<ConstValue>{materialize(initializer)}
                   : addressOfLvalue(initializer);
  if (value) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
    if (address && *address) {
      if (symbol_cast<VariableSymbol>(reference)) {
        auto extended = materializedTemporaries(traits, initializer);
        for (auto index = temporaryMark; index < temporaries_.size();) {
          auto& temporary = temporaries_[index];
          if (std::ranges::find(extended, temporary.expression) ==
              extended.end()) {
            ++index;
            continue;
          }
          frame.locals.insert_or_assign(reference, temporary.storage);
          temporaries_.erase(temporaries_.begin() + index);
        }
      }
      frame.referenceAddresses.insert_or_assign(reference, *value);
      if (auto slot = addressSlot(**address, 0, reference->type())) {
        frame.refs.insert_or_assign(reference, slot);
        return true;
      }
      if (auto referent = loadAddress(**address, 0, reference->type())) {
        frame.locals.insert_or_assign(
            reference, std::make_shared<ConstStorage>(reference->type(),
                                                      std::move(*referent)));
        return true;
      }
    }
  }

  auto slot = lvalue(initializer);
  if (!slot) return false;
  frame.refs.insert_or_assign(reference, slot);
  return true;
}

void ASTInterpreter::setLocal(const Symbol* sym, ConstValue value) {
  if (frames_.empty()) frames_.push_back({});
  frames_.back().locals.insert_or_assign(
      sym, std::make_shared<ConstStorage>(sym->type(), std::move(value)));
}

auto ASTInterpreter::definingDeclarationOf(FunctionSymbol* function)
    -> FunctionSymbol* {
  auto definition = function->resolvedDefinition();
  (void)ASTRewriter::completePendingBodyFor(unit_, definition);
  return definition;
}

auto ASTInterpreter::evaluateWithParameters(
    FunctionSymbol* function, std::span<ExpressionAST* const> arguments,
    ExpressionAST* condition) -> std::optional<ConstValue> {
  if (aborted_) return std::nullopt;
  auto guard = EvaluationScope{*this};
  pushFrame();
  auto parameters = function->parameters();
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    auto argument = index < arguments.size() ? arguments[index] : nullptr;
    if (!argument)
      argument = ASTRewriter::requireDefaultArgument(unit_, parameters[index]);
    const auto bound =
        argument &&
        bindOneParameter(frames_.back(), parameters[index], argument);
    if (std::exchange(aborted_, false) || !bound) {
      frames_.back().refs.erase(parameters[index]);
      frames_.back().referenceAddresses.erase(parameters[index]);
      frames_.back().locals.insert_or_assign(
          parameters[index],
          std::make_shared<ConstStorage>(parameters[index]->type(),
                                         IndeterminateValue{}));
    }
  }
  auto result = evaluate(condition);
  popFrame();
  return result;
}

auto ASTInterpreter::bindParameters(Frame& frame, FunctionSymbol* func,
                                    std::vector<ConstValue>& args) -> bool {
  auto params = definingDeclarationOf(func)->parameters();
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (i < args.size()) {
      auto value = traits.is_reference(params[i]->type()) ? args[i]
                                                          : cloneValue(args[i]);
      frame.locals.insert_or_assign(
          params[i],
          std::make_shared<ConstStorage>(params[i]->type(), std::move(value)));
      frame.automaticObjects.push_back(params[i]);
    } else {
      auto defaultArgument =
          ASTRewriter::requireDefaultArgument(unit_, params[i]);
      if (!defaultArgument) return false;
      if (!bindOneParameter(frame, params[i], defaultArgument)) return false;
    }
  }
  return true;
}

auto ASTInterpreter::bindOneParameter(Frame& frame, Symbol* paramSymbol,
                                      ExpressionAST* argExpr) -> bool {
  auto param = symbol_cast<ParameterSymbol>(paramSymbol);
  if (param && traits.is_reference(param->type()) &&
      bindReferenceTo(frame, param, argExpr)) {
    return true;
  }
  auto storage =
      std::make_shared<ConstStorage>(paramSymbol->type(), IndeterminateValue{});
  auto destination = std::make_shared<ConstAddress>(paramSymbol);
  destination->setStorage(storage);
  if (auto value =
          evaluateInitializer(paramSymbol->type(), argExpr, destination)) {
    storage->setValue(std::move(*value));
    frame.locals.insert_or_assign(paramSymbol, std::move(storage));
    frame.automaticObjects.push_back(paramSymbol);
    return true;
  }

  auto object = constexprUnknownObject(argExpr);
  if (!object) return false;
  frame.locals.insert_or_assign(
      paramSymbol,
      std::make_shared<ConstStorage>(paramSymbol->type(), std::move(object)));
  frame.automaticObjects.push_back(paramSymbol);
  return true;
}

auto ASTInterpreter::bindParametersFromExprs(
    Frame& frame, FunctionSymbol* function,
    std::span<ExpressionAST* const> arguments, bool rightToLeft) -> bool {
  auto parameters = definingDeclarationOf(function)->parameters();
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    auto index = i;
    if (rightToLeft) index = parameters.size() - i - 1;
    auto argument = index < arguments.size() ? arguments[index] : nullptr;
    if (!argument)
      argument = ASTRewriter::requireDefaultArgument(unit_, parameters[index]);
    if (!argument || !bindOneParameter(frame, parameters[index], argument))
      return false;
  }
  for (std::size_t i = parameters.size(); i < arguments.size(); ++i)
    if (!expression(arguments[i])) return false;
  return true;
}

auto ASTInterpreter::initializeDefaultedObject(
    const std::shared_ptr<ConstObject>& obj, ClassSymbol* classSymbol,
    std::shared_ptr<ConstAddress> destination) -> bool {
  if (!obj || !classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();
  ThisScope thisScope{*this, obj, destination};
  ConstructionScope construction{*this, obj, destination};

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!obj->subobject(base)) obj->setMember(base, IndeterminateValue{});
    auto address = subobjectDestination(destination, base);
    auto value = defaultConstruct(baseClass->type(), std::move(address));
    if (!value) return false;
    obj->setMember(base, std::move(*value));
  }

  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    if (classSymbol->isUnion() && !field->initializer()) continue;

    if (field->initializer()) {
      if (!obj->subobject(field)) obj->setMember(field, IndeterminateValue{});
      auto address = subobjectDestination(destination, field);
      auto value =
          initializationValue(field->type(), field->constructor(),
                              field->initializer(), std::move(address));
      if (!value) return false;
      obj->setMember(field, std::move(*value));
      if (classSymbol->isUnion()) break;
      continue;
    }

    const auto existing = obj->subobject(field);
    if (!existing ||
        traits.is_class(traits.remove_all_extents(field->type()))) {
      if (!existing) obj->setMember(field, IndeterminateValue{});
      auto address = subobjectDestination(destination, field);
      auto value = defaultConstruct(field->type(), std::move(address));
      if (!value) return false;
      obj->setMember(field, std::move(*value));
    }
    if (classSymbol->isUnion()) break;
  }

  return true;
}

auto ASTInterpreter::copyDefaultedObject(
    const std::shared_ptr<ConstObject>& obj,
    const std::shared_ptr<ConstObject>& source, ClassSymbol* classSymbol)
    -> bool {
  if (!obj || !source || !classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();

  if (classSymbol->isUnion()) {
    for (const auto& member : source->members())
      obj->addMember(member.symbol, cloneValue(member.value));
    return true;
  }

  for (auto base : classSymbol->baseClasses()) {
    auto value = source->subobject(base);
    if (!value) return false;
    obj->addMember(base, cloneValue(*value));
  }

  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    auto value = source->subobject(field);
    if (!value) return false;
    obj->setMember(field, cloneValue(*value));
  }

  return true;
}

auto ASTInterpreter::subobjectSlot(const std::shared_ptr<ConstObject>& object,
                                   const Symbol* symbol) -> ConstValue* {
  if (!object || !symbol) return nullptr;
  if (object->isConstexprUnknown()) return nullptr;

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

auto ASTInterpreter::copyArrayElements(
    const Type* type, FunctionSymbol* constructor, const ConstValue& source,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  if (!constructor) return cloneValue(source);
  auto array = unqualified_cast<BoundedArrayType>(type);
  if (!array)
    return evaluateConstructor(constructor, type, {source}, {},
                               std::move(destination));

  auto sourceElements = std::get_if<std::shared_ptr<InitializerList>>(&source);
  if (!sourceElements || !*sourceElements) return std::nullopt;
  auto elements = constructionArray(array, destination);
  if (!elements) return std::nullopt;
  for (std::size_t index = 0; index < (*sourceElements)->size(); ++index) {
    const auto& [value, elementType] = (*sourceElements)->element(index);
    auto address = arrayElementDestination(destination, type, index);
    auto element = copyArrayElements(array->elementType(), constructor, value,
                                     std::move(address));
    if (!element) return std::nullopt;
    elements->mutableElement(index) = {std::move(*element),
                                       array->elementType()};
  }
  return ConstValue{std::move(elements)};
}

auto ASTInterpreter::constructionArray(
    const BoundedArrayType* type,
    const std::shared_ptr<ConstAddress>& destination)
    -> std::shared_ptr<InitializerList> {
  ConstValue* slot = nullptr;
  if (destination) {
    slot = addressSlot(*destination, 0, type);
    if (!slot) return nullptr;
    auto existing = std::get_if<std::shared_ptr<InitializerList>>(slot);
    if (existing && *existing) return *existing;
  }
  auto elements = std::make_shared<InitializerList>();
  elements->setSize(type->size());
  if (slot) *slot = elements;
  return elements;
}

auto ASTInterpreter::constructSubobject(
    MemInitializerAST* ast, const Type* type,
    const std::vector<ExpressionAST*>& arguments,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  if (traits.is_array(type)) {
    if (arguments.size() == 1 &&
        isWholeArrayCopy(traits, arguments.front(), type)) {
      auto source = expression(arguments.front());
      if (!source) return std::nullopt;
      return copyArrayElements(type, ast->constructor, *source,
                               std::move(destination));
    }
  }

  auto paren = ast_cast<ParenMemInitializerAST>(ast);
  auto braced = ast_cast<BracedMemInitializerAST>(ast);
  const auto valueInitialized =
      arguments.empty() &&
      ((paren && paren->lparenLoc) || (braced && braced->bracedInitList));

  return classInitializationValue(type, ast->constructor, arguments,
                                  valueInitialized, std::move(destination));
}

void ASTInterpreter::applyMemInitializer(
    MemInitializerAST* ast, const std::vector<ExpressionAST*>& arguments) {
  if (!ast->symbol || !thisObject_) return;

  if (auto cls = symbol_cast<ClassSymbol>(ast->symbol)) {
    if (cls != currentConstructorClass_) return;
    if (!ast->constructor) return;
    auto result = evaluateConstructorFromExprs(
        ast->constructor, thisObject_->type(), arguments, thisAddress_);
    if (result) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&*result)) {
        if (*obj) *thisObject_ = **obj;
      }
    }
    return;
  }

  auto subobject = ast->symbol;
  if (auto base = symbol_cast<BaseClassSymbol>(subobject)) {
    if (base->isVirtual()) return;
    auto baseClassSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSym) return;
    initializeSubobject(ast, base, baseClassSym->type(), arguments);
    return;
  }

  auto field = symbol_cast<FieldSymbol>(subobject);
  if (!field) return;
  initializeSubobject(ast, field, field->type(), arguments);
}

void ASTInterpreter::initializeSubobject(
    MemInitializerAST* ast, Symbol* subobject, const Type* type,
    const std::vector<ExpressionAST*>& arguments) {
  if (!ast->constructor && arguments.empty()) return;

  if (!thisObject_->subobject(subobject))
    thisObject_->setMember(subobject, IndeterminateValue{});
  auto destination = subobjectDestination(thisAddress_, subobject);
  auto value = ast->constructor
                   ? constructSubobject(ast, type, arguments, destination)
                   : initialValue(type, arguments.front(), destination);
  if (!value) {
    aborted_ = true;
    return;
  }
  thisObject_->setMember(subobject, std::move(*value));
}

auto ASTInterpreter::defaultConstruct(const Type* type,
                                      std::shared_ptr<ConstAddress> destination,
                                      FunctionSymbol* constructor)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto elementType = traits.remove_all_extents(type);
  if (auto array = unqualified_cast<BoundedArrayType>(type);
      array && !traits.is_class(elementType)) {
    auto element = defaultConstruct(array->elementType());
    if (!element) return std::nullopt;
    return ConstValue{std::make_shared<InitializerList>(
        std::vector<InitializerList::Element>{}, array->size(),
        InitializerList::Element{std::move(*element), array->elementType()})};
  }
  if (!traits.is_class(elementType)) return ConstValue{IndeterminateValue{}};

  auto classType = unqualified_cast<ClassType>(elementType);
  if (!classType || !classType->symbol()) return std::nullopt;
  auto classSymbol = classType->symbol()->resolvedDefinition();
  if (!constructor) constructor = classSymbol->defaultConstructor();
  return classInitializationValue(type, constructor, {}, false,
                                  std::move(destination));
}

void ASTInterpreter::pushFrame() { frames_.push_back({}); }

void ASTInterpreter::popFrame() {
  if (frames_.empty()) return;
  if (!endAutomaticScope(0)) aborted_ = true;
  frames_.pop_back();
}

auto ASTInterpreter::beginAutomaticScope() const -> std::size_t {
  if (frames_.empty()) return 0;
  return frames_.back().automaticObjects.size();
}

void ASTInterpreter::registerAutomaticObject(VariableSymbol* variable) {
  if (!variable || variable->isStatic()) return;
  if (frames_.empty()) return;
  frames_.back().automaticObjects.push_back(variable);
}

auto ASTInterpreter::endAutomaticScope(std::size_t mark) -> bool {
  if (frames_.empty()) return true;
  auto& objects = frames_.back().automaticObjects;
  if (mark > objects.size()) return false;
  while (objects.size() > mark) {
    auto variable = objects.back();
    objects.pop_back();
    auto storage = lookupLocalStorage(variable);
    if (!storage) continue;
    if (destroyStorage(storage)) continue;
    aborted_ = true;
    return false;
  }
  return true;
}

void ASTInterpreter::endLifetime(const Type* type, ConstValue& value) {
  if (!traits.is_class(type) && !traits.is_array(type)) return;
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object || (*object)->lifetimeEnded()) return;
    (*object)->setLifetimeEnded(true);
    for (auto& member : (*object)->mutableMembers()) {
      if (!member.symbol) continue;
      endLifetime(member.symbol->type(), member.value);
    }
  } else if (auto list =
                 std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return;
    for (auto& [element, elementType] : (*list)->mutableInitializedElements())
      endLifetime(elementType, element);
  }
}

void ASTInterpreter::endTemporaries(std::size_t mark) {
  while (temporaries_.size() > mark) {
    auto storage = std::move(temporaries_.back().storage);
    temporaries_.pop_back();
    if (!destroyStorage(storage)) aborted_ = true;
  }
}

auto ASTInterpreter::destroyStorage(
    const std::shared_ptr<ConstStorage>& storage) -> bool {
  if (storage->lifetimeEnded()) return true;
  if (traits.is_reference(storage->type())) {
    storage->setLifetimeEnded(true);
    return true;
  }
  if (!destroyValue(storage->type(), storage->value())) return false;
  endLifetime(storage->type(), storage->value());
  storage->setLifetimeEnded(true);
  storage->setValue(IndeterminateValue{});
  return true;
}

auto ASTInterpreter::destroyValue(const Type* type, ConstValue& value) -> bool {
  auto unqual = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqual)) {
    auto elements = std::get_if<std::shared_ptr<InitializerList>>(&value);
    if (!elements || !*elements) return false;
    if (traits.has_trivial_destructor(
            traits.remove_all_extents(arrayType->elementType())))
      return true;
    if ((*elements)->size() == 0) return true;
    std::ignore = (*elements)->mutableElement((*elements)->size() - 1);
    auto initialized = (*elements)->mutableInitializedElements();
    for (auto it = initialized.rbegin(); it != initialized.rend(); ++it) {
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
  if (!destructor || isDeletedFunction(unit_, destructor)) return false;

  if (!destructor->isDefaulted()) {
    if (!destructor->isConstexpr()) return false;
    auto savedReturnValue = std::move(returnValue_);
    auto savedCaptureReturnAddress =
        std::exchange(captureReturnAddress_, false);
    auto savedReturnAddress = std::move(returnAddress_);
    (void)evaluateCall(destructor, {}, *object);
    returnValue_ = std::move(savedReturnValue);
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

auto ASTInterpreter::executeFunction(
    FunctionSymbol* function, Frame frame, CallResultKind kind,
    std::shared_ptr<ConstObject> object, bool constructor,
    std::shared_ptr<ConstAddress> objectAddress,
    std::shared_ptr<ConstAddress> resultAddress) -> CallResult {
  if (!function || !function->isConstexpr() || depth_ >= kMaxDepth) return {};
  if (object && object->lifetimeEnded()) return {};
  auto definition = definingDeclarationOf(function);
  auto declaration = definition->declaration();
  auto body = declaration ? body_statement(declaration->functionBody) : nullptr;
  if (!body) return {};

  auto functionType = type_cast<FunctionType>(function->type());
  const auto returnsReference =
      functionType && traits.is_reference(functionType->returnType());

  if (objectAddress) {
    objectAddress = callObjectAddress(function, object, *objectAddress);
    if (!objectAddress) return {};
  }

  auto savedValue = std::exchange(returnValue_, std::nullopt);
  auto savedResultAddress =
      std::exchange(resultAddress_, std::move(resultAddress));
  auto savedAddress = std::exchange(returnAddress_, std::nullopt);
  auto savedCaptureAddress =
      std::exchange(captureReturnAddress_,
                    kind != CallResultKind::kValue || returnsReference);
  auto savedAddressFallback = std::exchange(returnAddressFallsBackToValue_,
                                            kind == CallResultKind::kObject);
  auto savedFunction = std::exchange(currentFunction_, function);
  std::optional<ThisScope> thisScope;
  if (object)
    thisScope.emplace(*this, std::move(object), std::move(objectAddress));
  std::optional<ConstructionScope> construction;
  if (constructor) construction.emplace(*this, thisObject_, thisAddress_);
  auto savedConstructor = std::exchange(
      currentConstructorClass_,
      constructor ? symbol_cast<ClassSymbol>(function->parent()) : nullptr);
  auto savedContext =
      std::exchange(defaultInitializerContext_, DefaultInitializerContext{});
  ++depth_;
  frames_.push_back(std::move(frame));
  if (constructor) {
    for (auto initializer : ListView{ctor_initializers(function)}) {
      (void)memInitializer(initializer);
      if (aborted_) break;
    }
    if (!aborted_ && currentConstructorClass_ &&
        !currentConstructorClass_->isUnion()) {
      for (auto field :
           views::members(currentConstructorClass_->resolvedDefinition()) |
               views::non_static_fields) {
        if (thisObject_->subobject(field)) continue;
        if (anonymous_member_class(field)) continue;
        thisObject_->setMember(field, IndeterminateValue{});
        auto address = subobjectDestination(thisAddress_, field);
        auto value = defaultConstruct(field->type(), std::move(address));
        if (!value) {
          aborted_ = true;
          break;
        }
        thisObject_->setMember(field, std::move(*value));
      }
    }
  }
  if (!aborted_) (void)statement(body);
  if (auto type = type_cast<FunctionType>(function->type());
      type && traits.is_void(type->returnType()) && !returnValue_)
    returnValue_ = ConstValue{ConstInt{std::intmax_t{0}}};
  CallResult result;
  if (constructor)
    result.value = thisObject_;
  else if (kind == CallResultKind::kAddress || kind == CallResultKind::kLValue)
    result.value = returnAddress_;
  else if (kind == CallResultKind::kObject && returnAddress_) {
    result.value = returnAddress_;
    result.isAddress = true;
  } else if (returnsReference)
    result.value = returnAddress_;
  else
    result.value = returnValue_;
  popFrame();
  if (kind == CallResultKind::kValue && returnsReference && result.value) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*result.value);
    result.value = address && *address
                       ? loadAddress(**address, 0, functionType->returnType())
                       : std::nullopt;
  }
  if (kind == CallResultKind::kLValue && result.value) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*result.value);
    if (address && *address) result.lvalue = addressSlot(**address, 0);
    result.value.reset();
  }
  --depth_;
  returnValue_ = std::move(savedValue);
  returnAddress_ = std::move(savedAddress);
  captureReturnAddress_ = savedCaptureAddress;
  returnAddressFallsBackToValue_ = savedAddressFallback;
  currentFunction_ = savedFunction;
  resultAddress_ = std::move(savedResultAddress);
  thisScope.reset();
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
    const std::vector<ExpressionAST*>& arguments,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (!constructor || !constructor->isConstexpr()) return std::nullopt;
  auto declaration = definingDeclarationOf(constructor)->declaration();
  if (declaration &&
      ast_cast<DefaultFunctionBodyAST>(declaration->functionBody)) {
    std::vector<ConstValue> values;
    for (auto argument : arguments) {
      auto value = designatedValue(argument);
      if (!value) return std::nullopt;
      values.push_back(std::move(*value));
    }
    return evaluateConstructor(constructor, type, std::move(values), {},
                               std::move(destination));
  }
  Frame frame;
  if (!bindParametersFromExprs(frame, constructor, arguments))
    return std::nullopt;
  auto object = constructionObject(type, destination);
  if (!object) return std::nullopt;
  return executeFunction(constructor, std::move(frame), CallResultKind::kValue,
                         std::move(object), true, std::move(destination))
      .value;
}

auto ASTInterpreter::evaluateConstructor(
    FunctionSymbol* ctor, const Type* classType, std::vector<ConstValue> args,
    std::shared_ptr<ConstObject> object,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (!ctor) return std::nullopt;
  if (!ctor->isConstexpr()) return std::nullopt;

  auto defn = definingDeclarationOf(ctor);

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
      auto obj = constructionObject(classType, destination, std::move(object));
      if (!obj) return std::nullopt;
      ++depth_;
      auto initialized =
          initializeDefaultedObject(obj, defaultedClass, destination);
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
    auto obj = constructionObject(classType, destination, std::move(object));
    if (!obj) return std::nullopt;
    if (!copyDefaultedObject(obj, *source, classSymbol)) return std::nullopt;
    return ConstValue{std::move(obj)};
  }

  Frame frame;
  if (!bindParameters(frame, ctor, args)) return std::nullopt;
  auto obj = constructionObject(classType, destination, std::move(object));
  if (!obj) return std::nullopt;
  return executeFunction(ctor, std::move(frame), CallResultKind::kValue,
                         std::move(obj), true, std::move(destination))
      .value;
}
}  // namespace cxx
