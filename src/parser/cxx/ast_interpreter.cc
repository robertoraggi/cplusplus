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
#include <cxx/reflection.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <ranges>
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

  auto operator()(OutOfLifetimeValue) const -> std::optional<bool> {
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
    : unit_(unit), traits(unit), evaluationScope_(scope) {
  if (scope) currentFunction_ = scope->enclosingFunctionOrSelf();
}

auto ASTInterpreter::hasConstantEvaluationContext(
    ScopeSymbol* scope, ConstantEvaluationKind kind) const -> bool {
  if (!scope || kind == ConstantEvaluationKind::kNone) return false;
  if (kind == ConstantEvaluationKind::kImmediateInvocation) {
    auto point = scopeOfUse(scope);
    while (symbol_cast<BlockSymbol>(point)) point = point->parent();
    auto parameters = symbol_cast<FunctionParametersSymbol>(point);
    auto function = parameters
                        ? symbol_cast<FunctionSymbol>(parameters->parent())
                        : nullptr;
    if (function && function->isConsteval()) return false;
  }
  return true;
}

ASTInterpreter::EvaluationContextScope::EvaluationContextScope(
    ASTInterpreter& interpreter, ScopeSymbol* scope, SourceLocation point,
    ConstantEvaluationKind kind)
    : interpreter_(interpreter),
      saved_(interpreter.evaluationScope_),
      savedPoint_(interpreter.evaluationPoint_) {
  if (!scope || kind == ConstantEvaluationKind::kNone) return;
  if (!interpreter.hasConstantEvaluationContext(scope, kind)) return;
  interpreter_.evaluationScope_ = interpreter.scopeOfUse(scope);
  interpreter_.evaluationPoint_ = interpreter.pointOfUse(point);
}

ASTInterpreter::EvaluationScope::EvaluationScope(ASTInterpreter& interp,
                                                 ExpressionAST* ast)
    : interp_(interp) {
  if (interp_.activeEvaluations_++ || !ast) return;
  interp_.evaluationPoint_ = interp_.pointOfUse(ast->firstSourceLocation());
}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::evaluate(ExpressionAST* ast) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this, ast};
  return completeEvaluation(expression(ast));
}

auto ASTInterpreter::completeEvaluation(std::optional<ConstValue> result)
    -> std::optional<ConstValue> {
  if (activeEvaluations_ == 1) {
    endTemporaries(0);
    if (std::ranges::any_of(allocations_, [](const Allocation& allocation) {
          return !allocation.storage->lifetimeEnded();
        }))
      aborted_ = true;
  }
  if (aborted_) return std::nullopt;
  return result;
}

auto ASTInterpreter::storageDurationBeganWithinEvaluation(
    const ConstAddress& address) const -> bool {
  auto storage = address.storage();
  if (!storage) return isVariableUnderInitialization(address);
  if (storage->lifetimeEnded()) return false;
  if (std::ranges::contains(allocations_, storage, &Allocation::storage))
    return true;
  if (std::ranges::contains(exceptionObjects_, storage)) return true;
  if (std::ranges::contains(backingArrays_, storage)) return true;
  auto holdsStorage = [&](const Frame& frame) {
    return std::ranges::contains(frame.locals | std::views::values, storage);
  };
  if (std::ranges::any_of(frames_, holdsStorage)) return true;
  if (std::ranges::any_of(framesBeingBound_, [&](const Frame* frame) {
        return holdsStorage(*frame);
      }))
    return true;
  return std::ranges::any_of(temporaries_, [&](const auto& temporary) {
    return temporary.storage == storage;
  });
}

auto ASTInterpreter::allocatorElementType(FunctionSymbol* function) const
    -> const Type* {
  auto cls = symbol_cast<ClassSymbol>(function->parent());
  if (!cls || !cls->isSpecialization()) return nullptr;
  if (cls->name() != control()->getIdentifier("allocator")) return nullptr;
  if (!traits.is_std_namespace(cls->parent())) return nullptr;
  const auto allocate =
      function->name() == control()->getIdentifier("allocate");
  const auto deallocate =
      function->name() == control()->getIdentifier("deallocate");
  if (!allocate && !deallocate) return nullptr;
  auto arguments = template_arguments_of(cls);
  if (arguments.size() != 1) return nullptr;
  auto element = template_argument_as_type(arguments.front());
  if (!element || function->isStatic()) return nullptr;
  auto type = type_cast<FunctionType>(function->type());
  if (!type) return nullptr;
  const auto& parameters = type->parameterTypes();
  auto pointer = control()->getPointerType(element);
  if (allocate) {
    if (parameters != std::vector<const Type*>{control()->getSizeType()} ||
        type->returnType() != pointer)
      return nullptr;
  } else if (parameters !=
                 std::vector<const Type*>{pointer, control()->getSizeType()} ||
             !traits.is_void(type->returnType())) {
    return nullptr;
  }
  return element;
}

auto ASTInterpreter::isTrivialNominatingAssignment(AssignmentExpressionAST* ast)
    -> bool {
  if (isDefaultedUnionAssignment(ast->symbol)) return true;
  if (!ast->symbol->isDefaulted() || !nominatesSubobject(ast->leftExpression))
    return false;
  if (!traits.is_trivial_special_member(ast->symbol)) return false;
  return traits.is_same(traits.remove_cvref(ast->leftExpression->type),
                        traits.remove_cvref(ast->rightExpression->type));
}

auto ASTInterpreter::isDefaultedUnionAssignment(FunctionSymbol* function) const
    -> bool {
  auto cls = symbol_cast<ClassSymbol>(function->parent());
  if (!cls || !cls->isUnion() || !function->isDefaulted()) return false;
  if (traits.is_deleted_function(function)) return false;
  return traits.is_copy_assignment(function) ||
         traits.is_move_assignment(function);
}

auto ASTInterpreter::isVariableUnderInitialization(
    const ConstAddress& address) const -> bool {
  return std::ranges::contains(
      initializations_, address.root(),
      [](const auto& initialization) { return initialization.first; });
}

auto ASTInterpreter::isUnderConstruction(const ConstAddress& address) const
    -> bool {
  auto storage = address.storage();
  if (!storage) return isVariableUnderInitialization(address);
  auto underConstruction = [this](const ConstValue& value) {
    auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
    if (!object) return false;
    return std::ranges::contains(constructionPhases_, *object,
                                 &ConstructionPhase::completeObject) ||
           std::ranges::contains(objectsUnderDestruction_, *object);
  };
  const ConstValue* current = &storage->value();
  auto element = [&current](std::size_t index) {
    auto array = std::get_if<std::shared_ptr<InitializerList>>(current);
    if (!array || !*array || index >= (*array)->size()) return false;
    current = &std::get<0>((*array)->distinctElement(index));
    return true;
  };
  if (underConstruction(*current)) return true;
  for (const auto& subobject : address.subobjectPath()) {
    if (auto symbol = std::get_if<Symbol*>(&subobject)) {
      auto object = std::get_if<std::shared_ptr<ConstObject>>(current);
      current = object && *object ? (*object)->subobject(*symbol) : nullptr;
      if (!current) return false;
    } else if (!element(std::get<std::size_t>(subobject))) {
      return false;
    }
    if (underConstruction(*current)) return true;
  }
  if (!address.arrayType()) return false;
  if (!element(static_cast<std::size_t>(address.offset()))) return false;
  return underConstruction(*current);
}

auto ASTInterpreter::isTransparentlyReplaceable(const ConstAddress& address,
                                                const Type* type) const
    -> bool {
  if (!traits.is_same(traits.remove_cv(address.elementType()),
                      traits.remove_cv(type)))
    return false;
  auto storage = address.storage();
  if (!storage) return isVariableUnderInitialization(address);
  if (!traits.is_const(storage->type())) return true;
  if (isUnderConstruction(address)) return true;
  for (const auto& subobject : address.subobjectPath()) {
    auto symbol = std::get_if<Symbol*>(&subobject);
    auto field = symbol ? symbol_cast<FieldSymbol>(*symbol) : nullptr;
    if (field && field->isMutable()) return true;
  }
  return false;
}

auto ASTInterpreter::evaluateAllocatorCall(FunctionSymbol* function,
                                           const Frame& frame)
    -> std::optional<ConstValue> {
  auto element = allocatorElementType(function);
  if (!element) return std::nullopt;
  auto parameters = function->parameters();
  if (function->name() == control()->getIdentifier("allocate")) {
    auto count = toUInt(frame.locals.at(parameters.front())->value());
    auto size = control()->memoryLayout()->sizeOf(element);
    if (!count || !size || !*size ||
        *count > std::numeric_limits<std::intmax_t>::max() / *size)
      return std::nullopt;
    auto array = control()->getBoundedArrayType(element, *count);
    auto contents = std::make_shared<InitializerList>(
        std::vector<InitializerList::Element>{}, *count,
        InitializerList::Element{OutOfLifetimeValue{}, element});
    auto storage = newAllocation(array, AllocationForm::kAllocator);
    storage->setValue(std::move(contents));
    auto address = std::make_shared<ConstAddress>();
    address->setStorage(std::move(storage));
    address->setArrayType(array);
    address->setElementType(element);
    return address;
  }
  const auto& value = frame.locals.at(parameters.front())->value();
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  auto count = toUInt(frame.locals.at(parameters.back())->value());
  if (!address || !*address || !count || (*address)->lifetimeEnded() ||
      (*address)->offset() || !(*address)->subobjectPath().empty())
    return std::nullopt;
  auto storage = (*address)->storage();
  auto allocation =
      std::ranges::find(allocations_, storage, &Allocation::storage);
  if (allocation == allocations_.end() ||
      allocation->form != AllocationForm::kAllocator)
    return std::nullopt;
  auto array = type_cast<BoundedArrayType>(storage->type());
  if (!array || array->elementType() != element || array->size() != *count)
    return std::nullopt;
  storage->setLifetimeEnded(true);
  return ConstInt{0};
}

auto ASTInterpreter::evaluateAddress(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this, ast};
  return completeEvaluation(addressOfLvalue(ast));
}

auto ASTInterpreter::evaluateStaticDataMember(FieldSymbol* field)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this, field->initializer()};
  return completeEvaluation(staticFieldInitialValue(field));
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
  if (slot) {
    if (asConstexprUnknownObject(*slot)) return std::nullopt;
    return *slot;
  }
  auto binding = activeReference(sym);
  if (!binding || !binding->address) return std::nullopt;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*binding->address);
  if (!address || !*address) return std::nullopt;
  return loadAddress(**address, 0, traits.remove_reference(sym->type()));
}

auto ASTInterpreter::lookupLocalStorage(const Symbol* sym)
    -> std::shared_ptr<ConstStorage> {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return found->second;
  }
  return nullptr;
}

auto ASTInterpreter::activeReference(const Symbol* symbol) const
    -> const ReferenceBinding* {
  if (!traits.is_reference(symbol->type())) return nullptr;
  for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
    auto found = frame->references.find(symbol);
    if (found != frame->references.end()) return &found->second;
  }
  return nullptr;
}

auto ASTInterpreter::lookupLocalSlot(const Symbol* sym) -> ConstValue* {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    if (auto reference = it->references.find(sym);
        reference != it->references.end()) {
      auto pointer = reference->second.address
                         ? std::get_if<std::shared_ptr<ConstAddress>>(
                               &*reference->second.address)
                         : nullptr;
      if (pointer && *pointer && (*pointer)->lifetimeEnded()) return nullptr;
      if (pointer && *pointer) return addressSlot(**pointer, 0, sym->type());
      if (reference->second.slot) return reference->second.slot;
    }
    auto found = it->locals.find(sym);
    if (found == it->locals.end()) continue;
    if (found->second->lifetimeEnded()) return nullptr;
    return &found->second->value();
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
  if (!value) {
    std::erase_if(temporaries_, [&storage](const auto& temporary) {
      return temporary.storage == storage;
    });
    return nullptr;
  }
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
      auto target = typedAddress(**address);
      if (!target || target->isNull() || !validPosition(*target) ||
          isOnePastEnd(*target)) {
        aborted_ = true;
        return false;
      }
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
      frame.references.insert_or_assign(
          reference, ReferenceBinding{
                         *value, addressSlot(**address, 0, reference->type())});
      return true;
    }
  }

  auto slot = lvalue(initializer);
  if (!slot) return false;
  frame.references.insert_or_assign(reference,
                                    ReferenceBinding{std::nullopt, slot});
  return true;
}

auto ASTInterpreter::hasActiveReference(const Symbol* symbol) const -> bool {
  return activeReference(symbol) != nullptr;
}

void ASTInterpreter::setLocal(const Symbol* sym, ConstValue value) {
  if (frames_.empty()) frames_.push_back({});
  frames_.back().locals.insert_or_assign(
      sym, std::make_shared<ConstStorage>(sym->type(), std::move(value)));
}

auto ASTInterpreter::definingDeclarationOf(FunctionSymbol* function)
    -> FunctionSymbol* {
  if (Reflection{unit_}.builtin(function) != WellKnownName::T_NONE)
    return function;
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
      frames_.back().references.erase(parameters[index]);
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
  ParameterBindingScope binding{*this, frame};
  auto params = definingDeclarationOf(func)->parameters();
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (i < args.size()) {
      if (traits.is_reference(params[i]->type())) {
        auto address = std::get_if<std::shared_ptr<ConstAddress>>(&args[i]);
        if (address && *address) {
          auto target = typedAddress(**address);
          if (!target || target->isNull() || !validPosition(*target) ||
              isOnePastEnd(*target))
            return false;
          frame.references.insert_or_assign(
              params[i],
              ReferenceBinding{args[i],
                               addressSlot(**address, 0, params[i]->type())});
          continue;
        }
      }
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
  if (aborted_) return false;
  auto storage =
      std::make_shared<ConstStorage>(paramSymbol->type(), IndeterminateValue{});
  auto destination = std::make_shared<ConstAddress>(paramSymbol);
  destination->setStorage(storage);
  frame.locals.insert_or_assign(paramSymbol, storage);
  if (auto value =
          evaluateInitializer(paramSymbol->type(), argExpr, destination)) {
    storage->setValue(std::move(*value));
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
  ParameterBindingScope binding{*this, frame};
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
        traits.is_class_or_union(traits.remove_all_extents(field->type()))) {
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
    if (auto anonymous = anonymous_member_class(field); anonymous && !value) {
      for (const auto& candidate : source->members()) {
        if (candidate.symbol &&
            candidate.symbol->parent() == anonymous->resolvedDefinition())
          obj->addMember(candidate.symbol, cloneValue(candidate.value));
      }
      continue;
    }
    if (!value) return false;
    obj->setMember(field, cloneValue(*value));
  }

  return true;
}

void ASTInterpreter::gatherAnonymousMember(
    const std::shared_ptr<ConstObject>& object, const Symbol* symbol) {
  auto anonymous = anonymous_member_class(const_cast<Symbol*>(symbol));
  if (!anonymous || object->subobject(symbol)) return;
  auto gathered = std::make_shared<ConstObject>(anonymous->type());
  auto definition = anonymous->resolvedDefinition();
  auto& members = object->mutableMembers();
  for (auto& member : members) {
    if (member.symbol && member.symbol->parent() == definition)
      gathered->addMember(member.symbol, std::move(member.value));
  }
  std::erase_if(members, [definition](const auto& member) {
    return member.symbol && member.symbol->parent() == definition;
  });
  object->setMember(symbol, std::move(gathered));
}

auto ASTInterpreter::subobjectSlot(const std::shared_ptr<ConstObject>& object,
                                   const Symbol* symbol) -> ConstValue* {
  if (!object || !symbol) return nullptr;
  if (object->isConstexprUnknown()) return nullptr;

  gatherAnonymousMember(object, symbol);
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

  if (classSymbol) {
    auto path = classSymbol->subobjectPath(const_cast<Symbol*>(symbol));
    if (path && path->size() > 1) {
      auto enclosing = object->mutableSubobject(path->front());
      auto nested = enclosing
                        ? std::get_if<std::shared_ptr<ConstObject>>(enclosing)
                        : nullptr;
      if (nested && *nested) return subobjectSlot(*nested, symbol);
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

auto ASTInterpreter::hasEmptyInitializerSyntax(MemInitializerAST* ast) const
    -> bool {
  auto paren = ast_cast<ParenMemInitializerAST>(ast);
  auto braced = ast_cast<BracedMemInitializerAST>(ast);
  return (paren && paren->lparenLoc) || (braced && braced->bracedInitList);
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

  return classInitializationValue(
      type, ast->constructor, arguments,
      arguments.empty() && hasEmptyInitializerSyntax(ast),
      std::move(destination));
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
  if (!ast->constructor && arguments.empty()) {
    if (!hasEmptyInitializerSyntax(ast)) return;
    auto zero = zeroInitialize(type);
    if (!zero) {
      aborted_ = true;
      return;
    }
    thisObject_->setMember(subobject, std::move(*zero));
    return;
  }

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
  auto source = arguments.empty() ? nullptr : arguments.front();
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(source))
    source = cast->expression;
  if (!ast->constructor && source && is_glvalue(source))
    value = cloneValue(*value);
  thisObject_->setMember(subobject, std::move(*value));
}

auto ASTInterpreter::defaultConstruct(const Type* type,
                                      std::shared_ptr<ConstAddress> destination,
                                      FunctionSymbol* constructor)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto elementType = traits.remove_all_extents(type);
  if (auto array = unqualified_cast<BoundedArrayType>(type);
      array && !traits.is_class_or_union(elementType)) {
    auto element = defaultConstruct(array->elementType());
    if (!element) return std::nullopt;
    return ConstValue{std::make_shared<InitializerList>(
        std::vector<InitializerList::Element>{}, array->size(),
        InitializerList::Element{std::move(*element), array->elementType()})};
  }
  if (!traits.is_class_or_union(elementType))
    return ConstValue{IndeterminateValue{}};

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
  if (!traits.is_class_or_union(type) && !traits.is_array(type)) return;
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
  if (!thrown_) return destroyObject(storage);
  auto inFlight = std::exchange(thrown_, std::nullopt);
  aborted_ = false;
  const auto destroyed = destroyObject(storage);
  const auto failed = !destroyed || aborted_;
  aborted_ = true;
  if (failed) {
    thrown_.reset();
    return false;
  }
  thrown_ = std::move(inFlight);
  return true;
}

auto ASTInterpreter::destroyObject(const std::shared_ptr<ConstStorage>& storage)
    -> bool {
  if (storage->lifetimeEnded()) return true;
  if (traits.is_reference(storage->type())) {
    storage->setLifetimeEnded(true);
    return true;
  }
  auto address = std::make_shared<ConstAddress>();
  address->setStorage(storage);
  if (!destroyValue(storage->type(), storage->value(), std::move(address)))
    return false;
  endLifetime(storage->type(), storage->value());
  storage->setLifetimeEnded(true);
  storage->setValue(IndeterminateValue{});
  if (auto array = storage->backingArray()) return destroyObject(array);
  return true;
}

auto ASTInterpreter::destroyValue(const Type* type, ConstValue& value,
                                  std::shared_ptr<ConstAddress> address)
    -> bool {
  auto unqual = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqual)) {
    auto elements = std::get_if<std::shared_ptr<InitializerList>>(&value);
    if (!elements || !*elements) return false;
    if (traits.has_trivial_destructor(
            traits.remove_all_extents(arrayType->elementType())))
      return true;
    if ((*elements)->size() == 0) return true;
    std::ignore = (*elements)->mutableElement((*elements)->size() - 1);
    auto& initialized = (*elements)->mutableInitializedElements();
    for (auto index = initialized.size(); index-- > 0;) {
      auto& [element, elementType] = initialized[index];
      auto typeToDestroy = elementType;
      if (!typeToDestroy) typeToDestroy = arrayType->elementType();
      auto elementAddress = arrayElementDestination(address, unqual, index);
      if (!destroyValue(typeToDestroy, element, std::move(elementAddress)))
        return false;
    }
    return true;
  }

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return true;
  if (traits.has_trivial_destructor(unqual)) return true;

  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!object || !*object || (*object)->lifetimeEnded()) return false;
  auto classSymbol = classType->definition();
  traits.require_complete_class(classSymbol);
  if (!classSymbol || !classSymbol->isComplete()) return false;
  auto destructor = classSymbol->destructor();
  if (!destructor || unit_->typeTraits().is_deleted_function(destructor))
    return false;

  objectsUnderDestruction_.push_back(*object);
  struct DestructionEnd {
    std::vector<std::shared_ptr<ConstObject>>& objects;
    ~DestructionEnd() { objects.pop_back(); }
  } destructionEnd{objectsUnderDestruction_};

  if (!destructor->isDefaulted()) {
    if (!destructor->isConstexpr()) return false;
    auto savedReturnValue = std::move(returnValue_);
    auto savedCaptureReturnAddress =
        std::exchange(captureReturnAddress_, false);
    auto savedReturnAddress = std::move(returnAddress_);
    (void)evaluateCall(destructor, {}, *object, address);
    returnValue_ = std::move(savedReturnValue);
    captureReturnAddress_ = savedCaptureReturnAddress;
    returnAddress_ = std::move(savedReturnAddress);
    if (aborted_) return false;
  }

  auto& members = (*object)->mutableMembers();
  for (auto it = members.rbegin(); it != members.rend(); ++it) {
    auto memberType =
        traits.aggregate_element_type(const_cast<Symbol*>(it->symbol));
    if (!memberType || isUnionMember(it->symbol)) continue;
    auto memberAddress =
        subobjectDestination(address, const_cast<Symbol*>(it->symbol));
    if (!destroyValue(memberType, it->value, std::move(memberAddress)))
      return false;
  }

  return true;
}

void ASTInterpreter::destroyConstructedSubobjects(
    const std::shared_ptr<ConstObject>& object) {
  if (!object) return;
  auto& members = object->mutableMembers();
  for (auto it = members.rbegin(); it != members.rend(); ++it) {
    if (std::holds_alternative<IndeterminateValue>(it->value)) continue;
    if (isUnionMember(it->symbol)) continue;
    auto type = traits.aggregate_element_type(const_cast<Symbol*>(it->symbol));
    if (!type) continue;
    auto storage = std::make_shared<ConstStorage>(type, it->value);
    if (!destroyStorage(storage)) return;
  }
}

void ASTInterpreter::throwException(std::shared_ptr<ConstStorage> object,
                                    const Type* type) {
  if (!std::ranges::contains(exceptionObjects_, object))
    exceptionObjects_.push_back(object);
  thrown_ = ThrownException{std::move(object), type};
  aborted_ = true;
}

auto ASTInterpreter::bindHandlerVariable(HandlerAST* handler,
                                         const ThrownException& exception)
    -> bool {
  auto declaration =
      ast_cast<TypeExceptionDeclarationAST>(handler->exceptionDeclaration);
  auto variable = declaration ? declaration->symbol : nullptr;
  if (!variable) return true;
  auto address = std::make_shared<ConstAddress>();
  address->setStorage(exception.object);
  auto arena = unit_->arena();
  auto object = ConstExpressionAST::create(arena);
  object->type = exception.type;
  object->valueCategory = ValueCategory::kLValue;
  object->constValue = arena->make<ConstValue>(std::move(address));
  ExpressionAST* initializer = object;
  TypeChecker check{unit_};
  check.setScope(handler->symbol);
  check.check_variable_initializer(variable, initializer,
                                   handler->firstSourceLocation());
  return initializeAutomaticVariable(variable, initializer);
}

auto ASTInterpreter::catchException(List<HandlerAST*>* handlers,
                                    bool rethrowAtEnd) -> StatementResult {
  for (auto handler : ListView{handlers}) {
    auto declaration =
        ast_cast<TypeExceptionDeclarationAST>(handler->exceptionDeclaration);
    if (declaration &&
        !traits.handler_matches(declaration->symbol->type(), thrown_->type))
      continue;
    auto exception = *std::exchange(thrown_, std::nullopt);
    aborted_ = false;
    handled_.push_back(exception);
    auto mark = beginAutomaticScope();
    StatementResult result;
    if (bindHandlerVariable(handler, exception)) {
      result = statement(handler->statement);
    } else {
      aborted_ = true;
    }
    if (!endAutomaticScope(mark)) aborted_ = true;
    handled_.pop_back();
    if (rethrowAtEnd && !aborted_)
      throwException(exception.object, exception.type);
    const auto rethrown = thrown_ && thrown_->object == exception.object;
    if (!rethrown) {
      if (!destroyStorage(exception.object)) aborted_ = true;
      std::erase(exceptionObjects_, exception.object);
    }
    return result;
  }
  return {};
}

auto ASTInterpreter::executeFunction(
    FunctionSymbol* function, Frame frame, CallResultKind kind,
    std::shared_ptr<ConstObject> object, bool constructor,
    std::shared_ptr<ConstAddress> objectAddress,
    std::shared_ptr<ConstAddress> resultAddress) -> CallResult {
  if (!function || !function->isConstexpr() || depth_ >= kMaxDepth) return {};
  if (object && object->lifetimeEnded()) return {};
  if (allocatorElementType(function))
    return {evaluateAllocatorCall(function, frame), nullptr};
  Reflection reflection{unit_};
  auto builtin = reflection.builtin(function);
  if (builtin != WellKnownName::T_NONE) {
    auto parameters = function->parameters();
    std::vector<ConstValue> arguments;
    arguments.reserve(parameters.size());
    for (auto parameter : parameters) {
      if (traits.is_reference(parameter->type())) {
        auto binding = frame.references.find(parameter);
        if (binding == frame.references.end() || !binding->second.address)
          return {};
        arguments.push_back(*binding->second.address);
        continue;
      }
      auto argument = frame.locals.find(parameter);
      if (argument == frame.locals.end()) return {};
      arguments.push_back(argument->second->value());
    }
    auto value = reflection.call(builtin, arguments, evaluationScope_, this,
                                 object, function);
    if (!value) {
      if (!aborted_) reflection.throwFailure(function, evaluationScope_, this);
      return {};
    }
    auto type = type_cast<FunctionType>(function->type())->returnType();
    if (traits.is_reference(type))
      return builtinReferenceResult(kind, *value, type);
    auto elements = std::get_if<std::shared_ptr<InitializerList>>(&*value);
    if (!traits.is_class_or_union(type) || !elements || !*elements)
      return {std::move(value), nullptr};
    auto arena = unit_->arena();
    auto braced = BracedInitListAST::create(arena);
    auto tail = &braced->expressionList;
    for (const auto& [value, elementType] :
         (*elements)->initializedElements()) {
      auto expression = ConstExpressionAST::create(arena);
      expression->type = elementType;
      expression->valueCategory = ValueCategory::kPrValue;
      expression->constValue = arena->make<ConstValue>(value);
      *tail = make_list_node(arena, static_cast<ExpressionAST*>(expression));
      tail = &(*tail)->next;
    }
    ExpressionAST* initializer = braced;
    TypeChecker check{unit_};
    check.setScope(function->parent());
    auto selected = check.check_class_initializer(type, initializer, {});
    return {initializationValue(type, selected, initializer,
                                std::move(resultAddress)),
            nullptr};
  }
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
  auto savedNominates = std::exchange(nominatesUnionMembers_, false);
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
  if (thrown_ && constructor) {
    destroyConstructedSubobjects(thisObject_);
    if (thisObject_) thisObject_->setLifetimeEnded(true);
  }
  if (thrown_) {
    (void)catchException(body_handlers(declaration->functionBody),
                         constructor || function->isDestructor());
  }
  if (thrown_ && functionType && functionType->isNoexcept()) thrown_.reset();
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
  nominatesUnionMembers_ = savedNominates;
  resultAddress_ = std::move(savedResultAddress);
  thisScope.reset();
  currentConstructorClass_ = savedConstructor;
  defaultInitializerContext_ = savedContext;
  if (aborted_) return {};
  return result;
}

auto ASTInterpreter::builtinReferenceResult(CallResultKind kind,
                                            const ConstValue& value,
                                            const Type* type) -> CallResult {
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  if (!address || !*address) return {};
  if (kind == CallResultKind::kValue)
    return {loadAddress(**address, 0, type), nullptr};
  if (kind == CallResultKind::kLValue)
    return {std::nullopt, addressSlot(**address, 0)};
  return {value, nullptr, kind == CallResultKind::kObject};
}

auto ASTInterpreter::evaluateCall(FunctionSymbol* func,
                                  std::vector<ConstValue> args,
                                  std::shared_ptr<ConstObject> thisObject,
                                  std::shared_ptr<ConstAddress> thisAddress)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  Frame frame;
  if (!func || !func->isConstexpr() || !bindParameters(frame, func, args))
    return std::nullopt;
  return completeEvaluation(
      executeFunction(func, std::move(frame), CallResultKind::kValue,
                      std::move(thisObject), false, std::move(thisAddress))
          .value);
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
