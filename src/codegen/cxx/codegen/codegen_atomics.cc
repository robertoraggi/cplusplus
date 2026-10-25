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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/device_profile_limits.h>
#include <cxx/memory_layout.h>
#include <cxx/memory_order.h>
#include <cxx/names.h>
#include <cxx/types.h>

#include <algorithm>

namespace cxx {

namespace {

[[nodiscard]] auto toIr(MemoryOrder order) -> ir::MemoryOrder {
  switch (order) {
    case MemoryOrder::kRelaxed:
      return ir::MemoryOrder::Relaxed;
    case MemoryOrder::kAcquire:
      return ir::MemoryOrder::Acquire;
    case MemoryOrder::kRelease:
      return ir::MemoryOrder::Release;
    case MemoryOrder::kAcquireRelease:
      return ir::MemoryOrder::AcquireRelease;
    case MemoryOrder::kSequentiallyConsistent:
      return ir::MemoryOrder::SequentiallyConsistent;
  }
  return ir::MemoryOrder::SequentiallyConsistent;
}

[[nodiscard]] auto toIr(MemoryScope scope) -> ir::MemoryScope {
  switch (scope) {
    case MemoryScope::kSystem:
      return ir::MemoryScope::System;
    case MemoryScope::kDevice:
      return ir::MemoryScope::Device;
    case MemoryScope::kWorkgroup:
      return ir::MemoryScope::Workgroup;
    case MemoryScope::kSubgroup:
      return ir::MemoryScope::Subgroup;
    case MemoryScope::kInvocation:
      return ir::MemoryScope::Invocation;
  }
  return ir::MemoryScope::System;
}

[[nodiscard]] auto admitsLoad(ir::MemoryOrder order) -> bool {
  switch (order) {
    case ir::MemoryOrder::Release:
    case ir::MemoryOrder::AcquireRelease:
      return false;
    default:
      return true;
  }
}

[[nodiscard]] auto admitsStore(ir::MemoryOrder order) -> bool {
  switch (order) {
    case ir::MemoryOrder::Acquire:
    case ir::MemoryOrder::AcquireRelease:
      return false;
    default:
      return true;
  }
}

[[nodiscard]] auto constantOperand(ExpressionAST* expression)
    -> std::optional<std::intmax_t> {
  auto constant = ast_cast<ConstExpressionAST>(expression);
  if (!constant || !constant->constValue) return std::nullopt;
  auto integer = std::get_if<ConstInt>(constant->constValue);
  if (!integer) return std::nullopt;
  return integer->toIntMax();
}

[[nodiscard]] auto hasAddressSpace(std::intmax_t spaces, std::uint32_t space)
    -> bool {
  return (spaces >> space) & 1;
}

struct ReadModifyWriteBuiltin {
  BuiltinFunctionKind kind;
  ir::AtomicOperation operation;
  Codegen::AtomicResult result;
  bool isLegacy = false;
  bool isScaled = false;
};

using enum BuiltinFunctionKind;

constexpr auto kOld = Codegen::AtomicResult::kOld;
constexpr auto kUpdated = Codegen::AtomicResult::kUpdated;
constexpr auto kAdd = ir::AtomicOperation::Add;
constexpr auto kSub = ir::AtomicOperation::Sub;
constexpr auto kAnd = ir::AtomicOperation::And;
constexpr auto kOr = ir::AtomicOperation::Or;
constexpr auto kXor = ir::AtomicOperation::Xor;
constexpr auto kNand = ir::AtomicOperation::Nand;

constexpr ReadModifyWriteBuiltin kReadModifyWriteBuiltins[] = {
    {T___ATOMIC_FETCH_ADD, kAdd, kOld},
    {T___ATOMIC_FETCH_SUB, kSub, kOld},
    {T___ATOMIC_FETCH_AND, kAnd, kOld},
    {T___ATOMIC_FETCH_OR, kOr, kOld},
    {T___ATOMIC_FETCH_XOR, kXor, kOld},
    {T___ATOMIC_FETCH_NAND, kNand, kOld},
    {T___ATOMIC_ADD_FETCH, kAdd, kUpdated},
    {T___ATOMIC_SUB_FETCH, kSub, kUpdated},
    {T___ATOMIC_AND_FETCH, kAnd, kUpdated},
    {T___ATOMIC_OR_FETCH, kOr, kUpdated},
    {T___ATOMIC_XOR_FETCH, kXor, kUpdated},
    {T___ATOMIC_NAND_FETCH, kNand, kUpdated},
    {T___C11_ATOMIC_FETCH_ADD, kAdd, kOld, false, true},
    {T___C11_ATOMIC_FETCH_SUB, kSub, kOld, false, true},
    {T___C11_ATOMIC_FETCH_AND, kAnd, kOld},
    {T___C11_ATOMIC_FETCH_OR, kOr, kOld},
    {T___C11_ATOMIC_FETCH_XOR, kXor, kOld},
    {T___C11_ATOMIC_FETCH_NAND, kNand, kOld},
    {T___SYNC_FETCH_AND_ADD, kAdd, kOld, true},
    {T___SYNC_FETCH_AND_SUB, kSub, kOld, true},
    {T___SYNC_FETCH_AND_AND, kAnd, kOld, true},
    {T___SYNC_FETCH_AND_OR, kOr, kOld, true},
    {T___SYNC_FETCH_AND_XOR, kXor, kOld, true},
    {T___SYNC_FETCH_AND_NAND, kNand, kOld, true},
    {T___SYNC_ADD_AND_FETCH, kAdd, kUpdated, true},
    {T___SYNC_SUB_AND_FETCH, kSub, kUpdated, true},
    {T___SYNC_AND_AND_FETCH, kAnd, kUpdated, true},
    {T___SYNC_OR_AND_FETCH, kOr, kUpdated, true},
    {T___SYNC_XOR_AND_FETCH, kXor, kUpdated, true},
    {T___SYNC_NAND_AND_FETCH, kNand, kUpdated, true},
};

[[nodiscard]] auto readModifyWriteBuiltin(BuiltinFunctionKind kind)
    -> const ReadModifyWriteBuiltin* {
  auto found = std::ranges::find(kReadModifyWriteBuiltins, kind,
                                 &ReadModifyWriteBuiltin::kind);
  if (found == std::end(kReadModifyWriteBuiltins)) return nullptr;
  return found;
}

[[nodiscard]] auto pointerSizedOperand(ir::Emitter& emitter, SourceLocation loc,
                                       ir::ValueRef operand, unsigned bits)
    -> ir::ValueRef {
  if (emitter.typeKind(emitter.typeOf(operand)) == ir::TypeKind::Pointer)
    return emitter.pointerToInt(loc, emitter.integerType(bits), operand);
  auto width = emitter.scalarWidth(emitter.typeOf(operand));
  if (width == bits) return operand;
  auto kind = width < bits ? ir::CastKind::SignExtend : ir::CastKind::Truncate;
  return emitter.convert(loc, kind, operand, emitter.integerType(bits));
}

}  // namespace

struct Codegen::AtomicBuiltin {
  Codegen& gen;
  CallExpressionAST* ast;
  BuiltinFunctionKind kind;
  SourceLocation loc;
  std::vector<ExpressionAST*> operands;
  std::vector<ir::ValueRef> values;

  AtomicBuiltin(Codegen& gen, CallExpressionAST* ast)
      : gen(gen),
        ast(ast),
        kind(resolveBuiltinFunctionKind(
            ast_cast<IdExpressionAST>(ast->baseExpression))),
        loc(ast->lparenLoc) {}

  [[nodiscard]] auto emit() -> ExpressionResult {
    evaluateOperands();
    return emitOperation();
  }

 private:
  void evaluateOperands() {
    for (auto operand : ListView{ast->expressionList}) {
      operands.push_back(operand);
      auto result = gen.expression(operand);
      values.push_back(gen.builtinOperand(operand, result.value));
    }
  }

  [[nodiscard]] auto pointee() const -> const Type* {
    auto pointer =
        type_cast<PointerType>(gen.traits.decay(operands.front()->type));
    return pointer->elementType();
  }

  [[nodiscard]] auto objectType() const -> const Type* {
    return gen.traits.remove_atomic(unqualified_type(pointee()));
  }

  [[nodiscard]] auto valueType() const -> ir::TypeRef {
    return gen.convertType(objectType());
  }

  [[nodiscard]] auto alignment() const -> std::uint64_t {
    return gen.getAlignment(objectType());
  }

  [[nodiscard]] auto isVolatile() const -> bool {
    return gen.traits.is_volatile(pointee());
  }

  [[nodiscard]] auto address() const -> ir::ValueRef { return values.front(); }

  [[nodiscard]] auto zeroValue() -> ir::ValueRef {
    if (gen.traits.is_pointer(objectType()))
      return gen.emitter_.nullPointer(loc, valueType());
    return gen.emitter_.constantZero(loc, valueType());
  }

  [[nodiscard]] auto typedAddress(ir::ValueRef pointer) -> ir::ValueRef {
    auto type = gen.emitter_.rebindPointerType(gen.emitter_.typeOf(pointer),
                                               valueType());
    return gen.emitter_.bitcast(loc, type, pointer);
  }

  [[nodiscard]] auto byteAddress() -> ir::ValueRef {
    auto type = gen.emitter_.rebindPointerType(gen.emitter_.typeOf(address()),
                                               gen.emitter_.integerType(8));
    return gen.emitter_.bitcast(loc, type, address());
  }

  [[nodiscard]] auto order(unsigned index) const -> ir::MemoryOrder {
    auto value = constantOperand(operands[index]);
    auto order = value ? memoryOrderOf(*value) : std::nullopt;
    if (!order) return ir::MemoryOrder::SequentiallyConsistent;
    return toIr(*order);
  }

  [[nodiscard]] auto loadOrder(unsigned index) const -> ir::MemoryOrder {
    auto result = order(index);
    if (admitsLoad(result)) return result;
    return ir::MemoryOrder::SequentiallyConsistent;
  }

  [[nodiscard]] auto storeOrder(unsigned index) const -> ir::MemoryOrder {
    auto result = order(index);
    if (admitsStore(result)) return result;
    return ir::MemoryOrder::SequentiallyConsistent;
  }

  [[nodiscard]] auto scope(unsigned index) const -> ir::MemoryScope {
    auto value = constantOperand(operands[index]);
    auto scope = value ? memoryScopeOf(*value) : std::nullopt;
    if (!scope) return ir::MemoryScope::System;
    return toIr(*scope);
  }

  [[nodiscard]] auto regions(unsigned index) const -> ir::MemoryRegions {
    auto spaces = constantOperand(operands[index]).value_or(0);
    return {.storage = hasAddressSpace(spaces, kDeviceAddressSpace),
            .workgroup = hasAddressSpace(spaces, kThreadgroupAddressSpace)};
  }

  [[nodiscard]] auto access(ir::MemoryOrder order) const -> ir::AtomicAccess {
    return {.order = order, .isVolatile = isVolatile()};
  }

  [[nodiscard]] auto isWeak() const -> bool {
    switch (kind) {
      case T___C11_ATOMIC_COMPARE_EXCHANGE_WEAK:
        return true;
      case T___ATOMIC_COMPARE_EXCHANGE_N:
      case T___ATOMIC_COMPARE_EXCHANGE:
        return constantOperand(operands[3]).value_or(0) != 0;
      default:
        return false;
    }
  }

  [[nodiscard]] auto compareExchangeAccess(unsigned successIndex) const
      -> ir::CompareExchangeAccess {
    return {.success = order(successIndex),
            .failure = loadOrder(successIndex + 1),
            .isWeak = isWeak(),
            .isVolatile = isVolatile()};
  }

  [[nodiscard]] auto result(ir::ValueRef value) const -> ExpressionResult {
    return {.value = value};
  }

  void fence(MemoryScope fenceScope) {
    auto fenceOrder = order(0);
    if (fenceOrder == ir::MemoryOrder::Relaxed) return;
    gen.emitter_.memoryBarrier(
        loc, {.order = fenceOrder,
              .scope = toIr(fenceScope),
              .regions = {.storage = true, .workgroup = true}});
  }

  [[nodiscard]] auto compareExchange(ir::ValueRef expectedAddress,
                                     ir::ValueRef desired,
                                     const ir::CompareExchangeAccess& exchange)
      -> ExpressionResult {
    auto expected =
        gen.emitter_.load(loc, valueType(), expectedAddress, alignment());
    auto exchanged = gen.emitter_.atomicCompareExchange(
        loc, address(), expected, desired, exchange);
    auto failure = gen.newBlock();
    auto done = gen.newBlock();
    gen.emitter_.condBranch(loc, exchanged[1], done, failure);
    gen.emitter_.setInsertionBlock(failure);
    gen.emitter_.store(loc, exchanged[0], expectedAddress, alignment());
    gen.branch(loc, done);
    gen.emitter_.setInsertionBlock(done);
    return result(exchanged[1]);
  }

  [[nodiscard]] auto syncCompareExchange() -> ExpressionResult {
    auto exchanged = gen.emitter_.atomicCompareExchange(
        loc, address(), values[1], values[2], {.isVolatile = isVolatile()});
    if (kind == T___SYNC_BOOL_COMPARE_AND_SWAP) return result(exchanged[1]);
    return result(exchanged[0]);
  }

  [[nodiscard]] auto readModifyWrite(const ReadModifyWriteBuiltin& builtin)
      -> ExpressionResult {
    auto operand = values[1];
    if (builtin.isScaled)
      operand = gen.scaledAtomicOperand(loc, objectType(), operand);
    auto memoryOrder = ir::MemoryOrder::SequentiallyConsistent;
    if (!builtin.isLegacy) memoryOrder = order(2);
    return result(gen.emitAtomicUpdate(loc, builtin.operation, objectType(),
                                       address(), operand, access(memoryOrder),
                                       builtin.result));
  }

  [[nodiscard]] auto exchange(ir::ValueRef value, ir::MemoryOrder memoryOrder)
      -> ir::ValueRef {
    return gen.emitAtomicUpdate(loc, ir::AtomicOperation::Exchange,
                                objectType(), address(), value,
                                access(memoryOrder), kOld);
  }

  [[nodiscard]] auto testAndSet() -> ExpressionResult {
    auto byte = gen.emitter_.integerType(8);
    auto old = gen.emitter_.atomicReadModifyWrite(
        loc, ir::AtomicOperation::Exchange,
        gen.emitter_.constantInt(loc, byte, 1), byteAddress(),
        access(order(1)));
    return result(
        gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual, old,
                                gen.emitter_.constantInt(loc, byte, 0)));
  }

  [[nodiscard]] auto emitOperation() -> ExpressionResult;
};

auto Codegen::AtomicBuiltin::emitOperation() -> ExpressionResult {
  switch (kind) {
    case T___BUILTIN_CXX_CONTROL_BARRIER:
      gen.emitter_.controlBarrier(
          loc, scope(0),
          {.order = order(2), .scope = scope(1), .regions = regions(3)});
      return ExpressionResult{};
    case T___BUILTIN_CXX_MEMORY_BARRIER:
      gen.emitter_.memoryBarrier(
          loc, {.order = order(1), .scope = scope(0), .regions = regions(2)});
      return ExpressionResult{};
    case T___ATOMIC_THREAD_FENCE:
    case T___C11_ATOMIC_THREAD_FENCE:
      fence(MemoryScope::kSystem);
      return ExpressionResult{};
    case T___ATOMIC_SIGNAL_FENCE:
    case T___C11_ATOMIC_SIGNAL_FENCE:
      fence(MemoryScope::kInvocation);
      return ExpressionResult{};
    case T___SYNC_SYNCHRONIZE:
      gen.emitter_.memoryBarrier(
          loc, {.regions = {.storage = true, .workgroup = true}});
      return ExpressionResult{};
    case T___SYNC_LOCK_TEST_AND_SET:
      return result(exchange(values[1], ir::MemoryOrder::Acquire));
    case T___SYNC_LOCK_RELEASE:
      gen.emitter_.atomicStore(loc, zeroValue(), address(),
                               access(ir::MemoryOrder::Release));
      return ExpressionResult{};
    case T___SYNC_VAL_COMPARE_AND_SWAP:
    case T___SYNC_BOOL_COMPARE_AND_SWAP:
      return syncCompareExchange();
    case T___C11_ATOMIC_INIT:
      gen.emitter_.store(loc, values[1], address(),
                         gen.objectAccess(pointee(), alignment()));
      return ExpressionResult{};
    case T___ATOMIC_LOAD_N:
    case T___C11_ATOMIC_LOAD:
      return result(gen.emitter_.atomicLoad(loc, valueType(), address(),
                                            access(loadOrder(1))));
    case T___ATOMIC_LOAD: {
      auto loaded = gen.emitter_.atomicLoad(loc, valueType(), address(),
                                            access(loadOrder(2)));
      gen.emitter_.store(loc, loaded, typedAddress(values[1]), alignment());
      return ExpressionResult{};
    }
    case T___ATOMIC_STORE_N:
    case T___C11_ATOMIC_STORE:
      gen.emitter_.atomicStore(loc, values[1], address(),
                               access(storeOrder(2)));
      return ExpressionResult{};
    case T___ATOMIC_STORE: {
      auto stored = gen.emitter_.load(loc, valueType(), typedAddress(values[1]),
                                      alignment());
      gen.emitter_.atomicStore(loc, stored, address(), access(storeOrder(2)));
      return ExpressionResult{};
    }
    case T___ATOMIC_EXCHANGE_N:
    case T___C11_ATOMIC_EXCHANGE:
      return result(exchange(values[1], order(2)));
    case T___ATOMIC_EXCHANGE: {
      auto desired = gen.emitter_.load(loc, valueType(),
                                       typedAddress(values[1]), alignment());
      gen.emitter_.store(loc, exchange(desired, order(3)),
                         typedAddress(values[2]), alignment());
      return ExpressionResult{};
    }
    case T___ATOMIC_COMPARE_EXCHANGE_N:
      return compareExchange(values[1], values[2], compareExchangeAccess(4));
    case T___ATOMIC_COMPARE_EXCHANGE:
      return compareExchange(
          typedAddress(values[1]),
          gen.emitter_.load(loc, valueType(), typedAddress(values[2]),
                            alignment()),
          compareExchangeAccess(4));
    case T___C11_ATOMIC_COMPARE_EXCHANGE_STRONG:
    case T___C11_ATOMIC_COMPARE_EXCHANGE_WEAK:
      return compareExchange(values[1], values[2], compareExchangeAccess(3));
    case T___ATOMIC_TEST_AND_SET:
      return testAndSet();
    case T___ATOMIC_CLEAR:
      gen.emitter_.atomicStore(
          loc, gen.emitter_.constantInt(loc, gen.emitter_.integerType(8), 0),
          byteAddress(), access(storeOrder(1)));
      return ExpressionResult{};
    default:
      return readModifyWrite(*readModifyWriteBuiltin(kind));
  }
}

auto Codegen::emitAtomicBuiltin(CallExpressionAST* ast) -> ExpressionResult {
  return AtomicBuiltin{*this, ast}.emit();
}

auto Codegen::emitAtomicUpdate(SourceLocation loc,
                               ir::AtomicOperation operation,
                               const Type* objectType, ir::ValueRef address,
                               ir::ValueRef operand,
                               const ir::AtomicAccess& access,
                               AtomicResult result) -> ir::ValueRef {
  auto pointer = type_cast<PointerType>(objectType);
  if (!pointer || operation == ir::AtomicOperation::Exchange) {
    auto old = emitter_.atomicReadModifyWrite(loc, operation, operand, address,
                                              access);
    if (result == AtomicResult::kOld) return old;
    return atomicUpdatedValue(loc, operation, objectType, old, operand);
  }
  auto bits =
      static_cast<unsigned>(control()->memoryLayout()->sizeOfPointer() * 8);
  auto integer = emitter_.integerType(bits);
  operand = pointerSizedOperand(emitter_, loc, operand, bits);
  auto integerAddress = emitter_.bitcast(
      loc, emitter_.rebindPointerType(emitter_.typeOf(address), integer),
      address);
  auto value = emitter_.atomicReadModifyWrite(loc, operation, operand,
                                              integerAddress, access);
  if (result == AtomicResult::kUpdated)
    value = atomicUpdatedValue(loc, operation, control()->getSizeType(), value,
                               operand);
  return emitter_.intToPointer(loc, convertType(objectType), value);
}

auto Codegen::atomicUpdatedValue(SourceLocation loc,
                                 ir::AtomicOperation operation,
                                 const Type* objectType, ir::ValueRef old,
                                 ir::ValueRef operand) -> ir::ValueRef {
  auto isFloat = traits.is_floating_point(objectType);
  switch (operation) {
    case ir::AtomicOperation::Add:
      return emitter_.binaryOp(
          loc, isFloat ? ir::BinaryOp::AddFloat : ir::BinaryOp::AddInt, old,
          operand);
    case ir::AtomicOperation::Sub:
      return emitter_.binaryOp(
          loc, isFloat ? ir::BinaryOp::SubFloat : ir::BinaryOp::SubInt, old,
          operand);
    case ir::AtomicOperation::And:
      return emitter_.binaryOp(loc, ir::BinaryOp::AndInt, old, operand);
    case ir::AtomicOperation::Or:
      return emitter_.binaryOp(loc, ir::BinaryOp::OrInt, old, operand);
    case ir::AtomicOperation::Xor:
      return emitter_.binaryOp(loc, ir::BinaryOp::XorInt, old, operand);
    case ir::AtomicOperation::Nand: {
      auto both = emitter_.binaryOp(loc, ir::BinaryOp::AndInt, old, operand);
      auto ones = emitter_.constantInt(loc, emitter_.typeOf(both), -1);
      return emitter_.binaryOp(loc, ir::BinaryOp::XorInt, both, ones);
    }
    case ir::AtomicOperation::Exchange:
      return operand;
  }
  return operand;
}

auto Codegen::scaledAtomicOperand(SourceLocation loc, const Type* objectType,
                                  ir::ValueRef operand) -> ir::ValueRef {
  auto pointer = type_cast<PointerType>(objectType);
  if (!pointer) return operand;
  auto pointee = traits.remove_cv(pointer->elementType());
  if (traits.is_function(pointee) || traits.is_void(pointee)) return operand;
  auto size = control()->memoryLayout()->sizeOf(pointee);
  if (!size || *size == 1) return operand;
  auto stride = emitter_.constantInt(loc, emitter_.typeOf(operand),
                                     static_cast<std::int64_t>(*size));
  return emitter_.binaryOp(loc, ir::BinaryOp::MulInt, operand, stride);
}

}  // namespace cxx
