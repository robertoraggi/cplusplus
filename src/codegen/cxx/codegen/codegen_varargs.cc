#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/types.h>

#include <algorithm>

namespace cxx {

struct Codegen::VaArgBuilder {
  Codegen& gen;
  SourceLocation loc;
  ir::ValueRef list;
  const Type* type;
  ir::ValueRef result;
  bool isAggregate = false;
  bool isIndirect = false;
  bool usesVectorRegisters = false;
  const Type* homogeneousElement = nullptr;
  std::uint64_t elementCount = 1;

  [[nodiscard]] auto constant(std::int64_t value, unsigned bits = 64)
      -> ir::ValueRef {
    return gen.emitter_.constantInt(loc, gen.emitter_.integerType(bits), value);
  }

  [[nodiscard]] auto byteOffset(ir::ValueRef pointer, ir::ValueRef offset)
      -> ir::ValueRef {
    auto bytePointer = gen.emitter_.pointerType(gen.emitter_.integerType(8));
    pointer = gen.emitter_.bitcast(loc, bytePointer, pointer);
    return gen.emitter_.pointerAdd(loc, bytePointer, pointer, offset);
  }

  [[nodiscard]] auto align(ir::ValueRef value, std::uint64_t alignment)
      -> ir::ValueRef {
    auto added = gen.emitter_.binaryOp(
        loc, ir::BinaryOp::AddInt, value,
        gen.emitter_.constantInt(loc, gen.emitter_.typeOf(value),
                                 alignment - 1));
    return gen.emitter_.binaryOp(
        loc, ir::BinaryOp::AndInt, added,
        gen.emitter_.constantInt(loc, gen.emitter_.typeOf(value), -alignment));
  }

  void read(ir::ValueRef address) {
    auto valueType = gen.convertType(type);
    if (isIndirect)
      address =
          gen.emitter_.load(loc, gen.emitter_.pointerType(valueType), address,
                            gen.control()->memoryLayout()->sizeOfPointer());
    if (isAggregate) {
      gen.emitter_.memcpy(loc, result, address, gen.getSize(type));
      return;
    }
    auto value =
        gen.emitter_.load(loc, valueType, address, gen.getAlignment(type));
    gen.emitter_.store(loc, value, result, gen.getAlignment(type));
  }

  void readRegisters(ir::ValueRef address) {
    if (!homogeneousElement) {
      read(address);
      return;
    }
    auto elementType = gen.convertType(homogeneousElement);
    for (std::uint64_t index = 0; index < elementCount; ++index) {
      auto source = byteOffset(address, constant(index * 16));
      auto destination =
          byteOffset(result, constant(index * gen.getSize(homogeneousElement)));
      auto value = gen.emitter_.load(loc, elementType, source,
                                     gen.getAlignment(homogeneousElement));
      gen.emitter_.store(loc, value, destination,
                         gen.getAlignment(homogeneousElement));
    }
  }

  [[nodiscard]] auto emitPointer() -> ExpressionResult {
    isAggregate = gen.traits.is_class_or_union(type);
    auto abi = gen.classifyClassValueAbi(type, ClassValueAbiContext::Argument);
    isIndirect = abi.kind == ClassValueAbi::Kind::Indirect;
    result = gen.newTemp(type, loc);
    auto pointerType = gen.emitter_.pointerType(gen.emitter_.integerType(8));
    auto pointerSize = gen.control()->memoryLayout()->sizeOfPointer();
    auto stack = gen.emitter_.load(loc, pointerType, list, pointerSize);
    auto alignment = isIndirect ? pointerSize : gen.getAlignment(type);
    if (alignment > pointerSize) {
      auto integer = gen.emitter_.pointerToInt(
          loc, gen.convertType(gen.control()->getSizeType()), stack);
      auto aligned = align(integer, alignment);
      stack = gen.emitter_.intToPointer(loc, pointerType, aligned);
    }
    auto size = isIndirect ? pointerSize : gen.getSize(type);
    size = (size + pointerSize - 1) / pointerSize * pointerSize;
    auto offset = gen.emitter_.constantInt(
        loc, gen.convertType(gen.control()->getSizeType()), size);
    gen.emitter_.store(loc, byteOffset(stack, offset), list, pointerSize);
    read(stack);
    if (isAggregate) return {result};
    return {gen.emitter_.load(loc, gen.convertType(type), result,
                              gen.getAlignment(type))};
  }

  [[nodiscard]] auto emitAArch64() -> ExpressionResult {
    isAggregate = gen.traits.is_class_or_union(type);
    auto abi = gen.classifyClassValueAbi(type, ClassValueAbiContext::Argument);
    auto usage = classifyArgumentRegisterUsage(gen.unit_, type, abi);
    usesVectorRegisters = usage.vector != 0;
    if (usesClassValueAbi(type)) {
      isIndirect = abi.kind == ClassValueAbi::Kind::Indirect;
      if (abi.kind == ClassValueAbi::Kind::Coerce && abi.slots.size() == 1) {
        auto slotType = abi.slots.front().type;
        if (auto array = type_cast<BoundedArrayType>(slotType)) {
          homogeneousElement = array->elementType();
          elementCount = array->size();
        } else {
          homogeneousElement = slotType;
        }
        if (!usesVectorRegisters) homogeneousElement = nullptr;
      }
    }
    result = gen.newTemp(type, loc);
    auto intType = gen.emitter_.integerType(32);
    auto pointerType = gen.emitter_.pointerType(gen.emitter_.integerType(8));
    auto offsetAddress =
        gen.memberAddress(loc, list, intType, usesVectorRegisters ? 4 : 3);
    auto topAddress =
        gen.memberAddress(loc, list, pointerType, usesVectorRegisters ? 2 : 1);
    auto offset = gen.emitter_.load(loc, intType, offsetAddress, 4);
    auto registerBlock = gen.newBlock();
    auto readBlock = gen.newBlock();
    auto stackBlock = gen.newBlock();
    auto mergeBlock = gen.newBlock();
    auto available = gen.emitter_.compareInt(loc, ir::IntPredicate::SignedLess,
                                             offset, constant(0, 32));
    gen.emitter_.condBranch(loc, available, registerBlock, stackBlock);
    gen.emitter_.setInsertionBlock(registerBlock);
    if (!usesVectorRegisters && !isIndirect && gen.getAlignment(type) > 8) {
      auto wide =
          gen.emitter_.signExtend(loc, offset, gen.emitter_.integerType(64));
      offset = gen.emitter_.truncate(loc, align(wide, 16), intType);
    }
    std::uint64_t registerBytes = (gen.getSize(type) + 7) / 8 * 8;
    if (isIndirect) registerBytes = 8;
    if (usesVectorRegisters) registerBytes = elementCount * 16;
    auto nextOffset = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, offset,
                                            constant(registerBytes, 32));
    gen.emitter_.store(loc, nextOffset, offsetAddress, 4);
    auto fits = gen.emitter_.compareInt(loc, ir::IntPredicate::SignedLessEqual,
                                        nextOffset, constant(0, 32));
    gen.emitter_.condBranch(loc, fits, readBlock, stackBlock);
    gen.emitter_.setInsertionBlock(readBlock);
    auto top = gen.emitter_.load(loc, pointerType, topAddress, 8);
    auto wideOffset =
        gen.emitter_.signExtend(loc, offset, gen.emitter_.integerType(64));
    readRegisters(byteOffset(top, wideOffset));
    gen.branch(loc, mergeBlock);
    gen.emitter_.setInsertionBlock(stackBlock);
    auto stackAddress = gen.memberAddress(loc, list, pointerType, 0);
    auto stack = gen.emitter_.load(loc, pointerType, stackAddress, 8);
    if (!isIndirect && gen.getAlignment(type) > 8) {
      auto integer =
          gen.emitter_.pointerToInt(loc, gen.emitter_.integerType(64), stack);
      stack = gen.emitter_.intToPointer(loc, pointerType, align(integer, 16));
    }
    std::uint64_t stackBytes = (gen.getSize(type) + 7) / 8 * 8;
    if (isIndirect) stackBytes = 8;
    auto nextStack = byteOffset(stack, constant(stackBytes));
    gen.emitter_.store(loc, nextStack, stackAddress, 8);
    read(stack);
    gen.branch(loc, mergeBlock);
    gen.emitter_.setInsertionBlock(mergeBlock);
    if (isAggregate) return {result};
    return {gen.emitter_.load(loc, gen.convertType(type), result,
                              gen.getAlignment(type))};
  }

  [[nodiscard]] auto emitX86() -> ExpressionResult {
    isAggregate = gen.traits.is_class_or_union(type);
    std::vector<ClassValueAbiSlot> slots;
    auto abi = gen.classifyClassValueAbi(type, ClassValueAbiContext::Argument);
    auto usage = classifyArgumentRegisterUsage(gen.unit_, type, abi);
    const auto stackOnly = abi.kind != ClassValueAbi::Kind::Empty &&
                           usage.general == 0 && usage.vector == 0;
    if (usesClassValueAbi(type)) {
      isIndirect = abi.kind == ClassValueAbi::Kind::Indirect && !stackOnly;
      slots = abi.slots;
      if (isIndirect) slots = {{gen.control()->getPointerType(type), 0}};
    } else {
      slots = {{type, 0}};
    }
    const auto gpCount = usage.general;
    const auto fpCount = usage.vector;
    result = gen.newTemp(type, loc);
    auto intType = gen.emitter_.integerType(32);
    auto pointerType = gen.emitter_.pointerType(gen.emitter_.integerType(8));
    auto gpAddress = gen.memberAddress(loc, list, intType, 0);
    auto fpAddress = gen.memberAddress(loc, list, intType, 1);
    auto gpOffset = gen.emitter_.load(loc, intType, gpAddress, 4);
    auto fpOffset = gen.emitter_.load(loc, intType, fpAddress, 4);
    auto gpFits =
        gen.emitter_.compareInt(loc, ir::IntPredicate::UnsignedLessEqual,
                                gpOffset, constant(48 - gpCount * 8, 32));
    auto fpFits =
        gen.emitter_.compareInt(loc, ir::IntPredicate::UnsignedLessEqual,
                                fpOffset, constant(176 - fpCount * 16, 32));
    auto fits =
        gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt, gpFits, fpFits);
    auto registerBlock = gen.newBlock();
    auto stackBlock = gen.newBlock();
    auto mergeBlock = gen.newBlock();
    if (stackOnly)
      gen.branch(loc, stackBlock);
    else
      gen.emitter_.condBranch(loc, fits, registerBlock, stackBlock);
    gen.emitter_.setInsertionBlock(registerBlock);
    auto saveAddress = gen.memberAddress(loc, list, pointerType, 3);
    auto save = gen.emitter_.load(loc, pointerType, saveAddress, 8);
    std::uint64_t gpIndex = 0;
    std::uint64_t fpIndex = 0;
    for (auto slot : slots) {
      auto floating = gen.traits.is_floating_point(slot.type) ||
                      gen.traits.is_vector(slot.type);
      auto offset = floating ? fpOffset : gpOffset;
      auto adjustment = floating ? fpIndex * 16 : gpIndex * 8;
      offset = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, offset,
                                     constant(adjustment, 32));
      auto wide =
          gen.emitter_.zeroExtend(loc, offset, gen.emitter_.integerType(64));
      auto source = byteOffset(save, wide);
      if (isIndirect) {
        read(source);
      } else {
        auto destination = byteOffset(result, constant(slot.offset));
        gen.emitter_.memcpy(
            loc, destination, source,
            std::min(gen.getSize(slot.type), gen.getSize(type) - slot.offset));
      }
      if (floating)
        ++fpIndex;
      else
        gpIndex += (gen.getSize(slot.type) + 7) / 8;
    }
    if (gpCount) {
      auto next = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, gpOffset,
                                        constant(gpCount * 8, 32));
      gen.emitter_.store(loc, next, gpAddress, 4);
    }
    if (fpCount) {
      auto next = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, fpOffset,
                                        constant(fpCount * 16, 32));
      gen.emitter_.store(loc, next, fpAddress, 4);
    }
    gen.branch(loc, mergeBlock);
    gen.emitter_.setInsertionBlock(stackBlock);
    auto stackAddress = gen.memberAddress(loc, list, pointerType, 2);
    auto stack = gen.emitter_.load(loc, pointerType, stackAddress, 8);
    if (!isIndirect && gen.getAlignment(type) > 8) {
      auto integer =
          gen.emitter_.pointerToInt(loc, gen.emitter_.integerType(64), stack);
      stack = gen.emitter_.intToPointer(loc, pointerType,
                                        align(integer, gen.getAlignment(type)));
    }
    std::uint64_t stackBytes = (gen.getSize(type) + 7) / 8 * 8;
    if (isIndirect) stackBytes = 8;
    gen.emitter_.store(loc, byteOffset(stack, constant(stackBytes)),
                       stackAddress, 8);
    read(stack);
    gen.branch(loc, mergeBlock);
    gen.emitter_.setInsertionBlock(mergeBlock);
    if (isAggregate) return {result};
    return {gen.emitter_.load(loc, gen.convertType(type), result,
                              gen.getAlignment(type))};
  }
};

auto Codegen::emitPointerVaArg(SourceLocation loc, ir::ValueRef list,
                               const Type* type) -> ExpressionResult {
  return VaArgBuilder{*this, loc, list, type}.emitPointer();
}

auto Codegen::emitAArch64VaArg(SourceLocation loc, ir::ValueRef list,
                               const Type* type) -> ExpressionResult {
  return VaArgBuilder{*this, loc, list, type}.emitAArch64();
}

auto Codegen::emitX86VaArg(SourceLocation loc, ir::ValueRef list,
                           const Type* type) -> ExpressionResult {
  auto element = emitter_.elementType(emitter_.typeOf(list));
  if (emitter_.typeKind(element) == ir::TypeKind::Array)
    list = emitter_.bitcast(
        loc, emitter_.pointerType(emitter_.elementType(element)), list);
  return VaArgBuilder{*this, loc, list, type}.emitX86();
}

}  // namespace cxx
