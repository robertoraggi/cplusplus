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

#include <cxx/mlir/bitfield_accesses.h>
#include <cxx/mlir/cxx_dialect.h>
#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>

namespace mlir {
namespace {

constexpr unsigned kTypedWordBits = 32;

class BitfieldUnit {
 public:
  [[nodiscard]] static auto at(Operation* access, Value address,
                               std::int64_t bytes, std::int64_t alignment,
                               bool isVolatile, ::cxx::BitfieldMemory memory)
      -> FailureOr<BitfieldUnit> {
    auto storage = IntegerType::get(access->getContext(), 8 * bytes);
    OpBuilder builder(access);
    if (memory == ::cxx::BitfieldMemory::kUntyped)
      return BitfieldUnit(
          reinterpret(builder, access->getLoc(), address, storage), storage,
          storage, alignment, isVolatile);
    auto element = recordUnit(builder, access->getLoc(), address);
    auto word = IntegerType::get(access->getContext(), kTypedWordBits);
    auto type = cast<cxx::PointerType>(element.getType()).getElementType();
    if (type == storage && bytes * 8 <= kTypedWordBits)
      return BitfieldUnit(element, storage, word, alignment, isVolatile);
    auto bytesUnit = dyn_cast<cxx::ArrayType>(type);
    if (bytesUnit && bytesUnit.getElementType().isInteger(8) &&
        std::int64_t(bytesUnit.getSize()) == bytes)
      return BitfieldUnit(element, type, word, alignment, isVolatile);
    return access->emitError("device bit-field storage unit ")
           << type << " is not an integer of at most 32 bits or an i8 array";
  }

  [[nodiscard]] auto word() const -> IntegerType { return word_; }

  [[nodiscard]] auto load(OpBuilder& builder, Location loc, unsigned offset,
                          unsigned width) const -> FailureOr<Value> {
    if (isa<IntegerType>(element_)) {
      Value unit = cxx::LoadOp::create(builder, loc, element_, address_,
                                       std::uint64_t(alignment_), volatile_);
      return convertScalarToDtype(builder, loc, unit, word_, true);
    }
    auto span = byteSpan(offset, width);
    if (failed(span)) return failure();
    Value result = constant(builder, loc, 0);
    for (unsigned index = span->first; index <= span->last; ++index) {
      Value byte =
          cxx::LoadOp::create(builder, loc, builder.getI8Type(),
                              byteAddress(builder, loc, index), 1, volatile_);
      byte = convertScalarToDtype(builder, loc, byte, word_, true);
      byte = arith::ShLIOp::create(
          builder, loc, byte,
          constant(builder, loc, 8 * (index - span->first)));
      result = arith::OrIOp::create(builder, loc, result, byte);
    }
    return result;
  }

  void store(OpBuilder& builder, Location loc, Value word, unsigned offset,
             unsigned width) const {
    if (isa<IntegerType>(element_)) {
      cxx::StoreOp::create(
          builder, loc,
          convertScalarToDtype(builder, loc, word, element_, true), address_,
          std::uint64_t(alignment_), volatile_);
      return;
    }
    auto span = *byteSpan(offset, width);
    for (unsigned index = span.first; index <= span.last; ++index) {
      Value byte = arith::ShRUIOp::create(
          builder, loc, word, constant(builder, loc, 8 * (index - span.first)));
      byte =
          convertScalarToDtype(builder, loc, byte, builder.getI8Type(), true);
      cxx::StoreOp::create(builder, loc, byte, byteAddress(builder, loc, index),
                           1, volatile_);
    }
  }

  [[nodiscard]] auto wordOffset(unsigned offset) const -> unsigned {
    if (isa<IntegerType>(element_)) return offset;
    return offset % 8;
  }

  [[nodiscard]] auto constant(OpBuilder& builder, Location loc,
                              const llvm::APInt& value) const -> Value {
    return arith::ConstantIntOp::create(builder, loc, word_,
                                        value.zextOrTrunc(word_.getWidth()));
  }

  [[nodiscard]] auto constant(OpBuilder& builder, Location loc,
                              std::uint64_t value) const -> Value {
    return constant(builder, loc, llvm::APInt(64, value));
  }

 private:
  struct ByteSpan {
    unsigned first = 0;
    unsigned last = 0;
  };

  BitfieldUnit(Value address, Type element, IntegerType word,
               std::int64_t alignment, bool isVolatile)
      : address_(address),
        element_(element),
        word_(word),
        alignment_(alignment),
        volatile_(isVolatile) {}

  [[nodiscard]] static auto reinterpret(OpBuilder& builder, Location loc,
                                        Value address, Type element) -> Value {
    auto pointer = cast<cxx::PointerType>(address.getType());
    if (pointer.getElementType() == element) return address;
    auto type = cxx::PointerType::get(builder.getContext(), element,
                                      pointer.getAddressSpace());
    return cxx::BitcastOp::create(builder, loc, type, address);
  }

  [[nodiscard]] static auto recordUnit(OpBuilder& builder, Location loc,
                                       Value address) -> Value {
    auto member = address.getDefiningOp<cxx::MemberOp>();
    if (!member) return address;
    auto base = cast<cxx::PointerType>(member.getBase().getType());
    auto element = cxx::getAggregateElementType(base.getElementType(),
                                                member.getMemberIndex());
    auto type = cxx::PointerType::get(builder.getContext(), element,
                                      base.getAddressSpace());
    if (type == member.getType()) return address;
    return cxx::MemberOp::create(builder, loc, type, member.getBase(),
                                 member.getMemberIndex());
  }

  [[nodiscard]] auto byteSpan(unsigned offset, unsigned width) const
      -> FailureOr<ByteSpan> {
    ByteSpan span{offset / 8, (offset + width - 1) / 8};
    if (8 * (span.last - span.first + 1) > word_.getWidth()) return failure();
    return span;
  }

  [[nodiscard]] auto byteAddress(OpBuilder& builder, Location loc,
                                 unsigned index) const -> Value {
    auto base = cast<cxx::PointerType>(address_.getType());
    auto type = cxx::PointerType::get(builder.getContext(), builder.getI8Type(),
                                      base.getAddressSpace());
    return cxx::SubscriptOp::create(
        builder, loc, type, address_,
        arith::ConstantIntOp::create(builder, loc, index, 32));
  }

  Value address_;
  Type element_;
  IntegerType word_;
  std::int64_t alignment_;
  bool volatile_;
};

[[nodiscard]] auto expand(cxx::BitfieldLoadOp load,
                          ::cxx::BitfieldMemory memory) -> LogicalResult {
  auto unit =
      BitfieldUnit::at(load, load.getAddr(), load.getStorageSize(),
                       load.getAlignment(), load.getVolatile_(), memory);
  if (failed(unit)) return failure();
  OpBuilder builder(load);
  auto loc = load.getLoc();
  auto width = load.getBitWidth();
  auto word = unit->load(builder, loc, load.getBitOffset(), width);
  if (failed(word))
    return load.emitError("device bit-field spans more than 32 bits");
  auto bits = unit->word().getWidth();
  auto offset = unit->wordOffset(load.getBitOffset());
  Value field;
  if (load.getIsSigned()) {
    auto raised = arith::ShLIOp::create(
        builder, loc, *word,
        unit->constant(builder, loc, bits - offset - width));
    field = arith::ShRSIOp::create(builder, loc, raised,
                                   unit->constant(builder, loc, bits - width));
  } else {
    auto lowered = arith::ShRUIOp::create(builder, loc, *word,
                                          unit->constant(builder, loc, offset));
    field = arith::AndIOp::create(
        builder, loc, lowered,
        unit->constant(builder, loc, llvm::APInt::getLowBitsSet(bits, width)));
  }
  load.replaceAllUsesWith(convertScalarToDtype(
      builder, loc, field, load.getType(), !load.getIsSigned()));
  load.erase();
  return success();
}

[[nodiscard]] auto expand(cxx::BitfieldStoreOp store,
                          ::cxx::BitfieldMemory memory) -> LogicalResult {
  auto unit =
      BitfieldUnit::at(store, store.getAddr(), store.getStorageSize(),
                       store.getAlignment(), store.getVolatile_(), memory);
  if (failed(unit)) return failure();
  OpBuilder builder(store);
  auto loc = store.getLoc();
  auto width = store.getBitWidth();
  auto word = unit->load(builder, loc, store.getBitOffset(), width);
  if (failed(word))
    return store.emitError("device bit-field spans more than 32 bits");
  auto bits = unit->word().getWidth();
  auto offset = unit->wordOffset(store.getBitOffset());
  auto value =
      convertScalarToDtype(builder, loc, store.getValue(), unit->word(), true);
  auto masked = arith::AndIOp::create(
      builder, loc, value,
      unit->constant(builder, loc, llvm::APInt::getLowBitsSet(bits, width)));
  auto shifted = arith::ShLIOp::create(builder, loc, masked,
                                       unit->constant(builder, loc, offset));
  auto cleared = arith::AndIOp::create(
      builder, loc, *word,
      unit->constant(builder, loc,
                     ~llvm::APInt::getBitsSet(bits, offset, offset + width)));
  unit->store(builder, loc,
              arith::OrIOp::create(builder, loc, cleared, shifted),
              store.getBitOffset(), width);
  store.erase();
  return success();
}

[[nodiscard]] auto expandAccess(Operation* access, ::cxx::BitfieldMemory memory)
    -> LogicalResult {
  if (auto load = dyn_cast<cxx::BitfieldLoadOp>(access))
    return expand(load, memory);
  return expand(cast<cxx::BitfieldStoreOp>(access), memory);
}

}  // namespace
}  // namespace mlir

auto cxx::expandBitfieldAccesses(mlir::Operation* root, BitfieldMemory memory)
    -> mlir::LogicalResult {
  llvm::SmallVector<mlir::Operation*> accesses;
  root->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::cxx::BitfieldLoadOp, mlir::cxx::BitfieldStoreOp>(op))
      accesses.push_back(op);
  });
  for (auto* op : accesses)
    if (mlir::failed(mlir::expandAccess(op, memory))) return mlir::failure();
  return mlir::success();
}
