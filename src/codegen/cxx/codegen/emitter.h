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

#pragma once

#include <cxx/codegen/debug_emitter.h>
#include <cxx/codegen/emitter_handles.h>
#include <cxx/source_location.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cxx::ir {

enum class FloatKind {
  Half,
  Single,
  Double,
  X87DoubleExtended,
  Quad,
};

enum class TypeKind {
  Void,
  Integer,
  Floating,
  Pointer,
  Array,
  Class,
  Function,
  Unresolved,
  Other,
};

enum class IntPredicate {
  Equal,
  NotEqual,
  SignedLess,
  SignedLessEqual,
  SignedGreater,
  SignedGreaterEqual,
  UnsignedLess,
  UnsignedLessEqual,
  UnsignedGreater,
  UnsignedGreaterEqual,
};

enum class FloatPredicate {
  OrderedEqual,
  OrderedNotEqual,
  OrderedLess,
  OrderedLessEqual,
  OrderedGreater,
  OrderedGreaterEqual,
  UnorderedNotEqual,
  Unordered,
};

struct InsertionPoint {
  enum class Kind {
    BlockStart,
    BlockEnd,
    ModuleStart,
    ModuleEnd,
  };

  Kind kind = Kind::BlockEnd;
  BlockRef block;
};

enum class TodoKind {
  Expression,
  Statement,
};

enum class CastKind {
  Truncate,
  SignExtend,
  ZeroExtend,
  FloatExtend,
  FloatTruncate,
  SignedIntToFloat,
  UnsignedIntToFloat,
  FloatToSignedInt,
  FloatToUnsignedInt,
  ReinterpretBits,
  Bitcast,
  Reshape,
  ArrayToPointer,
  PointerToInt,
  IntToPointer,
};

enum class BinaryOp {
  AddInt,
  SubInt,
  MulInt,
  SignedDiv,
  UnsignedDiv,
  SignedRem,
  UnsignedRem,
  AndInt,
  OrInt,
  XorInt,
  ShiftLeft,
  ArithmeticShiftRight,
  LogicalShiftRight,
  AddFloat,
  SubFloat,
  MulFloat,
  DivFloat,
};

enum class UnaryOp {
  NegateFloat,
};

enum class Linkage {
  External,
  Internal,
  LinkOnceODR,
  WeakODR,
  AvailableExternally,
  Appending,
};

enum class Visibility {
  Default,
  Hidden,
  Protected,
};

enum class InlineKind {
  NoInline,
  InlineHint,
};

[[nodiscard]] inline auto singleValue(std::span<const ValueRef> values)
    -> ValueRef {
  return values.empty() ? ValueRef{} : values.front();
}

struct ModuleInfo {
  std::string_view name;
  std::string_view sourceFile;
  std::string_view targetTriple;
  std::string_view debugCompilationDirectory;
};

struct VTableTableInfo {
  std::span<const std::int64_t> virtualBaseOffsets;
  std::span<const std::int64_t> virtualCallOffsets;
  std::int64_t offsetToTop = 0;
  std::span<const FunctionRef> slots;
};

struct VTableInfo {
  std::string_view name;
  std::string_view typeInfo;
  std::span<const VTableTableInfo> tables;
  Linkage linkage = Linkage::LinkOnceODR;
};

enum class ParameterAbiKind { Default, StructReturn, ByValue };

struct ParameterAbi {
  ParameterAbiKind kind = ParameterAbiKind::Default;
  TypeRef indirectType;
  std::uint64_t alignment = 0;
};

struct FunctionInfo {
  std::string_view name;
  TypeRef type;
  Linkage linkage = Linkage::External;
  Visibility visibility = Visibility::Default;
  InlineKind inlineKind = InlineKind::NoInline;
  std::string_view aliasName;
  std::string_view importModule;
  std::string_view importName;
  std::string_view exportName;
  bool isUsed = false;
  std::span<const ParameterAbi> parameters;
};

struct Initializer {
  enum class Kind {
    None,
    Integer,
    Floating,
    Bytes,
    Aggregate,
    Null,
    Zero,
    ScalarZero,
    Undef,
    SignalingNaN,
  };
  Kind kind = Kind::None;
  TypeRef type;
  std::int64_t integer = 0;
  double floating = 0;
  std::string bytes;
  std::vector<Initializer> elements;
  explicit operator bool() const { return kind != Kind::None; }
  static auto integerValue(TypeRef type, std::int64_t value) -> Initializer {
    Initializer i;
    i.kind = Kind::Integer;
    i.type = type;
    i.integer = value;
    return i;
  }
  static auto floatingValue(TypeRef type, double value) -> Initializer {
    Initializer i;
    i.kind = Kind::Floating;
    i.type = type;
    i.floating = value;
    return i;
  }
  static auto byteString(std::string_view value) -> Initializer {
    Initializer i;
    i.kind = Kind::Bytes;
    i.bytes = value;
    return i;
  }
  static auto aggregate(std::vector<Initializer> value) -> Initializer {
    Initializer i;
    i.kind = Kind::Aggregate;
    i.elements = std::move(value);
    return i;
  }
  static auto null() -> Initializer {
    Initializer i;
    i.kind = Kind::Null;
    return i;
  }
  static auto zero() -> Initializer {
    Initializer i;
    i.kind = Kind::Zero;
    return i;
  }
  static auto scalarZero() -> Initializer {
    Initializer i;
    i.kind = Kind::ScalarZero;
    return i;
  }
  static auto undef() -> Initializer {
    Initializer i;
    i.kind = Kind::Undef;
    return i;
  }
  static auto signalingNaN() -> Initializer {
    Initializer i;
    i.kind = Kind::SignalingNaN;
    return i;
  }
};

struct GlobalInfo {
  std::string_view name;
  TypeRef type;
  Linkage linkage = Linkage::External;
  bool isConstant = false;
  std::uint64_t alignment = 0;
  Initializer initializer;
  bool unknownLocation = false;
  bool isUsed = false;
};

struct CleanupAction {
  ValueRef address;
  FunctionRef destructor;
  std::int64_t depth = 0;
  ValueRef activeFlag;
};

struct CleanupTarget {
  BlockRef block;
  std::string_view label;
};

struct BitfieldInfo {
  std::uint32_t bitOffset = 0;
  std::uint32_t bitWidth = 0;
  std::uint64_t alignment = 0;
};

struct Access {
  std::uint64_t alignment = 0;
  std::optional<BitfieldInfo> bitfield;
  bool isSigned = false;
};

enum class CallKind {
  Direct,
  Builtin,
};

struct CallInfo {
  CallKind kind = CallKind::Direct;
  std::string_view callee;
  ValueRef indirectCallee;
  std::span<const ValueRef> arguments;
  std::span<const TypeRef> results;
  std::span<const ParameterAbi> parameters;
  TypeRef variadicCalleeType;
};

class Emitter {
 public:
  // todo: remove
  [[nodiscard]] auto functionType(std::initializer_list<TypeRef> inputs,
                                  std::initializer_list<TypeRef> results,
                                  bool variadic) -> TypeRef {
    return functionType(
        std::span<const TypeRef>(inputs.begin(), inputs.size()),
        std::span<const TypeRef>(results.begin(), results.size()), variadic);
  }

  virtual ~Emitter() = default;

  auto asType(TypeKind kind, TypeRef type) -> TypeRef {
    return type && typeKind(type) == kind ? type : TypeRef{};
  }

  virtual auto debug() -> DebugEmitter* = 0;

  virtual auto saveInsertionPoint() -> InsertionPointRef = 0;
  virtual void restoreInsertionPoint(InsertionPointRef point) = 0;
  virtual void setInsertionPoint(InsertionPoint point) = 0;

  void setModuleInsertionPoint(bool atStart) {
    setInsertionPoint({.kind = atStart ? InsertionPoint::Kind::ModuleStart
                                       : InsertionPoint::Kind::ModuleEnd});
  }

  void setInsertionBlock(BlockRef block) {
    setInsertionPoint({.kind = InsertionPoint::Kind::BlockEnd, .block = block});
  }

  void setInsertionBlockStart(BlockRef block) {
    setInsertionPoint(
        {.kind = InsertionPoint::Kind::BlockStart, .block = block});
  }
  virtual void beginGlobalInitializer(GlobalRef global) = 0;
  virtual void globalConstructor(SourceLocation loc, FunctionRef function) = 0;
  virtual auto constant(SourceLocation loc, TypeRef type,
                        const Initializer& value) -> ValueRef = 0;

  auto constantLiteral(SourceLocation loc, TypeRef type,
                       const Initializer& value) -> ValueRef {
    return constant(loc, type, value);
  }

  [[nodiscard]] auto constantInt(SourceLocation loc, TypeRef type,
                                 std::int64_t value) -> ValueRef {
    return constant(loc, type, Initializer::integerValue(type, value));
  }

  [[nodiscard]] auto constantZero(SourceLocation loc, TypeRef type)
      -> ValueRef {
    return constant(loc, type, Initializer::scalarZero());
  }

  [[nodiscard]] auto zero(SourceLocation loc, TypeRef resultType) -> ValueRef {
    return constant(loc, resultType, Initializer::zero());
  }

  [[nodiscard]] auto undef(SourceLocation loc, TypeRef resultType) -> ValueRef {
    return constant(loc, resultType, Initializer::undef());
  }

  [[nodiscard]] auto nullPointer(SourceLocation loc, TypeRef pointerType)
      -> ValueRef {
    return constant(loc, pointerType, Initializer::null());
  }

  [[nodiscard]] auto signalingNaN(SourceLocation loc, TypeRef type)
      -> ValueRef {
    return constant(loc, type, Initializer::signalingNaN());
  }
  virtual auto isZeroConstant(ValueRef value) -> bool = 0;
  virtual auto symbolExists(std::string_view name) -> bool = 0;

  virtual auto todo(SourceLocation loc, TodoKind kind, std::string_view message)
      -> ValueRef = 0;

  auto todoExpression(SourceLocation loc, std::string_view message)
      -> ValueRef {
    return todo(loc, TodoKind::Expression, message);
  }

  auto todoStatement(SourceLocation loc, std::string_view message) -> ValueRef {
    return todo(loc, TodoKind::Statement, message);
  }

  auto negateFloat(SourceLocation loc, TypeRef type, ValueRef value)
      -> ValueRef {
    return unaryOp(loc, UnaryOp::NegateFloat, type, value);
  }

  virtual auto unaryOp(SourceLocation loc, UnaryOp op, TypeRef type,
                       ValueRef value) -> ValueRef = 0;

  auto builtinCall(SourceLocation loc, std::span<const TypeRef> results,
                   std::string_view name, std::span<const ValueRef> args)
      -> ValueRef {
    return singleValue(call(loc, {.kind = CallKind::Builtin,
                                  .callee = name,
                                  .arguments = args,
                                  .results = results}));
  }
  virtual void defineVTable(SourceLocation loc, const VTableInfo& info) = 0;

  [[nodiscard]] virtual auto beginModule(const ModuleInfo& info)
      -> ModuleRef = 0;

  virtual void endModule() = 0;

  virtual void beginFunctionBody(FunctionRef function) = 0;

  virtual void endFunctionBody(FunctionRef function) = 0;

  [[nodiscard]] virtual auto beginCleanupRegion() -> CleanupRegionRef = 0;

  virtual void endCleanupRegion(CleanupRegionRef region) = 0;

  [[nodiscard]] virtual auto activateConditionalCleanup(ValueRef address,
                                                        BlockRef entry,
                                                        CleanupRegionRef region)
      -> ValueRef = 0;

  virtual void switchBranch(SourceLocation loc, ValueRef flag,
                            BlockRef defaultDest,
                            std::span<const std::int64_t> caseValues,
                            std::span<const BlockRef> caseDestinations) = 0;

  virtual void defineLabel(SourceLocation loc, std::string_view name,
                           std::int64_t cleanupDepth) = 0;

  virtual void branchWithCleanups(SourceLocation loc, CleanupTarget target,
                                  std::span<const CleanupAction> cleanups) = 0;

  void gotoLabel(SourceLocation loc, std::string_view name,
                 std::span<const CleanupAction> cleanups) {
    branchWithCleanups(loc, {.label = name}, cleanups);
  }

  void branchWithCleanups(SourceLocation loc, BlockRef target,
                          std::span<const CleanupAction> cleanups) {
    branchWithCleanups(loc, CleanupTarget{.block = target}, cleanups);
  }

  virtual void indirectGoto(SourceLocation loc, ValueRef target) = 0;

  virtual auto labelAddress(SourceLocation loc, TypeRef type,
                            std::string_view name, FunctionRef function)
      -> ValueRef = 0;

  virtual void resolveFunctionControlFlow(FunctionRef function) = 0;

  [[nodiscard]] virtual auto functionParameterTypes(FunctionRef function)
      -> std::vector<TypeRef> = 0;

  [[nodiscard]] virtual auto functionResultTypes(FunctionRef function)
      -> std::vector<TypeRef> = 0;

  virtual auto addBlockParameter(BlockRef block, TypeRef type,
                                 SourceLocation loc) -> ValueRef = 0;
  [[nodiscard]] virtual auto blockParameter(BlockRef block, unsigned index)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto blockParameterCount(BlockRef block)
      -> unsigned = 0;

  // Creates a block, appended to `function`'s body when given and otherwise to
  // the region holding the current insertion point. Never moves the insertion
  // point: call setInsertionPoint() to emit into the new block.
  [[nodiscard]] virtual auto createBlock(FunctionRef function) -> BlockRef = 0;

  [[nodiscard]] virtual auto insertionBlock() -> BlockRef = 0;

  virtual void eraseBlock(BlockRef block) = 0;

  [[nodiscard]] virtual auto hasTerminator(BlockRef block) -> bool = 0;

  virtual void branch(SourceLocation loc, BlockRef target,
                      std::span<const ValueRef> operands = {}) = 0;
  virtual auto globalLinkage(GlobalRef global) -> Linkage = 0;

  virtual void condBranch(SourceLocation loc, ValueRef condition,
                          BlockRef trueDest, BlockRef falseDest) = 0;

  [[nodiscard]] auto addInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::AddInt, lhs, rhs);
  }

  [[nodiscard]] auto subInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::SubInt, lhs, rhs);
  }

  [[nodiscard]] auto mulInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::MulInt, lhs, rhs);
  }

  [[nodiscard]] auto signedDiv(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::SignedDiv, lhs, rhs);
  }

  [[nodiscard]] auto unsignedDiv(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::UnsignedDiv, lhs, rhs);
  }

  [[nodiscard]] auto signedRem(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::SignedRem, lhs, rhs);
  }

  [[nodiscard]] auto unsignedRem(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::UnsignedRem, lhs, rhs);
  }

  [[nodiscard]] auto addFloat(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::AddFloat, lhs, rhs);
  }

  [[nodiscard]] auto subFloat(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::SubFloat, lhs, rhs);
  }

  [[nodiscard]] auto mulFloat(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::MulFloat, lhs, rhs);
  }

  [[nodiscard]] auto divFloat(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::DivFloat, lhs, rhs);
  }

  [[nodiscard]] auto andInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::AndInt, lhs, rhs);
  }

  [[nodiscard]] auto orInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::OrInt, lhs, rhs);
  }

  [[nodiscard]] auto xorInt(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::XorInt, lhs, rhs);
  }

  [[nodiscard]] auto shiftLeft(SourceLocation loc, ValueRef lhs, ValueRef rhs)
      -> ValueRef {
    return binaryOp(loc, BinaryOp::ShiftLeft, lhs, rhs);
  }

  [[nodiscard]] auto arithmeticShiftRight(SourceLocation loc, ValueRef lhs,
                                          ValueRef rhs) -> ValueRef {
    return binaryOp(loc, BinaryOp::ArithmeticShiftRight, lhs, rhs);
  }

  [[nodiscard]] auto logicalShiftRight(SourceLocation loc, ValueRef lhs,
                                       ValueRef rhs) -> ValueRef {
    return binaryOp(loc, BinaryOp::LogicalShiftRight, lhs, rhs);
  }

  [[nodiscard]] virtual auto binaryOp(SourceLocation loc, BinaryOp op,
                                      ValueRef lhs, ValueRef rhs)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto compareInt(SourceLocation loc,
                                        IntPredicate predicate, ValueRef lhs,
                                        ValueRef rhs) -> ValueRef = 0;

  [[nodiscard]] virtual auto compareFloat(SourceLocation loc,
                                          FloatPredicate predicate,
                                          ValueRef lhs, ValueRef rhs)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto select(SourceLocation loc, ValueRef condition,
                                    ValueRef ifTrue, ValueRef ifFalse)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto voidType() -> TypeRef = 0;

  [[nodiscard]] virtual auto unresolvedType() -> TypeRef = 0;

  [[nodiscard]] virtual auto integerType(unsigned bits) -> TypeRef = 0;

  [[nodiscard]] virtual auto floatingType(FloatKind kind) -> TypeRef = 0;

  [[nodiscard]] virtual auto pointerType(TypeRef elementType) -> TypeRef = 0;

  [[nodiscard]] virtual auto arrayType(TypeRef elementType, std::uint64_t size)
      -> TypeRef = 0;

  [[nodiscard]] virtual auto vectorType(TypeRef elementType,
                                        std::uint64_t elementCount)
      -> TypeRef = 0;

  [[nodiscard]] virtual auto vectorSplat(SourceLocation loc, TypeRef vectorType,
                                         ValueRef scalar) -> ValueRef = 0;

  [[nodiscard]] virtual auto functionType(std::span<const TypeRef> parameters,
                                          std::span<const TypeRef> results,
                                          bool isVariadic) -> TypeRef = 0;

  [[nodiscard]] virtual auto declareClassType(std::string_view name)
      -> TypeRef = 0;

  virtual void defineClassType(TypeRef classType,
                               std::span<const TypeRef> members,
                               bool isPacked) = 0;

  [[nodiscard]] auto truncate(SourceLocation loc, ValueRef value, TypeRef type)
      -> ValueRef {
    return convert(loc, CastKind::Truncate, value, type);
  }

  [[nodiscard]] auto signExtend(SourceLocation loc, ValueRef value,
                                TypeRef type) -> ValueRef {
    return convert(loc, CastKind::SignExtend, value, type);
  }

  [[nodiscard]] auto zeroExtend(SourceLocation loc, ValueRef value,
                                TypeRef type) -> ValueRef {
    return convert(loc, CastKind::ZeroExtend, value, type);
  }

  [[nodiscard]] auto floatExtend(SourceLocation loc, ValueRef value,
                                 TypeRef type) -> ValueRef {
    return convert(loc, CastKind::FloatExtend, value, type);
  }

  [[nodiscard]] auto floatTruncate(SourceLocation loc, ValueRef value,
                                   TypeRef type) -> ValueRef {
    return convert(loc, CastKind::FloatTruncate, value, type);
  }

  [[nodiscard]] auto signedIntToFloat(SourceLocation loc, ValueRef value,
                                      TypeRef type) -> ValueRef {
    return convert(loc, CastKind::SignedIntToFloat, value, type);
  }

  [[nodiscard]] auto unsignedIntToFloat(SourceLocation loc, ValueRef value,
                                        TypeRef type) -> ValueRef {
    return convert(loc, CastKind::UnsignedIntToFloat, value, type);
  }

  [[nodiscard]] auto floatToSignedInt(SourceLocation loc, ValueRef value,
                                      TypeRef type) -> ValueRef {
    return convert(loc, CastKind::FloatToSignedInt, value, type);
  }

  [[nodiscard]] auto floatToUnsignedInt(SourceLocation loc, ValueRef value,
                                        TypeRef type) -> ValueRef {
    return convert(loc, CastKind::FloatToUnsignedInt, value, type);
  }

  [[nodiscard]] auto reinterpretBits(SourceLocation loc, ValueRef value,
                                     TypeRef type) -> ValueRef {
    return convert(loc, CastKind::ReinterpretBits, value, type);
  }

  [[nodiscard]] virtual auto convert(SourceLocation loc, CastKind kind,
                                     ValueRef value, TypeRef type)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto typeKind(TypeRef type) -> TypeKind = 0;

  [[nodiscard]] virtual auto scalarWidth(TypeRef type) -> unsigned = 0;

  [[nodiscard]] virtual auto elementType(TypeRef type) -> TypeRef = 0;

  [[nodiscard]] virtual auto typeOf(ValueRef value) -> TypeRef = 0;

  [[nodiscard]] virtual auto allocate(SourceLocation loc, TypeRef pointerType,
                                      ValueRef size, std::uint64_t alignment)
      -> ValueRef = 0;

  [[nodiscard]] auto allocate(SourceLocation loc, TypeRef pointerType,
                              std::uint64_t alignment) -> ValueRef {
    return allocate(loc, pointerType, ValueRef{}, alignment);
  }

  [[nodiscard]] auto dynamicAllocate(SourceLocation loc, TypeRef pointerType,
                                     ValueRef size, std::uint64_t alignment)
      -> ValueRef {
    return allocate(loc, pointerType, size, alignment);
  }

  [[nodiscard]] virtual auto load(SourceLocation loc, TypeRef valueType,
                                  ValueRef address, const Access& access)
      -> ValueRef = 0;

  [[nodiscard]] auto load(SourceLocation loc, TypeRef valueType,
                          ValueRef address, std::uint64_t alignment)
      -> ValueRef {
    return load(loc, valueType, address, Access{.alignment = alignment});
  }

  [[nodiscard]] auto loadBitfield(SourceLocation loc, TypeRef valueType,
                                  ValueRef address, BitfieldInfo field,
                                  bool isSigned) -> ValueRef {
    return load(loc, valueType, address,
                Access{.alignment = field.alignment,
                       .bitfield = field,
                       .isSigned = isSigned});
  }

  void storeBitfield(SourceLocation loc, ValueRef value, ValueRef address,
                     BitfieldInfo field) {
    store(loc, value, address,
          Access{.alignment = field.alignment, .bitfield = field});
  }

  [[nodiscard]] virtual auto pointerAdd(SourceLocation loc, TypeRef pointerType,
                                        ValueRef base, ValueRef offset)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto pointerDiff(SourceLocation loc, TypeRef resultType,
                                         ValueRef lhs, ValueRef rhs)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto subscript(SourceLocation loc, TypeRef pointerType,
                                       ValueRef base, ValueRef index)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto memberAddress(SourceLocation loc,
                                           TypeRef pointerType, ValueRef base,
                                           std::uint32_t index) -> ValueRef = 0;

  [[nodiscard]] auto bitcast(SourceLocation loc, TypeRef resultType,
                             ValueRef value) -> ValueRef {
    return convert(loc, CastKind::Bitcast, value, resultType);
  }

  [[nodiscard]] auto reshape(SourceLocation loc, TypeRef resultType,
                             ValueRef value) -> ValueRef {
    return convert(loc, CastKind::Reshape, value, resultType);
  }

  [[nodiscard]] auto arrayToPointer(SourceLocation loc, TypeRef pointerType,
                                    ValueRef value) -> ValueRef {
    return convert(loc, CastKind::ArrayToPointer, value, pointerType);
  }

  [[nodiscard]] auto pointerToInt(SourceLocation loc, TypeRef resultType,
                                  ValueRef value) -> ValueRef {
    return convert(loc, CastKind::PointerToInt, value, resultType);
  }

  [[nodiscard]] auto intToPointer(SourceLocation loc, TypeRef pointerType,
                                  ValueRef value) -> ValueRef {
    return convert(loc, CastKind::IntToPointer, value, pointerType);
  }

  [[nodiscard]] virtual auto extractValue(SourceLocation loc,
                                          TypeRef resultType,
                                          ValueRef container,
                                          std::int64_t position)
      -> ValueRef = 0;

  [[nodiscard]] virtual auto insertValue(SourceLocation loc, TypeRef resultType,
                                         ValueRef container, ValueRef value,
                                         std::int64_t position) -> ValueRef = 0;

  [[nodiscard]] virtual auto addressOfSymbol(SourceLocation loc,
                                             TypeRef resultType,
                                             std::string_view symbol)
      -> ValueRef = 0;

  virtual void store(SourceLocation loc, ValueRef value, ValueRef address,
                     const Access& access) = 0;

  void store(SourceLocation loc, ValueRef value, ValueRef address,
             std::uint64_t alignment) {
    store(loc, value, address, Access{.alignment = alignment});
  }

  virtual void memsetZero(SourceLocation loc, ValueRef address,
                          std::uint64_t size) = 0;

  virtual void memcpy(SourceLocation loc, ValueRef destination, ValueRef source,
                      std::uint64_t size) = 0;

  virtual void unreachable(SourceLocation loc) = 0;

  [[nodiscard]] virtual auto call(SourceLocation loc, const CallInfo& info)
      -> std::vector<ValueRef> = 0;

  virtual void ret(SourceLocation loc, std::span<const ValueRef> values) = 0;

  [[nodiscard]] virtual auto findFunction(std::string_view name)
      -> FunctionRef = 0;

  [[nodiscard]] virtual auto declareFunction(SourceLocation loc,
                                             const FunctionInfo& info)
      -> FunctionRef = 0;

  [[nodiscard]] virtual auto functionHasBody(FunctionRef function) -> bool = 0;

  [[nodiscard]] virtual auto findGlobal(std::string_view name) -> GlobalRef = 0;

  [[nodiscard]] virtual auto declareGlobal(SourceLocation loc,
                                           const GlobalInfo& info)
      -> GlobalRef = 0;
};

class FunctionBodyGuard {
 public:
  FunctionBodyGuard(Emitter& emitter, FunctionRef function)
      : emitter_(emitter), function_(function) {
    emitter_.beginFunctionBody(function_);
  }

  FunctionBodyGuard(const FunctionBodyGuard&) = delete;
  auto operator=(const FunctionBodyGuard&) -> FunctionBodyGuard& = delete;

  ~FunctionBodyGuard() { emitter_.endFunctionBody(function_); }

 private:
  Emitter& emitter_;
  FunctionRef function_;
};

class InsertionGuard {
 public:
  explicit InsertionGuard(Emitter& emitter)
      : emitter_(emitter), saved_(emitter.saveInsertionPoint()) {}

  InsertionGuard(const InsertionGuard&) = delete;
  auto operator=(const InsertionGuard&) -> InsertionGuard& = delete;

  ~InsertionGuard() { emitter_.restoreInsertionPoint(saved_); }

 private:
  Emitter& emitter_;
  InsertionPointRef saved_;
};

}  // namespace cxx::ir
