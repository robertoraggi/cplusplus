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

#include <cxx/cxx_fwd.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/mlir_debug_emitter.h>
#include <cxx/mlir/mlir_emitter.h>
#include <cxx/translation_unit.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>

#include <format>

namespace cxx::ir {
namespace {}  // namespace

MlirEmitter::MlirEmitter(mlir::MLIRContext& context, TranslationUnit* unit)
    : unit_(unit), builder_(&context) {}

MlirEmitter::~MlirEmitter() = default;
auto MlirEmitter::saveInsertionPoint() -> InsertionPointRef {
  std::uint32_t id = 0;

  if (!freeInsertionPoints_.empty()) {
    id = freeInsertionPoints_.back();
    freeInsertionPoints_.pop_back();
  } else {
    id = static_cast<std::uint32_t>(insertionPoints_.size());
    insertionPoints_.emplace_back();
  }

  insertionPoints_[id] = builder_.saveInsertionPoint();
  return HandleAccess::make<InsertionPointTag>(id);
}
void MlirEmitter::restoreInsertionPoint(InsertionPointRef point) {
  auto id = HandleAccess::id(point);
  builder_.restoreInsertionPoint(insertionPoints_.at(id));
  insertionPoints_[id] = {};
  freeInsertionPoints_.push_back(id);
}
void MlirEmitter::setInsertionPoint(InsertionPoint point) {
  switch (point.kind) {
    case InsertionPoint::Kind::BlockStart:
      builder_.setInsertionPointToStart(block(point.block));
      return;

    case InsertionPoint::Kind::BlockEnd:
      builder_.setInsertionPointToEnd(block(point.block));
      return;

    case InsertionPoint::Kind::ModuleStart:
      builder_.setInsertionPointToStart(module_.getBody());
      return;

    case InsertionPoint::Kind::ModuleEnd:
      builder_.setInsertionPointToEnd(module_.getBody());
      return;
  }
}
void MlirEmitter::beginGlobalInitializer(GlobalRef ref) {
  auto b = builder_.createBlock(&global(ref).getInitializer());
  builder_.setInsertionPointToStart(b);
}
void MlirEmitter::globalConstructor(SourceLocation loc, FunctionRef ref) {
  mlir::cxx::GlobalCtorOp::create(builder_, getLocation(loc),
                                  functionName(ref));
}
auto MlirEmitter::isZeroConstant(ValueRef ref) -> bool {
  auto op = value(ref).getDefiningOp();
  if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(op)) {
    if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return i.getValue().isZero();
  }
  return mlir::isa_and_nonnull<mlir::cxx::ZeroOp>(op);
}
auto MlirEmitter::symbolExists(std::string_view name) -> bool {
  auto module = module_;
  return module && module.lookupSymbol(llvm::StringRef{name});
}
auto MlirEmitter::initializerAttribute(const Initializer& init)
    -> mlir::Attribute {
  switch (init.kind) {
    case Initializer::Kind::None:
      return {};
    case Initializer::Kind::Integer: {
      auto t = type(init.type);
      auto intType = mlir::dyn_cast<mlir::IntegerType>(t);
      const auto width = intType ? intType.getWidth() : 0;

      if (width <= 64) {
        return builder_.getIntegerAttr(t, init.integer.toIntMax());
      }

      auto bits = llvm::APInt{width, init.integer.lowBits()};
      bits |= llvm::APInt{width, init.integer.highBits()}.shl(64);
      return builder_.getIntegerAttr(t, bits);
    }
    case Initializer::Kind::Floating: {
      auto t = mlir::cast<mlir::FloatType>(type(init.type));
      llvm::APFloat f{init.floating};
      bool losesInfo = false;
      f.convert(t.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
      return builder_.getFloatAttr(t, f);
    }
    case Initializer::Kind::Bytes:
      return builder_.getStringAttr(init.bytes);
    case Initializer::Kind::Null:
      return builder_.getUnitAttr();
    case Initializer::Kind::Zero:
      return mlir::LLVM::ZeroAttr::get(context());
    case Initializer::Kind::ScalarZero:
      return builder_.getZeroAttr(type(init.type));
    case Initializer::Kind::Undef:
      return {};
    case Initializer::Kind::SignalingNaN: {
      auto t = mlir::cast<mlir::FloatType>(type(init.type));
      return builder_.getFloatAttr(
          t, llvm::APFloat::getSNaN(t.getFloatSemantics()));
    }
    case Initializer::Kind::Aggregate: {
      std::vector<mlir::Attribute> values;
      for (const auto& v : init.elements)
        values.push_back(initializerAttribute(v));
      return builder_.getArrayAttr(values);
    }
  }
  return {};
}
auto MlirEmitter::constant(SourceLocation loc, TypeRef resultType,
                           const Initializer& init) -> ValueRef {
  const auto location = getLocation(loc);

  switch (init.kind) {
    case Initializer::Kind::None:
      return {};

    case Initializer::Kind::Null:
      return wrap(mlir::cxx::NullPtrConstantOp::create(
          builder_, location,
          mlir::cast<mlir::cxx::PointerType>(type(resultType))));

    case Initializer::Kind::Zero:
      return wrap(
          mlir::cxx::ZeroOp::create(builder_, location, type(resultType)));

    case Initializer::Kind::Undef:
      return wrap(
          mlir::cxx::UndefOp::create(builder_, location, type(resultType)));

    case Initializer::Kind::ScalarZero: {
      auto zeroType = type(resultType);
      return wrap(mlir::arith::ConstantOp::create(
          builder_, location, zeroType, builder_.getZeroAttr(zeroType)));
    }

    case Initializer::Kind::SignalingNaN: {
      auto floatType = mlir::cast<mlir::FloatType>(type(resultType));
      return wrap(mlir::arith::ConstantOp::create(
          builder_, location, floatType,
          mlir::FloatAttr::get(floatType, llvm::APFloat::getSNaN(
                                              floatType.getFloatSemantics()))));
    }

    case Initializer::Kind::Integer:
    case Initializer::Kind::Floating:
    case Initializer::Kind::Bytes:
    case Initializer::Kind::Aggregate:
      return wrap(mlir::arith::ConstantOp::create(
          builder_, location, type(resultType),
          mlir::cast<mlir::TypedAttr>(initializerAttribute(init))));
  }
}

auto MlirEmitter::todo(SourceLocation loc, TodoKind kind,
                       std::string_view message) -> ValueRef {
  const auto location = getLocation(loc);

  switch (kind) {
    case TodoKind::Expression:
      return wrap(mlir::cxx::TodoExprOp::create(builder_, location, message));

    case TodoKind::Statement:
      mlir::cxx::TodoStmtOp::create(builder_, location, message);
      return {};
  }
}
auto MlirEmitter::unaryOp(SourceLocation loc, UnaryOp op, TypeRef t, ValueRef v)
    -> ValueRef {
  switch (op) {
    case UnaryOp::NegateFloat:
      return wrap(mlir::arith::NegFOp::create(builder_, getLocation(loc),
                                              type(t), value(v)));
  }
}
namespace {
auto toMlir(Linkage linkage) -> mlir::cxx::LinkageKind;
}
void MlirEmitter::defineVTable(SourceLocation loc, const VTableInfo& info) {
  std::vector<mlir::Attribute> vbaseOffsets;
  std::vector<mlir::Attribute> vcallOffsets;
  std::vector<std::int64_t> offsetsToTop;
  std::vector<mlir::Attribute> slots;

  for (const auto& table : info.tables) {
    vbaseOffsets.push_back(builder_.getI64ArrayAttr(
        {table.virtualBaseOffsets.data(), table.virtualBaseOffsets.size()}));
    vcallOffsets.push_back(builder_.getI64ArrayAttr(
        {table.virtualCallOffsets.data(), table.virtualCallOffsets.size()}));
    offsetsToTop.push_back(table.offsetToTop);

    std::vector<mlir::Attribute> tableSlots;
    for (auto f : table.slots)
      tableSlots.push_back(f ? mlir::Attribute(mlir::FlatSymbolRefAttr::get(
                                   context(), functionName(f)))
                             : mlir::Attribute(builder_.getUnitAttr()));
    slots.push_back(builder_.getArrayAttr(tableSlots));
  }

  mlir::cxx::VTableOp::create(
      builder_, getLocation(loc), info.name,
      builder_.getArrayAttr(vbaseOffsets), builder_.getArrayAttr(vcallOffsets),
      builder_.getI64ArrayAttr(offsetsToTop),
      mlir::FlatSymbolRefAttr::get(context(), info.typeInfo),
      builder_.getArrayAttr(slots),
      mlir::cxx::LinkageKindAttr::get(context(), toMlir(info.linkage)));
}

auto MlirEmitter::debug() -> DebugEmitter* {
  if (!debug_) debug_ = std::make_unique<MlirDebugEmitter>(*this, unit_);
  return debug_.get();
}

template <typename T>
static auto mintHandleId(std::vector<T>& objects,
                         std::vector<std::uint8_t>& generations, T object,
                         std::string_view what) -> std::uint32_t {
  const auto index = static_cast<std::uint32_t>(objects.size());

  if (index > kHandleMaxIndex)
    cxx_runtime_error(std::format("too many {} in one module", what));

  if (index >= generations.size()) generations.resize(index + 1, 0);

  objects.push_back(object);

  return handleId(index, generations[index]);
}

template <typename T>
static auto resolveHandleId(const std::vector<T>& objects,
                            const std::vector<std::uint8_t>& generations,
                            std::uint32_t id, std::string_view what) -> T {
  if (!id) return T{};

  const auto index = handleIndex(id);

  if (index >= objects.size() || generations[index] != handleGeneration(id)) {
    cxx_runtime_error(std::format(
        "{} handle outlived the function body that created it", what));
  }

  return objects[index];
}

auto MlirEmitter::wrap(mlir::Block* block) -> BlockRef {
  if (!block) return {};

  if (auto it = blockIds_.find(block); it != blockIds_.end())
    return HandleAccess::make<BlockTag>(it->second);

  const auto id = mintHandleId(blocks_, blockGenerations_, block, "blocks");
  blockIds_.insert({block, id});

  return HandleAccess::make<BlockTag>(id);
}

auto MlirEmitter::block(BlockRef ref) -> mlir::Block* {
  return resolveHandleId(blocks_, blockGenerations_, HandleAccess::id(ref),
                         "block");
}

void MlirEmitter::beginFunctionBody(FunctionRef function) {
  functionBodies_.push_back(
      {.function = function,
       .blockMark = static_cast<std::uint32_t>(blocks_.size()),
       .valueMark = static_cast<std::uint32_t>(values_.size())});
}

void MlirEmitter::endFunctionBody(FunctionRef function) {
  if (functionBodies_.empty() || functionBodies_.back().function != function)
    cxx_runtime_error("unbalanced function body scope");

  const auto scope = functionBodies_.back();
  functionBodies_.pop_back();

  for (auto i = scope.valueMark; i < values_.size(); ++i) {
    valueIds_.erase(values_[i]);
    ++valueGenerations_[i];
  }
  values_.resize(scope.valueMark);

  for (auto i = scope.blockMark; i < blocks_.size(); ++i) {
    blockIds_.erase(blocks_[i]);
    ++blockGenerations_[i];
  }
  blocks_.resize(scope.blockMark);
}

auto MlirEmitter::functionParameterTypes(FunctionRef ref)
    -> std::vector<TypeRef> {
  std::vector<TypeRef> result;
  for (auto t : function(ref).getFunctionType().getInputs())
    result.push_back(wrap(t));
  return result;
}
auto MlirEmitter::functionResultTypes(FunctionRef ref) -> std::vector<TypeRef> {
  std::vector<TypeRef> result;
  for (auto t : function(ref).getFunctionType().getResults())
    result.push_back(wrap(t));
  return result;
}
auto MlirEmitter::addBlockParameter(BlockRef ref, TypeRef t, SourceLocation loc)
    -> ValueRef {
  return wrap(addBlockArgument(ref, type(t), getLocation(loc)));
}
auto MlirEmitter::blockParameter(BlockRef ref, unsigned index) -> ValueRef {
  return wrap(blockArgument(ref, index));
}
auto MlirEmitter::blockParameterCount(BlockRef ref) -> unsigned {
  return blockArgumentCount(ref);
}

auto MlirEmitter::createBlock(FunctionRef ref) -> BlockRef {
  mlir::Region* region = nullptr;

  if (ref) {
    region = &function(ref).getBody();
  } else if (auto insertionPoint = builder_.getBlock()) {
    region = insertionPoint->getParent();
  } else {
    cxx_runtime_error("cannot create a block outside of a region");
  }

  auto block = new mlir::Block();
  region->getBlocks().push_back(block);

  return wrap(block);
}

auto MlirEmitter::wrap(mlir::Value value) -> ValueRef {
  if (!value) return {};

  if (auto it = valueIds_.find(value); it != valueIds_.end())
    return HandleAccess::make<ValueTag>(it->second);

  const auto id = mintHandleId(values_, valueGenerations_, value, "values");
  valueIds_.insert({value, id});

  return HandleAccess::make<ValueTag>(id);
}

auto MlirEmitter::value(ValueRef ref) -> mlir::Value {
  return resolveHandleId(values_, valueGenerations_, HandleAccess::id(ref),
                         "value");
}

auto MlirEmitter::values(llvm::ArrayRef<ValueRef> refs)
    -> llvm::SmallVector<mlir::Value> {
  llvm::SmallVector<mlir::Value> result;
  result.reserve(refs.size());
  for (auto ref : refs) result.push_back(value(ref));
  return result;
}

auto MlirEmitter::wrap(mlir::Type type) -> TypeRef {
  if (!type) return {};

  auto [it, inserted] =
      typeIds_.try_emplace(type, static_cast<std::uint32_t>(types_.size()));

  if (inserted) types_.push_back(type);

  return HandleAccess::make<TypeTag>(it->second);
}

auto MlirEmitter::type(TypeRef ref) -> mlir::Type {
  const auto id = HandleAccess::id(ref);
  if (id >= types_.size()) return {};
  return types_[id];
}

auto MlirEmitter::types(llvm::ArrayRef<TypeRef> refs)
    -> llvm::SmallVector<mlir::Type> {
  llvm::SmallVector<mlir::Type> result;
  result.reserve(refs.size());
  for (auto ref : refs) result.push_back(type(ref));
  return result;
}

auto MlirEmitter::voidType() -> TypeRef {
  return wrap(mlir::cxx::VoidType::get(context()));
}

auto MlirEmitter::unresolvedType() -> TypeRef {
  return wrap(mlir::cxx::ExprType::get(context()));
}

auto MlirEmitter::integerType(unsigned bits) -> TypeRef {
  return wrap(mlir::IntegerType::get(context(), bits));
}

auto MlirEmitter::floatingType(FloatKind kind) -> TypeRef {
  switch (kind) {
    case FloatKind::Half:
      return wrap(mlir::Float16Type::get(context()));
    case FloatKind::Single:
      return wrap(mlir::Float32Type::get(context()));
    case FloatKind::Double:
      return wrap(mlir::Float64Type::get(context()));
    case FloatKind::X87DoubleExtended:
      return wrap(mlir::Float80Type::get(context()));
    case FloatKind::Quad:
      return wrap(mlir::Float128Type::get(context()));
  }
}

auto MlirEmitter::pointerType(TypeRef elementType) -> TypeRef {
  return wrap(mlir::cxx::PointerType::get(context(), type(elementType)));
}

auto MlirEmitter::arrayType(TypeRef elementType, std::uint64_t size)
    -> TypeRef {
  return wrap(mlir::cxx::ArrayType::get(context(), type(elementType), size));
}

auto MlirEmitter::vectorType(TypeRef elementType, std::uint64_t elementCount)
    -> TypeRef {
  return wrap(mlir::VectorType::get({static_cast<std::int64_t>(elementCount)},
                                    type(elementType)));
}

auto MlirEmitter::vectorSplat(SourceLocation loc, TypeRef vectorType,
                              ValueRef scalar) -> ValueRef {
  return wrap(mlir::vector::BroadcastOp::create(
      builder_, getLocation(loc), type(vectorType), value(scalar)));
}

auto MlirEmitter::functionType(std::span<const TypeRef> parameters,
                               std::span<const TypeRef> results,
                               bool isVariadic) -> TypeRef {
  return wrap(mlir::cxx::FunctionType::get(
      context(), types({parameters.data(), parameters.size()}),
      types({results.data(), results.size()}), isVariadic));
}

auto MlirEmitter::declareClassType(std::string_view name) -> TypeRef {
  return wrap(mlir::cxx::ClassType::getNamed(
      context(), mlir::StringRef{name.data(), name.size()}));
}

void MlirEmitter::defineClassType(TypeRef classType,
                                  std::span<const TypeRef> members,
                                  bool isPacked) {
  mlir::cast<mlir::cxx::ClassType>(type(classType))
      .setBody(types({members.data(), members.size()}), isPacked);
}

auto MlirEmitter::typeKind(TypeRef ref) -> TypeKind {
  auto target = type(ref);
  if (!target) return TypeKind::Other;
  if (mlir::isa<mlir::cxx::VoidType>(target)) return TypeKind::Void;
  if (mlir::isa<mlir::cxx::ExprType>(target)) return TypeKind::Unresolved;
  if (mlir::isa<mlir::IntegerType>(target)) return TypeKind::Integer;
  if (mlir::isa<mlir::FloatType>(target)) return TypeKind::Floating;
  if (mlir::isa<mlir::cxx::PointerType>(target)) return TypeKind::Pointer;
  if (mlir::isa<mlir::cxx::ArrayType>(target)) return TypeKind::Array;
  if (mlir::isa<mlir::cxx::ClassType>(target)) return TypeKind::Class;
  if (mlir::isa<mlir::cxx::FunctionType>(target)) return TypeKind::Function;
  return TypeKind::Other;
}

auto MlirEmitter::scalarWidth(TypeRef ref) -> unsigned {
  return type(ref).getIntOrFloatBitWidth();
}

auto MlirEmitter::elementType(TypeRef ref) -> TypeRef {
  auto target = type(ref);
  if (auto pointer = mlir::dyn_cast<mlir::cxx::PointerType>(target))
    return wrap(pointer.getElementType());
  if (auto array = mlir::dyn_cast<mlir::cxx::ArrayType>(target))
    return wrap(array.getElementType());
  return {};
}

auto MlirEmitter::typeOf(ValueRef ref) -> TypeRef {
  auto target = value(ref);
  return target ? wrap(target.getType()) : TypeRef{};
}

auto MlirEmitter::convert(SourceLocation loc, CastKind kind, ValueRef value,
                          TypeRef type) -> ValueRef {
  const auto location = getLocation(loc);
  auto resultType = this->type(type);
  auto operand = this->value(value);

  switch (kind) {
    case CastKind::Truncate:
      return wrap(mlir::arith::TruncIOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::SignExtend:
      return wrap(mlir::arith::ExtSIOp::create(builder_, location, resultType,
                                               operand));
    case CastKind::ZeroExtend:
      return wrap(mlir::arith::ExtUIOp::create(builder_, location, resultType,
                                               operand));
    case CastKind::FloatExtend:
      return wrap(
          mlir::arith::ExtFOp::create(builder_, location, resultType, operand));
    case CastKind::FloatTruncate:
      return wrap(mlir::arith::TruncFOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::SignedIntToFloat:
      return wrap(mlir::arith::SIToFPOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::UnsignedIntToFloat:
      return wrap(mlir::arith::UIToFPOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::FloatToSignedInt:
      return wrap(mlir::arith::FPToSIOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::FloatToUnsignedInt:
      return wrap(mlir::arith::FPToUIOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::ReinterpretBits:
      return wrap(mlir::arith::BitcastOp::create(builder_, location, resultType,
                                                 operand));
    case CastKind::Bitcast:
      return wrap(mlir::cxx::BitcastOp::create(builder_, location, resultType,
                                               operand));
    case CastKind::Reshape:
      return wrap(mlir::cxx::ReshapeOp::create(builder_, location, resultType,
                                               operand));
    case CastKind::ArrayToPointer:
      return wrap(mlir::cxx::ArrayToPointerOp::create(builder_, location,
                                                      resultType, operand));
    case CastKind::PointerToInt:
      return wrap(mlir::cxx::PtrToIntOp::create(builder_, location, resultType,
                                                operand));
    case CastKind::IntToPointer:
      return wrap(mlir::cxx::IntToPtrOp::create(builder_, location, resultType,
                                                operand));
  }
}

auto MlirEmitter::createBlock(mlir::Region& region) -> BlockRef {
  return wrap(builder_.createBlock(&region));
}

auto MlirEmitter::insertionBlock() -> BlockRef {
  return wrap(builder_.getInsertionBlock());
}

auto MlirEmitter::blockArgumentCount(BlockRef ref) -> unsigned {
  auto target = block(ref);
  return target ? target->getNumArguments() : 0;
}

auto MlirEmitter::blockArgument(BlockRef ref, unsigned index) -> mlir::Value {
  return block(ref)->getArgument(index);
}

auto MlirEmitter::addBlockArgument(BlockRef ref, mlir::Type type,
                                   mlir::Location loc) -> mlir::Value {
  return block(ref)->addArgument(type, loc);
}

void MlirEmitter::eraseBlock(BlockRef ref) {
  auto target = block(ref);
  if (!target) return;

  blockIds_.erase(target);
  blocks_[handleIndex(HandleAccess::id(ref))] = nullptr;

  target->erase();
}

auto MlirEmitter::hasTerminator(BlockRef ref) -> bool {
  auto target = block(ref);
  return target && target->mightHaveTerminator();
}

auto MlirEmitter::enclosingFunctionLocation() -> mlir::Location {
  auto function = enclosingFunction();
  if (!function) return mlir::UnknownLoc::get(builder_.getContext());

  auto loc = function.getLoc();
  while (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    if (fused.getLocations().empty()) break;
    loc = fused.getLocations().front();
  }
  return loc;
}

auto MlirEmitter::getLocation(SourceLocation loc) -> mlir::Location {
  if (!loc) return mlir::UnknownLoc::get(builder_.getContext());
  auto [filename, line, column] = unit_->tokenStartPosition(loc);
  return mlir::FileLineColLoc::get(builder_.getContext(), filename, line,
                                   column);
}

static auto fromMlir(mlir::cxx::LinkageKind kind) -> Linkage {
  switch (kind) {
    case mlir::cxx::LinkageKind::External:
      return Linkage::External;
    case mlir::cxx::LinkageKind::Internal:
      return Linkage::Internal;
    case mlir::cxx::LinkageKind::LinkOnceODR:
      return Linkage::LinkOnceODR;
    case mlir::cxx::LinkageKind::WeakODR:
      return Linkage::WeakODR;
    case mlir::cxx::LinkageKind::AvailableExternally:
      return Linkage::AvailableExternally;
    case mlir::cxx::LinkageKind::Appending:
      return Linkage::Appending;
  }
  llvm_unreachable("unknown linkage");
}

auto MlirEmitter::globalLinkage(GlobalRef ref) -> Linkage {
  return fromMlir(
      global(ref).getLinkageKind().value_or(mlir::cxx::LinkageKind::External));
}

auto MlirEmitter::functionLinkage(FunctionRef ref) -> Linkage {
  return fromMlir(function(ref).getLinkageKind().value_or(
      mlir::cxx::LinkageKind::External));
}
void MlirEmitter::branch(SourceLocation loc, BlockRef target,
                         std::span<const ValueRef> operands) {
  branch(getLocation(loc), target, values({operands.data(), operands.size()}));
}
void MlirEmitter::branch(mlir::Location loc, BlockRef target,
                         mlir::ValueRange operands) {
  mlir::cf::BranchOp::create(builder_, loc, block(target), operands);
}

auto MlirEmitter::binaryOp(SourceLocation loc, BinaryOp op, ValueRef lhs,
                           ValueRef rhs) -> ValueRef {
  const auto location = getLocation(loc);
  auto lhs_ = value(lhs);
  auto rhs_ = value(rhs);

  switch (op) {
    case BinaryOp::AddInt:
      return wrap(mlir::arith::AddIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::SubInt:
      return wrap(mlir::arith::SubIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::MulInt:
      return wrap(mlir::arith::MulIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::SignedDiv:
      return wrap(mlir::arith::DivSIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::UnsignedDiv:
      return wrap(mlir::arith::DivUIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::SignedRem:
      return wrap(mlir::arith::RemSIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::UnsignedRem:
      return wrap(mlir::arith::RemUIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::AndInt:
      return wrap(mlir::arith::AndIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::OrInt:
      return wrap(mlir::arith::OrIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::XorInt:
      return wrap(mlir::arith::XOrIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::ShiftLeft:
      return wrap(mlir::arith::ShLIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::ArithmeticShiftRight:
      return wrap(mlir::arith::ShRSIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::LogicalShiftRight:
      return wrap(mlir::arith::ShRUIOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::AddFloat:
      return wrap(mlir::arith::AddFOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::SubFloat:
      return wrap(mlir::arith::SubFOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::MulFloat:
      return wrap(mlir::arith::MulFOp::create(builder_, location, lhs_, rhs_));
    case BinaryOp::DivFloat:
      return wrap(mlir::arith::DivFOp::create(builder_, location, lhs_, rhs_));
  }
}

namespace {

auto toMlir(IntPredicate predicate) -> mlir::arith::CmpIPredicate {
  switch (predicate) {
    case IntPredicate::Equal:
      return mlir::arith::CmpIPredicate::eq;
    case IntPredicate::NotEqual:
      return mlir::arith::CmpIPredicate::ne;
    case IntPredicate::SignedLess:
      return mlir::arith::CmpIPredicate::slt;
    case IntPredicate::SignedLessEqual:
      return mlir::arith::CmpIPredicate::sle;
    case IntPredicate::SignedGreater:
      return mlir::arith::CmpIPredicate::sgt;
    case IntPredicate::SignedGreaterEqual:
      return mlir::arith::CmpIPredicate::sge;
    case IntPredicate::UnsignedLess:
      return mlir::arith::CmpIPredicate::ult;
    case IntPredicate::UnsignedLessEqual:
      return mlir::arith::CmpIPredicate::ule;
    case IntPredicate::UnsignedGreater:
      return mlir::arith::CmpIPredicate::ugt;
    case IntPredicate::UnsignedGreaterEqual:
      return mlir::arith::CmpIPredicate::uge;
  }
}

auto toMlir(FloatPredicate predicate) -> mlir::arith::CmpFPredicate {
  switch (predicate) {
    case FloatPredicate::OrderedEqual:
      return mlir::arith::CmpFPredicate::OEQ;
    case FloatPredicate::OrderedNotEqual:
      return mlir::arith::CmpFPredicate::ONE;
    case FloatPredicate::OrderedLess:
      return mlir::arith::CmpFPredicate::OLT;
    case FloatPredicate::OrderedLessEqual:
      return mlir::arith::CmpFPredicate::OLE;
    case FloatPredicate::OrderedGreater:
      return mlir::arith::CmpFPredicate::OGT;
    case FloatPredicate::OrderedGreaterEqual:
      return mlir::arith::CmpFPredicate::OGE;
    case FloatPredicate::UnorderedNotEqual:
      return mlir::arith::CmpFPredicate::UNE;
    case FloatPredicate::Unordered:
      return mlir::arith::CmpFPredicate::UNO;
  }
}

}  // namespace

auto MlirEmitter::compareInt(SourceLocation loc, IntPredicate predicate,
                             ValueRef lhs, ValueRef rhs) -> ValueRef {
  return wrap(mlir::arith::CmpIOp::create(
      builder_, getLocation(loc), toMlir(predicate), value(lhs), value(rhs)));
}

auto MlirEmitter::compareFloat(SourceLocation loc, FloatPredicate predicate,
                               ValueRef lhs, ValueRef rhs) -> ValueRef {
  return wrap(mlir::arith::CmpFOp::create(
      builder_, getLocation(loc), toMlir(predicate), value(lhs), value(rhs)));
}

auto MlirEmitter::select(SourceLocation loc, ValueRef condition,
                         ValueRef ifTrue, ValueRef ifFalse) -> ValueRef {
  return wrap(mlir::arith::SelectOp::create(builder_, getLocation(loc),
                                            value(condition), value(ifTrue),
                                            value(ifFalse)));
}

auto MlirEmitter::allocate(SourceLocation loc, TypeRef pointerType,
                           ValueRef size, std::uint64_t alignment) -> ValueRef {
  if (size)
    return dynamicAllocate(getLocation(loc), pointerType, size, alignment);
  return allocate(getLocation(loc), pointerType, alignment);
}

auto MlirEmitter::allocate(mlir::Location loc, TypeRef pointerType,
                           std::uint64_t alignment) -> ValueRef {
  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  if (auto entry = enclosingFunctionEntryBlock()) {
    auto insertionPoint = entry->begin();
    while (insertionPoint != entry->end() &&
           mlir::isa<mlir::cxx::AllocaOp>(*insertionPoint))
      ++insertionPoint;
    builder_.setInsertionPoint(entry, insertionPoint);
  }

  return wrap(
      mlir::cxx::AllocaOp::create(builder_, loc, type(pointerType), alignment));
}

auto MlirEmitter::enclosingFunction() -> mlir::cxx::FuncOp {
  auto block = builder_.getInsertionBlock();
  if (!block) return {};

  for (auto op = block->getParentOp(); op; op = op->getParentOp()) {
    if (auto function = mlir::dyn_cast<mlir::cxx::FuncOp>(op)) return function;
  }

  return {};
}

auto MlirEmitter::enclosingFunctionEntryBlock() -> mlir::Block* {
  auto function = enclosingFunction();
  if (!function) return nullptr;
  if (function.getBody().empty()) return nullptr;
  return &function.getBody().front();
}

auto MlirEmitter::dynamicAllocate(mlir::Location loc, TypeRef pointerType,
                                  ValueRef size, std::uint64_t alignment)
    -> ValueRef {
  return wrap(mlir::cxx::DynAllocaOp::create(builder_, loc, type(pointerType),
                                             value(size), alignment));
}

auto MlirEmitter::load(SourceLocation loc, TypeRef valueType, ValueRef address,
                       const Access& access) -> ValueRef {
  if (!access.bitfield)
    return load(getLocation(loc), valueType, address, access.alignment);

  const auto& field = *access.bitfield;
  return wrap(mlir::cxx::BitfieldLoadOp::create(
      builder_, getLocation(loc), type(valueType), value(address),
      builder_.getI32IntegerAttr(field.bitOffset),
      builder_.getI32IntegerAttr(field.bitWidth),
      builder_.getI64IntegerAttr(access.alignment),
      builder_.getBoolAttr(access.isSigned)));
}

auto MlirEmitter::load(mlir::Location loc, TypeRef valueType, ValueRef address,
                       std::uint64_t alignment) -> ValueRef {
  return wrap(mlir::cxx::LoadOp::create(builder_, loc, type(valueType),
                                        value(address), alignment));
}

auto MlirEmitter::pointerAdd(SourceLocation loc, TypeRef pointerType,
                             ValueRef base, ValueRef offset) -> ValueRef {
  return pointerAdd(getLocation(loc), pointerType, base, offset);
}

auto MlirEmitter::pointerAdd(mlir::Location loc, TypeRef pointerType,
                             ValueRef base, ValueRef offset) -> ValueRef {
  return wrap(mlir::cxx::PtrAddOp::create(builder_, loc, type(pointerType),
                                          value(base), value(offset)));
}

auto MlirEmitter::pointerDiff(SourceLocation loc, TypeRef resultType,
                              ValueRef lhs, ValueRef rhs) -> ValueRef {
  return pointerDiff(getLocation(loc), resultType, lhs, rhs);
}

auto MlirEmitter::pointerDiff(mlir::Location loc, TypeRef resultType,
                              ValueRef lhs, ValueRef rhs) -> ValueRef {
  return wrap(mlir::cxx::PtrDiffOp::create(builder_, loc, type(resultType),
                                           value(lhs), value(rhs)));
}

auto MlirEmitter::subscript(SourceLocation loc, TypeRef pointerType,
                            ValueRef base, ValueRef index) -> ValueRef {
  return subscript(getLocation(loc), pointerType, base, index);
}

auto MlirEmitter::subscript(mlir::Location loc, TypeRef pointerType,
                            ValueRef base, ValueRef index) -> ValueRef {
  return wrap(mlir::cxx::SubscriptOp::create(builder_, loc, type(pointerType),
                                             value(base), value(index)));
}

auto MlirEmitter::memberAddress(SourceLocation loc, TypeRef pointerType,
                                ValueRef base, std::uint32_t index)
    -> ValueRef {
  return memberAddress(getLocation(loc), pointerType, base, index);
}

auto MlirEmitter::memberAddress(mlir::Location loc, TypeRef pointerType,
                                ValueRef base, std::uint32_t index)
    -> ValueRef {
  return wrap(mlir::cxx::MemberOp::create(builder_, loc, type(pointerType),
                                          value(base), index));
}

auto MlirEmitter::extractValue(SourceLocation loc, TypeRef resultType,
                               ValueRef container, std::int64_t position)
    -> ValueRef {
  return extractValue(getLocation(loc), resultType, container, position);
}

auto MlirEmitter::extractValue(mlir::Location loc, TypeRef resultType,
                               ValueRef container, std::int64_t position)
    -> ValueRef {
  return wrap(mlir::cxx::ExtractValueOp::create(
      builder_, loc, type(resultType), this->value(container), position));
}

auto MlirEmitter::insertValue(SourceLocation loc, TypeRef resultType,
                              ValueRef container, ValueRef value,
                              std::int64_t position) -> ValueRef {
  return insertValue(getLocation(loc), resultType, container, value, position);
}

auto MlirEmitter::insertValue(mlir::Location loc, TypeRef resultType,
                              ValueRef container, ValueRef value,
                              std::int64_t position) -> ValueRef {
  return wrap(mlir::cxx::InsertValueOp::create(builder_, loc, type(resultType),
                                               this->value(container),
                                               this->value(value), position));
}

auto MlirEmitter::addressOfSymbol(SourceLocation loc, TypeRef resultType,
                                  std::string_view symbol) -> ValueRef {
  return addressOfSymbol(getLocation(loc), resultType, symbol);
}

auto MlirEmitter::addressOfSymbol(mlir::Location loc, TypeRef resultType,
                                  std::string_view symbol) -> ValueRef {
  return wrap(mlir::cxx::AddressOfOp::create(
      builder_, loc, type(resultType),
      mlir::StringRef{symbol.data(), symbol.size()}));
}

void MlirEmitter::store(SourceLocation loc, ValueRef value, ValueRef address,
                        const Access& access) {
  if (!access.bitfield) {
    store(getLocation(loc), value, address, access.alignment);
    return;
  }

  const auto& field = *access.bitfield;
  mlir::cxx::BitfieldStoreOp::create(
      builder_, getLocation(loc), this->value(value), this->value(address),
      builder_.getI32IntegerAttr(field.bitOffset),
      builder_.getI32IntegerAttr(field.bitWidth),
      builder_.getI64IntegerAttr(access.alignment));
}

void MlirEmitter::store(mlir::Location loc, ValueRef value, ValueRef address,
                        std::uint64_t alignment) {
  mlir::cxx::StoreOp::create(builder_, loc, this->value(value),
                             this->value(address), alignment);
}

void MlirEmitter::memsetZero(SourceLocation loc, ValueRef address,
                             std::uint64_t size) {
  memsetZero(getLocation(loc), address, size);
}

void MlirEmitter::memsetZero(mlir::Location loc, ValueRef address,
                             std::uint64_t size) {
  mlir::cxx::MemSetZeroOp::create(builder_, loc, value(address), size);
}

void MlirEmitter::memcpy(SourceLocation loc, ValueRef destination,
                         ValueRef source, std::uint64_t size) {
  memcpy(getLocation(loc), destination, source, size);
}

void MlirEmitter::memcpy(mlir::Location loc, ValueRef destination,
                         ValueRef source, std::uint64_t size) {
  mlir::cxx::MemCpyOp::create(builder_, loc, value(destination), value(source),
                              size);
}

void MlirEmitter::unreachable(SourceLocation loc) {
  unreachable(getLocation(loc));
}

void MlirEmitter::unreachable(mlir::Location loc) {
  mlir::cxx::UnreachableOp::create(builder_, loc);
}

namespace {

auto toMlir(Linkage linkage) -> mlir::cxx::LinkageKind {
  switch (linkage) {
    case Linkage::External:
      return mlir::cxx::LinkageKind::External;
    case Linkage::Internal:
      return mlir::cxx::LinkageKind::Internal;
    case Linkage::LinkOnceODR:
      return mlir::cxx::LinkageKind::LinkOnceODR;
    case Linkage::WeakODR:
      return mlir::cxx::LinkageKind::WeakODR;
    case Linkage::AvailableExternally:
      return mlir::cxx::LinkageKind::AvailableExternally;
    case Linkage::Appending:
      return mlir::cxx::LinkageKind::Appending;
  }
}

auto toMlir(Visibility visibility) -> mlir::cxx::Visibility {
  switch (visibility) {
    case Visibility::Default:
      return mlir::cxx::Visibility::Default;
    case Visibility::Hidden:
      return mlir::cxx::Visibility::Hidden;
    case Visibility::Protected:
      return mlir::cxx::Visibility::Protected;
  }
}

auto toMlir(InlineKind kind) -> mlir::cxx::InlineKind {
  switch (kind) {
    case InlineKind::NoInline:
      return mlir::cxx::InlineKind::NoInline;
    case InlineKind::InlineHint:
      return mlir::cxx::InlineKind::InlineHint;
  }
}

}  // namespace

auto MlirEmitter::wrap(mlir::cxx::FuncOp function) -> FunctionRef {
  if (!function) return {};

  auto [it, inserted] = functionIds_.try_emplace(
      function.getOperation(), static_cast<std::uint32_t>(functions_.size()));

  if (inserted) functions_.push_back(function);

  return HandleAccess::make<FunctionTag>(it->second);
}

auto MlirEmitter::function(FunctionRef ref) -> mlir::cxx::FuncOp {
  const auto id = HandleAccess::id(ref);
  if (id >= functions_.size()) return {};
  return functions_[id];
}

auto MlirEmitter::wrap(mlir::cxx::GlobalOp global) -> GlobalRef {
  if (!global) return {};

  auto [it, inserted] = globalIds_.try_emplace(
      global.getOperation(), static_cast<std::uint32_t>(globals_.size()));

  if (inserted) globals_.push_back(global);

  return HandleAccess::make<GlobalTag>(it->second);
}

auto MlirEmitter::global(GlobalRef ref) -> mlir::cxx::GlobalOp {
  const auto id = HandleAccess::id(ref);
  if (id >= globals_.size()) return {};
  return globals_[id];
}

auto MlirEmitter::findFunction(std::string_view name) -> FunctionRef {
  if (!module_) return {};
  auto symbol = mlir::SymbolTable::lookupSymbolIn(
      module_, mlir::StringAttr::get(
                   context(), mlir::StringRef{name.data(), name.size()}));
  return wrap(mlir::dyn_cast_or_null<mlir::cxx::FuncOp>(symbol));
}

auto MlirEmitter::declareFunction(SourceLocation loc, const FunctionInfo& info)
    -> FunctionRef {
  return declareFunction(getLocation(loc), info);
}

auto MlirEmitter::parameterAbiAttrs(std::span<const ParameterAbi> parameters,
                                    std::size_t count) -> mlir::ArrayAttr {
  if (parameters.empty()) return {};

  auto context = this->context();
  mlir::SmallVector<mlir::Attribute> entries;
  bool hasAttr = false;

  for (std::size_t i = 0; i < count; ++i) {
    auto parameter = i < parameters.size() ? parameters[i] : ParameterAbi{};
    mlir::SmallVector<mlir::NamedAttribute> attrs;
    if (parameter.kind != ParameterAbiKind::Default) {
      auto name = parameter.kind == ParameterAbiKind::StructReturn
                      ? mlir::LLVM::LLVMDialect::getStructRetAttrName()
                      : mlir::LLVM::LLVMDialect::getByValAttrName();
      attrs.emplace_back(mlir::StringAttr::get(context, name),
                         mlir::TypeAttr::get(type(parameter.indirectType)));
      if (parameter.alignment) {
        attrs.emplace_back(
            mlir::StringAttr::get(context,
                                  mlir::LLVM::LLVMDialect::getAlignAttrName()),
            builder_.getI64IntegerAttr(
                static_cast<std::int64_t>(parameter.alignment)));
      }
      hasAttr = true;
    }
    entries.push_back(mlir::DictionaryAttr::get(context, attrs));
  }

  if (!hasAttr) return {};

  return mlir::ArrayAttr::get(context, entries);
}

auto MlirEmitter::declareFunction(mlir::Location loc, const FunctionInfo& info)
    -> FunctionRef {
  auto context = this->context();

  auto optionalString = [&](std::string_view value) -> mlir::StringAttr {
    if (value.empty()) return {};
    return mlir::StringAttr::get(context,
                                 mlir::StringRef{value.data(), value.size()});
  };

  auto functionType = mlir::cast<mlir::cxx::FunctionType>(type(info.type));

  return wrap(mlir::cxx::FuncOp::create(
      builder_, loc, mlir::StringRef{info.name.data(), info.name.size()},
      functionType,
      mlir::cxx::LinkageKindAttr::get(context, toMlir(info.linkage)),
      mlir::cxx::InlineKindAttr::get(context, toMlir(info.inlineKind)),
      info.visibility == Visibility::Default
          ? mlir::cxx::VisibilityAttr{}
          : mlir::cxx::VisibilityAttr::get(context, toMlir(info.visibility)),
      optionalString(info.aliasee), optionalString(info.importModule),
      optionalString(info.importName), optionalString(info.exportName),
      info.isUsed,
      parameterAbiAttrs(info.parameters, functionType.getInputs().size()),
      mlir::ArrayAttr{}));
}

auto MlirEmitter::functionName(FunctionRef ref) -> std::string_view {
  auto name = function(ref).getSymName();
  return {name.data(), name.size()};
}

auto MlirEmitter::functionHasBody(FunctionRef ref) -> bool {
  return !function(ref).getBody().empty();
}

void MlirEmitter::setFunctionAliasee(FunctionRef ref,
                                     std::string_view aliasee) {
  auto func = function(ref);
  if (!func) return;
  func.setAliasee(mlir::StringRef{aliasee.data(), aliasee.size()});
}

auto MlirEmitter::findGlobal(std::string_view name) -> GlobalRef {
  if (!module_) return {};
  auto symbol = mlir::SymbolTable::lookupSymbolIn(
      module_, mlir::StringAttr::get(
                   context(), mlir::StringRef{name.data(), name.size()}));
  return wrap(mlir::dyn_cast_or_null<mlir::cxx::GlobalOp>(symbol));
}

auto MlirEmitter::declareGlobal(SourceLocation loc, const GlobalInfo& info)
    -> GlobalRef {
  auto alignmentAttr = info.alignment
                           ? builder_.getI64IntegerAttr(
                                 static_cast<std::int64_t>(info.alignment))
                           : mlir::IntegerAttr{};

  return wrap(mlir::cxx::GlobalOp::create(
      builder_,
      info.unknownLocation ? builder_.getUnknownLoc() : getLocation(loc),
      mlir::TypeRange(), type(info.type), info.isConstant,
      mlir::StringRef{info.name.data(), info.name.size()},
      initializerAttribute(info.initializer),
      mlir::cxx::LinkageKindAttr::get(context(), toMlir(info.linkage)),
      alignmentAttr, info.isUsed));
}

auto MlirEmitter::constantInt(mlir::Location loc, TypeRef typeRef,
                              std::int64_t value) -> ValueRef {
  auto integerType = type(typeRef);
  return wrap(mlir::arith::ConstantOp::create(
      builder_, loc, integerType, builder_.getIntegerAttr(integerType, value)));
}

auto MlirEmitter::call(SourceLocation loc, const CallInfo& info)
    -> std::vector<ValueRef> {
  return call(getLocation(loc), info);
}

auto MlirEmitter::call(mlir::Location loc, const CallInfo& info)
    -> std::vector<ValueRef> {
  if (mlir::isa<mlir::UnknownLoc>(loc)) loc = enclosingFunctionLocation();

  auto arguments = values({info.arguments.data(), info.arguments.size()});
  auto resultTypes = types({info.results.data(), info.results.size()});

  if (info.kind == CallKind::Builtin) {
    auto builtinOp = mlir::cxx::BuiltinCallOp::create(
        builder_, loc, resultTypes,
        mlir::StringRef{info.callee.data(), info.callee.size()}, arguments);

    std::vector<ValueRef> builtinResults;
    for (auto result : builtinOp.getResults())
      builtinResults.push_back(wrap(result));
    return builtinResults;
  }

  auto callOp =
      info.indirectCallee
          ? mlir::cxx::CallOp::create(builder_, loc, resultTypes,
                                      value(info.indirectCallee), arguments)
          : mlir::cxx::CallOp::create(
                builder_, loc, resultTypes,
                mlir::StringRef{info.callee.data(), info.callee.size()},
                arguments);

  if (auto argAttrs = parameterAbiAttrs(info.parameters, arguments.size())) {
    callOp.setArgAttrsAttr(argAttrs);
  }

  if (info.variadicCalleeType) {
    callOp.setVarCalleeType(
        mlir::cast<mlir::cxx::FunctionType>(type(info.variadicCalleeType)));
  }

  std::vector<ValueRef> results;
  results.reserve(callOp.getNumResults());
  for (auto result : callOp.getResults()) results.push_back(wrap(result));

  return results;
}

void MlirEmitter::ret(SourceLocation loc, std::span<const ValueRef> values) {
  ret(getLocation(loc), values);
}

void MlirEmitter::ret(mlir::Location loc, std::span<const ValueRef> refs) {
  mlir::cxx::ReturnOp::create(builder_, loc,
                              values({refs.data(), refs.size()}));
}

void MlirEmitter::condBranch(SourceLocation loc, ValueRef condition,
                             BlockRef trueDest, BlockRef falseDest) {
  condBranch(getLocation(loc), value(condition), trueDest, falseDest);
}

void MlirEmitter::condBranch(mlir::Location loc, mlir::Value condition,
                             BlockRef trueDest, BlockRef falseDest) {
  mlir::cf::CondBranchOp::create(builder_, loc, condition, block(trueDest),
                                 mlir::ValueRange{}, block(falseDest),
                                 mlir::ValueRange{});
}

void MlirEmitter::switchBranch(SourceLocation loc, ValueRef flag,
                               BlockRef defaultDest,
                               std::span<const std::int64_t> caseValues,
                               std::span<const BlockRef> caseDestinations) {
  switchBranch(getLocation(loc), value(flag),
               mlir::cast<mlir::IntegerType>(value(flag).getType()),
               defaultDest, {caseValues.data(), caseValues.size()},
               {caseDestinations.data(), caseDestinations.size()});
}

void MlirEmitter::switchBranch(mlir::Location loc, mlir::Value flag,
                               mlir::IntegerType flagType, BlockRef defaultDest,
                               llvm::ArrayRef<std::int64_t> caseValues,
                               llvm::ArrayRef<BlockRef> caseDestinations) {
  auto shapeType =
      mlir::VectorType::get(static_cast<std::int64_t>(caseValues.size()),
                            builder_.getIntegerType(64));

  auto caseValuesAttr = mlir::cast<mlir::DenseIntElementsAttr>(
      mlir::DenseIntElementsAttr::get(shapeType, caseValues)
          .mapValues(flagType, [&](mlir::APInt v) {
            return mlir::APInt(flagType.getIntOrFloatBitWidth(),
                               v.getZExtValue(), false, true);
          }));

  llvm::SmallVector<mlir::Block*> destinations;
  for (auto destination : caseDestinations) {
    destinations.push_back(block(destination));
  }

  std::vector<mlir::ValueRange> caseOperands(destinations.size(),
                                             mlir::ValueRange{});

  mlir::cf::SwitchOp::create(builder_, loc, flag, block(defaultDest), {},
                             caseValuesAttr, destinations, caseOperands);
}

}  // namespace cxx::ir
