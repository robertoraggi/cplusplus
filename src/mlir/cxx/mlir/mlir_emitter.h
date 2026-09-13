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

#include <cxx/codegen/emitter.h>
#include <cxx/mlir/cxx_dialect.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <memory>
#include <vector>

namespace cxx {
class TranslationUnit;
}

namespace cxx::ir {

class MlirDebugEmitter;

class MlirEmitter final : public Emitter {
 public:
  MlirEmitter(mlir::MLIRContext& context, TranslationUnit* unit);
  ~MlirEmitter() override;
  auto debug() -> DebugEmitter* override;

  auto saveInsertionPoint() -> InsertionPointRef override;
  void restoreInsertionPoint(InsertionPointRef point) override;
  void setInsertionPoint(InsertionPoint point) override;
  void beginGlobalInitializer(GlobalRef global) override;
  void globalConstructor(SourceLocation loc, FunctionRef function) override;
  auto constant(SourceLocation loc, TypeRef type, const Initializer& value)
      -> ValueRef override;
  auto isZeroConstant(ValueRef value) -> bool override;
  auto symbolExists(std::string_view name) -> bool override;
  auto initializerAttribute(const Initializer& value) -> mlir::Attribute;
  auto todo(SourceLocation loc, TodoKind kind, std::string_view message)
      -> ValueRef override;
  auto unaryOp(SourceLocation loc, UnaryOp op, TypeRef type, ValueRef value)
      -> ValueRef override;
  void defineVTable(SourceLocation loc, const VTableInfo& info) override;
  [[nodiscard]] auto beginModule(const ModuleInfo& info) -> ModuleRef override;
  void endModule() override;
  // Native artifact access for the MLIR drivers; handles do not own modules.
  [[nodiscard]] auto module(ModuleRef ref) -> mlir::ModuleOp;

  [[nodiscard]] auto beginCleanupRegion() -> CleanupRegionRef override;
  void endCleanupRegion(CleanupRegionRef region) override;
  [[nodiscard]] auto activateConditionalCleanup(ValueRef address,
                                                BlockRef entry,
                                                CleanupRegionRef region)
      -> ValueRef override;

  void switchBranch(SourceLocation loc, ValueRef flag, BlockRef defaultDest,
                    std::span<const std::int64_t> caseValues,
                    std::span<const BlockRef> caseDestinations) override;

  void defineLabel(SourceLocation loc, std::string_view name,
                   std::int64_t cleanupDepth) override;
  void branchWithCleanups(SourceLocation loc, CleanupTarget target,
                          std::span<const CleanupAction> cleanups) override;
  void indirectGoto(SourceLocation loc, ValueRef target) override;
  auto labelAddress(SourceLocation loc, TypeRef type, std::string_view name,
                    FunctionRef function) -> ValueRef override;
  void resolveFunctionControlFlow(FunctionRef function) override;

  [[nodiscard]] auto functionParameterTypes(FunctionRef function)
      -> std::vector<TypeRef> override;
  [[nodiscard]] auto functionResultTypes(FunctionRef function)
      -> std::vector<TypeRef> override;
  auto addBlockParameter(BlockRef block, TypeRef type, SourceLocation loc)
      -> ValueRef override;
  [[nodiscard]] auto blockParameter(BlockRef block, unsigned index)
      -> ValueRef override;
  [[nodiscard]] auto blockParameterCount(BlockRef block) -> unsigned override;

  [[nodiscard]] auto createBlock(FunctionRef function) -> BlockRef override;

  [[nodiscard]] auto createBlock(mlir::Region& region) -> BlockRef;

  [[nodiscard]] auto insertionBlock() -> BlockRef override;

  void eraseBlock(BlockRef block) override;

  [[nodiscard]] auto hasTerminator(BlockRef block) -> bool override;

  void branch(SourceLocation loc, BlockRef target,
              std::span<const ValueRef> operands) override;
  auto globalLinkage(GlobalRef global) -> Linkage override;

  void condBranch(SourceLocation loc, ValueRef condition, BlockRef trueDest,
                  BlockRef falseDest) override;

  [[nodiscard]] auto binaryOp(SourceLocation loc, BinaryOp op, ValueRef lhs,
                              ValueRef rhs) -> ValueRef override;

  [[nodiscard]] auto compareInt(SourceLocation loc, IntPredicate predicate,
                                ValueRef lhs, ValueRef rhs)
      -> ValueRef override;

  [[nodiscard]] auto compareFloat(SourceLocation loc, FloatPredicate predicate,
                                  ValueRef lhs, ValueRef rhs)
      -> ValueRef override;

  [[nodiscard]] auto select(SourceLocation loc, ValueRef condition,
                            ValueRef ifTrue, ValueRef ifFalse)
      -> ValueRef override;

  [[nodiscard]] auto voidType() -> TypeRef override;

  [[nodiscard]] auto unresolvedType() -> TypeRef override;

  [[nodiscard]] auto integerType(unsigned bits) -> TypeRef override;

  [[nodiscard]] auto floatingType(FloatKind kind) -> TypeRef override;

  [[nodiscard]] auto pointerType(TypeRef elementType) -> TypeRef override;

  [[nodiscard]] auto arrayType(TypeRef elementType, std::uint64_t size)
      -> TypeRef override;

  [[nodiscard]] auto vectorType(TypeRef elementType, std::uint64_t elementCount)
      -> TypeRef override;

  [[nodiscard]] auto vectorSplat(SourceLocation loc, TypeRef vectorType,
                                 ValueRef scalar) -> ValueRef override;

  [[nodiscard]] auto functionType(std::span<const TypeRef> parameters,
                                  std::span<const TypeRef> results,
                                  bool isVariadic) -> TypeRef override;

  [[nodiscard]] auto declareClassType(std::string_view name)
      -> TypeRef override;

  void defineClassType(TypeRef classType, std::span<const TypeRef> members,
                       bool isPacked) override;

  [[nodiscard]] auto typeKind(TypeRef type) -> TypeKind override;

  [[nodiscard]] auto scalarWidth(TypeRef type) -> unsigned override;

  [[nodiscard]] auto elementType(TypeRef type) -> TypeRef override;

  [[nodiscard]] auto typeOf(ValueRef value) -> TypeRef override;

  [[nodiscard]] auto allocate(SourceLocation loc, TypeRef pointerType,
                              ValueRef size, std::uint64_t alignment)
      -> ValueRef override;

  [[nodiscard]] auto enclosingFunctionEntryBlock() -> mlir::Block*;

  [[nodiscard]] auto allocate(mlir::Location loc, TypeRef pointerType,
                              std::uint64_t alignment) -> ValueRef;

  [[nodiscard]] auto dynamicAllocate(mlir::Location loc, TypeRef pointerType,
                                     ValueRef size, std::uint64_t alignment)
      -> ValueRef;

  [[nodiscard]] auto load(SourceLocation loc, TypeRef valueType,
                          ValueRef address, const Access& access)
      -> ValueRef override;

  [[nodiscard]] auto load(mlir::Location loc, TypeRef valueType,
                          ValueRef address, std::uint64_t alignment)
      -> ValueRef;

  [[nodiscard]] auto pointerAdd(SourceLocation loc, TypeRef pointerType,
                                ValueRef base, ValueRef offset)
      -> ValueRef override;

  [[nodiscard]] auto pointerAdd(mlir::Location loc, TypeRef pointerType,
                                ValueRef base, ValueRef offset) -> ValueRef;

  [[nodiscard]] auto pointerDiff(SourceLocation loc, TypeRef resultType,
                                 ValueRef lhs, ValueRef rhs)
      -> ValueRef override;

  [[nodiscard]] auto pointerDiff(mlir::Location loc, TypeRef resultType,
                                 ValueRef lhs, ValueRef rhs) -> ValueRef;

  [[nodiscard]] auto subscript(SourceLocation loc, TypeRef pointerType,
                               ValueRef base, ValueRef index)
      -> ValueRef override;

  [[nodiscard]] auto subscript(mlir::Location loc, TypeRef pointerType,
                               ValueRef base, ValueRef index) -> ValueRef;

  [[nodiscard]] auto memberAddress(SourceLocation loc, TypeRef pointerType,
                                   ValueRef base, std::uint32_t index)
      -> ValueRef override;

  [[nodiscard]] auto memberAddress(mlir::Location loc, TypeRef pointerType,
                                   ValueRef base, std::uint32_t index)
      -> ValueRef;

  [[nodiscard]] auto extractValue(SourceLocation loc, TypeRef resultType,
                                  ValueRef container, std::int64_t position)
      -> ValueRef override;

  [[nodiscard]] auto extractValue(mlir::Location loc, TypeRef resultType,
                                  ValueRef container, std::int64_t position)
      -> ValueRef;

  [[nodiscard]] auto insertValue(SourceLocation loc, TypeRef resultType,
                                 ValueRef container, ValueRef value,
                                 std::int64_t position) -> ValueRef override;

  [[nodiscard]] auto insertValue(mlir::Location loc, TypeRef resultType,
                                 ValueRef container, ValueRef value,
                                 std::int64_t position) -> ValueRef;

  [[nodiscard]] auto addressOfSymbol(SourceLocation loc, TypeRef resultType,
                                     std::string_view symbol)
      -> ValueRef override;

  [[nodiscard]] auto addressOfSymbol(mlir::Location loc, TypeRef resultType,
                                     std::string_view symbol) -> ValueRef;

  void store(SourceLocation loc, ValueRef value, ValueRef address,
             const Access& access) override;

  void store(mlir::Location loc, ValueRef value, ValueRef address,
             std::uint64_t alignment);

  void memsetZero(SourceLocation loc, ValueRef address,
                  std::uint64_t size) override;

  void memsetZero(mlir::Location loc, ValueRef address, std::uint64_t size);

  void memcpy(SourceLocation loc, ValueRef destination, ValueRef source,
              std::uint64_t size) override;

  void memcpy(mlir::Location loc, ValueRef destination, ValueRef source,
              std::uint64_t size);

  void unreachable(SourceLocation loc) override;

  void unreachable(mlir::Location loc);

  [[nodiscard]] auto call(SourceLocation loc, const CallInfo& info)
      -> std::vector<ValueRef> override;

  [[nodiscard]] auto call(mlir::Location loc, const CallInfo& info)
      -> std::vector<ValueRef>;

  void ret(SourceLocation loc, std::span<const ValueRef> values) override;

  void ret(mlir::Location loc, std::span<const ValueRef> values);

  [[nodiscard]] auto constantInt(mlir::Location loc, TypeRef type,
                                 std::int64_t value) -> ValueRef;

  [[nodiscard]] auto findFunction(std::string_view name)
      -> FunctionRef override;

  [[nodiscard]] auto parameterAbiAttrs(std::span<const ParameterAbi> parameters,
                                       std::size_t count) -> mlir::ArrayAttr;

  [[nodiscard]] auto declareFunction(SourceLocation loc,
                                     const FunctionInfo& info)
      -> FunctionRef override;

  [[nodiscard]] auto declareFunction(mlir::Location loc,
                                     const FunctionInfo& info) -> FunctionRef;

  [[nodiscard]] auto functionName(FunctionRef function) -> std::string_view;

  [[nodiscard]] auto functionLinkage(FunctionRef function) -> Linkage;

  [[nodiscard]] auto functionHasBody(FunctionRef function) -> bool override;

  [[nodiscard]] auto findGlobal(std::string_view name) -> GlobalRef override;

  [[nodiscard]] auto declareGlobal(SourceLocation loc, const GlobalInfo& info)
      -> GlobalRef override;

  [[nodiscard]] auto wrap(mlir::cxx::FuncOp function) -> FunctionRef;

  [[nodiscard]] auto function(FunctionRef ref) -> mlir::cxx::FuncOp;

  [[nodiscard]] auto wrap(mlir::cxx::GlobalOp global) -> GlobalRef;

  [[nodiscard]] auto global(GlobalRef ref) -> mlir::cxx::GlobalOp;

  [[nodiscard]] auto module() -> mlir::ModuleOp { return module_; }

  [[nodiscard]] auto convert(SourceLocation loc, CastKind kind, ValueRef value,
                             TypeRef type) -> ValueRef override;

  [[nodiscard]] auto builder() -> mlir::OpBuilder& { return builder_; }

  [[nodiscard]] auto context() -> mlir::MLIRContext* {
    return builder_.getContext();
  }

  [[nodiscard]] auto wrap(mlir::Block* block) -> BlockRef;

  [[nodiscard]] auto block(BlockRef ref) -> mlir::Block*;

  [[nodiscard]] auto wrap(mlir::Value value) -> ValueRef;

  [[nodiscard]] auto value(ValueRef ref) -> mlir::Value;

  [[nodiscard]] auto values(llvm::ArrayRef<ValueRef> refs)
      -> llvm::SmallVector<mlir::Value>;

  [[nodiscard]] auto wrap(mlir::Type type) -> TypeRef;

  [[nodiscard]] auto type(TypeRef ref) -> mlir::Type;

  [[nodiscard]] auto types(llvm::ArrayRef<TypeRef> refs)
      -> llvm::SmallVector<mlir::Type>;

  [[nodiscard]] auto getLocation(SourceLocation loc) -> mlir::Location;

  void beginFunctionBody(FunctionRef function) override;

  void endFunctionBody(FunctionRef function) override;

  [[nodiscard]] auto blockArgumentCount(BlockRef block) -> unsigned;

  [[nodiscard]] auto blockArgument(BlockRef block, unsigned index)
      -> mlir::Value;

  [[nodiscard]] auto addBlockArgument(BlockRef block, mlir::Type type,
                                      mlir::Location loc) -> mlir::Value;

  void branch(mlir::Location loc, BlockRef target,
              mlir::ValueRange operands = {});

  void condBranch(mlir::Location loc, mlir::Value condition, BlockRef trueDest,
                  BlockRef falseDest);

  void switchBranch(mlir::Location loc, mlir::Value flag,
                    mlir::IntegerType flagType, BlockRef defaultDest,
                    llvm::ArrayRef<std::int64_t> caseValues,
                    llvm::ArrayRef<BlockRef> caseDestinations);

 private:
  struct CleanupRegion {
    mlir::Block* startBlock = nullptr;
    mlir::Operation* startAnchor = nullptr;
  };
  llvm::DenseMap<std::uint32_t, CleanupRegion> cleanupRegions_;
  std::uint32_t nextCleanupRegionId_ = 1;

  std::vector<mlir::OpBuilder::InsertPoint> insertionPoints_{{}};
  std::vector<std::uint32_t> freeInsertionPoints_;
  std::unique_ptr<MlirDebugEmitter> debug_;
  TranslationUnit* unit_;
  mlir::OpBuilder builder_;
  struct FunctionBodyScope {
    FunctionRef function;
    std::uint32_t blockMark = 0;
    std::uint32_t valueMark = 0;
  };

  std::vector<FunctionBodyScope> functionBodies_;
  std::vector<mlir::Block*> blocks_{nullptr};
  std::vector<std::uint8_t> blockGenerations_{0};
  llvm::DenseMap<mlir::Block*, std::uint32_t> blockIds_;
  std::vector<mlir::Value> values_{mlir::Value{}};
  std::vector<std::uint8_t> valueGenerations_{0};
  llvm::DenseMap<mlir::Value, std::uint32_t> valueIds_;
  std::vector<mlir::Type> types_{mlir::Type{}};
  llvm::DenseMap<mlir::Type, std::uint32_t> typeIds_;
  std::vector<mlir::cxx::FuncOp> functions_{mlir::cxx::FuncOp{}};
  llvm::DenseMap<mlir::Operation*, std::uint32_t> functionIds_;
  std::vector<mlir::cxx::GlobalOp> globals_{mlir::cxx::GlobalOp{}};
  llvm::DenseMap<mlir::Operation*, std::uint32_t> globalIds_;
  std::vector<mlir::ModuleOp> modules_{mlir::ModuleOp{}};
  std::vector<mlir::ModuleOp> moduleStack_;
  mlir::ModuleOp module_;
};

}  // namespace cxx::ir

namespace llvm {

template <typename To, typename Tag>
struct CastInfo<To, cxx::ir::Handle<Tag>> {
  static_assert(sizeof(To) == 0,
                "an opaque codegen handle carries no backend type: unwrap it "
                "before casting");
};

template <typename To, typename Tag>
struct CastInfo<To, const cxx::ir::Handle<Tag>>
    : CastInfo<To, cxx::ir::Handle<Tag>> {};

}  // namespace llvm
