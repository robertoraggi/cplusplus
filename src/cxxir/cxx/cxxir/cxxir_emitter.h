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

#include <format>
#include <memory>

namespace cxx {
class TranslationUnit;
}

namespace cxx::ir {

class CxxIREmitter final : public Emitter {
 public:
  explicit CxxIREmitter(TranslationUnit* unit = nullptr);
  ~CxxIREmitter() override;

  CxxIREmitter(const CxxIREmitter&) = delete;
  auto operator=(const CxxIREmitter&) -> CxxIREmitter& = delete;

  struct ModuleView {
    const CxxIREmitter& emitter;
    ModuleRef ref;
    bool printLocations = false;
  };

  [[nodiscard]] auto module(ModuleRef ref, bool printLocations = false) const
      -> ModuleView;

  using Emitter::allocate;
  using Emitter::branchWithCleanups;
  using Emitter::functionType;
  using Emitter::load;
  using Emitter::store;

  void moduleAsm(SourceLocation loc, std::string_view assembly) override;
  void functionPersonality(FunctionRef function,
                           std::string_view personality) override;
  auto landingPad(SourceLocation loc, std::span<const ValueRef> clauses,
                  bool cleanup) -> std::vector<ValueRef> override;
  auto exceptionTypeId(SourceLocation loc, ValueRef typeInfo)
      -> ValueRef override;
  void resumeException(SourceLocation loc, ValueRef exception,
                       ValueRef selector) override;
  auto inlineAsm(SourceLocation loc, const InlineAsmInfo& info)
      -> std::vector<ValueRef> override;
  auto inlineAsmGoto(SourceLocation loc, const InlineAsmInfo& info,
                     BlockRef fallthrough,
                     std::span<const BlockRef> destinations)
      -> ValueRef override;
  [[nodiscard]] auto debug() -> DebugEmitter* override;
  auto saveInsertionPoint() -> InsertionPointRef override;
  void restoreInsertionPoint(InsertionPointRef point) override;
  void setInsertionPoint(InsertionPoint point) override;
  void beginGlobalInitializer(GlobalRef global) override;
  void globalConstructor(SourceLocation loc, FunctionRef function,
                         std::uint32_t priority = 65535) override;
  auto constant(SourceLocation loc, TypeRef type, const Initializer& value)
      -> ValueRef override;
  [[nodiscard]] auto isZeroConstant(ValueRef value) -> bool override;
  [[nodiscard]] auto symbolExists(std::string_view name) -> bool override;
  auto todo(SourceLocation loc, TodoKind kind, std::string_view message)
      -> ValueRef override;
  auto unaryOp(SourceLocation loc, UnaryOp op, TypeRef type, ValueRef value)
      -> ValueRef override;
  void defineVTable(SourceLocation loc, const VTableInfo& info) override;
  auto beginModule(const ModuleInfo& info) -> ModuleRef override;
  void endModule() override;
  void beginFunctionBody(FunctionRef function) override;
  void endFunctionBody(FunctionRef function) override;
  auto beginCleanupRegion() -> CleanupRegionRef override;
  void endCleanupRegion(CleanupRegionRef region) override;
  auto activateConditionalCleanup(ValueRef address, BlockRef entry,
                                  CleanupRegionRef region) -> ValueRef override;
  void switchBranch(SourceLocation loc, ValueRef flag, BlockRef defaultDest,
                    std::span<const ConstInt> caseValues,
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
  auto createBlock(FunctionRef function) -> BlockRef override;
  [[nodiscard]] auto insertionBlock() -> BlockRef override;
  void eraseBlock(BlockRef block) override;
  [[nodiscard]] auto hasTerminator(BlockRef block) -> bool override;
  void branch(SourceLocation loc, BlockRef target,
              std::span<const ValueRef> operands = {}) override;
  [[nodiscard]] auto globalLinkage(GlobalRef global) -> Linkage override;
  void condBranch(SourceLocation loc, ValueRef condition, BlockRef trueDest,
                  BlockRef falseDest) override;
  auto binaryOp(SourceLocation loc, BinaryOp op, ValueRef lhs, ValueRef rhs)
      -> ValueRef override;
  auto compareInt(SourceLocation loc, IntPredicate predicate, ValueRef lhs,
                  ValueRef rhs) -> ValueRef override;
  auto compareFloat(SourceLocation loc, FloatPredicate predicate, ValueRef lhs,
                    ValueRef rhs) -> ValueRef override;
  auto select(SourceLocation loc, ValueRef condition, ValueRef ifTrue,
              ValueRef ifFalse) -> ValueRef override;
  auto voidType() -> TypeRef override;
  auto unresolvedType() -> TypeRef override;
  auto integerType(unsigned bits) -> TypeRef override;
  auto floatingType(FloatKind kind) -> TypeRef override;
  auto pointerType(TypeRef elementType,
                   std::optional<std::uint32_t> addressSpace = std::nullopt)
      -> TypeRef override;
  auto rebindPointerType(TypeRef pointerType, TypeRef elementType)
      -> TypeRef override;
  auto arrayType(TypeRef elementType, std::uint64_t size) -> TypeRef override;
  auto vectorType(TypeRef elementType, std::uint64_t elementCount)
      -> TypeRef override;
  auto vectorSplat(SourceLocation loc, TypeRef vectorType, ValueRef scalar)
      -> ValueRef override;
  auto vectorFromElements(SourceLocation loc, TypeRef vectorType,
                          std::span<const ValueRef> elements)
      -> ValueRef override;
  auto vectorExtract(SourceLocation loc, TypeRef elementType, ValueRef vector,
                     ValueRef index, bool indexIsSigned) -> ValueRef override;
  auto vectorInsert(SourceLocation loc, ValueRef vector, ValueRef element,
                    ValueRef index, bool indexIsSigned) -> ValueRef override;
  auto vectorShuffle(SourceLocation loc, ValueRef first, ValueRef second,
                     std::span<const std::int64_t> mask) -> ValueRef override;
  auto functionType(std::span<const TypeRef> parameters,
                    std::span<const TypeRef> results, bool isVariadic)
      -> TypeRef override;
  auto declareClassType(std::string_view name) -> TypeRef override;
  void defineClassType(TypeRef classType, std::span<const TypeRef> members,
                       bool isPacked, std::uint64_t alignment,
                       std::span<const std::uint32_t> padding) override;
  auto convert(SourceLocation loc, CastKind kind, ValueRef value, TypeRef type)
      -> ValueRef override;
  [[nodiscard]] auto typeKind(TypeRef type) -> TypeKind override;
  [[nodiscard]] auto scalarWidth(TypeRef type) -> unsigned override;
  [[nodiscard]] auto elementType(TypeRef type) -> TypeRef override;
  [[nodiscard]] auto typeOf(ValueRef value) -> TypeRef override;
  auto allocate(SourceLocation loc, TypeRef pointerType, ValueRef size,
                std::uint64_t alignment) -> ValueRef override;
  auto load(SourceLocation loc, TypeRef valueType, ValueRef address,
            const Access& access) -> ValueRef override;
  auto allocateWorkgroup(SourceLocation loc, TypeRef pointerType,
                         std::uint64_t alignment) -> ValueRef override;
  auto atomicLoad(SourceLocation loc, TypeRef valueType, ValueRef address,
                  const AtomicAccess& access) -> ValueRef override;
  void atomicStore(SourceLocation loc, ValueRef value, ValueRef address,
                   const AtomicAccess& access) override;
  auto atomicReadModifyWrite(SourceLocation loc, AtomicOperation operation,
                             ValueRef value, ValueRef address,
                             const AtomicAccess& access) -> ValueRef override;
  auto atomicCompareExchange(SourceLocation loc, ValueRef address,
                             ValueRef expected, ValueRef desired,
                             const CompareExchangeAccess& access)
      -> std::vector<ValueRef> override;
  void memoryBarrier(SourceLocation loc,
                     const MemoryBarrierInfo& barrier) override;
  void controlBarrier(SourceLocation loc, MemoryScope executionScope,
                      const MemoryBarrierInfo& barrier) override;
  auto pointerAdd(SourceLocation loc, TypeRef pointerType, ValueRef base,
                  ValueRef offset) -> ValueRef override;
  auto pointerDiff(SourceLocation loc, TypeRef resultType, ValueRef lhs,
                   ValueRef rhs) -> ValueRef override;
  auto subscript(SourceLocation loc, TypeRef pointerType, ValueRef base,
                 ValueRef index) -> ValueRef override;
  auto memberAddress(SourceLocation loc, TypeRef pointerType, ValueRef base,
                     std::uint32_t index) -> ValueRef override;
  auto extractValue(SourceLocation loc, TypeRef resultType, ValueRef container,
                    std::int64_t position) -> ValueRef override;
  auto insertValue(SourceLocation loc, TypeRef resultType, ValueRef container,
                   ValueRef value, std::int64_t position) -> ValueRef override;
  auto addressOfSymbol(SourceLocation loc, TypeRef resultType,
                       std::string_view symbol) -> ValueRef override;
  void store(SourceLocation loc, ValueRef value, ValueRef address,
             const Access& access) override;
  void memsetZero(SourceLocation loc, ValueRef address,
                  std::uint64_t size) override;
  void memcpy(SourceLocation loc, ValueRef destination, ValueRef source,
              std::uint64_t size) override;
  void zeroInitialize(SourceLocation loc, ValueRef address,
                      TypeRef objectType) override;
  void copyObject(SourceLocation loc, ValueRef destination, ValueRef source,
                  TypeRef objectType) override;
  void unreachable(SourceLocation loc) override;
  auto call(SourceLocation loc, const CallInfo& info)
      -> std::vector<ValueRef> override;
  void ret(SourceLocation loc, std::span<const ValueRef> values) override;
  [[nodiscard]] auto findFunction(std::string_view name)
      -> FunctionRef override;
  auto declareFunction(SourceLocation loc, const FunctionInfo& info)
      -> FunctionRef override;
  [[nodiscard]] auto functionHasBody(FunctionRef function) -> bool override;
  void setFunctionAliasee(FunctionRef function,
                          std::string_view aliasee) override;
  [[nodiscard]] auto findGlobal(std::string_view name) -> GlobalRef override;
  auto declareGlobal(SourceLocation loc, const GlobalInfo& info)
      -> GlobalRef override;

 private:
  friend struct std::formatter<ModuleView>;

  auto formatTo(ModuleRef module, std::format_context::iterator out,
                bool printLocations) const -> std::format_context::iterator;

  struct Private;
  std::unique_ptr<Private> priv_;
};

}  // namespace cxx::ir

template <>
struct std::formatter<cxx::ir::CxxIREmitter::ModuleView> {
  [[nodiscard]] constexpr auto parse(std::format_parse_context& context) {
    return context.begin();
  }

  auto format(const cxx::ir::CxxIREmitter::ModuleView& module,
              std::format_context& context) const {
    return module.emitter.formatTo(module.ref, context.out(),
                                   module.printLocations);
  }
};
