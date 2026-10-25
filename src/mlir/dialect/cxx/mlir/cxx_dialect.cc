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

#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrAttrs.h>
#include <mlir/Dialect/Ptr/IR/PtrDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <functional>
#include <numeric>

namespace mlir::cxx {
template <typename Enum>
struct detail::EnumAttrStorage : public AttributeStorage {
  using KeyTy = Enum;

  explicit EnumAttrStorage(Enum value) : value(value) {}

  [[nodiscard]] auto operator==(KeyTy key) const -> bool {
    return value == key;
  }

  [[nodiscard]] static auto hashKey(KeyTy key) -> llvm::hash_code {
    return llvm::hash_value(key);
  }

  static auto construct(AttributeStorageAllocator& allocator, KeyTy key)
      -> EnumAttrStorage* {
    return new (allocator.allocate<EnumAttrStorage>()) EnumAttrStorage(key);
  }

  Enum value;
};

struct detail::ClassTypeStorage : public TypeStorage {
 public:
  using KeyTy = StringRef;

  explicit ClassTypeStorage(const KeyTy& key) : name_(key) {}

  auto getName() -> StringRef const { return name_; }
  auto getBody() const -> ArrayRef<Type> { return body_; }
  auto isPacked() const -> bool { return packed_; }
  [[nodiscard]] auto getAlignment() const -> uint64_t { return alignment_; }
  [[nodiscard]] auto getPadding() const -> ArrayRef<unsigned> {
    return padding_;
  }

  auto operator==(const KeyTy& key) const -> bool { return name_ == key; };

  static auto hashKey(const KeyTy& key) -> llvm::hash_code {
    return llvm::hash_value(key);
  }

  static ClassTypeStorage* construct(TypeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ClassTypeStorage>())
        ClassTypeStorage(allocator.copyInto(key));
  }

  auto mutate(TypeStorageAllocator& allocator, ArrayRef<Type> body, bool packed,
              uint64_t alignment, ArrayRef<unsigned> padding) -> LogicalResult {
    if (isInitialized_)
      return success(isDefinedAs(body, packed, alignment, padding));

    isInitialized_ = true;
    body_ = allocator.copyInto(body);
    packed_ = packed;
    alignment_ = alignment;
    padding_ = allocator.copyInto(padding);

    return success();
  }

 private:
  [[nodiscard]] auto isDefinedAs(ArrayRef<Type> body, bool packed,
                                 uint64_t alignment,
                                 ArrayRef<unsigned> padding) const -> bool {
    if (body != getBody() || padding != getPadding()) return false;
    return packed == packed_ && alignment == alignment_;
  }

  StringRef name_;
  ArrayRef<Type> body_;
  ArrayRef<unsigned> padding_;
  bool packed_ = false;
  uint64_t alignment_ = 0;
  bool isInitialized_ = false;
};

namespace {
[[nodiscard]] auto isValidPointerMemorySpace(Attribute memorySpace) -> bool {
  if (!memorySpace) return true;
  if (auto integer = dyn_cast<IntegerAttr>(memorySpace)) {
    auto type = dyn_cast<IntegerType>(integer.getType());
    if (!type) return false;
    if (!type.isUnsigned() && integer.getValue().isNegative()) return false;
    return integer.getValue().getActiveBits() <= 32;
  }
  return isa<ptr::MemorySpaceAttrInterface>(memorySpace);
}

[[nodiscard]] auto canonicalPointerMemorySpace(Attribute memorySpace)
    -> Attribute {
  if (!memorySpace) return {};
  if (isa<ptr::GenericSpaceAttr>(memorySpace)) return {};
  if (auto space = dyn_cast<LLVM::LLVMAddrSpaceAttrInterface>(memorySpace)) {
    auto value = space.getAddressSpace();
    if (!value) return {};
    return IntegerAttr::get(IntegerType::get(memorySpace.getContext(), 64),
                            value);
  }
  if (auto integer = dyn_cast<IntegerAttr>(memorySpace)) {
    if (integer.getValue().isZero()) return {};
    return IntegerAttr::get(IntegerType::get(memorySpace.getContext(), 64),
                            integer.getValue().getZExtValue());
  }
  return memorySpace;
}

[[nodiscard]] auto verifyPointerMemorySpaces(Operation* op, Type input,
                                             Type result) -> LogicalResult {
  auto inputPointer = dyn_cast<PtrLikeTypeInterface>(input);
  auto resultPointer = dyn_cast<PtrLikeTypeInterface>(result);
  if (!inputPointer || !resultPointer) return success();
  if (haveSamePointerMemorySpace(inputPointer, resultPointer)) return success();
  return op->emitOpError("requires matching pointer memory spaces");
}

class TypeGraphPrinter {
 public:
  explicit TypeGraphPrinter(AsmPrinter& printer) : printer_(printer) {}

  void printBody(ClassType type) {
    auto cyclicPrint = printer_.tryStartCyclicPrint(type);
    printer_ << '<';
    printer_.printString(type.getName());
    if (failed(cyclicPrint) || !classes_.insert(type).second) {
      printer_ << '>';
      return;
    }
    printer_ << ", ";
    if (type.isPacked()) printer_ << "packed ";
    if (auto alignment = type.getAlignment())
      printer_ << "align " << alignment << ' ';
    printTypeList(type.getBody(), type.getPadding());
    printer_ << '>';
  }

  void printBody(PointerType type) {
    printer_ << '<';
    printType(type.getElementType());
    if (auto memorySpace = type.getAddressSpace()) {
      printer_ << ", ";
      if (auto integer = dyn_cast<IntegerAttr>(memorySpace);
          integer && integer.getType().isSignlessInteger(64))
        printer_.printAttributeWithoutType(memorySpace);
      else
        printer_ << memorySpace;
    }
    printer_ << '>';
  }

  void printBody(ArrayType type) {
    printer_ << '<';
    printType(type.getElementType());
    printer_ << ", " << type.getSize() << '>';
  }

  void printBody(FunctionType type) {
    printer_ << '<';
    printTypeList(type.getInputs());
    printer_ << ", ";
    printTypeList(type.getResults());
    printer_ << ", " << (type.getVariadic() ? "true" : "false") << '>';
  }

 private:
  void printTypeList(ArrayRef<Type> types, ArrayRef<unsigned> padding = {}) {
    printer_ << '(';
    llvm::interleaveComma(llvm::enumerate(types), printer_.getStream(),
                          [&](auto element) {
                            if (llvm::is_contained(padding, element.index()))
                              printer_ << "padding ";
                            printType(element.value());
                          });
    printer_ << ')';
  }

  void printType(Type type) {
    if (succeeded(printer_.printAlias(type))) return;
    llvm::TypeSwitch<Type>(type)
        .Case<ClassType>([this](ClassType type) {
          printer_ << "!cxx.class";
          printBody(type);
        })
        .Case<PointerType>([this](PointerType type) {
          printer_ << "!cxx.ptr";
          printBody(type);
        })
        .Case<ArrayType>([this](ArrayType type) {
          printer_ << "!cxx.array";
          printBody(type);
        })
        .Case<FunctionType>([this](FunctionType type) {
          printer_ << "!cxx.function";
          printBody(type);
        })
        .Default([this](Type type) { printer_.printType(type); });
  }

  AsmPrinter& printer_;
  llvm::DenseSet<ClassType> classes_;
};

auto parseTypeList(AsmParser& parser, SmallVectorImpl<Type>& types,
                   SmallVectorImpl<unsigned>* padding = nullptr)
    -> ParseResult {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
    if (padding && succeeded(parser.parseOptionalKeyword("padding")))
      padding->push_back(types.size());
    Type type;
    if (parser.parseType(type)) return failure();
    types.push_back(type);
    return success();
  });
}

void parseLinkage(OpAsmParser& parser, OperationState& result) {
  StringRef keyword;
  if (failed(parser.parseOptionalKeyword(
          &keyword,
          {"external", "internal", "linkonce_odr", "weak_odr",
           "available_externally", "appending", "weak", "extern_weak"}))) {
    return;
  }
  result.addAttribute("linkage_kind",
                      LinkageKindAttr::get(parser.getContext(),
                                           *symbolizeLinkageKind(keyword)));
}

struct CxxGenerateAliases : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  auto getAlias(Type type, raw_ostream& os) const -> AliasResult override {
    if (auto classType = dyn_cast<ClassType>(type)) {
      if (!classType.getBody().empty()) {
        os << "class_" << classType.getName();
        return AliasResult::FinalAlias;
      }
    }

    if (auto ptrType = dyn_cast<PointerType>(type)) {
      os << "ptr_";
      return AliasResult::FinalAlias;
    }

    if (auto arrayType = dyn_cast<ArrayType>(type)) {
      os << "array_";
      return AliasResult::FinalAlias;
    }

    if (isa<VoidType>(type)) {
      os << "void";
      return AliasResult::FinalAlias;
    }

    return AliasResult::NoAlias;
  }
};
}  // namespace

auto haveSamePointerMemorySpace(PtrLikeTypeInterface left,
                                PtrLikeTypeInterface right) -> bool {
  return canonicalPointerMemorySpace(left.getMemorySpace()) ==
         canonicalPointerMemorySpace(right.getMemorySpace());
}

auto isVolatileAccess(Operation* op) -> bool {
  return llvm::TypeSwitch<Operation*, bool>(op)
      .Case<LoadOp, StoreOp, BitfieldLoadOp, BitfieldStoreOp, AtomicLoadOp,
            AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(
          [](auto access) { return bool(access.getVolatile_()); })
      .Default(false);
}

auto getLoadStoreAddress(Operation* op) -> Value {
  return llvm::TypeSwitch<Operation*, Value>(op)
      .Case<LoadOp, StoreOp>([](auto access) { return access.getAddr(); })
      .Default(Value());
}

auto canBitcastStorageValue(Type from, Type to) -> bool {
  if (from == to) return true;
  auto fromPointer = dyn_cast<PointerType>(from);
  auto toPointer = dyn_cast<PointerType>(to);
  if (fromPointer || toPointer)
    return fromPointer && toPointer &&
           haveSamePointerMemorySpace(fromPointer, toPointer);
  auto fromVector = dyn_cast<VectorType>(from);
  auto toVector = dyn_cast<VectorType>(to);
  if (bool(fromVector) != bool(toVector)) return false;
  if (fromVector) {
    if (fromVector.getShape() != toVector.getShape()) return false;
    from = fromVector.getElementType();
    to = toVector.getElementType();
  }
  return from.isIntOrFloat() && to.isIntOrFloat() &&
         from.getIntOrFloatBitWidth() == to.getIntOrFloatBitWidth();
}

auto getAggregateElementType(Type container, std::int64_t position) -> Type {
  if (position < 0) return {};
  if (isa<ClassType, TupleType>(container)) {
    auto fields = getRecordElementTypes(container);
    return std::uint64_t(position) < fields.size() ? fields[position] : Type();
  }
  if (auto array = dyn_cast<ArrayType>(container))
    return std::uint64_t(position) < array.getSize() ? array.getElementType()
                                                     : Type();
  return {};
}

auto getRecordElementTypes(Type record) -> ArrayRef<Type> {
  if (auto type = dyn_cast<ClassType>(record)) return type.getBody();
  if (auto type = dyn_cast<TupleType>(record)) return type.getTypes();
  return {};
}

auto isAggregateType(Type type) -> bool {
  return isa<ClassType, TupleType, ArrayType>(type);
}

namespace {

struct CxxInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  [[nodiscard]] auto isLegalToInline(Region*, Region*, bool, IRMapping&) const
      -> bool final {
    return true;
  }

  [[nodiscard]] auto isLegalToInline(Operation* op, Region*, bool,
                                     IRMapping&) const -> bool final {
    return !isa<AllocaOp, DynAllocaOp, LabelOp, GotoOp, LabelAddressOp,
                IndirectGotoOp, AsmGotoOp, LandingPadOp, ResumeOp>(op);
  }
};

}  // namespace

void CxxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <cxx/mlir/CxxOps.cpp.inc>
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include <cxx/mlir/CxxOpsTypes.cpp.inc>
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include <cxx/mlir/CxxOpsAttributes.cpp.inc>
      >();

  addInterface<CxxGenerateAliases>();
  addInterface<CxxInlinerInterface>();
}

void FuncOp::print(OpAsmPrinter& p) {
  const auto isVariadic = getFunctionType().getVariadic();

  p << ' ';

  if (auto linkage = getLinkageKind()) {
    if (*linkage != LinkageKind::External) {
      p << stringifyLinkageKind(*linkage) << ' ';
    }
  }

  if (auto inlineKind = getInlineKind()) {
    if (*inlineKind != InlineKind::NoInline) {
      p << stringifyInlineKind(*inlineKind) << ' ';
    }
  }

  p.printSymbolName(getSymName());

  ArrayRef<Type> argTypes = getArgumentTypes();
  ArrayRef<Type> resultTypes = getResultTypes();
  function_interface_impl::printFunctionSignature(p, *this, argTypes,
                                                  isVariadic, resultTypes);

  SmallVector<StringRef> elided = {
      getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName()};
  if (getLinkageKind() != LinkageKind::External) {
    elided.push_back(getLinkageKindAttrName());
  }
  if (getInlineKind() != InlineKind::NoInline) {
    elided.push_back(getInlineKindAttrName());
  }
  function_interface_impl::printFunctionAttributes(p, *this, elided);

  Region& body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

auto FuncOp::parse(OpAsmParser& parser, OperationState& result) -> ParseResult {
  parseLinkage(parser, result);

  StringRef inlineKind;
  if (succeeded(parser.parseOptionalKeyword(
          &inlineKind, {"no_inline", "inline", "always_inline"}))) {
    result.addAttribute("inline_kind",
                        InlineKindAttr::get(parser.getContext(),
                                            *symbolizeInlineKind(inlineKind)));
  }

  auto funcTypeBuilder = [](Builder& builder, llvm::ArrayRef<Type> argTypes,
                            ArrayRef<Type> results,
                            function_interface_impl::VariadicFlag variadic,
                            std::string&) {
    return FunctionType::get(builder.getContext(), argTypes, results,
                             variadic.isVariadic());
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, true, getFunctionTypeAttrName(result.name),
      funcTypeBuilder, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

namespace {
[[nodiscard]] auto verifyAsmArgumentElementTypes(Operation* op,
                                                 ValueRange inputs,
                                                 ArrayAttr elementTypes)
    -> LogicalResult {
  if (elementTypes.size() != inputs.size())
    return op->emitOpError("expects an element type entry for each input");
  for (auto [input, elementType] : llvm::zip(inputs, elementTypes)) {
    if (isa<UnitAttr>(elementType)) continue;
    if (!isa<TypeAttr>(elementType))
      return op->emitOpError(
          "expects type or unit attributes for input element types");
    if (isa<LLVM::LLVMPointerType>(input.getType())) continue;
    auto pointer = dyn_cast<PointerType>(input.getType());
    if (!pointer)
      return op->emitOpError("expects a pointer input for an indirect operand");
    if (pointer.getElementType() != cast<TypeAttr>(elementType).getValue())
      return op->emitOpError(
          "indirect operand element type does not match its pointer type");
  }
  return success();
}
}  // namespace

auto InlineAsmOp::verify() -> LogicalResult {
  return verifyAsmArgumentElementTypes(*this, getInputs(),
                                       getArgumentElementTypes());
}

auto AsmGotoOp::verify() -> LogicalResult {
  return verifyAsmArgumentElementTypes(*this, getInputs(),
                                       getArgumentElementTypes());
}

auto CallOp::getCallableForCallee() -> CallInterfaceCallable {
  if (auto callee = getCalleeAttr()) return callee;
  return getCalleeOperand();
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  if (auto symbol = dyn_cast<SymbolRefAttr>(callee)) {
    setCalleeAttr(cast<FlatSymbolRefAttr>(symbol));
    return;
  }
  getCalleeOperandMutable().assign(cast<Value>(callee));
}

auto CallOp::getArgOperands() -> Operation::operand_range {
  return getInputs();
}

auto CallOp::getArgOperandsMutable() -> MutableOperandRange {
  return getInputsMutable();
}

auto CallOp::verify() -> LogicalResult {
  if (getCalleeAttr() && getCalleeOperand()) {
    return emitOpError("expects either a callee symbol or a callee operand");
  }

  if (!getCalleeAttr() && !getCalleeOperand()) {
    return emitOpError("expects a callee symbol or a callee operand");
  }

  return success();
}

auto CopyOp::verify() -> LogicalResult {
  auto destination = cast<PointerType>(getDest().getType());
  auto source = cast<PointerType>(getSrc().getType());
  if (destination.getElementType() != source.getElementType())
    return emitOpError("copies between objects of different types (")
           << destination.getElementType() << " and " << source.getElementType()
           << ")";
  return success();
}

auto StoreOp::verify() -> LogicalResult {
#if false
  auto addrType = dyn_cast<PointerType>(getAddr().getType());
  if (!addrType) {
    return emitOpError("addr must be a pointer type");
  }

  auto valueType = getValue().getType();
  if (addrType.getElementType() != valueType) {
    return emitOpError("addr must be a pointer to the value type (")
           << valueType << " but found " << addrType << ")";
  }

#endif

  return success();
}

namespace {

[[nodiscard]] auto isPromotableSlotType(Type type) -> bool {
  return !isa<ClassType, ArrayType, FunctionType>(type);
}

[[nodiscard]] auto accessesSlotAddress(
    const SmallPtrSetImpl<OpOperand*>& blockingUses, OpOperand& address)
    -> bool {
  return blockingUses.size() == 1 && blockingUses.contains(&address);
}

}  // namespace

auto AllocaOp::getPromotableSlots() -> SmallVector<MemorySlot> {
  auto element = cast<PointerType>(getType()).getElementType();
  if (!isPromotableSlotType(element)) return {};
  return {MemorySlot{getResult(), element}};
}

auto AllocaOp::getDefaultValue(const MemorySlot& slot, OpBuilder& builder)
    -> Value {
  return UndefOp::create(builder, getLoc(), slot.elemType);
}

void AllocaOp::handleBlockArgument(const MemorySlot&, BlockArgument,
                                   OpBuilder&) {}

auto AllocaOp::handlePromotionComplete(const MemorySlot&, Value defaultValue,
                                       OpBuilder&)
    -> std::optional<PromotableAllocationOpInterface> {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  erase();
  return std::nullopt;
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable());
  if (!getVolatile_()) return;
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable());
  if (!getVolatile_()) return;
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

auto LoadOp::loadsFrom(const MemorySlot& slot) -> bool {
  return getAddr() == slot.ptr;
}

auto LoadOp::storesTo(const MemorySlot&) -> bool { return false; }

auto LoadOp::getStored(const MemorySlot&, OpBuilder&, Value, const DataLayout&)
    -> Value {
  llvm_unreachable("cxx.load does not store");
}

auto LoadOp::canUsesBeRemoved(const MemorySlot& slot,
                              const SmallPtrSetImpl<OpOperand*>& blockingUses,
                              SmallVectorImpl<OpOperand*>&, const DataLayout&)
    -> bool {
  if (getVolatile_()) return false;
  return accessesSlotAddress(blockingUses, getAddrMutable()) &&
         getType() == slot.elemType;
}

auto LoadOp::removeBlockingUses(const MemorySlot&,
                                const SmallPtrSetImpl<OpOperand*>&, OpBuilder&,
                                Value reachingDefinition, const DataLayout&)
    -> DeletionKind {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

auto StoreOp::loadsFrom(const MemorySlot&) -> bool { return false; }

auto StoreOp::storesTo(const MemorySlot& slot) -> bool {
  return getAddr() == slot.ptr;
}

auto StoreOp::getStored(const MemorySlot&, OpBuilder&, Value, const DataLayout&)
    -> Value {
  return getValue();
}

auto StoreOp::canUsesBeRemoved(const MemorySlot& slot,
                               const SmallPtrSetImpl<OpOperand*>& blockingUses,
                               SmallVectorImpl<OpOperand*>&, const DataLayout&)
    -> bool {
  if (getVolatile_()) return false;
  return accessesSlotAddress(blockingUses, getAddrMutable()) &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType;
}

auto StoreOp::removeBlockingUses(const MemorySlot&,
                                 const SmallPtrSetImpl<OpOperand*>&, OpBuilder&,
                                 Value, const DataLayout&) -> DeletionKind {
  return DeletionKind::Delete;
}

void PointerType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto PointerType::parse(AsmParser& parser) -> Type {
  auto location = parser.getCurrentLocation();
  Type elementType;
  Attribute memorySpace;
  if (parser.parseLess() || parser.parseType(elementType)) return {};
  if (succeeded(parser.parseOptionalComma()) &&
      parser.parseAttribute(memorySpace))
    return {};
  if (parser.parseGreater()) return {};
  return parser.getChecked<PointerType>(location, parser.getContext(),
                                        elementType, memorySpace);
}

auto PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                         Type elementType, Attribute memorySpace)
    -> LogicalResult {
  if (!elementType) return emitError() << "expected pointer element type";
  if (isValidPointerMemorySpace(memorySpace)) return success();
  return emitError() << "expected a nonnegative 32-bit integer memory space or "
                        "an attribute implementing MemorySpaceAttrInterface";
}

auto PointerType::getMemorySpace() const -> Attribute {
  auto space = getAddressSpace();
  if (!space) return ptr::GenericSpaceAttr::get(getContext());
  if (auto integer = dyn_cast<IntegerAttr>(space))
    return LLVM::AddressSpaceAttr::get(getContext(),
                                       integer.getValue().getZExtValue());
  return space;
}

auto PointerType::clonePtrWith(Attribute memorySpace,
                               std::optional<Type> elementType) const
    -> FailureOr<PtrLikeTypeInterface> {
  if (!isValidPointerMemorySpace(memorySpace)) return failure();
  auto element = elementType.value_or(getElementType());
  if (!element) return failure();
  return cast<PtrLikeTypeInterface>(get(getContext(), element, memorySpace));
}

namespace {
[[nodiscard]] auto getPointerLayoutType(PointerType pointer) -> Type {
  auto space = pointer.getMemorySpace();
  if (isa<ptr::GenericSpaceAttr>(space))
    return LLVM::LLVMPointerType::get(pointer.getContext());
  if (auto llvmSpace = dyn_cast<LLVM::LLVMAddrSpaceAttrInterface>(space))
    return LLVM::LLVMPointerType::get(pointer.getContext(),
                                      llvmSpace.getAddressSpace());
  return ptr::PtrType::get(cast<ptr::MemorySpaceAttrInterface>(space));
}
}  // namespace

auto PointerType::getTypeSizeInBits(const DataLayout& layout,
                                    DataLayoutEntryListRef) const
    -> llvm::TypeSize {
  return layout.getTypeSizeInBits(getPointerLayoutType(*this));
}

auto PointerType::getABIAlignment(const DataLayout& layout,
                                  DataLayoutEntryListRef) const -> uint64_t {
  return layout.getTypeABIAlignment(getPointerLayoutType(*this));
}

auto PointerType::getPreferredAlignment(const DataLayout& layout,
                                        DataLayoutEntryListRef) const
    -> uint64_t {
  return layout.getTypePreferredAlignment(getPointerLayoutType(*this));
}

auto PointerType::getIndexBitwidth(const DataLayout& layout,
                                   DataLayoutEntryListRef) const
    -> std::optional<uint64_t> {
  return layout.getTypeIndexBitwidth(getPointerLayoutType(*this));
}

auto PointerType::verifyEntries(DataLayoutEntryListRef entries,
                                Location location) const -> LogicalResult {
  if (entries.empty()) return success();
  return emitError(location,
                   "specify Cxx pointer layout using LLVM or Ptr layout keys");
}

auto SubscriptOp::verify() -> LogicalResult {
  return verifyPointerMemorySpaces(*this, getBase().getType(), getType());
}

auto MemberOp::verify() -> LogicalResult {
  auto source = cast<PointerType>(getBase().getType()).getElementType();
  auto field = getAggregateElementType(source, getMemberIndex());
  if (!field) return emitOpError("member index is outside its record");
  return verifyPointerMemorySpaces(*this, getBase().getType(), getType());
}

auto PtrAddOp::verify() -> LogicalResult {
  return verifyPointerMemorySpaces(*this, getBase().getType(), getType());
}

auto PtrDiffOp::verify() -> LogicalResult {
  return verifyPointerMemorySpaces(*this, getLhs().getType(),
                                   getRhs().getType());
}

auto ArrayToPointerOp::verify() -> LogicalResult {
  return verifyPointerMemorySpaces(*this, getValue().getType(), getType());
}

auto BitcastOp::verify() -> LogicalResult {
  return verifyPointerMemorySpaces(*this, getValue().getType(), getType());
}

auto BitcastOp::fold(FoldAdaptor) -> OpFoldResult {
  if (!isa<PointerType>(getType()) || !isa<PointerType>(getValue().getType()))
    return {};
  if (getValue().getType() == getType()) return getValue();
  auto previous = getValue().getDefiningOp<BitcastOp>();
  if (previous && previous.getValue().getType() == getType())
    return previous.getValue();
  return {};
}

void ArrayType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto ArrayType::parse(AsmParser& parser) -> Type {
  Type elementType;
  std::uint64_t size;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseComma() || parser.parseInteger(size) || parser.parseGreater())
    return {};
  return get(parser.getContext(), elementType, size);
}

void FunctionType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto FunctionType::parse(AsmParser& parser) -> Type {
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  if (parser.parseLess() || parseTypeList(parser, inputs) ||
      parser.parseComma() || parseTypeList(parser, results) ||
      parser.parseComma())
    return {};
  StringRef variadic;
  if (failed(parser.parseOptionalKeyword(&variadic, {"true", "false"}))) {
    parser.emitError(parser.getCurrentLocation(), "expected 'true' or 'false'");
    return {};
  }
  if (parser.parseGreater()) return {};
  return get(parser.getContext(), inputs, results, variadic == "true");
}

auto FunctionType::clone(TypeRange inputs, TypeRange results) const
    -> FunctionType {
  return get(getContext(), llvm::to_vector(inputs), llvm::to_vector(results),
             getVariadic());
}

void VTableOp::print(OpAsmPrinter& p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << ' ';

  if (auto linkage = getLinkageKind()) {
    p << stringifyLinkageKind(*linkage) << ' ';
  }

  if (auto typeInfo = getTypeInfo()) {
    p << "type_info ";
    p.printSymbolName(*typeInfo);
    p << ' ';
  }

  auto printOffsetArray = [&](llvm::StringRef label, Attribute offsets) {
    auto array = mlir::cast<ArrayAttr>(offsets);
    if (array.empty()) return;
    p << label << " [";
    llvm::interleaveComma(array, p, [&](Attribute entry) {
      p << mlir::cast<IntegerAttr>(entry).getInt();
    });
    p << "] ";
  };

  auto offsetsToTop = getOffsetsToTop();
  auto offsets = getOffsets();

  p << '{';
  llvm::interleaveComma(
      llvm::seq<std::size_t>(0, getSlots().size()), p, [&](std::size_t index) {
        p << ' ';
        printOffsetArray("offsets", offsets[index]);
        p << "offset_to_top "
          << mlir::cast<IntegerAttr>(offsetsToTop[index]).getInt() << " [";
        llvm::interleaveComma(
            mlir::cast<ArrayAttr>(getSlots()[index]), p, [&](Attribute entry) {
              if (auto symRef = mlir::dyn_cast<FlatSymbolRefAttr>(entry)) {
                p.printSymbolName(symRef.getValue());
              } else {
                p << "null";
              }
            });
        p << ']';
      });
  p << " }";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"sym_name", "linkage_kind", "type_info", "offsets",
                           "offsets_to_top", "slots"});
}

auto VTableOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  auto& builder = parser.getBuilder();
  StringAttr name;
  if (parser.parseSymbolName(name, "sym_name", result.attributes))
    return failure();
  parseLinkage(parser, result);
  if (succeeded(parser.parseOptionalKeyword("type_info"))) {
    FlatSymbolRefAttr typeInfo;
    if (parser.parseAttribute(typeInfo, "type_info", result.attributes))
      return failure();
  }
  if (parser.parseLBrace()) return failure();

  SmallVector<Attribute> offsets;
  SmallVector<Attribute> offsetsToTop;
  SmallVector<Attribute> slots;
  while (failed(parser.parseOptionalRBrace())) {
    SmallVector<Attribute> tableOffsets;
    if (succeeded(parser.parseOptionalKeyword("offsets"))) {
      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
            int64_t offset;
            if (parser.parseInteger(offset)) return failure();
            tableOffsets.push_back(builder.getI64IntegerAttr(offset));
            return success();
          }))
        return failure();
    }
    int64_t offsetToTop;
    if (parser.parseKeyword("offset_to_top") ||
        parser.parseInteger(offsetToTop)) {
      return failure();
    }
    SmallVector<Attribute> tableSlots;
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          if (succeeded(parser.parseOptionalKeyword("null"))) {
            tableSlots.push_back(builder.getUnitAttr());
            return success();
          }
          FlatSymbolRefAttr slot;
          if (parser.parseAttribute(slot)) return failure();
          tableSlots.push_back(slot);
          return success();
        }))
      return failure();
    offsets.push_back(builder.getArrayAttr(tableOffsets));
    offsetsToTop.push_back(builder.getI64IntegerAttr(offsetToTop));
    slots.push_back(builder.getArrayAttr(tableSlots));
    if (succeeded(parser.parseOptionalRBrace())) break;
    if (parser.parseComma()) return failure();
  }
  result.addAttribute("offsets", builder.getArrayAttr(offsets));
  result.addAttribute("offsets_to_top", builder.getArrayAttr(offsetsToTop));
  result.addAttribute("slots", builder.getArrayAttr(slots));
  return parser.parseOptionalAttrDict(result.attributes);
}

void GlobalOp::print(OpAsmPrinter& p) {
  p << ' ';
  p.printSymbolName(getSymName());

  if (auto linkage = getLinkageKind()) {
    if (*linkage != LinkageKind::External) {
      p << ' ' << stringifyLinkageKind(*linkage);
    }
  }

  if (getConstant()) {
    p << " constant";
  }

  p << " : ";
  p.printType(getGlobalType());

  if (auto val = getValue()) {
    p << " = ";
    p.printAttribute(*val);
  }

  auto& region = getInitializer();
  if (!region.empty()) {
    p << ' ';
    p.printRegion(region);
  }

  if (auto alignment = getAlignment()) {
    p << " alignment(" << *alignment << ")";
  }

  SmallVector<StringRef> elided = {"sym_name", "global_type", "constant",
                                   "value", "alignment"};
  if (getLinkageKind() != LinkageKind::External) {
    elided.push_back("linkage_kind");
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elided);
}

auto GlobalOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  auto& builder = parser.getBuilder();
  StringAttr name;
  if (parser.parseSymbolName(name, "sym_name", result.attributes))
    return failure();
  parseLinkage(parser, result);
  if (succeeded(parser.parseOptionalKeyword("constant"))) {
    result.addAttribute("constant", builder.getUnitAttr());
  }
  Type type;
  if (parser.parseColonType(type)) return failure();
  result.addAttribute("global_type", TypeAttr::get(type));
  if (succeeded(parser.parseOptionalEqual())) {
    Attribute value;
    if (parser.parseAttribute(value, "value", result.attributes))
      return failure();
  }
  auto region = result.addRegion();
  auto parsedRegion = parser.parseOptionalRegion(*region);
  if (parsedRegion.has_value() && failed(*parsedRegion)) return failure();
  if (succeeded(parser.parseOptionalKeyword("alignment"))) {
    int64_t alignment;
    if (parser.parseLParen() || parser.parseInteger(alignment) ||
        parser.parseRParen()) {
      return failure();
    }
    result.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
  }
  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

namespace {

[[nodiscard]] auto isElementPositionSet(ArrayRef<unsigned> positions,
                                        std::size_t elementCount) -> bool {
  if (positions.empty()) return true;
  if (positions.back() >= elementCount) return false;
  return llvm::adjacent_find(positions, std::greater_equal<>()) ==
         positions.end();
}

}  // namespace

auto ClassType::getNamed(MLIRContext* context, StringRef name) -> ClassType {
  return Base::get(context, name);
}

auto ClassType::setBody(llvm::ArrayRef<Type> body, bool packed,
                        uint64_t alignment, ArrayRef<unsigned> padding)
    -> LogicalResult {
  if (!isElementPositionSet(padding, body.size())) return failure();
  return Base::mutate(body, packed, alignment, padding);
}

void ClassType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto ClassType::parse(AsmParser& parser) -> Type {
  auto location = parser.getCurrentLocation();
  std::string name;
  if (parser.parseLess() || parser.parseString(&name)) return {};
  auto type = getNamed(parser.getContext(), name);
  if (succeeded(parser.parseOptionalGreater())) return type;
  if (parser.parseComma()) return {};
  auto cyclicParse = parser.tryStartCyclicParse(type);
  if (failed(cyclicParse)) {
    parser.emitError(location, "recursive class body definition");
    return {};
  }
  bool packed = succeeded(parser.parseOptionalKeyword("packed"));
  uint64_t alignment = 0;
  if (succeeded(parser.parseOptionalKeyword("align"))) {
    auto alignmentLocation = parser.getCurrentLocation();
    if (parser.parseInteger(alignment)) return {};
    if (!llvm::isPowerOf2_64(alignment)) {
      parser.emitError(alignmentLocation,
                       "class alignment must be a power of two");
      return {};
    }
  }
  SmallVector<Type> body;
  SmallVector<unsigned> padding;
  if (parseTypeList(parser, body, &padding) || parser.parseGreater()) return {};
  if (failed(type.setBody(body, packed, alignment, padding))) {
    parser.emitError(location, "conflicting class body definition");
    return {};
  }
  return type;
}

auto ClassType::getName() const -> StringRef { return getImpl()->getName(); }

auto ClassType::getBody() const -> ArrayRef<Type> {
  return getImpl()->getBody();
}

auto ClassType::isPacked() const -> bool { return getImpl()->isPacked(); }

auto ClassType::getPadding() const -> ArrayRef<unsigned> {
  return getImpl()->getPadding();
}

auto ClassType::isPaddingElement(unsigned position) const -> bool {
  return llvm::binary_search(getPadding(), position);
}

auto ClassType::getAlignment() const -> uint64_t {
  return getImpl()->getAlignment();
}

namespace {

[[nodiscard]] auto objectBytes(Type type, const DataLayout& layout)
    -> uint64_t {
  return layout.getTypeSize(type).getFixedValue();
}

[[nodiscard]] auto objectStride(Type type, const DataLayout& layout)
    -> uint64_t {
  return llvm::alignTo(objectBytes(type, layout),
                       layout.getTypeABIAlignment(type));
}

}  // namespace

auto ArrayType::getTypeSizeInBits(const DataLayout& layout,
                                  DataLayoutEntryListRef) const
    -> llvm::TypeSize {
  return llvm::TypeSize::getFixed(objectStride(getElementType(), layout) *
                                  getSize() * 8);
}

auto ArrayType::getABIAlignment(const DataLayout& layout,
                                DataLayoutEntryListRef) const -> uint64_t {
  return layout.getTypeABIAlignment(getElementType());
}

auto ArrayType::getPreferredAlignment(const DataLayout& layout,
                                      DataLayoutEntryListRef) const
    -> uint64_t {
  return layout.getTypePreferredAlignment(getElementType());
}

auto getClassElementOffsets(ClassType type, const DataLayout& layout)
    -> SmallVector<std::uint64_t> {
  SmallVector<std::uint64_t> offsets;
  uint64_t offset = 0;
  for (auto field : type.getBody()) {
    auto alignment = type.isPacked() ? 1 : layout.getTypeABIAlignment(field);
    offset = llvm::alignTo(offset, alignment);
    offsets.push_back(offset);
    offset += objectBytes(field, layout);
  }
  return offsets;
}

auto ClassType::getTypeSizeInBits(const DataLayout& layout,
                                  DataLayoutEntryListRef) const
    -> llvm::TypeSize {
  if (getBody().empty()) return llvm::TypeSize::getFixed(8);
  auto offsets = getClassElementOffsets(*this, layout);
  auto end = offsets.back() + objectBytes(getBody().back(), layout);
  auto alignment = getABIAlignment(layout, {});
  return llvm::TypeSize::getFixed(llvm::alignTo(end, alignment) * 8);
}

auto ClassType::getABIAlignment(const DataLayout& layout,
                                DataLayoutEntryListRef) const -> uint64_t {
  if (auto alignment = getAlignment()) return alignment;
  if (isPacked()) return 1;
  uint64_t alignment = 1;
  for (auto field : getBody())
    alignment = std::max(alignment, layout.getTypeABIAlignment(field));
  return alignment;
}

auto ClassType::getPreferredAlignment(const DataLayout& layout,
                                      DataLayoutEntryListRef) const
    -> uint64_t {
  return getABIAlignment(layout, {});
}
namespace {
class NumericBitcast final : public OpRewritePattern<BitcastOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(BitcastOp operation, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto source = operation.getValue().getType();
    auto target = operation.getType();
    if (!arith::BitcastOp::areCastCompatible(TypeRange{source},
                                             TypeRange{target}))
      return failure();
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(operation, target,
                                                  operation.getValue());
    return success();
  }
};

class StorageReferenceToPtr final : public OpRewritePattern<StorageRefOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(StorageRefOp operation, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto source = cast<PtrLikeTypeInterface>(operation.getStorage().getType());
    auto space = dyn_cast_or_null<ptr::MemorySpaceAttrInterface>(
        source.getMemorySpace());
    if (!space) return failure();
    auto result = cast<PointerType>(operation.getType());
    if (result.getMemorySpace() != space) return failure();
    auto pointer =
        ptr::ToPtrOp::create(rewriter, operation.getLoc(),
                             ptr::PtrType::get(space), operation.getStorage());
    rewriter.replaceOpWithNewOp<ptr::FromPtrOp>(operation, result, pointer,
                                                Value());
    return success();
  }
};
}  // namespace

void BitcastOp::getCanonicalizationPatterns(RewritePatternSet& patterns,
                                            MLIRContext* context) {
  patterns.add<NumericBitcast>(context);
}

void StorageRefOp::getCanonicalizationPatterns(RewritePatternSet& patterns,
                                               MLIRContext* context) {
  patterns.add<StorageReferenceToPtr>(context);
}

auto StorageRefOp::verify() -> LogicalResult {
  auto memory = dyn_cast<MemRefType>(getStorage().getType());
  if (!memory || memory.getRank() > 1 || !memory.getLayout().isIdentity())
    return emitOpError("requires a whole scalar or linear storage root");
  if (!isStorageRoot(getStorage()))
    return emitOpError(
        "requires an allocation or a function storage parameter");
  auto pointer = cast<PointerType>(getType());
  if (pointer.getElementType() != memory.getElementType())
    return emitOpError("reference element type differs from its storage");
  return success();
}

auto BindReferencesOp::verify() -> LogicalResult {
  auto pointer = cast<PointerType>(getType());
  auto record = dyn_cast<ClassType>(pointer.getElementType());
  if (!record || record.getBody().empty())
    return emitOpError("requires a reference environment record");
  ReferenceRecordLayout layout(record);
  if (layout.referenceFields().empty())
    return emitOpError("requires a record with reference fields");
  if (layout.referenceFields().size() != getFields().size())
    return emitOpError("requires one binding per reference field");
  for (auto [index, value] : llvm::zip(layout.referenceFields(), getFields())) {
    if (record.getBody()[index] != value.getType())
      return emitOpError("reference binding type differs from its field");
  }
  auto state = getState();
  auto stateType = state ? cast<MemRefType>(state.getType()) : MemRefType();
  if (auto expected = layout.stateType()) {
    if (!stateType || stateType.getRank() ||
        stateType.getElementType() != expected)
      return emitOpError("state storage differs from its record fields");
    if (!isStorageRoot(state))
      return emitOpError("requires whole state storage");
  } else if (state) {
    return emitOpError("reference-only record has no state storage");
  }
  return success();
}

}  // namespace mlir::cxx

#define GET_ATTRDEF_CLASSES
#include <cxx/mlir/CxxOpsAttributes.cpp.inc>
#include <cxx/mlir/CxxOpsDialect.cpp.inc>
#include <cxx/mlir/CxxOpsEnums.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <cxx/mlir/CxxOpsTypes.cpp.inc>

#define GET_OP_CLASSES
#include <cxx/mlir/CxxOps.cpp.inc>
