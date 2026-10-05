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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

#include <numeric>

namespace mlir::cxx {
struct detail::ClassTypeStorage : public TypeStorage {
 public:
  using KeyTy = StringRef;

  explicit ClassTypeStorage(const KeyTy& key) : name_(key) {}

  auto getName() -> StringRef const { return name_; }
  auto getBody() const -> ArrayRef<Type> { return body_; }
  auto isPacked() const -> bool { return packed_; }

  auto operator==(const KeyTy& key) const -> bool { return name_ == key; };

  static auto hashKey(const KeyTy& key) -> llvm::hash_code {
    return llvm::hash_value(key);
  }

  static ClassTypeStorage* construct(TypeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ClassTypeStorage>())
        ClassTypeStorage(allocator.copyInto(key));
  }

  auto mutate(TypeStorageAllocator& allocator, ArrayRef<Type> body, bool packed)
      -> LogicalResult {
    if (isInitialized_) return success(body == getBody() && packed == packed_);

    isInitialized_ = true;
    body_ = allocator.copyInto(body);
    packed_ = packed;

    return success();
  }

 private:
  StringRef name_;
  ArrayRef<Type> body_;
  bool packed_ = false;
  bool isInitialized_ = false;
};

namespace {
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
    printTypeList(type.getBody());
    printer_ << '>';
  }

  void printBody(PointerType type) {
    printer_ << '<';
    printType(type.getElementType());
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
  void printTypeList(ArrayRef<Type> types) {
    printer_ << '(';
    llvm::interleaveComma(types, printer_.getStream(),
                          [this](Type type) { printType(type); });
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

auto parseTypeList(AsmParser& parser, SmallVectorImpl<Type>& types)
    -> ParseResult {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
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

auto CallOp::verify() -> LogicalResult {
  if (getCalleeAttr() && getCalleeOperand()) {
    return emitOpError("expects either a callee symbol or a callee operand");
  }

  if (!getCalleeAttr() && !getCalleeOperand()) {
    return emitOpError("expects a callee symbol or a callee operand");
  }

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

void PointerType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto PointerType::parse(AsmParser& parser) -> Type {
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater())
    return {};
  return get(parser.getContext(), elementType);
}

void ArrayType::print(AsmPrinter& p) const {
  TypeGraphPrinter(p).printBody(*this);
}

auto ArrayType::parse(AsmParser& parser) -> Type {
  Type elementType;
  unsigned size;
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

auto ClassType::getNamed(MLIRContext* context, StringRef name) -> ClassType {
  return Base::get(context, name);
}

auto ClassType::setBody(llvm::ArrayRef<Type> body, bool packed)
    -> LogicalResult {
  return Base::mutate(body, packed);
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
  SmallVector<Type> body;
  if (parseTypeList(parser, body) || parser.parseGreater()) return {};
  if (failed(type.setBody(body, packed))) {
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
}  // namespace mlir::cxx

#define GET_ATTRDEF_CLASSES
#include <cxx/mlir/CxxOpsAttributes.cpp.inc>
#include <cxx/mlir/CxxOpsDialect.cpp.inc>
#include <cxx/mlir/CxxOpsEnums.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <cxx/mlir/CxxOpsTypes.cpp.inc>

#define GET_OP_CLASSES
#include <cxx/mlir/CxxOps.cpp.inc>
