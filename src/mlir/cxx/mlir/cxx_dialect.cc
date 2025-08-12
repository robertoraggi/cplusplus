// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

// mlir
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

namespace mlir::cxx {

struct detail::ClassTypeStorage : public TypeStorage {
 public:
  using KeyTy = StringRef;

  explicit ClassTypeStorage(const KeyTy &key) : name_(key) {}

  auto getName() -> StringRef const { return name_; }
  auto getBody() const -> ArrayRef<Type> { return body_; }

  auto operator==(const KeyTy &key) const -> bool { return name_ == key; };

  static auto hashKey(const KeyTy &key) -> llvm::hash_code {
    return llvm::hash_value(key);
  }

  static ClassTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ClassTypeStorage>())
        ClassTypeStorage(allocator.copyInto(key));
  }

  auto mutate(TypeStorageAllocator &allocator, ArrayRef<Type> body)
      -> LogicalResult {
    if (isInitialized_) return success(body == getBody());

    isInitialized_ = true;
    body_ = allocator.copyInto(body);

    return success();
  }

 private:
  StringRef name_;
  ArrayRef<Type> body_;
  bool isInitialized_ = false;
};

namespace {

struct CxxGenerateAliases : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  auto getAlias(Type type, raw_ostream &os) const -> AliasResult override {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      os << 'i' << intType.getWidth() << (intType.getIsSigned() ? 's' : 'u');
      return AliasResult::FinalAlias;
    }

    if (auto floatType = dyn_cast<FloatType>(type)) {
      os << 'f' << floatType.getWidth();
      return AliasResult::FinalAlias;
    }

    if (auto classType = dyn_cast<ClassType>(type)) {
      if (!classType.getBody().empty()) {
        os << "class_" << classType.getName();
        return AliasResult::FinalAlias;
      }
    }

    if (isa<VoidType>(type)) {
      os << "void";
      return AliasResult::FinalAlias;
    }

    if (isa<BoolType>(type)) {
      os << "bool";
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

  addInterface<CxxGenerateAliases>();
}

void FuncOp::print(OpAsmPrinter &p) {
  const auto isVariadic = getFunctionType().getVariadic();
  function_interface_impl::printFunctionOp(
      p, *this, isVariadic, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

auto FuncOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult {
  auto funcTypeBuilder =
      [](Builder &builder, llvm::ArrayRef<Type> argTypes,
         ArrayRef<Type> results, function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      funcTypeBuilder, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
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

auto FunctionType::clone(TypeRange inputs, TypeRange results) const
    -> FunctionType {
  return get(getContext(), llvm::to_vector(inputs), llvm::to_vector(results),
             getVariadic());
}

auto ClassType::getNamed(MLIRContext *context, StringRef name) -> ClassType {
  return Base::get(context, name);
}

auto ClassType::setBody(llvm::ArrayRef<Type> body) -> LogicalResult {
  Base::mutate(body);
}

void ClassType::print(AsmPrinter &p) const {
  FailureOr<AsmPrinter::CyclicPrintReset> cyclicPrint;

  p << "<";
  cyclicPrint = p.tryStartCyclicPrint(*this);

  p << '"';
  llvm::printEscapedString(getName(), p.getStream());
  p << '"';

  if (failed(cyclicPrint)) {
    p << '>';
    return;
  }

  p << ", ";

  p << '(';
  llvm::interleaveComma(getBody(), p.getStream(),
                        [&](Type subtype) { p << subtype; });
  p << ')';

  p << '>';
}

auto ClassType::parse(AsmParser &parser) -> Type {
  // todo: implement parsing for ClassType
  return {};
}

auto ClassType::getName() const -> StringRef { return getImpl()->getName(); }

auto ClassType::getBody() const -> ArrayRef<Type> {
  return getImpl()->getBody();
}

}  // namespace mlir::cxx

#include <cxx/mlir/CxxOpsDialect.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <cxx/mlir/CxxOpsTypes.cpp.inc>

#define GET_OP_CLASSES
#include <cxx/mlir/CxxOps.cpp.inc>
