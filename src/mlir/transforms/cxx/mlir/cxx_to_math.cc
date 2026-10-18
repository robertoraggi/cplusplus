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

#include <cxx/mlir/passes.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include "private/cxx_to_math-priv.h"

namespace mlir {
namespace {

using ::cxx::math_rewrites::FloatingKind;
using ::cxx::math_rewrites::kLibraryMathOps;
using ::cxx::math_rewrites::LibraryMathOp;

[[nodiscard]] auto libraryMathOp(StringRef callee) -> const LibraryMathOp* {
  auto found = llvm::find_if(kLibraryMathOps, [&](const LibraryMathOp& entry) {
    return entry.callee == callee;
  });
  return found == std::end(kLibraryMathOps) ? nullptr : found;
}

[[nodiscard]] auto hasFloatingKind(Type type, FloatingKind kind) -> bool {
  auto floating = dyn_cast<FloatType>(type);
  if (!floating) return false;
  switch (kind) {
    case FloatingKind::kFloat:
      return floating.isF32();
    case FloatingKind::kDouble:
      return floating.isF64();
    case FloatingKind::kLongDouble:
      return true;
  }
  return false;
}

[[nodiscard]] auto matchesSignature(CallOpInterface call,
                                    const LibraryMathOp& entry) -> bool {
  if (call->getNumResults() != 1) return false;
  auto operands = call.getArgOperands();
  if (operands.size() != entry.arity) return false;
  auto type = call->getResult(0).getType();
  if (!hasFloatingKind(type, entry.floating)) return false;
  return llvm::all_of(operands.getTypes(),
                      [&](Type operand) { return operand == type; });
}

class ConvertCxxToMathPass
    : public PassWrapper<ConvertCxxToMathPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxToMathPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-to-math";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Rewrite calls to C library math functions to the math and arith "
           "dialects";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    SymbolTableCollection symbols;
    SmallVector<CallOpInterface> calls;
    module.walk([&](CallOpInterface call) { calls.push_back(call); });
    llvm::SetVector<Operation*> callees;
    for (auto call : calls) {
      auto callee =
          llvm::dyn_cast_if_present<SymbolRefAttr>(call.getCallableForCallee());
      if (!callee) continue;
      auto function = dyn_cast_if_present<FunctionOpInterface>(
          symbols.lookupNearestSymbolFrom(call, callee));
      if (!function || !function.isExternal()) continue;
      auto entry = libraryMathOp(function.getName());
      if (!entry || !matchesSignature(call, *entry)) continue;
      OpBuilder builder(call);
      auto value = entry->create(builder, call.getLoc(), call.getArgOperands());
      call->getResult(0).replaceAllUsesWith(value);
      call->erase();
      callees.insert(function);
    }
    for (auto callee : callees) {
      if (SymbolTable::symbolKnownUseEmpty(callee, module)) callee->erase();
    }
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxToMathPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxToMathPass>();
}

void cxx::registerConvertCxxToMathPass() {
  mlir::registerPass(createConvertCxxToMathPass);
}
