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

#include <cxx/mlir/module_data_layout.h>
#include <llvm/IR/DataLayout.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Target/LLVMIR/Import.h>

namespace cxx {

void materializeDataLayout(mlir::ModuleOp module) {
  if (module->hasAttr(mlir::DLTIDialect::kDataLayoutAttrName)) return;
  auto layout = module->getAttrOfType<mlir::StringAttr>("cxx.data-layout");
  if (!layout) return;
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                  mlir::translateDataLayout(llvm::DataLayout(layout.getValue()),
                                            module.getContext()));
}

}  // namespace cxx
