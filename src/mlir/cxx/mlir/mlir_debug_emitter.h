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
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Builders.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace cxx::ir {
class MlirEmitter;

class MlirDebugEmitter final : public DebugEmitter {
 public:
  explicit MlirDebugEmitter(MlirEmitter& emitter);
  auto compileUnit(const DebugCompileUnitInfo& info) -> DebugScopeRef override;
  auto fileScope(std::string_view file) -> DebugScopeRef override;
  auto lexicalBlock(DebugScopeRef parent, DebugLocation location)
      -> DebugScopeRef override;
  auto typeScope(DebugTypeRef type) -> DebugScopeRef override;
  auto basicType(const DebugBasicTypeInfo& info) -> DebugTypeRef override;
  auto derivedType(const DebugDerivedTypeInfo& info) -> DebugTypeRef override;
  auto compositeType(const DebugCompositeTypeInfo& info)
      -> DebugTypeRef override;
  auto arrayType(const DebugArrayTypeInfo& info) -> DebugTypeRef override;
  auto subroutineType(std::span<const DebugTypeRef> types)
      -> DebugTypeRef override;
  auto defineFunction(FunctionRef function, SourceLocation loc,
                      const DebugFunctionInfo& info) -> DebugScopeRef override;
  void localVariable(ValueRef address, const DebugVariableInfo& info) override;

 private:
  [[nodiscard]] auto type(DebugTypeRef ref) const -> mlir::LLVM::DITypeAttr;
  [[nodiscard]] auto scope(DebugScopeRef ref) const -> mlir::LLVM::DIScopeAttr;
  auto getFileAttr(std::string_view file) -> mlir::LLVM::DIFileAttr;
  auto getOrCreateFileAttr(std::string_view file) -> mlir::LLVM::DIFileAttr;
  auto compositeTypeAttr(unsigned tag, std::string_view name,
                         DebugLocation location, DebugScopeRef parent,
                         DebugTypeRef baseType, std::uint64_t sizeInBits,
                         std::uint64_t alignInBits, bool scopedEnum,
                         llvm::ArrayRef<mlir::LLVM::DINodeAttr> elements)
      -> mlir::LLVM::DITypeAttr;
  MlirEmitter& emitter_;
  mlir::MLIRContext* context_;
  mlir::OpBuilder& builder_;
  std::string compilationDirectory_;
  std::unordered_map<std::string, mlir::LLVM::DIFileAttr> fileAttrs_;
  mlir::LLVM::DICompileUnitAttr compileUnitAttr_;
  std::vector<mlir::LLVM::DITypeAttr> types_{{}};
  std::vector<mlir::LLVM::DIScopeAttr> scopes_{{}};
};
}  // namespace cxx::ir
