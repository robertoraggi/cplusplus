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
#include <cxx/type_traits.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Builders.h>

#include <optional>
#include <unordered_map>
namespace cxx {
class Control;
class TranslationUnit;
}  // namespace cxx
namespace cxx::ir {
class MlirEmitter;
class MlirDebugEmitter final : public DebugEmitter {
 public:
  MlirDebugEmitter(MlirEmitter& emitter, TranslationUnit* unit);
  void defineFunction(FunctionSymbol*, FunctionRef, SourceLocation,
                      SourceLocation, SourceLocation) override;
  void localVariable(ValueRef, Symbol*, std::string_view, unsigned) override;
  void objectParameter(ValueRef, const Type*, FunctionSymbol*, std::string_view,
                       unsigned) override;

 private:
  struct ConvertDebugType;
  auto control() const -> Control*;
  auto convertDebugType(const Type*) -> mlir::LLVM::DITypeAttr;
  auto getOrCreateDIScope(Symbol*) -> mlir::LLVM::DIScopeAttr;
  auto buildSubroutineTypeAttr(FunctionSymbol*)
      -> mlir::LLVM::DISubroutineTypeAttr;
  auto getCompileUnitAttr() -> mlir::LLVM::DICompileUnitAttr;
  auto compilationDirectory() -> const std::string&;
  auto getOrCreateFileAttr(const std::string&) -> mlir::LLVM::DIFileAttr;
  auto getFileAttr(const std::string&) -> mlir::LLVM::DIFileAttr;
  auto getFileAttr(std::string_view) -> mlir::LLVM::DIFileAttr;
  auto getFileAttrAt(SourceLocation) -> mlir::LLVM::DIFileAttr;
  MlirEmitter& emitter_;
  mlir::MLIRContext* context_;
  mlir::OpBuilder& builder_;
  TranslationUnit* unit_;
  TypeTraits traits;
  std::unordered_map<FunctionSymbol*, FunctionRef> funcOps_;
  std::unordered_map<std::string, mlir::LLVM::DIFileAttr> fileAttrs_;
  mlir::LLVM::DICompileUnitAttr compileUnitAttr_;
  std::optional<std::string> compilationDirectory_;
  std::unordered_map<const Type*, mlir::LLVM::DITypeAttr> debugTypeCache_;
  std::unordered_map<Symbol*, mlir::LLVM::DIScopeAttr> diScopes_;
};
}  // namespace cxx::ir
