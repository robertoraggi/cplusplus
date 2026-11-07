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

#include <cxx/codegen/source_context_observer.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <memory>

namespace cxx {
class TimeTrace;
}

namespace cxx::ir {
class MlirEmitter;

struct SourceContextOptions {
  enum class Mode { StructuredIf, Structured };
  Mode mode = Mode::Structured;
  TimeTrace* timeTrace = nullptr;
};

class MlirSourceContexts final : public SourceContextObserver,
                                 public mlir::RewriterBase::Listener {
 public:
  explicit MlirSourceContexts(MlirEmitter& emitter,
                              SourceContextOptions options);
  ~MlirSourceContexts() override;
  void beginFunction(FunctionRef function);
  void endFunction(FunctionRef function);
  [[nodiscard]] auto contextOf(mlir::Operation* operation) const
      -> std::uint64_t;
  void blockCreated(mlir::Block* block);
  [[nodiscard]] auto beginContext(SourceContextKind kind)
      -> std::uint64_t override;
  void endContext(std::uint64_t context) override;
  [[nodiscard]] auto currentContext() const -> std::uint64_t override;
  void setContext(std::uint64_t context) override;
  void recordBlock(SourceBlockRole role, BlockRef block) override;
  [[nodiscard]] auto beginLifetime() -> std::uint64_t override;
  void endLifetime(std::uint64_t scope) override;
  void notifyOperationInserted(mlir::Operation* op,
                               mlir::OpBuilder::InsertPoint previous) override;
  void notifyOperationErased(mlir::Operation* op) override;
  void notifyBlockInserted(mlir::Block* block, mlir::Region* previous,
                           mlir::Region::iterator previousIt) override;
  void notifyBlockErased(mlir::Block* block) override;

 private:
  struct Data;
  std::unique_ptr<Data> data_;
};
}  // namespace cxx::ir
