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
}
