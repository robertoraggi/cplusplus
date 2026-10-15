#pragma once

#include <cxx/ast_kind.h>
#include <cxx/codegen/emitter.h>

#include <variant>

namespace cxx {

enum class SyntheticSourceContext {
  Function,
  Condition,
  Cleanup,
  ConditionalCleanup,
  Destructor,
  ArrayCleanup,
  Unwind,
};

using SourceContextKind = std::variant<ASTKind, SyntheticSourceContext>;

enum class SourceBlockRole {
  Condition,
  Then,
  Else,
  Body,
  Step,
  ConditionFalse,
  Continuation,
  ExitTarget,
};

class SourceContextObserver {
 public:
  virtual ~SourceContextObserver() = default;

  [[nodiscard]] virtual auto beginContext(SourceContextKind kind)
      -> std::uint64_t = 0;

  virtual void endContext(std::uint64_t context) = 0;

  [[nodiscard]] virtual auto currentContext() const -> std::uint64_t = 0;
  virtual void setContext(std::uint64_t context) = 0;

  virtual void recordBlock(SourceBlockRole role, ir::BlockRef block) = 0;

  [[nodiscard]] virtual auto beginLifetime() -> std::uint64_t = 0;
  virtual void endLifetime(std::uint64_t scope) = 0;
};

class SourceContextGuard {
 public:
  SourceContextGuard(SourceContextObserver* observer, SourceContextKind kind)
      : observer_(observer),
        context_(observer ? observer->beginContext(kind) : 0) {}

  ~SourceContextGuard() {
    if (observer_) observer_->endContext(context_);
  }

  SourceContextGuard(const SourceContextGuard&) = delete;
  auto operator=(const SourceContextGuard&) -> SourceContextGuard& = delete;

 private:
  SourceContextObserver* observer_;
  std::uint64_t context_;
};

class SourceContextActivation {
 public:
  SourceContextActivation(const SourceContextActivation&) = delete;

  SourceContextActivation(SourceContextObserver* observer,
                          std::uint64_t context)
      : observer_(observer),
        previous_(observer ? observer->currentContext() : 0) {
    if (observer_) observer_->setContext(context);
  }

  ~SourceContextActivation() {
    if (observer_) observer_->setContext(previous_);
  }

  auto operator=(const SourceContextActivation&)
      -> SourceContextActivation& = delete;

 private:
  SourceContextObserver* observer_;
  std::uint64_t previous_;
};

}  // namespace cxx
