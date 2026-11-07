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
