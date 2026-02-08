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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

struct ToInt {
  auto operator()(bool v) const -> std::optional<std::intmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(std::intmax_t v) const -> std::optional<std::intmax_t> {
    return v;
  }

  auto operator()(auto x) const -> std::optional<std::intmax_t> {
    return std::nullopt;  // Unsupported type for int conversion
  }
};

struct ToUInt {
  auto operator()(bool v) const -> std::optional<std::uintmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(std::intmax_t v) const -> std::optional<std::uintmax_t> {
    return std::bit_cast<std::uintmax_t>(v);
  }

  auto operator()(auto x) const -> std::optional<std::uintmax_t> {
    return std::nullopt;
  }
};

template <typename T>
struct ArithmeticCast {
  auto operator()(const StringLiteral*) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<Meta>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<InitializerList>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstObject>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(auto value) const -> T { return static_cast<T>(value); }
};

}  // namespace

struct ASTInterpreter::ToBool {
  ASTInterpreter& interp;

  auto operator()(const StringLiteral*) const -> std::optional<bool> {
    return true;
  }

  auto operator()(const Meta&) const -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const std::shared_ptr<ConstObject>&) const
      -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const auto& value) const -> std::optional<bool> {
    return bool(value);
  }
};

ASTInterpreter::ASTInterpreter(TranslationUnit* unit) : unit_(unit) {}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::evaluate(ExpressionAST* ast) -> std::optional<ConstValue> {
  auto result = expression(ast);
  return result;
}

auto ASTInterpreter::toBool(const ConstValue& value) -> std::optional<bool> {
  return std::visit(ToBool{*this}, value);
}

auto ASTInterpreter::toInt(const ConstValue& value)
    -> std::optional<std::intmax_t> {
  return std::visit(ToInt{}, value);
}

auto ASTInterpreter::toUInt(const ConstValue& value)
    -> std::optional<std::uintmax_t> {
  return std::visit(ToUInt{}, value);
}

auto ASTInterpreter::toFloat(const ConstValue& value) -> std::optional<float> {
  return std::visit(ArithmeticCast<float>{}, value);
}

auto ASTInterpreter::toDouble(const ConstValue& value)
    -> std::optional<double> {
  return std::visit(ArithmeticCast<double>{}, value);
}

auto ASTInterpreter::toLongDouble(const ConstValue& value)
    -> std::optional<long double> {
  return std::visit(ArithmeticCast<long double>{}, value);
}

auto ASTInterpreter::lookupLocal(const Symbol* sym) const
    -> std::optional<ConstValue> {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return found->second;
  }
  return std::nullopt;
}

void ASTInterpreter::setLocal(const Symbol* sym, ConstValue value) {
  if (frames_.empty()) frames_.push_back({});
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) {
      found->second = std::move(value);
      return;
    }
  }
  frames_.back().locals.emplace(sym, std::move(value));
}

void ASTInterpreter::pushFrame() { frames_.push_back({}); }

void ASTInterpreter::popFrame() {
  if (!frames_.empty()) frames_.pop_back();
}

auto ASTInterpreter::evaluateCall(FunctionSymbol* func,
                                  std::vector<ConstValue> args)
    -> std::optional<ConstValue> {
  if (!func || !func->isConstexpr()) return std::nullopt;

  auto* defn = func->definition();
  if (!defn) defn = func;

  auto* funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto* body = funcDef->functionBody;
  if (!body) return std::nullopt;

  auto* compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody || !compBody->statement) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  auto* params = func->functionParameters();
  if (params) {
    const auto& members = params->members();
    for (std::size_t i = 0; i < args.size() && i < members.size(); ++i) {
      setLocal(members[i], std::move(args[i]));
    }
  }

  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = takeReturnValue();

  popFrame();
  --depth_;

  return result;
}

auto ASTInterpreter::evaluateConstructor(FunctionSymbol* ctor,
                                         const Type* classType,
                                         std::vector<ConstValue> args)
    -> std::optional<ConstValue> {
  if (!ctor || !ctor->isConstexpr()) return std::nullopt;

  auto* defn = ctor->definition();
  if (!defn) defn = ctor;

  auto* funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto* body = funcDef->functionBody;
  if (!body) return std::nullopt;

  auto* compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  auto obj = std::make_shared<ConstObject>(classType);
  auto savedThis = thisObject_;
  thisObject_ = obj;

  auto* params = ctor->functionParameters();
  if (params) {
    const auto& members = params->members();
    for (std::size_t i = 0; i < args.size() && i < members.size(); ++i) {
      setLocal(members[i], std::move(args[i]));
    }
  }

  for (auto node : ListView{compBody->memInitializerList}) {
    (void)memInitializer(node);
  }

  returnValue_.reset();
  if (compBody->statement) {
    (void)statement(compBody->statement);
  }

  thisObject_ = savedThis;
  popFrame();
  --depth_;

  return ConstValue{std::move(obj)};
}

}  // namespace cxx
