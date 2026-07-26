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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
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

  auto operator()(float v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::intmax_t> {
    return static_cast<std::intmax_t>(v);
  }

  auto operator()(auto x) const -> std::optional<std::intmax_t> {
    return std::nullopt;
  }
};

struct ToUInt {
  auto operator()(bool v) const -> std::optional<std::uintmax_t> {
    return v ? 1 : 0;
  }

  auto operator()(std::intmax_t v) const -> std::optional<std::uintmax_t> {
    return std::bit_cast<std::uintmax_t>(v);
  }

  auto operator()(float v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
  }

  auto operator()(double v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
  }

  auto operator()(long double v) const -> std::optional<std::uintmax_t> {
    return static_cast<std::uintmax_t>(v);
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

  auto operator()(const std::shared_ptr<ConstAddress>&) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(const std::shared_ptr<ConstLabelAddress>&) const -> T {
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

  auto operator()(const std::shared_ptr<ConstAddress>&) const
      -> std::optional<bool> {
    return true;
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
    auto ref = it->refs.find(sym);
    if (ref != it->refs.end()) return *ref->second;
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return found->second;
  }
  return std::nullopt;
}

auto ASTInterpreter::lookupLocalSlot(const Symbol* sym) -> ConstValue* {
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto ref = it->refs.find(sym);
    if (ref != it->refs.end()) return ref->second;
    auto found = it->locals.find(sym);
    if (found != it->locals.end()) return &found->second;
  }
  return nullptr;
}

void ASTInterpreter::bindReference(const Symbol* sym, ConstValue* target) {
  if (frames_.empty()) frames_.push_back({});
  for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
    auto found = it->refs.find(sym);
    if (found != it->refs.end()) {
      found->second = target;
      return;
    }
  }
  frames_.back().refs.emplace(sym, target);
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

auto ASTInterpreter::bindParameters(FunctionSymbol* func,
                                    std::vector<ConstValue>& args) -> bool {
  auto params = func->functionParameters();
  if (!params) return true;

  const auto& members = params->members();
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (i < args.size()) {
      setLocal(members[i], std::move(args[i]));
      continue;
    }
    auto param = symbol_cast<ParameterSymbol>(members[i]);
    if (!param || !param->defaultArgument()) continue;
    auto value = evaluate(param->defaultArgument());
    if (!value) return false;
    setLocal(param, std::move(*value));
  }
  return true;
}

auto ASTInterpreter::bindOneParameter(Symbol* paramSymbol,
                                      ExpressionAST* argExpr) -> bool {
  auto param = symbol_cast<ParameterSymbol>(paramSymbol);
  if (param && unit_->typeTraits().is_reference(param->type())) {
    if (auto slot = lvalue(argExpr)) {
      bindReference(paramSymbol, slot);
      return true;
    }
  }
  auto value = evaluate(argExpr);
  if (!value) return false;
  setLocal(paramSymbol, std::move(*value));
  return true;
}

auto ASTInterpreter::bindParametersFromExprs(FunctionSymbol* func,
                                             List<ExpressionAST*>* argExprs)
    -> bool {
  auto params = func->functionParameters();
  if (!params) return true;

  const auto& members = params->members();
  std::size_t i = 0;
  for (auto node : ListView{argExprs}) {
    if (i >= members.size()) break;
    if (!bindOneParameter(members[i], node)) return false;
    ++i;
  }
  for (; i < members.size(); ++i) {
    auto param = symbol_cast<ParameterSymbol>(members[i]);
    if (!param || !param->defaultArgument()) continue;
    if (!bindOneParameter(members[i], param->defaultArgument())) return false;
  }
  return true;
}

void ASTInterpreter::applyNsdmis(const std::shared_ptr<ConstObject>& obj) {
  auto classType = type_cast<ClassType>(obj->type());
  if (!classType || !classType->symbol()) return;
  for (auto member : classType->symbol()->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic() || !field->initializer()) continue;
    auto value = evaluate(field->initializer());
    if (value) obj->setField(field, std::move(*value));
  }
}

void ASTInterpreter::applyMemInitializer(MemInitializerAST* ast,
                                         std::vector<ConstValue> args) {
  if (!ast->symbol || !thisObject_) return;

  if (auto cls = symbol_cast<ClassSymbol>(ast->symbol)) {
    if (cls != currentConstructorClass_) return;
    if (!ast->constructor) return;
    auto result = evaluateConstructor(ast->constructor, thisObject_->type(),
                                      std::move(args));
    if (result) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&*result)) {
        if (*obj) *thisObject_ = **obj;
      }
    }
    return;
  }

  if (auto base = symbol_cast<BaseClassSymbol>(ast->symbol)) {
    if (base->isVirtual()) return;
    auto baseClassSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSym) return;
    if (ast->constructor) {
      auto result = evaluateConstructor(ast->constructor, baseClassSym->type(),
                                        std::move(args));
      if (result) thisObject_->addBase(std::move(*result));
    } else if (!args.empty()) {
      thisObject_->addBase(std::move(args.front()));
    }
    return;
  }

  auto field = symbol_cast<FieldSymbol>(ast->symbol);
  if (!field) return;

  if (ast->constructor) {
    auto result =
        evaluateConstructor(ast->constructor, field->type(), std::move(args));
    if (result) thisObject_->setField(field, std::move(*result));
    return;
  }

  if (!args.empty()) thisObject_->setField(field, std::move(args.front()));
}

auto ASTInterpreter::defaultConstruct(const Type* type)
    -> std::optional<ConstValue> {
  auto classType = type_cast<ClassType>(unit_->typeTraits().remove_cv(type));
  if (!classType || !classType->symbol()) return std::nullopt;

  auto callableWithNoArgs = [](FunctionSymbol* ctor) {
    auto params = ctor->functionParameters();
    if (!params) return true;
    for (auto member : params->members()) {
      auto param = symbol_cast<ParameterSymbol>(member);
      if (param && !param->defaultArgument()) return false;
    }
    return true;
  };

  for (auto ctor : classType->symbol()->constructors()) {
    if (!ctor->isConstexpr() || !callableWithNoArgs(ctor)) continue;
    if (auto value = evaluateConstructor(ctor, type, {})) return value;
  }

  auto obj = std::make_shared<ConstObject>(type);
  auto savedThis = std::exchange(thisObject_, obj);

  for (auto base : classType->symbol()->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClassSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSym) continue;
    if (auto baseVal = defaultConstruct(baseClassSym->type()))
      obj->addBase(std::move(*baseVal));
  }

  applyNsdmis(obj);
  thisObject_ = std::move(savedThis);
  if (aborted_) return std::nullopt;
  return ConstValue{std::move(obj)};
}

void ASTInterpreter::pushFrame() { frames_.push_back({}); }

void ASTInterpreter::popFrame() {
  if (!frames_.empty()) frames_.pop_back();
}

auto ASTInterpreter::evaluateCall(FunctionSymbol* func,
                                  std::vector<ConstValue> args,
                                  std::shared_ptr<ConstObject> thisObject)
    -> std::optional<ConstValue> {
  if (!func || !func->isConstexpr()) return std::nullopt;

  auto defn = func->definition();
  if (!defn) defn = func;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto body = funcDef->functionBody;
  if (!body) return std::nullopt;

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody || !compBody->statement) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  std::string funcNameStr;
  if (auto id = name_cast<Identifier>(func->name())) {
    funcNameStr = id->value();
  }
  auto savedFunctionName =
      std::exchange(currentFunctionName_, std::move(funcNameStr));

  auto savedThis = thisObject_;
  if (thisObject) thisObject_ = std::move(thisObject);

  if (!bindParameters(func, args)) {
    thisObject_ = std::move(savedThis);
    currentFunctionName_ = std::move(savedFunctionName);
    popFrame();
    --depth_;
    return std::nullopt;
  }

  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = takeReturnValue();

  thisObject_ = std::move(savedThis);
  currentFunctionName_ = std::move(savedFunctionName);
  popFrame();
  --depth_;

  if (aborted_) return std::nullopt;

  return result;
}

auto ASTInterpreter::evaluateCallLValue(FunctionSymbol* func,
                                        std::vector<ConstValue> args)
    -> ConstValue* {
  if (!func || !func->isConstexpr()) return nullptr;

  auto defn = func->definition();
  if (!defn) defn = func;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return nullptr;

  auto body = funcDef->functionBody;
  if (!body) return nullptr;

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody || !compBody->statement) return nullptr;

  if (depth_ >= kMaxDepth) return nullptr;

  ++depth_;
  pushFrame();

  if (!bindParameters(func, args)) {
    popFrame();
    --depth_;
    return nullptr;
  }

  auto savedCapture = std::exchange(captureReturnLValue_, true);
  returnLValue_ = nullptr;
  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = returnLValue_;
  captureReturnLValue_ = savedCapture;
  returnLValue_ = nullptr;

  popFrame();
  --depth_;

  if (aborted_) return nullptr;

  return result;
}

auto ASTInterpreter::evaluateCallExprs(FunctionSymbol* func,
                                       List<ExpressionAST*>* argExprs)
    -> std::optional<ConstValue> {
  if (!func || !func->isConstexpr()) return std::nullopt;

  auto defn = func->definition();
  if (!defn) defn = func;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto body = funcDef->functionBody;
  if (!body) return std::nullopt;

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody || !compBody->statement) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  std::string funcNameStr;
  if (auto id = name_cast<Identifier>(func->name())) {
    funcNameStr = id->value();
  }
  auto savedFunctionName =
      std::exchange(currentFunctionName_, std::move(funcNameStr));

  if (!bindParametersFromExprs(func, argExprs)) {
    currentFunctionName_ = std::move(savedFunctionName);
    popFrame();
    --depth_;
    return std::nullopt;
  }

  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = takeReturnValue();

  currentFunctionName_ = std::move(savedFunctionName);
  popFrame();
  --depth_;

  if (aborted_) return std::nullopt;

  return result;
}

auto ASTInterpreter::evaluateCallLValueFromExprs(FunctionSymbol* func,
                                                 List<ExpressionAST*>* argExprs)
    -> ConstValue* {
  if (!func || !func->isConstexpr()) return nullptr;

  auto defn = func->definition();
  if (!defn) defn = func;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return nullptr;

  auto body = funcDef->functionBody;
  if (!body) return nullptr;

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody || !compBody->statement) return nullptr;

  if (depth_ >= kMaxDepth) return nullptr;

  ++depth_;
  pushFrame();

  if (!bindParametersFromExprs(func, argExprs)) {
    popFrame();
    --depth_;
    return nullptr;
  }

  auto savedCapture = std::exchange(captureReturnLValue_, true);
  returnLValue_ = nullptr;
  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = returnLValue_;
  captureReturnLValue_ = savedCapture;
  returnLValue_ = nullptr;

  popFrame();
  --depth_;

  if (aborted_) return nullptr;

  return result;
}

auto ASTInterpreter::evaluateConstructor(FunctionSymbol* ctor,
                                         const Type* classType,
                                         std::vector<ConstValue> args)
    -> std::optional<ConstValue> {
  if (!ctor || !ctor->isConstexpr()) return std::nullopt;

  auto defn = ctor->definition();
  if (!defn) defn = ctor;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto body = funcDef->functionBody;
  if (!body) return std::nullopt;

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  auto obj = std::make_shared<ConstObject>(classType);
  auto savedThis = thisObject_;
  thisObject_ = obj;
  auto savedConstructorClass = std::exchange(
      currentConstructorClass_,
      symbol_cast<ClassSymbol>(ctor->enclosingNonTemplateParametersScope()));

  if (!bindParameters(ctor, args)) {
    thisObject_ = savedThis;
    currentConstructorClass_ = savedConstructorClass;
    popFrame();
    --depth_;
    return std::nullopt;
  }

  for (auto node : ListView{compBody->memInitializerList}) {
    (void)memInitializer(node);
  }

  returnValue_.reset();
  if (compBody->statement) {
    (void)statement(compBody->statement);
  }

  thisObject_ = savedThis;
  currentConstructorClass_ = savedConstructorClass;
  popFrame();
  --depth_;

  if (aborted_) return std::nullopt;

  return ConstValue{std::move(obj)};
}
}  // namespace cxx
