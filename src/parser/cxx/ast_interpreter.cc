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

#include <algorithm>
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

  auto operator()(IndeterminateValue) const -> T {
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

  auto operator()(IndeterminateValue) const -> std::optional<bool> {
    return std::nullopt;
  }

  auto operator()(const auto& value) const -> std::optional<bool> {
    return bool(value);
  }
};

ASTInterpreter::ASTInterpreter(TranslationUnit* unit)
    : unit_(unit), traits(unit) {}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::cloneValue(const ConstValue& value) -> ConstValue {
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return value;
    auto copy = std::make_shared<ConstObject>((*object)->type());
    for (const auto& field : (*object)->fields())
      copy->addField(field.symbol, cloneValue(field.value));
    for (const auto& base : (*object)->bases()) copy->addBase(cloneValue(base));
    return ConstValue{std::move(copy)};
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return value;
    auto copy = std::make_shared<InitializerList>();
    for (const auto& [element, type] : (*list)->elements)
      copy->elements.emplace_back(cloneValue(element), type);
    return ConstValue{std::move(copy)};
  }

  return value;
}

auto ASTInterpreter::isFullyInitialized(const ConstValue& value) const -> bool {
  if (std::holds_alternative<IndeterminateValue>(value)) return false;

  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return false;
    for (const auto& field : (*object)->fields()) {
      if (!isFullyInitialized(field.value)) return false;
    }
    for (const auto& base : (*object)->bases()) {
      if (!isFullyInitialized(base)) return false;
    }
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return false;
    for (const auto& [element, type] : (*list)->elements) {
      if (!isFullyInitialized(element)) return false;
    }
  }

  return true;
}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::evaluate(ExpressionAST* ast) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  auto result = expression(ast);
  return result;
}

auto ASTInterpreter::evaluateAddress(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  return addressOfLvalue(ast);
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
  frames_.back().refs.insert_or_assign(sym, target);
}

void ASTInterpreter::setLocal(const Symbol* sym, ConstValue value) {
  if (frames_.empty()) frames_.push_back({});
  frames_.back().locals.insert_or_assign(sym, std::move(value));
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
  if (param && traits.is_reference(param->type())) {
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
  auto classType = unqualified_cast<ClassType>(obj->type());
  if (!classType || !classType->symbol()) return;
  auto savedThis = std::exchange(thisObject_, obj);
  for (auto member : classType->symbol()->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic() || !field->initializer()) continue;
    auto value = evaluate(field->initializer());
    if (value) obj->setField(field, std::move(*value));
  }
  thisObject_ = std::move(savedThis);
}

auto ASTInterpreter::initializeDefaultedObject(
    const std::shared_ptr<ConstObject>& obj, ClassSymbol* classSymbol) -> bool {
  if (!obj || !classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();
  auto savedThis = std::exchange(thisObject_, obj);

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    auto value = defaultConstruct(baseClass->type());
    if (!value) {
      thisObject_ = std::move(savedThis);
      return false;
    }
    obj->addBase(std::move(*value));
  }

  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;

    if (field->initializer()) {
      auto value = evaluate(field->initializer());
      if (!value) {
        thisObject_ = std::move(savedThis);
        return false;
      }
      obj->setField(field, std::move(*value));
      if (classSymbol->isUnion()) break;
      continue;
    }

    auto value = defaultConstruct(field->type());
    if (!value) {
      thisObject_ = std::move(savedThis);
      return false;
    }
    obj->setField(field, std::move(*value));
    if (classSymbol->isUnion()) break;
  }

  thisObject_ = std::move(savedThis);
  return true;
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
  EvaluationScope evaluationScope{*this};
  auto unqualified = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqualified)) {
    auto elements = std::make_shared<InitializerList>();
    elements->elements.reserve(arrayType->size());
    for (std::size_t index = 0; index < arrayType->size(); ++index) {
      auto value = defaultConstruct(arrayType->elementType());
      if (!value) return std::nullopt;
      elements->elements.emplace_back(std::move(*value),
                                      arrayType->elementType());
    }
    return ConstValue{std::move(elements)};
  }

  if (!traits.is_class(unqualified)) return ConstValue{IndeterminateValue{}};

  auto classType = type_cast<ClassType>(unqualified);
  if (!classType || !classType->symbol()) return std::nullopt;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  auto constructor = classSymbol->defaultConstructor();
  if (!constructor) return std::nullopt;
  if (!constructor->isConstexpr()) return std::nullopt;
  return evaluateConstructor(constructor, type, {});
}

void ASTInterpreter::pushFrame() { frames_.push_back({}); }

void ASTInterpreter::popFrame() {
  if (!frames_.empty()) frames_.pop_back();
}

void ASTInterpreter::retireFrame() {
  if (frames_.empty()) return;
  retiredFrames_.push_back(std::move(frames_.back()));
  frames_.pop_back();
}

auto ASTInterpreter::beginAutomaticScope() const -> std::size_t {
  if (frames_.empty()) return 0;
  return frames_.back().automaticObjects.size();
}

void ASTInterpreter::registerAutomaticObject(VariableSymbol* variable) {
  if (!variable || variable->isStatic()) return;
  if (frames_.empty()) return;
  auto type = traits.remove_cv(variable->type());
  if (!traits.is_class(type) && !traits.is_array(type)) return;
  frames_.back().automaticObjects.push_back(variable);
}

auto ASTInterpreter::endAutomaticScope(std::size_t mark) -> bool {
  if (frames_.empty()) return true;
  auto& objects = frames_.back().automaticObjects;
  if (mark > objects.size()) return false;
  while (objects.size() > mark) {
    auto variable = objects.back();
    objects.pop_back();
    auto value = lookupLocalSlot(variable);
    if (!value) continue;
    if (destroyValue(variable->type(), *value)) continue;
    aborted_ = true;
    return false;
  }
  return true;
}

auto ASTInterpreter::destroyValue(const Type* type, ConstValue& value) -> bool {
  auto unqual = traits.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqual)) {
    auto elements = std::get_if<std::shared_ptr<InitializerList>>(&value);
    if (!elements || !*elements) return false;
    for (auto it = (*elements)->elements.rbegin();
         it != (*elements)->elements.rend(); ++it) {
      auto& [element, elementType] = *it;
      auto typeToDestroy = elementType;
      if (!typeToDestroy) typeToDestroy = arrayType->elementType();
      if (!destroyValue(typeToDestroy, element)) return false;
    }
    return true;
  }

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return true;
  if (traits.has_trivial_destructor(unqual)) return true;

  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!object || !*object) return false;
  auto classSymbol = classType->definition();
  traits.requireCompleteClass(classSymbol);
  if (!classSymbol || !classSymbol->isComplete()) return false;
  auto destructor = classSymbol->destructor();
  if (!destructor || destructor->isDeleted()) return false;

  if (!destructor->isDefaulted()) {
    if (!destructor->isConstexpr()) return false;
    auto savedReturnValue = std::move(returnValue_);
    auto savedCaptureReturnLValue = std::exchange(captureReturnLValue_, false);
    auto savedReturnLValue = std::exchange(returnLValue_, nullptr);
    auto savedCaptureReturnAddress =
        std::exchange(captureReturnAddress_, false);
    auto savedReturnAddress = std::move(returnAddress_);
    (void)evaluateCall(destructor, {}, *object);
    returnValue_ = std::move(savedReturnValue);
    captureReturnLValue_ = savedCaptureReturnLValue;
    returnLValue_ = savedReturnLValue;
    captureReturnAddress_ = savedCaptureReturnAddress;
    returnAddress_ = std::move(savedReturnAddress);
    if (aborted_) return false;
  }

  auto& fields = (*object)->mutableFields();
  for (auto it = fields.rbegin(); it != fields.rend(); ++it) {
    if (!it->symbol || !it->symbol->type()) continue;
    if (!destroyValue(it->symbol->type(), it->value)) return false;
  }

  auto& bases = (*object)->mutableBases();
  auto baseClasses = classSymbol->baseClasses();
  auto count = std::min(bases.size(), baseClasses.size());
  for (auto index = count; index > 0; --index) {
    auto baseClass = symbol_cast<ClassSymbol>(baseClasses[index - 1]->symbol());
    if (!baseClass) continue;
    if (!destroyValue(baseClass->type(), bases[index - 1])) return false;
  }

  return true;
}

auto ASTInterpreter::evaluateCall(FunctionSymbol* func,
                                  std::vector<ConstValue> args,
                                  std::shared_ptr<ConstObject> thisObject)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
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

  std::string funcNameStr = to_string(func->name());

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

  if (result)
    retireFrame();
  else
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

  std::string funcNameStr = to_string(func->name());

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

  if (result)
    retireFrame();
  else
    popFrame();
  --depth_;

  if (aborted_) return nullptr;

  return result;
}

auto ASTInterpreter::evaluateCallAddressFromExprs(
    FunctionSymbol* func, List<ExpressionAST*>* argExprs)
    -> std::optional<ConstValue> {
  if (!func) return std::nullopt;
  if (!func->isConstexpr()) return std::nullopt;

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
  if (!compBody) return std::nullopt;
  if (!compBody->statement) return std::nullopt;
  if (depth_ >= kMaxDepth) return std::nullopt;

  ++depth_;
  pushFrame();

  if (!bindParametersFromExprs(func, argExprs)) {
    popFrame();
    --depth_;
    return std::nullopt;
  }

  auto savedCapture = std::exchange(captureReturnAddress_, true);
  auto savedAddress = std::move(returnAddress_);
  returnAddress_.reset();
  returnValue_.reset();
  (void)statement(compBody->statement);

  auto result = std::move(returnAddress_);
  returnAddress_ = std::move(savedAddress);
  captureReturnAddress_ = savedCapture;

  popFrame();
  --depth_;

  if (aborted_) return std::nullopt;
  return result;
}

auto ASTInterpreter::evaluateConstructor(FunctionSymbol* ctor,
                                         const Type* classType,
                                         std::vector<ConstValue> args)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (!ctor) return std::nullopt;
  if (!ctor->isConstexpr()) return std::nullopt;

  auto defn = ctor->definition();
  if (!defn) defn = ctor;

  if (defn->hasPendingBody()) {
    ASTRewriter::completePendingBodyFor(unit_, defn);
  }

  auto funcDef = defn->declaration();
  if (!funcDef) return std::nullopt;

  auto body = funcDef->functionBody;
  if (!body) return std::nullopt;

  if (depth_ >= kMaxDepth) return std::nullopt;

  if (ast_cast<DefaultFunctionBodyAST>(body)) {
    auto classSymbol = symbol_cast<ClassSymbol>(ctor->parent());
    if (!classSymbol) return std::nullopt;
    classSymbol = classSymbol->resolvedDefinition();

    auto obj = std::make_shared<ConstObject>(classType);
    auto defaultConstructor = classSymbol->defaultConstructor();
    auto isDefaultConstructor = defaultConstructor != nullptr;
    if (isDefaultConstructor)
      isDefaultConstructor =
          ctor->canonical() == defaultConstructor->canonical();
    if (isDefaultConstructor) {
      ++depth_;
      auto initialized = initializeDefaultedObject(obj, classSymbol);
      --depth_;
      if (!initialized) return std::nullopt;
      return ConstValue{std::move(obj)};
    }

    auto copy = classSymbol->copyConstructor();
    auto move = classSymbol->moveConstructor();
    auto canonical = ctor->canonical();
    auto copiesObject = copy && canonical == copy->canonical();
    if (move && canonical == move->canonical()) copiesObject = true;
    if (!copiesObject || args.size() != 1) return std::nullopt;
    auto source = std::get_if<std::shared_ptr<ConstObject>>(&args.front());
    if (!source || !*source) return std::nullopt;
    return cloneValue(args.front());
  }

  auto compBody = ast_cast<CompoundStatementFunctionBodyAST>(body);
  if (!compBody) return std::nullopt;

  ++depth_;
  pushFrame();

  auto obj = std::make_shared<ConstObject>(classType);
  auto savedThis = thisObject_;
  thisObject_ = obj;
  auto savedConstructorClass = std::exchange(
      currentConstructorClass_, symbol_cast<ClassSymbol>(ctor->parent()));

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
