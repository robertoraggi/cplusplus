#include <cxx/mlir/mlir_emitter.h>
#include <cxx/mlir/mlir_source_contexts.h>
#include <cxx/mlir/mlir_source_region.h>
#include <cxx/time_trace.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <algorithm>
#include <array>
#include <deque>
#include <utility>

namespace cxx::ir {
namespace {
struct OwnedOperation {
  mlir::Operation* operation;
  std::uint64_t owner;
};

struct Context {
  std::uint64_t id = 0;
  std::uint64_t parent = 0;
  std::uint64_t firstChild = 0;
  std::uint64_t nextSibling = 0;
  SourceContextKind kind = SyntheticSourceContext::Function;
  bool sealed = false;
  bool hasOperations = false;
  bool hasCompletion = false;
  bool hasGoto = false;
  bool hasException = false;
  std::size_t subtreeBegin = 0;
  std::size_t subtreeEnd = 0;
  std::uint64_t closestCandidate = 0;
  std::uint64_t parentCandidate = 0;
  unsigned level = 0;
  llvm::SmallVector<OwnedOperation*> members;
  std::array<std::uint64_t, std::to_underlying(SourceBlockRole::ExitTarget) + 1>
      roles{};

  [[nodiscard]] auto role(SourceBlockRole role) const -> std::uint64_t {
    return roles[std::to_underlying(role)];
  }

  [[nodiscard]] auto contains(const Context& other) const -> bool {
    return subtreeBegin <= other.subtreeBegin &&
           other.subtreeBegin < subtreeEnd;
  }
};

struct Function {
  FunctionRef ref;
  mlir::cxx::FuncOp operation;
  std::vector<Context> contexts;
  std::deque<OwnedOperation> operations;
  llvm::DenseMap<std::uint64_t, std::size_t> indices;
  llvm::DenseMap<std::uint64_t, bool> lifetimes;
  std::uint64_t current = 0;
  std::int64_t invalidEvents = 0;

  [[nodiscard]] auto context(std::uint64_t id) -> Context* {
    auto it = indices.find(id);
    if (it == indices.end()) return nullptr;
    return &contexts[it->second];
  }

  [[nodiscard]] auto treeOrder() -> llvm::SmallVector<std::uint64_t> {
    llvm::SmallVector<std::uint64_t> order;
    llvm::SmallVector<std::uint64_t> pending{contexts.front().id};
    while (!pending.empty()) {
      auto* current = context(pending.pop_back_val());
      current->subtreeBegin = order.size();
      current->subtreeEnd = order.size() + 1;
      order.push_back(current->id);
      for (auto child = current->firstChild; child;
           child = context(child)->nextSibling)
        pending.push_back(child);
    }
    for (auto id : llvm::reverse(order)) {
      auto* current = context(id);
      if (auto* parent = context(current->parent))
        parent->subtreeEnd = std::max(parent->subtreeEnd, current->subtreeEnd);
    }
    return order;
  }

  [[nodiscard]] auto surface(Context& current)
      -> llvm::SmallVector<mlir::Operation*> {
    llvm::SmallVector<mlir::Operation*> result;
    for (auto* record : current.members) {
      auto* op = record->operation;
      if (op && op->getParentRegion() == &operation.getBody())
        result.push_back(op);
    }
    return result;
  }

  void promote(Context& current) {
    if (auto* parent = context(current.parentCandidate)) {
      for (auto* record : current.members) {
        auto* op = record->operation;
        if (op && op->getParentRegion() == &operation.getBody())
          parent->members.push_back(record);
      }
    }
    current.members.clear();
  }
};

using ContextPlan =
    std::variant<SourceIfPlan, SourceLoopPlan, SourceRegionPlan>;

struct PendingContext {
  std::uint64_t id;
  ContextPlan plan;
  llvm::SmallVector<mlir::Operation*> targets;
};

struct FinalizeContext {
  MlirSourceContexts& listener;
  TimeTrace* timeTrace;

  void operator()(const SourceIfPlan& plan) const {
    TimeTrace::Scope trace(timeTrace, "SCF if");
    finalizeSourceIf(plan, &listener);
  }

  void operator()(const SourceLoopPlan& plan) const {
    TimeTrace::Scope trace(timeTrace, "SCF loop");
    finalizeSourceLoop(plan, &listener);
  }

  void operator()(const SourceRegionPlan& plan) const {
    TimeTrace::Scope trace(timeTrace, "SCF switch");
    finalizeSourceScope(plan, &listener);
  }
};
}

struct MlirSourceContexts::Data {
  MlirEmitter& emitter;
  std::deque<Function> functions;
  llvm::DenseMap<mlir::Operation*, OwnedOperation*> owners;
  llvm::DenseMap<mlir::Block*, std::uint64_t> blocks;
  std::uint64_t nextId = 1;
  std::int64_t invalidEvents = 0;
  SourceContextOptions options;

  Data(MlirEmitter& emitter, SourceContextOptions options)
      : emitter(emitter), options(options) {}

  void analyze(Function& function, llvm::ArrayRef<std::uint64_t> order) {
    TimeTrace::Scope trace(options.timeTrace, "SCF analyze");
    function.operation.getBody().walk([&](mlir::Operation* op) {
      auto* record = owners.lookup(op);
      if (!record) {
        ++function.invalidEvents;
        return;
      }
      auto* context = function.context(record->owner);
      context->hasOperations = true;
      context->hasException |=
          mlir::isa<mlir::cxx::InvokeOp, mlir::cxx::LandingPadOp,
                    mlir::cxx::ResumeOp>(op);
      context->hasGoto |=
          mlir::isa<mlir::cxx::IndirectGotoOp, mlir::cxx::AsmGotoOp,
                    mlir::cxx::LabelAddressOp>(op);
    });
    for (auto id : llvm::reverse(order)) {
      auto& context = *function.context(id);
      if (auto* kind = std::get_if<ASTKind>(&context.kind);
          kind && context.hasOperations) {
        context.hasCompletion |= *kind == ASTKind::BreakStatement ||
                                 *kind == ASTKind::ContinueStatement ||
                                 *kind == ASTKind::ReturnStatement;
        context.hasGoto |= *kind == ASTKind::GotoStatement ||
                           *kind == ASTKind::LabeledStatement;
      }
      auto* parent = function.context(context.parent);
      if (!parent) continue;
      parent->hasOperations |= context.hasOperations;
      parent->hasException |= context.hasException;
      parent->hasCompletion |= context.hasCompletion;
      parent->hasGoto |= context.hasGoto;
    }
  }

  struct IsCandidate {
    SourceContextOptions::Mode mode;

    [[nodiscard]] auto operator()(ASTKind kind) const -> bool {
      if (kind == ASTKind::IfStatement) return true;
      if (mode != SourceContextOptions::Mode::Structured) return false;
      switch (kind) {
        case ASTKind::SwitchStatement:
        case ASTKind::DoStatement:
        case ASTKind::WhileStatement:
        case ASTKind::ForStatement:
          return true;
        default:
          return false;
      }
    }

    [[nodiscard]] auto operator()(SyntheticSourceContext kind) const -> bool {
      return mode == SourceContextOptions::Mode::Structured &&
             kind == SyntheticSourceContext::ArrayCleanup;
    }
  };

  [[nodiscard]] auto collectCandidates(Function& function,
                                       llvm::ArrayRef<std::uint64_t> order)
      -> std::vector<llvm::SmallVector<std::uint64_t>> {
    std::vector<llvm::SmallVector<std::uint64_t>> levels;
    for (auto id : order) {
      auto& context = *function.context(id);
      if (auto* parent = function.context(context.parent))
        context.closestCandidate = parent->closestCandidate;
      if (!context.role(SourceBlockRole::Continuation) || context.hasGoto)
        continue;
      if (context.hasCompletion &&
          options.mode != SourceContextOptions::Mode::Structured)
        continue;
      if (!std::visit(IsCandidate{options.mode}, context.kind)) continue;
      context.parentCandidate = context.closestCandidate;
      if (auto* parent = function.context(context.parentCandidate))
        context.level = parent->level + 1;
      context.closestCandidate = context.id;
      if (levels.size() <= context.level) levels.resize(context.level + 1);
      levels[context.level].push_back(context.id);
    }
    for (auto& record : function.operations) {
      if (!record.operation) continue;
      auto candidate = function.context(record.owner)->closestCandidate;
      if (auto* context = function.context(candidate))
        context->members.push_back(&record);
    }
    return levels;
  }

  [[nodiscard]] auto preflightIf(
      Function& function, const Context& context, const SourceRegionPlan& plan,
      llvm::ArrayRef<mlir::Operation*> operations,
      const llvm::DenseMap<std::uint64_t, mlir::Block*>& liveBlocks,
      mlir::DominanceInfo& dominance) -> SourceIfPreflight {
    const Context* conditionContext = nullptr;
    for (auto id = context.firstChild; id;
         id = function.context(id)->nextSibling) {
      auto* child = function.context(id);
      auto* kind = std::get_if<SyntheticSourceContext>(&child->kind);
      if (!kind || *kind != SyntheticSourceContext::Condition) continue;
      if (conditionContext) return std::string("multiple-if-conditions");
      conditionContext = child;
    }
    if (!conditionContext) return std::string("missing-if-condition");
    llvm::SmallVector<mlir::Operation*> condition;
    for (auto* op : operations) {
      auto* record = owners.lookup(op);
      if (record &&
          conditionContext->contains(*function.context(record->owner)))
        condition.push_back(op);
    }
    auto* thenBlock = liveBlocks.lookup(context.role(SourceBlockRole::Then));
    auto* elseBlock = liveBlocks.lookup(context.role(SourceBlockRole::Else));
    auto* join = liveBlocks.lookup(context.role(SourceBlockRole::Continuation));
    return preflightSourceIf(plan, condition, thenBlock, elseBlock, join,
                             dominance);
  }

  struct PlanContext {
    Data& data;
    Function& function;
    const Context& context;
    SourceRegionPlan& region;
    llvm::ArrayRef<mlir::Operation*> operations;
    const llvm::DenseMap<std::uint64_t, mlir::Block*>& liveBlocks;
    mlir::DominanceInfo& dominance;

    [[nodiscard]] auto operator()(ASTKind kind) -> std::optional<ContextPlan> {
      if (kind == ASTKind::IfStatement) {
        auto checked = data.preflightIf(function, context, region, operations,
                                        liveBlocks, dominance);
        if (auto* plan = std::get_if<SourceIfPlan>(&checked))
          return std::move(*plan);
        return std::nullopt;
      }
      if (kind == ASTKind::SwitchStatement) return std::move(region);
      if (kind == ASTKind::DoStatement) return loop(SourceBlockRole::Body);
      return loop(SourceBlockRole::Condition);
    }

    [[nodiscard]] auto operator()(SyntheticSourceContext)
        -> std::optional<ContextPlan> {
      return loop(SourceBlockRole::Condition);
    }

    [[nodiscard]] auto loop(SourceBlockRole headerRole)
        -> std::optional<ContextPlan> {
      auto* header = liveBlocks.lookup(context.role(headerRole));
      auto* continuation =
          liveBlocks.lookup(context.role(SourceBlockRole::Continuation));
      auto checked =
          preflightSourceLoop(operations, header, continuation, dominance);
      if (auto* plan = std::get_if<SourceLoopPlan>(&checked))
        return std::move(*plan);
      return std::nullopt;
    }
  };

  [[nodiscard]] auto planContext(
      Function& function, Context& context,
      const llvm::DenseMap<std::uint64_t, mlir::Block*>& liveBlocks,
      mlir::DominanceInfo& dominance) -> std::optional<PendingContext> {
    auto operations = function.surface(context);
    auto checked = preflightSourceRegion(operations, dominance);
    auto* region = std::get_if<SourceRegionPlan>(&checked);
    if (!region) return std::nullopt;
    llvm::SmallVector<mlir::Operation*> targets;
    for (auto exit : region->exits) targets.push_back(exit.target);
    auto plan = std::visit(PlanContext{*this, function, context, *region,
                                       operations, liveBlocks, dominance},
                           context.kind);
    if (!plan) return std::nullopt;
    return PendingContext{context.id, std::move(*plan), std::move(targets)};
  }

  [[nodiscard]] auto orderPlans(Function& function,
                                llvm::ArrayRef<PendingContext> plans)
      -> llvm::SmallVector<unsigned> {
    llvm::DenseMap<mlir::Operation*, unsigned> membership;
    for (auto [index, plan] : llvm::enumerate(plans)) {
      for (auto* op : function.surface(*function.context(plan.id)))
        membership.try_emplace(op, index);
    }
    std::vector<llvm::SmallVector<unsigned>> successors(plans.size());
    llvm::SmallVector<unsigned> predecessors(plans.size(), 0);
    for (auto [index, plan] : llvm::enumerate(plans)) {
      for (auto* target : plan.targets) {
        auto found = membership.find(target);
        if (found == membership.end() || found->second == index) continue;
        successors[index].push_back(found->second);
        ++predecessors[found->second];
      }
    }
    llvm::SmallVector<unsigned> order;
    for (auto [index, count] : llvm::enumerate(predecessors)) {
      if (!count) order.push_back(index);
    }
    for (std::size_t next = 0; next < order.size(); ++next) {
      for (auto successor : successors[order[next]]) {
        if (!--predecessors[successor]) order.push_back(successor);
      }
    }
    return order;
  }

  void finalize(Function& function, const PendingContext& pending,
                MlirSourceContexts& listener) {
    const auto previous = function.current;
    function.current = pending.id;
    std::visit(FinalizeContext{listener, options.timeTrace}, pending.plan);
    function.current = previous;
  }

  void preflight(Function& function, llvm::ArrayRef<std::uint64_t> order,
                 MlirSourceContexts& listener) {
    TimeTrace::Scope trace(options.timeTrace, "SCF preflight");
    if (invalidEvents || function.invalidEvents) return;
    for (const auto& context : function.contexts) {
      if (!context.sealed) return;
    }
    for (const auto& lifetime : function.lifetimes) {
      if (!lifetime.second) return;
    }
    auto levels = collectCandidates(function, order);
    llvm::DenseMap<std::uint64_t, mlir::Block*> liveBlocks;
    function.operation.getBody().walk(
        [&](mlir::Block* block) { liveBlocks[blocks.lookup(block)] = block; });
    mlir::DominanceInfo dominance(function.operation);
    for (auto& level : llvm::reverse(levels)) {
      std::vector<PendingContext> plans;
      for (auto id : level) {
        if (auto plan = planContext(function, *function.context(id), liveBlocks,
                                    dominance))
          plans.push_back(std::move(*plan));
      }
      auto schedule = orderPlans(function, plans);
      if (schedule.size() == plans.size()) {
        for (auto index : schedule) finalize(function, plans[index], listener);
        if (!plans.empty()) dominance.invalidate();
      } else {
        for (auto id : level) {
          auto plan = planContext(function, *function.context(id), liveBlocks,
                                  dominance);
          if (!plan) continue;
          finalize(function, *plan, listener);
          dominance.invalidate();
        }
      }
      for (auto id : level) function.promote(*function.context(id));
    }
  }

  [[nodiscard]] auto active(mlir::Operation* op) -> Function* {
    if (functions.empty()) return nullptr;
    auto function = op->getParentOfType<mlir::cxx::FuncOp>();
    for (auto it = functions.rbegin(); it != functions.rend(); ++it) {
      if (function == it->operation) return &*it;
    }
    return nullptr;
  }
};

MlirSourceContexts::MlirSourceContexts(MlirEmitter& emitter,
                                       SourceContextOptions options)
    : data_(std::make_unique<Data>(emitter, options)) {}
MlirSourceContexts::~MlirSourceContexts() = default;

void MlirSourceContexts::beginFunction(FunctionRef ref) {
  Function function;
  function.ref = ref;
  function.operation = data_->emitter.function(ref);
  function.current = data_->nextId++;
  function.indices[function.current] = 0;
  Context context;
  context.id = function.current;
  function.contexts.push_back(std::move(context));
  data_->functions.push_back(std::move(function));
}

auto MlirSourceContexts::beginContext(SourceContextKind kind) -> std::uint64_t {
  if (data_->functions.empty()) {
    ++data_->invalidEvents;
    return 0;
  }
  auto& function = data_->functions.back();
  Context context;
  context.id = data_->nextId++;
  context.parent = function.current;
  context.kind = kind;
  auto* parent = function.context(function.current);
  context.nextSibling = parent->firstChild;
  parent->firstChild = context.id;
  function.indices[context.id] = function.contexts.size();
  function.current = context.id;
  function.contexts.push_back(std::move(context));
  return function.current;
}

void MlirSourceContexts::endContext(std::uint64_t id) {
  if (data_->functions.empty()) {
    ++data_->invalidEvents;
    return;
  }
  auto& function = data_->functions.back();
  auto* context = function.context(id);
  if (!context || function.current != id || context->sealed) {
    ++function.invalidEvents;
    return;
  }
  context->sealed = true;
  function.current = context->parent;
}

auto MlirSourceContexts::currentContext() const -> std::uint64_t {
  return data_->functions.empty() ? 0 : data_->functions.back().current;
}

void MlirSourceContexts::setContext(std::uint64_t id) {
  if (data_->functions.empty()) {
    ++data_->invalidEvents;
    return;
  }
  auto& function = data_->functions.back();
  if (!function.context(id)) {
    ++function.invalidEvents;
    return;
  }
  function.current = id;
}

auto MlirSourceContexts::contextOf(mlir::Operation* operation) const
    -> std::uint64_t {
  auto it = data_->owners.find(operation);
  return it == data_->owners.end() ? currentContext() : it->second->owner;
}

void MlirSourceContexts::recordBlock(SourceBlockRole role, BlockRef ref) {
  if (data_->functions.empty()) return;
  auto* block = data_->emitter.block(ref);
  blockCreated(block);
  auto& function = data_->functions.back();
  auto* context = function.context(function.current);
  auto& target = context->roles[std::to_underlying(role)];
  if (!target) target = data_->blocks.lookup(block);
}

auto MlirSourceContexts::beginLifetime() -> std::uint64_t {
  if (data_->functions.empty()) {
    ++data_->invalidEvents;
    return 0;
  }
  auto& function = data_->functions.back();
  const auto id = data_->nextId++;
  function.lifetimes.try_emplace(id, false);
  return id;
}

void MlirSourceContexts::endLifetime(std::uint64_t id) {
  if (data_->functions.empty()) {
    ++data_->invalidEvents;
    return;
  }
  auto& function = data_->functions.back();
  auto it = function.lifetimes.find(id);
  if (it == function.lifetimes.end() || it->second) {
    ++function.invalidEvents;
    return;
  }
  it->second = true;
}

void MlirSourceContexts::blockCreated(mlir::Block* block) {
  if (data_->functions.empty() || !block || data_->blocks.contains(block))
    return;
  data_->blocks[block] = data_->nextId++;
}

void MlirSourceContexts::notifyBlockInserted(mlir::Block* block, mlir::Region*,
                                             mlir::Region::iterator) {
  blockCreated(block);
}

void MlirSourceContexts::notifyOperationInserted(mlir::Operation* op,
                                                 mlir::OpBuilder::InsertPoint) {
  auto* function = data_->active(op);
  if (!function || data_->owners.contains(op)) return;
  const auto owner = mlir::isa<mlir::cxx::AllocaOp>(op)
                         ? function->contexts.front().id
                         : function->current;
  auto& record = function->operations.emplace_back(op, owner);
  data_->owners[op] = &record;
  auto candidate = function->context(owner)->closestCandidate;
  if (auto* context = function->context(candidate))
    context->members.push_back(&record);
  blockCreated(op->getBlock());
}

void MlirSourceContexts::notifyOperationErased(mlir::Operation* op) {
  auto it = data_->owners.find(op);
  if (it == data_->owners.end()) return;
  it->second->operation = nullptr;
  data_->owners.erase(it);
}

void MlirSourceContexts::notifyBlockErased(mlir::Block* block) {
  block->walk([&](mlir::Operation* op) { notifyOperationErased(op); });
  data_->blocks.erase(block);
}

void MlirSourceContexts::endFunction(FunctionRef ref) {
  TimeTrace::Scope trace(data_->options.timeTrace, "SCF contexts");
  if (data_->functions.empty() || data_->functions.back().ref != ref) {
    ++data_->invalidEvents;
    return;
  }
  auto& function = data_->functions.back();
  auto& root = function.contexts.front();
  if (function.current != root.id) ++function.invalidEvents;
  root.sealed = true;
  auto order = function.treeOrder();
  data_->analyze(function, order);
  data_->preflight(function, order, *this);
  if (data_->options.mode == SourceContextOptions::Mode::Structured &&
      (root.hasGoto || root.hasException))
    function.operation->setAttr(
        "cxx.source_scf_fallback",
        mlir::UnitAttr::get(function.operation.getContext()));
  for (auto& record : function.operations) {
    if (record.operation) data_->owners.erase(record.operation);
  }
  function.operation.getBody().walk(
      [&](mlir::Block* block) { data_->blocks.erase(block); });
  data_->functions.pop_back();
}
}
