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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <optional>
#include <ranges>
#include <tuple>
#include <vector>

namespace cxx {

namespace {

struct Subobject {
  ClassSymbol* classSymbol = nullptr;
  std::uint64_t offset = 0;
  bool isVirtual = false;
  std::vector<std::size_t> bases;
  std::vector<std::size_t> derived;
};

class SubobjectGraph {
 public:
  SubobjectGraph(ClassSymbol* completeClass, ClassSymbol* root,
                 std::uint64_t rootOffset, bool rootIsVirtual)
      : completeLayout_(completeClass->layout()) {
    nodes_.push_back({.classSymbol = root,
                      .offset = rootOffset,
                      .isVirtual = rootIsVirtual});
    preorder_.push_back(0);

    struct Frame {
      std::size_t node = 0;
      std::size_t nextBase = 0;
    };

    std::vector<Frame> frames{{0}};
    while (!frames.empty()) {
      const auto parent = frames.back().node;
      const auto& bases = nodes_[parent].classSymbol->baseClasses();
      if (frames.back().nextBase == bases.size()) {
        frames.pop_back();
        continue;
      }
      auto base = bases[frames.back().nextBase++];
      auto baseClass = resolved_base_class(base);
      if (!baseClass) continue;

      auto [child, created] = base->isVirtual()
                                  ? virtualNode(baseClass)
                                  : nonVirtualNode(parent, baseClass);
      nodes_[parent].bases.push_back(child);
      nodes_[child].derived.push_back(parent);
      if (!created) continue;
      preorder_.push_back(child);
      frames.push_back({child});
    }
  }

  [[nodiscard]] auto node(std::size_t index) const -> const Subobject& {
    return nodes_[index];
  }

  [[nodiscard]] auto preorder() const -> const std::vector<std::size_t>& {
    return preorder_;
  }

  [[nodiscard]] auto virtualNodeOf(ClassSymbol* classSymbol) const
      -> std::optional<std::size_t> {
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
      if (nodes_[index].isVirtual && nodes_[index].classSymbol == classSymbol)
        return index;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto primaryBaseOf(std::size_t index) const
      -> std::optional<std::size_t> {
    auto layout = nodes_[index].classSymbol->layout();
    if (!layout || !layout->primaryBase()) return std::nullopt;
    auto primary = layout->primaryBase();
    if (layout->primaryBaseIsVirtual()) return virtualNodeOf(primary);
    for (auto base : nodes_[index].bases) {
      if (nodes_[base].isVirtual) continue;
      if (nodes_[base].classSymbol == primary) return base;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto isNonVirtualPrimaryBase(std::size_t index) const -> bool {
    if (nodes_[index].isVirtual || nodes_[index].derived.empty()) return false;
    return primaryBaseOf(nodes_[index].derived.front()) == index;
  }

  [[nodiscard]] auto sharesVirtualPointer(std::size_t index) const -> bool {
    const auto& subobject = nodes_[index];
    for (std::size_t other = 0; other < nodes_.size(); ++other) {
      if (other == index) continue;
      if (nodes_[other].offset != subobject.offset) continue;
      auto layout = nodes_[other].classSymbol->layout();
      if (!layout || !layout->primaryBaseIsVirtual()) continue;
      if (layout->primaryBase() == subobject.classSymbol) return true;
    }
    return false;
  }

  [[nodiscard]] auto isReachableAlongVirtualPath(std::size_t index) const
      -> bool {
    for (auto current = index; current != 0;) {
      if (nodes_[current].isVirtual) return true;
      current = nodes_[current].derived.front();
    }
    return false;
  }

  [[nodiscard]] auto enclosingVirtualBase(std::size_t index) const
      -> ClassSymbol* {
    for (auto current = index; current != 0;) {
      if (nodes_[current].isVirtual) return nodes_[current].classSymbol;
      current = nodes_[current].derived.front();
    }
    return nullptr;
  }

  [[nodiscard]] auto contains(std::size_t outer, std::size_t inner) const
      -> bool {
    std::vector<std::size_t> pending{outer};
    std::vector<std::size_t> visited;
    while (!pending.empty()) {
      auto current = pending.back();
      pending.pop_back();
      if (current == inner) return true;
      if (std::ranges::contains(visited, current)) continue;
      visited.push_back(current);
      for (auto base : nodes_[current].bases) pending.push_back(base);
    }
    return false;
  }

  [[nodiscard]] auto containers(std::size_t index) const
      -> std::vector<std::size_t> {
    std::vector<std::size_t> result{index};
    for (std::size_t next = 0; next < result.size(); ++next) {
      for (auto derived : nodes_[result[next]].derived) {
        if (!std::ranges::contains(result, derived)) result.push_back(derived);
      }
    }
    return result;
  }

 private:
  [[nodiscard]] auto nonVirtualNode(std::size_t parent, ClassSymbol* base)
      -> std::pair<std::size_t, bool> {
    std::uint64_t offset = nodes_[parent].offset;
    if (auto layout = nodes_[parent].classSymbol->layout()) {
      if (auto info = layout->getBaseInfo(base)) offset += info->offset;
    }
    nodes_.push_back({.classSymbol = base, .offset = offset});
    return {nodes_.size() - 1, true};
  }

  [[nodiscard]] auto virtualNode(ClassSymbol* base)
      -> std::pair<std::size_t, bool> {
    if (auto existing = virtualNodeOf(base)) return {*existing, false};
    std::uint64_t offset = 0;
    if (completeLayout_) {
      if (auto info = completeLayout_->getVirtualBaseInfo(base))
        offset = info->offset;
    }
    nodes_.push_back(
        {.classSymbol = base, .offset = offset, .isVirtual = true});
    return {nodes_.size() - 1, true};
  }

  const ClassLayout* completeLayout_ = nullptr;
  std::vector<Subobject> nodes_;
  std::vector<std::size_t> preorder_;
};

[[nodiscard]] auto virtualFunctionDeclarationOrder(FunctionSymbol* function) {
  auto declaration = function->canonical()->declaration();
  const bool synthesized = declaration && !declaration->firstSourceLocation();
  return std::tuple{synthesized, function->canonical()->location()};
}

[[nodiscard]] auto declaredVirtualFunctions(ClassSymbol* classSymbol)
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> functions;
  for (auto member : classSymbol->members()) {
    for (auto function : views::declared_functions(member)) {
      if (function->parent() != classSymbol) continue;
      if (!function->isVirtual() || function->isConsteval()) continue;
      functions.push_back(function);
    }
  }
  std::ranges::stable_sort(functions, {}, virtualFunctionDeclarationOrder);
  return functions;
}

[[nodiscard]] auto declaringClassOf(FunctionSymbol* function) -> ClassSymbol* {
  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  if (!classSymbol) return nullptr;
  return classSymbol->resolvedDefinition();
}

[[nodiscard]] auto returnedClassOf(TypeTraits& traits, FunctionSymbol* function)
    -> ClassSymbol* {
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return nullptr;
  auto returnType = functionType->returnType();
  const Type* pointee = nullptr;
  if (traits.is_pointer(returnType))
    pointee = traits.get_element_type(returnType);
  else if (traits.is_reference(returnType))
    pointee = traits.remove_reference(returnType);
  auto classType = unqualified_cast<ClassType>(pointee);
  if (!classType || !classType->symbol()) return nullptr;
  return classType->symbol()->resolvedDefinition();
}

struct BasePath {
  ClassSymbol* virtualBase = nullptr;
  std::int64_t offset = 0;
};

[[nodiscard]] auto findBasePath(ClassSymbol* derived, ClassSymbol* base)
    -> std::optional<BasePath> {
  struct Step {
    ClassSymbol* classSymbol;
    BasePath path;
  };

  std::vector<Step> pending{{derived, {}}};
  while (!pending.empty()) {
    auto [current, path] = pending.back();
    pending.pop_back();
    if (current == base) return path;
    auto layout = current->layout();
    if (!layout) continue;
    for (auto baseClass : current->baseClasses() | std::views::reverse) {
      auto next = resolved_base_class(baseClass);
      if (!next) continue;
      if (baseClass->isVirtual()) {
        pending.push_back({next, {.virtualBase = next, .offset = 0}});
        continue;
      }
      auto info = layout->getBaseInfo(next);
      if (!info) continue;
      pending.push_back(
          {next,
           {.virtualBase = path.virtualBase,
            .offset = path.offset + static_cast<std::int64_t>(info->offset)}});
    }
  }
  return std::nullopt;
}

}  // namespace

struct Binder::BuildVTableLayout {
  struct VcallEntry {
    FunctionSymbol* function = nullptr;
    std::size_t node = 0;
  };

  struct Segment {
    std::vector<VcallEntry> vcalls;
    std::vector<ClassSymbol*> virtualBases;
  };

  struct Overrider {
    FunctionSymbol* function = nullptr;
    std::size_t node = 0;
  };

  Binder& binder;
  ClassSymbol* classSymbol;
  std::unique_ptr<VTableLayout> layout = std::make_unique<VTableLayout>();
  std::vector<VTableLayout::Slot> primarySlots;
  VTableLayout::Table primaryHeader;
  std::int64_t wordSize = 0;
  bool reportAmbiguities = false;
  std::vector<FunctionSymbol*> reportedAmbiguities;

  BuildVTableLayout(Binder& b, ClassSymbol* cls)
      : binder(b),
        classSymbol(cls),
        wordSize(static_cast<std::int64_t>(
            b.control()->memoryLayout()->sizeOfPointer())) {}

  void operator()() {
    buildPrimarySlots();

    SubobjectGraph graph{classSymbol, classSymbol, 0, false};

    reportAmbiguities = true;
    layout->main = buildGroup(graph);
    reportAmbiguities = false;

    buildAdjustingEntryPoints(graph);

    buildVTT();

    classSymbol->setVTableLayout(std::move(layout));
  }

  [[nodiscard]] auto requiresReturnAdjustment(FunctionSymbol* overrider,
                                              FunctionSymbol* overridden) const
      -> bool {
    auto from = returnedClassOf(binder.traits, overrider);
    auto to = returnedClassOf(binder.traits, overridden);
    if (!from || !to || from == to) return false;
    auto path = findBasePath(from, to);
    if (!path) return false;
    return path->virtualBase || path->offset != 0;
  }

  void buildPrimarySlots() {
    auto classLayout = classSymbol->layout();
    if (auto primaryBase = classLayout->primaryBase()) {
      if (auto primaryVTable = primaryBase->vtableLayout())
        primarySlots = primaryVTable->primary().slots;
    }

    for (auto function : declaredVirtualFunctions(classSymbol))
      addSlotsFor(function);
  }

  void addSlotsFor(FunctionSymbol* function) {
    if (!layout->keyFunction && !function->isPure() && !function->isInline())
      layout->keyFunction = function;

    if (function->isDestructor()) {
      addDestructorSlots(function);
      return;
    }

    for (std::size_t index = 0; index < primarySlots.size(); ++index) {
      const auto& slot = primarySlots[index];
      if (slot.kind != VTableLayout::SlotKind::kFunction) continue;
      if (!function->overrides(slot.introducingFunction)) continue;
      if (requiresReturnAdjustment(function, slot.introducingFunction))
        continue;
      function->setVtableSlotIndex(static_cast<int>(index));
      return;
    }

    function->setVtableSlotIndex(static_cast<int>(primarySlots.size()));
    primarySlots.push_back(
        {.function = function, .introducingFunction = function});
  }

  void addDestructorSlots(FunctionSymbol* destructor) {
    for (std::size_t index = 0; index < primarySlots.size(); ++index) {
      if (primarySlots[index].kind != VTableLayout::SlotKind::kCompleteDtor)
        continue;
      destructor->setVtableSlotIndex(static_cast<int>(index));
      return;
    }

    destructor->setVtableSlotIndex(static_cast<int>(primarySlots.size()));
    primarySlots.push_back({.function = destructor,
                            .kind = VTableLayout::SlotKind::kCompleteDtor,
                            .introducingFunction = destructor});
    primarySlots.push_back({.function = destructor,
                            .kind = VTableLayout::SlotKind::kDeletingDtor,
                            .introducingFunction = destructor});
  }

  [[nodiscard]] auto slotTemplate(ClassSymbol* tableClass) const
      -> const std::vector<VTableLayout::Slot>& {
    if (tableClass == classSymbol) return primarySlots;
    return tableClass->vtableLayout()->primary().slots;
  }

  [[nodiscard]] auto hasVirtualPointer(ClassSymbol* tableClass) const -> bool {
    auto classLayout = tableClass->layout();
    return classLayout && classLayout->hasVtable();
  }

  [[nodiscard]] auto finalOverrider(const SubobjectGraph& graph,
                                    std::size_t subobject,
                                    FunctionSymbol* function)
      -> std::optional<Overrider> {
    std::vector<Overrider> candidates;
    for (auto container : graph.containers(subobject)) {
      for (auto candidate :
           declaredVirtualFunctions(graph.node(container).classSymbol)) {
        if (candidate != function && !candidate->overrides(function)) continue;
        candidates.push_back({candidate, container});
      }
    }

    std::vector<Overrider> finals;
    for (const auto& candidate : candidates) {
      const auto dominated =
          std::ranges::any_of(candidates, [&](const Overrider& other) {
            return other.node != candidate.node &&
                   graph.contains(other.node, candidate.node);
          });
      if (dominated) continue;
      if (std::ranges::contains(finals, candidate.function,
                                &Overrider::function))
        continue;
      finals.push_back(candidate);
    }

    if (finals.size() == 1) return finals.front();
    if (finals.empty() || !reportAmbiguities) return std::nullopt;
    if (std::ranges::contains(reportedAmbiguities, function))
      return std::nullopt;

    reportedAmbiguities.push_back(function);
    binder.error(classSymbol->location(),
                 std::format("virtual function '{}' has more than one final "
                             "overrider in '{}'",
                             to_string(function->name()),
                             to_string(classSymbol->name())));
    for (const auto& overrider : finals)
      binder.note(overrider.function->location(), "final overrider is here");
    return std::nullopt;
  }

  [[nodiscard]] auto sameVcallSignature(FunctionSymbol* lhs,
                                        FunctionSymbol* rhs) const -> bool {
    return binder.traits.is_corresponding_overrider(lhs, rhs);
  }

  void appendDeclaredVcalls(const SubobjectGraph& graph, std::size_t node,
                            std::vector<VcallEntry>& vcalls) const {
    for (auto function :
         declaredVirtualFunctions(graph.node(node).classSymbol)) {
      const auto replicated =
          std::ranges::any_of(vcalls, [&](const VcallEntry& entry) {
            return sameVcallSignature(entry.function, function);
          });
      if (!replicated) vcalls.push_back({function, node});
    }
  }

  void appendNonVirtualBaseVcalls(const SubobjectGraph& graph, std::size_t node,
                                  std::vector<VcallEntry>& vcalls) const {
    std::vector<std::size_t> pending{node};
    while (!pending.empty()) {
      auto current = pending.back();
      pending.pop_back();
      appendDeclaredVcalls(graph, current, vcalls);
      for (auto base : graph.node(current).bases | std::views::reverse) {
        if (!graph.node(base).isVirtual) pending.push_back(base);
      }
    }
  }

  void appendVcalls(const SubobjectGraph& graph, std::size_t node,
                    std::vector<VcallEntry>& vcalls) const {
    auto primary = graph.primaryBaseOf(node);
    const auto nonVirtualPrimary =
        primary && !graph.node(*primary).isVirtual ? primary : std::nullopt;

    if (nonVirtualPrimary) appendVcalls(graph, *nonVirtualPrimary, vcalls);

    appendDeclaredVcalls(graph, node, vcalls);

    for (auto base : graph.node(node).bases) {
      if (graph.node(base).isVirtual || base == nonVirtualPrimary) continue;
      appendNonVirtualBaseVcalls(graph, base, vcalls);
    }
  }

  [[nodiscard]] auto virtualPrimaryOfChain(const SubobjectGraph& graph,
                                           std::size_t node) const
      -> std::optional<std::size_t> {
    for (auto current = node;;) {
      auto primary = graph.primaryBaseOf(current);
      if (!primary) return std::nullopt;
      if (graph.node(*primary).isVirtual) return primary;
      current = *primary;
    }
  }

  [[nodiscard]] auto segmentVirtualBases(const SubobjectGraph& graph,
                                         std::size_t node) const
      -> std::vector<ClassSymbol*> {
    std::vector<std::size_t> chain{node};
    for (auto primary = graph.primaryBaseOf(node);
         primary && !graph.node(*primary).isVirtual;
         primary = graph.primaryBaseOf(*primary)) {
      chain.push_back(*primary);
    }

    std::vector<ClassSymbol*> virtualBases;
    for (auto current : chain | std::views::reverse) {
      auto classLayout = graph.node(current).classSymbol->layout();
      if (!classLayout) continue;
      for (auto base : classLayout->virtualBases()) {
        if (!std::ranges::contains(virtualBases, base))
          virtualBases.push_back(base);
      }
    }
    return virtualBases;
  }

  [[nodiscard]] auto tableSegments(const SubobjectGraph& graph,
                                   std::size_t node) const
      -> std::vector<Segment> {
    std::vector<Segment> segments;
    auto asVirtual = graph.node(node).isVirtual;
    for (std::optional<std::size_t> current = node; current;) {
      Segment segment;
      if (asVirtual) appendVcalls(graph, *current, segment.vcalls);
      segment.virtualBases = segmentVirtualBases(graph, *current);
      segments.push_back(std::move(segment));
      current = virtualPrimaryOfChain(graph, *current);
      asVirtual = true;
    }

    for (std::size_t outer = 0; outer < segments.size(); ++outer) {
      for (std::size_t inner = outer + 1; inner < segments.size(); ++inner) {
        std::erase_if(segments[outer].virtualBases, [&](ClassSymbol* base) {
          return std::ranges::contains(segments[inner].virtualBases, base);
        });
        std::erase_if(segments[outer].vcalls, [&](const VcallEntry& entry) {
          return std::ranges::any_of(
              segments[inner].vcalls, [&](const VcallEntry& deeper) {
                return sameVcallSignature(deeper.function, entry.function);
              });
        });
      }
    }

    return segments;
  }

  [[nodiscard]] auto vcallWordsBeforeAddressPoint(
      const SubobjectGraph& graph, std::size_t virtualBase,
      FunctionSymbol* function) const -> std::int64_t {
    auto words = std::int64_t{2};
    auto segments = tableSegments(graph, virtualBase);
    for (const auto& segment : segments | std::views::reverse) {
      words += static_cast<std::int64_t>(segment.virtualBases.size());
      for (const auto& entry : segment.vcalls) {
        ++words;
        if (sameVcallSignature(entry.function, function)) return words;
      }
    }
    return 0;
  }

  [[nodiscard]] auto tableHeader(const SubobjectGraph& graph, std::size_t node)
      -> std::vector<VTableLayout::Offset> {
    const auto tableOffset = static_cast<std::int64_t>(graph.node(node).offset);
    std::vector<VTableLayout::Offset> offsets;
    for (const auto& segment : tableSegments(graph, node)) {
      for (const auto& entry : segment.vcalls | std::views::reverse) {
        auto overrider = finalOverrider(graph, entry.node, entry.function);
        std::int64_t value = 0;
        if (overrider)
          value =
              static_cast<std::int64_t>(graph.node(overrider->node).offset) -
              tableOffset;
        offsets.push_back({entry.function, value});
      }
      for (auto base : segment.virtualBases | std::views::reverse) {
        std::int64_t value = 0;
        if (auto baseNode = graph.virtualNodeOf(base))
          value = static_cast<std::int64_t>(graph.node(*baseNode).offset) -
                  tableOffset;
        offsets.push_back({base, value});
      }
    }
    return offsets;
  }

  [[nodiscard]] static auto declaresOverrider(ClassSymbol* classSymbol,
                                              FunctionSymbol* function)
      -> bool {
    return std::ranges::any_of(
        declaredVirtualFunctions(classSymbol), [&](FunctionSymbol* declared) {
          return declared == function || declared->overrides(function);
        });
  }

  [[nodiscard]] auto declaringNode(const SubobjectGraph& graph,
                                   std::size_t node,
                                   FunctionSymbol* function) const
      -> std::optional<std::size_t> {
    const auto address = graph.node(node).offset;
    for (std::optional<std::size_t> current = node;
         current && graph.node(*current).offset == address;
         current = graph.primaryBaseOf(*current)) {
      if (declaresOverrider(graph.node(*current).classSymbol, function))
        return current;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto thisAdjustment(const SubobjectGraph& graph,
                                    std::size_t introducing,
                                    std::size_t overrider,
                                    FunctionSymbol* function,
                                    std::int64_t tableOffset) const
      -> VTableLayout::CallOffset {
    auto current = introducing;
    while (current != overrider && !graph.node(current).isVirtual &&
           !graph.node(current).derived.empty()) {
      current = graph.node(current).derived.front();
    }

    const auto& target = graph.node(current);
    const auto targetOffset = static_cast<std::int64_t>(target.offset);
    if (current == overrider) return {.nonVirtual = targetOffset - tableOffset};

    return {.nonVirtual = targetOffset - tableOffset,
            .virtualOffset = -wordSize * vcallWordsBeforeAddressPoint(
                                             graph, current, function)};
  }

  [[nodiscard]] auto primaryVirtualBaseWords(ClassSymbol* derived,
                                             ClassSymbol* virtualBase) const
      -> std::int64_t {
    if (derived == classSymbol)
      return primaryHeader.offsetWordsBeforeAddressPoint(virtualBase);
    auto derivedVTable = derived->vtableLayout();
    if (!derivedVTable) return 0;
    return derivedVTable->primary().offsetWordsBeforeAddressPoint(virtualBase);
  }

  [[nodiscard]] auto returnAdjustment(FunctionSymbol* overrider,
                                      FunctionSymbol* introducing) const
      -> VTableLayout::CallOffset {
    if (!requiresReturnAdjustment(overrider, introducing)) return {};
    auto from = returnedClassOf(binder.traits, overrider);
    auto to = returnedClassOf(binder.traits, introducing);
    auto path = findBasePath(from, to);
    if (!path) return {};
    if (!path->virtualBase) return {.nonVirtual = path->offset};
    return {.nonVirtual = path->offset,
            .virtualOffset =
                -wordSize * primaryVirtualBaseWords(from, path->virtualBase)};
  }

  [[nodiscard]] auto tableSlot(const SubobjectGraph& graph, std::size_t node,
                               const VTableLayout::Slot& slotTemplate)
      -> VTableLayout::Slot {
    VTableLayout::Slot slot{
        .kind = slotTemplate.kind,
        .introducingFunction = slotTemplate.introducingFunction};

    const auto tableOffset = static_cast<std::int64_t>(graph.node(node).offset);
    auto declaring =
        declaringNode(graph, node, slotTemplate.introducingFunction);
    if (!declaring) return slot;

    auto overrider =
        finalOverrider(graph, *declaring, slotTemplate.introducingFunction);
    if (!overrider) return slot;

    slot.function = overrider->function;
    if (graph.node(overrider->node).offset != graph.node(node).offset) {
      slot.thisAdjustment =
          thisAdjustment(graph, *declaring, overrider->node,
                         slotTemplate.introducingFunction, tableOffset);
    }
    slot.returnAdjustment =
        returnAdjustment(overrider->function, slotTemplate.introducingFunction);
    return slot;
  }

  void appendNonVirtualTableNodes(const SubobjectGraph& graph, std::size_t node,
                                  std::vector<std::size_t>& nodes) const {
    std::vector<std::size_t> pending;
    for (auto base : graph.node(node).bases | std::views::reverse) {
      if (!graph.node(base).isVirtual) pending.push_back(base);
    }
    while (!pending.empty()) {
      auto current = pending.back();
      pending.pop_back();
      if (hasVirtualPointer(graph.node(current).classSymbol) &&
          !graph.isNonVirtualPrimaryBase(current))
        nodes.push_back(current);
      for (auto base : graph.node(current).bases | std::views::reverse) {
        if (!graph.node(base).isVirtual) pending.push_back(base);
      }
    }
  }

  [[nodiscard]] auto tableNodes(const SubobjectGraph& graph) const
      -> std::vector<std::size_t> {
    std::vector<std::size_t> nodes{0};
    appendNonVirtualTableNodes(graph, 0, nodes);
    for (auto node : graph.preorder()) {
      if (node == 0 || !graph.node(node).isVirtual) continue;
      if (!hasVirtualPointer(graph.node(node).classSymbol)) continue;
      if (graph.sharesVirtualPointer(node)) continue;
      nodes.push_back(node);
      appendNonVirtualTableNodes(graph, node, nodes);
    }
    return nodes;
  }

  [[nodiscard]] auto buildGroup(const SubobjectGraph& graph)
      -> VTableLayout::Group {
    const auto& root = graph.node(0);
    VTableLayout::Group group{.base = root.classSymbol, .offset = root.offset};

    const auto nodes = tableNodes(graph);
    for (auto node : nodes) {
      const auto& subobject = graph.node(node);
      if (node == 0 && subobject.classSymbol == classSymbol &&
          !subobject.isVirtual)
        primaryHeader.offsets = tableHeader(graph, node);
      group.tables.push_back(
          {.base = subobject.classSymbol,
           .offset = subobject.offset,
           .enclosingVirtualBase = graph.enclosingVirtualBase(node),
           .offsetToTop = static_cast<std::int64_t>(root.offset) -
                          static_cast<std::int64_t>(subobject.offset),
           .offsets = tableHeader(graph, node)});
    }

    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto& table = group.tables[index];
      for (const auto& slotTemplate : slotTemplate(table.base))
        table.slots.push_back(tableSlot(graph, nodes[index], slotTemplate));
    }

    return group;
  }

  [[nodiscard]] static auto sameCallOffset(const VTableLayout::CallOffset& lhs,
                                           const VTableLayout::CallOffset& rhs)
      -> bool {
    return lhs.nonVirtual == rhs.nonVirtual &&
           lhs.virtualOffset == rhs.virtualOffset;
  }

  void addAdjustingEntryPoint(const VTableLayout::EntryPoint& entryPoint) {
    const auto exists = std::ranges::any_of(
        layout->adjustingEntryPoints,
        [&](const VTableLayout::EntryPoint& entry) {
          return entry.function == entryPoint.function &&
                 entry.kind == entryPoint.kind &&
                 sameCallOffset(entry.thisAdjustment,
                                entryPoint.thisAdjustment) &&
                 sameCallOffset(entry.returnAdjustment,
                                entryPoint.returnAdjustment);
        });
    if (!exists) layout->adjustingEntryPoints.push_back(entryPoint);
  }

  void addAdjustingEntryPoints(const SubobjectGraph& graph,
                               FunctionSymbol* function, std::size_t base,
                               FunctionSymbol* overridden) {
    const auto baseOffset = static_cast<std::int64_t>(graph.node(base).offset);
    auto thisAdjust = thisAdjustment(graph, base, 0, overridden, baseOffset);
    auto returnAdjust = returnAdjustment(function, overridden);
    if (thisAdjust.isEmpty() && returnAdjust.isEmpty()) return;

    if (!function->isDestructor()) {
      addAdjustingEntryPoint({function, VTableLayout::SlotKind::kFunction,
                              thisAdjust, returnAdjust});
      return;
    }

    addAdjustingEntryPoint({function, VTableLayout::SlotKind::kCompleteDtor,
                            thisAdjust, returnAdjust});
    addAdjustingEntryPoint({function, VTableLayout::SlotKind::kDeletingDtor,
                            thisAdjust, returnAdjust});
  }

  void buildAdjustingEntryPoints(const SubobjectGraph& graph) {
    for (auto function : declaredVirtualFunctions(classSymbol)) {
      if (function->isPure()) continue;
      for (auto base : graph.preorder()) {
        if (base == 0) continue;
        for (auto overridden :
             declaredVirtualFunctions(graph.node(base).classSymbol)) {
          if (!function->overrides(overridden)) continue;
          addAdjustingEntryPoints(graph, function, base, overridden);
        }
      }
    }
  }

  [[nodiscard]] static auto requiresVTT(ClassSymbol* tableClass) -> bool {
    return tableClass->hasVirtualBaseSubobjects();
  }

  void appendSubVTT(ClassSymbol* base, std::uint64_t offset, bool isVirtual,
                    std::vector<VTableLayout::SubVTT>& starts) {
    starts.push_back({base, static_cast<std::uint32_t>(layout->vtt.size())});
    SubobjectGraph graph{classSymbol, base, offset, isVirtual};
    layout->constructionGroups.push_back(buildGroup(graph));
    const auto group =
        static_cast<std::int32_t>(layout->constructionGroups.size() - 1);
    appendVTT(graph, group, false);
  }

  [[nodiscard]] auto groupAt(std::int32_t group) const
      -> const VTableLayout::Group& {
    if (group < 0) return layout->main;
    return layout->constructionGroups[static_cast<std::size_t>(group)];
  }

  void appendVTT(const SubobjectGraph& graph, std::int32_t group,
                 bool isComplete) {
    layout->vtt.push_back({group, 0});

    std::vector<VTableLayout::SubVTT> nestedStarts;
    for (auto base : graph.node(0).bases) {
      const auto& subobject = graph.node(base);
      if (subobject.isVirtual || !requiresVTT(subobject.classSymbol)) continue;
      appendSubVTT(subobject.classSymbol, subobject.offset, false,
                   isComplete ? layout->baseSubVTTs : nestedStarts);
    }

    const auto& tables = groupAt(group);
    for (auto node : graph.preorder()) {
      if (node == 0) continue;
      const auto& subobject = graph.node(node);
      if (!hasVirtualPointer(subobject.classSymbol)) continue;
      if (graph.isNonVirtualPrimaryBase(node)) continue;
      if (!requiresVTT(subobject.classSymbol) &&
          !graph.isReachableAlongVirtualPath(node))
        continue;
      auto table = tables.tableAt(subobject.offset);
      if (table < 0) continue;
      if (isComplete) recordTableVTTIndex(static_cast<std::size_t>(table));
      layout->vtt.push_back({group, static_cast<std::uint32_t>(table)});
    }

    if (!isComplete) return;

    for (auto node : graph.preorder()) {
      const auto& subobject = graph.node(node);
      if (!subobject.isVirtual || !requiresVTT(subobject.classSymbol)) continue;
      appendSubVTT(subobject.classSymbol, subobject.offset, true,
                   layout->virtualBaseSubVTTs);
    }
  }

  void recordTableVTTIndex(std::size_t table) {
    if (layout->tableVTTIndices[table] >= 0) return;
    layout->tableVTTIndices[table] =
        static_cast<std::int32_t>(layout->vtt.size());
  }

  void buildVTT() {
    layout->tableVTTIndices.assign(layout->main.tables.size(), -1);
    if (!requiresVTT(classSymbol)) return;
    layout->tableVTTIndices.front() = 0;
    appendVTT(SubobjectGraph{classSymbol, classSymbol, 0, false}, -1, true);
  }
};

void Binder::buildVTableLayout(ClassSymbol* classSymbol) {
  BuildVTableLayout{*this, classSymbol}();
}

}  // namespace cxx
