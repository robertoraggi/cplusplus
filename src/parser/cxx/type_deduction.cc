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
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/memory_layout.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_deduction.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>

namespace cxx {

namespace {

enum class ParameterKind { kType, kValue, kTemplate };

struct TemplateParameterKind {
  [[nodiscard]] auto operator()(TypenameTypeParameterAST*) const
      -> ParameterKind {
    return ParameterKind::kType;
  }
  [[nodiscard]] auto operator()(ConstraintTypeParameterAST*) const
      -> ParameterKind {
    return ParameterKind::kType;
  }
  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST*) const
      -> ParameterKind {
    return ParameterKind::kValue;
  }
  [[nodiscard]] auto operator()(TemplateTypeParameterAST*) const
      -> ParameterKind {
    return ParameterKind::kTemplate;
  }
};

struct ParameterShape {
  ParameterKind kind = ParameterKind::kType;
  bool isPack = false;
  bool hasDefault = false;
};

[[nodiscard]] auto parameterShapes(List<TemplateParameterAST*>* parameters)
    -> std::vector<ParameterShape> {
  std::vector<ParameterShape> shapes;
  for (auto parameter : ListView{parameters}) {
    shapes.push_back({.kind = visit(TemplateParameterKind{}, parameter),
                      .isPack = isPackParameter(parameter),
                      .hasDefault = hasDefaultTemplateArgument(parameter)});
  }
  return shapes;
}

[[nodiscard]] auto isValidTemplateTemplateArgument(
    TemplateParameterAST* parameter, Symbol* templateSymbol) -> bool {
  auto templateParameter = ast_cast<TemplateTypeParameterAST>(parameter);
  if (!templateParameter) return false;
  auto declaration = template_declaration_of(templateSymbol);
  if (!declaration) return true;

  auto P = parameterShapes(templateParameter->templateParameterList);
  auto A = parameterShapes(declaration->templateParameterList);

  std::size_t index = 0;
  for (const auto& p : P) {
    if (p.isPack) {
      for (; index < A.size(); ++index) {
        if (A[index].kind != p.kind) return false;
      }
      return true;
    }
    if (index >= A.size()) return false;
    if (A[index].kind != p.kind) return false;
    if (!A[index].isPack) ++index;
  }

  for (; index < A.size(); ++index) {
    if (!A[index].isPack && !A[index].hasDefault) return false;
  }

  return true;
}

[[nodiscard]] auto hasDefaultsFrom(Symbol* templateSymbol, std::size_t position)
    -> bool {
  auto declaration = template_declaration_of(templateSymbol);
  if (!declaration) return false;
  auto shapes = parameterShapes(declaration->templateParameterList);
  for (std::size_t index = position; index < shapes.size(); ++index) {
    if (!shapes[index].hasDefault && !shapes[index].isPack) return false;
  }
  return true;
}

enum class ArgumentKind { kType, kValue, kTemplate, kPack };

struct SymbolArgumentKind {
  [[nodiscard]] auto operator()(ParameterPackSymbol*) const -> ArgumentKind {
    return ArgumentKind::kPack;
  }
  [[nodiscard]] auto operator()(VariableSymbol*) const -> ArgumentKind {
    return ArgumentKind::kValue;
  }
  [[nodiscard]] auto operator()(NonTypeParameterSymbol*) const -> ArgumentKind {
    return ArgumentKind::kValue;
  }
  [[nodiscard]] auto operator()(TemplateTypeParameterSymbol*) const
      -> ArgumentKind {
    return ArgumentKind::kTemplate;
  }
  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> ArgumentKind {
    if (symbol->isTemplatePattern()) return ArgumentKind::kTemplate;
    return ArgumentKind::kType;
  }
  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> ArgumentKind {
    if (symbol->templateParameters() && !symbol->isSpecialization())
      return ArgumentKind::kTemplate;
    return ArgumentKind::kType;
  }
  [[nodiscard]] auto operator()(Symbol*) const -> ArgumentKind {
    return ArgumentKind::kType;
  }
};

[[nodiscard]] auto argumentSymbol(const TemplateArgument& argument) -> Symbol* {
  auto symbol = std::get_if<Symbol*>(&argument);
  return symbol ? *symbol : nullptr;
}

[[nodiscard]] auto argumentKind(const TemplateArgument& argument)
    -> ArgumentKind {
  if (std::holds_alternative<const Type*>(argument)) return ArgumentKind::kType;
  auto symbol = argumentSymbol(argument);
  if (!symbol) return ArgumentKind::kValue;
  return visit(SymbolArgumentKind{}, symbol);
}

[[nodiscard]] auto argumentType(const TemplateArgument& argument)
    -> const Type* {
  if (argumentKind(argument) != ArgumentKind::kType) return nullptr;
  if (auto type = std::get_if<const Type*>(&argument)) return *type;
  return argumentSymbol(argument)->type();
}

[[nodiscard]] auto valueType(const TemplateArgument& argument) -> const Type* {
  auto symbol = argumentSymbol(argument);
  if (auto parameter = symbol_cast<NonTypeParameterSymbol>(symbol))
    return parameter->objectType();
  if (symbol) return symbol->type();
  if (auto expression = std::get_if<ExpressionAST*>(&argument))
    return *expression ? (*expression)->type : nullptr;
  return nullptr;
}

[[nodiscard]] auto nonTypeParameterOf(ExpressionAST* expression)
    -> NonTypeParameterSymbol* {
  auto id = ast_cast<IdExpressionAST>(expression);
  if (!id) return nullptr;
  return symbol_cast<NonTypeParameterSymbol>(id->symbol);
}

[[nodiscard]] auto valuePattern(ExpressionAST* expression) -> TemplateArgument {
  if (auto parameter = nonTypeParameterOf(expression))
    return TemplateArgument{static_cast<Symbol*>(parameter)};
  return TemplateArgument{expression};
}

[[nodiscard]] auto expandedValuePattern(const TemplateArgument& argument)
    -> std::optional<TemplateArgument> {
  auto variable = symbol_cast<VariableSymbol>(argumentSymbol(argument));
  if (!variable) return std::nullopt;
  auto expansion =
      ast_cast<PackExpansionExpressionAST>(variable->initializer());
  if (!expansion) return std::nullopt;
  return valuePattern(expansion->expression);
}

[[nodiscard]] auto templateParameterNamedBy(TypeIdAST* typeId)
    -> TemplateTypeParameterSymbol* {
  if (!typeId) return nullptr;
  for (auto specifier : ListView{typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(specifier);
    if (!named) continue;
    return symbol_cast<TemplateTypeParameterSymbol>(named->symbol);
  }
  return nullptr;
}

[[nodiscard]] auto expandedTemplateNamePattern(const TemplateArgument& argument)
    -> std::optional<TemplateArgument> {
  auto alias = symbol_cast<TypeAliasSymbol>(argumentSymbol(argument));
  if (!alias) return std::nullopt;
  auto parameter = templateParameterNamedBy(alias->expansionTypeId());
  if (!parameter) return std::nullopt;
  return TemplateArgument{static_cast<Symbol*>(parameter)};
}

[[nodiscard]] auto packExpansionPattern(const TemplateArgument& argument)
    -> std::optional<TemplateArgument> {
  if (auto pattern = expandedTemplateNamePattern(argument)) return pattern;
  if (auto expansion = type_cast<PackExpansionType>(argumentType(argument)))
    return TemplateArgument{expansion->pattern()};
  return expandedValuePattern(argument);
}

[[nodiscard]] auto canonicalTemplate(Symbol* symbol) -> Symbol* {
  auto templateName = template_name_symbol(symbol);
  if (!templateName) return symbol;
  return templateName->canonical();
}

}  // namespace

struct TypeDeduction::CollectSlots {
  enum class Mode { kDeducible, kExpandedPacks, kUsed };

  const TypeDeduction& deduction;
  std::vector<int>& slots;
  Mode mode = Mode::kDeducible;

  struct CollectReferencedParameters final : ASTVisitor {
    CollectSlots& collector;

    explicit CollectReferencedParameters(CollectSlots& collector)
        : collector(collector) {}

    void addSymbol(Symbol* symbol) {
      auto info = template_parameter_info(symbol);
      if (!info) return;
      collector.addParameter(info->depth, info->index, info->isPack);
    }

    void visit(NamedTypeSpecifierAST* ast) override {
      addSymbol(ast->symbol);
      ASTVisitor::visit(ast);
    }

    void visit(SimpleNestedNameSpecifierAST* ast) override {
      addSymbol(ast->symbol);
      ASTVisitor::visit(ast);
    }

    void visit(SimpleTemplateIdAST* ast) override {
      addSymbol(ast->symbol);
      ASTVisitor::visit(ast);
    }

    void visit(IdExpressionAST* ast) override {
      addSymbol(ast->symbol);
      ASTVisitor::visit(ast);
    }
  };

  [[nodiscard]] auto packsOnly() const -> bool {
    return mode == Mode::kExpandedPacks;
  }

  void add(int slot) {
    if (slot < 0) return;
    if (packsOnly() && std::ranges::contains(deduction.expandingSlots_, slot))
      return;
    if (std::ranges::contains(slots, slot)) return;
    slots.push_back(slot);
  }

  void addParameter(int depth, int index, bool isPack) {
    if (packsOnly() && !isPack) return;
    add(deduction.slotOf(depth, index));
  }

  void collectInNonDeducedContext(const Type* type) {
    if (mode != Mode::kUsed) return;
    collect(type);
  }

  void collectReferencedParameters(AST* ast) {
    if (mode != Mode::kUsed || !ast) return;
    CollectReferencedParameters collector{*this};
    collector.accept(ast);
  }

  void collect(const Type* type) {
    if (type) visit(*this, type);
  }

  void collect(const TemplateArgument& argument) {
    auto symbol = argumentSymbol(argument);
    switch (argumentKind(argument)) {
      case ArgumentKind::kPack:
        for (auto element :
             symbol_cast<ParameterPackSymbol>(symbol)->elements())
          collect(TemplateArgument{element});
        return;
      case ArgumentKind::kTemplate:
        if (auto parameter = symbol_cast<TemplateTypeParameterSymbol>(symbol))
          collect(parameter->type());
        return;
      case ArgumentKind::kValue:
        collectValue(argument);
        return;
      case ArgumentKind::kType:
        collect(argumentType(argument));
        return;
    }
  }

  void collectValue(const TemplateArgument& argument) {
    if (auto pattern = expandedValuePattern(argument)) {
      if (packsOnly()) return;
      collectValue(*pattern);
      return;
    }
    if (auto expression = std::get_if<ExpressionAST*>(&argument)) {
      collect(*expression);
      return;
    }
    auto symbol = argumentSymbol(argument);
    if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
      collectReferencedParameters(variable->initializer());
      return;
    }
    auto parameter = symbol_cast<NonTypeParameterSymbol>(symbol);
    if (!parameter) return;
    addParameter(parameter->depth(), parameter->index(),
                 parameter->isParameterPack());
  }

  void collect(ExpressionAST* expression) {
    if (mode == Mode::kUsed) {
      collectReferencedParameters(expression);
      return;
    }
    auto parameter = nonTypeParameterOf(expression);
    if (!parameter) return;
    addParameter(parameter->depth(), parameter->index(),
                 parameter->isParameterPack());
  }

  void operator()(const TypeParameterType* type) {
    addParameter(type->depth(), type->index(), type->isParameterPack());
  }

  void operator()(const TemplateTypeParameterType* type) {
    addParameter(type->depth(), type->index(), type->isParameterPack());
  }

  void operator()(const TemplateTypeParameterSpecializationType* type) {
    collect(type->templateParameter());
    if (packsOnly()) return;
    for (const auto& argument : type->templateArguments()) collect(argument);
  }

  void operator()(const QualType* type) { collect(type->elementType()); }
  void operator()(const PointerType* type) { collect(type->elementType()); }
  void operator()(const LvalueReferenceType* type) {
    collect(type->elementType());
  }
  void operator()(const RvalueReferenceType* type) {
    collect(type->elementType());
  }
  void operator()(const BoundedArrayType* type) {
    collect(type->elementType());
  }
  void operator()(const UnboundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const UnresolvedBoundedArrayType* type) {
    collect(type->elementType());
    collect(type->size());
  }

  void operator()(const PackExpansionType* type) {
    if (packsOnly()) return;
    collect(type->pattern());
  }

  void operator()(const FunctionType* type) {
    collect(type->returnType());
    for (auto parameter : type->parameterTypes()) collect(parameter);
    if (auto expression = type->noexceptExpression()) collect(expression);
  }

  void operator()(const MemberObjectPointerType* type) {
    collect(type->classType());
    collect(type->elementType());
  }

  void operator()(const MemberFunctionPointerType* type) {
    collect(type->classType());
    collect(type->functionType());
  }

  void operator()(const ClassType* type) {
    auto classSymbol = type->symbol();
    if (!class_template_of(classSymbol)) return;
    if (packsOnly() && !classSymbol->isSpecialization()) return;
    for (const auto& argument :
         class_template_arguments(deduction.unit_, classSymbol))
      collect(argument);
  }

  void operator()(const UnresolvedNameType* type) {
    collectReferencedParameters(type->nestedNameSpecifier());
    collectReferencedParameters(type->unqualifiedId());
  }

  void operator()(const UnresolvedUnderlyingType* type) {
    collectReferencedParameters(type->typeId());
  }

  void operator()(const UnresolvedBuiltinType* type) {
    collectReferencedParameters(type->typeId());
  }

  void operator()(const UnresolvedBitIntType* type) {
    collectReferencedParameters(type->sizeExpression());
  }

  void operator()(const UnresolvedVectorType* type) {
    collect(type->elementType());
    collect(type->sizeExpression());
  }

  void operator()(const VectorType* type) {
    collectInNonDeducedContext(type->elementType());
  }

  void operator()(const ComplexType* type) {
    collectInNonDeducedContext(type->elementType());
  }

  void operator()(const AtomicType* type) {
    collectInNonDeducedContext(type->elementType());
  }

  void operator()(const Type*) {}
};

struct TypeDeduction::DeduceType {
  TypeDeduction& deduction;
  const Type* A;
  bool allowsQualificationConversion = false;

  [[nodiscard]] auto deduceNextLevel(const Type* P, const Type* A) -> bool {
    return deduction.deduceAtLevel(P, A, allowsQualificationConversion);
  }

  [[nodiscard]] auto operator()(const TypeParameterType* P) -> bool {
    if (P == deduction.control()->getDependentType())
      return deduction.skipNonDeducedContext();
    auto slot = deduction.slotOf(P);
    if (slot < 0) return deduction.unit_->typeTraits().is_same(P, A);
    return deduction.record(slot, deduction.typeArgument(A));
  }

  [[nodiscard]] auto operator()(const QualType* P) -> bool {
    return deduction.deduceQualifiedType(P, A, allowsQualificationConversion);
  }

  [[nodiscard]] auto operator()(const PointerType* P) -> bool {
    auto pointer = type_cast<PointerType>(A);
    if (!pointer) return false;
    return deduceNextLevel(P->elementType(), pointer->elementType());
  }

  [[nodiscard]] auto specifiedElementType(const Type* element) const
      -> const Type* {
    auto qualified = type_cast<QualType>(element);
    auto slot =
        deduction.slotOf(qualified ? qualified->elementType() : element);
    if (slot < 0 || !deduction.isSpecified(slot)) return nullptr;
    auto alias = symbol_cast<TypeAliasSymbol>(deduction.deduced(slot));
    if (!alias || !alias->type()) return nullptr;
    auto traits = deduction.unit_->typeTraits();
    if (!qualified || traits.is_reference(alias->type())) return alias->type();
    return traits.add_cv(alias->type(), qualified->cvQualifiers());
  }

  [[nodiscard]] auto operator()(const LvalueReferenceType* P) -> bool {
    if (auto specified = specifiedElementType(P->elementType())) {
      auto traits = deduction.unit_->typeTraits();
      return deduction.deduce(traits.add_lvalue_reference(specified), A);
    }
    auto reference = type_cast<LvalueReferenceType>(A);
    if (!reference) return false;
    return deduction.deduce(P->elementType(), reference->elementType());
  }

  [[nodiscard]] auto operator()(const RvalueReferenceType* P) -> bool {
    if (auto specified = specifiedElementType(P->elementType())) {
      auto traits = deduction.unit_->typeTraits();
      return deduction.deduce(traits.add_rvalue_reference(specified), A);
    }
    auto reference = type_cast<RvalueReferenceType>(A);
    if (!reference) return false;
    return deduction.deduce(P->elementType(), reference->elementType());
  }

  [[nodiscard]] auto operator()(const BoundedArrayType* P) -> bool {
    auto array = type_cast<BoundedArrayType>(A);
    if (!array || array->size() != P->size()) return false;
    return deduceNextLevel(P->elementType(), array->elementType());
  }

  [[nodiscard]] auto operator()(const UnboundedArrayType* P) -> bool {
    auto array = type_cast<UnboundedArrayType>(A);
    if (!array) return false;
    return deduceNextLevel(P->elementType(), array->elementType());
  }

  [[nodiscard]] auto operator()(const UnresolvedBoundedArrayType* P) -> bool {
    if (auto array = type_cast<BoundedArrayType>(A)) {
      if (!deduction.deduceArrayBound(P->size(), array)) return false;
      return deduceNextLevel(P->elementType(), array->elementType());
    }
    if (auto array = type_cast<UnresolvedBoundedArrayType>(A)) {
      if (!deduction.deduceArrayBound(P->size(), array)) return false;
      return deduceNextLevel(P->elementType(), array->elementType());
    }
    return false;
  }

  [[nodiscard]] auto operator()(const UnresolvedVectorType* P) -> bool {
    auto vector = type_cast<VectorType>(A);
    if (!vector || vector->vectorKind() != P->vectorKind()) return false;
    auto size = deduction.vectorSizeIn(P->sizeKind(), vector);
    if (!size) return false;
    if (!deduction.deduceArrayBound(P->sizeExpression(), *size)) return false;
    return deduceNextLevel(P->elementType(), vector->elementType());
  }

  [[nodiscard]] auto operator()(const FunctionType* P) -> bool {
    return deduction.deduceFunctionType(P, A, allowsQualificationConversion);
  }

  [[nodiscard]] auto operator()(const MemberObjectPointerType* P) -> bool {
    auto pointer = decomposeMemberPointer(A);
    if (!pointer) return false;
    if (!deduction.deduce(P->classType(), pointer.classType)) return false;
    return deduceNextLevel(P->elementType(), pointer.pointeeType);
  }

  [[nodiscard]] auto operator()(const MemberFunctionPointerType* P) -> bool {
    auto pointer = type_cast<MemberFunctionPointerType>(A);
    if (!pointer) return false;
    if (!deduction.deduce(P->classType(), pointer->classType())) return false;
    return deduceNextLevel(P->functionType(), pointer->functionType());
  }

  [[nodiscard]] auto operator()(const ClassType* P) -> bool {
    return deduction.deduceClassSpecialization(P, A);
  }

  [[nodiscard]] auto operator()(const PackExpansionType* P) -> bool {
    auto expansion = type_cast<PackExpansionType>(A);
    if (!expansion) return false;
    return deduction.deduce(P->pattern(), expansion->pattern());
  }

  [[nodiscard]] auto operator()(
      const TemplateTypeParameterSpecializationType* P) -> bool {
    return deduction.deduceTemplateParameterSpecialization(P, A);
  }

  [[nodiscard]] auto operator()(const Type*) -> bool {
    return deduction.skipNonDeducedContext();
  }
};

TypeDeduction::TypeDeduction(TranslationUnit* unit,
                             TemplateDeclarationAST* templateDeclaration)
    : unit_(unit) {
  if (templateDeclaration) {
    for (auto parameter : ListView{templateDeclaration->templateParameterList})
      parameters_.push_back(parameter);
  }
  deduced_.assign(parameters_.size(), nullptr);
  specified_.assign(parameters_.size(), false);
  packPrefixes_.assign(parameters_.size(), nullptr);
}

auto TypeDeduction::control() const -> Control* { return unit_->control(); }

auto TypeDeduction::slotOf(int depth, int index) const -> int {
  for (std::size_t slot = 0; slot < parameters_.size(); ++slot) {
    if (parameters_[slot]->depth == depth && parameters_[slot]->index == index)
      return static_cast<int>(slot);
  }
  return -1;
}

auto TypeDeduction::slotOf(Symbol* parameter) const -> int {
  if (!parameter) return -1;
  if (auto nonType = symbol_cast<NonTypeParameterSymbol>(parameter))
    return slotOf(nonType->depth(), nonType->index());
  return slotOf(parameter->type());
}

auto TypeDeduction::slotOf(const Type* type) const -> int {
  auto info = getTypeParamInfo(type);
  if (!info) return -1;
  return slotOf(info->depth, info->index);
}

void TypeDeduction::specify(int slot, Symbol* argument) {
  deduced_[slot] = argument;
  specified_[slot] = true;
}

void TypeDeduction::specifyPackPrefix(int slot, ParameterPackSymbol* prefix) {
  deduced_[slot] = prefix;
  packPrefixes_[slot] = prefix;
}

auto TypeDeduction::specifiedPackLength(int slot) const -> std::size_t {
  if (!packPrefixes_[slot]) return 0;
  return packPrefixes_[slot]->elements().size();
}

auto TypeDeduction::state() const -> State {
  return {.deduced = deduced_,
          .skippedNonDeducedContext = skippedNonDeducedContext_};
}

void TypeDeduction::restore(State state) {
  deduced_ = std::move(state.deduced);
  skippedNonDeducedContext_ = state.skippedNonDeducedContext;
}

auto TypeDeduction::record(int slot, Symbol* value) -> bool {
  if (!value) return false;
  auto& deduced = deduced_[slot];
  if (!deduced) {
    deduced = value;
    return true;
  }
  return compare_single_arg(unit_, deduced, value);
}

auto TypeDeduction::typeArgument(const Type* type) const -> Symbol* {
  return control()->getTypeArgumentSymbol(type);
}

auto TypeDeduction::sizeArgument(std::size_t size) const -> Symbol* {
  auto value = unit_->typeTraits().integral_constant(
      control()->getSizeType(), static_cast<ConstInt::Wide>(size));
  if (!value) return nullptr;
  return control()->getConstantArgumentSymbol(control()->getSizeType(),
                                              ConstValue{*value});
}

auto TypeDeduction::skipNonDeducedContext() -> bool {
  skippedNonDeducedContext_ = true;
  return true;
}

auto TypeDeduction::mentionsDeducibleParameter(const Type* type) const -> bool {
  std::vector<int> slots;
  CollectSlots collector{*this, slots};
  collector.collect(type);
  return std::ranges::any_of(slots,
                             [&](int slot) { return !specified_[slot]; });
}

auto TypeDeduction::mentionedSlots(const Type* type) const -> std::vector<int> {
  std::vector<int> slots;
  CollectSlots collector{*this, slots, CollectSlots::Mode::kUsed};
  collector.collect(type);
  return slots;
}

auto TypeDeduction::packSlots(const TemplateArgument& pattern) const
    -> std::vector<int> {
  std::vector<int> slots;
  CollectSlots collector{*this, slots, CollectSlots::Mode::kExpandedPacks};
  collector.collect(pattern);
  return slots;
}

auto TypeDeduction::isPackExpansion(const TemplateArgument& argument) const
    -> bool {
  return packExpansionPattern(argument).has_value();
}

auto TypeDeduction::deduce(std::span<const TemplateArgument> P,
                           std::span<const TemplateArgument> A) -> bool {
  auto patterns = expand_template_arguments(P);
  auto arguments = expand_template_arguments(A);

  for (std::size_t index = 0; index < patterns.size(); ++index) {
    if (auto pattern = packExpansionPattern(patterns[index])) {
      if (index + 1 != patterns.size()) return skipNonDeducedContext();
      auto rest =
          std::span{arguments}.subspan(std::min(index, arguments.size()));
      return deducePackExpansion(*pattern, rest);
    }
    if (index >= arguments.size()) return false;
    if (isPackExpansion(arguments[index])) return false;
    if (!deduce(patterns[index], arguments[index])) return false;
  }

  return patterns.size() == arguments.size();
}

auto TypeDeduction::beginExpansion(const TemplateArgument& pattern)
    -> Expansion {
  Expansion expansion{.slots = packSlots(pattern),
                      .expandingSlots = expandingSlots_.size()};
  for (auto slot : expansion.slots) {
    expansion.previous.push_back(deduced_[slot]);
    expansion.packs.push_back(control()->newParameterPackSymbol(nullptr, {}));
  }
  expandingSlots_.insert(expandingSlots_.end(), expansion.slots.begin(),
                         expansion.slots.end());
  return expansion;
}

auto TypeDeduction::beginElement(const Expansion& expansion, std::size_t index)
    -> bool {
  auto specifiedElements = 0uz;
  for (auto slot : expansion.slots) {
    auto prefix = packPrefixes_[slot];
    if (prefix && index < prefix->elements().size()) {
      deduced_[slot] = prefix->elements()[index];
      ++specifiedElements;
    } else {
      deduced_[slot] = nullptr;
    }
  }
  return specifiedElements != expansion.slots.size();
}

auto TypeDeduction::endElement(Expansion& expansion) -> bool {
  for (std::size_t i = 0; i < expansion.slots.size(); ++i) {
    auto element = deduced_[expansion.slots[i]];
    if (!element) return false;
    expansion.packs[i]->addElement(element);
  }
  return true;
}

auto TypeDeduction::endExpansion(Expansion& expansion) -> bool {
  expandingSlots_.resize(expansion.expandingSlots);
  for (std::size_t i = 0; i < expansion.slots.size(); ++i) {
    auto slot = expansion.slots[i];
    auto pack = expansion.packs[i];
    if (auto prefix = packPrefixes_[slot]) {
      for (auto index = pack->elements().size();
           index < prefix->elements().size(); ++index)
        pack->addElement(prefix->elements()[index]);
    }
    deduced_[slot] = expansion.previous[i];
    if (deduced_[slot] == packPrefixes_[slot]) deduced_[slot] = nullptr;
    if (!record(slot, pack)) return false;
  }
  return true;
}

auto TypeDeduction::deducePackExpansion(
    const TemplateArgument& pattern,
    std::span<const TemplateArgument> arguments) -> bool {
  auto expansion = beginExpansion(pattern);
  if (expansion.slots.empty()) {
    expandingSlots_.resize(expansion.expandingSlots);
    return skipNonDeducedContext();
  }
  if (!deducePackElements(pattern, arguments, expansion)) {
    expandingSlots_.resize(expansion.expandingSlots);
    return false;
  }
  return endExpansion(expansion);
}

auto TypeDeduction::deducePackElements(
    const TemplateArgument& pattern,
    std::span<const TemplateArgument> arguments, Expansion& expansion) -> bool {
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    auto argument =
        packExpansionPattern(arguments[index]).value_or(arguments[index]);
    if (beginElement(expansion, index) && !deduce(pattern, argument))
      return false;
    if (!endElement(expansion)) return false;
  }
  return true;
}

auto TypeDeduction::deduce(const TemplateArgument& P, const TemplateArgument& A)
    -> bool {
  auto symbol = argumentSymbol(P);

  switch (argumentKind(P)) {
    case ArgumentKind::kPack: {
      auto other = symbol_cast<ParameterPackSymbol>(argumentSymbol(A));
      if (!other) return false;
      const auto& elements =
          symbol_cast<ParameterPackSymbol>(symbol)->elements();
      std::vector<TemplateArgument> patterns(elements.begin(), elements.end());
      std::vector<TemplateArgument> arguments(other->elements().begin(),
                                              other->elements().end());
      return deduce(patterns, arguments);
    }

    case ArgumentKind::kTemplate: {
      auto templateSymbol = argumentSymbol(A);
      if (argumentKind(A) != ArgumentKind::kTemplate) return false;
      auto slot = slotOf(symbol_cast<TemplateTypeParameterSymbol>(symbol));
      if (slot < 0)
        return canonicalTemplate(symbol) == canonicalTemplate(templateSymbol);
      return deduceTemplateName(slot, templateSymbol);
    }

    case ArgumentKind::kValue:
      if (auto parameter = symbol_cast<NonTypeParameterSymbol>(symbol))
        return deduceTemplateArgumentValue(parameter, A);
      return deduceValue(P, A);

    case ArgumentKind::kType: {
      auto type = argumentType(A);
      if (!type) return false;
      return deduce(argumentType(P), type);
    }
  }

  return false;
}

auto TypeDeduction::deduce(const Type* P, const Type* A) -> bool {
  return deduceAtLevel(P, A, false);
}

auto TypeDeduction::deduceAllowingConversions(const Type* P, const Type* A)
    -> bool {
  return deduceAtLevel(P, A, true);
}

auto TypeDeduction::deduceAtLevel(const Type* P, const Type* A,
                                  bool allowsQualificationConversion) -> bool {
  if (!P || !A) return false;
  if (!isDependent(unit_, P))
    return matchesNonDependentType(P, A, allowsQualificationConversion);
  return visit(DeduceType{*this, A, allowsQualificationConversion}, P);
}

auto TypeDeduction::matchesNonDependentType(
    const Type* P, const Type* A, bool allowsQualificationConversion) const
    -> bool {
  auto traits = unit_->typeTraits();
  if (traits.is_same(P, A)) return true;
  if (!allowsQualificationConversion) return false;
  return traits.is_qualification_convertible(traits.add_pointer(A),
                                             traits.add_pointer(P));
}

auto TypeDeduction::deduceQualifiedType(const QualType* P, const Type* A,
                                        bool allowsQualificationConversion)
    -> bool {
  auto cv = cv_qualifiers(A);
  if (!allowsQualificationConversion &&
      !is_at_least_as_cv_qualified(cv, P->cvQualifiers()))
    return false;
  auto traits = unit_->typeTraits();
  auto remainder = residual_cv_qualifiers(cv, P->cvQualifiers());
  auto element = traits.add_cv(traits.remove_cv(A), remainder);
  return deduceAtLevel(P->elementType(), element,
                       allowsQualificationConversion);
}

auto TypeDeduction::deduceFunctionType(const FunctionType* P, const Type* A,
                                       bool allowsFunctionPointerConversion)
    -> bool {
  auto function = type_cast<FunctionType>(A);
  if (!function) return false;
  if (P->isVariadic() != function->isVariadic()) return false;
  if (P->cvQualifiers() != function->cvQualifiers()) return false;
  if (P->refQualifier() != function->refQualifier()) return false;
  if (!deduceExceptionSpecification(P, function,
                                    allowsFunctionPointerConversion))
    return false;
  if (!deduce(P->returnType(), function->returnType())) return false;

  std::vector<TemplateArgument> patterns;
  for (auto type : P->parameterTypes()) patterns.push_back(type);
  std::vector<TemplateArgument> arguments;
  for (auto type : function->parameterTypes()) arguments.push_back(type);
  return deduce(patterns, arguments);
}

auto TypeDeduction::deduceExceptionSpecification(
    const FunctionType* P, const FunctionType* A,
    bool allowsFunctionPointerConversion) -> bool {
  auto expression = P->noexceptExpression();
  if (!expression) {
    if (P->isNoexcept() == A->isNoexcept()) return true;
    return allowsFunctionPointerConversion;
  }

  auto parameter = nonTypeParameterOf(expression);
  auto slot = slotOf(parameter);
  if (slot < 0) return skipNonDeducedContext();

  auto value = unit_->typeTraits().integral_constant(control()->getBoolType(),
                                                     A->isNoexcept());
  if (!value) return false;
  return record(slot, control()->getConstantArgumentSymbol(
                          control()->getBoolType(), ConstValue{*value}));
}

auto TypeDeduction::deduceArrayBound(ExpressionAST* P, std::size_t size)
    -> bool {
  auto parameter = nonTypeParameterOf(P);
  if (slotOf(parameter) < 0) return skipNonDeducedContext();
  return deduceNonTypeParameter(parameter, sizeArgument(size));
}

auto TypeDeduction::vectorSizeIn(VectorSizeKind sizeKind,
                                 const VectorType* vector) const
    -> std::optional<std::size_t> {
  if (sizeKind == VectorSizeKind::kElements) return vector->elementCount();
  auto elementSize = control()->memoryLayout()->sizeOf(vector->elementType());
  if (!elementSize) return std::nullopt;
  return vector->elementCount() * *elementSize;
}

auto TypeDeduction::deduceArrayBound(ExpressionAST* P, const Type* A) -> bool {
  if (auto array = type_cast<BoundedArrayType>(A))
    return deduceArrayBound(P, array->size());

  auto parameter = nonTypeParameterOf(P);
  if (slotOf(parameter) < 0) return skipNonDeducedContext();

  auto array = type_cast<UnresolvedBoundedArrayType>(A);
  if (!array) return false;
  auto boundId = ast_cast<IdExpressionAST>(array->size());
  auto bound = boundId ? boundId->symbol : nullptr;
  if (!bound) return skipNonDeducedContext();
  return deduceNonTypeParameter(parameter, bound);
}

auto TypeDeduction::deduceTemplateName(int slot, Symbol* templateSymbol)
    -> bool {
  templateSymbol = canonicalTemplate(templateSymbol);
  if (!symbol_cast<TemplateTypeParameterSymbol>(templateSymbol) &&
      !isValidTemplateTemplateArgument(parameters_[slot], templateSymbol))
    return false;
  return record(slot, templateSymbol);
}

auto TypeDeduction::deduceSpecializationArguments(
    std::span<const TemplateArgument> P, Symbol* templateSymbol,
    std::span<const TemplateArgument> A) -> bool {
  auto patterns = expand_template_arguments(P);
  auto arguments = expand_template_arguments(A);

  const auto expandsTail =
      !patterns.empty() && isPackExpansion(patterns.back());

  if (!expandsTail && patterns.size() < arguments.size() &&
      hasDefaultsFrom(templateSymbol, patterns.size())) {
    arguments.resize(patterns.size());
    (void)skipNonDeducedContext();
  }

  return deduce(patterns, arguments);
}

auto TypeDeduction::deduceClassSpecialization(const ClassType* P, const Type* A)
    -> bool {
  auto classType = type_cast<ClassType>(A);
  if (!classType) return false;

  auto patternTemplate = class_template_of(P->symbol());
  auto argumentTemplate = class_template_of(classType->symbol());
  if (!patternTemplate || !argumentTemplate)
    return unit_->typeTraits().is_same(P, A);
  if (patternTemplate->canonical() != argumentTemplate->canonical())
    return false;

  auto patterns = class_template_arguments(unit_, P->symbol());
  auto arguments = class_template_arguments(unit_, classType->symbol());
  return deduce(patterns, arguments);
}

auto TypeDeduction::deduceTemplateParameterSpecialization(
    const TemplateTypeParameterSpecializationType* P, const Type* A) -> bool {
  Symbol* templateSymbol = nullptr;
  std::vector<TemplateArgument> arguments;

  if (auto classType = type_cast<ClassType>(A)) {
    templateSymbol = class_template_of(classType->symbol());
    if (!templateSymbol) return false;
    arguments = class_template_arguments(unit_, classType->symbol());
  } else if (auto specialization =
                 type_cast<TemplateTypeParameterSpecializationType>(A)) {
    templateSymbol = typeArgument(specialization->templateParameter());
    arguments = specialization->templateArguments();
  } else {
    return false;
  }

  auto slot = slotOf(P->templateParameter());
  if (slot < 0) {
    (void)skipNonDeducedContext();
  } else if (!deduceTemplateName(slot, templateSymbol)) {
    return false;
  }

  return deduceSpecializationArguments(P->templateArguments(), templateSymbol,
                                       arguments);
}

auto TypeDeduction::deduceNonTypeParameter(NonTypeParameterSymbol* parameter,
                                           const TemplateArgument& A) -> bool {
  if (argumentKind(A) != ArgumentKind::kValue) return false;

  auto slot = slotOf(parameter);
  if (slot < 0) return deduceValue(TemplateArgument{parameter}, A);

  if (auto type = parameter->objectType(); type && isDependent(unit_, type)) {
    auto argumentType = valueType(A);
    if (argumentType && !deduce(type, argumentType)) return false;
  }

  if (auto symbol = argumentSymbol(A)) return record(slot, symbol);

  auto value = template_argument_value(A);
  if (!value) return skipNonDeducedContext();
  return record(slot, control()->getConstantArgumentSymbol(
                          parameter->objectType(), *value));
}

auto TypeDeduction::deduceTemplateArgumentValue(
    NonTypeParameterSymbol* parameter, const TemplateArgument& A) -> bool {
  auto slot = slotOf(parameter);
  if (slot >= 0 && holdsSpecifiedArgument(slot))
    return matchesSpecifiedArgument(slot, A);
  if (!deduceNonTypeParameter(parameter, A)) return false;
  if (slot < 0) return true;
  return hasCorrespondingParameterType(parameter, A);
}

auto TypeDeduction::holdsSpecifiedArgument(int slot) const -> bool {
  if (specified_[slot]) return true;
  auto prefix = packPrefixes_[slot];
  if (!prefix || !deduced_[slot]) return false;
  return std::ranges::contains(prefix->elements(), deduced_[slot]);
}

auto TypeDeduction::matchesSpecifiedArgument(int slot,
                                             const TemplateArgument& A) const
    -> bool {
  auto specified = template_argument_value(deduced_[slot]);
  if (!specified) return compare_single_arg(unit_, deduced_[slot], A);
  auto value = template_argument_value(A);
  if (!value) return compare_single_arg(unit_, deduced_[slot], A);
  auto converted =
      unit_->typeTraits().converted_constant_value(valueType(A), *specified);
  if (!converted) return false;
  return equivalent_values(*converted, *value);
}

auto TypeDeduction::hasCorrespondingParameterType(
    NonTypeParameterSymbol* parameter, const TemplateArgument& A) const
    -> bool {
  auto type = parameter->objectType();
  if (!type) return true;
  if (containsPlaceholderType(type)) return true;
  if (isDependent(unit_, type)) return true;
  if (!argumentSymbol(A)) return true;
  auto argumentType = valueType(A);
  if (!argumentType || isDependent(unit_, argumentType)) return true;
  auto traits = unit_->typeTraits();
  return traits.is_same(traits.remove_cv(type), traits.remove_cv(argumentType));
}

auto TypeDeduction::deduceValue(const TemplateArgument& P,
                                const TemplateArgument& A) -> bool {
  if (argumentKind(A) != ArgumentKind::kValue) return false;
  auto patternValue = template_argument_value(P);
  if (!patternValue) {
    if (compare_single_arg(unit_, P, A)) return true;
    return skipNonDeducedContext();
  }
  auto value = template_argument_value(A);
  return value && *value == *patternValue;
}

auto TypeDeduction::templateArguments() const
    -> std::optional<std::vector<TemplateArgument>> {
  std::vector<TemplateArgument> arguments;
  for (auto value : deduced_) {
    if (!value) return std::nullopt;
    arguments.push_back(value);
  }
  return arguments;
}

auto TypeDeduction::undeducedParameter() const -> TemplateParameterAST* {
  for (std::size_t slot = 0; slot < deduced_.size(); ++slot) {
    if (!deduced_[slot]) return parameters_[slot];
  }
  return nullptr;
}

}  // namespace cxx
