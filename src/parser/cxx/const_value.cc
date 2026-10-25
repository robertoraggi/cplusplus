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

#include <cxx/const_value.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <ranges>
#include <span>

namespace cxx {

auto ConstFloat16::value() const -> float {
  if ((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0) {
    auto sign = std::uint32_t(bits & 0x8000) << 16;
    auto payload = std::uint32_t(bits & 0x03ff) << 13;
    return std::bit_cast<float>(sign | 0x7fc00000u | payload);
  }

  constexpr auto format =
      FloatingPointFormat{.exponentBits = 5, .significandDigits = 11};
  return static_cast<float>(format.value(bits));
}

namespace {
[[nodiscard]] auto isTransparentSubobject(const Symbol* symbol) -> bool {
  if (symbol_cast<BaseClassSymbol>(const_cast<Symbol*>(symbol))) return true;
  auto field = symbol_cast<FieldSymbol>(const_cast<Symbol*>(symbol));
  return field && anonymous_member_class(field);
}
}  // namespace

auto asConstexprUnknownObject(const ConstValue& value)
    -> std::shared_ptr<ConstObject> {
  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!object || !*object) return {};
  if (!(*object)->isConstexprUnknown()) return {};
  return *object;
}

namespace {

enum class ConstantResultKind {
  kInitialized,
  kRepresentable,
  kStaticInitialization,
  kTemplateArgument
};

class ConstantResult {
 public:
  ConstantResult(const TypeTraits* traits, ScopeSymbol* point,
                 ConstantResultKind kind)
      : traits_(traits), point_(point), kind_(kind) {}

  [[nodiscard]] auto check(const ConstValue& value, const Type* type) -> bool {
    std::vector<std::pair<const ConstValue*, const Type*>> pending{
        {&value, type}};
    while (!pending.empty()) {
      auto [current, currentType] = pending.back();
      pending.pop_back();
      if (std::holds_alternative<IndeterminateValue>(*current)) return false;
      if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(current)) {
        if (!*address || !checkAddress(**address, currentType, pending))
          return false;
      } else if (auto object =
                     std::get_if<std::shared_ptr<ConstObject>>(current)) {
        if (!*object || (*object)->isConstexprUnknown() ||
            (*object)->lifetimeEnded())
          return false;
        if (auto classType = unqualified_cast<ClassType>((*object)->type())) {
          auto definition = classType->definition();
          if (!definition) return false;
          for (auto base : definition->baseClasses())
            if (!(*object)->subobject(base)) return false;
        }
        for (const auto& member : (*object)->members())
          pending.emplace_back(&member.value,
                               traits_ ? traits_->aggregate_element_type(
                                             const_cast<Symbol*>(member.symbol))
                                       : member.symbol->type());
      } else if (auto list =
                     std::get_if<std::shared_ptr<InitializerList>>(current)) {
        if (!*list) return false;
        for (const auto& [element, elementType] :
             (*list)->initializedElements())
          pending.emplace_back(&element, elementType);
        if ((*list)->hasFiller()) {
          const auto& [filler, fillerType] = (*list)->filler();
          pending.emplace_back(&filler, fillerType);
        }
      } else if (auto complex =
                     std::get_if<std::shared_ptr<ConstComplex>>(current)) {
        if (!*complex) return false;
        pending.emplace_back(
            &(*complex)->real(),
            traits_ ? traits_->get_element_type(currentType) : nullptr);
        pending.emplace_back(
            &(*complex)->imag(),
            traits_ ? traits_->get_element_type(currentType) : nullptr);
      } else if (kind_ == ConstantResultKind::kTemplateArgument &&
                 std::holds_alternative<const StringLiteral*>(*current)) {
        return false;
      }
    }
    return true;
  }

 private:
  [[nodiscard]] auto referenceable(Symbol* referent) const -> bool {
    if (has_static_storage_duration(referent)) return true;
    if (has_thread_storage_duration(referent)) return false;
    if (!symbol_cast<VariableSymbol>(referent) &&
        !symbol_cast<ParameterSymbol>(referent))
      return false;
    if (kind_ != ConstantResultKind::kRepresentable || !point_) return false;
    auto function = referent->enclosingFunction();
    return function && function == point_->enclosingFunctionOrSelf();
  }

  [[nodiscard]] auto checkAddress(
      const ConstAddress& address, const Type* type,
      std::vector<std::pair<const ConstValue*, const Type*>>& pending) -> bool {
    if (address.lifetimeEnded()) return false;
    auto owner = address.owner();
    if (owner && owner->isConstexprUnknown()) return false;
    if (kind_ == ConstantResultKind::kInitialized) return true;
    if (address.isNull() || traits_->is_member_pointer(type)) return true;
    if (address.stringLiteral() || address.typeInfoFor())
      return kind_ != ConstantResultKind::kTemplateArgument;

    auto referent = address.root() ? address.root() : address.symbol();
    if (!referent) return false;
    if (symbol_cast<FunctionSymbol>(referent)) return true;
    if (!referenceable(referent)) return false;

    auto temporary = symbol_cast<VariableSymbol>(referent);
    if (!temporary) return true;
    if (!temporary->lifetimeExtendingDeclaration() && temporary->parent())
      return true;
    if (kind_ == ConstantResultKind::kTemplateArgument) return false;
    if (!traits_->is_reference(type)) return true;
    if (std::ranges::contains(extendedTemporaries_, temporary)) return true;
    extendedTemporaries_.push_back(temporary);
    if (!temporary->constValue()) return false;
    pending.emplace_back(&*temporary->constValue(), temporary->type());
    return true;
  }

  const TypeTraits* traits_;
  ScopeSymbol* point_;
  ConstantResultKind kind_;
  std::vector<VariableSymbol*> extendedTemporaries_;
};

}  // namespace

auto isFullyInitialized(const ConstValue& value) -> bool {
  return ConstantResult{nullptr, nullptr, ConstantResultKind::kInitialized}
      .check(value, nullptr);
}

auto isConstexprRepresentable(const TypeTraits& traits, const ConstValue& value,
                              const Type* type, ScopeSymbol* point) -> bool {
  return ConstantResult{&traits, point, ConstantResultKind::kRepresentable}
      .check(value, type);
}

auto isConstantInitialization(const TypeTraits& traits, Symbol* variable,
                              const ConstValue& value) -> bool {
  auto staticResult = has_static_storage_duration(variable) ||
                      has_thread_storage_duration(variable);
  if (traits.unit()->preprocessor()->cplusplusVersion() <= 202302L)
    staticResult = true;
  auto kind = staticResult ? ConstantResultKind::kStaticInitialization
                           : ConstantResultKind::kRepresentable;
  return ConstantResult{&traits, variable->parent(), kind}.check(
      value, variable->type());
}

auto isConstantTemplateArgument(const TypeTraits& traits,
                                const ConstValue& value, const Type* type)
    -> bool {
  return ConstantResult{&traits, nullptr, ConstantResultKind::kTemplateArgument}
      .check(value, type);
}

auto ConstObject::makeConstexprUnknown(const Type* type)
    -> std::shared_ptr<ConstObject> {
  auto object = std::make_shared<ConstObject>(type);
  object->setConstexprUnknown(true);
  return object;
}

auto ConstObject::isUnion() const -> bool {
  auto classType = unqualified_cast<ClassType>(type_);
  auto classSymbol = classType ? classType->symbol() : nullptr;
  return classSymbol && classSymbol->isUnion();
}

auto hasObjectIdentity(const ConstValue& value) -> bool {
  return std::holds_alternative<std::shared_ptr<ConstObject>>(value) ||
         std::holds_alternative<std::shared_ptr<InitializerList>>(value) ||
         std::holds_alternative<std::shared_ptr<ConstComplex>>(value);
}

namespace {

[[nodiscard]] auto cloneValueOfType(const ConstValue& value, const Type* type)
    -> ConstValue {
  if (unqualified_cast<PointerType>(type)) return value;
  return cloneValue(value);
}

}  // namespace

auto cloneValue(const ConstValue& value) -> ConstValue {
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return value;
    auto copy = std::make_shared<ConstObject>((*object)->type());
    for (const auto& member : (*object)->members())
      copy->addMember(member.symbol,
                      cloneValueOfType(member.value, member.symbol->type()));
    return ConstValue{std::move(copy)};
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return value;
    std::vector<InitializerList::Element> elements;
    elements.reserve((*list)->initializedElements().size());
    for (const auto& [element, type] : (*list)->initializedElements())
      elements.emplace_back(cloneValueOfType(element, type), type);
    if (!(*list)->hasFiller())
      return ConstValue{std::make_shared<InitializerList>(std::move(elements))};
    const auto& [filler, fillerType] = (*list)->filler();
    return ConstValue{std::make_shared<InitializerList>(
        std::move(elements), (*list)->size(),
        InitializerList::Element{cloneValueOfType(filler, fillerType),
                                 fillerType})};
  }

  if (auto complexValue = std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
    if (!*complexValue) return value;
    return ConstValue{
        std::make_shared<ConstComplex>(cloneValue((*complexValue)->real()),
                                       cloneValue((*complexValue)->imag()))};
  }

  return value;
}

InitializerList::InitializerList(std::vector<Element> elements,
                                 std::size_t size, Element filler)
    : elements_(std::make_move_iterator(elements.begin()),
                std::make_move_iterator(elements.end())),
      size_(std::max(size, elements_.size())),
      filler_(std::move(filler)) {}

auto InitializerList::element(std::size_t index) const -> const Element& {
  if (index < elements_.size()) return elements_[index];
  return filler_;
}

auto InitializerList::mutableElement(std::size_t index) -> Element& {
  if (index >= elements_.size()) {
    while (elements_.size() <= index) {
      const auto& [filler, type] = filler_;
      elements_.emplace_back(cloneValueOfType(filler, type), type);
    }
  }
  return elements_[index];
}

auto InitializerList::distinctElement(std::size_t index) -> const Element& {
  if (index >= elements_.size() && hasObjectIdentity(std::get<0>(filler_)))
    return mutableElement(index);
  return element(index);
}

void InitializerList::append(ConstValue value, const Type* type) {
  elements_.emplace_back(std::move(value), type);
  size_ = std::max(size_, elements_.size());
}

void InitializerList::setSize(std::size_t size) {
  size_ = std::max(size, elements_.size());
}

void InitializerList::setFiller(Element filler) { filler_ = std::move(filler); }

auto ConstObject::addMember(const Symbol* symbol, ConstValue value)
    -> ConstValue* {
  if (isUnion()) members_.clear();
  members_.push_back({symbol, std::move(value)});
  return &members_.back().value;
}

void ConstObject::setMember(const Symbol* symbol, ConstValue value) {
  if (!isUnion()) {
    for (auto& member : members_) {
      if (member.symbol == symbol) {
        member.value = std::move(value);
        return;
      }
    }
  }
  addMember(symbol, std::move(value));
}

namespace {
struct LocatedSubobject {
  const ConstValue* value = nullptr;
  std::vector<Symbol*> path;
};

[[nodiscard]] auto locateSubobject(const ConstObject& object,
                                   const Symbol* symbol) -> LocatedSubobject {
  struct Candidate {
    const ConstObject* object;
    std::vector<Symbol*> path;
  };
  std::vector<Candidate> pending{{&object, {}}};
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
    for (const auto& member : current.object->members()) {
      if (member.symbol != symbol) continue;
      current.path.push_back(const_cast<Symbol*>(symbol));
      return {&member.value, std::move(current.path)};
    }

    for (const auto& member : std::views::reverse(current.object->members())) {
      if (!isTransparentSubobject(member.symbol)) continue;
      auto nested = std::get_if<std::shared_ptr<ConstObject>>(&member.value);
      if (!nested || !*nested) continue;
      auto path = current.path;
      path.push_back(const_cast<Symbol*>(member.symbol));
      pending.push_back({nested->get(), std::move(path)});
    }
  }
  return {};
}
}  // namespace

auto ConstObject::subobject(const Symbol* symbol) const -> const ConstValue* {
  return locateSubobject(*this, symbol).value;
}

auto ConstObject::mutableSubobject(const Symbol* symbol) -> ConstValue* {
  return const_cast<ConstValue*>(subobject(symbol));
}

auto ConstObject::subobjectPath(const Symbol* symbol) const
    -> std::optional<std::vector<Symbol*>> {
  auto found = locateSubobject(*this, symbol);
  if (!found.value) return std::nullopt;
  return std::move(found.path);
}

auto ConstAddress::memberPointer(Symbol* symbol, std::intmax_t adjustment)
    -> std::shared_ptr<ConstAddress> {
  auto result = std::make_shared<ConstAddress>(symbol);
  result->setMemberAdjustment(adjustment);
  return result;
}

ConstAddress::ConstAddress(Symbol* symbol, std::intmax_t offset)
    : symbol_(symbol ? symbol->canonical() : nullptr),
      offset_(offset),
      root_(symbol_) {}

auto ConstAddress::sameTarget(const ConstAddress& other) const -> bool {
  if (root_ || other.root_)
    return root_ == other.root_ && storage_ == other.storage_;
  if (symbol_ != other.symbol_ || owner_ != other.owner_) return false;
  if (string_ != other.string_ || typeInfoFor_ != other.typeInfoFor_)
    return false;
  return owner_ || storage_ == other.storage_;
}

auto ConstAddress::sameArray(const ConstAddress& other) const -> bool {
  return sameTarget(other) && arrayType_ == other.arrayType_ &&
         unqualified_type(elementType_) ==
             unqualified_type(other.elementType_) &&
         subobjectPath_ == other.subobjectPath_ &&
         classDisplacement_ == other.classDisplacement_;
}

auto ConstAddress::sameContainingObject(const ConstAddress& other) const
    -> bool {
  if (!root_ && !other.root_) return owner_ && owner_ == other.owner_;
  if (root_ != other.root_ || storage_ != other.storage_) return false;
  if (subobjectPath_.empty() || other.subobjectPath_.empty()) return false;
  if (!std::holds_alternative<Symbol*>(subobjectPath_.back()) ||
      !std::holds_alternative<Symbol*>(other.subobjectPath_.back()))
    return false;
  return std::ranges::equal(
      std::span{subobjectPath_}.first(subobjectPath_.size() - 1),
      std::span{other.subobjectPath_}.first(other.subobjectPath_.size() - 1));
}

auto ConstAddress::sameLocation(const ConstAddress& other) const -> bool {
  if (sameArray(other) && offset_ == other.offset_) return true;
  if (displacement_ != other.displacement_) return false;
  if (root_ && root_ == other.root_) return storage_ == other.storage_;
  if (root_ || other.root_) return false;
  return owner_ && owner_ == other.owner_;
}

auto ConstAddress::lifetimeEnded() const -> bool {
  if (storage_ && storage_->lifetimeEnded()) return true;
  return owner_ && owner_->lifetimeEnded();
}

auto ConstAddress::isNull() const -> bool {
  return !symbol_ && !owner_ && !storage_ && !string_ && !typeInfoFor_;
}

auto ConstObject::operator==(const ConstObject& other) const -> bool {
  if (unqualified_type(type_) != unqualified_type(other.type_)) return false;
  if (members_.size() != other.members_.size()) return false;
  for (std::size_t i = 0; i < members_.size(); ++i) {
    if (members_[i].symbol != other.members_[i].symbol) return false;
    if (!equivalent_values(members_[i].value, other.members_[i].value))
      return false;
  }
  return true;
}

namespace {

struct EquivalentValues {
  const ConstValue& rhs;

  template <typename T>
  [[nodiscard]] auto other() const -> const T* {
    return std::get_if<T>(&rhs);
  }

  [[nodiscard]] auto operator()(const ConstInt& lhs) const -> bool {
    auto value = other<ConstInt>();
    return value && lhs == *value;
  }

  [[nodiscard]] auto operator()(float lhs) const -> bool {
    auto value = other<float>();
    return value && std::bit_cast<std::uint32_t>(lhs) ==
                        std::bit_cast<std::uint32_t>(*value);
  }

  [[nodiscard]] auto operator()(double lhs) const -> bool {
    auto value = other<double>();
    return value && std::bit_cast<std::uint64_t>(lhs) ==
                        std::bit_cast<std::uint64_t>(*value);
  }

  [[nodiscard]] auto operator()(long double lhs) const -> bool {
    auto value = other<long double>();
    return value && lhs == *value;
  }

  [[nodiscard]] auto operator()(
      const std::shared_ptr<InitializerList>& lhs) const -> bool {
    auto value = other<std::shared_ptr<InitializerList>>();
    if (!value || !*value || !lhs) return false;
    const auto& rhs = **value;
    if (lhs->size() != rhs.size()) return false;
    const auto initialized = std::max(lhs->initializedElements().size(),
                                      rhs.initializedElements().size());
    for (std::size_t index = 0; index < initialized; ++index) {
      if (!equivalent_values(std::get<0>(lhs->element(index)),
                             std::get<0>(rhs.element(index))))
        return false;
    }
    if (initialized == lhs->size()) return true;
    return equivalent_values(std::get<0>(lhs->filler()),
                             std::get<0>(rhs.filler()));
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstObject>& lhs) const
      -> bool {
    auto value = other<std::shared_ptr<ConstObject>>();
    return value && *value && lhs && *lhs == **value;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstAddress>& lhs) const
      -> bool {
    auto value = other<std::shared_ptr<ConstAddress>>();
    if (!value || !*value || !lhs) return false;
    return lhs->sameArray(**value) && lhs->offset() == (*value)->offset() &&
           lhs->displacement() == (*value)->displacement() &&
           lhs->memberAdjustment() == (*value)->memberAdjustment();
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstComplex>& lhs) const
      -> bool {
    auto value = other<std::shared_ptr<ConstComplex>>();
    if (!value || !*value || !lhs) return false;
    return equivalent_values(lhs->real(), (*value)->real()) &&
           equivalent_values(lhs->imag(), (*value)->imag());
  }

  template <typename T>
  [[nodiscard]] auto operator()(const T& lhs) const -> bool {
    auto value = other<T>();
    return value && lhs == *value;
  }
};

}  // namespace

auto equivalent_values(const ConstValue& lhs, const ConstValue& rhs) -> bool {
  return std::visit(EquivalentValues{rhs}, lhs);
}

}  // namespace cxx
