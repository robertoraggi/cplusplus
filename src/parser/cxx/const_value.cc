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
#include <cxx/reflection.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>

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
      if (std::holds_alternative<IndeterminateValue>(*current) ||
          std::holds_alternative<OutOfLifetimeValue>(*current))
        return false;
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
    if (traits_->is_member_pointer(type)) return true;
    if (address.isNull()) return !traits_->is_reference(type);
    if (traits_->is_reference(type)) {
      auto extent = unqualified_cast<BoundedArrayType>(address.arrayType());
      auto size = extent ? extent->size() : std::size_t{1};
      if (address.offset() < 0 || std::uintmax_t(address.offset()) >= size)
        return false;
    }
    if (address.stringLiteral() || address.typeInfoFor())
      return kind_ != ConstantResultKind::kTemplateArgument;

    auto referent = address.root() ? address.root() : address.symbol();
    if (!referent) return false;
    if (symbol_cast<FunctionSymbol>(referent)) return true;
    if (!referenceable(referent)) return false;

    auto temporary = symbol_cast<VariableSymbol>(referent);
    if (!temporary) return true;
    if (kind_ == ConstantResultKind::kTemplateArgument &&
        temporary->isFunctionLocalPredefined())
      return false;
    if (temporary->isTemplateParameterObject()) return true;
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

namespace {

class ImmediateObject {
 public:
  [[nodiscard]] auto check(const ConstValue& value) -> bool {
    pending_.push_back(&value);
    while (!pending_.empty()) {
      auto current = pending_.back();
      pending_.pop_back();
      if (std::visit(*this, *current)) return true;
    }
    return false;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<Meta>& reflection)
      -> bool {
    return reflection && !Reflection::equal(*reflection, Meta{});
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstObject>& object)
      -> bool {
    if (object)
      for (const auto& member : object->members())
        pending_.push_back(&member.value);
    return false;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<InitializerList>& list)
      -> bool {
    if (!list) return false;
    for (const auto& [element, type] : list->initializedElements())
      pending_.push_back(&element);
    if (list->hasFiller()) pending_.push_back(&std::get<0>(list->filler()));
    return false;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstAddress>& address)
      -> bool {
    if (!address || address->isNull()) return false;
    auto symbol = address->root() ? address->root() : address->symbol();
    if (auto function = symbol_cast<FunctionSymbol>(symbol))
      return function->isConsteval();
    if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
      if (variable->constValue()) follow(&*variable->constValue());
    } else if (auto field = symbol_cast<FieldSymbol>(symbol)) {
      if (field->constValue()) follow(&*field->constValue());
    }
    if (address->storage()) follow(&address->storage()->value());
    auto owner = address->owner();
    if (!owner || !visited_.insert(owner.get()).second) return false;
    return (*this)(owner);
  }

  [[nodiscard]] auto operator()(const auto&) -> bool { return false; }

 private:
  void follow(const ConstValue* value) {
    if (visited_.insert(value).second) pending_.push_back(value);
  }

  std::vector<const ConstValue*> pending_;
  std::unordered_set<const void*> visited_;
};

struct ManifestlyConstantEvaluatedInitializer {
  const TypeTraits& traits;

  template <typename S>
  [[nodiscard]] auto check(S* var) const -> bool {
    if (var->isConstexpr()) return true;
    if (traits.is_usable_in_constant_expressions(var)) return true;
    if (!has_static_storage_duration(var) && !has_thread_storage_duration(var))
      return false;
    return var->constValue() &&
           traits.is_constant_initialization(var, *var->constValue());
  }

  [[nodiscard]] auto operator()(VariableSymbol* var) const -> bool {
    return check(var);
  }
  [[nodiscard]] auto operator()(FieldSymbol* var) const -> bool {
    return check(var);
  }
  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

}  // namespace

auto isImmediateObject(const ConstValue& value) -> bool {
  return ImmediateObject{}.check(value);
}

auto declaresImmediateObject(VariableSymbol* variable) -> bool {
  auto definition = symbol_cast<VariableSymbol>(variable->resolvedDefinition());
  if (!definition || !definition->isConstexpr()) return false;
  auto value = definition->constValue();
  return value && isImmediateObject(*value);
}

auto TypeTraits::is_constexpr_representable(const ConstValue& value,
                                            const Type* type,
                                            ScopeSymbol* point) const -> bool {
  return ConstantResult{this, point, ConstantResultKind::kRepresentable}.check(
      value, type);
}

auto TypeTraits::is_constant_initialization(Symbol* variable,
                                            const ConstValue& value) const
    -> bool {
  auto staticResult = has_static_storage_duration(variable) ||
                      has_thread_storage_duration(variable);
  if (unit_->preprocessor()->cplusplusVersion() <= 202302L) staticResult = true;
  auto kind = staticResult ? ConstantResultKind::kStaticInitialization
                           : ConstantResultKind::kRepresentable;
  return ConstantResult{this, variable->parent(), kind}.check(value,
                                                              variable->type());
}

auto TypeTraits::has_manifestly_constant_evaluated_initializer(
    Symbol* variable) const -> bool {
  return visit(ManifestlyConstantEvaluatedInitializer{*this}, variable);
}

auto TypeTraits::is_constant_template_argument(const ConstValue& value,
                                               const Type* type) const -> bool {
  return ConstantResult{this, nullptr, ConstantResultKind::kTemplateArgument}
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

template <typename Transform>
class ValueCloner {
 public:
  explicit ValueCloner(Transform transform) : transform_(transform) {}

  [[nodiscard]] auto copy(const ConstValue& value) -> ConstValue {
    auto result = copyValue(value);
    if constexpr (!std::is_same_v<Transform, std::nullptr_t>) {
      transform_(result);
      if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
          object && *object)
        objects_.emplace_back(object->get(),
                              std::get<std::shared_ptr<ConstObject>>(result));
    }
    return result;
  }

  void finish() {
    for (auto address : addresses_) {
      if (!address->owner()) continue;
      for (const auto& [source, object] : objects_) {
        if (address->owner().get() != source) continue;
        address->setOwner(object);
        break;
      }
    }
  }

 private:
  [[nodiscard]] auto copyElement(const ConstValue& value, const Type* type)
      -> ConstValue {
    if constexpr (std::is_same_v<Transform, std::nullptr_t>)
      if (unqualified_cast<PointerType>(type)) return value;
    return copy(value);
  }

  [[nodiscard]] auto copyValue(const ConstValue& value) -> ConstValue {
    if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
      if (!*object) return value;
      auto result = std::make_shared<ConstObject>((*object)->type());
      for (const auto& member : (*object)->members())
        result->addMember(member.symbol,
                          copyElement(member.value, member.symbol->type()));
      return result;
    }
    if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      if (!*list) return value;
      std::vector<InitializerList::Element> elements;
      elements.reserve((*list)->initializedElements().size());
      for (const auto& [element, type] : (*list)->initializedElements())
        elements.emplace_back(copyElement(element, type), type);
      if (!(*list)->hasFiller())
        return std::make_shared<InitializerList>(std::move(elements));
      const auto& [filler, type] = (*list)->filler();
      return std::make_shared<InitializerList>(
          std::move(elements), (*list)->size(),
          InitializerList::Element{copyElement(filler, type), type});
    }
    if (auto complex = std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
      if (!*complex) return value;
      return std::make_shared<ConstComplex>(copy((*complex)->real()),
                                            copy((*complex)->imag()));
    }
    if constexpr (!std::is_same_v<Transform, std::nullptr_t>) {
      if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
        if (!*address) return value;
        auto result = std::make_shared<ConstAddress>(**address);
        addresses_.push_back(result);
        return result;
      }
      if (auto meta = std::get_if<std::shared_ptr<Meta>>(&value)) {
        if (!*meta) return value;
        auto result = std::make_shared<Meta>(**meta);
        if (auto expression = std::get_if<Meta::ConstExpr>(&result->value))
          expression->value = copy(expression->value);
        if (auto object =
                std::get_if<std::shared_ptr<ConstAddress>>(&result->value)) {
          auto copied = copy(ConstValue{*object});
          *object = std::get<std::shared_ptr<ConstAddress>>(std::move(copied));
        }
        return result;
      }
    }
    return value;
  }

  Transform transform_;
  std::vector<std::pair<const ConstObject*, std::shared_ptr<ConstObject>>>
      objects_;
  std::vector<std::shared_ptr<ConstAddress>> addresses_;
};

}  // namespace

auto cloneValue(const ConstValue& value) -> ConstValue {
  return ValueCloner{nullptr}.copy(value);
}

auto cloneValue(const ConstValue& value,
                const std::function<void(ConstValue&)>& transform)
    -> ConstValue {
  auto cloner = ValueCloner{transform};
  auto result = cloner.copy(value);
  cloner.finish();
  return result;
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
  if (isUnion()) {
    members_.clear();
  } else if (auto owner = symbol_cast<ClassSymbol>(symbol->parent());
             owner && owner->isUnion()) {
    std::erase_if(members_, [owner](const Member& member) {
      return member.symbol && member.symbol->parent() == owner;
    });
  }
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
  if (storage_ && other.storage_) return storage_ == other.storage_;
  if (root_ || other.root_)
    return root_ == other.root_ && storage_ == other.storage_;
  if (symbol_ != other.symbol_ || owner_ != other.owner_) return false;
  if (string_ != other.string_ || typeInfoFor_ != other.typeInfoFor_)
    return false;
  return owner_ || storage_ == other.storage_;
}

auto ConstAddress::sameArray(const ConstAddress& other) const -> bool {
  if (!sameTarget(other)) return false;
  if (bool(arrayType_) != bool(other.arrayType_)) return false;
  return unqualified_type(elementType_) ==
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

namespace {

[[nodiscard]] auto designationPath(const ConstAddress& address)
    -> std::vector<ConstAddress::Subobject> {
  std::vector<ConstAddress::Subobject> path;
  for (const auto& step : address.subobjectPath()) {
    auto symbol = std::get_if<Symbol*>(&step);
    if (symbol && symbol_cast<BaseClassSymbol>(*symbol)) continue;
    path.push_back(step);
  }
  if (address.arrayType()) path.emplace_back(std::size_t(address.offset()));
  return path;
}

[[nodiscard]] auto memberObjectType(const Type* enclosing, Symbol* member,
                                    const TypeTraits& traits) -> const Type* {
  auto field = symbol_cast<FieldSymbol>(member);
  auto type = field ? field->type() : traits.aggregate_element_type(member);
  if (!type) return nullptr;
  if (traits.is_reference(type)) return traits.remove_reference(type);
  auto cv = cv_qualifiers(enclosing);
  if (field && field->isMutable()) cv = cv & ~CvQualifiers::kConst;
  return traits.add_cv(type, cv);
}

[[nodiscard]] auto declaredObjectType(const ConstAddress& address)
    -> const Type* {
  if (address.storage()) return address.storage()->type();
  if (address.root()) return address.root()->type();
  if (address.owner()) return address.owner()->type();
  return address.symbol() ? address.symbol()->type() : nullptr;
}

}  // namespace

auto ConstAddress::sameObject(const ConstAddress& other) const -> bool {
  if (!sameTarget(other)) return false;
  if (displacement_ != other.displacement_) return false;
  if (unqualified_type(designatedType()) !=
      unqualified_type(other.designatedType()))
    return false;
  return designationPath(*this) == designationPath(other);
}

auto ConstAddress::designatedType() const -> const Type* {
  if (elementType_) return elementType_;
  return symbol_ ? symbol_->type() : nullptr;
}

auto ConstAddress::objectHash() const -> std::size_t {
  std::size_t seed = 0;
  if (root_) {
    hash_combine(seed, root_);
    hash_combine(seed, storage_.get());
  } else {
    hash_combine(seed, symbol_);
    hash_combine(seed, owner_.get());
    hash_combine(seed, string_);
    hash_combine(seed, typeInfoFor_);
    if (!owner_) hash_combine(seed, storage_.get());
  }
  for (const auto& step : designationPath(*this))
    std::visit([&](auto value) { hash_combine(seed, value); }, step);
  hash_combine(seed, displacement_);
  return seed;
}

auto ConstAddress::objectType(const TypeTraits& traits) const -> const Type* {
  auto type = declaredObjectType(*this);
  if (!type) return nullptr;
  type = traits.remove_reference(type);
  if (subobjectPath_.empty() && owner_ && symbol_)
    type = memberObjectType(type, symbol_, traits);
  for (const auto& step : subobjectPath_) {
    auto member = std::get_if<Symbol*>(&step);
    type = member ? memberObjectType(type, *member, traits)
                  : traits.get_element_type(type);
    if (!type) return nullptr;
  }
  if (arrayType_) type = traits.get_element_type(type);
  if (!type || !elementType_) return type;
  return traits.add_cv(traits.remove_cv(elementType_), cv_qualifiers(type));
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

[[nodiscard]] auto floatingIdentity(long double value) {
  auto bytes =
      std::bit_cast<std::array<unsigned char, sizeof(long double)>>(value);
#if LDBL_MANT_DIG == 64
  if constexpr (std::endian::native == std::endian::little)
    std::fill(bytes.begin() + 10, bytes.end(), 0);
  else
    std::fill(bytes.begin(), bytes.end() - 10, 0);
#endif
  return bytes;
}

[[nodiscard]] auto memberPointerIdentity(const ConstAddress& address)
    -> Symbol* {
  if (!address.root() || address.root() != address.symbol() ||
      address.owner() || address.storage())
    return nullptr;
  if (auto field = symbol_cast<FieldSymbol>(address.symbol());
      field && !field->isStatic())
    return field;
  if (auto function = symbol_cast<FunctionSymbol>(address.symbol());
      function && function->isImplicitObjectMemberFunction())
    return function;
  return nullptr;
}

struct EquivalentValues {
  const ConstValue& rhs;

  template <typename T>
  [[nodiscard]] auto other() const -> const T* {
    return std::get_if<T>(&rhs);
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<Meta>& lhs) const
      -> bool {
    auto value = other<std::shared_ptr<Meta>>();
    return value && *value && lhs && Reflection::equal(*lhs, **value);
  }

  [[nodiscard]] auto operator()(const ConstInt& lhs) const -> bool {
    auto value = other<ConstInt>();
    if (value) return lhs == *value;
    auto address = other<std::shared_ptr<ConstAddress>>();
    return lhs.isZero() && address && *address && (*address)->isNull();
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
    return value && floatingIdentity(lhs) == floatingIdentity(*value);
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
    if (!lhs) return false;
    if (auto integer = other<ConstInt>())
      return lhs->isNull() && integer->isZero();
    auto value = other<std::shared_ptr<ConstAddress>>();
    if (!value || !*value) return false;
    if (lhs->isNull() || (*value)->isNull())
      return lhs->isNull() && (*value)->isNull();
    if (auto member = memberPointerIdentity(*lhs))
      return member == memberPointerIdentity(**value);
    return lhs->sameObject(**value);
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

class ValueHasher {
 public:
  [[nodiscard]] auto hash(const ConstValue& value)
      -> std::optional<std::size_t> {
    return std::visit(*this, value);
  }

  [[nodiscard]] auto operator()(const ConstInt& value)
      -> std::optional<std::size_t> {
    return value.hash();
  }

  [[nodiscard]] auto operator()(float value) -> std::optional<std::size_t> {
    return std::hash<std::uint32_t>{}(std::bit_cast<std::uint32_t>(value));
  }

  [[nodiscard]] auto operator()(double value) -> std::optional<std::size_t> {
    return std::hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(value));
  }

  [[nodiscard]] auto operator()(long double value)
      -> std::optional<std::size_t> {
    std::size_t seed = 0;
    for (auto byte : floatingIdentity(value)) hash_combine(seed, byte);
    return seed;
  }

  [[nodiscard]] auto operator()(const ConstFloat16& value)
      -> std::optional<std::size_t> {
    return value.bits;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<Meta>& value)
      -> std::optional<std::size_t> {
    if (!value) return std::nullopt;
    return Reflection::hash(*value);
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstAddress>& value)
      -> std::optional<std::size_t> {
    if (!value) return std::nullopt;
    if (value->isNull()) return ConstInt{0}.hash();
    if (auto member = memberPointerIdentity(*value))
      return std::hash<Symbol*>{}(member);
    return value->objectHash();
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstObject>& value)
      -> std::optional<std::size_t> {
    if (!value) return std::nullopt;
    if (auto it = objects_.find(value.get()); it != objects_.end())
      return it->second;
    std::size_t seed = 0;
    hash_combine(seed, unqualified_type(value->type()));
    for (const auto& member : value->members()) {
      hash_combine(seed, member.symbol);
      if (!append(seed, member.value)) return std::nullopt;
    }
    objects_.emplace(value.get(), seed);
    return seed;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<InitializerList>& value)
      -> std::optional<std::size_t> {
    if (!value) return std::nullopt;
    std::size_t seed = value->size();
    for (std::size_t index = 0; index < value->size(); ++index)
      if (!append(seed, std::get<0>(value->element(index))))
        return std::nullopt;
    return seed;
  }

  [[nodiscard]] auto operator()(const std::shared_ptr<ConstComplex>& value)
      -> std::optional<std::size_t> {
    if (!value) return std::nullopt;
    std::size_t seed = 0;
    if (!append(seed, value->real()) || !append(seed, value->imag()))
      return std::nullopt;
    return seed;
  }

  [[nodiscard]] auto operator()(const StringLiteral* value)
      -> std::optional<std::size_t> {
    return std::hash<const void*>{}(value);
  }

  [[nodiscard]] auto operator()(OutOfLifetimeValue)
      -> std::optional<std::size_t> {
    return 0;
  }

  template <typename T>
  [[nodiscard]] auto operator()(const T&) -> std::optional<std::size_t> {
    return std::nullopt;
  }

 private:
  [[nodiscard]] auto append(std::size_t& seed, const ConstValue& value)
      -> bool {
    auto key = hash(value);
    if (!key) return false;
    hash_combine(seed, *key);
    return true;
  }

  std::unordered_map<const ConstObject*, std::size_t> objects_;
};

}  // namespace

auto equivalent_values(const ConstValue& lhs, const ConstValue& rhs) -> bool {
  return std::visit(EquivalentValues{rhs}, lhs);
}

auto hash_value(const ConstValue& value) -> std::optional<std::size_t> {
  return ValueHasher{}.hash(value);
}

}  // namespace cxx
