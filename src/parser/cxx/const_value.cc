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
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace cxx {

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

auto ConstObject::subobject(const Symbol* symbol) const -> const ConstValue* {
  for (const auto& member : members_) {
    if (member.symbol == symbol) return &member.value;
  }
  for (const auto& member : members_) {
    if (!isTransparentSubobject(member.symbol)) continue;
    auto nested = std::get_if<std::shared_ptr<ConstObject>>(&member.value);
    if (!nested || !*nested) continue;
    if (auto found = (*nested)->subobject(symbol)) return found;
  }
  return nullptr;
}

auto ConstObject::mutableSubobject(const Symbol* symbol) -> ConstValue* {
  for (auto& member : members_) {
    if (member.symbol == symbol) return &member.value;
  }
  for (auto& member : members_) {
    if (!isTransparentSubobject(member.symbol)) continue;
    auto nested = std::get_if<std::shared_ptr<ConstObject>>(&member.value);
    if (!nested || !*nested) continue;
    if (auto found = (*nested)->mutableSubobject(symbol)) return found;
  }
  return nullptr;
}

auto ConstAddress::sameTarget(const ConstAddress& other) const -> bool {
  return symbol_ == other.symbol_ && owner_ == other.owner_ &&
         string_ == other.string_ && typeInfoFor_ == other.typeInfoFor_ &&
         root_ == other.root_;
}

auto ConstObject::operator==(const ConstObject& other) const -> bool {
  if (type_ != other.type_) return false;
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
    return lhs->sameTarget(**value) && lhs->offset() == (*value)->offset() &&
           lhs->displacement() == (*value)->displacement();
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