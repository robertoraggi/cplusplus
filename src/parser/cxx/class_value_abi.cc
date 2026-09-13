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

#include <cxx/class_value_abi.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <array>
#include <functional>

namespace cxx {
namespace {

struct Subobject {
  const Type* type = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  bool isBitField = false;
  bool isUnnamedBitField = false;
};

using SubobjectVisitor = std::function<bool(const Subobject&)>;

struct ClassValueAbiRules {
  Control* control;
  MemoryLayout* memoryLayout;
  TypeTraits traits;

  explicit ClassValueAbiRules(TranslationUnit* unit)
      : control(unit->control()),
        memoryLayout(unit->control()->memoryLayout()),
        traits(unit->typeTraits()) {}

  [[nodiscard]] auto sizeOf(const Type* type) const -> std::uint64_t {
    return memoryLayout->sizeOf(type).value_or(0);
  }

  [[nodiscard]] auto alignmentOf(const Type* type) const -> std::uint64_t {
    return memoryLayout->alignmentOf(type).value_or(0);
  }

  [[nodiscard]] auto abiType(const Type* type) const -> const Type* {
    type = traits.remove_cv(type);
    if (traits.is_reference(type)) return traits.add_pointer(type);
    return type;
  }

  [[nodiscard]] auto definitionOf(const Type* type) const -> ClassSymbol* {
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType) return nullptr;
    auto classSymbol = classType->symbol();
    return classSymbol ? classSymbol->resolvedDefinition() : nullptr;
  }

  auto forEachSubobject(const Type* type, std::uint64_t offset,
                        const SubobjectVisitor& visit) const -> bool;

  auto forEachClassSubobject(ClassSymbol* classSymbol, std::uint64_t offset,
                             const SubobjectVisitor& visit) const -> bool;

  [[nodiscard]] auto isEmptyRecord(const Type* type) -> bool;
  [[nodiscard]] auto isEmptyField(FieldSymbol* field) -> bool;
  [[nodiscard]] auto isAggregateForAbi(const Type* type) const -> bool;
  [[nodiscard]] auto singleElement(const Type* type) -> const Type*;

  [[nodiscard]] auto subobjectAt(const Type* type, std::uint64_t offset) const
      -> const Type*;

  [[nodiscard]] auto homogeneousAggregate(const Type* type,
                                          std::size_t maxMembers,
                                          const Type*& base,
                                          std::size_t& members) const -> bool;

  [[nodiscard]] auto coerceTo(std::vector<const Type*> slotTypes,
                              std::uint64_t stride = 0) const -> ClassValueAbi;

  [[nodiscard]] auto indirect(ClassValueAbiContext context, const Type* type,
                              bool byValue,
                              std::uint64_t minimumAlignment = 0) const
      -> ClassValueAbi;

  [[nodiscard]] auto classifySingleScalar(const Type* type,
                                          ClassValueAbiContext context)
      -> ClassValueAbi;
  [[nodiscard]] auto classifyAArch64(const Type* type,
                                     ClassValueAbiContext context)
      -> ClassValueAbi;
  [[nodiscard]] auto classifyX86_64(const Type* type,
                                    ClassValueAbiContext context) const
      -> ClassValueAbi;
};

auto ClassValueAbiRules::forEachClassSubobject(
    ClassSymbol* classSymbol, std::uint64_t offset,
    const SubobjectVisitor& visit) const -> bool {
  if (!classSymbol->isComplete()) return false;
  if (classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses())
    return false;

  auto layout = classSymbol->layout();
  if (!layout) return false;

  for (auto [base, info] : layout->sortedBaseInfos()) {
    if (!forEachSubobject(base->type(), offset + info.offset, visit))
      return false;
  }

  for (auto [field, info] : layout->sortedFieldInfos()) {
    if (field->isStatic()) continue;

    if (field->isBitField()) {
      if (!visit({.type = field->type(),
                  .offset = offset + info.offset,
                  .size = info.allocUnitSizeBytes,
                  .isBitField = true,
                  .isUnnamedBitField = field->name() == nullptr}))
        return false;
      continue;
    }

    if (!forEachSubobject(field->type(), offset + info.offset, visit))
      return false;
  }

  return true;
}

auto ClassValueAbiRules::forEachSubobject(const Type* type,
                                          std::uint64_t offset,
                                          const SubobjectVisitor& visit) const
    -> bool {
  type = abiType(type);

  if (auto arrayType = type_cast<BoundedArrayType>(type)) {
    auto elementType = arrayType->elementType();
    auto elementSize = sizeOf(elementType);
    if (!elementSize) return false;
    for (std::size_t i = 0; i < arrayType->size(); ++i) {
      if (!forEachSubobject(elementType, offset + i * elementSize, visit))
        return false;
    }
    return true;
  }

  if (traits.is_class_or_union(type)) {
    auto classSymbol = definitionOf(type);
    if (!classSymbol) return false;
    return forEachClassSubobject(classSymbol, offset, visit);
  }

  if (auto complexType = unqualified_cast<ComplexType>(type)) {
    auto elementType = complexType->elementType();
    auto elementSize = sizeOf(elementType);
    if (!elementSize) return false;
    return forEachSubobject(elementType, offset, visit) &&
           forEachSubobject(elementType, offset + elementSize, visit);
  }

  auto size = sizeOf(type);
  if (!size) return false;

  return visit({.type = type, .offset = offset, .size = size});
}

auto ClassValueAbiRules::isEmptyRecord(const Type* type) -> bool {
  auto classSymbol = definitionOf(type);
  if (!classSymbol || !classSymbol->isComplete()) return false;
  if (classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses())
    return false;

  for (auto base : classSymbol->baseClasses()) {
    if (!isEmptyRecord(base->symbol()->type())) return false;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!isEmptyField(field)) return false;
  }

  return true;
}

auto ClassValueAbiRules::isEmptyField(FieldSymbol* field) -> bool {
  if (field->isBitField() && !field->name()) return true;

  auto fieldType = traits.remove_cv(field->type());

  while (auto arrayType = type_cast<BoundedArrayType>(fieldType)) {
    if (!arrayType->size()) return true;
    fieldType = traits.remove_cv(arrayType->elementType());
  }

  if (!traits.is_class_or_union(fieldType)) return false;
  if (!traits.is_zero_size_subobject(field)) return false;

  return isEmptyRecord(fieldType);
}

auto ClassValueAbiRules::isAggregateForAbi(const Type* type) const -> bool {
  if (traits.is_class_or_union(type)) return true;
  if (traits.is_array(type)) return true;
  if (traits.is_complex(type)) return true;
  return traits.is_member_function_pointer(type);
}

auto ClassValueAbiRules::singleElement(const Type* type) -> const Type* {
  auto classSymbol = definitionOf(type);
  if (!classSymbol || !classSymbol->isComplete()) return nullptr;
  if (classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses())
    return nullptr;

  const Type* element = nullptr;

  for (auto base : classSymbol->baseClasses()) {
    auto baseType = base->symbol()->type();
    if (isEmptyRecord(baseType)) continue;
    if (element) return nullptr;
    element = singleElement(baseType);
    if (!element) return nullptr;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (isEmptyField(field)) continue;
    if (element) return nullptr;

    auto fieldType = abiType(field->type());

    while (auto arrayType = type_cast<BoundedArrayType>(fieldType)) {
      if (arrayType->size() != 1) return nullptr;
      fieldType = abiType(arrayType->elementType());
    }

    if (isAggregateForAbi(fieldType)) {
      element = singleElement(fieldType);
      if (!element) return nullptr;
    } else {
      element = fieldType;
    }
  }

  if (element && sizeOf(element) != sizeOf(type)) return nullptr;

  return element;
}

auto ClassValueAbiRules::subobjectAt(const Type* type,
                                     std::uint64_t offset) const
    -> const Type* {
  const Type* found = nullptr;

  (void)forEachSubobject(type, 0, [&](const Subobject& sub) {
    if (sub.isBitField) return true;
    if (sub.offset != offset) return true;
    found = sub.type;
    return false;
  });

  return found;
}

auto ClassValueAbiRules::homogeneousAggregate(const Type* type,
                                              std::size_t maxMembers,
                                              const Type*& base,
                                              std::size_t& members) const
    -> bool {
  base = nullptr;
  std::uint64_t lastOffset = 0;
  bool first = true;

  const auto ok = forEachSubobject(type, 0, [&](const Subobject& sub) {
    if (sub.isBitField) return false;
    if (!traits.is_floating_point(sub.type)) return false;
    if (!base) {
      base = sub.type;
    } else if (base->kind() != sub.type->kind()) {
      return false;
    }
    if (first || sub.offset != lastOffset) {
      if (++members > maxMembers) return false;
      lastOffset = sub.offset;
      first = false;
    }
    return true;
  });

  if (!ok || !base || !members) return false;

  return sizeOf(base) * members == sizeOf(type);
}

auto ClassValueAbiRules::coerceTo(std::vector<const Type*> slotTypes,
                                  std::uint64_t stride) const -> ClassValueAbi {
  ClassValueAbi abi{.kind = ClassValueAbi::Kind::Coerce};

  std::uint64_t offset = 0;
  std::uint64_t alignment = 1;

  for (auto slotType : slotTypes) {
    alignment = std::max(alignment, alignmentOf(slotType));
    abi.slots.push_back({.type = slotType, .offset = offset});
    offset += std::max(stride, sizeOf(slotType));
  }

  abi.coerceAlignment = alignment;
  abi.coerceSize = (offset + alignment - 1) & ~(alignment - 1);

  return abi;
}

auto ClassValueAbiRules::indirect(ClassValueAbiContext context,
                                  const Type* type, bool byValue,
                                  std::uint64_t minimumAlignment) const
    -> ClassValueAbi {
  const bool passedInMemory =
      byValue && context == ClassValueAbiContext::Argument;

  return {.kind = ClassValueAbi::Kind::Indirect,
          .passedInMemory = passedInMemory,
          .indirectAlignment =
              passedInMemory
                  ? std::max<std::uint64_t>(alignmentOf(type), minimumAlignment)
                  : alignmentOf(type)};
}

auto ClassValueAbiRules::classifySingleScalar(const Type* type,
                                              ClassValueAbiContext context)
    -> ClassValueAbi {
  if (isEmptyRecord(type)) return {.kind = ClassValueAbi::Kind::Empty};

  if (auto element = singleElement(type)) return coerceTo({element});

  return indirect(context, type, /*byValue=*/true);
}

auto ClassValueAbiRules::classifyAArch64(const Type* type,
                                         ClassValueAbiContext context)
    -> ClassValueAbi {
  if (isEmptyRecord(type)) return {.kind = ClassValueAbi::Kind::Empty};

  const Type* base = nullptr;
  std::size_t members = 0;

  if (homogeneousAggregate(type, 4, base, members)) {
    if (context == ClassValueAbiContext::Return)
      return {.kind = ClassValueAbi::Kind::Direct};
    return coerceTo({control->getBoundedArrayType(base, members)});
  }

  const auto size = sizeOf(type);
  const auto alignment = alignmentOf(type);
  const auto registerSize = memoryLayout->sizeOfPointer();

  if (!size || !alignment || size > 2 * registerSize)
    return indirect(context, type, /*byValue=*/false);

  if (alignment >= 2 * registerSize)
    return coerceTo({control->getUnsignedBitIntType(
        static_cast<int>(2 * registerSize) * 8)});

  if (context == ClassValueAbiContext::Return && size <= registerSize)
    return coerceTo(
        {control->getUnsignedBitIntType(static_cast<int>(size) * 8)});

  if (size <= registerSize)
    return coerceTo(
        {control->getUnsignedBitIntType(static_cast<int>(registerSize) * 8)});

  return coerceTo({control->getBoundedArrayType(
      control->getUnsignedBitIntType(static_cast<int>(registerSize) * 8), 2)});
}

enum class X86Class { NoClass, Integer, Sse, SseUp, X87, X87Up, Memory };

auto merge(X86Class lhs, X86Class rhs) -> X86Class {
  if (lhs == rhs) return lhs;
  if (lhs == X86Class::NoClass) return rhs;
  if (rhs == X86Class::NoClass) return lhs;
  if (lhs == X86Class::Memory || rhs == X86Class::Memory)
    return X86Class::Memory;
  if (lhs == X86Class::Integer || rhs == X86Class::Integer)
    return X86Class::Integer;
  if (lhs == X86Class::X87 || lhs == X86Class::X87Up || rhs == X86Class::X87 ||
      rhs == X86Class::X87Up)
    return X86Class::Memory;
  return X86Class::Sse;
}

auto ClassValueAbiRules::classifyX86_64(const Type* type,
                                        ClassValueAbiContext context) const
    -> ClassValueAbi {
  const auto memoryClass = [&] {
    return indirect(context, type, /*byValue=*/true,
                    memoryLayout->sizeOfPointer());
  };

  constexpr std::uint64_t kEightByte = 8;
  constexpr std::size_t kMaxEightBytes = 2;

  const auto size = sizeOf(type);
  if (!size || size > kMaxEightBytes * kEightByte) return memoryClass();

  std::array<X86Class, kMaxEightBytes> classes{X86Class::NoClass,
                                               X86Class::NoClass};

  const auto ok = forEachSubobject(type, 0, [&](const Subobject& sub) {
    if (sub.isUnnamedBitField) return true;

    const auto alignment = alignmentOf(sub.type);
    if (!sub.isBitField && (!alignment || sub.offset % alignment != 0))
      return false;

    const auto first = sub.offset / kEightByte;
    const auto last = (sub.offset + sub.size - 1) / kEightByte;
    if (last >= kMaxEightBytes) return false;

    if (!sub.isBitField && traits.is_floating_point(sub.type) &&
        sub.size > kEightByte) {
      classes[first] = merge(classes[first], X86Class::X87);
      classes[last] = merge(classes[last], X86Class::X87Up);
      return true;
    }

    const auto memberClass =
        !sub.isBitField && traits.is_floating_point(sub.type)
            ? X86Class::Sse
            : X86Class::Integer;

    for (auto unit = first; unit <= last; ++unit)
      classes[unit] = merge(classes[unit], memberClass);

    return true;
  });

  if (!ok) return memoryClass();

  if (classes[0] == X86Class::NoClass && classes[1] == X86Class::NoClass)
    return {.kind = ClassValueAbi::Kind::Empty};

  if (classes[0] == X86Class::Memory || classes[1] == X86Class::Memory)
    return memoryClass();

  if (classes[1] == X86Class::X87Up && classes[0] != X86Class::X87)
    return memoryClass();

  if (classes[0] == X86Class::X87) {
    if (context == ClassValueAbiContext::Argument) return memoryClass();
    return coerceTo({control->getLongDoubleType()});
  }

  std::vector<const Type*> slots;

  for (std::uint64_t offset = 0; offset < size; offset += kEightByte) {
    const auto remaining = size - offset;
    const auto unit = offset / kEightByte;

    if (classes[unit] != X86Class::Sse) {
      const auto bits = static_cast<int>(std::min(remaining, kEightByte) * 8);
      slots.push_back(control->getUnsignedBitIntType(bits));
      continue;
    }

    auto leaf = subobjectAt(type, offset);
    const bool isSingleFloat =
        leaf && sizeOf(leaf) * 2 == kEightByte &&
        (remaining <= kEightByte / 2 || !subobjectAt(type, offset + 4));

    slots.push_back(isSingleFloat
                        ? control->getFloatType()
                        : static_cast<const Type*>(control->getDoubleType()));
  }

  return coerceTo(std::move(slots), kEightByte);
}

}  // namespace

auto isClassValueDestroyedInCallee(const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return false;
  return classType->symbol()->resolvedDefinition()->isTrivialAbi();
}

auto usesClassValueAbi(const Type* type) -> bool {
  if (!type) return false;
  if (unqualified_cast<ClassType>(type)) return true;
  return unqualified_cast<ComplexType>(type) != nullptr;
}

auto classifyClassValueAbi(TranslationUnit* unit, const Type* type,
                           ClassValueAbiContext context) -> ClassValueAbi {
  if (!usesClassValueAbi(type)) return {};

  auto complexType = unqualified_cast<ComplexType>(type);
  auto classType = unqualified_cast<ClassType>(type);

  const Type* valueType = classType ? static_cast<const Type*>(classType)
                                    : static_cast<const Type*>(complexType);

  ClassValueAbiRules rules{unit};

  const auto abiKind = rules.memoryLayout->classValueAbiKind();
  if (abiKind == ClassValueAbiKind::kDefault) return {};

  if (classType) {
    auto classSymbol = rules.definitionOf(classType);

    if (!classSymbol || !classSymbol->isComplete())
      return rules.indirect(context, classType, /*byValue=*/false);

    if (rules.traits.is_non_trivial_for_calls(classType))
      return rules.indirect(context, classType, /*byValue=*/false);
  }

  switch (abiKind) {
    case ClassValueAbiKind::kSingleScalar:
      return rules.classifySingleScalar(valueType, context);

    case ClassValueAbiKind::kAArch64:
      return rules.classifyAArch64(valueType, context);

    case ClassValueAbiKind::kX86_64:
      return rules.classifyX86_64(valueType, context);

    case ClassValueAbiKind::kDefault:
      break;
  }

  return {};
}

}  // namespace cxx
