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

namespace cxx {
namespace {

auto isEmptyClassForAbi(TypeTraits& traits, ClassSymbol* classSymbol) -> bool {
  if (!classSymbol || !classSymbol->isComplete()) return false;
  if (classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses())
    return false;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass) baseClass = baseClass->resolvedDefinition();
    if (!isEmptyClassForAbi(traits, baseClass)) return false;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->isBitField()) return false;
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    auto fieldClassType = type_cast<ClassType>(fieldType);
    if (!fieldClassType) return false;
    auto fieldClass = fieldClassType->symbol();
    if (fieldClass) fieldClass = fieldClass->resolvedDefinition();
    if (!isEmptyClassForAbi(traits, fieldClass)) return false;
  }

  return true;
}

auto singleElementForAbi(TypeTraits& traits, ClassSymbol* classSymbol)
    -> const Type* {
  if (!classSymbol || !classSymbol->isComplete()) return nullptr;
  if (classSymbol->isUnion()) return nullptr;
  if (classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses())
    return nullptr;

  const Type* element = nullptr;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass) baseClass = baseClass->resolvedDefinition();
    if (isEmptyClassForAbi(traits, baseClass)) continue;
    if (element) return nullptr;
    element = singleElementForAbi(traits, baseClass);
    if (!element) return nullptr;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->isBitField()) return nullptr;

    auto fieldType = traits.remove_cv(field->type());

    auto leafType = traits.remove_all_extents(fieldType);
    if (auto leafClassType = type_cast<ClassType>(leafType)) {
      auto leafClass = leafClassType->symbol();
      if (leafClass) leafClass = leafClass->resolvedDefinition();
      if (isEmptyClassForAbi(traits, leafClass)) continue;
    }

    if (element) return nullptr;

    while (auto arrayType = type_cast<BoundedArrayType>(fieldType)) {
      if (arrayType->size() != 1) return nullptr;
      fieldType = traits.remove_cv(arrayType->elementType());
    }

    if (auto fieldClassType = type_cast<ClassType>(fieldType)) {
      auto fieldClass = fieldClassType->symbol();
      if (fieldClass) fieldClass = fieldClass->resolvedDefinition();
      element = singleElementForAbi(traits, fieldClass);
      if (!element) return nullptr;
    } else if (traits.is_scalar(fieldType)) {
      element = fieldType;
    } else {
      return nullptr;
    }
  }

  return element;
}

}  // namespace

auto classifyClassValueAbi(TranslationUnit* unit, const Type* type)
    -> ClassValueAbi {
  if (!type) return {};

  auto traits = unit->typeTraits();
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return {};

  auto memoryLayout = unit->control()->memoryLayout();
  if (!memoryLayout->usesSingleScalarClassAbi()) return {};

  auto classSymbol = classType->symbol();
  if (classSymbol) classSymbol = classSymbol->resolvedDefinition();

  if (!classSymbol || !classSymbol->isComplete()) {
    return {.kind = ClassValueAbi::Kind::Indirect};
  }

  if (isEmptyClassForAbi(traits, classSymbol)) {
    return {.kind = ClassValueAbi::Kind::Empty};
  }

  if (!traits.is_trivially_copyable(classType) ||
      classSymbol->isPolymorphic() || classSymbol->hasVirtualBaseClasses()) {
    return {.kind = ClassValueAbi::Kind::Indirect};
  }

  if (auto element = singleElementForAbi(traits, classSymbol)) {
    auto elementSize = memoryLayout->sizeOf(element);
    auto classSize = memoryLayout->sizeOf(classType);
    if (elementSize && classSize && *elementSize == *classSize) {
      return {.kind = ClassValueAbi::Kind::Scalar, .scalarType = element};
    }
  }

  return {.kind = ClassValueAbi::Kind::Indirect};
}

}  // namespace cxx
