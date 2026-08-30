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

#include <cxx/control.h>
#include <cxx/implicit_conversion_sequence.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {
namespace {

using Step = ImplicitConversionSequence::Step;
using ReferenceBinding = ImplicitConversionSequence::ReferenceBinding;

[[nodiscard]] auto isLvalueTransformation(ImplicitCastKind kind) -> bool {
  switch (kind) {
    case ImplicitCastKind::kLValueToRValueConversion:
    case ImplicitCastKind::kArrayToPointerConversion:
    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kTemporaryMaterializationConversion:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto decayedSourceType(const TypeTraits& traits,
                                     const ImplicitConversionSequence& seq)
    -> const Type* {
  auto source = traits.remove_reference(seq.sourceType);
  if (traits.is_array(source) || traits.is_function(source))
    source = traits.decay(source);
  return traits.remove_cv(source);
}

[[nodiscard]] auto conversionTargetType(const TypeTraits& traits,
                                        const ImplicitConversionSequence& seq)
    -> const Type* {
  if (seq.binding.binds() && seq.binding.referencedType)
    return traits.remove_cv(seq.binding.referencedType);
  if (!seq.steps.empty() && seq.steps.back().type)
    return traits.remove_cv(seq.steps.back().type);
  return traits.remove_cv(traits.remove_reference(seq.destinationType));
}

[[nodiscard]] auto referenceBindingConversionKind(
    const TypeTraits& traits, const ImplicitConversionSequence& seq)
    -> ImplicitCastKind {
  auto target = traits.remove_cv(seq.binding.referencedType);
  auto source = traits.remove_cv(traits.remove_reference(seq.sourceType));

  if (traits.is_same(source, target)) return ImplicitCastKind::kIdentity;

  if (traits.is_base_of(target, source))
    return ImplicitCastKind::kDerivedToBaseConversion;

  if (traits.is_unbounded_array(target) && traits.is_bounded_array(source) &&
      traits.is_same(traits.remove_cv(traits.get_element_type(target)),
                     traits.remove_cv(traits.get_element_type(source))))
    return ImplicitCastKind::kIdentity;

  if (traits.is_function(target))
    return ImplicitCastKind::kFunctionPointerConversion;

  return ImplicitCastKind::kQualificationConversion;
}

[[nodiscard]] auto canonicalConversions(const TypeTraits& traits,
                                        const ImplicitConversionSequence& seq)
    -> std::vector<ImplicitCastKind> {
  if (seq.binding.isDirect && seq.binding.referencedType) {
    auto kind = referenceBindingConversionKind(traits, seq);
    if (kind == ImplicitCastKind::kIdentity) return {};
    return {kind};
  }

  std::vector<ImplicitCastKind> kinds;
  for (const auto& step : seq.steps) {
    if (isLvalueTransformation(step.kind)) continue;
    if (step.kind == ImplicitCastKind::kIdentity) continue;
    kinds.push_back(step.kind);
  }
  return kinds;
}

[[nodiscard]] auto isProperSubsequence(
    const std::vector<ImplicitCastKind>& shorter,
    const std::vector<ImplicitCastKind>& longer) -> bool {
  if (shorter.size() >= longer.size()) return false;

  std::size_t matched = 0;
  for (auto kind : longer) {
    if (matched < shorter.size() && shorter[matched] == kind) ++matched;
  }
  return matched == shorter.size();
}

[[nodiscard]] auto withoutQualificationConversions(
    std::vector<ImplicitCastKind> kinds) -> std::vector<ImplicitCastKind> {
  std::erase_if(kinds, [](ImplicitCastKind kind) {
    return kind == ImplicitCastKind::kQualificationConversion ||
           kind == ImplicitCastKind::kFunctionPointerConversion;
  });
  return kinds;
}

[[nodiscard]] auto yieldedType(const TypeTraits& traits,
                               const ImplicitConversionSequence& seq)
    -> const Type* {
  return conversionTargetType(traits, seq);
}

[[nodiscard]] auto convertsToBoolFromPointer(
    const TypeTraits& traits, const ImplicitConversionSequence& seq) -> bool {
  auto target = conversionTargetType(traits, seq);
  if (!target || target->kind() != TypeKind::kBool) return false;

  auto source = decayedSourceType(traits, seq);
  return traits.is_pointer(source) || traits.is_member_pointer(source) ||
         traits.is_null_pointer(source);
}

[[nodiscard]] auto pointeeClass(const TypeTraits& traits, const Type* type)
    -> const Type* {
  auto pointerType = type_cast<PointerType>(type);
  if (!pointerType) return nullptr;
  auto element = traits.remove_cv(pointerType->elementType());
  if (!traits.is_class(element)) return nullptr;
  return element;
}

[[nodiscard]] auto isPointerToVoid(const TypeTraits& traits, const Type* type)
    -> bool {
  auto pointerType = type_cast<PointerType>(type);
  if (!pointerType) return false;
  return traits.is_void(pointerType->elementType());
}

[[nodiscard]] auto memberPointerClass(const TypeTraits& traits,
                                      const Type* type) -> const Type* {
  if (auto objectPointer = type_cast<MemberObjectPointerType>(type))
    return traits.remove_cv(objectPointer->classType());
  if (auto functionPointer = type_cast<MemberFunctionPointerType>(type))
    return traits.remove_cv(functionPointer->classType());
  return nullptr;
}

[[nodiscard]] auto classOperand(const TypeTraits& traits, const Type* type)
    -> const Type* {
  if (!traits.is_class(type)) return nullptr;
  return type;
}

[[nodiscard]] auto nearerBase(const TypeTraits& traits, const Type* left,
                              const Type* right) -> std::optional<bool> {
  if (!left || !right) return std::nullopt;
  if (traits.is_same(left, right)) return std::nullopt;
  if (traits.is_base_of(right, left)) return true;
  if (traits.is_base_of(left, right)) return false;
  return std::nullopt;
}

[[nodiscard]] auto sameSourceHierarchyBetter(const TypeTraits& traits,
                                             const Type* source,
                                             const Type* target1,
                                             const Type* target2)
    -> std::optional<bool> {
  if (auto sourceClass = pointeeClass(traits, source)) {
    auto class1 = pointeeClass(traits, target1);
    auto class2 = pointeeClass(traits, target2);

    if (class1 && isPointerToVoid(traits, target2) &&
        traits.is_base_of(class1, sourceClass))
      return true;
    if (class2 && isPointerToVoid(traits, target1) &&
        traits.is_base_of(class2, sourceClass))
      return false;

    if (class1 && class2 && traits.is_base_of(class1, sourceClass) &&
        traits.is_base_of(class2, sourceClass))
      return nearerBase(traits, class1, class2);
  }

  if (auto sourceClass = memberPointerClass(traits, source)) {
    auto class1 = memberPointerClass(traits, target1);
    auto class2 = memberPointerClass(traits, target2);

    if (class1 && class2 && traits.is_base_of(sourceClass, class1) &&
        traits.is_base_of(sourceClass, class2))
      return nearerBase(traits, class2, class1);
  }

  if (auto sourceClass = classOperand(traits, source)) {
    auto class1 = classOperand(traits, target1);
    auto class2 = classOperand(traits, target2);

    if (class1 && class2 && traits.is_base_of(class1, sourceClass) &&
        traits.is_base_of(class2, sourceClass))
      return nearerBase(traits, class1, class2);
  }

  return std::nullopt;
}

[[nodiscard]] auto sameTargetHierarchyBetter(const TypeTraits& traits,
                                             const Type* target,
                                             const Type* source1,
                                             const Type* source2)
    -> std::optional<bool> {
  auto class1 = pointeeClass(traits, source1);
  auto class2 = pointeeClass(traits, source2);

  if (class1 && class2) {
    if (isPointerToVoid(traits, target))
      return nearerBase(traits, class2, class1);

    if (auto targetClass = pointeeClass(traits, target)) {
      if (traits.is_base_of(targetClass, class1) &&
          traits.is_base_of(targetClass, class2))
        return nearerBase(traits, class2, class1);
    }
  }

  auto memberClass1 = memberPointerClass(traits, source1);
  auto memberClass2 = memberPointerClass(traits, source2);

  if (memberClass1 && memberClass2) {
    if (auto targetClass = memberPointerClass(traits, target)) {
      if (traits.is_base_of(memberClass1, targetClass) &&
          traits.is_base_of(memberClass2, targetClass))
        return nearerBase(traits, memberClass1, memberClass2);
    }
  }

  auto valueClass1 = classOperand(traits, source1);
  auto valueClass2 = classOperand(traits, source2);

  if (valueClass1 && valueClass2) {
    if (auto targetClass = classOperand(traits, target)) {
      if (traits.is_base_of(targetClass, valueClass1) &&
          traits.is_base_of(targetClass, valueClass2))
        return nearerBase(traits, valueClass2, valueClass1);
    }
  }

  return std::nullopt;
}

[[nodiscard]] auto classHierarchyBetter(
    const TypeTraits& traits, const Type* source1, const Type* target1,
    const Type* source2, const Type* target2) -> std::optional<bool> {
  if (!source1 || !source2 || !target1 || !target2) return std::nullopt;

  if (traits.is_same(source1, source2))
    return sameSourceHierarchyBetter(traits, source1, target1, target2);

  if (traits.is_same(target1, target2))
    return sameTargetHierarchyBetter(traits, target1, source1, source2);

  return std::nullopt;
}

[[nodiscard]] auto promotesFixedEnumerationToUnderlyingType(
    const TypeTraits& traits, const ImplicitConversionSequence& seq)
    -> std::optional<bool> {
  auto enumType = type_cast<EnumType>(decayedSourceType(traits, seq));
  if (!enumType) return std::nullopt;

  auto [underlyingType, promotedType] =
      traits.promoted_enumeration_types(enumType);
  if (!underlyingType || !promotedType) return std::nullopt;

  auto target = conversionTargetType(traits, seq);
  if (traits.is_same(target, traits.remove_cv(underlyingType))) return true;
  if (traits.is_same(target, traits.remove_cv(promotedType))) return false;
  return std::nullopt;
}

[[nodiscard]] auto initializedElementCount(
    const TypeTraits& traits, const ImplicitConversionSequence& seq)
    -> std::size_t {
  auto arrayType =
      traits.remove_cv(traits.remove_reference(seq.destinationType));
  if (auto bounded = type_cast<BoundedArrayType>(arrayType))
    return bounded->size();
  return seq.list.elementCount;
}

[[nodiscard]] auto listArrayElementType(const TypeTraits& traits,
                                        const ImplicitConversionSequence& seq)
    -> const Type* {
  auto arrayType =
      traits.remove_cv(traits.remove_reference(seq.destinationType));
  if (!traits.is_array(arrayType)) return nullptr;
  return traits.remove_cv(traits.get_element_type(arrayType));
}

[[nodiscard]] auto listInitializationBetter(
    const TypeTraits& traits, const ImplicitConversionSequence& lhs,
    const ImplicitConversionSequence& rhs) -> std::optional<bool> {
  if (!lhs.list.isListInitialization || !rhs.list.isListInitialization)
    return std::nullopt;

  const bool lhsToInitializerList =
      lhs.list.initializerListElementType != nullptr;
  const bool rhsToInitializerList =
      rhs.list.initializerListElementType != nullptr;
  if (lhsToInitializerList != rhsToInitializerList) return lhsToInitializerList;

  auto lhsElementType = listArrayElementType(traits, lhs);
  auto rhsElementType = listArrayElementType(traits, rhs);
  if (!lhsElementType || !rhsElementType) return std::nullopt;
  if (!traits.is_same(lhsElementType, rhsElementType)) return std::nullopt;

  auto lhsCount = initializedElementCount(traits, lhs);
  auto rhsCount = initializedElementCount(traits, rhs);
  if (lhsCount != rhsCount) return lhsCount < rhsCount;

  if (lhs.list.targetIsUnboundedArray != rhs.list.targetIsUnboundedArray)
    return !lhs.list.targetIsUnboundedArray;

  return std::nullopt;
}

[[nodiscard]] auto sameRankBetter(const TypeTraits& traits,
                                  const ImplicitConversionSequence& lhs,
                                  const ImplicitConversionSequence& rhs)
    -> std::optional<bool> {
  const bool lhsToBool = convertsToBoolFromPointer(traits, lhs);
  const bool rhsToBool = convertsToBoolFromPointer(traits, rhs);
  if (lhsToBool != rhsToBool) return !lhsToBool;

  auto lhsSource = decayedSourceType(traits, lhs);
  auto rhsSource = decayedSourceType(traits, rhs);

  if (traits.is_same(lhsSource, rhsSource)) {
    if (auto better = promotesFixedEnumerationToUnderlyingType(traits, lhs)) {
      auto other = promotesFixedEnumerationToUnderlyingType(traits, rhs);
      if (other && *better != *other) return *better;
    }
  }

  return classHierarchyBetter(traits, lhsSource,
                              conversionTargetType(traits, lhs), rhsSource,
                              conversionTargetType(traits, rhs));
}

[[nodiscard]] auto referenceBindingBetter(const TypeTraits& traits,
                                          const ImplicitConversionSequence& lhs,
                                          const ImplicitConversionSequence& rhs)
    -> std::optional<bool> {
  const bool comparesReferenceBindings =
      !lhs.binding.isUnqualifiedImplicitObjectParameter &&
      !rhs.binding.isUnqualifiedImplicitObjectParameter;

  if (comparesReferenceBindings) {
    if (lhs.binding.bindsRvalueReferenceToRvalue() &&
        rhs.binding.bindsLvalueReference())
      return true;
    if (lhs.binding.bindsLvalueReference() &&
        rhs.binding.bindsRvalueReferenceToRvalue())
      return false;

    if (lhs.binding.bindsLvalueReferenceToFunctionLvalue() &&
        rhs.binding.bindsRvalueReferenceToFunctionLvalue())
      return true;
    if (lhs.binding.bindsRvalueReferenceToFunctionLvalue() &&
        rhs.binding.bindsLvalueReferenceToFunctionLvalue())
      return false;
  }

  return std::nullopt;
}

[[nodiscard]] auto qualificationConversionBetter(
    const TypeTraits& traits, const ImplicitConversionSequence& lhs,
    const ImplicitConversionSequence& rhs) -> std::optional<bool> {
  auto lhsKinds = canonicalConversions(traits, lhs);
  auto rhsKinds = canonicalConversions(traits, rhs);
  if (withoutQualificationConversions(lhsKinds) !=
      withoutQualificationConversions(rhsKinds))
    return std::nullopt;

  auto lhsType = yieldedType(traits, lhs);
  auto rhsType = yieldedType(traits, rhs);
  if (!lhsType || !rhsType) return std::nullopt;
  if (traits.is_same(lhsType, rhsType)) return std::nullopt;
  if (!traits.is_similar(lhsType, rhsType)) return std::nullopt;

  if (traits.is_reference_compatible(traits.add_const(rhsType), lhsType))
    return true;
  if (traits.is_reference_compatible(traits.add_const(lhsType), rhsType))
    return false;

  return std::nullopt;
}

[[nodiscard]] auto boundReferenceTypeBetter(
    const TypeTraits& traits, const ImplicitConversionSequence& lhs,
    const ImplicitConversionSequence& rhs) -> std::optional<bool> {
  if (!lhs.binding.binds() || !rhs.binding.binds()) return std::nullopt;

  auto lhsType = lhs.binding.referencedType;
  auto rhsType = rhs.binding.referencedType;
  if (!lhsType || !rhsType) return std::nullopt;

  if (!traits.is_same(lhsType, rhsType)) {
    if (traits.is_reference_compatible(rhsType, lhsType)) return true;
    if (traits.is_reference_compatible(lhsType, rhsType)) return false;
    return std::nullopt;
  }

  return classHierarchyBetter(
      traits, traits.add_pointer(decayedSourceType(traits, lhs)),
      traits.add_pointer(traits.remove_cv(lhsType)),
      traits.add_pointer(decayedSourceType(traits, rhs)),
      traits.add_pointer(traits.remove_cv(rhsType)));
}

[[nodiscard]] auto standardSequenceBetter(const TypeTraits& traits,
                                          const ImplicitConversionSequence& lhs,
                                          const ImplicitConversionSequence& rhs)
    -> bool {
  auto lhsKinds = canonicalConversions(traits, lhs);
  auto rhsKinds = canonicalConversions(traits, rhs);

  if (isProperSubsequence(lhsKinds, rhsKinds)) return true;
  if (isProperSubsequence(rhsKinds, lhsKinds)) return false;

  if (lhs.rank() != rhs.rank()) return lhs.rank() > rhs.rank();

  if (auto better = sameRankBetter(traits, lhs, rhs)) return *better;

  if (auto better = referenceBindingBetter(traits, lhs, rhs)) return *better;

  if (auto better = qualificationConversionBetter(traits, lhs, rhs))
    return *better;

  if (auto better = boundReferenceTypeBetter(traits, lhs, rhs)) return *better;

  return false;
}

}  // namespace

auto ImplicitConversionSequence::isBetterThan(
    const ImplicitConversionSequence& other, const TypeTraits& traits) const
    -> bool {
  if (isStaticMemberObjectParameter || other.isStaticMemberObjectParameter)
    return false;

  if (formOrder(form) != formOrder(other.form))
    return formOrder(form) < formOrder(other.form);

  if (auto better = listInitializationBetter(traits, *this, other))
    return *better;

  if (form == ConversionSequenceForm::kStandard)
    return standardSequenceBetter(traits, *this, other);

  if (form == ConversionSequenceForm::kUserDefined) {
    if (!comparesSecondStandardSequenceWith(other)) return false;

    if (udc.secondRank != other.udc.secondRank)
      return udc.secondRank > other.udc.secondRank;

    auto better = referenceBindingBetter(traits, *this, other);
    return better.value_or(false);
  }

  return false;
}
}  // namespace cxx
