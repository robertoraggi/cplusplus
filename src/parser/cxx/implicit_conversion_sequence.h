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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/types_fwd.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <vector>

namespace cxx {
class FunctionSymbol;
class TypeTraits;

enum class ConversionRank {
  kNone,
  kConversion,
  kPromotion,
  kExactMatch,
};

[[nodiscard]] constexpr auto conversionRank(ImplicitCastKind kind)
    -> ConversionRank {
  switch (kind) {
    case ImplicitCastKind::kIdentity:
    case ImplicitCastKind::kLValueToRValueConversion:
    case ImplicitCastKind::kArrayToPointerConversion:
    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
    case ImplicitCastKind::kFunctionPointerConversion:
    case ImplicitCastKind::kTemporaryMaterializationConversion:
      return ConversionRank::kExactMatch;

    case ImplicitCastKind::kIntegralPromotion:
    case ImplicitCastKind::kFloatingPointPromotion:
      return ConversionRank::kPromotion;

    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kFloatingPointConversion:
    case ImplicitCastKind::kFloatingIntegralConversion:
    case ImplicitCastKind::kPointerConversion:
    case ImplicitCastKind::kPointerToMemberConversion:
    case ImplicitCastKind::kDerivedToBaseConversion:
    case ImplicitCastKind::kBaseToDerivedConversion:
    case ImplicitCastKind::kBooleanConversion:
    case ImplicitCastKind::kUserDefinedConversion:
      return ConversionRank::kConversion;
    default:
      return ConversionRank::kNone;
  }
}

enum class ConversionSequenceForm {
  kNone,
  kStandard,
  kUserDefined,
  kAmbiguous,
  kEllipsis,
};

[[nodiscard]] constexpr auto formOrder(ConversionSequenceForm form) -> int {
  switch (form) {
    case ConversionSequenceForm::kStandard:
      return 0;
    case ConversionSequenceForm::kUserDefined:
    case ConversionSequenceForm::kAmbiguous:
      return 1;
    case ConversionSequenceForm::kEllipsis:
      return 2;
    case ConversionSequenceForm::kNone:
      return 3;
    default:
      return 0;
  }
}

struct ImplicitConversionSequence {
  struct Step {
    ImplicitCastKind kind;
    const Type* type = nullptr;
  };

  struct ReferenceBinding {
    enum class Kind {
      kNone,
      kDirectToLvalue,
      kDirectToXvalue,
      kToTemporary,
    };

    Kind kind = Kind::kNone;
    const Type* referencedType = nullptr;
    CvQualifiers cv = CvQualifiers::kNone;
    bool isRvalueRef = false;
    bool isDirect = false;
    bool referencesFunctionType = false;
    bool isUnqualifiedImplicitObjectParameter = false;

    [[nodiscard]] auto binds() const -> bool { return kind != Kind::kNone; }

    [[nodiscard]] auto bindsToGlvalue() const -> bool {
      return kind == Kind::kDirectToLvalue || kind == Kind::kDirectToXvalue;
    }

    [[nodiscard]] auto bindsToLvalue() const -> bool {
      return kind == Kind::kDirectToLvalue;
    }

    [[nodiscard]] auto bindsToTemporary() const -> bool {
      return kind == Kind::kToTemporary;
    }

    [[nodiscard]] auto bindsRvalueReferenceToRvalue() const -> bool {
      if (!isRvalueRef) return false;
      return !bindsToLvalue();
    }

    [[nodiscard]] auto bindsLvalueReference() const -> bool {
      if (!binds()) return false;
      return !isRvalueRef;
    }

    [[nodiscard]] auto bindsRvalueReferenceToFunctionLvalue() const -> bool {
      if (!isRvalueRef) return false;
      if (!bindsToLvalue()) return false;
      return referencesFunctionType;
    }

    [[nodiscard]] auto bindsLvalueReferenceToFunctionLvalue() const -> bool {
      if (!bindsLvalueReference()) return false;
      if (!bindsToLvalue()) return false;
      return referencesFunctionType;
    }
  };

  struct UserDefinedConversion {
    FunctionSymbol* function = nullptr;
    const Type* aggregateInitializedClass = nullptr;
    const Type* secondTarget = nullptr;
    std::vector<Step> secondSteps;
    ConversionRank secondRank = ConversionRank::kNone;
  };

  struct ListInitialization {
    bool isListInitialization = false;
    bool fromSingleElement = false;
    bool narrowsElement = false;
    bool targetIsUnboundedArray = false;
    const Type* initializerListElementType = nullptr;
    std::size_t elementCount = 0;
    ConversionRank elementRank = ConversionRank::kExactMatch;
  };

  ConversionSequenceForm form = ConversionSequenceForm::kNone;

  const Type* sourceType = nullptr;
  const Type* destinationType = nullptr;
  FunctionSymbol* copyConstructor = nullptr;
  bool requiresCopyConstruction = false;
  bool isStaticMemberObjectParameter = false;

  std::vector<Step> steps;
  ReferenceBinding binding;
  UserDefinedConversion udc;
  ListInitialization list;

  const Type* pointeeUnqual = nullptr;
  CvQualifiers pointeeCv = CvQualifiers::kNone;

  [[nodiscard]] auto rank() const -> ConversionRank {
    switch (form) {
      case ConversionSequenceForm::kNone:
        return ConversionRank::kNone;

      case ConversionSequenceForm::kUserDefined:
      case ConversionSequenceForm::kAmbiguous:
      case ConversionSequenceForm::kEllipsis:
        return ConversionRank::kConversion;

      case ConversionSequenceForm::kStandard:
        break;
    }

    auto worst = list.isListInitialization ? list.elementRank
                                           : ConversionRank::kExactMatch;
    for (const auto& step : steps)
      worst = std::min(worst, conversionRank(step.kind));
    return worst;
  }

  [[nodiscard]] auto hasPointerQualificationConversion() const -> bool {
    if (!pointeeUnqual) return false;
    return std::ranges::any_of(steps, [](const Step& step) {
      return step.kind == ImplicitCastKind::kQualificationConversion;
    });
  }

  [[nodiscard]] auto comparesSecondStandardSequenceWith(
      const ImplicitConversionSequence& other) const -> bool {
    if (udc.function || other.udc.function)
      return udc.function == other.udc.function;
    return udc.aggregateInitializedClass &&
           udc.aggregateInitializedClass == other.udc.aggregateInitializedClass;
  }

  [[nodiscard]] auto isBetterThan(const ImplicitConversionSequence& other,
                                  const TypeTraits& traits) const -> bool;

  explicit operator bool() const {
    return form != ConversionSequenceForm::kNone;
  }
};
}  // namespace cxx
