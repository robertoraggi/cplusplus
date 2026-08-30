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

#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {
namespace {

struct ContainsPlaceholderType {
  auto operator()(const AutoType*) const -> bool { return true; }
  auto operator()(const DecltypeAutoType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const BoundedArrayType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const PointerType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const LvalueReferenceType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const RvalueReferenceType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const FunctionType* type) const -> bool {
    if (containsPlaceholderType(type->returnType())) return true;
    for (auto parameterType : type->parameterTypes()) {
      if (containsPlaceholderType(parameterType)) return true;
    }
    return false;
  }

  auto operator()(const MemberObjectPointerType* type) const -> bool {
    return containsPlaceholderType(type->elementType());
  }

  auto operator()(const MemberFunctionPointerType* type) const -> bool {
    return containsPlaceholderType(type->functionType());
  }

  template <typename T>
  auto operator()(const T*) const -> bool {
    return false;
  }
};

}  // namespace

auto containsPlaceholderType(const Type* type) -> bool {
  if (!type) return false;
  return visit(ContainsPlaceholderType{}, type);
}

auto unqualified_type(const Type* type) -> const Type* {
  while (auto qualType = type_cast<QualType>(type)) {
    type = qualType->elementType();
  }
  return type;
}

auto cv_qualifiers(const Type* type) -> CvQualifiers {
  auto cv = CvQualifiers::kNone;
  while (type) {
    if (auto qualType = type_cast<QualType>(type)) {
      cv |= qualType->cvQualifiers();
      type = qualType->elementType();
    } else if (auto arrayType = type_cast<BoundedArrayType>(type)) {
      type = arrayType->elementType();
    } else if (auto arrayType = type_cast<UnboundedArrayType>(type)) {
      type = arrayType->elementType();
    } else {
      break;
    }
  }
  return cv;
}

auto residual_cv_qualifiers(CvQualifiers argumentCv, CvQualifiers parameterCv)
    -> CvQualifiers {
  return argumentCv & ~parameterCv;
}

auto EnumType::underlyingType() const -> const Type* {
  return symbol()->underlyingType();
}

auto ScopedEnumType::underlyingType() const -> const Type* {
  return symbol()->underlyingType();
}

auto ClassType::definition() const -> ClassSymbol* {
  return symbol()->resolvedDefinition();
}

auto ClassType::isComplete() const -> bool {
  return definition()->isComplete();
}

auto ClassType::isUnion() const -> bool { return definition()->isUnion(); }

auto memberPointerBaseAdjustment(const MemberObjectPointerType* sourceType,
                                 const MemberObjectPointerType* targetType)
    -> std::optional<std::int64_t> {
  if (!sourceType || !targetType) return std::nullopt;

  auto sourceClassType = type_cast<ClassType>(sourceType->classType());
  auto targetClassType = type_cast<ClassType>(targetType->classType());
  if (!sourceClassType || !targetClassType) return std::nullopt;

  auto baseClass = sourceClassType->symbol();
  auto derivedClass = targetClassType->symbol();
  if (!baseClass || !derivedClass) return std::nullopt;

  auto offset = derivedClass->resolvedDefinition()->baseClassOffset(baseClass);
  if (!offset.has_value()) return std::nullopt;

  return static_cast<std::int64_t>(*offset);
}
}  // namespace cxx
