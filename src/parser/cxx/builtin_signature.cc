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

#include <cxx/builtin_signature.h>
#include <cxx/control.h>
#include <cxx/types.h>

#include <iterator>
#include <vector>

#include "private/builtins_signatures-priv.h"

namespace cxx {

namespace {

class BuiltinSignatureDecoder {
 public:
  BuiltinSignatureDecoder(Control* control, std::size_t offset)
      : control_(control), pos_(offset) {}

  [[nodiscard]] auto decodeFunctionType(BuiltinFlags flags)
      -> const FunctionType* {
    auto returnType = decodeType();
    if (!returnType) return nullptr;

    std::vector<const Type*> parameterTypes;
    bool isVariadic = false;

    while (peek() != BuiltinTypeOp::kEnd) {
      if (peek() == BuiltinTypeOp::kVariadic) {
        isVariadic = true;
        ++pos_;
        break;
      }

      auto parameterType = decodeType();
      if (!parameterType) return nullptr;
      parameterTypes.push_back(parameterType);
    }

    if (peek() != BuiltinTypeOp::kEnd) return nullptr;
    ++pos_;

    return control_->getFunctionType(
        returnType, std::move(parameterTypes), isVariadic, CvQualifiers::kNone,
        RefQualifier::kNone, contains(flags, BuiltinFlags::kNoexcept));
  }

 private:
  [[nodiscard]] auto peek() const -> BuiltinTypeOp {
    if (pos_ >= std::size(kBuiltinSignatureOps)) return BuiltinTypeOp::kEnd;
    return static_cast<BuiltinTypeOp>(kBuiltinSignatureOps[pos_]);
  }

  [[nodiscard]] auto decodeType() -> const Type* {
    const auto op = peek();

    switch (op) {
      case BuiltinTypeOp::kPointer: {
        ++pos_;
        auto elementType = decodeType();
        if (!elementType) return nullptr;
        return control_->getPointerType(elementType);
      }

      case BuiltinTypeOp::kLvalueReference: {
        ++pos_;
        auto elementType = decodeType();
        if (!elementType) return nullptr;
        return control_->getLvalueReferenceType(elementType);
      }

      case BuiltinTypeOp::kRvalueReference: {
        ++pos_;
        auto elementType = decodeType();
        if (!elementType) return nullptr;
        return control_->getRvalueReferenceType(elementType);
      }

      case BuiltinTypeOp::kConst:
      case BuiltinTypeOp::kVolatile: {
        auto cvQualifiers = CvQualifiers::kNone;
        while (peek() == BuiltinTypeOp::kConst ||
               peek() == BuiltinTypeOp::kVolatile) {
          cvQualifiers = cvQualifiers | (peek() == BuiltinTypeOp::kConst
                                             ? CvQualifiers::kConst
                                             : CvQualifiers::kVolatile);
          ++pos_;
        }
        auto elementType = decodeType();
        if (!elementType) return nullptr;
        return control_->getQualType(elementType, cvQualifiers);
      }

      case BuiltinTypeOp::kComplex: {
        ++pos_;
        auto elementType = decodeType();
        if (!elementType) return nullptr;
        return control_->getComplexType(elementType);
      }

      case BuiltinTypeOp::kEnd:
      case BuiltinTypeOp::kVariadic:
        return nullptr;

      default:
        ++pos_;
        return decodeBuiltinLeafType(control_, op);
    }
  }

  Control* control_;
  std::size_t pos_;
};

}  // namespace

auto builtinSignatureOf(BuiltinFunctionKind kind) -> BuiltinSignature {
  const auto index = static_cast<std::size_t>(kind);
  if (index >= std::size(kBuiltinSignatures)) return {};
  return kBuiltinSignatures[index];
}

auto decodeBuiltinSignature(Control* control, BuiltinFunctionKind kind,
                            std::size_t index) -> const FunctionType* {
  if (kind == BuiltinFunctionKind::T_NONE) return nullptr;

  auto signature = builtinSignatureOf(kind);
  if (index >= signature.count) return nullptr;

  BuiltinSignatureDecoder decoder{control, signature.offset};

  const FunctionType* functionType = nullptr;
  for (std::size_t i = 0; i <= index; ++i) {
    functionType = decoder.decodeFunctionType(signature.flags);
    if (!functionType) return nullptr;
  }

  return functionType;
}

}  // namespace cxx
