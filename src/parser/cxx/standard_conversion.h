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
#include <cxx/implicit_conversion_sequence.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

namespace cxx {

class Arena;
class Control;
class TranslationUnit;

class StandardConversion {
 public:
  explicit StandardConversion(TranslationUnit* unit, bool isC = false);

  [[nodiscard]] auto initializerListElementType(const Type* targetType) const
      -> const Type*;

  [[nodiscard]] auto computeConversionSequence(ExpressionAST* expr,
                                               const Type* targetType)
      -> ImplicitConversionSequence;

  void applyConversionSequence(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  void adjustCv(ExpressionAST* expr);

  [[nodiscard]] auto checkCvQualifiers(CvQualifiers target,
                                       CvQualifiers source) const -> bool;

  [[nodiscard]] auto ensurePrvalue(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto temporaryMaterialization(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto usualArithmeticConversion(ExpressionAST*& expr,
                                               ExpressionAST*& other)
      -> const Type*;

  [[nodiscard]] auto convertImplicitly(ExpressionAST*& expr,
                                       const Type* destinationType) -> bool;

  [[nodiscard]] auto integralPromotion(ExpressionAST*& expr,
                                       const Type* destinationType = nullptr)
      -> bool;

  [[nodiscard]] auto floatingPointPromotion(
      ExpressionAST*& expr, const Type* destinationType = nullptr) -> bool;

  [[nodiscard]] auto isNullPointerConstant(ExpressionAST* expr) const -> bool;

  [[nodiscard]] auto lvalueToRvalue(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto functionToPointer(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto arrayToPointer(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto compositePointerType(ExpressionAST*& expr,
                                          ExpressionAST*& other) -> const Type*;

  [[nodiscard]] auto isIntegralPromotion(const Type* sourceType,
                                         const Type* targetType) const -> bool;

  [[nodiscard]] auto isFloatingPointPromotion(const Type* sourceType,
                                              const Type* targetType) const
      -> bool;

 private:
  [[nodiscard]] auto integralConversion(ExpressionAST*& expr,
                                        const Type* destinationType) -> bool;
  [[nodiscard]] auto floatingPointConversion(ExpressionAST*& expr,
                                             const Type* destinationType)
      -> bool;
  [[nodiscard]] auto floatingIntegralConversion(ExpressionAST*& expr,
                                                const Type* destinationType)
      -> bool;

  [[nodiscard]] auto isNarrowingConversion(const Type* from, const Type* to)
      -> bool;

  [[nodiscard]] auto isReferenceCompatible(const Type* targetType,
                                           const Type* sourceType) const
      -> bool;

  [[nodiscard]] auto getQualificationCombinedType(const Type* left,
                                                  const Type* right)
      -> const Type*;

  [[nodiscard]] auto getQualificationCombinedType(
      const Type* left, const Type* right, bool& didChangeTypeOrQualifiers)
      -> const Type*;

  [[nodiscard]] auto stripCv(const Type*& type) -> CvQualifiers;

  [[nodiscard]] auto mergeCv(CvQualifiers cv1, CvQualifiers cv2) const
      -> CvQualifiers;

 private:
  TranslationUnit* unit_;
  Control* control_;
  Arena* arena_;
  bool isC_ = false;
};

}  // namespace cxx
