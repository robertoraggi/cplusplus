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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/builtin_arithmetic_overflow.h>
#include <cxx/builtin_bit_operations.h>
#include <cxx/const_int.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

// cstd
#include <bit>
#include <cctype>
#include <cmath>
#include <limits>

namespace cxx {

namespace {

[[nodiscard]] auto argumentAt(CallExpressionAST* ast, int index)
    -> ExpressionAST* {
  int current = 0;
  for (auto argument : ListView{ast->expressionList}) {
    if (current++ == index) return argument;
  }
  return nullptr;
}

[[nodiscard]] auto strippedPointerType(ExpressionAST* ast)
    -> const PointerType* {
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
    auto pointer = type_cast<PointerType>(ast->type);
    if (pointer && !unqualified_cast<VoidType>(pointer->elementType()))
      return pointer;
    ast = cast->expression;
  }
  if (!ast) return nullptr;
  return type_cast<PointerType>(ast->type);
}

class ObjectRepresentation {
 public:
  ObjectRepresentation(ASTInterpreter& interp, std::size_t size)
      : interp_(interp),
        traits_(interp.translationUnit()->typeTraits()),
        bytes_(size),
        defined_(size) {}

  [[nodiscard]] auto store(const Type* type, const ConstValue& value,
                           std::uint64_t offset) -> bool;

  [[nodiscard]] auto load(const Type* type, std::uint64_t offset)
      -> std::optional<ConstValue>;

 private:
  struct BitRange {
    std::uint64_t offset = 0;
    std::uint32_t position = 0;
    std::uint32_t width = 0;
  };

  struct Placement {
    const Type* type = nullptr;
    std::uint64_t offset = 0;
    std::optional<BitRange> bitField;
  };

  [[nodiscard]] auto placement(const ClassLayout& layout, Symbol* element,
                               std::uint64_t offset) const
      -> std::optional<Placement>;

  [[nodiscard]] auto valueRange(const Type* type, std::uint64_t offset)
      -> std::optional<BitRange>;

  [[nodiscard]] auto valueBits(const Type* type, const ConstValue& value)
      -> std::optional<ConstInt::UWide>;

  [[nodiscard]] auto scalarValue(const Type* type, ConstInt::UWide bits)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto storeBits(BitRange range, ConstInt::UWide bits) -> bool;

  [[nodiscard]] auto loadBits(BitRange range) const
      -> std::optional<ConstInt::UWide>;

  [[nodiscard]] auto storeScalar(const Type* type, const ConstValue& value,
                                 std::uint64_t offset) -> bool;

  [[nodiscard]] auto loadScalar(const Type* type, std::uint64_t offset)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto storeArray(const BoundedArrayType* type,
                                const ConstValue& value, std::uint64_t offset)
      -> bool;

  [[nodiscard]] auto loadArray(const BoundedArrayType* type,
                               std::uint64_t offset)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto storeObject(const ClassType* type, const ConstValue& value,
                                 std::uint64_t offset) -> bool;

  [[nodiscard]] auto loadObject(const ClassType* type, std::uint64_t offset)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto storeElement(const Placement& placement,
                                  const ConstValue& value) -> bool;

  [[nodiscard]] auto loadElement(const Placement& placement)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto receivesIndeterminateValue(const Type* type) -> bool;

  [[nodiscard]] auto memoryLayout() const -> MemoryLayout* {
    return interp_.control()->memoryLayout();
  }

  ASTInterpreter& interp_;
  TypeTraits traits_;
  std::vector<std::uint8_t> bytes_;
  std::vector<std::uint8_t> defined_;
};

auto ObjectRepresentation::store(const Type* type, const ConstValue& value,
                                 std::uint64_t offset) -> bool {
  if (std::holds_alternative<IndeterminateValue>(value)) return true;
  auto unqualified = traits_.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqualified))
    return storeArray(arrayType, value, offset);
  if (auto classType = type_cast<ClassType>(unqualified))
    return storeObject(classType, value, offset);
  return storeScalar(unqualified, value, offset);
}

auto ObjectRepresentation::load(const Type* type, std::uint64_t offset)
    -> std::optional<ConstValue> {
  auto unqualified = traits_.remove_cv(type);
  if (auto arrayType = type_cast<BoundedArrayType>(unqualified))
    return loadArray(arrayType, offset);
  if (auto classType = type_cast<ClassType>(unqualified))
    return loadObject(classType, offset);
  return loadScalar(unqualified, offset);
}

auto ObjectRepresentation::placement(const ClassLayout& layout, Symbol* element,
                                     std::uint64_t offset) const
    -> std::optional<Placement> {
  if (auto base = symbol_cast<BaseClassSymbol>(element)) {
    auto baseClass = resolved_base_class(base);
    if (!baseClass) return std::nullopt;
    auto info = layout.getBaseInfo(baseClass);
    if (!info) return std::nullopt;
    return Placement{baseClass->type(), offset + info->offset};
  }

  auto field = symbol_cast<FieldSymbol>(element);
  if (!field) return std::nullopt;
  auto info = layout.getFieldInfo(field);
  if (!info) return std::nullopt;
  Placement result{field->type(), offset + info->offset};
  if (field->isBitField())
    result.bitField = BitRange{result.offset, info->bitOffset, info->bitWidth};
  return result;
}

auto ObjectRepresentation::valueRange(const Type* type, std::uint64_t offset)
    -> std::optional<BitRange> {
  if (auto representation = traits_.integral_representation(type)) {
    return BitRange{offset, 0,
                    static_cast<std::uint32_t>(representation->bits)};
  }
  if (auto format = memoryLayout()->floatingPointFormat(type)) {
    auto width = 1 + format->exponentBits + format->fractionBits();
    return BitRange{offset, 0, static_cast<std::uint32_t>(width)};
  }
  return std::nullopt;
}

auto ObjectRepresentation::valueBits(const Type* type, const ConstValue& value)
    -> std::optional<ConstInt::UWide> {
  if (auto format = memoryLayout()->floatingPointFormat(type)) {
    if (auto stored = std::get_if<ConstFloat16>(&value)) return stored->bits;
    if (auto stored = std::get_if<float>(&value);
        stored && unqualified_cast<FloatType>(type))
      return std::bit_cast<std::uint32_t>(*stored);
    if (auto stored = std::get_if<double>(&value);
        stored && unqualified_cast<DoubleType>(type))
      return std::bit_cast<std::uint64_t>(*stored);
    auto real = interp_.toDouble(value);
    if (!real) return std::nullopt;
    return format->representation(*real);
  }
  auto converted = interp_.toArithmeticType(value, type);
  if (!converted) return std::nullopt;
  auto integer = std::get_if<ConstInt>(&*converted);
  if (!integer) return std::nullopt;
  return static_cast<ConstInt::UWide>(integer->toWide());
}

auto ObjectRepresentation::scalarValue(const Type* type, ConstInt::UWide bits)
    -> std::optional<ConstValue> {
  if (unqualified_cast<HalfType>(type) || unqualified_cast<Float16Type>(type))
    return ConstValue{ConstFloat16{static_cast<std::uint16_t>(bits)}};
  if (unqualified_cast<FloatType>(type))
    return ConstValue{std::bit_cast<float>(static_cast<std::uint32_t>(bits))};
  if (unqualified_cast<DoubleType>(type))
    return ConstValue{std::bit_cast<double>(static_cast<std::uint64_t>(bits))};
  if (auto format = memoryLayout()->floatingPointFormat(type))
    return interp_.toArithmeticType(ConstValue{format->value(bits)}, type);
  auto integer =
      traits_.integral_constant(type, static_cast<ConstInt::Wide>(bits));
  if (!integer) return std::nullopt;
  if (traits_.is_bool(type) && integer->toUWide() > 1) return std::nullopt;
  return ConstValue{*integer};
}

auto ObjectRepresentation::storeBits(BitRange range, ConstInt::UWide bits)
    -> bool {
  for (std::uint32_t index = 0; index < range.width; ++index) {
    auto position = range.position + index;
    auto byte = range.offset + position / 8;
    if (byte >= bytes_.size()) return false;
    auto mask = static_cast<std::uint8_t>(1u << (position % 8));
    defined_[byte] |= mask;
    if ((bits >> index) & 1) bytes_[byte] |= mask;
  }
  return true;
}

auto ObjectRepresentation::loadBits(BitRange range) const
    -> std::optional<ConstInt::UWide> {
  ConstInt::UWide bits = 0;
  for (std::uint32_t index = 0; index < range.width; ++index) {
    auto position = range.position + index;
    auto byte = range.offset + position / 8;
    if (byte >= bytes_.size()) return std::nullopt;
    auto mask = static_cast<std::uint8_t>(1u << (position % 8));
    if (!(defined_[byte] & mask)) return std::nullopt;
    if (bytes_[byte] & mask) bits |= ConstInt::UWide(1) << index;
  }
  return bits;
}

auto ObjectRepresentation::storeScalar(const Type* type,
                                       const ConstValue& value,
                                       std::uint64_t offset) -> bool {
  if (traits_.is_null_pointer(type)) return true;
  auto range = valueRange(type, offset);
  if (!range) return false;
  auto bits = valueBits(type, value);
  if (!bits) return false;
  return storeBits(*range, *bits);
}

auto ObjectRepresentation::loadScalar(const Type* type, std::uint64_t offset)
    -> std::optional<ConstValue> {
  if (traits_.is_null_pointer(type)) return interp_.zeroInitialize(type);
  auto range = valueRange(type, offset);
  if (!range) return std::nullopt;
  auto bits = loadBits(*range);
  if (!bits) {
    if (receivesIndeterminateValue(type)) return IndeterminateValue{};
    return std::nullopt;
  }
  return scalarValue(type, *bits);
}

auto ObjectRepresentation::storeArray(const BoundedArrayType* type,
                                      const ConstValue& value,
                                      std::uint64_t offset) -> bool {
  auto list = std::get_if<std::shared_ptr<InitializerList>>(&value);
  if (!list || !*list) return false;
  auto stride = memoryLayout()->sizeOf(type->elementType());
  if (!stride) return false;
  for (std::size_t index = 0; index < (*list)->size(); ++index) {
    const auto& element = std::get<0>((*list)->element(index));
    if (!store(type->elementType(), element, offset + index * *stride))
      return false;
  }
  return true;
}

auto ObjectRepresentation::loadArray(const BoundedArrayType* type,
                                     std::uint64_t offset)
    -> std::optional<ConstValue> {
  auto stride = memoryLayout()->sizeOf(type->elementType());
  if (!stride) return std::nullopt;
  auto list = std::make_shared<InitializerList>();
  for (std::size_t index = 0; index < type->size(); ++index) {
    auto element = load(type->elementType(), offset + index * *stride);
    if (!element) return std::nullopt;
    list->append(std::move(*element), type->elementType());
  }
  return ConstValue{std::move(list)};
}

auto ObjectRepresentation::storeObject(const ClassType* type,
                                       const ConstValue& value,
                                       std::uint64_t offset) -> bool {
  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!object || !*object || (*object)->isConstexprUnknown()) return false;
  auto layout = type->symbol()->resolvedDefinition()->layout();
  if (!layout) return false;
  for (const auto& member : (*object)->members()) {
    auto where = placement(*layout, const_cast<Symbol*>(member.symbol), offset);
    if (!where || !storeElement(*where, member.value)) return false;
  }
  return true;
}

auto ObjectRepresentation::loadObject(const ClassType* type,
                                      std::uint64_t offset)
    -> std::optional<ConstValue> {
  auto classSymbol = type->symbol()->resolvedDefinition();
  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;
  auto object = std::make_shared<ConstObject>(type);
  for (auto element : traits_.aggregate_elements(classSymbol)) {
    auto where = placement(*layout, element, offset);
    if (!where) return std::nullopt;
    auto value = loadElement(*where);
    if (!value) return std::nullopt;
    object->addMember(element, std::move(*value));
  }
  return ConstValue{std::move(object)};
}

auto ObjectRepresentation::storeElement(const Placement& placement,
                                        const ConstValue& value) -> bool {
  if (!placement.bitField)
    return store(placement.type, value, placement.offset);
  if (std::holds_alternative<IndeterminateValue>(value)) return true;
  auto bits = valueBits(traits_.remove_cv(placement.type), value);
  if (!bits) return false;
  return storeBits(*placement.bitField, *bits);
}

auto ObjectRepresentation::loadElement(const Placement& placement)
    -> std::optional<ConstValue> {
  if (!placement.bitField) return load(placement.type, placement.offset);
  auto type = traits_.remove_cv(placement.type);
  auto bits = loadBits(*placement.bitField);
  if (!bits) return std::nullopt;
  auto representation = traits_.integral_representation(type);
  if (!representation) return std::nullopt;
  auto narrow = ConstInt::make(static_cast<ConstInt::Wide>(*bits),
                               static_cast<int>(placement.bitField->width),
                               representation->isSigned);
  if (!narrow) return std::nullopt;
  return scalarValue(type, static_cast<ConstInt::UWide>(narrow->toWide()));
}

auto ObjectRepresentation::receivesIndeterminateValue(const Type* type)
    -> bool {
  if (traits_.is_narrow_char_type(type) && traits_.is_unsigned(type))
    return true;
  auto enumType = type_cast<ScopedEnumType>(type);
  if (!enumType) return false;
  auto byte = lookupStandardLibraryType(interp_.translationUnit(),
                                        WellKnownName::T_BYTE);
  return byte && enumType->symbol() == byte;
}
}  // namespace

auto ASTInterpreter::memoryRange(const ConstAddress& address,
                                 const Type* elementType)
    -> std::optional<MemoryRange> {
  auto located = typedAddress(address);
  if (!validPosition(*located)) return std::nullopt;
  auto type = located->elementType();
  if (traits.is_void(elementType)) elementType = nullptr;
  if (elementType && !traits.is_similar(type, elementType)) return std::nullopt;
  if (traits.is_volatile(type)) return std::nullopt;
  bool readOnly = traits.is_const(type) || located->stringLiteral();
  if (located->root()) {
    auto rootType = located->root()->type();
    if (traits.is_volatile(rootType)) return std::nullopt;
    auto field = symbol_cast<FieldSymbol>(located->symbol());
    if (!field || !field->isMutable()) readOnly |= traits.is_const(rootType);
  }
  auto extent = *arrayExtent(*located);
  auto offset = static_cast<std::uintmax_t>(located->offset());
  return MemoryRange{traits.remove_cv(type), extent - offset, readOnly};
}

auto ASTInterpreter::assignMemoryValue(ConstValue& destination,
                                       const ConstValue& source) -> bool {
  std::vector<std::pair<ConstValue*, const ConstValue*>> pending{
      {&destination, &source}};
  while (!pending.empty()) {
    auto [target, value] = pending.back();
    pending.pop_back();
    if (auto object = std::get_if<std::shared_ptr<ConstObject>>(target)) {
      auto sourceObject = std::get_if<std::shared_ptr<ConstObject>>(value);
      if (!*object || !sourceObject || !*sourceObject) return false;
      for (const auto& member : (*sourceObject)->members()) {
        auto slot = (*object)->mutableSubobject(member.symbol);
        if (!slot)
          (*object)->setMember(member.symbol, cloneValue(member.value));
        else
          pending.emplace_back(slot, &member.value);
      }
      continue;
    }
    if (auto list = std::get_if<std::shared_ptr<InitializerList>>(target)) {
      auto sourceList = std::get_if<std::shared_ptr<InitializerList>>(value);
      if (!*list || !sourceList || !*sourceList) return false;
      if ((*list)->size() != (*sourceList)->size()) return false;
      auto count = std::max((*list)->initializedElements().size(),
                            (*sourceList)->initializedElements().size());
      for (std::size_t index = 0; index < count; ++index)
        pending.emplace_back(&std::get<0>((*list)->mutableElement(index)),
                             &std::get<0>((*sourceList)->element(index)));
      if ((*sourceList)->hasFiller())
        (*list)->setFiller((*sourceList)->filler());
      continue;
    }
    *target = cloneValue(*value);
  }
  return true;
}

auto ASTInterpreter::evaluateBuiltinMemmove(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto destinationExpression = argumentAt(ast, 0);
  auto sourceExpression = argumentAt(ast, 1);
  auto sizeExpression = argumentAt(ast, 2);
  if (!destinationExpression || !sourceExpression || !sizeExpression)
    return std::nullopt;

  auto destinationValue = expression(destinationExpression);
  auto sourceValue = expression(sourceExpression);
  auto sizeValue = expression(sizeExpression);
  if (!destinationValue || !sourceValue || !sizeValue) return std::nullopt;
  auto size = toUInt(*sizeValue);
  if (!size) return std::nullopt;
  if (*size == 0) return destinationValue;

  auto destination =
      std::get_if<std::shared_ptr<ConstAddress>>(&*destinationValue);
  auto source = std::get_if<std::shared_ptr<ConstAddress>>(&*sourceValue);
  if (!destination || !*destination || !source || !*source) return std::nullopt;

  auto destinationType = strippedPointerType(destinationExpression);
  auto sourceType = strippedPointerType(sourceExpression);
  if (!destinationType || !sourceType) return std::nullopt;
  auto destinationRange = memoryRange(
      **destination, traits.remove_cv(destinationType->elementType()));
  auto sourceRange =
      memoryRange(**source, traits.remove_cv(sourceType->elementType()));
  if (!destinationRange || !sourceRange) return std::nullopt;
  if (destinationRange->readOnly) return std::nullopt;
  auto elementType = destinationRange->elementType;
  if (!traits.is_same(elementType, sourceRange->elementType))
    return std::nullopt;
  if (!traits.is_trivially_copyable(elementType)) return std::nullopt;

  auto elementSize = control()->memoryLayout()->sizeOf(elementType);
  if (!size || !elementSize || !*elementSize) return std::nullopt;
  if (*size % *elementSize) return std::nullopt;
  auto count = *size / *elementSize;
  if (count > destinationRange->count || count > sourceRange->count)
    return std::nullopt;

  std::vector<ConstValue> values;
  std::vector<ConstValue*> slots;
  for (std::uintmax_t index = 0; index < count; ++index) {
    auto value = loadAddress(**source, index, elementType);
    auto slot = addressSlot(**destination, index, elementType);
    if (!value || !slot) return std::nullopt;
    values.push_back(cloneValue(*value));
    slots.push_back(slot);
  }
  for (std::size_t index = 0; index < values.size(); ++index)
    if (!assignMemoryValue(*slots[index], values[index])) return std::nullopt;
  return destinationValue;
}

auto ASTInterpreter::evaluateBuiltinCharacterLength(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;
  auto value = expression(argument);
  if (!value) return std::nullopt;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
  auto pointer = strippedPointerType(argument);
  if (!address || !*address || !pointer) return std::nullopt;
  auto range = memoryRange(**address, pointer->elementType());
  if (!range) return std::nullopt;

  for (std::uintmax_t index = 0; index < range->count; ++index) {
    if (!std::in_range<std::intmax_t>(index)) return std::nullopt;
    auto element = loadAddress(**address, static_cast<std::intmax_t>(index),
                               range->elementType);
    auto unit = element ? toInt(*element) : std::nullopt;
    if (!unit) return std::nullopt;
    if (*unit == 0) return ConstValue{ConstInt{index}};
  }
  return std::nullopt;
}

auto ASTInterpreter::evaluateBuiltinCharacterComparison(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  if (!leftExpression || !rightExpression) return std::nullopt;
  auto left = expression(leftExpression);
  auto right = expression(rightExpression);
  if (!left || !right) return std::nullopt;

  auto kind = resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression));
  bool stopAtNull = true;
  bool ignoreCase = false;
  bool booleanResult = false;
  switch (kind) {
    case BuiltinFunctionKind::T___BUILTIN_MEMCMP:
      stopAtNull = false;
      break;
    case BuiltinFunctionKind::T___BUILTIN_BCMP:
      stopAtNull = false;
      booleanResult = true;
      break;
    case BuiltinFunctionKind::T___BUILTIN_STRCASECMP:
    case BuiltinFunctionKind::T___BUILTIN_STRNCASECMP:
      ignoreCase = true;
      break;
    default:
      break;
  }
  auto count = std::numeric_limits<std::uintmax_t>::max();
  if (auto sizeExpression = argumentAt(ast, 2)) {
    auto sizeValue = expression(sizeExpression);
    if (!sizeValue) return std::nullopt;
    auto size = toUInt(*sizeValue);
    if (!size) return std::nullopt;
    count = *size;
  }
  for (std::uintmax_t index = 0; index < count; ++index) {
    auto leftUnit = pointeeCharacter(*left, index);
    auto rightUnit = pointeeCharacter(*right, index);
    if (!leftUnit || !rightUnit) return std::nullopt;
    int leftCharacter = static_cast<unsigned char>(*leftUnit);
    int rightCharacter = static_cast<unsigned char>(*rightUnit);
    if (ignoreCase) {
      leftCharacter = std::tolower(leftCharacter);
      rightCharacter = std::tolower(rightCharacter);
    }
    if (leftCharacter != rightCharacter) {
      if (booleanResult) return ConstValue{std::intmax_t{1}};
      if (leftCharacter < rightCharacter) return ConstValue{std::intmax_t{-1}};
      return ConstValue{std::intmax_t{1}};
    }
    if (stopAtNull && leftCharacter == 0) return ConstValue{std::intmax_t{0}};
  }
  return ConstValue{std::intmax_t{0}};
}

auto ASTInterpreter::evaluateBuiltinArithmeticOverflow(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  auto outputExpression = argumentAt(ast, 2);
  if (!leftExpression || !rightExpression || !outputExpression)
    return std::nullopt;

  auto leftValue = evaluate(leftExpression);
  auto rightValue = evaluate(rightExpression);
  auto outputValue = evaluate(outputExpression);
  if (!leftValue || !rightValue || !outputValue) return std::nullopt;

  auto left = toInt(*leftValue);
  auto right = toInt(*rightValue);
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*outputValue);
  if (!left || !right || !address || !*address) return std::nullopt;

  auto slot = addressSlot(**address, 0);
  auto pointer = type_cast<PointerType>(outputExpression->type);
  if (!slot || !pointer) return std::nullopt;

  auto outputType = traits.remove_cv(pointer->elementType());
  auto size = control()->memoryLayout()->sizeOf(outputType);
  if (!size || *size > sizeof(std::intmax_t)) return std::nullopt;

  const auto isBoolOutput =
      traits.is_same(outputType, control()->getBoolType());
  const auto width = isBoolOutput ? 1 : static_cast<int>(*size * 8);
  const auto isSignedOutput = traits.is_signed(outputType);

  auto widen = [&](std::intmax_t value, const Type* type) -> ConstInt::Wide {
    if (traits.is_signed(type)) return static_cast<ConstInt::Wide>(value);
    return static_cast<ConstInt::Wide>(
        static_cast<ConstInt::UWide>(static_cast<std::uintmax_t>(value)));
  };

  const auto lhs = widen(*left, leftExpression->type);
  const auto rhs = widen(*right, rightExpression->type);

  auto operation = arithmeticOverflowOperation(resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression)));
  if (!operation) return std::nullopt;

  const auto value = applyArithmeticOverflowOperation(*operation, lhs, rhs);
  const auto overflow = !fitsIntegerWidth(value, width, isSignedOutput);

  auto result = ConstInt::make(value, width, isSignedOutput);
  if (!result) return std::nullopt;

  *slot = result->toIntMax();
  return ConstValue{overflow};
}

auto ASTInterpreter::evaluateBuiltinCarryArithmetic(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto operation = carryArithmeticOperation(resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression)));
  if (!operation) return std::nullopt;

  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  auto carryInExpression = argumentAt(ast, 2);
  auto carryOutExpression = argumentAt(ast, 3);
  if (!leftExpression || !rightExpression || !carryInExpression ||
      !carryOutExpression)
    return std::nullopt;

  auto leftValue = evaluate(leftExpression);
  auto rightValue = evaluate(rightExpression);
  auto carryInValue = evaluate(carryInExpression);
  auto carryOutValue = evaluate(carryOutExpression);
  if (!leftValue || !rightValue || !carryInValue || !carryOutValue)
    return std::nullopt;

  auto left = toInt(*leftValue);
  auto right = toInt(*rightValue);
  auto carryIn = toInt(*carryInValue);
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*carryOutValue);
  if (!left || !right || !carryIn || !address || !*address) return std::nullopt;

  auto slot = addressSlot(**address, 0);
  auto pointer = type_cast<PointerType>(carryOutExpression->type);
  if (!slot || !pointer) return std::nullopt;

  auto type = traits.remove_cv(pointer->elementType());
  auto size = control()->memoryLayout()->sizeOf(type);
  if (!size || *size > sizeof(std::intmax_t)) return std::nullopt;
  const auto width = static_cast<int>(*size * 8);

  auto widen = [](std::intmax_t value) {
    return static_cast<ConstInt::Wide>(
        static_cast<ConstInt::UWide>(static_cast<std::uintmax_t>(value)));
  };

  const auto first =
      applyArithmeticOverflowOperation(*operation, widen(*left), widen(*right));
  auto wrappedFirst = ConstInt::make(first, width, /*isSigned=*/false);
  if (!wrappedFirst) return std::nullopt;

  const auto second = applyArithmeticOverflowOperation(
      *operation, widen(wrappedFirst->toIntMax()), widen(*carryIn));
  auto wrappedSecond = ConstInt::make(second, width, /*isSigned=*/false);
  if (!wrappedSecond) return std::nullopt;

  const auto carry = !fitsIntegerWidth(first, width, /*isSigned=*/false) ||
                     !fitsIntegerWidth(second, width, /*isSigned=*/false);

  *slot = std::intmax_t{carry ? 1 : 0};
  return ConstValue{wrappedSecond->toIntMax()};
}

auto ASTInterpreter::evaluateBuiltinFloatComparison(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  if (!leftExpression || !rightExpression) return std::nullopt;

  auto leftValue = evaluate(leftExpression);
  auto rightValue = evaluate(rightExpression);
  if (!leftValue || !rightValue) return std::nullopt;

  auto left = toFloat(*leftValue);
  auto right = toFloat(*rightValue);
  if (!left || !right) return std::nullopt;

  const auto unordered = std::isnan(*left) || std::isnan(*right);

  switch (resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression))) {
    case BuiltinFunctionKind::T___BUILTIN_ISGREATER:
      return ConstValue{!unordered && *left > *right};
    case BuiltinFunctionKind::T___BUILTIN_ISGREATEREQUAL:
      return ConstValue{!unordered && *left >= *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESS:
      return ConstValue{!unordered && *left < *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESSEQUAL:
      return ConstValue{!unordered && *left <= *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESSGREATER:
      return ConstValue{!unordered && *left != *right};
    case BuiltinFunctionKind::T___BUILTIN_ISUNORDERED:
      return ConstValue{unordered};
    default:
      return std::nullopt;
  }
}

auto ASTInterpreter::evaluateBuiltinAddressof(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateAddress(argumentAt(ast, 0));
}

auto ASTInterpreter::sourceLocation(CallExpressionAST* ast) const
    -> SourceLocation {
  return defaultInitializerContext_.location
             ? defaultInitializerContext_.location
             : ast->firstSourceLocation();
}

auto ASTInterpreter::sourceFunction(CallExpressionAST* ast) const
    -> FunctionSymbol* {
  if (!defaultInitializerContext_.location) return currentFunction_;
  auto scope = defaultInitializerContext_.scope;
  if (auto function = symbol_cast<FunctionSymbol>(scope)) return function;
  return scope ? scope->enclosingFunction() : nullptr;
}

auto ASTInterpreter::evaluateBuiltinColumn(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  return ConstValue{static_cast<std::intmax_t>(pos.column)};
}

auto ASTInterpreter::evaluateBuiltinSourceLocation(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pointer = type_cast<PointerType>(ast->type);
  auto type =
      pointer ? unqualified_cast<ClassType>(pointer->elementType()) : nullptr;
  if (!type) return std::nullopt;
  auto object = std::make_shared<ConstObject>(type);
  auto function = sourceFunction(ast);
  auto functionName =
      function ? to_string(function->type(), to_string(function->name()))
               : std::string{};
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  for (auto member : type->symbol()->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    auto name = to_string(field->name());
    if (name == "_M_file_name")
      object->addMember(field, control()->stringLiteralFromValue(pos.fileName));
    else if (name == "_M_function_name")
      object->addMember(field, control()->stringLiteralFromValue(functionName));
    else if (name == "_M_line")
      object->addMember(field, static_cast<std::intmax_t>(pos.line));
    else if (name == "_M_column")
      object->addMember(field, static_cast<std::intmax_t>(pos.column));
    else
      return std::nullopt;
  }
  auto variable =
      control()->newVariableSymbol(unit_->globalScope(), sourceLocation(ast));
  variable->setName(control()->newAnonymousId("source_location"));
  variable->setType(traits.add_const(type));
  variable->setConstexpr(true);
  variable->setStatic(true);
  variable->setConstValue(ConstValue{std::move(object)});
  return ConstValue{std::make_shared<ConstAddress>(variable)};
}

auto ASTInterpreter::evaluateBuiltinLine(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  return ConstValue{static_cast<std::intmax_t>(pos.line)};
}

auto ASTInterpreter::evaluateBuiltinFile(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  auto lit = control()->stringLiteralFromValue(pos.fileName);
  return ConstValue{lit};
}

auto ASTInterpreter::evaluateBuiltinFunction(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto function = sourceFunction(ast);
  auto lit = control()->stringLiteralFromValue(
      function ? to_string(function->name()) : std::string{});
  return ConstValue{lit};
}

auto ASTInterpreter::evaluateBuiltinComplex(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto complexType = unqualified_cast<ComplexType>(ast->type);
  if (!complexType) return std::nullopt;

  std::vector<ConstValue> parts;
  for (auto argument : ListView{ast->expressionList}) {
    auto value = expression(argument);
    if (!value) return std::nullopt;
    auto converted = toArithmeticType(*value, complexType->elementType());
    if (!converted) return std::nullopt;
    parts.push_back(*converted);
  }

  if (parts.size() != 2) return std::nullopt;

  return ConstValue{std::make_shared<ConstComplex>(parts[0], parts[1])};
}

auto ASTInterpreter::evaluateBuiltinHugeVal(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<double>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinHugeValf(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<float>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinHugeVall(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<long double>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinNanPayload(CallExpressionAST* ast)
    -> std::optional<std::string> {
  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;

  auto value = evaluate(argument);
  if (!value) return std::nullopt;

  return nullTerminatedString(*value);
}

auto ASTInterpreter::evaluateBuiltinNan(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{std::nan(payload->c_str())};
}

auto ASTInterpreter::evaluateBuiltinNanf(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{static_cast<double>(std::nanf(payload->c_str()))};
}

auto ASTInterpreter::evaluateBuiltinNanl(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{static_cast<double>(std::nanl(payload->c_str()))};
}

auto ASTInterpreter::evaluateBuiltinLockFree(CallExpressionAST* ast,
                                             bool alwaysLockFree,
                                             bool ignoresPointerOperand)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;

  auto sizeArgument = argumentAt(ast, 0);
  if (!sizeArgument) return std::nullopt;

  auto sizeValue = evaluate(sizeArgument);
  if (!sizeValue) return std::nullopt;

  auto size = toUInt(*sizeValue);
  if (!size) return std::nullopt;

  auto isLockFree = [&]() -> bool {
    if (*size == 0 || !std::has_single_bit(*size)) return false;

    const auto inlineWidth =
        control()->memoryLayout()->maxAtomicInlineWidth() / 8;
    if (*size > inlineWidth) return false;

    if (ignoresPointerOperand || *size == 1) return true;

    auto pointerArgument = argumentAt(ast, 1);
    if (!pointerArgument) return false;

    if (auto pointerValue = evaluate(pointerArgument)) {
      if (auto address = toUInt(*pointerValue)) {
        if (*address % *size == 0) return true;
      }
    }

    auto pointerType = strippedPointerType(pointerArgument);
    if (!pointerType) return false;

    auto pointeeType = pointerType->elementType();
    if (!unit_->typeTraits().is_complete(pointeeType)) return false;

    auto alignment = control()->memoryLayout()->alignmentOf(pointeeType);
    return alignment && *alignment >= *size;
  };

  if (isLockFree()) return ConstValue{std::intmax_t(1)};

  if (!alwaysLockFree) return std::nullopt;

  return ConstValue{std::intmax_t(0)};
}

auto ASTInterpreter::fieldOffsetIn(const Type* type, FieldSymbol* field)
    -> std::optional<std::uintmax_t> {
  if (!field) return std::nullopt;
  auto classType = unqualified_cast<ClassType>(traits.remove_reference(type));
  if (!classType) return std::nullopt;
  auto classSymbol = classType->symbol();
  traits.requireCompleteClass(classSymbol);
  auto layout = classSymbol->resolvedDefinition()->layout();
  if (!layout) return std::nullopt;
  auto info = layout->getFieldInfo(field);
  if (!info) return std::nullopt;
  return info->offset;
}

auto ASTInterpreter::offsetOfDesignated(BuiltinOffsetofExpressionAST* ast)
    -> std::optional<std::uintmax_t> {
  if (!ast->symbol || !ast->typeId) return std::nullopt;
  auto offset = fieldOffsetIn(ast->typeId->type, ast->symbol);
  if (!offset) return std::nullopt;

  const Type* designatedType = ast->symbol->type();
  for (auto designator : ListView{ast->designatorList}) {
    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto fieldOffset = fieldOffsetIn(designatedType, dot->symbol);
      if (!fieldOffset) return std::nullopt;
      *offset += *fieldOffset;
      designatedType = dot->symbol->type();
      continue;
    }

    auto subscript = ast_cast<SubscriptDesignatorAST>(designator);
    if (!subscript) return std::nullopt;

    auto indexValue = evaluate(subscript->expression);
    auto index = indexValue ? toInt(*indexValue) : std::nullopt;
    if (!index) return std::nullopt;

    auto elementType = traits.remove_extent(designatedType);
    auto elementSize = control()->memoryLayout()->sizeOf(elementType);
    if (!elementSize) return std::nullopt;

    *offset += static_cast<std::uintmax_t>(*index) * *elementSize;
    designatedType = elementType;
  }

  return offset;
}

auto ASTInterpreter::evaluateBuiltinAssume(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;

  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;

  auto value = evaluate(argument);
  if (!value) return std::nullopt;

  auto holds = toBool(*value);
  if (!holds || !*holds) return std::nullopt;

  return ConstValue{ConstInt{std::intmax_t{0}}};
}

auto ASTInterpreter::evaluateBuiltinBitCount(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;

  auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
  auto operation = bitCountOperation(resolveBuiltinFunctionKind(idExpr));
  if (!operation) return std::nullopt;

  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;

  auto value = evaluate(argument);
  if (!value) return std::nullopt;

  auto representation = traits.integral_representation(argument->type);
  if (!representation) return std::nullopt;
  if (!ConstInt::isRepresentableWidth(representation->bits))
    return std::nullopt;

  auto stored = std::get_if<ConstInt>(&*value);
  if (!stored) return std::nullopt;

  auto operand = ConstInt::make(stored->toWideValue(), representation->bits,
                                /*isSigned=*/false);
  if (!operand) return std::nullopt;

  const auto width = representation->bits;

  if (operand->isZero()) {
    if (auto fallback = argumentAt(ast, 1)) {
      auto fallbackValue = evaluate(fallback);
      if (!fallbackValue) return std::nullopt;
      auto result = toInt(*fallbackValue);
      if (!result) return std::nullopt;
      return ConstValue{*result};
    }
  }

  switch (*operation) {
    case BitCountOperation::kCountLeadingZeros:
      if (operand->isZero()) return std::nullopt;
      return ConstValue{
          static_cast<std::intmax_t>(operand->countLeadingZeros())};

    case BitCountOperation::kCountTrailingZeros:
      if (operand->isZero()) return std::nullopt;
      return ConstValue{
          static_cast<std::intmax_t>(operand->countTrailingZeros())};

    case BitCountOperation::kPopulationCount:
      return ConstValue{static_cast<std::intmax_t>(operand->popcount())};

    case BitCountOperation::kParity:
      return ConstValue{static_cast<std::intmax_t>(operand->popcount() & 1)};

    case BitCountOperation::kFindFirstSet:
      if (operand->isZero()) return ConstValue{std::intmax_t(0)};
      return ConstValue{
          static_cast<std::intmax_t>(operand->countTrailingZeros() + 1)};

    case BitCountOperation::kCountLeadingRedundantSignBits: {
      auto signBit = ConstInt::make(1, width, /*isSigned=*/false);
      if (!signBit) return std::nullopt;
      auto shift = ConstInt::make(width - 1, width, /*isSigned=*/false);
      if (!shift) return std::nullopt;

      auto magnitude = *operand;
      if (!((*operand >> *shift) & *signBit).isZero()) {
        auto allOnes = ConstInt::make(-1, width, /*isSigned=*/false);
        if (!allOnes) return std::nullopt;
        magnitude = *operand ^ *allOnes;
      }

      return ConstValue{
          static_cast<std::intmax_t>(magnitude.countLeadingZeros() - 1)};
    }
  }

  return std::nullopt;
}

auto ASTInterpreter::evaluateBuiltinAtomicAlwaysLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/true,
                                 /*ignoresPointerOperand=*/false);
}

auto ASTInterpreter::evaluateBuiltinAtomicIsLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/false,
                                 /*ignoresPointerOperand=*/false);
}

auto ASTInterpreter::evaluateBuiltinC11AtomicIsLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/false,
                                 /*ignoresPointerOperand=*/true);
}

auto ASTInterpreter::bitCast(const ConstValue& value, const Type* sourceType,
                             const Type* targetType)
    -> std::optional<ConstValue> {
  if (traits.has_constexpr_unknown_representation(sourceType)) return {};
  if (traits.has_constexpr_unknown_representation(targetType)) return {};
  auto size = control()->memoryLayout()->sizeOf(targetType);
  if (!size) return std::nullopt;
  ObjectRepresentation representation{*this, *size};
  if (!representation.store(sourceType, value, 0)) return std::nullopt;
  return representation.load(targetType, 0);
}

}  // namespace cxx
