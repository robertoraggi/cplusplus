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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/class_value_abi.h>
#include <cxx/codegen/codegen.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <filesystem>
#include <format>
#include <limits>
#include <map>

namespace cxx {
static auto isMemberOfClassTemplateSpecialization(Symbol* symbol) -> bool {
  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto cls = symbol_cast<ClassSymbol>(scope)) {
      if (cls->isSpecialization()) return true;
    }
  }
  return false;
}

auto Codegen::hasVagueFunctionEmission(FunctionSymbol* function) const -> bool {
  if (!function) return false;
  if (is_inline_or_templated(function)) return true;
  if (!function->isSpecialization()) function = function->canonical();
  if (function->isDefaulted()) return true;
  if (function->isStructorVariant()) return true;
  return hasVagueFunctionEmission(function->enclosingFunction());
}

auto Codegen::hasVagueEmission(Symbol* symbol) const -> bool {
  if (!symbol) return false;
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return hasVagueFunctionEmission(function);
  if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
    if (variable->isInline()) return true;
    if (variable->isSpecialization()) return true;
  }
  if (isMemberOfClassTemplateSpecialization(symbol)) return true;
  return hasVagueFunctionEmission(symbol->enclosingFunction());
}

auto Codegen::hasInternalLinkage(Symbol* symbol) const -> bool {
  if (!symbol) return false;
  if (has_internal_linkage(symbol)) return true;
  if (is_specialized_on_internal_type(symbol)) return true;
  if (is_declared_with_internal_type(symbol)) return true;

  auto enclosingFunction = symbol->enclosingFunction();
  if (!enclosingFunction) return false;

  if (hasInternalLinkage(enclosingFunction)) return true;
  return !hasVagueFunctionEmission(enclosingFunction);
}

auto Codegen::symbolLinkage(Symbol* symbol) const -> ir::Linkage {
  if (hasInternalLinkage(symbol)) return ir::Linkage::Internal;
  if (symbol && symbol->isWeak()) return ir::Linkage::Weak;
  if (unit_->isExplicitInstantiationDefinition(
          symbol_cast<FunctionSymbol>(symbol))) {
    return ir::Linkage::WeakODR;
  }
  if (hasVagueEmission(symbol)) return ir::Linkage::LinkOnceODR;
  return ir::Linkage::External;
}

static auto isMemberOfExplicitInstantiationDeclaredClass(TranslationUnit* unit,
                                                         Symbol* symbol)
    -> bool {
  if (symbol->isExcludedFromExplicitInstantiation()) return false;

  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto cls = symbol_cast<ClassSymbol>(scope)) {
      if (cls->isExplicitInstantiationDeclared(unit)) return true;
    }
  }
  return false;
}

[[nodiscard]] static auto suppressesOutOfLineCopy(TranslationUnit* unit,
                                                  FunctionSymbol* function)
    -> bool {
  if (unit->isExplicitInstantiationDefinition(function)) return false;
  if (function->isExplicitInstantiationDeclared(unit)) return true;
  if (function->isInline()) return false;
  return isMemberOfExplicitInstantiationDeclaredClass(unit, function);
}

Codegen::Codegen(ir::Emitter& emitter, TranslationUnit* unit, Options options)
    : emitter_(emitter),
      unit_(unit),
      traits(unit),
      options_(std::move(options)),
      debugInfo_(options_.debugInfo) {
  const auto triple = unit->control()->memoryLayout()->triple();
  const auto arch = triple.substr(0, triple.find('-'));
  isWasmTarget_ = arch == "wasm32" || arch == "wasm64";
}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::getAlignment(const Type* type) -> uint64_t {
  return control()->memoryLayout()->alignmentOf(type).value_or(1);
}

auto Codegen::getAlignment(VariableSymbol* var) -> uint64_t {
  auto alignment = getAlignment(var->type());
  auto requested = static_cast<uint64_t>(var->explicitAlignment());
  return std::max(alignment, requested);
}

namespace {

[[nodiscard]] auto isReadOnlyStorage(TypeTraits& traits, const Type* type,
                                     bool isConstantInitialized) -> bool {
  if (!isConstantInitialized) return false;
  if (traits.is_reference(type)) return true;
  if (!traits.is_const(type)) return false;
  if (traits.has_mutable_subobject(type)) return false;
  return traits.has_trivial_destructor(type);
}

[[nodiscard]] auto alignmentAtOffset(uint64_t alignment, uint64_t offset)
    -> uint64_t {
  if (offset == 0) return alignment;
  return std::min(alignment, offset & (~offset + 1));
}

[[nodiscard]] auto withoutParentheses(ExpressionAST* expression)
    -> ExpressionAST* {
  while (auto nested = ast_cast<NestedExpressionAST>(expression))
    expression = nested->expression;
  return expression;
}

}  // namespace

auto Codegen::lvalueAlignment(ExpressionAST* expression) -> uint64_t {
  expression = withoutParentheses(expression);

  if (auto member = ast_cast<MemberExpressionAST>(expression))
    return memberAlignment(member);

  if (auto subscript = ast_cast<SubscriptExpressionAST>(expression))
    return elementAlignment(subscript);

  if (ast_cast<TargetExpressionAST>(expression) && targetExpression_)
    return lvalueAlignment(targetExpression_);

  if (auto id = ast_cast<IdExpressionAST>(expression)) {
    if (auto variable = objectVariable(id->symbol))
      return getAlignment(variable);
  }

  return getAlignment(expression->type);
}

auto Codegen::objectVariable(Symbol* symbol) -> VariableSymbol* {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable || traits.is_reference(variable->type())) return nullptr;
  return variable;
}

auto Codegen::memberAlignment(MemberExpressionAST* member) -> uint64_t {
  if (auto variable = objectVariable(member->symbol))
    return getAlignment(variable);

  auto field = symbol_cast<FieldSymbol>(member->symbol);
  if (!field || field->isStatic()) return getAlignment(member->type);
  if (traits.is_reference(field->type())) return getAlignment(member->type);

  auto baseAlignment = getAlignment(member->type);
  if (member->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointer =
        type_cast<PointerType>(traits.remove_cv(member->baseExpression->type));
    if (!pointer) return baseAlignment;
    baseAlignment = getAlignment(pointer->elementType());
  } else {
    baseAlignment = lvalueAlignment(member->baseExpression);
  }

  if (auto owner = symbol_cast<ClassSymbol>(field->parent()))
    baseAlignment = std::min(baseAlignment, getAlignment(owner->type()));

  return alignmentAtOffset(baseAlignment,
                           static_cast<uint64_t>(field->localOffset()));
}

auto Codegen::elementAlignment(SubscriptExpressionAST* subscript) -> uint64_t {
  auto base = withoutParentheses(subscript->baseExpression);
  if (auto decay = ast_cast<ImplicitCastExpressionAST>(base);
      decay && decay->castKind == ImplicitCastKind::kArrayToPointerConversion)
    base = decay->expression;

  if (!traits.is_array(traits.remove_cv(base->type)))
    return getAlignment(subscript->type);

  auto elementSize =
      control()->memoryLayout()->sizeOf(subscript->type).value_or(1);
  return alignmentAtOffset(lvalueAlignment(base), elementSize);
}

auto Codegen::pointerSize() const -> std::int64_t {
  return static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
}

auto Codegen::pointerSizedIntType() -> ir::TypeRef {
  return emitter_.integerType(static_cast<unsigned>(pointerSize() * 8));
}

auto Codegen::currentBlockMightHaveTerminator() -> bool {
  auto block = emitter_.insertionBlock();
  if (!block) {
    cxx_runtime_error("current block is null");
  }
  return emitter_.hasTerminator(block);
}

auto Codegen::newBlock() -> ir::BlockRef {
  return emitter_.createBlock(ir::FunctionRef{});
}

auto Codegen::newUniqueSymbolName(std::string_view prefix) -> std::string {
  auto& uniqueName = uniqueSymbolNames_[prefix];
  if (uniqueName == 0) {
    uniqueName = 1;
    return std::format("{}{}", prefix, uniqueName);
  }
  return std::format("{}{}", prefix, ++uniqueName);
}

auto Codegen::makeFloatInitializer(const Type* type, double value)
    -> ir::Initializer {
  return ir::Initializer::floatingValue(convertType(type), value);
}

auto Codegen::getFloatAttr(const std::optional<ConstValue>& value,
                           const Type* type) -> std::optional<ir::Initializer> {
  if (!value.has_value()) return {};

  auto ty = traits.remove_cvref(type);
  if (!traits.is_floating_point(ty)) return {};

  auto interp = ASTInterpreter{unit_};
  return interp.toDouble(*value).transform(
      [&](double converted) { return makeFloatInitializer(ty, converted); });
}

auto Codegen::nullMemberObjectPointer() const -> std::int64_t {
  return control()->memoryLayout()->nullMemberObjectPointer();
}

auto Codegen::classConstantSlots(const ConstValue& value,
                                 const ClassType* classType)
    -> std::optional<std::vector<ConstantSlot>> {
  if (auto listPtr = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*listPtr) return std::nullopt;
    std::vector<ConstantSlot> slots;
    slots.reserve((*listPtr)->elements.size());
    for (const auto& element : (*listPtr)->elements) slots.push_back(element);
    return slots;
  }

  auto objectPtr = std::get_if<std::shared_ptr<ConstObject>>(&value);
  if (!objectPtr || !*objectPtr) return std::nullopt;

  auto classSymbol = classType ? classType->symbol() : nullptr;
  if (!classSymbol) return std::nullopt;

  const auto& members = (*objectPtr)->members();

  if (classSymbol->isUnion()) {
    if (members.empty()) return std::nullopt;
    auto activeMember =
        symbol_cast<FieldSymbol>(const_cast<Symbol*>(members.front().symbol));
    if (!activeMember) return std::nullopt;
    std::vector<ConstantSlot> slots;
    slots.push_back(std::tuple{members.front().value, activeMember->type()});
    return slots;
  }

  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;
  if (layout->hasDirectVtable()) return std::nullopt;

  std::map<std::uint32_t, ConstantSlot> slotMap;
  std::map<std::uint32_t, std::tuple<std::intmax_t, const Type*>> bitFieldMap;
  auto interp = ASTInterpreter{unit_};

  for (const auto& member : members) {
    auto symbol = const_cast<Symbol*>(member.symbol);

    if (auto base = symbol_cast<BaseClassSymbol>(symbol)) {
      auto baseSymbol = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseSymbol) return std::nullopt;
      auto info = layout->getBaseInfo(baseSymbol, base->isVirtual());
      if (!info) continue;
      auto baseLayout = baseSymbol->layout();
      if (!baseLayout) return std::nullopt;
      if (baseLayout->isAbiEmpty()) continue;
      if (!baseLayout->virtualBases().empty()) return std::nullopt;
      slotMap[info->index] = std::tuple{member.value, baseSymbol->type()};
      continue;
    }

    auto field = symbol_cast<FieldSymbol>(symbol);
    if (!field) return std::nullopt;
    auto info = layout->getFieldInfo(field);
    if (!info) continue;

    if (info->bitWidth == 0) {
      slotMap[info->index] = std::tuple{member.value, field->type()};
      continue;
    }

    auto bits = interp.toInt(member.value);
    if (!bits) return std::nullopt;
    const auto width = std::min<std::uint32_t>(
        info->bitWidth, std::numeric_limits<std::uintmax_t>::digits);
    const auto mask = width == std::numeric_limits<std::uintmax_t>::digits
                          ? ~std::uintmax_t{0}
                          : (std::uintmax_t{1} << width) - 1;
    auto& [packed, packedType] = bitFieldMap[info->index];
    packed |= static_cast<std::intmax_t>(
        (static_cast<std::uintmax_t>(*bits) & mask) << info->bitOffset);
    packedType = field->type();
  }

  for (const auto& [index, packed] : bitFieldMap) {
    const auto& [bits, packedType] = packed;
    slotMap[index] = std::tuple{ConstValue{bits}, packedType};
  }

  if (slotMap.empty()) return std::vector<ConstantSlot>{};

  std::vector<ConstantSlot> slots(slotMap.rbegin()->first + 1);
  for (const auto& [index, slot] : slotMap) slots[index] = slot;
  return slots;
}

auto Codegen::constValueToInitializer(const ConstValue& value, const Type* type)
    -> std::optional<ir::Initializer> {
  auto interp = ASTInterpreter{unit_};

  if (traits.is_integral_or_enum(type)) {
    auto constValue = interp.toIntegralType(value, type);
    if (!constValue) return ir::Initializer::integerValue(convertType(type), 0);
    auto number = std::get<ConstInt>(*constValue);
    return ir::Initializer::integerValue(emitter_.integerType(number.width()),
                                         number);
  }

  if (type_cast<MemberObjectPointerType>(type)) {
    auto constValue = interp.memberObjectPointerOffset(value);
    return ir::Initializer::integerValue(
        emitter_.integerType(64),
        constValue.value_or(nullMemberObjectPointer()));
  }

  if (auto attr = getFloatAttr(value, type)) {
    return *attr;
  }

  if (traits.is_pointer(type) || traits.is_reference(type)) {
    if (std::get_if<std::shared_ptr<ConstLabelAddress>>(&value))
      return std::nullopt;
    if (auto intVal = std::get_if<ConstInt>(&value)) {
      if (intVal->isZero()) return ir::Initializer::null();
    }
    return std::nullopt;
  }

  if (traits.is_class(type)) {
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType) return std::nullopt;
    auto slots = classConstantSlots(value, classType);
    if (!slots) return std::nullopt;

    if (classType->isUnion()) {
      auto unionType = emitter_.asType(ir::TypeKind::Class, convertType(type));
      if (slots->empty() || !(*slots)[0] || !unionType ||
          this->classMembers(unionType).empty())
        return std::nullopt;
      if (convertType(std::get<1>(*(*slots)[0])) !=
          this->classMembers(unionType)[0])
        return std::nullopt;
    }

    std::vector<ir::Initializer> elements;
    for (const auto& slot : *slots) {
      if (!slot) {
        elements.push_back(ir::Initializer::zero());
        continue;
      }
      auto attr =
          constValueToInitializer(std::get<0>(*slot), std::get<1>(*slot));
      if (!attr) return std::nullopt;
      elements.push_back(*attr);
    }
    return ir::Initializer::aggregate(elements);
  }

  if (traits.is_array(type)) {
    auto constArrayPtr = std::get_if<std::shared_ptr<InitializerList>>(&value);
    if (!constArrayPtr || !*constArrayPtr) return std::nullopt;
    std::vector<ir::Initializer> elements;
    for (const auto& [elemValue, elemType] : (*constArrayPtr)->elements) {
      auto attr = constValueToInitializer(elemValue, elemType);
      if (!attr) return std::nullopt;
      elements.push_back(*attr);
    }
    return ir::Initializer::aggregate(elements);
  }

  return std::nullopt;
}

auto Codegen::emitStringLiteralAddress(SourceLocation loc,
                                       ir::TypeRef pointerType,
                                       const StringLiteral* literal)
    -> ir::ValueRef {
  literal->initialize(literal->encoding());
  std::string bytes(literal->stringValue());
  bytes.append(literal->codeUnitSize(), '\0');

  auto i8Type = emitter_.integerType(8);
  auto arrayType = this->arrayType(i8Type, bytes.size());
  auto initializer =
      ir::Initializer::byteString(std::string_view(bytes.data(), bytes.size()));
  auto name = newUniqueSymbolName(".str");

  {
    auto guard = ir::InsertionGuard(emitter_);
    emitter_.setModuleInsertionPoint(true);
    (void)this->declareGlobal(loc, {.name = name,
                                    .type = arrayType,
                                    .linkage = ir::Linkage::Internal,
                                    .isConstant = true,
                                    .alignment = static_cast<std::uint64_t>(0),
                                    .initializer = initializer,
                                    .unknownLocation = false});
  }

  return emitter_.addressOfSymbol(loc, pointerType, name);
}

auto Codegen::emitConstInitValue(SourceLocation loc, const Type* type,
                                 const ConstValue& value) -> ir::ValueRef {
  auto interp = ASTInterpreter{unit_};

  if (traits.is_integral_or_enum(type)) {
    auto irType = convertType(type);
    auto constValue = interp.toIntegralType(value, type);
    if (!constValue)
      return emitter_.constantLiteral(loc, irType,
                                      ir::Initializer::integerValue(irType, 0));
    return emitter_.constantLiteral(
        loc, irType,
        ir::Initializer::integerValue(irType, std::get<ConstInt>(*constValue)));
  }

  if (type_cast<MemberObjectPointerType>(type)) {
    auto irType = convertType(type);
    auto constValue = interp.memberObjectPointerOffset(value);
    return emitter_.constantLiteral(
        loc, irType,
        ir::Initializer::integerValue(
            irType, constValue.value_or(nullMemberObjectPointer())));
  }

  if (auto memberFunctionPointerType =
          type_cast<MemberFunctionPointerType>(type)) {
    FunctionSymbol* function = nullptr;
    std::int64_t adjustment = 0;
    if (auto addrPtr = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
      function = symbol_cast<FunctionSymbol>((*addrPtr)->symbol());
      adjustment = static_cast<std::int64_t>((*addrPtr)->offset());
    }
    return emitMemberFunctionPointerValue(loc, memberFunctionPointerType,
                                          function, adjustment);
  }

  if (traits.is_floating_point(type)) {
    auto irType = convertType(type);
    auto floatType = irType;
    auto constValue = interp.toDouble(value);
    return emitter_.constantLiteral(
        loc, floatType,
        ir::Initializer::floatingValue(floatType, constValue.value_or(0.0)));
  }

  if (traits.is_pointer(type) || traits.is_reference(type)) {
    auto ptrType = convertType(type);
    auto irPtrType = ptrType;

    if (auto addrPtr = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
      if (auto typeInfoFor = (*addrPtr)->typeInfoFor()) {
        auto typeInfoName = findOrCreateTypeInfo(typeInfoFor);
        return emitter_.addressOfSymbol(loc, irPtrType, typeInfoName);
      }
      auto symbol = (*addrPtr)->symbol();
      auto offset = (*addrPtr)->offset();
      if (auto literal = (*addrPtr)->stringLiteral()) {
        if (offset == 0)
          return emitStringLiteralAddress(loc, irPtrType, literal);
        auto bytePointerType = emitter_.pointerType(emitter_.integerType(8));
        auto base = emitStringLiteralAddress(loc, bytePointerType, literal);
        auto byteOffset =
            offset * static_cast<std::intmax_t>(literal->codeUnitSize());
        auto offsetType = emitter_.integerType(64);
        auto offsetValue = emitter_.constantLiteral(
            loc, offsetType,
            ir::Initializer::integerValue(offsetType, byteOffset));
        return emitter_.bitcast(
            loc, irPtrType,
            emitter_.pointerAdd(loc, bytePointerType, base, offsetValue));
      }
      if (auto result = staticStorageAddress(loc, symbol)) {
        if (offset != 0) {
          auto offsetVal = emitter_.constantLiteral(
              loc, emitter_.integerType(64),
              ir::Initializer::integerValue(emitter_.integerType(64), offset));
          result = emitter_.pointerAdd(loc, irPtrType, result, offsetVal);
        }
        return result;
      }
      if (auto funcSym = symbol_cast<FunctionSymbol>(symbol)) {
        auto funcOp = findOrCreateFunction(funcSym);
        return emitter_.addressOfSymbol(loc, irPtrType,
                                        this->functionName(funcOp));
      }
    }

    if (auto labelAddrPtr =
            std::get_if<std::shared_ptr<ConstLabelAddress>>(&value)) {
      return emitter_.labelAddress(loc, irPtrType, (*labelAddrPtr)->name(),
                                   function_);
    }

    if (auto strLitPtr = std::get_if<const StringLiteral*>(&value)) {
      return emitStringLiteralAddress(loc, irPtrType, *strLitPtr);
    }

    return emitter_.nullPointer(loc, irPtrType);
  }

  if (traits.is_class_or_union(type)) {
    auto classType = unqualified_cast<ClassType>(type);
    auto irType = convertType(type);
    auto slots = classConstantSlots(value, classType);

    if (classType && classType->isUnion()) {
      if (slots && !slots->empty() && (*slots)[0]) {
        auto& [elemValue, elemType] = *(*slots)[0];
        bool isZero = false;
        if (auto intVal = std::get_if<ConstInt>(&elemValue)) {
          isZero = intVal->isZero();
        } else if (auto floatVal = std::get_if<float>(&elemValue)) {
          isZero = (*floatVal == 0.0f);
        } else if (auto doubleVal = std::get_if<double>(&elemValue)) {
          isZero = (*doubleVal == 0.0);
        }

        if (isZero) {
          return emitter_.zero(loc, irType);
        }

        auto elemVal = emitConstInitValue(loc, elemType, elemValue);

        auto unionClassType = emitter_.asType(ir::TypeKind::Class, irType);
        if (unionClassType && !this->classMembers(unionClassType).empty() &&
            emitter_.typeOf(elemVal) == this->classMembers(unionClassType)[0]) {
          auto undef = emitter_.undef(loc, irType);
          return emitter_.insertValue(loc, irType, undef, elemVal,
                                      static_cast<int64_t>(0));
        }

        if (unionClassType && !this->classMembers(unionClassType).empty()) {
          auto dstFieldType = this->classMembers(unionClassType)[0];

          if (auto srcArr = emitter_.asType(ir::TypeKind::Array,
                                            emitter_.typeOf(elemVal))) {
            if (auto dstArr =
                    emitter_.asType(ir::TypeKind::Array, dstFieldType)) {
              if (emitter_.elementType(srcArr) ==
                  emitter_.elementType(dstArr)) {
                ir::ValueRef resized = elemVal;
                if (this->arraySize(srcArr) != this->arraySize(dstArr))
                  resized = emitter_.reshape(loc, dstArr, elemVal);
                auto undef = emitter_.undef(loc, irType);
                return emitter_.insertValue(loc, irType, undef, resized,
                                            static_cast<int64_t>(0));
              }
            }
          }

          auto classSymbol = classType->symbol();
          unsigned unionBits =
              static_cast<unsigned>(classSymbol->sizeInBytes()) * 8;
          auto intUnionType = emitter_.integerType(unionBits);

          ir::ValueRef intRepr;
          if (auto srcInt = emitter_.asType(ir::TypeKind::Integer,
                                            emitter_.typeOf(elemVal))) {
            unsigned srcBits = emitter_.scalarWidth(srcInt);
            if (srcBits < unionBits)
              intRepr = emitter_.zeroExtend(loc, elemVal, intUnionType);
            else if (srcBits > unionBits)
              intRepr = emitter_.truncate(loc, elemVal, intUnionType);
            else
              intRepr = elemVal;
          } else if (auto srcFloat = emitter_.asType(
                         ir::TypeKind::Floating, emitter_.typeOf(elemVal))) {
            unsigned srcBits = emitter_.scalarWidth(srcFloat);
            auto srcIntTy = emitter_.integerType(srcBits);
            ir::ValueRef asInt =
                emitter_.reinterpretBits(loc, elemVal, srcIntTy);
            if (srcBits < unionBits)
              intRepr = emitter_.zeroExtend(loc, asInt, intUnionType);
            else
              intRepr = asInt;
          }

          if (intRepr) {
            ir::ValueRef fieldVal;
            if (auto dstPointer =
                    emitter_.asType(ir::TypeKind::Pointer, dstFieldType)) {
              auto pointerBits = pointerSize() * 8;
              ir::ValueRef bits = intRepr;
              if (unionBits != pointerBits) {
                auto pointerIntType =
                    emitter_.integerType(static_cast<unsigned>(pointerBits));
                bits = emitter_.truncate(loc, bits, pointerIntType);
              }
              fieldVal = emitter_.intToPointer(loc, dstPointer, bits);
            } else if (auto dstFloat = emitter_.asType(ir::TypeKind::Floating,
                                                       dstFieldType)) {
              unsigned dstBits = emitter_.scalarWidth(dstFloat);
              ir::ValueRef bits = intRepr;
              if (unionBits != dstBits) {
                auto dstIntTy = emitter_.integerType(dstBits);
                bits = emitter_.truncate(loc, bits, dstIntTy);
              }
              fieldVal = emitter_.reinterpretBits(loc, bits, dstFloat);
            } else if (emitter_.asType(ir::TypeKind::Integer, dstFieldType)) {
              fieldVal = intRepr;
              if (emitter_.typeOf(intRepr) != dstFieldType)
                fieldVal = emitter_.truncate(loc, intRepr, dstFieldType);
            }
            if (fieldVal) {
              auto undef = emitter_.undef(loc, irType);
              return emitter_.insertValue(loc, irType, undef, fieldVal,
                                          static_cast<int64_t>(0));
            }
          }
        }

        return emitter_.bitcast(loc, irType, elemVal);
      }
      return emitter_.zero(loc, irType);
    }

    if (slots) {
      ir::ValueRef result = emitter_.zero(loc, irType);
      auto fieldTypes = this->classMembers(irType);
      for (size_t i = 0; i < slots->size(); ++i) {
        if (!(*slots)[i]) continue;
        auto& [elemValue, elemType] = *(*slots)[i];
        auto elemVal = emitConstInitValue(loc, elemType, elemValue);
        if (i < fieldTypes.size() &&
            emitter_.typeOf(elemVal) != fieldTypes[i]) {
          auto srcArr =
              emitter_.asType(ir::TypeKind::Array, emitter_.typeOf(elemVal));
          auto dstArr = emitter_.asType(ir::TypeKind::Array, fieldTypes[i]);
          if (srcArr && dstArr &&
              emitter_.elementType(srcArr) == emitter_.elementType(dstArr) &&
              this->arraySize(srcArr) != this->arraySize(dstArr)) {
            elemVal = emitter_.reshape(loc, dstArr, elemVal);
          } else if (auto unionClassType =
                         emitter_.asType(ir::TypeKind::Class, fieldTypes[i]);
                     unionClassType && isUnionClassType(unionClassType)) {
            elemVal = emitter_.bitcast(loc, unionClassType, elemVal);
          } else {
            bool isZeroVal = emitter_.isZeroConstant(elemVal);
            if (isZeroVal) continue;

            if (auto intSrc = emitter_.asType(ir::TypeKind::Integer,
                                              emitter_.typeOf(elemVal))) {
              if (auto intDst =
                      emitter_.asType(ir::TypeKind::Integer, fieldTypes[i])) {
                if (emitter_.scalarWidth(intSrc) > emitter_.scalarWidth(intDst))
                  elemVal = emitter_.truncate(loc, elemVal, intDst);
                else if (emitter_.scalarWidth(intSrc) <
                         emitter_.scalarWidth(intDst))
                  elemVal = emitter_.zeroExtend(loc, elemVal, intDst);
              }
            }
          }
        }
        result = emitter_.insertValue(loc, irType, result, elemVal,
                                      static_cast<int64_t>(i));
      }
      return result;
    }

    return emitter_.zero(loc, irType);
  }

  if (traits.is_array(type)) {
    auto irType = convertType(type);
    auto cxxArrType = emitter_.asType(ir::TypeKind::Array, irType);

    if (auto strLitPtr = std::get_if<const StringLiteral*>(&value)) {
      auto stringLiteral = *strLitPtr;
      stringLiteral->initialize(stringLiteral->encoding());
      std::string str(stringLiteral->stringValue());
      str.push_back('\0');
      auto destSize =
          cxxArrType ? (size_t)this->arraySize(cxxArrType) : str.size();
      str.resize(destSize, '\0');
      auto i8Type = emitter_.integerType(8);
      ir::ValueRef result = emitter_.undef(loc, irType);
      for (size_t i = 0; i < str.size(); ++i) {
        auto elem = emitter_.constantLiteral(
            loc, i8Type,
            ir::Initializer::integerValue(i8Type, (unsigned char)str[i]));
        result = emitter_.insertValue(loc, irType, result, elem,
                                      static_cast<int64_t>(i));
      }
      return result;
    }

    if (auto initListPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto& initList = *initListPtr;
      ir::ValueRef result = emitter_.zero(loc, irType);
      for (size_t i = 0; i < initList->elements.size(); ++i) {
        auto& [elemValue, elemType] = initList->elements[i];
        auto elemVal = emitConstInitValue(loc, elemType, elemValue);
        if (cxxArrType) {
          auto dstElemType = emitter_.elementType(cxxArrType);
          if (emitter_.typeOf(elemVal) != dstElemType) {
            auto srcArr =
                emitter_.asType(ir::TypeKind::Array, emitter_.typeOf(elemVal));
            auto dstArr = emitter_.asType(ir::TypeKind::Array, dstElemType);
            if (srcArr && dstArr &&
                emitter_.elementType(srcArr) == emitter_.elementType(dstArr) &&
                this->arraySize(srcArr) != this->arraySize(dstArr)) {
              elemVal = emitter_.reshape(loc, dstArr, elemVal);
            }
          }
        }
        result = emitter_.insertValue(loc, irType, result, elemVal,
                                      static_cast<int64_t>(i));
      }
      return result;
    }
    return emitter_.zero(loc, irType);
  }

  auto irType = convertType(type);
  return emitter_.zero(loc, irType);
}

void Codegen::branch(SourceLocation loc, ir::BlockRef block,
                     std::vector<ir::ValueRef> operands) {
  if (currentBlockMightHaveTerminator()) return;
  emitter_.branch(loc, block, operands);
}

auto Codegen::findOrCreateLocal(Symbol* symbol) -> std::optional<ir::ValueRef> {
  if (auto local = locals_.find(symbol); local != locals_.end()) {
    return local->second;
  }

  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return std::nullopt;

  if (var->isStatic()) return std::nullopt;
  if (!var->parent()->isBlock()) return std::nullopt;

  auto loc = var->location();

  if (auto vlaType = type_cast<UnresolvedBoundedArrayType>(var->type())) {
    auto countResult = expression(vlaType->size());
    if (!countResult.value) return std::nullopt;

    auto countVal = countResult.value;
    if (emitter_.typeKind(emitter_.typeOf(countVal)) == ir::TypeKind::Pointer) {
      auto valueType = convertType(vlaType->size()->type);
      countVal = emitter_.load(loc, valueType, countVal,
                               getAlignment(vlaType->size()->type));
    }

    auto totalElements = countVal;
    const Type* elemType = vlaType->elementType();

    while (auto inner = type_cast<UnresolvedBoundedArrayType>(elemType)) {
      auto innerResult = expression(inner->size());
      if (!innerResult.value) return std::nullopt;
      auto innerVal = innerResult.value;
      if (emitter_.typeKind(emitter_.typeOf(innerVal)) ==
          ir::TypeKind::Pointer) {
        auto valueType = convertType(inner->size()->type);
        innerVal = emitter_.load(loc, valueType, innerVal,
                                 getAlignment(inner->size()->type));
      }
      const auto countType = emitter_.typeOf(totalElements);
      if (emitter_.typeOf(innerVal) != countType)
        innerVal = emitter_.signExtend(loc, innerVal, countType);
      totalElements =
          emitter_.binaryOp(loc, ir::BinaryOp::MulInt, totalElements, innerVal);
      elemType = inner->elementType();
    }

    auto elementType = convertType(elemType);
    auto ptrType = emitter_.pointerType(elementType);
    auto alignment = getAlignment(elemType);

    auto leafSizeBytes = static_cast<int64_t>(
        control()->memoryLayout()->sizeOf(elemType).value_or(1));
    auto totalBytes = totalElements;
    if (leafSizeBytes > 1) {
      auto countType = emitter_.typeOf(totalElements);
      auto sizeConst = emitter_.constantInt(loc, countType, leafSizeBytes);
      totalBytes = emitter_.binaryOp(loc, ir::BinaryOp::MulInt, totalElements,
                                     sizeConst);
    }

    auto allocaOp =
        emitter_.dynamicAllocate(loc, ptrType, totalBytes, alignment);
    auto address = allocaOp;
    locals_.emplace(var, address);
    return address;
  }

  auto type = convertType(var->type());
  auto ptrType = emitter_.pointerType(type);

  auto allocaOp = emitter_.allocate(loc, ptrType, getAlignment(var));

  attachDebugInfo(allocaOp, var);

  auto address = allocaOp;
  locals_.emplace(var, address);

  return address;
}

auto Codegen::newTemp(const Type* type, SourceLocation loc) -> ir::ValueRef {
  return emitter_.allocate(loc, emitter_.pointerType(convertType(type)),
                           getAlignment(type));
}

void Codegen::pushCleanup() { cleanupStack_.emplace_back(); }

void Codegen::pushFullExpressionCleanup() {
  auto& scope = cleanupStack_.emplace_back();
  scope.isFullExpression = true;
  scope.region = emitter_.beginCleanupRegion();
}

void Codegen::popCleanup(SourceLocation loc) {
  auto& scope = cleanupStack_.back();
  if (scope.entries.empty() || currentBlockMightHaveTerminator()) {
    emitter_.endCleanupRegion(scope.region);
    cleanupStack_.pop_back();
    return;
  }
  auto mergeBlock = newBlock();
  emitBranchWithCleanups(loc, mergeBlock, cleanupStack_.size() - 1);
  emitter_.setInsertionBlock(mergeBlock);
  emitter_.endCleanupRegion(scope.region);
  cleanupStack_.pop_back();
}

void Codegen::emitBranchWithCleanups(SourceLocation loc, ir::BlockRef target,
                                     std::size_t targetDepth) {
  if (currentBlockMightHaveTerminator()) return;

  auto snapshot = collectCleanupSnapshot(targetDepth);

  if (snapshot.empty()) {
    emitter_.branch(loc, target);
    return;
  }

  emitter_.branchWithCleanups(implicitLocation(loc), target, snapshot);
}

void Codegen::addCleanup(ir::ValueRef address, FunctionSymbol* dtor,
                         std::int64_t elementCount) {
  for (auto i = cleanupStack_.size(); i > 0; --i) {
    auto& scope = cleanupStack_[i - 1];
    if (scope.isFullExpression) continue;
    scope.entries.push_back(
        {.address = address, .destructor = dtor, .elementCount = elementCount});
    return;
  }
}

auto Codegen::objectDestructor(const Type* type) const -> FunctionSymbol* {
  auto shape = classSubobjectShape(type);
  if (!shape) return nullptr;
  return shape->classSymbol->destructor();
}

void Codegen::addArrayCleanup(ir::ValueRef address, const Type* arrayType) {
  auto shape = classSubobjectShape(arrayType);
  if (!shape) return;
  if (traits.has_trivial_destructor(shape->elementType)) return;

  auto dtor = objectDestructor(arrayType);
  if (!dtor) return;

  addCleanup(address, completeObjectDtor(dtor),
             static_cast<std::int64_t>(shape->elementCount));
}

void Codegen::addTemporaryCleanup(ir::ValueRef address, const Type* type) {
  if (cleanupStack_.empty() || !cleanupStack_.back().isFullExpression) return;
  if (traits.has_trivial_destructor(type)) return;
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return;
  auto dtor = classType->symbol()->resolvedDefinition()->destructor();
  if (!dtor) return;

  auto& scope = cleanupStack_.back();

  ir::ValueRef activeFlag;

  if (conditionalEvaluationDepth_ > 0) {
    activeFlag =
        emitter_.activateConditionalCleanup(address, entryBlock_, scope.region);
  }

  scope.entries.push_back({address, completeObjectDtor(dtor), activeFlag});
}

void Codegen::cancelCleanup(ir::ValueRef address) {
  if (!address) return;

  for (auto i = cleanupStack_.size(); i > 0; --i) {
    auto& entries = cleanupStack_[i - 1].entries;
    auto found =
        std::ranges::find(entries, address, &CleanupScope::Entry::address);
    if (found == entries.end()) continue;
    entries.erase(found);
    return;
  }
}

auto Codegen::loadThisPointer(SourceLocation loc, ClassSymbol* classSymbol)
    -> ir::ValueRef {
  if (!thisValue_ || !classSymbol) return {};

  auto pointerType = control()->getPointerType(classSymbol->type());

  return emitter_.load(loc,
                       emitter_.pointerType(convertType(classSymbol->type())),
                       thisValue_, getAlignment(pointerType));
}

auto Codegen::loadEnclosingObject(SourceLocation loc, ClassSymbol* targetClass,
                                  ClassSymbol*& objectClass) -> ir::ValueRef {
  objectClass = targetClass;
  if (currentFunctionSymbol_) {
    if (auto enclosing =
            symbol_cast<ClassSymbol>(currentFunctionSymbol_->parent())) {
      objectClass = enclosing;
    }
  }

  auto object = loadThisPointer(loc, objectClass);
  if (!object) return object;

  while (objectClass != targetClass && objectClass->isClosureType()) {
    auto capturedThis = objectClass->capturedThisField();
    if (!capturedThis) break;

    auto layout = objectClass->layout();
    auto fieldInfo = layout ? layout->getFieldInfo(capturedThis) : std::nullopt;

    auto enclosingType = unqualified_cast<ClassType>(
        traits.get_element_type(traits.remove_cv(capturedThis->type())));

    if (!fieldInfo || !enclosingType || !enclosingType->symbol()) {
      cxx_runtime_error(std::format(
          "closure capturing 'this' has no usable '__this' field for '{}'",
          to_string(targetClass->type())));
    }

    auto fieldAddress =
        memberAddress(loc, object, capturedThis->type(), fieldInfo->index);

    objectClass = enclosingType->symbol()->resolvedDefinition();

    object =
        emitter_.load(loc, emitter_.pointerType(convertType(enclosingType)),
                      fieldAddress, getAlignment(capturedThis->type()));
  }

  return object;
}

auto Codegen::classSubobjectShape(const Type* type) const
    -> std::optional<ClassSubobjectShape> {
  if (!type) return std::nullopt;

  ClassSubobjectShape shape;

  auto elementType = traits.remove_cv(type);
  while (traits.is_array(elementType)) {
    auto arrayType = type_cast<BoundedArrayType>(elementType);
    if (!arrayType) return std::nullopt;
    shape.elementCount *= arrayType->size();
    elementType = traits.remove_cv(arrayType->elementType());
  }

  auto classType = type_cast<ClassType>(elementType);
  if (!classType || !classType->symbol()) return std::nullopt;

  shape.classSymbol = classType->symbol()->resolvedDefinition();
  shape.elementType = elementType;
  return shape;
}

auto Codegen::subobjectType(Symbol* subobject) const -> const Type* {
  if (auto field = symbol_cast<FieldSymbol>(subobject)) return field->type();
  if (auto base = symbol_cast<BaseClassSymbol>(subobject))
    return base->symbol() ? base->symbol()->type() : nullptr;
  return nullptr;
}

auto Codegen::subobjectIndex(ClassSymbol* classSymbol, Symbol* subobject) const
    -> std::optional<int> {
  auto layout = classSymbol->layout();

  if (auto field = symbol_cast<FieldSymbol>(subobject)) {
    if (layout) {
      if (auto fi = layout->getFieldInfo(field)) return fi->index;
    }
  } else if (auto base = symbol_cast<BaseClassSymbol>(subobject)) {
    if (layout) {
      if (auto baseSym = symbol_cast<ClassSymbol>(base->symbol())) {
        if (auto bi = layout->getBaseInfo(baseSym, base->isVirtual()))
          return bi->index;
      }
    }
  }

  int index = 0;
  for (auto base : classSymbol->baseClasses()) {
    if (base == subobject) return index;
    ++index;
  }
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field == subobject) return index;
    ++index;
  }
  return std::nullopt;
}

auto Codegen::subobjectAddress(SourceLocation loc, ir::ValueRef objectPtr,
                               ClassSymbol* classSymbol, Symbol* subobject)
    -> ir::ValueRef {
  if (auto base = symbol_cast<BaseClassSymbol>(subobject)) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    auto layout = classSymbol->layout();
    if (!baseClass || !layout) return {};
    auto info = layout->getBaseInfo(baseClass, base->isVirtual());
    if (!info) return {};
    return subobjectAddress(loc, objectPtr, baseClass, info->offset);
  }

  auto type = subobjectType(subobject);
  if (!type) return {};

  auto declaringClass = symbol_cast<ClassSymbol>(subobject->parent());
  if (declaringClass) declaringClass = declaringClass->resolvedDefinition();

  const auto declaredHere = !declaringClass || declaringClass == classSymbol ||
                            declaringClass == classSymbol->definition();

  if (!declaredHere && !declaringClass->name()) {
    auto enclosingClass = symbol_cast<ClassSymbol>(declaringClass->parent());
    if (!enclosingClass) return {};
    enclosingClass = enclosingClass->resolvedDefinition();

    for (auto field :
         views::members(enclosingClass) | views::non_static_fields) {
      auto fieldClass = unqualified_cast<ClassType>(field->type());
      if (!fieldClass || !fieldClass->symbol()) continue;
      if (fieldClass->symbol()->resolvedDefinition() != declaringClass)
        continue;

      auto enclosing = subobjectAddress(loc, objectPtr, classSymbol, field);
      if (!enclosing) return {};

      auto index = subobjectIndex(declaringClass, subobject);
      if (!index) return {};
      return memberAddress(loc, enclosing, type, *index);
    }

    return {};
  }

  auto index = subobjectIndex(classSymbol, subobject);
  if (!index) return {};
  return memberAddress(loc, objectPtr, type, *index);
}

void Codegen::forEachSubobjectElement(
    SourceLocation loc, ir::ValueRef subobjectPtr,
    const ClassSubobjectShape& shape, bool reverse,
    const std::function<void(ir::ValueRef)>& body) {
  if (shape.elementCount == 1) {
    body(subobjectPtr);
    return;
  }

  auto count =
      emitter_.constantInt(loc, convertType(control()->getSizeType()),
                           static_cast<std::int64_t>(shape.elementCount));
  emitArrayLoop(loc, subobjectPtr, shape.elementType, count, reverse, body);
}

auto Codegen::subobjectsInDeclarationOrder(ClassSymbol* classSymbol) const
    -> std::vector<Symbol*> {
  std::vector<Symbol*> subobjects;

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    subobjects.push_back(base);
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields)
    subobjects.push_back(field);

  return subobjects;
}

auto Codegen::isImplicitlyInitializedSubobject(ClassSymbol* classSymbol,
                                               Symbol* subobject) const
    -> bool {
  if (!classSymbol->isUnion()) return true;
  return !symbol_cast<FieldSymbol>(subobject);
}

auto Codegen::defaultConstructorArguments(FunctionSymbol* constructor)
    -> std::vector<ExpressionResult> {
  std::vector<ExpressionResult> args;

  for (auto parameter : constructor->parameters()) {
    auto defaultArgument =
        ASTRewriter::requireDefaultArgument(unit_, parameter);
    if (!defaultArgument) continue;
    args.push_back(expression(defaultArgument));
  }

  return args;
}

void Codegen::emitSubobjectDestruction(SourceLocation loc,
                                       ir::ValueRef objectPtr,
                                       ClassSymbol* classSymbol,
                                       Symbol* subobject) {
  if (!isImplicitlyInitializedSubobject(classSymbol, subobject)) return;

  auto shape = classSubobjectShape(subobjectType(subobject));
  if (!shape) return;

  auto dtor = shape->classSymbol->destructor();
  if (!dtor) return;

  const auto isField = symbol_cast<FieldSymbol>(subobject) != nullptr;
  if (isField) dtor = completeObjectDtor(dtor);

  auto subobjectPtr = subobjectAddress(loc, objectPtr, classSymbol, subobject);
  if (!subobjectPtr) return;

  forEachSubobjectElement(
      loc, subobjectPtr, *shape, /*reverse=*/true, [&](ir::ValueRef element) {
        (void)emitCall(loc, dtor, {element}, {}, /*isVirtualDispatch=*/false,
                       /*resultOwner=*/nullptr,
                       /*baseObjectStructor=*/!isField);
      });
}

void Codegen::emitSubobjectDefaultConstruction(SourceLocation loc,
                                               ir::ValueRef objectPtr,
                                               ClassSymbol* classSymbol,
                                               Symbol* subobject) {
  if (!isImplicitlyInitializedSubobject(classSymbol, subobject)) return;

  auto shape = classSubobjectShape(subobjectType(subobject));
  if (!shape) return;

  auto defaultConstructor = shape->classSymbol->defaultConstructor();
  if (!defaultConstructor) return;

  auto subobjectPtr = subobjectAddress(loc, objectPtr, classSymbol, subobject);
  if (!subobjectPtr) return;

  const auto completeObject = symbol_cast<FieldSymbol>(subobject) != nullptr;

  forEachSubobjectElement(
      loc, subobjectPtr, *shape, /*reverse=*/false, [&](ir::ValueRef element) {
        (void)emitCtorCall(loc, defaultConstructor, element,
                           defaultConstructorArguments(defaultConstructor),
                           completeObject);
      });
}

Codegen::FullExpression::FullExpression(Codegen& gen, SourceLocation endLoc)
    : gen_(gen), endLoc_(endLoc) {
  gen_.pushFullExpressionCleanup();
}

Codegen::FullExpression::~FullExpression() { gen_.popCleanup(endLoc_); }

auto Codegen::takeResultObject(ExpressionAST* ast) -> ir::ValueRef {
  if (!ast || resultObjectOwner_ != ast) return {};
  resultObjectOwner_ = nullptr;
  resultObjectInitialized_ = true;
  return std::exchange(resultObjectAddress_, ir::ValueRef{});
}

auto Codegen::emitIntoResultObject(ExpressionAST* owner, ExpressionAST* operand)
    -> ir::ValueRef {
  auto object = takeResultObject(owner);
  if (!object) return {};
  (void)emitPrvalueInto(object, owner->type, operand,
                        owner->firstSourceLocation());
  return object;
}

auto Codegen::emitAggregateObject(ExpressionAST* owner, const Type* type,
                                  BracedInitListAST* initializer,
                                  SourceLocation loc) -> ir::ValueRef {
  auto object = takeResultObject(owner);
  const bool ownsTemporary = !object;
  if (ownsTemporary) object = newTemp(type, loc);
  if (initializer) emitAggregateInit(object, type, initializer);
  if (ownsTemporary) addTemporaryCleanup(object, type);
  return object;
}

auto Codegen::takeIndirectResultObject(ExpressionAST* ast,
                                       const FunctionType* functionType)
    -> ir::ValueRef {
  if (!ast || !functionType) return {};

  const auto abi = classifyClassValueAbi(functionType->returnType(),
                                         ClassValueAbiContext::Return);

  if (abi.kind != ClassValueAbi::Kind::Indirect) return {};

  return takeResultObject(ast);
}

auto Codegen::emitPrvalueInto(ir::ValueRef object, const Type* objectType,
                              ExpressionAST* ast, SourceLocation loc) -> bool {
  if (traits.is_reference(objectType)) {
    auto result = expression(ast);
    if (!result.value) return false;
    emitter_.store(loc, result.value, object, getAlignment(objectType));
    return true;
  }

  auto resultObject = ResultObject{*this, ast, object};

  auto result = expression(ast);

  if (resultObject.wasConsumed()) return true;
  if (!result.value) return false;

  auto objectIrType = convertType(objectType);
  auto value = result.value;

  const bool yieldedClassAddress =
      traits.is_class(objectType) && emitter_.typeOf(value) != objectIrType &&
      (emitter_.typeKind(emitter_.typeOf(value)) == ir::TypeKind::Pointer);

  if (yieldedClassAddress) {
    value = emitter_.load(loc, objectIrType, value, getAlignment(objectType));
  }

  emitter_.store(loc, value, object, getAlignment(objectType));

  return true;
}

Codegen::ResultObject::ResultObject(Codegen& gen, ExpressionAST* ast,
                                    ir::ValueRef address)
    : gen_(gen),
      savedOwner_(std::exchange(gen.resultObjectOwner_, ast)),
      savedAddress_(std::exchange(gen.resultObjectAddress_, address)),
      savedInitialized_(std::exchange(gen.resultObjectInitialized_, false)) {}

Codegen::ResultObject::~ResultObject() {
  gen_.resultObjectOwner_ = savedOwner_;
  gen_.resultObjectAddress_ = savedAddress_;
  gen_.resultObjectInitialized_ = savedInitialized_;
}

auto Codegen::ResultObject::wasConsumed() const -> bool {
  return gen_.resultObjectInitialized_;
}

auto Codegen::collectCleanupSnapshot(std::size_t targetDepth)
    -> CleanupSnapshot {
  CleanupSnapshot snapshot;

  for (auto i = cleanupStack_.size(); i > targetDepth; --i) {
    auto& scope = cleanupStack_[i - 1];
    for (auto jt = scope.entries.rbegin(); jt != scope.entries.rend(); ++jt) {
      auto destructor = findOrCreateFunction(jt->destructor);
      snapshot.push_back({jt->address, destructor,
                          static_cast<std::int64_t>(i - 1), jt->activeFlag,
                          jt->elementCount});
    }
  }

  return snapshot;
}

auto Codegen::structorReturnsThis(FunctionSymbol* symbol) -> bool {
  if (!symbol) return false;
  if (!symbol->isConstructor() && !name_cast<DestructorId>(symbol->name())) {
    return false;
  }
  if (symbol->isDeletingDtorVariant()) return false;
  return control()->memoryLayout()->structorsReturnThis();
}

auto Codegen::classifyClassValueAbi(const Type* type,
                                    ClassValueAbiContext context)
    -> ClassValueAbi {
  return cxx::classifyClassValueAbi(unit_, type, context);
}

auto Codegen::getSize(const Type* type) -> std::uint64_t {
  return control()->memoryLayout()->sizeOf(type).value_or(0);
}

auto Codegen::abiSlotAddress(SourceLocation loc, ir::ValueRef address,
                             const ClassValueAbiSlot& slot) -> ir::ValueRef {
  auto slotPtrType = emitter_.pointerType(convertType(slot.type));
  if (!slot.offset) return emitter_.bitcast(loc, slotPtrType, address);

  auto byteType = emitter_.integerType(8);
  auto bytePtrType = emitter_.pointerType(byteType);
  auto base = emitter_.bitcast(loc, bytePtrType, address);
  auto offset = emitter_.constantInt(loc, emitter_.integerType(64),
                                     static_cast<std::int64_t>(slot.offset));

  return emitter_.bitcast(loc, slotPtrType,
                          emitter_.pointerAdd(loc, bytePtrType, base, offset));
}

auto Codegen::abiCoerceStorage(SourceLocation loc, const Type* valueType,
                               const ClassValueAbi& abi, ir::ValueRef address)
    -> ir::ValueRef {
  if (getSize(valueType) >= abi.coerceSize) return address;

  auto byteType = emitter_.integerType(8);
  auto storageType = this->arrayType(byteType, abi.coerceSize);

  return emitter_.allocate(loc, emitter_.pointerType(storageType),
                           abi.coerceAlignment);
}

void Codegen::abiLoadClassValue(SourceLocation loc, const Type* valueType,
                                const ClassValueAbi& abi, ir::ValueRef address,
                                std::vector<ir::ValueRef>& values) {
  auto storage = abiCoerceStorage(loc, valueType, abi, address);
  if (storage != address)
    emitter_.memcpy(loc, storage, address, getSize(valueType));

  for (const auto& slot : abi.slots) {
    values.push_back(emitter_.load(loc, convertType(slot.type),
                                   abiSlotAddress(loc, storage, slot),
                                   getAlignment(slot.type)));
  }
}

void Codegen::abiStoreClassValue(SourceLocation loc, const Type* valueType,
                                 const ClassValueAbi& abi,
                                 std::span<const ir::ValueRef> values,
                                 ir::ValueRef address) {
  auto storage = abiCoerceStorage(loc, valueType, abi, address);

  for (std::size_t i = 0; i < values.size() && i < abi.slots.size(); ++i) {
    emitter_.store(loc, values[i], abiSlotAddress(loc, storage, abi.slots[i]),
                   getAlignment(abi.slots[i].type));
  }

  if (storage != address)
    emitter_.memcpy(loc, address, storage, getSize(valueType));
}

auto Codegen::hasNoValueRepresentation(const Type* type) -> bool {
  return classifyClassValueAbi(type, ClassValueAbiContext::Argument).kind ==
         ClassValueAbi::Kind::Empty;
}

auto Codegen::classValueAddress(SourceLocation loc, const Type* type,
                                ir::ValueRef value) -> ir::ValueRef {
  if ((emitter_.typeKind(emitter_.typeOf(value)) == ir::TypeKind::Pointer))
    return value;
  auto temp = newTemp(type, loc);
  emitter_.store(loc, value, temp, getAlignment(type));
  return temp;
}

auto Codegen::classValueLoad(SourceLocation loc, const Type* type,
                             ir::ValueRef value) -> ir::ValueRef {
  auto expectedType = convertType(type);
  if (emitter_.typeOf(value) == expectedType) return value;
  if (emitter_.typeKind(emitter_.typeOf(value)) != ir::TypeKind::Pointer)
    return value;
  return emitter_.load(loc, expectedType, value, getAlignment(type));
}

void Codegen::abiLowerClassArgument(SourceLocation loc, const Type* paramType,
                                    ir::ValueRef value,
                                    std::vector<ir::ValueRef>& args) {
  const auto abi =
      classifyClassValueAbi(paramType, ClassValueAbiContext::Argument);

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct:
      args.push_back(classValueLoad(loc, paramType, value));
      break;

    case ClassValueAbi::Kind::Empty:
      break;

    case ClassValueAbi::Kind::Indirect:
      args.push_back(classValueAddress(loc, paramType, value));
      break;

    case ClassValueAbi::Kind::Coerce:
      abiLoadClassValue(loc, paramType, abi,
                        classValueAddress(loc, paramType, value), args);
      break;
  }
}

auto Codegen::abiPrepareResult(SourceLocation loc, const Type* returnType,
                               std::vector<ir::TypeRef>& resultTypes,
                               ir::ValueRef resultObject) -> ir::ValueRef {
  if (traits.is_void(returnType)) return {};

  const auto abi =
      classifyClassValueAbi(returnType, ClassValueAbiContext::Return);

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct:
      resultTypes.push_back(convertType(returnType));
      return {};

    case ClassValueAbi::Kind::Coerce:
      for (const auto& slot : abi.slots)
        resultTypes.push_back(convertType(slot.type));
      return {};

    case ClassValueAbi::Kind::Empty:
      return {};

    case ClassValueAbi::Kind::Indirect:
      if (resultObject) return resultObject;
      return newTemp(returnType, loc);
  }

  return {};
}

auto Codegen::abiFinishResult(SourceLocation loc, const Type* returnType,
                              std::span<const ir::ValueRef> callResults,
                              ir::ValueRef sretTemp) -> ExpressionResult {
  if (traits.is_void(returnType)) return {};

  const auto abi =
      classifyClassValueAbi(returnType, ClassValueAbiContext::Return);

  auto loadFrom = [&](ir::ValueRef address) -> ExpressionResult {
    return {emitter_.load(loc, convertType(returnType), address,
                          getAlignment(returnType))};
  };

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct:
      return {ir::singleValue(callResults)};

    case ClassValueAbi::Kind::Indirect:
      return {sretTemp};

    case ClassValueAbi::Kind::Coerce: {
      auto temp = newTemp(returnType, loc);
      abiStoreClassValue(loc, returnType, abi, callResults, temp);
      return loadFrom(temp);
    }

    case ClassValueAbi::Kind::Empty:
      return loadFrom(newTemp(returnType, loc));
  }

  return {};
}

namespace {
auto classDefinition(ClassSymbol* classSymbol) -> ClassSymbol* {
  return classSymbol ? classSymbol->resolvedDefinition() : nullptr;
}

auto findBaseClassPath(ClassSymbol* from, ClassSymbol* target,
                       std::vector<BaseClassSymbol*>& path) -> bool {
  for (auto base : from->baseClasses()) {
    auto baseClass = classDefinition(symbol_cast<ClassSymbol>(base->symbol()));
    if (!baseClass) continue;
    path.push_back(base);
    if (baseClass == target) return true;
    if (findBaseClassPath(baseClass, target, path)) return true;
    path.pop_back();
  }
  return false;
}
}  // namespace

auto Codegen::emitVirtualBaseAddress(SourceLocation loc, ir::ValueRef objectPtr,
                                     ClassSymbol* fromClass,
                                     ClassSymbol* vbaseClass) -> ir::ValueRef {
  std::int64_t words = 0;
  if (auto fromVTable = fromClass->vtableLayout())
    words = fromVTable->primary().offsetWordsBeforeAddressPoint(vbaseClass);

  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  const auto slotByteOffset = -wordSize * words;

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  auto objectI8 = emitter_.bitcast(loc, i8PtrType, objectPtr);
  auto adjusted = adjustByVtableWord(loc, objectI8, slotByteOffset);

  auto vbasePtrType = emitter_.pointerType(convertType(vbaseClass->type()));
  return emitter_.bitcast(loc, vbasePtrType, adjusted);
}

auto Codegen::adjustByVtableWord(SourceLocation loc, ir::ValueRef objectPtrI8,
                                 std::int64_t byteOffset) -> ir::ValueRef {
  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto i8PtrPtrType = emitter_.pointerType(i8PtrType);

  const auto wordSize = pointerSize();
  auto wordType = pointerSizedIntType();

  auto vptr =
      emitter_.load(loc, i8PtrType, vptrAddress(loc, objectPtrI8), wordSize);

  auto offsetConstOp = emitter_.constantInt(loc, wordType, byteOffset);
  auto slotAddr = emitter_.pointerAdd(loc, i8PtrType, vptr, offsetConstOp);
  auto wordPtrType = emitter_.pointerType(wordType);
  auto slotPtr = emitter_.bitcast(loc, wordPtrType, slotAddr);
  auto word = emitter_.load(loc, wordType, slotPtr, wordSize);

  return emitter_.pointerAdd(loc, i8PtrType, objectPtrI8, word);
}

auto Codegen::emitBaseClassAddress(SourceLocation loc, ir::ValueRef objectPtr,
                                   ClassSymbol* fromClass,
                                   ClassSymbol* targetClass) -> ir::ValueRef {
  if (!objectPtr || !fromClass || !targetClass) return objectPtr;
  if (!(emitter_.typeKind(emitter_.typeOf(objectPtr)) == ir::TypeKind::Pointer))
    return objectPtr;

  fromClass = classDefinition(fromClass);
  targetClass = classDefinition(targetClass);
  if (fromClass == targetClass) return objectPtr;

  std::vector<BaseClassSymbol*> path;
  if (!findBaseClassPath(fromClass, targetClass, path)) return objectPtr;

  auto current = objectPtr;
  auto currentClass = fromClass;

  for (auto step : path) {
    auto baseClass = classDefinition(symbol_cast<ClassSymbol>(step->symbol()));

    if (step->isVirtual()) {
      current = emitVirtualBaseAddress(loc, current, currentClass, baseClass);
    } else {
      std::uint64_t offset = 0;
      if (auto layout = currentClass->layout()) {
        if (auto baseInfo = layout->getBaseInfo(baseClass)) {
          offset = baseInfo->offset;
        }
      }
      current = subobjectAddress(loc, current, baseClass, offset);
    }

    currentClass = baseClass;
  }

  return current;
}

auto Codegen::emitDerivedClassAddress(SourceLocation loc,
                                      ir::ValueRef objectPtr,
                                      ClassSymbol* fromClass,
                                      ClassSymbol* targetClass)
    -> ir::ValueRef {
  if (!objectPtr || !fromClass || !targetClass) return objectPtr;
  if (!(emitter_.typeKind(emitter_.typeOf(objectPtr)) == ir::TypeKind::Pointer))
    return objectPtr;

  fromClass = classDefinition(fromClass);
  targetClass = classDefinition(targetClass);
  if (fromClass == targetClass) return objectPtr;

  std::vector<BaseClassSymbol*> path;
  if (!findBaseClassPath(targetClass, fromClass, path)) return objectPtr;

  std::int64_t byteOffset = 0;
  auto currentClass = targetClass;

  for (auto step : path) {
    auto baseClass = classDefinition(symbol_cast<ClassSymbol>(step->symbol()));
    if (step->isVirtual())
      cxx_runtime_error("base-to-derived conversion through a virtual base");

    if (auto layout = currentClass->layout()) {
      if (auto baseInfo = layout->getBaseInfo(baseClass))
        byteOffset += static_cast<std::int64_t>(baseInfo->offset);
    }

    currentClass = baseClass;
  }

  auto derivedPtrType = emitter_.pointerType(convertType(targetClass->type()));

  if (byteOffset == 0) {
    return emitter_.bitcast(loc, derivedPtrType, objectPtr);
  }

  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  auto wordType = emitter_.integerType(static_cast<unsigned>(wordSize * 8));

  auto objectI8 = emitter_.bitcast(loc, i8PtrType, objectPtr);
  auto offsetOp = emitter_.constantInt(loc, wordType, -byteOffset);
  auto adjusted = emitter_.pointerAdd(loc, i8PtrType, objectI8, offsetOp);

  return emitter_.bitcast(loc, derivedPtrType, adjusted);
}

auto Codegen::makeMemberFunctionPointer(
    SourceLocation loc, const MemberFunctionPointerType* pointerType,
    ir::ValueRef pointerField, ir::ValueRef adjustmentField) -> ir::ValueRef {
  auto representation = convertType(pointerType);

  ir::ValueRef value = emitter_.zero(loc, representation);
  value = emitter_.insertValue(loc, representation, value, pointerField,
                               static_cast<std::int64_t>(0));
  return emitter_.insertValue(loc, representation, value, adjustmentField,
                              static_cast<std::int64_t>(1));
}

auto Codegen::memberFunctionPointerFields(
    SourceLocation loc, const MemberFunctionPointerType* pointerType,
    ir::ValueRef value) -> std::pair<ir::ValueRef, ir::ValueRef> {
  auto wordType = pointerSizedIntType();
  const auto alignment = static_cast<int>(pointerSize());

  auto address = value;
  if (!(emitter_.typeKind(emitter_.typeOf(address)) == ir::TypeKind::Pointer)) {
    auto ptrType = emitter_.pointerType(convertType(pointerType));
    auto temp = emitter_.allocate(loc, ptrType, alignment);
    emitter_.store(loc, value, temp, alignment);
    address = temp;
  }

  auto load = [&](std::uint32_t index) -> ir::ValueRef {
    auto fieldPtrType = emitter_.pointerType(wordType);
    auto fieldPtr = emitter_.memberAddress(loc, fieldPtrType, address, index);
    return emitter_.load(loc, wordType, fieldPtr, alignment);
  };

  return {load(0), load(1)};
}

auto Codegen::emitMemberFunctionPointerValue(
    SourceLocation loc, const MemberFunctionPointerType* pointerType,
    FunctionSymbol* function, std::int64_t adjustmentBytes) -> ir::ValueRef {
  auto wordType = pointerSizedIntType();

  auto constantWord = [&](std::int64_t word) -> ir::ValueRef {
    return emitter_.constantLiteral(
        loc, wordType, ir::Initializer::integerValue(wordType, word));
  };

  auto adjustmentField = adjustmentBytes * 2;
  ir::ValueRef pointerField;

  if (!function) {
    pointerField = constantWord(0);
    adjustmentField = 0;
  } else if (function->isVirtual()) {
    adjustmentField |= 1;
    pointerField = constantWord(vtableSlotIndex(function) * pointerSize());
  } else {
    auto funcOp = findOrCreateFunction(function);
    auto funcPtrType = convertType(control()->getPointerType(function->type()));
    auto funcAddress =
        emitter_.addressOfSymbol(loc, funcPtrType, this->functionName(funcOp));
    pointerField = emitter_.pointerToInt(loc, wordType, funcAddress);
  }

  return makeMemberFunctionPointer(loc, pointerType, pointerField,
                                   constantWord(adjustmentField));
}

auto Codegen::computeFunctionSignature(FunctionSymbol* functionSymbol)
    -> ir::TypeRef {
  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  if (!functionType) return {};
  return computeFunctionSignature(functionType, functionSymbol);
}

auto Codegen::computeFunctionSignature(const FunctionType* functionType,
                                       FunctionSymbol* functionSymbol)
    -> ir::TypeRef {
  return computeFunctionAbi(functionType, functionSymbol).signature;
}

auto Codegen::computeParameterAbi(const FunctionType* functionType,
                                  FunctionSymbol* functionSymbol)
    -> std::vector<ir::ParameterAbi> {
  return computeFunctionAbi(functionType, functionSymbol).parameters;
}

auto Codegen::computeFunctionAbi(const FunctionType* functionType,
                                 FunctionSymbol* functionSymbol)
    -> FunctionAbi {
  FunctionAbi abi;

  const auto returnType = functionType->returnType();
  const auto returnAbi =
      classifyClassValueAbi(returnType, ClassValueAbiContext::Return);

  std::vector<ir::TypeRef> inputTypes;
  std::vector<ir::TypeRef> resultTypes;

  bool hasParameterAttribute = false;

  auto addInput = [&](ir::TypeRef type, ir::ParameterAbi parameter = {}) {
    inputTypes.push_back(type);
    hasParameterAttribute |= parameter.kind != ir::ParameterAbiKind::Default;
    abi.parameters.push_back(parameter);
  };

  const bool returnsThis =
      functionSymbol && structorReturnsThis(functionSymbol);

  if (returnAbi.kind == ClassValueAbi::Kind::Indirect) {
    auto returnIrType = convertType(returnType);
    addInput(emitter_.pointerType(returnIrType),
             returnsThis
                 ? ir::ParameterAbi{}
                 : ir::ParameterAbi{.kind = ir::ParameterAbiKind::StructReturn,
                                    .indirectType = returnIrType,
                                    .alignment = returnAbi.indirectAlignment});
  }

  if (functionSymbol && functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    addInput(emitter_.pointerType(convertType(classSymbol->type())));
  }

  for (auto paramTy : functionType->parameterTypes()) {
    const auto paramAbi =
        classifyClassValueAbi(paramTy, ClassValueAbiContext::Argument);

    switch (paramAbi.kind) {
      case ClassValueAbi::Kind::Direct:
        addInput(convertType(paramTy));
        break;

      case ClassValueAbi::Kind::Empty:
        break;

      case ClassValueAbi::Kind::Coerce:
        for (const auto& slot : paramAbi.slots)
          addInput(convertType(slot.type));
        break;

      case ClassValueAbi::Kind::Indirect: {
        auto paramIrType = convertType(paramTy);
        addInput(emitter_.pointerType(paramIrType),
                 paramAbi.passedInMemory
                     ? ir::ParameterAbi{.kind = ir::ParameterAbiKind::ByValue,
                                        .indirectType = paramIrType,
                                        .alignment = paramAbi.indirectAlignment}
                     : ir::ParameterAbi{});
        break;
      }
    }
  }

  if (functionSymbol &&
      (functionSymbol->isConstructor() || functionSymbol->isDestructor()) &&
      !functionSymbol->isStructorVariant()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (requiresVTT(classSymbol)) {
      auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
      addInput(emitter_.pointerType(i8PtrType));
    }
  }

  if (returnsThis) {
    resultTypes.push_back(inputTypes.front());
  } else if (!traits.is_void(returnType)) {
    switch (returnAbi.kind) {
      case ClassValueAbi::Kind::Direct:
        resultTypes.push_back(convertType(returnType));
        break;
      case ClassValueAbi::Kind::Coerce:
        for (const auto& slot : returnAbi.slots)
          resultTypes.push_back(convertType(slot.type));
        break;
      case ClassValueAbi::Kind::Indirect:
      case ClassValueAbi::Kind::Empty:
        break;
    }
  }

  if (!hasParameterAttribute) abi.parameters.clear();

  abi.signature = emitter_.functionType(inputTypes, resultTypes,
                                        functionType->isVariadic());

  return abi;
}

auto Codegen::implicitLocation(SourceLocation loc) const -> SourceLocation {
  if (loc) return loc;
  if (!currentFunctionSymbol_) return loc;
  return currentFunctionSymbol_->location();
}

auto Codegen::declareFunction(SourceLocation loc, const ir::FunctionInfo& info)
    -> ir::FunctionRef {
  auto function = emitter_.declareFunction(loc, info);
  if (function) {
    functionNames_[function] = info.name;
    functionTypes_[function] = info.type;
  }
  return function;
}

auto Codegen::findFunction(std::string_view name) -> ir::FunctionRef {
  auto function = emitter_.findFunction(name);
  if (function) functionNames_[function] = name;
  return function;
}

auto Codegen::functionName(ir::FunctionRef function) const -> std::string_view {
  auto it = functionNames_.find(function);
  return it == functionNames_.end() ? std::string_view{} : it->second;
}

auto Codegen::functionType(ir::FunctionRef function) const -> ir::TypeRef {
  auto it = functionTypes_.find(function);
  return it == functionTypes_.end() ? ir::TypeRef{} : it->second;
}

auto Codegen::declareGlobal(SourceLocation loc, const ir::GlobalInfo& info)
    -> ir::GlobalRef {
  auto global = emitter_.declareGlobal(loc, info);
  if (global) globalNames_[global] = info.name;
  return global;
}

auto Codegen::findGlobal(std::string_view name) -> ir::GlobalRef {
  auto global = emitter_.findGlobal(name);
  if (global) globalNames_[global] = name;
  return global;
}

auto Codegen::globalName(ir::GlobalRef global) const -> std::string_view {
  auto it = globalNames_.find(global);
  return it == globalNames_.end() ? std::string_view{} : it->second;
}

auto Codegen::emittedFunctionSymbol(FunctionSymbol* functionSymbol)
    -> FunctionSymbol* {
  if (functionSymbol->isSpecialization()) return functionSymbol;
  return functionSymbol->canonical();
}

auto Codegen::findOrCreateFunction(FunctionSymbol* functionSymbol)
    -> ir::FunctionRef {
  auto emittedSymbol = emittedFunctionSymbol(functionSymbol);

  if (auto it = funcOps_.find(emittedSymbol); it != funcOps_.end()) {
    return it->second;
  }

  const auto functionType = type_cast<FunctionType>(emittedSymbol->type());
  if (!functionType) {
    return {};
  }

  auto functionAbi = computeFunctionAbi(functionType, emittedSymbol);

  std::string name;

  if (auto externalName = emittedSymbol->externalName()) {
    name = externalName->name();
  } else if (emittedSymbol->hasCLinkage()) {
    name = to_string(emittedSymbol->name());
  } else {
    ExternalNameEncoder encoder{unit_};
    name = encoder.encode(emittedSymbol);
  }

  auto visibility = emittedSymbol->hasHiddenVisibility()
                        ? ir::Visibility::Hidden
                        : ir::Visibility::Default;

  if (auto existingFunc = findFunction(name)) {
    functionTypes_[existingFunc] = functionAbi.signature;
    funcOps_.insert_or_assign(emittedSymbol, existingFunc);
    if (auto aliasName = aliasNameOf(emittedSymbol)) {
      (void)findOrCreateSecondaryFunctionName(emittedSymbol, *aliasName, name);
    }
    enqueueFunctionBody(emittedSymbol);
    return existingFunc;
  }

  const auto loc = functionSymbol->location();

  auto guard = ir::InsertionGuard(emitter_);

  emitter_.setModuleInsertionPoint(true);

  auto inlineKind = emittedSymbol->isInline() ? ir::InlineKind::InlineHint
                                              : ir::InlineKind::NoInline;

  auto linkage = symbolLinkage(emittedSymbol);

  auto identifierName = [](const Identifier* id) -> std::string_view {
    return id ? id->name() : std::string_view{};
  };

  auto func = this->declareFunction(
      loc, ir::FunctionInfo{
               .name = name,
               .type = functionAbi.signature,
               .linkage = linkage,
               .visibility = visibility,
               .inlineKind = inlineKind,
               .importModule = identifierName(emittedSymbol->importModule()),
               .importName = identifierName(emittedSymbol->importName()),
               .exportName = identifierName(emittedSymbol->exportName()),
               .isUsed = emittedSymbol->isUsed(),
               .parameters = functionAbi.parameters});

  funcOps_.insert_or_assign(emittedSymbol, func);

  if (auto aliasName = aliasNameOf(emittedSymbol)) {
    (void)findOrCreateSecondaryFunctionName(emittedSymbol, *aliasName, name);
  }

  enqueueFunctionBody(emittedSymbol);

  return func;
}

auto Codegen::baseObjectStructorName(FunctionSymbol* functionSymbol)
    -> std::optional<std::string> {
  auto emittedSymbol = emittedFunctionSymbol(functionSymbol);

  if (!emittedSymbol->hasBaseObjectVariant()) return std::nullopt;

  ExternalNameEncoder encoder{unit_};
  encoder.setStructorVariant(ExternalNameEncoder::StructorVariant::Base);
  return encoder.encode(emittedSymbol);
}

auto Codegen::aliasNameOf(FunctionSymbol* emittedSymbol)
    -> std::optional<std::string> {
  if (auto alias = emittedSymbol->aliasName()) {
    return std::string{alias->name()};
  }

  return std::nullopt;
}

auto Codegen::findOrCreateSecondaryFunctionName(FunctionSymbol* functionSymbol,
                                                std::string_view name,
                                                std::string_view aliaseeName)
    -> ir::FunctionRef {
  auto emittedSymbol = emittedFunctionSymbol(functionSymbol);

  if (auto existingFunc = findFunction(name)) {
    if (!aliaseeName.empty() && !emitter_.functionHasBody(existingFunc)) {
      emitter_.setFunctionAliasee(existingFunc, aliaseeName);
    }
    return existingFunc;
  }

  const auto functionType = type_cast<FunctionType>(emittedSymbol->type());
  if (!functionType) return {};

  auto functionAbi = computeFunctionAbi(functionType, emittedSymbol);

  auto visibility = ir::Visibility::Default;
  if (emittedSymbol->hasHiddenVisibility()) visibility = ir::Visibility::Hidden;

  auto inlineKind = ir::InlineKind::NoInline;
  if (emittedSymbol->isInline()) inlineKind = ir::InlineKind::InlineHint;

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  return this->declareFunction(
      emittedSymbol->location(),
      ir::FunctionInfo{.name = name,
                       .type = functionAbi.signature,
                       .linkage = symbolLinkage(emittedSymbol),
                       .visibility = visibility,
                       .inlineKind = inlineKind,
                       .aliasee = aliaseeName,
                       .isUsed = emittedSymbol->isUsed(),
                       .parameters = functionAbi.parameters});
}

auto Codegen::findOrCreateBaseObjectStructor(FunctionSymbol* functionSymbol)
    -> ir::FunctionRef {
  auto func = findOrCreateFunction(functionSymbol);
  if (!func) return func;

  auto name = baseObjectStructorName(functionSymbol);
  if (!name) return func;

  return findOrCreateSecondaryFunctionName(functionSymbol, *name, {});
}

void Codegen::emitBaseObjectStructor(FunctionSymbol* functionSymbol,
                                     ir::FunctionRef completeObjectFunc) {
  auto name = baseObjectStructorName(functionSymbol);
  if (!name) return;

  auto baseObjectFunc =
      findOrCreateSecondaryFunctionName(functionSymbol, *name, {});
  if (!baseObjectFunc) return;
  if (emitter_.functionHasBody(baseObjectFunc)) return;

  auto emittedSymbol = emittedFunctionSymbol(functionSymbol);

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  emitForwardingBody(baseObjectFunc, emittedSymbol, completeObjectFunc, {}, {});
}

void Codegen::enqueueFunctionBody(FunctionSymbol* symbol) {
  auto target = emittedFunctionSymbol(symbol);
  target = target->resolvedDefinition();
  if (!target->declaration()) return;
  if (suppressesOutOfLineCopy(unit_, target)) return;
  if (!enqueuedFunctions_.insert(target).second) return;
  pendingFunctions_.push_back(target);
}

void Codegen::processPendingFunctions() {
  while (!pendingFunctions_.empty()) {
    auto sym = pendingFunctions_.back();
    pendingFunctions_.pop_back();

    auto target = sym->resolvedDefinition();

    if (auto funcDecl = target->declaration()) {
      (void)declaration(funcDecl);
    }

    emitAdjustingEntryPoints(sym);

    if (auto classSymbol = classRequiringVTable(sym))
      generateVTable(classSymbol);
  }
}

auto Codegen::findOrCreateGlobal(Symbol* symbol)
    -> std::optional<ir::GlobalRef> {
  auto variableSymbol = symbol_cast<VariableSymbol>(symbol);
  if (!variableSymbol) return {};

  auto canonicalVar = variableSymbol->canonical();

  if (auto it = globalOps_.find(canonicalVar); it != globalOps_.end()) {
    return it->second;
  }

  if (!variableSymbol->isStatic() && !variableSymbol->parent()->isNamespace()) {
    return {};
  }

  VariableSymbol* defVar = canonicalVar;
  if (!defVar->constValue().has_value()) {
    for (auto redecl : canonicalVar->redeclarations()) {
      if (redecl->constValue().has_value()) {
        defVar = redecl;
        break;
      }
    }
  }
  if (!defVar->constValue().has_value()) {
    defVar = canonicalVar->resolvedDefinition();
  }

  auto varType = convertType(defVar->type());

  const auto loc = variableSymbol->location();

  auto guard = ir::InsertionGuard(emitter_);

  emitter_.setModuleInsertionPoint(true);

  ir::Linkage linkageKind = symbolLinkage(variableSymbol);

  auto linkageAttr = linkageKind;

  std::string name;

  if (!variableSymbol->name()) {
    name = newUniqueSymbolName(".compoundliteral");
  } else if (unit_->language() != LanguageKind::kCXX &&
             !symbol->enclosingFunction()) {
    name = to_string(symbol->name());
  } else {
    std::string suffix;
    if (variableSymbol->isStatic()) {
      if (auto function = symbol->enclosingFunction()) {
        auto& count = staticLocalCounts_[symbol->name()];
        if (count > 0) {
          suffix = std::format("_{}", count - 1);
        }
        ++count;
      }
    }

    ExternalNameEncoder encoder{unit_};
    name = encoder.encode(symbol, suffix);
  }

  std::vector<ir::TypeRef> resultTypes;
  resultTypes.push_back(varType);

  ir::Initializer initializer;
  bool needsRegionInit = false;

  auto value = defVar->constValue();

  if (value.has_value()) {
    auto interp = ASTInterpreter{unit_};

    if (traits.is_integral_or_enum(defVar->type()) ||
        type_cast<MemberObjectPointerType>(defVar->type())) {
      if (auto attr = constValueToInitializer(*value, defVar->type()))
        initializer = *attr;
    } else if (auto attr = getFloatAttr(value, defVar->type())) {
      initializer = attr.value();
    } else if (traits.is_array(defVar->type())) {
      if (auto constArrayPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&*value)) {
        auto constArray = *constArrayPtr;
        std::vector<ir::Initializer> elements;
        bool allConverted = true;

        for (const auto& [elemValue, elemType] : constArray->elements) {
          if (auto attr = constValueToInitializer(elemValue, elemType)) {
            elements.push_back(*attr);
          } else {
            allConverted = false;
            break;
          }
        }

        if (allConverted) {
          initializer = ir::Initializer::aggregate(elements);
        } else {
          needsRegionInit = true;
        }
      } else if (auto constStringPtr =
                     std::get_if<const StringLiteral*>(&*value)) {
        auto stringLiteral = *constStringPtr;
        stringLiteral->initialize(stringLiteral->encoding());
        std::string str(stringLiteral->stringValue());

        switch (stringLiteral->encoding()) {
          case StringLiteralEncoding::kUtf16:
            str.push_back('\0');
            str.push_back('\0');
            break;
          case StringLiteralEncoding::kUtf32:
          case StringLiteralEncoding::kWide:
            str.push_back('\0');
            str.push_back('\0');
            str.push_back('\0');
            str.push_back('\0');
            break;
          default:
            str.push_back('\0');
            break;
        }

        initializer = ir::Initializer::byteString(
            std::string_view(str.data(), str.size()));

        if (auto arr = type_cast<BoundedArrayType>(defVar->type())) {
          auto destSize = static_cast<size_t>(arr->size());
          if (str.size() != destSize) {
            str.resize(destSize, '\0');
            initializer = ir::Initializer::byteString(
                std::string_view(str.data(), str.size()));
          }
        }
      }
    } else if (traits.is_class(defVar->type())) {
      needsRegionInit = true;
    } else if (type_cast<MemberFunctionPointerType>(defVar->type())) {
      needsRegionInit = true;
    } else if (traits.is_pointer(defVar->type()) ||
               traits.is_reference(defVar->type())) {
      if (auto attr = constValueToInitializer(*value, defVar->type())) {
        initializer = *attr;
      } else {
        needsRegionInit = true;
      }
    }
  }

  auto isExternalOnly = variableSymbol->isExtern();
  if (isExternalOnly) {
    if (auto canon = variableSymbol->canonical()) {
      if (canon->definition() || !canon->isExtern()) isExternalOnly = false;
    }
  }

  if (isExternalOnly && linkageAttr == ir::Linkage::Weak) {
    linkageAttr = ir::Linkage::ExternalWeak;
  }

  if (!initializer && !isExternalOnly && !needsRegionInit) {
    if (type_cast<MemberObjectPointerType>(defVar->type()))
      initializer = ir::Initializer::integerValue(emitter_.integerType(64),
                                                  nullMemberObjectPointer());
    else
      initializer = ir::Initializer::zero();
  }

  const auto isConstant = isReadOnlyStorage(
      traits, defVar->type(), value.has_value() || isExternalOnly);

  auto alignmentAttr = ir::Initializer::integerValue(
      emitter_.integerType(64), static_cast<int64_t>(getAlignment(defVar)));

  auto var = this->declareGlobal(loc, {.name = std::string_view(name),
                                       .type = varType,
                                       .linkage = linkageAttr,
                                       .isConstant = isConstant,
                                       .alignment = static_cast<std::uint64_t>(
                                           alignmentAttr.integer.toUIntMax()),
                                       .initializer = initializer,
                                       .unknownLocation = false,
                                       .isUsed = defVar->isUsed()});

  globalOps_.insert_or_assign(canonicalVar, var);

  if (needsRegionInit && value.has_value()) {
    auto initGuard = ir::InsertionGuard(emitter_);
    emitter_.beginGlobalInitializer(var);
    const auto initLoc = loc;
    auto result = emitConstInitValue(initLoc, defVar->type(), *value);
    emitter_.ret(initLoc, {&result, 1});
  }

  if (variableSymbol->isSpecialization())
    emitGlobalVarInit(variableSymbol, var);

  return var;
}

auto Codegen::staticStorageAddress(SourceLocation loc, Symbol* symbol)
    -> ir::ValueRef {
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    if (!field->isStatic()) return {};
    if (auto definition = field->definition())
      return staticStorageAddress(loc, definition);
    auto global = findOrCreateStaticField(field);
    auto pointerType = emitter_.pointerType(convertType(field->type()));
    return emitter_.addressOfSymbol(loc, pointerType, globalName(global));
  }

  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable) return {};
  auto global = findOrCreateGlobal(variable);
  if (!global) return {};
  auto pointerType = emitter_.pointerType(convertType(variable->type()));
  return emitter_.addressOfSymbol(loc, pointerType, globalName(*global));
}

auto Codegen::findOrCreateStaticField(FieldSymbol* field) -> ir::GlobalRef {
  if (auto it = staticFieldGlobalOps_.find(field);
      it != staticFieldGlobalOps_.end()) {
    return it->second;
  }

  auto varType = convertType(field->type());
  const auto loc = field->location();

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  const bool isDefinition = field->isInline() || field->isConstexpr();

  auto linkage = ir::Linkage::External;
  if (isDefinition) {
    linkage = hasInternalLinkage(field) ? ir::Linkage::Internal
                                        : ir::Linkage::LinkOnceODR;
  }
  auto linkageAttr = linkage;

  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encode(field);

  std::optional<ConstValue> value;
  ir::Initializer initializer;
  bool needsRegionInit = false;
  bool needsDynamicInit = false;
  if (isDefinition) {
    value = field->constValue();
    if (!value && field->initializer())
      value = ASTInterpreter{unit_}.evaluate(field->initializer());
    if (!value && field->initializer()) {
      if (field->isConstexpr() || field->isConstinit()) {
        cxx_runtime_error(std::format(
            "cannot emit constant initializer for static data member '{}'",
            to_string(field->name())));
      }
      needsDynamicInit = true;
    } else if (value) {
      if (auto attr = constValueToInitializer(*value, field->type())) {
        initializer = *attr;
      } else {
        needsRegionInit = true;
      }
    }
    if (!field->initializer() && field->constructor()) needsDynamicInit = true;
    if (!initializer && !needsRegionInit) initializer = ir::Initializer::zero();
  }

  const auto isConstant =
      isReadOnlyStorage(traits, field->type(), !needsDynamicInit);

  ir::Initializer alignmentAttr;

  auto var = this->declareGlobal(loc, {.name = std::string_view(name),
                                       .type = varType,
                                       .linkage = linkageAttr,
                                       .isConstant = isConstant,
                                       .alignment = static_cast<std::uint64_t>(
                                           alignmentAttr.integer.toUIntMax()),
                                       .initializer = initializer,
                                       .unknownLocation = false,
                                       .isUsed = field->isUsed()});

  staticFieldGlobalOps_.insert_or_assign(field, var);

  if (needsRegionInit) {
    auto initGuard = ir::InsertionGuard(emitter_);
    emitter_.beginGlobalInitializer(var);
    const auto initLoc = loc;
    auto result = emitConstInitValue(initLoc, field->type(), *value);
    emitter_.ret(initLoc, {&result, 1});
  }

  FunctionSymbol* destructor = nullptr;
  if (auto classType = unqualified_cast<ClassType>(field->type())) {
    auto classSymbol = classType->symbol();
    if (classSymbol)
      destructor = classSymbol->resolvedDefinition()->destructor();
  }
  const auto needsDestruction =
      destructor && !traits.has_trivial_destructor(field->type());
  if (needsDynamicInit || needsDestruction) {
    ExpressionAST* initializer = nullptr;
    FunctionSymbol* constructor = nullptr;
    if (needsDynamicInit) {
      initializer = field->initializer();
      constructor = field->constructor();
    }
    FunctionSymbol* cleanup = nullptr;
    if (needsDestruction) cleanup = completeObjectDtor(destructor);
    emitGlobalInit(field, field->type(), initializer, constructor, cleanup, var,
                   linkage == ir::Linkage::LinkOnceODR);
  }

  return var;
}

struct Codegen::ConstructorArgumentsVisitor {
  Codegen& gen;

  auto operator()(EqualInitializerAST* ast) -> std::vector<ExpressionResult> {
    if (auto braced = ast_cast<BracedInitListAST>(ast->expression))
      return (*this)(braced);
    return {gen.expression(ast->expression)};
  }

  auto operator()(ParenInitializerAST* ast) -> std::vector<ExpressionResult> {
    std::vector<ExpressionResult> result;
    for (auto it = ast->expressionList; it; it = it->next)
      result.push_back(gen.expression(it->value));
    return result;
  }

  auto operator()(BracedInitListAST* ast) -> std::vector<ExpressionResult> {
    if (gen.traits.initializer_list_element_type(ast->type))
      return {gen.expression(ast)};
    std::vector<ExpressionResult> result;
    for (auto it = ast->expressionList; it; it = it->next)
      result.push_back(gen.expression(it->value));
    return result;
  }

  auto operator()(ExpressionAST* ast) -> std::vector<ExpressionResult> {
    return {gen.expression(ast)};
  }
};

struct Codegen::InitializerExpressionVisitor {
  auto operator()(EqualInitializerAST* ast) -> ExpressionAST* {
    return ast->expression;
  }

  auto operator()(ParenInitializerAST* ast) -> ExpressionAST* {
    if (ast->expressionList && !ast->expressionList->next)
      return ast->expressionList->value;
    return nullptr;
  }

  auto operator()(ExpressionAST* ast) -> ExpressionAST* { return ast; }
};

auto Codegen::constructorArguments(ExpressionAST* initializer)
    -> std::vector<ExpressionResult> {
  if (!initializer) return {};
  return visit(ConstructorArgumentsVisitor{*this}, initializer);
}

auto Codegen::constructorArgumentList(BracedInitListAST* bracedInitList)
    -> List<ExpressionAST*>* {
  if (!bracedInitList) return nullptr;
  if (traits.initializer_list_element_type(bracedInitList->type))
    return make_list_node<ExpressionAST>(unit_->arena(), bracedInitList);
  return bracedInitList->expressionList;
}

auto Codegen::initializerExpression(ExpressionAST* initializer)
    -> ExpressionAST* {
  if (!initializer) return nullptr;
  return visit(InitializerExpressionVisitor{}, initializer);
}

void Codegen::emitGlobalVarInit(VariableSymbol* var, ir::GlobalRef global) {
  auto canonicalVar = var->canonical();
  auto defVar = canonicalVar->resolvedDefinition();
  if (defVar->isExtern()) return;

  auto destructor = objectDestructor(defVar->type());

  const auto linkage = emitter_.globalLinkage(global);
  const auto isConstantInitialized = defVar->constValue().has_value();
  const auto needsDestruction =
      destructor && !traits.has_trivial_destructor(defVar->type());
  ExpressionAST* initializer = nullptr;
  FunctionSymbol* constructor = nullptr;
  if (!isConstantInitialized) {
    initializer = defVar->initializer();
    constructor = defVar->constructor();
  }
  FunctionSymbol* cleanup = nullptr;
  if (needsDestruction) cleanup = completeObjectDtor(destructor);
  emitGlobalInit(canonicalVar, defVar->type(), initializer, constructor,
                 cleanup, global, linkage == ir::Linkage::LinkOnceODR);
}

auto Codegen::findOrCreateGuardVariable(Symbol* symbol, ir::Linkage linkage,
                                        SourceLocation loc) -> ir::GlobalRef {
  ExternalNameEncoder encoder{unit_};
  auto guardName = encoder.encodeGuardVariable(symbol);

  if (auto existing = this->findGlobal(guardName)) return existing;

  auto insertionGuard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  const bool isInternal = linkage == ir::Linkage::Internal;
  auto guardType = isInternal ? emitter_.integerType(8) : pointerSizedIntType();
  auto alignment = isInternal ? std::size_t(1) : pointerSize();

  return this->declareGlobal(
      loc, {.name = guardName,
            .type = guardType,
            .linkage = linkage,
            .isConstant = false,
            .alignment = static_cast<std::uint64_t>(alignment),
            .initializer = ir::Initializer::integerValue(guardType, 0),
            .unknownLocation = false});
}

void Codegen::emitStaticLocalVarInit(VariableSymbol* var, ir::GlobalRef global,
                                     ExpressionAST* initializer) {
  auto canonicalVar = var->canonical();
  auto defVar = canonicalVar->resolvedDefinition();
  if (defVar->isExtern()) return;
  if (defVar->constValue().has_value()) return;

  auto constructor = defVar->constructor();
  if (!initializer) initializer = defVar->initializer();

  auto destructor = objectDestructor(defVar->type());

  const bool needsDestruction =
      destructor && !traits.has_trivial_destructor(defVar->type());

  if (!constructor && !initializer && !needsDestruction) return;

  const auto loc = var->location();
  const auto initLoc =
      initializer ? initializer->firstSourceLocation() : var->location();

  auto initGuard = findOrCreateGuardVariable(
      canonicalVar, emitter_.globalLinkage(global), loc);

  auto guardByteType = emitter_.integerType(8);
  auto guardBytePtrType = emitter_.pointerType(guardByteType);
  auto guardAddress = emitter_.addressOfSymbol(loc, guardBytePtrType,
                                               this->globalName(initGuard));
  auto guardValue = emitter_.load(loc, guardByteType, guardAddress, 1);
  auto zero = emitter_.constantInt(loc, guardByteType, 0);
  auto needsInitialization =
      emitter_.compareInt(loc, ir::IntPredicate::Equal, guardValue, zero);

  auto initBlock = newBlock();
  auto continueBlock = newBlock();
  emitter_.condBranch(loc, needsInitialization, initBlock, continueBlock);

  emitter_.setInsertionBlock(initBlock);

  auto ptrType = emitter_.pointerType(convertType(defVar->type()));
  auto addr = emitter_.addressOfSymbol(loc, ptrType, this->globalName(global));

  {
    auto fullExpression = FullExpression{*this, initLoc};

    if (constructor) {
      emitConstructorInitialization(initLoc, addr, defVar->type(), constructor,
                                    initializer);
    } else if (auto expression = initializerExpression(initializer)) {
      (void)emitPrvalueInto(addr, defVar->type(), expression, initLoc);
    }
  }

  if (needsDestruction) {
    emitGlobalVarDtorRegistration(canonicalVar, defVar->type(),
                                  completeObjectDtor(destructor), global, loc);
  }

  auto one = emitter_.constantInt(loc, guardByteType, 1);
  emitter_.store(loc, one, guardAddress, 1);

  branch(initLoc, continueBlock);

  emitter_.setInsertionBlock(continueBlock);
}

void Codegen::emitGlobalInit(Symbol* symbol, const Type* type,
                             ExpressionAST* initializer,
                             FunctionSymbol* constructor,
                             FunctionSymbol* destructor, ir::GlobalRef global,
                             bool guarded) {
  if (!constructor && !initializer && !destructor) return;
  if (!emittedGlobalInits_.insert(symbol).second) return;

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(false);

  const auto loc = symbol->location();

  ir::GlobalRef initGuard;
  if (guarded)
    initGuard =
        findOrCreateGuardVariable(symbol, ir::Linkage::LinkOnceODR, loc);

  std::string name = "__cxx_global_var_init";
  if (globalVarInitCount_ > 0) {
    name = std::format("__cxx_global_var_init.{}", globalVarInitCount_);
  }
  ++globalVarInitCount_;

  auto funcType =
      emitter_.functionType(std::vector<ir::TypeRef>{},
                            std::vector<ir::TypeRef>{}, /*isVariadic=*/false);

  auto linkageAttr = ir::Linkage::Internal;

  auto func = this->declareFunction(
      loc,
      ir::FunctionInfo{
          .name = name, .type = funcType, .linkage = ir::Linkage::Internal});

  emitter_.globalConstructor(loc, func);

  auto functionBodyGuard = ir::FunctionBodyGuard{emitter_, func};

  auto guardBlock = emitter_.createBlock(func);
  auto entryBlock = guarded ? emitter_.createBlock(func) : guardBlock;
  auto exitBlock = emitter_.createBlock(func);

  ir::ValueRef exitValue;
  std::unordered_map<Symbol*, ir::ValueRef> locals;
  std::unordered_map<const Name*, int> staticLocalCounts;
  std::vector<CleanupScope> cleanupStack;
  FunctionSymbol* functionSymbol = nullptr;
  ir::ValueRef thisValue;

  std::swap(function_, func);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(exitValue_, exitValue);
  std::swap(locals_, locals);
  std::swap(staticLocalCounts_, staticLocalCounts);
  std::swap(cleanupStack_, cleanupStack);
  std::swap(currentFunctionSymbol_, functionSymbol);
  std::swap(thisValue_, thisValue);

  if (guarded) {
    emitter_.setInsertionBlock(guardBlock);
    auto guardType = pointerSizedIntType();
    auto guardPtrType = emitter_.pointerType(guardType);
    auto guardStorage = emitter_.addressOfSymbol(loc, guardPtrType,
                                                 this->globalName(initGuard));
    auto guardByteType = emitter_.integerType(8);
    auto guardBytePtrType = emitter_.pointerType(guardByteType);
    auto guardAddress = emitter_.bitcast(loc, guardBytePtrType, guardStorage);
    auto guardValue = emitter_.load(loc, guardByteType, guardAddress, 1);
    auto one = emitter_.constantInt(loc, guardByteType, 1);
    auto initializedBit =
        emitter_.binaryOp(loc, ir::BinaryOp::AndInt, guardValue, one);
    auto zero = emitter_.constantInt(loc, guardByteType, 0);
    auto needsInitialization =
        emitter_.compareInt(loc, ir::IntPredicate::Equal, initializedBit, zero);
    emitter_.condBranch(loc, needsInitialization, entryBlock_, exitBlock_);

    emitter_.setInsertionBlock(entryBlock_);
    emitter_.store(loc, one, guardAddress, 1);
  } else {
    emitter_.setInsertionBlock(entryBlock_);
  }

  auto ptrType = emitter_.pointerType(convertType(type));
  auto addr = emitter_.addressOfSymbol(loc, ptrType, this->globalName(global));

  const auto initLoc =
      initializer ? initializer->firstSourceLocation() : symbol->location();

  {
    auto fullExpression = FullExpression{*this, initLoc};

    if (constructor) {
      emitConstructorInitialization(initLoc, addr, type, constructor,
                                    initializer);
    } else if (auto expression = initializerExpression(initializer)) {
      (void)emitPrvalueInto(addr, type, expression, initLoc);
    }
  }

  if (destructor && !traits.has_trivial_destructor(type))
    emitGlobalVarDtorRegistration(symbol, type, destructor, global, loc);

  emitBranchWithCleanups(initLoc, exitBlock_, 0);

  emitter_.setInsertionBlock(exitBlock_);
  emitter_.ret(loc, {});

  emitter_.resolveFunctionControlFlow(function_);

  std::swap(function_, func);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(exitValue_, exitValue);
  std::swap(locals_, locals);
  std::swap(staticLocalCounts_, staticLocalCounts);
  std::swap(cleanupStack_, cleanupStack);
  std::swap(currentFunctionSymbol_, functionSymbol);
  std::swap(thisValue_, thisValue);
}

auto Codegen::emitTodoStmt(SourceLocation location, std::string_view message)
    -> ir::ValueRef {
  unit_->error(
      location,
      std::format("unable to generate code for this statement ({})", message));
  const auto loc = location;
  auto op = emitter_.todoStatement(loc, message);
  return op;
}

auto Codegen::emitTodoExpr(SourceLocation location, std::string_view message)
    -> ir::ValueRef {
  unit_->error(
      location,
      std::format("unable to generate code for this expression ({})", message));
  const auto loc = location;
  auto op = emitter_.todoExpression(loc, message);
  return op;
}

auto Codegen::vtableSlotIndex(FunctionSymbol* function) -> int {
  if (function->vtableSlotIndex() >= 0) return function->vtableSlotIndex();
  if (auto canonical = function->canonical();
      canonical && canonical->vtableSlotIndex() >= 0)
    return canonical->vtableSlotIndex();
  if (auto definition = function->definition();
      definition && definition->vtableSlotIndex() >= 0)
    return definition->vtableSlotIndex();
  return 0;
}

void Codegen::emitVTableOp(SourceLocation loc, std::string_view name,
                           ClassSymbol* classSymbol,
                           const VTableLayout::Group& group,
                           ir::Linkage linkage) {
  auto typeInfoAttr = findOrCreateTypeInfo(classSymbol->type());

  std::vector<std::vector<std::int64_t>> offsets(group.tables.size());
  std::vector<std::vector<ir::FunctionRef>> slots(group.tables.size());
  std::vector<ir::VTableTableInfo> tableInfos;

  for (std::size_t index = 0; index < group.tables.size(); ++index) {
    const auto& table = group.tables[index];
    for (const auto& offset : table.offsets)
      offsets[index].push_back(offset.value);
    for (const auto& slot : table.slots)
      slots[index].push_back(vtableEntry(loc, slot));
    tableInfos.push_back({.offsets = offsets[index],
                          .offsetToTop = table.offsetToTop,
                          .slots = slots[index]});
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  emitter_.defineVTable(loc, {.name = name,
                              .typeInfo = typeInfoAttr,
                              .tables = tableInfos,
                              .linkage = linkage});
}

void Codegen::emitVTableGroup(SourceLocation loc, std::string_view name,
                              ClassSymbol* classSymbol,
                              const VTableLayout::Group& group,
                              const VTableEmission& emission) {
  if (emitter_.symbolExists(name)) return;

  if (emission.emitDefinition)
    emitVTableOp(loc, name, classSymbol, group, emission.linkage);
  else
    declareExternalVTable(loc, name, group.wordCount());
}

void Codegen::declareExternalVTable(SourceLocation loc, std::string_view name,
                                    std::size_t wordCount) {
  if (emitter_.symbolExists(name)) return;

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto arrayType = this->arrayType(i8PtrType, wordCount);
  auto linkageAttr = ir::Linkage::External;

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);
  (void)this->declareGlobal(loc, {.name = name,
                                  .type = arrayType,
                                  .linkage = linkageAttr,
                                  .isConstant = true,
                                  .alignment = static_cast<std::uint64_t>(0),
                                  .initializer = ir::Initializer(),
                                  .unknownLocation = false});
}

auto Codegen::vtableEntryTarget(const VTableLayout::Slot& slot)
    -> FunctionSymbol* {
  if (slot.kind == VTableLayout::SlotKind::kDeletingDtor) {
    if (auto deletingDtor = slot.function->deletingDtorVariant())
      return deletingDtor;
    return completeObjectDtor(slot.function);
  }
  if (slot.kind == VTableLayout::SlotKind::kCompleteDtor)
    return completeObjectDtor(slot.function);
  return slot.function;
}

auto Codegen::vtableEntry(SourceLocation loc, const VTableLayout::Slot& slot)
    -> ir::FunctionRef {
  if (!slot.function) return {};

  if (slot.function->isPure())
    return findOrCreateUnimplementedVirtual(loc, "__cxa_pure_virtual");

  if (slot.function->isDeleted())
    return findOrCreateUnimplementedVirtual(loc, "__cxa_deleted_virtual");

  auto target = vtableEntryTarget(slot);
  if (slot.thisAdjustment.isEmpty() && slot.returnAdjustment.isEmpty())
    return findOrCreateFunction(target);

  return findOrCreateThunk(target, slot.thisAdjustment, slot.returnAdjustment);
}

auto Codegen::definesFunctionBody(FunctionSymbol* function) -> bool {
  auto principal = function->structorPrincipal();
  if (!principal) principal = function;
  auto definition = emittedFunctionSymbol(principal)->resolvedDefinition();
  return definition->declaration() != nullptr;
}

auto Codegen::applyCallOffset(SourceLocation loc, ir::ValueRef pointerI8,
                              const VTableLayout::CallOffset& callOffset,
                              bool virtualFirst) -> ir::ValueRef {
  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  auto wordType = pointerSizedIntType();

  auto addNonVirtual = [&](ir::ValueRef pointer) {
    if (!callOffset.nonVirtual) return pointer;
    auto offset = emitter_.constantInt(loc, wordType, callOffset.nonVirtual);
    return emitter_.pointerAdd(loc, i8PtrType, pointer, offset);
  };

  if (!callOffset.virtualOffset) return addNonVirtual(pointerI8);

  if (virtualFirst) {
    auto adjusted =
        adjustByVtableWord(loc, pointerI8, callOffset.virtualOffset);
    return addNonVirtual(adjusted);
  }

  return adjustByVtableWord(loc, addNonVirtual(pointerI8),
                            callOffset.virtualOffset);
}

void Codegen::emitForwardingBody(
    ir::FunctionRef function, FunctionSymbol* target,
    ir::FunctionRef targetFuncOp,
    const VTableLayout::CallOffset& thisAdjustment,
    const VTableLayout::CallOffset& returnAdjustment) {
  auto functionBodyGuard = ir::FunctionBodyGuard{emitter_, function};
  auto loc = target->location();

  auto entryBlock = emitter_.createBlock(function);
  std::vector<ir::ValueRef> callArgs;
  for (auto inputType : emitter_.functionParameterTypes(targetFuncOp))
    callArgs.push_back(emitter_.addBlockParameter(entryBlock, inputType, loc));
  emitter_.setInsertionBlock(entryBlock);

  auto functionType = type_cast<FunctionType>(target->type());
  const auto returnAbi = classifyClassValueAbi(functionType->returnType(),
                                               ClassValueAbiContext::Return);
  const std::size_t thisIndex =
      returnAbi.kind == ClassValueAbi::Kind::Indirect ? 1 : 0;

  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  auto rawThis = callArgs[thisIndex];
  auto adjustedThis = applyCallOffset(
      loc, emitter_.bitcast(loc, i8PtrType, rawThis), thisAdjustment,
      /*virtualFirst=*/false);
  callArgs[thisIndex] =
      emitter_.bitcast(loc, emitter_.typeOf(rawThis), adjustedThis);

  auto results = emitter_.call(
      loc, {.callee = this->functionName(targetFuncOp),
            .arguments = callArgs,
            .results = emitter_.functionResultTypes(targetFuncOp)});

  if (returnAdjustment.isEmpty() || results.empty()) {
    emitter_.ret(loc, results);
    return;
  }

  auto result = results.front();
  auto resultType = emitter_.typeOf(result);

  auto adjustResult = [&] {
    auto adjusted =
        applyCallOffset(loc, emitter_.bitcast(loc, i8PtrType, result),
                        returnAdjustment, /*virtualFirst=*/true);
    return emitter_.bitcast(loc, resultType, adjusted);
  };

  if (!traits.is_pointer(functionType->returnType())) {
    auto adjusted = adjustResult();
    emitter_.ret(loc, {&adjusted, 1});
    return;
  }

  auto wordType = pointerSizedIntType();
  auto isNull =
      emitter_.compareInt(loc, ir::IntPredicate::Equal,
                          emitter_.pointerToInt(loc, wordType, result),
                          emitter_.constantInt(loc, wordType, 0));

  auto adjustBlock = emitter_.createBlock(function);
  auto nullBlock = emitter_.createBlock(function);
  auto returnBlock = emitter_.createBlock(function);
  auto returnValue = emitter_.addBlockParameter(returnBlock, resultType, loc);
  emitter_.condBranch(loc, isNull, nullBlock, adjustBlock);

  emitter_.setInsertionBlock(nullBlock);
  emitter_.branch(loc, returnBlock, {&result, 1});

  emitter_.setInsertionBlock(adjustBlock);
  auto adjusted = adjustResult();
  emitter_.branch(loc, returnBlock, {&adjusted, 1});

  emitter_.setInsertionBlock(returnBlock);
  emitter_.ret(loc, {&returnValue, 1});
}

auto Codegen::findOrCreateThunk(
    FunctionSymbol* target, const VTableLayout::CallOffset& thisAdjustment,
    const VTableLayout::CallOffset& returnAdjustment) -> ir::FunctionRef {
  auto targetFuncOp = findOrCreateFunction(target);
  if (!targetFuncOp) return {};

  ExternalNameEncoder encoder{unit_};
  auto thunkName = encoder.encodeThunk(emittedFunctionSymbol(target),
                                       thisAdjustment, returnAdjustment);

  auto linkage = symbolLinkage(emittedFunctionSymbol(target));
  const auto definesBody =
      linkage != ir::Linkage::External || definesFunctionBody(target);

  if (auto existing = this->findFunction(thunkName)) {
    if (definesBody && !emitter_.functionHasBody(existing))
      emitForwardingBody(existing, target, targetFuncOp, thisAdjustment,
                         returnAdjustment);
    return existing;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto thunk = this->declareFunction(
      target->location(), ir::FunctionInfo{.name = thunkName,
                                           .type = functionType(targetFuncOp),
                                           .linkage = linkage});

  if (definesBody)
    emitForwardingBody(thunk, target, targetFuncOp, thisAdjustment,
                       returnAdjustment);

  return thunk;
}

void Codegen::emitAdjustingEntryPoints(FunctionSymbol* function) {
  auto principal = function->structorPrincipal();
  if (!principal) principal = function;
  if (!principal->isVirtual()) return;
  if (symbolLinkage(emittedFunctionSymbol(function)) != ir::Linkage::External)
    return;

  auto classSymbol = symbol_cast<ClassSymbol>(principal->parent());
  if (!classSymbol) return;
  auto vtableLayout = classSymbol->resolvedDefinition()->vtableLayout();
  if (!vtableLayout) return;

  auto emitted = emittedFunctionSymbol(principal);
  for (const auto& entryPoint : vtableLayout->adjustingEntryPoints) {
    if (emittedFunctionSymbol(entryPoint.function) != emitted) continue;
    auto target = vtableEntryTarget(
        {.function = entryPoint.function, .kind = entryPoint.kind});
    (void)findOrCreateThunk(target, entryPoint.thisAdjustment,
                            entryPoint.returnAdjustment);
  }
}

auto Codegen::classRequiringVTable(FunctionSymbol* function) -> ClassSymbol* {
  auto principal = function->structorPrincipal();
  if (!principal) principal = function;
  auto classSymbol = symbol_cast<ClassSymbol>(principal->parent());
  if (!classSymbol) return nullptr;
  classSymbol = classSymbol->resolvedDefinition();

  if (unit_->isExplicitInstantiationDefinition(principal)) return classSymbol;

  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout || !vtableLayout->keyFunction) return nullptr;
  if (vtableLayout->keyFunction->canonical() != principal->canonical())
    return nullptr;
  return classSymbol;
}

void Codegen::emitCtorVtableInit(FunctionSymbol* functionSymbol,
                                 SourceLocation loc) {
  if ((!functionSymbol->isConstructor() && !functionSymbol->isDestructor()) ||
      !thisValue_)
    return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return;

  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) return;

  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return;

  ExternalNameEncoder encoder{unit_};
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto addressPointType = emitter_.pointerType(i8PtrType);
  auto activeVTT = structorVTTValue_;
  const auto entryArgumentCount = emitter_.blockParameterCount(entryBlock_);
  if (!activeVTT && !functionSymbol->isStructorVariant() && entryBlock_ &&
      entryArgumentCount > 1)
    activeVTT = emitter_.blockParameter(entryBlock_, entryArgumentCount - 1);
  const auto usesVTT = activeVTT && requiresVTT(classSymbol) &&
                       !functionSymbol->isStructorVariant();

  auto intTy = convertType(control()->getIntType());

  auto loadVTTEntry = [&](std::size_t index) -> ir::ValueRef {
    auto offset = emitter_.constantInt(loc, intTy, index);
    auto entry =
        emitter_.pointerAdd(loc, emitter_.typeOf(activeVTT), activeVTT, offset);
    auto address = emitter_.load(loc, i8PtrType, entry, pointerSize());
    return emitter_.bitcast(loc, addressPointType, address);
  };

  const auto& group = vtableLayout->main;
  auto vtableArrayType = this->arrayType(i8PtrType, group.wordCount());

  auto addressPointOf = [&](std::size_t index) {
    generateVTable(classSymbol);
    auto vtableAddr = emitter_.addressOfSymbol(
        loc, emitter_.pointerType(vtableArrayType), vtableName);
    auto offset = emitter_.constantInt(
        loc, intTy, static_cast<int64_t>(group.addressPointIndex(index)));
    return emitter_.pointerAdd(loc, addressPointType, vtableAddr, offset);
  };

  auto thisPtr = loadThisPointer(loc, classSymbol);

  for (std::size_t index = 0; index < group.tables.size(); ++index) {
    const auto& table = group.tables[index];

    ir::ValueRef tableAddress;
    const auto vttIndex = vtableLayout->tableVTTIndices[index];
    if (usesVTT && vttIndex >= 0)
      tableAddress = loadVTTEntry(static_cast<std::size_t>(vttIndex));
    else
      tableAddress = addressPointOf(index);

    ir::ValueRef subobjectPtr = thisPtr;
    if (index != 0)
      subobjectPtr =
          tableSubobjectAddress(loc, thisPtr, classSymbol, table, usesVTT);

    emitter_.store(loc, tableAddress, vptrAddress(loc, subobjectPtr), 8);
  }
}

auto Codegen::tableSubobjectAddress(SourceLocation loc, ir::ValueRef thisPtr,
                                    ClassSymbol* classSymbol,
                                    const VTableLayout::Table& table,
                                    bool usesVTT) -> ir::ValueRef {
  auto virtualBase = table.enclosingVirtualBase;
  if (!usesVTT || !virtualBase)
    return subobjectAddress(loc, thisPtr, table.base, table.offset);

  auto virtualBaseAddress =
      emitBaseClassAddress(loc, thisPtr, classSymbol, virtualBase);
  auto info = classSymbol->layout()->getVirtualBaseInfo(virtualBase);
  const auto delta = info ? table.offset - info->offset : 0;
  return subobjectAddress(loc, virtualBaseAddress, table.base, delta);
}

auto Codegen::subobjectAddress(SourceLocation loc, ir::ValueRef objectPtr,
                               ClassSymbol* subobjectClass,
                               std::uint64_t byteOffset) -> ir::ValueRef {
  auto subobjectPtrType =
      emitter_.pointerType(convertType(subobjectClass->type()));

  if (byteOffset == 0) {
    return emitter_.bitcast(loc, subobjectPtrType, objectPtr);
  }

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  auto objectI8 = emitter_.bitcast(loc, i8PtrType, objectPtr);

  auto offsetType = pointerSizedIntType();
  auto offset = emitter_.constantInt(loc, offsetType,
                                     static_cast<std::int64_t>(byteOffset));

  auto adjusted = emitter_.pointerAdd(loc, i8PtrType, objectI8, offset);

  return emitter_.bitcast(loc, subobjectPtrType, adjusted);
}

auto Codegen::memberAddress(SourceLocation loc, ir::ValueRef objectPtr,
                            const Type* memberType, std::uint32_t index)
    -> ir::ValueRef {
  return memberAddress(loc, objectPtr, convertType(memberType), index);
}

auto Codegen::vtableEmission(ClassSymbol* classSymbol) -> VTableEmission {
  if (classSymbol->isExplicitInstantiationDeclared(unit_))
    return {.emitDefinition = false, .linkage = ir::Linkage::External};

  if (hasInternalLinkage(classSymbol))
    return {.emitDefinition = true, .linkage = ir::Linkage::Internal};

  if (classSymbol->templateDeclaration() || classSymbol->isSpecialization())
    return {};

  auto vtableLayout = classSymbol->vtableLayout();
  auto keyFunction = vtableLayout ? vtableLayout->keyFunction : nullptr;
  if (!keyFunction) return {};

  auto definition = keyFunction->resolvedDefinition();
  if (!definition || !definition->isDefined())
    return {.emitDefinition = false, .linkage = ir::Linkage::External};

  if (keyFunction->isInline() || definition->isInline()) return {};

  return {.emitDefinition = true, .linkage = ir::Linkage::External};
}

auto Codegen::memberAddress(SourceLocation loc, ir::ValueRef objectPtr,
                            ir::TypeRef memberType, std::uint32_t index)
    -> ir::ValueRef {
  auto ptrType = emitter_.pointerType(memberType);
  return emitter_.memberAddress(loc, ptrType, objectPtr, index);
}

auto Codegen::vptrAddress(SourceLocation loc, ir::ValueRef objectPtr)
    -> ir::ValueRef {
  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  return emitter_.bitcast(loc, emitter_.pointerType(i8PtrType), objectPtr);
}

auto Codegen::requiresVTT(ClassSymbol* classSymbol) const -> bool {
  if (!classSymbol) return false;
  return classSymbol->hasVirtualBaseSubobjects();
}

auto Codegen::constructionVTableName(ClassSymbol* completeClass,
                                     const VTableLayout::Group& group)
    -> std::string {
  ExternalNameEncoder encoder{unit_};
  return encoder.encodeConstructionVTable(
      completeClass, static_cast<std::int64_t>(group.offset), group.base);
}

void Codegen::generateVTT(ClassSymbol* completeClass,
                          const VTableEmission& emission) {
  if (!requiresVTT(completeClass)) return;

  auto vtableLayout = completeClass->vtableLayout();
  if (!vtableLayout || vtableLayout->vtt.empty()) return;

  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeVTT(completeClass);
  if (emitter_.symbolExists(name)) return;

  auto loc = completeClass->location();

  std::vector<std::string> groupNames;
  for (const auto& group : vtableLayout->constructionGroups) {
    groupNames.push_back(constructionVTableName(completeClass, group));
    emitVTableGroup(loc, groupNames.back(), group.base, group, emission);
  }

  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  auto arrayType = this->arrayType(i8PtrType, vtableLayout->vtt.size());

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);
  auto global =
      this->declareGlobal(loc, {.name = name,
                                .type = arrayType,
                                .linkage = emission.linkage,
                                .isConstant = true,
                                .alignment = static_cast<std::uint64_t>(0),
                                .initializer = ir::Initializer(),
                                .unknownLocation = false});
  if (!emission.emitDefinition) return;

  auto mainName = ExternalNameEncoder{unit_}.encodeVTable(completeClass);

  emitter_.beginGlobalInitializer(global);
  auto value = emitter_.undef(loc, arrayType);
  auto indexType = convertType(control()->getIntType());

  for (std::size_t index = 0; index < vtableLayout->vtt.size(); ++index) {
    const auto& entry = vtableLayout->vtt[index];
    const auto& group =
        entry.group < 0
            ? vtableLayout->main
            : vtableLayout
                  ->constructionGroups[static_cast<std::size_t>(entry.group)];
    const auto& tableName =
        entry.group < 0 ? mainName
                        : groupNames[static_cast<std::size_t>(entry.group)];
    auto tableType = this->arrayType(i8PtrType, group.wordCount());
    auto table = emitter_.addressOfSymbol(loc, emitter_.pointerType(tableType),
                                          tableName);
    auto addressPointIndex = emitter_.constantInt(
        loc, indexType, group.addressPointIndex(entry.table));
    auto addressPoint = emitter_.pointerAdd(
        loc, emitter_.pointerType(i8PtrType), table, addressPointIndex);
    auto address = emitter_.bitcast(loc, i8PtrType, addressPoint);
    value = emitter_.insertValue(loc, arrayType, value, address,
                                 static_cast<std::int64_t>(index));
  }
  emitter_.ret(loc, {&value, 1});
}

auto Codegen::vttAddress(SourceLocation loc, ClassSymbol* completeClass,
                         std::size_t index) -> ir::ValueRef {
  generateVTable(completeClass);
  auto vtableLayout = completeClass->vtableLayout();
  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeVTT(completeClass);
  auto i8PtrType = emitter_.pointerType(emitter_.integerType(8));
  auto arrayType = this->arrayType(i8PtrType, vtableLayout->vtt.size());
  auto arrayPtrType = emitter_.pointerType(arrayType);
  auto address = emitter_.addressOfSymbol(loc, arrayPtrType, name);
  auto indexType = convertType(control()->getIntType());
  auto offset = emitter_.constantInt(loc, indexType, index);
  return emitter_.pointerAdd(loc, emitter_.pointerType(i8PtrType), address,
                             offset);
}

void Codegen::generateVTable(ClassSymbol* classSymbol) {
  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) return;

  if (!emittedVTables_.insert(classSymbol).second) return;

  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return;

  ExternalNameEncoder encoder{unit_};
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto emission = vtableEmission(classSymbol);
  emitVTableGroup(classSymbol->location(), vtableName, classSymbol,
                  vtableLayout->main, emission);

  generateVTT(classSymbol, emission);
}

auto Codegen::findOrCreateUnimplementedVirtual(SourceLocation loc,
                                               std::string_view name)
    -> ir::FunctionRef {
  if (auto existingFunc = this->findFunction(name)) {
    return existingFunc;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto funcType = emitter_.functionType({}, {}, /*isVariadic=*/false);

  return this->declareFunction(
      loc, {.name = name, .type = funcType, .linkage = ir::Linkage::External});
}

auto Codegen::findOrCreateCxaAtexit(SourceLocation loc) -> ir::FunctionRef {
  const std::string_view name = "__cxa_atexit";

  if (auto existingFunc = this->findFunction(name)) {
    return existingFunc;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto i32Type = emitter_.integerType(32);

  std::vector<ir::TypeRef> paramTypes{i8PtrType, i8PtrType, i8PtrType};
  std::vector<ir::TypeRef> resultTypes{i32Type};
  auto funcType = emitter_.functionType(paramTypes, resultTypes, /*isVariadic=*/
                                        false);
  auto linkageAttr = ir::Linkage::External;

  return this->declareFunction(
      loc, {.name = name, .type = funcType, .linkage = linkageAttr});
}

auto Codegen::findOrCreateDsoHandle(SourceLocation loc) -> ir::GlobalRef {
  const std::string_view name = "__dso_handle";

  if (auto existing = this->findGlobal(name)) {
    return existing;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto i8Type = emitter_.integerType(8);
  auto linkageAttr = ir::Linkage::External;

  ir::Initializer alignmentAttr;

  return this->declareGlobal(
      loc, {.name = name,
            .type = i8Type,
            .linkage = linkageAttr,
            .isConstant = /*isConstant=*/false,
            .alignment =
                static_cast<std::uint64_t>(alignmentAttr.integer.toUIntMax()),
            .initializer = /*initializer=*/ir::Initializer(),
            .unknownLocation = false});
}

void Codegen::emitGlobalVarDtorRegistration(Symbol* symbol, const Type* type,
                                            FunctionSymbol* dtor,
                                            ir::GlobalRef global,
                                            SourceLocation loc) {
  auto savedInsertionPoint = emitter_.saveInsertionPoint();

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  std::string thunkName = "__cxx_global_array_dtor";
  if (globalVarDtorCount_ > 0) {
    thunkName = std::format("__cxx_global_array_dtor.{}", globalVarDtorCount_);
  }
  ++globalVarDtorCount_;

  std::vector<ir::TypeRef> paramTypes{i8PtrType};
  std::vector<ir::TypeRef> resultTypes;
  auto funcType =
      emitter_.functionType(paramTypes, resultTypes, /*isVariadic=*/false);
  emitter_.setModuleInsertionPoint(false);

  auto thunkFunc = this->declareFunction(
      loc, ir::FunctionInfo{.name = thunkName,
                            .type = funcType,
                            .linkage = ir::Linkage::Internal});

  auto functionBodyGuard = ir::FunctionBodyGuard{emitter_, thunkFunc};

  auto entryBlock = emitter_.createBlock(thunkFunc);
  (void)emitter_.addBlockParameter(entryBlock, i8PtrType, loc);
  emitter_.setInsertionBlock(entryBlock);

  auto ptrType = emitter_.pointerType(convertType(type));
  auto addr = emitter_.addressOfSymbol(loc, ptrType, this->globalName(global));

  if (traits.is_array(type)) {
    auto count =
        arrayElementCount(loc, type, convertType(control()->getSizeType()));
    emitArrayLoop(loc, addr, traits.remove_all_extents(type), count,
                  /*reverse=*/true, [&](ir::ValueRef element) {
                    (void)emitCall(symbol->location(), dtor, {element}, {});
                  });
  } else {
    (void)emitCall(symbol->location(), dtor, {addr}, {});
  }

  emitter_.ret(loc, {});

  emitter_.setModuleInsertionPoint(false);

  auto atexitFunc = findOrCreateCxaAtexit(loc);
  auto dsoHandle = findOrCreateDsoHandle(loc);

  emitter_.restoreInsertionPoint(savedInsertionPoint);

  auto thunkPtr =
      emitter_.addressOfSymbol(loc, i8PtrType, this->functionName(thunkFunc));
  auto nullPtr = emitter_.nullPointer(loc, i8PtrType);
  auto dsoHandlePtr =
      emitter_.addressOfSymbol(loc, i8PtrType, this->globalName(dsoHandle));

  std::vector<ir::ValueRef> args{thunkPtr, nullPtr, dsoHandlePtr};
  std::vector<ir::TypeRef> callResultTypes{emitter_.integerType(32)};
  (void)emitter_.call(implicitLocation(loc),
                      {.callee = this->functionName(atexitFunc),
                       .arguments = args,
                       .results = callResultTypes});
}

auto Codegen::completeObjectDtor(FunctionSymbol* dtor) -> FunctionSymbol* {
  if (auto variant = dtor->completeObjectVariant()) return variant;
  return dtor;
}
}  // namespace cxx
