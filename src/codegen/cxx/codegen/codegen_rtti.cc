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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

constexpr std::uint32_t kNonDiamondRepeatMask = 0x1;
constexpr std::uint32_t kDiamondShapedMask = 0x2;

constexpr std::int64_t kVirtualBaseMask = 0x1;
constexpr std::int64_t kPublicBaseMask = 0x2;
constexpr std::int64_t kBaseOffsetShift = 8;

constexpr std::uint32_t kPointeeConstMask = 0x1;
constexpr std::uint32_t kPointeeVolatileMask = 0x2;
constexpr std::uint32_t kPointeeIncompleteMask = 0x8;
constexpr std::uint32_t kPointeeIncompleteClassMask = 0x10;
constexpr std::uint32_t kPointeeNoexceptMask = 0x40;

auto isFundamentalTypeInfoInRuntime(const Type* type) -> bool {
  switch (type->kind()) {
    case TypeKind::kVoid:
    case TypeKind::kNullptr:
    case TypeKind::kBool:
    case TypeKind::kSignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kInt:
    case TypeKind::kLongInt:
    case TypeKind::kLongLongInt:
    case TypeKind::kInt128:
    case TypeKind::kUnsignedChar:
    case TypeKind::kUnsignedShortInt:
    case TypeKind::kUnsignedInt:
    case TypeKind::kUnsignedLongInt:
    case TypeKind::kUnsignedLongLongInt:
    case TypeKind::kUnsignedInt128:
    case TypeKind::kChar:
    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar:
    case TypeKind::kFloat:
    case TypeKind::kDouble:
    case TypeKind::kLongDouble:
    case TypeKind::kFloat16:
      return true;
    default:
      return false;
  }
}

auto abiTypeInfoClassName(const Type* type) -> std::string_view {
  switch (type->kind()) {
    case TypeKind::kEnum:
    case TypeKind::kScopedEnum:
      return "16__enum_type_info";
    case TypeKind::kFunction:
      return "20__function_type_info";
    case TypeKind::kBoundedArray:
    case TypeKind::kUnboundedArray:
      return "17__array_type_info";
    case TypeKind::kPointer:
      return "19__pointer_type_info";
    case TypeKind::kMemberObjectPointer:
    case TypeKind::kMemberFunctionPointer:
      return "29__pointer_to_member_type_info";
    default:
      return "23__fundamental_type_info";
  }
}

auto isPointerDereference(ExpressionAST* expression) -> bool {
  while (auto nested = ast_cast<NestedExpressionAST>(expression)) {
    expression = nested->expression;
  }
  auto unary = ast_cast<UnaryExpressionAST>(expression);
  return unary && unary->op == TokenKind::T_STAR;
}

auto canUseSingleInheritanceTypeInfo(ClassSymbol* classSymbol) -> bool {
  if (classSymbol->baseClasses().size() != 1) return false;
  auto base = classSymbol->baseClasses().front();
  if (base->isVirtual()) return false;
  if (base->accessSpecifier() != AccessSpecifier::kPublic) return false;
  auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
  if (!baseClass) return false;
  auto baseDefinition = baseClass->resolvedDefinition();
  auto layout = classSymbol->layout();
  if (!layout) return false;
  auto baseInfo = layout->getBaseInfo(baseDefinition);
  return baseInfo && baseInfo->offset == 0;
}

}  // namespace

struct Codegen::TypeInfoIncompleteClassVisitor {
  Codegen& gen;

  auto operator()(const ClassType* type) const -> bool {
    return !type->definition()->isComplete();
  }

  auto operator()(const PointerType* type) const -> bool {
    return gen.typeInfoHasIncompleteClass(type->elementType());
  }

  auto operator()(const MemberObjectPointerType* type) const -> bool {
    return gen.typeInfoHasIncompleteClass(type->elementType()) ||
           gen.typeInfoHasIncompleteClass(type->classType());
  }

  auto operator()(const MemberFunctionPointerType* type) const -> bool {
    return gen.typeInfoHasIncompleteClass(type->functionType()) ||
           gen.typeInfoHasIncompleteClass(type->classType());
  }

  auto operator()(const BoundedArrayType* type) const -> bool {
    return gen.typeInfoHasIncompleteClass(type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) const -> bool {
    return gen.typeInfoHasIncompleteClass(type->elementType());
  }

  template <typename T>
  auto operator()(const T*) const -> bool {
    return false;
  }
};

struct Codegen::TypeInfoInternalLinkageVisitor {
  const Codegen& gen;

  auto operator()(const ClassType* type) const -> bool {
    return gen.hasInternalLinkage(type->definition());
  }

  auto operator()(const EnumType* type) const -> bool {
    return gen.hasInternalLinkage(type->symbol());
  }

  auto operator()(const ScopedEnumType* type) const -> bool {
    return gen.hasInternalLinkage(type->symbol());
  }

  auto operator()(const PointerType* type) const -> bool {
    return gen.typeInfoHasInternalLinkage(type->elementType());
  }

  auto operator()(const MemberObjectPointerType* type) const -> bool {
    return gen.typeInfoHasInternalLinkage(type->elementType()) ||
           gen.typeInfoHasInternalLinkage(type->classType());
  }

  auto operator()(const MemberFunctionPointerType* type) const -> bool {
    return gen.typeInfoHasInternalLinkage(type->functionType()) ||
           gen.typeInfoHasInternalLinkage(type->classType());
  }

  auto operator()(const FunctionType* type) const -> bool {
    if (gen.typeInfoHasInternalLinkage(type->returnType())) return true;
    return std::ranges::any_of(
        type->parameterTypes(), [&](const Type* parameterType) {
          return gen.typeInfoHasInternalLinkage(parameterType);
        });
  }

  auto operator()(const BoundedArrayType* type) const -> bool {
    return gen.typeInfoHasInternalLinkage(type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) const -> bool {
    return gen.typeInfoHasInternalLinkage(type->elementType());
  }

  template <typename T>
  auto operator()(const T*) const -> bool {
    return false;
  }
};

auto Codegen::typeInfoHasIncompleteClass(const Type* type) -> bool {
  type = traits.remove_cv(traits.remove_reference(type));
  return visit(TypeInfoIncompleteClassVisitor{*this}, type);
}

auto Codegen::typeInfoHasInternalLinkage(const Type* type) const -> bool {
  type = traits.remove_cv(traits.remove_reference(type));
  return visit(TypeInfoInternalLinkageVisitor{*this}, type);
}

auto Codegen::typeInfoEmission(const Type* type) -> VTableEmission {
  type = traits.remove_cv(traits.remove_reference(type));

  if (typeInfoHasIncompleteClass(type) || typeInfoHasInternalLinkage(type))
    return {.emitDefinition = true, .linkage = ir::Linkage::Internal};

  auto classType = type_cast<ClassType>(type);
  if (!classType) return {};

  auto classSymbol = classType->definition();
  if (hasInternalLinkage(classSymbol))
    return {.emitDefinition = true, .linkage = ir::Linkage::Internal};

  if (!classSymbol->isPolymorphic()) return {};
  return vtableEmission(classSymbol);
}

auto Codegen::findOrCreateAbiTypeInfoVTable(std::string_view abiClassName)
    -> ir::GlobalRef {
  auto name =
      std::format("_ZTVN10__cxxabiv1{}E", std::string_view{abiClassName});

  if (auto existing = this->findGlobal(name)) {
    return existing;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto arrayType = this->arrayType(i8PtrType, 0);

  auto linkageAttr = ir::Linkage::External;

  return this->declareGlobal(SourceLocation{},
                             {.name = name,
                              .type = arrayType,
                              .linkage = linkageAttr,
                              .isConstant = true,
                              .alignment = static_cast<std::uint64_t>(0),
                              .initializer = ir::Initializer(),
                              .unknownLocation = true});
}

auto Codegen::findOrCreateTypeInfoName(const Type* type) -> std::string {
  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeTypeInfoName(type);

  if (this->findGlobal(name)) return name;

  ExternalNameEncoder contentEncoder{unit_};
  auto content = contentEncoder.encode(type);
  content.push_back('\0');

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto i8Type = emitter_.integerType(8);
  auto arrayType = this->arrayType(i8Type, content.size());
  auto linkage = typeInfoEmission(type).linkage;
  auto linkageAttr = linkage;

  (void)this->declareGlobal(
      SourceLocation{}, {.name = name,
                         .type = arrayType,
                         .linkage = linkageAttr,
                         .isConstant = true,
                         .alignment = static_cast<std::uint64_t>(0),
                         .initializer = ir::Initializer::byteString(
                             std::string_view(content.data(), content.size())),
                         .unknownLocation = true});

  return name;
}

auto Codegen::emitTypeInfoObject(
    SourceLocation loc, std::string_view name, std::string_view abiClassName,
    std::string_view typeInfoNameSymbol, ir::Linkage linkage,
    const std::function<void(std::vector<ir::TypeRef>& fieldTypes,
                             std::vector<ir::ValueRef>& fields)>&
        emitTrailingFields) -> ir::GlobalRef {
  auto abiVTable = findOrCreateAbiTypeInfoVTable(abiClassName);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto wordPtrType = emitter_.pointerType(i8PtrType);

  auto recordType = declareClassType(name, /*isUnion=*/false);

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto linkageAttr = linkage;

  auto global =
      this->declareGlobal(loc, {.name = name,
                                .type = recordType,
                                .linkage = linkageAttr,
                                .isConstant = true,
                                .alignment = static_cast<std::uint64_t>(0),
                                .initializer = ir::Initializer(),
                                .unknownLocation = false});

  emitter_.beginGlobalInitializer(global);

  auto abiVTableAddr =
      emitter_.addressOfSymbol(loc, wordPtrType, this->globalName(abiVTable));

  auto intType = convertType(control()->getIntType());
  auto addressPointIndex = emitter_.constantInt(loc, intType, 2);

  std::vector<ir::TypeRef> fieldTypes{i8PtrType, i8PtrType};
  std::vector<ir::ValueRef> fields{
      emitter_.pointerAdd(loc, wordPtrType, abiVTableAddr, addressPointIndex),
      emitter_.addressOfSymbol(loc, i8PtrType, typeInfoNameSymbol)};

  emitTrailingFields(fieldTypes, fields);

  (void)this->defineClassType(recordType, fieldTypes, false);

  auto record = emitter_.undef(loc, recordType);
  for (std::int64_t index = 0; index < std::int64_t(fields.size()); ++index) {
    record =
        emitter_.insertValue(loc, recordType, record, fields[index], index);
  }

  emitter_.ret(loc, {&record, 1});

  return global;
}

auto Codegen::virtualBaseOffsetSlotOffset(ClassSymbol* classSymbol,
                                          ClassSymbol* virtualBase)
    -> std::optional<std::int64_t> {
  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return std::nullopt;

  auto& primary = vtableLayout->primary;
  const auto wordSize = pointerSize();

  for (std::size_t index = 0; index < primary.vbaseOffsets.size(); ++index) {
    if (primary.vbaseOffsets[index].first->resolvedDefinition() !=
        virtualBase->resolvedDefinition())
      continue;
    const auto distanceWords =
        static_cast<std::int64_t>(primary.vbaseOffsets.size() + 2 - index);
    return -wordSize * distanceWords;
  }

  return std::nullopt;
}

auto Codegen::classTypeInfoBaseDescriptors(ClassSymbol* classSymbol)
    -> std::vector<Codegen::TypeInfoBaseDescriptor> {
  std::vector<TypeInfoBaseDescriptor> descriptors;

  auto layout = classSymbol->layout();
  if (!layout) {
    cxx_runtime_error(std::format("missing class layout for RTTI '{}'",
                                  to_string(classSymbol->name())));
  }

  for (auto baseClass : classSymbol->baseClasses()) {
    auto baseSymbol = symbol_cast<ClassSymbol>(baseClass->symbol());
    if (!baseSymbol) continue;
    auto baseDefinition = baseSymbol->resolvedDefinition();

    std::int64_t offsetFlags = 0;
    if (baseClass->accessSpecifier() == AccessSpecifier::kPublic)
      offsetFlags |= kPublicBaseMask;

    if (baseClass->isVirtual()) {
      offsetFlags |= kVirtualBaseMask;
      auto slotOffset =
          virtualBaseOffsetSlotOffset(classSymbol, baseDefinition);
      if (!slotOffset) {
        cxx_runtime_error(std::format(
            "missing virtual-base RTTI slot for '{}' in '{}'",
            to_string(baseDefinition->name()), to_string(classSymbol->name())));
      }
      offsetFlags |= *slotOffset << kBaseOffsetShift;
    } else {
      auto baseInfo = layout->getBaseInfo(baseDefinition);
      if (!baseInfo) {
        cxx_runtime_error(std::format("missing base layout for '{}' in '{}'",
                                      to_string(baseDefinition->name()),
                                      to_string(classSymbol->name())));
      }
      offsetFlags |= static_cast<std::int64_t>(baseInfo->offset)
                     << kBaseOffsetShift;
    }

    descriptors.push_back(
        {.typeInfo = findOrCreateTypeInfo(baseDefinition->type()),
         .offsetFlags = offsetFlags});
  }

  return descriptors;
}

void Codegen::emitClassTypeInfoBases(
    ClassSymbol* classSymbol,
    const std::vector<TypeInfoBaseDescriptor>& descriptors,
    std::vector<ir::TypeRef>& fieldTypes, std::vector<ir::ValueRef>& fields,
    SourceLocation loc) {
  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto i32Type = emitter_.integerType(32);
  auto offsetFlagsType = pointerSizedIntType();

  auto repetition = classSymbol->baseClassRepetition();
  std::int64_t flags = 0;
  if (repetition.nonDiamondRepeat) flags |= kNonDiamondRepeatMask;
  if (repetition.diamondShaped) flags |= kDiamondShapedMask;

  const auto appendInt = [&](ir::TypeRef type, std::int64_t value) {
    fieldTypes.push_back(type);
    fields.push_back(emitter_.constantInt(loc, type, value));
  };

  appendInt(i32Type, flags);
  appendInt(i32Type, static_cast<std::int64_t>(descriptors.size()));

  for (auto& descriptor : descriptors) {
    fieldTypes.push_back(i8PtrType);
    fields.push_back(
        emitter_.addressOfSymbol(loc, i8PtrType, descriptor.typeInfo));

    appendInt(offsetFlagsType, descriptor.offsetFlags);
  }
}

auto Codegen::findOrCreateTypeInfo(const Type* type) -> std::string {
  type = traits.remove_cv(type);

  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeTypeInfo(type);

  if (!emittedTypeInfos_.insert(name).second) return name;
  if (this->findGlobal(name)) return name;

  auto guard = ir::InsertionGuard(emitter_);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  const auto declareExternal = [&] {
    emitter_.setModuleInsertionPoint(true);
    auto linkageAttr = ir::Linkage::External;
    (void)this->declareGlobal(SourceLocation{},
                              {.name = name,
                               .type = i8PtrType,
                               .linkage = linkageAttr,
                               .isConstant = true,
                               .alignment = static_cast<std::uint64_t>(0),
                               .initializer = ir::Initializer(),
                               .unknownLocation = true});
    return name;
  };

  if (isFundamentalTypeInfoInRuntime(type)) return declareExternal();

  if (auto pointerType = type_cast<PointerType>(type)) {
    auto pointee = traits.remove_cv(pointerType->elementType());
    if (isFundamentalTypeInfoInRuntime(pointee) &&
        !has_volatile(cv_qualifiers(pointerType->elementType()))) {
      return declareExternal();
    }
  }

  auto emission = typeInfoEmission(type);
  if (!emission.emitDefinition) return declareExternal();

  auto typeInfoNameSymbol = findOrCreateTypeInfoName(type);

  auto classType = type_cast<ClassType>(type);
  auto classSymbol = classType ? classType->definition() : nullptr;

  auto loc = classSymbol ? classSymbol->location() : SourceLocation{};

  if (classSymbol) {
    if (classSymbol->baseClasses().empty()) {
      (void)emitTypeInfoObject(loc, name, "17__class_type_info",
                               typeInfoNameSymbol, emission.linkage,
                               [](auto& fieldTypes, auto& fields) {});
      return name;
    }

    if (canUseSingleInheritanceTypeInfo(classSymbol)) {
      auto base = symbol_cast<ClassSymbol>(
          classSymbol->baseClasses().front()->symbol());
      auto baseTypeInfo =
          findOrCreateTypeInfo(base->resolvedDefinition()->type());
      (void)emitTypeInfoObject(loc, name, "20__si_class_type_info",
                               typeInfoNameSymbol, emission.linkage,
                               [&](auto& fieldTypes, auto& fields) {
                                 fieldTypes.push_back(i8PtrType);
                                 fields.push_back(emitter_.addressOfSymbol(
                                     loc, i8PtrType, baseTypeInfo));
                               });
      return name;
    }

    auto descriptors = classTypeInfoBaseDescriptors(classSymbol);
    (void)emitTypeInfoObject(loc, name, "21__vmi_class_type_info",
                             typeInfoNameSymbol, emission.linkage,
                             [&](auto& fieldTypes, auto& fields) {
                               emitClassTypeInfoBases(classSymbol, descriptors,
                                                      fieldTypes, fields, loc);
                             });
    return name;
  }

  const Type* pointee = nullptr;
  const Type* memberPointerClass = nullptr;

  if (auto pointerType = type_cast<PointerType>(type)) {
    pointee = pointerType->elementType();
  } else if (auto memberObjectPointer =
                 type_cast<MemberObjectPointerType>(type)) {
    pointee = memberObjectPointer->elementType();
    memberPointerClass = memberObjectPointer->classType();
  } else if (auto memberFunctionPointer =
                 type_cast<MemberFunctionPointerType>(type)) {
    pointee = memberFunctionPointer->functionType();
    memberPointerClass = memberFunctionPointer->classType();
  }

  if (!pointee) {
    (void)emitTypeInfoObject(loc, name, abiTypeInfoClassName(type),
                             typeInfoNameSymbol, emission.linkage,
                             [](auto& fieldTypes, auto& fields) {});
    return name;
  }

  const auto cv = cv_qualifiers(pointee);
  std::uint32_t pointeeFlags = 0;
  if (has_const(cv)) pointeeFlags |= kPointeeConstMask;
  if (has_volatile(cv)) pointeeFlags |= kPointeeVolatileMask;
  if (!traits.is_complete(traits.remove_cv(pointee)))
    pointeeFlags |= kPointeeIncompleteMask;
  if (memberPointerClass && !traits.is_complete(memberPointerClass))
    pointeeFlags |= kPointeeIncompleteClassMask;
  if (auto functionType = unqualified_cast<FunctionType>(pointee);
      functionType && functionType->isNoexcept())
    pointeeFlags |= kPointeeNoexceptMask;

  auto pointeeTypeInfo = findOrCreateTypeInfo(pointee);
  auto contextTypeInfo =
      memberPointerClass ? findOrCreateTypeInfo(memberPointerClass) : "";

  auto i32Type = emitter_.integerType(32);

  (void)emitTypeInfoObject(
      loc, name, abiTypeInfoClassName(type), typeInfoNameSymbol,
      emission.linkage, [&](auto& fieldTypes, auto& fields) {
        fieldTypes.push_back(i32Type);
        fields.push_back(emitter_.constantInt(
            loc, i32Type, static_cast<std::int64_t>(pointeeFlags)));

        fieldTypes.push_back(i8PtrType);
        fields.push_back(
            emitter_.addressOfSymbol(loc, i8PtrType, pointeeTypeInfo));

        if (contextTypeInfo.empty()) return;

        fieldTypes.push_back(i8PtrType);
        fields.push_back(
            emitter_.addressOfSymbol(loc, i8PtrType, contextTypeInfo));
      });

  return name;
}

auto Codegen::typeInfoAddress(SourceLocation loc, const Type* type)
    -> ir::ValueRef {
  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto name = findOrCreateTypeInfo(type);
  return emitter_.addressOfSymbol(loc, i8PtrType, name);
}

auto Codegen::findOrCreateNoreturnRuntimeCall(SourceLocation loc,
                                              std::string_view name)
    -> ir::FunctionRef {
  if (auto existing = this->findFunction(name)) {
    return existing;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto funcType = emitter_.functionType({}, {}, false);
  auto linkageAttr = ir::Linkage::External;

  return this->declareFunction(
      loc, {.name = name, .type = funcType, .linkage = linkageAttr});
}

auto Codegen::findOrCreateDynamicCast(SourceLocation loc) -> ir::FunctionRef {
  const std::string_view name = "__dynamic_cast";

  if (auto existing = this->findFunction(name)) {
    return existing;
  }

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  auto funcType = emitter_.functionType(
      {i8PtrType, i8PtrType, i8PtrType, pointerSizedIntType()}, {i8PtrType},
      false);
  auto linkageAttr = ir::Linkage::External;

  return this->declareFunction(
      loc, {.name = name, .type = funcType, .linkage = linkageAttr});
}

auto Codegen::dynamicCastOffsetHint(ClassSymbol* sourceClass,
                                    ClassSymbol* targetClass) -> std::int64_t {
  constexpr std::int64_t kNoHint = -1;
  constexpr std::int64_t kSourceIsNotAPublicBase = -2;
  constexpr std::int64_t kSourceIsARepeatedPublicBase = -3;

  auto info = targetClass->baseSubobjectInfo(sourceClass);

  if (info.publicPathCount == 0) return kSourceIsNotAPublicBase;
  if (info.anyPublicPathIsVirtual) return kNoHint;
  if (info.publicPathCount > 1) return kSourceIsARepeatedPublicBase;
  return static_cast<std::int64_t>(info.publicNonVirtualOffset);
}

auto Codegen::dynamicCastNeedsRuntimeCheck(CppCastExpressionAST* ast) -> bool {
  if (ast->castOp != TokenKind::T_DYNAMIC_CAST) return false;
  if (!ast->type || !ast->expression || !ast->expression->type) return false;

  auto targetObjectType = ast->type;
  auto sourceObjectType = ast->expression->type;

  if (ast->valueCategory == ValueCategory::kPrValue) {
    auto targetPointer = unqualified_cast<PointerType>(ast->type);
    auto sourcePointer = unqualified_cast<PointerType>(ast->expression->type);
    if (!targetPointer || !sourcePointer) return false;
    targetObjectType = targetPointer->elementType();
    sourceObjectType = sourcePointer->elementType();
  }

  return !traits.is_same(traits.remove_cv(targetObjectType),
                         traits.remove_cv(sourceObjectType));
}

auto Codegen::emitDynamicCast(CppCastExpressionAST* ast) -> ir::ValueRef {
  const auto loc = ast->firstSourceLocation();

  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);

  const bool isPointerCast = ast->valueCategory == ValueCategory::kPrValue;

  const auto objectTypeOf = [&](const Type* type) {
    if (!isPointerCast) return type;
    return unqualified_cast<PointerType>(type)->elementType();
  };

  auto sourceObjectType = objectTypeOf(ast->expression->type);
  auto targetObjectType = objectTypeOf(ast->type);

  auto sourceValue = expression(ast->expression).value;
  auto sourceI8 = emitter_.bitcast(loc, i8PtrType, sourceValue);

  const auto resultType = isPointerCast
                              ? convertType(ast->type)
                              : emitter_.pointerType(convertType(ast->type));

  const auto emitCast = [&]() -> ir::ValueRef {
    if (traits.is_void(traits.remove_cv(targetObjectType))) {
      return adjustByVtableWord(loc, sourceI8, -2 * pointerSize());
    }

    auto callee = findOrCreateDynamicCast(loc);
    auto hintType = pointerSizedIntType();

    auto sourceClass =
        unqualified_cast<ClassType>(sourceObjectType)->definition();
    auto targetClass =
        unqualified_cast<ClassType>(targetObjectType)->definition();

    auto hint = emitter_.constantLiteral(
        loc, hintType,
        ir::Initializer::integerValue(
            hintType, dynamicCastOffsetHint(sourceClass, targetClass)));

    std::vector<ir::ValueRef> args{
        sourceI8, typeInfoAddress(loc, sourceClass->type()),
        typeInfoAddress(loc, targetClass->type()), hint};

    return ir::singleValue(
        emitter_.call(implicitLocation(loc),
                      {.callee = this->functionName(callee),
                       .arguments = args,
                       .results = std::vector<ir::TypeRef>{i8PtrType}}));
  };

  if (!isPointerCast) {
    auto castedI8 = emitCast();

    auto failedBlock = newBlock();
    auto succeededBlock = newBlock();

    emitter_.condBranch(loc, emitPointerIsNull(loc, castedI8), failedBlock,
                        succeededBlock);

    emitter_.setInsertionBlock(failedBlock);
    (void)emitter_.call(
        implicitLocation(loc),
        {.callee = this->functionName(
             findOrCreateNoreturnRuntimeCall(loc, "__cxa_bad_cast")),
         .arguments = std::vector<ir::ValueRef>{},
         .results = std::vector<ir::TypeRef>{}});
    emitter_.unreachable(loc);

    emitter_.setInsertionBlock(succeededBlock);
    return emitter_.bitcast(loc, resultType, castedI8);
  }

  auto nullBlock = newBlock();
  auto castBlock = newBlock();
  auto endBlock = newBlock();
  auto endBlockResult = emitter_.addBlockParameter(endBlock, i8PtrType, loc);

  emitter_.condBranch(loc, emitPointerIsNull(loc, sourceI8), nullBlock,
                      castBlock);

  emitter_.setInsertionBlock(nullBlock);
  branch(loc, endBlock,
         std::vector<ir::ValueRef>{emitter_.nullPointer(loc, i8PtrType)});

  emitter_.setInsertionBlock(castBlock);
  branch(loc, endBlock, std::vector<ir::ValueRef>{emitCast()});

  emitter_.setInsertionBlock(endBlock);
  return emitter_.bitcast(loc, resultType, endBlockResult);
}

auto Codegen::emitTypeid(TypeidExpressionAST* ast) -> ir::ValueRef {
  const auto loc = ast->firstSourceLocation();

  auto operand = ast->expression;

  const bool isPolymorphicGlvalue =
      operand->valueCategory != ValueCategory::kPrValue &&
      traits.is_polymorphic(operand->type);

  if (!isPolymorphicGlvalue) {
    return typeInfoAddress(loc, traits.remove_reference(operand->type));
  }

  auto objectPtr = expression(operand).value;

  if (!isPointerDereference(operand)) {
    return emitTypeidOfPolymorphicGlvalue(loc, objectPtr);
  }

  auto failedBlock = newBlock();
  auto succeededBlock = newBlock();

  emitter_.condBranch(loc, emitPointerIsNull(loc, objectPtr), failedBlock,
                      succeededBlock);

  emitter_.setInsertionBlock(failedBlock);
  (void)emitter_.call(
      implicitLocation(loc),
      {.callee = this->functionName(
           findOrCreateNoreturnRuntimeCall(loc, "__cxa_bad_typeid")),
       .arguments = std::vector<ir::ValueRef>{},
       .results = std::vector<ir::TypeRef>{}});
  emitter_.unreachable(loc);

  emitter_.setInsertionBlock(succeededBlock);
  return emitTypeidOfPolymorphicGlvalue(loc, objectPtr);
}

auto Codegen::emitPointerIsNull(SourceLocation loc, ir::ValueRef pointer)
    -> ir::ValueRef {
  auto intType = pointerSizedIntType();
  auto address = emitter_.pointerToInt(loc, intType, pointer);
  auto zero = emitter_.constantInt(loc, intType, 0);
  return emitter_.compareInt(loc, ir::IntPredicate::Equal, address, zero);
}

auto Codegen::emitTypeidOfPolymorphicGlvalue(SourceLocation loc,
                                             ir::ValueRef objectPtr)
    -> ir::ValueRef {
  auto i8Type = emitter_.integerType(8);
  auto i8PtrType = emitter_.pointerType(i8Type);
  auto wordPtrType = emitter_.pointerType(i8PtrType);

  const auto wordSize = pointerSize();

  auto vptrAddr = emitter_.bitcast(loc, wordPtrType, objectPtr);
  auto vptr = emitter_.load(loc, i8PtrType, vptrAddr, wordSize);

  auto wordType = pointerSizedIntType();
  auto offset = emitter_.constantInt(loc, wordType, -wordSize);

  auto slotAddr = emitter_.pointerAdd(loc, i8PtrType, vptr, offset);
  auto slotPtr = emitter_.bitcast(loc, wordPtrType, slotAddr);

  return emitter_.load(loc, i8PtrType, slotPtr, wordSize);
}

}  // namespace cxx
