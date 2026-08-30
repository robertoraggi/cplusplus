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
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

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
    return {.emitDefinition = true,
            .linkage = mlir::cxx::LinkageKind::Internal};

  auto classType = type_cast<ClassType>(type);
  if (!classType) return {};

  auto classSymbol = classType->definition();
  if (hasInternalLinkage(classSymbol))
    return {.emitDefinition = true,
            .linkage = mlir::cxx::LinkageKind::Internal};

  if (!classSymbol->isPolymorphic()) return {};
  return vtableEmission(classSymbol);
}

auto Codegen::findOrCreateAbiTypeInfoVTable(llvm::StringRef abiClassName)
    -> mlir::cxx::GlobalOp {
  auto name =
      std::format("_ZTVN10__cxxabiv1{}E", std::string_view{abiClassName});

  if (auto existing = module_.lookupSymbol<mlir::cxx::GlobalOp>(name)) {
    return existing;
  }

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto arrayType = mlir::cxx::ArrayType::get(context_, i8PtrType, 0);

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);

  return mlir::cxx::GlobalOp::create(
      builder_, builder_.getUnknownLoc(), mlir::TypeRange(), arrayType, true,
      name, mlir::Attribute(), linkageAttr, mlir::IntegerAttr{});
}

auto Codegen::findOrCreateTypeInfoName(const Type* type) -> std::string {
  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeTypeInfoName(type);

  if (module_.lookupSymbol<mlir::cxx::GlobalOp>(name)) return name;

  ExternalNameEncoder contentEncoder{unit_};
  auto content = contentEncoder.encode(type);
  content.push_back('\0');

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto i8Type = builder_.getI8Type();
  auto arrayType = mlir::cxx::ArrayType::get(context_, i8Type, content.size());
  auto linkage = typeInfoEmission(type).linkage;
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkage);

  mlir::cxx::GlobalOp::create(
      builder_, builder_.getUnknownLoc(), mlir::TypeRange(), arrayType, true,
      name,
      builder_.getStringAttr(llvm::StringRef(content.data(), content.size())),
      linkageAttr, mlir::IntegerAttr{});

  return name;
}

auto Codegen::emitTypeInfoObject(
    mlir::Location loc, llvm::StringRef name, llvm::StringRef abiClassName,
    llvm::StringRef typeInfoNameSymbol, mlir::cxx::LinkageKind linkage,
    const std::function<void(mlir::SmallVector<mlir::Type>& fieldTypes,
                             mlir::SmallVector<mlir::Value>& fields)>&
        emitTrailingFields) -> mlir::cxx::GlobalOp {
  auto abiVTable = findOrCreateAbiTypeInfoVTable(abiClassName);

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto wordPtrType = mlir::cxx::PointerType::get(context_, i8PtrType);

  auto recordType = mlir::cxx::ClassType::getNamed(context_, name);

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkage);

  auto global = mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), recordType, true, name,
      mlir::Attribute(), linkageAttr, mlir::IntegerAttr{});

  auto block = builder_.createBlock(&global.getInitializer());
  builder_.setInsertionPointToStart(block);

  auto abiVTableAddr = mlir::cxx::AddressOfOp::create(
      builder_, loc, wordPtrType,
      mlir::FlatSymbolRefAttr::get(context_, abiVTable.getSymName()));

  auto intType = convertType(control()->getIntType());
  auto addressPointIndex = mlir::arith::ConstantOp::create(
      builder_, loc, intType, builder_.getIntegerAttr(intType, 2));

  mlir::SmallVector<mlir::Type> fieldTypes{i8PtrType, i8PtrType};
  mlir::SmallVector<mlir::Value> fields{
      mlir::cxx::PtrAddOp::create(builder_, loc, wordPtrType, abiVTableAddr,
                                  addressPointIndex),
      mlir::cxx::AddressOfOp::create(
          builder_, loc, i8PtrType,
          mlir::FlatSymbolRefAttr::get(context_, typeInfoNameSymbol))};

  emitTrailingFields(fieldTypes, fields);

  (void)recordType.setBody(fieldTypes);

  mlir::Value record = mlir::cxx::UndefOp::create(builder_, loc, recordType);
  for (std::int64_t index = 0; index < std::int64_t(fields.size()); ++index) {
    record = mlir::cxx::InsertValueOp::create(builder_, loc, recordType, record,
                                              fields[index], index);
  }

  mlir::cxx::ReturnOp::create(builder_, loc, record);

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
        static_cast<std::int64_t>(primary.headerWordCount() - index);
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
    mlir::SmallVector<mlir::Type>& fieldTypes,
    mlir::SmallVector<mlir::Value>& fields, mlir::Location loc) {
  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto i32Type = builder_.getI32Type();
  auto offsetFlagsType = pointerSizedIntType();

  auto repetition = classSymbol->baseClassRepetition();
  std::int64_t flags = 0;
  if (repetition.nonDiamondRepeat) flags |= kNonDiamondRepeatMask;
  if (repetition.diamondShaped) flags |= kDiamondShapedMask;

  const auto appendInt = [&](mlir::Type type, std::int64_t value) {
    fieldTypes.push_back(type);
    fields.push_back(mlir::arith::ConstantOp::create(
        builder_, loc, type, builder_.getIntegerAttr(type, value)));
  };

  appendInt(i32Type, flags);
  appendInt(i32Type, static_cast<std::int64_t>(descriptors.size()));

  for (auto& descriptor : descriptors) {
    fieldTypes.push_back(i8PtrType);
    fields.push_back(mlir::cxx::AddressOfOp::create(
        builder_, loc, i8PtrType,
        mlir::FlatSymbolRefAttr::get(context_, descriptor.typeInfo)));

    appendInt(offsetFlagsType, descriptor.offsetFlags);
  }
}

auto Codegen::findOrCreateTypeInfo(const Type* type) -> std::string {
  type = traits.remove_cv(type);

  ExternalNameEncoder encoder{unit_};
  auto name = encoder.encodeTypeInfo(type);

  if (!emittedTypeInfos_.insert(name).second) return name;
  if (module_.lookupSymbol<mlir::cxx::GlobalOp>(name)) return name;

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  const auto declareExternal = [&] {
    builder_.setInsertionPointToStart(module_.getBody());
    auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
        context_, mlir::cxx::LinkageKind::External);
    mlir::cxx::GlobalOp::create(
        builder_, builder_.getUnknownLoc(), mlir::TypeRange(), i8PtrType, true,
        name, mlir::Attribute(), linkageAttr, mlir::IntegerAttr{});
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

  auto loc = classSymbol ? getLocation(classSymbol->location())
                         : builder_.getUnknownLoc();

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
      (void)emitTypeInfoObject(
          loc, name, "20__si_class_type_info", typeInfoNameSymbol,
          emission.linkage, [&](auto& fieldTypes, auto& fields) {
            fieldTypes.push_back(i8PtrType);
            fields.push_back(mlir::cxx::AddressOfOp::create(
                builder_, loc, i8PtrType,
                mlir::FlatSymbolRefAttr::get(context_, baseTypeInfo)));
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

  auto i32Type = builder_.getI32Type();

  (void)emitTypeInfoObject(
      loc, name, abiTypeInfoClassName(type), typeInfoNameSymbol,
      emission.linkage, [&](auto& fieldTypes, auto& fields) {
        fieldTypes.push_back(i32Type);
        fields.push_back(mlir::arith::ConstantOp::create(
            builder_, loc, i32Type,
            builder_.getIntegerAttr(i32Type,
                                    static_cast<std::int64_t>(pointeeFlags))));

        fieldTypes.push_back(i8PtrType);
        fields.push_back(mlir::cxx::AddressOfOp::create(
            builder_, loc, i8PtrType,
            mlir::FlatSymbolRefAttr::get(context_, pointeeTypeInfo)));

        if (contextTypeInfo.empty()) return;

        fieldTypes.push_back(i8PtrType);
        fields.push_back(mlir::cxx::AddressOfOp::create(
            builder_, loc, i8PtrType,
            mlir::FlatSymbolRefAttr::get(context_, contextTypeInfo)));
      });

  return name;
}

auto Codegen::typeInfoAddress(mlir::Location loc, const Type* type)
    -> mlir::Value {
  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto name = findOrCreateTypeInfo(type);
  return mlir::cxx::AddressOfOp::create(
      builder_, loc, i8PtrType, mlir::FlatSymbolRefAttr::get(context_, name));
}

auto Codegen::findOrCreateNoreturnRuntimeCall(mlir::Location loc,
                                              llvm::StringRef name)
    -> mlir::cxx::FuncOp {
  if (auto existing = module_.lookupSymbol<mlir::cxx::FuncOp>(name)) {
    return existing;
  }

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto funcType = mlir::cxx::FunctionType::get(context_, {}, {}, false);
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);

  return mlir::cxx::FuncOp::create(builder_, loc, name, funcType, linkageAttr,
                                   inlineAttr, mlir::cxx::VisibilityAttr{},
                                   mlir::StringAttr{}, mlir::ArrayAttr{},
                                   mlir::ArrayAttr{});
}

auto Codegen::findOrCreateDynamicCast(mlir::Location loc) -> mlir::cxx::FuncOp {
  const llvm::StringRef name = "__dynamic_cast";

  if (auto existing = module_.lookupSymbol<mlir::cxx::FuncOp>(name)) {
    return existing;
  }

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  auto funcType = mlir::cxx::FunctionType::get(
      context_, {i8PtrType, i8PtrType, i8PtrType, pointerSizedIntType()},
      {i8PtrType}, false);
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);

  return mlir::cxx::FuncOp::create(builder_, loc, name, funcType, linkageAttr,
                                   inlineAttr, mlir::cxx::VisibilityAttr{},
                                   mlir::StringAttr{}, mlir::ArrayAttr{},
                                   mlir::ArrayAttr{});
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

auto Codegen::emitDynamicCast(CppCastExpressionAST* ast) -> mlir::Value {
  const auto loc = getLocation(ast->firstSourceLocation());

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  const bool isPointerCast = ast->valueCategory == ValueCategory::kPrValue;

  const auto objectTypeOf = [&](const Type* type) {
    if (!isPointerCast) return type;
    return unqualified_cast<PointerType>(type)->elementType();
  };

  auto sourceObjectType = objectTypeOf(ast->expression->type);
  auto targetObjectType = objectTypeOf(ast->type);

  auto sourceValue = expression(ast->expression).value;
  auto sourceI8 =
      mlir::cxx::BitcastOp::create(builder_, loc, i8PtrType, sourceValue);

  const auto resultType = isPointerCast
                              ? convertType(ast->type)
                              : mlir::Type{mlir::cxx::PointerType::get(
                                    context_, convertType(ast->type))};

  const auto emitCast = [&]() -> mlir::Value {
    if (traits.is_void(traits.remove_cv(targetObjectType))) {
      return adjustByVtableWord(loc, sourceI8, -2 * pointerSize());
    }

    auto callee = findOrCreateDynamicCast(loc);
    auto hintType = pointerSizedIntType();

    auto sourceClass =
        unqualified_cast<ClassType>(sourceObjectType)->definition();
    auto targetClass =
        unqualified_cast<ClassType>(targetObjectType)->definition();

    auto hint = mlir::arith::ConstantOp::create(
        builder_, loc, hintType,
        builder_.getIntegerAttr(
            hintType, dynamicCastOffsetHint(sourceClass, targetClass)));

    mlir::SmallVector<mlir::Value> args{
        sourceI8, typeInfoAddress(loc, sourceClass->type()),
        typeInfoAddress(loc, targetClass->type()), hint};

    return mlir::cxx::CallOp::create(builder_, loc, mlir::TypeRange{i8PtrType},
                                     callee.getSymName(), args)
        .getResult();
  };

  if (!isPointerCast) {
    auto castedI8 = emitCast();

    auto failedBlock = newBlock();
    auto succeededBlock = newBlock();

    mlir::cf::CondBranchOp::create(builder_, loc,
                                   emitPointerIsNull(loc, castedI8),
                                   failedBlock, succeededBlock);

    builder_.setInsertionPointToEnd(failedBlock);
    (void)mlir::cxx::CallOp::create(
        builder_, loc, mlir::TypeRange{},
        findOrCreateNoreturnRuntimeCall(loc, "__cxa_bad_cast").getSymName(),
        mlir::ValueRange{});
    mlir::cxx::UnreachableOp::create(builder_, loc);

    builder_.setInsertionPointToEnd(succeededBlock);
    return mlir::cxx::BitcastOp::create(builder_, loc, resultType, castedI8);
  }

  auto nullBlock = newBlock();
  auto castBlock = newBlock();
  auto endBlock = newBlock();
  endBlock->addArgument(i8PtrType, loc);

  mlir::cf::CondBranchOp::create(
      builder_, loc, emitPointerIsNull(loc, sourceI8), nullBlock, castBlock);

  builder_.setInsertionPointToEnd(nullBlock);
  branch(loc, endBlock,
         mlir::ValueRange{
             mlir::cxx::NullPtrConstantOp::create(builder_, loc, i8PtrType)});

  builder_.setInsertionPointToEnd(castBlock);
  branch(loc, endBlock, mlir::ValueRange{emitCast()});

  builder_.setInsertionPointToEnd(endBlock);
  return mlir::cxx::BitcastOp::create(builder_, loc, resultType,
                                      endBlock->getArgument(0));
}

auto Codegen::emitTypeid(TypeidExpressionAST* ast) -> mlir::Value {
  const auto loc = getLocation(ast->firstSourceLocation());

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

  mlir::cf::CondBranchOp::create(builder_, loc,
                                 emitPointerIsNull(loc, objectPtr), failedBlock,
                                 succeededBlock);

  builder_.setInsertionPointToEnd(failedBlock);
  (void)mlir::cxx::CallOp::create(
      builder_, loc, mlir::TypeRange{},
      findOrCreateNoreturnRuntimeCall(loc, "__cxa_bad_typeid").getSymName(),
      mlir::ValueRange{});
  mlir::cxx::UnreachableOp::create(builder_, loc);

  builder_.setInsertionPointToEnd(succeededBlock);
  return emitTypeidOfPolymorphicGlvalue(loc, objectPtr);
}

auto Codegen::emitPointerIsNull(mlir::Location loc, mlir::Value pointer)
    -> mlir::Value {
  auto intType = pointerSizedIntType();
  auto address = mlir::cxx::PtrToIntOp::create(builder_, loc, intType, pointer);
  auto zero = mlir::arith::ConstantOp::create(
      builder_, loc, intType, builder_.getIntegerAttr(intType, 0));
  return mlir::arith::CmpIOp::create(
      builder_, loc, mlir::arith::CmpIPredicate::eq, address, zero);
}

auto Codegen::emitTypeidOfPolymorphicGlvalue(mlir::Location loc,
                                             mlir::Value objectPtr)
    -> mlir::Value {
  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto wordPtrType = mlir::cxx::PointerType::get(context_, i8PtrType);

  const auto wordSize = pointerSize();

  auto vptrAddr =
      mlir::cxx::BitcastOp::create(builder_, loc, wordPtrType, objectPtr);
  auto vptr =
      mlir::cxx::LoadOp::create(builder_, loc, i8PtrType, vptrAddr, wordSize);

  auto wordType = pointerSizedIntType();
  auto offset = mlir::arith::ConstantOp::create(
      builder_, loc, wordType, builder_.getIntegerAttr(wordType, -wordSize));

  auto slotAddr =
      mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, vptr, offset);
  auto slotPtr =
      mlir::cxx::BitcastOp::create(builder_, loc, wordPtrType, slotAddr);

  return mlir::cxx::LoadOp::create(builder_, loc, i8PtrType, slotPtr, wordSize);
}

}  // namespace cxx
