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
#include <cxx/class_value_abi.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

#include <filesystem>
#include <format>

namespace cxx {
static auto isInAnonymousNamespace(Symbol* symbol) -> bool {
  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto ns = symbol_cast<NamespaceSymbol>(scope)) {
      if (ns->anonNamespaceIndex().has_value()) return true;
    }
  }
  return false;
}

static auto isMemberOfClassTemplateSpecialization(Symbol* symbol) -> bool {
  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto cls = symbol_cast<ClassSymbol>(scope)) {
      if (cls->isSpecialization()) return true;
    }
  }
  return false;
}

static auto isMemberOfExplicitInstantiationDeclaredClass(Symbol* symbol)
    -> bool {
  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto cls = symbol_cast<ClassSymbol>(scope)) {
      if (cls->isExplicitInstantiationDeclared()) return true;
    }
  }
  return false;
}

static auto targetNeedsAppleNameTable(mlir::ModuleOp module) -> bool {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!tripleAttr) return false;
  llvm::Triple triple(tripleAttr.getValue());
  return triple.isAppleMachO();
}

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit,
                 bool debugInfo)
    : context_(&context),
      builder_(&context),
      unit_(unit),
      traits(unit),
      debugInfo_(debugInfo) {
  const auto triple = llvm::Triple(unit->control()->memoryLayout()->triple());
  isWasmTarget_ = triple.getArch() == llvm::Triple::wasm32 ||
                  triple.getArch() == llvm::Triple::wasm64;
}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::getAlignment(const Type* type) -> uint64_t {
  return control()->memoryLayout()->alignmentOf(type).value_or(1);
}

auto Codegen::currentBlockMightHaveTerminator() -> bool {
  auto block = builder_.getInsertionBlock();
  if (!block) {
    cxx_runtime_error("current block is null");
  }
  return block->mightHaveTerminator();
}

auto Codegen::newBlock() -> mlir::Block* {
  auto region = builder_.getBlock()->getParent();
  auto newBlock = new mlir::Block();
  region->getBlocks().push_back(newBlock);
  return newBlock;
}

auto Codegen::newUniqueSymbolName(std::string_view prefix) -> std::string {
  auto& uniqueName = uniqueSymbolNames_[prefix];
  if (uniqueName == 0) {
    uniqueName = 1;
    return std::format("{}{}", prefix, uniqueName);
  }
  return std::format("{}{}", prefix, ++uniqueName);
}

auto Codegen::getFloatAttr(const std::optional<ConstValue>& value,
                           const Type* type) -> std::optional<mlir::FloatAttr> {
  if (value.has_value()) {
    auto ty = traits.remove_cvref(type);

    auto interp = ASTInterpreter{unit_};

    switch (ty->kind()) {
      case TypeKind::kFloat:
        return interp.toFloat(*value).transform(
            [&](float value) { return builder_.getF32FloatAttr(value); });

      case TypeKind::kDouble:
        return interp.toDouble(*value).transform(
            [&](double value) { return builder_.getF64FloatAttr(value); });

      case TypeKind::kLongDouble:
        return interp.toDouble(*value).transform(
            [&](double value) { return builder_.getF64FloatAttr(value); });

      default:
        break;
    }
  }

  return {};
}

auto Codegen::nullMemberObjectPointer() const -> std::int64_t {
  return control()->memoryLayout()->nullMemberObjectPointer();
}

auto Codegen::constValueToAttr(const ConstValue& value, const Type* type)
    -> std::optional<mlir::Attribute> {
  auto interp = ASTInterpreter{unit_};

  if (traits.is_integral_or_unscoped_enum(type)) {
    auto constValue = interp.toInt(value);
    return builder_.getI64IntegerAttr(constValue.value_or(0));
  }

  if (type_cast<MemberObjectPointerType>(type)) {
    auto constValue = interp.toInt(value);
    return builder_.getI64IntegerAttr(
        constValue.value_or(nullMemberObjectPointer()));
  }

  if (auto attr = getFloatAttr(value, type)) {
    return *attr;
  }

  if (traits.is_pointer(type)) {
    if (std::get_if<std::shared_ptr<ConstLabelAddress>>(&value))
      return std::nullopt;
    if (auto intVal = std::get_if<std::intmax_t>(&value)) {
      if (*intVal == 0) return builder_.getUnitAttr();
    }
    return std::nullopt;
  }

  if (traits.is_array(type) || traits.is_class(type)) {
    if (auto constArrayPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto constArray = *constArrayPtr;
      std::vector<mlir::Attribute> elements;
      for (const auto& [elemValue, elemType] : constArray->elements) {
        if (auto attr = constValueToAttr(elemValue, elemType)) {
          elements.push_back(*attr);
        } else {
          return std::nullopt;
        }
      }
      return builder_.getArrayAttr(elements);
    }
  }

  return std::nullopt;
}

auto Codegen::emitConstInitValue(mlir::OpBuilder& builder, mlir::Location loc,
                                 const Type* type, const ConstValue& value)
    -> mlir::Value {
  auto interp = ASTInterpreter{unit_};

  if (traits.is_integral_or_unscoped_enum(type)) {
    auto mlirType = convertType(type);
    auto constValue = interp.toInt(value);
    return mlir::arith::ConstantOp::create(
        builder, loc, mlirType,
        builder.getIntegerAttr(mlirType, constValue.value_or(0)));
  }

  if (type_cast<MemberObjectPointerType>(type)) {
    auto mlirType = convertType(type);
    auto constValue = interp.toInt(value);
    return mlir::arith::ConstantOp::create(
        builder, loc, mlirType,
        builder.getIntegerAttr(mlirType,
                               constValue.value_or(nullMemberObjectPointer())));
  }

  if (traits.is_floating_point(type)) {
    auto mlirType = convertType(type);
    auto floatType = mlir::cast<mlir::FloatType>(mlirType);
    auto constValue = interp.toDouble(value);
    return mlir::arith::ConstantOp::create(
        builder, loc, floatType,
        mlir::FloatAttr::get(floatType, constValue.value_or(0.0)));
  }

  if (traits.is_pointer(type)) {
    auto ptrType = convertType(type);
    auto mlirPtrType = mlir::cast<mlir::cxx::PointerType>(ptrType);

    if (auto addrPtr = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
      auto symbol = (*addrPtr)->symbol();
      auto offset = (*addrPtr)->offset();
      if (symbol_cast<VariableSymbol>(symbol)) {
        if (auto glo = findOrCreateGlobal(symbol)) {
          mlir::Value result = mlir::cxx::AddressOfOp::create(
              builder, loc, mlirPtrType, glo->getSymName());
          if (offset != 0) {
            auto offsetVal = mlir::arith::ConstantOp::create(
                builder, loc, builder.getI64Type(),
                builder.getI64IntegerAttr(offset));
            result = mlir::cxx::PtrAddOp::create(builder, loc, mlirPtrType,
                                                 result, offsetVal);
          }
          return result;
        }
      } else if (auto funcSym = symbol_cast<FunctionSymbol>(symbol)) {
        auto funcOp = findOrCreateFunction(funcSym);
        return mlir::cxx::AddressOfOp::create(builder, loc, mlirPtrType,
                                              funcOp.getSymName());
      }
    }

    if (auto labelAddrPtr =
            std::get_if<std::shared_ptr<ConstLabelAddress>>(&value)) {
      auto funcNameAttr =
          function_ ? mlir::StringAttr::get(context_, function_.getSymName())
                    : mlir::StringAttr{};
      return mlir::cxx::LabelAddressOp::create(
          builder, loc, mlirPtrType, (*labelAddrPtr)->name(),
          mlir::IntegerAttr{}, funcNameAttr);
    }

    if (auto strLitPtr = std::get_if<const StringLiteral*>(&value)) {
      auto stringLiteral = *strLitPtr;
      stringLiteral->initialize(stringLiteral->encoding());
      std::string str(stringLiteral->stringValue());
      str.push_back('\0');

      auto i8Type = mlir::IntegerType::get(context_, 8);
      auto arrayType = mlir::cxx::ArrayType::get(context_, i8Type, str.size());
      auto strAttr =
          builder.getStringAttr(llvm::StringRef(str.data(), str.size()));
      auto strName = builder.getStringAttr(newUniqueSymbolName(".str"));

      {
        auto guard = mlir::OpBuilder::InsertionGuard(builder);
        builder.setInsertionPointToStart(module_.getBody());
        auto linkage = mlir::cxx::LinkageKindAttr::get(
            context_, mlir::cxx::LinkageKind::Internal);
        mlir::cxx::GlobalOp::create(builder, loc, mlir::TypeRange(), arrayType,
                                    true, strName.getValue(), strAttr, linkage,
                                    mlir::IntegerAttr{});
      }

      return mlir::cxx::AddressOfOp::create(builder, loc, mlirPtrType, strName);
    }

    return mlir::cxx::NullPtrConstantOp::create(builder, loc, mlirPtrType);
  }

  if (traits.is_class_or_union(type)) {
    auto classType = type_cast<ClassType>(traits.remove_cv(type));
    auto mlirType = convertType(type);

    if (classType && classType->isUnion()) {
      if (auto initListPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&value)) {
        auto& initList = *initListPtr;
        if (!initList->elements.empty()) {
          auto& [elemValue, elemType] = initList->elements[0];

          bool isZero = false;
          if (auto intVal = std::get_if<std::intmax_t>(&elemValue)) {
            isZero = (*intVal == 0);
          } else if (auto floatVal = std::get_if<float>(&elemValue)) {
            isZero = (*floatVal == 0.0f);
          } else if (auto doubleVal = std::get_if<double>(&elemValue)) {
            isZero = (*doubleVal == 0.0);
          }

          if (isZero) {
            return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
          }

          auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);

          auto unionClassType = mlir::dyn_cast<mlir::cxx::ClassType>(mlirType);
          if (unionClassType && !unionClassType.getBody().empty() &&
              elemVal.getType() == unionClassType.getBody()[0]) {
            auto undef = mlir::cxx::UndefOp::create(builder, loc, mlirType);
            return mlir::cxx::InsertValueOp::create(builder, loc, mlirType,
                                                    undef, elemVal,
                                                    static_cast<int64_t>(0));
          }

          if (unionClassType && !unionClassType.getBody().empty()) {
            auto dstFieldType = unionClassType.getBody()[0];

            if (auto srcArr =
                    mlir::dyn_cast<mlir::cxx::ArrayType>(elemVal.getType())) {
              if (auto dstArr =
                      mlir::dyn_cast<mlir::cxx::ArrayType>(dstFieldType)) {
                if (srcArr.getElementType() == dstArr.getElementType()) {
                  mlir::Value resized = elemVal;
                  if (srcArr.getSize() != dstArr.getSize())
                    resized = mlir::cxx::ReshapeOp::create(builder, loc, dstArr,
                                                           elemVal);
                  auto undef =
                      mlir::cxx::UndefOp::create(builder, loc, mlirType);
                  return mlir::cxx::InsertValueOp::create(
                      builder, loc, mlirType, undef, resized,
                      static_cast<int64_t>(0));
                }
              }
            }

            auto classSymbol = classType->symbol();
            unsigned unionBits =
                static_cast<unsigned>(classSymbol->sizeInBytes()) * 8;
            auto intUnionType = mlir::IntegerType::get(context_, unionBits);

            mlir::Value intRepr;
            if (auto srcInt =
                    mlir::dyn_cast<mlir::IntegerType>(elemVal.getType())) {
              unsigned srcBits = srcInt.getWidth();
              if (srcBits < unionBits)
                intRepr = mlir::arith::ExtUIOp::create(builder, loc,
                                                       intUnionType, elemVal);
              else if (srcBits > unionBits)
                intRepr = mlir::arith::TruncIOp::create(builder, loc,
                                                        intUnionType, elemVal);
              else
                intRepr = elemVal;
            } else if (auto srcFloat =
                           mlir::dyn_cast<mlir::FloatType>(elemVal.getType())) {
              unsigned srcBits = srcFloat.getWidth();
              auto srcIntTy = mlir::IntegerType::get(context_, srcBits);
              mlir::Value asInt = mlir::arith::BitcastOp::create(
                  builder, loc, srcIntTy, elemVal);
              if (srcBits < unionBits)
                intRepr = mlir::arith::ExtUIOp::create(builder, loc,
                                                       intUnionType, asInt);
              else
                intRepr = asInt;
            }

            if (intRepr) {
              mlir::Value fieldVal;
              if (auto dstFloat =
                      mlir::dyn_cast<mlir::FloatType>(dstFieldType)) {
                unsigned dstBits = dstFloat.getWidth();
                mlir::Value bits = intRepr;
                if (unionBits != dstBits) {
                  auto dstIntTy = mlir::IntegerType::get(context_, dstBits);
                  bits = mlir::arith::TruncIOp::create(builder, loc, dstIntTy,
                                                       bits);
                }
                fieldVal = mlir::arith::BitcastOp::create(builder, loc,
                                                          dstFloat, bits);
              } else if (mlir::dyn_cast<mlir::IntegerType>(dstFieldType)) {
                fieldVal = intRepr;
                if (intRepr.getType() != dstFieldType)
                  fieldVal = mlir::arith::TruncIOp::create(
                      builder, loc, dstFieldType, intRepr);
              }
              if (fieldVal) {
                auto undef = mlir::cxx::UndefOp::create(builder, loc, mlirType);
                return mlir::cxx::InsertValueOp::create(
                    builder, loc, mlirType, undef, fieldVal,
                    static_cast<int64_t>(0));
              }
            }
          }

          return mlir::cxx::BitcastOp::create(builder, loc, mlirType, elemVal);
        }
      }
      return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
    }

    if (auto initListPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto& initList = *initListPtr;
      mlir::Value result = mlir::cxx::ZeroOp::create(builder, loc, mlirType);
      auto fieldTypes =
          mlir::dyn_cast<mlir::cxx::ClassType>(mlirType).getBody();
      for (size_t i = 0; i < initList->elements.size(); ++i) {
        auto& [elemValue, elemType] = initList->elements[i];
        auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);
        if (i < fieldTypes.size() && elemVal.getType() != fieldTypes[i]) {
          auto srcArr = mlir::dyn_cast<mlir::cxx::ArrayType>(elemVal.getType());
          auto dstArr = mlir::dyn_cast<mlir::cxx::ArrayType>(fieldTypes[i]);
          if (srcArr && dstArr &&
              srcArr.getElementType() == dstArr.getElementType() &&
              srcArr.getSize() != dstArr.getSize()) {
            elemVal =
                mlir::cxx::ReshapeOp::create(builder, loc, dstArr, elemVal);
          } else if (auto unionClassType =
                         mlir::dyn_cast<mlir::cxx::ClassType>(fieldTypes[i]);
                     unionClassType &&
                     unionClassType.getName().starts_with("union.")) {
            elemVal = mlir::cxx::BitcastOp::create(builder, loc, unionClassType,
                                                   elemVal);
          } else {
            bool isZeroVal = false;
            if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
                    elemVal.getDefiningOp())) {
              if (auto intAttr =
                      mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                isZeroVal = intAttr.getValue().isZero();
            } else if (mlir::isa_and_nonnull<mlir::cxx::ZeroOp>(
                           elemVal.getDefiningOp())) {
              isZeroVal = true;
            }
            if (isZeroVal) continue;

            if (auto intSrc =
                    mlir::dyn_cast<mlir::IntegerType>(elemVal.getType())) {
              if (auto intDst =
                      mlir::dyn_cast<mlir::IntegerType>(fieldTypes[i])) {
                if (intSrc.getWidth() > intDst.getWidth())
                  elemVal = mlir::arith::TruncIOp::create(builder, loc, intDst,
                                                          elemVal);
                else if (intSrc.getWidth() < intDst.getWidth())
                  elemVal = mlir::arith::ExtUIOp::create(builder, loc, intDst,
                                                         elemVal);
              }
            }
          }
        }
        result = mlir::cxx::InsertValueOp::create(
            builder, loc, mlirType, result, elemVal, static_cast<int64_t>(i));
      }
      return result;
    }

    if (auto objectPtr = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
      auto& object = *objectPtr;
      auto classSymbol = classType ? classType->symbol() : nullptr;

      if (classSymbol) {
        auto layout = classSymbol->layout();
        mlir::Value result = mlir::cxx::ZeroOp::create(builder, loc, mlirType);

        for (const auto& field : object->fields()) {
          auto fieldSymbol =
              symbol_cast<FieldSymbol>(const_cast<Symbol*>(field.symbol));
          if (!fieldSymbol) continue;

          int index = 0;
          if (layout) {
            if (auto fi = layout->getFieldInfo(fieldSymbol)) index = fi->index;
          }

          auto fieldVal = emitConstInitValue(builder, loc, fieldSymbol->type(),
                                             field.value);
          result = mlir::cxx::InsertValueOp::create(
              builder, loc, mlirType, result, fieldVal,
              static_cast<int64_t>(index));
        }

        return result;
      }
    }

    return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
  }

  if (traits.is_array(type)) {
    auto mlirType = convertType(type);
    auto cxxArrType = mlir::dyn_cast<mlir::cxx::ArrayType>(mlirType);

    if (auto strLitPtr = std::get_if<const StringLiteral*>(&value)) {
      auto stringLiteral = *strLitPtr;
      stringLiteral->initialize(stringLiteral->encoding());
      std::string str(stringLiteral->stringValue());
      str.push_back('\0');
      auto destSize = cxxArrType ? (size_t)cxxArrType.getSize() : str.size();
      str.resize(destSize, '\0');
      auto i8Type = mlir::IntegerType::get(context_, 8);
      mlir::Value result = mlir::cxx::UndefOp::create(builder, loc, mlirType);
      for (size_t i = 0; i < str.size(); ++i) {
        auto elem = mlir::arith::ConstantOp::create(
            builder, loc, i8Type,
            builder.getIntegerAttr(i8Type, (unsigned char)str[i]));
        result = mlir::cxx::InsertValueOp::create(
            builder, loc, mlirType, result, elem, static_cast<int64_t>(i));
      }
      return result;
    }

    if (auto initListPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto& initList = *initListPtr;
      mlir::Value result = mlir::cxx::ZeroOp::create(builder, loc, mlirType);
      for (size_t i = 0; i < initList->elements.size(); ++i) {
        auto& [elemValue, elemType] = initList->elements[i];
        auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);
        if (cxxArrType) {
          auto dstElemType = cxxArrType.getElementType();
          if (elemVal.getType() != dstElemType) {
            auto srcArr =
                mlir::dyn_cast<mlir::cxx::ArrayType>(elemVal.getType());
            auto dstArr = mlir::dyn_cast<mlir::cxx::ArrayType>(dstElemType);
            if (srcArr && dstArr &&
                srcArr.getElementType() == dstArr.getElementType() &&
                srcArr.getSize() != dstArr.getSize()) {
              elemVal =
                  mlir::cxx::ReshapeOp::create(builder, loc, dstArr, elemVal);
            }
          }
        }
        result = mlir::cxx::InsertValueOp::create(
            builder, loc, mlirType, result, elemVal, static_cast<int64_t>(i));
      }
      return result;
    }
    return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
  }

  auto mlirType = convertType(type);
  return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
}

void Codegen::branch(mlir::Location loc, mlir::Block* block,
                     mlir::ValueRange operands) {
  if (currentBlockMightHaveTerminator()) return;
  mlir::cf::BranchOp::create(builder_, loc, block, operands);
}

auto Codegen::findOrCreateLocal(Symbol* symbol) -> std::optional<mlir::Value> {
  if (auto local = locals_.find(symbol); local != locals_.end()) {
    return local->second;
  }

  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return std::nullopt;

  if (var->isStatic()) return std::nullopt;
  if (!var->parent()->isBlock()) return std::nullopt;

  auto loc = getLocation(var->location());

  if (auto vlaType = type_cast<UnresolvedBoundedArrayType>(var->type())) {
    auto countResult = expression(vlaType->size());
    if (!countResult.value) return std::nullopt;

    auto countVal = countResult.value;
    if (mlir::isa<mlir::cxx::PointerType>(countVal.getType())) {
      auto valueType = convertType(vlaType->size()->type);
      countVal = mlir::cxx::LoadOp::create(builder_, loc, valueType, countVal,
                                           getAlignment(vlaType->size()->type));
    }

    mlir::Value totalElements = countVal;
    const Type* elemType = vlaType->elementType();

    while (auto inner = type_cast<UnresolvedBoundedArrayType>(elemType)) {
      auto innerResult = expression(inner->size());
      if (!innerResult.value) return std::nullopt;
      auto innerVal = innerResult.value;
      if (mlir::isa<mlir::cxx::PointerType>(innerVal.getType())) {
        auto valueType = convertType(inner->size()->type);
        innerVal = mlir::cxx::LoadOp::create(builder_, loc, valueType, innerVal,
                                             getAlignment(inner->size()->type));
      }
      if (innerVal.getType() != totalElements.getType())
        innerVal = mlir::arith::ExtSIOp::create(
            builder_, loc, totalElements.getType(), innerVal);
      totalElements = mlir::arith::MulIOp::create(
          builder_, loc, totalElements.getType(), totalElements, innerVal);
      elemType = inner->elementType();
    }

    auto elementType = convertType(elemType);
    auto ptrType = mlir::cxx::PointerType::get(context_, elementType);
    auto alignment = getAlignment(elemType);

    auto leafSizeBytes = static_cast<int64_t>(
        control()->memoryLayout()->sizeOf(elemType).value_or(1));
    mlir::Value totalBytes = totalElements;
    if (leafSizeBytes > 1) {
      auto sizeConst = mlir::arith::ConstantOp::create(
          builder_, loc, totalElements.getType(),
          builder_.getIntegerAttr(totalElements.getType(), leafSizeBytes));
      totalBytes = mlir::arith::MulIOp::create(
          builder_, loc, totalElements.getType(), totalElements, sizeConst);
    }

    auto allocaOp = mlir::cxx::DynAllocaOp::create(builder_, loc, ptrType,
                                                   totalBytes, alignment);
    locals_.emplace(var, allocaOp);
    return allocaOp;
  }

  auto type = convertType(var->type());
  auto ptrType = mlir::cxx::PointerType::get(context_, type);

  auto allocaOp = mlir::cxx::AllocaOp::create(builder_, loc, ptrType,
                                              getAlignment(var->type()));

  attachDebugInfo(allocaOp, var);

  locals_.emplace(var, allocaOp);

  return allocaOp;
}

auto Codegen::getOrCreateDIScope(Symbol* symbol) -> mlir::LLVM::DIScopeAttr {
  if (!symbol) return {};

  if (auto it = diScopes_.find(symbol); it != diScopes_.end())
    return it->second;

  if (symbol_cast<FunctionParametersSymbol>(symbol))
    return getOrCreateDIScope(symbol->parent());

  if (auto block = symbol_cast<BlockSymbol>(symbol)) {
    if (symbol_cast<FunctionParametersSymbol>(block->parent()) ||
        symbol_cast<FunctionSymbol>(block->parent()))
      return getOrCreateDIScope(block->parent());

    auto parentScope = getOrCreateDIScope(block->parent());
    if (!parentScope) return {};
    auto [filename, line, column] =
        unit_->tokenStartPosition(block->location());
    auto fileAttr = getFileAttr(filename);
    auto lexicalBlock = mlir::LLVM::DILexicalBlockAttr::get(
        context_, parentScope, fileAttr, line, column);
    diScopes_[symbol] = lexicalBlock;
    return lexicalBlock;
  }

  if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto it = funcOps_.find(func); it != funcOps_.end()) {
      if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(it->second.getLoc())) {
        if (auto sp = mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(
                fusedLoc.getMetadata())) {
          diScopes_[symbol] = sp;
          return sp;
        }
      }
    }
  }

  return getFileAttrAt(symbol->location());
}

void Codegen::attachDebugInfo(mlir::cxx::AllocaOp allocaOp, Symbol* symbol,
                              std::string_view name, unsigned arg) {
  if (!debugInfo_) return;
  if (!function_) return;

  auto scope = getOrCreateDIScope(symbol->parent());
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(
      ctx, name.empty() ? to_string(symbol->name()) : name);
  auto file = getFileAttrAt(symbol->location());
  unsigned line = unit_->tokenStartPosition(symbol->location()).line;
  auto typeAttr = convertDebugType(symbol->type());

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr,
      mlir::LLVM::DIFlags::Zero);

  allocaOp->setAttr("cxx.di_local", localVar);
}

void Codegen::attachDebugInfo(mlir::cxx::AllocaOp allocaOp, const Type* type,
                              std::string_view name, unsigned arg,
                              mlir::LLVM::DIFlags flags) {
  if (!debugInfo_) return;
  if (!function_) return;

  auto scope = getOrCreateDIScope(currentFunctionSymbol_);
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(ctx, name);
  auto typeAttr = convertDebugType(type);

  mlir::LLVM::DIFileAttr file;
  unsigned line = 0;
  if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(scope)) {
    file = sp.getFile();
    line = sp.getLine();
  }

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr, flags);

  allocaOp->setAttr("cxx.di_local", localVar);
}

auto Codegen::buildSubroutineTypeAttr(FunctionSymbol* functionSymbol)
    -> mlir::LLVM::DISubroutineTypeAttr {
  auto functionType = type_cast<FunctionType>(functionSymbol->type());

  mlir::SmallVector<mlir::LLVM::DITypeAttr> signatureType;
  signatureType.push_back(convertDebugType(functionType->returnType()));

  if (auto classType = type_cast<ClassType>(functionSymbol->parent()->type());
      classType && !functionSymbol->isStatic()) {
    signatureType.push_back(convertDebugType(traits.add_pointer(classType)));
  }

  for (auto paramType : functionType->parameterTypes()) {
    signatureType.push_back(convertDebugType(paramType));
  }

  return mlir::LLVM::DISubroutineTypeAttr::get(context_, signatureType);
}

void Codegen::buildSubprogramAttr(FunctionSymbol* functionSymbol,
                                  FunctionDefinitionAST* ast,
                                  mlir::cxx::FuncOp func, mlir::Location loc) {
  auto ctx = context_;

  mlir::DistinctAttr id = mlir::DistinctAttr::create(builder_.getUnitAttr());

  mlir::LLVM::DIScopeAttr scope;

  if (functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    scope = mlir::dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(
        convertDebugType(classSymbol->type()));
  }

  mlir::StringAttr name = mlir::StringAttr::get(ctx, func.getSymName());
  mlir::StringAttr linkageName = name;

  auto declaratorId = getDeclaratorId(ast->declarator);

  auto compileUnitAttr = getCompileUnitAttr();

  mlir::LLVM::DIFileAttr fileAttr;
  unsigned line = 0;
  unsigned scopeLine = 0;

  if (declaratorId && declaratorId->firstSourceLocation()) {
    auto funcLoc =
        unit_->tokenStartPosition(declaratorId->firstSourceLocation());
    fileAttr = getFileAttr(funcLoc.fileName);
    line = funcLoc.line;
  }

  if (ast->functionBody) {
    auto bodyLoc = ast->functionBody->firstSourceLocation();
    if (bodyLoc) {
      scopeLine = unit_->tokenStartPosition(bodyLoc).line;
    }
  }

  if (!fileAttr) {
    auto symbolLoc = functionSymbol->location();
    fileAttr = getFileAttrAt(symbolLoc);
    if (symbolLoc) {
      line = unit_->tokenStartPosition(symbolLoc).line;
      scopeLine = line;
    }
  }

  if (!scope) scope = fileAttr;

  auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;

  auto type = buildSubroutineTypeAttr(functionSymbol);

  mlir::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
  mlir::SmallVector<mlir::LLVM::DINodeAttr> annotations;

  auto subprogram = mlir::LLVM::DISubprogramAttr::get(
      ctx, id, compileUnitAttr, scope, name, linkageName, fileAttr, line,
      scopeLine, subprogramFlags, type, retainedNodes, annotations);

  func->setLoc(mlir::FusedLoc::get({loc}, subprogram, ctx));

  diScopes_[functionSymbol] = subprogram;
}

auto Codegen::newTemp(const Type* type, SourceLocation loc)
    -> mlir::cxx::AllocaOp {
  auto ptrType = mlir::cxx::PointerType::get(context_, convertType(type));
  return mlir::cxx::AllocaOp::create(builder_, getLocation(loc), ptrType,
                                     getAlignment(type));
}

void Codegen::pushCleanup() { cleanupStack_.emplace_back(); }

void Codegen::pushFullExpressionCleanup() {
  auto& scope = cleanupStack_.emplace_back();
  scope.isFullExpression = true;
  scope.startBlock = builder_.getInsertionBlock();

  if (scope.startBlock) {
    auto insertionPoint = builder_.getInsertionPoint();
    if (insertionPoint != scope.startBlock->begin())
      scope.startAnchor = &*std::prev(insertionPoint);
  }
}

void Codegen::popCleanup(SourceLocation loc) {
  auto& scope = cleanupStack_.back();
  if (scope.entries.empty() || currentBlockMightHaveTerminator()) {
    cleanupStack_.pop_back();
    return;
  }
  auto mergeBlock = newBlock();
  emitBranchWithCleanups(loc, mergeBlock, cleanupStack_.size() - 1);
  builder_.setInsertionPointToEnd(mergeBlock);
  cleanupStack_.pop_back();
}

void Codegen::emitBranchWithCleanups(SourceLocation loc, mlir::Block* target,
                                     std::size_t targetDepth) {
  if (currentBlockMightHaveTerminator()) return;

  auto mlirLoc = getLocation(loc);

  auto snapshot = collectCleanupSnapshot(targetDepth);

  if (snapshot.addresses.empty()) {
    mlir::cf::BranchOp::create(builder_, mlirLoc, target);
    return;
  }

  mlir::cxx::CleanupBranchOp::create(
      builder_, mlirLoc, snapshot.addresses, snapshot.activeFlags,
      mlir::ArrayAttr::get(context_, snapshot.destructors),
      builder_.getDenseI32ArrayAttr(snapshot.activeFlagIndices), target);
}

void Codegen::addCleanup(mlir::Value address, FunctionSymbol* dtor) {
  for (auto i = cleanupStack_.size(); i > 0; --i) {
    auto& scope = cleanupStack_[i - 1];
    if (scope.isFullExpression) continue;
    scope.entries.push_back({address, dtor});
    return;
  }
}

auto Codegen::allocaInEntryBlock(mlir::Location loc, mlir::Type elementType,
                                 std::uint64_t alignment) -> mlir::Value {
  mlir::OpBuilder::InsertionGuard guard(builder_);
  builder_.setInsertionPointToStart(entryBlock_);
  return mlir::cxx::AllocaOp::create(
      builder_, loc, mlir::cxx::PointerType::get(context_, elementType),
      alignment);
}

void Codegen::hoistAllocaToEntryBlock(mlir::Value address) {
  auto allocaOp = address.getDefiningOp<mlir::cxx::AllocaOp>();
  if (!allocaOp || !entryBlock_) return;
  if (allocaOp->getBlock() == entryBlock_) return;

  for (auto& scope : cleanupStack_) {
    if (scope.startAnchor != allocaOp.getOperation()) continue;

    auto position = allocaOp->getIterator();
    scope.startAnchor = position == allocaOp->getBlock()->begin()
                            ? nullptr
                            : &*std::prev(position);
  }

  allocaOp->moveBefore(entryBlock_, entryBlock_->begin());
}

auto Codegen::createConditionalCleanupFlag(mlir::Location loc,
                                           CleanupScope& scope) -> mlir::Value {
  auto boolType = builder_.getIntegerType(1);
  auto flag = allocaInEntryBlock(loc, boolType, 1);

  mlir::OpBuilder::InsertionGuard guard(builder_);

  if (scope.startAnchor) {
    builder_.setInsertionPointAfter(scope.startAnchor);
  } else if (scope.startBlock &&
             scope.startBlock != flag.getDefiningOp()->getBlock()) {
    builder_.setInsertionPointToStart(scope.startBlock);
  } else {
    builder_.setInsertionPointAfter(flag.getDefiningOp());
  }

  auto inactive = mlir::arith::ConstantOp::create(
      builder_, loc, boolType, builder_.getIntegerAttr(boolType, 0));
  mlir::cxx::StoreOp::create(builder_, loc, inactive, flag, 1);

  return flag;
}

void Codegen::addTemporaryCleanup(mlir::Value address, const Type* type) {
  if (cleanupStack_.empty() || !cleanupStack_.back().isFullExpression) return;
  auto classType = type_cast<ClassType>(traits.remove_cv(type));
  if (!classType || !classType->symbol()) return;
  auto dtor = classType->symbol()->resolvedDefinition()->destructor();
  if (!dtor) return;

  auto& scope = cleanupStack_.back();

  mlir::Value activeFlag;

  if (conditionalEvaluationDepth_ > 0) {
    hoistAllocaToEntryBlock(address);

    auto loc = address.getLoc();
    activeFlag = createConditionalCleanupFlag(loc, scope);

    auto boolType = builder_.getIntegerType(1);
    auto active = mlir::arith::ConstantOp::create(
        builder_, loc, boolType, builder_.getIntegerAttr(boolType, 1));
    mlir::cxx::StoreOp::create(builder_, loc, active, activeFlag, 1);
  }

  scope.entries.push_back({address, completeObjectDtor(dtor), activeFlag});
}

auto Codegen::loadThisPointer(mlir::Location loc, ClassSymbol* classSymbol)
    -> mlir::Value {
  if (!thisValue_ || !classSymbol) return {};

  auto pointerType = control()->getPointerType(classSymbol->type());

  return mlir::cxx::LoadOp::create(
      builder_, loc,
      mlir::cxx::PointerType::get(context_, convertType(classSymbol->type())),
      thisValue_, getAlignment(pointerType));
}

auto Codegen::loadEnclosingObject(mlir::Location loc, ClassSymbol* targetClass,
                                  ClassSymbol*& objectClass) -> mlir::Value {
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

    auto enclosingType = type_cast<ClassType>(traits.remove_cv(
        traits.get_element_type(traits.remove_cv(capturedThis->type()))));

    if (!fieldInfo || !enclosingType || !enclosingType->symbol()) {
      cxx_runtime_error(std::format(
          "closure capturing 'this' has no usable '__this' field for '{}'",
          to_string(targetClass->type())));
    }

    auto fieldAddress =
        memberAddress(loc, object, capturedThis->type(), fieldInfo->index);

    objectClass = enclosingType->symbol()->resolvedDefinition();

    object = mlir::cxx::LoadOp::create(
        builder_, loc,
        mlir::cxx::PointerType::get(context_, convertType(enclosingType)),
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
        if (auto bi = layout->getBaseInfo(baseSym)) return bi->index;
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

auto Codegen::subobjectAddress(mlir::Location loc, mlir::Value objectPtr,
                               ClassSymbol* classSymbol, Symbol* subobject)
    -> mlir::Value {
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
      auto fieldClass = type_cast<ClassType>(traits.remove_cv(field->type()));
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

auto Codegen::subobjectElementAddresses(mlir::Location loc,
                                        mlir::Value subobjectPtr,
                                        const ClassSubobjectShape& shape)
    -> std::vector<mlir::Value> {
  if (shape.elementCount == 1) return {subobjectPtr};

  auto elementPtrType =
      mlir::cxx::PointerType::get(context_, convertType(shape.elementType));
  auto basePtr =
      mlir::cxx::BitcastOp::create(builder_, loc, elementPtrType, subobjectPtr);

  auto indexType = builder_.getIntegerType(64);

  std::vector<mlir::Value> addresses;
  addresses.reserve(shape.elementCount);

  for (std::uint64_t i = 0; i < shape.elementCount; ++i) {
    auto offset = mlir::arith::ConstantOp::create(
        builder_, loc, indexType,
        builder_.getIntegerAttr(indexType, static_cast<std::int64_t>(i)));
    addresses.push_back(mlir::cxx::PtrAddOp::create(
        builder_, loc, elementPtrType, basePtr, offset));
  }

  return addresses;
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

  auto params = constructor->functionParameters();
  if (!params) return args;

  for (auto member : views::members(params)) {
    auto param = symbol_cast<ParameterSymbol>(member);
    if (!param || !param->defaultArgument()) continue;
    args.push_back(expression(param->defaultArgument()));
  }

  return args;
}

void Codegen::emitSubobjectDestruction(SourceLocation loc,
                                       mlir::Value objectPtr,
                                       ClassSymbol* classSymbol,
                                       Symbol* subobject) {
  if (!isImplicitlyInitializedSubobject(classSymbol, subobject)) return;

  auto shape = classSubobjectShape(subobjectType(subobject));
  if (!shape) return;

  auto dtor = shape->classSymbol->destructor();
  if (!dtor) return;

  if (symbol_cast<FieldSymbol>(subobject)) dtor = completeObjectDtor(dtor);

  auto mlirLoc = getLocation(loc);

  auto subobjectPtr =
      subobjectAddress(mlirLoc, objectPtr, classSymbol, subobject);
  if (!subobjectPtr) return;

  auto addresses = subobjectElementAddresses(mlirLoc, subobjectPtr, *shape);

  for (auto it = addresses.rbegin(); it != addresses.rend(); ++it)
    (void)emitCall(loc, dtor, {*it}, {});
}

void Codegen::emitSubobjectDefaultConstruction(SourceLocation loc,
                                               mlir::Value objectPtr,
                                               ClassSymbol* classSymbol,
                                               Symbol* subobject) {
  if (!isImplicitlyInitializedSubobject(classSymbol, subobject)) return;

  auto shape = classSubobjectShape(subobjectType(subobject));
  if (!shape) return;

  auto defaultConstructor = shape->classSymbol->defaultConstructor();
  if (!defaultConstructor) return;

  auto mlirLoc = getLocation(loc);

  auto subobjectPtr =
      subobjectAddress(mlirLoc, objectPtr, classSymbol, subobject);
  if (!subobjectPtr) return;

  const auto completeObject = symbol_cast<FieldSymbol>(subobject) != nullptr;

  for (auto address : subobjectElementAddresses(mlirLoc, subobjectPtr, *shape))
    (void)emitCtorCall(loc, defaultConstructor, address,
                       defaultConstructorArguments(defaultConstructor),
                       completeObject);
}

Codegen::FullExpression::FullExpression(Codegen& gen, SourceLocation endLoc)
    : gen_(gen), endLoc_(endLoc) {
  gen_.pushFullExpressionCleanup();
}

Codegen::FullExpression::~FullExpression() { gen_.popCleanup(endLoc_); }

auto Codegen::takeResultObject(ExpressionAST* ast) -> mlir::Value {
  if (!ast || resultObjectOwner_ != ast) return {};
  resultObjectOwner_ = nullptr;
  resultObjectInitialized_ = true;
  return std::exchange(resultObjectAddress_, mlir::Value{});
}

auto Codegen::emitPrvalueInto(mlir::Value object, const Type* objectType,
                              ExpressionAST* ast, SourceLocation loc) -> bool {
  auto resultObject = ResultObject{*this, ast, object};

  auto result = expression(ast);

  if (resultObject.wasConsumed()) return true;
  if (!result.value) return false;

  auto objectMlirType = convertType(objectType);
  auto value = result.value;

  const bool yieldedClassAddress =
      traits.is_class(objectType) && value.getType() != objectMlirType &&
      mlir::isa<mlir::cxx::PointerType>(value.getType());

  if (yieldedClassAddress) {
    value =
        mlir::cxx::LoadOp::create(builder_, getLocation(loc), objectMlirType,
                                  value, getAlignment(objectType));
  }

  mlir::cxx::StoreOp::create(builder_, getLocation(loc), value, object,
                             getAlignment(objectType));

  return true;
}

Codegen::ResultObject::ResultObject(Codegen& gen, ExpressionAST* ast,
                                    mlir::Value address)
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
    auto depthAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context_, 64), i - 1);
    auto& scope = cleanupStack_[i - 1];
    for (auto jt = scope.entries.rbegin(); jt != scope.entries.rend(); ++jt) {
      auto funcOp = findOrCreateFunction(jt->destructor);
      snapshot.addresses.push_back(jt->address);
      snapshot.destructors.push_back(
          mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
      snapshot.depths.push_back(depthAttr);

      if (!jt->activeFlag) {
        snapshot.activeFlagIndices.push_back(-1);
        continue;
      }

      snapshot.activeFlagIndices.push_back(
          static_cast<std::int32_t>(snapshot.activeFlags.size()));
      snapshot.activeFlags.push_back(jt->activeFlag);
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
  return isWasmTarget_;
}

auto Codegen::classifyClassValueAbi(const Type* type) -> ClassValueAbi {
  return cxx::classifyClassValueAbi(unit_, type);
}

auto Codegen::classValueAddress(SourceLocation loc, const Type* type,
                                mlir::Value value) -> mlir::Value {
  if (mlir::isa<mlir::cxx::PointerType>(value.getType())) return value;
  auto temp = newTemp(type, loc);
  mlir::cxx::StoreOp::create(builder_, getLocation(loc), value, temp,
                             getAlignment(type));
  return temp;
}

auto Codegen::abiLowerClassArgument(SourceLocation loc, const Type* paramType,
                                    mlir::Value value) -> mlir::Value {
  auto mlirLoc = getLocation(loc);
  const auto abi = classifyClassValueAbi(paramType);

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct: {
      auto expectedType = convertType(paramType);
      if (value.getType() != expectedType &&
          mlir::isa<mlir::cxx::PointerType>(value.getType())) {
        return mlir::cxx::LoadOp::create(builder_, mlirLoc, expectedType, value,
                                         getAlignment(paramType));
      }
      return value;
    }

    case ClassValueAbi::Kind::Empty:
      return {};

    case ClassValueAbi::Kind::Indirect:
      return classValueAddress(loc, paramType, value);

    case ClassValueAbi::Kind::Scalar: {
      auto address = classValueAddress(loc, paramType, value);
      auto scalarType = convertType(abi.scalarType);
      auto scalarPtrType = mlir::cxx::PointerType::get(context_, scalarType);
      auto castOp = mlir::cxx::BitcastOp::create(builder_, mlirLoc,
                                                 scalarPtrType, address);
      return mlir::cxx::LoadOp::create(builder_, mlirLoc, scalarType, castOp,
                                       getAlignment(abi.scalarType));
    }
  }

  return value;
}

auto Codegen::abiPrepareResult(SourceLocation loc, const Type* returnType,
                               mlir::SmallVector<mlir::Type>& resultTypes,
                               mlir::Value resultObject) -> mlir::Value {
  if (traits.is_void(returnType)) return {};

  const auto abi = classifyClassValueAbi(returnType);

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct:
      resultTypes.push_back(convertType(returnType));
      return {};

    case ClassValueAbi::Kind::Scalar:
      resultTypes.push_back(convertType(abi.scalarType));
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
                              mlir::cxx::CallOp callOp, mlir::Value sretTemp)
    -> ExpressionResult {
  if (traits.is_void(returnType)) return {};

  auto mlirLoc = getLocation(loc);
  const auto abi = classifyClassValueAbi(returnType);

  switch (abi.kind) {
    case ClassValueAbi::Kind::Direct:
      return {callOp.getResult()};

    case ClassValueAbi::Kind::Indirect:
      return {mlir::cxx::LoadOp::create(builder_, mlirLoc,
                                        convertType(returnType), sretTemp,
                                        getAlignment(returnType))};

    case ClassValueAbi::Kind::Scalar: {
      auto temp = newTemp(returnType, loc);
      auto scalarType = convertType(abi.scalarType);
      auto scalarPtrType = mlir::cxx::PointerType::get(context_, scalarType);
      auto castOp =
          mlir::cxx::BitcastOp::create(builder_, mlirLoc, scalarPtrType, temp);
      mlir::cxx::StoreOp::create(builder_, mlirLoc, callOp.getResult(), castOp,
                                 getAlignment(abi.scalarType));
      return {mlir::cxx::LoadOp::create(builder_, mlirLoc,
                                        convertType(returnType), temp,
                                        getAlignment(returnType))};
    }

    case ClassValueAbi::Kind::Empty: {
      auto temp = newTemp(returnType, loc);
      return {mlir::cxx::LoadOp::create(builder_, mlirLoc,
                                        convertType(returnType), temp,
                                        getAlignment(returnType))};
    }
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

auto Codegen::emitVirtualBaseAddress(mlir::Location loc, mlir::Value objectPtr,
                                     ClassSymbol* fromClass,
                                     ClassSymbol* vbaseClass) -> mlir::Value {
  int vbaseIndex = 0;
  if (auto fromLayout = fromClass->layout()) {
    for (auto baseClass : fromLayout->virtualBases()) {
      if (baseClass == vbaseClass) break;
      ++vbaseIndex;
    }
  }

  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  const auto slotByteOffset = -wordSize * (3 + vbaseIndex);

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  auto objectI8 =
      mlir::cxx::BitcastOp::create(builder_, loc, i8PtrType, objectPtr);
  auto adjusted = adjustByVtableWord(loc, objectI8, slotByteOffset);

  auto vbasePtrType =
      mlir::cxx::PointerType::get(context_, convertType(vbaseClass->type()));
  return mlir::cxx::BitcastOp::create(builder_, loc, vbasePtrType, adjusted);
}

auto Codegen::adjustByVtableWord(mlir::Location loc, mlir::Value objectPtrI8,
                                 std::int64_t byteOffset) -> mlir::Value {
  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto i8PtrPtrType = mlir::cxx::PointerType::get(context_, i8PtrType);

  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  auto wordType =
      mlir::IntegerType::get(context_, static_cast<unsigned>(wordSize * 8));

  auto vptrAddr =
      mlir::cxx::BitcastOp::create(builder_, loc, i8PtrPtrType, objectPtrI8);
  auto vptr =
      mlir::cxx::LoadOp::create(builder_, loc, i8PtrType, vptrAddr, wordSize);

  auto offsetConstOp = mlir::arith::ConstantOp::create(
      builder_, loc, wordType, builder_.getIntegerAttr(wordType, byteOffset));
  auto slotAddr = mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, vptr,
                                              offsetConstOp);
  auto wordPtrType = mlir::cxx::PointerType::get(context_, wordType);
  auto slotPtr =
      mlir::cxx::BitcastOp::create(builder_, loc, wordPtrType, slotAddr);
  auto word =
      mlir::cxx::LoadOp::create(builder_, loc, wordType, slotPtr, wordSize);

  return mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, objectPtrI8,
                                     word);
}

auto Codegen::emitBaseClassAddress(mlir::Location loc, mlir::Value objectPtr,
                                   ClassSymbol* fromClass,
                                   ClassSymbol* targetClass) -> mlir::Value {
  if (!objectPtr || !fromClass || !targetClass) return objectPtr;
  if (!mlir::isa<mlir::cxx::PointerType>(objectPtr.getType())) return objectPtr;

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
      int index = 0;
      if (auto layout = currentClass->layout()) {
        if (auto baseInfo = layout->getBaseInfo(baseClass)) {
          index = baseInfo->index;
        }
      }
      current = memberAddress(loc, current, baseClass->type(), index);
    }

    currentClass = baseClass;
  }

  return current;
}

auto Codegen::emitDerivedClassAddress(mlir::Location loc, mlir::Value objectPtr,
                                      ClassSymbol* fromClass,
                                      ClassSymbol* targetClass) -> mlir::Value {
  if (!objectPtr || !fromClass || !targetClass) return objectPtr;
  if (!mlir::isa<mlir::cxx::PointerType>(objectPtr.getType())) return objectPtr;

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

  auto derivedPtrType =
      mlir::cxx::PointerType::get(context_, convertType(targetClass->type()));

  if (byteOffset == 0) {
    return mlir::cxx::BitcastOp::create(builder_, loc, derivedPtrType,
                                        objectPtr);
  }

  auto i8PtrType = mlir::cxx::PointerType::get(context_, builder_.getI8Type());
  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  auto wordType =
      mlir::IntegerType::get(context_, static_cast<unsigned>(wordSize * 8));

  auto objectI8 =
      mlir::cxx::BitcastOp::create(builder_, loc, i8PtrType, objectPtr);
  auto offsetOp = mlir::arith::ConstantOp::create(
      builder_, loc, wordType, builder_.getIntegerAttr(wordType, -byteOffset));
  auto adjusted =
      mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, objectI8, offsetOp);

  return mlir::cxx::BitcastOp::create(builder_, loc, derivedPtrType, adjusted);
}

auto Codegen::computeFunctionSignature(FunctionSymbol* functionSymbol)
    -> mlir::cxx::FunctionType {
  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  if (!functionType) return {};
  return computeFunctionSignature(functionType, functionSymbol);
}

auto Codegen::computeFunctionSignature(const FunctionType* functionType,
                                       FunctionSymbol* functionSymbol)
    -> mlir::cxx::FunctionType {
  const auto returnType = functionType->returnType();
  const auto needsExitValue = !traits.is_void(returnType);
  const auto returnAbi = classifyClassValueAbi(returnType);

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  if (returnAbi.kind == ClassValueAbi::Kind::Indirect) {
    inputTypes.push_back(
        mlir::cxx::PointerType::get(context_, convertType(returnType)));
  }

  if (functionSymbol && functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    inputTypes.push_back(mlir::cxx::PointerType::get(
        context_, convertType(classSymbol->type())));
  }

  for (auto paramTy : functionType->parameterTypes()) {
    const auto paramAbi = classifyClassValueAbi(paramTy);
    switch (paramAbi.kind) {
      case ClassValueAbi::Kind::Direct:
        inputTypes.push_back(convertType(paramTy));
        break;
      case ClassValueAbi::Kind::Empty:
        break;
      case ClassValueAbi::Kind::Scalar:
        inputTypes.push_back(convertType(paramAbi.scalarType));
        break;
      case ClassValueAbi::Kind::Indirect:
        inputTypes.push_back(
            mlir::cxx::PointerType::get(context_, convertType(paramTy)));
        break;
    }
  }

  if (functionSymbol && structorReturnsThis(functionSymbol)) {
    resultTypes.push_back(inputTypes.front());
  } else if (needsExitValue) {
    switch (returnAbi.kind) {
      case ClassValueAbi::Kind::Direct:
        resultTypes.push_back(convertType(returnType));
        break;
      case ClassValueAbi::Kind::Scalar:
        resultTypes.push_back(convertType(returnAbi.scalarType));
        break;
      case ClassValueAbi::Kind::Indirect:
      case ClassValueAbi::Kind::Empty:
        break;
    }
  }

  return mlir::cxx::FunctionType::get(context_, inputTypes, resultTypes,
                                      functionType->isVariadic());
}

void Codegen::reportDeferredBodyDiagnostics(FunctionSymbol* functionSymbol) {
  if (!functionSymbol) return;

  for (const auto& diagnostic :
       unit_->takeDeferredBodyDiagnostics(functionSymbol))
    unit_->diagnosticsClient()->report(diagnostic);
}

auto Codegen::findOrCreateFunction(FunctionSymbol* functionSymbol)
    -> mlir::cxx::FuncOp {
  auto canonicalSymbol = functionSymbol->canonical();
  auto emittedSymbol = functionSymbol;

  if (!functionSymbol->isSpecialization()) {
    emittedSymbol = canonicalSymbol;
  }

  if (auto it = funcOps_.find(emittedSymbol); it != funcOps_.end()) {
    return it->second;
  }

  const auto functionType = type_cast<FunctionType>(emittedSymbol->type());
  if (!functionType) {
    return {};
  }

  auto funcType = computeFunctionSignature(emittedSymbol);

  std::string name;
  bool isStructor = false;

  if (auto externalName = emittedSymbol->externalName()) {
    name = externalName->name();
  } else if (emittedSymbol->hasCLinkage()) {
    name = to_string(emittedSymbol->name());
  } else {
    ExternalNameEncoder encoder;
    name = encoder.encode(emittedSymbol);
    isStructor = emittedSymbol->isConstructor() ||
                 name_cast<DestructorId>(emittedSymbol->name());
  }

  mlir::cxx::VisibilityAttr visibilityAttr;
  if (emittedSymbol->hasHiddenVisibility()) {
    visibilityAttr =
        mlir::cxx::VisibilityAttr::get(context_, mlir::cxx::Visibility::Hidden);
  }

  mlir::StringAttr aliasNameAttr;
  if (auto aliasName = emittedSymbol->aliasName()) {
    aliasNameAttr = mlir::StringAttr::get(context_, aliasName->name());
  } else if (isStructor &&
             (emittedSymbol->isDefined() || emittedSymbol->definition()) &&
             !emittedSymbol->completeObjectVariant() &&
             !emittedSymbol->isStructorVariant()) {
    ExternalNameEncoder encoder;
    encoder.setStructorVariant(ExternalNameEncoder::StructorVariant::Base);
    aliasNameAttr =
        mlir::StringAttr::get(context_, encoder.encode(emittedSymbol));
  }

  if (auto existingFunc = module_.lookupSymbol<mlir::cxx::FuncOp>(name)) {
    funcOps_.insert_or_assign(emittedSymbol, existingFunc);
    enqueueFunctionBody(emittedSymbol);
    return existingFunc;
  }

  const auto loc = getLocation(functionSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  if (emittedSymbol->isInline()) {
    inlineKind = mlir::cxx::InlineKind::InlineHint;
  }

  auto inlineAttr = mlir::cxx::InlineKindAttr::get(context_, inlineKind);

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (emittedSymbol->isStatic() && !emittedSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (isInAnonymousNamespace(emittedSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (emittedSymbol->isInline()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (emittedSymbol->isSpecialization()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (isMemberOfClassTemplateSpecialization(emittedSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (emittedSymbol->isDefaulted()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (emittedSymbol->isStructorVariant()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  }

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkageKind);

  auto func = mlir::cxx::FuncOp::create(
      builder_, loc, name, funcType, linkageAttr, inlineAttr, visibilityAttr,
      aliasNameAttr, mlir::ArrayAttr{}, mlir::ArrayAttr{});

  funcOps_.insert_or_assign(emittedSymbol, func);

  enqueueFunctionBody(emittedSymbol);

  return func;
}

void Codegen::enqueueFunctionBody(FunctionSymbol* symbol) {
  auto target = symbol->isSpecialization() ? symbol : symbol->canonical();
  target = target->resolvedDefinition();
  if (!target->declaration()) return;
  if (!target->isInline() &&
      isMemberOfExplicitInstantiationDeclaredClass(target)) {
    return;
  }
  if (!enqueuedFunctions_.insert(target).second) return;
  pendingFunctions_.push_back(target);
}

void Codegen::processPendingFunctions() {
  while (!pendingFunctions_.empty()) {
    auto sym = pendingFunctions_.back();
    pendingFunctions_.pop_back();

    auto target = sym->resolvedDefinition();

    if (auto funcDecl = target->declaration()) {
      declaration(funcDecl);
    }

    if (sym->parent() && sym->parent()->isClass()) {
      auto classSymbol = symbol_cast<ClassSymbol>(sym->parent());
      if (classSymbol) {
        generateVTable(classSymbol);
      }
    }
  }
}

auto Codegen::findOrCreateGlobal(Symbol* symbol)
    -> std::optional<mlir::cxx::GlobalOp> {
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

  const auto loc = getLocation(variableSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (variableSymbol->isStatic() && !variableSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (isInAnonymousNamespace(variableSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (variableSymbol->isInline()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (variableSymbol->isSpecialization()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (isMemberOfClassTemplateSpecialization(variableSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (variableSymbol->isStatic() &&
             variableSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::External;
  }

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkageKind);

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

    ExternalNameEncoder encoder;
    name = encoder.encode(symbol, suffix);
  }

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(varType);

  mlir::Attribute initializer;
  bool needsRegionInit = false;

  auto value = defVar->constValue();

  if (value.has_value()) {
    auto interp = ASTInterpreter{unit_};

    if (traits.is_integral_or_unscoped_enum(defVar->type())) {
      auto constValue = interp.toInt(*value);
      initializer = builder_.getI64IntegerAttr(constValue.value_or(0));
    } else if (type_cast<MemberObjectPointerType>(defVar->type())) {
      if (auto attr = constValueToAttr(*value, defVar->type()))
        initializer = *attr;
    } else if (auto attr = getFloatAttr(value, defVar->type())) {
      initializer = attr.value();
    } else if (traits.is_array(defVar->type())) {
      if (auto constArrayPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&*value)) {
        auto constArray = *constArrayPtr;
        std::vector<mlir::Attribute> elements;
        bool allConverted = true;

        for (const auto& [elemValue, elemType] : constArray->elements) {
          if (auto attr = constValueToAttr(elemValue, elemType)) {
            elements.push_back(*attr);
          } else {
            allConverted = false;
            break;
          }
        }

        if (allConverted) {
          initializer = builder_.getArrayAttr(elements);
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

        initializer =
            builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));

        if (auto arr = type_cast<BoundedArrayType>(defVar->type())) {
          auto destSize = static_cast<size_t>(arr->size());
          if (str.size() != destSize) {
            str.resize(destSize, '\0');
            initializer =
                builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
          }
        }
      }
    } else if (traits.is_class(defVar->type())) {
      needsRegionInit = true;
    } else if (traits.is_pointer(defVar->type())) {
      if (auto attr = constValueToAttr(*value, defVar->type())) {
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

  if (!initializer && !isExternalOnly) {
    if (type_cast<MemberObjectPointerType>(defVar->type()))
      initializer = builder_.getI64IntegerAttr(nullMemberObjectPointer());
    else
      initializer = mlir::LLVM::ZeroAttr::get(context_);
  }

  const auto isConstant =
      variableSymbol->isConstexpr() || traits.is_const(defVar->type());

  mlir::IntegerAttr alignmentAttr;

  auto var = mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), varType, isConstant,
      llvm::StringRef(name), initializer, linkageAttr, alignmentAttr);

  globalOps_.insert_or_assign(canonicalVar, var);

  if (needsRegionInit && value.has_value()) {
    auto& region = var.getInitializer();
    auto block = new mlir::Block();
    region.push_back(block);
    mlir::OpBuilder initBuilder(block, block->begin());
    auto result = emitConstInitValue(initBuilder, loc, defVar->type(), *value);
    mlir::cxx::ReturnOp::create(initBuilder, loc, result);
  }

  return var;
}

auto Codegen::findOrCreateExternField(FieldSymbol* field)
    -> mlir::cxx::GlobalOp {
  if (auto it = externFieldGlobalOps_.find(field);
      it != externFieldGlobalOps_.end()) {
    return it->second;
  }

  auto varType = convertType(field->type());
  const auto loc = getLocation(field->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);

  ExternalNameEncoder encoder;
  auto name = encoder.encode(field);

  const auto isConstant =
      field->isConstexpr() || traits.is_const(field->type());

  mlir::IntegerAttr alignmentAttr;

  auto var = mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), varType, isConstant,
      llvm::StringRef(name), /*initializer=*/mlir::Attribute(), linkageAttr,
      alignmentAttr);

  externFieldGlobalOps_.insert_or_assign(field, var);

  return var;
}

void Codegen::emitGlobalVarInit(VariableSymbol* var,
                                mlir::cxx::GlobalOp global) {
  auto canonicalVar = var->canonical();

  VariableSymbol* defVar = canonicalVar->resolvedDefinition();

  if (defVar->isExtern()) return;
  if (defVar->constValue().has_value()) return;
  if (!defVar->constructor() && !defVar->initializer()) return;

  if (!emittedGlobalVarInits_.insert(canonicalVar).second) return;

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToEnd(module_.getBody());

  const auto loc = getLocation(defVar->location());

  std::string name = "__cxx_global_var_init";
  if (globalVarInitCount_ > 0) {
    name = std::format("__cxx_global_var_init.{}", globalVarInitCount_);
  }
  ++globalVarInitCount_;

  auto funcType = mlir::cxx::FunctionType::get(
      context_, llvm::ArrayRef<mlir::Type>{}, llvm::ArrayRef<mlir::Type>{},
      /*isVariadic=*/false);

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::Internal);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);

  auto func = mlir::cxx::FuncOp::create(
      builder_, loc, name, funcType, linkageAttr, inlineAttr,
      mlir::cxx::VisibilityAttr{}, mlir::StringAttr{}, mlir::ArrayAttr{},
      mlir::ArrayAttr{});

  mlir::cxx::GlobalCtorOp::create(builder_, loc, func.getSymName());

  auto entryBlock = builder_.createBlock(&func.getBody());
  auto exitBlock = builder_.createBlock(&func.getBody());

  mlir::cxx::AllocaOp exitValue;
  std::unordered_map<Symbol*, mlir::Value> locals;
  std::unordered_map<const Name*, int> staticLocalCounts;
  std::vector<CleanupScope> cleanupStack;
  FunctionSymbol* functionSymbol = nullptr;
  mlir::Value thisValue;

  std::swap(function_, func);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(exitValue_, exitValue);
  std::swap(locals_, locals);
  std::swap(staticLocalCounts_, staticLocalCounts);
  std::swap(cleanupStack_, cleanupStack);
  std::swap(currentFunctionSymbol_, functionSymbol);
  std::swap(thisValue_, thisValue);

  builder_.setInsertionPointToEnd(entryBlock_);

  auto ptrType =
      mlir::cxx::PointerType::get(context_, convertType(defVar->type()));
  mlir::Value addr = mlir::cxx::AddressOfOp::create(builder_, loc, ptrType,
                                                    global.getSymName());

  auto initializer = defVar->initializer();

  const auto initLoc =
      initializer ? initializer->firstSourceLocation() : defVar->location();

  {
    auto fullExpression = FullExpression{*this, initLoc};

    if (auto ctor = defVar->constructor()) {
      std::vector<ExpressionResult> args;

      if (initializer) {
        if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
          for (auto it = paren->expressionList; it; it = it->next) {
            args.push_back(expression(it->value));
          }
        } else if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
          for (auto it = braced->expressionList; it; it = it->next) {
            args.push_back(expression(it->value));
          }
        } else if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
          if (auto braced = ast_cast<BracedInitListAST>(equal->expression)) {
            for (auto it = braced->expressionList; it; it = it->next) {
              args.push_back(expression(it->value));
            }
          } else {
            args.push_back(expression(equal->expression));
          }
        }
      }

      (void)emitCtorCall(initLoc, ctor, addr, std::move(args),
                         /*completeObject=*/true);
    } else if (initializer) {
      ExpressionAST* initExpr = nullptr;
      if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
        initExpr = equal->expression;
      } else if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
        if (paren->expressionList && !paren->expressionList->next) {
          initExpr = paren->expressionList->value;
        }
      } else {
        initExpr = initializer;
      }

      if (initExpr) {
        (void)emitPrvalueInto(addr, defVar->type(), initExpr, initLoc);
      }
    }
  }

  if (traits.is_class(defVar->type())) {
    auto classType = type_cast<ClassType>(traits.remove_cv(defVar->type()));
    if (classType && classType->symbol()) {
      if (auto dtor = classType->symbol()->destructor()) {
        emitGlobalVarDtorRegistration(defVar, completeObjectDtor(dtor), global,
                                      loc);
      }
    }
  }

  emitBranchWithCleanups(initLoc, exitBlock_, 0);

  builder_.setInsertionPointToEnd(exitBlock_);
  mlir::cxx::ReturnOp::create(builder_, loc);

  resolveLabels();

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

auto Codegen::getCompileUnitAttr() -> mlir::LLVM::DICompileUnitAttr {
  if (compileUnitAttr_) return compileUnitAttr_;

  auto ctx = context_;

  auto distinct = mlir::DistinctAttr::create(builder_.getUnitAttr());

  auto sourceLanguage = unit_->language() == LanguageKind::kCXX
                            ? llvm::dwarf::DW_LANG_C_plus_plus_20
                            : llvm::dwarf::DW_LANG_C;

  auto fileAttr = getOrCreateFileAttr(unit_->fileName());
  auto producer = mlir::StringAttr::get(ctx, "cxx");
  auto isOptimized = false;
  auto emissionKind = mlir::LLVM::DIEmissionKind::Full;

  mlir::LLVM::DINameTableKind nameTableKind =
      mlir::LLVM::DINameTableKind::Default;

  if (targetNeedsAppleNameTable(module_)) {
    nameTableKind = mlir::LLVM::DINameTableKind::Apple;
  }

  auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(
      distinct, sourceLanguage, fileAttr, producer, isOptimized, emissionKind,
      nameTableKind);

  compileUnitAttr_ = compileUnit;

  return compileUnit;
}

auto Codegen::getOrCreateFileAttr(const std::string& filename)
    -> mlir::LLVM::DIFileAttr {
  if (auto it = fileAttrs_.find(filename); it != fileAttrs_.end()) {
    return it->second;
  }

  auto filePath = std::filesystem::path{filename};

  auto attr = mlir::LLVM::DIFileAttr::get(
      context_, filePath.filename().string(), filePath.parent_path().string());

  fileAttrs_.insert_or_assign(filename, attr);

  return attr;
}

auto Codegen::getFileAttr(const std::string& filename)
    -> mlir::LLVM::DIFileAttr {
  if (filename.empty()) return getCompileUnitAttr().getFile();

  return getOrCreateFileAttr(filename);
}

auto Codegen::getFileAttr(std::string_view filename) -> mlir::LLVM::DIFileAttr {
  return getFileAttr(std::string{filename});
}

auto Codegen::getFileAttrAt(SourceLocation location) -> mlir::LLVM::DIFileAttr {
  if (!location) return getCompileUnitAttr().getFile();

  return getFileAttr(unit_->tokenStartPosition(location).fileName);
}

auto Codegen::getLocation(SourceLocation location) -> mlir::Location {
  auto [filename, line, column] = unit_->tokenStartPosition(location);

  auto loc = mlir::FileLineColLoc::get(context_, filename, line, column);

  return loc;
}

auto Codegen::emitTodoStmt(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoStmtOp {
  unit_->error(
      location,
      std::format("unable to generate code for this statement ({})", message));
  const auto loc = getLocation(location);
  auto op = mlir::cxx::TodoStmtOp::create(builder_, loc, message);
  return op;
}

auto Codegen::emitTodoExpr(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoExprOp {
  unit_->error(
      location,
      std::format("unable to generate code for this expression ({})", message));
  const auto loc = getLocation(location);
  auto op = mlir::cxx::TodoExprOp::create(builder_, loc, message);
  return op;
}

auto Codegen::encodeSecondaryVTableName(ClassSymbol* classSymbol,
                                        ClassSymbol* base) -> std::string {
  ExternalNameEncoder classEncoder;
  ExternalNameEncoder baseEncoder;
  return std::format("__cxx_secondary_vtable${}${}",
                     classEncoder.encodeVTable(classSymbol),
                     baseEncoder.encodeVTable(base));
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

void Codegen::emitVTableGroupOp(mlir::Location loc, llvm::StringRef name,
                                const VTableLayout::Group& group) {
  mlir::SmallVector<std::int64_t> vbaseOffsets;
  for (auto& vbaseOffset : group.vbaseOffsets) {
    vbaseOffsets.push_back(vbaseOffset.second);
  }

  mlir::SmallVector<std::int64_t> vcallOffsets;
  for (auto& vcallOffset : group.vcallOffsets) {
    vcallOffsets.push_back(vcallOffset.second);
  }

  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  auto vcallSlotByteOffset = [&](int index) -> std::int64_t {
    const auto distWords =
        2 + static_cast<std::int64_t>(vbaseOffsets.size()) +
        (static_cast<std::int64_t>(vcallOffsets.size()) - index);
    return -wordSize * distWords;
  };

  mlir::SmallVector<mlir::Attribute> slotAttrs;
  for (auto& slot : group.slots) {
    if (!slot.function) {
      slotAttrs.push_back(builder_.getUnitAttr());
      continue;
    }

    FunctionSymbol* target = slot.function;
    if (slot.kind == VTableLayout::SlotKind::kDeletingDtor) {
      auto deletingDtor = slot.function->deletingDtorVariant();
      target = deletingDtor ? deletingDtor : completeObjectDtor(slot.function);
    } else if (slot.kind == VTableLayout::SlotKind::kCompleteDtor) {
      target = completeObjectDtor(slot.function);
    }

    mlir::cxx::FuncOp funcOp;
    if (slot.vcallOffsetIndex >= 0) {
      funcOp = findOrCreateVirtualThunk(
          target, vcallSlotByteOffset(slot.vcallOffsetIndex));
    } else if (slot.thisAdjustment != 0) {
      funcOp = findOrCreateThisAdjustingThunk(
          target, static_cast<std::int64_t>(slot.thisAdjustment));
    } else {
      funcOp = findOrCreateFunction(target);
    }

    slotAttrs.push_back(
        mlir::FlatSymbolRefAttr::get(context_, funcOp.getSymName()));
  }

  auto linkage = mlir::cxx::LinkageKind::LinkOnceODR;
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkage);

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::VTableOp::create(builder_, loc, name,
                              builder_.getI64ArrayAttr(vbaseOffsets),
                              builder_.getI64ArrayAttr(vcallOffsets),
                              -static_cast<std::int64_t>(group.offset),
                              builder_.getArrayAttr(slotAttrs), linkageAttr);
}

void Codegen::generateSecondaryVTables(ClassSymbol* classSymbol) {
  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return;

  for (auto& group : vtableLayout->secondary) {
    auto secondaryVtableName =
        encodeSecondaryVTableName(classSymbol, group.base);

    if (module_.lookupSymbol<mlir::cxx::VTableOp>(secondaryVtableName)) {
      continue;
    }

    auto loc = getLocation(classSymbol->location());
    emitVTableGroupOp(loc, secondaryVtableName, group);
  }
}

auto Codegen::findOrCreateThunk(
    FunctionSymbol* target, llvm::StringRef thunkName,
    const std::function<mlir::Value(mlir::Value rawThisI8, mlir::Location loc)>&
        computeAdjustedThisI8) -> mlir::cxx::FuncOp {
  auto targetFuncOp = findOrCreateFunction(target);
  if (!targetFuncOp) return {};

  if (auto existing = module_.lookupSymbol<mlir::cxx::FuncOp>(thunkName)) {
    return existing;
  }

  auto funcType = targetFuncOp.getFunctionType();

  auto functionType = type_cast<FunctionType>(target->type());
  const auto returnAbi = classifyClassValueAbi(functionType->returnType());
  const size_t thisIndex =
      returnAbi.kind == ClassValueAbi::Kind::Indirect ? 1 : 0;

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto loc = getLocation(target->location());

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::LinkOnceODR);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);

  auto thunkFunc = mlir::cxx::FuncOp::create(
      builder_, loc, thunkName, funcType, linkageAttr, inlineAttr,
      mlir::cxx::VisibilityAttr{}, mlir::StringAttr{}, mlir::ArrayAttr{},
      mlir::ArrayAttr{});

  auto entryBlock = builder_.createBlock(&thunkFunc.getBody());
  mlir::SmallVector<mlir::Value> blockArgs;
  for (auto inputType : funcType.getInputs()) {
    blockArgs.push_back(entryBlock->addArgument(inputType, loc));
  }
  builder_.setInsertionPointToEnd(entryBlock);

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  auto rawThis = blockArgs[thisIndex];
  auto rawThisI8 =
      mlir::cxx::BitcastOp::create(builder_, loc, i8PtrType, rawThis);

  auto adjustedThisI8 = computeAdjustedThisI8(rawThisI8, loc);

  auto adjustedThis = mlir::cxx::BitcastOp::create(
      builder_, loc, rawThis.getType(), adjustedThisI8);

  mlir::SmallVector<mlir::Value> callArgs(blockArgs.begin(), blockArgs.end());
  callArgs[thisIndex] = adjustedThis;

  auto callOp = mlir::cxx::CallOp::create(builder_, loc, funcType.getResults(),
                                          targetFuncOp.getSymName(), callArgs);

  if (funcType.getResults().empty()) {
    mlir::cxx::ReturnOp::create(builder_, loc);
  } else {
    mlir::cxx::ReturnOp::create(builder_, loc, callOp->getResults());
  }

  return thunkFunc;
}

auto Codegen::findOrCreateThisAdjustingThunk(FunctionSymbol* target,
                                             std::int64_t offset)
    -> mlir::cxx::FuncOp {
  auto targetFuncOp = findOrCreateFunction(target);
  if (!targetFuncOp) return {};

  auto thunkName = std::format("__cxx_thunk.{}.{}", offset,
                               std::string(targetFuncOp.getSymName()));

  return findOrCreateThunk(
      target, thunkName, [&](mlir::Value rawThisI8, mlir::Location loc) {
        auto offsetType = convertType(control()->getIntType());
        auto offsetOp = mlir::arith::ConstantOp::create(
            builder_, loc, offsetType,
            builder_.getIntegerAttr(offsetType, static_cast<int64_t>(-offset)));
        return mlir::cxx::PtrAddOp::create(builder_, loc, rawThisI8.getType(),
                                           rawThisI8, offsetOp)
            .getResult();
      });
}

auto Codegen::findOrCreateVirtualThunk(FunctionSymbol* target,
                                       std::int64_t vcallSlotByteOffset)
    -> mlir::cxx::FuncOp {
  auto targetFuncOp = findOrCreateFunction(target);
  if (!targetFuncOp) return {};

  auto thunkName = std::format("__cxx_vthunk.{}.{}", vcallSlotByteOffset,
                               std::string(targetFuncOp.getSymName()));

  return findOrCreateThunk(
      target, thunkName, [&](mlir::Value rawThisI8, mlir::Location loc) {
        return adjustByVtableWord(loc, rawThisI8, vcallSlotByteOffset);
      });
}

void Codegen::emitCtorVtableInit(FunctionSymbol* functionSymbol,
                                 mlir::Location loc) {
  if (!functionSymbol->isConstructor() || !thisValue_) return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return;

  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) return;

  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return;

  ExternalNameEncoder encoder;
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto& primary = vtableLayout->primary;

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto vtableArrayType =
      mlir::cxx::ArrayType::get(context_, i8PtrType, primary.wordCount());
  auto vtablePtrType = mlir::cxx::PointerType::get(context_, vtableArrayType);

  auto vtableAddr = mlir::cxx::AddressOfOp::create(
      builder_, loc, vtablePtrType,
      mlir::FlatSymbolRefAttr::get(context_, vtableName));

  auto intTy = convertType(control()->getIntType());
  auto twoOp = mlir::arith::ConstantOp::create(
      builder_, loc, intTy,
      builder_.getIntegerAttr(intTy,
                              static_cast<int64_t>(primary.headerWordCount())));

  auto vtableDataPtr = mlir::cxx::PtrAddOp::create(
      builder_, loc, mlir::cxx::PointerType::get(context_, i8PtrType),
      vtableAddr, twoOp);

  auto thisPtr = loadThisPointer(loc, classSymbol);

  auto vptrFieldPtr = resolveVptrField(thisPtr, classSymbol, loc);

  mlir::cxx::StoreOp::create(builder_, loc, vtableDataPtr, vptrFieldPtr, 8);

  for (auto& group : vtableLayout->secondary) {
    auto secondaryVtableName =
        encodeSecondaryVTableName(classSymbol, group.base);

    auto groupArrayType =
        mlir::cxx::ArrayType::get(context_, i8PtrType, group.wordCount());
    auto groupPtrType = mlir::cxx::PointerType::get(context_, groupArrayType);

    auto groupVtableAddr = mlir::cxx::AddressOfOp::create(
        builder_, loc, groupPtrType,
        mlir::FlatSymbolRefAttr::get(context_, secondaryVtableName));

    auto groupHeaderOp = mlir::arith::ConstantOp::create(
        builder_, loc, intTy,
        builder_.getIntegerAttr(intTy,
                                static_cast<int64_t>(group.headerWordCount())));

    auto groupDataPtr = mlir::cxx::PtrAddOp::create(
        builder_, loc, mlir::cxx::PointerType::get(context_, i8PtrType),
        groupVtableAddr, groupHeaderOp);

    auto baseSubobjectPtr =
        memberAddress(loc, thisPtr, group.base->type(),
                      layout->getBaseInfo(group.base)->index);

    auto baseVptrFieldPtr = resolveVptrField(baseSubobjectPtr, group.base, loc);

    mlir::cxx::StoreOp::create(builder_, loc, groupDataPtr, baseVptrFieldPtr,
                               8);
  }
}

auto Codegen::memberAddress(mlir::Location loc, mlir::Value objectPtr,
                            const Type* memberType, std::uint32_t index)
    -> mlir::Value {
  return memberAddress(loc, objectPtr, convertType(memberType), index);
}

auto Codegen::memberAddress(mlir::Location loc, mlir::Value objectPtr,
                            mlir::Type memberType, std::uint32_t index)
    -> mlir::Value {
  auto ptrType = mlir::cxx::PointerType::get(context_, memberType);
  return mlir::cxx::MemberOp::create(builder_, loc, ptrType, objectPtr, index);
}

auto Codegen::resolveVptrField(mlir::Value basePtr, ClassSymbol* baseClassSym,
                               mlir::Location loc) -> mlir::Value {
  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  auto layout = baseClassSym->layout();
  if (!layout) return {};

  if (layout->hasDirectVtable()) {
    return memberAddress(loc, basePtr, i8PtrType, layout->vtableIndex());
  }

  mlir::Value current = basePtr;
  auto currentClass = baseClassSym;
  auto currentLayout = layout;

  while (currentLayout && !currentLayout->hasDirectVtable()) {
    auto baseIdx = currentLayout->vtableIndex();
    ClassSymbol* baseSym = nullptr;
    for (auto base : currentClass->baseClasses()) {
      auto bs = symbol_cast<ClassSymbol>(base->symbol());
      if (!bs) continue;
      auto bi = currentLayout->getBaseInfo(bs);
      if (bi && bi->index == baseIdx) {
        baseSym = bs;
        break;
      }
    }
    if (!baseSym) break;

    current = memberAddress(loc, current, baseSym->type(), baseIdx);
    currentClass = baseSym;
    currentLayout = baseSym->layout();
  }

  auto vtableIdx = currentLayout ? currentLayout->vtableIndex() : 0;
  return memberAddress(loc, current, i8PtrType, vtableIdx);
}

void Codegen::generateVTable(ClassSymbol* classSymbol) {
  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) {
    return;
  }

  if (classSymbol->isExplicitInstantiationDeclared()) return;

  if (!emittedVTables_.insert(classSymbol).second) return;

  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout) return;

  ExternalNameEncoder encoder;
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto loc = getLocation(classSymbol->location());

  emitVTableGroupOp(loc, vtableName, vtableLayout->primary);

  generateSecondaryVTables(classSymbol);
}

auto Codegen::findOrCreateCxaAtexit(mlir::Location loc) -> mlir::cxx::FuncOp {
  const llvm::StringRef name = "__cxa_atexit";

  if (auto existingFunc = module_.lookupSymbol<mlir::cxx::FuncOp>(name)) {
    return existingFunc;
  }

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto i32Type = builder_.getI32Type();

  mlir::SmallVector<mlir::Type> paramTypes{i8PtrType, i8PtrType, i8PtrType};
  mlir::SmallVector<mlir::Type> resultTypes{i32Type};
  auto funcType =
      mlir::cxx::FunctionType::get(context_, paramTypes, resultTypes,
                                   /*isVariadic=*/false);
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);
  return mlir::cxx::FuncOp::create(builder_, loc, name, funcType, linkageAttr,
                                   inlineAttr, mlir::cxx::VisibilityAttr{},
                                   mlir::StringAttr{}, mlir::ArrayAttr{},
                                   mlir::ArrayAttr{});
}

auto Codegen::findOrCreateDsoHandle(mlir::Location loc) -> mlir::cxx::GlobalOp {
  const llvm::StringRef name = "__dso_handle";

  if (auto existing = module_.lookupSymbol<mlir::cxx::GlobalOp>(name)) {
    return existing;
  }

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);
  builder_.setInsertionPointToStart(module_.getBody());

  auto i8Type = builder_.getI8Type();
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::External);

  mlir::IntegerAttr alignmentAttr;

  return mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), i8Type, /*isConstant=*/false, name,
      /*initializer=*/mlir::Attribute(), linkageAttr, alignmentAttr);
}

void Codegen::emitGlobalVarDtorRegistration(VariableSymbol* defVar,
                                            FunctionSymbol* dtor,
                                            mlir::cxx::GlobalOp global,
                                            mlir::Location loc) {
  auto savedInsertionPoint = builder_.saveInsertionPoint();
  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);

  std::string thunkName = "__cxx_global_array_dtor";
  if (globalVarDtorCount_ > 0) {
    thunkName = std::format("__cxx_global_array_dtor.{}", globalVarDtorCount_);
  }
  ++globalVarDtorCount_;

  mlir::SmallVector<mlir::Type> paramTypes{i8PtrType};
  mlir::SmallVector<mlir::Type> resultTypes;
  auto funcType =
      mlir::cxx::FunctionType::get(context_, paramTypes, resultTypes,
                                   /*isVariadic=*/false);
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
      context_, mlir::cxx::LinkageKind::Internal);
  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(context_, mlir::cxx::InlineKind::NoInline);

  builder_.setInsertionPointToEnd(module_.getBody());

  auto thunkFunc = mlir::cxx::FuncOp::create(
      builder_, loc, thunkName, funcType, linkageAttr, inlineAttr,
      mlir::cxx::VisibilityAttr{}, mlir::StringAttr{}, mlir::ArrayAttr{},
      mlir::ArrayAttr{});

  auto entryBlock = builder_.createBlock(&thunkFunc.getBody());
  entryBlock->addArgument(i8PtrType, loc);
  builder_.setInsertionPointToEnd(entryBlock);

  auto ptrType =
      mlir::cxx::PointerType::get(context_, convertType(defVar->type()));
  auto addr = mlir::cxx::AddressOfOp::create(builder_, loc, ptrType,
                                             global.getSymName());

  (void)emitCall(defVar->location(), dtor, {addr}, {});

  mlir::cxx::ReturnOp::create(builder_, loc);

  builder_.setInsertionPointToEnd(module_.getBody());

  auto atexitFunc = findOrCreateCxaAtexit(loc);
  auto dsoHandle = findOrCreateDsoHandle(loc);

  builder_.restoreInsertionPoint(savedInsertionPoint);

  auto thunkPtr = mlir::cxx::AddressOfOp::create(builder_, loc, i8PtrType,
                                                 thunkFunc.getSymName());
  auto nullPtr = mlir::cxx::NullPtrConstantOp::create(builder_, loc, i8PtrType);
  auto dsoHandlePtr = mlir::cxx::AddressOfOp::create(builder_, loc, i8PtrType,
                                                     dsoHandle.getSymName());

  mlir::SmallVector<mlir::Value> args{thunkPtr, nullPtr, dsoHandlePtr};
  mlir::SmallVector<mlir::Type> callResultTypes{builder_.getI32Type()};
  mlir::cxx::CallOp::create(builder_, loc, callResultTypes,
                            atexitFunc.getSymName(), args);
}

auto Codegen::completeObjectDtor(FunctionSymbol* dtor) -> FunctionSymbol* {
  if (auto variant = dtor->completeObjectVariant()) return variant;
  return dtor;
}
}  // namespace cxx
