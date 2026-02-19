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

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

struct [[nodiscard]] Binder::CompleteClass {
  Binder& binder;
  ClassSpecifierAST* ast;
  ClassSymbol* classSymbol;
  Arena* pool;

  CompleteClass(Binder& b, ClassSpecifierAST* a)
      : binder(b), ast(a), classSymbol(a->symbol), pool(b.unit_->arena()) {}

  auto control() const -> Control* { return binder.control(); }

  void complete();

  void markComplete();
  auto shouldSynthesizeSpecialMembers() const -> bool;
  void synthesizeSpecialMembers();
  auto hasVirtualBaseDestructor() const -> bool;

  auto buildRecordLayout() -> std::expected<bool, std::string>;

  auto newDefaultedFunction(const Name* name, const Type* type)
      -> FunctionSymbol*;
  void attachDeclaration(FunctionSymbol* symbol, UnqualifiedIdAST* id);
  auto makeCtorNameId() -> NameIdAST*;
  void addDefaultConstructor();
  void addCopyConstructor();
  void addMoveConstructor();
  void addCopyAssignmentOperator();
  void addMoveAssignmentOperator();
  void addDestructor();
};

void Binder::complete(ClassSpecifierAST* ast) {
  CompleteClass{*this, ast}.complete();
}

void Binder::CompleteClass::markComplete() { ast->symbol->setComplete(true); }

auto Binder::CompleteClass::shouldSynthesizeSpecialMembers() const -> bool {
  if (!binder.is_parsing_cxx()) return false;
  if (!classSymbol->name()) return false;
  return true;
}

void Binder::CompleteClass::synthesizeSpecialMembers() {
  addDefaultConstructor();
  addCopyConstructor();
  addMoveConstructor();
  addCopyAssignmentOperator();
  addMoveAssignmentOperator();
  addDestructor();
}

auto Binder::CompleteClass::hasVirtualBaseDestructor() const -> bool {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;

    auto dtor = baseClass->destructor();
    if (dtor && dtor->isVirtual()) return true;
  }

  return false;
}

auto Binder::CompleteClass::buildRecordLayout()
    -> std::expected<bool, std::string> {
  return binder.buildRecordLayout(classSymbol);
}

void Binder::CompleteClass::complete() {
  if (binder.inTemplate()) {
    markComplete();
    return;
  }

  if (shouldSynthesizeSpecialMembers()) synthesizeSpecialMembers();

  auto status = buildRecordLayout();
  if (!status.has_value())
    binder.error(classSymbol->location(), status.error());

  binder.computeClassFlags(classSymbol);
  markComplete();
}

auto Binder::CompleteClass::newDefaultedFunction(const Name* name,
                                                 const Type* type)
    -> FunctionSymbol* {
  auto symbol =
      control()->newFunctionSymbol(classSymbol, classSymbol->location());
  symbol->setName(name);
  symbol->setType(type);
  symbol->setDefined(true);
  symbol->setDefaulted(true);
  symbol->setLanguageLinkage(LanguageKind::kCXX);
  return symbol;
}

void Binder::CompleteClass::attachDeclaration(FunctionSymbol* symbol,
                                              UnqualifiedIdAST* id) {
  auto idDecl = IdDeclaratorAST::create(pool);
  idDecl->unqualifiedId = id;

  auto funcChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, nullptr, idDecl,
      make_list_node<DeclaratorChunkAST>(pool, funcChunk));

  auto funcDef = FunctionDefinitionAST::create(pool);
  funcDef->declarator = declarator;
  funcDef->functionBody = DefaultFunctionBodyAST::create(pool);
  funcDef->symbol = symbol;
  symbol->setDeclaration(funcDef);
}

auto Binder::CompleteClass::makeCtorNameId() -> NameIdAST* {
  return NameIdAST::create(pool, name_cast<Identifier>(classSymbol->name()));
}

void Binder::CompleteClass::addDefaultConstructor() {
  if (!classSymbol->constructors().empty()) return;

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addCopyConstructor() {
  if (classSymbol->copyConstructor()) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {constRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addMoveConstructor() {
  if (classSymbol->moveConstructor()) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {rvalRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addCopyAssignmentOperator() {
  if (classSymbol->copyAssignmentOperator()) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {constRefType}));
  classSymbol->addSymbol(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

void Binder::CompleteClass::addMoveAssignmentOperator() {
  if (classSymbol->moveAssignmentOperator()) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {rvalRefType}));
  classSymbol->addSymbol(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

void Binder::CompleteClass::addDestructor() {
  if (classSymbol->destructor()) return;

  auto symbol = newDefaultedFunction(
      control()->getDestructorId(classSymbol->name()),
      control()->getFunctionType(control()->getVoidType(), {}));

  if (hasVirtualBaseDestructor()) symbol->setVirtual(true);

  classSymbol->addSymbol(symbol);

  auto dtorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(classSymbol->name()))
    dtorId->id = NameIdAST::create(pool, id);
  attachDeclaration(symbol, dtorId);
}

struct [[nodiscard]] Binder::BuildRecordLayout {
  Binder& binder;
  ClassSymbol* classSymbol;
  const MemoryLayout* memoryLayout;
  std::unique_ptr<ClassLayout> layout;

  int calculatedSize = 0;
  int calculatedAlignment = 1;
  std::uint32_t currentIndex = 0;

  int nextBitPos = 0;
  int runStartByte = 0;
  std::uint32_t runIndex = 0;
  bool inBitfieldRun = false;
  std::vector<FieldSymbol*> runFields;

  BuildRecordLayout(Binder& b, ClassSymbol* cls)
      : binder(b),
        classSymbol(cls),
        memoryLayout(b.control()->memoryLayout()),
        layout(std::make_unique<ClassLayout>()) {}

  auto control() const -> Control* { return binder.control(); }

  auto operator()() -> std::expected<bool, std::string>;
  auto validate() -> std::expected<bool, std::string>;
  void layoutVtable();
  void layoutBases();
  auto layoutFields() -> std::expected<bool, std::string>;
  auto layoutBitfield(FieldSymbol* field) -> std::expected<bool, std::string>;
  auto layoutRegularField(FieldSymbol* field)
      -> std::expected<bool, std::string>;
  void closeBitfieldRun();
  void propagateBaseFields();
  void propagateAnonymousFields(ClassSymbol* cls, std::uint64_t baseOffset);
  void finalize();
};

auto Binder::buildRecordLayout(ClassSymbol* classSymbol)
    -> std::expected<bool, std::string> {
  return BuildRecordLayout{*this, classSymbol}();
}

auto Binder::BuildRecordLayout::operator()()
    -> std::expected<bool, std::string> {
  if (auto status = validate(); !status) return status;

  layoutVtable();
  layoutBases();

  if (auto status = layoutFields(); !status) return status;

  propagateBaseFields();
  finalize();

  return true;
}

auto Binder::BuildRecordLayout::validate() -> std::expected<bool, std::string> {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) {
      return std::unexpected(
          std::format("base class '{}' not found", to_string(base->name())));
    }
    if (!baseClassSymbol->isComplete()) {
      return std::unexpected(std::format("base class '{}' is incomplete",
                                         to_string(baseClassSymbol->name())));
    }
  }
  return true;
}

void Binder::BuildRecordLayout::layoutVtable() {
  if (classSymbol->isUnion()) return;

  bool hasPolymorphicBase = false;
  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass && baseClass->layout() && baseClass->layout()->hasVtable()) {
      hasPolymorphicBase = true;
      break;
    }
  }

  bool needsOwnVptr = false;
  if (!hasPolymorphicBase) {
    needsOwnVptr =
        views::any_function(classSymbol->members(),
                            [](FunctionSymbol* f) { return f->isVirtual(); });
  }

  if (needsOwnVptr) {
    layout->setHasVtable(true);
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(currentIndex++);

    auto ptrSize = static_cast<int>(memoryLayout->sizeOfPointer());
    calculatedSize = ptrSize;
    calculatedAlignment = ptrSize;
    nextBitPos = calculatedSize * 8;
  } else if (hasPolymorphicBase) {
    layout->setHasVtable(true);
    layout->setVtableIndex(0);
  }
}

void Binder::BuildRecordLayout::layoutBases() {
  if (classSymbol->isUnion()) return;

  bool foundPolymorphicBase = false;
  for (auto* base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    auto baseAlignment = baseClassSymbol->alignment();
    if (baseAlignment > 0) {
      calculatedSize = align_to(calculatedSize, baseAlignment);
    }

    ClassLayout::MemberInfo baseInfo;
    baseInfo.offset = calculatedSize;
    baseInfo.index = currentIndex++;
    layout->setBaseInfo(baseClassSymbol, baseInfo);

    auto baseLayout = baseClassSymbol->layout();
    if (!foundPolymorphicBase && baseLayout && baseLayout->hasVtable()) {
      layout->setVtableIndex(baseInfo.index);
      foundPolymorphicBase = true;
    }

    calculatedSize += baseClassSymbol->sizeInBytes();
    calculatedAlignment =
        std::max(calculatedAlignment, baseClassSymbol->alignment());
  }

  // Keep nextBitPos in sync after bases
  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::closeBitfieldRun() {
  if (!inBitfieldRun) return;

  // Advance calculatedSize to the next byte boundary after all bits used
  calculatedSize = (nextBitPos + 7) / 8;

  // Compute the storage unit size for this run
  auto allocUnitSizeBytes =
      static_cast<std::uint32_t>(calculatedSize - runStartByte);

  // Update all fields in this run with the final storage unit size
  for (auto* f : runFields) {
    if (auto info = layout->getFieldInfo(f)) {
      auto updated = *info;
      updated.allocUnitSizeBytes = allocUnitSizeBytes;
      layout->setFieldInfo(f, updated);
    }
  }

  runFields.clear();
  inBitfieldRun = false;
  currentIndex++;
}

auto Binder::BuildRecordLayout::layoutBitfield(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  int bitWidth = 0;
  if (auto& bfw = field->bitFieldWidth()) {
    if (auto* iv = std::get_if<std::intmax_t>(&*bfw)) {
      bitWidth = static_cast<int>(*iv);
    }
  }

  auto fieldAlign = field->alignment();
  auto fieldSizeBytes =
      static_cast<int>(memoryLayout->sizeOf(field->type()).value_or(0));
  auto fieldSizeBits = fieldSizeBytes * 8;

  // Zero-width bitfield: forces alignment to the type boundary
  if (bitWidth == 0) {
    if (inBitfieldRun) {
      closeBitfieldRun();
    }
    if (!isUnion && fieldSizeBits > 0) {
      // Align nextBitPos to the type's alignment boundary (in bits)
      auto alignBits = fieldAlign * 8;
      nextBitPos = align_to(nextBitPos, alignBits);
      calculatedSize = (nextBitPos + 7) / 8;
    }
    return true;
  }

  if (isUnion) {
    field->setLocalOffset(0);
    field->setBitFieldOffset(0);

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = 0;
    fieldInfo.index = 0;
    fieldInfo.bitOffset = 0;
    fieldInfo.bitWidth = bitWidth;
    fieldInfo.allocUnitSizeBytes = (bitWidth + 7) / 8;
    layout->setFieldInfo(field, fieldInfo);

    auto fieldSizeForUnion = std::max(fieldSizeBytes, (bitWidth + 7) / 8);
    calculatedSize = std::max(calculatedSize, fieldSizeForUnion);
    calculatedAlignment = std::max(calculatedAlignment, fieldAlign);
    return true;
  }

  if (fieldSizeBits > 0) {
    auto startUnit = nextBitPos / fieldSizeBits;
    auto endUnit = (nextBitPos + bitWidth - 1) / fieldSizeBits;
    if (startUnit != endUnit) {
      if (inBitfieldRun) {
        closeBitfieldRun();
      }
      nextBitPos = align_to(nextBitPos, fieldSizeBits);
      calculatedSize = nextBitPos / 8;
    }
  }

  if (!inBitfieldRun) {
    runStartByte = calculatedSize;
    nextBitPos = runStartByte * 8;
    runIndex = currentIndex;
    inBitfieldRun = true;
    runFields.clear();
  }

  auto bitOffsetInRun = nextBitPos - runStartByte * 8;

  field->setLocalOffset(runStartByte);
  field->setBitFieldOffset(bitOffsetInRun);

  ClassLayout::MemberInfo fieldInfo;
  fieldInfo.offset = runStartByte;
  fieldInfo.index = runIndex;
  fieldInfo.bitOffset = bitOffsetInRun;
  fieldInfo.bitWidth = bitWidth;
  layout->setFieldInfo(field, fieldInfo);

  runFields.push_back(field);
  nextBitPos += bitWidth;
  calculatedAlignment = std::max(calculatedAlignment, fieldAlign);

  return true;
}

auto Binder::BuildRecordLayout::layoutRegularField(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  closeBitfieldRun();

  std::optional<std::size_t> size;
  if (control()->is_unbounded_array(field->type())) {
    size = 0;
  } else {
    size = memoryLayout->sizeOf(field->type());
  }

  if (!size.has_value()) {
    return std::unexpected(
        std::format("size of incomplete type '{}'",
                    to_string(field->type(), field->name())));
  }

  if (isUnion) {
    field->setLocalOffset(0);
    calculatedSize = std::max(calculatedSize, int(size.value()));

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = 0;
    fieldInfo.index = 0;
    layout->setFieldInfo(field, fieldInfo);
  } else {
    calculatedSize = align_to(calculatedSize, field->alignment());
    field->setLocalOffset(calculatedSize);

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = calculatedSize;
    fieldInfo.index = currentIndex++;
    layout->setFieldInfo(field, fieldInfo);

    calculatedSize += size.value();
  }

  // Keep nextBitPos in sync
  nextBitPos = calculatedSize * 8;

  calculatedAlignment = std::max(calculatedAlignment, field->alignment());
  return true;
}

auto Binder::BuildRecordLayout::layoutFields()
    -> std::expected<bool, std::string> {
  FieldSymbol* lastField = nullptr;

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (lastField && control()->is_unbounded_array(lastField->type())) {
      return std::unexpected(
          std::format("size of incomplete type '{}'",
                      to_string(lastField->type(), lastField->name())));
    }

    if (!field->alignment()) {
      return std::unexpected(
          std::format("alignment of incomplete type '{}'",
                      to_string(field->type(), field->name())));
    }

    if (field->isBitField()) {
      if (auto status = layoutBitfield(field); !status) return status;
    } else {
      if (auto status = layoutRegularField(field); !status) return status;
    }

    lastField = field;
  }

  closeBitfieldRun();
  return true;
}

void Binder::BuildRecordLayout::propagateBaseFields() {
  for (auto* base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    auto baseLayout = baseClassSymbol->layout();
    if (!baseLayout) continue;

    auto baseInfo = layout->getBaseInfo(baseClassSymbol);
    if (!baseInfo) continue;

    for (auto field :
         views::members(baseClassSymbol) | views::non_static_fields) {
      auto baseFieldInfo = baseLayout->getFieldInfo(field);
      if (baseFieldInfo) {
        ClassLayout::MemberInfo adjustedInfo;
        adjustedInfo.offset = baseInfo->offset + baseFieldInfo->offset;
        adjustedInfo.index = baseFieldInfo->index;
        adjustedInfo.bitOffset = baseFieldInfo->bitOffset;
        adjustedInfo.bitWidth = baseFieldInfo->bitWidth;
        adjustedInfo.allocUnitSizeBytes = baseFieldInfo->allocUnitSizeBytes;
        layout->setFieldInfo(field, adjustedInfo);
      }
    }
  }

  propagateAnonymousFields(classSymbol, 0);
}

void Binder::BuildRecordLayout::propagateAnonymousFields(
    ClassSymbol* cls, std::uint64_t baseOffset) {
  for (auto member : cls->members()) {
    auto nestedClass = symbol_cast<ClassSymbol>(member);
    if (!nestedClass) continue;
    if (nestedClass->name()) continue;  // not anonymous
    if (!nestedClass->isComplete()) continue;

    auto nestedLayout = nestedClass->layout();
    if (!nestedLayout) continue;

    // Find the FieldSymbol for this anonymous class in cls
    FieldSymbol* anonField = nullptr;
    for (auto m : cls->members()) {
      auto f = symbol_cast<FieldSymbol>(m);
      if (!f) continue;
      if (auto ct = type_cast<ClassType>(f->type())) {
        if (ct->symbol() == nestedClass) {
          anonField = f;
          break;
        }
      }
    }
    if (!anonField) continue;

    auto anonFieldInfo = layout->getFieldInfo(anonField);
    if (!anonFieldInfo) continue;

    std::uint64_t anonOffset = baseOffset + anonFieldInfo->offset;

    // Propagate each field of the anonymous class into the parent layout.
    for (auto field : views::members(nestedClass) | views::non_static_fields) {
      if (auto nestedFieldInfo = nestedLayout->getFieldInfo(field)) {
        // Don't propagate implicit anonymous field symbols — they are
        // structural fields, not user-visible members accessed via lookup.
        if (!field->name()) {
          auto fieldType = type_cast<ClassType>(field->type());
          if (fieldType && !fieldType->symbol()->name()) continue;
        }

        ClassLayout::MemberInfo adjustedInfo;
        adjustedInfo.offset = anonOffset + nestedFieldInfo->offset;
        adjustedInfo.index = nestedFieldInfo->index;
        adjustedInfo.bitOffset = nestedFieldInfo->bitOffset;
        adjustedInfo.bitWidth = nestedFieldInfo->bitWidth;
        adjustedInfo.allocUnitSizeBytes = nestedFieldInfo->allocUnitSizeBytes;
        layout->setFieldInfo(field, adjustedInfo);
      }
    }

    // Recurse into nested anonymous classes
    propagateAnonymousFields(nestedClass, anonOffset);
  }
}

void Binder::BuildRecordLayout::finalize() {
  calculatedSize = align_to(calculatedSize, calculatedAlignment);
  if (calculatedSize == 0) calculatedSize = 1;

  classSymbol->setAlignment(calculatedAlignment);
  classSymbol->setSizeInBytes(calculatedSize);

  layout->setSize(calculatedSize);
  layout->setAlignment(calculatedAlignment);

  classSymbol->setLayout(std::move(layout));
}

}  // namespace cxx