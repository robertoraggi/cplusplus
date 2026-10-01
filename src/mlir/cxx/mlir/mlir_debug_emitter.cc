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

#include <cxx/cxx_fwd.h>
#include <cxx/mlir/mlir_debug_emitter.h>
#include <cxx/mlir/mlir_emitter.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/TargetParser/Triple.h>

#include <array>
#include <filesystem>
#include <limits>

namespace cxx::ir {
namespace {

[[nodiscard]] static auto targetNeedsAppleNameTable(mlir::ModuleOp module)
    -> bool {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!tripleAttr) return false;
  llvm::Triple triple(tripleAttr.getValue());
  return triple.isAppleMachO();
}

[[nodiscard]] static auto debugFilePath(const std::string& filename,
                                        const std::string& compilationDirectory)
    -> std::pair<std::string, std::string> {
  const auto filePath = std::filesystem::path{filename};

  if (!filePath.is_absolute()) return {filename, compilationDirectory};

  const auto currentPath = std::filesystem::path{compilationDirectory};

  auto fileIt = filePath.begin();
  const auto fileEnd = filePath.end();
  auto dirIt = currentPath.begin();
  const auto dirEnd = currentPath.end();

  std::filesystem::path commonPrefix;
  for (; dirIt != dirEnd && fileIt != fileEnd && *dirIt == *fileIt;
       ++dirIt, ++fileIt) {
    commonPrefix /= *dirIt;
  }

  if (commonPrefix == commonPrefix.root_path()) return {filename, {}};

  std::filesystem::path relativePath;
  for (; fileIt != fileEnd; ++fileIt) relativePath /= *fileIt;

  return {relativePath.string(), commonPrefix.string()};
}

[[nodiscard]] static auto subprogramOf(mlir::LLVM::DIScopeAttr scope)
    -> mlir::LLVM::DISubprogramAttr {
  while (scope) {
    if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(scope))
      return sp;
    auto block = mlir::dyn_cast<mlir::LLVM::DILexicalBlockAttr>(scope);
    if (!block) break;
    scope = block.getScope();
  }
  return {};
}

[[nodiscard]] static auto enclosingSubprogram(mlir::Operation* op)
    -> mlir::LLVM::DISubprogramAttr {
  for (; op; op = op->getParentOp()) {
    auto fused = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc());
    if (!fused) continue;
    if (auto sp = mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(
            fused.getMetadata()))
      return sp;
  }
  return {};
}

[[nodiscard]] static auto scopeForOperation(mlir::Operation* op,
                                            mlir::LLVM::DIScopeAttr declared)
    -> mlir::LLVM::DIScopeAttr {
  auto enclosing = enclosingSubprogram(op);
  if (enclosing && subprogramOf(declared) != enclosing) return enclosing;
  return declared;
}

template <typename Tag, typename Attr>
auto retain(std::vector<Attr>& values, Attr value) -> Handle<Tag> {
  if (!value) return {};
  if (values.size() >= std::numeric_limits<std::uint32_t>::max())
    cxx_runtime_error("too many debug metadata nodes");
  auto ref = HandleAccess::make<Tag>(static_cast<std::uint32_t>(values.size()));
  values.push_back(value);
  return ref;
}

[[nodiscard]] auto encoding(DebugEncoding value) -> unsigned {
  constexpr std::array values{0u,
                              unsigned(llvm::dwarf::DW_ATE_boolean),
                              unsigned(llvm::dwarf::DW_ATE_signed),
                              unsigned(llvm::dwarf::DW_ATE_unsigned),
                              unsigned(llvm::dwarf::DW_ATE_UTF),
                              unsigned(llvm::dwarf::DW_ATE_float),
                              unsigned(llvm::dwarf::DW_ATE_complex_float)};
  return values.at(static_cast<unsigned>(value));
}

[[nodiscard]] auto derivedTag(DebugDerivedKind value) -> unsigned {
  constexpr std::array values{
      unsigned(llvm::dwarf::DW_TAG_pointer_type),
      unsigned(llvm::dwarf::DW_TAG_reference_type),
      unsigned(llvm::dwarf::DW_TAG_rvalue_reference_type),
      unsigned(llvm::dwarf::DW_TAG_const_type),
      unsigned(llvm::dwarf::DW_TAG_volatile_type),
      unsigned(llvm::dwarf::DW_TAG_atomic_type),
      unsigned(llvm::dwarf::DW_TAG_ptr_to_member_type),
      unsigned(llvm::dwarf::DW_TAG_inheritance),
      unsigned(llvm::dwarf::DW_TAG_member)};
  return values.at(static_cast<unsigned>(value));
}

[[nodiscard]] auto compositeTag(DebugCompositeKind value) -> unsigned {
  constexpr std::array values{unsigned(llvm::dwarf::DW_TAG_structure_type),
                              unsigned(llvm::dwarf::DW_TAG_union_type),
                              unsigned(llvm::dwarf::DW_TAG_enumeration_type)};
  return values.at(static_cast<unsigned>(value));
}

}  // namespace

MlirDebugEmitter::MlirDebugEmitter(MlirEmitter& emitter)
    : emitter_(emitter),
      context_(emitter.context()),
      builder_(emitter.builder()) {}

auto MlirDebugEmitter::type(DebugTypeRef ref) const -> mlir::LLVM::DITypeAttr {
  return types_.at(HandleAccess::id(ref));
}

auto MlirDebugEmitter::scope(DebugScopeRef ref) const
    -> mlir::LLVM::DIScopeAttr {
  return scopes_.at(HandleAccess::id(ref));
}

auto MlirDebugEmitter::compileUnit(const DebugCompileUnitInfo& info)
    -> DebugScopeRef {
  compilationDirectory_ = info.directory;
  auto distinct = mlir::DistinctAttr::create(builder_.getUnitAttr());
  auto language =
      info.isCxx ? llvm::dwarf::DW_LANG_C_plus_plus_20 : llvm::dwarf::DW_LANG_C;
  auto nameTable = targetNeedsAppleNameTable(emitter_.module())
                       ? mlir::LLVM::DINameTableKind::Apple
                       : mlir::LLVM::DINameTableKind::Default;
  compileUnitAttr_ = mlir::LLVM::DICompileUnitAttr::get(
      distinct, language, getOrCreateFileAttr(info.file),
      mlir::StringAttr::get(context_, "cxx"), false,
      mlir::LLVM::DIEmissionKind::Full,
#if LLVM_VERSION_MAJOR > 22
      false,
#endif
      nameTable);
  return retain<DebugScopeTag>(scopes_,
                               mlir::LLVM::DIScopeAttr{compileUnitAttr_});
}

auto MlirDebugEmitter::fileScope(std::string_view file) -> DebugScopeRef {
  return retain<DebugScopeTag>(scopes_,
                               mlir::LLVM::DIScopeAttr{getFileAttr(file)});
}

auto MlirDebugEmitter::lexicalBlock(DebugScopeRef parent,
                                    DebugLocation location) -> DebugScopeRef {
  return retain<DebugScopeTag>(
      scopes_, mlir::LLVM::DIScopeAttr{mlir::LLVM::DILexicalBlockAttr::get(
                   context_, scope(parent), getFileAttr(location.file),
                   location.line, location.column)});
}

auto MlirDebugEmitter::typeScope(DebugTypeRef ref) -> DebugScopeRef {
  return retain<DebugScopeTag>(
      scopes_, mlir::dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(type(ref)));
}

auto MlirDebugEmitter::basicType(const DebugBasicTypeInfo& info)
    -> DebugTypeRef {
  auto tag = info.encoding == DebugEncoding::Unspecified
                 ? llvm::dwarf::DW_TAG_unspecified_type
                 : llvm::dwarf::DW_TAG_base_type;
  return retain<DebugTypeTag>(
      types_,
      mlir::LLVM::DITypeAttr{mlir::LLVM::DIBasicTypeAttr::get(
          context_, tag, info.name, info.sizeInBits, encoding(info.encoding))});
}

auto MlirDebugEmitter::derivedType(const DebugDerivedTypeInfo& info)
    -> DebugTypeRef {
  auto name = mlir::StringAttr::get(context_, info.name);
  auto tag = derivedTag(info.kind);
#if LLVM_VERSION_MAJOR < 23
  auto attr = mlir::LLVM::DIDerivedTypeAttr::get(
      context_, tag, name, type(info.baseType), info.sizeInBits,
      info.alignInBits, info.offsetInBits, {}, type(info.classType));
#else
  auto attr = mlir::LLVM::DIDerivedTypeAttr::get(
      context_, tag, name, {}, 0, {}, type(info.baseType), info.sizeInBits,
      info.alignInBits, info.offsetInBits, std::nullopt, mlir::LLVM::DIFlags{},
      type(info.classType));
#endif
  return retain<DebugTypeTag>(types_, mlir::LLVM::DITypeAttr{attr});
}

auto MlirDebugEmitter::compositeType(const DebugCompositeTypeInfo& info)
    -> DebugTypeRef {
  mlir::SmallVector<mlir::LLVM::DINodeAttr> elements;
  for (auto element : info.elements) elements.push_back(type(element));
  return retain<DebugTypeTag>(
      types_,
      compositeTypeAttr(compositeTag(info.kind), info.name, info.location,
                        info.scope, info.baseType, info.sizeInBits,
                        info.alignInBits, info.isScopedEnum, elements));
}

auto MlirDebugEmitter::arrayType(const DebugArrayTypeInfo& info)
    -> DebugTypeRef {
  auto count = mlir::IntegerAttr::get(
      mlir::IntegerType::get(context_, info.countBitWidth), info.count);
  mlir::SmallVector<mlir::LLVM::DINodeAttr> elements{
      mlir::LLVM::DISubrangeAttr::get(context_, count, {}, {}, {})};
  return retain<DebugTypeTag>(
      types_, compositeTypeAttr(llvm::dwarf::DW_TAG_array_type, {}, {}, {},
                                info.elementType, info.sizeInBits,
                                info.alignInBits, false, elements));
}

auto MlirDebugEmitter::compositeTypeAttr(
    unsigned tag, std::string_view name, DebugLocation location,
    DebugScopeRef parent, DebugTypeRef baseType, std::uint64_t sizeInBits,
    std::uint64_t alignInBits, bool scopedEnum,
    llvm::ArrayRef<mlir::LLVM::DINodeAttr> elements) -> mlir::LLVM::DITypeAttr {
  auto file =
      location.line ? getFileAttr(location.file) : mlir::LLVM::DIFileAttr{};
  auto flags =
      scopedEnum ? mlir::LLVM::DIFlags::EnumClass : mlir::LLVM::DIFlags::Zero;
  return mlir::LLVM::DICompositeTypeAttr::get(
      context_, tag, mlir::StringAttr::get(context_, name), file, location.line,
      scope(parent), type(baseType), flags, sizeInBits, alignInBits,
#if LLVM_VERSION_MAJOR < 22
      elements, {}, {}, {}, {}
#else
      {}, {}, {}, {},
#if LLVM_VERSION_MAJOR > 22
      {}, {},
#endif
      elements
#endif
  );
}

auto MlirDebugEmitter::subroutineType(std::span<const DebugTypeRef> types)
    -> DebugTypeRef {
  mlir::SmallVector<mlir::LLVM::DITypeAttr> signature;
  for (auto ref : types) signature.push_back(type(ref));
  return retain<DebugTypeTag>(
      types_, mlir::LLVM::DITypeAttr{
                  mlir::LLVM::DISubroutineTypeAttr::get(context_, signature)});
}

auto MlirDebugEmitter::defineFunction(FunctionRef function, SourceLocation loc,
                                      const DebugFunctionInfo& info)
    -> DebugScopeRef {
  auto symbolName = emitter_.functionName(function);
  auto name = mlir::StringAttr::get(context_,
                                    info.name.empty() ? symbolName : info.name);
  mlir::StringAttr linkageName;
  if (std::string_view{name.getValue()} != symbolName)
    linkageName = mlir::StringAttr::get(context_, symbolName);
  auto file = getFileAttr(info.location.file);
  auto parent = scope(info.scope);
  if (!parent) parent = file;
  auto flags = mlir::LLVM::DISubprogramFlags::Definition;
  if (emitter_.functionLinkage(function) == Linkage::Internal)
    flags = flags | mlir::LLVM::DISubprogramFlags::LocalToUnit;
#if LLVM_VERSION_MAJOR < 23
  mlir::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
#else
  mlir::SmallVector<mlir::Attribute> retainedNodes;
#endif
  mlir::SmallVector<mlir::LLVM::DINodeAttr> annotations;
  auto subprogram = mlir::LLVM::DISubprogramAttr::get(
      context_, mlir::DistinctAttr::create(builder_.getUnitAttr()),
      compileUnitAttr_, parent, name, linkageName, file, info.location.line,
      info.scopeLine, flags,
      mlir::cast<mlir::LLVM::DISubroutineTypeAttr>(type(info.type)),
      retainedNodes, annotations);
  emitter_.function(function)->setLoc(
      mlir::FusedLoc::get({emitter_.getLocation(loc)}, subprogram, context_));
  return retain<DebugScopeTag>(scopes_, mlir::LLVM::DIScopeAttr{subprogram});
}

void MlirDebugEmitter::localVariable(ValueRef address,
                                     const DebugVariableInfo& info) {
  auto op = emitter_.value(address).getDefiningOp();
  if (!op) return;
  auto parent = scopeForOperation(op, scope(info.scope));
  if (!parent) return;
  auto file = getFileAttr(info.location.file);
  auto line = info.location.line;
  auto flags = mlir::LLVM::DIFlags::Zero;
  if (info.isObjectParameter) {
    flags =
        mlir::LLVM::DIFlags::Artificial | mlir::LLVM::DIFlags::ObjectPointer;
    file = {};
    line = 0;
    if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(parent)) {
      file = sp.getFile();
      line = sp.getLine();
    }
  }
  auto variable = mlir::LLVM::DILocalVariableAttr::get(
      context_, parent, mlir::StringAttr::get(context_, info.name), file, line,
      info.argument, 0, type(info.type), flags);
  op->setAttr("cxx.di_local", variable);
}

auto MlirDebugEmitter::getOrCreateFileAttr(std::string_view filename)
    -> mlir::LLVM::DIFileAttr {
  auto key = std::string{filename};
  if (auto it = fileAttrs_.find(key); it != fileAttrs_.end()) return it->second;
  auto [file, directory] = debugFilePath(key, compilationDirectory_);
  auto attr = mlir::LLVM::DIFileAttr::get(context_, file, directory);
  fileAttrs_.emplace(std::move(key), attr);
  return attr;
}

auto MlirDebugEmitter::getFileAttr(std::string_view filename)
    -> mlir::LLVM::DIFileAttr {
  if (filename.empty()) return compileUnitAttr_.getFile();
  return getOrCreateFileAttr(filename);
}

}  // namespace cxx::ir
