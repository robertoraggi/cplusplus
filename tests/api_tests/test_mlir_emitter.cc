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

// cxx
#include <cxx/codegen/debug_emitter.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/mlir/mlir_emitter.h>
#include <cxx/mlir/mlir_source_contexts.h>

// mlir
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

// gtest
#undef GTEST_HAS_CXXABI_H_
#define GTEST_HAS_CXXABI_H_ 0

#include <gtest/gtest.h>

#include <array>

namespace {

using namespace cxx;

class EmitterFixture : public ::testing::Test {
 protected:
  EmitterFixture() {
    context_.loadDialect<mlir::cxx::CxxDialect, mlir::arith::ArithDialect>();

    auto& builder = emitter_.builder();
    const auto loc = builder.getUnknownLoc();

    module_ = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module_.getBody());

    auto functionType = mlir::cxx::FunctionType::get(&context_, {}, {}, false);

    func_ = mlir::cxx::FuncOp::create(
        builder, loc, "f", functionType, mlir::cxx::LinkageKindAttr{},
        mlir::cxx::InlineKindAttr{}, mlir::cxx::VisibilityAttr{},
        mlir::StringAttr{}, mlir::StringAttr{}, mlir::StringAttr{},
        mlir::StringAttr{}, /*used=*/false, mlir::ArrayAttr{},
        mlir::ArrayAttr{}, mlir::StringAttr{},
        mlir::cxx::CallingConventionAttr{});

    builder.setInsertionPointToEnd(builder.createBlock(&func_.getBody()));
  }

  mlir::MLIRContext context_;
  ir::MlirEmitter emitter_{context_, nullptr};
  mlir::ModuleOp module_;
  mlir::cxx::FuncOp func_;
};

TEST_F(EmitterFixture, ClassDefinitionsPreserveLayout) {
  auto classType = emitter_.declareClassType("Layout");
  const ir::TypeRef members[] = {emitter_.integerType(32)};
  emitter_.defineClassType(classType, members, true, 0, {});
  emitter_.defineClassType(classType, members, true, 0, {});
  auto type = mlir::cast<mlir::cxx::ClassType>(emitter_.type(classType));
  ASSERT_EQ(type.getBody().size(), 1u);
  EXPECT_EQ(type.getBody().front(), emitter_.type(members[0]));
  EXPECT_TRUE(type.isPacked());
}

TEST_F(EmitterFixture, ClassDefinitionsPreserveAlignment) {
  auto classType = emitter_.declareClassType("Aligned");
  const ir::TypeRef members[] = {emitter_.integerType(32)};
  emitter_.defineClassType(classType, members, false, 16, {});
  auto type = mlir::cast<mlir::cxx::ClassType>(emitter_.type(classType));
  EXPECT_EQ(type.getAlignment(), 16u);
  EXPECT_FALSE(type.isPacked());
}

TEST_F(EmitterFixture, ClassDefinitionsPreservePadding) {
  auto classType = emitter_.declareClassType("Padded");
  const ir::TypeRef members[] = {emitter_.integerType(8),
                                 emitter_.integerType(8),
                                 emitter_.integerType(32)};
  const std::uint32_t padding[] = {1};
  emitter_.defineClassType(classType, members, false, 0, padding);
  auto type = mlir::cast<mlir::cxx::ClassType>(emitter_.type(classType));
  EXPECT_FALSE(type.isPaddingElement(0));
  EXPECT_TRUE(type.isPaddingElement(1));
  EXPECT_FALSE(type.isPaddingElement(2));
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::Type(type).print(stream);
  EXPECT_EQ(text, "!cxx.class<\"Padded\", (i8, padding i8, i32)>");
}

#if GTEST_HAS_DEATH_TEST
TEST_F(EmitterFixture, ConflictingClassDefinitionsFail) {
  auto classType = emitter_.declareClassType("Conflict");
  const ir::TypeRef members[] = {emitter_.integerType(32)};
  const ir::TypeRef differentMembers[] = {emitter_.integerType(64)};
  emitter_.defineClassType(classType, members, false, 0, {});
  EXPECT_DEATH(
      emitter_.defineClassType(classType, differentMembers, false, 0, {}),
      "conflicting definition of class 'Conflict'");
  EXPECT_DEATH(emitter_.defineClassType(classType, members, true, 0, {}),
               "conflicting definition of class 'Conflict'");
  const std::uint32_t padding[] = {0};
  EXPECT_DEATH(emitter_.defineClassType(classType, members, false, 0, padding),
               "conflicting definition of class 'Conflict'");
  auto type = mlir::cast<mlir::cxx::ClassType>(emitter_.type(classType));
  EXPECT_EQ(type.getBody().front(), emitter_.type(members[0]));
  EXPECT_FALSE(type.isPacked());
}

TEST_F(EmitterFixture, InvalidFloatingTypeFails) {
  EXPECT_DEATH((void)emitter_.floatingType(static_cast<ir::FloatKind>(99)),
               "invalid floating type kind");
}
#endif

TEST_F(EmitterFixture, DefaultBlockRefIsNull) {
  ir::BlockRef ref;
  ASSERT_FALSE(static_cast<bool>(ref));
  ASSERT_EQ(emitter_.block(ref), nullptr);
}

TEST_F(EmitterFixture, RoundTripsABlockRef) {
  const auto ref = emitter_.createBlock(ir::FunctionRef{});

  ASSERT_TRUE(static_cast<bool>(ref));

  auto block = emitter_.block(ref);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->getParent(), &func_.getBody());
  ASSERT_EQ(emitter_.wrap(block), ref);
}

TEST_F(EmitterFixture, DistinctBlocksGetDistinctRefs) {
  const auto first = emitter_.createBlock(ir::FunctionRef{});
  const auto second = emitter_.createBlock(ir::FunctionRef{});

  ASSERT_NE(first, second);
  ASSERT_NE(emitter_.block(first), emitter_.block(second));
}

TEST_F(EmitterFixture, RebindingPointersPreservesAnnotations) {
  auto integer = emitter_.integerType(32);
  auto byte = emitter_.integerType(8);
  auto plain = emitter_.pointerType(integer);
  auto zero = emitter_.pointerType(integer, 0);
  auto one = emitter_.pointerType(integer, 1);
  EXPECT_EQ(emitter_.rebindPointerType(plain, byte),
            emitter_.pointerType(byte));
  EXPECT_EQ(emitter_.rebindPointerType(zero, byte),
            emitter_.pointerType(byte, 0));
  EXPECT_EQ(emitter_.rebindPointerType(one, byte),
            emitter_.pointerType(byte, 1));
}

TEST_F(EmitterFixture, TracksTheInsertionBlock) {
  const auto entry = emitter_.insertionBlock();
  ASSERT_EQ(emitter_.block(entry), &func_.getBody().front());

  const auto block = emitter_.createBlock(ir::FunctionRef{});
  emitter_.setInsertionBlock(block);

  ASSERT_EQ(emitter_.insertionBlock(), block);
  ASSERT_EQ(emitter_.builder().getInsertionBlock(), emitter_.block(block));

  emitter_.setInsertionBlock(entry);
  ASSERT_EQ(emitter_.insertionBlock(), entry);
}

TEST_F(EmitterFixture, CleanupRegionsStayInTheirBlockWhenAnAnchorIsHoisted) {
  auto& builder = emitter_.builder();
  const auto loc = builder.getUnknownLoc();
  const auto entry = emitter_.insertionBlock();
  const auto body = emitter_.createBlock(ir::FunctionRef{});
  emitter_.setInsertionBlock(body);
  auto before = emitter_.constantInt(loc, emitter_.integerType(32), 7);
  auto address =
      emitter_.allocate(loc, emitter_.pointerType(emitter_.integerType(32)), 4);

  // Both live regions are anchored to the allocation about to be hoisted.
  ir::Emitter& backend = emitter_;
  const auto outer = backend.beginCleanupRegion();
  const auto inner = backend.beginCleanupRegion();
  const auto evaluation = emitter_.createBlock(ir::FunctionRef{});
  emitter_.setInsertionBlock(evaluation);
  const auto first = backend.activateConditionalCleanup(address, entry, inner);
  backend.endCleanupRegion(inner);
  const auto second = backend.activateConditionalCleanup(address, entry, outer);
  backend.endCleanupRegion(outer);

  ASSERT_EQ(emitter_.value(address).getDefiningOp()->getBlock(),
            emitter_.block(entry));
  ASSERT_EQ(emitter_.insertionBlock(), evaluation);
  for (auto flag : {first, second}) {
    ASSERT_EQ(emitter_.value(flag).getDefiningOp()->getBlock(),
              emitter_.block(entry));
    unsigned stores = 0;
    for (auto* user : emitter_.value(flag).getUsers()) {
      auto store = mlir::dyn_cast<mlir::cxx::StoreOp>(user);
      ASSERT_TRUE(store);
      auto constant =
          store->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
      ASSERT_TRUE(constant);
      auto bit = mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt();
      ASSERT_EQ(store->getBlock(), emitter_.block(bit ? evaluation : body));
      if (!bit)
        ASSERT_TRUE(
            emitter_.value(before).getDefiningOp()->isBeforeInBlock(store));
      ++stores;
    }
    ASSERT_EQ(stores, 2u);
  }
}

TEST_F(EmitterFixture, CleanupFlagInitializationFollowsItsEntryAllocation) {
  auto& builder = emitter_.builder();
  const auto loc = builder.getUnknownLoc();
  const auto entry = emitter_.insertionBlock();
  ir::Emitter& backend = emitter_;
  const auto region = backend.beginCleanupRegion();
  auto address =
      emitter_.allocate(loc, emitter_.pointerType(emitter_.integerType(32)), 4);
  auto sentinel = emitter_.constantInt(loc, emitter_.integerType(32), 9);
  auto sentinelOp = emitter_.value(sentinel).getDefiningOp();
  builder.setInsertionPoint(sentinelOp);
  const auto flag = backend.activateConditionalCleanup(address, entry, region);
  backend.endCleanupRegion(region);

  ASSERT_EQ(&*builder.getInsertionPoint(), sentinelOp);
  auto flagOp = emitter_.value(flag).getDefiningOp();
  unsigned stores = 0;
  for (auto* user : emitter_.value(flag).getUsers()) {
    ASSERT_TRUE(mlir::isa<mlir::cxx::StoreOp>(user));
    ASSERT_EQ(user->getBlock(), emitter_.block(entry));
    ASSERT_TRUE(flagOp->isBeforeInBlock(user));
    ASSERT_TRUE(user->isBeforeInBlock(sentinelOp));
    ++stores;
  }
  ASSERT_EQ(stores, 2u);
}

TEST_F(EmitterFixture, ModuleLifecycleRestoresSymbolLookupAndKeepsArtifacts) {
  ir::Emitter& backend = emitter_;
  ASSERT_FALSE(emitter_.module(ir::ModuleRef{}));
  const auto outer = backend.beginModule({.name = "outer"});
  auto outerModule = emitter_.module(outer);
  ASSERT_EQ(emitter_.module(), outerModule);
  ASSERT_EQ(emitter_.builder().getInsertionBlock(), outerModule.getBody());
  ASSERT_TRUE(mlir::isa<mlir::UnknownLoc>(outerModule.getLoc()));
  ASSERT_FALSE(outerModule->hasAttr("cxx.triple"));
  ASSERT_FALSE(outerModule->hasAttr("cxx.data-layout"));
  ASSERT_FALSE(outerModule->hasAttr(mlir::DLTIDialect::kDataLayoutAttrName));

  const auto loc = emitter_.builder().getUnknownLoc();
  auto type = emitter_.functionType({}, {}, false);
  const auto outerFunction =
      emitter_.declareFunction(loc, {.name = "same", .type = type});
  ASSERT_EQ(backend.findFunction("same"), outerFunction);

  const auto inner = backend.beginModule({.name = "inner"});
  ASSERT_NE(inner, outer);
  ASSERT_FALSE(backend.findFunction("same"));
  const auto innerFunction =
      emitter_.declareFunction(loc, {.name = "same", .type = type});
  ASSERT_NE(innerFunction, outerFunction);
  ASSERT_EQ(backend.findFunction("same"), innerFunction);
  backend.endModule();
  ASSERT_EQ(backend.findFunction("same"), outerFunction);
  ASSERT_EQ(emitter_.module(), outerModule);
  backend.endModule();
  ASSERT_FALSE(emitter_.module());
  ASSERT_FALSE(backend.findFunction("same"));
  ASSERT_EQ(emitter_.module(outer), outerModule);
  ASSERT_TRUE(emitter_.module(inner));
}

TEST_F(EmitterFixture, SymbolLookupTracksNewFunctionsGlobalsAndVTables) {
  ir::Emitter& backend = emitter_;
  emitter_.builder().setInsertionPointToEnd(module_.getBody());
  auto module = backend.beginModule({.name = "symbols"});
  auto integer = backend.integerType(32);
  auto type = backend.functionType({}, {}, false);
  // Populate the index before adding symbols, including failed lookups.
  EXPECT_FALSE(backend.findFunction("function"));
  EXPECT_FALSE(backend.findGlobal("global"));
  EXPECT_FALSE(backend.symbolExists("vtable"));

  auto function =
      backend.declareFunction({}, {.name = "function", .type = type});
  backend.setModuleInsertionPoint(true);
  auto global = backend.declareGlobal({}, {.name = "global", .type = integer});
  std::array slots{function};
  std::array tables{ir::VTableTableInfo{.slots = slots}};
  backend.defineVTable({}, {.name = "vtable", .tables = tables});

  EXPECT_EQ(backend.findFunction("function"), function);
  EXPECT_EQ(backend.findGlobal("global"), global);
  EXPECT_TRUE(backend.symbolExists("function"));
  EXPECT_TRUE(backend.symbolExists("global"));
  EXPECT_TRUE(backend.symbolExists("vtable"));
  EXPECT_FALSE(backend.findFunction("global"));
  EXPECT_FALSE(backend.findGlobal("function"));
  EXPECT_FALSE(backend.findFunction("vtable"));
  EXPECT_FALSE(backend.findGlobal("vtable"));
  backend.endModule();
  EXPECT_FALSE(backend.symbolExists("function"));
  EXPECT_TRUE(succeeded(mlir::verify(emitter_.module(module))));
}

TEST_F(EmitterFixture, DuplicateSymbolsStillFailVerification) {
  ir::Emitter& backend = emitter_;
  emitter_.builder().setInsertionPointToEnd(module_.getBody());
  auto module = backend.beginModule({.name = "duplicates"});
  auto type = backend.functionType({}, {}, false);
  auto first = backend.declareFunction({}, {.name = "same", .type = type});
  auto second = backend.declareFunction({}, {.name = "same", .type = type});
  EXPECT_NE(first, second);
  EXPECT_EQ(emitter_.function(first).getSymName(), "same");
  EXPECT_EQ(emitter_.function(second).getSymName(), "same");
  backend.endModule();
  bool diagnosed = false;
  mlir::ScopedDiagnosticHandler diagnostics(&context_, [&](mlir::Diagnostic&) {
    diagnosed = true;
    return mlir::success();
  });
  EXPECT_TRUE(failed(mlir::verify(emitter_.module(module))));
  EXPECT_TRUE(diagnosed);
}

TEST_F(EmitterFixture, NestedFunctionsPreserveSourceOwnership) {
  ir::Emitter& backend = emitter_;
  emitter_.builder().setInsertionPointToEnd(module_.getBody());
  auto module = backend.beginModule({.name = "source-contexts"});
  auto type = emitter_.functionType({}, {}, false);
  auto outer = backend.declareFunction({}, {.name = "outer", .type = type});
  auto inner = backend.declareFunction({}, {.name = "inner", .type = type});
  auto* observer =
      static_cast<ir::MlirSourceContexts*>(emitter_.enableSourceContexts({}));
  auto& builder = emitter_.builder();
  auto loc = builder.getUnknownLoc();
  emitter_.beginFunctionBody(outer);
  auto block = emitter_.createBlock(outer);
  emitter_.setInsertionBlock(block);
  auto context = observer->beginContext(ASTKind::CompoundStatement);
  auto first = emitter_.constantInt(loc, emitter_.integerType(32), 7);
  auto* firstOp = emitter_.value(first).getDefiningOp();
  EXPECT_EQ(observer->contextOf(firstOp), context);
  emitter_.beginFunctionBody(inner);
  emitter_.setInsertionBlock(emitter_.createBlock(inner));
  auto innerContext = observer->currentContext();
  auto nested = emitter_.constantInt(loc, emitter_.integerType(32), 9);
  EXPECT_EQ(observer->contextOf(emitter_.value(nested).getDefiningOp()),
            innerContext);
  mlir::cxx::ReturnOp::create(builder, loc, mlir::ValueRange{});
  emitter_.endFunctionBody(inner);
  emitter_.setInsertionBlock(block);
  EXPECT_EQ(observer->currentContext(), context);
  EXPECT_EQ(observer->contextOf(firstOp), context);
  auto second = emitter_.constantInt(loc, emitter_.integerType(32), 11);
  EXPECT_EQ(observer->contextOf(emitter_.value(second).getDefiningOp()),
            context);
  mlir::cxx::ReturnOp::create(builder, loc, mlir::ValueRange{});
  observer->endContext(context);
  emitter_.endFunctionBody(outer);
  EXPECT_EQ(observer->currentContext(), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(emitter_.module(module))));
  backend.endModule();
}

TEST_F(EmitterFixture, ModuleMetadataUsesTheRequestedSourceAndTarget) {
  ir::Emitter& backend = emitter_;
  const auto ref = backend.beginModule({.name = "example.cc",
                                        .sourceFile = "example.cc",
                                        .targetTriple = "wasm32-unknown-wasi"});
  auto module = emitter_.module(ref);
  auto loc = mlir::dyn_cast<mlir::FileLineColLoc>(module.getLoc());
  ASSERT_TRUE(loc);
  ASSERT_EQ(loc.getFilename().getValue(), "example.cc");
  ASSERT_EQ(loc.getLine(), 0u);
  ASSERT_EQ(loc.getColumn(), 0u);
  ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("cxx.triple").getValue(),
            "wasm32-unknown-wasi");
  auto layout = module->getAttrOfType<mlir::StringAttr>("cxx.data-layout");
  ASSERT_TRUE(layout);
  ASSERT_EQ(llvm::DataLayout(layout.getValue()).getPointerSizeInBits(), 32u);
  ASSERT_TRUE(module->hasAttr(mlir::DLTIDialect::kDataLayoutAttrName));
  backend.endModule();
}

TEST_F(EmitterFixture, InsertionGuardRestoresPositionWithinBlockAndUnsetState) {
  auto& builder = emitter_.builder();
  const auto loc = builder.getUnknownLoc();
  const auto type = emitter_.integerType(32);
  auto first = emitter_.constantInt(loc, type, 1);
  auto last = emitter_.constantInt(loc, type, 3);
  builder.setInsertionPoint(emitter_.value(last).getDefiningOp());
  {
    ir::InsertionGuard guard(emitter_);
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
  }
  auto middle = emitter_.constantInt(loc, type, 2);
  EXPECT_EQ(emitter_.value(first).getDefiningOp()->getNextNode(),
            emitter_.value(middle).getDefiningOp());
  EXPECT_EQ(emitter_.value(middle).getDefiningOp()->getNextNode(),
            emitter_.value(last).getDefiningOp());
  auto block = emitter_.insertionBlock();
  builder.clearInsertionPoint();
  {
    ir::InsertionGuard guard(emitter_);
    emitter_.setInsertionBlock(block);
  }
  EXPECT_FALSE(emitter_.insertionBlock());
}

TEST_F(EmitterFixture,
       LiteralInitializersPreserveNestedValuesAndEmbeddedZeros) {
  context_.loadDialect<mlir::LLVM::LLVMDialect>();
  ir::Emitter& backend = emitter_;
  (void)backend.beginModule({.name = "literals"});
  auto i32 = backend.integerType(32);
  auto f32 = backend.floatingType(ir::FloatKind::Single);
  auto init = ir::Initializer::aggregate(
      {ir::Initializer::integerValue(i32, -7),
       ir::Initializer::floatingValue(f32, 1.25),
       ir::Initializer::byteString(std::string_view{"a\0b", 3}),
       ir::Initializer::aggregate(
           {ir::Initializer::null(), ir::Initializer::zero()})});
  auto ref = backend.declareGlobal({}, {.name = "literal",
                                        .type = i32,
                                        .linkage = ir::Linkage::Internal,
                                        .isConstant = true,
                                        .alignment = 8,
                                        .initializer = init,
                                        .unknownLocation = true});
  auto global = emitter_.global(ref);
  auto values = global->getAttrOfType<mlir::ArrayAttr>("value");
  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(values[0]).getInt(), -7);
  EXPECT_DOUBLE_EQ(mlir::cast<mlir::FloatAttr>(values[1]).getValueAsDouble(),
                   1.25);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(values[2]).getValue(),
            llvm::StringRef("a\0b", 3));
  auto nested = mlir::cast<mlir::ArrayAttr>(values[3]);
  EXPECT_TRUE(mlir::isa<mlir::UnitAttr>(nested[0]));
  EXPECT_TRUE(mlir::isa<mlir::LLVM::ZeroAttr>(nested[1]));
  EXPECT_EQ(backend.globalLinkage(ref), ir::Linkage::Internal);
  EXPECT_EQ(global->getAttrOfType<mlir::IntegerAttr>("alignment").getInt(), 8);
  backend.endModule();
}

TEST_F(EmitterFixture, DebugMetadataDoesNotRequireAParserTranslationUnit) {
  ir::Emitter& backend = emitter_;
  auto module = backend.beginModule({.name = "debug.cc"});
  auto function =
      backend.declareFunction({}, {.name = "debugged",
                                   .type = backend.functionType({}, {}, false),
                                   .linkage = ir::Linkage::Internal});
  auto debug = backend.debug();
  ASSERT_NE(debug, nullptr);
  auto unit =
      debug->compileUnit({.file = "/source/debug.cc", .directory = "/source"});
  auto integer = debug->basicType(
      {.name = "int", .sizeInBits = 32, .encoding = ir::DebugEncoding::Signed});
  std::array signature{integer};
  auto subprogram =
      debug->defineFunction(function, {},
                            {.name = "sourceName",
                             .type = debug->subroutineType(signature),
                             .location = {"/source/debug.cc", 7, 1},
                             .scopeLine = 8});
  auto fused = mlir::cast<mlir::FusedLoc>(emitter_.function(function).getLoc());
  auto metadata = mlir::cast<mlir::LLVM::DISubprogramAttr>(fused.getMetadata());
  EXPECT_EQ(metadata.getCompileUnit().getFile().getName().getValue(),
            "debug.cc");
  EXPECT_EQ(metadata.getCompileUnit().getFile().getDirectory().getValue(),
            "/source");
  EXPECT_EQ(metadata.getName().getValue(), "sourceName");
  EXPECT_EQ(metadata.getLinkageName().getValue(), "debugged");
  EXPECT_EQ(metadata.getLine(), 7u);
  EXPECT_EQ(metadata.getScopeLine(), 8u);
  EXPECT_EQ(metadata.getSubprogramFlags(),
            mlir::LLVM::DISubprogramFlags::Definition |
                mlir::LLVM::DISubprogramFlags::LocalToUnit);
  backend.beginFunctionBody(function);
  auto block = debug->lexicalBlock(subprogram, {"/source/debug.cc", 9, 3});
  auto address =
      backend.allocate({}, backend.pointerType(backend.integerType(32)), 4);
  debug->localVariable(address, {.name = "local",
                                 .scope = block,
                                 .type = integer,
                                 .location = {"/source/debug.cc", 10, 5}});
  auto variable =
      emitter_.value(address)
          .getDefiningOp()
          ->getAttrOfType<mlir::LLVM::DILocalVariableAttr>("cxx.di_local");
  ASSERT_TRUE(variable);
  EXPECT_EQ(variable.getName().getValue(), "local");
  EXPECT_EQ(variable.getLine(), 10u);
  auto lexical =
      mlir::cast<mlir::LLVM::DILexicalBlockAttr>(variable.getScope());
  EXPECT_EQ(lexical.getScope(), metadata);
  EXPECT_EQ(lexical.getLine(), 9u);
  EXPECT_EQ(lexical.getColumn(), 3u);
  debug->localVariable(address, {.name = "this",
                                 .scope = subprogram,
                                 .type = integer,
                                 .argument = 1,
                                 .isObjectParameter = true});
  variable =
      emitter_.value(address)
          .getDefiningOp()
          ->getAttrOfType<mlir::LLVM::DILocalVariableAttr>("cxx.di_local");
  EXPECT_EQ(variable.getLine(), 7u);
  EXPECT_EQ(variable.getArg(), 1u);
  EXPECT_EQ(variable.getFlags(), mlir::LLVM::DIFlags::Artificial |
                                     mlir::LLVM::DIFlags::ObjectPointer);
  EXPECT_TRUE(static_cast<bool>(unit));
  EXPECT_TRUE(static_cast<bool>(module));
  backend.endFunctionBody(function);
  backend.endModule();
}

class IRVerification : public ::testing::Test {
 protected:
  IRVerification() { context_.loadDialect<mlir::LLVM::LLVMDialect>(); }

  auto invalidModule() -> mlir::OwningOpRef<mlir::ModuleOp> {
    // Parse a structurally valid operation with an invalid return type, so only
    // the optional IR verifiers reject it.
    return mlir::parseSourceString<mlir::ModuleOp>(
        "module { llvm.func @bad() -> i32 { llvm.return } }",
        mlir::ParserConfig(&context_, /*verifyAfterParse=*/false));
  }

  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};
};

TEST_F(IRVerification, ExplicitPipelinesHonorVerificationSetting) {
  mlir::ScopedDiagnosticHandler diagnostics(&context_,
                                            [](mlir::Diagnostic&) {});
  for (bool verify : {true, false}) {
    auto module = invalidModule();
    ASSERT_TRUE(module);
    EXPECT_EQ(succeeded(runMLIRPassPipeline(*module, "canonicalize", verify)),
              !verify);
  }
}

TEST_F(IRVerification, LoweringHonorsVerificationSetting) {
  mlir::ScopedDiagnosticHandler diagnostics(&context_,
                                            [](mlir::Diagnostic&) {});
  for (bool verify : {true, false}) {
    auto module = invalidModule();
    ASSERT_TRUE(module);
    EXPECT_EQ(succeeded(lowerToMLIR(*module, {}, {.verifyIR = verify})),
              !verify);
  }
}

TEST_F(IRVerification, LLVMExportHonorsVerificationSetting) {
  for (bool verify : {true, false}) {
    auto module = invalidModule();
    ASSERT_TRUE(module);
    llvm::LLVMContext context;
    testing::internal::CaptureStderr();
    auto result = exportToLLVMIR(*module, context, verify);
    auto diagnostics = testing::internal::GetCapturedStderr();
    EXPECT_EQ(bool(result), !verify);
    EXPECT_EQ(diagnostics.empty(), !verify);
  }
}

}  // namespace
