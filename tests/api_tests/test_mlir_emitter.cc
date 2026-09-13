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

#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/mlir_emitter.h>
#include <gtest/gtest.h>
#include <llvm/IR/DataLayout.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

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
        mlir::ArrayAttr{});

    builder.setInsertionPointToEnd(builder.createBlock(&func_.getBody()));
  }

  mlir::MLIRContext context_;
  ir::MlirEmitter emitter_{context_, nullptr};
  mlir::ModuleOp module_;
  mlir::cxx::FuncOp func_;
};

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

}  // namespace
