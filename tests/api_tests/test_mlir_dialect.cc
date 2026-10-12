#include <cxx/mlir/cxx_dialect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <initializer_list>
#include <limits>

#undef GTEST_HAS_CXXABI_H_
#define GTEST_HAS_CXXABI_H_ 0
#include <gtest/gtest.h>

template <typename Attr, typename Enum>
void verifyEnumAttributes(mlir::MLIRContext& context,
                          std::initializer_list<Enum> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  for (auto value : values) {
    auto attribute = Attr::get(&context, value);
    EXPECT_EQ(attribute.getValue(), value);
    EXPECT_EQ(attribute, Attr::get(&context, value));
    for (auto previous : attributes) EXPECT_NE(attribute, previous);
    attributes.push_back(attribute);

    std::string text;
    llvm::raw_string_ostream stream(text);
    mlir::Attribute(attribute).print(stream);
    EXPECT_EQ(mlir::parseAttribute(text, &context), attribute);
  }
}

TEST(CxxDialect, EnumAttributesPreserveValuesAndIdentity) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::cxx::CxxDialect>();
  using namespace mlir::cxx;
  verifyEnumAttributes<InlineKindAttr>(
      context,
      {InlineKind::NoInline, InlineKind::InlineHint, InlineKind::AlwaysInline});
  verifyEnumAttributes<LinkageKindAttr>(
      context,
      {LinkageKind::External, LinkageKind::Internal, LinkageKind::LinkOnceODR,
       LinkageKind::WeakODR, LinkageKind::AvailableExternally,
       LinkageKind::Appending, LinkageKind::Weak, LinkageKind::ExternalWeak});
  verifyEnumAttributes<VisibilityAttr>(
      context,
      {Visibility::Default, Visibility::Hidden, Visibility::Protected});
  verifyEnumAttributes<CallingConventionAttr>(
      context, {CallingConvention::C, CallingConvention::CxxFastTls});
}

TEST(CxxDialect, ArraySizesRoundTripWithoutTruncation) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::cxx::CxxDialect>();
  auto element = mlir::IntegerType::get(&context, 8);
  constexpr std::uint64_t sizes[] = {0, 2, std::uint64_t{1} << 32,
                                     std::numeric_limits<std::uint64_t>::max()};
  for (auto size : sizes) {
    auto type = mlir::cxx::ArrayType::get(&context, element, size);
    EXPECT_EQ(type.getSize(), size);
    EXPECT_EQ(type, mlir::cxx::ArrayType::get(&context, element, size));
    std::string text;
    llvm::raw_string_ostream stream(text);
    mlir::Type(type).print(stream);
    EXPECT_EQ(mlir::parseType(text, &context), type);
  }
  EXPECT_NE(
      mlir::cxx::ArrayType::get(&context, element, 0),
      mlir::cxx::ArrayType::get(&context, element, std::uint64_t{1} << 32));
}

constexpr unsigned kClassCount = 18;

class TypeGraphFixture : public ::testing::Test {
 protected:
  TypeGraphFixture() {
    context_.loadDialect<mlir::cxx::CxxDialect>();
    for (unsigned i = 0; i < kClassCount; ++i) {
      classes_.push_back(mlir::cxx::ClassType::getNamed(
          &context_, "Class" + std::to_string(i)));
    }
    for (unsigned i = 0; i < kClassCount; ++i) {
      auto pointer = mlir::cxx::PointerType::get(
          &context_, classes_[(i + 1) % kClassCount]);
      EXPECT_TRUE(succeeded(classes_[i].setBody({pointer, pointer}, i % 2)));
    }
  }

  void verifyLayouts(mlir::MLIRContext& context) {
    for (unsigned i = 0; i < kClassCount; ++i) {
      auto type =
          mlir::cxx::ClassType::getNamed(&context, "Class" + std::to_string(i));
      auto next = mlir::cxx::ClassType::getNamed(
          &context, "Class" + std::to_string((i + 1) % kClassCount));
      auto pointer = mlir::cxx::PointerType::get(&context, next);
      ASSERT_EQ(type.getBody().size(), 2u);
      EXPECT_EQ(type.getBody()[0], pointer);
      EXPECT_EQ(type.getBody()[1], pointer);
      EXPECT_EQ(type.isPacked(), bool(i % 2));
    }
  }

  mlir::MLIRContext context_;
  llvm::SmallVector<mlir::cxx::ClassType> classes_;
};

TEST_F(TypeGraphFixture, StandaloneTypesPrintSharedBodiesOnce) {
  auto pointer = mlir::cxx::PointerType::get(&context_, classes_.front());
  auto array = mlir::cxx::ArrayType::get(&context_, classes_.front(), 2);
  auto function =
      mlir::cxx::FunctionType::get(&context_, {pointer}, {array}, true);
  llvm::SmallVector<mlir::Type> roots = {classes_.front(), pointer, array,
                                         function};
  for (auto type : roots) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    type.print(stream);
    ASSERT_LT(text.size(), 8192u);
    mlir::MLIRContext parsedContext;
    parsedContext.loadDialect<mlir::cxx::CxxDialect>();
    ASSERT_TRUE(mlir::parseType(text, &parsedContext));
    verifyLayouts(parsedContext);
  }
}

TEST_F(TypeGraphFixture, ModuleAliasesPreserveSharedClassLayouts) {
  mlir::OpBuilder builder(&context_);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());
  mlir::OperationState state(builder.getUnknownLoc(), "cxx.global");
  state.addAttribute("sym_name", builder.getStringAttr("root"));
  state.addAttribute("global_type", mlir::TypeAttr::get(classes_.front()));
  state.addRegion();
  builder.create(state);
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  module.erase();
  ASSERT_LT(text.size(), 65536u);
  mlir::MLIRContext parsedContext;
  parsedContext.loadDialect<mlir::cxx::CxxDialect>();
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &parsedContext);
  ASSERT_TRUE(parsed);
  verifyLayouts(parsedContext);
}
