#include <cxx/lsp/types.h>
#include <gtest/gtest.h>

#include "cxx/lsp/enums.h"
#include "cxx/lsp/fwd.h"

TEST(SampleTest, AssertionTrue) {
  using namespace cxx::lsp;

  // storage
  auto store1 = json::object();
  auto store2 = json::object();
  auto store3 = json::object();

  InitializeResult initializeResult{store1};
  ASSERT_TRUE(!initializeResult);

  auto serverInfo = ServerInfo{store2}.name("cxx").version("0.1.0");
  ASSERT_EQ(serverInfo.name(), "cxx");
  ASSERT_EQ(serverInfo.version(), "0.1.0");
  ASSERT_TRUE(serverInfo);

  auto capabilities = ServerCapabilities{store3}.textDocumentSync(
      TextDocumentSyncKind::kIncremental);

  ASSERT_TRUE(capabilities);

  ASSERT_TRUE(capabilities.textDocumentSync().has_value());

  ASSERT_TRUE(std::holds_alternative<TextDocumentSyncKind>(
      *capabilities.textDocumentSync()));

  ASSERT_EQ(std::get<TextDocumentSyncKind>(*capabilities.textDocumentSync()),
            TextDocumentSyncKind::kIncremental);

  initializeResult.serverInfo(std::move(serverInfo));
  ASSERT_TRUE(initializeResult.serverInfo());

  initializeResult.capabilities(std::move(capabilities));
  ASSERT_TRUE(initializeResult.capabilities());

  ASSERT_TRUE(initializeResult);
}
