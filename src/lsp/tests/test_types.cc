#include <cxx/lsp/types.h>
#include <gtest/gtest.h>

#include <iostream>

#include "cxx/lsp/enums.h"
#include "cxx/lsp/fwd.h"

using namespace cxx::lsp;

TEST(LSP, Initialization) {
  // storage
  auto storage = json::object();

  InitializeResult initializeResult{storage};
  ASSERT_TRUE(!initializeResult);

  auto serverInfo =
      initializeResult.serverInfo<ServerInfo>().name("cxx").version("0.1.0");

  ASSERT_TRUE(initializeResult.serverInfo().has_value());
  ASSERT_EQ(serverInfo.name(), "cxx");
  ASSERT_EQ(serverInfo.version(), "0.1.0");

  auto capabilities = initializeResult.capabilities().textDocumentSync(
      TextDocumentSyncKind::kIncremental);

  ASSERT_TRUE(capabilities);

  ASSERT_TRUE(capabilities.textDocumentSync().has_value());

  ASSERT_TRUE(std::holds_alternative<TextDocumentSyncKind>(
      *capabilities.textDocumentSync()));

  ASSERT_EQ(std::get<TextDocumentSyncKind>(*capabilities.textDocumentSync()),
            TextDocumentSyncKind::kIncremental);

  capabilities.completionProvider<CompletionOptions>().triggerCharacters(
      std::vector<std::string>{".", ">", ":"});

  ASSERT_TRUE(capabilities.completionProvider().has_value());

  auto completionProvider =
      capabilities.completionProvider<CompletionOptions>();

  ASSERT_TRUE(completionProvider.triggerCharacters().has_value());

  auto triggerCharacters = *completionProvider.triggerCharacters();

  ASSERT_EQ(triggerCharacters.size(), 3);
  ASSERT_EQ(triggerCharacters.at(0), ".");
  ASSERT_EQ(triggerCharacters.at(1), ">");
  ASSERT_EQ(triggerCharacters.at(2), ":");

  ASSERT_TRUE(initializeResult.serverInfo());
  ASSERT_TRUE(initializeResult.capabilities());

  ASSERT_TRUE(initializeResult);
}

TEST(LSP, ArrayProperty) {
  json storage = json::object();

  ConfigurationParams configurationParams{storage};

  ASSERT_FALSE(configurationParams);

  auto items = configurationParams.items();

  ASSERT_TRUE(items.empty());

  ASSERT_TRUE(configurationParams);
}

TEST(LSP, MapProperty) {
  json storage = json::object();

  DocumentDiagnosticReportPartialResult documentDiagnosticReportPartialResult{
      storage};

  ASSERT_FALSE(documentDiagnosticReportPartialResult);

  auto relatedDocuments =
      documentDiagnosticReportPartialResult.relatedDocuments();

  ASSERT_TRUE(relatedDocuments.empty());

  ASSERT_TRUE(documentDiagnosticReportPartialResult);
}

TEST(LSP, StringProperty) {
  json storage = json::object();

  Location location{storage};

  ASSERT_FALSE(location);

  ASSERT_EQ(location.uri(), "");

  location.uri("file:///path/to/file");
  ASSERT_EQ(location.uri(), "file:///path/to/file");

  auto range = location.range();
  ASSERT_EQ(range.start().line(), 0);
  ASSERT_EQ(range.start().character(), 0);

  range.start().line(1);
  range.start().character(2);
  range.end().line(3);
  range.end().character(4);

  ASSERT_EQ(range.start().line(), 1);
  ASSERT_EQ(range.start().character(), 2);
  ASSERT_EQ(range.end().line(), 3);
  ASSERT_EQ(range.end().character(), 4);

  ASSERT_TRUE(location);
}

TEST(LSP, StringArrayProperty) {
  json storage = json::object();

  auto textDocumentContentRegistrationOptions =
      TextDocumentContentRegistrationOptions{storage};

  auto schemas = textDocumentContentRegistrationOptions.schemes();
  ASSERT_TRUE(schemas.empty());

  schemas.emplace_back("file");
  schemas.emplace_back("http");

  ASSERT_EQ(schemas.at(0), "file");
  ASSERT_EQ(schemas.at(1), "http");

  ASSERT_EQ(textDocumentContentRegistrationOptions.schemes().at(0), "file");
  ASSERT_EQ(textDocumentContentRegistrationOptions.schemes().at(1), "http");
}

TEST(LSP, VariantArrayProperty) {
  auto storage = json::object();

  NotebookDocumentSyncRegistrationOptions
      notebookDocumentSyncRegistrationOptions{storage};

  auto notebookSelector =
      notebookDocumentSyncRegistrationOptions.notebookSelector();

  ASSERT_TRUE(notebookSelector.empty());

  auto item = notebookSelector.emplace_back<NotebookDocumentFilterWithCells>();

  ASSERT_FALSE(item.notebook().has_value());

  item.notebook("a_notebook");

  ASSERT_TRUE(item.notebook().has_value());

  ASSERT_EQ(std::get<std::string>(*item.notebook()), "a_notebook");

  ASSERT_EQ(notebookSelector.size(), 1);
}

TEST(LSP, CompletionList) {
  const char *kResponse = R"(
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "isIncomplete": false,
    "items": [
      {
        "filterText": "include",
        "insertText": "include \"$0\"",
        "insertTextFormat": 2,
        "kind": 15,
        "label": " include",
        "sortText": "000000001",
        "labelDetails": {
          "detail": " \"header\""
        },
        "textEdit": {
          "newText": "include \"$0\"",
          "range": {
            "end": {
              "character": 4,
              "line": 0
            },
            "start": {
              "character": 1,
              "line": 0
            }
          }
        }
      }
    ]
  }
}
)";

  auto response = json::parse(kResponse)["result"];

  CompletionList completionList{response};

  ASSERT_TRUE(completionList);
  ASSERT_FALSE(completionList.isIncomplete());

  ASSERT_EQ(completionList.items().size(), 1);

  auto items = completionList.items();
  auto item = items.at(0);
  ASSERT_EQ(*item.filterText(), "include");
  ASSERT_EQ(*item.insertText(), "include \"$0\"");
  ASSERT_EQ(item.insertTextFormat(), InsertTextFormat::kSnippet);
  ASSERT_EQ(*item.kind(), CompletionItemKind::kSnippet);
  ASSERT_EQ(item.label(), " include");
  ASSERT_EQ(*item.labelDetails()->detail(), " \"header\"");
  ASSERT_EQ(item.sortText(), "000000001");

  ASSERT_TRUE(std::holds_alternative<TextEdit>(*item.textEdit()));
  auto textEdit = std::get<TextEdit>(*item.textEdit());

  ASSERT_EQ(textEdit.newText(), "include \"$0\"");
  auto range = textEdit.range();
  ASSERT_EQ(range.start().line(), 0);
  ASSERT_EQ(range.start().character(), 1);
  ASSERT_EQ(range.end().line(), 0);
  ASSERT_EQ(range.end().character(), 4);
}

TEST(LSP, CreateCompletionList) {
  auto storage = json::object();

  CompletionList completionList{storage};

  completionList.isIncomplete(false);

  auto item = completionList.items().emplace_back();
  item.filterText("include");
  item.insertText("include \"$0\"");
  item.insertTextFormat(InsertTextFormat::kSnippet);
  item.kind(CompletionItemKind::kSnippet);
  item.label(" include");

  item.labelDetails<CompletionItemLabelDetails>().detail(" \"header\"");
  item.sortText("000000001");

  auto textEdit = item.textEdit<TextEdit>();

  textEdit.newText("include \"$0\"");
  auto range = textEdit.range();
  range.start().line(0);
  range.start().character(1);
  range.end().line(0);
  range.end().character(4);

  ASSERT_TRUE(completionList);

  // std::cout << completionList.get().dump(2) << std::endl;
}