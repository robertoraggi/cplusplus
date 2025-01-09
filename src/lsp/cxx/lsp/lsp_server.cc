// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "lsp_server.h"

#include <cxx/ast.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/requests.h>
#include <cxx/lsp/types.h>
#include <cxx/name_printer.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>
#include <utf8/unchecked.h>

#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "cxx_document.h"

namespace cxx::lsp {

// TODO: move to a separate file
template <typename It>
inline auto skipBOM(It& it, It end) -> bool {
  if (it < end && *it == '\xEF') {
    if (it + 1 < end && it[1] == '\xBB') {
      if (it + 2 < end && it[2] == '\xBF') {
        it += 3;
        return true;
      }
    }
  }
  return false;
}

auto Server::Text::offsetAt(std::size_t line, std::size_t column) const
    -> std::size_t {
  if (line >= lineStartOffsets.size()) {
    return std::string::npos;
  }

  const auto lineStart = lineStartOffsets.at(line);
  const auto nextLineStart = line + 1 < lineStartOffsets.size()
                                 ? lineStartOffsets.at(line + 1)
                                 : value.size();

  const auto columnOffset = std::min(column, nextLineStart - lineStart);

  return lineStart + columnOffset;
}

void Server::Text::computeLineStartOffsets() {
  auto begin = value.begin();
  auto end = value.end();

  auto it = begin;
  skipBOM(it, end);

  lineStartOffsets.clear();
  lineStartOffsets.push_back(it - begin);

  while (it != end) {
    const auto ch = utf8::unchecked::next(it);
    if (ch == '\n') {
      lineStartOffsets.push_back(it - begin);
    }
  }
}

Server::Server(const CLI& cli)
    : cli(cli), input(std::cin), output(std::cout), log(std::cerr) {}

Server::~Server() {}

auto Server::start() -> int {
  trace_ = TraceValue::kOff;

  logTrace(std::format("Starting LSP server"));

  startWorkersIfNeeded();

  while (!done_ && input.good()) {
    if (done_) {
      break;
    }

    if (auto req = nextRequest()) {
      auto request = LSPRequest(req.value());
      visit(*this, request);
    }
  }

  stopWorkersIfNeeded();

  return 0;
}

auto Server::nextRequest() -> std::optional<json> {
  if (cli.opt_lsp_test) {
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty()) {
        continue;
      } else if (line.starts_with("#")) {
        continue;
      }
      return json::parse(line);
    }
    return std::nullopt;
  }

  const auto headers = readHeaders(input);

  // Get Content-Length
  const auto it = headers.find("Content-Length");

  if (it == headers.end()) {
    return std::nullopt;
  }

  const auto contentLength = std::stoi(it->second);

  // Read content
  std::string content(contentLength, '\0');
  input.read(content.data(), content.size());

  // Parse JSON
  auto request = json::parse(content);
  return request;
}

auto Server::readHeaders(std::istream& input)
    -> std::unordered_map<std::string, std::string> {
  std::unordered_map<std::string, std::string> headers;

  std::string line;

  while (std::getline(input, line)) {
    if (line.empty() || line == "\r") {
      break;
    }

    const auto pos = line.find_first_of(':');

    if (pos == std::string::npos) {
      continue;
    }

    auto name = line.substr(0, pos);
    auto value = line.substr(pos + 1);

    // trim whitespace
    name.erase(name.find_last_not_of(" \t\r\n") + 1);
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);

    headers.insert_or_assign(std::move(name), std::move(value));
  }

  return headers;
}

void Server::sendMessage(const json& message) {
#ifndef CXX_NO_THREADS
  auto locker = std::unique_lock(outputMutex_);
#endif

  if (cli.opt_lsp_test) {
    output << message.dump(2) << std::endl;
  } else {
    const auto text = message.dump();
    output << std::format("Content-Length: {}\r\n\r\n{}", text.size(), text);
  }

  output.flush();
}

void Server::sendToClient(LSPResponse response) {
  json& message = response.get();
  message["jsonrpc"] = "2.0";
  sendMessage(message);
}

void Server::sendToClient(LSPRequest notification) {
  json response = notification;
  response["jsonrpc"] = "2.0";
  sendMessage(response);
}

void Server::logTrace(std::string message, std::optional<std::string> verbose) {
  if (trace_ == TraceValue::kOff) {
    return;
  }

  withUnsafeJson([&](json storage) {
    LogTraceNotification logTrace(storage);
    logTrace.method("$/logTrace");
    logTrace.params().message(std::move(message));
    if (verbose.has_value()) {
      logTrace.params().verbose(std::move(*verbose));
    }
    sendToClient(logTrace);
  });
}

auto Server::pathFromUri(const std::string& uri) -> std::string {
  if (uri.starts_with("file://")) {
    return uri.substr(7);
  } else if (cli.opt_lsp_test && uri.starts_with("test://")) {
    return uri.substr(7);
  }

  lsp_runtime_error(std::format("Unsupported URI scheme: {}", uri));
}

void Server::startWorkersIfNeeded() {
#ifndef CXX_NO_THREADS
  const auto threadCountOption = cli.getSingle("-j");

  if (!threadCountOption.has_value()) {
    return;
  }

  auto workerCount = std::stoi(threadCountOption.value());

  if (workerCount <= 0) {
    workerCount = int(std::thread::hardware_concurrency());
  }

  for (int i = 0; i < workerCount; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        auto task = syncQueue_.pop();
        if (syncQueue_.closed()) break;
        task();
      }
    });
  }
#endif
}

void Server::stopWorkersIfNeeded() {
#ifndef CXX_NO_THREADS
  if (workers_.empty()) {
    return;
  }

  syncQueue_.close();

  for (int i = 0; i < workers_.size(); ++i) {
    syncQueue_.push([] {});
  }

  std::ranges::for_each(workers_, &std::thread::join);
#endif
}

void Server::run(std::function<void()> task) {
#ifndef CXX_NO_THREADS
  if (!workers_.empty()) {
    syncQueue_.push(std::move(task));
    return;
  }
#endif

  task();
}

void Server::cancelPendingParserRequests(const std::string& fileName) {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif

  std::vector<std::shared_ptr<CxxDocument>> pendingParserRequests;
  std::swap(pendingParserRequests_, pendingParserRequests);

  for (const auto& doc : pendingParserRequests) {
    if (doc->fileName() == fileName) {
      doc->cancel();
    } else {
      pendingParserRequests_.push_back(doc);
    }
  }
}

void Server::parse(const std::string& uri) {
  const auto& doc = documentContents_.at(uri);

  auto text = doc.value;
  auto version = doc.version;

  run([text = std::move(text), uri = std::move(uri), version, this] {
    auto fileName = pathFromUri(uri);

    cancelPendingParserRequests(fileName);

    auto doc = std::make_shared<CxxDocument>(cli, std::move(fileName), version);
    pendingParserRequests_.push_back(doc);

    doc->parse(std::move(text));

    {
#ifndef CXX_NO_THREADS
      auto locker = std::unique_lock(outputMutex_);
#endif

      if (auto it = std::ranges::find(pendingParserRequests_, doc);
          it != pendingParserRequests_.end()) {
        pendingParserRequests_.erase(it);
      }

      if (documents_.contains(uri) && documents_.at(uri)->version() > version) {
        return;
      }

      documents_[uri] = doc;
    }

    withUnsafeJson([&](json storage) {
      PublishDiagnosticsNotification publishDiagnostics(storage);
      publishDiagnostics.method("textDocument/publishDiagnostics");
      publishDiagnostics.params().uri(uri);
      publishDiagnostics.params().diagnostics(doc->diagnostics());
      publishDiagnostics.params().version(version);

      sendToClient(publishDiagnostics);
    });
  });
}

void Server::operator()(InitializeRequest request) {
  logTrace(std::format("Did receive InitializeRequest"));

  withUnsafeJson([&](json storage) {
    InitializeResponse response{storage};

    response.id(*request.id());

    auto serverInfo = response.result().serverInfo<ServerInfo>();

    serverInfo.name("cxx-lsp").version(CXX_VERSION);

    auto capabilities = response.result().capabilities();
    capabilities.textDocumentSync(TextDocumentSyncKind::kIncremental);
    capabilities.documentSymbolProvider(true);

    auto completionOptions =
        capabilities.completionProvider<CompletionOptions>();

    completionOptions.triggerCharacters({":", ".", ">"});

    sendToClient(response);
  });
}

void Server::operator()(InitializedNotification notification) {
  logTrace(std::format("Did receive InitializedNotification"));
}

void Server::operator()(ShutdownRequest request) {
  logTrace(std::format("Did receive ShutdownRequest"));

  withUnsafeJson([&](json storage) {
    LSPResponse response(storage);
    response.id(request.id());
    response.get().emplace("result", nullptr);
    sendToClient(response);
  });
}

void Server::operator()(ExitNotification notification) {
  logTrace(std::format("Did receive ExitNotification"));
  done_ = true;
}

void Server::operator()(DidOpenTextDocumentNotification notification) {
  logTrace(std::format("Did receive DidOpenTextDocumentNotification"));

  auto textDocument = notification.params().textDocument();

  auto text = textDocument.text();

  auto& content = documentContents_[textDocument.uri()];
  content.value = std::move(text);
  content.version = textDocument.version();
  content.computeLineStartOffsets();

  parse(textDocument.uri());
}

void Server::operator()(DidCloseTextDocumentNotification notification) {
  logTrace(std::format("Did receive DidCloseTextDocumentNotification"));

  const auto uri = notification.params().textDocument().uri();
  documents_.erase(uri);
}

void Server::operator()(DidChangeTextDocumentNotification notification) {
  logTrace(std::format("Did receive DidChangeTextDocumentNotification"));

  const auto textDocument = notification.params().textDocument();
  const auto uri = textDocument.uri();
  const auto version = textDocument.version();

  auto& text = documentContents_[uri];
  text.version = version;

  struct {
    Text& text;

    void operator()(const TextDocumentContentChangeWholeDocument& change) {
      text.value = change.text();
    }

    void operator()(const TextDocumentContentChangePartial& change) {
      auto range = change.range();
      auto start = range.start();
      auto end = range.end();
      auto startOffset = text.offsetAt(start.line(), start.character());
      auto endOffset = text.offsetAt(end.line(), end.character());
      text.value.replace(startOffset, endOffset - startOffset, change.text());
      text.computeLineStartOffsets();
    }
  } visit{text};

  auto contentChanges = notification.params().contentChanges();
  const auto contentChangeCount = int(contentChanges.size());

  for (int i = 0; i < contentChangeCount; ++i) {
    std::visit(visit, contentChanges.at(i));
  }

  parse(textDocument.uri());
}

auto Server::latestDocument(const std::string& uri)
    -> std::shared_ptr<CxxDocument> {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif

  if (!documents_.contains(uri)) {
    return {};
  }

  return documents_[uri];
}

void Server::operator()(DocumentSymbolRequest request) {
  logTrace(std::format("Did receive DocumentSymbolRequest"));

  auto uri = request.params().textDocument().uri();
  auto doc = latestDocument(uri);
  auto id = request.id();

  run([=, this] {
    withUnsafeJson([&](json storage) {
      DocumentSymbolResponse response(storage);
      response.id(id);
      (void)response.result();
      sendToClient(response);
    });
  });
}

void Server::operator()(CompletionRequest request) {
  logTrace(std::format("Did receive CompletionRequest"));

  auto textDocument = request.params().textDocument();
  auto uri = textDocument.uri();
  auto id = request.id();
  auto line = request.params().position().line();
  auto column = request.params().position().character();

  const auto& text = documentContents_.at(uri);
  auto source = text.value;

  run([=, this, fileName = pathFromUri(uri)] {
    withUnsafeJson([&](json storage) {
      CompletionResponse response(storage);
      response.id(request.id());

      // the version is not relevant for code completion requests as we don't
      // need to store the document in the cache.
      auto cxxDocument = std::make_shared<CxxDocument>(cli, std::move(fileName),
                                                       /*version=*/0);

      auto completionItems = response.result<Vector<CompletionItem>>();

      // cxx expects 1-based line and column numbers
      cxxDocument->codeCompletionAt(std::move(source), std::uint32_t(line + 1),
                                    std::uint32_t(column + 1), completionItems);

      sendToClient(response);
    });
  });
}

void Server::operator()(CancelNotification notification) {
  const auto id = notification.params().id();

  if (std::holds_alternative<std::string>(id)) {
    logTrace(
        std::format("Did receive CancelNotification for request with id {}",
                    std::get<std::string>(id)));
  } else {
    logTrace(
        std::format("Did receive CancelNotification for request with id {}",
                    std::get<long>(id)));
  }
}

void Server::operator()(SetTraceNotification notification) {
  logTrace(std::format("Did receive SetTraceNotification"));

  trace_ = notification.params().value();

  if (trace_ != TraceValue::kOff) {
    logTrace(std::format("Trace level set to {}", to_string(trace_)));
    return;
  }
}

void Server::operator()(LSPRequest request) {
  if (!request.id().has_value()) {
    // nothing to do for notifications
    logTrace(std::format("Did receive notification {}", request.method()));
    return;
  }

  logTrace(std::format("Did receive request {}", request.method()));

  // send an empty response.
  withUnsafeJson([&](json storage) {
    LSPResponse response(storage);
    response.id(request.id());
    request.get().emplace("result", nullptr);
    sendToClient(response);
  });
}

}  // namespace cxx::lsp
