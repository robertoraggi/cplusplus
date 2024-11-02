// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/lsp/enums.h>
#include <cxx/lsp/requests.h>
#include <cxx/lsp/types.h>
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

void Server::sendToClient(const json& message) {
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

void Server::sendToClient(const LSPObject& result,
                          std::optional<std::variant<long, std::string>> id) {
  auto response = json::object();
  response["jsonrpc"] = "2.0";

  if (id.has_value()) {
    if (std::holds_alternative<long>(id.value())) {
      response["id"] = std::get<long>(id.value());
    } else {
      response["id"] = std::get<std::string>(id.value());
    }
  }

  response["result"] = result;

  sendToClient(response);
}

void Server::sendNotification(const LSPRequest& notification) {
  json response = notification;
  response["jsonrpc"] = "2.0";

  sendToClient(response);
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
    sendNotification(logTrace);
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

void Server::parse(const std::string& uri) {
  const auto& doc = documentContents_.at(uri);

  auto text = doc.value;
  auto version = doc.version;

  run([text = std::move(text), uri = std::move(uri), version, this] {
    auto doc = std::make_shared<CxxDocument>(cli, version);
    doc->parse(std::move(text), pathFromUri(uri));

    {
#ifndef CXX_NO_THREADS
      auto locker = std::unique_lock(outputMutex_);
#endif

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

      sendNotification(publishDiagnostics);
    });
  });
}

void Server::operator()(const InitializeRequest& request) {
  logTrace(std::format("Did receive InitializeRequest"));

  withUnsafeJson([&](json storage) {
    InitializeResult result(storage);
    result.serverInfo<ServerInfo>().name("cxx-lsp").version(CXX_VERSION);
    auto capabilities = result.capabilities();
    capabilities.textDocumentSync(TextDocumentSyncKind::kIncremental);
    sendToClient(result, request.id());
  });
}

void Server::operator()(const InitializedNotification& notification) {
  logTrace(std::format("Did receive InitializedNotification"));
}

void Server::operator()(const ShutdownRequest& request) {
  logTrace(std::format("Did receive ShutdownRequest"));

  withUnsafeJson([&](json storage) {
    LSPObject result(storage);
    sendToClient(result, request.id());
  });
}

void Server::operator()(const ExitNotification& notification) {
  logTrace(std::format("Did receive ExitNotification"));
  done_ = true;
}

void Server::operator()(const DidOpenTextDocumentNotification& notification) {
  logTrace(std::format("Did receive DidOpenTextDocumentNotification"));

  auto textDocument = notification.params().textDocument();

  auto text = textDocument.text();

  auto& content = documentContents_[textDocument.uri()];
  content.value = std::move(text);
  content.version = textDocument.version();
  content.computeLineStartOffsets();

  parse(textDocument.uri());
}

void Server::operator()(const DidCloseTextDocumentNotification& notification) {
  logTrace(std::format("Did receive DidCloseTextDocumentNotification"));

  const auto uri = notification.params().textDocument().uri();
  documents_.erase(uri);
}

void Server::operator()(const DidChangeTextDocumentNotification& notification) {
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
  const auto contentChangeCount = contentChanges.size();

  for (std::size_t i = 0; i < contentChangeCount; ++i) {
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

void Server::operator()(const DocumentDiagnosticRequest& request) {
  logTrace(std::format("Did receive DocumentDiagnosticRequest"));
}

void Server::operator()(const CancelNotification& notification) {
  auto id = notification.params().id<long>();
  logTrace(
      std::format("Did receive CancelNotification for request with id {}", id));
}

void Server::operator()(const SetTraceNotification& notification) {
  logTrace(std::format("Did receive SetTraceNotification"));

  trace_ = notification.params().value();
}

void Server::operator()(const LSPRequest& request) {
  logTrace(std::format("Did receive LSPRequest {}", request.method()));

  if (!request.id().has_value()) {
    // nothing to do for notifications
    return;
  }

  // send an empty response.
  withUnsafeJson([&](json storage) {
    LSPObject result(storage);
    sendToClient(result, request.id());
  });
}

}  // namespace cxx::lsp
