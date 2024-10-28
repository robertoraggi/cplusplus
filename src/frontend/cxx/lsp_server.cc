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

#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "cxx_document.h"

namespace cxx::lsp {

Server::Server(const CLI& cli)
    : cli(cli), input(std::cin), output(std::cout), log(std::cerr) {
  // create workers
  const auto workerCount = 4;

  auto worker = [] {

  };
}

Server::~Server() {}

auto Server::start() -> int {
  log << std::format("Starting LSP server\n");

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
  };

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
    output << message.dump(2) << "\n";
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

auto Server::pathFromUri(const std::string& uri) -> std::string {
  if (uri.starts_with("file://")) {
    return uri.substr(7);
  } else if (cli.opt_lsp_test && uri.starts_with("test://")) {
    return uri.substr(7);
  }

  lsp_runtime_error(std::format("Unsupported URI scheme: {}\n", uri));
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

void Server::parse(std::string uri, std::string text, long version) {
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
  log << std::format("Did receive InitializeRequest\n");

  withUnsafeJson([&](json storage) {
    InitializeResult result(storage);
    result.serverInfo<ServerInfo>().name("cxx-lsp").version(CXX_VERSION);
    result.capabilities().textDocumentSync(TextDocumentSyncKind::kFull);
    sendToClient(result, request.id());
  });
}

void Server::operator()(const InitializedNotification& notification) {
  log << std::format("Did receive InitializedNotification\n");
}

void Server::operator()(const ShutdownRequest& request) {
  log << std::format("Did receive ShutdownRequest\n");

  withUnsafeJson([&](json storage) {
    LSPObject result(storage);
    sendToClient(result, request.id());
  });
}

void Server::operator()(const ExitNotification& notification) {
  log << std::format("Did receive ExitNotification\n");
  done_ = true;
}

void Server::operator()(const DidOpenTextDocumentNotification& notification) {
  log << std::format("Did receive DidOpenTextDocumentNotification\n");

  auto textDocument = notification.params().textDocument();
  parse(textDocument.uri(), textDocument.text(), textDocument.version());
}

void Server::operator()(const DidCloseTextDocumentNotification& notification) {
  log << std::format("Did receive DidCloseTextDocumentNotification\n");

  const auto uri = notification.params().textDocument().uri();
  documents_.erase(uri);
}

void Server::operator()(const DidChangeTextDocumentNotification& notification) {
  log << std::format("Did receive DidChangeTextDocumentNotification\n");

  const auto textDocument = notification.params().textDocument();
  const auto uri = textDocument.uri();
  const auto version = textDocument.version();

  // update the document
  auto contentChanges = notification.params().contentChanges();
  const auto contentChangeCount = contentChanges.size();

  std::string text;

  for (std::size_t i = 0; i < contentChangeCount; ++i) {
    auto contentChange = contentChanges.at(i);

    if (auto change = std::get_if<TextDocumentContentChangeWholeDocument>(
            &contentChange)) {
      text = change->text();
    } else {
      lsp_runtime_error("Unsupported content change\n");
    }
  }

  parse(textDocument.uri(), std::move(text), textDocument.version());
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
  log << std::format("Did receive DocumentDiagnosticRequest\n");
}

void Server::operator()(const CancelNotification& notification) {
  auto id = notification.params().id<long>();
  log << std::format("Did receive CancelNotification for request with id {}\n",
                     id);
}

void Server::operator()(const LSPRequest& request) {
  log << std::format("Did receive LSPRequest {}\n", request.method());

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
