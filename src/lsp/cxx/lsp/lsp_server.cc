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

#include "lsp_server.h"

#include <cxx/ast.h>
#include <cxx/lexer.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/requests.h>
#include <cxx/lsp/types.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <utf8/unchecked.h>

#include <chrono>
#include <format>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cxx_document.h"
#include "server_host.h"

namespace cxx::lsp {

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

[[nodiscard]] auto isSupportedTextDocument(const json& textDocument) -> bool {
  static const std::unordered_set<std::string_view> supportedLanguages{
      "c", "cpp", "objective-c", "objective-cpp"};

  if (!textDocument.contains("languageId")) return true;

  const auto& languageId = textDocument.at("languageId");
  if (!languageId.is_string()) return true;

  return supportedLanguages.contains(languageId.get<std::string>());
}

auto elapsedMilliseconds(std::chrono::steady_clock::time_point start)
    -> double {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

auto Server::Text::offsetAt(std::size_t line, std::size_t column) const
    -> std::size_t {
  if (line >= lineStartOffsets.size()) {
    return std::string::npos;
  }

  const auto lineStart = lineStartOffsets.at(line);

  auto lineEnd = value.size();
  if (line + 1 < lineStartOffsets.size()) {
    lineEnd = lineStartOffsets.at(line + 1);
  }

  auto it = value.begin() + std::ptrdiff_t(lineStart);
  const auto end = value.begin() + std::ptrdiff_t(lineEnd);

  std::size_t utf16Offset = 0;

  while (it != end && utf16Offset < column) {
    const auto codepoint = utf8::unchecked::next(it);

    if (codepoint > 0xFFFF) {
      utf16Offset += 2;
    } else {
      utf16Offset += 1;
    }
  }

  return std::size_t(it - value.begin());
}

auto Server::Text::completionPrefixStartAt(std::size_t line,
                                           std::size_t column) const
    -> std::size_t {
  const auto cursorOffset = offsetAt(line, column);
  if (cursorOffset == std::string::npos) return column;

  const auto lineStart = lineStartOffsets.at(line);
  const auto prefixSize = cursorOffset - lineStart;
  const auto linePrefix = std::string_view(value).substr(lineStart, prefixSize);

  Lexer lexer{linePrefix};
  TokenKind lastTokenKind = TokenKind::T_EOF_SYMBOL;
  std::size_t lastTokenStart = 0;
  std::size_t lastTokenEnd = 0;

  for (;;) {
    const auto tokenKind = lexer.next();
    if (tokenKind == TokenKind::T_EOF_SYMBOL) break;
    lastTokenKind = tokenKind;
    lastTokenStart = std::size_t(lexer.tokenPos());
    lastTokenEnd = lastTokenStart + lexer.tokenLength();
  }

  if (lastTokenKind != TokenKind::T_IDENTIFIER) return column;
  if (lastTokenEnd != linePrefix.size()) return column;

  auto it = linePrefix.begin() + std::ptrdiff_t(lastTokenStart);
  const auto end = linePrefix.end();
  std::size_t prefixLength = 0;

  while (it != end) {
    const auto codepoint = utf8::unchecked::next(it);
    if (codepoint > 0xFFFF) {
      prefixLength += 2;
    } else {
      prefixLength += 1;
    }
  }

  return column - prefixLength;
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

Server::Server(ServerHost& host, std::unique_ptr<Transport> transport)
    : host_(&host), transport_(std::move(transport)) {}

Server::~Server() {}

auto Server::start() -> int {
  startProcessing();

  while (!done_ && transport_->isOpen()) {
    continueProcessing();
  }

  stopProcessing();

  return 0;
}

void Server::startProcessing() {
  logTrace(std::format("Starting LSP server"));

  host_->start();
}

void Server::continueProcessing() {
  if (auto message = transport_->nextMessage()) {
    auto request = LSPRequest(message.value());
    visit(*this, request);
  }
}

void Server::stopProcessing() { host_->stop(); }

void Server::sendToClient(LSPResponse response) {
  json& message = response.get();
  message["jsonrpc"] = "2.0";
  transport_->sendMessage(message);
}

void Server::sendToClient(LSPRequest notification) {
  json response = notification;
  response["jsonrpc"] = "2.0";
  transport_->sendMessage(response);
}

void Server::logTrace(std::string message, std::optional<std::string> verbose) {
  host_->trace(message, verbose);

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

void Server::sendNullResult(std::optional<std::variant<long, std::string>> id) {
  withUnsafeJson([&](json storage) {
    LSPResponse response(storage);
    response.id(std::move(id));
    response.get().emplace("result", nullptr);
    sendToClient(response);
  });
}

void Server::sendEmittedCode(EmitCodeResponse response, CxxDocument& document,
                             EmitCodeFormat format, bool debugInfo) {
  const auto startedAt = std::chrono::steady_clock::now();

  logTrace(std::format(
      "emit event=started file={} version={} format={} interruptible=false",
      document.fileName(), document.version(), to_string(format)));

  auto text = host_->emitCode(document, format, debugInfo);

  logTrace(std::format(
      "emit event=finished file={} version={} format={} duration_ms={:.1f}",
      document.fileName(), document.version(), to_string(format),
      elapsedMilliseconds(startedAt)));

  if (!text.has_value()) {
    response.get().emplace("result", nullptr);
  } else {
    auto result = response.result<EmitCodeResult>();
    result.format(format);
    result.text(std::move(*text));
  }

  sendToClient(response);
}

auto Server::cancelPendingParserRequests(const std::string& fileName,
                                         std::optional<long> minVersion)
    -> std::size_t {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif

  std::size_t cancelledCount = 0;

  for (auto& request : pendingParserRequests_) {
    auto& document = request->document;

    if (document->fileName() != fileName) continue;
    if (minVersion.has_value()) {
      if (document->version() >= *minVersion) continue;
    }
    if (document->isCancelled()) continue;

    document->cancel();
    ++cancelledCount;
  }

  return cancelledCount;
}

auto Server::cancelPendingParserRequest(const ParserRequestId& id) -> bool {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif

  for (auto& request : pendingParserRequests_) {
    if (!request->id.has_value()) continue;
    if (*request->id != id) continue;

    request->document->cancel();
    return true;
  }

  return false;
}

auto Server::registerPendingParserRequest(std::shared_ptr<CxxDocument> document,
                                          std::string uri, std::string kind,
                                          std::size_t sourceBytes,
                                          std::optional<ParserRequestId> id)
    -> std::shared_ptr<PendingParserRequest> {
  auto request = std::make_shared<PendingParserRequest>(PendingParserRequest{
      .document = std::move(document),
      .uri = std::move(uri),
      .kind = std::move(kind),
      .id = std::move(id),
      .queuedAt = std::chrono::steady_clock::now(),
  });

  std::size_t pendingCount;

  {
#ifndef CXX_NO_THREADS
    auto lock = std::unique_lock(documentsMutex_);
#endif

    pendingParserRequests_.push_back(request);
    pendingCount = pendingParserRequests_.size();
  }

  logTrace(std::format(
      "parse event=queued kind={} file={} version={} bytes={} pending={}",
      request->kind, request->document->fileName(),
      request->document->version(), sourceBytes, pendingCount));

  return request;
}

auto Server::unregisterPendingParserRequest(
    const std::shared_ptr<PendingParserRequest>& request) -> std::size_t {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif

  auto matchesRequest =
      [&request](const std::shared_ptr<PendingParserRequest>& entry) {
        return entry == request;
      };

  if (auto it = std::ranges::find_if(pendingParserRequests_, matchesRequest);
      it != pendingParserRequests_.end()) {
    pendingParserRequests_.erase(it);
  }

  return pendingParserRequests_.size();
}

auto Server::parserRequestIsInvalidated(
    const std::shared_ptr<PendingParserRequest>& request) -> bool {
  if (request->document->isCancelled()) return true;

  auto currentVersion = documentVersion(request->uri);
  if (!currentVersion.has_value()) return true;
  return *currentVersion != request->document->version();
}

void Server::skipParserRequest(
    const std::shared_ptr<PendingParserRequest>& request) {
  request->document->cancel();
  const auto pendingCount = unregisterPendingParserRequest(request);

  logTrace(std::format(
      "parse event=skipped kind={} file={} version={} queue_ms={:.1f} "
      "reason=invalidated pending={}",
      request->kind, request->document->fileName(),
      request->document->version(), elapsedMilliseconds(request->queuedAt),
      pendingCount));

  if (request->id.has_value()) sendNullResult(request->id);
}

auto Server::startParserRequest(
    const std::shared_ptr<PendingParserRequest>& request)
    -> std::chrono::steady_clock::time_point {
  const auto startedAt = std::chrono::steady_clock::now();

  logTrace(std::format(
      "parse event=started kind={} file={} version={} queue_ms={:.1f}",
      request->kind, request->document->fileName(),
      request->document->version(), elapsedMilliseconds(request->queuedAt)));

  return startedAt;
}

auto Server::finishParserRequest(
    const std::shared_ptr<PendingParserRequest>& request,
    std::chrono::steady_clock::time_point startedAt) -> bool {
  const auto pendingCount = unregisterPendingParserRequest(request);

  logTrace(std::format(
      "parse event=finished kind={} file={} version={} duration_ms={:.1f} "
      "cancelled={} pending={}",
      request->kind, request->document->fileName(),
      request->document->version(), elapsedMilliseconds(startedAt),
      request->document->isCancelled(), pendingCount));

  if (!parserRequestIsInvalidated(request)) return true;
  if (request->id.has_value()) sendNullResult(request->id);
  return false;
}

void Server::reuseParserRequest(
    const std::shared_ptr<PendingParserRequest>& request) {
  const auto pendingCount = unregisterPendingParserRequest(request);

  logTrace(std::format(
      "parse event=reused kind={} file={} version={} queue_ms={:.1f} "
      "pending={}",
      request->kind, request->document->fileName(),
      request->document->version(), elapsedMilliseconds(request->queuedAt),
      pendingCount));
}

void Server::scheduleParse(const std::string& uri) {
  constexpr auto kDiagnosticsDebounce = std::chrono::milliseconds(250);

  std::int64_t generation;

  {
#ifndef CXX_NO_THREADS
    auto lock = std::unique_lock(documentsMutex_);
#endif
    generation = ++pendingParseGeneration_[uri];
  }

  host_->runLater(kDiagnosticsDebounce, [this, uri, generation] {
    {
#ifndef CXX_NO_THREADS
      auto lock = std::unique_lock(documentsMutex_);
#endif
      if (pendingParseGeneration_[uri] != generation) return;
    }

    parse(uri);
  });
}

auto Server::snapshotDocument(const std::string& uri) -> std::optional<Text> {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentContentsMutex_);
#endif

  auto content = documentContents_.find(uri);
  if (content == documentContents_.end()) return std::nullopt;

  return content->second;
}

auto Server::documentVersion(const std::string& uri)
    -> std::optional<std::int64_t> {
#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentContentsMutex_);
#endif

  auto content = documentContents_.find(uri);
  if (content == documentContents_.end()) return std::nullopt;
  return content->second.version;
}

void Server::parse(const std::string& uri) {
  auto snapshot = snapshotDocument(uri);
  if (!snapshot.has_value()) return;

  auto version = snapshot->version;
  auto fileName = host_->pathFromUri(uri);

  if (!fileName.has_value()) {
    logTrace(std::format("Unsupported URI scheme: {}", uri));
    return;
  }

  const auto cancelledCount = cancelPendingParserRequests(*fileName, version);
  if (cancelledCount != 0) {
    logTrace(
        std::format("parse event=superseded file={} version={} cancelled={}",
                    *fileName, version, cancelledCount));
  }

  auto document = std::make_shared<CxxDocument>(std::move(*fileName), version);
  auto parserRequest = registerPendingParserRequest(
      document, uri, "diagnostics", snapshot->value.size());

  host_->run([text = std::move(snapshot->value), parserRequest, uri, version,
              this]() mutable {
    if (parserRequestIsInvalidated(parserRequest)) {
      skipParserRequest(parserRequest);
      return;
    }

    const auto startedAt = startParserRequest(parserRequest);

    host_->process(
        *parserRequest->document, std::move(text),
        [this, parserRequest, uri, version, startedAt,
         storage = std::make_shared<json>()] {
          if (!finishParserRequest(parserRequest, startedAt)) return;

          {
#ifndef CXX_NO_THREADS
            auto locker = std::unique_lock(documentsMutex_);
#endif

            if (documents_.contains(uri)) {
              if (documents_.at(uri)->version() > version) return;
            }

            documents_[uri] = parserRequest->document;
          }

          PublishDiagnosticsNotification publishDiagnostics(*storage);
          publishDiagnostics.method("textDocument/publishDiagnostics");
          publishDiagnostics.params().uri(uri);
          publishDiagnostics.params().diagnostics(
              parserRequest->document->diagnostics());
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

    if (host_->supportsEmitCode()) {
      auto experimental = json::object();
      experimental.emplace("cxxEmitCode", true);
      capabilities.get().emplace("experimental", std::move(experimental));
    }

    auto completionOptions =
        capabilities.completionProvider<CompletionOptions>();

    completionOptions.triggerCharacters({":", ".", ">"});

    auto signatureHelpOptions =
        capabilities.signatureHelpProvider<SignatureHelpOptions>();

    signatureHelpOptions.triggerCharacters({"(", ",", "<", "{"});

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

  if (!isSupportedTextDocument(textDocument.get())) {
    logTrace(std::format("Ignoring the document {}", textDocument.uri()));
    return;
  }

  auto text = textDocument.text();

  {
#ifndef CXX_NO_THREADS
    auto lock = std::unique_lock(documentContentsMutex_);
#endif

    auto& content = documentContents_[textDocument.uri()];
    content.value = std::move(text);
    content.version = textDocument.version();
    content.computeLineStartOffsets();
  }

  parse(textDocument.uri());
}

void Server::operator()(DidCloseTextDocumentNotification notification) {
  logTrace(std::format("Did receive DidCloseTextDocumentNotification"));

  const auto uri = notification.params().textDocument().uri();
  auto fileName = host_->pathFromUri(uri);

  {
#ifndef CXX_NO_THREADS
    auto lock = std::unique_lock(documentContentsMutex_);
#endif
    documentContents_.erase(uri);
  }

  if (fileName.has_value()) {
    const auto cancelledCount = cancelPendingParserRequests(*fileName);
    logTrace(std::format("parse event=closed file={} cancelled={}", *fileName,
                         cancelledCount));
  }

#ifndef CXX_NO_THREADS
  auto lock = std::unique_lock(documentsMutex_);
#endif
  documents_.erase(uri);
  pendingParseGeneration_.erase(uri);
}

void Server::operator()(DidChangeTextDocumentNotification notification) {
  logTrace(std::format("Did receive DidChangeTextDocumentNotification"));

  const auto textDocument = notification.params().textDocument();
  const auto uri = textDocument.uri();
  const auto version = textDocument.version();

  {
#ifndef CXX_NO_THREADS
    auto lock = std::unique_lock(documentContentsMutex_);
#endif

    auto content = documentContents_.find(uri);
    if (content == documentContents_.end()) return;

    auto& text = content->second;
    text.version = version;

    struct {
      Text& text;

      void operator()(const TextDocumentContentChangeWholeDocument& change) {
        text.value = change.text();
        text.computeLineStartOffsets();
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
  }

  auto fileName = host_->pathFromUri(uri);
  if (fileName.has_value()) {
    const auto cancelledCount = cancelPendingParserRequests(*fileName, version);
    logTrace(
        std::format("parse event=invalidated file={} version={} cancelled={}",
                    *fileName, version, cancelledCount));
  }

  scheduleParse(uri);
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

  host_->run([=, this] {
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

  auto fileName = host_->pathFromUri(uri);

  if (!fileName.has_value()) {
    logTrace(std::format("Unsupported URI scheme: {}", uri));
    sendNullResult(id);
    return;
  }

  auto snapshot = snapshotDocument(uri);

  if (!snapshot.has_value()) {
    logTrace(std::format("No content for the document {}", uri));
    sendNullResult(id);
    return;
  }

  auto version = snapshot->version;
  const auto completionStartColumn =
      snapshot->completionPrefixStartAt(line, column);
  auto storage = std::make_shared<json>();

  CompletionResponse response(*storage);
  response.id(id);

  auto document = std::make_shared<CxxDocument>(std::move(*fileName), version);
  auto completionItems = response.result<Vector<CompletionItem>>();
  document->requestCodeCompletionAt(
      std::uint32_t(line + 1), std::uint32_t(column + 1),
      CompletionEditRange{
          .line = std::uint32_t(line),
          .startColumn = std::uint32_t(completionStartColumn),
          .endColumn = std::uint32_t(column),
      },
      completionItems);

  auto parserRequest = registerPendingParserRequest(document, uri, "completion",
                                                    snapshot->value.size(), id);

  host_->run([this, storage, parserRequest, response,
              source = std::move(snapshot->value)]() mutable {
    if (parserRequestIsInvalidated(parserRequest)) {
      skipParserRequest(parserRequest);
      return;
    }

    const auto startedAt = startParserRequest(parserRequest);

    host_->process(*parserRequest->document, std::move(source),
                   [this, storage, parserRequest, response, startedAt] {
                     if (!finishParserRequest(parserRequest, startedAt)) return;
                     sendToClient(response);
                   });
  });
}

void Server::operator()(SignatureHelpRequest request) {
  logTrace(std::format("Did receive SignatureHelpRequest"));

  auto textDocument = request.params().textDocument();
  auto uri = textDocument.uri();
  auto id = request.id();
  auto line = request.params().position().line();
  auto column = request.params().position().character();

  auto fileName = host_->pathFromUri(uri);

  if (!fileName.has_value()) {
    logTrace(std::format("Unsupported URI scheme: {}", uri));
    sendNullResult(id);
    return;
  }

  auto snapshot = snapshotDocument(uri);

  if (!snapshot.has_value()) {
    logTrace(std::format("No content for the document {}", uri));
    sendNullResult(id);
    return;
  }

  auto version = snapshot->version;
  auto storage = std::make_shared<json>();

  SignatureHelpResponse response(*storage);
  response.id(id);

  auto document = std::make_shared<CxxDocument>(std::move(*fileName), version);
  auto signatureHelp = response.result<SignatureHelp>();
  document->requestSignatureHelpAt(std::uint32_t(line + 1),
                                   std::uint32_t(column + 1), signatureHelp);

  auto parserRequest = registerPendingParserRequest(document, uri, "signature",
                                                    snapshot->value.size(), id);

  host_->run([this, storage, parserRequest, response,
              source = std::move(snapshot->value)]() mutable {
    if (parserRequestIsInvalidated(parserRequest)) {
      skipParserRequest(parserRequest);
      return;
    }

    const auto startedAt = startParserRequest(parserRequest);

    host_->process(*parserRequest->document, std::move(source),
                   [this, storage, parserRequest, response, startedAt] {
                     if (!finishParserRequest(parserRequest, startedAt)) return;
                     sendToClient(response);
                   });
  });
}

void Server::operator()(EmitCodeRequest request) {
  logTrace(std::format("Did receive EmitCodeRequest"));

  auto params = request.params();
  auto uri = params.textDocument().uri();
  auto id = request.id();
  auto format = params.format();
  auto debugInfo = params.debugInfo().value_or(false);

  auto fileName = host_->pathFromUri(uri);

  if (!fileName.has_value()) {
    logTrace(std::format("Unsupported URI scheme: {}", uri));
    sendNullResult(id);
    return;
  }

  if (!host_->supportsEmitCode()) {
    sendNullResult(id);
    return;
  }

  auto snapshot = snapshotDocument(uri);

  if (!snapshot.has_value()) {
    logTrace(std::format("No content for the document {}", uri));
    sendNullResult(id);
    return;
  }

  auto version = snapshot->version;
  auto storage = std::make_shared<json>();

  EmitCodeResponse response(*storage);
  response.id(id);

  auto document = std::make_shared<CxxDocument>(std::move(*fileName), version);
  auto parserRequest = registerPendingParserRequest(document, uri, "emit",
                                                    snapshot->value.size(), id);

  host_->run([this, storage, parserRequest, response, uri, version, format,
              debugInfo, source = std::move(snapshot->value)]() mutable {
    if (parserRequestIsInvalidated(parserRequest)) {
      skipParserRequest(parserRequest);
      return;
    }

    auto parsedDocument = latestDocument(uri);
    auto canReuse = bool(parsedDocument);
    if (canReuse) canReuse = parsedDocument->version() == version;

    if (canReuse) {
      reuseParserRequest(parserRequest);
      sendEmittedCode(response, *parsedDocument, format, debugInfo);
      return;
    }

    const auto startedAt = startParserRequest(parserRequest);

    host_->process(
        *parserRequest->document, std::move(source),
        [this, storage, parserRequest, response, format, debugInfo, startedAt] {
          if (!finishParserRequest(parserRequest, startedAt)) return;
          sendEmittedCode(response, *parserRequest->document, format,
                          debugInfo);
        });
  });
}

void Server::operator()(CancelNotification notification) {
  const auto id = notification.params().id();
  const auto cancelled = cancelPendingParserRequest(id);

  if (std::holds_alternative<std::string>(id)) {
    logTrace(std::format("parse event=request-cancelled id={} matched={}",
                         std::get<std::string>(id), cancelled));
  } else {
    logTrace(std::format("parse event=request-cancelled id={} matched={}",
                         std::get<long>(id), cancelled));
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
    logTrace(std::format("Did receive notification {}", request.method()));
    return;
  }

  logTrace(std::format("Did receive request {}", request.method()));

  sendNullResult(request.id());
}

}  // namespace cxx::lsp
