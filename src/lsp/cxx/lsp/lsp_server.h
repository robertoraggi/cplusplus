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

#pragma once

#include <cxx/lsp/fwd.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifndef CXX_NO_THREADS
#include <mutex>
#endif

#include "transport.h"

namespace cxx::lsp {

class CxxDocument;
class ServerHost;

class Server {
 public:
  Server(ServerHost& host, std::unique_ptr<Transport> transport);
  ~Server();

  auto start() -> int;

  void startProcessing();
  void continueProcessing();
  void stopProcessing();

  void operator()(InitializeRequest request);
  void operator()(InitializedNotification notification);

  void operator()(ShutdownRequest request);
  void operator()(ExitNotification notification);

  void operator()(DidOpenTextDocumentNotification notification);
  void operator()(DidCloseTextDocumentNotification notification);
  void operator()(DidChangeTextDocumentNotification notification);

  void operator()(DocumentSymbolRequest request);
  void operator()(CompletionRequest request);
  void operator()(SignatureHelpRequest request);
  void operator()(EmitCodeRequest request);

  void operator()(SetTraceNotification notification);

  void operator()(CancelNotification notification);
  void operator()(LSPRequest request);

 private:
  using ParserRequestId = std::variant<long, std::string>;
  struct PendingParserRequest;

  [[nodiscard]] auto cancelPendingParserRequests(
      const std::string& fileName, std::optional<long> minVersion = {})
      -> std::size_t;
  [[nodiscard]] auto cancelPendingParserRequest(const ParserRequestId& id)
      -> bool;
  [[nodiscard]] auto registerPendingParserRequest(
      std::shared_ptr<CxxDocument> document, std::string uri, std::string kind,
      std::size_t sourceBytes, std::optional<ParserRequestId> id = {})
      -> std::shared_ptr<PendingParserRequest>;
  [[nodiscard]] auto unregisterPendingParserRequest(
      const std::shared_ptr<PendingParserRequest>& request) -> std::size_t;
  [[nodiscard]] auto parserRequestIsInvalidated(
      const std::shared_ptr<PendingParserRequest>& request) -> bool;
  void skipParserRequest(const std::shared_ptr<PendingParserRequest>& request);
  [[nodiscard]] auto startParserRequest(
      const std::shared_ptr<PendingParserRequest>& request)
      -> std::chrono::steady_clock::time_point;
  [[nodiscard]] auto finishParserRequest(
      const std::shared_ptr<PendingParserRequest>& request,
      std::chrono::steady_clock::time_point startedAt) -> bool;
  void reuseParserRequest(const std::shared_ptr<PendingParserRequest>& request);

  void parse(const std::string& uri);
  void scheduleParse(const std::string& uri);

  [[nodiscard]] auto latestDocument(const std::string& uri)
      -> std::shared_ptr<CxxDocument>;

  void sendToClient(LSPRequest notification);
  void sendToClient(LSPResponse response);

  void sendNullResult(std::optional<std::variant<long, std::string>> id);

  void sendEmittedCode(EmitCodeResponse response, CxxDocument& document,
                       EmitCodeFormat format, bool debugInfo);

  void logTrace(std::string message, std::optional<std::string> verbose = {});

  struct Text {
    std::string value;
    std::vector<std::size_t> lineStartOffsets;
    std::int64_t version = 0;

    auto offsetAt(std::size_t line, std::size_t column) const -> std::size_t;
    auto completionPrefixStartAt(std::size_t line, std::size_t column) const
        -> std::size_t;

    void computeLineStartOffsets();
  };

  [[nodiscard]] auto snapshotDocument(const std::string& uri)
      -> std::optional<Text>;
  [[nodiscard]] auto documentVersion(const std::string& uri)
      -> std::optional<std::int64_t>;

  struct PendingParserRequest {
    std::shared_ptr<CxxDocument> document;
    std::string uri;
    std::string kind;
    std::optional<ParserRequestId> id;
    std::chrono::steady_clock::time_point queuedAt;
  };

 private:
  ServerHost* host_;
  std::unique_ptr<Transport> transport_;
  std::unordered_map<std::string, std::shared_ptr<CxxDocument>> documents_;
  std::unordered_map<std::string, Text> documentContents_;
  std::vector<std::shared_ptr<PendingParserRequest>> pendingParserRequests_;
  std::unordered_map<std::string, std::int64_t> pendingParseGeneration_;
#ifndef CXX_NO_THREADS
  std::mutex documentsMutex_;
  std::mutex documentContentsMutex_;
#endif
  TraceValue trace_{};
  bool done_ = false;
};

}  // namespace cxx::lsp
