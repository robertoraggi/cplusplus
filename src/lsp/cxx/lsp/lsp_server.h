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

#include <istream>
#include <ostream>

#ifndef CXX_NO_THREADS
#include <thread>
#include <unordered_map>
#endif

#include <cxx/cli.h>

#include <vector>

#include "sync_queue.h"

namespace cxx::lsp {

class CxxDocument;

class Server {
 public:
  Server(const CLI& cli);
  ~Server();

  auto start() -> int;

  void operator()(InitializeRequest request);
  void operator()(InitializedNotification notification);

  void operator()(ShutdownRequest request);
  void operator()(ExitNotification notification);

  void operator()(DidOpenTextDocumentNotification notification);
  void operator()(DidCloseTextDocumentNotification notification);
  void operator()(DidChangeTextDocumentNotification notification);

  void operator()(DocumentSymbolRequest request);
  void operator()(CompletionRequest request);

  void operator()(SetTraceNotification notification);

  void operator()(CancelNotification notification);
  void operator()(LSPRequest request);

 private:
  void startWorkersIfNeeded();
  void stopWorkersIfNeeded();

  void cancelPendingParserRequests(const std::string& fileName);

  void run(std::function<void()> task);

  void parse(const std::string& uri);

  [[nodiscard]] auto latestDocument(const std::string& uri)
      -> std::shared_ptr<CxxDocument>;

  [[nodiscard]] auto nextRequest() -> std::optional<json>;

  void sendToClient(LSPRequest notification);
  void sendToClient(LSPResponse response);
  void sendMessage(const json& message);

  void logTrace(std::string message, std::optional<std::string> verbose = {});

  [[nodiscard]] auto pathFromUri(const std::string& uri) -> std::string;

  [[nodiscard]] auto readHeaders(std::istream& input)
      -> std::unordered_map<std::string, std::string>;

  struct Text {
    std::string value;
    std::vector<std::size_t> lineStartOffsets;
    std::int64_t version = 0;

    auto offsetAt(std::size_t line, std::size_t column) const -> std::size_t;

    void computeLineStartOffsets();
  };

 private:
  const CLI& cli;
  std::istream& input;
  std::ostream& output;
  std::ostream& log;
  std::unordered_map<std::string, std::shared_ptr<CxxDocument>> documents_;
  std::unordered_map<std::string, Text> documentContents_;
  std::vector<std::shared_ptr<CxxDocument>> pendingParserRequests_;
#ifndef CXX_NO_THREADS
  SyncQueue syncQueue_;
  std::vector<std::thread> workers_;
  std::mutex documentsMutex_;
  std::mutex outputMutex_;
#endif
  TraceValue trace_{};
  bool done_ = false;
};

}  // namespace cxx::lsp
