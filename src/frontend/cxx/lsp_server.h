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

#pragma once

#include <cxx/lsp/fwd.h>

#include <istream>
#include <ostream>

#ifndef CXX_NO_THREADS
#include <thread>
#include <unordered_map>
#endif

#include <vector>

#include "cli.h"
#include "sync_queue.h"

namespace cxx::lsp {

class CxxDocument;

class Server {
 public:
  Server(const CLI& cli);
  ~Server();

  auto start() -> int;

  void operator()(const InitializeRequest& request);
  void operator()(const InitializedNotification& notification);

  void operator()(const ShutdownRequest& request);
  void operator()(const ExitNotification& notification);

  void operator()(const DidOpenTextDocumentNotification& notification);
  void operator()(const DidCloseTextDocumentNotification& notification);
  void operator()(const DidChangeTextDocumentNotification& notification);

  void operator()(const DocumentDiagnosticRequest& request);

  void operator()(const CancelNotification& notification);
  void operator()(const LSPRequest& request);

 private:
  void startWorkersIfNeeded();
  void stopWorkersIfNeeded();

  void run(std::function<void()> task);

  void parse(std::string uri, std::string text, long version);

  [[nodiscard]] auto latestDocument(const std::string& uri)
      -> std::shared_ptr<CxxDocument>;

  [[nodiscard]] auto nextRequest() -> std::optional<json>;

  void sendNotification(const LSPRequest& notification);

  void sendToClient(const json& message);

  void sendToClient(
      const LSPObject& result,
      std::optional<std::variant<long, std::string>> id = std::nullopt);

  [[nodiscard]] auto pathFromUri(const std::string& uri) -> std::string;

  [[nodiscard]] auto readHeaders(std::istream& input)
      -> std::unordered_map<std::string, std::string>;

 private:
  const CLI& cli;
  std::istream& input;
  std::ostream& output;
  std::ostream& log;
  std::unordered_map<std::string, std::shared_ptr<CxxDocument>> documents_;
#ifndef CXX_NO_THREADS
  SyncQueue syncQueue_;
  std::vector<std::thread> workers_;
  std::mutex documentsMutex_;
  std::mutex outputMutex_;
#endif
  bool done_ = false;
};

}  // namespace cxx::lsp
