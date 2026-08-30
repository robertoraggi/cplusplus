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

#include <vector>

#ifndef CXX_NO_THREADS
#include <thread>
#endif

#include "server_host.h"
#include "sync_queue.h"

namespace cxx {
class CLI;
}

namespace cxx::lsp {

class CxxServerHost final : public ServerHost {
 public:
  explicit CxxServerHost(const CLI& cli);
  ~CxxServerHost() override;

  void process(CxxDocument& document, std::string source,
               std::function<void()> done) override;

  [[nodiscard]] auto pathFromUri(const std::string& uri)
      -> std::optional<std::string> override;

  void run(std::function<void()> task) override;

  void runLater(std::chrono::milliseconds delay,
                std::function<void()> task) override;

  void start() override;

  void stop() override;

 private:
  const CLI& cli_;
#ifndef CXX_NO_THREADS
  SyncQueue syncQueue_;
  std::vector<std::thread> workers_;
#endif
};

}  // namespace cxx::lsp
