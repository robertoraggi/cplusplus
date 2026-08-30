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

#include "cxx_server_host.h"

#include <cxx/cli.h>
#include <cxx/control.h>
#include <cxx/preprocessor.h>
#include <cxx/toolchain.h>
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>

#include <algorithm>

#include "cxx_document.h"

namespace cxx::lsp {

CxxServerHost::CxxServerHost(const CLI& cli) : cli_(cli) {}

CxxServerHost::~CxxServerHost() {}

void CxxServerHost::process(CxxDocument& document, std::string source,
                            std::function<void()> done) {
  auto unit = document.translationUnit();
  auto preprocessor = unit->preprocessor();

  std::string error;
  auto toolchain = createToolchain(cli_, preprocessor, error);

  if (toolchain && error.empty()) {
    unit->control()->setMemoryLayout(toolchain->memoryLayout());
    document.setToolchain(std::move(toolchain));
  }

  DefaultPreprocessorState state{*preprocessor};

  unit->beginPreprocessing(std::move(source), document.fileName());

  while (state) {
    if (document.isCancelled()) break;
    std::visit(state, unit->continuePreprocessing());
  }

  unit->endPreprocessing();

  if (!document.isCancelled()) {
    unit->parse(document.parserConfiguration());
  }

  done();
}

auto CxxServerHost::pathFromUri(const std::string& uri)
    -> std::optional<std::string> {
  if (uri.starts_with("file://")) return uri.substr(7);
  if (cli_.opt_lsp_test && uri.starts_with("test://")) return uri.substr(7);
  return std::nullopt;
}

void CxxServerHost::run(std::function<void()> task) {
#ifndef CXX_NO_THREADS
  if (!workers_.empty()) {
    syncQueue_.push(std::move(task));
    return;
  }
#endif

  task();
}

void CxxServerHost::runLater(std::chrono::milliseconds delay,
                             std::function<void()> task) {
#ifndef CXX_NO_THREADS
  if (!cli_.opt_lsp_test) {
    std::thread([delay, task = std::move(task)] {
      std::this_thread::sleep_for(delay);
      task();
    }).detach();
    return;
  }
#endif

  task();
}

void CxxServerHost::start() {
#ifndef CXX_NO_THREADS
  if (cli_.opt_lsp_test) return;

  const auto threadCountOption = cli_.getSingle("-j");

  int workerCount = 0;

  if (threadCountOption.has_value()) {
    workerCount = std::stoi(threadCountOption.value());
  }

  if (workerCount <= 0) {
    workerCount = int(std::thread::hardware_concurrency());
  }

  if (workerCount <= 0) {
    workerCount = 1;
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

void CxxServerHost::stop() {
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

}  // namespace cxx::lsp
