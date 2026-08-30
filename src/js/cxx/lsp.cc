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

#include <cxx/lsp/cxx_document.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/json.h>
#include <cxx/lsp/lsp_server.h>
#include <cxx/lsp/server_host.h>
#include <cxx/lsp/transport.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <emscripten/bind.h>
#include <emscripten/eventloop.h>
#include <emscripten/val.h>

#include <deque>
#include <format>
#include <memory>
#include <optional>
#include <string>

#include "async_parse.h"
#include "emit_code.h"
#include "toolchain_options.h"

using namespace emscripten;

namespace {

EMSCRIPTEN_DECLARE_VAL_TYPE(LanguageServerOptions);

class JsServerHost final : public cxx::lsp::ServerHost {
 public:
  explicit JsServerHost(val options) : options_(options) {}

  void run(std::function<void()> task) override {
    tasks_.push_back(std::move(task));
    if (draining_) return;
    draining_ = true;
    pending_.push_back(drain());
  }

  void process(cxx::lsp::CxxDocument& document, std::string source,
               std::function<void()> done) override {
    inFlight_ = processAsync(document, std::move(source), std::move(done));
  }

  [[nodiscard]] auto pathFromUri(const std::string& uri)
      -> std::optional<std::string> override {
    if (uri.starts_with("file://")) return uri.substr(7);
    return uri;
  }

  [[nodiscard]] auto supportsEmitCode() const -> bool override {
    return cxx::js::hasCodeGenerator();
  }

  [[nodiscard]] auto emitCode(cxx::lsp::CxxDocument& document,
                              cxx::lsp::EmitCodeFormat format, bool debugInfo)
      -> std::optional<std::string> override {
    if (document.hasErrors()) return std::string{};

    auto generated = cxx::js::generateCode(document.translationUnit(),
                                           to_string(format), debugInfo);

    if (!generated) return std::nullopt;

    return std::move(generated->text);
  }

  void runLater(std::chrono::milliseconds delay,
                std::function<void()> task) override {
    auto scheduled = new std::function<void()>(std::move(task));

    emscripten_set_timeout(
        [](void* userData) {
          std::unique_ptr<std::function<void()>> task{
              static_cast<std::function<void()>*>(userData)};
          (*task)();
        },
        double(delay.count()), scheduled);
  }

  void trace(const std::string& message,
             const std::optional<std::string>& verbose) override {
    if (options_.isUndefined()) return;

    auto onTrace = options_["onTrace"];
    if (onTrace.isUndefined()) return;

    if (verbose.has_value()) {
      onTrace(message, *verbose);
      return;
    }

    onTrace(message, val::undefined());
  }

  [[nodiscard]] auto takePending() -> std::optional<val> {
    if (pending_.empty()) return std::nullopt;
    auto promise = pending_.front();
    pending_.pop_front();
    return promise;
  }

 private:
  auto drain() -> val {
    while (!tasks_.empty()) {
      auto task = std::move(tasks_.front());
      tasks_.pop_front();

      task();

      if (inFlight_.has_value()) {
        auto parsing = std::move(*inFlight_);
        inFlight_.reset();
        co_await parsing;
      }
    }

    draining_ = false;

    co_return val::undefined();
  }

  auto processAsync(cxx::lsp::CxxDocument& document, std::string source,
                    std::function<void()> done) -> val {
    auto unit = document.translationUnit();
    const auto fileName = document.fileName();
    const auto version = document.version();

    document.setToolchain(cxx::js::configureToolchain(unit, options_));

    cxx::js::AsyncParseRequest request{
        .unit = unit,
        .source = std::move(source),
        .fileName = document.fileName(),
        .config = document.parserConfiguration(),
    };

    if (!options_.isUndefined()) {
      request.exists = options_["exists"];
      request.readFile = options_["readFile"];
      request.shouldContinue = options_["shouldContinue"];
    }

    request.didFinishPhase = [this, fileName, version](std::string_view phase,
                                                       double elapsedMs,
                                                       bool cancelled) {
      trace(std::format("parse phase={} file={} version={} duration_ms={:.1f} "
                        "cancelled={}",
                        phase, fileName, version, elapsedMs, cancelled),
            {});
    };

    auto completed = co_await cxx::js::asyncParse(std::move(request));
    if (!completed.as<bool>()) document.cancel();

    done();

    co_return val::undefined();
  }

  val options_;
  std::deque<std::function<void()>> tasks_;
  std::deque<val> pending_;
  std::optional<val> inFlight_;
  bool draining_ = false;
};

class JsTransport final : public cxx::lsp::Transport {
 public:
  explicit JsTransport(val options) : options_(options) {}

  [[nodiscard]] auto isOpen() const -> bool override { return true; }

  [[nodiscard]] auto nextMessage() -> std::optional<cxx::lsp::json> override {
    if (inbox_.empty()) return std::nullopt;
    auto message = std::move(inbox_.front());
    inbox_.pop_front();
    return message;
  }

  void sendMessage(const cxx::lsp::json& message) override {
    if (options_.isUndefined()) return;

    val onMessage = options_["onMessage"];
    if (onMessage.isUndefined()) return;

    onMessage(message.dump());
  }

  void push(cxx::lsp::json message) { inbox_.push_back(std::move(message)); }

 private:
  val options_;
  std::deque<cxx::lsp::json> inbox_;
};

class WrappedLanguageServer {
 public:
  explicit WrappedLanguageServer(LanguageServerOptions options)
      : host_(options),
        transport_(new JsTransport(options)),
        server_(host_, std::unique_ptr<cxx::lsp::Transport>(transport_)) {
    server_.startProcessing();
  }

  ~WrappedLanguageServer() { server_.stopProcessing(); }

  auto receive(std::string message) -> val {
    transport_->push(cxx::lsp::json::parse(message));

    server_.continueProcessing();

    while (auto pending = host_.takePending()) {
      co_await *pending;
    }

    co_return val::undefined();
  }

 private:
  JsServerHost host_;
  JsTransport* transport_;
  cxx::lsp::Server server_;
};

auto createLanguageServer(LanguageServerOptions options)
    -> WrappedLanguageServer* {
  return new WrappedLanguageServer(options);
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx_lsp) {
  register_type<LanguageServerOptions>(
      "LanguageServerOptions",
      R"({ appdir?: string | undefined; sysroot?: string | undefined; std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined; defines?: string[] | undefined; undefines?: string[] | undefined; quoteIncludePaths?: string[] | undefined; includePaths?: string[] | undefined; systemIncludePaths?: string[] | undefined; exists?: ((path: string) => boolean) | undefined; readFile?: ((path: string) => Promise<string | undefined>) | undefined; shouldContinue?: (() => Promise<boolean>) | undefined; onTrace?: ((message: string, verbose: string | undefined) => void) | undefined; onMessage: (message: string) => void })");

  class_<WrappedLanguageServer>("LanguageServer")
      .function("receive", &WrappedLanguageServer::receive);

  function("createLanguageServer", &createLanguageServer, allow_raw_pointers());
}
