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
#include <cxx/lsp/types.h>
#include <cxx/preprocessor.h>
#include <cxx/toolchain.h>
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
#include <string_view>
#include <vector>

#include "async_parse.h"
#include "emit_code.h"
#include "toolchain_options.h"

using namespace emscripten;

namespace {

EMSCRIPTEN_DECLARE_VAL_TYPE(LanguageServerOptions);

class JsServerHost final : public cxx::lsp::ServerHost {
 public:
  explicit JsServerHost(val options) : options_(options) {}

  void retainedBy(std::weak_ptr<void> owner) { owner_ = std::move(owner); }

  void close() { *closed_ = true; }

  [[nodiscard]] auto isClosed() const -> bool { return *closed_; }

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
                              cxx::lsp::EmitCodeFormat format, bool debugInfo,
                              int optimizationLevel)
      -> std::optional<cxx::lsp::EmittedCode> override {
    if (document.hasErrors()) return cxx::lsp::EmittedCode{};

    if (auto toolchain = document.toolchain()) {
      toolchain->applyEntryPointAbi(document.translationUnit());
    }

    if (format == cxx::lsp::EmitCodeFormat::kExecutable) {
      return emitExecutable(document, debugInfo, optimizationLevel);
    }

    auto generated =
        cxx::js::generateCode(document.translationUnit(), to_string(format),
                              debugInfo, optimizationLevel);

    if (!generated) return std::nullopt;

    return cxx::lsp::EmittedCode{.text = std::move(generated->text)};
  }

  void runLater(std::chrono::milliseconds delay,
                std::function<void()> task) override {
    auto scheduled = new ScheduledTask{std::move(task), closed_};

    emscripten_set_timeout(
        [](void* userData) {
          std::unique_ptr<ScheduledTask> scheduled{
              static_cast<ScheduledTask*>(userData)};
          if (*scheduled->closed) return;
          scheduled->task();
        },
        double(delay.count()), scheduled);
  }

  void trace(const std::string& message,
             const std::optional<std::string>& verbose) override {
    if (isClosed() || options_.isUndefined()) return;

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
  [[nodiscard]] auto emitExecutable(cxx::lsp::CxxDocument& document,
                                    bool debugInfo, int optimizationLevel)
      -> std::optional<cxx::lsp::EmittedCode> {
    if (options_.isUndefined()) return std::nullopt;

    val link = options_["link"];
    if (link.isUndefined()) return std::nullopt;

    auto generated = cxx::js::generateCode(document.translationUnit(), "obj",
                                           debugInfo, optimizationLevel);

    if (!generated || generated->objectCode.empty()) {
      return cxx::lsp::EmittedCode{.error = "code generation failed"};
    }

    auto& objectCode = generated->objectCode;
    auto object = val::global("Uint8Array").new_(objectCode.size());
    object.call<void>(
        "set", val(typed_memory_view(objectCode.size(), objectCode.data())));

    val linked = link(object, debugInfo);

    if (auto error = linked["error"].as<std::string>(); !error.empty()) {
      return cxx::lsp::EmittedCode{.error = std::move(error)};
    }

    return cxx::lsp::EmittedCode{.text = base64Encode(linked["output"])};
  }

  [[nodiscard]] static auto base64Encode(const val& bytes) -> std::string {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    const auto size = bytes["length"].as<std::size_t>();
    std::vector<std::uint8_t> data(size);
    val(typed_memory_view(size, data.data())).call<void>("set", bytes);

    std::string text;
    text.reserve((size + 2) / 3 * 4);

    const auto byteAt = [&](std::size_t index) -> std::uint32_t {
      return index < size ? data[index] : 0;
    };

    for (std::size_t i = 0; i < size; i += 3) {
      const auto remaining = size - i;
      const auto chunk = byteAt(i) << 16 | byteAt(i + 1) << 8 | byteAt(i + 2);
      text.push_back(alphabet[chunk >> 18 & 63]);
      text.push_back(alphabet[chunk >> 12 & 63]);
      text.push_back(remaining > 1 ? alphabet[chunk >> 6 & 63] : '=');
      text.push_back(remaining > 2 ? alphabet[chunk & 63] : '=');
    }

    return text;
  }

  auto drain() -> val {
    auto retained = owner_.lock();

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
    auto retained = owner_.lock();
    auto unit = document.translationUnit();
    const auto fileName = document.fileName();
    const auto version = document.version();

    document.setToolchain(cxx::js::configureToolchain(unit, options_));

    cxx::js::AsyncParseRequest request{
        .unit = unit,
        .source = {},
        .fileName = document.fileName(),
        .config = document.parserConfiguration(),
    };

    if (!options_.isUndefined()) {
      request.exists = options_["exists"];
      request.readFile = options_["readFile"];
      request.readDirectory = options_["readDirectory"];
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

    bool usePreamble = true;
    if (!options_.isUndefined()) {
      auto enabled = options_["preamble"];
      if (!enabled.isUndefined()) usePreamble = enabled.as<bool>();
    }
    if (document.preambleCache && usePreamble &&
        !document.completionIsInIncludeDirective(source)) {
      auto cached = document.preambleCache->get(source);
      if (!cached) {
        const auto started = std::chrono::steady_clock::now();
        cxx::lsp::CxxDocument producer(fileName, version);
        producer.setToolchain(
            cxx::js::configureToolchain(producer.translationUnit(), options_));
        producer.translationUnit()->preprocessor()->setPreambleOnly(true);
        auto prefixRequest = request;
        prefixRequest.unit = producer.translationUnit();
        prefixRequest.source = source;
        prefixRequest.fileName = producer.fileName();
        prefixRequest.config.complete = {};
        auto built = co_await cxx::js::asyncParse(std::move(prefixRequest));
        if (!built.as<bool>()) {
          document.cancel();
          done();
          co_return val::undefined();
        }
        auto preambleSize =
            producer.translationUnit()->preprocessor()->preambleSize();
        if (preambleSize && producer.diagnostics().empty()) {
          cxx::PrecompiledHeaderWriter writer(producer.translationUnit(),
                                              cxx::lsp::preambleKeys());
          writer.setPreprocessorState(
              producer.translationUnit()->preprocessor()->preambleState());
          auto bytes = writer();
          if (writer.errors().empty()) {
            cached = std::make_shared<cxx::lsp::Preamble>(
                source.substr(0, *preambleSize), std::move(bytes));
            document.preambleCache->put(cached);
          } else {
            for (const auto& error : writer.errors())
              trace("preamble event=failed reason=" + error, {});
          }
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        auto event = "skipped";
        if (cached) event = "built";
        trace(std::format("preamble event={} file={} duration_ms={:.1f}", event,
                          fileName, elapsed),
              {});
      }
      if (cached && !document.completionIsWithin(cached->source.size())) {
        const auto started = std::chrono::steady_clock::now();
        cxx::PrecompiledHeaderReader reader(unit, cxx::lsp::preambleKeys());
        if (reader(cached->bytes)) {
          cxx::lsp::maskPreamble(source, cached->source.size());
          const auto elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
          trace(std::format(
                    "preamble event=reused file={} bytes={} duration_ms={:.1f}",
                    fileName, cached->bytes.size(), elapsed),
                {});
        } else {
          trace("preamble event=failed reason=" + reader.error(), {});
        }
      }
    }

    request.source = std::move(source);
    auto completed = co_await cxx::js::asyncParse(std::move(request));
    if (!completed.as<bool>()) document.cancel();

    done();

    co_return val::undefined();
  }

  struct ScheduledTask {
    std::function<void()> task;
    std::shared_ptr<bool> closed;
  };

  val options_;
  std::shared_ptr<bool> closed_ = std::make_shared<bool>(false);
  std::weak_ptr<void> owner_;
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

  void close() { closed_ = true; }

  void sendMessage(const cxx::lsp::json& message) override {
    if (closed_ || options_.isUndefined()) return;

    val onMessage = options_["onMessage"];
    if (onMessage.isUndefined()) return;

    onMessage(message.dump());
  }

  void push(cxx::lsp::json message) { inbox_.push_back(std::move(message)); }

 private:
  val options_;
  bool closed_ = false;
  std::deque<cxx::lsp::json> inbox_;
};

struct LanguageServerCore {
  explicit LanguageServerCore(LanguageServerOptions options)
      : host(options),
        transport(new JsTransport(options)),
        server(host, std::unique_ptr<cxx::lsp::Transport>(transport)) {}

  JsServerHost host;
  JsTransport* transport;
  cxx::lsp::Server server;
};

class WrappedLanguageServer {
 public:
  explicit WrappedLanguageServer(LanguageServerOptions options)
      : core_(std::make_shared<LanguageServerCore>(options)) {
    core_->host.retainedBy(core_);
    core_->server.startProcessing();
  }

  ~WrappedLanguageServer() {
    core_->host.close();
    core_->transport->close();
    core_->server.stopProcessing();
  }

  auto receive(std::string message) -> val {
    auto core = core_;

    core->transport->push(cxx::lsp::json::parse(message));

    core->server.continueProcessing();

    while (auto pending = core->host.takePending()) {
      co_await *pending;
    }

    co_return val::undefined();
  }

 private:
  std::shared_ptr<LanguageServerCore> core_;
};

auto createLanguageServer(LanguageServerOptions options)
    -> WrappedLanguageServer* {
  return new WrappedLanguageServer(options);
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx_lsp) {
  register_type<LanguageServerOptions>(
      "LanguageServerOptions",
      R"({ preamble?: boolean | undefined; appdir?: string | undefined; sysroot?: string | undefined; std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined; defines?: string[] | undefined; undefines?: string[] | undefined; quoteIncludePaths?: string[] | undefined; includePaths?: string[] | undefined; systemIncludePaths?: string[] | undefined; exists?: ((path: string) => boolean) | undefined; readFile?: ((path: string) => Promise<string | undefined>) | undefined; readDirectory?: ((path: string) => string[] | undefined) | undefined; link?: ((object: Uint8Array, debugInfo: boolean) => { output: Uint8Array; error: string }) | undefined; shouldContinue?: (() => Promise<boolean>) | undefined; onTrace?: ((message: string, verbose: string | undefined) => void) | undefined; onMessage: (message: string) => void })");

  class_<WrappedLanguageServer>("LanguageServer")
      .function("receive", &WrappedLanguageServer::receive);

  function("createLanguageServer", &createLanguageServer, allow_raw_pointers());
}
