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

#include <cxx/linker/linker.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <string>
#include <vector>

using emscripten::val;

namespace {

EMSCRIPTEN_DECLARE_VAL_TYPE(LinkerOptions);
EMSCRIPTEN_DECLARE_VAL_TYPE(LinkInputList);
EMSCRIPTEN_DECLARE_VAL_TYPE(LinkResult);

[[nodiscard]] auto copyBytes(const val& array)
    -> std::shared_ptr<cxx::linker::Bytes> {
  const auto size = array["length"].as<std::size_t>();
  auto bytes = std::make_shared<cxx::linker::Bytes>(size);
  val view{emscripten::typed_memory_view(size, bytes->data())};
  view.call<void>("set", array);
  return bytes;
}

class LinkInput {
 public:
  LinkInput(std::string name, const val& data)
      : name_(std::move(name)), content_(copyBytes(data)) {
    if (!cxx::linker::Archive::isArchive(*content_)) return;
    archive_ = cxx::linker::Archive::create(name_, content_, error_);
  }

  [[nodiscard]] auto getName() const -> std::string { return name_; }
  [[nodiscard]] auto getError() const -> std::string { return error_; }
  [[nodiscard]] auto getSize() const -> std::size_t { return content_->size(); }
  [[nodiscard]] auto isArchive() const -> bool { return archive_ != nullptr; }

  [[nodiscard]] auto addTo(cxx::linker::Linker& linker,
                           std::string& error) const -> bool {
    if (!archive_) return linker.addObject(name_, content_, error);
    linker.addArchive(archive_);
    return true;
  }

 private:
  std::string name_;
  std::shared_ptr<const cxx::linker::Bytes> content_;
  std::shared_ptr<const cxx::linker::Archive> archive_;
  std::string error_;
};

[[nodiscard]] auto stringOption(const val& options, const char* name,
                                std::string fallback) -> std::string {
  const val value = options[name];
  return value.isString() ? value.as<std::string>() : std::move(fallback);
}

[[nodiscard]] auto numberOption(const val& options, const char* name,
                                std::uint32_t fallback) -> std::uint32_t {
  const val value = options[name];
  return value.isNumber() ? value.as<std::uint32_t>() : fallback;
}

[[nodiscard]] auto boolOption(const val& options, const char* name,
                              bool fallback) -> bool {
  const val value = options[name];
  const bool isBoolean = value.isTrue() || value.isFalse();
  return isBoolean ? value.as<bool>() : fallback;
}

[[nodiscard]] auto linkOptionsFrom(const val& options)
    -> cxx::linker::LinkOptions {
  cxx::linker::LinkOptions result;
  if (options.isUndefined()) return result;
  result.entry = stringOption(options, "entry", result.entry);
  if (options["exports"].isArray()) {
    result.exports =
        emscripten::vecFromJSArray<std::string>(options["exports"]);
  }
  result.globalBase = numberOption(options, "globalBase", result.globalBase);
  result.stackSize = numberOption(options, "stackSize", result.stackSize);
  result.initialMemory =
      numberOption(options, "initialMemory", result.initialMemory);
  result.maxMemory = numberOption(options, "maxMemory", result.maxMemory);
  result.stackFirst = boolOption(options, "stackFirst", result.stackFirst);
  result.gcSections = boolOption(options, "gcSections", result.gcSections);
  result.noEntry = boolOption(options, "noEntry", result.noEntry);
  result.allowUndefined =
      boolOption(options, "allowUndefined", result.allowUndefined);
  const auto strip = stringOption(options, "strip", "none");
  result.stripDebug = strip == "debug";
  result.stripAll = strip == "all";
  return result;
}

[[nodiscard]] auto toUint8Array(const cxx::linker::Bytes& bytes) -> val {
  auto result = val::global("Uint8Array").new_(bytes.size());
  val view{emscripten::typed_memory_view(bytes.size(), bytes.data())};
  result.call<void>("set", view);
  return result;
}

[[nodiscard]] auto failure(std::string error) -> LinkResult {
  val result = val::object();
  result.set("output", val::global("Uint8Array").new_(0));
  result.set("error", val(std::move(error)));
  return LinkResult(result);
}

auto link(LinkInputList inputs, LinkerOptions options) -> LinkResult {
  cxx::linker::Linker linker{linkOptionsFrom(options)};
  std::string error;
  const val list = inputs;
  const auto count = list["length"].as<std::size_t>();
  for (std::size_t i = 0; i < count; ++i) {
    const auto* input =
        list[i].as<LinkInput*>(emscripten::allow_raw_pointers());
    if (!input->addTo(linker, error)) return failure(std::move(error));
  }
  cxx::linker::Bytes output;
  if (!linker.link(output, error)) return failure(std::move(error));
  val result = val::object();
  result.set("output", toUint8Array(output));
  result.set("error", val(std::string{}));
  return LinkResult(result);
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx_link) {
  emscripten::register_type<LinkerOptions>(
      "LinkerOptions",
      R"({ entry?: string | undefined; exports?: string[] | undefined; globalBase?: number | undefined; stackSize?: number | undefined; initialMemory?: number | undefined; maxMemory?: number | undefined; stackFirst?: boolean | undefined; gcSections?: boolean | undefined; noEntry?: boolean | undefined; allowUndefined?: boolean | undefined; strip?: "none" | "debug" | "all" | undefined })");

  emscripten::register_type<LinkInputList>("LinkInputList", "LinkInput[]");

  emscripten::register_type<LinkResult>(
      "LinkResult", R"({ output: Uint8Array; error: string })");

  emscripten::class_<LinkInput>("LinkInput")
      .constructor<std::string, val>()
      .function("getName", &LinkInput::getName)
      .function("getError", &LinkInput::getError)
      .function("getSize", &LinkInput::getSize)
      .function("isArchive", &LinkInput::isArchive);

  emscripten::function("link", &link);
}
