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

#include "toolchain_options.h"

#include <cxx/cli.h>
#include <cxx/toolchain.h>
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <optional>

namespace cxx::js {

using emscripten::val;

namespace {

[[nodiscard]] auto stringOption(const val& options, const char* name)
    -> std::optional<std::string> {
  val value = options[name];
  if (!value.isString()) return std::nullopt;
  return value.as<std::string>();
}

void addFlag(std::vector<std::string>& args, const val& options,
             const char* name, const char* enabled, const char* disabled) {
  val value = options[name];
  if (value.isTrue()) args.push_back(enabled);
  if (value.isFalse()) args.push_back(disabled);
}

void addJoined(std::vector<std::string>& args, const val& options,
               const char* name, const std::string& flag) {
  if (auto value = stringOption(options, name)) args.push_back(flag + *value);
}

void addEach(std::vector<std::string>& args, const val& options,
             const char* name, const char* flag) {
  for (auto& value : stringArrayOption(options, name)) {
    args.push_back(flag);
    args.push_back(std::move(value));
  }
}

[[nodiscard]] auto commandLineOf(const val& options)
    -> std::vector<std::string> {
  std::vector<std::string> args{"cxx"};
  if (options.isUndefined()) return args;
  if (auto appdir = stringOption(options, "appdir"))
    args.front() = (std::filesystem::path(*appdir) / "cxx").string();
  addJoined(args, options, "target", "--target=");
  addJoined(args, options, "deviceProfile", "--device-profile=");
  addJoined(args, options, "sysroot", "--sysroot=");
  addJoined(args, options, "std", "-std=");
  addFlag(args, options, "exceptions", "-fexceptions", "-fno-exceptions");
  addFlag(args, options, "rtti", "-frtti", "-fno-rtti");
  addEach(args, options, "systemIncludePaths", "-isystem");
  addEach(args, options, "quoteIncludePaths", "-iquote");
  addEach(args, options, "includePaths", "-I");
  addEach(args, options, "undefines", "-U");
  addEach(args, options, "defines", "-D");
  return args;
}

}  // namespace

auto stringArrayOption(const val& options, const char* name)
    -> std::vector<std::string> {
  if (options.isUndefined()) return {};
  val value = options[name];
  if (!value.isArray()) return {};
  return emscripten::vecFromJSArray<std::string>(value);
}

auto configureToolchain(TranslationUnit* unit, const std::string& fileName,
                        const val& options, std::vector<std::string> overrides)
    -> ConfiguredToolchain {
  auto preprocessor = unit->preprocessor();
  if (!preprocessor) return {};

  auto args = commandLineOf(options);
  std::ranges::move(overrides, std::back_inserter(args));
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  char** arguments = argv.data();

  CLI cli;
  if (!cli.parse(argc, arguments)) return {nullptr, "invalid unit options"};

  ConfiguredToolchain configured;
  configured.toolchain = createToolchain(
      cli, preprocessor, languageOf(cli, fileName), configured.error);
  if (!configured.toolchain && configured.error.empty())
    configured.error = describeUnsupportedTarget(cli);
  return configured;
}

}  // namespace cxx::js
