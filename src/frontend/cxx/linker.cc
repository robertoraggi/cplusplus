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

#include "linker.h"

#include <cxx/cli.h>
#include <cxx/linker/driver.h>
#include <cxx/toolchain.h>

#include <cerrno>
#include <cstring>
#include <format>
#include <iostream>
#include <optional>

#if !defined(__wasi__) && !defined(__EMSCRIPTEN__) && \
    (defined(__APPLE__) || defined(__linux__))
#include <spawn.h>
#include <sys/wait.h>

extern char** environ;
#endif

namespace cxx {
namespace {

void printLinkCommand(const CLI& cli, const std::vector<std::string>& args) {
  if (!cli.opt_v) return;
  std::cerr << "cxx: linking with";
  for (const auto& arg : args) std::cerr << ' ' << arg;
  std::cerr << std::endl;
}

void addLinkInputs(const CLI& cli, const std::vector<std::string>& inputs,
                   std::vector<std::string>& args, bool external) {
  auto input = inputs.begin();
  for (const auto& match : cli) {
    if (std::get_if<CLIPositional>(&match)) {
      if (input != inputs.end()) args.push_back(*input++);
      continue;
    }
    if (auto forwarded = std::get_if<CLIForwarded>(&match)) {
      args.push_back(std::get<0>(*forwarded));
      continue;
    }
    auto option = std::get_if<CLIOption>(&match);
    if (!option) continue;
    const auto& name = std::get<0>(*option);
    const auto& value = std::get<1>(*option);
    if (name == "-L" || name == "-l") {
      args.push_back(name + value);
      continue;
    }
    if (name == "-Xlinker") {
      if (external) args.push_back(name);
      args.push_back(value);
      continue;
    }
    if (!external) continue;
    if (name == "--sysroot" || name == "-stdlib") {
      args.push_back(name + "=" + value);
    } else if (name == "-isysroot" || name == "-arch" || name == "-F" ||
               name == "-framework") {
      args.push_back(name);
      args.push_back(value);
    }
  }
}

[[nodiscard]] auto defaultLinkDriver(const Toolchain* toolchain)
    -> std::optional<std::string> {
  const bool cxx = toolchain->language() == LanguageKind::kCXX;
  switch (toolchain->linkerFlavor()) {
    case LinkerFlavor::kDarwin:
      return cxx ? "clang++" : "clang";
    case LinkerFlavor::kGnu:
      return cxx ? "g++" : "gcc";
    default:
      return std::nullopt;
  }
}

auto runLinkDriver(std::vector<std::string>& args) -> bool {
#if !defined(__wasi__) && !defined(__EMSCRIPTEN__) && \
    (defined(__APPLE__) || defined(__linux__))
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);

  pid_t process;
  const auto error = posix_spawnp(&process, argv.front(), nullptr, nullptr,
                                  argv.data(), environ);
  if (error) {
    std::cerr << std::format("cxx: cannot execute '{}': {}\n", args.front(),
                             std::strerror(error));
    return false;
  }

  int status = 0;
  while (waitpid(process, &status, 0) < 0) {
    if (errno == EINTR) continue;
    std::cerr << std::format("cxx: cannot wait for '{}': {}\n", args.front(),
                             std::strerror(errno));
    return false;
  }
  if (WIFSIGNALED(status)) {
    std::cerr << std::format(
        "cxx: linker driver '{}' terminated by signal {}\n", args.front(),
        WTERMSIG(status));
    return false;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
  std::cerr
      << "cxx: external linker processes are unavailable on this platform\n";
  return false;
#endif
}

auto linkExternal(const CLI& cli, Toolchain* toolchain,
                  const std::vector<std::string>& inputs,
                  const std::string& outputPath) -> bool {
  auto driver = cli.getSingle("-fuse-ld");
  if (!driver) driver = defaultLinkDriver(toolchain);
  if (!driver || driver->empty()) {
    std::cerr << "cxx: no external linker driver for the selected toolchain\n";
    return false;
  }
  std::vector<std::string> args{*driver};
  if (toolchain->linkerFlavor() == LinkerFlavor::kDarwin) {
    args.emplace_back("-target");
    args.push_back(toolchain->triple().str());
    if (!cli.getSingle("-isysroot") && !cli.getSingle("--sysroot")) {
      args.emplace_back("-isysroot");
      args.push_back(toolchain->sysroot());
    }
  }
  addLinkInputs(cli, inputs, args, true);
  args.emplace_back("-o");
  args.push_back(outputPath);
  printLinkCommand(cli, args);
  return runLinkDriver(args);
}

auto linkEmbedded(const CLI& cli, Toolchain* toolchain,
                  const std::vector<std::string>& inputs,
                  const std::string& outputPath) -> bool {
  if (toolchain->linkerFlavor() != LinkerFlavor::kWasm) {
    std::cerr
        << "cxx: embedded linker does not support the selected toolchain\n";
    return false;
  }
  std::vector<std::string> args;
  toolchain->addLinkerStartArgs(args);
  addLinkInputs(cli, inputs, args, false);
  toolchain->addLinkerEndArgs(args);
  args.emplace_back("-o");
  args.push_back(outputPath);
  printLinkCommand(cli, args);

  std::string error;
  if (linker::link(args, error)) return true;
  std::cerr << std::format("cxx-link: error: {}\n", error);
  return false;
}

}  // namespace

auto link(const CLI& cli, Toolchain* toolchain,
          const std::vector<std::string>& inputs, const std::string& outputPath)
    -> bool {
  if (cli.getSingle("-fuse-ld")) {
    return linkExternal(cli, toolchain, inputs, outputPath);
  }
  if (toolchain->linkerFlavor() == LinkerFlavor::kWasm) {
    return linkEmbedded(cli, toolchain, inputs, outputPath);
  }
  return linkExternal(cli, toolchain, inputs, outputPath);
}
}  // namespace cxx
