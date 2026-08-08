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

#include <cxx/lsp/lsp_server.h>

#include <chrono>
#include <format>
#include <iostream>

#include "frontend.h"
#include "linker.h"

#ifndef CXX_NO_FILESYSTEM
#include <filesystem>
#include <system_error>
#endif

namespace {
#ifndef CXX_NO_FILESYSTEM

namespace fs = std::filesystem;

auto isSourceInput(const std::string& fileName) -> bool {
  if (fileName == "-") return true;
  for (const char* ext :
       {".c", ".cc", ".cp", ".cpp", ".cxx", ".c++", ".C", ".i", ".ii"}) {
    if (fileName.ends_with(ext)) return true;
  }
  return false;
}

auto makeTempObject(const std::string& source, int index) -> std::string {
  auto stem = fs::path{source == "-" ? "stdin" : source}.stem().string();
  auto token = std::chrono::steady_clock::now().time_since_epoch().count();
  auto name = std::format("cxx-{}-{}-{}.o", stem, token, index);
  return (fs::temp_directory_path() / name).string();
}

auto compileAndLink(cxx::CLI& cli, const std::vector<std::string>& inputFiles)
    -> int {
  if (!cxx::haveEmbeddedLinker()) {
    std::cerr << "cxx: -flink requires a build with an embedded linker (lld)"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> objectsToLink;
  std::vector<std::string> tempObjects;
  cxx::Toolchain* toolchain = nullptr;
  auto exitStatus = EXIT_SUCCESS;

  std::vector<std::unique_ptr<cxx::Frontend>> frontends;

  int index = 0;
  for (const auto& fileName : inputFiles) {
    if (!isSourceInput(fileName)) {
      objectsToLink.push_back(fileName);
      continue;
    }

    auto objectFile = makeTempObject(fileName, index++);

    auto runOnFile = std::make_unique<cxx::Frontend>(cli, fileName);
    runOnFile->setObjectOutput(objectFile);

    if (!(*runOnFile)()) {
      exitStatus = EXIT_FAILURE;
      continue;
    }

    toolchain = runOnFile->toolchain();
    tempObjects.push_back(objectFile);
    objectsToLink.push_back(objectFile);
    frontends.push_back(std::move(runOnFile));
  }

  if (exitStatus == EXIT_SUCCESS) {
    if (!toolchain) {
      std::cerr << "cxx: nothing to link" << std::endl;
      exitStatus = EXIT_FAILURE;
    } else {
      auto outputPath = cli.getSingle("-o").value_or("a.out");
      if (!cxx::link(cli, toolchain, objectsToLink, outputPath)) {
        exitStatus = EXIT_FAILURE;
      }
    }
  }

  std::error_code ec;
  for (const auto& tmp : tempObjects) fs::remove(tmp, ec);

  return exitStatus;
}

#else

auto compileAndLink(cxx::CLI&, const std::vector<std::string>&) -> int {
  std::cerr << "cxx: -flink is not supported in this build" << std::endl;
  return EXIT_FAILURE;
}

#endif
}  // namespace

auto main(int argc, char* argv[]) -> int {
  cxx::CLI cli;
  cli.parse(argc, argv);

  if (cli.opt_help) {
    cli.showHelp();
    return EXIT_SUCCESS;
  }

  const auto& inputFiles = cli.positionals();

  if (cli.opt_fsyntax_only) {
    cli.opt_fcheck = true;
  }

  if (cli.opt_lsp_test) {
    cli.opt_lsp = true;
  }

  if (!cli.opt_lsp && inputFiles.empty()) {
    std::cerr << "cxx: no input files" << std::endl
              << "Usage: cxx [options] file..." << std::endl;
    return EXIT_FAILURE;
  }

  if (cli.opt_lsp) {
    auto server = cxx::lsp::Server{cli};

    return server.start();
  }

  const bool stopBeforeLink = cli.opt_E || cli.opt_Eonly ||
                              cli.opt_fsyntax_only || cli.opt_S || cli.opt_c ||
                              cli.opt_emit_ast || cli.opt_emit_cxx_ir ||
                              cli.opt_emit_mlir || cli.opt_emit_llvm;

  auto output = cli.getSingle("-o");
  const bool wantsExecutable = output.has_value() && output != "-";
  const bool doLink = cli.opt_link || (wantsExecutable && !stopBeforeLink &&
                                       cxx::haveEmbeddedLinker());

  if (doLink && !stopBeforeLink) {
    return compileAndLink(cli, inputFiles);
  }

  auto exitStatus = EXIT_SUCCESS;
  for (const auto& fileName : inputFiles) {
    cxx::Frontend runOnFile(cli, fileName);
    if (!runOnFile()) exitStatus = EXIT_FAILURE;
  }

  return exitStatus;
}
