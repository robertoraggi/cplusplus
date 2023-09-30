#// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

// cxx
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/lexer.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/format.h>
#include <cxx/private/path.h>
#include <cxx/recursive_ast_visitor.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#include "ast_printer.h"
#include "verify_diagnostics_client.h"

// fmt
#include <fmt/format.h>

// std
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <regex>
#include <sstream>
#include <string>

#include "cli.h"

namespace {
using namespace cxx;

auto readAll(const std::string& fileName, std::istream& in) -> std::string {
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);
  return code;
}

auto readAll(const std::string& fileName) -> std::string {
  if (fileName == "-" || fileName.empty()) return readAll("<stdin>", std::cin);
  std::ifstream stream(fileName);
  return readAll(fileName, stream);
}

void dumpTokens(const CLI& cli, TranslationUnit& unit, std::ostream& output) {
  std::string flags;

  for (SourceLocation loc(1);; loc = loc.next()) {
    const auto& tk = unit.tokenAt(loc);

    flags.clear();

    if (tk.startOfLine()) {
      flags += " [start-of-line]";
    }

    if (tk.leadingSpace()) {
      flags += " [leading-space]";
    }

    auto kind = tk.kind();
    if (kind == TokenKind::T_IDENTIFIER) {
      kind = Lexer::classifyKeyword(tk.spell());
    }

    fmt::print(output, "{} '{}'{}\n", Token::name(kind), tk.spell(), flags);

    if (tk.is(TokenKind::T_EOF_SYMBOL)) break;
  }
}

auto runOnFile(const CLI& cli, const std::string& fileName) -> bool {
  Control control;
  VerifyDiagnosticsClient diagnosticsClient;
  TranslationUnit unit(&control, &diagnosticsClient);

  auto preprocesor = unit.preprocessor();

  std::unique_ptr<Toolchain> toolchain;

  if (cli.opt_verify) {
    diagnosticsClient.setVerify(true);
    preprocesor->setCommentHandler(&diagnosticsClient);
  }

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
    toolchainId = "wasm32";
  }

  if (toolchainId == "darwin") {
    toolchain = std::make_unique<MacOSToolchain>(preprocesor);
  } else if (toolchainId == "wasm32") {
    auto wasmToolchain = std::make_unique<Wasm32WasiToolchain>(preprocesor);

    fs::path app_dir;

#if __wasi__
    app_dir = fs::path("/usr/bin/");
#elif !defined(CXX_NO_FILESYSTEM)
    app_dir = std::filesystem::canonical(
        std::filesystem::path(cli.app_name).remove_filename());
#elif __unix__ || __APPLE__
    char* app_name = realpath(cli.app_name.c_str(), nullptr);
    app_dir = fs::path(app_name).remove_filename().string();
    std::free(app_name);
#endif

    wasmToolchain->setAppdir(app_dir.string());

    if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      wasmToolchain->setSysroot(paths.back());
    } else {
      auto sysroot_dir = app_dir / "../lib/wasi-sysroot";
      wasmToolchain->setSysroot(sysroot_dir.string());
    }

    toolchain = std::move(wasmToolchain);
  } else if (toolchainId == "linux") {
    toolchain = std::make_unique<GCCLinuxToolchain>(preprocesor);
  } else if (toolchainId == "windows") {
    auto windowsToolchain = std::make_unique<WindowsToolchain>(preprocesor);

    if (auto paths = cli.get("-vctoolsdir"); !paths.empty()) {
      windowsToolchain->setVctoolsdir(paths.back());
    }

    if (auto paths = cli.get("-winsdkdir"); !paths.empty()) {
      windowsToolchain->setWinsdkdir(paths.back());
    }

    if (auto versions = cli.get("-winsdkversion"); !versions.empty()) {
      windowsToolchain->setWinsdkversion(versions.back());
    }

    toolchain = std::move(windowsToolchain);
  }

  if (toolchain) {
    if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();

    if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();

    toolchain->addPredefinedMacros();
  }

  for (const auto& path : cli.get("-I")) {
    preprocesor->addSystemIncludePath(path);
  }

  if (cli.opt_v) {
    fmt::print(std::cerr, "#include <...> search starts here:\n");
    const auto& paths = preprocesor->systemIncludePaths();
    for (auto it = rbegin(paths); it != rend(paths); ++it) {
      fmt::print(std::cerr, " {}\n", *it);
    }
    fmt::print(std::cerr, "End of search list.\n");
  }

  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of("=");

    if (sep == std::string::npos) {
      preprocesor->defineMacro(macro, "1");
    } else {
      preprocesor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }

  for (const auto& macro : cli.get("-U")) {
    preprocesor->undefMacro(macro);
  }

  auto outputs = cli.get("-o");

  auto outfile = !outputs.empty() && outputs.back() != "-"
                     ? std::optional{std::ofstream{outputs.back()}}
                     : std::nullopt;

  auto& output = outfile ? *outfile : std::cout;

  bool shouldExit = false;

  if (cli.opt_P) {
    preprocesor->setOmitLineMarkers(true);
  }

  if (cli.opt_H && (cli.opt_E || cli.opt_Eonly)) {
    preprocesor->setOnWillIncludeHeader(
        [&](const std::string& header, int level) {
          std::string fill(level, '.');
          fmt::print(stdout, "{} {}\n", fill, header);
        });
  }

  if (cli.opt_E && !cli.opt_dM) {
    preprocesor->preprocess(readAll(fileName), fileName, output);
    shouldExit = true;
  } else {
    unit.setSource(readAll(fileName), fileName);

    if (cli.opt_dM) {
      preprocesor->printMacros(output);
      shouldExit = true;
    } else if (cli.opt_dump_tokens) {
      dumpTokens(cli, unit, output);
      shouldExit = true;
    } else if (cli.opt_Eonly) {
      shouldExit = true;
    }
  }

  if (!shouldExit) {
    preprocesor->squeeze();

    unit.parse(cli.checkTypes());

    if (cli.opt_emit_ast) {
      unit.serialize(output);
    }

    if (cli.opt_ast_dump) {
      ASTPrinter printAST(&unit, std::cout);
      printAST(unit.ast());
    }
  }

  diagnosticsClient.verifyExpectedDiagnostics();

  return !diagnosticsClient.hasErrors();
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  using namespace cxx;

  CLI cli;
  cli.parse(argc, argv);

  if (cli.opt_help) {
    cli.showHelp();
    exit(0);
  }

  const auto& inputFiles = cli.positionals();

  if (inputFiles.empty()) {
    std::cerr << "cxx: no input files" << std::endl
              << "Usage: cxx [options] file..." << std::endl;
    return EXIT_FAILURE;
  }

  int existStatus = EXIT_SUCCESS;

  for (const auto& fileName : inputFiles) {
    if (!runOnFile(cli, fileName)) {
      existStatus = EXIT_FAILURE;
    }
  }

  return existStatus;
}
