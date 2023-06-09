// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/ir/codegen.h>
#include <cxx/ir/ir.h>
#include <cxx/ir/ir_printer.h>
#include <cxx/ir/x64_instruction_selection.h>
#include <cxx/lexer.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/format.h>
#include <cxx/private/path.h>
#include <cxx/recursive_ast_visitor.h>
#include <cxx/scope.h>
#include <cxx/symbol_printer.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#include "ast_printer.h"

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

namespace cxx {

struct ExpectedDiagnostic {
  Severity severity = Severity::Error;
  std::string_view fileName;
  unsigned line = 0;
  std::string message;
};

struct VerifyCommentHandler : CommentHandler {
  std::regex rx{
      R"(^//\s*expected-(error|warning)(?:@([+-]?\d+))?\s*\{\{(.+)\}\})"};
  std::list<ExpectedDiagnostic> expectedDiagnostics;

  void handleComment(Preprocessor* preprocessor, const Token& token) override {
    const std::string text{preprocessor->getTokenText(token)};

    std::smatch match;

    if (std::regex_match(text, match, rx)) {
      std::string_view fileName;
      unsigned line = 0;
      unsigned column = 0;

      preprocessor->getTokenStartPosition(token, &line, &column, &fileName);

      Severity severity = Severity::Error;

      if (match[1] == "warning") severity = Severity::Warning;

      std::string offset = match[2];

      if (!offset.empty()) line += std::stoi(offset);

      const auto& message = match[3];

      expectedDiagnostics.push_back({severity, fileName, line, message});
    }
  }
};

struct VerifyDiagnostics : DiagnosticsClient {
  std::list<Diagnostic> reportedDiagnostics;
  bool verify = false;

  [[nodiscard]] auto hasErrors() const -> bool {
    if (verify) return !reportedDiagnostics.empty();

    for (const auto& d : reportedDiagnostics) {
      if (d.severity() == Severity::Error || d.severity() == Severity::Fatal) {
        return true;
      }
    }

    return false;
  }

  void report(const Diagnostic& diagnostic) override {
    if (!verify) {
      DiagnosticsClient::report(diagnostic);
      return;
    }

    reportedDiagnostics.push_back(diagnostic);
  }

  void verifyExpectedDiagnostics(
      const std::list<ExpectedDiagnostic>& expectedDiagnostics) {
    if (!verify) return;

    for (const auto& expected : expectedDiagnostics) {
      if (auto it = findDiagnostic(expected); it != cend(reportedDiagnostics)) {
        reportedDiagnostics.erase(it);
      }
    }

    for (const auto& diag : reportedDiagnostics) {
      DiagnosticsClient::report(diag);
    }
  }

 private:
  [[nodiscard]] auto findDiagnostic(const ExpectedDiagnostic& expected) const
      -> std::list<Diagnostic>::const_iterator {
    return std::find_if(reportedDiagnostics.begin(), reportedDiagnostics.end(),
                        [&](const Diagnostic& d) {
                          if (d.severity() != expected.severity) {
                            return false;
                          }

                          unsigned line = 0;
                          unsigned column = 0;
                          std::string_view fileName;

                          preprocessor()->getTokenStartPosition(
                              d.token(), &line, &column, &fileName);

                          if (line != expected.line) return false;

                          if (fileName != expected.fileName) return false;

                          if (d.message() != expected.message) return false;

                          return true;
                        });
  }
};

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
  VerifyDiagnostics diagnosticsClient;
  TranslationUnit unit(&control, &diagnosticsClient);

  auto preprocesor = unit.preprocessor();

  std::unique_ptr<Toolchain> toolchain;

  VerifyCommentHandler verifyCommentHandler;

  diagnosticsClient.verify = cli.opt_verify;

  if (cli.opt_verify) {
    preprocesor->setCommentHandler(&verifyCommentHandler);
  }

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
#if defined(__APPLE__)
    toolchainId = "darwin";
#elif defined(__wasi__)
    toolchainId = "wasm32";
#elif defined(__linux__)
    toolchainId = "linux";
#elif defined(_MSC_VER)
    toolchainId = "windows";
#endif
  }

  if (toolchainId == "darwin") {
    toolchain = std::make_unique<MacOSToolchain>(preprocesor);
  } else if (toolchainId == "wasm32") {
    auto wasmToolchain = std::make_unique<Wasm32WasiToolchain>(preprocesor);

    if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      wasmToolchain->setSysroot(paths.back());
    } else {
#if __wasi__
      wasmToolchain->setSysroot("/wasi-sysroot");
#elif __unix__
      char* app_name = realpath(cli.app_name.c_str(), nullptr);

      const fs::path app_dir = fs::path(app_name).remove_filename();
      wasmToolchain->setAppdir(app_dir.string());

      const auto sysroot_dir = app_dir / "../lib/wasi-sysroot";
      wasmToolchain->setSysroot(sysroot_dir.string());

      if (app_name) {
        std::free(app_name);
      }
#endif
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

  if (shouldExit) {
    diagnosticsClient.verifyExpectedDiagnostics(
        verifyCommentHandler.expectedDiagnostics);

    return !diagnosticsClient.hasErrors();
  }

  preprocesor->squeeze();

  unit.parse(cli.checkTypes());

  if (cli.opt_dump_symbols) {
    SymbolPrinter printSymbol(std::cout);
    printSymbol(unit.ast()->symbol);
  }

  if (cli.opt_ast_dump) {
    ASTPrinter toJSON(&unit);
    fmt::print(std::cout, "{}",
               toJSON(unit.ast(), /*print locations*/ true).dump(2));
  }

  if (cli.opt_S || cli.opt_ir_dump || cli.opt_c) {
    ir::Codegen cg;

    auto module = cg(&unit);

    if (cli.opt_S || cli.opt_c) {
      ir::X64InstructionSelection isel;
      isel(module.get(), output);
    } else if (cli.opt_ir_dump) {
      ir::IRPrinter printer;
      printer.print(module.get(), output);
    }
  }

  diagnosticsClient.verifyExpectedDiagnostics(
      verifyCommentHandler.expectedDiagnostics);

  return !diagnosticsClient.hasErrors();
}

}  // namespace cxx

auto main(int argc, char* argv[]) -> int {
  using namespace cxx;

  CLI cli;
  cli.parse(argc, argv);

  if (cli.opt_help) {
    cli.showHelp();
    exit(0);
  }

  // for (const auto& opt : cli) {
  //   fmt::print("  {}\n", to_string(opt));
  // }

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
