// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/ast_pretty_printer.h>
#include <cxx/ast_printer.h>
#include <cxx/ast_visitor.h>
#include <cxx/cli.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/lexer.h>
#include <cxx/lsp/lsp_server.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#ifdef CXX_WITH_MLIR
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#endif

#include <format>
#include <fstream>
#include <iostream>
#include <list>
#include <regex>
#include <string>

#include "verify_diagnostics_client.h"

namespace {

using namespace cxx;

class CheckExpressionTypes final : private ASTVisitor {
 public:
  [[nodiscard]] auto operator()(TranslationUnit* unit) {
    std::size_t missingTypes = 0;
    std::swap(unit_, unit);
    std::swap(missingTypes_, missingTypes);

    accept(unit_->ast());

    std::swap(unit_, unit);
    std::swap(missingTypes_, missingTypes);

    return missingTypes == 0;
  }

 private:
  using ASTVisitor::visit;

  auto preVisit(AST* ast) -> bool override {
    if (ast_cast<TemplateDeclarationAST>(ast)) {
      // skip template declarations, as they are not instantiated yet
      return false;
    }

    if (auto expression = ast_cast<ExpressionAST>(ast)) {
      if (!expression->type) {
        const auto loc = expression->firstSourceLocation();

        unit_->warning(loc, std::format("untyped expression of kind '{}'",
                                        to_string(expression->kind())));

        ++missingTypes_;
        return false;
      }
    }

    return true;  // visit children
  }

 private:
  TranslationUnit* unit_ = nullptr;
  std::size_t missingTypes_ = 0;
};

auto readAll(const std::string& fileName, std::istream& in)
    -> std::optional<std::string> {
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);
  return code;
}

auto readAll(const std::string& fileName) -> std::optional<std::string> {
  if (fileName == "-" || fileName.empty()) return readAll("<stdin>", std::cin);
  if (std::ifstream stream(fileName); stream) return readAll(fileName, stream);
  return std::nullopt;
}

void dumpTokens(const CLI& cli, TranslationUnit& unit, std::ostream& output) {
  auto lang = LanguageKind::kCXX;

  if (auto x = cli.getSingle("x")) {
    if (x == "c") lang = LanguageKind::kC;
  } else if (unit.fileName().ends_with(".c")) {
    lang = LanguageKind::kC;
  }

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
      kind = Lexer::classifyKeyword(tk.spell(), lang);
    }

    output << std::format("{} '{}'{}", Token::name(kind), tk.spell(), flags);

    auto pos = unit.tokenStartPosition(loc);

    output << std::format(" at {}:{}:{}\n", pos.fileName, pos.line, pos.column);

    if (tk.is(TokenKind::T_EOF_SYMBOL)) break;
  }
}

auto runOnFile(const CLI& cli, const std::string& fileName) -> bool {
  VerifyDiagnosticsClient diagnosticsClient;
  TranslationUnit unit(&diagnosticsClient);

  auto preprocessor = unit.preprocessor();

  const auto lang = cli.getSingle("-x");

  if (lang == "c" || (!lang.has_value() && fileName.ends_with(".c"))) {
    // set the language to C
    preprocessor->setLanguage(LanguageKind::kC);
  }

  std::unique_ptr<Toolchain> toolchain;

  if (cli.opt_verify) {
    diagnosticsClient.setVerify(true);
    preprocessor->setCommentHandler(&diagnosticsClient);
  }

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
    toolchainId = "wasm32";
  }

  if (toolchainId == "darwin" || toolchainId == "macos") {
    auto macosToolchain = std::make_unique<MacOSToolchain>(preprocessor);
    std::string host;
#ifdef __aarch64__
    host = "aarch64";
#elif __x86_64__
    host = "x86_64";
#endif
    macosToolchain->setArch(cli.getSingle("-arch").value_or(host));
    toolchain = std::move(macosToolchain);

  } else if (toolchainId == "wasm32") {
    auto wasmToolchain = std::make_unique<Wasm32WasiToolchain>(preprocessor);

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
      auto sysroot_dir = app_dir / std::string("../lib/wasi-sysroot");
      wasmToolchain->setSysroot(sysroot_dir.string());
    }

    toolchain = std::move(wasmToolchain);
  } else if (toolchainId == "linux") {
    std::string host;
#ifdef __aarch64__
    host = "aarch64";
#elif __x86_64__
    host = "x86_64";
#endif

    std::string arch = cli.getSingle("-arch").value_or(host);
    toolchain = std::make_unique<GCCLinuxToolchain>(preprocessor, arch);
  } else if (toolchainId == "windows") {
    auto windowsToolchain = std::make_unique<WindowsToolchain>(preprocessor);

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
    unit.control()->setMemoryLayout(toolchain->memoryLayout());

    if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();

    if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();

    toolchain->addPredefinedMacros();
  }

  for (const auto& path : cli.get("-I")) {
    preprocessor->addSystemIncludePath(path);
  }

  if (cli.opt_v) {
    std::cerr << std::format("#include <...> search starts here:\n");
    const auto& paths = preprocessor->systemIncludePaths();
    for (auto it = rbegin(paths); it != rend(paths); ++it) {
      std::cerr << std::format(" {}\n", *it);
    }
    std::cerr << std::format("End of search list.\n");
  }

  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of("=");

    if (sep == std::string::npos) {
      preprocessor->defineMacro(macro, "1");
    } else {
      preprocessor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }

  for (const auto& macro : cli.get("-U")) {
    preprocessor->undefMacro(macro);
  }

  auto outputs = cli.get("-o");

  auto outfile = !outputs.empty() && outputs.back() != "-"
                     ? std::optional{std::ofstream{outputs.back()}}
                     : std::nullopt;

  auto& output = outfile ? *outfile : std::cout;

  bool shouldExit = false;

  if (cli.opt_P) {
    preprocessor->setOmitLineMarkers(true);
  }

  if (cli.opt_H && (cli.opt_E || cli.opt_Eonly)) {
    preprocessor->setOnWillIncludeHeader(
        [&](const std::string& header, int level) {
          std::string fill(level, '.');
          std::cout << std::format("{} {}\n", fill, header);
        });
  }

  if (auto source = readAll(fileName)) {
    if (cli.opt_E && !cli.opt_dM) {
      std::vector<Token> tokens;
      preprocessor->preprocess(std::move(*source), fileName, tokens);
      preprocessor->getPreprocessedText(tokens, output);
      shouldExit = true;
    } else {
      unit.setSource(std::move(*source), fileName);
      if (cli.opt_dM) {
        preprocessor->printMacros(output);
        shouldExit = true;
      } else if (cli.opt_dump_tokens) {
        dumpTokens(cli, unit, output);
        shouldExit = true;
      } else if (cli.opt_Eonly) {
        shouldExit = true;
      }
    }
  } else {
    std::cerr << std::format("cxx: No such file or directory: '{}'\n",
                             fileName);
    return false;
  }

  if (!shouldExit) {
    unit.parse(ParserConfiguration{
        .checkTypes = cli.opt_fcheck,
        .fuzzyTemplateResolution = true,
        .reflect = !cli.opt_fno_reflect,
    });

    if (cli.opt_freport_missing_types) {
      CheckExpressionTypes checkExpressionTypes;
      const auto missingTypes = checkExpressionTypes(&unit);
    }

    if (cli.opt_dump_symbols && unit.globalScope()) {
      dump(std::cout, unit.globalScope()->owner());
    }

    if (cli.opt_emit_ast) {
      (void)unit.serialize(output);
    }

    if (cli.opt_ast_dump) {
      ASTPrinter printAST(&unit, std::cout);
      printAST(unit.ast());
    }

    if (cli.opt_ast_print) {
      ASTPrettyPrinter prettyPrinter(&unit, std::cout);
      prettyPrinter(unit.ast());
    }

#ifdef CXX_WITH_MLIR
    if (cli.opt_emit_ir) {
      mlir::MLIRContext context;
      context.loadDialect<mlir::cxx::CxxDialect>();

      cxx::Codegen codegen(context, &unit);

      auto ir = codegen(unit.ast());

      mlir::OpPrintingFlags flags;
      if (cli.opt_g) {
        flags.enableDebugInfo(true, true);
      }
      ir.module->print(llvm::outs(), flags);
    }
#endif
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

  if (cli.opt_lsp_test) {
    cli.opt_lsp = true;
  }

  if (!cli.opt_lsp && inputFiles.empty()) {
    std::cerr << "cxx: no input files" << std::endl
              << "Usage: cxx [options] file..." << std::endl;
    return EXIT_FAILURE;
  }

  int existStatus = EXIT_SUCCESS;

  if (cli.opt_lsp) {
    lsp::Server server(cli);
    existStatus = server.start();
  } else {
    for (const auto& fileName : inputFiles) {
      if (!runOnFile(cli, fileName)) {
        existStatus = EXIT_FAILURE;
      }
    }
  }

  return existStatus;
}
