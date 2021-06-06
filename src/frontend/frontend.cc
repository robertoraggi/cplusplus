// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/ast_printer.h>
#include <cxx/ast_visitor.h>
#include <cxx/codegen.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/ir.h>
#include <cxx/ir_printer.h>
#include <cxx/lexer.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/recursive_ast_visitor.h>
#include <cxx/target/wasm/wasm_codegen.h>
#include <cxx/translation_unit.h>

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

// std
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "cli.h"

namespace cxx {

std::string readAll(const std::string& fileName, std::istream& in) {
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);
  return code;
}

std::string readAll(const std::string& fileName) {
  if (fileName == "-" || fileName.empty()) return readAll("<stdin>", std::cin);
  std::ifstream stream(fileName);
  return readAll(fileName, stream);
}

void dumpTokens(const CLI& cli, TranslationUnit& unit) {
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
    if (kind == TokenKind::T_IDENTIFIER)
      kind = Lexer::classifyKeyword(tk.spell());

    fmt::print("{} '{}'{}\n", Token::name(kind), tk.spell(), flags);

    if (tk.is(TokenKind::T_EOF_SYMBOL)) break;
  }
}

bool runOnFile(const CLI& cli, const std::string& fileName) {
  Control control;
  TranslationUnit unit(&control);

  auto preprocesor = unit.preprocessor();

  std::unique_ptr<Toolchain> toolchain;

#if defined(__APPLE__)
  toolchain = std::make_unique<MacOSToolchain>(preprocesor);
#elif defined(__linux__)
  toolchain = std::make_unique<GCCLinuxToolchain>(preprocesor);
#endif

  if (toolchain) {
    if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();

    if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();

    toolchain->addPredefinedMacros();
  }

  for (const auto& path : cli.get("-I")) {
    preprocesor->addSystemIncludePath(path);
  }

  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of("=");

    if (sep == std::string::npos) {
      preprocesor->defineMacro(macro, "1");
    } else {
      preprocesor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }

  if (cli.opt_E && !cli.opt_dM) {
    preprocesor->preprocess(readAll(fileName), fileName, std::cout);
    return true;
  }

  unit.setSource(readAll(fileName), fileName);

  if (cli.opt_dM) {
    preprocesor->printMacros(std::cout);
    return true;
  }

  if (cli.opt_dump_tokens) {
    dumpTokens(cli, unit);
    return true;
  }

  if (cli.opt_Eonly) {
    return true;
  }

  preprocesor->squeeze();

  const auto result = unit.parse(cli.checkTypes());

  if (cli.opt_ast_dump) {
    ASTPrinter print(&unit);
    std::cout << std::setw(4) << print(unit.ast());
    return result;
  }

  if (cli.opt_S || cli.opt_c || cli.opt_dump_ir) {
    Codegen cg;

    auto module = cg(&unit);

    if (cli.opt_dump_ir) {
      ir::IRPrinter printer;
      printer.print(module.get(), std::cout);
    }

    if (cli.opt_c || cli.opt_S) {
      cxx::target::wasm::Codegen compile;

      if (auto code = compile(module.get())) {
        BinaryenModulePrint(code);
        BinaryenModuleDispose(code);
      }
    }
  }

  return result;
}

}  // namespace cxx

int main(int argc, char* argv[]) {
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
    std::cerr << "cxx-frontend: no input files" << std::endl
              << "Usage: cxx-frontend [options] file..." << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto& fileName : inputFiles) {
    runOnFile(cli, fileName);
  }

  return EXIT_SUCCESS;
}
