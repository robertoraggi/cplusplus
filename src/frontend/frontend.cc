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
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/preprocessor.h>
#include <cxx/recursive_ast_visitor.h>
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

bool parseFile(const CLI& cli, const std::string& fileName, bool preprocessed,
               bool printAST) {
  Control control;
  TranslationUnit unit(&control);

  if (!preprocessed) {
    auto preprocesor = std::make_unique<Preprocessor>(&control);

    preprocesor->addSystemIncludePaths();
    preprocesor->addPredefinedMacros();

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

    unit.setPreprocessor(std::move(preprocesor));
  }

  unit.setFileName(fileName);
  unit.setSource(readAll(fileName));
  unit.setPreprocessed(preprocessed);

  const auto result = unit.parse();

  if (printAST) {
    ASTPrinter print(&unit);
    std::cout << std::setw(4) << print(unit.ast());
  }

  return result;
}

void dumpTokens(const std::string& fileName, bool preprocessed) {
  Control control;

  TranslationUnit unit(&control);

  unit.setPreprocessed(preprocessed);
  unit.setSource(readAll(fileName));
  unit.setFileName(std::move(fileName));
  unit.tokenize(/*preprocessing=*/true);

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

}  // namespace cxx

int main(int argc, char* argv[]) {
  using namespace cxx;

  CLI cli;
  cli.parse(argc, argv);

  if (cli.count("--help")) {
    cli.showHelp();
    exit(0);
  }

  // for (const auto& opt : cli) {
  //   fmt::print("  {}\n", to_string(opt));
  // }

  const auto& inputFiles = cli.positionals();
  const auto shouldDumpTokens = cli.count("-dump-tokens");
  const auto shouldDumpMacros = cli.count("-dM");
  const auto shouldDumpAST = cli.count("-ast-dump");
  const auto preprocessOnly = cli.count("-Eonly");
  const auto shouldPreprocess = preprocessOnly || cli.count("-E");
  const auto preprocessed = cli.count("-fpreprocessed");

  if (inputFiles.empty()) {
    std::cerr << "cxx-frontend: no input files" << std::endl
              << "Usage: cxx-frontend [options] file..." << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto& fileName : inputFiles) {
    if (shouldPreprocess) {
      Control control;
      Preprocessor preprocess(&control);

      preprocess.addSystemIncludePaths();
      preprocess.addPredefinedMacros();

      for (const auto& path : cli.get("-I")) {
        preprocess.addSystemIncludePath(path);
      }

      for (const auto& macro : cli.get("-D")) {
        auto sep = macro.find_first_of("=");
        if (sep == std::string::npos) {
          preprocess.defineMacro(macro, "1");
        } else {
          preprocess.defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
        }
      }

      const auto source = readAll(fileName);

      if (preprocessOnly || shouldDumpMacros) {
        std::vector<Token> tokens;
        preprocess.preprocess(source, fileName, tokens);
        if (shouldDumpMacros) preprocess.printMacros(std::cout);
      } else {
        std::ostringstream out;
        preprocess(source, fileName, out);
        fmt::print("{}\n", out.str());
      }
    } else if (shouldDumpTokens) {
      dumpTokens(fileName, preprocessed);
    } else {
      parseFile(cli, fileName, preprocessed, shouldDumpAST);
    }
  }

  return EXIT_SUCCESS;
}
