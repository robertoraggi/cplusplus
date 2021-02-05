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
#include <cxx/recursive_ast_visitor.h>
#include <cxx/translation_unit.h>

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cxxopts.hpp>

// std
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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

bool parseFile(const std::string& fileName, bool printAST) {
  Control control;
  TranslationUnit unit(&control);
  unit.setFileName(fileName);
  unit.setSource(readAll(fileName));
  const auto result = unit.parse();
  if (printAST) {
    ASTPrinter print(&unit);
    std::cout << std::setw(4) << print(unit.ast());
  }
  return result;
}

void dumpTokens(const std::string& fileName) {
  const auto source = readAll(fileName);
  Lexer lexer(source);
  lexer.setPreprocessing(true);

  do {
    lexer.next();

    std::string flags;

    if (lexer.tokenStartOfLine()) {
      flags += " [start-of-line]";
    }

    if (lexer.tokenLeadingSpace()) {
      flags += " [leading-space]";
    }

    fmt::print("{} '{}'{}\n", Token::name(lexer.tokenKind()), lexer.tokenText(),
               flags);
  } while (lexer.tokenKind() != TokenKind::T_EOF_SYMBOL);
}

}  // namespace cxx

int main(int argc, char* argv[]) {
  using namespace cxx;

  cxxopts::Options options("cxx-frontend", "cxx-frontend tool\n");

  // clang-format off
  options.add_options()
      ("h,help", "Display this information")
      ("input", "Input Files", cxxopts::value<std::vector<std::string>>())
      ("dump-tokens", "Dump the tokens")
      ("dump-ast", "Dump the AST");
  // clang-format on

  options.positional_help("file...");
  options.parse_positional("input");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    fmt::print("{}\n", options.help());
    return 0;
  }

  const auto& inputFiles = result["input"].as<std::vector<std::string>>();
  const auto shouldDumpTokens = result["dump-tokens"].as<bool>();
  const auto shouldDumpAST = result["dump-ast"].as<bool>();

  if (inputFiles.empty()) {
    std::cerr << "cxx-frontend: no input files" << std::endl
              << "Usage: cxx-frontend [OPTION...] file..." << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto& fileName : inputFiles) {
    if (shouldDumpTokens)
      dumpTokens(fileName);
    else
      parseFile(fileName, shouldDumpAST);
  }

  return EXIT_SUCCESS;
}
