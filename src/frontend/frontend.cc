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

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/recursive_ast_visitor.h>
#include <cxx/translation_unit.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#if defined(__has_include)
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define WITH_CXXABI
#endif
#endif

namespace cxx {

class DumpAST final : RecursiveASTVisitor {
  TranslationUnit* unit_;
  int depth_ = 0;

  bool preVisit(AST* ast) override {
    std::string ind(depth_ * 2, ' ');

    std::string_view name = typeid(*ast).name();

#ifdef WITH_CXXABI
    int status = 0;
    auto mangled = abi::__cxa_demangle(name.data(), nullptr, 0, &status);
    name = mangled;
#endif

#if 0
    if (ast->firstSourceLocation())
      unit_->report(ast->firstSourceLocation(), Severity::Message, "{}{}", ind,
                    name);
#else
    fmt::print("{}{}\n", ind, name);
#endif

#ifdef WITH_CXXABI
    std::free(mangled);
#endif

    ++depth_;

    return true;
  }

  void postVisit(AST*) override { --depth_; }

 public:
  explicit DumpAST(TranslationUnit* unit) : unit_(unit) {}

  void operator()() { accept(unit_->ast()); }
};

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
    DumpAST dump(&unit);
    dump();
  }
  return result;
}

}  // namespace cxx

int main(int argc, char* argv[]) {
  using namespace cxx;

  std::vector<std::string> inputFiles;
  bool dumpTokens = false;
  bool dumpAST = false;

  int index = 1;
  while (index < argc) {
    std::string arg{argv[index++]};
    if (arg == "--help") {
      std::cerr << "Usage: cplusplus [options] files..." << std::endl
                << " The options are:" << std::endl
                << "  --help            display this output" << std::endl
                << "  -dump-tokens      dump tokens" << std::endl
                << "  -dump-ast         dump AST" << std::endl;
      exit(EXIT_SUCCESS);
    } else if (arg == "--dump-tokens") {
      dumpTokens = true;
    } else if (arg == "--dump-ast") {
      dumpAST = true;
    } else {
      inputFiles.push_back(std::move(arg));
    }
  }

  if (inputFiles.empty()) {
    std::cerr << "cplusplus: no input files" << std::endl
              << "Usage: cplusplus [options] files..." << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto& fileName : inputFiles) {
    if (dumpTokens) {
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

        fmt::print("{} '{}'{}\n", Token::name(lexer.tokenKind()),
                   lexer.tokenText(), flags);
      } while (lexer.tokenKind() != TokenKind::T_EOF_SYMBOL);
      continue;
    }

    parseFile(fileName, dumpAST);
  }

  return EXIT_SUCCESS;
}
