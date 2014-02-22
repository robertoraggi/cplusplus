// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Control.h"
#include "TranslationUnit.h"
#include "ASTVisitor.h"
#include "AST.h"
#include "Symbols.h"
#include "Codegen.h"
#include "IR.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>

class DumpIR: protected RecursiveASTVisitor {
public:
  DumpIR(TranslationUnit* unit)
    : RecursiveASTVisitor(unit) {}

  void operator()(AST* ast) { accept(ast); }

protected:
  void visit(FunctionDefinitionAST* ast) {
    auto fun = ast->symbol;
    printf("%s {\n", typeToString(fun->type(), fun->name()).c_str());
    if (auto code = fun->code())
      code->dump(std::cout);
    printf("}\n\n");
  }
private:
  TypeToString typeToString;
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
  if (fileName == "-" || fileName.empty())
    return readAll("<stdin>", std::cin);
  std::ifstream stream(fileName);
  return readAll(fileName, stream);
}

bool parseFile(const std::string& fileName,
               bool resolveSymbols,
               const std::function<void (TranslationUnit*,
                                         TranslationUnitAST*)>& consume) {
  Control control;
  TranslationUnit unit(&control);
  unit.setFileName(fileName);
  unit.setSource(readAll(fileName));
  unit.setResolveSymbols(resolveSymbols);
  unit.setFatalErrors(true);
  return unit.parse([&unit, consume](TranslationUnitAST* ast) {
    consume(&unit, ast);
  });
}

int main(int argc, char* argv[]) {
  std::vector<std::string> inputFiles;
  bool dumpAST{false};
  bool dumpSymbols{false};
  bool dumpIR{false};
  bool resolveSymbols{false};

  int index = 1;
  while (index < argc) {
    std::string arg{argv[index++]};
    if (arg == "--help") {
      std::cerr << "Usage: cplusplus [options] files..." << std::endl
                << " The options are:" << std::endl
                << "  --ast             dump the AST" << std::endl
                << "  --ir              dump the IR code" << std::endl
                << "  --symbols         dump the symbols" << std::endl
                << "  --lookup          resolve symbols" << std::endl
                << "  --help            display this output" << std::endl;
      exit(EXIT_SUCCESS);
    } else if (arg == "--ast") {
      dumpAST = true;
    } else if (arg == "--symbols") {
      dumpSymbols = true;
    } else if (arg == "--ir") {
      dumpIR = true;
    } else if (arg == "--lookup") {
      resolveSymbols = true;
    } else {
      inputFiles.push_back(std::move(arg));
    }
  }

  if (inputFiles.empty()) {
    std::cerr << "cplusplus: no input files" << std::endl
              << "Usage: cplusplus [options] files..." << std::endl;
    return EXIT_FAILURE;
  }

  for (auto&& fileName: inputFiles) {
    parseFile(fileName, resolveSymbols,
              [=](TranslationUnit* unit, TranslationUnitAST* ast) {
      if (dumpAST) {
        DumpAST dump{unit};
        dump(ast);
      }
      if (dumpSymbols && ast) {
        assert(ast->globalScope);
        ast->globalScope->dump(std::cout, 0);
      }
      if (dumpIR && ast) {
        DumpIR dump{unit};
        dump(ast);
      }
    });
  }

  return EXIT_SUCCESS;
}
