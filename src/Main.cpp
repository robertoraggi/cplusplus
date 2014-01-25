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
#include <fstream>
#include <iostream>
#include <string>

std::string readAll(const std::string& fileName, std::istream& in)
{
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

bool parseFile(const std::string& fileName, const std::function<void (TranslationUnit*, TranslationUnitAST*)>& consume) {
  Control control;
  TranslationUnit unit(&control);
  unit.setFileName(fileName);
  unit.setSource(readAll(fileName));
  return unit.parse([&unit, consume](TranslationUnitAST* ast) {
    consume(&unit, ast);
  });
}

int main(int argc, char* argv[]) {
  std::vector<std::string> inputFiles;
  bool astDump = false;

  int index = 1;
  while (index < argc) {
    std::string arg{argv[index++]};
    if (arg == "-ast-dump" || arg == "--ast-dump") {
      astDump = true;
    } else {
      inputFiles.push_back(std::move(arg));
    }
  }

  if (inputFiles.empty()) {
    std::cerr << "cplusplus: no input files" << std::endl;
    return EXIT_FAILURE;
  }

  for (auto&& fileName: inputFiles) {
    parseFile(fileName, [astDump](TranslationUnit* unit, TranslationUnitAST* ast) {
      if (astDump) {
        RecursiveASTVisitor dump{unit};
        dump(ast);
      }
    });
  }

  return EXIT_SUCCESS;
}
