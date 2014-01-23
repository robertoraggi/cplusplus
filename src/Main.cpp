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
#include <fstream>
#include <iostream>
#include <string>

void parse_file(const char *file, std::istream &in)
{
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);

  Control control;
  TranslationUnit unit(&control);
  unit.setSource(std::move(code));
  unit.setFileName(file);
  unit.tokenize();
  unit.parse();
}

void parse_file(const char* file) {

  if (file[0] == '-' && file[1] == '\0') {
    parse_file("<stdin>", std::cin);
  } else {
    std::fstream in(file);
    parse_file(file, in);
  }
}

int main(int argc, char* argv[]) {
  for (auto it = argv + 1; it != argv + argc; ++it)
    parse_file(*it);
  return 0;
}
