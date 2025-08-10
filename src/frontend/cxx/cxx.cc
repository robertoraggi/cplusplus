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

#include <cxx/lsp/lsp_server.h>

#include <iostream>

#include "frontend.h"

auto main(int argc, char* argv[]) -> int {
  cxx::CLI cli;
  cli.parse(argc, argv);

  if (cli.opt_help) {
    cli.showHelp();
    return EXIT_SUCCESS;
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

  if (cli.opt_lsp) {
    auto server = cxx::lsp::Server{cli};

    return server.start();
  }

  auto existStatus = EXIT_SUCCESS;

  for (const auto& fileName : inputFiles) {
    cxx::Frontend runOnFile(cli, fileName);

    if (!runOnFile()) {
      existStatus = EXIT_FAILURE;
    }
  }

  return existStatus;
}
