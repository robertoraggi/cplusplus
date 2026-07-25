// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "linker.h"

#include <cxx/cli.h>
#include <cxx/toolchain.h>

#include <array>
#include <format>
#include <iostream>
#include <optional>

#ifdef CXX_WITH_LLD
#include <lld/Common/Driver.h>
#include <llvm/Support/raw_ostream.h>

LLD_HAS_DRIVER(wasm)
#endif

namespace cxx {
auto haveEmbeddedLinker() -> bool {
#ifdef CXX_WITH_LLD
  return true;
#else
  return false;
#endif
}

#ifdef CXX_WITH_LLD

namespace {
struct LldDriver {
  lld::Flavor flavor;
  const char* name;
};

auto lldDriverFor(LinkerFlavor flavor) -> std::optional<LldDriver> {
  switch (flavor) {
    case LinkerFlavor::kWasm:
      return LldDriver{lld::Wasm, "wasm-ld"};
    default:
      return std::nullopt;
  }
}
}  // namespace

auto link(const CLI& cli, Toolchain* toolchain,
          const std::vector<std::string>& inputs, const std::string& outputPath)
    -> bool {
  auto driver = lldDriverFor(toolchain->linkerFlavor());
  if (!driver) {
    std::cerr << "cxx: embedded linker does not support the selected toolchain"
              << std::endl;
    return false;
  }

  std::vector<std::string> args;
  args.emplace_back(driver->name);

  toolchain->addLinkerStartArgs(args);

  for (const auto& dir : cli.get("-L"))
    args.push_back(std::format("-L{}", dir));

  for (const auto& input : inputs) args.push_back(input);

  for (const auto& lib : cli.get("-l"))
    args.push_back(std::format("-l{}", lib));
  for (const auto& arg : cli.linkerArgs()) args.push_back(arg);

  toolchain->addLinkerEndArgs(args);

  args.emplace_back("-o");
  args.push_back(outputPath);

  if (cli.opt_v) {
    std::cerr << "cxx: linking with";
    for (const auto& arg : args) std::cerr << ' ' << arg;
    std::cerr << std::endl;
  }

  std::vector<const char*> argv;
  argv.reserve(args.size());
  for (const auto& arg : args) argv.push_back(arg.c_str());

  const std::array<lld::DriverDef, 1> drivers = {
      lld::DriverDef{driver->flavor, &lld::wasm::link}};

  auto result = lld::lldMain(argv, llvm::outs(), llvm::errs(), drivers);
  return result.retCode == 0;
}

#else

auto link(const CLI&, Toolchain*, const std::vector<std::string>&,
          const std::string&) -> bool {
  std::cerr << "cxx: this build does not have an embedded linker (rebuild with "
               "lld)"
            << std::endl;
  return false;
}

#endif
}  // namespace cxx
