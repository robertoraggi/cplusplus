// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "cli.h"

#include <cxx/private/format.h>
#include <cxx/private/path.h>

#include <array>
#include <iostream>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace cxx {

auto to_string(const CLIMatch& match) -> std::string {
  struct Process {
    auto operator()(const CLIFlag& o) const -> std::string {
      return cxx::format("{}=true", std::get<0>(o));
    }

    auto operator()(const CLIOption& o) const -> std::string {
      return cxx::format("{}={}", std::get<0>(o), std::get<1>(o));
    }

    auto operator()(const CLIPositional& o) const -> std::string {
      return cxx::format("{}", std::get<0>(o));
    }
  };
  return std::visit(Process(), match);
}

namespace {

enum struct CLIOptionDescrKind {
  kFlag,
  kJoined,
  kSeparated,
};

enum struct CLIOptionVisibility : bool {
  kDefault,
  kExperimental,
};

struct CLIOptionDescr {
  std::string option;
  std::string arg;
  std::string help;
  CLIOptionDescrKind kind;
  bool CLI::*flag = nullptr;
  CLIOptionVisibility visibility{CLIOptionVisibility::kDefault};

  CLIOptionDescr(std::string option, std::string arg, std::string help,
                 CLIOptionDescrKind kind,
                 CLIOptionVisibility visibility = CLIOptionVisibility::kDefault)
      : option(std::move(option)),
        arg(std::move(arg)),
        help(std::move(help)),
        kind(kind),
        visibility(visibility) {}

  CLIOptionDescr(std::string option, std::string help,
                 CLIOptionDescrKind kind = CLIOptionDescrKind::kFlag,
                 CLIOptionVisibility visibility = CLIOptionVisibility::kDefault)
      : option(std::move(option)),
        help(std::move(help)),
        kind(kind),
        visibility(visibility) {}

  CLIOptionDescr(std::string option, std::string help, bool CLI::*flag,
                 CLIOptionVisibility visibility = CLIOptionVisibility::kDefault)
      : option(std::move(option)),
        help(std::move(help)),
        kind(CLIOptionDescrKind::kFlag),
        flag(flag),
        visibility(visibility) {}
};

std::vector<CLIOptionDescr> options{
    {"--help", "Display this information", &CLI::opt_help},

    {"-D", "<macro>[=<val>]",
     "Define a <macro> with <val> as its value. If just <macro> is given, "
     "<val> is taken to be 1",
     CLIOptionDescrKind::kSeparated},

    {"-I", "<dir>", "Add <dir> to the end of the main include path",
     CLIOptionDescrKind::kSeparated},

    {"-L", "<dir>", "Add <dir> to the end of the library path",
     CLIOptionDescrKind::kSeparated},

    {"-U", "<macro>", "Undefine <macro>", CLIOptionDescrKind::kSeparated},

    {"-std", "<standard>", "Assume that the input sources are for <standard>",
     CLIOptionDescrKind::kJoined},

    {"--sysroot", "<directory>",
     "Use <directory> as the root directory for headers and libraries",
     CLIOptionDescrKind::kJoined},

    {"-E", "Preprocess only; do not compile, assemble or link", &CLI::opt_E},

    {"-Eonly", "Just run preprocessor, no output (for timings)",
     &CLI::opt_Eonly},

    {"-P", "Omit line markers", &CLI::opt_P},

    {"-H", "Print the name of each header file used to the standard output",
     &CLI::opt_H},

    {"-dM", "Print macro definitions in -E mode instead of normal output",
     &CLI::opt_dM},

    {"-S", "Only run preprocess and compilation steps", &CLI::opt_S,
     CLIOptionVisibility::kExperimental},

    {"-c", "Compile and assemble, but do not link", &CLI::opt_c,
     CLIOptionVisibility::kExperimental},

    {"-o", "<file>", "Place output into <file>",
     CLIOptionDescrKind::kSeparated},

    {"-x", "Specify the language from the compiler driver",
     CLIOptionDescrKind::kSeparated},

    {"-fcheck", "Enable type checker (WIP)", &CLI::opt_fcheck},

    {"-fsyntax-only", "Check only the syntax", &CLI::opt_fsyntax_only},

    {"-fstatic-assert", "Enable static asserts", &CLI::opt_fstatic_assert,
     CLIOptionVisibility::kExperimental},

    {"-freflect", "Enable reflection", &CLI::opt_freflect},

    {"-emit-ast", "Emit AST files for source inputs", &CLI::opt_emit_ast},

    {"-ast-dump", "Build ASTs and then debug dump them", &CLI::opt_ast_dump},

    {"-ir-dump", "Dump the IR", &CLI::opt_ir_dump},

    {"-dump-symbols", "Dump the symbol tables", &CLI::opt_dump_symbols},

    {"-dump-tokens", "Run preprocessor, dump internal rep of tokens",
     &CLI::opt_dump_tokens},

    {"-nostdinc",
     "Disable standard #include directories for the C standard library",
     &CLI::opt_nostdinc},

    {"-nostdinc++",
     "Disable standard #include directories for the C++ standard library",
     &CLI::opt_nostdincpp},

    {"-winsdkdir", "<dir>", "Path to the the Windows SDK",
     CLIOptionDescrKind::kSeparated},

    {"-vctoolsdir", "<dir>", "Path to the Visual Studio Tools",
     CLIOptionDescrKind::kSeparated},

    {"-winsdkversion", "<version>", "Version of the Windows SDK",
     CLIOptionDescrKind::kSeparated},

    {"-toolchain", "<id>",
     "Set the toolchain to 'linux', 'darwin', 'wasm32', or 'windows'. Defaults "
     "to wasm32.",
     CLIOptionDescrKind::kSeparated},

    {"-arch", "<arch>",
     "Set the architecture to 'x86_64', 'aarch64', 'wasm32'. Defaults to the "
     "host architecture.",
     CLIOptionDescrKind::kSeparated},

    {"-verify", "Verify the diagnostic messages", &CLI::opt_verify},

    {"-v", "Show commands to run and use verbose output", &CLI::opt_v},

};

#ifndef CXX_NO_FILESYSTEM

/**
 * Retuns the system path found in the PATH environment variable.
 */
auto getSystemPaths() -> std::vector<fs::path> {
  std::vector<fs::path> paths;

  char sep = ':';

  // if on windows use ';' as separator
#ifdef _WIN32
  sep = ';';
#endif

  if (auto path = getenv("PATH")) {
    std::istringstream iss(path);
    std::string token;

    while (std::getline(iss, token, sep)) {
      paths.push_back(fs::path(token));
    }
  }

  return paths;
}

#endif

}  // namespace

CLI::CLI() = default;

auto CLI::count(const std::string& flag) const -> int {
  int n = 0;
  for (const auto& match : result_) {
    if (auto opt = std::get_if<CLIFlag>(&match)) {
      if (std::get<0>(*opt) == flag) ++n;
    }
  }
  return n;
}

auto CLI::getSingle(const std::string& opt) const
    -> std::optional<std::string> {
  auto value = get(opt);
  return !value.empty() ? std::optional{value.back()} : std::nullopt;
}

auto CLI::get(const std::string& opt) const -> std::vector<std::string> {
  std::vector<std::string> result;
  for (const auto& match : result_) {
    if (auto p = std::get_if<CLIOption>(&match)) {
      if (std::get<0>(*p) == opt) result.push_back(std::get<1>(*p));
    }
  }
  return result;
}

auto CLI::positionals() const -> std::vector<std::string> {
  std::vector<std::string> result;
  for (const auto& match : result_) {
    if (auto p = std::get_if<CLIPositional>(&match)) {
      result.push_back(std::get<0>(*p));
    }
  }
  return result;
}

void CLI::parse(int& argc, char**& argv) {
  app_name = argv[0];

#ifndef CXX_NO_FILESYSTEM
  if (fs::path(app_name).remove_filename().empty()) {
    for (auto path : getSystemPaths()) {
      if (fs::exists(path / app_name)) {
        app_name = (path / app_name).string();
        break;
      }
    }
  }

  while (fs::is_symlink(app_name)) {
    app_name = fs::read_symlink(app_name).string();
  }
#endif

  for (int i = 1; i < argc;) {
    const std::string arg(argv[i++]);

    if (!arg.starts_with("-") || arg == "-") {
      result_.emplace_back(CLIPositional(arg));
      continue;
    }

    const auto eq = arg.find_first_of('=');

    if (eq) {
      const auto name = arg.substr(0, eq);
      const auto value = arg.substr(eq + 1);

      auto it = std::find_if(
          options.begin(), options.end(), [&](const CLIOptionDescr& o) {
            return o.kind == CLIOptionDescrKind::kJoined && o.option == name;
          });

      if (it != options.end()) {
        result_.emplace_back(CLIOption(name, value));
        continue;
      }
    }

    auto it =
        std::find_if(options.begin(), options.end(),
                     [&](const CLIOptionDescr& o) { return o.option == arg; });

    if (it != options.end()) {
      if (it->kind == CLIOptionDescrKind::kFlag) {
        if (auto flag = it->flag) this->*flag = true;
        result_.emplace_back(CLIFlag(arg));
        continue;
      }

      if (it->kind == CLIOptionDescrKind::kSeparated) {
        if (i < argc) {
          result_.emplace_back(CLIOption(arg, argv[i++]));
          continue;
        }

        std::cerr << cxx::format("missing argument after '{}'\n", arg);
        continue;
      }
    }

    it = std::find_if(
        options.begin(), options.end(), [&](const CLIOptionDescr& o) {
          return o.kind == CLIOptionDescrKind::kSeparated &&
                 o.option.length() == 2 && arg.starts_with(o.option);
        });

    if (it != options.end()) {
      result_.emplace_back(CLIOption(it->option, arg.substr(2)));
      continue;
    }

    std::cerr << cxx::format("unsupported option '{}'\n", arg);
  }
}

void CLI::showHelp() {
  std::cerr << cxx::format("Usage: cxx [options] file...\n");
  std::cerr << cxx::format("Options:\n");
  for (const auto& opt : options) {
    if (opt.visibility == CLIOptionVisibility::kExperimental) {
      continue;
    }

    std::string info;
    switch (opt.kind) {
      case CLIOptionDescrKind::kSeparated: {
        if (opt.arg.empty()) {
          info = opt.option;
        } else {
          info = cxx::format("{} {}", opt.option, opt.arg);
        }
        break;
      }
      case CLIOptionDescrKind::kJoined: {
        info = cxx::format("{}={}", opt.option, opt.arg);
        break;
      }
      case CLIOptionDescrKind::kFlag: {
        info = cxx::format("{}", opt.option);
        break;
      }
    }  // switch
    std::cerr << cxx::format("  {:<28} {}\n", info, opt.help);
  }
}

}  // namespace cxx
