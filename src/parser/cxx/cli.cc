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

#include <cxx/cli.h>
#include <cxx/private/path.h>

#include <algorithm>
#include <cstdio>
#include <format>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

namespace cxx {
auto to_string(const CLIMatch& match) -> std::string {
  struct Process {
    auto operator()(const CLIFlag& o) const -> std::string {
      return std::format("{}=true", std::get<0>(o));
    }

    auto operator()(const CLIOption& o) const -> std::string {
      return std::format("{}={}", std::get<0>(o), std::get<1>(o));
    }

    auto operator()(const CLIPositional& o) const -> std::string {
      return std::format("{}", std::get<0>(o));
    }

    auto operator()(const CLIForwarded& o) const -> std::string {
      return std::get<0>(o);
    }
  };
  return std::visit(Process(), match);
}

namespace {
enum class CLIOptionDescrKind {
  kFlag,
  kJoined,
  kSeparated,
  kPrefixed,
};

enum class CLIOptionVisibility {
  kDefault,
  kExperimental,
  kDeprecated,
};

struct CLIOptionDescr {
  std::string option;
  std::string arg;
  std::string help;
  CLIOptionDescrKind kind;
  bool CLI::* flag = nullptr;
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

  CLIOptionDescr(std::string option, std::string help, bool CLI::* flag,
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

    {"-F", "<dir>", "Add <dir> to the framework search path",
     CLIOptionDescrKind::kSeparated},

    {"-framework", "<name>", "Link the named framework",
     CLIOptionDescrKind::kSeparated},

    {"-U", "<macro>", "Undefine <macro>", CLIOptionDescrKind::kSeparated},

    {"-W", "<warning>", "Configure the diagnostic <warning>",
     CLIOptionDescrKind::kPrefixed},

    {"-std", "<standard>",
     "Assume that the input sources are for <standard>, one of 'c++14', "
     "'c++17', 'c++20', 'c++23', 'c++26', or 'c23'",
     CLIOptionDescrKind::kJoined},

    {"--sysroot", "<directory>",
     "Use <directory> as the root directory for headers and libraries",
     CLIOptionDescrKind::kJoined},

    {"--sysroot", "<directory>",
     "Use <directory> as the root directory for headers and libraries",
     CLIOptionDescrKind::kSeparated},

    {"-isysroot", "<directory>",
     "Set the system root directory (usually used for SDK paths)",
     CLIOptionDescrKind::kSeparated},

    {"-resource-dir", "<directory>",
     "Use <directory> as the resource directory",
     CLIOptionDescrKind::kSeparated},

    {"-print-resource-dir", "Print the resource directory pathname",
     &CLI::opt_print_resource_dir},

    {"-print-target-triple", "Print the normalized target triple",
     &CLI::opt_print_target_triple},

    {"-isystem", "<directory>", "Add directory to SYSTEM include search path",
     CLIOptionDescrKind::kSeparated},

    {"-iwithsysroot", "<directory>",
     "Add directory to SYSTEM include search path, absolute paths are "
     "relative to -isysroot",
     CLIOptionDescrKind::kSeparated},

    {"-stdlib", "<library>",
     "C++ standard library to use, 'libc++' or 'libstdc++'",
     CLIOptionDescrKind::kJoined},

    {"-stdlib++-isystem", "<directory>",
     "Use directory as the C++ standard library include path",
     CLIOptionDescrKind::kSeparated},

    {"-iquote", "<directory>", "Add directory to QUOTE include search path",
     CLIOptionDescrKind::kSeparated},

    {"-E", "Preprocess only; do not compile, assemble or link", &CLI::opt_E},

    {"-Eonly", "Just run preprocessor, no output (for timings)",
     &CLI::opt_Eonly},

    {"-P", "Omit line markers", &CLI::opt_P},

    {"-H", "Print the name of each header file used to the standard output",
     &CLI::opt_H},

    {"-dM", "Print macro definitions in -E mode instead of normal output",
     &CLI::opt_dM},

    {"-S",
     "Only run preprocess and compilation steps; device targets print the "
     "SPIR-V module in MLIR's SPIR-V dialect",
     &CLI::opt_S, CLIOptionVisibility::kExperimental},

    {"-c",
     "Compile and assemble, but do not link; device targets write the SPIR-V "
     "module, to <input>.o unless -o names the output",
     &CLI::opt_c, CLIOptionVisibility::kExperimental},

    {"-flink", "Compile the inputs and link them into an executable",
     &CLI::opt_link, CLIOptionVisibility::kExperimental},

    {"-fuse-ld", "<driver>", "Use an external compiler driver for linking",
     CLIOptionDescrKind::kJoined},

    {"-fuse-ld", "<driver>", "Use an external compiler driver for linking",
     CLIOptionDescrKind::kSeparated},

    {"-l", "<library>", "Link against the library named <library>",
     CLIOptionDescrKind::kSeparated, CLIOptionVisibility::kExperimental},

    {"-g", "Generate debug information", &CLI::opt_g,
     CLIOptionVisibility::kExperimental},

    {"-fdebug-compilation-dir", "<dir>",
     "The compilation directory recorded in the debug information",
     CLIOptionDescrKind::kSeparated, CLIOptionVisibility::kExperimental},

    {"-O", "Optimize the generated code", &CLI::opt_O},

    {"-O0", "Disable optimizations", &CLI::opt_O0},

    {"-O1", "Optimize the generated code", &CLI::opt_O1},

    {"-O2", "Optimize the generated code further", &CLI::opt_O2},

    {"-O3", "Optimize the generated code aggressively", &CLI::opt_O3},

    {"-Os", "Optimize the generated code for size", &CLI::opt_Os},

    {"-Oz", "Optimize the generated code aggressively for size", &CLI::opt_Oz},

    {"-o", "<file>", "Place output into <file>",
     CLIOptionDescrKind::kSeparated},

    {"-x", "Specify the language from the compiler driver, e.g. c, or c++",
     CLIOptionDescrKind::kSeparated},

    {"-fno-check", "Disable type checker", &CLI::opt_fno_check},

    {"-fvalidate-ast", "Validate completed template instantiations",
     &CLI::opt_fvalidate_ast},

    {"-ftime-trace", "<file>",
     "Write declaration timings in Chrome JSON format",
     CLIOptionDescrKind::kJoined},

    {"-fexperimental-scf",
     "Construct structured MLIR control flow from source contexts"},

    {"-fno-experimental-scf", "Disable experimental structured MLIR emission"},

    {"-fexperimental-cxxir",
     "Generate IR with the experimental text emitter and read it into MLIR",
     CLIOptionDescrKind::kFlag, CLIOptionVisibility::kExperimental},

    {"-fno-experimental-cxxir", "Disable experimental CXXIR lowering",
     CLIOptionDescrKind::kFlag, CLIOptionVisibility::kExperimental},

    {"-fverify-ir", "Verify MLIR and LLVM IR (the default in debug builds)"},

    {"-fno-verify-ir", "Disable MLIR and LLVM IR verification"},

    {"-mlir-pass-pipeline", "<pipeline>",
     "Run an MLIR pass pipeline before lowering to LLVM",
     CLIOptionDescrKind::kJoined},

    {"-fsyntax-only", "Check only the syntax", &CLI::opt_fsyntax_only},

    {"-freport-missing-types", "Report missing types",
     &CLI::opt_freport_missing_types},

    {"-fno-reflect", "Disable reflection", &CLI::opt_fno_reflect},

    {"-fexceptions", "Enable C++ exceptions (the default for native targets)"},

    {"-fno-exceptions", "Disable C++ exceptions"},

    {"-frtti", "Enable run-time type information (the default)"},

    {"-fno-rtti", "Disable run-time type information"},

    {"-fPIC", "Generate position-independent code"},

    {"-fpic", "Generate position-independent code (small GOT)"},

    {"-fPIE", "Generate position-independent code for executables"},

    {"-fpie", "Generate position-independent code for executables (small GOT)"},

    {"-fno-pic", "Do not generate position-independent code"},

    {"-fno-pie", "Do not generate position-independent executable code"},

    {"-fno-strict-prototypes",
     "Allow unprototyped C declarations (pre-C23 behavior)",
     &CLI::opt_fno_strict_prototypes},

    {"-emit-pch", "Emit a precompiled header for the source input",
     &CLI::opt_emit_pch, CLIOptionVisibility::kExperimental},

    {"-include-pch", "<file>", "Resume compilation from the precompiled header",
     CLIOptionDescrKind::kSeparated, CLIOptionVisibility::kExperimental},

    {"-emit-ir", "Emit the CXX dialect IR for the source inputs",
     &CLI::opt_emit_cxx_ir},

    {"-emit-mlir",
     "Emit the MLIR handed to the target backend: the LLVM dialect for host "
     "targets, the GPU dialect for device targets",
     &CLI::opt_emit_mlir},

    {"-emit-spirv", "Emit the device module as SPIR-V", &CLI::opt_emit_spirv},
    {"-emit-wgsl", "Emit the device module as WGSL", &CLI::opt_emit_wgsl},

    {"--kernel", "<name>", "Select one checked device entry point",
     CLIOptionDescrKind::kJoined},

    {"--resource-manifest", "<path>",
     "Also write the device module resource manifest",
     CLIOptionDescrKind::kJoined},

    {"-emit-llvm", "Emit the LLVM IR for the source inputs",
     &CLI::opt_emit_llvm},

    {"-ast-dump", "Build ASTs and then debug dump them", &CLI::opt_ast_dump},

    {"-ast-print", "Print the AST", &CLI::opt_ast_print},

    {"-dump-symbols", "Dump the symbol tables", &CLI::opt_dump_symbols},

    {"-dump-record-layouts", "Dump class/struct layout information",
     &CLI::opt_dump_record_layouts},

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

    {"--target", "<triple>",
     "Generate code for the given target triple, e.g. "
     "'x86_64-unknown-linux-gnu'",
     CLIOptionDescrKind::kJoined},

    {"--device-profile", "<profile>",
     "Device ABI profile: 'vulkan1.1-f32' (default for spirv32), 'none'",
     CLIOptionDescrKind::kJoined},

    {"-target", "<triple>",
     "Generate code for the given target triple, e.g. "
     "'x86_64-unknown-linux-gnu'",
     CLIOptionDescrKind::kSeparated},

    {"-toolchain", "<id>",
     "Set the toolchain to 'linux', 'darwin', 'wasm32', 'windows', 'spirv32', "
     "or 'spirv64'. Defaults "
     "to wasm32.",
     CLIOptionDescrKind::kSeparated, CLIOptionVisibility::kDeprecated},

    {"-arch", "<arch>",
     "Set the architecture to 'x86_64', 'aarch64', 'wasm32', 'spirv32', or "
     "'spirv64'. Defaults to the "
     "host architecture.",
     CLIOptionDescrKind::kSeparated},

    {"-verify", "Verify the diagnostic messages", &CLI::opt_verify},

    {"-lsp", "Start Language Server", &CLI::opt_lsp},

    {"-lsp-test", "Start Language Server in testing mode", &CLI::opt_lsp_test},

    {"-j", "<n>", "Run <n> jobs in parallel.", CLIOptionDescrKind::kSeparated},

    {"-v", "Show commands to run and use verbose output", &CLI::opt_v},

    {"-M", "Like -MD, but also implies -E and writes to stdout by default",
     &CLI::opt_M},

    {"-MD", "Write a depfile containing user and system headers", &CLI::opt_MD},

    {"-MM", "Like -MMD, but also implies -E and writes to stdout by default",
     &CLI::opt_MM},

    {"-MMD", "Write a depfile containing user headers", &CLI::opt_MMD},

    {"-MF", "<file>",
     "Write depfile output from -MMD, -MD, -MM, or -M to <file>",
     CLIOptionDescrKind::kSeparated},

    {"-MT", "<value>", "Specify name of main file output in depfile",
     CLIOptionDescrKind::kSeparated},

    {"-MQ", "<value>", "Specify name of main file output to quote in depfile",
     CLIOptionDescrKind::kSeparated},

    {"-MP", "Create phony target for each dependency (other than main file)",
     &CLI::opt_MP},

    {"-ferror-limit", "<N>",
     "Set the maximum number of errors to emit before stopping (0 = no limit)",
     CLIOptionDescrKind::kSeparated},
};

auto getSystemPaths() -> std::vector<fs::path> {
  std::vector<fs::path> paths;

  char sep = ':';

#ifdef _WIN32
  sep = ';';
#endif

  if (auto path = getenv("PATH")) {
    auto value = std::string_view(path);
    for (auto token : value | std::views::split(sep)) {
      if (token.empty() && token.end() == value.end()) continue;
      paths.emplace_back(std::string_view(token));
    }
  }

  return paths;
}

}  // namespace

CLI::CLI() = default;

void CLI::addOption(std::string option, std::string value) {
  result_.emplace_back(CLIOption(std::move(option), std::move(value)));
}

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

auto CLI::getLast(std::initializer_list<std::string_view> spellings) const
    -> std::optional<std::string> {
  std::optional<std::string> result;
  for (const auto& match : result_) {
    auto option = std::get_if<CLIOption>(&match);
    if (option && std::ranges::contains(spellings, std::get<0>(*option)))
      result = std::get<1>(*option);
  }
  return result;
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
      auto arg = std::get<0>(*p);
      if (arg.empty()) continue;
      result.push_back(std::move(arg));
    }
  }
  return result;
}

namespace {

struct PositionIndependenceFlag {
  std::string_view name;
  PositionIndependence value;
};

constexpr PositionIndependenceFlag positionIndependenceFlags[] = {
    {"-fPIC", {.picLevel = 2}},
    {"-fpic", {.picLevel = 1}},
    {"-fPIE", {.picLevel = 2, .pieLevel = 2}},
    {"-fpie", {.picLevel = 1, .pieLevel = 1}},
    {"-fno-pic", {}},
    {"-fno-pie", {}},
};

[[nodiscard]] auto findPositionIndependenceFlag(std::string_view name)
    -> const PositionIndependenceFlag* {
  auto it = std::ranges::find(positionIndependenceFlags, name,
                              &PositionIndependenceFlag::name);
  return it == std::end(positionIndependenceFlags) ? nullptr : &*it;
}

}  // namespace

auto CLI::exceptionsEnabled(bool defaults) const -> bool {
  auto flag = lastFlag([](std::string_view name) {
    return name == "-fexceptions" || name == "-fno-exceptions";
  });
  if (!flag) return defaults;
  return flag == "-fexceptions";
}

auto CLI::stopsBeforeLink() const -> bool {
  return opt_E || opt_Eonly || opt_fsyntax_only || opt_S || opt_c ||
         opt_emit_pch || opt_emit_cxx_ir || opt_emit_mlir || opt_emit_llvm ||
         opt_emit_spirv || opt_emit_wgsl;
}

auto CLI::links() const -> bool {
  if (stopsBeforeLink()) return false;
  auto output = getSingle("-o");
  const bool wantsExecutable = output.has_value() && output != "-";
  return opt_link || getSingle("-fuse-ld").has_value() || wantsExecutable;
}

auto CLI::hostOnlyOutputOption() const -> std::optional<std::string> {
  if (opt_emit_llvm) return "-emit-llvm";
  if (links()) return "linking";
  return std::nullopt;
}

auto CLI::experimentalSCFEnabled(bool defaults) const -> bool {
  auto flag = lastFlag([](std::string_view name) {
    return name == "-fexperimental-scf" || name == "-fno-experimental-scf";
  });
  if (!flag) return defaults;
  return flag == "-fexperimental-scf";
}

auto CLI::experimentalCxxIREnabled() const -> bool {
  auto flag = lastFlag([](std::string_view name) {
    return name == "-fexperimental-cxxir" || name == "-fno-experimental-cxxir";
  });
  return flag == "-fexperimental-cxxir";
}

auto CLI::rttiEnabled() const -> bool {
  auto flag = lastFlag([](std::string_view name) {
    return name == "-frtti" || name == "-fno-rtti";
  });
  return flag != "-fno-rtti";
}

auto CLI::irVerificationEnabled() const -> bool {
  auto flag = lastFlag([](std::string_view name) {
    return name == "-fverify-ir" || name == "-fno-verify-ir";
  });
  return flag == "-fverify-ir";
}

auto CLI::positionIndependence(PositionIndependence defaults) const
    -> PositionIndependence {
  auto flag = lastFlag([](std::string_view name) {
    return findPositionIndependenceFlag(name) != nullptr;
  });
  if (!flag) return defaults;
  return findPositionIndependenceFlag(*flag)->value;
}

auto CLI::optimizationLevel() const -> int {
  if (opt_O3) return 3;
  if (opt_O2) return 2;
  if (opt_Os || opt_Oz) return 2;
  if (opt_O1 || opt_O) return 1;
  return 0;
}

auto CLI::parse(int& argc, char**& argv) -> bool {
  bool valid = true;
  app_name = argv[0];

  if constexpr (fs::kHasProcessWorkingDirectory) {
    if (fs::path(app_name).remove_filename().empty()) {
      for (auto path : getSystemPaths()) {
        if (fs::exists(path / app_name)) {
          app_name = (path / app_name).string();
          break;
        }
      }
    }
    app_name = fs::file_identity(app_name).string();
  }

  for (int i = 1; i < argc;) {
    const std::string arg(argv[i++]);

    if (!arg.starts_with("-") || arg == "-") {
      result_.emplace_back(CLIPositional(arg));
      continue;
    }

    if (arg.starts_with("-Wl,")) {
      auto payload = std::string_view(arg).substr(4);
      for (auto token : payload | std::views::split(',')) {
        if (token.empty() && token.end() == payload.end()) continue;
        result_.emplace_back(
            CLIOption("-Xlinker", std::string(std::string_view(token))));
      }
      continue;
    }

    if (arg == "-Xlinker") {
      if (i < argc) {
        result_.emplace_back(CLIOption("-Xlinker", argv[i++]));
      } else {
        std::print(stderr, "missing argument after '{}'\n", arg);
        valid = false;
      }
      continue;
    }

    auto prefixed = std::find_if(
        options.begin(), options.end(), [&](const CLIOptionDescr& o) {
          return o.kind == CLIOptionDescrKind::kPrefixed &&
                 arg.starts_with(o.option);
        });

    if (prefixed != options.end()) {
      result_.emplace_back(
          CLIOption(prefixed->option, arg.substr(prefixed->option.length())));
      continue;
    }

    const auto eq = arg.find_first_of('=');

    if (eq != std::string::npos) {
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

    auto it = std::find_if(
        options.begin(), options.end(), [&](const CLIOptionDescr& o) {
          return o.kind != CLIOptionDescrKind::kJoined && o.option == arg;
        });

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

        std::print(stderr, "missing argument after '{}'\n", arg);
        valid = false;
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

    result_.emplace_back(CLIForwarded(arg));
  }
  return valid;
}

void CLI::showHelp() {
  std::print("Usage: cxx [options] file...\n");
  std::print("Options:\n");
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
          info = std::format("{} {}", opt.option, opt.arg);
        }
        break;
      }
      case CLIOptionDescrKind::kJoined: {
        info = std::format("{}={}", opt.option, opt.arg);
        break;
      }
      case CLIOptionDescrKind::kPrefixed: {
        info = std::format("{}{}", opt.option, opt.arg);
        break;
      }
      case CLIOptionDescrKind::kFlag: {
        info = std::format("{}", opt.option);
        break;
      }
    }
    auto help = opt.help;
    if (opt.visibility == CLIOptionVisibility::kDeprecated) {
      help = std::format("[deprecated, use --target=<triple>] {}", help);
    }

    std::print("  {:<28} {}\n", info, help);
  }
}
}  // namespace cxx
