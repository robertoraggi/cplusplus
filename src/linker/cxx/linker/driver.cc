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

#include <cxx/linker/driver.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>

namespace cxx::linker {

namespace {

[[nodiscard]] auto parseNumber(std::string_view text, std::uint32_t& out)
    -> bool {
  const std::string copy{text};
  char* end = nullptr;
  const auto value = std::strtoul(copy.c_str(), &end, 0);
  if (copy.empty() || *end != '\0') return false;
  out = static_cast<std::uint32_t>(value);
  return true;
}

struct Argument {
  std::string_view name;
  std::string_view value;
  bool hasValue = false;
};

[[nodiscard]] auto isShortOptionWithValue(char letter) -> bool {
  return letter == 'l' || letter == 'L' || letter == 'o' || letter == 'e' ||
         letter == 'z';
}

[[nodiscard]] auto splitArgument(std::string_view text) -> Argument {
  if (text.size() > 2 && isShortOptionWithValue(text[1]) && text[0] == '-' &&
      text[1] != '-') {
    return {text.substr(1, 1), text.substr(2), true};
  }
  text.remove_prefix(text.starts_with("--") ? 2 : 1);
  const auto equals = text.find('=');
  if (equals == std::string_view::npos) return {text, {}, false};
  return {text.substr(0, equals), text.substr(equals + 1), true};
}

[[nodiscard]] auto takesValue(std::string_view name) -> bool {
  return name.size() == 1
             ? isShortOptionWithValue(name[0])
             : name == "entry" || name == "export" || name == "stack-size" ||
                   name == "global-base" || name == "initial-memory" ||
                   name == "max-memory";
}

class ArgumentParser {
 public:
  ArgumentParser(std::span<const std::string> arguments, LinkCommand& command,
                 std::string& error)
      : arguments_(arguments),
        command_(command),
        options_(command.options),
        error_(error) {}

  [[nodiscard]] auto parse() -> bool {
    while (index_ < arguments_.size()) {
      if (!parseOne(arguments_[index_++])) return false;
    }
    return true;
  }

 private:
  std::span<const std::string> arguments_;
  LinkCommand& command_;
  LinkOptions& options_;
  std::string& error_;
  std::size_t index_ = 0;

  [[nodiscard]] auto parseOne(const std::string& text) -> bool {
    if (text.size() < 2 || text[0] != '-') {
      command_.inputs.push_back(text);
      return true;
    }
    auto argument = splitArgument(text);
    if (takesValue(argument.name) && !argument.hasValue) {
      if (index_ == arguments_.size()) {
        error_ = concat("missing argument to ", text);
        return false;
      }
      argument.value = arguments_[index_++];
    }
    return apply(argument.name, argument.value, text);
  }

  [[nodiscard]] auto applyNumber(std::string_view value, std::uint32_t& out,
                                 const std::string& text) -> bool {
    if (parseNumber(value, out)) return true;
    error_ = concat("invalid argument: ", text);
    return false;
  }

  [[nodiscard]] auto applyZ(std::string_view value, const std::string& text)
      -> bool {
    constexpr std::string_view kStackSize = "stack-size=";
    if (!value.starts_with(kStackSize)) return true;
    value.remove_prefix(kStackSize.size());
    return applyNumber(value, options_.stackSize, text);
  }

  [[nodiscard]] auto applyFlag(std::string_view name) -> bool {
    if (name == "no-entry") {
      options_.noEntry = true;
    } else if (name == "allow-undefined") {
      options_.allowUndefined = true;
    } else if (name == "gc-sections" || name == "no-gc-sections") {
      options_.gcSections = name == "gc-sections";
    } else if (name == "stack-first" || name == "no-stack-first") {
      options_.stackFirst = name == "stack-first";
    } else if (name == "S" || name == "strip-debug") {
      options_.stripDebug = true;
    } else if (name == "s" || name == "strip-all") {
      options_.stripAll = true;
    } else {
      return false;
    }
    return true;
  }

  [[nodiscard]] auto apply(std::string_view name, std::string_view value,
                           const std::string& text) -> bool {
    if (name == "o") {
      command_.output = value;
    } else if (name == "L") {
      command_.libraryPaths.emplace_back(value);
    } else if (name == "l") {
      command_.libraries.emplace_back(value);
    } else if (name == "e" || name == "entry") {
      options_.entry = value;
    } else if (name == "export") {
      options_.exports.emplace_back(value);
    } else {
      return applyValued(name, value, text);
    }
    return true;
  }

  [[nodiscard]] auto applyValued(std::string_view name, std::string_view value,
                                 const std::string& text) -> bool {
    if (name == "z") return applyZ(value, text);
    if (name == "stack-size")
      return applyNumber(value, options_.stackSize, text);
    if (name == "global-base") {
      return applyNumber(value, options_.globalBase, text);
    }
    if (name == "initial-memory") {
      return applyNumber(value, options_.initialMemory, text);
    }
    if (name == "max-memory") {
      return applyNumber(value, options_.maxMemory, text);
    }
    if (applyFlag(name)) return true;
    error_ = concat("unknown argument: ", text);
    return false;
  }
};

[[nodiscard]] auto readFile(const std::string& path, Bytes& out,
                            std::string& error) -> bool {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) {
    error = concat("cannot open ", path);
    return false;
  }
  std::fseek(file, 0, SEEK_END);
  out.resize(static_cast<std::size_t>(std::ftell(file)));
  std::fseek(file, 0, SEEK_SET);
  const auto count = std::fread(out.data(), 1, out.size(), file);
  std::fclose(file);
  if (count == out.size()) return true;
  error = concat("cannot read ", path);
  return false;
}

[[nodiscard]] auto writeFile(const std::string& path, const Bytes& bytes,
                             std::string& error) -> bool {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) {
    error = concat("cannot open ", path);
    return false;
  }
  const auto count = std::fwrite(bytes.data(), 1, bytes.size(), file);
  std::fclose(file);
  if (count == bytes.size()) return true;
  error = concat("cannot write ", path);
  return false;
}

[[nodiscard]] auto findLibrary(const LinkCommand& command,
                               const std::string& name, std::string& path)
    -> bool {
  for (const auto& directory : command.libraryPaths) {
    path = concat(directory, "/lib", name, ".a");
    if (std::FILE* file = std::fopen(path.c_str(), "rb")) {
      std::fclose(file);
      return true;
    }
  }
  return false;
}

[[nodiscard]] auto addInput(Linker& linker, const std::string& path,
                            std::string& error) -> bool {
  auto content = std::make_shared<Bytes>();
  if (!readFile(path, *content, error)) return false;
  if (!Archive::isArchive(*content)) {
    return linker.addObject(path, std::move(content), error);
  }
  auto archive = Archive::create(path, std::move(content), error);
  if (!archive) return false;
  linker.addArchive(std::move(archive));
  return true;
}

[[nodiscard]] auto addInputs(Linker& linker, const LinkCommand& command,
                             std::string& error) -> bool {
  for (const auto& path : command.inputs) {
    if (!addInput(linker, path, error)) return false;
  }
  for (const auto& library : command.libraries) {
    std::string path;
    if (!findLibrary(command, library, path)) {
      error = concat("unable to find library -l", library);
      return false;
    }
    if (!addInput(linker, path, error)) return false;
  }
  return true;
}

}  // namespace

auto parseArguments(std::span<const std::string> arguments,
                    LinkCommand& command, std::string& error) -> bool {
  return ArgumentParser{arguments, command, error}.parse();
}

auto link(std::span<const std::string> arguments, std::string& error) -> bool {
  LinkCommand command;
  if (!parseArguments(arguments, command, error)) return false;
  Linker linker{command.options};
  Bytes module;
  return addInputs(linker, command, error) && linker.link(module, error) &&
         writeFile(command.output, module, error);
}

}  // namespace cxx::linker
