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

#include <cxx/triple.h>

#include <cctype>
#include <cstdio>
#include <vector>

namespace cxx {
namespace {

auto splitComponents(std::string_view text) -> std::vector<std::string> {
  std::vector<std::string> components;
  while (true) {
    auto sep = text.find('-');
    if (sep == std::string_view::npos) {
      components.emplace_back(text);
      break;
    }
    components.emplace_back(text.substr(0, sep));
    text.remove_prefix(sep + 1);
  }
  return components;
}

auto stripVersion(std::string_view name) -> std::string_view {
  auto end = name.size();
  while (end > 0) {
    auto ch = static_cast<unsigned char>(name[end - 1]);
    if (!std::isdigit(ch) && ch != '.') break;
    --end;
  }
  return name.substr(0, end);
}

auto parseArch(std::string_view name) -> TripleArch {
  if (name == "aarch64" || name == "arm64") return TripleArch::kAArch64;
  if (name == "x86_64" || name == "amd64") return TripleArch::kX86_64;
  if (name == "wasm32") return TripleArch::kWasm32;
  return TripleArch::kUnknown;
}

auto parseVendor(std::string_view name) -> TripleVendor {
  if (name == "apple") return TripleVendor::kApple;
  if (name == "pc") return TripleVendor::kPC;
  return TripleVendor::kUnknown;
}

auto parseOS(std::string_view name) -> TripleOS {
  if (name == "wasi") return TripleOS::kWasi;
  if (name == "wasip1") return TripleOS::kWasiPreview1;
  if (name == "wasip2") return TripleOS::kWasiPreview2;
  if (name == "emscripten") return TripleOS::kEmscripten;

  auto base = stripVersion(name);
  if (base == "darwin") return TripleOS::kDarwin;
  if (base == "macosx" || base == "macos") return TripleOS::kMacOSX;
  if (base == "ios") return TripleOS::kIOS;
  if (base == "linux") return TripleOS::kLinux;
  if (base == "windows" || base == "win32") return TripleOS::kWindows;
  return TripleOS::kUnknown;
}

auto parseEnvironment(std::string_view name) -> TripleEnvironment {
  if (name == "threads") return TripleEnvironment::kThreads;
  if (name == "msvc") return TripleEnvironment::kMSVC;
  if (name == "mingw32") return TripleEnvironment::kMinGW;

  auto base = stripVersion(name);
  if (base == "gnu" || base == "gnueabi" || base == "gnueabihf") {
    return TripleEnvironment::kGNU;
  }
  return TripleEnvironment::kUnknown;
}

}  // namespace

Triple::Triple(std::string_view text) {
  const auto components = splitComponents(text);

  printedComponents_ = static_cast<int>(components.size());
  if (printedComponents_ > kComponentCount) {
    printedComponents_ = kComponentCount;
  }

  int slot = 0;

  for (std::size_t index = 0; index != components.size(); ++index) {
    const auto& component = components[index];

    int found = -1;
    if (slot <= 0 && parseArch(component) != TripleArch::kUnknown) {
      found = 0;
    } else if (slot <= 1 && parseVendor(component) != TripleVendor::kUnknown) {
      found = 1;
    } else if (slot <= 2 && parseOS(component) != TripleOS::kUnknown) {
      found = 2;
    } else if (slot <= 3 &&
               parseEnvironment(component) != TripleEnvironment::kUnknown) {
      found = 3;
    }

    if (found < 0) {
      if (slot >= kComponentCount) continue;
      found = slot;
    }

    names_[found] = component;
    slot = found + 1;

    if (found + 1 > printedComponents_) printedComponents_ = found + 1;
  }

  arch_ = parseArch(names_[0]);
  vendor_ = parseVendor(names_[1]);
  os_ = parseOS(names_[2]);
  environment_ = parseEnvironment(names_[3]);
}

auto Triple::isDarwin() const -> bool {
  return os_ == TripleOS::kDarwin || os_ == TripleOS::kMacOSX ||
         os_ == TripleOS::kIOS;
}

auto Triple::isWebAssembly() const -> bool {
  if (arch_ == TripleArch::kWasm32) return true;

  switch (os_) {
    case TripleOS::kWasi:
    case TripleOS::kWasiPreview1:
    case TripleOS::kWasiPreview2:
    case TripleOS::kEmscripten:
      return true;
    default:
      return false;
  }
}

auto Triple::osVersion() const -> std::optional<std::pair<int, int>> {
  const auto& name = names_[2];
  auto base = stripVersion(name);
  if (base.size() == name.size()) return std::nullopt;

  int major = 0;
  int minor = 0;
  if (std::sscanf(name.c_str() + base.size(), "%d.%d", &major, &minor) < 1) {
    return std::nullopt;
  }

  return std::pair{major, minor};
}

auto Triple::str() const -> std::string {
  std::string result;
  for (int index = 0; index != printedComponents_; ++index) {
    if (index) result += '-';
    if (names_[index].empty()) {
      result += "unknown";
    } else {
      result += names_[index];
    }
  }
  return result;
}

auto to_string(TripleArch arch) -> std::string_view {
  switch (arch) {
    case TripleArch::kAArch64:
      return "aarch64";
    case TripleArch::kX86_64:
      return "x86_64";
    case TripleArch::kWasm32:
      return "wasm32";
    case TripleArch::kUnknown:
      return "unknown";
  }
  return "unknown";
}

}  // namespace cxx
