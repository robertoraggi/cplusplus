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

#pragma once

#include <cxx/cli.h>
#include <cxx/translation_unit.h>

#include <functional>
#include <memory>
#include <string>

#ifdef CXX_WITH_MLIR
#include <llvm/Support/raw_ostream.h>
#endif

namespace cxx {

class VerifyDiagnosticsClient;
class Toolchain;

class Frontend {
 public:
  Frontend(const CLI& cli, std::string fileName);
  ~Frontend();

  [[nodiscard]] auto operator()() -> bool;

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  [[nodiscard]] auto toolchain() const -> Toolchain*;
  [[nodiscard]] auto fileName() const -> const std::string&;

  void addAction(std::function<void()> action);

 private:
  void prepare();
  void preparePreprocessor();
  void preprocess();
  void parse();
  void showSearchPaths(std::ostream& out);
  void dumpTokens(std::ostream& out);
  void dumpSymbols(std::ostream& out);
  void serializeAst();
  void dumpAst();
  void printAstIfNeeded();
  void emitIR();
  void printPreprocessedText();
  void dumpMacros(std::ostream& out);

  void withOutputStream(const std::optional<std::string>& extension,
                        const std::function<void(std::ostream&)>& action);

#ifdef CXX_WITH_MLIR
  void withRawOutputStream(
      const std::optional<std::string>& extension,
      const std::function<void(llvm::raw_ostream&)>& action);
#endif

  [[nodiscard]] auto readAll(const std::string& fileName, std::istream& in)
      -> std::optional<std::string>;

  [[nodiscard]] auto readAll(const std::string& fileName)
      -> std::optional<std::string>;

 private:
  const CLI& cli;
  std::string fileName_;
  std::unique_ptr<TranslationUnit> unit_;
  std::unique_ptr<VerifyDiagnosticsClient> diagnosticsClient_;
  std::unique_ptr<Toolchain> toolchain_;
  std::vector<std::function<void()>> actions_;
  bool shouldExit_ = false;
  int exitStatus_ = 0;
};

}  // namespace cxx