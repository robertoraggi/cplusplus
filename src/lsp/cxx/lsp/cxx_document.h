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

#pragma once

#include <cxx/lsp/fwd.h>
#include <cxx/parser_fwd.h>
#include <cxx/translation_unit.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace cxx {
class Toolchain;
}

namespace cxx::lsp {

struct CompletionEditRange {
  std::uint32_t line = 0;
  std::uint32_t startColumn = 0;
  std::uint32_t endColumn = 0;
};

class CxxDocument {
 public:
  CxxDocument(std::string fileName, long version);
  ~CxxDocument();

  [[nodiscard]] auto fileName() const -> const std::string&;
  [[nodiscard]] auto version() const -> long;

  [[nodiscard]] auto isCancelled() const -> bool;
  void cancel();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  [[nodiscard]] auto parserConfiguration() const -> ParserConfiguration;

  void setToolchain(std::shared_ptr<Toolchain> toolchain);

  void requestCodeCompletionAt(std::uint32_t line, std::uint32_t column,
                               CompletionEditRange editRange,
                               Vector<CompletionItem> result);

  void requestSignatureHelpAt(std::uint32_t line, std::uint32_t column,
                              SignatureHelp result);

  [[nodiscard]] auto diagnostics() const -> Vector<Diagnostic>;
  [[nodiscard]] auto hasErrors() const -> bool;

  [[nodiscard]] auto textOf(AST* ast) -> std::optional<std::string_view>;

  [[nodiscard]] auto textInRange(SourceLocation start, SourceLocation end)
      -> std::optional<std::string_view>;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx::lsp
