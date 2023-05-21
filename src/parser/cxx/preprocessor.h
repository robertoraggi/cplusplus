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

#pragma once

#include <cxx/cxx_fwd.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class Token;
class DiagnosticsClient;
class CommentHandler;
class Preprocessor;
class PreprocessorDelegate;

class CommentHandler {
 public:
  virtual ~CommentHandler() = default;

  virtual void handleComment(Preprocessor *preprocessor,
                             const Token &token) = 0;
};

class PreprocessorDelegate {
 public:
  virtual ~PreprocessorDelegate() = default;
};

class Preprocessor {
 public:
  Preprocessor(Preprocessor &&) noexcept = default;
  auto operator=(Preprocessor &&) noexcept -> Preprocessor & = default;

  explicit Preprocessor(Control *control, DiagnosticsClient *diagnosticsClient);
  ~Preprocessor();

  [[nodiscard]] auto diagnosticsClient() const -> DiagnosticsClient *;

  [[nodiscard]] auto commentHandler() const -> CommentHandler *;
  void setCommentHandler(CommentHandler *commentHandler);

  [[nodiscard]] auto delegate() const -> PreprocessorDelegate *;
  void setDelegate(PreprocessorDelegate *delegate);

  [[nodiscard]] auto canResolveFiles() const -> bool;
  void setCanResolveFiles(bool canResolveFiles);

  [[nodiscard]] auto currentPath() const -> std::string;
  void setCurrentPath(std::string currentPath);

  void operator()(std::string source, std::string fileName, std::ostream &out);

  void preprocess(std::string source, std::string fileName, std::ostream &out);

  void preprocess(std::string source, std::string fileName,
                  std::vector<Token> &tokens);

  [[nodiscard]] auto systemIncludePaths() const
      -> const std::vector<std::string> &;

  void addSystemIncludePath(std::string path);

  void defineMacro(const std::string &name, const std::string &body);

  void undefMacro(const std::string &name);

  void printMacros(std::ostream &out) const;

  void getTokenStartPosition(const Token &token, unsigned *line,
                             unsigned *column,
                             std::string_view *fileName) const;

  void getTokenEndPosition(const Token &token, unsigned *line, unsigned *column,
                           std::string_view *fileName) const;

  [[nodiscard]] auto getTextLine(const Token &token) const -> std::string_view;

  [[nodiscard]] auto getTokenText(const Token &token) const -> std::string_view;

  void squeeze();

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
