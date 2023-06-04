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

/**
 * @brief Interface for handling comments during preprocessing.
 */
class CommentHandler {
 public:
  virtual ~CommentHandler() = default;

  /**
   * @brief Handles a comment token during preprocessing.
   * @param preprocessor The preprocessor instance.
   * @param token The comment token.
   */
  virtual void handleComment(Preprocessor *preprocessor,
                             const Token &token) = 0;
};

/**
 * @brief Interface for a preprocessor delegate.
 */
class PreprocessorDelegate {
 public:
  virtual ~PreprocessorDelegate() = default;
};

/**
 * @brief A C++ preprocessor.
 */
class Preprocessor {
 public:
  Preprocessor(Preprocessor &&) noexcept = default;
  auto operator=(Preprocessor &&) noexcept -> Preprocessor & = default;

  /**
   * @brief Constructs a preprocessor instance.
   * @param control The control instance.
   * @param diagnosticsClient The diagnostics client instance.
   */
  explicit Preprocessor(Control *control, DiagnosticsClient *diagnosticsClient);

  /**
   * @brief Destructs the preprocessor instance.
   */
  ~Preprocessor();

  /**
   * @brief Returns the diagnostics client instance.
   * @return The diagnostics client instance.
   */
  [[nodiscard]] auto diagnosticsClient() const -> DiagnosticsClient *;

  /**
   * @brief Returns the comment handler instance.
   * @return The comment handler instance.
   */
  [[nodiscard]] auto commentHandler() const -> CommentHandler *;

  /**
   * @brief Sets the comment handler instance.
   * @param commentHandler The comment handler instance.
   */
  void setCommentHandler(CommentHandler *commentHandler);

  /**
   * @brief Returns the preprocessor delegate instance.
   * @return The preprocessor delegate instance.
   */
  [[nodiscard]] auto delegate() const -> PreprocessorDelegate *;

  /**
   * @brief Sets the preprocessor delegate instance.
   * @param delegate The preprocessor delegate instance.
   */
  void setDelegate(PreprocessorDelegate *delegate);

  /**
   * @brief Returns whether the preprocessor can resolve files.
   * @return Whether the preprocessor can resolve files.
   */
  [[nodiscard]] auto canResolveFiles() const -> bool;

  /**
   * @brief Sets whether the preprocessor can resolve files.
   * @param canResolveFiles Whether the preprocessor can resolve files.
   */
  void setCanResolveFiles(bool canResolveFiles);

  /**
   * @brief Returns the current path.
   * @return The current path.
   */
  [[nodiscard]] auto currentPath() const -> std::string;

  /**
   * @brief Sets the current path.
   * @param currentPath The current path.
   */
  void setCurrentPath(std::string currentPath);

  /**
   * @brief Preprocesses the given source code and writes the output to the
   * given output stream.
   * @param source The source code to preprocess.
   * @param fileName The name of the file being preprocessed.
   * @param out The output stream to write the preprocessed code to.
   */
  void operator()(std::string source, std::string fileName, std::ostream &out);

  /**
   * @brief Preprocesses the given source code and writes the output to the
   * given output stream.
   * @param source The source code to preprocess.
   * @param fileName The name of the file being preprocessed.
   * @param out The output stream to write the preprocessed code to.
   */
  void preprocess(std::string source, std::string fileName, std::ostream &out);

  /**
   * @brief Preprocesses the given source code and writes the output to the
   * given token vector.
   * @param source The source code to preprocess.
   * @param fileName The name of the file being preprocessed.
   * @param tokens The token vector to write the preprocessed code to.
   */
  void preprocess(std::string source, std::string fileName,
                  std::vector<Token> &tokens);

  /**
   * @brief Returns the system include paths.
   * @return The system include paths.
   */
  [[nodiscard]] auto systemIncludePaths() const
      -> const std::vector<std::string> &;

  /**
   * @brief Adds a system include path.
   * @param path The system include path to add.
   */
  void addSystemIncludePath(std::string path);

  /**
   * @brief Defines a macro.
   * @param name The name of the macro.
   * @param body The body of the macro.
   */
  void defineMacro(const std::string &name, const std::string &body);

  /**
   * @brief Undefines a macro.
   * @param name The name of the macro to undefine.
   */
  void undefMacro(const std::string &name);

  /**
   * @brief Prints the defined macros to the given output stream.
   * @param out The output stream to write the defined macros to.
   */
  void printMacros(std::ostream &out) const;

  /**
   * @brief Returns the start position of the given token.
   * @param token The token to get the start position of.
   * @param line The line number of the start position.
   * @param column The column number of the start position.
   * @param fileName The name of the file the token is in.
   */
  void getTokenStartPosition(const Token &token, unsigned *line,
                             unsigned *column,
                             std::string_view *fileName) const;

  /**
   * @brief Returns the end position of the given token.
   * @param token The token to get the end position of.
   * @param line The line number of the end position.
   * @param column The column number of the end position.
   * @param fileName The name of the file the token is in.
   */
  void getTokenEndPosition(const Token &token, unsigned *line, unsigned *column,
                           std::string_view *fileName) const;

  /**
   * @brief Returns the text line of the given token.
   * @param token The token to get the text line of.
   * @return The text line of the given token.
   */
  [[nodiscard]] auto getTextLine(const Token &token) const -> std::string_view;

  /**
   * @brief Returns the text of the given token.
   * @param token The token to get the text of.
   * @return The text of the given token.
   */
  [[nodiscard]] auto getTokenText(const Token &token) const -> std::string_view;

  void squeeze();

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
