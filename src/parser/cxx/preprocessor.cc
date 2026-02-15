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

#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/private/pp_directives-priv.h>
#include <cxx/util.h>
#include <utf8/unchecked.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace {

#include "private/builtins-priv.h"

std::unordered_set<std::string_view> enabledBuiltins{
    "__is_trivially_destructible",
    "__builtin_is_constant_evaluated",

    "__has_unique_object_representations",
    "__has_virtual_destructor",
    "__is_abstract",
    "__is_aggregate",
    "__is_arithmetic",
    "__is_array",
    "__is_assignable",
    "__is_base_of",
    "__is_bounded_array",
    "__is_class",
    "__is_compound",
    "__is_const",
    "__is_empty",
    "__is_enum",
    "__is_final",
    "__is_floating_point",
    "__is_function",
    "__is_fundamental",
    "__is_integral",
    "__is_layout_compatible",
    "__is_literal_type",
    "__is_lvalue_reference",
    "__is_member_function_pointer",
    "__is_member_object_pointer",
    "__is_member_pointer",
    "__is_null_pointer",
    "__is_object",
    "__is_pod",
    "__is_pointer",
    "__is_polymorphic",
    "__is_reference",
    "__is_rvalue_reference",
    "__is_same_as",
    "__is_same",
    "__is_scalar",
    "__is_scoped_enum",
    "__is_signed",
    "__is_standard_layout",
    "__is_swappable_with",
    "__is_trivial",
    "__is_unbounded_array",
    "__is_union",
    "__is_unsigned",
    "__is_void",
    "__is_volatile",

    "__make_integer_seq",
    "__remove_reference_t",
    "__type_pack_element",

#define VISIT_BUILTIN_FUNCTION(_, name) name,
    FOR_EACH_BUILTIN_FUNCTION(VISIT_BUILTIN_FUNCTION)
#undef VISIT_BUILTIN_FUNCTION

#if false
      "__add_lvalue_reference", "__add_pointer", "__add_rvalue_reference",
      "__array_extent", "__array_rank", "__atomic_add_fetch", "__atomic_load_n",
      "__builtin_assume", "__builtin_bswap128", "__builtin_char_memchr",
      "__builtin_coro_noop", "__builtin_isfinite", "__builtin_isinf",
      "__builtin_isnan", "__builtin_operator_delete", "__builtin_operator_new",
      "__builtin_source_location", "__builtin_va_copy", "__builtin_wcslen",
      "__builtin_wmemcmp", "__decay", "__has_trivial_destructor", "__is_array",
      "__is_compound", "__is_const", "__is_convertible_to", "__is_destructible",
      "__is_function", "__is_fundamental", "__is_integral",
      "__is_lvalue_reference", "__is_member_function_pointer",
      "__is_member_object_pointer", "__is_member_pointer",
      "__is_nothrow_constructible", "__is_object", "__is_pointer",
      "__is_reference", "__is_referenceable", "__is_rvalue_reference",
      "__is_scalar", "__is_signed", "__is_trivially_destructible",
      "__is_unsigned", "__is_void", "__is_volatile", "__make_integer_seq",
      "__make_signed", "__make_unsigned", "__remove_all_extents",
      "__remove_const", "__remove_cv", "__remove_cvref", "__remove_extent",
      "__remove_pointer", "__remove_reference_t", "__remove_volatile",
      "__type_pack_element",
#endif
};

std::unordered_set<std::string_view> enabledExtensions{
    "__cxx_binary_literals__",
    "cxx_string_literal_templates",
};

std::unordered_set<std::string_view> enabledFeatures{
    "__cxx_aggregate_nsdmi__",
    "__cxx_alias_templates__",
    "__cxx_binary_literals__",
    "__cxx_constexpr__",
    "__cxx_decltype__",
    "__cxx_decltype_auto__",
    "__cxx_deleted_functions__",
    "__cxx_generic_lambdas__",
    "__cxx_init_captures__",
    "__cxx_noexcept__",
    "__cxx_nullptr__",
    "__cxx_reference_qualified_functions__",
    "__cxx_relaxed_constexpr__",
    "__cxx_return_type_deduction__",
    "__cxx_rvalue_references__",
    "__cxx_static_assert__",
    "__cxx_variable_templates__",
    "__cxx_variadic_templates__",
    "cxx_alias_templates",
    "cxx_alignas",
    "cxx_alignof",
    "cxx_atomic",
    "cxx_attributes",
    "cxx_auto_type",
    "cxx_constexpr_string_builtins",
    "cxx_constexpr",
    "cxx_decltype_incomplete_return_types",
    "cxx_decltype",
    "cxx_default_function_template_args",
    "cxx_defaulted_functions",
    "cxx_delegating_constructors",
    "cxx_deleted_functions",
    "cxx_exceptions",
    "cxx_explicit_conversions",
    "cxx_generalized_initializers",
    "cxx_inline_namespaces",
    "cxx_lambdas",
    "cxx_local_type_template_args",
    "cxx_noexcept",
    "cxx_nullptr",
    "cxx_override_control",
    "cxx_range_for",
    "cxx_raw_string_literals",
    "cxx_reference_qualified_functions",
    "cxx_rtti",
    "cxx_rvalue_references",
    "cxx_static_assert",
    "cxx_strong_enums",
    "cxx_thread_local",
    "cxx_trailing_return",
    "cxx_unicode_literals",
    "cxx_unrestricted_unions",
    "cxx_user_literals",
    "cxx_variadic_templates",
};

inline auto getHeaderName(const cxx::Include& include) -> std::string {
  return std::visit([](const auto& include) { return include.fileName; },
                    include);
}

}  // namespace

namespace cxx {

namespace {

struct SourceFile;

struct Tok {
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t sourceFile = 0;
  std::uint32_t textIndex = 0;
  TokenKind kind = TokenKind::T_EOF_SYMBOL;
  std::uint8_t bol : 1 = false;
  std::uint8_t space : 1 = false;
  std::uint8_t generated : 1 = false;
  std::uint8_t isFromMacroBody : 1 = false;
  std::uint8_t noexpand : 1 = false;
  std::uint8_t dirty : 1 = false;

  [[nodiscard]] auto is(TokenKind k) const -> bool { return kind == k; }
  [[nodiscard]] auto isNot(TokenKind k) const -> bool { return kind != k; }
};

using TokVector = std::vector<Tok>;
using TokSpan = std::span<const Tok>;

struct SourceFile {
  std::string fileName;
  std::string source;
  mutable std::vector<int> lines;
  mutable bool linesComputed = false;
  TokVector tokens;
  std::string headerGuardName;
  int headerProtectionLevel = 0;
  int id = 0;
  bool pragmaOnceProtected = false;
  bool isSystemHeader = false;

  SourceFile() noexcept = default;
  SourceFile(const SourceFile&) = default;
  auto operator=(const SourceFile&) -> SourceFile& = default;
  SourceFile(SourceFile&&) noexcept = default;
  auto operator=(SourceFile&&) noexcept -> SourceFile& = default;

  SourceFile(std::string fileName, std::string source,
             std::uint32_t id) noexcept
      : fileName(std::move(fileName)), source(std::move(source)), id(id) {}

  void ensureLineMap() const {
    if (linesComputed) return;
    linesComputed = true;
    std::size_t offset = 0;
    lines.push_back(0);
    while (offset < source.length()) {
      const auto index = source.find_first_of('\n', offset);
      if (index == std::string::npos) break;
      offset = index + 1;
      lines.push_back(static_cast<int>(offset));
    }
  }

  [[nodiscard]] auto getTokenStartPosition(unsigned offset) const
      -> SourcePosition {
    ensureLineMap();
    auto it = std::lower_bound(lines.cbegin(), lines.cend(),
                               static_cast<int>(offset));
    if (*it != static_cast<int>(offset)) --it;
    assert(*it <= int(offset));
    auto line = std::uint32_t(std::distance(cbegin(lines), it) + 1);
    const auto start = cbegin(source) + *it;
    const auto end = cbegin(source) + offset;
    const auto column =
        std::uint32_t(utf8::unchecked::distance(start, end) + 1);
    return SourcePosition{fileName, line, column};
  }

  [[nodiscard]] auto offsetAt(std::uint32_t line, std::uint32_t column) const
      -> std::uint32_t {
    ensureLineMap();
    if (line == 0 && column == 0) return 0;
    if (line > lines.size()) return static_cast<std::uint32_t>(source.size());
    const auto start = source.data();
    const auto offsetOfTheLine = lines[line - 1];
    auto it = start + offsetOfTheLine;
    for (std::uint32_t i = 1; i < column; ++i) {
      utf8::unchecked::next(it);
    }
    return static_cast<std::uint32_t>(it - start);
  }

  void releaseTokens() {
    tokens.clear();
    tokens.shrink_to_fit();
  }
};

struct ObjectMacro {
  std::string name;
  TokVector body;

  ObjectMacro(std::string name, TokVector body)
      : name(std::move(name)), body(std::move(body)) {}
};

struct FunctionMacro {
  std::string name;
  std::vector<std::string> formals;
  TokVector body;
  bool variadic = false;

  FunctionMacro(std::string name, std::vector<std::string> formals,
                TokVector body, bool variadic)
      : name(std::move(name)),
        formals(std::move(formals)),
        body(std::move(body)),
        variadic(variadic) {}
};

struct MacroExpansionContext {
  const Tok* tok = nullptr;
  const Tok* pos = nullptr;
  const Tok* end = nullptr;
};

struct BuiltinObjectMacro {
  std::string name;
  std::function<auto(MacroExpansionContext)->TokVector> expand;

  BuiltinObjectMacro(
      std::string name,
      std::function<auto(MacroExpansionContext)->TokVector> expand)
      : name(std::move(name)), expand(std::move(expand)) {}
};

struct BuiltinFunctionMacro {
  std::string name;
  std::function<auto(MacroExpansionContext)->TokVector> expand;

  BuiltinFunctionMacro(
      std::string name,
      std::function<auto(MacroExpansionContext)->TokVector> expand)
      : name(std::move(name)), expand(std::move(expand)) {}
};

using Macro = std::variant<ObjectMacro, FunctionMacro, BuiltinObjectMacro,
                           BuiltinFunctionMacro>;

struct TransparentStringHash {
  using is_transparent = void;
  auto operator()(std::string_view s) const noexcept -> std::size_t {
    return std::hash<std::string_view>{}(s);
  }
};

struct TransparentStringEqual {
  using is_transparent = void;
  auto operator()(std::string_view a, std::string_view b) const noexcept
      -> bool {
    return a == b;
  }
};

[[nodiscard]] inline auto getMacroName(const Macro& macro) -> std::string_view {
  return std::visit([](const auto& m) -> std::string_view { return m.name; },
                    macro);
}

[[nodiscard]] inline auto getMacroBody(const Macro& macro) -> const TokVector* {
  struct Visitor {
    auto operator()(const ObjectMacro& m) const -> const TokVector* {
      return &m.body;
    }
    auto operator()(const FunctionMacro& m) const -> const TokVector* {
      return &m.body;
    }
    auto operator()(const BuiltinObjectMacro&) const -> const TokVector* {
      return nullptr;
    }
    auto operator()(const BuiltinFunctionMacro&) const -> const TokVector* {
      return nullptr;
    }
  };
  return std::visit(Visitor{}, macro);
}

[[nodiscard]] inline auto isObjectLikeMacro(const Macro& macro) -> bool {
  return std::holds_alternative<ObjectMacro>(macro) ||
         std::holds_alternative<BuiltinObjectMacro>(macro);
}

[[nodiscard]] inline auto isFunctionLikeMacro(const Macro& macro) -> bool {
  return std::holds_alternative<FunctionMacro>(macro) ||
         std::holds_alternative<BuiltinFunctionMacro>(macro);
}

static auto isSameBody(
    const TokVector& a, const TokVector& b,
    const std::vector<std::string>& texts,
    const std::vector<std::unique_ptr<SourceFile>>& sourceFiles) -> bool {
  if (a.size() != b.size()) return false;
  auto getTextOf = [&](const Tok& tk) -> std::string_view {
    if (tk.dirty && tk.textIndex > 0) return texts[tk.textIndex - 1];
    if (tk.sourceFile > 0 && tk.sourceFile <= sourceFiles.size()) {
      const auto& src = sourceFiles[tk.sourceFile - 1]->source;
      return std::string_view(src).substr(tk.offset, tk.length);
    }
    return {};
  };
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].kind != b[i].kind) return false;
    if (getTextOf(a[i]) != getTextOf(b[i])) return false;
  }
  return true;
}

}  // namespace

struct Preprocessor::Private {
  struct Cursor {
    enum Kind { FileCursor, ExpansionCursor };
    Kind kind = FileCursor;
    SourceFile* sourceFile = nullptr;
    fs::path currentPath;
    int includeDepth = 0;
    const Identifier* untaintOnPop = nullptr;
    TokVector ownedTokens;
    const Tok* begin = nullptr;
    const Tok* end = nullptr;
    const Tok* pos = nullptr;

    void initFromSourceFile() {
      if (sourceFile && !sourceFile->tokens.empty()) {
        begin = sourceFile->tokens.data();
        end = begin + sourceFile->tokens.size();
        pos = begin;
      } else {
        begin = end = pos = nullptr;
      }
    }

    void initFromOwned() {
      if (!ownedTokens.empty()) {
        begin = ownedTokens.data();
        end = begin + ownedTokens.size();
        pos = begin;
      } else {
        begin = end = pos = nullptr;
      }
    }

    [[nodiscard]] auto atEnd() const -> bool {
      return pos == nullptr || pos >= end;
    }

    [[nodiscard]] auto current() const -> const Tok& { return *pos; }

    void advance() { ++pos; }
  };

  Preprocessor* preprocessor_ = nullptr;
  Control* control_ = nullptr;
  DiagnosticsClient* diagnosticsClient_ = nullptr;
  CommentHandler* commentHandler_ = nullptr;
  LanguageKind language_ = LanguageKind::kCXX;
  bool canResolveFiles_ = true;
  bool disableCurrentDirSearch_ = false;
  std::vector<std::string> systemIncludePaths_;
  std::vector<std::string> quoteIncludePaths_;
  std::vector<std::string> userIncludePaths_;
  std::vector<std::pair<std::string, bool>> includedFiles_;
  std::unordered_map<std::string, Macro, TransparentStringHash,
                     TransparentStringEqual>
      macros_;
  std::unordered_set<const cxx::Identifier*> taintedIdents_;
  std::unordered_map<std::string, std::string> ifndefProtectedFiles_;
  std::vector<std::unique_ptr<SourceFile>> sourceFiles_;
  fs::path currentPath_;
  std::string currentFileName_;
  std::vector<bool> evaluating_;
  std::vector<bool> skipping_;
  std::string date_;
  std::string time_;
  std::function<void(const std::string&, int)> willIncludeHeader_;
  std::deque<Cursor> cursors_;

  struct Dep {
    std::string local;
    Include include;
    bool isIncludeNext = false;
    bool exists = false;
  };
  std::vector<Dep> dependencies_;
  std::function<auto()->std::optional<PreprocessingState>> continuation_;
  std::optional<SourcePosition> codeCompletionLocation_;
  std::uint32_t codeCompletionOffset_ = 0;
  int localCount_ = 0;

  int counter_ = 0;
  int includeDepth_ = 0;
  int builtinsFileId_ = 0;
  int mainSourceFileId_ = 0;
  bool omitLineMarkers_ = false;
  std::unordered_map<std::string, SourceFile*> sourceFileIndex_;

  std::vector<std::string> texts_;

  std::vector<TokVector> expansionPool_;

  struct ResolveResult {
    std::string fileName;
    bool isSystemHeader = false;
  };

  struct IncludeCacheKey {
    bool isQuoted = false;
    bool isIncludeNext = false;
    std::string currentPath;
    std::string headerName;

    auto operator<=>(const IncludeCacheKey& other) const = default;
  };

  mutable std::map<IncludeCacheKey, std::optional<ResolveResult>> resolveCache_;

  Private();

  void initialize();

  [[nodiscard]] auto addText(std::string s) -> std::uint32_t {
    texts_.push_back(std::move(s));
    return static_cast<std::uint32_t>(texts_.size());
  }

  [[nodiscard]] auto getText(const Tok& tk) const -> std::string_view {
    if (tk.dirty && tk.textIndex > 0) {
      return texts_[tk.textIndex - 1];
    }
    if (tk.sourceFile > 0 && tk.sourceFile <= sourceFiles_.size()) {
      const auto& src = sourceFiles_[tk.sourceFile - 1]->source;
      return std::string_view(src).substr(tk.offset, tk.length);
    }
    return {};
  }

  [[nodiscard]] auto acquireExpansionBuffer() -> TokVector {
    if (!expansionPool_.empty()) {
      auto buf = std::move(expansionPool_.back());
      expansionPool_.pop_back();
      buf.clear();
      return buf;
    }
    return TokVector{};
  }

  void releaseExpansionBuffer(TokVector buf) {
    buf.clear();
    expansionPool_.push_back(std::move(buf));
  }

  [[nodiscard]] auto makeTok(const Lexer& lex, int sourceFile) -> Tok {
    Tok tk;
    tk.sourceFile = sourceFile;
    tk.kind = lex.tokenKind();
    tk.offset = lex.tokenPos();
    tk.length = lex.tokenLength();
    tk.bol = lex.tokenStartOfLine();
    tk.space = lex.tokenLeadingSpace();
    return tk;
  }

  [[nodiscard]] auto genTok(TokenKind kind, const std::string_view& text)
      -> Tok {
    Tok tk;
    tk.kind = kind;
    tk.generated = true;
    tk.dirty = true;
    tk.length = static_cast<std::uint32_t>(text.length());
    tk.textIndex = addText(std::string(text));
    return tk;
  }

  [[nodiscard]] auto copyTok(const Tok& src) -> Tok { return src; }

  [[nodiscard]] auto tokenForDiagnostic(const Tok& tk) const -> Token {
    Token token(tk.kind, tk.offset, tk.length);
    token.setFileId(tk.sourceFile);
    token.setLeadingSpace(tk.space);
    token.setStartOfLine(tk.bol);
    return token;
  }

  void error(const Tok* tk, std::string message) const {
    if (!tk) {
      cxx_runtime_error(std::format("no source location: {}", message));
    } else {
      diagnosticsClient_->report(tokenForDiagnostic(*tk), Severity::Error,
                                 std::move(message));
    }
  }

  void warning(const Tok* tk, std::string message) const {
    if (!tk) {
      cxx_runtime_error(std::format("no source location: {}", message));
    } else {
      diagnosticsClient_->report(tokenForDiagnostic(*tk), Severity::Warning,
                                 std::move(message));
    }
  }

  void error(const Token& token, std::string message) const {
    diagnosticsClient_->report(token, Severity::Error, std::move(message));
  }

  void warning(const Token& token, std::string message) const {
    diagnosticsClient_->report(token, Severity::Warning, std::move(message));
  }

  [[nodiscard]] auto state() const -> std::tuple<bool, bool> {
    return std::tuple(skipping_.back(), evaluating_.back());
  }

  void pushState(std::tuple<bool, bool> state) {
    auto [skipping, evaluating] = state;
    skipping_.push_back(skipping);
    evaluating_.push_back(evaluating);
  }

  void setState(std::tuple<bool, bool> state) {
    auto [skipping, evaluating] = state;
    skipping_.back() = skipping;
    evaluating_.back() = evaluating;
  }

  void popState() {
    skipping_.pop_back();
    evaluating_.pop_back();
  }

  [[nodiscard]] auto findSourceFile(const std::string& fileName)
      -> SourceFile* {
    auto it = sourceFileIndex_.find(fileName);
    if (it != sourceFileIndex_.end()) return it->second;
    return nullptr;
  }

  [[nodiscard]] auto createSourceFile(std::string fileName, std::string source)
      -> SourceFile*;

  [[nodiscard]] auto tokenize(const std::string_view& source, int sourceFile,
                              bool bol) -> TokVector;

  void skipLine(const Tok*& pos, const Tok* end);

  struct ParsedIncludeDirective {
    Include header;
    bool includeNext = false;
    const Tok* loc = nullptr;
  };

  struct ParsedIfDirective {
    std::function<auto()->std::optional<PreprocessingState>> resume;
  };

  using ParsedDirective =
      std::variant<std::monostate, ParsedIncludeDirective, ParsedIfDirective>;

  [[nodiscard]] auto parseDirective(SourceFile* source,
                                    const Tok* directiveLine,
                                    const Tok* directiveEnd) -> ParsedDirective;

  [[nodiscard]] auto parseIncludeDirective(const Tok* directive, const Tok* ts,
                                           const Tok* lineEnd)
      -> std::optional<ParsedIncludeDirective>;

  [[nodiscard]] auto parseHeaderName(const Tok*& ts, const Tok* lineEnd)
      -> std::optional<Include>;

  template <typename EmitToken>
  [[nodiscard]] auto expand(const EmitToken& emitToken) -> PreprocessingState;

  [[nodiscard]] auto expandTokens(const Tok* begin, const Tok* end,
                                  bool inConditionalExpression) -> TokVector;

  template <typename EmitToken>
  void expandOne(Cursor& cursor, bool inConditionalExpression,
                 const EmitToken& emitToken);

  template <typename EmitToken>
  auto replaceIsDefinedMacro(Cursor& cursor, bool inConditionalExpression,
                             const EmitToken& emitToken) -> bool;

  [[nodiscard]] auto expandMacro(Cursor& cursor) -> bool;

  [[nodiscard]] auto expandObjectLikeMacro(Cursor& cursor, const Macro* macro,
                                           const cxx::Identifier* ident)
      -> bool;

  [[nodiscard]] auto expandFunctionLikeMacro(Cursor& cursor, const Macro* macro,
                                             const cxx::Identifier* ident)
      -> bool;

  [[nodiscard]] auto expandFunctionLikeMacroAcrossBoundary(
      const Macro* macro, const cxx::Identifier* ident) -> bool;

  using TokRange = std::pair<const Tok*, const Tok*>;

  [[nodiscard]] auto substitute(const Tok& pointOfSubstitution,
                                const Macro* macro,
                                const std::vector<TokRange>& actuals,
                                const std::vector<TokRange>& expandedActuals)
      -> TokVector;

  [[nodiscard]] auto merge(const Tok& left, const Tok& right) -> Tok;

  [[nodiscard]] auto stringize(const Tok* begin, const Tok* end) -> Tok;

  [[nodiscard]] auto lookupMacro(const Tok& tk) const
      -> std::pair<const Macro*, const cxx::Identifier*>;

  [[nodiscard]] auto lookupMacroArgument(const Tok*& ts, const Tok* lineEnd,
                                         const Macro* macro,
                                         const std::vector<TokRange>& actuals)
      -> std::optional<TokRange>;

  [[nodiscard]] auto copyLine(const Tok* ts, const Tok* lineEnd,
                              bool inMacroBody = false) -> TokVector;

  [[nodiscard]] auto prepareConstantExpression(const Tok* ts,
                                               const Tok* lineEnd) -> TokVector;

  [[nodiscard]] auto evaluateConstantExpression(const Tok*& ts, const Tok* end)
      -> long;
  [[nodiscard]] auto conditionalExpression(const Tok*& ts, const Tok* end)
      -> long;
  [[nodiscard]] auto binaryExpression(const Tok*& ts, const Tok* end) -> long;
  [[nodiscard]] auto binaryExpressionHelper(const Tok*& ts, const Tok* end,
                                            long lhs, int minPrec) -> long;
  [[nodiscard]] auto unaryExpression(const Tok*& ts, const Tok* end) -> long;
  [[nodiscard]] auto primaryExpression(const Tok*& ts, const Tok* end) -> long;

  struct ParsedArgs {
    std::vector<TokRange> args;
    const Tok* rest = nullptr;
  };

  [[nodiscard]] auto parseArguments(const Tok* pos, const Tok* end,
                                    std::size_t formalCount, bool ignoreComma)
      -> ParsedArgs;

  [[nodiscard]] auto parseMacroDefinition(const Tok* ts, const Tok* lineEnd)
      -> Macro;

  void defineMacro(const Tok* ts, const Tok* lineEnd);

  [[nodiscard]] auto fileExists(const fs::path& file) const -> bool {
    return fs::exists(file) && !fs::is_directory(file);
  }

  [[nodiscard]] auto checkHeaderProtection(const TokVector& tokens) const
      -> std::string;

  [[nodiscard]] auto checkPragmaOnceProtected(const TokVector& tokens) const
      -> bool;

  [[nodiscard]] auto resolve(const Include& include, bool isIncludeNext) const
      -> std::optional<ResolveResult>;

  [[nodiscard]] auto resolveUncached(const Include& include, bool isIncludeNext,
                                     const std::string& headerName,
                                     bool isQuoted) const
      -> std::optional<ResolveResult>;

  template <typename TryCandidate>
  [[nodiscard]] auto resolveNormal(const TryCandidate& tryCandidate,
                                   const std::string& curDir,
                                   bool isQuoted) const
      -> std::optional<ResolveResult>;

  template <typename TryCandidate>
  [[nodiscard]] auto resolveNext(const TryCandidate& tryCandidate,
                                 const std::string& curDir,
                                 bool isQuoted) const
      -> std::optional<ResolveResult>;

  [[nodiscard]] auto buildSearchDirs(const std::string& curDir,
                                     bool isQuoted) const
      -> std::vector<std::pair<const std::string*, bool>>;

  [[nodiscard]] auto isDefined(std::string_view id) const -> bool {
    return macros_.contains(id);
  }

  [[nodiscard]] auto isDefined(const Tok& tok) const -> bool {
    return tok.is(TokenKind::T_IDENTIFIER) && isDefined(getText(tok));
  }

  [[nodiscard]] auto isTainted(const cxx::Identifier* id) const -> bool {
    return taintedIdents_.contains(id);
  }

  void taint(const cxx::Identifier* id) { taintedIdents_.insert(id); }
  void untaint(const cxx::Identifier* id) { taintedIdents_.erase(id); }

  [[nodiscard]] auto isStringLiteral(TokenKind kind) const -> bool {
    switch (kind) {
      case TokenKind::T_STRING_LITERAL:
      case TokenKind::T_WIDE_STRING_LITERAL:
      case TokenKind::T_UTF8_STRING_LITERAL:
      case TokenKind::T_UTF16_STRING_LITERAL:
      case TokenKind::T_UTF32_STRING_LITERAL:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] auto updateStringLiteralValue(Token& lastToken, const Tok& tk)
      -> bool {
    if (!isStringLiteral(lastToken.kind())) return false;
    if (tk.isNot(TokenKind::T_STRING_LITERAL) && tk.kind != lastToken.kind())
      return false;

    auto newText = lastToken.value().literalValue->value();
    if (newText.ends_with('"')) newText.pop_back();

    auto tkText = getText(tk);
    newText += tkText.substr(tkText.find_first_of('"') + 1);

    TokenValue value = lastToken.value();
    auto internedText = std::string(newText);

    switch (lastToken.kind()) {
      case TokenKind::T_STRING_LITERAL:
        value.literalValue = control_->stringLiteral(internedText);
        break;
      case TokenKind::T_WIDE_STRING_LITERAL:
        value.literalValue = control_->wideStringLiteral(internedText);
        break;
      case TokenKind::T_UTF8_STRING_LITERAL:
        value.literalValue = control_->utf8StringLiteral(internedText);
        break;
      case TokenKind::T_UTF16_STRING_LITERAL:
        value.literalValue = control_->utf16StringLiteral(internedText);
        break;
      case TokenKind::T_UTF32_STRING_LITERAL:
        value.literalValue = control_->utf32StringLiteral(internedText);
        break;
      default:
        break;
    }

    lastToken.setValue(value);
    return true;
  }

  void adddBuiltinMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokVector> expand) {
    macros_.insert_or_assign(
        std::string(name),
        BuiltinObjectMacro(std::string(name), std::move(expand)));
  }

  void adddBuiltinFunctionMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokVector> expand) {
    macros_.insert_or_assign(
        std::string(name),
        BuiltinFunctionMacro(std::string(name), std::move(expand)));
  }

  void finalizeToken(std::vector<Token>& tokens, const Tok& tk);

  void print(const Tok* begin, const Tok* end, std::ostream& out) const;
  void printLine(const Tok* begin, const Tok* end, std::ostream& out,
                 bool nl = true) const;
};

Preprocessor::Private::Private() {
  skipping_.push_back(false);
  evaluating_.push_back(true);

  time_t t;
  time(&t);

  char buffer[32];
  strftime(buffer, sizeof(buffer), "\"%b %e %Y\"", localtime(&t));
  date_ = buffer;

  strftime(buffer, sizeof(buffer), "\"%T\"", localtime(&t));
  time_ = buffer;
}

void Preprocessor::Private::initialize() {
  adddBuiltinMacro("__FILE__",
                   [this](const MacroExpansionContext& context) -> TokVector {
                     TokVector result;
                     auto tk = genTok(TokenKind::T_STRING_LITERAL,
                                      std::format("\"{}\"", currentFileName_));
                     tk.space = true;
                     tk.sourceFile = context.tok->sourceFile;
                     result.push_back(tk);
                     return result;
                   });

  adddBuiltinMacro(
      "__LINE__", [this](const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        const auto start =
            preprocessor_->tokenStartPosition(tokenForDiagnostic(*context.tok));
        auto tk =
            genTok(TokenKind::T_INTEGER_LITERAL, std::to_string(start.line));
        tk.sourceFile = context.tok->sourceFile;
        tk.space = true;
        result.push_back(tk);
        return result;
      });

  adddBuiltinMacro(
      "__COUNTER__", [this](const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        auto tk =
            genTok(TokenKind::T_INTEGER_LITERAL, std::to_string(counter_++));
        tk.sourceFile = context.tok->sourceFile;
        tk.space = true;
        result.push_back(tk);
        return result;
      });

  adddBuiltinMacro("__DATE__",
                   [this](const MacroExpansionContext& context) -> TokVector {
                     TokVector result;
                     auto tk = genTok(TokenKind::T_STRING_LITERAL, date_);
                     tk.sourceFile = context.tok->sourceFile;
                     tk.space = true;
                     result.push_back(tk);
                     return result;
                   });

  adddBuiltinMacro("__TIME__",
                   [this](const MacroExpansionContext& context) -> TokVector {
                     TokVector result;
                     auto tk = genTok(TokenKind::T_STRING_LITERAL, time_);
                     tk.sourceFile = context.tok->sourceFile;
                     tk.space = true;
                     result.push_back(tk);
                     return result;
                   });

  auto replaceWithBoolLiteral = [this](const Tok& token, bool value) -> Tok {
    auto tk = genTok(TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
    tk.sourceFile = token.sourceFile;
    tk.space = token.space;
    tk.bol = token.bol;
    return tk;
  };

  auto replaceWithLocal = [this](const Tok& token,
                                 std::string_view local) -> Tok {
    auto tk = genTok(TokenKind::T_PP_INTERNAL_VARIABLE, local);
    tk.sourceFile = token.sourceFile;
    tk.space = token.space;
    tk.bol = token.bol;
    return tk;
  };

  adddBuiltinFunctionMacro(
      "__has_feature",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        auto pos = context.pos;
        auto end = context.end;
        auto macroId = *context.tok;
        if (pos < end && pos->is(TokenKind::T_LPAREN)) ++pos;
        std::string_view id;
        if (pos < end && pos->is(TokenKind::T_IDENTIFIER)) {
          id = getText(*pos);
          ++pos;
        }
        if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;
        result.push_back(
            replaceWithBoolLiteral(macroId, enabledFeatures.contains(id)));
        return result;
      });

  adddBuiltinFunctionMacro(
      "__has_builtin",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        auto pos = context.pos;
        auto end = context.end;
        auto macroId = *context.tok;
        if (pos < end && pos->is(TokenKind::T_LPAREN)) ++pos;
        std::string_view id;
        if (pos < end && !pos->is(TokenKind::T_RPAREN) &&
            !pos->is(TokenKind::T_EOF_SYMBOL)) {
          id = getText(*pos);
          ++pos;
        }
        if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;
        result.push_back(
            replaceWithBoolLiteral(macroId, enabledBuiltins.contains(id)));
        return result;
      });

  adddBuiltinFunctionMacro(
      "__has_extension",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        auto pos = context.pos;
        auto end = context.end;
        auto macroId = *context.tok;
        if (pos < end && pos->is(TokenKind::T_LPAREN)) ++pos;
        std::string_view id;
        if (pos < end && pos->is(TokenKind::T_IDENTIFIER)) {
          id = getText(*pos);
          ++pos;
        }
        if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;
        result.push_back(
            replaceWithBoolLiteral(macroId, enabledExtensions.contains(id)));
        return result;
      });

  adddBuiltinFunctionMacro(
      "__has_attribute",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext& context) -> TokVector {
        TokVector result;
        auto pos = context.pos;
        auto end = context.end;
        auto macroId = *context.tok;
        if (pos < end && pos->is(TokenKind::T_LPAREN)) ++pos;
        if (pos < end && pos->is(TokenKind::T_IDENTIFIER)) ++pos;
        if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;
        result.push_back(replaceWithBoolLiteral(macroId, true));
        return result;
      });

  auto hasInclude =
      [this, replaceWithLocal, replaceWithBoolLiteral](
          const MacroExpansionContext& context) mutable -> TokVector {
    TokVector result;
    auto pos = context.pos;
    auto end = context.end;
    auto macroTok = *context.tok;

    const auto isIncludeNext = getText(macroTok) == "__has_include_next";

    if (pos < end && pos->is(TokenKind::T_LPAREN)) ++pos;

    auto argBegin = pos;
    int depth = 0;
    while (pos < end) {
      if (pos->is(TokenKind::T_LPAREN))
        ++depth;
      else if (pos->is(TokenKind::T_RPAREN)) {
        if (depth == 0) break;
        --depth;
      }
      ++pos;
    }
    auto argEnd = pos;
    if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;

    auto expandedArg = expandTokens(argBegin, argEnd, false);

    Include include;
    const Tok* ea = expandedArg.data();
    const Tok* eaEnd = ea + expandedArg.size();

    if (ea < eaEnd && ea->is(TokenKind::T_STRING_LITERAL)) {
      auto text = getText(*ea);
      std::string fn(text.substr(1, text.length() - 2));
      include = QuoteInclude(std::move(fn));
    } else if (ea < eaEnd && ea->is(TokenKind::T_LESS)) {
      ++ea;
      std::string fn;
      while (ea < eaEnd && ea->isNot(TokenKind::T_GREATER)) {
        fn += getText(*ea);
        ++ea;
      }
      include = SystemInclude(std::move(fn));
    }

    auto local = std::format("@{}", localCount_++);
    dependencies_.push_back({local, include, isIncludeNext});
    result.push_back(replaceWithLocal(macroTok, local));
    return result;
  };

  adddBuiltinFunctionMacro("__has_include", hasInclude);
  adddBuiltinFunctionMacro("__has_include_next", hasInclude);
}

auto Preprocessor::Private::createSourceFile(std::string fileName,
                                             std::string source)
    -> SourceFile* {
  if (sourceFiles_.size() >= 4096) {
    cxx_runtime_error("too many source files");
  }

  const int sourceFileId = static_cast<int>(sourceFiles_.size() + 1);

  SourceFile* sourceFile =
      &*sourceFiles_.emplace_back(std::make_unique<SourceFile>(
          std::move(fileName), std::move(source), sourceFileId));

  sourceFileIndex_[sourceFile->fileName] = sourceFile;

  sourceFile->tokens = tokenize(sourceFile->source, sourceFileId, true);

  return sourceFile;
}

auto Preprocessor::Private::tokenize(const std::string_view& source,
                                     int sourceFile, bool bol) -> TokVector {
  cxx::Lexer lex(source, language_);
  lex.setKeepComments(true);
  lex.setPreprocessing(true);

  TokVector tokens;
  tokens.reserve(source.size() / 4);

  do {
    lex();

    if (lex.tokenKind() == TokenKind::T_COMMENT) {
      if (commentHandler_) {
        TokenValue tokenValue{};

        if (sourceFile) {
          const SourceFile* file = sourceFiles_[sourceFile - 1].get();
          auto tokenText =
              file->source.substr(lex.tokenPos(), lex.tokenLength());
          tokenValue.literalValue = control_->commentLiteral(tokenText);
        }

        Token token(lex.tokenKind(), lex.tokenPos(), lex.tokenLength(),
                    tokenValue);
        token.setFileId(sourceFile);
        token.setLeadingSpace(lex.tokenLeadingSpace());
        token.setStartOfLine(lex.tokenStartOfLine());
        commentHandler_->handleComment(preprocessor_, token);
      }
      continue;
    }

    auto tk = makeTok(lex, sourceFile);
    if (sourceFile == 0 || !lex.tokenIsClean()) {
      tk.dirty = true;
      auto text =
          lex.tokenIsClean()
              ? std::string(source.substr(lex.tokenPos(), lex.tokenLength()))
              : std::move(lex.text());
      tk.textIndex = addText(std::move(text));
    }

    tokens.push_back(tk);
  } while (lex.tokenKind() != cxx::TokenKind::T_EOF_SYMBOL);

  return tokens;
}

void Preprocessor::Private::skipLine(const Tok*& pos, const Tok* end) {
  while (pos < end) {
    if (pos->kind == TokenKind::T_EOF_SYMBOL) break;
    if (pos->bol) break;
    ++pos;
  }
}

auto Preprocessor::Private::copyLine(const Tok* ts, const Tok* lineEnd,
                                     bool inMacroBody) -> TokVector {
  TokVector line;
  const Tok* lastTok = ts;
  while (ts < lineEnd && ts->isNot(TokenKind::T_EOF_SYMBOL) && !ts->bol) {
    auto tok = *ts;
    if (inMacroBody) {
      tok.isFromMacroBody = true;
    }
    line.push_back(tok);
    lastTok = ts;
    ++ts;
  }
  Tok eol;
  eol.kind = TokenKind::T_EOF_SYMBOL;
  eol.sourceFile = lastTok->sourceFile;
  eol.offset = lastTok->offset + lastTok->length;
  line.push_back(eol);
  return line;
}

auto Preprocessor::Private::prepareConstantExpression(const Tok* ts,
                                                      const Tok* lineEnd)
    -> TokVector {
  auto line = copyLine(ts, lineEnd);
  dependencies_.clear();
  return expandTokens(line.data(), line.data() + line.size(), true);
}

auto Preprocessor::Private::evaluateConstantExpression(const Tok*& ts,
                                                       const Tok* end) -> long {
  return conditionalExpression(ts, end);
}

auto Preprocessor::Private::conditionalExpression(const Tok*& ts,
                                                  const Tok* end) -> long {
  if (ts >= end) return 0;
  const auto value = binaryExpression(ts, end);
  if (ts < end && ts->is(TokenKind::T_QUESTION)) {
    ++ts;
    const auto iftrue = conditionalExpression(ts, end);
    if (ts < end && ts->is(TokenKind::T_COLON)) ++ts;
    const auto iffalse = conditionalExpression(ts, end);
    return value ? iftrue : iffalse;
  }
  return value;
}

static auto prec(const Tok* ts, const Tok* end) -> int {
  enum Prec {
    kLogicalOr,
    kLogicalAnd,
    kInclusiveOr,
    kExclusiveOr,
    kAnd,
    kEquality,
    kRelational,
    kShift,
    kAdditive,
    kMultiplicative,
  };

  if (ts >= end) return -1;

  switch (ts->kind) {
    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PERCENT:
      return Prec::kMultiplicative;
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
      return Prec::kAdditive;
    case TokenKind::T_LESS_LESS:
    case TokenKind::T_GREATER_GREATER:
      return Prec::kShift;
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_GREATER:
      return Prec::kRelational;
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
      return Prec::kEquality;
    case TokenKind::T_AMP:
      return Prec::kAnd;
    case TokenKind::T_CARET:
      return Prec::kExclusiveOr;
    case TokenKind::T_BAR:
      return Prec::kInclusiveOr;
    case TokenKind::T_AMP_AMP:
      return Prec::kLogicalAnd;
    case TokenKind::T_BAR_BAR:
      return Prec::kLogicalOr;
    default:
      return -1;
  }
}

auto Preprocessor::Private::binaryExpression(const Tok*& ts, const Tok* end)
    -> long {
  auto e = unaryExpression(ts, end);
  return binaryExpressionHelper(ts, end, e, 0);
}

auto Preprocessor::Private::binaryExpressionHelper(const Tok*& ts,
                                                   const Tok* end, long lhs,
                                                   int minPrec) -> long {
  while (prec(ts, end) >= minPrec) {
    const auto p = prec(ts, end);
    const auto op = ts->kind;
    ++ts;
    auto rhs = unaryExpression(ts, end);
    while (prec(ts, end) > p) {
      rhs = binaryExpressionHelper(ts, end, rhs, prec(ts, end));
    }
    switch (op) {
      case TokenKind::T_STAR:
        lhs = lhs * rhs;
        break;
      case TokenKind::T_SLASH:
        lhs = rhs != 0 ? lhs / rhs : 0;
        break;
      case TokenKind::T_PERCENT:
        lhs = rhs != 0 ? lhs % rhs : 0;
        break;
      case TokenKind::T_PLUS:
        lhs = lhs + rhs;
        break;
      case TokenKind::T_MINUS:
        lhs = lhs - rhs;
        break;
      case TokenKind::T_LESS_LESS:
        lhs = lhs << rhs;
        break;
      case TokenKind::T_GREATER_GREATER:
        lhs = lhs >> rhs;
        break;
      case TokenKind::T_LESS_EQUAL:
        lhs = lhs <= rhs;
        break;
      case TokenKind::T_GREATER_EQUAL:
        lhs = lhs >= rhs;
        break;
      case TokenKind::T_LESS:
        lhs = lhs < rhs;
        break;
      case TokenKind::T_GREATER:
        lhs = lhs > rhs;
        break;
      case TokenKind::T_EQUAL_EQUAL:
        lhs = lhs == rhs;
        break;
      case TokenKind::T_EXCLAIM_EQUAL:
        lhs = lhs != rhs;
        break;
      case TokenKind::T_AMP:
        lhs = lhs & rhs;
        break;
      case TokenKind::T_CARET:
        lhs = lhs ^ rhs;
        break;
      case TokenKind::T_BAR:
        lhs = lhs | rhs;
        break;
      case TokenKind::T_AMP_AMP:
        lhs = lhs && rhs;
        break;
      case TokenKind::T_BAR_BAR:
        lhs = lhs || rhs;
        break;
      default:
        cxx_runtime_error(
            std::format("invalid operator '{}'", Token::spell(op)));
    }
  }
  return lhs;
}

auto Preprocessor::Private::unaryExpression(const Tok*& ts, const Tok* end)
    -> long {
  if (ts >= end) return 0;
  if (ts->is(TokenKind::T_MINUS)) {
    ++ts;
    return -unaryExpression(ts, end);
  }
  if (ts->is(TokenKind::T_PLUS)) {
    ++ts;
    return unaryExpression(ts, end);
  }
  if (ts->is(TokenKind::T_TILDE)) {
    ++ts;
    return ~unaryExpression(ts, end);
  }
  if (ts->is(TokenKind::T_EXCLAIM)) {
    ++ts;
    return !unaryExpression(ts, end);
  }
  return primaryExpression(ts, end);
}

auto Preprocessor::Private::primaryExpression(const Tok*& ts, const Tok* end)
    -> long {
  if (ts >= end) return 0;
  const auto& tk = *ts;
  auto text = getText(tk);

  if (tk.is(TokenKind::T_INTEGER_LITERAL)) {
    ++ts;
    return IntegerLiteral::Components::from(text).value;
  } else if (tk.is(TokenKind::T_CHARACTER_LITERAL)) {
    ++ts;
    return CharLiteral::Components::from(text).value;
  } else if (tk.is(TokenKind::T_LPAREN)) {
    ++ts;
    auto result = conditionalExpression(ts, end);
    if (ts < end && ts->is(TokenKind::T_RPAREN)) ++ts;
    return result;
  } else if (tk.is(TokenKind::T_IDENTIFIER) && text == "true") {
    ++ts;
    return 1;
  } else if (tk.is(TokenKind::T_IDENTIFIER) && text == "false") {
    ++ts;
    return 0;
  } else if (tk.is(TokenKind::T_PP_INTERNAL_VARIABLE)) {
    for (const auto& dep : dependencies_) {
      if (dep.local == text) {
        ++ts;
        return dep.exists;
      }
    }
  }

  ++ts;
  return 0;
}

auto Preprocessor::Private::parseArguments(const Tok* pos, const Tok* end,
                                           std::size_t formalCount,
                                           bool ignoreComma) -> ParsedArgs {
  ParsedArgs result;
  if (pos >= end || pos->isNot(TokenKind::T_LPAREN)) return result;
  auto lparen = pos;
  ++pos;

  if (pos < end && pos->is(TokenKind::T_RPAREN) &&
      !(ignoreComma && formalCount == 0)) {
    ++pos;
    result.rest = pos;
    return result;
  }

  auto skipArg = [&](bool skipComma) -> const Tok* {
    int depth = 0;
    while (pos < end) {
      if (pos->is(TokenKind::T_LPAREN)) {
        ++depth;
      } else if (pos->is(TokenKind::T_RPAREN)) {
        if (depth == 0) break;
        --depth;
      } else if (pos->is(TokenKind::T_COMMA)) {
        if (depth == 0 && !skipComma) break;
      }
      ++pos;
    }
    return pos;
  };

  auto startArg = pos;
  skipArg(ignoreComma && formalCount == 0);
  result.args.push_back({startArg, pos});

  while (pos < end && pos->is(TokenKind::T_COMMA)) {
    ++pos;
    startArg = pos;
    skipArg(ignoreComma && result.args.size() >= formalCount);
    result.args.push_back({startArg, pos});
  }

  if (pos < end && pos->is(TokenKind::T_RPAREN)) ++pos;

  result.rest = pos;
  return result;
}

auto Preprocessor::Private::stringize(const Tok* begin, const Tok* end) -> Tok {
  std::string s;
  for (auto it = begin; it < end; ++it) {
    if (!s.empty() && (it->space || it->bol)) s += ' ';
    s += getText(*it);
  }

  std::string o;
  o += '"';
  for (auto c : s) {
    if (c == '\\')
      o += "\\\\";
    else if (c == '"')
      o += "\\\"";
    else
      o += c;
  }
  o += '"';

  auto tk = genTok(TokenKind::T_STRING_LITERAL, o);
  if (begin < end) {
    tk.sourceFile = begin->sourceFile;
    tk.offset = begin->offset;
  }
  return tk;
}

auto Preprocessor::Private::merge(const Tok& left, const Tok& right) -> Tok {
  auto leftText = getText(left);
  auto rightText = getText(right);
  auto mergedText = std::string(leftText) + std::string(rightText);
  Lexer lex(std::string_view(mergedText), language_);
  lex.setPreprocessing(true);
  lex.next();
  auto tok = genTok(lex.tokenKind(), lex.tokenText());
  tok.sourceFile = left.sourceFile;
  tok.offset = left.offset;
  tok.noexpand = false;
  return tok;
}

auto Preprocessor::Private::lookupMacro(const Tok& tk) const
    -> std::pair<const Macro*, const cxx::Identifier*> {
  if (tk.isNot(TokenKind::T_IDENTIFIER)) return {nullptr, nullptr};
  if (tk.noexpand) return {nullptr, nullptr};

  auto text = getText(tk);
  if (auto it = macros_.find(text); it != macros_.end()) {
    auto ident = control_->getIdentifier(text);
    if (!isTainted(ident)) {
      return {&it->second, ident};
    }
  }
  return {nullptr, nullptr};
}

auto Preprocessor::Private::lookupMacroArgument(
    const Tok*& ts, const Tok* lineEnd, const Macro* m,
    const std::vector<TokRange>& actuals) -> std::optional<TokRange> {
  if (!isFunctionLikeMacro(*m)) return std::nullopt;
  const auto* macro = std::get_if<FunctionMacro>(m);
  if (!macro) return std::nullopt;

  if (ts >= lineEnd || ts->isNot(TokenKind::T_IDENTIFIER)) return std::nullopt;

  auto text = getText(*ts);

  if (macro->variadic) {
    if (text == "__VA_ARGS__") {
      ++ts;
      if (actuals.size() > macro->formals.size()) {
        return actuals.back();
      }
      return TokRange{nullptr, nullptr};
    }

    if (text == "__VA_OPT__" && ts + 1 < lineEnd &&
        (ts + 1)->is(TokenKind::T_LPAREN)) {
      ++ts;
      ++ts;
      int depth = 1;
      while (ts < lineEnd && depth > 0) {
        if (ts->is(TokenKind::T_LPAREN))
          ++depth;
        else if (ts->is(TokenKind::T_RPAREN))
          --depth;
        if (depth > 0) ++ts;
      }
      if (ts < lineEnd && ts->is(TokenKind::T_RPAREN)) ++ts;

      return TokRange{nullptr, nullptr};
    }
  }

  for (std::size_t i = 0; i < macro->formals.size(); ++i) {
    if (macro->formals[i] == text) {
      ++ts;
      if (i < actuals.size()) return actuals[i];
      return TokRange{nullptr, nullptr};
    }
  }

  return std::nullopt;
}

auto Preprocessor::Private::substitute(
    const Tok& pointOfSubstitution, const Macro* macro,
    const std::vector<TokRange>& actuals,
    const std::vector<TokRange>& expandedActuals) -> TokVector {
  TokVector os;

  auto appendToken = [&](const Tok& tk) {
    if (tk.isFromMacroBody) {
      auto copyTk = tk;
      copyTk.sourceFile = pointOfSubstitution.sourceFile;
      copyTk.offset = pointOfSubstitution.offset;
      copyTk.length = pointOfSubstitution.length;
      os.push_back(copyTk);
    } else {
      os.push_back(tk);
    }
  };

  auto appendTokens = [&](const Tok* begin, const Tok* end) {
    for (auto it = begin; it < end; ++it) appendToken(*it);
  };

  const TokVector* macroBody = getMacroBody(*macro);
  if (!macroBody) return os;

  const Tok* ts = macroBody->data();
  const Tok* tsEnd = ts + macroBody->size();

  while (ts < tsEnd && ts->isNot(TokenKind::T_EOF_SYMBOL)) {
    if (ts->is(TokenKind::T_HASH) && ts + 1 < tsEnd &&
        (ts + 1)->is(TokenKind::T_IDENTIFIER)) {
      const auto* saved = ts;
      ++ts;
      if (auto actual = lookupMacroArgument(ts, tsEnd, macro, actuals)) {
        auto [ab, ae] = *actual;
        if (ab && ab < ae) {
          appendToken(stringize(ab, ae));
        }
        continue;
      }
      ts = saved;
    }

    if (ts->is(TokenKind::T_HASH_HASH) && ts + 1 < tsEnd &&
        (ts + 1)->is(TokenKind::T_IDENTIFIER)) {
      const auto* saved = ts;
      ++ts;
      if (auto actual = lookupMacroArgument(ts, tsEnd, macro, actuals)) {
        auto [ab, ae] = *actual;
        if (ab && ab < ae && !os.empty()) {
          os.back() = merge(os.back(), *ab);
          for (auto it = ab + 1; it < ae; ++it) appendToken(*it);
        }
        continue;
      }
      ts = saved;
    }

    if (ts->is(TokenKind::T_HASH_HASH) && ts + 1 < tsEnd) {
      if (!os.empty()) {
        os.back() = merge(os.back(), *(ts + 1));
      }
      ts += 2;
      continue;
    }

    if (ts->is(TokenKind::T_COMMA) && ts + 1 < tsEnd &&
        (ts + 1)->is(TokenKind::T_HASH_HASH) && ts + 2 < tsEnd) {
      auto commaText = getText(*(ts + 2));
      if (commaText == "__VA_ARGS__") {
        auto comma = *ts;
        ts += 2;
        if (auto actual = lookupMacroArgument(ts, tsEnd, macro, actuals)) {
          auto [ab, ae] = *actual;
          if (ab && ab < ae) {
            appendToken(comma);
            appendTokens(ab, ae);
          }
          continue;
        }
      }
    }

    if (ts->is(TokenKind::T_IDENTIFIER) && ts + 1 < tsEnd &&
        (ts + 1)->is(TokenKind::T_HASH_HASH)) {
      if (auto actual = lookupMacroArgument(ts, tsEnd, macro, actuals)) {
        auto [ab, ae] = *actual;
        if (ab && ab < ae) {
          appendTokens(ab, ae);
        } else {
          auto emptyTk = genTok(TokenKind::T_IDENTIFIER, "");
          appendToken(emptyTk);
        }
        continue;
      }
    }

    if (ts->is(TokenKind::T_IDENTIFIER) && getText(*ts) == "__VA_OPT__" &&
        ts + 1 < tsEnd && (ts + 1)->is(TokenKind::T_LPAREN)) {
      const auto* funcMacro = std::get_if<FunctionMacro>(macro);
      if (funcMacro && funcMacro->variadic) {
        ++ts;
        ++ts;
        auto contentBegin = ts;
        int depth = 1;
        while (ts < tsEnd && depth > 0) {
          if (ts->is(TokenKind::T_LPAREN))
            ++depth;
          else if (ts->is(TokenKind::T_RPAREN))
            --depth;
          if (depth > 0) ++ts;
        }
        auto contentEnd = ts;
        if (ts < tsEnd && ts->is(TokenKind::T_RPAREN)) ++ts;

        bool hasVarArgs = false;
        if (actuals.size() > funcMacro->formals.size()) {
          auto [ab, ae] = actuals.back();
          if (ab != ae && ab != nullptr) hasVarArgs = true;
        }

        if (hasVarArgs) {
          TokVector contentTokens(contentBegin, contentEnd);
          Tok eol;
          eol.kind = TokenKind::T_EOF_SYMBOL;
          contentTokens.push_back(eol);

          const Tok* cTs = contentTokens.data();
          const Tok* cEnd = cTs + contentTokens.size();
          while (cTs < cEnd && cTs->isNot(TokenKind::T_EOF_SYMBOL)) {
            if (auto actual =
                    lookupMacroArgument(cTs, cEnd, macro, expandedActuals)) {
              auto [ab, ae] = *actual;
              if (ab && ab < ae) appendTokens(ab, ae);
              continue;
            }
            appendToken(*cTs);
            ++cTs;
          }
        }
        continue;
      }
    }

    if (auto actual = lookupMacroArgument(ts, tsEnd, macro, expandedActuals)) {
      auto [ab, ae] = *actual;
      if (ab && ab < ae) appendTokens(ab, ae);
      continue;
    }

    appendToken(*ts);
    ++ts;
  }

  for (auto& tok : os) {
    if (tok.is(TokenKind::T_IDENTIFIER) && !tok.noexpand) {
      auto text = getText(tok);
      if (!text.empty() && macros_.contains(text) &&
          isTainted(control_->getIdentifier(text))) {
        tok.noexpand = true;
      }
    }
  }

  return os;
}

auto Preprocessor::Private::expandMacro(Cursor& cursor) -> bool {
  auto& tk = cursor.current();
  auto [macro, ident] = lookupMacro(tk);
  if (!macro) return false;

  struct ExpandMacro {
    Private& self;
    Cursor& cursor;
    const Macro* macro = nullptr;
    const cxx::Identifier* ident = nullptr;

    auto operator()(const ObjectMacro&) -> bool {
      return self.expandObjectLikeMacro(cursor, macro, ident);
    }

    auto operator()(const FunctionMacro&) -> bool {
      const Tok* peek = cursor.pos + 1;
      while (peek < cursor.end && peek->is(TokenKind::T_IDENTIFIER) &&
             peek->noexpand)
        ++peek;
      if (peek >= cursor.end) {
        return self.expandFunctionLikeMacroAcrossBoundary(macro, ident);
      }
      if (peek->isNot(TokenKind::T_LPAREN)) return false;
      return self.expandFunctionLikeMacro(cursor, macro, ident);
    }

    auto operator()(const BuiltinObjectMacro& m) -> bool {
      MacroExpansionContext ctx{
          .tok = &cursor.current(),
          .pos = cursor.pos + 1,
          .end = cursor.end,
      };
      auto expanded = m.expand(ctx);
      cursor.advance();

      if (!expanded.empty()) {
        Cursor ec;
        ec.kind = Cursor::ExpansionCursor;
        ec.ownedTokens = std::move(expanded);
        ec.initFromOwned();
        self.cursors_.push_back(std::move(ec));
      }
      return true;
    }

    auto operator()(const BuiltinFunctionMacro& m) -> bool {
      const Tok* peek = cursor.pos + 1;
      if (peek >= cursor.end || peek->isNot(TokenKind::T_LPAREN)) return false;

      auto argsResult =
          self.parseArguments(cursor.pos + 1, cursor.end, 0, true);

      MacroExpansionContext ctx{
          .tok = &cursor.current(),
          .pos = cursor.pos + 1,
          .end = argsResult.rest ? argsResult.rest : cursor.end,
      };
      auto expanded = m.expand(ctx);

      cursor.pos = argsResult.rest ? argsResult.rest : cursor.end;

      if (!expanded.empty()) {
        Cursor ec;
        ec.kind = Cursor::ExpansionCursor;
        ec.ownedTokens = std::move(expanded);
        ec.initFromOwned();
        self.cursors_.push_back(std::move(ec));
      }
      return true;
    }
  };

  return std::visit(ExpandMacro{*this, cursor, macro, ident}, *macro);
}

auto Preprocessor::Private::expandObjectLikeMacro(Cursor& cursor,
                                                  const Macro* m,
                                                  const cxx::Identifier* ident)
    -> bool {
  const auto& tk = cursor.current();

  taint(ident);

  auto expanded = substitute(tk, m, {}, {});

  if (!expanded.empty()) {
    expanded.front().space = tk.space;
    expanded.front().bol = tk.bol;
  }

  cursor.advance();

  Cursor ec;
  ec.kind = Cursor::ExpansionCursor;
  ec.untaintOnPop = ident;
  ec.ownedTokens = std::move(expanded);
  ec.initFromOwned();
  cursors_.push_back(std::move(ec));

  return true;
}

auto Preprocessor::Private::expandFunctionLikeMacro(
    Cursor& cursor, const Macro* m, const cxx::Identifier* ident) -> bool {
  const auto* macro = std::get_if<FunctionMacro>(m);
  if (!macro) return false;

  auto cursorIndex = cursors_.size() - 1;

  auto tk = cursor.current();

  auto argsResult = parseArguments(cursor.pos + 1, cursor.end,
                                   macro->formals.size(), macro->variadic);
  if (!argsResult.rest) return false;

  auto actuals = argsResult.args;
  auto rest = argsResult.rest;

  std::vector<TokRange> expandedArgs;
  expandedArgs.reserve(actuals.size());

  std::vector<TokVector> expandedArgStorage;
  expandedArgStorage.reserve(actuals.size());
  for (const auto& [ab, ae] : actuals) {
    expandedArgStorage.push_back(expandTokens(ab, ae, false));
    auto& v = expandedArgStorage.back();
    expandedArgs.push_back({v.data(), v.data() + v.size()});
  }

  taint(ident);

  auto expanded = substitute(tk, m, actuals, expandedArgs);

  if (!expanded.empty()) {
    expanded.front().space = tk.space;
    expanded.front().bol = tk.bol;
  }

  cursors_[cursorIndex].pos = rest;

  Cursor ec;
  ec.kind = Cursor::ExpansionCursor;
  ec.untaintOnPop = ident;
  ec.ownedTokens = std::move(expanded);
  ec.initFromOwned();
  cursors_.push_back(std::move(ec));

  return true;
}

auto Preprocessor::Private::expandFunctionLikeMacroAcrossBoundary(
    const Macro* macro, const cxx::Identifier* ident) -> bool {
  auto curIdx = cursors_.size() - 1;
  if (curIdx == 0) return false;  // no parent cursor

  auto& parent = cursors_[curIdx - 1];
  const Tok* parentPeek = parent.pos;
  while (parentPeek < parent.end && parentPeek->is(TokenKind::T_IDENTIFIER) &&
         parentPeek->noexpand)
    ++parentPeek;
  if (parentPeek >= parent.end || parentPeek->isNot(TokenKind::T_LPAREN))
    return false;

  auto& cur = cursors_[curIdx];
  TokVector combined;
  combined.reserve(static_cast<std::size_t>(cur.end - cur.pos) +
                   static_cast<std::size_t>(parent.end - parent.pos));
  combined.insert(combined.end(), cur.pos, cur.end);
  combined.insert(combined.end(), parent.pos, parent.end);

  auto untaintId = cur.untaintOnPop;
  cursors_.pop_back();
  if (untaintId) untaint(untaintId);

  auto& target = cursors_.back();
  target.ownedTokens = std::move(combined);
  target.initFromOwned();

  return expandFunctionLikeMacro(target, macro, ident);
}

auto Preprocessor::Private::expandTokens(const Tok* begin, const Tok* end,
                                         bool inConditionalExpression)
    -> TokVector {
  TokVector result;

  auto baseDepth = cursors_.size();

  Cursor tempCursor;
  tempCursor.kind = Cursor::ExpansionCursor;
  tempCursor.begin = begin;
  tempCursor.end = end;
  tempCursor.pos = begin;

  cursors_.push_back(std::move(tempCursor));

  while (cursors_.size() > baseDepth) {
    auto& cur = cursors_.back();
    if (cur.atEnd() || cur.current().is(TokenKind::T_EOF_SYMBOL)) {
      auto id = cur.untaintOnPop;
      cursors_.pop_back();
      if (id) untaint(id);
      continue;
    }

    expandOne(cursors_.back(), inConditionalExpression,
              [&](const Tok& tok) { result.push_back(tok); });
  }

  return result;
}

template <typename EmitToken>
void Preprocessor::Private::expandOne(Cursor& cursor,
                                      bool inConditionalExpression,
                                      const EmitToken& emitToken) {
  if (cursor.atEnd()) return;

  if (inConditionalExpression) {
    if (replaceIsDefinedMacro(cursor, inConditionalExpression, emitToken))
      return;
  }

  if (cursor.current().is(TokenKind::T_IDENTIFIER) &&
      !cursor.current().noexpand) {
    if (expandMacro(cursor)) return;
  }

  emitToken(cursor.current());
  cursor.advance();
}

template <typename EmitToken>
auto Preprocessor::Private::replaceIsDefinedMacro(Cursor& cursor,
                                                  bool inConditionalExpression,
                                                  const EmitToken& emitToken)
    -> bool {
  if (cursor.atEnd()) return false;
  auto text = getText(cursor.current());
  if (text != "defined") return false;

  auto start = cursor.current();
  cursor.advance();

  bool value = false;

  if (!cursor.atEnd() && cursor.current().is(TokenKind::T_LPAREN)) {
    cursor.advance();
    if (!cursor.atEnd()) {
      value = isDefined(cursor.current());
      cursor.advance();
    }
    if (!cursor.atEnd() && cursor.current().is(TokenKind::T_RPAREN))
      cursor.advance();
  } else if (!cursor.atEnd()) {
    value = isDefined(cursor.current());
    cursor.advance();
  }

  auto tk = genTok(TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
  tk.sourceFile = start.sourceFile;
  tk.space = start.space;
  tk.bol = start.bol;
  emitToken(tk);

  return true;
}

template <typename EmitToken>
auto Preprocessor::Private::expand(const EmitToken& emitToken)
    -> PreprocessingState {
  if (cursors_.empty()) return ProcessingComplete{};

  auto cursor = std::move(cursors_.back());
  cursors_.pop_back();

  if (cursor.kind != Cursor::FileCursor) {
    cursors_.push_back(std::move(cursor));
    while (!cursors_.empty()) {
      auto& cur = cursors_.back();
      if (cur.atEnd() || cur.current().is(TokenKind::T_EOF_SYMBOL)) {
        auto id = cur.untaintOnPop;
        cursors_.pop_back();
        if (id) untaint(id);
        continue;
      }
      expandOne(cur, false, emitToken);
    }
    return ProcessingComplete{};
  }

  auto source = cursor.sourceFile;
  currentFileName_ = source->fileName;
  currentPath_ = cursor.currentPath;
  includeDepth_ = cursor.includeDepth;

  while (!cursor.atEnd()) {
    if (cursor.current().is(TokenKind::T_EOF_SYMBOL)) break;

    const auto [skipping, evaluating] = state();

    if (cursor.current().bol && cursor.current().is(TokenKind::T_HASH)) {
      auto directiveStart = cursor.pos;
      cursor.advance();

      auto lineStart = cursor.pos;
      skipLine(cursor.pos, cursor.end);
      auto lineEnd = cursor.pos;

      auto parsedDirective = parseDirective(source, directiveStart, lineEnd);

      if (auto pi = std::get_if<ParsedIncludeDirective>(&parsedDirective)) {
        PendingInclude nextState{
            .preprocessor = *preprocessor_,
            .include = pi->header,
            .isIncludeNext = pi->includeNext,
            .loc = const_cast<void*>(static_cast<const void*>(pi->loc)),
        };

        cursors_.push_back(std::move(cursor));

        return nextState;
      } else if (auto pif = std::get_if<ParsedIfDirective>(&parsedDirective)) {
        continuation_ = std::move(pif->resume);

        cursors_.push_back(std::move(cursor));

        std::vector<PendingHasIncludes::Request> dependencies;
        for (auto& dep : dependencies_) {
          dependencies.push_back({
              .include = dep.include,
              .isIncludeNext = dep.isIncludeNext,
              .exists = dep.exists,
          });
        }

        return PendingHasIncludes{
            .preprocessor = *preprocessor_,
            .requests = dependencies,
        };
      }
    } else if (skipping) {
      cursor.advance();
      skipLine(cursor.pos, cursor.end);
    } else {
      cursors_.push_back(std::move(cursor));

      while (!cursors_.empty()) {
        auto& cur = cursors_.back();
        if (cur.atEnd() || cur.current().is(TokenKind::T_EOF_SYMBOL)) {
          auto id = cur.untaintOnPop;
          auto kind = cur.kind;
          cursors_.pop_back();
          if (id) untaint(id);
          if (kind == Cursor::FileCursor) break;
          continue;
        }

        if (cur.kind == Cursor::FileCursor && cur.current().bol &&
            cur.current().is(TokenKind::T_HASH)) {
          break;
        }

        if (cur.kind == Cursor::FileCursor && std::get<0>(state())) {
          break;
        }

        expandOne(cur, false, emitToken);
      }

      if (cursors_.empty()) return ProcessingComplete{};

      cursor = std::move(cursors_.back());
      cursors_.pop_back();

      if (cursor.kind != Cursor::FileCursor) {
        cursors_.push_back(std::move(cursor));
        continue;
      }

      source = cursor.sourceFile;
      currentFileName_ = source->fileName;
      currentPath_ = cursor.currentPath;
      includeDepth_ = cursor.includeDepth;
    }
  }

  if (cursors_.empty()) return ProcessingComplete{};
  return CanContinuePreprocessing{};
}

auto Preprocessor::Private::parseDirective(SourceFile* source,
                                           const Tok* directiveLine,
                                           const Tok* directiveEnd)
    -> ParsedDirective {
  auto ts = directiveLine + 1;
  if (ts >= directiveEnd) return std::monostate{};

  if (ts->isNot(TokenKind::T_IDENTIFIER)) return std::monostate{};

  dependencies_.clear();

  auto directiveText = getText(*ts);
  const auto directiveKind =
      classifyDirective(directiveText.data(), int(directiveText.length()));

  ++ts;

  const auto [skipping, evaluating] = state();

  switch (directiveKind) {
    case PreprocessorDirectiveKind::T_INCLUDE_NEXT:
    case PreprocessorDirectiveKind::T_INCLUDE: {
      if (skipping) break;
      auto includeDirective =
          parseIncludeDirective(directiveLine + 1, ts, directiveEnd);
      if (includeDirective.has_value()) {
        return *includeDirective;
      }
      break;
    }

    case PreprocessorDirectiveKind::T_DEFINE: {
      if (skipping) break;
      defineMacro(ts, directiveEnd);
      break;
    }

    case PreprocessorDirectiveKind::T_UNDEF: {
      if (skipping) break;
      if (ts >= directiveEnd || ts->bol) {
        error(ts < directiveEnd ? ts : directiveLine,
              std::format("missing macro name"));
        break;
      }
      if (ts->is(TokenKind::T_IDENTIFIER)) {
        auto name = getText(*ts);
        auto it = macros_.find(name);
        if (it != macros_.end()) macros_.erase(it);
      }
      break;
    }

    case PreprocessorDirectiveKind::T_IFDEF: {
      if (ts < directiveEnd) {
        const auto value = isDefined(*ts);
        if (value)
          pushState(std::tuple(skipping, false));
        else
          pushState(std::tuple(true, !skipping));
      }
      break;
    }

    case PreprocessorDirectiveKind::T_IFNDEF: {
      if (ts < directiveEnd) {
        const auto value = !isDefined(*ts);
        if (value)
          pushState(std::tuple(skipping, false));
        else
          pushState(std::tuple(true, !skipping));
      }
      break;
    }

    case PreprocessorDirectiveKind::T_IF: {
      if (skipping) {
        pushState(std::tuple(true, false));
      } else {
        auto expression = prepareConstantExpression(ts, directiveEnd);

        auto resume = [expression = std::move(expression), skipping,
                       this]() mutable -> std::optional<PreprocessingState> {
          const Tok* ep = expression.data();
          const Tok* ee = ep + expression.size();
          const auto value = evaluateConstantExpression(ep, ee);
          if (value)
            pushState(std::tuple(skipping, false));
          else
            pushState(std::tuple(true, !skipping));
          return std::nullopt;
        };

        if (dependencies_.empty()) {
          resume();
        } else {
          return ParsedIfDirective{.resume = std::move(resume)};
        }
      }
      break;
    }

    case PreprocessorDirectiveKind::T_ELIF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else {
        auto expression = prepareConstantExpression(ts, directiveEnd);

        auto resume = [expression = std::move(expression), evaluating,
                       this]() mutable -> std::optional<PreprocessingState> {
          const Tok* ep = expression.data();
          const Tok* ee = ep + expression.size();
          const auto value = evaluateConstantExpression(ep, ee);
          if (value)
            setState(std::tuple(!evaluating, false));
          else
            setState(std::tuple(true, evaluating));
          return std::nullopt;
        };

        if (dependencies_.empty()) {
          resume();
        } else {
          return ParsedIfDirective{.resume = std::move(resume)};
        }
      }
      break;
    }

    case PreprocessorDirectiveKind::T_ELIFDEF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else if (ts < directiveEnd) {
        const auto value = isDefined(*ts);
        if (value)
          setState(std::tuple(!evaluating, false));
        else
          setState(std::tuple(true, evaluating));
      }
      break;
    }

    case PreprocessorDirectiveKind::T_ELIFNDEF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else if (ts < directiveEnd) {
        const auto value = isDefined(*ts);
        if (!value)
          setState(std::tuple(!evaluating, false));
        else
          setState(std::tuple(true, evaluating));
      }
      break;
    }

    case PreprocessorDirectiveKind::T_ELSE: {
      setState(std::tuple(!evaluating, false));
      break;
    }

    case PreprocessorDirectiveKind::T_ENDIF: {
      popState();
      if (evaluating_.empty()) {
        error(directiveLine + 1, "unexpected '#endif'");
      }
      if (!source->headerGuardName.empty() &&
          evaluating_.size() == source->headerProtectionLevel) {
        bool hasTrailingTokens = false;
        for (auto p = ts; p < directiveEnd; ++p) {
          if (p->is(TokenKind::T_EOF_SYMBOL)) break;
          if (p->bol) break;
          hasTrailingTokens = true;
          break;
        }
        if (hasTrailingTokens) {
          ifndefProtectedFiles_.erase(currentFileName_);
        }
      }
      break;
    }

    case PreprocessorDirectiveKind::T_LINE: {
      break;
    }

    case PreprocessorDirectiveKind::T_PRAGMA: {
      if (skipping) break;
      break;
    }

    case PreprocessorDirectiveKind::T_ERROR: {
      if (skipping) break;
      std::ostringstream out;
      printLine(directiveLine, directiveEnd, out, false);
      error(directiveLine + 1, std::format("{}", out.str()));
      break;
    }

    case PreprocessorDirectiveKind::T_WARNING: {
      if (skipping) break;
      std::ostringstream out;
      printLine(directiveLine, directiveEnd, out, false);
      warning(directiveLine + 1, std::format("{}", out.str()));
      break;
    }

    default:
      break;
  }

  return std::monostate{};
}

auto Preprocessor::Private::parseIncludeDirective(const Tok* directive,
                                                  const Tok* ts,
                                                  const Tok* lineEnd)
    -> std::optional<ParsedIncludeDirective> {
  if (ts < lineEnd && ts->is(TokenKind::T_IDENTIFIER)) {
    auto expanded = expandTokens(ts, lineEnd, false);
    if (!expanded.empty()) {
      const Tok* ep = expanded.data();
      const Tok* ee = ep + expanded.size();

      auto directiveText = getText(*directive);
      const bool isIncludeNext = directiveText == "include_next";

      const Tok* loc = (ep < ee) ? ep : directive;

      if (auto headerFile = parseHeaderName(ep, ee); headerFile.has_value()) {
        return ParsedIncludeDirective{
            .header = *headerFile,
            .includeNext = isIncludeNext,
            .loc = loc,
        };
      }
      return std::nullopt;
    }
  }

  const Tok* loc = ts;
  if (ts >= lineEnd || ts->is(TokenKind::T_EOF_SYMBOL)) loc = directive;

  auto directiveText = getText(*directive);
  const bool isIncludeNext = directiveText == "include_next";

  if (auto headerFile = parseHeaderName(ts, lineEnd); headerFile.has_value()) {
    return ParsedIncludeDirective{
        .header = *headerFile,
        .includeNext = isIncludeNext,
        .loc = loc,
    };
  }

  return std::nullopt;
}

auto Preprocessor::Private::parseHeaderName(const Tok*& ts, const Tok* lineEnd)
    -> std::optional<Include> {
  if (ts < lineEnd && ts->is(TokenKind::T_STRING_LITERAL)) {
    auto text = getText(*ts);
    auto file = text.substr(1, text.length() - 2);
    ++ts;
    return QuoteInclude(std::string(file));
  }

  if (ts < lineEnd && ts->is(TokenKind::T_LESS)) {
    ++ts;
    std::string file;
    while (ts < lineEnd && ts->isNot(TokenKind::T_EOF_SYMBOL) && !ts->bol) {
      if (ts->is(TokenKind::T_GREATER)) {
        ++ts;
        break;
      }
      file += getText(*ts);
      ++ts;
    }
    return SystemInclude(file);
  }

  return std::nullopt;
}

void Preprocessor::Private::finalizeToken(std::vector<Token>& tokens,
                                          const Tok& tk) {
  auto kind = tk.kind;
  const auto fileId = tk.sourceFile;
  TokenValue value{};
  auto text = getText(tk);

  if (tk.sourceFile == 1 && codeCompletionLocation_.has_value()) {
    if (codeCompletionOffset_ < tk.offset ||
        (codeCompletionOffset_ >= tk.offset &&
         codeCompletionOffset_ < tk.offset + tk.length)) {
      auto& completionToken =
          tokens.emplace_back(TokenKind::T_CODE_COMPLETION, tk.offset, 0);
      completionToken.setFileId(fileId);
      codeCompletionLocation_ = std::nullopt;
    }
  }

  switch (tk.kind) {
    case TokenKind::T_IDENTIFIER: {
      kind = Lexer::classifyKeyword(text, language_);
      if (kind == TokenKind::T_IDENTIFIER) {
        value.idValue = control_->getIdentifier(text);
      }
      break;
    }

    case TokenKind::T_CHARACTER_LITERAL:
      value.literalValue = control_->charLiteral(text);
      break;

    case TokenKind::T_WIDE_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) return;
      value.literalValue = control_->wideStringLiteral(text);
      break;

    case TokenKind::T_UTF8_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) return;
      value.literalValue = control_->utf8StringLiteral(text);
      break;

    case TokenKind::T_UTF16_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) return;
      value.literalValue = control_->utf16StringLiteral(text);
      break;

    case TokenKind::T_UTF32_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) return;
      value.literalValue = control_->utf32StringLiteral(text);
      break;

    case TokenKind::T_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) return;
      value.literalValue = control_->stringLiteral(text);
      break;

    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
      value.literalValue = control_->stringLiteral(text);
      break;

    case TokenKind::T_INTEGER_LITERAL:
      value.literalValue = control_->integerLiteral(text);
      break;

    case TokenKind::T_FLOATING_POINT_LITERAL:
      value.literalValue = control_->floatLiteral(text);
      break;

    default:
      break;
  }

  if (tk.kind == TokenKind::T_GREATER_GREATER) {
    value.tokenKindValue = tk.kind;

    Token token(TokenKind::T_GREATER, tk.offset, 1);
    token.setFileId(fileId);
    token.setLeadingSpace(tk.space);
    token.setStartOfLine(tk.bol);
    tokens.push_back(token);

    token = Token(TokenKind::T_GREATER, tk.offset + 1, 1);
    token.setFileId(fileId);
    token.setLeadingSpace(false);
    token.setStartOfLine(false);
    tokens.push_back(token);
  } else {
    Token token(kind, tk.offset, tk.length, value);
    token.setFileId(fileId);
    token.setLeadingSpace(tk.space);
    token.setStartOfLine(tk.bol);
    tokens.push_back(token);
  }
}

auto Preprocessor::Private::checkPragmaOnceProtected(
    const TokVector& tokens) const -> bool {
  const Tok* ts = tokens.data();
  const Tok* end = ts + tokens.size();
  if (ts >= end) return false;
  if (ts->isNot(TokenKind::T_HASH)) return false;
  ++ts;
  if (ts >= end || ts->bol || getText(*ts) != "pragma") return false;
  ++ts;
  if (ts >= end || ts->bol || getText(*ts) != "once") return false;
  return true;
}

auto Preprocessor::Private::checkHeaderProtection(const TokVector& tokens) const
    -> std::string {
  const Tok* ts = tokens.data();
  const Tok* end = ts + tokens.size();
  if (ts >= end) return {};
  if (ts->isNot(TokenKind::T_HASH)) return {};
  ++ts;
  if (ts >= end || ts->bol) return {};
  if (getText(*ts) != "ifndef") return {};
  ++ts;
  if (ts >= end || ts->bol || ts->isNot(TokenKind::T_IDENTIFIER)) return {};
  auto protName = std::string(getText(*ts));
  ++ts;
  if (ts >= end || !ts->bol || ts->isNot(TokenKind::T_HASH)) return {};
  ++ts;
  if (ts >= end || ts->bol) return {};
  if (getText(*ts) != "define") return {};
  ++ts;
  if (ts >= end || ts->bol) return {};
  if (getText(*ts) != protName) return {};
  return protName;
}

auto Preprocessor::Private::resolve(const Include& include,
                                    bool isIncludeNext) const
    -> std::optional<ResolveResult> {
  if (!canResolveFiles_) return std::nullopt;

  const auto headerName = getHeaderName(include);
  const bool isQuoted = std::holds_alternative<QuoteInclude>(include);

  auto cacheKey = IncludeCacheKey{
      .isQuoted = isQuoted,
      .isIncludeNext = isIncludeNext,
      .currentPath = currentPath_.string(),
      .headerName = headerName,
  };

  if (auto cacheIt = resolveCache_.find(cacheKey);
      cacheIt != resolveCache_.end()) {
    return cacheIt->second;
  }

  auto result = resolveUncached(include, isIncludeNext, headerName, isQuoted);
  resolveCache_.emplace(std::move(cacheKey), result);
  return result;
}

auto Preprocessor::Private::resolveUncached(const Include& include,
                                            bool isIncludeNext,
                                            const std::string& headerName,
                                            bool isQuoted) const
    -> std::optional<ResolveResult> {
  auto tryCandidate =
      [&](const std::string& dir) -> std::optional<std::string> {
    auto candidate = fs::path(dir) / headerName;
    if (fileExists(candidate)) return candidate.string();
    return std::nullopt;
  };

  auto curDir = currentPath_.string();

  if (!isIncludeNext) {
    if (auto r = resolveNormal(tryCandidate, curDir, isQuoted)) return r;
  } else {
    if (auto r = resolveNext(tryCandidate, curDir, isQuoted)) return r;
  }

  return std::nullopt;
}

auto Preprocessor::Private::buildSearchDirs(const std::string& curDir,
                                            bool isQuoted) const
    -> std::vector<std::pair<const std::string*, bool>> {
  std::vector<std::pair<const std::string*, bool>> dirs;
  if (isQuoted) {
    if (!disableCurrentDirSearch_ && !curDir.empty())
      dirs.push_back({&curDir, false});
    for (auto it = quoteIncludePaths_.rbegin();
         it != quoteIncludePaths_.rend(); ++it)
      dirs.push_back({&*it, false});
  }
  for (auto it = userIncludePaths_.rbegin();
       it != userIncludePaths_.rend(); ++it)
    dirs.push_back({&*it, false});
  for (auto it = systemIncludePaths_.rbegin();
       it != systemIncludePaths_.rend(); ++it)
    dirs.push_back({&*it, true});
  return dirs;
}

template <typename TryCandidate>
auto Preprocessor::Private::resolveNormal(const TryCandidate& tryCandidate,
                                          const std::string& curDir,
                                          bool isQuoted) const
    -> std::optional<ResolveResult> {
  for (auto& [dir, isSys] : buildSearchDirs(curDir, isQuoted)) {
    if (auto r = tryCandidate(*dir))
      return ResolveResult{std::move(*r), isSys};
  }
  return std::nullopt;
}

template <typename TryCandidate>
auto Preprocessor::Private::resolveNext(const TryCandidate& tryCandidate,
                                        const std::string& curDir,
                                        bool isQuoted) const
    -> std::optional<ResolveResult> {
  auto dirs = buildSearchDirs(curDir, isQuoted);

  std::size_t startIndex = 0;
  for (std::size_t i = 0; i < dirs.size(); ++i) {
    if (*dirs[i].first == curDir) {
      startIndex = i + 1;
      break;
    }
  }

  for (std::size_t i = startIndex; i < dirs.size(); ++i) {
    if (auto r = tryCandidate(*dirs[i].first))
      return ResolveResult{std::move(*r), dirs[i].second};
  }
  return std::nullopt;
}

auto Preprocessor::Private::parseMacroDefinition(const Tok* ts,
                                                 const Tok* lineEnd) -> Macro {
  auto name = std::string(getText(*ts));
  ++ts;

  if (ts < lineEnd && ts->is(TokenKind::T_LPAREN) && !ts->space) {
    ++ts;

    std::vector<std::string> formals;
    bool variadic = false;

    if (ts < lineEnd && ts->isNot(TokenKind::T_RPAREN)) {
      if (ts->is(TokenKind::T_DOT_DOT_DOT)) {
        variadic = true;
        ++ts;
      } else {
        if (ts->is(TokenKind::T_IDENTIFIER)) {
          formals.push_back(std::string(getText(*ts)));
          ++ts;
        }
        while (ts < lineEnd && ts->is(TokenKind::T_COMMA)) {
          ++ts;
          if (ts < lineEnd && ts->is(TokenKind::T_DOT_DOT_DOT)) {
            variadic = true;
            ++ts;
            break;
          }
          if (ts < lineEnd && ts->is(TokenKind::T_IDENTIFIER)) {
            formals.push_back(std::string(getText(*ts)));
            ++ts;
          }
        }
        if (!variadic && ts < lineEnd && ts->is(TokenKind::T_DOT_DOT_DOT)) {
          variadic = true;
          ++ts;
        }
      }
      if (ts < lineEnd && ts->is(TokenKind::T_RPAREN)) ++ts;
    } else if (ts < lineEnd && ts->is(TokenKind::T_RPAREN)) {
      ++ts;
    }

    TokVector body;
    while (ts < lineEnd && ts->isNot(TokenKind::T_EOF_SYMBOL) && !ts->bol) {
      auto tok = *ts;
      tok.isFromMacroBody = true;
      if (!tok.dirty) {
        tok.dirty = true;
        tok.textIndex = addText(std::string(getText(*ts)));
      }
      body.push_back(tok);
      ++ts;
    }
    Tok eol;
    eol.kind = TokenKind::T_EOF_SYMBOL;
    body.push_back(eol);

    return FunctionMacro(name, std::move(formals), std::move(body), variadic);
  }

  TokVector body;
  while (ts < lineEnd && ts->isNot(TokenKind::T_EOF_SYMBOL) && !ts->bol) {
    auto tok = *ts;
    tok.isFromMacroBody = true;
    if (!tok.dirty) {
      tok.dirty = true;
      tok.textIndex = addText(std::string(getText(*ts)));
    }
    body.push_back(tok);
    ++ts;
  }
  Tok eol;
  eol.kind = TokenKind::T_EOF_SYMBOL;
  body.push_back(eol);

  return ObjectMacro(name, std::move(body));
}

void Preprocessor::Private::defineMacro(const Tok* ts, const Tok* lineEnd) {
  if (ts >= lineEnd || ts->isNot(TokenKind::T_IDENTIFIER)) return;

  auto macro = parseMacroDefinition(ts, lineEnd);
  auto name = std::string(getMacroName(macro));

  if (auto body = getMacroBody(macro); body && !body->empty()) {
    // strip leading space/bol from first body token — but body is in the
    // Macro variant, so we need to modify it in place after insert
  }

  if (auto it = macros_.find(name); it != macros_.end()) {
    auto previousBody = getMacroBody(it->second);
    auto newBody = getMacroBody(macro);
    if (previousBody && newBody &&
        !isSameBody(*newBody, *previousBody, texts_, sourceFiles_)) {
      warning(ts, std::format("'{}' macro redefined", name));
    }
    macros_.erase(it);
  }

  macros_.insert_or_assign(name, std::move(macro));
}

static auto wantSpace(TokenKind kind) -> bool {
  switch (kind) {
    case TokenKind::T_IDENTIFIER:
    case TokenKind::T_INTEGER_LITERAL:
    case TokenKind::T_FLOATING_POINT_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_WIDE_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
      return true;
    default:
      return false;
  }
}

static auto needSpace(const Tok* prev, const Tok& current) -> bool {
  if (!prev || current.space) return current.space;
  return wantSpace(prev->kind) && wantSpace(current.kind);
}

void Preprocessor::Private::print(const Tok* begin, const Tok* end,
                                  std::ostream& out) const {
  bool first = true;
  const Tok* prevTk = nullptr;
  for (auto it = begin; it < end; ++it) {
    auto text = getText(*it);
    if (text.empty()) continue;
    if (it->bol) {
      out << "\n";
    } else if (!first && needSpace(prevTk, *it)) {
      out << " ";
    }
    out << text;
    prevTk = it;
    first = false;
  }
}

void Preprocessor::Private::printLine(const Tok* begin, const Tok* end,
                                      std::ostream& out, bool nl) const {
  bool first = true;
  const Tok* prevTk = nullptr;
  for (auto it = begin; it < end; ++it) {
    auto text = getText(*it);
    if (text.empty()) continue;
    if (!first && needSpace(prevTk, *it)) out << " ";
    out << text;
    prevTk = it;
    first = false;
    if (it + 1 < end && (it + 1)->bol) break;
  }
  if (nl) out << "\n";
}

Preprocessor::Preprocessor(Control* control,
                           DiagnosticsClient* diagnosticsClient)
    : d(std::make_unique<Private>()) {
  d->preprocessor_ = this;
  d->control_ = control;
  d->diagnosticsClient_ = diagnosticsClient;
  d->initialize();
}

Preprocessor::~Preprocessor() = default;

auto Preprocessor::control() const -> Control* { return d->control_; }

auto Preprocessor::diagnosticsClient() const -> DiagnosticsClient* {
  return d->diagnosticsClient_;
}

auto Preprocessor::language() const -> LanguageKind { return d->language_; }

void Preprocessor::setLanguage(LanguageKind lang) { d->language_ = lang; }

auto Preprocessor::preprocessorDelegate() const -> PreprocessorDelegate* {
  return nullptr;
}

auto Preprocessor::commentHandler() const -> CommentHandler* {
  return d->commentHandler_;
}

void Preprocessor::setCommentHandler(CommentHandler* commentHandler) {
  d->commentHandler_ = commentHandler;
}

auto Preprocessor::canResolveFiles() const -> bool {
  return d->canResolveFiles_;
}

void Preprocessor::setCanResolveFiles(bool canResolveFiles) {
  d->canResolveFiles_ = canResolveFiles;
}

auto Preprocessor::currentPath() const -> std::string {
  return d->currentPath_.string();
}

void Preprocessor::setCurrentPath(std::string currentPath) {
  d->currentPath_ = std::move(currentPath);
}

auto Preprocessor::omitLineMarkers() const -> bool {
  return d->omitLineMarkers_;
}

void Preprocessor::setOmitLineMarkers(bool omitLineMarkers) {
  d->omitLineMarkers_ = omitLineMarkers;
}

void Preprocessor::setOnWillIncludeHeader(
    std::function<void(const std::string&, int)> willIncludeHeader) {
  d->willIncludeHeader_ = std::move(willIncludeHeader);
}

void Preprocessor::squeeze() {
  d->texts_.clear();
  d->texts_.shrink_to_fit();
  d->expansionPool_.clear();
  d->expansionPool_.shrink_to_fit();
}

auto Preprocessor::sourceFileName(uint32_t sourceFileId) const
    -> const std::string& {
  assert(sourceFileId > 0);
  return d->sourceFiles_[sourceFileId - 1]->fileName;
}

auto Preprocessor::source(uint32_t sourceFileId) const -> const std::string& {
  assert(sourceFileId > 0);
  return d->sourceFiles_[sourceFileId - 1]->source;
}

void Preprocessor::preprocess(std::string source, std::string fileName,
                              std::vector<Token>& tokens) {
  beginPreprocessing(std::move(source), std::move(fileName), tokens);

  DefaultPreprocessorState state{*this};

  while (state) {
    std::visit(state, continuePreprocessing(tokens));
  }

  endPreprocessing(tokens);
}

void Preprocessor::beginPreprocessing(std::string source, std::string fileName,
                                      std::vector<Token>& tokens) {
  assert(!d->findSourceFile(fileName));

  auto sourceFile = d->createSourceFile(std::move(fileName), std::move(source));

  d->mainSourceFileId_ = sourceFile->id;

  auto dirpath = fs::path(sourceFile->fileName);
  dirpath.remove_filename();

  Private::Cursor mainCursor;
  mainCursor.kind = Private::Cursor::FileCursor;
  mainCursor.sourceFile = sourceFile;
  mainCursor.currentPath = dirpath;
  mainCursor.includeDepth = d->includeDepth_;
  mainCursor.initFromSourceFile();
  d->cursors_.push_back(std::move(mainCursor));

  {
    auto builtinsSourceFile =
        d->createSourceFile("<builtins>", std::string(builtinsSource));

    d->builtinsFileId_ = builtinsSourceFile->id;

    Private::Cursor builtinsCursor;
    builtinsCursor.kind = Private::Cursor::FileCursor;
    builtinsCursor.sourceFile = builtinsSourceFile;
    builtinsCursor.currentPath = fs::path{};
    builtinsCursor.includeDepth = d->includeDepth_;
    builtinsCursor.initFromSourceFile();
    d->cursors_.push_back(std::move(builtinsCursor));
  }

  if (!tokens.empty()) {
    assert(tokens.back().is(TokenKind::T_EOF_SYMBOL));
    tokens.pop_back();
  }

  if (tokens.empty()) {
    tokens.emplace_back(TokenKind::T_ERROR);
  }

  if (auto loc = d->codeCompletionLocation_) {
    d->codeCompletionOffset_ = sourceFile->offsetAt(loc->line, loc->column);
  }
}

void Preprocessor::endPreprocessing(std::vector<Token>& tokens) {
  std::function<auto()->std::optional<PreprocessingState>> continuation;
  std::swap(continuation, d->continuation_);
  if (continuation) continuation();

  if (tokens.empty()) return;

  const auto mainSourceFileId = d->mainSourceFileId_;
  if (mainSourceFileId == 0) return;

  const auto offset = d->sourceFiles_[mainSourceFileId - 1]->source.size();

  if (d->codeCompletionLocation_.has_value()) {
    auto& tk = tokens.emplace_back(TokenKind::T_CODE_COMPLETION,
                                   static_cast<unsigned>(offset), 0);
    tk.setFileId(mainSourceFileId);
    d->codeCompletionLocation_ = std::nullopt;
  }

  auto& tk = tokens.emplace_back(TokenKind::T_EOF_SYMBOL,
                                 static_cast<unsigned>(offset));
  tk.setFileId(mainSourceFileId);
}

auto Preprocessor::builtinsFileId() const -> int { return d->builtinsFileId_; }

auto Preprocessor::mainSourceFileId() const -> int {
  return d->mainSourceFileId_;
}

auto Preprocessor::continuePreprocessing(std::vector<Token>& tokens)
    -> PreprocessingState {
  std::function<std::optional<PreprocessingState>()> continuation;
  std::swap(continuation, d->continuation_);
  if (continuation) {
    auto next = continuation();
    if (next) return *next;
  }

  auto emitToken = [&](const Tok& tk) { d->finalizeToken(tokens, tk); };

  return d->expand(emitToken);
}

void Preprocessor::getPreprocessedText(const std::vector<Token>& tokens,
                                       std::ostream& out) const {
  struct FileEntry {
    std::uint32_t fileId;
    bool isSystemHeader;
  };

  std::vector<FileEntry> fileStack;
  if (d->mainSourceFileId_) {
    fileStack.push_back(
        {static_cast<std::uint32_t>(d->mainSourceFileId_), false});
  }
  std::uint32_t lastFileId = std::numeric_limits<std::uint32_t>::max();
  bool atStartOfLine = true;

  auto emitLineMarker = [&](const Token& token, std::uint32_t fileId) {
    if (lastFileId != std::numeric_limits<std::uint32_t>::max()) out << '\n';
    const auto pos = tokenStartPosition(token);
    const bool isSys = isSystemHeader(fileId);
    std::string flags;

    if (fileStack.empty() || fileStack.back().fileId != fileId) {
      bool returning = false;
      for (int i = static_cast<int>(fileStack.size()) - 1; i >= 0; --i) {
        if (fileStack[i].fileId == fileId) {
          fileStack.resize(i + 1);
          returning = true;
          break;
        }
      }
      if (returning) {
        flags += " 2";
      } else {
        flags += " 1";
        fileStack.push_back({fileId, isSys});
      }
    }

    if (isSys) flags += " 3";
    out << std::format("# {} \"{}\"{}\n", pos.line, pos.fileName, flags);
    lastFileId = fileId;
    atStartOfLine = true;
  };

  auto emitTokenSpacing = [&](const Token& token, std::size_t index,
                              const std::vector<Token>& toks) {
    if (token.startOfLine()) {
      atStartOfLine = true;
      out << '\n';
    } else if (token.leadingSpace()) {
      atStartOfLine = false;
      out << ' ';
    } else if (index > 2) {
      const auto& prevToken = toks[index - 2];
      std::string s = prevToken.spell();
      s += token.spell();
      Lexer lex(s, d->language_);
      lex.next();
      if (lex.tokenKind() != prevToken.kind() ||
          lex.tokenLength() != prevToken.length()) {
        out << ' ';
      }
    }
  };

  auto emitColumnPadding = [&](const Token& token) {
    if (!atStartOfLine) return;
    const auto pos = tokenStartPosition(token);
    if (pos.column > 0) {
      for (std::uint32_t i = 0; i < pos.column - 1; ++i) out << ' ';
    }
    atStartOfLine = false;
  };

  std::size_t index = 1;
  while (index + 1 < tokens.size()) {
    const auto& token = tokens[index++];
    if (d->builtinsFileId_ && token.fileId() == d->builtinsFileId_) continue;

    const auto fileId = token.fileId();
    if (!d->omitLineMarkers_ && fileId && fileId != lastFileId) {
      emitLineMarker(token, fileId);
    } else {
      emitTokenSpacing(token, index, tokens);
    }

    emitColumnPadding(token);
    out << token.spell();
  }

  out << '\n';
}

auto Preprocessor::systemIncludePaths() const
    -> const std::vector<std::string>& {
  return d->systemIncludePaths_;
}

static void stripTrailingSep(std::string& path) {
  while (path.length() > 1 && path.ends_with(fs::path::preferred_separator)) {
    path.pop_back();
  }
}

void Preprocessor::addSystemIncludePath(std::string path) {
  stripTrailingSep(path);
  d->systemIncludePaths_.push_back(std::move(path));
}

void Preprocessor::addQuoteIncludePath(std::string path) {
  stripTrailingSep(path);
  d->quoteIncludePaths_.push_back(std::move(path));
}

auto Preprocessor::quoteIncludePaths() const
    -> const std::vector<std::string>& {
  return d->quoteIncludePaths_;
}

void Preprocessor::addUserIncludePath(std::string path) {
  stripTrailingSep(path);
  d->userIncludePaths_.push_back(std::move(path));
}

auto Preprocessor::userIncludePaths() const
    -> const std::vector<std::string>& {
  return d->userIncludePaths_;
}

auto Preprocessor::includedFiles() const
    -> const std::vector<std::pair<std::string, bool>>& {
  return d->includedFiles_;
}

auto Preprocessor::isSystemHeader(std::uint32_t sourceFileId) const -> bool {
  if (sourceFileId == 0 || sourceFileId > d->sourceFiles_.size()) return false;
  return d->sourceFiles_[sourceFileId - 1]->isSystemHeader;
}

void Preprocessor::setDisableCurrentDirSearch(bool disable) {
  d->disableCurrentDirSearch_ = disable;
}

void Preprocessor::defineMacro(const std::string& name,
                               const std::string& body) {
  auto s = name + " " + body;
  auto tokens = d->tokenize(s, 0, false);
  for (auto& tok : tokens) {
    tok.isFromMacroBody = true;
  }
  const Tok* begin = tokens.data();
  const Tok* end = begin + tokens.size();
  d->defineMacro(begin, end);
}

void Preprocessor::undefMacro(const std::string& name) {
  auto it = d->macros_.find(name);
  if (it != d->macros_.end()) d->macros_.erase(it);
}

void Preprocessor::printMacros(std::ostream& out) const {
  struct {
    const Preprocessor& self;
    std::ostream& out;

    void operator()(const FunctionMacro& macro) {
      out << std::format("#define {}", macro.name);
      out << "(";
      for (std::size_t i = 0; i < macro.formals.size(); ++i) {
        if (i > 0) out << ",";
        out << macro.formals[i];
      }
      if (macro.variadic) {
        if (!macro.formals.empty()) out << ",";
        out << "...";
      }
      out << ")";
      if (!macro.body.empty()) {
        out << " ";
        auto d = self.d.get();
        d->print(macro.body.data(), macro.body.data() + macro.body.size(), out);
      }
      out << "\n";
    }

    void operator()(const ObjectMacro& macro) {
      out << std::format("#define {}", macro.name);
      if (!macro.body.empty()) {
        out << " ";
        auto d = self.d.get();
        d->print(macro.body.data(), macro.body.data() + macro.body.size(), out);
      }
      out << "\n";
    }

    void operator()(const BuiltinObjectMacro&) {}
    void operator()(const BuiltinFunctionMacro&) {}

  } printMacro{*this, out};

  for (const auto& [name, macro] : d->macros_) {
    std::visit(printMacro, macro);
  }
}

auto Preprocessor::sources() const -> std::vector<Source> {
  std::vector<Source> sources;
  for (const auto& sourceFile : d->sourceFiles_) {
    sourceFile->ensureLineMap();
    sources.push_back(Source{.fileName = sourceFile->fileName,
                             .lineOffsets = sourceFile->lines});
  }
  return sources;
}

auto Preprocessor::tokenStartPosition(const Token& token) const
    -> SourcePosition {
  if (token.fileId() == 0) return {};
  auto& sourceFile = *d->sourceFiles_[token.fileId() - 1];
  return sourceFile.getTokenStartPosition(token.offset());
}

auto Preprocessor::tokenEndPosition(const Token& token) const
    -> SourcePosition {
  if (token.fileId() == 0) return {};
  auto& sourceFile = *d->sourceFiles_[token.fileId() - 1];
  return sourceFile.getTokenStartPosition(token.offset() + token.length());
}

auto Preprocessor::getTextLine(const Token& token) const -> std::string_view {
  if (token.fileId() == 0) return {};
  const SourceFile* file = d->sourceFiles_[token.fileId() - 1].get();
  file->ensureLineMap();
  const auto pos = tokenStartPosition(token);
  std::string_view source = file->source;
  const auto& lines = file->lines;
  const auto start = lines.at(pos.line - 1);
  const auto end =
      pos.line < lines.size() ? lines.at(pos.line) : source.length();
  auto textLine = source.substr(start, end - start);
  while (!textLine.empty()) {
    auto ch = textLine.back();
    if (!std::isspace(ch)) break;
    textLine.remove_suffix(1);
  }
  return textLine;
}

auto Preprocessor::getTokenText(const Token& token) const -> std::string_view {
  if (token.fileId() == 0) return {};
  const SourceFile* file = d->sourceFiles_[token.fileId() - 1].get();
  std::string_view source = file->source;
  return source.substr(token.offset(), token.length());
}

auto Preprocessor::resolve(const Include& include, bool isIncludeNext) const
    -> std::optional<std::string> {
  auto result = d->resolve(include, isIncludeNext);
  if (!result) return std::nullopt;
  return result->fileName;
}

void Preprocessor::requestCodeCompletionAt(std::uint32_t line,
                                           std::uint32_t column) {
  d->codeCompletionLocation_ = SourcePosition{{}, line, column};
}

void PendingInclude::resolveWith(
    std::optional<std::string> resolvedFileName,
    bool isSystemHeader) const {
  auto d = preprocessor.d.get();

  if (!resolvedFileName.has_value()) {
    const auto& header = getHeaderName(include);
    Token errorTok;
    if (loc) {
      auto tokPtr = static_cast<const Tok*>(loc);
      errorTok = d->tokenForDiagnostic(*tokPtr);
    }
    d->error(errorTok, std::format("file '{}' not found", header));
    return;
  }

  auto fileName = resolvedFileName.value();

  auto resume = [=, this]() -> std::optional<PreprocessingState> {
    auto sourceFile = d->findSourceFile(fileName);
    if (!sourceFile) {
      PendingFileContent request{
          .preprocessor = preprocessor,
          .fileName = fileName,
          .isSystemHeader = isSystemHeader,
      };
      return request;
    }

    if (sourceFile->pragmaOnceProtected) return std::nullopt;

    if (auto it = d->ifndefProtectedFiles_.find(fileName);
        it != d->ifndefProtectedFiles_.end() &&
        d->macros_.contains(it->second)) {
      return std::nullopt;
    }

    auto dirpath = fs::path(sourceFile->fileName).parent_path();

    Preprocessor::Private::Cursor fileCursor;
    fileCursor.kind = Preprocessor::Private::Cursor::FileCursor;
    fileCursor.sourceFile = sourceFile;
    fileCursor.currentPath = dirpath;
    fileCursor.includeDepth = d->includeDepth_ + 1;
    fileCursor.initFromSourceFile();
    d->cursors_.push_back(std::move(fileCursor));
    d->includedFiles_.emplace_back(fileName, isSystemHeader);

    if (d->willIncludeHeader_) {
      d->willIncludeHeader_(fileName, d->includeDepth_ + 1);
    }

    return std::nullopt;
  };

  auto sourceFile = d->findSourceFile(fileName);
  if (sourceFile) {
    resume();
    return;
  }

  d->continuation_ = std::move(resume);
}

void PendingFileContent::setContent(std::optional<std::string> content) const {
  auto d = preprocessor.d.get();

  if (!content.has_value()) return;

  auto sourceFile = d->createSourceFile(fileName, std::move(*content));
  sourceFile->isSystemHeader = isSystemHeader;

  sourceFile->pragmaOnceProtected =
      d->checkPragmaOnceProtected(sourceFile->tokens);

  sourceFile->headerGuardName = d->checkHeaderProtection(sourceFile->tokens);

  if (!sourceFile->headerGuardName.empty()) {
    sourceFile->headerProtectionLevel = int(d->evaluating_.size());
    d->ifndefProtectedFiles_.insert_or_assign(sourceFile->fileName,
                                              sourceFile->headerGuardName);
  }

  auto dirpath = fs::path(sourceFile->fileName);
  dirpath.remove_filename();

  Preprocessor::Private::Cursor fileCursor;
  fileCursor.kind = Preprocessor::Private::Cursor::FileCursor;
  fileCursor.sourceFile = sourceFile;
  fileCursor.currentPath = dirpath;
  fileCursor.includeDepth = d->includeDepth_ + 1;
  fileCursor.initFromSourceFile();
  d->cursors_.push_back(std::move(fileCursor));
  d->includedFiles_.emplace_back(fileName, sourceFile->isSystemHeader);

  if (d->willIncludeHeader_) {
    d->willIncludeHeader_(fileName, d->includeDepth_ + 1);
  }
}

void DefaultPreprocessorState::operator()(const ProcessingComplete&) {
  done = true;
}

void DefaultPreprocessorState::operator()(const CanContinuePreprocessing&) {}

void DefaultPreprocessorState::operator()(const PendingInclude& status) {
  auto d = self.d.get();
  auto result = d->resolve(status.include, status.isIncludeNext);
  if (result) {
    status.resolveWith(result->fileName, result->isSystemHeader);
  } else {
    status.resolveWith(std::nullopt);
  }
}

void DefaultPreprocessorState::operator()(const PendingHasIncludes& status) {
  using Request = PendingHasIncludes::Request;

  std::ranges::for_each(status.requests, [&](const Request& dep) {
    auto resolved = self.resolve(dep.include, dep.isIncludeNext);
    dep.setExists(resolved.has_value());
  });
}

void DefaultPreprocessorState::operator()(const PendingFileContent& request) {
  std::ifstream in(request.fileName, std::ios::binary | std::ios::ate);
  if (!in) {
    request.setContent(std::nullopt);
    return;
  }

  auto size = in.tellg();
  if (size <= 0) {
    request.setContent(std::string{});
    return;
  }

  std::string content(static_cast<std::size_t>(size), '\0');
  in.seekg(0);
  in.read(content.data(), size);
  request.setContent(std::move(content));
}

void DefaultPreprocessorState::operator()(const EnteringFile&) {}

void DefaultPreprocessorState::operator()(const LeavingFile&) {}

}  // namespace cxx
