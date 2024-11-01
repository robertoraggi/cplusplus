// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/preprocessor.h>

// cxx
#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/private/path.h>

// fmt
#include <format>

// utf8
#include <utf8/unchecked.h>

// stl
#include <cxx/preprocessor.h>

#include <algorithm>
#include <cassert>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "pp_keywords-priv.h"

namespace {

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

class Hideset {
 public:
  Hideset(const Hideset &other) = default;
  auto operator=(const Hideset &other) -> Hideset & = default;

  Hideset(Hideset &&other) = default;
  auto operator=(Hideset &&other) -> Hideset & = default;

  Hideset() = default;

  explicit Hideset(std::set<std::string_view> names)
      : names_(std::move(names)) {}

  [[nodiscard]] auto contains(const std::string_view &name) const -> bool {
    return names_.contains(name);
  }

  [[nodiscard]] auto names() const -> const std::set<std::string_view> & {
    return names_;
  };

  auto operator==(const Hideset &other) const -> bool {
    return names_ == other.names_;
  }

 private:
  std::set<std::string_view> names_;
};

struct SystemInclude {
  std::string fileName;

  SystemInclude() = default;

  explicit SystemInclude(std::string fileName)
      : fileName(std::move(fileName)) {}
};

struct QuoteInclude {
  std::string fileName;

  QuoteInclude() = default;

  explicit QuoteInclude(std::string fileName) : fileName(std::move(fileName)) {}
};

using Include = std::variant<SystemInclude, QuoteInclude>;

inline auto getHeaderName(const Include &include) -> std::string {
  return std::visit([](const auto &include) { return include.fileName; },
                    include);
}

}  // namespace

template <>
struct std::less<Hideset> {
  using is_transparent = void;

  auto operator()(const Hideset &hideset, const Hideset &other) const -> bool {
    return hideset.names() < other.names();
  }

  auto operator()(const Hideset &hideset,
                  const std::set<std::string_view> &names) const -> bool {
    return hideset.names() < names;
  }

  auto operator()(const std::set<std::string_view> &names,
                  const Hideset &hideset) const -> bool {
    return names < hideset.names();
  }

  auto operator()(const Hideset &hideset, const std::string_view &name) const
      -> bool {
    return std::lexicographical_compare(begin(hideset.names()),
                                        end(hideset.names()), &name, &name + 1);
  }

  auto operator()(const std::string_view &name, const Hideset &hideset) const
      -> bool {
    return std::lexicographical_compare(
        &name, &name + 1, begin(hideset.names()), end(hideset.names()));
  }
};

template <>
struct std::hash<Hideset> {
  using is_transparent = void;

  template <typename T>
  void hash_combine(std::size_t &seed, const T &val) const {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  auto operator()(const Hideset &hideset) const -> std::size_t {
    return operator()(hideset.names());
  }

  auto operator()(const std::set<std::string_view> &names) const
      -> std::size_t {
    std::size_t seed = 0;
    for (const auto &name : names) hash_combine(seed, name);
    return seed;
  }

  auto operator()(const std::string_view &name) const -> std::size_t {
    std::size_t seed = 0;
    hash_combine(seed, name);
    return seed;
  }
};

template <>
struct std::equal_to<Hideset> {
  using is_transparent = void;

  auto operator()(const Hideset &hideset, const Hideset &other) const -> bool {
    return hideset.names() == other.names();
  }

  auto operator()(const Hideset &hideset,
                  const std::set<std::string_view> &names) const -> bool {
    return hideset.names() == names;
  }

  auto operator()(const std::set<std::string_view> &names,
                  const Hideset &hideset) const -> bool {
    return hideset.names() == names;
  }

  auto operator()(const Hideset &hideset, const std::string_view &name) const
      -> bool {
    return hideset.names().size() == 1 && *hideset.names().begin() == name;
  }

  auto operator()(const std::string_view &name, const Hideset &hideset) const
      -> bool {
    return hideset.names().size() == 1 && *hideset.names().begin() == name;
  }
};

namespace cxx {

namespace {

struct SourceFile;
struct TokList;

struct Tok final : Managed {
  std::string_view text;
  const Hideset *hideset = nullptr;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t sourceFile = 0;
  TokenKind kind = TokenKind::T_EOF_SYMBOL;
  std::uint16_t bol : 1 = false;
  std::uint16_t space : 1 = false;
  std::uint16_t generated : 1 = false;

  Tok(const Tok &other) = default;
  auto operator=(const Tok &other) -> Tok & = default;

  Tok(Tok &&other) = default;
  auto operator=(Tok &&other) -> Tok & = default;

  [[nodiscard]] auto is(TokenKind k) const -> bool { return kind == k; }

  [[nodiscard]] auto isNot(TokenKind k) const -> bool { return kind != k; }

  [[nodiscard]] static auto WithHideset(Arena *pool, const Tok *tok,
                                        const Hideset *hideset) -> Tok * {
    return new (pool) Tok(tok, hideset);
  }

  [[nodiscard]] static auto FromCurrentToken(Arena *pool, const Lexer &lex,
                                             int sourceFile) -> Tok * {
    auto tk = new (pool) Tok();
    tk->sourceFile = sourceFile;
    tk->kind = lex.tokenKind();
    tk->text = lex.tokenText();
    tk->offset = lex.tokenPos();
    tk->length = lex.tokenLength();
    tk->bol = lex.tokenStartOfLine();
    tk->space = lex.tokenLeadingSpace();
    return tk;
  }

  [[nodiscard]] static auto Copy(Arena *pool, const Tok *tok) {
    auto tk = new (pool) Tok();
    *tk = *tok;
    return tk;
  }

  [[nodiscard]] static auto Gen(Arena *pool, TokenKind kind,
                                const std::string_view &text,
                                const Hideset *hideset = nullptr) -> Tok * {
    auto tk = new (pool) Tok();
    tk->kind = kind;
    tk->text = text;
    tk->hideset = hideset;
    tk->generated = true;
    tk->length = static_cast<std::uint32_t>(text.length());
    // tk->space = true;
    return tk;
  }

  [[nodiscard]] auto token() const -> Token {
    Token token(kind, offset, length);
    token.setFileId(sourceFile);
    token.setLeadingSpace(space);
    token.setStartOfLine(bol);
    return token;
  }

 private:
  Tok() = default;

  Tok(const Tok *tok, const Hideset *hs) {
    kind = tok->kind;
    text = tok->text;
    sourceFile = tok->sourceFile;
    bol = tok->bol;
    space = tok->space;
    generated = tok->generated;
    offset = tok->offset;
    length = tok->length;
    hideset = hs;
  }
};

struct TokList final : Managed {
  const Tok *tok = nullptr;
  TokList *next = nullptr;

  explicit TokList(const Tok *tok, TokList *next = nullptr)
      : tok(tok), next(next) {}

  [[nodiscard]] static auto isSame(TokList *ls, TokList *rs) -> bool {
    if (ls == rs) return true;
    if (!ls || !rs) return false;
    if (ls->tok->kind != rs->tok->kind) return false;
    if (ls->tok->text != rs->tok->text) return false;
    return isSame(ls->next, rs->next);
  }
};

class TokIterator {
 public:
  using value_type = const Tok *;
  using difference_type = std::ptrdiff_t;

  TokIterator() = default;
  explicit TokIterator(TokList *ts) : ts_(ts) {}

  auto operator==(const TokIterator &other) const -> bool = default;

  auto operator*() const -> const Tok & { return *ts_->tok; }

  auto operator->() const -> const Tok * { return ts_->tok; }

  auto operator++() -> TokIterator & {
    ts_ = ts_->next;
    return *this;
  }

  auto operator++(int) -> TokIterator {
    auto it = *this;
    ts_ = ts_->next;
    return it;
  }

  [[nodiscard]] auto toTokList() const -> TokList * { return ts_; }

 private:
  TokList *ts_ = nullptr;
};

struct EndOfFileSentinel {
  auto operator==(TokIterator it) const -> bool { return !it.toTokList(); }
};

static_assert(std::sentinel_for<EndOfFileSentinel, TokIterator>);

struct EndOfTokLineSentinel {
  TokIterator startOfLine;

  EndOfTokLineSentinel() = default;

  explicit EndOfTokLineSentinel(TokIterator startOfLine)
      : startOfLine(startOfLine) {}

  auto operator==(TokIterator it) const -> bool {
    auto list = it.toTokList();
    if (!list || !list->tok) return true;
    if (list->tok->bol) return list != startOfLine.toTokList();
    return false;
  }
};

static_assert(std::sentinel_for<EndOfTokLineSentinel, TokIterator>);

using TokRange = std::ranges::subrange<TokIterator, TokIterator>;

using TokLine = std::ranges::subrange<TokIterator, EndOfTokLineSentinel>;

struct LookAt {
  template <typename I, typename S>
  [[nodiscard]] auto operator()(I first, S last, auto... tokens) const -> bool {
    return Helper<I, S>{}(first, last, tokens...);
  }

 private:
  template <typename I, typename S>
  struct Helper {
    [[nodiscard]] auto operator()(I, S) const -> bool { return true; }

    [[nodiscard]] auto operator()(I first, S last, std::string_view text,
                                  auto... rest) const -> bool {
      if (first == last) return false;
      if (first->text != text) return false;
      return (*this)(++first, last, rest...);
    }

    [[nodiscard]] auto operator()(I first, S last, TokenKind kind,
                                  auto... rest) const -> bool {
      if (first == last) return false;
      if (first->isNot(kind)) return false;
      return (*this)(++first, last, rest...);
    }
  };
};

inline constexpr LookAt lookat{};

struct ObjectMacro {
  std::string_view name;
  TokList *body = nullptr;

  ObjectMacro(std::string_view name, TokList *body) : name(name), body(body) {}
};

struct FunctionMacro {
  std::string_view name;
  std::vector<std::string_view> formals;
  TokList *body = nullptr;
  bool variadic = false;

  FunctionMacro(std::string_view name, std::vector<std::string_view> formals,
                TokList *body, bool variadic)
      : name(name),
        formals(std::move(formals)),
        body(body),
        variadic(variadic) {}
};

struct MacroExpansionContext {
  TokList *ts = nullptr;
};

struct BuiltinObjectMacro {
  std::string_view name;
  std::function<auto(MacroExpansionContext)->TokList *> expand;

  BuiltinObjectMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokList *> expand)
      : name(name), expand(std::move(expand)) {}
};

struct BuiltinFunctionMacro {
  std::string_view name;
  std::function<auto(MacroExpansionContext)->TokList *> expand;

  BuiltinFunctionMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokList *> expand)
      : name(name), expand(std::move(expand)) {}
};

using Macro = std::variant<ObjectMacro, FunctionMacro, BuiltinObjectMacro,
                           BuiltinFunctionMacro>;

[[nodiscard]] inline auto getMacroName(const Macro &macro) -> std::string_view {
  struct {
    auto operator()(const ObjectMacro &macro) const -> std::string_view {
      return macro.name;
    }

    auto operator()(const FunctionMacro &macro) const -> std::string_view {
      return macro.name;
    }

    auto operator()(const BuiltinObjectMacro &macro) const -> std::string_view {
      return macro.name;
    }

    auto operator()(const BuiltinFunctionMacro &macro) const
        -> std::string_view {
      return macro.name;
    }
  } visitor;

  return std::visit(visitor, macro);
}

[[nodiscard]] inline auto getMacroBody(const Macro &macro) -> TokList * {
  struct {
    auto operator()(const ObjectMacro &macro) const -> TokList * {
      return macro.body;
    }

    auto operator()(const FunctionMacro &macro) const -> TokList * {
      return macro.body;
    }

    auto operator()(const BuiltinObjectMacro &macro) const -> TokList * {
      return nullptr;
    }

    auto operator()(const BuiltinFunctionMacro &macro) const -> TokList * {
      return nullptr;
    }
  } visitor;

  return std::visit(visitor, macro);
}

[[nodiscard]] inline auto isObjectLikeMacro(const Macro &macro) -> bool {
  struct {
    auto operator()(const ObjectMacro &) const -> bool { return true; }
    auto operator()(const BuiltinObjectMacro &) const -> bool { return true; }

    auto operator()(const FunctionMacro &) const -> bool { return false; }
    auto operator()(const BuiltinFunctionMacro &) const -> bool {
      return false;
    }
  } visitor;

  return std::visit(visitor, macro);
}

[[nodiscard]] inline auto isFunctionLikeMacro(const Macro &macro) -> bool {
  struct {
    auto operator()(const FunctionMacro &) const -> bool { return true; }
    auto operator()(const BuiltinFunctionMacro &) const -> bool { return true; }

    auto operator()(const ObjectMacro &) const -> bool { return false; }
    auto operator()(const BuiltinObjectMacro &) const -> bool { return false; }
  } visitor;

  return std::visit(visitor, macro);
}

struct SourceFile {
  std::string fileName;
  std::string source;
  std::vector<int> lines;
  TokList *tokens = nullptr;
  TokList *headerProtection = nullptr;
  int headerProtectionLevel = 0;
  int id = 0;
  bool pragmaOnceProtected = false;

  SourceFile() noexcept = default;
  SourceFile(const SourceFile &) noexcept = default;
  auto operator=(const SourceFile &) noexcept -> SourceFile & = default;
  SourceFile(SourceFile &&) noexcept = default;
  auto operator=(SourceFile &&) noexcept -> SourceFile & = default;

  SourceFile(std::string fileName, std::string source,
             std::uint32_t id) noexcept
      : fileName(std::move(fileName)), source(std::move(source)), id(id) {
    initLineMap();
  }

  void getTokenStartPosition(unsigned offset, unsigned *line, unsigned *column,
                             std::string_view *fileName) const {
    auto it = std::lower_bound(lines.cbegin(), lines.cend(),
                               static_cast<int>(offset));
    if (*it != static_cast<int>(offset)) --it;

    assert(*it <= int(offset));

    if (line) *line = int(std::distance(cbegin(lines), it) + 1);

    if (column) {
      const auto start = cbegin(source) + *it;
      const auto end = cbegin(source) + offset;

      *column = utf8::unchecked::distance(start, end) + 1;
    }

    if (fileName) *fileName = this->fileName;
  }

 private:
  void initLineMap() {
    std::size_t offset = 0;

    lines.push_back(0);

    while (offset < source.length()) {
      const auto index = source.find_first_of('\n', offset);

      if (index == std::string::npos) break;

      offset = index + 1;

      lines.push_back(static_cast<int>(offset));
    }
  }
};

}  // namespace

struct Preprocessor::Private {
  struct Buffer {
    SourceFile *source = nullptr;
    fs::path currentPath;
    TokList *ts = nullptr;
    int includeDepth = 0;
  };

  Preprocessor *preprocessor_ = nullptr;
  Control *control_ = nullptr;
  DiagnosticsClient *diagnosticsClient_ = nullptr;
  CommentHandler *commentHandler_ = nullptr;
  PreprocessorDelegate *delegate_ = nullptr;
  bool canResolveFiles_ = true;
  std::vector<std::string> systemIncludePaths_;
  std::vector<std::string> quoteIncludePaths_;
  std::unordered_map<std::string_view, Macro> macros_;
  std::set<Hideset> hidesets;
  std::forward_list<std::string> scratchBuffer_;
  std::unordered_map<std::string, std::string> ifndefProtectedFiles_;
  std::vector<std::unique_ptr<SourceFile>> sourceFiles_;
  fs::path currentPath_;
  std::string currentFileName_;
  std::vector<bool> evaluating_;
  std::vector<bool> skipping_;
  std::string_view date_;
  std::string_view time_;
  std::function<bool(std::string)> fileExists_;
  std::function<std::string(std::string)> readFile_;
  std::function<void(const std::string &, int)> willIncludeHeader_;
  std::vector<Buffer> buffers_;
  int counter_ = 0;
  int includeDepth_ = 0;
  bool omitLineMarkers_ = false;
  Arena pool_;

  Private();

  void initialize();

  void error(TokList *ts, std::string message) const {
    if (!ts || !ts->tok) {
      cxx_runtime_error(std::format("no source location: {}", message));
    } else {
      diagnosticsClient_->report(ts->tok->token(), Severity::Error,
                                 std::move(message));
    }
  }

  void warning(TokList *ts, std::string message) const {
    if (!ts || !ts->tok) {
      cxx_runtime_error(std::format("no source location: {}", message));
    } else {
      diagnosticsClient_->report(ts->tok->token(), Severity::Warning,
                                 std::move(message));
    }
  }

  void error(const Token &token, std::string message) const {
    diagnosticsClient_->report(token, Severity::Error, std::move(message));
  }

  void warning(const Token &token, std::string message) const {
    diagnosticsClient_->report(token, Severity::Warning, std::move(message));
  }

  [[nodiscard]] auto state() const -> std::tuple<bool, bool> {
    return std::tuple(skipping_.back(), evaluating_.back());
  }

  [[nodiscard]] auto clone(TokList *ts) -> TokList * {
    if (!ts) return nullptr;
    return cons(ts->tok, clone(ts->next));
  }

  [[nodiscard]] auto cons(const Tok *tok, TokList *next = nullptr)
      -> TokList * {
    return new (&pool_) TokList(tok, next);
  }

  [[nodiscard]] auto snoc(TokList *first, TokList *second) -> TokList * {
    if (!first) return second;
    if (!second) return first;

    TokList *tail = first;

    while (tail->next) {
      tail = tail->next;
    }

    (tail)->next = second;

    return first;
  }

  template <typename I, std::sentinel_for<I> S>
  [[nodiscard]] auto toTokList(I first, S last) -> TokList * {
    TokList *list = nullptr;
    TokList **it = &list;
    for (; first != last; ++first) {
      *it = cons(&*first);
      it = (&(*it)->next);
    }
    return list;
  }

  template <typename R>
  [[nodiscard]] auto toTokList(const R &range) -> TokList * {
    return toTokList(std::ranges::begin(range), std::ranges::end(range));
  }

  [[nodiscard]] auto withHideset(const Tok *tok, const Hideset *hideset)
      -> Tok * {
    return Tok::WithHideset(&pool_, tok, hideset);
  }

  [[nodiscard]] auto fromCurrentToken(const Lexer &lex, int sourceFile)
      -> Tok * {
    return Tok::FromCurrentToken(&pool_, lex, sourceFile);
  }

  [[nodiscard]] auto gen(TokenKind kind, const std::string_view &text,
                         const Hideset *hideset = nullptr) -> Tok * {
    return Tok::Gen(&pool_, kind, text, hideset);
  }

  [[nodiscard]] auto copy(const Tok *tok) -> Tok * {
    auto copy = Tok::Copy(&pool_, tok);
    return copy;
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

  [[nodiscard]] auto findSourceFile(const std::string &fileName)
      -> SourceFile * {
    for (auto &sourceFile : sourceFiles_) {
      if (sourceFile->fileName == fileName) {
        return sourceFile.get();
      }
    }
    return nullptr;
  }

  [[nodiscard]] auto createSourceFile(std::string fileName, std::string source)
      -> SourceFile * {
    if (sourceFiles_.size() >= 4096) {
      cxx_runtime_error("too many source files");
    }

    const int sourceFileId = static_cast<int>(sourceFiles_.size() + 1);

    SourceFile *sourceFile =
        &*sourceFiles_.emplace_back(std::make_unique<SourceFile>(
            std::move(fileName), std::move(source), sourceFileId));

    sourceFile->tokens = tokenize(sourceFile->source, sourceFileId, true);

    return sourceFile;
  }

  [[nodiscard]] auto bol(TokList *ts) const -> bool {
    return ts && ts->tok->bol;
  }

  [[nodiscard]] auto lookat(TokList *ts, auto... tokens) const -> bool {
    return cxx::lookat(TokIterator{ts}, EndOfFileSentinel{}, tokens...);
  }

  [[nodiscard]] auto match(TokList *&ts, TokenKind k) const -> bool {
    if (lookat(ts, k)) {
      ts = ts->next;
      return true;
    }
    return false;
  }

  [[nodiscard]] auto match(TokIterator &it, TokIterator last, TokenKind k) const
      -> bool {
    if (it == last) return false;
    if (it->isNot(k)) return false;
    ++it;
    return true;
  }

  [[nodiscard]] auto matchId(TokList *&ts, const std::string_view &s) const
      -> bool {
    if (lookat(ts, TokenKind::T_IDENTIFIER) && ts->tok->text == s) {
      ts = ts->next;
      return true;
    }
    return false;
  }

  void expect(TokList *&ts, TokenKind k) const {
    if (!match(ts, k)) {
      error(ts, std::format("expected '{}'", Token::spell(k)));
    }
  }

  [[nodiscard]] auto expectId(TokList *&ts) const -> std::string_view {
    if (lookat(ts, TokenKind::T_IDENTIFIER)) {
      auto id = ts->tok->text;
      ts = ts->next;
      return id;
    }
    assert(ts);
    error(ts, "expected an identifier");
    return {};
  }

  [[nodiscard]] auto makeUnion(const Hideset *hs, const std::string_view &name)
      -> const Hideset * {
    if (!hs) return get(name);
    if (hs->names().contains(name)) return hs;
    auto names = hs->names();
    names.insert(name);
    return get(std::move(names));
  }

  [[nodiscard]] auto makeIntersection(const Hideset *hs, const Hideset *other)
      -> const Hideset * {
    if (!other || !hs) return nullptr;
    if (other == hs) return hs;

    std::set<std::string_view> names;

    std::set_intersection(begin(hs->names()), end(hs->names()),
                          begin(other->names()), end(other->names()),
                          std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  [[nodiscard]] auto get(std::set<std::string_view> names) -> const Hideset * {
    if (names.empty()) return nullptr;
    if (auto it = hidesets.find(names); it != hidesets.end()) return &*it;
    return &*hidesets.emplace(std::move(names)).first;
  }

  [[nodiscard]] auto get(const std::string_view &name) -> const Hideset * {
    if (auto it = hidesets.find(name); it != hidesets.end()) return &*it;
    return &*hidesets.emplace(std::set{name}).first;
  }

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
    }  // switch
  }

  [[nodiscard]] auto updateStringLiteralValue(Token &lastToken, const Tok *tk)
      -> bool {
    if (!isStringLiteral(lastToken.kind())) {
      return false;
    }

    if (tk->isNot(TokenKind::T_STRING_LITERAL) &&
        tk->kind != lastToken.kind()) {
      return false;
    }

    auto newText = lastToken.value().literalValue->value();

    if (newText.ends_with('"')) {
      newText.pop_back();
    }

    newText += tk->text.substr(tk->text.find_first_of('"') + 1);

    TokenValue value = lastToken.value();

    switch (lastToken.kind()) {
      case TokenKind::T_STRING_LITERAL:
        value.literalValue = control_->stringLiteral(string(newText));
        break;
      case TokenKind::T_WIDE_STRING_LITERAL:
        value.literalValue = control_->wideStringLiteral(string(newText));
        break;
      case TokenKind::T_UTF8_STRING_LITERAL:
        value.literalValue = control_->utf8StringLiteral(string(newText));
        break;
      case TokenKind::T_UTF16_STRING_LITERAL:
        value.literalValue = control_->utf16StringLiteral(string(newText));
        break;
      case TokenKind::T_UTF32_STRING_LITERAL:
        value.literalValue = control_->utf32StringLiteral(string(newText));
        break;
      default:
        break;
    }  // switch

    lastToken.setValue(value);

    return true;
  }

  [[nodiscard]] auto fileExists(const fs::path &file) const -> bool {
    if (fileExists_) return fileExists_(file.string());
    return fs::exists(file);
  }

  [[nodiscard]] auto readFile(const fs::path &file) const -> std::string {
    if (readFile_) return readFile_(file.string());
    std::ifstream in(file);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
  }

  [[nodiscard]] auto checkHeaderProtection(TokList *ts) const -> TokList *;

  [[nodiscard]] auto checkPragmaOnceProtected(TokList *ts) const -> bool;

  [[nodiscard]] auto resolve(const Include &include, bool next) const
      -> std::optional<fs::path> {
    if (!canResolveFiles_) return std::nullopt;

    struct Resolve {
      const Private *d;
      bool next;

      Resolve(const Private *d, bool next) : d(d), next(next) {}

      [[nodiscard]] auto operator()(const SystemInclude &include) const
          -> std::optional<fs::path> {
        bool hit = false;
        for (const auto &includePath :
             d->systemIncludePaths_ | std::views::reverse) {
          const auto path = fs::path(includePath) / include.fileName;
          if (d->fileExists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }
        return {};
      }

      [[nodiscard]] auto operator()(const QuoteInclude &include) const
          -> std::optional<fs::path> {
        bool hit = false;

        if (auto path = d->currentPath_ / include.fileName;
            d->fileExists(path)) {
          if (!next) return path;
          hit = true;
        }

        for (const auto &includePath :
             d->quoteIncludePaths_ | std::views::reverse) {
          const auto path = fs::path(includePath) / include.fileName;
          if (d->fileExists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }

        for (const auto &includePath :
             d->systemIncludePaths_ | std::views::reverse) {
          const auto path = fs::path(includePath) / include.fileName;
          if (d->fileExists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }
        return {};
      }
    };

    return std::visit(Resolve(this, next), include);
  }

  [[nodiscard]] auto isDefined(const std::string_view &id) const -> bool {
    const auto defined = macros_.contains(id);
    return defined;
  }

  [[nodiscard]] auto isDefined(const Tok *tok) const -> bool {
    if (!tok) return false;
    return tok->is(TokenKind::T_IDENTIFIER) && isDefined(tok->text);
  }

  void defineMacro(TokList *ts);

  void adddBuiltinMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokList *> expand) {
    macros_.insert_or_assign(name, BuiltinObjectMacro(name, std::move(expand)));
  }

  void adddBuiltinFunctionMacro(
      std::string_view name,
      std::function<auto(MacroExpansionContext)->TokList *> expand) {
    macros_.insert_or_assign(name,
                             BuiltinFunctionMacro(name, std::move(expand)));
  }

  [[nodiscard]] auto tokenize(const std::string_view &source, int sourceFile,
                              bool bol) -> TokList *;

  [[nodiscard]] auto skipLine(TokIterator it, TokIterator last) -> TokIterator;

  [[nodiscard]] auto parseMacroDefinition(TokList *ts) -> Macro;

  [[nodiscard]] auto expand(const std::function<void(const Tok *)> &emitToken)
      -> Status;

  [[nodiscard]] auto expandTokens(TokIterator it, TokIterator last,
                                  bool inConditionalExpression) -> TokIterator;

  [[nodiscard]] auto expandOne(
      TokIterator first, TokIterator last, bool inConditionalExpression,
      const std::function<void(const Tok *)> &emitToken) -> TokIterator;

  [[nodiscard]] auto replaceIsDefinedMacro(
      TokList *ts, bool inConditionalExpression,
      const std::function<void(const Tok *)> &emitToken) -> TokList *;

  [[nodiscard]] auto expandMacro(TokList *ts) -> TokList *;

  [[nodiscard]] auto expandObjectLikeMacro(TokList *ts, const Macro *macro)
      -> TokList *;

  [[nodiscard]] auto expandFunctionLikeMacro(TokList *ts, const Macro *macro)
      -> TokList *;

  struct ParsedIncludeDirective {
    Include header;
    bool includeNext = false;
    TokList *loc = nullptr;
  };

  [[nodiscard]] auto parseDirective(SourceFile *source, TokList *start)
      -> std::optional<ParsedIncludeDirective>;

  [[nodiscard]] auto parseIncludeDirective(TokList *directive, TokList *ts)
      -> std::optional<ParsedIncludeDirective>;

  [[nodiscard]] auto resolveIncludeDirective(
      const ParsedIncludeDirective &directive) -> SourceFile *;

  [[nodiscard]] auto parseHeaderName(TokList *ts)
      -> std::tuple<TokList *, std::optional<Include>>;

  [[nodiscard]] auto substitute(TokList *pointOfSubstitution,
                                const Macro *macro,
                                const std::vector<TokRange> &actuals,
                                const Hideset *hideset) -> TokList *;

  [[nodiscard]] auto merge(const Tok *left, const Tok *right) -> const Tok *;

  [[nodiscard]] auto stringize(TokList *ts) -> const Tok *;

  [[nodiscard]] auto instantiate(TokList *ts, const Hideset *hideset)
      -> TokList *;

  [[nodiscard]] auto lookupMacro(const Tok *tk) const -> const Macro *;

  [[nodiscard]] auto lookupMacroArgument(TokList *&ts, const Macro *macro,
                                         const std::vector<TokRange> &actuals)
      -> std::optional<TokRange>;

  [[nodiscard]] auto copyLine(TokList *ts) -> TokList *;

  [[nodiscard]] auto constantExpression(TokList *ts) -> long;
  [[nodiscard]] auto conditionalExpression(TokList *&ts) -> long;
  [[nodiscard]] auto binaryExpression(TokList *&ts) -> long;
  [[nodiscard]] auto binaryExpressionHelper(TokList *&ts, long lhs, int minPrec)
      -> long;
  [[nodiscard]] auto unaryExpression(TokList *&ts) -> long;
  [[nodiscard]] auto primaryExpression(TokList *&ts) -> long;

  [[nodiscard]] auto parseArguments(TokList *ts, int formalCount,
                                    bool ignoreComma = false)
      -> std::tuple<std::vector<TokRange>, TokList *, const Hideset *>;

  [[nodiscard]] auto string(std::string s) -> std::string_view;

  void print(TokList *ts, std::ostream &out) const;

  void printLine(TokList *ts, std::ostream &out, bool nl = true) const;

  void finalizeToken(std::vector<Token> &tokens, const Tok *tk);
};

struct Preprocessor::ParseArguments {
  struct Result {
    std::vector<TokRange> args;
    TokIterator it;
    const Hideset *hideset = nullptr;
  };

  Private &d;

  template <std::sentinel_for<TokIterator> S>
  auto operator()(TokIterator it, S last, int formalCount, bool ignoreComma)
      -> std::optional<Result> {
    if (!cxx::lookat(it, last, TokenKind::T_LPAREN)) {
      cxx_runtime_error("expected '('");
      return std::nullopt;
    }

    auto lparen = it++;

    std::vector<TokRange> args;

    if (!cxx::lookat(it, last, TokenKind::T_RPAREN) || ignoreComma) {
      auto startArgument = it;

      it = skipArgument(it, last, ignoreComma && formalCount == 0);
      args.push_back(TokRange{startArgument, it});

      while (it != last && it->is(TokenKind::T_COMMA)) {
        ++it;

        auto startArgument = it;

        it = skipArgument(it, last, ignoreComma && args.size() >= formalCount);
        args.push_back(TokRange{startArgument, it});
      }
    }

    if (!cxx::lookat(it, last, TokenKind::T_RPAREN)) {
      d.error(lparen.toTokList(),
              std::format("unterminated function-like macro invocation"));
      return std::nullopt;
    }

    auto rparen = it++;

    return Result{
        .args = std::move(args),
        .it = it,
        .hideset = rparen->hideset,
    };
  }

  template <typename I, std::sentinel_for<I> S>
  auto skipArgument(I it, S last, bool ignoreComma) -> TokIterator {
    for (int depth = 0; it != last; ++it) {
      const auto &tk = *it;
      if (tk.is(TokenKind::T_LPAREN)) {
        ++depth;
      } else if (tk.is(TokenKind::T_RPAREN)) {
        if (depth == 0) break;
        --depth;
      } else if (tk.is(TokenKind::T_COMMA)) {
        if (depth == 0 && !ignoreComma) break;
      }
    }

    return it;
  }
};

Preprocessor::Private::Private() {
  skipping_.push_back(false);
  evaluating_.push_back(true);

  time_t t;
  time(&t);

  char buffer[32];

  strftime(buffer, sizeof(buffer), "\"%b %e %Y\"", localtime(&t));
  date_ = string(buffer);

  strftime(buffer, sizeof(buffer), "\"%T\"", localtime(&t));
  time_ = string(buffer);
}

void Preprocessor::Private::initialize() {
  // add built-in object-like macros

  adddBuiltinMacro(
      "__FILE__", [this](const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        auto tk = gen(TokenKind::T_STRING_LITERAL,
                      string(std::format("\"{}\"", currentFileName_)));
        tk->space = true;
        tk->sourceFile = ts->tok->sourceFile;
        return cons(tk, ts->next);
      });

  adddBuiltinMacro(
      "__LINE__", [this](const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        const auto start = preprocessor_->tokenStartPosition(ts->tok->token());
        auto tk = gen(TokenKind::T_INTEGER_LITERAL,
                      string(std::to_string(start.line)));
        tk->sourceFile = ts->tok->sourceFile;
        tk->space = true;
        return cons(tk, ts->next);
      });

  adddBuiltinMacro("__COUNTER__",
                   [this](const MacroExpansionContext &context) -> TokList * {
                     auto tk = gen(TokenKind::T_INTEGER_LITERAL,
                                   string(std::to_string(counter_++)));
                     tk->sourceFile = context.ts->tok->sourceFile;
                     tk->space = true;
                     return cons(tk, context.ts->next);
                   });

  adddBuiltinMacro("__DATE__",
                   [this](const MacroExpansionContext &context) -> TokList * {
                     auto ts = context.ts;
                     auto tk = gen(TokenKind::T_STRING_LITERAL, date_);
                     tk->sourceFile = ts->tok->sourceFile;
                     tk->space = true;
                     return cons(tk, ts->next);
                   });

  adddBuiltinMacro("__TIME__",
                   [this](const MacroExpansionContext &context) -> TokList * {
                     auto ts = context.ts;
                     auto tk = gen(TokenKind::T_STRING_LITERAL, time_);
                     tk->sourceFile = ts->tok->sourceFile;
                     tk->space = true;
                     return cons(tk, ts->next);
                   });

  // add built-in function-like macros

  auto replaceWithBoolLiteral = [this](const Tok *token, bool value,
                                       TokList *ts) {
    auto tk = gen(TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
    tk->sourceFile = token->sourceFile;
    tk->space = token->space;
    tk->bol = token->bol;
    return cons(tk, ts);
  };

  adddBuiltinFunctionMacro(
      "__has_feature",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        auto macroId = ts->tok;
        ts = ts->next;
        expect(ts, TokenKind::T_LPAREN);
        const auto id = expectId(ts);
        expect(ts, TokenKind::T_RPAREN);
        const auto enabled = enabledFeatures.contains(id);
        return replaceWithBoolLiteral(macroId, enabled, ts);
      });

  adddBuiltinFunctionMacro(
      "__has_builtin",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        auto macroId = ts->tok;
        ts = ts->next;
        expect(ts, TokenKind::T_LPAREN);
        const auto id = expectId(ts);
        expect(ts, TokenKind::T_RPAREN);
        const auto enabled = enabledBuiltins.contains(id);
        return replaceWithBoolLiteral(macroId, enabled, ts);
      });

  adddBuiltinFunctionMacro(
      "__has_extension",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        auto macroId = ts->tok;
        ts = ts->next;
        expect(ts, TokenKind::T_LPAREN);
        const auto id = expectId(ts);
        expect(ts, TokenKind::T_RPAREN);
        const auto enabled = enabledExtensions.contains(id);
        return replaceWithBoolLiteral(macroId, enabled, ts);
      });

  adddBuiltinFunctionMacro(
      "__has_attribute",
      [this, replaceWithBoolLiteral](
          const MacroExpansionContext &context) -> TokList * {
        auto ts = context.ts;
        auto macroId = ts->tok;
        ts = ts->next;
        expect(ts, TokenKind::T_LPAREN);
        const auto id = expectId(ts);
        expect(ts, TokenKind::T_RPAREN);
        const auto enabled = true;
        return replaceWithBoolLiteral(macroId, enabled, ts);
      });

  auto hasInclude = [this, replaceWithBoolLiteral](
                        const MacroExpansionContext &context) -> TokList * {
    auto ts = context.ts;

    const auto macroName = ts->tok;
    ts = ts->next;

    const auto isIncludeNext = macroName->text == "__has_include_next";

    expect(ts, TokenKind::T_LPAREN);

    auto [args, rest, hideset] = parseArguments(context.ts, 0, true);

    if (args.empty()) {
      error(macroName->token(), std::format("expected a header name"));
      return replaceWithBoolLiteral(macroName, false, rest);
    }

    auto [startArg, endArg] = args[0];

    auto arg = expandTokens(TokIterator{startArg}, TokIterator{endArg},
                            /*expression*/ false)
                   .toTokList();

    Include include;

    if (auto literal = arg; match(arg, TokenKind::T_STRING_LITERAL)) {
      std::string fn(
          literal->tok->text.substr(1, literal->tok->text.length() - 2));
      include = QuoteInclude(std::move(fn));
    } else if (arg) {
      auto ts = arg;
      expect(ts, TokenKind::T_LESS);

      std::string fn;
      for (; ts && !lookat(ts, TokenKind::T_GREATER); ts = ts->next) {
        fn += ts->tok->text;
      }

      expect(ts, TokenKind::T_GREATER);
      include = SystemInclude(std::move(fn));
    }

    const auto value = resolve(include, isIncludeNext);

    return replaceWithBoolLiteral(macroName, value.has_value(), rest);
  };

  adddBuiltinFunctionMacro("__has_include", hasInclude);
  adddBuiltinFunctionMacro("__has_include_next", hasInclude);
}

void Preprocessor::Private::finalizeToken(std::vector<Token> &tokens,
                                          const Tok *tk) {
  auto kind = tk->kind;
  const auto fileId = tk->sourceFile;
  TokenValue value{};

  switch (tk->kind) {
    case TokenKind::T_IDENTIFIER: {
      kind = Lexer::classifyKeyword(tk->text);

      if (kind == TokenKind::T_IDENTIFIER) {
        value.idValue = control_->getIdentifier(tk->text);
      }

      break;
    }

    case TokenKind::T_CHARACTER_LITERAL:
      value.literalValue = control_->charLiteral(tk->text);
      break;

    case TokenKind::T_WIDE_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) {
        return;
      }
      value.literalValue = control_->wideStringLiteral(tk->text);
      break;

    case TokenKind::T_UTF8_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) {
        return;
      }
      value.literalValue = control_->utf8StringLiteral(tk->text);
      break;

    case TokenKind::T_UTF16_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) {
        return;
      }
      value.literalValue = control_->utf16StringLiteral(tk->text);
      break;

    case TokenKind::T_UTF32_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) {
        return;
      }
      value.literalValue = control_->utf32StringLiteral(tk->text);
      break;

    case TokenKind::T_STRING_LITERAL:
      if (updateStringLiteralValue(tokens.back(), tk)) {
        return;
      }
      value.literalValue = control_->stringLiteral(tk->text);
      break;

    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
      value.literalValue = control_->stringLiteral(tk->text);
      break;

    case TokenKind::T_INTEGER_LITERAL:
      value.literalValue = control_->integerLiteral(tk->text);
      break;

    case TokenKind::T_FLOATING_POINT_LITERAL:
      value.literalValue = control_->floatLiteral(tk->text);
      break;

    default:
      break;
  }  // switch

  if (tk->kind == TokenKind::T_GREATER_GREATER) {
    value.tokenKindValue = tk->kind;

    Token token(TokenKind::T_GREATER, tk->offset, 1);
    token.setFileId(fileId);
    token.setLeadingSpace(tk->space);
    token.setStartOfLine(tk->bol);
    tokens.push_back(token);

    token = Token(TokenKind::T_GREATER, tk->offset + 1, 1);
    token.setFileId(fileId);
    token.setLeadingSpace(false);
    token.setStartOfLine(false);
    tokens.push_back(token);
  } else {
    Token token(kind, tk->offset, tk->length, value);
    token.setFileId(fileId);
    token.setLeadingSpace(tk->space);
    token.setStartOfLine(tk->bol);
    tokens.push_back(token);
  }
};

auto Preprocessor::Private::tokenize(const std::string_view &source,
                                     int sourceFile, bool bol) -> TokList * {
  cxx::Lexer lex(source);
  lex.setKeepComments(true);
  lex.setPreprocessing(true);
  TokList *ts = nullptr;
  auto it = &ts;
  do {
    lex();

    if (lex.tokenKind() == TokenKind::T_COMMENT) {
      if (commentHandler_) {
        TokenValue tokenValue{};

        if (sourceFile) {
          const SourceFile *file = sourceFiles_[sourceFile - 1].get();

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
    auto tk = fromCurrentToken(lex, sourceFile);
    if (!lex.tokenIsClean()) tk->text = string(std::move(lex.text()));
    *it = cons(tk);
    it = (&(*it)->next);
  } while (lex.tokenKind() != cxx::TokenKind::T_EOF_SYMBOL);
  return ts;
}

auto Preprocessor::Private::expandTokens(TokIterator it, TokIterator last,
                                         bool inConditionalExpression)
    -> TokIterator {
  TokList *tokens = nullptr;
  auto out = &tokens;

  while (it != last) {
    if (it->is(TokenKind::T_EOF_SYMBOL)) {
      break;
    }
    it = expandOne(it, last, inConditionalExpression, [&](auto tok) {
      *out = cons(tok);
      out = (&(*out)->next);
    });
  }

  return TokIterator{tokens};
}

auto Preprocessor::Private::expand(
    const std::function<void(const Tok *)> &emitToken) -> Status {
  if (buffers_.empty()) return IsDone{};

  auto buffer = buffers_.back();
  buffers_.pop_back();

  // reconstruct the state from the active buffer
  auto source = buffer.source;
  currentFileName_ = source->fileName;
  currentPath_ = buffer.currentPath;
  includeDepth_ = buffer.includeDepth;

  auto it = TokIterator{buffer.ts};
  auto last = TokIterator{};

  while (it != last) {
    if (it->is(TokenKind::T_EOF_SYMBOL)) {
      break;
    }

    const auto [skipping, evaluating] = state();

    if (const auto start = it; it->bol && match(it, last, TokenKind::T_HASH)) {
      // skip the rest of the line
      it = skipLine(it, last);

      if (auto parsedInclude = parseDirective(source, start.toTokList())) {
        NeedToResolveInclude status{
            .preprocessor = *preprocessor_,
            .fileName = getHeaderName(parsedInclude->header),
            .isQuoted =
                std::holds_alternative<QuoteInclude>(parsedInclude->header),
            .isIncludeNext = parsedInclude->includeNext,
            .loc = parsedInclude->loc,
        };

        // suspend the current file and start processing the continuation
        buffers_.push_back(Buffer{
            .source = source,
            .currentPath = currentPath_,
            .ts = it.toTokList(),
            .includeDepth = includeDepth_,
        });

        // reset the token stream, so we can start processing the continuation
        it = last;

        return status;
      }
    } else if (skipping) {
      it = skipLine(++it, last);
    } else {
      it = expandOne(it, last,
                     /*inConditionalExpression*/ false, emitToken);
    }
  }

  if (buffers_.empty()) return IsDone{};

  return CanContinue{};
}

auto Preprocessor::Private::parseDirective(SourceFile *source, TokList *start)
    -> std::optional<ParsedIncludeDirective> {
  auto directive = start->next;

  if (!lookat(directive, TokenKind::T_IDENTIFIER)) return std::nullopt;

  const auto directiveKind = classifyDirective(directive->tok->text.data(),
                                               directive->tok->text.length());

  TokList *ts = directive->next;

  const auto [skipping, evaluating] = state();

  switch (directiveKind) {
    case PreprocessorDirectiveKind::T_INCLUDE_NEXT:
    case PreprocessorDirectiveKind::T_INCLUDE: {
      if (skipping) break;
      return parseIncludeDirective(directive, ts);
    }

    case PreprocessorDirectiveKind::T_DEFINE: {
      if (skipping) break;
      defineMacro(copyLine(ts));
      break;
    }

    case PreprocessorDirectiveKind::T_UNDEF: {
      if (skipping) break;

      if (bol(ts)) {
        error(ts, std::format("missing macro name"));
        break;
      }

      auto macroName = expectId(ts);

      if (!macroName.empty()) {
        auto it = macros_.find(macroName);

        if (it != macros_.end()) {
          macros_.erase(it);
        }
      }

      break;
    }

    case PreprocessorDirectiveKind::T_IFDEF: {
      const auto value = isDefined(ts->tok);
      if (value) {
        pushState(std::tuple(skipping, false));
      } else {
        pushState(std::tuple(true, !skipping));
      }

      break;
    }

    case PreprocessorDirectiveKind::T_IFNDEF: {
      const auto value = !isDefined(ts->tok);
      if (value) {
        pushState(std::tuple(skipping, false));
      } else {
        pushState(std::tuple(true, !skipping));
      }

      break;
    }

    case PreprocessorDirectiveKind::T_IF: {
      if (skipping) {
        pushState(std::tuple(true, false));
      } else {
        const auto value = constantExpression(ts);
        if (value) {
          pushState(std::tuple(skipping, false));
        } else {
          pushState(std::tuple(true, !skipping));
        }
      }

      break;
    }

    case PreprocessorDirectiveKind::T_ELIF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else {
        const auto value = constantExpression(ts);
        if (value) {
          setState(std::tuple(!evaluating, false));
        } else {
          setState(std::tuple(true, evaluating));
        }
      }

      break;
    }

    case PreprocessorDirectiveKind::T_ELIFDEF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else {
        const auto value = isDefined(ts->tok);
        if (value) {
          setState(std::tuple(!evaluating, false));
        } else {
          setState(std::tuple(true, evaluating));
        }
      }

      break;
    }

    case PreprocessorDirectiveKind::T_ELIFNDEF: {
      if (!evaluating) {
        setState(std::tuple(true, false));
      } else {
        const auto value = isDefined(ts->tok);
        if (!value) {
          setState(std::tuple(!evaluating, false));
        } else {
          setState(std::tuple(true, evaluating));
        }
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
        error(directive, "unexpected '#endif'");
      }
      if (source->headerProtection &&
          evaluating_.size() == source->headerProtectionLevel) {
        if (!lookat(ts, TokenKind::T_EOF_SYMBOL)) {
          ifndefProtectedFiles_.erase(currentFileName_);
        }
      }
      break;
    }

    case PreprocessorDirectiveKind::T_LINE: {
      // ###
      std::ostringstream out;
      printLine(start, out);
      break;
    }

    case PreprocessorDirectiveKind::T_PRAGMA: {
      if (skipping) break;
#if 0
        std::ostringstream out;
        printLine(start, out);
        std::cerr << std::format("** todo pragma: ");
        printLine(ts, std::cerr);
        std::cerr << std::format("\n");
        // cxx_runtime_error(out.str());
#endif
      break;
    }
    case PreprocessorDirectiveKind::T_ERROR: {
      if (skipping) break;

      std::ostringstream out;
      printLine(start, out, /*nl=*/false);
      error(directive, std::format("{}", out.str()));

      break;
    }

    case PreprocessorDirectiveKind::T_WARNING: {
      if (skipping) break;

      std::ostringstream out;
      printLine(start, out, /*nl=*/false);
      warning(directive, std::format("{}", out.str()));

      break;
    }

    default:
      break;
  }  // switch

  return std::nullopt;
}

auto Preprocessor::Private::parseIncludeDirective(TokList *directive,
                                                  TokList *ts)
    -> std::optional<ParsedIncludeDirective> {
  if (lookat(ts, TokenKind::T_IDENTIFIER)) {
    auto eol = skipLine(TokIterator{ts}, TokIterator{});
    ts = expandTokens(TokIterator{ts}, eol, /*expression*/ false).toTokList();
  }

  auto loc = ts;
  if (!ts || lookat(ts, TokenKind::T_EOF_SYMBOL)) loc = directive;

  const bool isIncludeNext = directive->tok->text == "include_next";

  if (auto [rest, headerFile] = parseHeaderName(ts); headerFile.has_value()) {
    auto parsedInclude = ParsedIncludeDirective{
        .header = *headerFile,
        .includeNext = isIncludeNext,
        .loc = loc,
    };

    return parsedInclude;
  }

  return std::nullopt;
}

auto Preprocessor::Private::resolveIncludeDirective(
    const ParsedIncludeDirective &directive) -> SourceFile * {
  const auto path = resolve(directive.header, directive.includeNext);

  if (!path) {
    const auto file = getHeaderName(directive.header);
    error(directive.loc, std::format("file '{}' not found", file));
    return nullptr;
  }

  std::string currentFileName = path->string();

  if (auto it = ifndefProtectedFiles_.find(currentFileName);
      it != ifndefProtectedFiles_.end() && macros_.contains(it->second)) {
    return nullptr;
  }

  auto sourceFile = findSourceFile(currentFileName);

  if (sourceFile && sourceFile->pragmaOnceProtected) {
    return nullptr;
  }

  if (!sourceFile) {
    sourceFile = createSourceFile(path->string(), readFile(*path));

    sourceFile->pragmaOnceProtected =
        checkPragmaOnceProtected(sourceFile->tokens);

    sourceFile->headerProtection = checkHeaderProtection(sourceFile->tokens);

    if (sourceFile->headerProtection) {
      sourceFile->headerProtectionLevel = evaluating_.size();

      ifndefProtectedFiles_.insert_or_assign(
          sourceFile->fileName, sourceFile->headerProtection->tok->text);
    }
  }

  if (willIncludeHeader_) {
    willIncludeHeader_(currentFileName, includeDepth_ + 1);
  }

  return sourceFile;
}

auto Preprocessor::Private::parseHeaderName(TokList *ts)
    -> std::tuple<TokList *, std::optional<Include>> {
  if (lookat(ts, TokenKind::T_STRING_LITERAL)) {
    auto file = ts->tok->text.substr(1, ts->tok->text.length() - 2);
    Include headerFile = QuoteInclude(std::string(file));
    return {ts->next, headerFile};
  }

  if (match(ts, TokenKind::T_LESS)) {
    std::string file;
    while (ts && !lookat(ts, TokenKind::T_EOF_SYMBOL) && !bol(ts)) {
      if (match(ts, TokenKind::T_GREATER)) break;
      file += ts->tok->text;
      ts = ts->next;
    }
    Include headerFile = SystemInclude(file);
    return {ts, headerFile};
  }

  return {ts, std::nullopt};
}

auto Preprocessor::Private::expandOne(
    TokIterator it, TokIterator last, bool inConditionalExpression,
    const std::function<void(const Tok *)> &emitToken) -> TokIterator {
  if (it == last) return last;

  if (auto continuation = replaceIsDefinedMacro(
          it.toTokList(), inConditionalExpression, emitToken)) {
    return TokIterator{continuation};
  }

  if (auto continuation = expandMacro(it.toTokList())) {
    return TokIterator{continuation};
  }

  emitToken(&*it);

  ++it;

  return it;
}

auto Preprocessor::Private::replaceIsDefinedMacro(
    TokList *ts, bool inConditionalExpression,
    const std::function<void(const Tok *)> &emitToken) -> TokList * {
  if (!inConditionalExpression) {
    return nullptr;
  }

  auto start = ts->tok;

  if (!matchId(ts, "defined")) {
    return nullptr;
  }

  bool value = false;

  if (match(ts, TokenKind::T_LPAREN)) {
    value = isDefined(ts->tok);
    ts = ts->next;
    expect(ts, TokenKind::T_RPAREN);
  } else {
    value = isDefined(ts->tok);
    ts = ts->next;
  }

  auto tk = gen(TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
  tk->sourceFile = start->sourceFile;
  tk->space = start->space;
  tk->bol = start->bol;
  emitToken(tk);

  return ts;
}

auto Preprocessor::Private::expandMacro(TokList *ts) -> TokList * {
  struct ExpandMacro {
    Private &self;
    TokList *ts = nullptr;
    const Macro *macro = nullptr;

    auto operator()(const ObjectMacro &) -> TokList * {
      return self.expandObjectLikeMacro(ts, macro);
    }

    auto operator()(const FunctionMacro &) -> TokList * {
      if (!self.lookat(ts->next, TokenKind::T_LPAREN)) return nullptr;

      return self.expandFunctionLikeMacro(ts, macro);
    }

    auto operator()(const BuiltinObjectMacro &macro) -> TokList * {
      return macro.expand(MacroExpansionContext{.ts = ts});
    }

    auto operator()(const BuiltinFunctionMacro &macro) -> TokList * {
      if (!self.lookat(ts->next, TokenKind::T_LPAREN)) return nullptr;

      return macro.expand(MacroExpansionContext{.ts = ts});
    }
  };

  if (auto macro = lookupMacro(ts->tok)) {
    return std::visit(ExpandMacro{.self = *this, .ts = ts, .macro = macro},
                      *macro);
  }

  return nullptr;
}

auto Preprocessor::Private::expandObjectLikeMacro(TokList *ts, const Macro *m)
    -> TokList * {
  auto macro = &std::get<ObjectMacro>(*m);
  const Tok *tk = ts->tok;

  const auto hideset = makeUnion(tk->hideset, tk->text);
  auto expanded = substitute(ts, m, {}, hideset);

  if (!expanded) {
    return ts->next;
  }

  // assert(expanded->tok->generated);
  const_cast<Tok *>(expanded->tok)->space = tk->space;
  const_cast<Tok *>(expanded->tok)->bol = tk->bol;

  auto it = expanded;

  while (it->next) it = it->next;
  (it)->next = ts->next;

  return expanded;
}

auto Preprocessor::Private::expandFunctionLikeMacro(TokList *ts, const Macro *m)
    -> TokList * {
  auto macro = &std::get<FunctionMacro>(*m);

  assert(lookat(ts->next, TokenKind::T_LPAREN));

  const Tok *tk = ts->tok;

  auto [args, rest, hideset] =
      parseArguments(ts, macro->formals.size(), macro->variadic);

  auto hs = makeUnion(makeIntersection(tk->hideset, hideset), tk->text);

  auto expanded = substitute(ts, m, args, hs);

  if (!expanded) {
    return rest;
  }

  // assert(expanded->tok->generated);

  const_cast<Tok *>(expanded->tok)->space = tk->space;
  const_cast<Tok *>(expanded->tok)->bol = tk->bol;

  auto it = expanded;
  while (it->next) it = it->next;
  (it)->next = rest;

  return expanded;
}

auto Preprocessor::Private::substitute(TokList *pointOfSubstitution,
                                       const Macro *macro,
                                       const std::vector<TokRange> &actuals,
                                       const Hideset *hideset) -> TokList * {
  TokList *os = nullptr;
  auto **ip = (&os);

  std::unordered_set<const Tok *> macroTokens;

  auto appendTokens = [&](TokList *rs) {
    if (!*ip) {
      *ip = (rs);
    } else {
      (*ip)->next = (rs);
    }
    while (*ip && (*ip)->next) ip = (&(*ip)->next);
  };

  auto appendToken = [&](const Tok *tk) {
    if (macroTokens.contains(tk)) {
      // todo: make a copy of tk and override its location to be the same as
      // the point of substitution.
      auto copyTk = copy(tk);
      copyTk->sourceFile = pointOfSubstitution->tok->sourceFile;
      copyTk->offset = pointOfSubstitution->tok->offset;
      copyTk->length = pointOfSubstitution->tok->length;
      appendTokens(cons(copyTk));
    } else {
      appendTokens(cons(tk));
    }
  };

  TokList *macroBody = getMacroBody(*macro);

  for (auto it = macroBody; it; it = it->next) {
    macroTokens.insert(it->tok);
  }

  // std::cerr << std::format("macro body has {} tokens\n", macroTokens.size());

  // set the token stream to the macro body
  TokList *ts = macroBody;

  while (ts && !lookat(ts, TokenKind::T_EOF_SYMBOL)) {
    if (lookat(ts, TokenKind::T_HASH, TokenKind::T_IDENTIFIER)) {
      const auto saved = ts;
      ts = ts->next;

      if (auto actual = lookupMacroArgument(ts, macro, actuals)) {
        auto [startArg, endArg] = *actual;
        if (startArg != endArg) {
          appendToken(stringize(toTokList(startArg, endArg)));
        }
        continue;
      }
      ts = saved;
    }

    if (lookat(ts, TokenKind::T_HASH_HASH, TokenKind::T_IDENTIFIER)) {
      const auto saved = ts;
      ts = ts->next;

      if (auto actual = lookupMacroArgument(ts, macro, actuals)) {
        auto [startArg, endArg] = *actual;
        if (startArg != endArg) {
          (*ip)->tok = merge((*ip)->tok, &*startArg);
          for (auto it = ++startArg; it != endArg; ++it) {
            appendToken(&*it);
          }
        }
        continue;
      }
      ts = saved;
    }

    if (ts->next && lookat(ts, TokenKind::T_HASH_HASH)) {
      (*ip)->tok = merge((*ip)->tok, ts->next->tok);
      ts = ts->next->next;
      continue;
    }

    if (lookat(ts, TokenKind::T_IDENTIFIER, TokenKind::T_HASH_HASH)) {
      if (auto actual = lookupMacroArgument(ts, macro, actuals)) {
        auto [startArg, endArg] = *actual;
        if (startArg != endArg) {
          for (auto it = startArg; it != endArg; ++it) {
            appendToken(&*it);
          }
        } else {
          // placemarker
          appendToken(gen(TokenKind::T_IDENTIFIER, ""));
        }
        continue;
      }
    }

    if (auto actual = lookupMacroArgument(ts, macro, actuals)) {
      auto [startArg, endArg] = *actual;
      auto line = expandTokens(startArg, endArg,
                               /*expression*/ false);
      if (line != TokIterator{}) {
        appendTokens(line.toTokList());
      }
      continue;
    }

    appendToken(ts->tok);
    ts = ts->next;
  }

  return instantiate(os, hideset);
}

auto Preprocessor::Private::lookupMacroArgument(
    TokList *&ts, const Macro *m, const std::vector<TokRange> &actuals)
    -> std::optional<TokRange> {
  if (!isFunctionLikeMacro(*m)) return std::nullopt;

  const FunctionMacro *macro = &std::get<FunctionMacro>(*m);

  if (!lookat(ts, TokenKind::T_IDENTIFIER)) {
    return std::nullopt;
  }

  if (macro->variadic) {
    if (matchId(ts, "__VA_ARGS__")) {
      if (actuals.size() > macro->formals.size()) {
        auto arg = actuals.back();
        return arg;
      }

      return {TokRange{}};
    }

    if (lookat(ts, "__VA_OPT__", TokenKind::T_LPAREN)) {
      const auto [args, rest, hideset] =
          parseArguments(ts, /*formal count*/ 0, /*ignore comma*/ true);

      ts = rest;

      if (!args.empty() && actuals.size() > macro->formals.size()) {
        auto arg = args.front();
        ts = snoc(toTokList(arg), ts);
        return {TokRange{}};
      }

      return {TokRange{}};
    }
  }

  const auto formal = ts->tok->text;

  for (std::size_t i = 0; i < macro->formals.size(); ++i) {
    if (macro->formals[i] == formal) {
      ts = ts->next;

      if (i < actuals.size()) {
        return actuals[i];
      }

      return {TokRange{}};
    }
  }

  return std::nullopt;
}

auto Preprocessor::Private::checkPragmaOnceProtected(TokList *ts) const
    -> bool {
  if (!ts) return false;
  if (!match(ts, TokenKind::T_HASH)) return false;
  if (bol(ts) || !matchId(ts, "pragma")) return false;
  if (bol(ts) || !matchId(ts, "once")) return false;
  return true;
}

auto Preprocessor::Private::checkHeaderProtection(TokList *ts) const
    -> TokList * {
  if (!ts) return nullptr;
  if (!match(ts, TokenKind::T_HASH)) return nullptr;
  if (bol(ts) || !matchId(ts, "ifndef")) return nullptr;
  TokList *prot = ts;
  if (bol(ts) || !match(ts, TokenKind::T_IDENTIFIER)) return nullptr;
  if (!bol(ts) || !match(ts, TokenKind::T_HASH)) return nullptr;
  if (bol(ts) || !matchId(ts, "define")) return nullptr;
  if (bol(ts) || !matchId(ts, prot->tok->text)) return nullptr;
  return prot;
}

auto Preprocessor::Private::copyLine(TokList *ts) -> TokList * {
  assert(ts);
  TokList *line = nullptr;
  auto it = &line;
  auto lastTok = ts->tok;
  for (; ts && !lookat(ts, TokenKind::T_EOF_SYMBOL) && !bol(ts);
       ts = ts->next) {
    *it = cons(ts->tok);
    lastTok = ts->tok;
    it = (&(*it)->next);
  }
  auto eol = gen(TokenKind::T_EOF_SYMBOL, std::string_view());
  eol->sourceFile = lastTok->sourceFile;
  eol->offset = lastTok->offset + lastTok->length;
  *it = cons(eol);
  return line;
}

auto Preprocessor::Private::constantExpression(TokList *ts) -> long {
  auto line = copyLine(ts);
  auto it = expandTokens(TokIterator{line}, TokIterator{},
                         /*inConditionalExpression*/ true);
  auto e = it.toTokList();
  return conditionalExpression(e);
}

auto Preprocessor::Private::conditionalExpression(TokList *&ts) -> long {
  if (!ts) return 0;
  const auto value = binaryExpression(ts);
  if (!match(ts, TokenKind::T_QUESTION)) return value;
  const auto iftrue = conditionalExpression(ts);
  expect(ts, TokenKind::T_COLON);
  const auto iffalse = conditionalExpression(ts);
  return value ? iftrue : iffalse;
}

static auto prec(TokList *ts) -> int {
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

  if (!ts) return -1;

  switch (ts->tok->kind) {
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
  }  // switch
}

auto Preprocessor::Private::binaryExpression(TokList *&ts) -> long {
  auto e = unaryExpression(ts);
  return binaryExpressionHelper(ts, e, 0);
}

auto Preprocessor::Private::binaryExpressionHelper(TokList *&ts, long lhs,
                                                   int minPrec) -> long {
  while (prec(ts) >= minPrec) {
    const auto p = prec(ts);
    const auto op = ts->tok->kind;
    ts = ts->next;
    auto rhs = unaryExpression(ts);
    while (prec(ts) > p) {
      rhs = binaryExpressionHelper(ts, rhs, prec(ts));
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
    }  // switch
  }
  return lhs;
}

auto Preprocessor::Private::unaryExpression(TokList *&ts) -> long {
  if (match(ts, TokenKind::T_MINUS)) {
    return -unaryExpression(ts);
  }
  if (match(ts, TokenKind::T_PLUS)) {
    return unaryExpression(ts);
  }
  if (match(ts, TokenKind::T_TILDE)) {
    return ~unaryExpression(ts);
  }
  if (match(ts, TokenKind::T_EXCLAIM)) {
    return !unaryExpression(ts);
  }
  return primaryExpression(ts);
}

auto Preprocessor::Private::primaryExpression(TokList *&ts) -> long {
  const auto tk = ts->tok;

  if (match(ts, TokenKind::T_INTEGER_LITERAL)) {
    return IntegerLiteral::Components::from(tk->text.data()).value;
  } else if (matchId(ts, "true")) {
    return 1;
  } else if (matchId(ts, "false")) {
    return 0;
  } else if (match(ts, TokenKind::T_LPAREN)) {
    auto result = conditionalExpression(ts);
    expect(ts, TokenKind::T_RPAREN);
    return result;
  }

  ts = ts->next;
  return 0;
}

auto Preprocessor::Private::instantiate(TokList *ts, const Hideset *hideset)
    -> TokList * {
  for (auto ip = ts; ip; ip = ip->next) {
    if (ip->tok->hideset != hideset) {
      (ip)->tok = withHideset(ip->tok, hideset);
    }
  }
  return ts;
}

auto Preprocessor::Private::parseArguments(TokList *ts, int formalCount,
                                           bool ignoreComma)
    -> std::tuple<std::vector<TokRange>, TokList *, const Hideset *> {
  assert(lookat(ts, TokenKind::T_IDENTIFIER, TokenKind::T_LPAREN));

  auto parsedArgs = ParseArguments{*this}(
      TokIterator{ts->next}, EndOfFileSentinel{}, formalCount, ignoreComma);

  if (!parsedArgs) {
    return std::tuple(std::vector<TokRange>(), ts->next, nullptr);
  }

  return std::tuple(std::move(parsedArgs->args), parsedArgs->it.toTokList(),
                    parsedArgs->hideset);
}

auto Preprocessor::Private::stringize(TokList *ts) -> const Tok * {
  std::string s;

  const auto start = ts;

  for (; ts; ts = ts->next) {
    if (!s.empty() && (ts->tok->space || bol(ts))) s += ' ';
    s += ts->tok->text;
  }

  std::string o;

  o += '"';
  for (auto c : s) {
    if (c == '\\') {
      o += "\\\\";
    } else if (c == '"') {
      o += "\\\"";
    } else {
      o += c;
    }
  }
  o += '"';

  auto tk = gen(TokenKind::T_STRING_LITERAL, string(o));
  if (start) {
    tk->sourceFile = start->tok->sourceFile;
    tk->offset = start->tok->offset;
  }

  return tk;
}

auto Preprocessor::Private::string(std::string s) -> std::string_view {
  return std::string_view(scratchBuffer_.emplace_front(std::move(s)));
}

auto Preprocessor::Private::parseMacroDefinition(TokList *ts) -> Macro {
  const auto name = ts->tok->text;
  ts = ts->next;

  if (lookat(ts, TokenKind::T_LPAREN) && !ts->tok->space) {
    // parse function like macro
    ts = ts->next;

    std::vector<std::string_view> formals;
    bool variadic = false;

    if (!match(ts, TokenKind::T_RPAREN)) {
      variadic = match(ts, TokenKind::T_DOT_DOT_DOT);
      if (!variadic) {
        auto formal = expectId(ts);
        if (!formal.empty()) formals.push_back(formal);
        while (match(ts, TokenKind::T_COMMA)) {
          variadic = match(ts, TokenKind::T_DOT_DOT_DOT);
          if (variadic) break;
          auto formal = expectId(ts);
          if (!formal.empty()) formals.push_back(formal);
        }
        if (!variadic) variadic = match(ts, TokenKind::T_DOT_DOT_DOT);
      }
      expect(ts, TokenKind::T_RPAREN);
    }

    return FunctionMacro(name, std::move(formals), ts, variadic);
  }

  return ObjectMacro(name, ts);
}

void Preprocessor::Private::defineMacro(TokList *ts) {
#if 0
  std::cout << std::format("*** defining macro: ");
  printLine(ts, std::cout);
  std::cout << std::format("\n");
#endif

  auto macro = parseMacroDefinition(ts);
  const auto name = getMacroName(macro);

  if (auto body = getMacroBody(macro)) {
    const_cast<Tok *>(body->tok)->space = false;
    const_cast<Tok *>(body->tok)->bol = false;
  }

  if (auto it = macros_.find(name); it != macros_.end()) {
    auto previousMacroBody = getMacroBody(it->second);
    if (!TokList::isSame(getMacroBody(macro), previousMacroBody)) {
      warning(ts, std::format("'{}' macro redefined", name));
    }

    macros_.erase(it);
  }

  macros_.insert_or_assign(name, std::move(macro));
}

auto Preprocessor::Private::merge(const Tok *left, const Tok *right)
    -> const Tok * {
  if (!left) return right;
  if (!right) return left;
  const auto hideset = makeIntersection(left->hideset, right->hideset);
  auto text = string(std::string(left->text) + std::string(right->text));
  Lexer lex(text);
  lex.setPreprocessing(true);
  lex.next();
  auto tok = gen(lex.tokenKind(), lex.tokenText(), hideset);
  tok->sourceFile = left->sourceFile;
  tok->offset = left->offset;
  return tok;
}

auto Preprocessor::Private::skipLine(TokIterator it, TokIterator last)
    -> TokIterator {
  for (; it != last; ++it) {
    if (it->kind == TokenKind::T_EOF_SYMBOL) break;
    if (it->bol) break;
  }
  return it;
}

auto Preprocessor::Private::lookupMacro(const Tok *tk) const -> const Macro * {
  if (!tk || tk->isNot(TokenKind::T_IDENTIFIER)) {
    return nullptr;
  }

  if (auto it = macros_.find(tk->text); it != macros_.end()) {
    const auto disabled = tk->hideset && tk->hideset->contains(tk->text);
    if (!disabled) {
      return &it->second;
    }
  }
  return nullptr;
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
  }  // switch
}

static auto needSpace(const Tok *prev, const Tok *current) -> bool {
  if (!prev || current->space) return current->space;
  return wantSpace(prev->kind) && wantSpace(current->kind);
}

void Preprocessor::Private::print(TokList *ts, std::ostream &out) const {
  bool first = true;
  for (const Tok *prevTk = nullptr; ts; ts = ts->next) {
    auto tk = ts->tok;
    if (tk->text.empty()) continue;
    if (tk->bol) {
      out << "\n";
    } else if (!first && needSpace(prevTk, tk)) {
      out << " ";
    }
    out << std::format("{}", tk->text);
    prevTk = tk;
    first = false;
  }
}

void Preprocessor::Private::printLine(TokList *ts, std::ostream &out,
                                      bool nl) const {
  bool first = true;
  for (const Tok *prevTk = nullptr; ts; ts = ts->next) {
    auto tk = ts->tok;
    if (tk->text.empty()) continue;
    if (!first && needSpace(prevTk, tk)) out << std::format(" ");
    out << std::format("{}", tk->text);
    prevTk = tk;
    first = false;
    if (ts->next && bol(ts->next)) break;
  }
  if (nl) out << std::format("\n");
}

Preprocessor::Preprocessor(Control *control,
                           DiagnosticsClient *diagnosticsClient)
    : d(std::make_unique<Private>()) {
  d->preprocessor_ = this;
  d->control_ = control;
  d->diagnosticsClient_ = diagnosticsClient;
  d->initialize();
}

Preprocessor::~Preprocessor() = default;

auto Preprocessor::diagnosticsClient() const -> DiagnosticsClient * {
  return d->diagnosticsClient_;
}

auto Preprocessor::commentHandler() const -> CommentHandler * {
  return d->commentHandler_;
}

void Preprocessor::setCommentHandler(CommentHandler *commentHandler) {
  d->commentHandler_ = commentHandler;
}

auto Preprocessor::delegate() const -> PreprocessorDelegate * {
  return d->delegate_;
}

void Preprocessor::setDelegate(PreprocessorDelegate *delegate) {
  d->delegate_ = delegate;
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

void Preprocessor::setFileExistsFunction(
    std::function<bool(std::string)> fileExists) {
  d->fileExists_ = std::move(fileExists);
}

void Preprocessor::setReadFileFunction(
    std::function<std::string(std::string)> readFile) {
  d->readFile_ = std::move(readFile);
}

void Preprocessor::setOnWillIncludeHeader(
    std::function<void(const std::string &, int)> willIncludeHeader) {
  d->willIncludeHeader_ = std::move(willIncludeHeader);
}

void Preprocessor::squeeze() { d->pool_.reset(); }

void Preprocessor::preprocess(std::string source, std::string fileName,
                              std::vector<Token> &tokens) {
  struct {
    Preprocessor &self;
    bool done = false;

    void operator()(const IsDone &) { done = true; }

    void operator()(const CanContinue &) {
      // keep going
    }

    void operator()(const NeedToResolveInclude &status) {
      // the header file resolution may be asynchronous
      Include header;

      if (status.isQuoted) {
        header = QuoteInclude(status.fileName);
      } else {
        header = SystemInclude(status.fileName);
      }

      Private::ParsedIncludeDirective parsedInclude{
          .header = header,
          .includeNext = status.isIncludeNext,
          .loc = static_cast<TokList *>(const_cast<void *>(status.loc)),
      };

      auto continuation = self.d->resolveIncludeDirective(parsedInclude);
      if (!continuation) return;

      // make the continuation the current file
      fs::path dirpath = fs::path(continuation->fileName);
      dirpath.remove_filename();

      self.d->buffers_.push_back(Private::Buffer{
          .source = continuation,
          .currentPath = dirpath,
          .ts = continuation->tokens,
          .includeDepth = self.d->includeDepth_ + 1,
      });
    }

    void operator()(const NeedToKnowIfFileExists &status) {
      auto exists = self.d->fileExists(status.fileName);
      status.setFileExists(exists);
    }

    void operator()(const NeedToReadFile &status) {
      auto source = self.d->readFile(status.fileName);
      status.setContents(std::move(source));
    }
  } state{*this};

  beginPreprocessing(std::move(source), std::move(fileName), tokens);

  while (!state.done) {
    std::visit(state, continuePreprocessing(tokens));
  }

  endPreprocessing(tokens);
}

void Preprocessor::NeedToKnowIfFileExists::setFileExists(bool exists) const {}

void Preprocessor::NeedToReadFile::setContents(std::string contents) const {}

void Preprocessor::beginPreprocessing(std::string source, std::string fileName,
                                      std::vector<Token> &tokens) {
  assert(!d->findSourceFile(fileName));

  auto sourceFile = d->createSourceFile(std::move(fileName), std::move(source));

  auto dirpath = fs::path(sourceFile->fileName);
  dirpath.remove_filename();

  d->buffers_.push_back(Private::Buffer{
      .source = sourceFile,
      .currentPath = dirpath,
      .ts = sourceFile->tokens,
      .includeDepth = d->includeDepth_,
  });

  if (!tokens.empty()) {
    assert(tokens.back().is(TokenKind::T_EOF_SYMBOL));
    tokens.pop_back();
  }

  if (tokens.empty()) {
    tokens.emplace_back(TokenKind::T_ERROR);
  }
}

void Preprocessor::endPreprocessing(std::vector<Token> &tokens) {
  if (tokens.empty()) return;

  // assume the main source file is the first one
  const auto mainSourceFileId = 1;

  // place the EOF token at the end of the main source file
  const auto offset = d->sourceFiles_[mainSourceFileId - 1]->source.size();
  auto &tk = tokens.emplace_back(TokenKind::T_EOF_SYMBOL, offset);
  tk.setFileId(mainSourceFileId);
}

auto Preprocessor::continuePreprocessing(std::vector<Token> &tokens) -> Status {
  auto emitToken = [&](const Tok *tk) { d->finalizeToken(tokens, tk); };
  return d->expand(emitToken);
}

void Preprocessor::getPreprocessedText(const std::vector<Token> &tokens,
                                       std::ostream &out) const {
  // ### print tokens
  std::size_t index = 1;
  std::uint32_t lastFileId = std::numeric_limits<std::uint32_t>::max();

  bool atStartOfLine = true;

  while (index + 1 < tokens.size()) {
    const auto &token = tokens[index++];

    if (const auto fileId = token.fileId();
        !d->omitLineMarkers_ && fileId && fileId != lastFileId) {
      if (lastFileId != std::numeric_limits<std::uint32_t>::max()) {
        out << '\n';
      }
      const auto pos = tokenStartPosition(token);
      out << std::format("# {} \"{}\"\n", pos.line, pos.fileName);
      lastFileId = fileId;
      atStartOfLine = true;
    } else if (token.startOfLine()) {
      atStartOfLine = true;
      out << '\n';
    } else if (token.leadingSpace()) {
      atStartOfLine = false;
      out << ' ';
    } else if (index > 2) {
      const auto &prevToken = tokens[index - 2];
      std::string s = prevToken.spell();
      s += token.spell();
      Lexer lex(s);
      // lex.setPreprocessing(true);
      lex.next();
      if (lex.tokenKind() != prevToken.kind()) {
        out << ' ';
      } else if (lex.tokenLength() != prevToken.length()) {
        out << ' ';
      }
    }

    if (atStartOfLine) {
      const auto pos = tokenStartPosition(token);
      if (pos.column > 0) {
        for (std::uint32_t i = 0; i < pos.column - 1; ++i) {
          out << ' ';
        }
      }
      atStartOfLine = false;
    }

    out << token.spell();
  }

  out << '\n';
}

auto Preprocessor::systemIncludePaths() const
    -> const std::vector<std::string> & {
  return d->systemIncludePaths_;
}

void Preprocessor::addSystemIncludePath(std::string path) {
  d->systemIncludePaths_.push_back(std::move(path));
}

void Preprocessor::defineMacro(const std::string &name,
                               const std::string &body) {
  auto s = d->string(name + " " + body);
  auto tokens = d->tokenize(s, /*sourceFile=*/0, false);
  d->defineMacro(tokens);
}

void Preprocessor::undefMacro(const std::string &name) {
  auto it = d->macros_.find(name);
  if (it != d->macros_.end()) d->macros_.erase(it);
}

void Preprocessor::printMacros(std::ostream &out) const {
  struct {
    const Preprocessor &self;
    std::ostream &out;

    void operator()(const FunctionMacro &macro) {
      auto d = self.d.get();

      out << std::format("#define {}", macro.name);

      out << std::format("(");
      for (std::size_t i = 0; i < macro.formals.size(); ++i) {
        if (i > 0) out << ",";
        out << std::format("{}", macro.formals[i]);
      }

      if (macro.variadic) {
        if (!macro.formals.empty()) out << std::format(",");
        out << std::format("...");
      }

      out << std::format(")");

      if (macro.body) {
        out << std::format(" ");
        d->print(macro.body, out);
      }

      out << std::format("\n");
    }

    void operator()(const ObjectMacro &macro) {
      auto d = self.d.get();

      out << std::format("#define {}", macro.name);

      if (macro.body) {
        out << std::format(" ");
        d->print(macro.body, out);
      }

      out << std::format("\n");
    }

    void operator()(const BuiltinObjectMacro &) {}
    void operator()(const BuiltinFunctionMacro &) {}

  } printMacro{*this, out};

  for (const auto &[name, macro] : d->macros_) {
    std::visit(printMacro, macro);
  }
}

auto Preprocessor::tokenStartPosition(const Token &token) const
    -> SourcePosition {
  if (token.fileId() == 0) {
    return {};
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];
  SourcePosition pos;
  sourceFile.getTokenStartPosition(token.offset(), &pos.line, &pos.column,
                                   &pos.fileName);
  return pos;
}

auto Preprocessor::tokenEndPosition(const Token &token) const
    -> SourcePosition {
  if (token.fileId() == 0) {
    return {};
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];

  SourcePosition pos;
  sourceFile.getTokenStartPosition(token.offset() + token.length(), &pos.line,
                                   &pos.column, &pos.fileName);
  return pos;
}

auto Preprocessor::getTextLine(const Token &token) const -> std::string_view {
  if (token.fileId() == 0) return {};
  const SourceFile *file = d->sourceFiles_[token.fileId() - 1].get();
  const auto pos = tokenStartPosition(token);
  std::string_view source = file->source;
  const auto &lines = file->lines;
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

auto Preprocessor::getTokenText(const Token &token) const -> std::string_view {
  if (token.fileId() == 0) return {};
  const SourceFile *file = d->sourceFiles_[token.fileId() - 1].get();
  std::string_view source = file->source;
  return source.substr(token.offset(), token.length());
}

}  // namespace cxx
