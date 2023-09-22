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

#include <cxx/preprocessor.h>

// cxx
#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/private/path.h>

// fmt
#include <cxx/private/format.h>

// utf8
#include <utf8/unchecked.h>

// stl
#include <cxx/preprocessor.h>

#include <cassert>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace {

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

using Include = std::variant<std::monostate, SystemInclude, QuoteInclude>;

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

struct Tok final : Managed {
  std::string_view text;
  const Hideset *hideset = nullptr;
  uint32_t offset = 0;
  uint32_t length = 0;
  uint32_t sourceFile = 0;
  TokenKind kind = TokenKind::T_EOF_SYMBOL;
  uint16_t bol : 1 = false;
  uint16_t space : 1 = false;
  uint16_t generated : 1 = false;

  Tok(const Tok &other) = default;
  auto operator=(const Tok &other) -> Tok & = default;

  Tok(Tok &&other) = default;
  auto operator=(Tok &&other) -> Tok & = default;

  [[nodiscard]] auto is(TokenKind k) const -> bool { return kind == k; }

  [[nodiscard]] auto isNot(TokenKind k) const -> bool { return kind != k; }

  static auto WithHideset(Arena *pool, const Tok *tok, const Hideset *hideset)
      -> Tok * {
    return new (pool) Tok(tok, hideset);
  }

  static auto FromCurrentToken(Arena *pool, const Lexer &lex, int sourceFile)
      -> Tok * {
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

  static auto Gen(Arena *pool, TokenKind kind, const std::string_view &text,
                  const Hideset *hideset = nullptr) -> Tok * {
    auto tk = new (pool) Tok();
    tk->kind = kind;
    tk->text = text;
    tk->hideset = hideset;
    tk->generated = true;
    tk->space = true;
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
  const Tok *head = nullptr;
  const TokList *tail = nullptr;

  explicit TokList(const Tok *head, const TokList *tail = nullptr)
      : head(head), tail(tail) {}
};

struct Macro {
  std::vector<std::string_view> formals;
  const TokList *body = nullptr;
  bool objLike = true;
  bool variadic = false;

  auto operator!=(const Macro &other) const -> bool {
    return !operator==(other);
  }

  auto operator==(const Macro &other) const -> bool {
    if (formals != other.formals) return false;
    if (objLike != other.objLike) return false;
    if (variadic != other.variadic) return false;
    return isSame(body, other.body);
  }

  auto isSame(const TokList *ls, const TokList *rs) const -> bool {
    if (ls == rs) return true;
    if (!ls || !rs) return false;
    if (ls->head->kind != rs->head->kind) return false;
    if (ls->head->text != rs->head->text) return false;
    return isSame(ls->tail, rs->tail);
  }
};

struct SourceFile {
  std::string fileName;
  std::string source;
  std::vector<int> lines;
  const TokList *tokens = nullptr;
  int id;

  SourceFile() noexcept = default;
  SourceFile(const SourceFile &) noexcept = default;
  auto operator=(const SourceFile &) noexcept -> SourceFile & = default;
  SourceFile(SourceFile &&) noexcept = default;
  auto operator=(SourceFile &&) noexcept -> SourceFile & = default;

  SourceFile(std::string fileName, std::string source, uint32_t id) noexcept
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
  std::unordered_set<std::string> pragmaOnceProtected_;
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
  int counter_ = 0;
  Arena pool_;

  Private() {
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

  void error(const Token &token, std::string message) const {
    diagnosticsClient_->report(token, Severity::Error, std::move(message));
  }

  void warning(const Token &token, std::string message) const {
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

  auto bol(const TokList *ts) const -> bool { return ts && ts->head->bol; }

  auto match(const TokList *&ts, TokenKind k) const -> bool {
    if (ts && ts->head->is(k)) {
      ts = ts->tail;
      return true;
    }
    return false;
  }

  auto matchId(const TokList *&ts, const std::string_view &s) const -> bool {
    if (ts && ts->head->is(TokenKind::T_IDENTIFIER) && ts->head->text == s) {
      ts = ts->tail;
      return true;
    }
    return false;
  }

  void expect(const TokList *&ts, TokenKind k) const {
    if (!match(ts, k)) {
      error(ts->head->token(), fmt::format("expected '{}'", Token::spell(k)));
    }
  }

  auto expectId(const TokList *&ts) const -> std::string_view {
    if (ts && ts->head->is(TokenKind::T_IDENTIFIER)) {
      auto id = ts->head->text;
      ts = ts->tail;
      return id;
    }
    assert(ts);
    error(ts->head->token(), "expected an identifier");
    return {};
  }

  auto makeUnion(const Hideset *hs, const std::string_view &name)
      -> const Hideset * {
    if (!hs) return get(name);
    if (hs->names().contains(name)) return hs;
    auto names = hs->names();
    names.insert(name);
    return get(std::move(names));
  }

  auto makeIntersection(const Hideset *hs, const Hideset *other)
      -> const Hideset * {
    if (!other || !hs) return nullptr;
    if (other == hs) return hs;

    std::set<std::string_view> names;

    std::set_intersection(begin(hs->names()), end(hs->names()),
                          begin(other->names()), end(other->names()),
                          std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  auto get(std::set<std::string_view> names) -> const Hideset * {
    if (names.empty()) return nullptr;
    if (auto it = hidesets.find(names); it != hidesets.end()) return &*it;
    return &*hidesets.emplace(std::move(names)).first;
  }

  auto get(const std::string_view &name) -> const Hideset * {
    if (auto it = hidesets.find(name); it != hidesets.end()) return &*it;
    return &*hidesets.emplace(std::set{name}).first;
  }

  auto isStringLiteral(TokenKind kind) {
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

  auto updateStringLiteralValue(Token &lastToken, const Tok *tk) -> bool {
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

  auto checkHeaderProtection(const TokList *ts) const -> const TokList *;

  auto checkPragmaOnceProtected(const TokList *ts) const -> bool;

  [[nodiscard]] auto resolve(const Include &include, bool next) const
      -> std::optional<fs::path> {
    if (!canResolveFiles_) return std::nullopt;

    struct Resolve {
      const Private *d;
      bool next;

      explicit Resolve(const Private *d, bool next) : d(d), next(next) {}

      auto operator()(std::monostate) const -> std::optional<fs::path> {
        return {};
      }

      auto operator()(const SystemInclude &include) const
          -> std::optional<fs::path> {
        bool hit = false;
        for (auto it = rbegin(d->systemIncludePaths_);
             it != rend(d->systemIncludePaths_); ++it) {
          const auto p = fs::path(*it);
          auto path = p / include.fileName;
          if (d->fileExists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }
        return {};
      }

      auto operator()(const QuoteInclude &include) const
          -> std::optional<fs::path> {
        bool hit = false;

        if (d->fileExists(d->currentPath_ / include.fileName)) {
          if (!next) return d->currentPath_ / include.fileName;
          hit = true;
        }

        for (auto it = rbegin(d->quoteIncludePaths_);
             it != rend(d->quoteIncludePaths_); ++it) {
          auto p = fs::path(*it);
          auto path = p / include.fileName;
          if (d->fileExists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }

        for (auto it = rbegin(d->systemIncludePaths_);
             it != rend(d->systemIncludePaths_); ++it) {
          auto p = fs::path(*it);
          auto path = p / include.fileName;
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

  void defineMacro(const TokList *ts);

  auto tokenize(const std::string_view &source, int sourceFile, bool bol)
      -> const TokList *;

  auto skipLine(const TokList *ts) -> const TokList *;

  auto expand(const TokList *ts, bool evaluateDirectives) -> const TokList *;

  void expand(const TokList *ts, bool evaluateDirectives, TokList **&out);

  void expand(const TokList *ts, bool evaluateDirectives,
              const std::function<void(const Tok *)> &emitToken);

  auto expandOne(const TokList *ts,
                 const std::function<void(const Tok *)> &emitToken)
      -> const TokList *;

  auto substitude(const TokList *ts, const Macro *macro,
                  const std::vector<const TokList *> &actuals,
                  const Hideset *hideset, const TokList *os) -> const TokList *;

  auto merge(const Tok *left, const Tok *right) -> const Tok *;

  auto stringize(const TokList *ts) -> const Tok *;

  auto instantiate(const TokList *ts, const Hideset *hideset)
      -> const TokList *;

  auto lookupMacro(const Tok *tk, const Macro *&macro) const -> bool;

  auto lookupMacroArgument(const Macro *macro,
                           const std::vector<const TokList *> &actuals,
                           const Tok *tk, const TokList *&actual) const -> bool;

  auto copyLine(const TokList *ts) -> const TokList *;

  auto constantExpression(const TokList *ts) -> long;
  auto conditionalExpression(const TokList *&ts) -> long;
  auto binaryExpression(const TokList *&ts) -> long;
  auto binaryExpressionHelper(const TokList *&ts, long lhs, int minPrec)
      -> long;
  auto unaryExpression(const TokList *&ts) -> long;
  auto primaryExpression(const TokList *&ts) -> long;

  auto readArguments(const TokList *ts, const Macro *macro)
      -> std::tuple<std::vector<const TokList *>, const TokList *,
                    const Hideset *>;

  auto string(std::string s) -> std::string_view;

  void print(const TokList *ts, std::ostream &out) const;

  void printLine(const TokList *ts, std::ostream &out, bool nl = true) const;
};

static auto clone(Arena *pool, const TokList *ts) -> const TokList * {
  if (!ts) return nullptr;
  return new (pool) TokList(ts->head, clone(pool, ts->tail));
}

static auto depth(const TokList *ts) -> int {
  if (!ts) return 0;
  return depth(ts->tail) + 1;
}

auto Preprocessor::Private::tokenize(const std::string_view &source,
                                     int sourceFile, bool bol)
    -> const TokList * {
  cxx::Lexer lex(source);
  lex.setKeepComments(true);
  lex.setPreprocessing(true);
  const TokList *ts = nullptr;
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
    auto tk = Tok::FromCurrentToken(&pool_, lex, sourceFile);
    if (!lex.tokenIsClean()) tk->text = string(std::move(lex.text()));
    *it = new (&pool_) TokList(tk);
    it = const_cast<const TokList **>(&(*it)->tail);
  } while (lex.tokenKind() != cxx::TokenKind::T_EOF_SYMBOL);
  return ts;
}

auto Preprocessor::Private::expand(const TokList *ts, bool evaluateDirectives)
    -> const TokList * {
  TokList *tokens = nullptr;
  auto out = &tokens;
  expand(ts, evaluateDirectives, out);
  return tokens;
}

void Preprocessor::Private::expand(const TokList *ts, bool evaluateDirectives,
                                   TokList **&out) {
  expand(ts, evaluateDirectives, [&](auto tok) {
    *out = new (&pool_) TokList(tok);
    out = const_cast<TokList **>(&(*out)->tail);
  });
}

void Preprocessor::Private::expand(
    const TokList *ts, bool evaluateDirectives,
    const std::function<void(const Tok *)> &emitToken) {
  while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL)) {
    const auto tk = ts->head;
    const auto start = ts;

    const auto [skipping, evaluating] = state();

    if (evaluateDirectives && tk->bol && match(ts, TokenKind::T_HASH)) {
      auto directive = ts;

#if 0
      fmt::print("*** ({}) ", currentPath_);
      printLine(directive, std::cerr);
      fmt::print(std::cerr, "\n");
#endif

      if (!skipping && matchId(ts, "define")) {
        defineMacro(copyLine(ts));
      } else if (!skipping && matchId(ts, "undef")) {
        auto line = copyLine(ts);
        auto name = expectId(line);
        if (!name.empty()) {
          // warning(ts->head->token(), "undef '{}'", name);
          auto it = macros_.find(name);
          if (it != macros_.end()) macros_.erase(it);
        }
      } else if (!skipping &&
                 (matchId(ts, "include") || matchId(ts, "include_next"))) {
        if (ts->head->is(TokenKind::T_IDENTIFIER)) {
          ts = expand(copyLine(ts), /*directives=*/false);
        }

        auto loc = ts;
        if (loc->head->is(TokenKind::T_EOF_SYMBOL)) loc = directive;

        const bool next = directive->head->text == "include_next";
        std::optional<fs::path> path;
        std::string file;
        if (ts->head->is(TokenKind::T_STRING_LITERAL)) {
          file = ts->head->text.substr(1, ts->head->text.length() - 2);
          path = resolve(QuoteInclude(file), next);
        } else if (match(ts, TokenKind::T_LESS)) {
          while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL) &&
                 !ts->head->bol) {
            if (match(ts, TokenKind::T_GREATER)) break;
            file += ts->head->text;
            ts = ts->tail;
          }
          path = resolve(SystemInclude(file), next);
        }

        if (!path) {
          error(loc->head->token(), fmt::format("file '{}' not found", file));
          ts = skipLine(directive);
          continue;
        }

        const auto fn = path->string();

        if (pragmaOnceProtected_.find(fn) != pragmaOnceProtected_.end()) {
          ts = skipLine(directive);
          continue;
        }

        auto it = ifndefProtectedFiles_.find(fn);

        if (it != ifndefProtectedFiles_.end()) {
          if (macros_.find(it->second) != macros_.end()) {
            ts = skipLine(directive);
            continue;
          }
        }

        std::string currentFileName = path->string();

        auto dirpath = *path;
        dirpath.remove_filename();

        std::swap(currentPath_, dirpath);
        std::swap(currentFileName_, currentFileName);

        SourceFile *sourceFile = nullptr;

        auto fileName = path->string();

        for (const auto &source : sourceFiles_) {
          if (source->fileName == fileName) {
            sourceFile = source.get();
            break;
          }
        }

        if (!sourceFile) {
          const int sourceFileId = static_cast<int>(sourceFiles_.size() + 1);

          sourceFile = sourceFiles_
                           .emplace_back(std::make_unique<SourceFile>(
                               path->string(), readFile(*path), sourceFileId))
                           .get();

          sourceFile->tokens =
              tokenize(sourceFile->source, sourceFile->id, true);

          if (checkPragmaOnceProtected(sourceFile->tokens)) {
            pragmaOnceProtected_.insert(fn);
          }
        }

        const auto prot = checkHeaderProtection(sourceFile->tokens);

        if (prot) ifndefProtectedFiles_.emplace(fn, prot->head->text);

        expand(sourceFile->tokens, /*directives=*/true, emitToken);

        if (prot && macros_.find(prot->head->text) == macros_.end()) {
          auto it = ifndefProtectedFiles_.find(std::string(prot->head->text));
          if (it != ifndefProtectedFiles_.end()) {
            ifndefProtectedFiles_.erase(it);
          }
        }

        std::swap(currentPath_, dirpath);
        std::swap(currentFileName_, currentFileName);
        ts = skipLine(directive);
      } else if (matchId(ts, "ifdef")) {
        const Macro *macro = nullptr;
        const auto value = lookupMacro(ts->head, macro);
        if (value) {
          pushState(std::tuple(skipping, false));
        } else {
          pushState(std::tuple(true, !skipping));
        }
      } else if (matchId(ts, "ifndef")) {
        const Macro *macro = nullptr;
        const auto value = !lookupMacro(ts->head, macro);
        if (value) {
          pushState(std::tuple(skipping, false));
        } else {
          pushState(std::tuple(true, !skipping));
        }
      } else if (matchId(ts, "if")) {
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
      } else if (matchId(ts, "elif")) {
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
      } else if (matchId(ts, "elifdef")) {
        if (!evaluating) {
          setState(std::tuple(true, false));
        } else {
          const Macro *macro = nullptr;
          const auto value = lookupMacro(ts->head, macro);
          if (value) {
            setState(std::tuple(!evaluating, false));
          } else {
            setState(std::tuple(true, evaluating));
          }
        }
      } else if (matchId(ts, "elifndef")) {
        if (!evaluating) {
          setState(std::tuple(true, false));
        } else {
          const Macro *macro = nullptr;
          const auto value = lookupMacro(ts->head, macro);
          if (!value) {
            setState(std::tuple(!evaluating, false));
          } else {
            setState(std::tuple(true, evaluating));
          }
        }
      } else if (matchId(ts, "else")) {
        setState(std::tuple(!evaluating, false));
      } else if (matchId(ts, "endif")) {
        popState();
        if (evaluating_.empty()) {
          error(directive->head->token(), "unexpected '#endif'");
        }
      } else if (matchId(ts, "line")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        warning(directive->head->token(), "skipped #line directive");
      } else if (matchId(ts, "pragma")) {
        // ###
#if 0
        std::ostringstream out;
        printLine(start, out);
        fmt::print(std::cerr, "** todo pragma: ");
        printLine(ts, std::cerr);
        fmt::print(std::cerr, "\n");
        // cxx_runtime_error(out.str());
#endif
      } else if (!skipping && matchId(ts, "error")) {
        std::ostringstream out;
        printLine(start, out, /*nl=*/false);
        error(directive->head->token(), fmt::format("{}", out.str()));
      } else if (!skipping && matchId(ts, "warning")) {
        std::ostringstream out;
        printLine(start, out, /*nl=*/false);
        warning(directive->head->token(), fmt::format("{}", out.str()));
      }
      ts = skipLine(ts);
    } else if (evaluateDirectives && skipping) {
      ts = skipLine(ts->tail);
    } else if (!evaluateDirectives && matchId(ts, "defined")) {
      const Macro *macro = nullptr;
      if (match(ts, TokenKind::T_LPAREN)) {
        lookupMacro(ts->head, macro);
        ts = ts->tail;
        expect(ts, TokenKind::T_RPAREN);
      } else {
        lookupMacro(ts->head, macro);
        ts = ts->tail;
      }
      auto t =
          Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, macro ? "1" : "0");
      emitToken(t);
    } else if (!evaluateDirectives && matchId(ts, "__has_include")) {
      std::string fn;
      expect(ts, TokenKind::T_LPAREN);
      auto literal = ts;
      Include include;
      if (match(ts, TokenKind::T_STRING_LITERAL)) {
        fn = literal->head->text.substr(1, literal->head->text.length() - 2);
        include = QuoteInclude(fn);
      } else {
        expect(ts, TokenKind::T_LESS);
        for (; ts && !ts->head->is(TokenKind::T_GREATER); ts = ts->tail) {
          fn += ts->head->text;
        }
        expect(ts, TokenKind::T_GREATER);
        include = SystemInclude(fn);
      }
      expect(ts, TokenKind::T_RPAREN);
      const auto value = resolve(include, /*next*/ false);
      auto t =
          Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
      emitToken(t);
    } else if (!evaluateDirectives && matchId(ts, "__has_include_next")) {
      std::string fn;
      expect(ts, TokenKind::T_LPAREN);
      auto literal = ts;
      Include include;
      if (match(ts, TokenKind::T_STRING_LITERAL)) {
        fn = literal->head->text.substr(1, literal->head->text.length() - 2);
        include = QuoteInclude(fn);
      } else {
        expect(ts, TokenKind::T_LESS);
        for (; ts && !ts->head->is(TokenKind::T_GREATER); ts = ts->tail) {
          fn += ts->head->text;
        }
        expect(ts, TokenKind::T_GREATER);
        include = SystemInclude(fn);
      }
      expect(ts, TokenKind::T_RPAREN);
      const auto value = resolve(include, /*next*/ true);
      auto t =
          Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, value ? "1" : "0");
      emitToken(t);
    } else if (!evaluateDirectives && matchId(ts, "__has_extension")) {
      expect(ts, TokenKind::T_LPAREN);
      const auto id = expectId(ts);
      expect(ts, TokenKind::T_RPAREN);
      bool enabled = true;
      if (id == "blocks") {
        enabled = false;
      } else if (id.starts_with("obj_")) {
        enabled = false;
      }
      auto t =
          Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, enabled ? "1" : "0");
      emitToken(t);
    } else if (!evaluateDirectives && matchId(ts, "__has_feature")) {
      expect(ts, TokenKind::T_LPAREN);
      auto id = expectId(ts);
      (void)id;
      expect(ts, TokenKind::T_RPAREN);
      auto t = Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, "1");
      emitToken(t);
    } else {
      ts = expandOne(ts, emitToken);
    }
  }
}

auto Preprocessor::Private::expandOne(
    const TokList *ts, const std::function<void(const Tok *)> &emitToken)
    -> const TokList * {
  if (ts->head->is(TokenKind::T_EOF_SYMBOL)) return ts;

  const Macro *macro = nullptr;

  if (ts->head->text == "__FILE__") {
    auto tk = Tok::Gen(&pool_, TokenKind::T_STRING_LITERAL,
                       string(fmt::format("\"{}\"", currentFileName_)));
    tk->bol = ts->head->bol;
    tk->space = ts->head->space;
    emitToken(tk);
    return ts->tail;
  }
  if (ts->head->text == "__LINE__") {
    unsigned line = 0;
    preprocessor_->getTokenStartPosition(ts->head->token(), &line, nullptr,
                                         nullptr);
    auto tk = Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL,
                       string(std::to_string(line)));
    tk->bol = ts->head->bol;
    tk->space = ts->head->space;
    tk->sourceFile = ts->head->sourceFile;
    emitToken(tk);
    return ts->tail;
  }
  if (ts->head->text == "__DATE__") {
    auto tk = Tok::Gen(&pool_, TokenKind::T_STRING_LITERAL, date_);
    tk->bol = ts->head->bol;
    tk->space = ts->head->space;
    tk->sourceFile = ts->head->sourceFile;
    emitToken(tk);
    return ts->tail;
  }
  if (ts->head->text == "__TIME__") {
    auto tk = Tok::Gen(&pool_, TokenKind::T_STRING_LITERAL, time_);
    tk->bol = ts->head->bol;
    tk->space = ts->head->space;
    tk->sourceFile = ts->head->sourceFile;
    emitToken(tk);
    return ts->tail;
  }
  if (ts->head->text == "__COUNTER__") {
    auto tk = Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL,
                       string(std::to_string(counter_++)));
    tk->bol = ts->head->bol;
    tk->space = ts->head->space;
    tk->sourceFile = ts->head->sourceFile;
    emitToken(tk);
    return ts->tail;
  }
  if (lookupMacro(ts->head, macro)) {
    const auto tk = ts->head;

    if (macro->objLike) {
      const auto hideset = makeUnion(tk->hideset, tk->text);
      auto expanded = substitude(macro->body, macro, {}, hideset, nullptr);
      if (expanded) {
        const_cast<Tok *>(expanded->head)->space = tk->space;
        const_cast<Tok *>(expanded->head)->bol = tk->bol;
      }
      if (!expanded) return ts->tail;
      auto it = expanded;
      while (it->tail) it = it->tail;
      const_cast<TokList *>(it)->tail = ts->tail;
      return expanded;
    }
    if (ts->tail && ts->tail->head->is(TokenKind::T_LPAREN)) {
      auto [args, p, hideset] = readArguments(ts, macro);
      auto hs = makeUnion(makeIntersection(tk->hideset, hideset), tk->text);
      auto expanded = substitude(macro->body, macro, args, hs, nullptr);
      if (expanded) {
        const_cast<Tok *>(expanded->head)->space = tk->space;
        const_cast<Tok *>(expanded->head)->bol = tk->bol;
      }
      if (!expanded) return p;
      auto it = expanded;
      while (it->tail) it = it->tail;
      const_cast<TokList *>(it)->tail = p;
      return expanded;
    }
  }

  emitToken(ts->head);
  return ts->tail;
}

auto Preprocessor::Private::substitude(
    const TokList *ts, const Macro *macro,
    const std::vector<const TokList *> &actuals, const Hideset *hideset,
    const TokList *os) -> const TokList * {
  auto **ip = const_cast<TokList **>(&os);

  auto appendTokens = [&](const TokList *rs) {
    if (!*ip) {
      *ip = const_cast<TokList *>(rs);
    } else {
      (*ip)->tail = const_cast<TokList *>(rs);
    }
    while (*ip && (*ip)->tail) ip = const_cast<TokList **>(&(*ip)->tail);
  };

  auto appendToken = [&](const Tok *tk) {
    appendTokens(new (&pool_) TokList(tk));
  };

  while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL)) {
    auto tk = ts->head;
    const TokList *actual = nullptr;

    if (ts->tail && tk->is(TokenKind::T_HASH) &&
        lookupMacroArgument(macro, actuals, ts->tail->head, actual)) {
      appendToken(stringize(actual));
      ts = ts->tail->tail;
    } else if (ts->tail && tk->is(TokenKind::T_HASH_HASH) &&
               lookupMacroArgument(macro, actuals, ts->tail->head, actual)) {
      if (actual) {
        (*ip)->head = merge((*ip)->head, actual->head);
        appendTokens(clone(&pool_, actual->tail));
      }
      ts = ts->tail->tail;
    } else if (ts->tail && tk->is(TokenKind::T_HASH_HASH)) {
      (*ip)->head = merge((*ip)->head, ts->tail->head);
      ts = ts->tail->tail;
    } else if (ts->tail && lookupMacroArgument(macro, actuals, tk, actual) &&
               ts->tail->head->is(TokenKind::T_HASH_HASH)) {
      appendTokens(clone(&pool_, actual));
      ts = ts->tail;
    } else if (lookupMacroArgument(macro, actuals, tk, actual)) {
      appendTokens(expand(actual, /*directives=*/false));
      ts = ts->tail;
    } else {
      appendToken(tk);
      ts = ts->tail;
    }
  }

  return instantiate(os, hideset);
}

auto Preprocessor::Private::checkPragmaOnceProtected(const TokList *ts) const
    -> bool {
  if (!ts) return false;
  if (!match(ts, TokenKind::T_HASH)) return false;
  if (bol(ts) || !matchId(ts, "pragma")) return false;
  if (bol(ts) || !matchId(ts, "once")) return false;
  return true;
}

auto Preprocessor::Private::checkHeaderProtection(const TokList *ts) const
    -> const TokList * {
  if (!ts) return nullptr;
  if (!match(ts, TokenKind::T_HASH)) return nullptr;
  if (bol(ts) || !matchId(ts, "ifndef")) return nullptr;
  const TokList *prot = ts;
  if (bol(ts) || !match(ts, TokenKind::T_IDENTIFIER)) return nullptr;
  if (!bol(ts) || !match(ts, TokenKind::T_HASH)) return nullptr;
  if (bol(ts) || !matchId(ts, "define")) return nullptr;
  if (bol(ts) || !matchId(ts, prot->head->text)) return nullptr;
  return prot;
}

auto Preprocessor::Private::copyLine(const TokList *ts) -> const TokList * {
  assert(ts);
  TokList *line = nullptr;
  auto it = &line;
  auto lastTok = ts->head;
  for (; ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL) && !ts->head->bol;
       ts = ts->tail) {
    *it = new (&pool_) TokList(ts->head);
    lastTok = ts->head;
    it = const_cast<TokList **>(&(*it)->tail);
  }
  auto eol = Tok::Gen(&pool_, TokenKind::T_EOF_SYMBOL, std::string_view());
  eol->sourceFile = lastTok->sourceFile;
  eol->offset = lastTok->offset + lastTok->length;
  *it = new (&pool_) TokList(eol);
  return line;
}

auto Preprocessor::Private::constantExpression(const TokList *ts) -> long {
  auto line = copyLine(ts);
#if 0
  fmt::print("\n**evaluating: ");
  print(line, std::cout);
  fmt::print("\n");
  fmt::print("\n**expanded to: ");
#endif
  auto e = expand(line, /*directives=*/false);
#if 0
  print(e, std::cout);
  fmt::print("\n");
#endif
  return conditionalExpression(e);
}

auto Preprocessor::Private::conditionalExpression(const TokList *&ts) -> long {
  if (!ts) return 0;
  const auto value = binaryExpression(ts);
  if (!match(ts, TokenKind::T_QUESTION)) return value;
  const auto iftrue = conditionalExpression(ts);
  expect(ts, TokenKind::T_COLON);
  const auto iffalse = conditionalExpression(ts);
  return value ? iftrue : iffalse;
}

static auto prec(const TokList *ts) -> int {
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

  switch (ts->head->kind) {
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

auto Preprocessor::Private::binaryExpression(const TokList *&ts) -> long {
  auto e = unaryExpression(ts);
  return binaryExpressionHelper(ts, e, 0);
}

auto Preprocessor::Private::binaryExpressionHelper(const TokList *&ts, long lhs,
                                                   int minPrec) -> long {
  while (prec(ts) >= minPrec) {
    const auto p = prec(ts);
    const auto op = ts->head->kind;
    ts = ts->tail;
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
            fmt::format("invalid operator '{}'", Token::spell(op)));
    }  // switch
  }
  return lhs;
}

auto Preprocessor::Private::unaryExpression(const TokList *&ts) -> long {
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

auto Preprocessor::Private::primaryExpression(const TokList *&ts) -> long {
  const auto tk = ts->head;
  if (match(ts, TokenKind::T_INTEGER_LITERAL)) {
    return IntegerLiteral::interpretText(tk->text.data());
  }
  if (match(ts, TokenKind::T_LPAREN)) {
    auto result = conditionalExpression(ts);
    expect(ts, TokenKind::T_RPAREN);
    return result;
  }
  ts = ts->tail;
  return 0;
}

auto Preprocessor::Private::lookupMacroArgument(
    const Macro *macro, const std::vector<const TokList *> &actuals,
    const Tok *tk, const TokList *&actual) const -> bool {
  if (!tk) return false;
  if (macro->variadic && tk->text == "__VA_ARGS__") {
    actual = !actuals.empty() ? actuals.back() : nullptr;
    return true;
  }

  for (std::size_t i = 0; i < macro->formals.size(); ++i) {
    if (macro->formals[i] == tk->text) {
      actual = i < actuals.size() ? actuals[i] : nullptr;
      return true;
    }
  }
  return false;
}

auto Preprocessor::Private::instantiate(const TokList *ts,
                                        const Hideset *hideset)
    -> const TokList * {
  for (auto ip = ts; ip; ip = ip->tail) {
    if (ip->head->hideset != hideset) {
      const_cast<TokList *>(ip)->head =
          Tok::WithHideset(&pool_, ip->head, hideset);
    }
  }
  return ts;
}

auto Preprocessor::Private::readArguments(const TokList *ts, const Macro *macro)
    -> std::tuple<std::vector<const TokList *>, const TokList *,
                  const Hideset *> {
  auto it = ts->tail->tail;
  int depth = 1;
  int argc = 0;
  std::vector<const TokList *> args;

  const auto formalCount = macro->formals.size();

  const Tok *rp = nullptr;
  if (it->head->isNot(TokenKind::T_RPAREN)) {
    TokList *arg = nullptr;
    auto argIt = &arg;
    while (it && it->head->isNot(TokenKind::T_EOF_SYMBOL)) {
      auto tk = it->head;
      it = it->tail;
      if (depth == 1 && tk->is(TokenKind::T_COMMA) &&
          args.size() < formalCount) {
        args.push_back(arg);
        arg = nullptr;
        argIt = &arg;
        ++argc;
        continue;
      }
      if (tk->is(TokenKind::T_LPAREN)) {
        ++depth;
      } else if (tk->is(TokenKind::T_RPAREN) && !--depth) {
        rp = tk;
        break;
      }

      *argIt = new (&pool_) TokList(tk);
      argIt = const_cast<TokList **>(&(*argIt)->tail);
    }

    args.push_back(arg);
  } else {
    rp = it->head;
    it = it->tail;
  }

  assert(rp);

  return std::tuple(std::move(args), it, rp->hideset);
}

auto Preprocessor::Private::stringize(const TokList *ts) -> const Tok * {
  std::string s;

  const auto start = ts;

  for (; ts; ts = ts->tail) {
    if (!s.empty() && (ts->head->space || ts->head->bol)) s += ' ';
    s += ts->head->text;
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

  auto tk = Tok::Gen(&pool_, TokenKind::T_STRING_LITERAL, string(o));
  if (start) {
    tk->sourceFile = start->head->sourceFile;
    tk->offset = start->head->offset;
  }

  return tk;
}

auto Preprocessor::Private::string(std::string s) -> std::string_view {
  return std::string_view(scratchBuffer_.emplace_front(std::move(s)));
}

void Preprocessor::Private::defineMacro(const TokList *ts) {
#if 0
  fmt::print("*** defining macro: ");
  printLine(ts, std::cout);
  fmt::print("\n");
#endif

  auto name = ts->head->text;

  Macro m;

  if (ts->tail && !ts->tail->head->space &&
      ts->tail->head->is(TokenKind::T_LPAREN)) {
    ts = ts->tail->tail;  // skip macro name and '('

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

    m.objLike = false;
    m.body = ts;
    m.formals = std::move(formals);
    m.variadic = variadic;
  } else {
    m.objLike = true;
    m.body = ts->tail;
  }

  if (auto it = macros_.find(name); it != macros_.end()) {
    if (it->second != m) {
      warning(ts->head->token(), fmt::format("'{}' macro redefined", name));
    }

    macros_.erase(it);
  }

  macros_.emplace(name, std::move(m));
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
  auto tok = Tok::Gen(&pool_, lex.tokenKind(), lex.tokenText(), hideset);
  tok->sourceFile = left->sourceFile;
  tok->offset = left->offset;
  return tok;
}

auto Preprocessor::Private::skipLine(const TokList *ts) -> const TokList * {
  while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL) && !ts->head->bol) {
    ts = ts->tail;
  }
  return ts;
}

auto Preprocessor::Private::lookupMacro(const Tok *tk,
                                        const Macro *&macro) const -> bool {
  if (!tk || tk->isNot(TokenKind::T_IDENTIFIER)) return false;
  auto it = macros_.find(tk->text);
  if (it != macros_.end()) {
    const auto disabled = tk->hideset && tk->hideset->contains(tk->text);
    if (!disabled) {
      macro = &it->second;
      return true;
    }
  }
  return false;
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

void Preprocessor::Private::print(const TokList *ts, std::ostream &out) const {
  bool first = true;
  for (const Tok *prevTk = nullptr; ts; ts = ts->tail) {
    auto tk = ts->head;
    if (tk->text.empty()) continue;
    if (tk->bol) {
      fmt::print(out, "\n");
    } else if (!first && needSpace(prevTk, tk)) {
      fmt::print(out, " ");
    }
    fmt::print(out, "{}", tk->text);
    prevTk = tk;
    first = false;
  }
}

void Preprocessor::Private::printLine(const TokList *ts, std::ostream &out,
                                      bool nl) const {
  bool first = true;
  for (const Tok *prevTk = nullptr; ts; ts = ts->tail) {
    auto tk = ts->head;
    if (tk->text.empty()) continue;
    if (!first && needSpace(prevTk, tk)) fmt::print(out, " ");
    fmt::print(out, "{}", tk->text);
    prevTk = tk;
    first = false;
    if (ts->tail && ts->tail->head->bol) break;
  }
  if (nl) fmt::print(out, "\n");
}

Preprocessor::Preprocessor(Control *control,
                           DiagnosticsClient *diagnosticsClient)
    : d(std::make_unique<Private>()) {
  d->preprocessor_ = this;
  d->control_ = control;
  d->diagnosticsClient_ = diagnosticsClient;
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

void Preprocessor::setFileExistsFunction(
    std::function<bool(std::string)> fileExists) {
  d->fileExists_ = std::move(fileExists);
}

void Preprocessor::setReadFileFunction(
    std::function<std::string(std::string)> readFile) {
  d->readFile_ = std::move(readFile);
}

void Preprocessor::squeeze() { d->pool_.reset(); }

void Preprocessor::operator()(std::string source, std::string fileName,
                              std::ostream &out) {
  preprocess(std::move(source), std::move(fileName), out);
}

void Preprocessor::preprocess(std::string source, std::string fileName,
                              std::ostream &out) {
  const int sourceFileId = static_cast<int>(d->sourceFiles_.size() + 1);
  auto &sourceFile = *d->sourceFiles_.emplace_back(std::make_unique<SourceFile>(
      std::move(fileName), std::move(source), sourceFileId));

  std::string currentFileName = sourceFile.fileName;

  fs::path path(sourceFile.fileName);
  path.remove_filename();

  std::swap(d->currentPath_, path);
  std::swap(d->currentFileName_, currentFileName);

  const auto ts = d->tokenize(sourceFile.source, sourceFileId, true);

  const auto os = d->expand(ts, /*directives*/ true);

  std::swap(d->currentFileName_, currentFileName);
  std::swap(d->currentPath_, path);

  uint32_t outFile = 0;
  uint32_t outLine = -1;

  const Tok *prevTk = nullptr;

  for (auto it = os; it; it = it->tail) {
    auto tk = it->head;
    auto file =
        tk->sourceFile > 0 ? &*d->sourceFiles_[tk->sourceFile - 1] : nullptr;
    if ((tk->bol || it == os) && file) {
      std::string_view fileName;
      uint32_t line = 0;
      file->getTokenStartPosition(tk->offset, &line, nullptr, &fileName);
      if (outFile == tk->sourceFile && line == outLine) {
        ++outLine;
        fmt::print(out, "\n");
      } else {
        if (it != os) fmt::print(out, "\n");
        fmt::print(out, "# {} \"{}\"\n", line, fileName);
        outLine = line + 1;
        outFile = tk->sourceFile;
      }
    } else if (needSpace(prevTk, tk) || tk->space || !tk->sourceFile) {
      fmt::print(out, " ");
    }
    fmt::print(out, "{}", tk->text);
    prevTk = tk;
  }

  fmt::print(out, "\n");
}

void Preprocessor::preprocess(std::string source, std::string fileName,
                              std::vector<Token> &tokens) {
  const int sourceFileId = static_cast<int>(d->sourceFiles_.size() + 1);
  auto &sourceFile = *d->sourceFiles_.emplace_back(std::make_unique<SourceFile>(
      std::move(fileName), std::move(source), sourceFileId));

  std::string currentFileName = sourceFile.fileName;

  fs::path path(sourceFile.fileName);
  path.remove_filename();

  std::swap(d->currentPath_, path);
  std::swap(d->currentFileName_, currentFileName);

  const auto ts = d->tokenize(sourceFile.source, sourceFileId, true);

  sourceFile.tokens = ts;

  tokens.emplace_back(TokenKind::T_ERROR);

  d->expand(ts, /*directives*/ true, [&](const Tok *tk) {
    auto kind = tk->kind;
    const auto fileId = tk->sourceFile;
    TokenValue value{};

    switch (tk->kind) {
      case TokenKind::T_IDENTIFIER: {
        kind = Lexer::classifyKeyword(tk->text);
        if (kind == TokenKind::T_IDENTIFIER) {
          value.idValue = d->control_->getIdentifier(tk->text);
        }
        break;
      }

      case TokenKind::T_CHARACTER_LITERAL:
        value.literalValue = d->control_->charLiteral(tk->text);
        break;

      case TokenKind::T_WIDE_STRING_LITERAL:
        if (d->updateStringLiteralValue(tokens.back(), tk)) {
          return;
        }
        value.literalValue = d->control_->wideStringLiteral(tk->text);
        break;

      case TokenKind::T_UTF8_STRING_LITERAL:
        if (d->updateStringLiteralValue(tokens.back(), tk)) {
          return;
        }
        value.literalValue = d->control_->utf8StringLiteral(tk->text);
        break;

      case TokenKind::T_UTF16_STRING_LITERAL:
        if (d->updateStringLiteralValue(tokens.back(), tk)) {
          return;
        }
        value.literalValue = d->control_->utf16StringLiteral(tk->text);
        break;

      case TokenKind::T_UTF32_STRING_LITERAL:
        if (d->updateStringLiteralValue(tokens.back(), tk)) {
          return;
        }
        value.literalValue = d->control_->utf32StringLiteral(tk->text);
        break;

      case TokenKind::T_STRING_LITERAL:
        if (d->updateStringLiteralValue(tokens.back(), tk)) {
          return;
        }
        value.literalValue = d->control_->stringLiteral(tk->text);
        break;

      case TokenKind::T_USER_DEFINED_STRING_LITERAL:
        value.literalValue = d->control_->stringLiteral(tk->text);
        break;

      case TokenKind::T_INTEGER_LITERAL:
        value.literalValue = d->control_->integerLiteral(tk->text);
        break;

      case TokenKind::T_FLOATING_POINT_LITERAL:
        value.literalValue = d->control_->floatLiteral(tk->text);
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
  });

  tokens.emplace_back(TokenKind::T_EOF_SYMBOL,
                      static_cast<uint32_t>(sourceFile.source.size()));

  tokens.back().setFileId(sourceFileId);

  std::swap(d->currentPath_, path);
  std::swap(d->currentFileName_, currentFileName);
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
  for (const auto &[name, macro] : d->macros_) {
    fmt::print(out, "#define {}", name);
    if (!macro.objLike) {
      fmt::print(out, "(");
      for (std::size_t i = 0; i < macro.formals.size(); ++i) {
        if (i > 0) fmt::print(",");
        fmt::print(out, "{}", macro.formals[i]);
      }
      fmt::print(out, ")");
    }
    if (macro.body) fmt::print(out, " ");
    d->print(macro.body, out);
    fmt::print(out, "\n");
  }
}

void Preprocessor::getTokenStartPosition(const Token &token, unsigned *line,
                                         unsigned *column,
                                         std::string_view *fileName) const {
  if (token.fileId() == 0) {
    if (line) *line = 0;
    if (column) *column = 0;
    if (fileName) *fileName = std::string_view();
    return;
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];
  sourceFile.getTokenStartPosition(token.offset(), line, column, fileName);
}

void Preprocessor::getTokenEndPosition(const Token &token, unsigned *line,
                                       unsigned *column,
                                       std::string_view *fileName) const {
  if (token.fileId() == 0) {
    if (line) *line = 0;
    if (column) *column = 0;
    if (fileName) *fileName = std::string_view();
    return;
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];
  sourceFile.getTokenStartPosition(token.offset() + token.length(), line,
                                   column, fileName);
}

auto Preprocessor::getTextLine(const Token &token) const -> std::string_view {
  if (token.fileId() == 0) return {};
  const SourceFile *file = d->sourceFiles_[token.fileId() - 1].get();
  unsigned line = 0;
  getTokenStartPosition(token, &line, nullptr, nullptr);
  std::string_view source = file->source;
  const auto &lines = file->lines;
  const auto start = lines.at(line - 1);
  const auto end = line < lines.size() ? lines.at(line) : source.length();
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
