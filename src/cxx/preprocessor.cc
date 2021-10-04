// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

// utf8
#include <utf8.h>

// stl
#include <cassert>
#include <filesystem>
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

namespace fs = std::filesystem;

namespace {

class Hideset {
 public:
  Hideset(const Hideset &other) = default;
  Hideset &operator=(const Hideset &other) = default;

  Hideset(Hideset &&other) = default;
  Hideset &operator=(Hideset &&other) = default;

  Hideset() = default;

  explicit Hideset(std::set<std::string_view> names)
      : names_(std::move(names)) {}

  bool contains(const std::string_view &name) const {
    return names_.contains(name);
  }

  const std::set<std::string_view> &names() const { return names_; };

  bool operator==(const Hideset &other) const { return names_ == other.names_; }

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
struct std::hash<Hideset> {
  template <typename T>
  void hash_combine(std::size_t &seed, const T &val) const {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  std::size_t operator()(const Hideset &hideset) const {
    std::size_t seed = 0;
    for (const auto &name : hideset.names()) hash_combine(seed, name);
    return seed;
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
  Tok &operator=(const Tok &other) = default;

  Tok(Tok &&other) = default;
  Tok &operator=(Tok &&other) = default;

  bool is(TokenKind k) const { return kind == k; }

  bool isNot(TokenKind k) const { return kind != k; }

  static Tok *WithHideset(Arena *pool, const Tok *tok, const Hideset *hideset) {
    return new (pool) Tok(tok, hideset);
  }

  static Tok *FromCurrentToken(Arena *pool, const Lexer &lex, int sourceFile) {
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

  static Tok *Gen(Arena *pool, TokenKind kind, const std::string_view &text,
                  const Hideset *hideset = nullptr) {
    auto tk = new (pool) Tok();
    tk->kind = kind;
    tk->text = text;
    tk->hideset = hideset;
    tk->generated = true;
    tk->space = true;
    return tk;
  }

  Token token() const {
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
};

struct SourceFile {
  std::string fileName;
  std::string source;
  std::vector<int> lines;
  const TokList *tokens = nullptr;
  int id;

  SourceFile() noexcept = default;
  SourceFile(const SourceFile &) noexcept = default;
  SourceFile &operator=(const SourceFile &) noexcept = default;
  SourceFile(SourceFile &&) noexcept = default;
  SourceFile &operator=(SourceFile &&) noexcept = default;

  SourceFile(std::string fileName, std::string source, uint32_t id) noexcept
      : fileName(std::move(fileName)), source(std::move(source)), id(id) {
    initLineMap();
  }

  void getTokenStartPosition(unsigned offset, unsigned *line, unsigned *column,
                             std::string_view *fileName) const {
    auto it = std::lower_bound(lines.cbegin(), lines.cend(), int(offset));
    if (*it != offset) --it;

    assert(*it <= int(offset));

    if (line) *line = int(std::distance(cbegin(lines), it) + 1);

    if (column) {
      const auto start = cbegin(source) + *it;
      const auto end = cbegin(source) + offset;

      *column = utf8::distance(start, end) + 1;
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

      lines.push_back(int(offset));
    }
  }
};

}  // namespace

struct Preprocessor::Private {
  Control *control_ = nullptr;
  DiagnosticsClient *diagnosticsClient_ = nullptr;
  std::vector<fs::path> systemIncludePaths_;
  std::vector<fs::path> quoteIncludePaths_;
  std::unordered_map<std::string_view, Macro> macros_;
  std::unordered_set<Hideset> hidesets;
  std::forward_list<std::string> scratchBuffer_;
  std::unordered_set<std::string> pragmaOnceProtected_;
  std::unordered_map<std::string, std::string> ifndefProtectedFiles_;
  std::vector<std::unique_ptr<SourceFile>> sourceFiles_;
  fs::path currentPath_;
  std::vector<bool> evaluating_;
  std::vector<bool> skipping_;
  Arena pool_;

  Private() {
    currentPath_ = fs::current_path();

    skipping_.push_back(false);
    evaluating_.push_back(true);
  }

  template <typename... Args>
  void error(const Token &token, const std::string_view &format,
             const Args &...args) const {
    diagnosticsClient_->report(token, Severity::Error, format, args...);
  }

  template <typename... Args>
  void warning(const Token &token, const std::string_view &format,
               const Args &...args) const {
    diagnosticsClient_->report(token, Severity::Warning, format, args...);
  }

  std::tuple<bool, bool> state() const {
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

  bool bol(const TokList *ts) const { return ts && ts->head->bol; }

  bool match(const TokList *&ts, TokenKind k) const {
    if (ts && ts->head->is(k)) {
      ts = ts->tail;
      return true;
    }
    return false;
  }

  bool matchId(const TokList *&ts, const std::string_view &s) const {
    if (ts && ts->head->is(TokenKind::T_IDENTIFIER) && ts->head->text == s) {
      ts = ts->tail;
      return true;
    }
    return false;
  }

  void expect(const TokList *&ts, TokenKind k) const {
    if (!match(ts, k))
      error(ts->head->token(), "expected '{}'", Token::spell(k));
  }

  std::string_view expectId(const TokList *&ts) const {
    if (ts && ts->head->is(TokenKind::T_IDENTIFIER)) {
      auto id = ts->head->text;
      ts = ts->tail;
      return id;
    }
    assert(ts);
    error(ts->head->token(), "expected an identifier");
    return std::string_view();
  }

  const Hideset *makeUnion(const Hideset *hs, const std::string_view &name) {
    if (!hs) return get(std::set<std::string_view>{name});
    auto names = hs->names();
    names.insert(name);
    return get(std::move(names));
  }

  const Hideset *makeUnion(const Hideset *hs, const Hideset *other) {
    if (!other) return hs;
    if (!hs) return other;

    std::set<std::string_view> names;

    std::set_union(begin(hs->names()), end(hs->names()), begin(other->names()),
                   end(other->names()), std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  const Hideset *makeIntersection(const Hideset *hs, const Hideset *other) {
    if (!other) return hs;
    if (!hs) return other;

    std::set<std::string_view> names;

    std::set_intersection(begin(hs->names()), end(hs->names()),
                          begin(other->names()), end(other->names()),
                          std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  const Hideset *get(std::set<std::string_view> name) {
    if (name.empty()) return nullptr;
    return &*hidesets.emplace(name).first;
  }

  std::string readFile(const fs::path &file) const {
    std::ifstream in(file);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
  }

  const TokList *checkHeaderProtection(const TokList *ts) const;

  bool checkPragmaOnceProtected(const TokList *ts) const;

  std::optional<fs::path> resolve(const Include &include, bool next) const {
    struct Resolve {
      const Private *d;
      bool next;

      explicit Resolve(const Private *d, bool next) : d(d), next(next) {}

      std::optional<fs::path> operator()(std::monostate) const { return {}; }

      std::optional<fs::path> operator()(const SystemInclude &include) const {
        bool hit = false;
        for (auto it = rbegin(d->systemIncludePaths_);
             it != rend(d->systemIncludePaths_); ++it) {
          const auto &p = *it;
          auto path = p / include.fileName;
          if (exists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }
        return {};
      }

      std::optional<fs::path> operator()(const QuoteInclude &include) const {
        bool hit = false;

        if (exists(d->currentPath_ / include.fileName)) {
          if (!next) return d->currentPath_ / include.fileName;
          hit = true;
        }

        for (auto it = rbegin(d->quoteIncludePaths_);
             it != rend(d->quoteIncludePaths_); ++it) {
          const auto &p = *it;
          auto path = p / include.fileName;
          if (exists(path)) {
            if (!next || hit) return path;
            hit = true;
          }
        }

        for (auto it = rbegin(d->systemIncludePaths_);
             it != rend(d->systemIncludePaths_); ++it) {
          const auto &p = *it;
          auto path = p / include.fileName;
          if (exists(path)) {
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

  const TokList *tokenize(const std::string_view &source, int sourceFile,
                          bool bol);

  const TokList *skipLine(const TokList *ts);

  const TokList *expand(const TokList *ts, bool evaluateDirectives);

  void expand(const TokList *ts, bool evaluateDirectives, TokList **&out);

  void expand(const TokList *ts, bool evaluateDirectives,
              const std::function<void(const Tok *)> &emitToken);

  const TokList *expandOne(const TokList *ts,
                           const std::function<void(const Tok *)> &emitToken);

  const TokList *substitude(const TokList *ts, const Macro *macro,
                            const std::vector<const TokList *> &actuals,
                            const Hideset *hideset, const TokList *os);

  const TokList *glue(const TokList *ls, const TokList *rs);

  const Tok *merge(const Tok *left, const Tok *right);

  const Tok *stringize(const TokList *ts);

  const TokList *instantiate(const TokList *ts, const Hideset *hideset);

  bool lookupMacro(const Tok *tk, const Macro *&macro) const;

  bool lookupMacroArgument(const Macro *macro,
                           const std::vector<const TokList *> &actuals,
                           const Tok *tk, const TokList *&actual) const;

  const TokList *copyLine(const TokList *ts);

  long constantExpression(const TokList *ts);
  long conditionalExpression(const TokList *&ts);
  long binaryExpression(const TokList *&ts);
  long binaryExpressionHelper(const TokList *&ts, long lhs, int minPrec);
  long unaryExpression(const TokList *&ts);
  long primaryExpression(const TokList *&ts);

  std::tuple<std::vector<const TokList *>, const TokList *, const Hideset *>
  readArguments(const TokList *ts, const Macro *macro);

  std::string_view string(std::string s);

  void print(const TokList *ts, std::ostream &out) const;

  void printLine(const TokList *ts, std::ostream &out) const;
};

static const TokList *concat(Arena *pool, const TokList *ls,
                             const TokList *rs) {
  if (!ls) return rs;
  if (!rs) return ls;
  return new (pool) TokList(ls->head, concat(pool, ls->tail, rs));
}

static const TokList *concat(Arena *pool, const TokList *ts, const Tok *t) {
  return concat(pool, ts, new (pool) TokList(t));
}

const TokList *Preprocessor::Private::tokenize(const std::string_view &source,
                                               int sourceFile, bool bol) {
  cxx::Lexer lex(source);
  lex.setPreprocessing(true);
  const TokList *ts = nullptr;
  auto it = &ts;
  do {
    lex();
    auto tk = Tok::FromCurrentToken(&pool_, lex, sourceFile);
    if (!lex.tokenIsClean()) tk->text = string(std::move(lex.text()));
    *it = new (&pool_) TokList(tk);
    it = const_cast<const TokList **>(&(*it)->tail);
  } while (lex.tokenKind() != cxx::TokenKind::T_EOF_SYMBOL);
  return ts;
}

const TokList *Preprocessor::Private::expand(const TokList *ts,
                                             bool evaluateDirectives) {
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
          error(loc->head->token(), "file '{}' not found", file);
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

        auto dirpath = *path;
        dirpath.remove_filename();
        std::swap(currentPath_, dirpath);
        const int sourceFileId = int(sourceFiles_.size() + 1);
        auto &sourceFile =
            *sourceFiles_.emplace_back(std::make_unique<SourceFile>(
                path->string(), readFile(*path), sourceFileId));
        auto tokens = tokenize(sourceFile.source, sourceFileId, true);
        sourceFile.tokens = tokens;
        if (checkPragmaOnceProtected(tokens)) {
          pragmaOnceProtected_.insert(fn);
        }
        auto prot = checkHeaderProtection(tokens);
        if (prot) ifndefProtectedFiles_.emplace(fn, prot->head->text);
        expand(tokens, /*directives=*/true, emitToken);
        if (prot && macros_.find(prot->head->text) == macros_.end()) {
          auto it = ifndefProtectedFiles_.find(std::string(prot->head->text));
          if (it != ifndefProtectedFiles_.end())
            ifndefProtectedFiles_.erase(it);
        }
        std::swap(currentPath_, dirpath);
        ts = skipLine(directive);
      } else if (matchId(ts, "ifdef")) {
        const Macro *macro = nullptr;
        const auto value = lookupMacro(ts->head, macro);
        if (value)
          pushState(std::tuple(skipping, false));
        else
          pushState(std::tuple(true, !skipping));
      } else if (matchId(ts, "ifndef")) {
        const Macro *macro = nullptr;
        const auto value = !lookupMacro(ts->head, macro);
        if (value)
          pushState(std::tuple(skipping, false));
        else
          pushState(std::tuple(true, !skipping));
      } else if (matchId(ts, "if")) {
        if (skipping)
          pushState(std::tuple(true, false));
        else {
          const auto value = constantExpression(ts);
          if (value)
            pushState(std::tuple(skipping, false));
          else
            pushState(std::tuple(true, !skipping));
        }
      } else if (matchId(ts, "elif")) {
        if (!evaluating)
          setState(std::tuple(true, false));
        else {
          const auto value = constantExpression(ts);
          if (value)
            setState(std::tuple(!evaluating, false));
          else
            setState(std::tuple(true, evaluating));
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
        // throw std::runtime_error(out.str());
#endif
      } else if (!skipping && matchId(ts, "error")) {
        std::ostringstream out;
        printLine(start, out);
        error(directive->head->token(), "{}", out.str());
      } else if (!skipping && matchId(ts, "warning")) {
        std::ostringstream out;
        printLine(start, out);
        warning(directive->head->token(), "{}", out.str());
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
    } else if (!evaluateDirectives && matchId(ts, "__has_feature")) {
      expect(ts, TokenKind::T_LPAREN);
      auto idLoc = ts;
      auto id = expectId(ts);
      expect(ts, TokenKind::T_RPAREN);
      auto t = Tok::Gen(&pool_, TokenKind::T_INTEGER_LITERAL, "1");
      emitToken(t);
    } else {
      ts = expandOne(ts, emitToken);
    }
  }
}

const TokList *Preprocessor::Private::expandOne(
    const TokList *ts, const std::function<void(const Tok *)> &emitToken) {
  if (ts->head->is(TokenKind::T_EOF_SYMBOL)) return ts;

  const Macro *macro = nullptr;

  if (lookupMacro(ts->head, macro)) {
    const auto tk = ts->head;

    if (macro->objLike) {
      const auto hideset = makeUnion(tk->hideset, tk->text);
      auto expanded = substitude(macro->body, macro, {}, hideset, nullptr);
      if (expanded) {
        const_cast<Tok *>(expanded->head)->space = tk->space;
        const_cast<Tok *>(expanded->head)->bol = tk->bol;
      }
      return concat(&pool_, expanded, ts->tail);
    }

    if (!macro->objLike && ts->tail &&
        ts->tail->head->is(TokenKind::T_LPAREN)) {
      auto [args, p, hideset] = readArguments(ts, macro);
      auto hs = makeUnion(makeIntersection(tk->hideset, hideset), tk->text);
      auto expanded = substitude(macro->body, macro, args, hs, nullptr);
      if (expanded) {
        const_cast<Tok *>(expanded->head)->space = tk->space;
        const_cast<Tok *>(expanded->head)->bol = tk->bol;
      }
      return concat(&pool_, expanded, p);
    }
  }

  emitToken(ts->head);
  return ts->tail;
}

const TokList *Preprocessor::Private::substitude(
    const TokList *ts, const Macro *macro,
    const std::vector<const TokList *> &actuals, const Hideset *hideset,
    const TokList *os) {
  while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL)) {
    auto tk = ts->head;
    const TokList *actual = nullptr;

    if (ts->tail && tk->is(TokenKind::T_HASH) &&
        lookupMacroArgument(macro, actuals, ts->tail->head, actual)) {
      auto s = stringize(actual);
      os = concat(&pool_, os, s);
      ts = ts->tail->tail;
    } else if (ts->tail && tk->is(TokenKind::T_HASH_HASH) &&
               lookupMacroArgument(macro, actuals, ts->tail->head, actual)) {
      os = actual ? glue(os, actual) : os;
      ts = ts->tail->tail;
    } else if (ts->tail && tk->is(TokenKind::T_HASH_HASH)) {
      os = glue(os, new (&pool_) TokList(ts->tail->head));
      ts = ts->tail->tail;
    } else if (ts->tail && lookupMacroArgument(macro, actuals, tk, actual) &&
               ts->tail->head->is(TokenKind::T_HASH_HASH)) {
      os = concat(&pool_, os, actual);
      ts = ts->tail;
    } else if (lookupMacroArgument(macro, actuals, tk, actual)) {
      os = concat(&pool_, os, expand(actual, /*directives=*/false));
      ts = ts->tail;
    } else {
      os = concat(&pool_, os, tk);
      ts = ts->tail;
    }
  }

  return instantiate(os, hideset);
}

bool Preprocessor::Private::checkPragmaOnceProtected(const TokList *ts) const {
  if (!ts) return false;
  if (!match(ts, TokenKind::T_HASH)) return false;
  if (bol(ts) || !matchId(ts, "pragma")) return false;
  if (bol(ts) || !matchId(ts, "once")) return false;
  return true;
}

const TokList *Preprocessor::Private::checkHeaderProtection(
    const TokList *ts) const {
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

const TokList *Preprocessor::Private::copyLine(const TokList *ts) {
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

long Preprocessor::Private::constantExpression(const TokList *ts) {
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

long Preprocessor::Private::conditionalExpression(const TokList *&ts) {
  const auto value = binaryExpression(ts);
  if (!match(ts, TokenKind::T_QUESTION)) return value;
  const auto iftrue = conditionalExpression(ts);
  expect(ts, TokenKind::T_COLON);
  const auto iffalse = conditionalExpression(ts);
  return value ? iftrue : iffalse;
}

static int prec(const TokList *ts) {
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

long Preprocessor::Private::binaryExpression(const TokList *&ts) {
  auto e = unaryExpression(ts);
  return binaryExpressionHelper(ts, e, 0);
}

long Preprocessor::Private::binaryExpressionHelper(const TokList *&ts, long lhs,
                                                   int minPrec) {
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
        throw std::runtime_error(
            fmt::format("invalid operator '{}'", Token::spell(op)));
    }  // switch
  }
  return lhs;
}

long Preprocessor::Private::unaryExpression(const TokList *&ts) {
  if (match(ts, TokenKind::T_MINUS)) {
    return -unaryExpression(ts);
  } else if (match(ts, TokenKind::T_PLUS)) {
    return unaryExpression(ts);
  } else if (match(ts, TokenKind::T_TILDE)) {
    return ~unaryExpression(ts);
  } else if (match(ts, TokenKind::T_EXCLAIM)) {
    return !unaryExpression(ts);
  } else {
    return primaryExpression(ts);
  }
}

long Preprocessor::Private::primaryExpression(const TokList *&ts) {
  const auto tk = ts->head;
  if (match(ts, TokenKind::T_INTEGER_LITERAL)) {
    return std::strtol(tk->text.data(), nullptr, 0);
  } else if (match(ts, TokenKind::T_LPAREN)) {
    auto result = conditionalExpression(ts);
    expect(ts, TokenKind::T_RPAREN);
    return result;
  } else {
    ts = ts->tail;
    return 0;
  }
}

bool Preprocessor::Private::lookupMacroArgument(
    const Macro *macro, const std::vector<const TokList *> &actuals,
    const Tok *tk, const TokList *&actual) const {
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

const TokList *Preprocessor::Private::instantiate(const TokList *ts,
                                                  const Hideset *hideset) {
  if (!ts) return nullptr;

  return new (&pool_) TokList(Tok::WithHideset(&pool_, ts->head, hideset),
                              instantiate(ts->tail, hideset));
}

std::tuple<std::vector<const TokList *>, const TokList *, const Hideset *>
Preprocessor::Private::readArguments(const TokList *ts, const Macro *macro) {
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
      } else if (tk->is(TokenKind::T_LPAREN)) {
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

const Tok *Preprocessor::Private::stringize(const TokList *ts) {
  std::string s;

  const auto start = ts;

  for (; ts; ts = ts->tail) {
    if (!s.empty() && (ts->head->space || ts->head->bol)) s += ' ';
    s += ts->head->text;
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

  auto tk = Tok::Gen(&pool_, TokenKind::T_STRING_LITERAL, string(o));
  if (start) {
    tk->sourceFile = start->head->sourceFile;
    tk->offset = start->head->offset;
  }

  return tk;
}

std::string_view Preprocessor::Private::string(std::string s) {
  return std::string_view(scratchBuffer_.emplace_front(std::move(s)));
}

void Preprocessor::Private::defineMacro(const TokList *ts) {
#if 0
  fmt::print("*** defining macro: ");
  printLine(ts, std::cout);
  fmt::print("\n");
#endif

  auto name = ts->head->text;

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

    Macro m;
    m.objLike = false;
    m.body = ts;
    m.formals = std::move(formals);
    m.variadic = variadic;
    macros_.emplace(name, std::move(m));
    return;
  }

  Macro m;
  m.objLike = true;
  m.body = ts->tail;
  macros_.emplace(name, std::move(m));
}

const Tok *Preprocessor::Private::merge(const Tok *left, const Tok *right) {
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

const TokList *Preprocessor::Private::glue(const TokList *ls,
                                           const TokList *rs) {
  if (!ls->tail && rs)
    return new (&pool_) TokList(merge(ls->head, rs->head), rs->tail);
  return new (&pool_) TokList(ls->head, glue(ls->tail, rs));
}

const TokList *Preprocessor::Private::skipLine(const TokList *ts) {
  while (ts && ts->head->isNot(TokenKind::T_EOF_SYMBOL) && !ts->head->bol)
    ts = ts->tail;
  return ts;
}

bool Preprocessor::Private::lookupMacro(const Tok *tk,
                                        const Macro *&macro) const {
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

static bool wantSpace(TokenKind kind) {
  switch (kind) {
    case TokenKind::T_IDENTIFIER:
    case TokenKind::T_INTEGER_LITERAL:
    case TokenKind::T_FLOATING_POINT_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_STRING_LITERAL:
      return true;
    default:
      return false;
  }  // switch
}

static bool needSpace(const Tok *prev, const Tok *current) {
  if (!prev || current->space) return current->space;
  return wantSpace(prev->kind) && wantSpace(current->kind);
}

void Preprocessor::Private::print(const TokList *ts, std::ostream &out) const {
  bool first = true;
  for (const Tok *prevTk = nullptr; ts; ts = ts->tail) {
    auto tk = ts->head;
    if (tk->text.empty()) continue;
    if (tk->bol)
      fmt::print(out, "\n");
    else if (!first && needSpace(prevTk, tk))
      fmt::print(out, " ");
    fmt::print(out, "{}", tk->text);
    prevTk = tk;
    first = false;
  }
}

void Preprocessor::Private::printLine(const TokList *ts,
                                      std::ostream &out) const {
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
  fmt::print(out, "\n");
}

Preprocessor::Preprocessor(Control *control,
                           DiagnosticsClient *diagnosticsClient)
    : d(std::make_unique<Private>()) {
  d->control_ = control;
  d->diagnosticsClient_ = diagnosticsClient;
}

Preprocessor::~Preprocessor() {}

DiagnosticsClient *Preprocessor::diagnosticsClient() const {
  return d->diagnosticsClient_;
}

void Preprocessor::squeeze() { d->pool_.reset(); }

void Preprocessor::operator()(std::string source, std::string fileName,
                              std::ostream &out) {
  preprocess(std::move(source), std::move(fileName), out);
}

void Preprocessor::preprocess(std::string source, std::string fileName,
                              std::ostream &out) {
  const int sourceFileId = int(d->sourceFiles_.size() + 1);
  auto &sourceFile = *d->sourceFiles_.emplace_back(std::make_unique<SourceFile>(
      std::move(fileName), std::move(source), sourceFileId));

  fs::path path(sourceFile.fileName);
  path.remove_filename();

  std::swap(d->currentPath_, path);

  const auto ts = d->tokenize(sourceFile.source, sourceFileId, true);

  const auto os = d->expand(ts, /*directives*/ true);

  std::swap(d->currentPath_, path);

  int outFile = 0;
  int outLine = -1;

  const Tok *prevTk = nullptr;

  for (auto it = os; it; it = it->tail) {
    auto tk = it->head;
    auto file =
        tk->sourceFile > 0 ? &*d->sourceFiles_[tk->sourceFile - 1] : nullptr;
    if ((tk->bol || it == os) && file) {
      std::string_view fileName;
      unsigned line = 0;
      file->getTokenStartPosition(tk->offset, &line, nullptr, &fileName);
      if (outFile == tk->sourceFile && line == outLine) {
        ++outLine;
        fmt::print(out, "\n");
      } else {
        if (it != os) fmt::print(out, "\n");
        if (tk->sourceFile == outFile)
          fmt::print(out, "#line {}\n", line);
        else
          fmt::print(out, "#line {} \"{}\"\n", line, fileName);
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
  const int sourceFileId = int(d->sourceFiles_.size() + 1);
  auto &sourceFile = *d->sourceFiles_.emplace_back(std::make_unique<SourceFile>(
      std::move(fileName), std::move(source), sourceFileId));

  fs::path path(sourceFile.fileName);
  path.remove_filename();

  std::swap(d->currentPath_, path);

  const auto ts = d->tokenize(sourceFile.source, sourceFileId, true);

  sourceFile.tokens = ts;

  tokens.emplace_back(TokenKind::T_ERROR);

  d->expand(ts, /*directives*/ true, [&](const Tok *tk) {
    auto kind = tk->kind;
    const auto fileId = tk->sourceFile;

    TokenValue value;

    switch (tk->kind) {
      case TokenKind::T_IDENTIFIER: {
        kind = Lexer::classifyKeyword(tk->text);
        if (kind == TokenKind::T_IDENTIFIER)
          value.idValue = d->control_->identifier(std::string(tk->text));
        break;
      }

      case TokenKind::T_CHARACTER_LITERAL:
        value.literalValue = d->control_->charLiteral(std::string(tk->text));
        break;

      case TokenKind::T_STRING_LITERAL:
        value.literalValue = d->control_->stringLiteral(std::string(tk->text));
        break;

      case TokenKind::T_INTEGER_LITERAL:
      case TokenKind::T_FLOATING_POINT_LITERAL:
        value.literalValue = d->control_->numericLiteral(std::string(tk->text));
        break;

      default:
        break;
    }  // switch

    if (tk->kind == TokenKind::T_GREATER_EQUAL) {
      value.tokenKindValue = tk->kind;

      Token token(TokenKind::T_GREATER, tk->offset, 1, value);
      token.setFileId(fileId);
      token.setLeadingSpace(tk->space);
      token.setStartOfLine(tk->bol);
      tokens.push_back(token);

      token = Token(TokenKind::T_EQUAL, tk->offset + 1, 1);
      token.setFileId(fileId);
      token.setLeadingSpace(false);
      token.setStartOfLine(false);
      tokens.push_back(token);
    } else if (tk->kind == TokenKind::T_GREATER_GREATER) {
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
    } else if (tk->kind == TokenKind::T_GREATER_GREATER_EQUAL) {
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

      token = Token(TokenKind::T_EQUAL, tk->offset + 2, 1);
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
                      uint32_t(sourceFile.source.size()));

  tokens.back().setFileId(sourceFileId);

  std::swap(d->currentPath_, path);
}

const std::vector<std::filesystem::path> &Preprocessor::systemIncludePaths()
    const {
  return d->systemIncludePaths_;
}

void Preprocessor::addSystemIncludePath(const std::string &path) {
  d->systemIncludePaths_.push_back(fs::absolute(path));
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
    *line = 0;
    *column = 0;
    *fileName = "??";
    return;
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];
  sourceFile.getTokenStartPosition(token.offset(), line, column, fileName);
}

void Preprocessor::getTokenEndPosition(const Token &token, unsigned *line,
                                       unsigned *column,
                                       std::string_view *fileName) const {
  if (token.fileId() == 0) {
    *line = 0;
    *column = 0;
    *fileName = "??";
    return;
  }

  auto &sourceFile = *d->sourceFiles_[token.fileId() - 1];
  sourceFile.getTokenStartPosition(token.offset() + token.length(), line,
                                   column, fileName);
}

std::string_view Preprocessor::getTextLine(const Token &token) const {
  if (token.fileId() == 0) return std::string_view();
  const SourceFile *file = d->sourceFiles_[token.fileId() - 1].get();
  unsigned line = 0;
  getTokenStartPosition(token, &line, nullptr, nullptr);
  std::string_view source = file->source;
  const auto &lines = file->lines;
  const auto start = lines.at(line - 1);
  const auto end = line < lines.size() ? lines.at(line) : source.length();
  return source.substr(start, end - start);
}

}  // namespace cxx
