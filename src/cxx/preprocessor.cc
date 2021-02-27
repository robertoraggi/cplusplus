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

#include <cxx/arena.h>
#include <cxx/lexer.h>
#include <cxx/preprocessor.h>

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

// stl
#include <filesystem>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <optional>
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

  explicit Hideset(std::unordered_set<std::string_view> names)
      : names_(std::move(names)) {}

  bool contains(const std::string_view &name) const {
    return names_.contains(name);
  }

  const std::unordered_set<std::string_view> &names() const { return names_; };

  bool operator==(const Hideset &other) const { return false; }

 private:
  std::unordered_set<std::string_view> names_;
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

struct Tok final : Managed {
  std::string_view text;
  const Hideset *hideset = nullptr;
  bool bol = false;
  bool space = false;

  Tok() = default;

  Tok(const Tok &other) = default;
  Tok &operator=(const Tok &other) = default;

  Tok(Tok &&other) = default;
  Tok &operator=(Tok &&other) = default;

  Tok(const Tok *tok, const Hideset *hs) {
    text = tok->text;
    bol = tok->bol;
    space = tok->space;
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
};

struct Preprocessor::Private {
  std::vector<fs::path> systemIncludePaths_;
  std::vector<fs::path> quoteIncludePaths_;
  std::unordered_map<std::string_view, Macro> macros_;
  std::unordered_set<Hideset> hidesets;
  std::forward_list<std::string> scratchBuffer_;
  fs::path currentPath_;
  Arena pool_;

  const Hideset *makeUnion(const Hideset *hs, const std::string_view &name) {
    if (!hs) return get(std::unordered_set<std::string_view>{name});
    auto names = hs->names();
    names.insert(name);
    return get(std::move(names));
  }

  const Hideset *makeUnion(const Hideset *hs, const Hideset *other) {
    if (!other) return hs;
    if (!hs) return other;

    std::unordered_set<std::string_view> names;

    std::set_union(begin(hs->names()), end(hs->names()), begin(other->names()),
                   end(other->names()), std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  const Hideset *makeIntersection(const Hideset *hs, const Hideset *other) {
    if (!other) return hs;
    if (!hs) return other;

    std::unordered_set<std::string_view> names;

    std::set_intersection(begin(hs->names()), end(hs->names()),
                          begin(other->names()), end(other->names()),
                          std::inserter(names, names.begin()));

    return get(std::move(names));
  }

  const Hideset *get(std::unordered_set<std::string_view> name) {
    if (name.empty()) return nullptr;
    return &*hidesets.emplace(name).first;
  }

  std::string readFile(const fs::path &file) const {
    std::ifstream in(file);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
  }

  std::optional<fs::path> resolve(const Include &include) const {
    struct Resolve {
      const Private *d;

      explicit Resolve(const Private *d) : d(d) {}

      std::optional<fs::path> operator()(std::monostate) const { return {}; }

      std::optional<fs::path> operator()(const SystemInclude &include) const {
        for (const auto &p : d->systemIncludePaths_) {
          auto path = p / include.fileName;
          if (exists(path)) return path;
        }
        return {};
      }

      std::optional<fs::path> operator()(const QuoteInclude &include) const {
        if (exists(d->currentPath_ / include.fileName))
          return d->currentPath_ / include.fileName;

        for (const auto &p : d->quoteIncludePaths_) {
          auto path = p / include.fileName;
          if (exists(path)) return path;
        }
        for (const auto &p : d->systemIncludePaths_) {
          auto path = p / include.fileName;
          if (exists(path)) return path;
        }
        return {};
      }
    };

    return std::visit(Resolve(this), include);
  }

  void defineMacro(const TokList *ts);

  const TokList *tokenize(const std::string_view &source, bool bol);

  const TokList *skipLine(const TokList *ts);

  const TokList *expand(const TokList *ts);

  const TokList *substitude(const TokList *ts, const Macro *macro,
                            const std::vector<const TokList *> &actuals,
                            const Hideset *hideset, const TokList *os);

  const TokList *glue(const TokList *ls, const TokList *rs);

  const Tok *merge(const Tok *left, const Tok *right);

  const Tok *stringize(const TokList *ts);

  const TokList *instantiate(const TokList *ts, const Hideset *hideset);

  const Macro *lookupMacro(const Tok *tk) const;

  bool lookupFormal(const Macro *macro, const Tok *tk,
                    std::size_t &index) const;

  std::tuple<std::vector<const TokList *>, const TokList *, const Hideset *>
  readArguments(const TokList *ts);

  std::string_view string(std::string s);

  bool match(const TokList *&ts, const std::string_view &s) const;

  void expect(const TokList *&ts, const std::string_view &s) const;

  std::string_view expectId(const TokList *&ts) const;

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

bool Preprocessor::Private::match(const TokList *&ts,
                                  const std::string_view &s) const {
  if (ts && ts->head->text == s) {
    ts = ts->tail;
    return true;
  }
  return false;
};

void Preprocessor::Private::expect(const TokList *&ts,
                                   const std::string_view &s) const {
  if (!match(ts, s)) throw std::runtime_error(fmt::format("expected '{}'", s));
};

std::string_view Preprocessor::Private::expectId(const TokList *&ts) const {
  auto ch = ts->head->text[0];
  if (std::isalpha(ch) || ch == '_') {
    auto id = ts->head->text;
    ts = ts->tail;
    return id;
  }
  throw std::runtime_error("expected an identifier");
};

const TokList *Preprocessor::Private::tokenize(const std::string_view &source,
                                               bool bol) {
  cxx::Lexer lex(source);
  lex.setPreprocessing(true);
  const TokList *ts = nullptr;
  auto it = &ts;
  while (lex() != cxx::TokenKind::T_EOF_SYMBOL) {
    auto tk = new (&pool_) Tok();
    tk->text = lex.tokenText();
    tk->bol = lex.tokenStartOfLine();
    tk->space = lex.tokenLeadingSpace();
    *it = new (&pool_) TokList(tk);
    it = const_cast<const TokList **>(&(*it)->tail);
  }
  return ts;
}

const TokList *Preprocessor::Private::expand(const TokList *ts) {
  TokList *tokens = nullptr;
  auto out = &tokens;

  while (ts) {
    const auto tk = ts->head;
    const auto start = ts;

    if (tk->bol && match(ts, "#")) {
      if (match(ts, "define")) {
        TokList *def = nullptr;
        auto it = &def;
        for (; ts && !ts->head->bol; ts = ts->tail) {
          *it = new (&pool_) TokList(ts->head);
          it = const_cast<TokList **>(&(*it)->tail);
        }
        defineMacro(def);
      } else if (match(ts, "undef")) {
        auto name = expectId(ts);
        auto it = macros_.find(name);
        if (it != macros_.end()) macros_.erase(it);
      } else if (match(ts, "include")) {
        // ###
        std::optional<fs::path> path;
        std::string file;
        if (ts->head->text[0] == '"') {
          file = ts->head->text.substr(1, ts->head->text.length() - 2);
          path = resolve(QuoteInclude(file));
        } else if (match(ts, "<")) {
          while (ts && !ts->head->bol) {
            if (match(ts, ">")) break;
            file += ts->head->text;
            ts = ts->tail;
          }
          path = resolve(QuoteInclude(file));
        }
        if (!path)
          throw std::runtime_error(fmt::format("file '{}' not found", file));
        auto dirpath = *path;
        dirpath.remove_filename();
        std::swap(currentPath_, dirpath);
        auto source = string(readFile(*path));
        auto tokens = tokenize(source, true);
        std::swap(currentPath_, dirpath);
        for (auto t = tokens; t; t = t->tail) {
          if (!t->tail) {
            const_cast<TokList *>(t)->tail = skipLine(ts);
            break;
          }
        }
        ts = tokens;
      } else if (match(ts, "include_next")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "ifdef")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "ifndef")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "if")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "elif")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "else")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "endif")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "line")) {
        // ###
      } else if (match(ts, "pragma")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "error")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      } else if (match(ts, "warning")) {
        // ###
        std::ostringstream out;
        printLine(start, out);
        throw std::runtime_error(out.str());
      }
      ts = skipLine(ts);
    } else if (auto macro = lookupMacro(tk)) {
      if (macro->objLike) {
        auto expanded = substitude(macro->body, macro, {},
                                   makeUnion(tk->hideset, tk->text), nullptr);
        ts = concat(&pool_, expanded, ts->tail);
      } else if (!macro->objLike && ts->tail && ts->tail->head->text == "(") {
        auto [args, p, hideset] = readArguments(ts);

        auto hs = makeUnion(makeIntersection(tk->hideset, hideset), tk->text);

        auto expanded = substitude(macro->body, macro, args, hs, nullptr);

        ts = concat(&pool_, expanded, p);
      } else {
        *out = new (&pool_) TokList(ts->head);
        out = const_cast<TokList **>(&(*out)->tail);
        ts = ts->tail;
      }
    } else {
      *out = new (&pool_) TokList(ts->head);
      out = const_cast<TokList **>(&(*out)->tail);
      ts = ts->tail;
    }
  }

  return tokens;
}

const TokList *Preprocessor::Private::substitude(
    const TokList *is, const Macro *macro,
    const std::vector<const TokList *> &actuals, const Hideset *hideset,
    const TokList *os) {
  while (is) {
    auto t = is->head;
    std::size_t index = 0;

    if (is->tail && is->head->text == "#" &&
        lookupFormal(macro, is->tail->head, index)) {
      os = concat(&pool_, os, stringize(actuals[index]));
      is = is->tail->tail;
    } else if (is->tail && t->text == "##" &&
               lookupFormal(macro, is->tail->head, index)) {
      auto actual = actuals[index];
      os = actual ? glue(os, actual) : os;
      is = is->tail->tail;
    } else if (is->tail && t->text == "##") {
      os = glue(os, new (&pool_) TokList(is->tail->head));
      is = is->tail->tail;
    } else if (is->tail && lookupFormal(macro, t, index) &&
               is->tail->head->text == "##") {
      auto actual = actuals[index];
      os = concat(&pool_, os, actual);
      is = is->tail;
    } else if (lookupFormal(macro, t, index)) {
      os = concat(&pool_, os, actuals[index]);
      is = is->tail;
    } else {
      os = concat(&pool_, os, t);
      is = is->tail;
    }
  }

  return instantiate(os, hideset);
}

bool Preprocessor::Private::lookupFormal(const Macro *macro, const Tok *tk,
                                         std::size_t &index) const {
  if (!tk) return false;
  for (std::size_t i = 0; i < macro->formals.size(); ++i) {
    if (macro->formals[i] == tk->text) {
      index = i;
      return true;
    }
  }
  return false;
}

const TokList *Preprocessor::Private::instantiate(const TokList *ts,
                                                  const Hideset *hideset) {
  if (!ts) return nullptr;

  return new (&pool_) TokList(new (&pool_) Tok(ts->head, hideset),
                              instantiate(ts->tail, hideset));
}

std::tuple<std::vector<const TokList *>, const TokList *, const Hideset *>
Preprocessor::Private::readArguments(const TokList *ts) {
  auto it = ts->tail->tail;
  int depth = 1;
  int argc = 0;
  std::vector<const TokList *> args;
  const Tok *rp = nullptr;
  if (it->head->text != ")") {
    TokList *arg = nullptr;
    auto argIt = &arg;
    while (it) {
      auto tk = it->head;
      it = it->tail;
      if (depth == 1 && tk->text == ",") {
        args.push_back(arg);
        arg = nullptr;
        argIt = &arg;
        ++argc;
        continue;
      } else if (tk->text == "(") {
        ++depth;
      } else if (tk->text == ")" && !--depth) {
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
  for (; ts; ts = ts->tail) {
    s += ts->head->text;
    if (ts->tail) s += ' ';
  }
  std::string o;
  for (auto c : s) {
    if (c == '\\')
      o += "\\\\";
    else if (c == '"')
      o += "\\\"";
    else
      o += c;
  }
  Tok *r = new (&pool_) Tok();
  r->text = std::move(o);
  r->space = true;
  return r;
}

std::string_view Preprocessor::Private::string(std::string s) {
  return std::string_view(scratchBuffer_.emplace_front(std::move(s)));
}

void Preprocessor::Private::defineMacro(const TokList *ts) {
  auto name = ts->head->text;

  if (ts->tail && !ts->tail->head->space && ts->tail->head->text == "(") {
    ts = ts->tail->tail;  // skip macro name and '('

    std::vector<std::string_view> formals;

    if (!match(ts, ")")) {
      formals.push_back(expectId(ts));
      while (match(ts, ",")) formals.push_back(expectId(ts));
      expect(ts, ")");
    }

    Macro m;
    m.objLike = false;
    m.body = ts;
    m.formals = std::move(formals);
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
  Tok *tok = new (&pool_) Tok();
  tok->text = string(std::string(left->text) + std::string(right->text));
  tok->bol = left->bol;
  tok->space = left->space;
  tok->hideset = makeIntersection(left->hideset, right->hideset);
  return tok;
}

const TokList *Preprocessor::Private::glue(const TokList *ls,
                                           const TokList *rs) {
  if (!ls->tail && rs)
    return new (&pool_) TokList(merge(ls->head, rs->head), rs->tail);
  return new (&pool_) TokList(ls->head, glue(ls->tail, rs));
}

const TokList *Preprocessor::Private::skipLine(const TokList *ts) {
  while (ts && !ts->head->bol) ts = ts->tail;
  return ts;
}

const Macro *Preprocessor::Private::lookupMacro(const Tok *tk) const {
  if (std::isalpha(tk->text[0]) || tk->text[0] == '_') {
    auto it = macros_.find(tk->text);
    if (it != macros_.end()) {
      const auto disabled = tk->hideset && tk->hideset->contains(tk->text);
      if (!disabled) return &it->second;
    }
  }
  return nullptr;
}

void Preprocessor::Private::print(const TokList *ts, std::ostream &out) const {
  for (; ts; ts = ts->tail) {
    auto tk = ts->head;
    if (tk->text.empty()) continue;
    if (tk->bol)
      fmt::print(out, "\n");
    else if (tk->space)
      fmt::print(out, " ");
    fmt::print(out, "{}", tk->text);
  }
}

void Preprocessor::Private::printLine(const TokList *ts,
                                      std::ostream &out) const {
  for (; ts; ts = ts->tail) {
    auto tk = ts->head;
    if (tk->text.empty()) continue;
    if (tk->space) fmt::print(out, " ");
    fmt::print(out, "{}", tk->text);
    if (ts->tail && ts->tail->head->bol) break;
  }
  fmt::print(out, "\n");
}

Preprocessor::Preprocessor() {
  d = new Private();

  d->currentPath_ = fs::current_path();
}

Preprocessor::~Preprocessor() { delete d; }

void Preprocessor::operator()(const std::string_view &source,
                              const std::string &fileName, std::ostream &out) {
  preprocess(source, fileName, out);
}

void Preprocessor::preprocess(const std::string_view &source,
                              const std::string &fileName, std::ostream &out) {
  fs::path path(fileName);
  path.remove_filename();

  std::swap(d->currentPath_, path);

  const auto ts = d->tokenize(source, true);
  const auto os = d->expand(ts);

  std::swap(d->currentPath_, path);

  d->print(os, out);
  fmt::print(out, "\n");
}

void Preprocessor::defineMacro(const std::string &name,
                               const std::string &body) {
  auto s = d->string(name + " " + body);
  auto tokens = d->tokenize(s, false);
  d->defineMacro(tokens);
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
    d->print(macro.body, out);
    fmt::print(out, "\n");
  }
}

void Preprocessor::addSystemIncludePaths() {
  // clang-format off
  d->systemIncludePaths_.emplace_back("/usr/include/c++/9");
  d->systemIncludePaths_.emplace_back("/usr/include/x86_64-linux-gnu/c++/9");
  d->systemIncludePaths_.emplace_back("/usr/include/c++/9/backward");
  d->systemIncludePaths_.emplace_back("/usr/lib/gcc/x86_64-linux-gnu/9/include");
  d->systemIncludePaths_.emplace_back("/usr/local/include");
  d->systemIncludePaths_.emplace_back("/usr/include/x86_64-linux-gnu");
  d->systemIncludePaths_.emplace_back("/usr/include");
  // clang-format on
}

void Preprocessor::addPredefinedMacros() {
  // clang-format off
  defineMacro("_GNU_SOURCE", "1");
  defineMacro("_LP64", "1");
  defineMacro("__ATOMIC_ACQUIRE", "2");
  defineMacro("__ATOMIC_ACQ_REL", "4");
  defineMacro("__ATOMIC_CONSUME", "1");
  defineMacro("__ATOMIC_RELAXED", "0");
  defineMacro("__ATOMIC_RELEASE", "3");
  defineMacro("__ATOMIC_SEQ_CST", "5");
  defineMacro("__BIGGEST_ALIGNMENT__", "16");
  defineMacro("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__CHAR16_TYPE__", "unsigned short");
  defineMacro("__CHAR32_TYPE__", "unsigned int");
  defineMacro("__CHAR_BIT__", "8");
  defineMacro("__CLANG_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__CONSTANT_CFSTRINGS__", "1");
  defineMacro("__DBL_DECIMAL_DIG__", "17");
  defineMacro("__DBL_DENORM_MIN__", "4.9406564584124654e-324");
  defineMacro("__DBL_DIG__", "15");
  defineMacro("__DBL_EPSILON__", "2.2204460492503131e-16");
  defineMacro("__DBL_HAS_DENORM__", "1");
  defineMacro("__DBL_HAS_INFINITY__", "1");
  defineMacro("__DBL_HAS_QUIET_NAN__", "1");
  defineMacro("__DBL_MANT_DIG__", "53");
  defineMacro("__DBL_MAX_10_EXP__", "308");
  defineMacro("__DBL_MAX_EXP__", "1024");
  defineMacro("__DBL_MAX__", "1.7976931348623157e+308");
  defineMacro("__DBL_MIN_10_EXP__", "(-307)");
  defineMacro("__DBL_MIN_EXP__", "(-1021)");
  defineMacro("__DBL_MIN__", "2.2250738585072014e-308");
  defineMacro("__DECIMAL_DIG__", "__LDBL_DECIMAL_DIG__");
  defineMacro("__DEPRECATED", "1");
  defineMacro("__ELF__", "1");
  defineMacro("__EXCEPTIONS", "1");
  defineMacro("__FINITE_MATH_ONLY__", "0");
  defineMacro("__FLOAT128__", "1");
  defineMacro("__FLT_DECIMAL_DIG__", "9");
  defineMacro("__FLT_DENORM_MIN__", "1.40129846e-45F");
  defineMacro("__FLT_DIG__", "6");
  defineMacro("__FLT_EPSILON__", "1.19209290e-7F");
  defineMacro("__FLT_EVAL_METHOD__", "0");
  defineMacro("__FLT_HAS_DENORM__", "1");
  defineMacro("__FLT_HAS_INFINITY__", "1");
  defineMacro("__FLT_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT_MANT_DIG__", "24");
  defineMacro("__FLT_MAX_10_EXP__", "38");
  defineMacro("__FLT_MAX_EXP__", "128");
  defineMacro("__FLT_MAX__", "3.40282347e+38F");
  defineMacro("__FLT_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT_MIN_EXP__", "(-125)");
  defineMacro("__FLT_MIN__", "1.17549435e-38F");
  defineMacro("__FLT_RADIX__", "2");
  defineMacro("__FXSR__", "1");
  defineMacro("__GCC_ASM_FLAG_OUTPUTS__", "1");
  defineMacro("__GCC_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_TEST_AND_SET_TRUEVAL", "1");
  defineMacro("__GCC_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8", "1");
  defineMacro("__GNUC_GNU_INLINE__", "1");
  defineMacro("__GNUC_MINOR__", "2");
  defineMacro("__GNUC_PATCHLEVEL__", "1");
  defineMacro("__GNUC__", "4");
  defineMacro("__GNUG__", "4");
  defineMacro("__GXX_ABI_VERSION", "1002");
  defineMacro("__GXX_EXPERIMENTAL_CXX0X__", "1");
  defineMacro("__GXX_RTTI", "1");
  defineMacro("__GXX_WEAK__", "1");
  defineMacro("__INT16_C_SUFFIX__", "");
  defineMacro("__INT16_FMTd__", "\"hd\"");
  defineMacro("__INT16_FMTi__", "\"hi\"");
  defineMacro("__INT16_MAX__", "32767");
  defineMacro("__INT16_TYPE__", "short");
  defineMacro("__INT32_C_SUFFIX__", "");
  defineMacro("__INT32_FMTd__", "\"d\"");
  defineMacro("__INT32_FMTi__", "\"i\"");
  defineMacro("__INT32_MAX__", "2147483647");
  defineMacro("__INT32_TYPE__", "int");
  defineMacro("__INT64_C_SUFFIX__", "L");
  defineMacro("__INT64_FMTd__", "\"ld\"");
  defineMacro("__INT64_FMTi__", "\"li\"");
  defineMacro("__INT64_MAX__", "9223372036854775807L");
  defineMacro("__INT64_TYPE__", "long int");
  defineMacro("__INT8_C_SUFFIX__", "");
  defineMacro("__INT8_FMTd__", "\"hhd\"");
  defineMacro("__INT8_FMTi__", "\"hhi\"");
  defineMacro("__INT8_MAX__", "127");
  defineMacro("__INT8_TYPE__", "signed char");
  defineMacro("__INTMAX_C_SUFFIX__", "L");
  defineMacro("__INTMAX_FMTd__", "\"ld\"");
  defineMacro("__INTMAX_FMTi__", "\"li\"");
  defineMacro("__INTMAX_MAX__", "9223372036854775807L");
  defineMacro("__INTMAX_TYPE__", "long int");
  defineMacro("__INTMAX_WIDTH__", "64");
  defineMacro("__INTPTR_FMTd__", "\"ld\"");
  defineMacro("__INTPTR_FMTi__", "\"li\"");
  defineMacro("__INTPTR_MAX__", "9223372036854775807L");
  defineMacro("__INTPTR_TYPE__", "long int");
  defineMacro("__INTPTR_WIDTH__", "64");
  defineMacro("__INT_FAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_FAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_FAST16_MAX__", "32767");
  defineMacro("__INT_FAST16_TYPE__", "short");
  defineMacro("__INT_FAST32_FMTd__", "\"d\"");
  defineMacro("__INT_FAST32_FMTi__", "\"i\"");
  defineMacro("__INT_FAST32_MAX__", "2147483647");
  defineMacro("__INT_FAST32_TYPE__", "int");
  defineMacro("__INT_FAST64_FMTd__", "\"ld\"");
  defineMacro("__INT_FAST64_FMTi__", "\"li\"");
  defineMacro("__INT_FAST64_MAX__", "9223372036854775807L");
  defineMacro("__INT_FAST64_TYPE__", "long int");
  defineMacro("__INT_FAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_FAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_FAST8_MAX__", "127");
  defineMacro("__INT_FAST8_TYPE__", "signed char");
  defineMacro("__INT_LEAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_LEAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_LEAST16_MAX__", "32767");
  defineMacro("__INT_LEAST16_TYPE__", "short");
  defineMacro("__INT_LEAST32_FMTd__", "\"d\"");
  defineMacro("__INT_LEAST32_FMTi__", "\"i\"");
  defineMacro("__INT_LEAST32_MAX__", "2147483647");
  defineMacro("__INT_LEAST32_TYPE__", "int");
  defineMacro("__INT_LEAST64_FMTd__", "\"ld\"");
  defineMacro("__INT_LEAST64_FMTi__", "\"li\"");
  defineMacro("__INT_LEAST64_MAX__", "9223372036854775807L");
  defineMacro("__INT_LEAST64_TYPE__", "long int");
  defineMacro("__INT_LEAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_LEAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_LEAST8_MAX__", "127");
  defineMacro("__INT_LEAST8_TYPE__", "signed char");
  defineMacro("__INT_MAX__", "2147483647");
  defineMacro("__LDBL_DECIMAL_DIG__", "21");
  defineMacro("__LDBL_DENORM_MIN__", "3.64519953188247460253e-4951L");
  defineMacro("__LDBL_DIG__", "18");
  defineMacro("__LDBL_EPSILON__", "1.08420217248550443401e-19L");
  defineMacro("__LDBL_HAS_DENORM__", "1");
  defineMacro("__LDBL_HAS_INFINITY__", "1");
  defineMacro("__LDBL_HAS_QUIET_NAN__", "1");
  defineMacro("__LDBL_MANT_DIG__", "64");
  defineMacro("__LDBL_MAX_10_EXP__", "4932");
  defineMacro("__LDBL_MAX_EXP__", "16384");
  defineMacro("__LDBL_MAX__", "1.18973149535723176502e+4932L");
  defineMacro("__LDBL_MIN_10_EXP__", "(-4931)");
  defineMacro("__LDBL_MIN_EXP__", "(-16381)");
  defineMacro("__LDBL_MIN__", "3.36210314311209350626e-4932L");
  defineMacro("__LITTLE_ENDIAN__", "1");
  defineMacro("__LONG_LONG_MAX__", "9223372036854775807LL");
  defineMacro("__LONG_MAX__", "9223372036854775807L");
  defineMacro("__LP64__", "1");
  defineMacro("__MMX__", "1");
  defineMacro("__NO_INLINE__", "1");
  defineMacro("__NO_MATH_INLINES", "1");
  defineMacro("__OBJC_BOOL_IS_BOOL", "0");
  defineMacro("__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES", "3");
  defineMacro("__OPENCL_MEMORY_SCOPE_DEVICE", "2");
  defineMacro("__OPENCL_MEMORY_SCOPE_SUB_GROUP", "4");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_GROUP", "1");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_ITEM", "0");
  defineMacro("__ORDER_BIG_ENDIAN__", "4321");
  defineMacro("__ORDER_LITTLE_ENDIAN__", "1234");
  defineMacro("__ORDER_PDP_ENDIAN__", "3412");
  defineMacro("__POINTER_WIDTH__", "64");
  defineMacro("__PRAGMA_REDEFINE_EXTNAME", "1");
  defineMacro("__PTRDIFF_FMTd__", "\"ld\"");
  defineMacro("__PTRDIFF_FMTi__", "\"li\"");
  defineMacro("__PTRDIFF_MAX__", "9223372036854775807L");
  defineMacro("__PTRDIFF_TYPE__", "long int");
  defineMacro("__PTRDIFF_WIDTH__", "64");
  defineMacro("__REGISTER_PREFIX__", "");
  defineMacro("__SCHAR_MAX__", "127");
  defineMacro("__SEG_FS", "1");
  defineMacro("__SEG_GS", "1");
  defineMacro("__SHRT_MAX__", "32767");
  defineMacro("__SIG_ATOMIC_MAX__", "2147483647");
  defineMacro("__SIG_ATOMIC_WIDTH__", "32");
  defineMacro("__SIZEOF_DOUBLE__", "8");
  defineMacro("__SIZEOF_FLOAT128__", "16");
  defineMacro("__SIZEOF_FLOAT__", "4");
  defineMacro("__SIZEOF_INT128__", "16");
  defineMacro("__SIZEOF_INT__", "4");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "16");
  defineMacro("__SIZEOF_LONG_LONG__", "8");
  defineMacro("__SIZEOF_LONG__", "8");
  defineMacro("__SIZEOF_POINTER__", "8");
  defineMacro("__SIZEOF_PTRDIFF_T__", "8");
  defineMacro("__SIZEOF_SHORT__", "2");
  defineMacro("__SIZEOF_SIZE_T__", "8");
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__SIZE_FMTX__", "\"lX\"");
  defineMacro("__SIZE_FMTo__", "\"lo\"");
  defineMacro("__SIZE_FMTu__", "\"lu\"");
  defineMacro("__SIZE_FMTx__", "\"lx\"");
  defineMacro("__SIZE_MAX__", "18446744073709551615UL");
  defineMacro("__SIZE_TYPE__", "long unsigned int");
  defineMacro("__SIZE_WIDTH__", "64");
  defineMacro("__SSE2_MATH__", "1");
  defineMacro("__SSE2__", "1");
  defineMacro("__SSE_MATH__", "1");
  defineMacro("__SSE__", "1");
  defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__", "16UL");
  defineMacro("__STDC_HOSTED__", "1");
  defineMacro("__STDC_UTF_16__", "1");
  defineMacro("__STDC_UTF_32__", "1");
  defineMacro("__STDC__", "1");
  defineMacro("__STRICT_ANSI__", "1");
  defineMacro("__UINT16_C_SUFFIX__", "");
  defineMacro("__UINT16_FMTX__", "\"hX\"");
  defineMacro("__UINT16_FMTo__", "\"ho\"");
  defineMacro("__UINT16_FMTu__", "\"hu\"");
  defineMacro("__UINT16_FMTx__", "\"hx\"");
  defineMacro("__UINT16_MAX__", "65535");
  defineMacro("__UINT16_TYPE__", "unsigned short");
  defineMacro("__UINT32_C_SUFFIX__", "U");
  defineMacro("__UINT32_FMTX__", "\"X\"");
  defineMacro("__UINT32_FMTo__", "\"o\"");
  defineMacro("__UINT32_FMTu__", "\"u\"");
  defineMacro("__UINT32_FMTx__", "\"x\"");
  defineMacro("__UINT32_MAX__", "4294967295U");
  defineMacro("__UINT32_TYPE__", "unsigned int");
  defineMacro("__UINT64_C_SUFFIX__", "UL");
  defineMacro("__UINT64_FMTX__", "\"lX\"");
  defineMacro("__UINT64_FMTo__", "\"lo\"");
  defineMacro("__UINT64_FMTu__", "\"lu\"");
  defineMacro("__UINT64_FMTx__", "\"lx\"");
  defineMacro("__UINT64_MAX__", "18446744073709551615UL");
  defineMacro("__UINT64_TYPE__", "long unsigned int");
  defineMacro("__UINT8_C_SUFFIX__", "");
  defineMacro("__UINT8_FMTX__", "\"hhX\"");
  defineMacro("__UINT8_FMTo__", "\"hho\"");
  defineMacro("__UINT8_FMTu__", "\"hhu\"");
  defineMacro("__UINT8_FMTx__", "\"hhx\"");
  defineMacro("__UINT8_MAX__", "255");
  defineMacro("__UINT8_TYPE__", "unsigned char");
  defineMacro("__UINTMAX_C_SUFFIX__", "UL");
  defineMacro("__UINTMAX_FMTX__", "\"lX\"");
  defineMacro("__UINTMAX_FMTo__", "\"lo\"");
  defineMacro("__UINTMAX_FMTu__", "\"lu\"");
  defineMacro("__UINTMAX_FMTx__", "\"lx\"");
  defineMacro("__UINTMAX_MAX__", "18446744073709551615UL");
  defineMacro("__UINTMAX_TYPE__", "long unsigned int");
  defineMacro("__UINTMAX_WIDTH__", "64");
  defineMacro("__UINTPTR_FMTX__", "\"lX\"");
  defineMacro("__UINTPTR_FMTo__", "\"lo\"");
  defineMacro("__UINTPTR_FMTu__", "\"lu\"");
  defineMacro("__UINTPTR_FMTx__", "\"lx\"");
  defineMacro("__UINTPTR_MAX__", "18446744073709551615UL");
  defineMacro("__UINTPTR_TYPE__", "long unsigned int");
  defineMacro("__UINTPTR_WIDTH__", "64");
  defineMacro("__UINT_FAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_FAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_FAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_FAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_FAST16_MAX__", "65535");
  defineMacro("__UINT_FAST16_TYPE__", "unsigned short");
  defineMacro("__UINT_FAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_FAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_FAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_FAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_FAST32_MAX__", "4294967295U");
  defineMacro("__UINT_FAST32_TYPE__", "unsigned int");
  defineMacro("__UINT_FAST64_FMTX__", "\"lX\"");
  defineMacro("__UINT_FAST64_FMTo__", "\"lo\"");
  defineMacro("__UINT_FAST64_FMTu__", "\"lu\"");
  defineMacro("__UINT_FAST64_FMTx__", "\"lx\"");
  defineMacro("__UINT_FAST64_MAX__", "18446744073709551615UL");
  defineMacro("__UINT_FAST64_TYPE__", "long unsigned int");
  defineMacro("__UINT_FAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_FAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_FAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_FAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_FAST8_MAX__", "255");
  defineMacro("__UINT_FAST8_TYPE__", "unsigned char");
  defineMacro("__UINT_LEAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_LEAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_LEAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_LEAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_LEAST16_MAX__", "65535");
  defineMacro("__UINT_LEAST16_TYPE__", "unsigned short");
  defineMacro("__UINT_LEAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_LEAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_LEAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_LEAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_LEAST32_MAX__", "4294967295U");
  defineMacro("__UINT_LEAST32_TYPE__", "unsigned int");
  defineMacro("__UINT_LEAST64_FMTX__", "\"lX\"");
  defineMacro("__UINT_LEAST64_FMTo__", "\"lo\"");
  defineMacro("__UINT_LEAST64_FMTu__", "\"lu\"");
  defineMacro("__UINT_LEAST64_FMTx__", "\"lx\"");
  defineMacro("__UINT_LEAST64_MAX__", "18446744073709551615UL");
  defineMacro("__UINT_LEAST64_TYPE__", "long unsigned int");
  defineMacro("__UINT_LEAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_LEAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_LEAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_LEAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_LEAST8_MAX__", "255");
  defineMacro("__UINT_LEAST8_TYPE__", "unsigned char");
  defineMacro("__USER_LABEL_PREFIX__", "");
  defineMacro("__VERSION__", "\"Clang 10.0.0 \"");
  defineMacro("__WCHAR_MAX__", "2147483647");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__WINT_MAX__", "4294967295U");
  defineMacro("__WINT_TYPE__", "unsigned int");
  defineMacro("__WINT_UNSIGNED__", "1");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__amd64", "1");
  defineMacro("__amd64__", "1");
  defineMacro("__clang__", "1");
  defineMacro("__clang_major__", "10");
  defineMacro("__clang_minor__", "0");
  defineMacro("__clang_patchlevel__", "0");
  defineMacro("__clang_version__", "\"10.0.0 \"");
  defineMacro("__code_model_small_", "1");
  defineMacro("__cplusplus", "202002L");
  defineMacro("__cpp_aggregate_bases", "201603L");
  defineMacro("__cpp_aggregate_nsdmi", "201304L");
  defineMacro("__cpp_alias_templates", "200704L");
  defineMacro("__cpp_aligned_new", "201606L");
  defineMacro("__cpp_attributes", "200809L");
  defineMacro("__cpp_binary_literals", "201304L");
  defineMacro("__cpp_capture_star_this", "201603L");
  defineMacro("__cpp_char8_t", "201811L");
  defineMacro("__cpp_concepts", "201907L");
  defineMacro("__cpp_conditional_explicit", "201806L");
  defineMacro("__cpp_constexpr", "201907L");
  defineMacro("__cpp_constexpr_dynamic_alloc", "201907L");
  defineMacro("__cpp_constexpr_in_decltype", "201711L");
  defineMacro("__cpp_constinit", "201907L");
  defineMacro("__cpp_coroutines", "201703L");
  defineMacro("__cpp_decltype", "200707L");
  defineMacro("__cpp_decltype_auto", "201304L");
  defineMacro("__cpp_deduction_guides", "201703L");
  defineMacro("__cpp_delegating_constructors", "200604L");
  defineMacro("__cpp_designated_initializers", "201707L");
  defineMacro("__cpp_digit_separators", "201309L");
  defineMacro("__cpp_enumerator_attributes", "201411L");
  defineMacro("__cpp_exceptions", "199711L");
  defineMacro("__cpp_fold_expressions", "201603L");
  defineMacro("__cpp_generic_lambdas", "201707L");
  defineMacro("__cpp_guaranteed_copy_elision", "201606L");
  defineMacro("__cpp_hex_float", "201603L");
  defineMacro("__cpp_if_constexpr", "201606L");
  defineMacro("__cpp_impl_destroying_delete", "201806L");
  defineMacro("__cpp_impl_three_way_comparison", "201907L");
  defineMacro("__cpp_inheriting_constructors", "201511L");
  defineMacro("__cpp_init_captures", "201803L");
  defineMacro("__cpp_initializer_lists", "200806L");
  defineMacro("__cpp_inline_variables", "201606L");
  defineMacro("__cpp_lambdas", "200907L");
  defineMacro("__cpp_namespace_attributes", "201411L");
  defineMacro("__cpp_nested_namespace_definitions", "201411L");
  defineMacro("__cpp_noexcept_function_type", "201510L");
  defineMacro("__cpp_nontype_template_args", "201411L");
  defineMacro("__cpp_nontype_template_parameter_auto", "201606L");
  defineMacro("__cpp_nsdmi", "200809L");
  defineMacro("__cpp_range_based_for", "201603L");
  defineMacro("__cpp_raw_strings", "200710L");
  defineMacro("__cpp_ref_qualifiers", "200710L");
  defineMacro("__cpp_return_type_deduction", "201304L");
  defineMacro("__cpp_rtti", "199711L");
  defineMacro("__cpp_rvalue_references", "200610L");
  defineMacro("__cpp_static_assert", "201411L");
  defineMacro("__cpp_structured_bindings", "201606L");
  defineMacro("__cpp_template_auto", "201606L");
  defineMacro("__cpp_threadsafe_static_init", "200806L");
  defineMacro("__cpp_unicode_characters", "200704L");
  defineMacro("__cpp_unicode_literals", "200710L");
  defineMacro("__cpp_user_defined_literals", "200809L");
  defineMacro("__cpp_variable_templates", "201304L");
  defineMacro("__cpp_variadic_templates", "200704L");
  defineMacro("__cpp_variadic_using", "201611L");
  defineMacro("__gnu_linux__", "1");
  defineMacro("__k8", "1");
  defineMacro("__k8__", "1");
  defineMacro("__linux", "1");
  defineMacro("__linux__", "1");
  defineMacro("__llvm__", "1");
  defineMacro("__private_extern__", "extern");
  defineMacro("__seg_fs", "__attribute__((address_space(257)))");
  defineMacro("__seg_gs", "__attribute__((address_space(256)))");
  defineMacro("__tune_k8__", "1");
  defineMacro("__unix", "1");
  defineMacro("__unix__", "1");
  defineMacro("__x86_64", "1");
  defineMacro("__x86_64__", "1");
  // clang-format on
}

}  // namespace cxx
