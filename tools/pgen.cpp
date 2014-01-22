// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <sstream>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <cstring>

namespace {
std::set<std::string> non_terminals;
std::set<std::string> terminals;
std::set<std::string> names;
std::string parser_name;
std::string input_file;
std::set<std::string> nullable_non_terminals;
}

std::string token_name(const std::string& t) {
  std::string name = "T_";
  for (auto&& x: t)
    name += std::toupper(x);
  return name;
}

std::string read_file(const std::string& filename) {
  char buffer[4 * 1024];
  std::string code;
  std::ifstream in(filename);
  while (in) {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  }
  return code;
}

std::string indent(const std::string& text) {
  std::string prefix;
  size_t i{0};
  for (; i < text.size(); ++i) {
    if (text[i] != '\n')
      break;
  }
  for (; i < text.size(); ++i) {
    if (text[i] == '\n' || !std::isspace(text[i]))
      break;
    prefix += text[i];
  }
  std::string out;
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '\n' && text.compare(i + 1, prefix.size(), prefix) == 0) {
      i += prefix.size() + 1;
      out += "\n\t";
    } else {
      out += text[i++];
    }
  }
  return out;
}

enum struct TOK {
  EOF_SYMBOL,
  ERROR,
  IDENTIFIER,
  BAR,
  EXTRA,
  CODE,
  COLON,
  LPAREN,
  PLUS,
  POUND,
  QUESTION,
  COMMA,
  RPAREN,
  SEMICOLON,
  SLASH,
  STAR,
  TEXT,
  EXTERN,
  TOKEN,
  CLASS,
};

struct Token {
  TOK kind{TOK::EOF_SYMBOL};
  int line{0};
  std::string text;
  bool is(TOK k) const { return kind == k; }
  bool isNot(TOK k) const { return kind != k; }
  Token() = default;
  Token(Token&& other) { *this = std::move(other); }
  Token& operator=(Token&& other) {
    kind = other.kind;
    line = other.line;
    std::swap(text, other.text);
    return *this;
  }
};

std::ostream& operator<<(std::ostream& out, const Token& tk) {
  return out << tk.text;
}

class Lexer {
  using buffer_type = std::string;
  using iterator = buffer_type::const_iterator;
  buffer_type yycode;
  iterator yypos;
  std::string yytext;
  char yychar{'\n'};
  unsigned yylineno{1};
  unsigned yytokenlineno{1};
  inline void yyinp() {
    if (yypos != yycode.cend()) {
      yychar = *yypos++;
      if (yychar == '\n')
        ++yylineno;
    } else {
      yychar = char();
    }
  }
  TOK yylex0() {
  again:
    while (isspace(yychar))
      yyinp();
    yytokenlineno = yylineno;
    yytext.clear();
    if (! yychar)
      return TOK::EOF_SYMBOL;
    auto ch = yychar;
    yytext += ch;
    yyinp();
    if (ch == '%' && yychar == '{') {
      yyinp();
      skip('{', '{', '}');
      yytext = std::string(yytext, 2, yytext.length() - 4);
      return TOK::TEXT;
    }
    if (ch == '%' && isalpha(yychar)) {
      while (isalnum(yychar) || yychar == '_') {
        yytext += yychar;
        yyinp();
      }
      if (yytext == "%extern") return TOK::EXTERN;
      if (yytext == "%token") return TOK::TOKEN;
      if (yytext == "%class") return TOK::CLASS;
      return TOK::ERROR;
    }
    if (ch == '/') {
      if (yychar == '/') {
        for (; yychar; yyinp()) {
          if (yychar == '\n')
            break;
        }
        goto again;
      }
      if (yychar == '*') {
        yyinp();
        while (yychar) {
          auto ch = yychar;
          yyinp();
          if (ch == '*' && yychar == '/') {
            yyinp();
            break;
          }
        }
        goto again;
      }
      return TOK::SLASH;
    }
    if (ch == '{') {
      skip(ch, '{', '}');
      return TOK::CODE;
    }
    if (ch == '(') {
      auto pos = yypos - 2;
      assert(*pos == '(');
      if (pos != yycode.cbegin() && (isalnum(pos[-1]) || pos[-1] == '_')) {
        skip(ch, '(', ')');
        yytext = std::string(yytext, 1, yytext.length() - 2);
        return TOK::EXTRA;
      }
      return TOK::LPAREN;
    }
    if (ch == '#') return TOK::POUND;
    if (ch == ')') return TOK::RPAREN;
    if (ch == '*') return TOK::STAR;
    if (ch == '+') return TOK::PLUS;
    if (ch == ',') return TOK::COMMA;
    if (ch == ':') return TOK::COLON;
    if (ch == ';') return TOK::SEMICOLON;
    if (ch == '?') return TOK::QUESTION;
    if (ch == '|') return TOK::BAR;
    if (isalpha(ch) || ch == '_') {
      while (isalnum(yychar) || yychar == '_') {
        yytext += yychar;
        yyinp();
      }
      return TOK::IDENTIFIER;
    }
    return TOK::ERROR;
  }
  void skip(char ch, char left, char right) {
    if (ch == left) {
      int count = 1;
      auto yynext = [this] { yytext += yychar; yyinp(); };
      while (auto ch = yychar) {
        yynext();
        if (ch == '/' && yychar == '/') {
          while (yychar && yychar != '\n')
            yynext();
          continue;
        }
        if (ch == '/' && yychar == '*') {
          yynext();
          while (auto ch = yychar) {
            yynext();
            if (ch == '*' && yychar == '/') {
              yynext();
              break;
            }
          }
          continue;
        }
        if (ch == left) {
          ++count;
        } else if (ch == right) {
          if (! --count)
            break;
        } if (ch == '"' || ch == '\'') {
          auto quote = ch;
          while (auto ch = yychar) {
            yynext();
            if (ch == quote)
              break;
            if (ch == '\\' && yychar)
              yynext();
          }
        }
      }
    }
  }
  Lexer(std::string&& code): yycode(std::move(code)) { yypos = yycode.cbegin(); }
  Token operator()() {
    Token tk;
    tk.kind = yylex0();
    tk.line = yytokenlineno;
    std::swap(tk.text, yytext);
    return tk;
  }
public:
  static std::vector<Token> tokenize(std::string&& code,
                                     std::vector<Token>* verbatim = 0) {
    Lexer yylex(std::move(code));
    std::vector<Token> tokens;
    Token tk;
    do {
      tk = yylex();
      if (tk.is(TOK::TEXT)) {
        if (verbatim)
          verbatim->push_back(std::move(tk));
        continue;
      }
      tokens.push_back(std::move(tk));
    } while (tk.isNot(TOK::EOF_SYMBOL));
    return tokens;
  }
};

struct Item;
struct Symbol;
struct Lookahead;
struct Code;
template <typename> struct Postfix;
template <typename> struct Pair;

struct And;
struct Or;
struct Plus;
struct Star;
struct Question;

struct ItemVisitor {
  virtual ~ItemVisitor() = default;
  virtual void visit(Symbol*) = 0;
  virtual void visit(Code*) = 0;
  virtual void visit(Pair<And>*) = 0;
  virtual void visit(Pair<Or>*) = 0;
  virtual void visit(Pair<Lookahead>*) = 0;
  virtual void visit(Postfix<Plus>*) = 0;
  virtual void visit(Postfix<Star>*) = 0;
  virtual void visit(Postfix<Question>*) = 0;
};

struct Item {
  bool nullable{false};
  virtual ~Item() = default;
  virtual void accept(ItemVisitor*) = 0;
  virtual Symbol* asSymbol() { return 0; }
};

template <typename>
struct Postfix final: Item {
  Item* item;
  Postfix(Item* item): item(item) { assert(item); }
  void accept(ItemVisitor* v) override { v->visit(this); }
};

template <typename>
struct Pair final: Item {
  Item* head;
  Item* tail;
  Pair(Item* head, Item* tail)
    : head(head), tail(tail) { assert(head); assert(tail); }
  void accept(ItemVisitor* v) override { v->visit(this); }
};

struct Symbol final: Item {
  std::string name;
  std::string extra;
  int line;
  Symbol(const std::string& name, const std::string& extra, int line)
    : name(name), extra(extra), line(line) {}
  void accept(ItemVisitor* v) override { v->visit(this); }
  Symbol* asSymbol() override { return this; }
  bool isTerminal() const { return terminals.find(name) != terminals.end(); }
};

struct Code final: Item {
  std::string text;
  int line;
  Code(const std::string& text, int line)
    : text(text), line(line) {}
  void accept(ItemVisitor* v) override { v->visit(this); }
};

struct Rule final {
  int line{0};
  std::string lhs;
  std::string extra;
  std::string init;
  Item* def{nullptr};
};

class Parser {
  std::string yyfilename;
  std::vector<Token> yytokens;
  using iterator = decltype(yytokens)::const_iterator;
  iterator yytoken;
  bool yymatch(TOK tk) {
    if (yytoken->isNot(tk))
      return false;
    ++yytoken;
    return true;
  }
  iterator yyexpect(TOK tk) {
    if (yytoken->isNot(tk)) {
      std::cerr << yyfilename << ":" << yytoken->line
                << ": error: unexpected token `" << *yytoken << "'"
                << std::endl;
      exit(EXIT_FAILURE);
      return yytokens.end();
    }
    return yytoken++;
  }
  iterator yyconsume() {
    return yytoken++;
  }
public:
  std::map<std::string, std::string> externals;
  Parser(const std::string& yyfilename, std::vector<Token>&& tokens)
    : yyfilename(yyfilename), yytokens(std::move(tokens)) {
    yytoken = yytokens.cbegin();
  }
  std::vector<Rule*> operator()() {
    parseDirectives();
    return parseRules();
  }
  void parseDirectives() {
    while (true) {
      if (yymatch(TOK::EXTERN))
        parseExternSymbols();
      else if (yymatch(TOK::TOKEN))
        parseTokens();
      else if (yymatch(TOK::CLASS))
        parseClass();
      else
        break;
    }
  }
  void parseTokens() {
    do {
      auto sym = yyexpect(TOK::IDENTIFIER);
      terminals.insert(sym->text);
    } while (yymatch(TOK::COMMA));
  }
  void parseClass() {
    auto name = yyexpect(TOK::IDENTIFIER);
    parser_name = name->text;
  }
  void parseExternSymbols() {
    do {
      auto sym = yyexpect(TOK::IDENTIFIER);
      if (yytoken->is(TOK::EXTRA)) {
        externals.insert(std::make_pair(sym->text, yytoken->text));
        ++yytoken;
      } else {
        externals.insert(std::make_pair(sym->text, std::string()));
      }
    } while (yymatch(TOK::COMMA));
  }
  std::vector<Rule*> parseRules() {
    std::vector<Rule*> rules;
    while (yytoken->is(TOK::IDENTIFIER))
      rules.push_back(parseRule());
    return rules;
  }
  Rule* parseRule() {
    auto rule = new Rule();
    rule->line = yytoken->line;
    rule->lhs = yyexpect(TOK::IDENTIFIER)->text;
    if (yytoken->is(TOK::EXTRA)) {
      rule->line = yytoken->line;
      rule->extra = yytoken->text;
      ++yytoken;
    }
    if (yytoken->is(TOK::CODE)) {
      rule->init = yytoken->text;
      ++yytoken;
    }
    yyexpect(TOK::COLON);
    if (yytoken->isNot(TOK::SEMICOLON))
      rule->def = parseDefinition();
    yyexpect(TOK::SEMICOLON);
    return rule;
  }
  Item* parseDefinition() {
    return parseOr();
  }
  Item* parseOr() {
    auto item = parseAnd();
    if (yymatch(TOK::BAR))
      item = new Pair<Or>(item, parseDefinition());
    return item;
  }
  Item* parseAnd() {
    auto item = parsePostfix();
    if (lookAtItem())
      item = new Pair<And>(item, parseAnd());
    return item;
  }
  Item* parsePostfix() {
    auto item = parsePrimary();
    if (! item) {
      return 0;
    } else if (yymatch(TOK::QUESTION)) {
      return new Postfix<Question>(item);
    } else if (yymatch(TOK::PLUS)) {
      return new Postfix<Plus>(item);
    } else if (yymatch(TOK::STAR)) {
      return new Postfix<Star>(item);
    } else if (yymatch(TOK::SLASH)) {
      auto la = parsePrimary();
      item = new Pair<Lookahead>(item, la);
    }
    return item;
  }
  Item* parseId() {
    if (yytoken->is(TOK::CODE)) {
      auto code = new Code(yytoken->text, yytoken->line);
      ++yytoken;
      return code;
    }
    auto id = yyexpect(TOK::IDENTIFIER);
    std::string extra;
    if (yytoken->is(TOK::EXTRA)) {
      extra = yytoken->text;
      ++yytoken;
    }
    return new Symbol(id->text, extra, id->line);
  }
  Item* parsePrimary() {
    if (yymatch(TOK::LPAREN)) {
      auto item = parseOr();
      yyexpect(TOK::RPAREN);
      return item;
    }
    return parseId();
  }
  bool lookAtItem() {
    if (yytoken->is(TOK::EOF_SYMBOL))
      return false;
    if (yytoken->is(TOK::SEMICOLON))
      return false;
    if (yytoken->is(TOK::BAR))
      return false;
    if (yytoken->is(TOK::RPAREN))
      return false;
    return true;
  }
};

namespace {
std::vector<Rule*> all_rules;
std::vector<std::pair<std::string, std::string>> all_externals;
std::set<std::string> undef;
std::vector<Token> verbatim;
}

namespace IR {
struct Function;
struct BasicBlock;
// statements
struct Stmt;
struct Exp;
struct Code;
struct Move;
struct Save;
struct Restore;
struct Ret;
struct Jump;
struct CJump;
// expressions
struct Expr;
struct Name;
struct Temp;

struct StmtVisitor {
  virtual ~StmtVisitor() = default;
  virtual void visit(Exp*) = 0;
  virtual void visit(Move*) = 0;
  virtual void visit(Save*) = 0;
  virtual void visit(Restore*) = 0;
  virtual void visit(Ret*) = 0;
  virtual void visit(Jump*) = 0;
  virtual void visit(CJump*) = 0;
};

struct ExprVisitor {
  virtual ~ExprVisitor() = default;
  virtual void visit(Name*) = 0;
  virtual void visit(Temp*) = 0;
  virtual void visit(Code*) = 0;
};

struct Expr {
  virtual ~Expr() = default;
  virtual void accept(ExprVisitor*) = 0;
  virtual Name* asName() { return 0; }
  virtual Temp* asTemp() { return 0; }
  virtual Code* asCode() { return 0; }
};

struct Stmt {
  virtual ~Stmt() = default;
  virtual void accept(StmtVisitor*) = 0;
  virtual bool isTerminator() const { return false; }
};

struct Name final: Expr {
  Symbol* sym;
  Name(Symbol* sym): sym(sym) {}
  void accept(ExprVisitor* v) override { v->visit(this); }
  Name* asName() override final { return this; }
};

struct Temp final: Expr {
  std::string ty;
  std::string id;
  Temp(const std::string& type, int index)
    : ty(type), id("yy" + std::to_string(index)) {}
  Temp(const std::string& type, const std::string& id)
    : ty(type), id(id) {}
  void accept(ExprVisitor* v) override { v->visit(this); }
  Temp* asTemp() override final { return this; }
  std::string name() { return id; }
  std::string type() const {
    if (ty.empty())
      return "unsigned";
    return ty;
  }
};

struct Code final: Expr {
  std::string text;
  int line;
  Code(const std::string& text, int line = -1): text(text), line(line) {}
  void accept(ExprVisitor* v) override { v->visit(this); }
  Code* asCode() override { return this; }
};

struct Exp final: Stmt {
  Expr* expr;
  Exp(Expr* expr): expr(expr) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
};

struct Move final: Stmt {
  Expr* target;
  Expr* source;
  Move(Expr* target, Expr* source)
    : target(target), source(source) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
};

struct Save final: Stmt {
  Temp* target;
  Save(Temp* target): target(target) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
};

struct Restore final: Stmt {
  Temp* source;
  Restore(Temp* source): source(source) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
};

struct Ret final: Stmt {
  bool result;
  Ret(bool result): result(result) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
  bool isTerminator() const override { return true; }
};

struct Jump final: Stmt {
  BasicBlock* target;
  Jump(BasicBlock* target): target(target) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
  bool isTerminator() const override { return true; }
};

struct CJump final: Stmt {
  Expr* cond;
  BasicBlock* iftrue;
  BasicBlock* iffalse;
  CJump(Expr* cond, BasicBlock* iftrue, BasicBlock* iffalse)
    : cond(cond), iftrue(iftrue), iffalse(iffalse) {}
  void accept(StmtVisitor* v) override { v->visit(this); }
  bool isTerminator() const override { return true; }
};

struct BasicBlock: std::vector<Stmt*> {
  Function* function;
  int index{-1};
  BasicBlock(Function* function): function(function) {}
  bool isTerminated() const { return terminator() != 0; }
  Stmt* terminator() const {
    if (empty())
      return 0;
    if (back()->isTerminator())
      return back();
    return 0;
  }
  Name* NAME(Symbol* sym) { return new Name(sym); }
  Code* CODE(const std::string& text, int line = -1) { return new Code(text, line); }
  void EXP(Expr* expr) {
    if (isTerminated())
      return;
    push_back(new Exp(expr));
  }
  void EXP_CODE(const std::string& text, int line = -1) {
    if (isTerminated())
      return;
    push_back(new Exp(new Code(text, line)));
  }
  void SAVE(Temp* target) {
    if (isTerminated())
      return;
    push_back(new Save(target));
  }
  void RESTORE(Temp* source) {
    if (isTerminated())
      return;
    push_back(new Restore(source));
  }
  void MOVE(Expr* target, Expr* source) {
    if (isTerminated())
      return;
    push_back(new Move(target, source));
  }
  void JUMP(BasicBlock* target) {
    if (isTerminated())
      return;
    push_back(new Jump(target));
  }
  void CJUMP(Expr* cond, BasicBlock* iftrue, BasicBlock* iffalse) {
    if (isTerminated())
      return;
    push_back(new CJump(cond, iftrue, iffalse));
  }
  void RET(bool result) {
    if (isTerminated())
      return;
    push_back(new Ret(result));
  }
};

class Print final: ExprVisitor, StmtVisitor {
  std::ostream& out;
  IR::BasicBlock* nextBlock{nullptr};
  bool lookat{false};

  void outCode(Code* c) {
    out << "([&]() -> bool {" << indent(c->text) << " return true; })()";
  }
  void visit(Exp* e) override {
    if (auto c = e->expr->asCode()) {
      out << '\t' << c->text << ';' << std::endl;
    } else {
      assert(!"unreachable");
    }
  }
  void visit(Code*) override {
    assert(!"unreachable");
  }
  void visit(Move* m) override {
    m->target->accept(this);
    out << " = ";
    m->source->accept(this);
    out << ';' << std::endl;
  }
  void visit(Save* s) override {
    out << "\t";
    s->target->accept(this);
    out << " = yycursor;" << std::endl;
  }
  void visit(Restore* s) override {
    out << "\tif (";
    s->source->accept(this);
    out << " > yyparsed) yyparsed = ";
    s->source->accept(this);
    out << ";" << std::endl;
    out << "\tyycursor = ";
    s->source->accept(this);
    out << ";" << std::endl;
  }
  void visit(Ret* r) override {
    if (r->result) {
      out << "\treturn true;" << std::endl;
      return;
    }
    if (! lookat)
      out << "\tyycursor = yyparsed;" << std::endl;
    out << "\treturn false;" << std::endl;
  }
  void visit(Jump* j) override {
    if (nextBlock == j->target) {
      out << std::endl;
      return;
    }
    out << "\tgoto L" << j->target->index << ';' << std::endl;
  }

  void outCJump(CJump* cj, Name* name) {
    IR::BasicBlock* target = cj->iftrue;
    IR::BasicBlock* cont = cj->iffalse;
    std::string unop = "";
    std::string binop = "==";

    if (nextBlock == cj->iftrue) {
      std::swap(target, cont);
      unop = "!";
      binop = "!=";
    }

    if (! lookat)
      out << std::endl << "#line " << name->sym->line << " \"" << input_file << "\"" << std::endl;
    out << "\tif (";
    auto&& id = name->sym->name;
    if (terminals.find(id) != terminals.end())
      out << "yytoken() " << binop << " " << token_name(id);
    else if (lookat)
      out << unop << "lookat_" << id << "()";
    else {
      out << unop << "(lookat_" << id << "() && ";
      out << "parse_" << id << "(" << name->sym->extra << "))";
    }
    out << ") goto L" << target->index << ';' << std::endl;
    if (cont != nextBlock)
      out << "\tgoto L" << cont->index << ';' << std::endl;
  }
  void outCJump(CJump* cj, Code* c) {
    IR::BasicBlock* target = cj->iftrue;
    IR::BasicBlock* cont = cj->iffalse;
    std::string unop = "";

    if (nextBlock == cj->iftrue) {
      std::swap(target, cont);
      unop = "!";
    }

    if (! lookat)
      out << std::endl << "#line " << c->line << " \"" << input_file << "\"" << std::endl;
    out << "\tif (";
    out << unop;
    outCode(c);
    out << ") goto L" << target->index << ';' << std::endl;
    if (cont != nextBlock)
      out << "\tgoto L" << cont->index << ';' << std::endl;
  }

  void visit(CJump* cj) override {
    if (auto name = cj->cond->asName()) {
      outCJump(cj, name);
    } else if (auto c = cj->cond->asCode()) {
      outCJump(cj, c);
    } else {
      assert(!"unreachable");
    }
  }
  void visit(Name*) override {
    assert(!"unreachable");
  }
  void visit(Temp* temp) override {
    out << temp->name();
  }
public:
  Print(std::ostream& out, bool lookat = false)
    : out(out), lookat(lookat) {}
  void operator()(Stmt* s) {
    s->accept(this);
  }
  void setNextBasicBlock(IR::BasicBlock* block) {
    nextBlock = block;
  }
};

struct Function: std::vector<BasicBlock*> {
  std::vector<Temp*> temps;
  Temp* newTemp(const std::string& type = std::string()) {
    auto index = temps.size();
    auto t = new Temp(type, index);
    temps.push_back(t);
    return t;
  }
  Temp* newTemp(const std::string& type, const std::string& name) {
    auto t = new Temp(type, name);
    temps.push_back(t);
    return t;
  }
  BasicBlock* newBasicBlock() {
    auto block = new BasicBlock(this);
    return block;
  }
  void place(BasicBlock* block) {
    assert(block->index == -1);
    block->index = size();
    push_back(block);
  }
  void print(std::ostream& out, bool lookat = false) {
    Print print{out, lookat};
    for (auto&& block: *this) {
      out << "L" << block->index << ":";
      if (size_t(block->index + 1) != size())
        print.setNextBasicBlock(at(block->index + 1));
      else
        print.setNextBasicBlock(0);
      for (auto&& s: *block)
        print(s);
    }
  }
};
} // ::IR

struct ClassifySymbols final: protected ItemVisitor {
  void operator()(Rule* rule) {
    non_terminals.insert(rule->lhs);
    if (rule->def)
      rule->def->accept(this);
  }
protected:
  void visit(Symbol* sym) override {
    names.insert(sym->name);
  }
  void visit(Code*) override {
  }
  void visit(Pair<And>* p) override {
    p->head->accept(this);
    p->tail->accept(this);
  }
  void visit(Pair<Or>* p) override {
    p->head->accept(this);
    p->tail->accept(this);
  }
  void visit(Pair<Lookahead>* p) override {
    p->head->accept(this);
    p->tail->accept(this);
  }
  void visit(Postfix<Plus>* p) override {
    p->item->accept(this);
  }
  void visit(Postfix<Star>* p) override {
    p->item->accept(this);
  }
  void visit(Postfix<Question>* p) {
    p->item->accept(this);
  }
};

struct Nullables: ItemVisitor {
  void operator()() {
    do {
      changed = false;
      for (auto&& rule: all_rules) {
        rule->def->accept(this);
        if (rule->def->nullable)
          changed |= nullable_non_terminals.insert(rule->lhs).second;
      }
    } while (changed);
  }

private:
  bool changed{false};

  void visit(Symbol* sym) override {
    if (! sym->nullable) {
      if (nullable_non_terminals.find(sym->name) != nullable_non_terminals.end()) {
        sym->nullable = true;
        changed = true;
      }
    }
  }
  void visit(Code* code) override {
    if (! code->nullable) {
      code->nullable = true;
      changed = true;
    }
  }
  void visit(Pair<And>* p) override {
    p->head->accept(this);
    p->tail->accept(this);
    if (! p->nullable) {
      p->nullable = p->head->nullable && p->tail->nullable;
      changed = p->nullable;
    }
  }
  void visit(Pair<Or>* p) override {
    p->head->accept(this);
    p->tail->accept(this);
    if (! p->nullable) {
      p->nullable = p->head->nullable || p->tail->nullable;
      changed = p->nullable;
    }
  }
  void visit(Pair<Lookahead>* p) override {
    p->head->accept(this);
    if (! p->nullable) {
      p->nullable = p->head->nullable;
      changed = p->nullable;
    }
  }
  void visit(Postfix<Plus>* p) override {
    p->item->accept(this);
    if (! p->nullable) {
      p->nullable = p->item->nullable;
      changed = p->nullable;
    }
  }
  void visit(Postfix<Star>* p) override {
    p->item->accept(this);
    if (! p->nullable) {
      p->nullable = true;
      changed = true;
    }
  }
  void visit(Postfix<Question>* p) override {
    p->item->accept(this);
    if (! p->nullable) {
      p->nullable = true;
      changed = true;
    }
  }
};

class CodeGenerator: public ItemVisitor {
protected:
  struct Result {
    IR::BasicBlock* iftrue{nullptr};
    IR::BasicBlock* iffalse{nullptr};
    Result() = default;
    Result(IR::BasicBlock* iftrue, IR::BasicBlock* iffalse)
      : iftrue(iftrue), iffalse(iffalse) {}
  };

  std::ostream& out;
  Result code;
  IR::Function* function{nullptr};
  IR::BasicBlock* block{nullptr};

  IR::Temp* newTemp(const std::string& type = std::string()) {
    return function->newTemp(type);
  }
  IR::Temp* newTemp(const std::string& type, const std::string& name) {
    return function->newTemp(type, name);
  }
  IR::BasicBlock* newBasicBlock() {
    return function->newBasicBlock();
  }
  void place(IR::BasicBlock* block) {
    function->place(block);
    this->block = block;
  }
  void condition(Item* item, IR::BasicBlock* iftrue, IR::BasicBlock* iffalse) {
    Result r(iftrue, iffalse);
    std::swap(code, r);
    item->accept(this);
    std::swap(code, r);
    assert(block->isTerminated());
  }

public:
  CodeGenerator(std::ostream& out): out(out) {}
  virtual ~CodeGenerator() = default;
};

class GenLookatMethod final: CodeGenerator {
  void visit(Symbol* sym) override {
    if (nullable_non_terminals.find(sym->name) != nullable_non_terminals.end())
      block->JUMP(code.iftrue);
    else
      block->CJUMP(block->NAME(sym), code.iftrue, code.iffalse);
  }
  void visit(Code*) override {
    block->JUMP(code.iftrue);
  }
  void visit(Pair<And>* p) override {
    if (p->nullable) {
      block->JUMP(code.iftrue);
    } else if (p->head->nullable) {
      auto iffalse = newBasicBlock();
      condition(p->head, code.iftrue, iffalse);
      place(iffalse);
      condition(p->tail, code.iftrue, code.iffalse);
    } else {
      condition(p->head, code.iftrue, code.iffalse); // ### TODO: follows
    }
  }
  void visit(Pair<Or>* p) override {
    if (p->nullable) {
      block->JUMP(code.iftrue);
    } else {
      auto iffalse = newBasicBlock();
      condition(p->head, code.iftrue, iffalse);
      place(iffalse);
      condition(p->tail, code.iftrue, code.iffalse);
    }
  }
  void visit(Pair<Lookahead>* item) override {
    condition(item->head, code.iftrue, code.iffalse);
  }
  void visit(Postfix<Plus>* item) override {
    condition(item->item, code.iftrue, code.iffalse);
  }
  void visit(Postfix<Star>*) override {
    block->JUMP(code.iftrue);
  }
  void visit(Postfix<Question>*) override {
    block->JUMP(code.iftrue);
  }
public:
   GenLookatMethod(std::ostream& out)
     : CodeGenerator(out) {}

   void operator()(Rule* rule) {
     IR::Function f;
     function = &f;

     auto entryBlock = newBasicBlock();
     auto iftrue = newBasicBlock();
     auto iffalse = newBasicBlock();

     place(entryBlock);
     condition(rule->def, iftrue, iffalse);
     place(iffalse);
     block->RET(false);
     place(iftrue);
     block->RET(true);

     out << std::endl;
     out << "inline bool " << parser_name << "::lookat_" << rule->lhs << "() {" << std::endl;
     std::set<std::string> p;
     for (auto&& temp: function->temps) {
       if (! p.insert(temp->name()).second)
         continue;
       out << "\t" << temp->type() << " " << temp->name() << ";" << std::endl;
     }
     out << "\tgoto L0;" << std::endl;
     f.print(out, /*lookat=*/ true);
     out << "}" << std::endl;
   }
};

class GenParseMethod final: CodeGenerator {
  void visit(Symbol* sym) override {
    if (sym->isTerminal()) {
      auto iftrue = newBasicBlock();
      block->CJUMP(block->NAME(sym), iftrue, code.iffalse);
      place(iftrue);
      block->EXP_CODE("++yycursor");
      block->JUMP(code.iftrue);
      return;
    }
    block->CJUMP(block->NAME(sym), code.iftrue, code.iffalse);
  }
  void visit(Code* c) override {
    block->CJUMP(block->CODE(c->text, c->line), code.iftrue, code.iffalse);
  }
  void visit(Pair<Lookahead>* p) override {
    auto iftrue = newBasicBlock();
    auto cleanup_iftrue = newBasicBlock();
    auto cleanup_iffalse = newBasicBlock();

    condition(p->head, iftrue, code.iffalse);

    auto target = newTemp();
    place(iftrue);
    block->SAVE(target);
    condition(p->tail, cleanup_iftrue, cleanup_iffalse);

    place(cleanup_iftrue);
    block->RESTORE(target);
    block->JUMP(code.iftrue);

    place(cleanup_iffalse);
    block->RESTORE(target);
    block->JUMP(code.iffalse);
  }
  void visit(Pair<And>* p) override {
    auto iftrue = newBasicBlock();
    condition(p->head, iftrue, code.iffalse);
    place(iftrue);
    condition(p->tail, code.iftrue, code.iffalse);
  }
  void visit(Pair<Or>* p) override {
    auto target = newTemp();
    block->SAVE(target);
    auto iffalse = newBasicBlock();
    condition(p->head, code.iftrue, iffalse);
    place(iffalse);
    block->RESTORE(target);
    condition(p->tail, code.iftrue, code.iffalse);
  }
  void visit(Postfix<Plus>* p) override {
    auto more = newBasicBlock();
    condition(p->item, more, code.iffalse);
    place(more);

    auto iftrue = newBasicBlock();
    auto iffalse = newBasicBlock();
    auto target = newTemp();

    block->JUMP(iftrue);
    place(iftrue);
    block->SAVE(target);
    condition(p->item, iftrue, iffalse);
    place(iffalse);
    block->RESTORE(target);
    block->JUMP(code.iftrue);
  }
  void visit(Postfix<Star>* p) override {
    auto iftrue = newBasicBlock();
    auto iffalse = newBasicBlock();
    auto target = newTemp();

    block->JUMP(iftrue);
    place(iftrue);
    block->SAVE(target);
    condition(p->item, iftrue, iffalse);
    place(iffalse);
    block->RESTORE(target);
    block->JUMP(code.iftrue);
  }
  void visit(Postfix<Question>* p) override {
    auto target = newTemp();
    auto iffalse = newBasicBlock();
    block->SAVE(target);
    condition(p->item, code.iftrue, iffalse);
    place(iffalse);
    block->RESTORE(target);
    block->JUMP(code.iftrue);
  }
public:
  GenParseMethod(std::ostream& out)
    : CodeGenerator(out) {}

  void operator()(Rule* rule) {
    IR::Function f;
    function = &f;

    auto entryBlock = newBasicBlock();
    auto iftrue = newBasicBlock();
    auto iffalse = newBasicBlock();

    place(entryBlock);
    auto target = newTemp();
    block->SAVE(target);
    condition(rule->def, iftrue, iffalse);
    place(iffalse);
    block->RET(false);
    place(iftrue);
    block->RET(true);

    out << std::endl;
    out << std::endl << "#line " << rule->line << " \"" << input_file << "\"" << std::endl;
    out << "bool " << parser_name << "::parse_" << rule->lhs << "(";
    out << rule->extra;
    out << ") {" << std::endl;
    std::set<std::string> p;
    if (! rule->init.empty())
      out << "\t" << indent(std::string(rule->init, 1, rule->init.length() - 2)) << std::endl;
    for (auto&& temp: function->temps) {
      if (! p.insert(temp->name()).second)
        continue;
      out << "\t" << temp->type() << " " << temp->name() << ";" << std::endl;
    }
    out << "\tgoto L0;" << std::endl;
    f.print(out);
    out << "}" << std::endl;
  }
};

int main(int argc, char* argv[]) {
  std::vector<std::string> input_files;
  std::string output_file;
  for (auto it = argv + 1; it != argv + argc;) {
    std::string arg(*it++);
    if (arg.empty())
      continue;
    if (arg[0] == '-') {
      if (arg == "-o" && it != argv + argc) {
        output_file = *it++;
        continue;
      }
      std::cerr << "skip unrecognized option " << arg << std::endl;
      continue;
    }
    input_files.push_back(std::move(arg));
  }

  if (input_files.empty()) {
    std::cerr << "pgen: no input files" << std::endl;
    exit(EXIT_FAILURE);
  }
  assert(input_files.size() == 1);
  input_file = input_files.front();

  for (auto&& filename: input_files) {
    Parser parse(filename, Lexer::tokenize(read_file(filename), &verbatim));
    auto&& rules = parse();
    auto&& externals = parse.externals;

    all_rules.insert(all_rules.end(),
                     rules.begin(),
                     rules.end());
    all_externals.insert(all_externals.end(),
                         externals.begin(),
                         externals.end());

    ClassifySymbols classifySymbols;
    for (auto&& rule: rules)
      classifySymbols(rule);
    for (auto&& n: names) {
      if (non_terminals.find(n) != non_terminals.end())
        continue;
      if (externals.find(n) != externals.end())
        continue;
      if (terminals.find(n) != terminals.end())
        continue;
      undef.insert(n);
    }
  }

  if (! undef.empty()) {
    for (auto&& u: undef)
      std::cerr << "undefined symbol `" << u << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (output_file.empty()) {
    std::cerr << "pgen: no output file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (parser_name.empty())
    parser_name = "Parser";

  for (auto it = all_externals.begin(); it != all_externals.end(); ++it)
    nullable_non_terminals.insert(it->first);

  Nullables updateNullableSet;
  updateNullableSet();

  std::ofstream out(output_file);
  out << "struct " << parser_name << " {" << std::endl;
  if (! verbatim.empty()) {
    for (auto&& tk: verbatim) {
      out << tk.text << std::endl;
    }
  }

  for (auto&& decl: all_externals) {
    out << "  inline bool lookat_" << decl.first << "();" << std::endl;
    out << "  bool parse_" << decl.first << "(" << decl.second << ");" << std::endl;
    out << std::endl;
  }
  if (! all_externals.empty())
    std::cout << std::endl;
  for (auto&& rule: all_rules) {
    out << "  inline bool lookat_" << rule->lhs << "();" << std::endl;
    out << "#line " << rule->line << " \"" << input_file << "\"" << std::endl;
    out << "  bool parse_" << rule->lhs << "(" << rule->extra << ");" << std::endl;
  }
  out << std::endl;
  out << "  int yytoken(int index = 0);" << std::endl;
  out << std::endl;
  out << "  unsigned yyparsed{0};" << std::endl;
  out << "  int yydepth{-1};" << std::endl;
  out << "  unsigned yycursor{0};" << std::endl;
  out << "};" << std::endl;

  GenLookatMethod genLookatMethod(out);
  for (auto&& rule: all_rules)
    genLookatMethod(rule);

  GenParseMethod genParseMethod(out);
  for (auto&& rule: all_rules)
    genParseMethod(rule);

  return EXIT_SUCCESS;
}
