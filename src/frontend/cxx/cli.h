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

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace cxx {

struct CLIOption : std::tuple<std::string, std::string> {
  using tuple::tuple;
};

struct CLIPositional : std::tuple<std::string> {
  using tuple::tuple;
};

struct CLIFlag : std::tuple<std::string> {
  using tuple::tuple;
};

using CLIMatch = std::variant<CLIFlag, CLIOption, CLIPositional>;

auto to_string(const CLIMatch& match) -> std::string;

class CLI {
  std::vector<CLIMatch> result_;

 public:
  CLI();

  std::string app_name;
  bool opt_ast_dump = false;
  bool opt_ir_dump = false;
  bool opt_dM = false;
  bool opt_dump_symbols = false;
  bool opt_dump_tokens = false;
  bool opt_E = false;
  bool opt_Eonly = false;
  bool opt_P = false;
  bool opt_H = false;
  bool opt_help = false;
  bool opt_nostdinc = false;
  bool opt_nostdincpp = false;
  bool opt_S = false;
  bool opt_c = false;
  bool opt_fsyntax_only = false;
  bool opt_fstatic_assert = false;
  bool opt_fcheck = false;
  bool opt_ftemplates = false;
  bool opt_fno_reflect = false;
  bool opt_verify = false;
  bool opt_v = false;
  bool opt_emit_ast = false;
  bool opt_lsp = false;

  void parse(int& argc, char**& argv);

  [[nodiscard]] auto count(const std::string& flag) const -> int;
  [[nodiscard]] auto getSingle(const std::string& opt) const
      -> std::optional<std::string>;
  [[nodiscard]] auto get(const std::string& opt) const
      -> std::vector<std::string>;
  [[nodiscard]] auto positionals() const -> std::vector<std::string>;

  void showHelp();

  [[nodiscard]] auto begin() const { return result_.begin(); }
  [[nodiscard]] auto end() const { return result_.end(); }
};

}  // namespace cxx
