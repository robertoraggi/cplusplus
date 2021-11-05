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

std::string to_string(const CLIMatch& match);

class CLI {
  std::vector<CLIMatch> result_;

 public:
  CLI();

  bool opt_ast_dump = false;
  bool opt_ir_dump = false;
  bool opt_dM = false;
  bool opt_dump_tokens = false;
  bool opt_E = false;
  bool opt_Eonly = false;
  bool opt_help = false;
  bool opt_nostdinc = false;
  bool opt_nostdincpp = false;
  bool opt_S = false;
  bool opt_c = false;
  bool opt_verify = false;
  bool opt_v = false;

  bool checkTypes() const { return opt_ir_dump || opt_S || opt_c; }

  void parse(int& argc, char**& argv);

  int count(const std::string& flag) const;
  std::optional<std::string> getSingle(const std::string& opt) const;
  std::vector<std::string> get(const std::string& opt) const;
  std::vector<std::string> positionals() const;

  void showHelp();

  auto begin() const { return result_.begin(); }
  auto end() const { return result_.end(); }
};

}  // namespace cxx