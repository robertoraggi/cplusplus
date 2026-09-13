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

#pragma once

#include <cxx/token_fwd.h>

#include <string>
#include <vector>

namespace cxx {

struct PreprocessingTokenRecord {
  std::string spelling;
  TokenKind kind = TokenKind::T_EOF_SYMBOL;
  bool startOfLine = false;
  bool leadingSpace = false;
  bool isFromMacroBody = false;
  bool noexpand = false;
};

struct MacroRecord {
  std::string name;
  std::vector<std::string> formals;
  std::vector<PreprocessingTokenRecord> body;
  bool isFunctionLike = false;
  bool isVariadic = false;
};

struct ProtectedFileRecord {
  std::string fileName;
  std::string headerGuardName;
  int headerProtectionLevel = 0;
  bool pragmaOnceProtected = false;
  bool isSystemHeader = false;
};

struct PreprocessorSnapshot {
  std::vector<MacroRecord> macros;
  std::vector<std::string> undefinedBuiltins;
  std::vector<ProtectedFileRecord> protectedFiles;
  std::vector<std::pair<std::string, bool>> includedFiles;
  std::vector<int> packStack;
  std::string date;
  std::string time;
  int counter = 0;
  int currentPack = 0;
};

}  // namespace cxx
