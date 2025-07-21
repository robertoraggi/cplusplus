// Generated file by: kwgen.ts
// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

enum class PreprocessorDirectiveKind {
  T_IDENTIFIER,
  T_DEFINE,
  T_ELIF,
  T_ELIFDEF,
  T_ELIFNDEF,
  T_ELSE,
  T_ENDIF,
  T_ERROR,
  T_IF,
  T_IFDEF,
  T_IFNDEF,
  T_INCLUDE_NEXT,
  T_INCLUDE,
  T_LINE,
  T_PRAGMA,
  T_UNDEF,
  T_WARNING,
};

static inline auto classifyDirective2(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'i') {
    if (s[1] == 'f') {
      return PreprocessorDirectiveKind::T_IF;
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective4(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'e') {
    if (s[1] == 'l') {
      if (s[2] == 'i') {
        if (s[3] == 'f') {
          return PreprocessorDirectiveKind::T_ELIF;
        }
      } else if (s[2] == 's') {
        if (s[3] == 'e') {
          return PreprocessorDirectiveKind::T_ELSE;
        }
      }
    }
  } else if (s[0] == 'l') {
    if (s[1] == 'i') {
      if (s[2] == 'n') {
        if (s[3] == 'e') {
          return PreprocessorDirectiveKind::T_LINE;
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective5(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'e') {
    if (s[1] == 'n') {
      if (s[2] == 'd') {
        if (s[3] == 'i') {
          if (s[4] == 'f') {
            return PreprocessorDirectiveKind::T_ENDIF;
          }
        }
      }
    } else if (s[1] == 'r') {
      if (s[2] == 'r') {
        if (s[3] == 'o') {
          if (s[4] == 'r') {
            return PreprocessorDirectiveKind::T_ERROR;
          }
        }
      }
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'f') {
      if (s[2] == 'd') {
        if (s[3] == 'e') {
          if (s[4] == 'f') {
            return PreprocessorDirectiveKind::T_IFDEF;
          }
        }
      }
    }
  } else if (s[0] == 'u') {
    if (s[1] == 'n') {
      if (s[2] == 'd') {
        if (s[3] == 'e') {
          if (s[4] == 'f') {
            return PreprocessorDirectiveKind::T_UNDEF;
          }
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective6(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'd') {
    if (s[1] == 'e') {
      if (s[2] == 'f') {
        if (s[3] == 'i') {
          if (s[4] == 'n') {
            if (s[5] == 'e') {
              return PreprocessorDirectiveKind::T_DEFINE;
            }
          }
        }
      }
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'f') {
      if (s[2] == 'n') {
        if (s[3] == 'd') {
          if (s[4] == 'e') {
            if (s[5] == 'f') {
              return PreprocessorDirectiveKind::T_IFNDEF;
            }
          }
        }
      }
    }
  } else if (s[0] == 'p') {
    if (s[1] == 'r') {
      if (s[2] == 'a') {
        if (s[3] == 'g') {
          if (s[4] == 'm') {
            if (s[5] == 'a') {
              return PreprocessorDirectiveKind::T_PRAGMA;
            }
          }
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective7(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'e') {
    if (s[1] == 'l') {
      if (s[2] == 'i') {
        if (s[3] == 'f') {
          if (s[4] == 'd') {
            if (s[5] == 'e') {
              if (s[6] == 'f') {
                return PreprocessorDirectiveKind::T_ELIFDEF;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'n') {
      if (s[2] == 'c') {
        if (s[3] == 'l') {
          if (s[4] == 'u') {
            if (s[5] == 'd') {
              if (s[6] == 'e') {
                return PreprocessorDirectiveKind::T_INCLUDE;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'w') {
    if (s[1] == 'a') {
      if (s[2] == 'r') {
        if (s[3] == 'n') {
          if (s[4] == 'i') {
            if (s[5] == 'n') {
              if (s[6] == 'g') {
                return PreprocessorDirectiveKind::T_WARNING;
              }
            }
          }
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective8(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'e') {
    if (s[1] == 'l') {
      if (s[2] == 'i') {
        if (s[3] == 'f') {
          if (s[4] == 'n') {
            if (s[5] == 'd') {
              if (s[6] == 'e') {
                if (s[7] == 'f') {
                  return PreprocessorDirectiveKind::T_ELIFNDEF;
                }
              }
            }
          }
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static inline auto classifyDirective12(const char* s)
    -> PreprocessorDirectiveKind {
  if (s[0] == 'i') {
    if (s[1] == 'n') {
      if (s[2] == 'c') {
        if (s[3] == 'l') {
          if (s[4] == 'u') {
            if (s[5] == 'd') {
              if (s[6] == 'e') {
                if (s[7] == '_') {
                  if (s[8] == 'n') {
                    if (s[9] == 'e') {
                      if (s[10] == 'x') {
                        if (s[11] == 't') {
                          return PreprocessorDirectiveKind::T_INCLUDE_NEXT;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return PreprocessorDirectiveKind::T_IDENTIFIER;
}

static auto classifyDirective(const char* s, int n)
    -> PreprocessorDirectiveKind {
  switch (n) {
    case 2:
      return classifyDirective2(s);
    case 4:
      return classifyDirective4(s);
    case 5:
      return classifyDirective5(s);
    case 6:
      return classifyDirective6(s);
    case 7:
      return classifyDirective7(s);
    case 8:
      return classifyDirective8(s);
    case 12:
      return classifyDirective12(s);
    default:
      return PreprocessorDirectiveKind::T_IDENTIFIER;
  }  // switch
}