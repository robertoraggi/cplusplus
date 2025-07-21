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

static inline auto classify2(const char* s) -> cxx::TokenKind {
  if (s[0] == 'd') {
    if (s[1] == 'o') {
      return cxx::TokenKind::T_DO;
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'f') {
      return cxx::TokenKind::T_IF;
    }
  } else if (s[0] == 'o') {
    if (s[1] == 'r') {
      return cxx::TokenKind::T_OR;
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify3(const char* s) -> cxx::TokenKind {
  if (s[0] == 'a') {
    if (s[1] == 's') {
      if (s[2] == 'm') {
        return cxx::TokenKind::T_ASM;
      }
    } else if (s[1] == 'n') {
      if (s[2] == 'd') {
        return cxx::TokenKind::T_AND;
      }
    }
  } else if (s[0] == 'f') {
    if (s[1] == 'o') {
      if (s[2] == 'r') {
        return cxx::TokenKind::T_FOR;
      }
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'n') {
      if (s[2] == 't') {
        return cxx::TokenKind::T_INT;
      }
    }
  } else if (s[0] == 'n') {
    if (s[1] == 'e') {
      if (s[2] == 'w') {
        return cxx::TokenKind::T_NEW;
      }
    } else if (s[1] == 'o') {
      if (s[2] == 't') {
        return cxx::TokenKind::T_NOT;
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'r') {
      if (s[2] == 'y') {
        return cxx::TokenKind::T_TRY;
      }
    }
  } else if (s[0] == 'x') {
    if (s[1] == 'o') {
      if (s[2] == 'r') {
        return cxx::TokenKind::T_XOR;
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify4(const char* s) -> cxx::TokenKind {
  if (s[0] == 'a') {
    if (s[1] == 'u') {
      if (s[2] == 't') {
        if (s[3] == 'o') {
          return cxx::TokenKind::T_AUTO;
        }
      }
    }
  } else if (s[0] == 'b') {
    if (s[1] == 'o') {
      if (s[2] == 'o') {
        if (s[3] == 'l') {
          return cxx::TokenKind::T_BOOL;
        }
      }
    }
  } else if (s[0] == 'c') {
    if (s[1] == 'a') {
      if (s[2] == 's') {
        if (s[3] == 'e') {
          return cxx::TokenKind::T_CASE;
        }
      }
    } else if (s[1] == 'h') {
      if (s[2] == 'a') {
        if (s[3] == 'r') {
          return cxx::TokenKind::T_CHAR;
        }
      }
    }
  } else if (s[0] == 'e') {
    if (s[1] == 'l') {
      if (s[2] == 's') {
        if (s[3] == 'e') {
          return cxx::TokenKind::T_ELSE;
        }
      }
    } else if (s[1] == 'n') {
      if (s[2] == 'u') {
        if (s[3] == 'm') {
          return cxx::TokenKind::T_ENUM;
        }
      }
    }
  } else if (s[0] == 'g') {
    if (s[1] == 'o') {
      if (s[2] == 't') {
        if (s[3] == 'o') {
          return cxx::TokenKind::T_GOTO;
        }
      }
    }
  } else if (s[0] == 'l') {
    if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 'g') {
          return cxx::TokenKind::T_LONG;
        }
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'h') {
      if (s[2] == 'i') {
        if (s[3] == 's') {
          return cxx::TokenKind::T_THIS;
        }
      }
    } else if (s[1] == 'r') {
      if (s[2] == 'u') {
        if (s[3] == 'e') {
          return cxx::TokenKind::T_TRUE;
        }
      }
    }
  } else if (s[0] == 'v') {
    if (s[1] == 'o') {
      if (s[2] == 'i') {
        if (s[3] == 'd') {
          return cxx::TokenKind::T_VOID;
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify5(const char* s) -> cxx::TokenKind {
  if (s[0] == 'b') {
    if (s[1] == 'r') {
      if (s[2] == 'e') {
        if (s[3] == 'a') {
          if (s[4] == 'k') {
            return cxx::TokenKind::T_BREAK;
          }
        }
      }
    } else if (s[1] == 'i') {
      if (s[2] == 't') {
        if (s[3] == 'o') {
          if (s[4] == 'r') {
            return cxx::TokenKind::T_BITOR;
          }
        }
      }
    }
  } else if (s[0] == 'c') {
    if (s[1] == 'a') {
      if (s[2] == 't') {
        if (s[3] == 'c') {
          if (s[4] == 'h') {
            return cxx::TokenKind::T_CATCH;
          }
        }
      }
    } else if (s[1] == 'l') {
      if (s[2] == 'a') {
        if (s[3] == 's') {
          if (s[4] == 's') {
            return cxx::TokenKind::T_CLASS;
          }
        }
      }
    } else if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 's') {
          if (s[4] == 't') {
            return cxx::TokenKind::T_CONST;
          }
        }
      } else if (s[2] == 'm') {
        if (s[3] == 'p') {
          if (s[4] == 'l') {
            return cxx::TokenKind::T_COMPL;
          }
        }
      }
    }
  } else if (s[0] == 'f') {
    if (s[1] == 'a') {
      if (s[2] == 'l') {
        if (s[3] == 's') {
          if (s[4] == 'e') {
            return cxx::TokenKind::T_FALSE;
          }
        }
      }
    } else if (s[1] == 'l') {
      if (s[2] == 'o') {
        if (s[3] == 'a') {
          if (s[4] == 't') {
            return cxx::TokenKind::T_FLOAT;
          }
        }
      }
    }
  } else if (s[0] == 's') {
    if (s[1] == 'h') {
      if (s[2] == 'o') {
        if (s[3] == 'r') {
          if (s[4] == 't') {
            return cxx::TokenKind::T_SHORT;
          }
        }
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'h') {
      if (s[2] == 'r') {
        if (s[3] == 'o') {
          if (s[4] == 'w') {
            return cxx::TokenKind::T_THROW;
          }
        }
      }
    }
  } else if (s[0] == 'u') {
    if (s[1] == 'n') {
      if (s[2] == 'i') {
        if (s[3] == 'o') {
          if (s[4] == 'n') {
            return cxx::TokenKind::T_UNION;
          }
        }
      }
    } else if (s[1] == 's') {
      if (s[2] == 'i') {
        if (s[3] == 'n') {
          if (s[4] == 'g') {
            return cxx::TokenKind::T_USING;
          }
        }
      }
    }
  } else if (s[0] == 'w') {
    if (s[1] == 'h') {
      if (s[2] == 'i') {
        if (s[3] == 'l') {
          if (s[4] == 'e') {
            return cxx::TokenKind::T_WHILE;
          }
        }
      }
    }
  } else if (s[0] == 'o') {
    if (s[1] == 'r') {
      if (s[2] == '_') {
        if (s[3] == 'e') {
          if (s[4] == 'q') {
            return cxx::TokenKind::T_OR_EQ;
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 's') {
          if (s[4] == 'm') {
            return cxx::TokenKind::T___ASM;
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify6(const char* s) -> cxx::TokenKind {
  if (s[0] == 'd') {
    if (s[1] == 'e') {
      if (s[2] == 'l') {
        if (s[3] == 'e') {
          if (s[4] == 't') {
            if (s[5] == 'e') {
              return cxx::TokenKind::T_DELETE;
            }
          }
        }
      }
    } else if (s[1] == 'o') {
      if (s[2] == 'u') {
        if (s[3] == 'b') {
          if (s[4] == 'l') {
            if (s[5] == 'e') {
              return cxx::TokenKind::T_DOUBLE;
            }
          }
        }
      }
    }
  } else if (s[0] == 'e') {
    if (s[1] == 'x') {
      if (s[2] == 'p') {
        if (s[3] == 'o') {
          if (s[4] == 'r') {
            if (s[5] == 't') {
              return cxx::TokenKind::T_EXPORT;
            }
          }
        }
      } else if (s[2] == 't') {
        if (s[3] == 'e') {
          if (s[4] == 'r') {
            if (s[5] == 'n') {
              return cxx::TokenKind::T_EXTERN;
            }
          }
        }
      }
    }
  } else if (s[0] == 'f') {
    if (s[1] == 'r') {
      if (s[2] == 'i') {
        if (s[3] == 'e') {
          if (s[4] == 'n') {
            if (s[5] == 'd') {
              return cxx::TokenKind::T_FRIEND;
            }
          }
        }
      }
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'n') {
      if (s[2] == 'l') {
        if (s[3] == 'i') {
          if (s[4] == 'n') {
            if (s[5] == 'e') {
              return cxx::TokenKind::T_INLINE;
            }
          }
        }
      }
    }
  } else if (s[0] == 'p') {
    if (s[1] == 'u') {
      if (s[2] == 'b') {
        if (s[3] == 'l') {
          if (s[4] == 'i') {
            if (s[5] == 'c') {
              return cxx::TokenKind::T_PUBLIC;
            }
          }
        }
      }
    }
  } else if (s[0] == 'r') {
    if (s[1] == 'e') {
      if (s[2] == 't') {
        if (s[3] == 'u') {
          if (s[4] == 'r') {
            if (s[5] == 'n') {
              return cxx::TokenKind::T_RETURN;
            }
          }
        }
      }
    }
  } else if (s[0] == 's') {
    if (s[1] == 'i') {
      if (s[2] == 'g') {
        if (s[3] == 'n') {
          if (s[4] == 'e') {
            if (s[5] == 'd') {
              return cxx::TokenKind::T_SIGNED;
            }
          }
        }
      } else if (s[2] == 'z') {
        if (s[3] == 'e') {
          if (s[4] == 'o') {
            if (s[5] == 'f') {
              return cxx::TokenKind::T_SIZEOF;
            }
          }
        }
      }
    } else if (s[1] == 't') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'i') {
            if (s[5] == 'c') {
              return cxx::TokenKind::T_STATIC;
            }
          }
        }
      } else if (s[2] == 'r') {
        if (s[3] == 'u') {
          if (s[4] == 'c') {
            if (s[5] == 't') {
              return cxx::TokenKind::T_STRUCT;
            }
          }
        }
      }
    } else if (s[1] == 'w') {
      if (s[2] == 'i') {
        if (s[3] == 't') {
          if (s[4] == 'c') {
            if (s[5] == 'h') {
              return cxx::TokenKind::T_SWITCH;
            }
          }
        }
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'y') {
      if (s[2] == 'p') {
        if (s[3] == 'e') {
          if (s[4] == 'i') {
            if (s[5] == 'd') {
              return cxx::TokenKind::T_TYPEID;
            }
          }
        }
      }
    }
  } else if (s[0] == 'a') {
    if (s[1] == 'n') {
      if (s[2] == 'd') {
        if (s[3] == '_') {
          if (s[4] == 'e') {
            if (s[5] == 'q') {
              return cxx::TokenKind::T_AND_EQ;
            }
          }
        }
      }
    }
  } else if (s[0] == 'b') {
    if (s[1] == 'i') {
      if (s[2] == 't') {
        if (s[3] == 'a') {
          if (s[4] == 'n') {
            if (s[5] == 'd') {
              return cxx::TokenKind::T_BITAND;
            }
          }
        }
      }
    }
  } else if (s[0] == 'n') {
    if (s[1] == 'o') {
      if (s[2] == 't') {
        if (s[3] == '_') {
          if (s[4] == 'e') {
            if (s[5] == 'q') {
              return cxx::TokenKind::T_NOT_EQ;
            }
          }
        }
      }
    }
  } else if (s[0] == 'x') {
    if (s[1] == 'o') {
      if (s[2] == 'r') {
        if (s[3] == '_') {
          if (s[4] == 'e') {
            if (s[5] == 'q') {
              return cxx::TokenKind::T_XOR_EQ;
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify7(const char* s) -> cxx::TokenKind {
  if (s[0] == 'a') {
    if (s[1] == 'l') {
      if (s[2] == 'i') {
        if (s[3] == 'g') {
          if (s[4] == 'n') {
            if (s[5] == 'a') {
              if (s[6] == 's') {
                return cxx::TokenKind::T_ALIGNAS;
              }
            } else if (s[5] == 'o') {
              if (s[6] == 'f') {
                return cxx::TokenKind::T_ALIGNOF;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'c') {
    if (s[1] == 'h') {
      if (s[2] == 'a') {
        if (s[3] == 'r') {
          if (s[4] == '8') {
            if (s[5] == '_') {
              if (s[6] == 't') {
                return cxx::TokenKind::T_CHAR8_T;
              }
            }
          }
        }
      }
    } else if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 'c') {
          if (s[4] == 'e') {
            if (s[5] == 'p') {
              if (s[6] == 't') {
                return cxx::TokenKind::T_CONCEPT;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'd') {
    if (s[1] == 'e') {
      if (s[2] == 'f') {
        if (s[3] == 'a') {
          if (s[4] == 'u') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                return cxx::TokenKind::T_DEFAULT;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'm') {
    if (s[1] == 'u') {
      if (s[2] == 't') {
        if (s[3] == 'a') {
          if (s[4] == 'b') {
            if (s[5] == 'l') {
              if (s[6] == 'e') {
                return cxx::TokenKind::T_MUTABLE;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'n') {
    if (s[1] == 'u') {
      if (s[2] == 'l') {
        if (s[3] == 'l') {
          if (s[4] == 'p') {
            if (s[5] == 't') {
              if (s[6] == 'r') {
                return cxx::TokenKind::T_NULLPTR;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'p') {
    if (s[1] == 'r') {
      if (s[2] == 'i') {
        if (s[3] == 'v') {
          if (s[4] == 'a') {
            if (s[5] == 't') {
              if (s[6] == 'e') {
                return cxx::TokenKind::T_PRIVATE;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'y') {
      if (s[2] == 'p') {
        if (s[3] == 'e') {
          if (s[4] == 'd') {
            if (s[5] == 'e') {
              if (s[6] == 'f') {
                return cxx::TokenKind::T_TYPEDEF;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'v') {
    if (s[1] == 'i') {
      if (s[2] == 'r') {
        if (s[3] == 't') {
          if (s[4] == 'u') {
            if (s[5] == 'a') {
              if (s[6] == 'l') {
                return cxx::TokenKind::T_VIRTUAL;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'w') {
    if (s[1] == 'c') {
      if (s[2] == 'h') {
        if (s[3] == 'a') {
          if (s[4] == 'r') {
            if (s[5] == '_') {
              if (s[6] == 't') {
                return cxx::TokenKind::T_WCHAR_T;
              }
            }
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'i') {
        if (s[3] == 'n') {
          if (s[4] == 't') {
            if (s[5] == '6') {
              if (s[6] == '4') {
                return cxx::TokenKind::T___INT64;
              }
            }
          }
        }
      } else if (s[2] == 'a') {
        if (s[3] == 's') {
          if (s[4] == 'm') {
            if (s[5] == '_') {
              if (s[6] == '_') {
                return cxx::TokenKind::T___ASM__;
              }
            }
          }
        }
      }
    } else if (s[1] == 'A') {
      if (s[2] == 't') {
        if (s[3] == 'o') {
          if (s[4] == 'm') {
            if (s[5] == 'i') {
              if (s[6] == 'c') {
                return cxx::TokenKind::T__ATOMIC;
              }
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify8(const char* s) -> cxx::TokenKind {
  if (s[0] == 'c') {
    if (s[1] == 'h') {
      if (s[2] == 'a') {
        if (s[3] == 'r') {
          if (s[4] == '1') {
            if (s[5] == '6') {
              if (s[6] == '_') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_CHAR16_T;
                }
              }
            }
          } else if (s[4] == '3') {
            if (s[5] == '2') {
              if (s[6] == '_') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_CHAR32_T;
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'o') {
      if (s[2] == '_') {
        if (s[3] == 'a') {
          if (s[4] == 'w') {
            if (s[5] == 'a') {
              if (s[6] == 'i') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_CO_AWAIT;
                }
              }
            }
          }
        } else if (s[3] == 'y') {
          if (s[4] == 'i') {
            if (s[5] == 'e') {
              if (s[6] == 'l') {
                if (s[7] == 'd') {
                  return cxx::TokenKind::T_CO_YIELD;
                }
              }
            }
          }
        }
      } else if (s[2] == 'n') {
        if (s[3] == 't') {
          if (s[4] == 'i') {
            if (s[5] == 'n') {
              if (s[6] == 'u') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T_CONTINUE;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'd') {
    if (s[1] == 'e') {
      if (s[2] == 'c') {
        if (s[3] == 'l') {
          if (s[4] == 't') {
            if (s[5] == 'y') {
              if (s[6] == 'p') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T_DECLTYPE;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'e') {
    if (s[1] == 'x') {
      if (s[2] == 'p') {
        if (s[3] == 'l') {
          if (s[4] == 'i') {
            if (s[5] == 'c') {
              if (s[6] == 'i') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_EXPLICIT;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'n') {
    if (s[1] == 'o') {
      if (s[2] == 'e') {
        if (s[3] == 'x') {
          if (s[4] == 'c') {
            if (s[5] == 'e') {
              if (s[6] == 'p') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_NOEXCEPT;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'o') {
    if (s[1] == 'p') {
      if (s[2] == 'e') {
        if (s[3] == 'r') {
          if (s[4] == 'a') {
            if (s[5] == 't') {
              if (s[6] == 'o') {
                if (s[7] == 'r') {
                  return cxx::TokenKind::T_OPERATOR;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'r') {
    if (s[1] == 'e') {
      if (s[2] == 'q') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'r') {
              if (s[6] == 'e') {
                if (s[7] == 's') {
                  return cxx::TokenKind::T_REQUIRES;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 't') {
    if (s[1] == 'e') {
      if (s[2] == 'm') {
        if (s[3] == 'p') {
          if (s[4] == 'l') {
            if (s[5] == 'a') {
              if (s[6] == 't') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T_TEMPLATE;
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'y') {
      if (s[2] == 'p') {
        if (s[3] == 'e') {
          if (s[4] == 'n') {
            if (s[5] == 'a') {
              if (s[6] == 'm') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T_TYPENAME;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'u') {
    if (s[1] == 'n') {
      if (s[2] == 's') {
        if (s[3] == 'i') {
          if (s[4] == 'g') {
            if (s[5] == 'n') {
              if (s[6] == 'e') {
                if (s[7] == 'd') {
                  return cxx::TokenKind::T_UNSIGNED;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'v') {
    if (s[1] == 'o') {
      if (s[2] == 'l') {
        if (s[3] == 'a') {
          if (s[4] == 't') {
            if (s[5] == 'i') {
              if (s[6] == 'l') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T_VOLATILE;
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'i') {
        if (s[3] == 'm') {
          if (s[4] == 'a') {
            if (s[5] == 'g') {
              if (s[6] == '_') {
                if (s[7] == '_') {
                  return cxx::TokenKind::T___IMAG__;
                }
              }
            }
          }
        } else if (s[3] == 'n') {
          if (s[4] == 't') {
            if (s[5] == '1') {
              if (s[6] == '2') {
                if (s[7] == '8') {
                  return cxx::TokenKind::T___INT128;
                }
              }
            }
          } else if (s[4] == 'l') {
            if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 'e') {
                  return cxx::TokenKind::T___INLINE;
                }
              }
            }
          }
        }
      } else if (s[2] == 'r') {
        if (s[3] == 'e') {
          if (s[4] == 'a') {
            if (s[5] == 'l') {
              if (s[6] == '_') {
                if (s[7] == '_') {
                  return cxx::TokenKind::T___REAL__;
                }
              }
            }
          }
        }
      } else if (s[2] == 't') {
        if (s[3] == 'h') {
          if (s[4] == 'r') {
            if (s[5] == 'e') {
              if (s[6] == 'a') {
                if (s[7] == 'd') {
                  return cxx::TokenKind::T___THREAD;
                }
              }
            }
          }
        } else if (s[3] == 'y') {
          if (s[4] == 'p') {
            if (s[5] == 'e') {
              if (s[6] == 'o') {
                if (s[7] == 'f') {
                  return cxx::TokenKind::T___TYPEOF;
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'C') {
      if (s[2] == 'o') {
        if (s[3] == 'm') {
          if (s[4] == 'p') {
            if (s[5] == 'l') {
              if (s[6] == 'e') {
                if (s[7] == 'x') {
                  return cxx::TokenKind::T__COMPLEX;
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'G') {
      if (s[2] == 'e') {
        if (s[3] == 'n') {
          if (s[4] == 'e') {
            if (s[5] == 'r') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  return cxx::TokenKind::T__GENERIC;
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'A') {
      if (s[2] == 'l') {
        if (s[3] == 'i') {
          if (s[4] == 'g') {
            if (s[5] == 'n') {
              if (s[6] == 'o') {
                if (s[7] == 'f') {
                  return cxx::TokenKind::T__ALIGNOF;
                }
              }
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify9(const char* s) -> cxx::TokenKind {
  if (s[0] == 'c') {
    if (s[1] == 'o') {
      if (s[2] == '_') {
        if (s[3] == 'r') {
          if (s[4] == 'e') {
            if (s[5] == 't') {
              if (s[6] == 'u') {
                if (s[7] == 'r') {
                  if (s[8] == 'n') {
                    return cxx::TokenKind::T_CO_RETURN;
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'n') {
        if (s[3] == 's') {
          if (s[4] == 't') {
            if (s[5] == 'e') {
              if (s[6] == 'v') {
                if (s[7] == 'a') {
                  if (s[8] == 'l') {
                    return cxx::TokenKind::T_CONSTEVAL;
                  }
                }
              } else if (s[6] == 'x') {
                if (s[7] == 'p') {
                  if (s[8] == 'r') {
                    return cxx::TokenKind::T_CONSTEXPR;
                  }
                }
              }
            } else if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 'i') {
                  if (s[8] == 't') {
                    return cxx::TokenKind::T_CONSTINIT;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'n') {
    if (s[1] == 'a') {
      if (s[2] == 'm') {
        if (s[3] == 'e') {
          if (s[4] == 's') {
            if (s[5] == 'p') {
              if (s[6] == 'a') {
                if (s[7] == 'c') {
                  if (s[8] == 'e') {
                    return cxx::TokenKind::T_NAMESPACE;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == 'p') {
    if (s[1] == 'r') {
      if (s[2] == 'o') {
        if (s[3] == 't') {
          if (s[4] == 'e') {
            if (s[5] == 'c') {
              if (s[6] == 't') {
                if (s[7] == 'e') {
                  if (s[8] == 'd') {
                    return cxx::TokenKind::T_PROTECTED;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'f') {
        if (s[3] == 'l') {
          if (s[4] == 'o') {
            if (s[5] == 'a') {
              if (s[6] == 't') {
                if (s[7] == '8') {
                  if (s[8] == '0') {
                    return cxx::TokenKind::T___FLOAT80;
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'a') {
        if (s[3] == 'l') {
          if (s[4] == 'i') {
            if (s[5] == 'g') {
              if (s[6] == 'n') {
                if (s[7] == 'o') {
                  if (s[8] == 'f') {
                    return cxx::TokenKind::T___ALIGNOF;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify10(const char* s) -> cxx::TokenKind {
  if (s[0] == 'c') {
    if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 's') {
          if (s[4] == 't') {
            if (s[5] == '_') {
              if (s[6] == 'c') {
                if (s[7] == 'a') {
                  if (s[8] == 's') {
                    if (s[9] == 't') {
                      return cxx::TokenKind::T_CONST_CAST;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'f') {
        if (s[3] == 'l') {
          if (s[4] == 'o') {
            if (s[5] == 'a') {
              if (s[6] == 't') {
                if (s[7] == '1') {
                  if (s[8] == '2') {
                    if (s[9] == '8') {
                      return cxx::TokenKind::T___FLOAT128;
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'i') {
        if (s[3] == 'n') {
          if (s[4] == 't') {
            if (s[5] == '1') {
              if (s[6] == '2') {
                if (s[7] == '8') {
                  if (s[8] == '_') {
                    if (s[9] == 't') {
                      return cxx::TokenKind::T___INT128_T;
                    }
                  }
                }
              }
            }
          } else if (s[4] == 'l') {
            if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 'e') {
                  if (s[8] == '_') {
                    if (s[9] == '_') {
                      return cxx::TokenKind::T___INLINE__;
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'd') {
        if (s[3] == 'e') {
          if (s[4] == 'c') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'y') {
                  if (s[8] == 'p') {
                    if (s[9] == 'e') {
                      return cxx::TokenKind::T___DECLTYPE;
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'r') {
        if (s[3] == 'e') {
          if (s[4] == 's') {
            if (s[5] == 't') {
              if (s[6] == 'r') {
                if (s[7] == 'i') {
                  if (s[8] == 'c') {
                    if (s[9] == 't') {
                      return cxx::TokenKind::T___RESTRICT;
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 't') {
        if (s[3] == 'y') {
          if (s[4] == 'p') {
            if (s[5] == 'e') {
              if (s[6] == 'o') {
                if (s[7] == 'f') {
                  if (s[8] == '_') {
                    if (s[9] == '_') {
                      return cxx::TokenKind::T___TYPEOF__;
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'v') {
        if (s[3] == 'o') {
          if (s[4] == 'l') {
            if (s[5] == 'a') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'l') {
                    if (s[9] == 'e') {
                      return cxx::TokenKind::T___VOLATILE;
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
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify11(const char* s) -> cxx::TokenKind {
  if (s[0] == 's') {
    if (s[1] == 't') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'i') {
            if (s[5] == 'c') {
              if (s[6] == '_') {
                if (s[7] == 'c') {
                  if (s[8] == 'a') {
                    if (s[9] == 's') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T_STATIC_CAST;
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
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'c') {
        if (s[3] == 'o') {
          if (s[4] == 'm') {
            if (s[5] == 'p') {
              if (s[6] == 'l') {
                if (s[7] == 'e') {
                  if (s[8] == 'x') {
                    if (s[9] == '_') {
                      if (s[10] == '_') {
                        return cxx::TokenKind::T___COMPLEX__;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'u') {
        if (s[3] == 'i') {
          if (s[4] == 'n') {
            if (s[5] == 't') {
              if (s[6] == '1') {
                if (s[7] == '2') {
                  if (s[8] == '8') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___UINT128_T;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'a') {
        if (s[3] == 'l') {
          if (s[4] == 'i') {
            if (s[5] == 'g') {
              if (s[6] == 'n') {
                if (s[7] == 'o') {
                  if (s[8] == 'f') {
                    if (s[9] == '_') {
                      if (s[10] == '_') {
                        return cxx::TokenKind::T___ALIGNOF__;
                      }
                    }
                  }
                }
              }
            }
          }
        } else if (s[3] == 't') {
          if (s[4] == 't') {
            if (s[5] == 'r') {
              if (s[6] == 'i') {
                if (s[7] == 'b') {
                  if (s[8] == 'u') {
                    if (s[9] == 't') {
                      if (s[10] == 'e') {
                        return cxx::TokenKind::T___ATTRIBUTE;
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
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify12(const char* s) -> cxx::TokenKind {
  if (s[0] == 'd') {
    if (s[1] == 'y') {
      if (s[2] == 'n') {
        if (s[3] == 'a') {
          if (s[4] == 'm') {
            if (s[5] == 'i') {
              if (s[6] == 'c') {
                if (s[7] == '_') {
                  if (s[8] == 'c') {
                    if (s[9] == 'a') {
                      if (s[10] == 's') {
                        if (s[11] == 't') {
                          return cxx::TokenKind::T_DYNAMIC_CAST;
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
  } else if (s[0] == 't') {
    if (s[1] == 'h') {
      if (s[2] == 'r') {
        if (s[3] == 'e') {
          if (s[4] == 'a') {
            if (s[5] == 'd') {
              if (s[6] == '_') {
                if (s[7] == 'l') {
                  if (s[8] == 'o') {
                    if (s[9] == 'c') {
                      if (s[10] == 'a') {
                        if (s[11] == 'l') {
                          return cxx::TokenKind::T_THREAD_LOCAL;
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
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'r') {
        if (s[3] == 'e') {
          if (s[4] == 's') {
            if (s[5] == 't') {
              if (s[6] == 'r') {
                if (s[7] == 'i') {
                  if (s[8] == 'c') {
                    if (s[9] == 't') {
                      if (s[10] == '_') {
                        if (s[11] == '_') {
                          return cxx::TokenKind::T___RESTRICT__;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'd') {
        if (s[3] == 'e') {
          if (s[4] == 'c') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'y') {
                  if (s[8] == 'p') {
                    if (s[9] == 'e') {
                      if (s[10] == '_') {
                        if (s[11] == '_') {
                          return cxx::TokenKind::T___DECLTYPE__;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (s[2] == 'v') {
        if (s[3] == 'o') {
          if (s[4] == 'l') {
            if (s[5] == 'a') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'l') {
                    if (s[9] == 'e') {
                      if (s[10] == '_') {
                        if (s[11] == '_') {
                          return cxx::TokenKind::T___VOLATILE__;
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
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify13(const char* s) -> cxx::TokenKind {
  if (s[0] == 's') {
    if (s[1] == 't') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'i') {
            if (s[5] == 'c') {
              if (s[6] == '_') {
                if (s[7] == 'a') {
                  if (s[8] == 's') {
                    if (s[9] == 's') {
                      if (s[10] == 'e') {
                        if (s[11] == 'r') {
                          if (s[12] == 't') {
                            return cxx::TokenKind::T_STATIC_ASSERT;
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
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 't') {
            if (s[5] == 'r') {
              if (s[6] == 'i') {
                if (s[7] == 'b') {
                  if (s[8] == 'u') {
                    if (s[9] == 't') {
                      if (s[10] == 'e') {
                        if (s[11] == '_') {
                          if (s[12] == '_') {
                            return cxx::TokenKind::T___ATTRIBUTE__;
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
      } else if (s[2] == 'e') {
        if (s[3] == 'x') {
          if (s[4] == 't') {
            if (s[5] == 'e') {
              if (s[6] == 'n') {
                if (s[7] == 's') {
                  if (s[8] == 'i') {
                    if (s[9] == 'o') {
                      if (s[10] == 'n') {
                        if (s[11] == '_') {
                          if (s[12] == '_') {
                            return cxx::TokenKind::T___EXTENSION__;
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
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify14(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == 'S') {
      if (s[2] == 't') {
        if (s[3] == 'a') {
          if (s[4] == 't') {
            if (s[5] == 'i') {
              if (s[6] == 'c') {
                if (s[7] == '_') {
                  if (s[8] == 'a') {
                    if (s[9] == 's') {
                      if (s[10] == 's') {
                        if (s[11] == 'e') {
                          if (s[12] == 'r') {
                            if (s[13] == 't') {
                              return cxx::TokenKind::T__STATIC_ASSERT;
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
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify16(const char* s) -> cxx::TokenKind {
  if (s[0] == 'r') {
    if (s[1] == 'e') {
      if (s[2] == 'i') {
        if (s[3] == 'n') {
          if (s[4] == 't') {
            if (s[5] == 'e') {
              if (s[6] == 'r') {
                if (s[7] == 'p') {
                  if (s[8] == 'r') {
                    if (s[9] == 'e') {
                      if (s[10] == 't') {
                        if (s[11] == '_') {
                          if (s[12] == 'c') {
                            if (s[13] == 'a') {
                              if (s[14] == 's') {
                                if (s[15] == 't') {
                                  return cxx::TokenKind::T_REINTERPRET_CAST;
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
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'v') {
                        if (s[11] == 'a') {
                          if (s[12] == '_') {
                            if (s[13] == 'a') {
                              if (s[14] == 'r') {
                                if (s[15] == 'g') {
                                  return cxx::TokenKind::T___BUILTIN_VA_ARG;
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
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify17(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'v') {
                        if (s[11] == 'a') {
                          if (s[12] == '_') {
                            if (s[13] == 'l') {
                              if (s[14] == 'i') {
                                if (s[15] == 's') {
                                  if (s[16] == 't') {
                                    return cxx::TokenKind::T___BUILTIN_VA_LIST;
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
          }
        }
      } else if (s[2] == 'u') {
        if (s[3] == 'n') {
          if (s[4] == 'd') {
            if (s[5] == 'e') {
              if (s[6] == 'r') {
                if (s[7] == 'l') {
                  if (s[8] == 'y') {
                    if (s[9] == 'i') {
                      if (s[10] == 'n') {
                        if (s[11] == 'g') {
                          if (s[12] == '_') {
                            if (s[13] == 't') {
                              if (s[14] == 'y') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'e') {
                                    return cxx::TokenKind::T___UNDERLYING_TYPE;
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
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classify18(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'b') {
                        if (s[11] == 'i') {
                          if (s[12] == 't') {
                            if (s[13] == '_') {
                              if (s[14] == 'c') {
                                if (s[15] == 'a') {
                                  if (s[16] == 's') {
                                    if (s[17] == 't') {
                                      return cxx::TokenKind::
                                          T___BUILTIN_BIT_CAST;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'o') {
                        if (s[11] == 'f') {
                          if (s[12] == 'f') {
                            if (s[13] == 's') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'o') {
                                    if (s[17] == 'f') {
                                      return cxx::TokenKind::
                                          T___BUILTIN_OFFSETOF;
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
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static auto classify(const char* s, int n) -> cxx::TokenKind {
  switch (n) {
    case 2:
      return classify2(s);
    case 3:
      return classify3(s);
    case 4:
      return classify4(s);
    case 5:
      return classify5(s);
    case 6:
      return classify6(s);
    case 7:
      return classify7(s);
    case 8:
      return classify8(s);
    case 9:
      return classify9(s);
    case 10:
      return classify10(s);
    case 11:
      return classify11(s);
    case 12:
      return classify12(s);
    case 13:
      return classify13(s);
    case 14:
      return classify14(s);
    case 16:
      return classify16(s);
    case 17:
      return classify17(s);
    case 18:
      return classify18(s);
    default:
      return cxx::TokenKind::T_IDENTIFIER;
  }  // switch
}