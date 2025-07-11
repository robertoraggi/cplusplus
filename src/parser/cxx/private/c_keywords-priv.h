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

static inline auto classifyC2(const char* s) -> cxx::TokenKind {
  if (s[0] == 'd') {
    if (s[1] == 'o') {
      return cxx::TokenKind::T_DO;
    }
  } else if (s[0] == 'i') {
    if (s[1] == 'f') {
      return cxx::TokenKind::T_IF;
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC3(const char* s) -> cxx::TokenKind {
  if (s[0] == 'a') {
    if (s[1] == 's') {
      if (s[2] == 'm') {
        return cxx::TokenKind::T_ASM;
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
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC4(const char* s) -> cxx::TokenKind {
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
    if (s[1] == 'r') {
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
  } else if (s[0] == '_') {
    if (s[1] == 'a') {
      if (s[2] == 's') {
        if (s[3] == 'm') {
          return cxx::TokenKind::T__ASM;
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC5(const char* s) -> cxx::TokenKind {
  if (s[0] == 'b') {
    if (s[1] == 'r') {
      if (s[2] == 'e') {
        if (s[3] == 'a') {
          if (s[4] == 'k') {
            return cxx::TokenKind::T_BREAK;
          }
        }
      }
    }
  } else if (s[0] == 'c') {
    if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 's') {
          if (s[4] == 't') {
            return cxx::TokenKind::T_CONST;
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
  } else if (s[0] == 'u') {
    if (s[1] == 'n') {
      if (s[2] == 'i') {
        if (s[3] == 'o') {
          if (s[4] == 'n') {
            return cxx::TokenKind::T_UNION;
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
  } else if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 's') {
          if (s[4] == 'm') {
            return cxx::TokenKind::T___ASM;
          }
        }
      }
    } else if (s[1] == 'B') {
      if (s[2] == 'o') {
        if (s[3] == 'o') {
          if (s[4] == 'l') {
            return cxx::TokenKind::T__BOOL;
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC6(const char* s) -> cxx::TokenKind {
  if (s[0] == 'd') {
    if (s[1] == 'o') {
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
      if (s[2] == 't') {
        if (s[3] == 'e') {
          if (s[4] == 'r') {
            if (s[5] == 'n') {
              return cxx::TokenKind::T_EXTERN;
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
          if (s[4] == 'o') {
            if (s[5] == 'f') {
              return cxx::TokenKind::T_TYPEOF;
            }
          }
        }
      }
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC7(const char* s) -> cxx::TokenKind {
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
  } else if (s[0] == '_') {
    if (s[1] == 'A') {
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
    } else if (s[1] == 'B') {
      if (s[2] == 'i') {
        if (s[3] == 't') {
          if (s[4] == 'I') {
            if (s[5] == 'n') {
              if (s[6] == 't') {
                return cxx::TokenKind::T__BITINT;
              }
            }
          }
        }
      }
    } else if (s[1] == '_') {
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
    }
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC8(const char* s) -> cxx::TokenKind {
  if (s[0] == 'c') {
    if (s[1] == 'o') {
      if (s[2] == 'n') {
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
  } else if (s[0] == 'r') {
    if (s[1] == 'e') {
      if (s[2] == 'g') {
        if (s[3] == 'i') {
          if (s[4] == 's') {
            if (s[5] == 't') {
              if (s[6] == 'e') {
                if (s[7] == 'r') {
                  return cxx::TokenKind::T_REGISTER;
                }
              }
            }
          }
        }
      } else if (s[2] == 's') {
        if (s[3] == 't') {
          if (s[4] == 'r') {
            if (s[5] == 'i') {
              if (s[6] == 'c') {
                if (s[7] == 't') {
                  return cxx::TokenKind::T_RESTRICT;
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
    if (s[1] == 'C') {
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
    } else if (s[1] == '_') {
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
    } else if (s[1] == 'A') {
      if (s[2] == 'l') {
        if (s[3] == 'i') {
          if (s[4] == 'g') {
            if (s[5] == 'n') {
              if (s[6] == 'a') {
                if (s[7] == 's') {
                  return cxx::TokenKind::T__ALIGNAS;
                }
              } else if (s[6] == 'o') {
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

static inline auto classifyC9(const char* s) -> cxx::TokenKind {
  if (s[0] == 'c') {
    if (s[1] == 'o') {
      if (s[2] == 'n') {
        if (s[3] == 's') {
          if (s[4] == 't') {
            if (s[5] == 'e') {
              if (s[6] == 'x') {
                if (s[7] == 'p') {
                  if (s[8] == 'r') {
                    return cxx::TokenKind::T_CONSTEXPR;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (s[0] == '_') {
    if (s[1] == 'N') {
      if (s[2] == 'o') {
        if (s[3] == 'r') {
          if (s[4] == 'e') {
            if (s[5] == 't') {
              if (s[6] == 'u') {
                if (s[7] == 'r') {
                  if (s[8] == 'n') {
                    return cxx::TokenKind::T__NORETURN;
                  }
                }
              }
            }
          }
        }
      }
    } else if (s[1] == '_') {
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

static inline auto classifyC10(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == 'D') {
      if (s[2] == 'e') {
        if (s[3] == 'c') {
          if (s[4] == 'i') {
            if (s[5] == 'm') {
              if (s[6] == 'a') {
                if (s[7] == 'l') {
                  if (s[8] == '3') {
                    if (s[9] == '2') {
                      return cxx::TokenKind::T__DECIMAL32;
                    }
                  } else if (s[8] == '6') {
                    if (s[9] == '4') {
                      return cxx::TokenKind::T__DECIMAL64;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (s[1] == 'I') {
      if (s[2] == 'm') {
        if (s[3] == 'a') {
          if (s[4] == 'g') {
            if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 'a') {
                  if (s[8] == 'r') {
                    if (s[9] == 'y') {
                      return cxx::TokenKind::T__IMAGINARY;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (s[1] == '_') {
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
      } else if (s[2] == 'i') {
        if (s[3] == 'n') {
          if (s[4] == 'l') {
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

static inline auto classifyC11(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == 'D') {
      if (s[2] == 'e') {
        if (s[3] == 'c') {
          if (s[4] == 'i') {
            if (s[5] == 'm') {
              if (s[6] == 'a') {
                if (s[7] == 'l') {
                  if (s[8] == '1') {
                    if (s[9] == '2') {
                      if (s[10] == '8') {
                        return cxx::TokenKind::T__DECIMAL128;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (s[1] == '_') {
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

static inline auto classifyC12(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
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
  }
  return cxx::TokenKind::T_IDENTIFIER;
}

static inline auto classifyC13(const char* s) -> cxx::TokenKind {
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
  } else if (s[0] == 't') {
    if (s[1] == 'y') {
      if (s[2] == 'p') {
        if (s[3] == 'e') {
          if (s[4] == 'o') {
            if (s[5] == 'f') {
              if (s[6] == '_') {
                if (s[7] == 'u') {
                  if (s[8] == 'n') {
                    if (s[9] == 'q') {
                      if (s[10] == 'u') {
                        if (s[11] == 'a') {
                          if (s[12] == 'l') {
                            return cxx::TokenKind::T_TYPEOF_UNQUAL;
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
    } else if (s[1] == 'T') {
      if (s[2] == 'h') {
        if (s[3] == 'r') {
          if (s[4] == 'e') {
            if (s[5] == 'a') {
              if (s[6] == 'd') {
                if (s[7] == '_') {
                  if (s[8] == 'l') {
                    if (s[9] == 'o') {
                      if (s[10] == 'c') {
                        if (s[11] == 'a') {
                          if (s[12] == 'l') {
                            return cxx::TokenKind::T__THREAD_LOCAL;
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

static inline auto classifyC14(const char* s) -> cxx::TokenKind {
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

static inline auto classifyC16(const char* s) -> cxx::TokenKind {
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

static inline auto classifyC17(const char* s) -> cxx::TokenKind {
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

static inline auto classifyC18(const char* s) -> cxx::TokenKind {
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

static auto classifyC(const char* s, int n) -> cxx::TokenKind {
  switch (n) {
    case 2:
      return classifyC2(s);
    case 3:
      return classifyC3(s);
    case 4:
      return classifyC4(s);
    case 5:
      return classifyC5(s);
    case 6:
      return classifyC6(s);
    case 7:
      return classifyC7(s);
    case 8:
      return classifyC8(s);
    case 9:
      return classifyC9(s);
    case 10:
      return classifyC10(s);
    case 11:
      return classifyC11(s);
    case 12:
      return classifyC12(s);
    case 13:
      return classifyC13(s);
    case 14:
      return classifyC14(s);
    case 16:
      return classifyC16(s);
    case 17:
      return classifyC17(s);
    case 18:
      return classifyC18(s);
    default:
      return cxx::TokenKind::T_IDENTIFIER;
  }  // switch
}