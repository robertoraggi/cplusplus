// Generated file by: kwgen.ts
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

static inline auto classifySveType10(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'S') {
        if (s[3] == 'V') {
          if (s[4] == 'B') {
            if (s[5] == 'o') {
              if (s[6] == 'o') {
                if (s[7] == 'l') {
                  if (s[8] == '_') {
                    if (s[9] == 't') {
                      return cxx::TokenKind::T___SVBOOL_T;
                    }
                  }
                }
              }
            }
          } else if (s[4] == 'I') {
            if (s[5] == 'n') {
              if (s[6] == 't') {
                if (s[7] == '8') {
                  if (s[8] == '_') {
                    if (s[9] == 't') {
                      return cxx::TokenKind::T___SVINT8_T;
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

static inline auto classifySveType11(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'S') {
        if (s[3] == 'V') {
          if (s[4] == 'C') {
            if (s[5] == 'o') {
              if (s[6] == 'u') {
                if (s[7] == 'n') {
                  if (s[8] == 't') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___SVCOUNT_T;
                      }
                    }
                  }
                }
              }
            }
          } else if (s[4] == 'I') {
            if (s[5] == 'n') {
              if (s[6] == 't') {
                if (s[7] == '1') {
                  if (s[8] == '6') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___SVINT16_T;
                      }
                    }
                  }
                } else if (s[7] == '3') {
                  if (s[8] == '2') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___SVINT32_T;
                      }
                    }
                  }
                } else if (s[7] == '6') {
                  if (s[8] == '4') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___SVINT64_T;
                      }
                    }
                  }
                }
              }
            }
          } else if (s[4] == 'U') {
            if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 't') {
                  if (s[8] == '8') {
                    if (s[9] == '_') {
                      if (s[10] == 't') {
                        return cxx::TokenKind::T___SVUINT8_T;
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

static inline auto classifySveType12(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'S') {
        if (s[3] == 'V') {
          if (s[4] == 'U') {
            if (s[5] == 'i') {
              if (s[6] == 'n') {
                if (s[7] == 't') {
                  if (s[8] == '1') {
                    if (s[9] == '6') {
                      if (s[10] == '_') {
                        if (s[11] == 't') {
                          return cxx::TokenKind::T___SVUINT16_T;
                        }
                      }
                    }
                  } else if (s[8] == '3') {
                    if (s[9] == '2') {
                      if (s[10] == '_') {
                        if (s[11] == 't') {
                          return cxx::TokenKind::T___SVUINT32_T;
                        }
                      }
                    }
                  } else if (s[8] == '6') {
                    if (s[9] == '4') {
                      if (s[10] == '_') {
                        if (s[11] == 't') {
                          return cxx::TokenKind::T___SVUINT64_T;
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

static inline auto classifySveType13(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'S') {
        if (s[3] == 'V') {
          if (s[4] == 'F') {
            if (s[5] == 'l') {
              if (s[6] == 'o') {
                if (s[7] == 'a') {
                  if (s[8] == 't') {
                    if (s[9] == '1') {
                      if (s[10] == '6') {
                        if (s[11] == '_') {
                          if (s[12] == 't') {
                            return cxx::TokenKind::T___SVFLOAT16_T;
                          }
                        }
                      }
                    } else if (s[9] == '3') {
                      if (s[10] == '2') {
                        if (s[11] == '_') {
                          if (s[12] == 't') {
                            return cxx::TokenKind::T___SVFLOAT32_T;
                          }
                        }
                      }
                    } else if (s[9] == '6') {
                      if (s[10] == '4') {
                        if (s[11] == '_') {
                          if (s[12] == 't') {
                            return cxx::TokenKind::T___SVFLOAT64_T;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          } else if (s[4] == 'M') {
            if (s[5] == 'f') {
              if (s[6] == 'l') {
                if (s[7] == 'o') {
                  if (s[8] == 'a') {
                    if (s[9] == 't') {
                      if (s[10] == '8') {
                        if (s[11] == '_') {
                          if (s[12] == 't') {
                            return cxx::TokenKind::T___SVMFLOAT8_T;
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

static inline auto classifySveType14(const char* s) -> cxx::TokenKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'S') {
        if (s[3] == 'V') {
          if (s[4] == 'B') {
            if (s[5] == 'f') {
              if (s[6] == 'l') {
                if (s[7] == 'o') {
                  if (s[8] == 'a') {
                    if (s[9] == 't') {
                      if (s[10] == '1') {
                        if (s[11] == '6') {
                          if (s[12] == '_') {
                            if (s[13] == 't') {
                              return cxx::TokenKind::T___SVBFLOAT16_T;
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

static auto classifySveType(const char* s, int n) -> cxx::TokenKind {
  switch (n) {
    case 10:
      return classifySveType10(s);
    case 11:
      return classifySveType11(s);
    case 12:
      return classifySveType12(s);
    case 13:
      return classifySveType13(s);
    case 14:
      return classifySveType14(s);
    default:
      return cxx::TokenKind::T_IDENTIFIER;
  }  // switch
}