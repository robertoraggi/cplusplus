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

static inline auto classifyBuiltinFunction13(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'l') {
                      if (s[10] == 'o') {
                        if (s[11] == 'a') {
                          if (s[12] == 'd') {
                            return cxx::BuiltinFunctionKind::T___ATOMIC_LOAD;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'a') {
                        if (s[11] == 'b') {
                          if (s[12] == 's') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_ABS;
                          }
                        }
                      } else if (s[10] == 'c') {
                        if (s[11] == 'o') {
                          if (s[12] == 's') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_COS;
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'z') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_CTZ;
                          }
                        }
                      } else if (s[10] == 'e') {
                        if (s[11] == 'r') {
                          if (s[12] == 'f') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_ERF;
                          }
                        } else if (s[11] == 'x') {
                          if (s[12] == 'p') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_EXP;
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'm') {
                          if (s[12] == 'a') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_FMA;
                          }
                        }
                      } else if (s[10] == 'i') {
                        if (s[11] == 'n') {
                          if (s[12] == 'f') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_INF;
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'o') {
                          if (s[12] == 'g') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_LOG;
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_NAN;
                          }
                        }
                      } else if (s[10] == 'p') {
                        if (s[11] == 'o') {
                          if (s[12] == 'w') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_POW;
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_SIN;
                          }
                        }
                      } else if (s[10] == 't') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            return cxx::BuiltinFunctionKind::T___BUILTIN_TAN;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction14(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'c') {
                      if (s[10] == 'l') {
                        if (s[11] == 'e') {
                          if (s[12] == 'a') {
                            if (s[13] == 'r') {
                              return cxx::BuiltinFunctionKind::T___ATOMIC_CLEAR;
                            }
                          }
                        }
                      }
                    } else if (s[9] == 's') {
                      if (s[10] == 't') {
                        if (s[11] == 'o') {
                          if (s[12] == 'r') {
                            if (s[13] == 'e') {
                              return cxx::BuiltinFunctionKind::T___ATOMIC_STORE;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'F') {
                        if (s[11] == 'I') {
                          if (s[12] == 'L') {
                            if (s[13] == 'E') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FILE;
                            }
                          }
                        }
                      } else if (s[10] == 'L') {
                        if (s[11] == 'I') {
                          if (s[12] == 'N') {
                            if (s[13] == 'E') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LINE;
                            }
                          }
                        }
                      } else if (s[10] == 'a') {
                        if (s[11] == 'c') {
                          if (s[12] == 'o') {
                            if (s[13] == 's') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ACOS;
                            }
                          }
                        } else if (s[11] == 's') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ASIN;
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'a') {
                            if (s[13] == 'n') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ATAN;
                            }
                          }
                        }
                      } else if (s[10] == 'b') {
                        if (s[11] == 'c') {
                          if (s[12] == 'm') {
                            if (s[13] == 'p') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_BCMP;
                            }
                          }
                        }
                      } else if (s[10] == 'c') {
                        if (s[11] == 'b') {
                          if (s[12] == 'r') {
                            if (s[13] == 't') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_CBRT;
                            }
                          }
                        } else if (s[11] == 'e') {
                          if (s[12] == 'i') {
                            if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_CEIL;
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'z') {
                            if (s[13] == 'g') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_CLZG;
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 's') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_COSF;
                            } else if (s[13] == 'h') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_COSH;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_COSL;
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'z') {
                            if (s[13] == 'g') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_CTZG;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_CTZL;
                            }
                          }
                        }
                      } else if (s[10] == 'e') {
                        if (s[11] == 'r') {
                          if (s[12] == 'f') {
                            if (s[13] == 'c') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ERFC;
                            } else if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ERFF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_ERFL;
                            }
                          }
                        } else if (s[11] == 'x') {
                          if (s[12] == 'p') {
                            if (s[13] == '2') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_EXP2;
                            } else if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_EXPF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_EXPL;
                            }
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'a') {
                          if (s[12] == 'b') {
                            if (s[13] == 's') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FABS;
                            }
                          }
                        } else if (s[11] == 'd') {
                          if (s[12] == 'i') {
                            if (s[13] == 'm') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FDIM;
                            }
                          }
                        } else if (s[11] == 'm') {
                          if (s[12] == 'a') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FMAF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FMAL;
                            } else if (s[13] == 'x') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FMAX;
                            }
                          } else if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FMIN;
                            }
                          } else if (s[12] == 'o') {
                            if (s[13] == 'd') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_FMOD;
                            }
                          }
                        }
                      } else if (s[10] == 'i') {
                        if (s[11] == 'n') {
                          if (s[12] == 'f') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_INFF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_INFL;
                            }
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'a') {
                          if (s[12] == 'b') {
                            if (s[13] == 's') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LABS;
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'g') {
                            if (s[13] == '2') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LOG2;
                            } else if (s[13] == 'b') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LOGB;
                            } else if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LOGF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_LOGL;
                            }
                          }
                        }
                      } else if (s[10] == 'm') {
                        if (s[11] == 'o') {
                          if (s[12] == 'd') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_MODF;
                            }
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_NANF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_NANL;
                            } else if (s[13] == 's') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_NANS;
                            }
                          }
                        }
                      } else if (s[10] == 'p') {
                        if (s[11] == 'o') {
                          if (s[12] == 'w') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_POWF;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_POWL;
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 't') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_RINT;
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_SINF;
                            } else if (s[13] == 'h') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_SINH;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_SINL;
                            }
                          }
                        } else if (s[11] == 'q') {
                          if (s[12] == 'r') {
                            if (s[13] == 't') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_SQRT;
                            }
                          }
                        }
                      } else if (s[10] == 't') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            if (s[13] == 'f') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_TANF;
                            } else if (s[13] == 'h') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_TANH;
                            } else if (s[13] == 'l') {
                              return cxx::BuiltinFunctionKind::T___BUILTIN_TANL;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction15(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'l') {
                      if (s[10] == 'o') {
                        if (s[11] == 'a') {
                          if (s[12] == 'd') {
                            if (s[13] == '_') {
                              if (s[14] == 'n') {
                                return cxx::BuiltinFunctionKind::
                                    T___ATOMIC_LOAD_N;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'a') {
                        if (s[11] == 'c') {
                          if (s[12] == 'o') {
                            if (s[13] == 's') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ACOSF;
                              } else if (s[14] == 'h') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ACOSH;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ACOSL;
                              }
                            }
                          }
                        } else if (s[11] == 's') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ASINF;
                              } else if (s[14] == 'h') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ASINH;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ASINL;
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'a') {
                            if (s[13] == 'n') {
                              if (s[14] == '2') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ATAN2;
                              } else if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ATANF;
                              } else if (s[14] == 'h') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ATANH;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ATANL;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'b') {
                        if (s[11] == 'c') {
                          if (s[12] == 'o') {
                            if (s[13] == 'p') {
                              if (s[14] == 'y') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_BCOPY;
                              }
                            }
                          }
                        } else if (s[11] == 'z') {
                          if (s[12] == 'e') {
                            if (s[13] == 'r') {
                              if (s[14] == 'o') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_BZERO;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'c') {
                        if (s[11] == 'b') {
                          if (s[12] == 'r') {
                            if (s[13] == 't') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_CBRTF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_CBRTL;
                              }
                            }
                          }
                        } else if (s[11] == 'e') {
                          if (s[12] == 'i') {
                            if (s[13] == 'l') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_CEILF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_CEILL;
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 's') {
                            if (s[13] == 'h') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_COSHF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_COSHL;
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'z') {
                            if (s[13] == 'l') {
                              if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_CTZLL;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'e') {
                        if (s[11] == 'r') {
                          if (s[12] == 'f') {
                            if (s[13] == 'c') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ERFCF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ERFCL;
                              }
                            }
                          }
                        } else if (s[11] == 'x') {
                          if (s[12] == 'p') {
                            if (s[13] == '2') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_EXP2F;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_EXP2L;
                              }
                            } else if (s[13] == 'm') {
                              if (s[14] == '1') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_EXPM1;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'a') {
                          if (s[12] == 'b') {
                            if (s[13] == 's') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FABSF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FABSL;
                              }
                            }
                          }
                        } else if (s[11] == 'd') {
                          if (s[12] == 'i') {
                            if (s[13] == 'm') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FDIMF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FDIML;
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'o') {
                            if (s[13] == 'o') {
                              if (s[14] == 'r') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FLOOR;
                              }
                            }
                          }
                        } else if (s[11] == 'm') {
                          if (s[12] == 'a') {
                            if (s[13] == 'x') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMAXF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMAXL;
                              }
                            }
                          } else if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMINF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMINL;
                              }
                            }
                          } else if (s[12] == 'o') {
                            if (s[13] == 'd') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMODF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FMODL;
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_FREXP;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'h') {
                        if (s[11] == 'y') {
                          if (s[12] == 'p') {
                            if (s[13] == 'o') {
                              if (s[14] == 't') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_HYPOT;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'i') {
                        if (s[11] == 'l') {
                          if (s[12] == 'o') {
                            if (s[13] == 'g') {
                              if (s[14] == 'b') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ILOGB;
                              }
                            }
                          }
                        } else if (s[11] == 'n') {
                          if (s[12] == 'd') {
                            if (s[13] == 'e') {
                              if (s[14] == 'x') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_INDEX;
                              }
                            }
                          }
                        } else if (s[11] == 's') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ISINF;
                              }
                            }
                          } else if (s[12] == 'n') {
                            if (s[13] == 'a') {
                              if (s[14] == 'n') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ISNAN;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'd') {
                          if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LDEXP;
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'a') {
                            if (s[13] == 'b') {
                              if (s[14] == 's') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LLABS;
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'g') {
                            if (s[13] == '1') {
                              if (s[14] == '0') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOG10;
                              } else if (s[14] == 'p') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOG1P;
                              }
                            } else if (s[13] == '2') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOG2F;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOG2L;
                              }
                            } else if (s[13] == 'b') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOGBF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LOGBL;
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 't') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_LRINT;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'm') {
                        if (s[11] == 'o') {
                          if (s[12] == 'd') {
                            if (s[13] == 'f') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_MODFF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_MODFL;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            if (s[13] == 's') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_NANSF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_NANSL;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 't') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_RINTF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_RINTL;
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'd') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_ROUND;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'h') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_SINHF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_SINHL;
                              }
                            }
                          }
                        } else if (s[11] == 'q') {
                          if (s[12] == 'r') {
                            if (s[13] == 't') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_SQRTF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_SQRTL;
                              }
                            }
                          }
                        }
                      } else if (s[10] == 't') {
                        if (s[11] == 'a') {
                          if (s[12] == 'n') {
                            if (s[13] == 'h') {
                              if (s[14] == 'f') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_TANHF;
                              } else if (s[14] == 'l') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_TANHL;
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'c') {
                                return cxx::BuiltinFunctionKind::
                                    T___BUILTIN_TRUNC;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction16(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 's') {
                      if (s[10] == 't') {
                        if (s[11] == 'o') {
                          if (s[12] == 'r') {
                            if (s[13] == 'e') {
                              if (s[14] == '_') {
                                if (s[15] == 'n') {
                                  return cxx::BuiltinFunctionKind::
                                      T___ATOMIC_STORE_N;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'a') {
                        if (s[11] == 'c') {
                          if (s[12] == 'o') {
                            if (s[13] == 's') {
                              if (s[14] == 'h') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ACOSHF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ACOSHL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'l') {
                            if (s[13] == 'o') {
                              if (s[14] == 'c') {
                                if (s[15] == 'a') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ALLOCA;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 's') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'h') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ASINHF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ASINHL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'a') {
                            if (s[13] == 'n') {
                              if (s[14] == '2') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ATAN2F;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ATAN2L;
                                }
                              } else if (s[14] == 'h') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ATANHF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ATANHL;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'e') {
                        if (s[11] == 'x') {
                          if (s[12] == 'p') {
                            if (s[13] == 'e') {
                              if (s[14] == 'c') {
                                if (s[15] == 't') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_EXPECT;
                                }
                              }
                            } else if (s[13] == 'm') {
                              if (s[14] == '1') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_EXPM1F;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_EXPM1L;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'i') {
                              if (s[14] == 't') {
                                if (s[15] == 'e') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_FINITE;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'o') {
                            if (s[13] == 'o') {
                              if (s[14] == 'r') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_FLOORF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_FLOORL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_FREXPF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_FREXPL;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'h') {
                        if (s[11] == 'y') {
                          if (s[12] == 'p') {
                            if (s[13] == 'o') {
                              if (s[14] == 't') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_HYPOTF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_HYPOTL;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'i') {
                        if (s[11] == 'l') {
                          if (s[12] == 'o') {
                            if (s[13] == 'g') {
                              if (s[14] == 'b') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ILOGBF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ILOGBL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'n') {
                          if (s[12] == 'v') {
                            if (s[13] == 'o') {
                              if (s[14] == 'k') {
                                if (s[15] == 'e') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_INVOKE;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'd') {
                          if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LDEXPF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LDEXPL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'g') {
                          if (s[12] == 'a') {
                            if (s[13] == 'm') {
                              if (s[14] == 'm') {
                                if (s[15] == 'a') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LGAMMA;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'r') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 't') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LLRINT;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'g') {
                            if (s[13] == '1') {
                              if (s[14] == '0') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LOG10F;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LOG10L;
                                }
                              } else if (s[14] == 'p') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LOG1PF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LOG1PL;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 't') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LRINTF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LRINTL;
                                }
                              }
                            }
                          } else if (s[12] == 'o') {
                            if (s[13] == 'u') {
                              if (s[14] == 'n') {
                                if (s[15] == 'd') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_LROUND;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'm') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'c') {
                              if (s[14] == 'h') {
                                if (s[15] == 'r') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_MEMCHR;
                                }
                              } else if (s[14] == 'm') {
                                if (s[15] == 'p') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_MEMCMP;
                                }
                              } else if (s[14] == 'p') {
                                if (s[15] == 'y') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_MEMCPY;
                                }
                              }
                            } else if (s[13] == 's') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_MEMSET;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'q') {
                              if (s[14] == 'u') {
                                if (s[15] == 'o') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_REMQUO;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'd') {
                              if (s[14] == 'e') {
                                if (s[15] == 'x') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_RINDEX;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'd') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ROUNDF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_ROUNDL;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'c') {
                          if (s[12] == 'a') {
                            if (s[13] == 'l') {
                              if (s[14] == 'b') {
                                if (s[15] == 'n') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_SCALBN;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 's') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_SINCOS;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'p') {
                            if (s[13] == 'c') {
                              if (s[14] == 'p') {
                                if (s[15] == 'y') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STPCPY;
                                }
                              }
                            }
                          } else if (s[12] == 'r') {
                            if (s[13] == 'c') {
                              if (s[14] == 'a') {
                                if (s[15] == 't') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRCAT;
                                }
                              } else if (s[14] == 'h') {
                                if (s[15] == 'r') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRCHR;
                                }
                              } else if (s[14] == 'm') {
                                if (s[15] == 'p') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRCMP;
                                }
                              } else if (s[14] == 'p') {
                                if (s[15] == 'y') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRCPY;
                                }
                              }
                            } else if (s[13] == 'd') {
                              if (s[14] == 'u') {
                                if (s[15] == 'p') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRDUP;
                                }
                              }
                            } else if (s[13] == 'l') {
                              if (s[14] == 'e') {
                                if (s[15] == 'n') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRLEN;
                                }
                              }
                            } else if (s[13] == 's') {
                              if (s[14] == 'p') {
                                if (s[15] == 'n') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRSPN;
                                }
                              } else if (s[14] == 't') {
                                if (s[15] == 'r') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRSTR;
                                }
                              }
                            } else if (s[13] == 't') {
                              if (s[14] == 'o') {
                                if (s[15] == 'k') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_STRTOK;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 't') {
                        if (s[11] == 'g') {
                          if (s[12] == 'a') {
                            if (s[13] == 'm') {
                              if (s[14] == 'm') {
                                if (s[15] == 'a') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_TGAMMA;
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'c') {
                                if (s[15] == 'f') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_TRUNCF;
                                } else if (s[15] == 'l') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_TRUNCL;
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'v') {
                        if (s[11] == 'a') {
                          if (s[12] == '_') {
                            if (s[13] == 'e') {
                              if (s[14] == 'n') {
                                if (s[15] == 'd') {
                                  return cxx::BuiltinFunctionKind::
                                      T___BUILTIN_VA_END;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction17(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'e') {
                      if (s[10] == 'x') {
                        if (s[11] == 'c') {
                          if (s[12] == 'h') {
                            if (s[13] == 'a') {
                              if (s[14] == 'n') {
                                if (s[15] == 'g') {
                                  if (s[16] == 'e') {
                                    return cxx::BuiltinFunctionKind::
                                        T___ATOMIC_EXCHANGE;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'f') {
                      if (s[10] == 'e') {
                        if (s[11] == 't') {
                          if (s[12] == 'c') {
                            if (s[13] == 'h') {
                              if (s[14] == '_') {
                                if (s[15] == 'o') {
                                  if (s[16] == 'r') {
                                    return cxx::BuiltinFunctionKind::
                                        T___ATOMIC_FETCH_OR;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'o') {
                      if (s[10] == 'r') {
                        if (s[11] == '_') {
                          if (s[12] == 'f') {
                            if (s[13] == 'e') {
                              if (s[14] == 't') {
                                if (s[15] == 'c') {
                                  if (s[16] == 'h') {
                                    return cxx::BuiltinFunctionKind::
                                        T___ATOMIC_OR_FETCH;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == '_') {
                        if (s[11] == '_') {
                          if (s[12] == 'c') {
                            if (s[13] == 'o') {
                              if (s[14] == 's') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN___COSPI;
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                if (s[15] == '1') {
                                  if (s[16] == '0') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN___EXP10;
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 's') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN___SINPI;
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 't') {
                            if (s[13] == 'a') {
                              if (s[14] == 'n') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN___TANPI;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'b') {
                        if (s[11] == 's') {
                          if (s[12] == 'w') {
                            if (s[13] == 'a') {
                              if (s[14] == 'p') {
                                if (s[15] == '3') {
                                  if (s[16] == '2') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_BSWAP32;
                                  }
                                } else if (s[15] == '6') {
                                  if (s[16] == '4') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_BSWAP64;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'i') {
                          if (s[12] == 'n') {
                            if (s[13] == 'i') {
                              if (s[14] == 't') {
                                if (s[15] == 'e') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_FINITEF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_FINITEL;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'g') {
                          if (s[12] == 'a') {
                            if (s[13] == 'm') {
                              if (s[14] == 'm') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LGAMMAF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LGAMMAL;
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'l') {
                          if (s[12] == 'r') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 't') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LLRINTF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LLRINTL;
                                  }
                                }
                              }
                            } else if (s[13] == 'o') {
                              if (s[14] == 'u') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'd') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LLROUND;
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'r') {
                          if (s[12] == 'o') {
                            if (s[13] == 'u') {
                              if (s[14] == 'n') {
                                if (s[15] == 'd') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LROUNDF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_LROUNDL;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'm') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'c') {
                              if (s[14] == 'c') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_MEMCCPY;
                                  }
                                }
                              }
                            } else if (s[13] == 'm') {
                              if (s[14] == 'o') {
                                if (s[15] == 'v') {
                                  if (s[16] == 'e') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_MEMMOVE;
                                  }
                                }
                              }
                            } else if (s[13] == 'p') {
                              if (s[14] == 'c') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_MEMPCPY;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'q') {
                              if (s[14] == 'u') {
                                if (s[15] == 'o') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_REMQUOF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_REMQUOL;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'c') {
                          if (s[12] == 'a') {
                            if (s[13] == 'l') {
                              if (s[14] == 'b') {
                                if (s[15] == 'l') {
                                  if (s[16] == 'n') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SCALBLN;
                                  }
                                } else if (s[15] == 'n') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SCALBNF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SCALBNL;
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'i') {
                          if (s[12] == 'g') {
                            if (s[13] == 'n') {
                              if (s[14] == 'b') {
                                if (s[15] == 'i') {
                                  if (s[16] == 't') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SIGNBIT;
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'n') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 's') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SINCOSF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_SINCOSL;
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'p') {
                            if (s[13] == 'n') {
                              if (s[14] == 'c') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STPNCPY;
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'r') {
                            if (s[13] == 'c') {
                              if (s[14] == 's') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'n') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRCSPN;
                                  }
                                }
                              }
                            } else if (s[13] == 'l') {
                              if (s[14] == 'c') {
                                if (s[15] == 'a') {
                                  if (s[16] == 't') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRLCAT;
                                  }
                                } else if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRLCPY;
                                  }
                                }
                              }
                            } else if (s[13] == 'n') {
                              if (s[14] == 'c') {
                                if (s[15] == 'a') {
                                  if (s[16] == 't') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRNCAT;
                                  }
                                } else if (s[15] == 'm') {
                                  if (s[16] == 'p') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRNCMP;
                                  }
                                } else if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRNCPY;
                                  }
                                }
                              } else if (s[14] == 'd') {
                                if (s[15] == 'u') {
                                  if (s[16] == 'p') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRNDUP;
                                  }
                                }
                              }
                            } else if (s[13] == 'p') {
                              if (s[14] == 'b') {
                                if (s[15] == 'r') {
                                  if (s[16] == 'k') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRPBRK;
                                  }
                                }
                              }
                            } else if (s[13] == 'r') {
                              if (s[14] == 'c') {
                                if (s[15] == 'h') {
                                  if (s[16] == 'r') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRRCHR;
                                  }
                                }
                              }
                            } else if (s[13] == 'x') {
                              if (s[14] == 'f') {
                                if (s[15] == 'r') {
                                  if (s[16] == 'm') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_STRXFRM;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 't') {
                        if (s[11] == 'g') {
                          if (s[12] == 'a') {
                            if (s[13] == 'm') {
                              if (s[14] == 'm') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'f') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_TGAMMAF;
                                  } else if (s[16] == 'l') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_TGAMMAL;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'v') {
                        if (s[11] == 'a') {
                          if (s[12] == '_') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'y') {
                                    return cxx::BuiltinFunctionKind::
                                        T___BUILTIN_VA_COPY;
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
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'i') {
                                  if (s[16] == 't') {
                                    return cxx::BuiltinFunctionKind::
                                        T___C11_ATOMIC_INIT;
                                  }
                                }
                              }
                            } else if (s[13] == 'l') {
                              if (s[14] == 'o') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'd') {
                                    return cxx::BuiltinFunctionKind::
                                        T___C11_ATOMIC_LOAD;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction18(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'a') {
                      if (s[10] == 'd') {
                        if (s[11] == 'd') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_ADD_FETCH;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'd') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_AND_FETCH;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'f') {
                      if (s[10] == 'e') {
                        if (s[11] == 't') {
                          if (s[12] == 'c') {
                            if (s[13] == 'h') {
                              if (s[14] == '_') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'd') {
                                    if (s[17] == 'd') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_FETCH_ADD;
                                    }
                                  } else if (s[16] == 'n') {
                                    if (s[17] == 'd') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_FETCH_AND;
                                    }
                                  }
                                } else if (s[15] == 's') {
                                  if (s[16] == 'u') {
                                    if (s[17] == 'b') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_FETCH_SUB;
                                    }
                                  }
                                } else if (s[15] == 'x') {
                                  if (s[16] == 'o') {
                                    if (s[17] == 'r') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_FETCH_XOR;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 's') {
                      if (s[10] == 'u') {
                        if (s[11] == 'b') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_SUB_FETCH;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'x') {
                      if (s[10] == 'o') {
                        if (s[11] == 'r') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      return cxx::BuiltinFunctionKind::
                                          T___ATOMIC_XOR_FETCH;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'F') {
                        if (s[11] == 'U') {
                          if (s[12] == 'N') {
                            if (s[13] == 'C') {
                              if (s[14] == 'T') {
                                if (s[15] == 'I') {
                                  if (s[16] == 'O') {
                                    if (s[17] == 'N') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_FUNCTION;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == '_') {
                        if (s[11] == '_') {
                          if (s[12] == 'c') {
                            if (s[13] == 'o') {
                              if (s[14] == 's') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN___COSPIF;
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'e') {
                            if (s[13] == 'x') {
                              if (s[14] == 'p') {
                                if (s[15] == '1') {
                                  if (s[16] == '0') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN___EXP10F;
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'f') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'i') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'e') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN___FINITE;
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 's') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN___SINPIF;
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 't') {
                            if (s[13] == 'a') {
                              if (s[14] == 'n') {
                                if (s[15] == 'p') {
                                  if (s[16] == 'i') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN___TANPIF;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'c') {
                        if (s[11] == 'o') {
                          if (s[12] == 'p') {
                            if (s[13] == 'y') {
                              if (s[14] == 's') {
                                if (s[15] == 'i') {
                                  if (s[16] == 'g') {
                                    if (s[17] == 'n') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_COPYSIGN;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'h') {
                        if (s[11] == 'u') {
                          if (s[12] == 'g') {
                            if (s[13] == 'e') {
                              if (s[14] == '_') {
                                if (s[15] == 'v') {
                                  if (s[16] == 'a') {
                                    if (s[17] == 'l') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_HUGE_VAL;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'i') {
                        if (s[11] == 's') {
                          if (s[12] == 'f') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'i') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'e') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_ISFINITE;
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'n') {
                            if (s[13] == 'o') {
                              if (s[14] == 'r') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'a') {
                                    if (s[17] == 'l') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_ISNORMAL;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'l') {
                        if (s[11] == 'l') {
                          if (s[12] == 'r') {
                            if (s[13] == 'o') {
                              if (s[14] == 'u') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'd') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_LLROUNDF;
                                    } else if (s[17] == 'l') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_LLROUNDL;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 'c') {
                          if (s[12] == 'a') {
                            if (s[13] == 'l') {
                              if (s[14] == 'b') {
                                if (s[15] == 'l') {
                                  if (s[16] == 'n') {
                                    if (s[17] == 'f') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_SCALBLNF;
                                    } else if (s[17] == 'l') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_SCALBLNL;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 't') {
                          if (s[12] == 'r') {
                            if (s[13] == 'e') {
                              if (s[14] == 'r') {
                                if (s[15] == 'r') {
                                  if (s[16] == 'o') {
                                    if (s[17] == 'r') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_STRERROR;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'v') {
                        if (s[11] == 'a') {
                          if (s[12] == '_') {
                            if (s[13] == 's') {
                              if (s[14] == 't') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'r') {
                                    if (s[17] == 't') {
                                      return cxx::BuiltinFunctionKind::
                                          T___BUILTIN_VA_START;
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
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 's') {
                              if (s[14] == 't') {
                                if (s[15] == 'o') {
                                  if (s[16] == 'r') {
                                    if (s[17] == 'e') {
                                      return cxx::BuiltinFunctionKind::
                                          T___C11_ATOMIC_STORE;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction19(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'e') {
                      if (s[10] == 'x') {
                        if (s[11] == 'c') {
                          if (s[12] == 'h') {
                            if (s[13] == 'a') {
                              if (s[14] == 'n') {
                                if (s[15] == 'g') {
                                  if (s[16] == 'e') {
                                    if (s[17] == '_') {
                                      if (s[18] == 'n') {
                                        return cxx::BuiltinFunctionKind::
                                            T___ATOMIC_EXCHANGE_N;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'f') {
                      if (s[10] == 'e') {
                        if (s[11] == 't') {
                          if (s[12] == 'c') {
                            if (s[13] == 'h') {
                              if (s[14] == '_') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'a') {
                                    if (s[17] == 'n') {
                                      if (s[18] == 'd') {
                                        return cxx::BuiltinFunctionKind::
                                            T___ATOMIC_FETCH_NAND;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else if (s[9] == 'n') {
                      if (s[10] == 'a') {
                        if (s[11] == 'n') {
                          if (s[12] == 'd') {
                            if (s[13] == '_') {
                              if (s[14] == 'f') {
                                if (s[15] == 'e') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'c') {
                                      if (s[18] == 'h') {
                                        return cxx::BuiltinFunctionKind::
                                            T___ATOMIC_NAND_FETCH;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == '_') {
                        if (s[11] == '_') {
                          if (s[12] == 'f') {
                            if (s[13] == 'i') {
                              if (s[14] == 'n') {
                                if (s[15] == 'i') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'f') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN___FINITEF;
                                      } else if (s[18] == 'l') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN___FINITEL;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'a') {
                        if (s[11] == 'd') {
                          if (s[12] == 'd') {
                            if (s[13] == 'r') {
                              if (s[14] == 'e') {
                                if (s[15] == 's') {
                                  if (s[16] == 's') {
                                    if (s[17] == 'o') {
                                      if (s[18] == 'f') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_ADDRESSOF;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'c') {
                        if (s[11] == 'o') {
                          if (s[12] == 'p') {
                            if (s[13] == 'y') {
                              if (s[14] == 's') {
                                if (s[15] == 'i') {
                                  if (s[16] == 'g') {
                                    if (s[17] == 'n') {
                                      if (s[18] == 'f') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_COPYSIGNF;
                                      } else if (s[18] == 'l') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_COPYSIGNL;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'h') {
                        if (s[11] == 'u') {
                          if (s[12] == 'g') {
                            if (s[13] == 'e') {
                              if (s[14] == '_') {
                                if (s[15] == 'v') {
                                  if (s[16] == 'a') {
                                    if (s[17] == 'l') {
                                      if (s[18] == 'f') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_HUGE_VALF;
                                      } else if (s[18] == 'l') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_HUGE_VALL;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'e') {
                          if (s[12] == 'a') {
                            if (s[13] == 'r') {
                              if (s[14] == 'b') {
                                if (s[15] == 'y') {
                                  if (s[16] == 'i') {
                                    if (s[17] == 'n') {
                                      if (s[18] == 't') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_NEARBYINT;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'x') {
                            if (s[13] == 't') {
                              if (s[14] == 'a') {
                                if (s[15] == 'f') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'r') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_NEXTAFTER;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'a') {
                              if (s[14] == 'i') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'd') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'r') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_REMAINDER;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'd') {
                                if (s[15] == 'e') {
                                  if (s[16] == 'v') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'n') {
                                        return cxx::BuiltinFunctionKind::
                                            T___BUILTIN_ROUNDEVEN;
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
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction20(const char* s)
    -> cxx::BuiltinFunctionKind {
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
                      if (s[10] == 'c') {
                        if (s[11] == 'o') {
                          if (s[12] == 'n') {
                            if (s[13] == 's') {
                              if (s[14] == 't') {
                                if (s[15] == 'a') {
                                  if (s[16] == 'n') {
                                    if (s[17] == 't') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'p') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_CONSTANT_P;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'f') {
                        if (s[11] == 'p') {
                          if (s[12] == 'c') {
                            if (s[13] == 'l') {
                              if (s[14] == 'a') {
                                if (s[15] == 's') {
                                  if (s[16] == 's') {
                                    if (s[17] == 'i') {
                                      if (s[18] == 'f') {
                                        if (s[19] == 'y') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_FPCLASSIFY;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'n') {
                        if (s[11] == 'e') {
                          if (s[12] == 'a') {
                            if (s[13] == 'r') {
                              if (s[14] == 'b') {
                                if (s[15] == 'y') {
                                  if (s[16] == 'i') {
                                    if (s[17] == 'n') {
                                      if (s[18] == 't') {
                                        if (s[19] == 'f') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_NEARBYINTF;
                                        } else if (s[19] == 'l') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_NEARBYINTL;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'x') {
                            if (s[13] == 't') {
                              if (s[14] == 'a') {
                                if (s[15] == 'f') {
                                  if (s[16] == 't') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'f') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_NEXTAFTERF;
                                        } else if (s[19] == 'l') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_NEXTAFTERL;
                                        }
                                      }
                                    }
                                  }
                                }
                              } else if (s[14] == 't') {
                                if (s[15] == 'o') {
                                  if (s[16] == 'w') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'd') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_NEXTTOWARD;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 'r') {
                        if (s[11] == 'e') {
                          if (s[12] == 'm') {
                            if (s[13] == 'a') {
                              if (s[14] == 'i') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'd') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'f') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_REMAINDERF;
                                        } else if (s[19] == 'l') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_REMAINDERL;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        } else if (s[11] == 'o') {
                          if (s[12] == 'u') {
                            if (s[13] == 'n') {
                              if (s[14] == 'd') {
                                if (s[15] == 'e') {
                                  if (s[16] == 'v') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'n') {
                                        if (s[19] == 'f') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_ROUNDEVENF;
                                        } else if (s[19] == 'l') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_ROUNDEVENL;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else if (s[10] == 's') {
                        if (s[11] == 't') {
                          if (s[12] == 'r') {
                            if (s[13] == 'c') {
                              if (s[14] == 'a') {
                                if (s[15] == 's') {
                                  if (s[16] == 'e') {
                                    if (s[17] == 'c') {
                                      if (s[18] == 'm') {
                                        if (s[19] == 'p') {
                                          return cxx::BuiltinFunctionKind::
                                              T___BUILTIN_STRCASECMP;
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
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction21(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'i') {
                      if (s[10] == 's') {
                        if (s[11] == '_') {
                          if (s[12] == 'l') {
                            if (s[13] == 'o') {
                              if (s[14] == 'c') {
                                if (s[15] == 'k') {
                                  if (s[16] == '_') {
                                    if (s[17] == 'f') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'e') {
                                          if (s[20] == 'e') {
                                            return cxx::BuiltinFunctionKind::
                                                T___ATOMIC_IS_LOCK_FREE;
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
                    } else if (s[9] == 's') {
                      if (s[10] == 'i') {
                        if (s[11] == 'g') {
                          if (s[12] == 'n') {
                            if (s[13] == 'a') {
                              if (s[14] == 'l') {
                                if (s[15] == '_') {
                                  if (s[16] == 'f') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'n') {
                                        if (s[19] == 'c') {
                                          if (s[20] == 'e') {
                                            return cxx::BuiltinFunctionKind::
                                                T___ATOMIC_SIGNAL_FENCE;
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
                    } else if (s[9] == 't') {
                      if (s[10] == 'e') {
                        if (s[11] == 's') {
                          if (s[12] == 't') {
                            if (s[13] == '_') {
                              if (s[14] == 'a') {
                                if (s[15] == 'n') {
                                  if (s[16] == 'd') {
                                    if (s[17] == '_') {
                                      if (s[18] == 's') {
                                        if (s[19] == 'e') {
                                          if (s[20] == 't') {
                                            return cxx::BuiltinFunctionKind::
                                                T___ATOMIC_TEST_AND_SET;
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
                      } else if (s[10] == 'h') {
                        if (s[11] == 'r') {
                          if (s[12] == 'e') {
                            if (s[13] == 'a') {
                              if (s[14] == 'd') {
                                if (s[15] == '_') {
                                  if (s[16] == 'f') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'n') {
                                        if (s[19] == 'c') {
                                          if (s[20] == 'e') {
                                            return cxx::BuiltinFunctionKind::
                                                T___ATOMIC_THREAD_FENCE;
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
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'n') {
                        if (s[11] == 'e') {
                          if (s[12] == 'x') {
                            if (s[13] == 't') {
                              if (s[14] == 't') {
                                if (s[15] == 'o') {
                                  if (s[16] == 'w') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'd') {
                                          if (s[20] == 'f') {
                                            return cxx::BuiltinFunctionKind::
                                                T___BUILTIN_NEXTTOWARDF;
                                          } else if (s[20] == 'l') {
                                            return cxx::BuiltinFunctionKind::
                                                T___BUILTIN_NEXTTOWARDL;
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
                      } else if (s[10] == 's') {
                        if (s[11] == 't') {
                          if (s[12] == 'r') {
                            if (s[13] == 'n') {
                              if (s[14] == 'c') {
                                if (s[15] == 'a') {
                                  if (s[16] == 's') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'c') {
                                        if (s[19] == 'm') {
                                          if (s[20] == 'p') {
                                            return cxx::BuiltinFunctionKind::
                                                T___BUILTIN_STRNCASECMP;
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
                      } else if (s[10] == 'u') {
                        if (s[11] == 'n') {
                          if (s[12] == 'r') {
                            if (s[13] == 'e') {
                              if (s[14] == 'a') {
                                if (s[15] == 'c') {
                                  if (s[16] == 'h') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'b') {
                                        if (s[19] == 'l') {
                                          if (s[20] == 'e') {
                                            return cxx::BuiltinFunctionKind::
                                                T___BUILTIN_UNREACHABLE;
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
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'e') {
                              if (s[14] == 'x') {
                                if (s[15] == 'c') {
                                  if (s[16] == 'h') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'n') {
                                        if (s[19] == 'g') {
                                          if (s[20] == 'e') {
                                            return cxx::BuiltinFunctionKind::
                                                T___C11_ATOMIC_EXCHANGE;
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            } else if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'o') {
                                          if (s[20] == 'r') {
                                            return cxx::BuiltinFunctionKind::
                                                T___C11_ATOMIC_FETCH_OR;
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
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction22(const char* s)
    -> cxx::BuiltinFunctionKind {
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
                      if (s[10] == 'c') {
                        if (s[11] == '2') {
                          if (s[12] == '3') {
                            if (s[13] == '_') {
                              if (s[14] == 'v') {
                                if (s[15] == 'a') {
                                  if (s[16] == '_') {
                                    if (s[17] == 's') {
                                      if (s[18] == 't') {
                                        if (s[19] == 'a') {
                                          if (s[20] == 'r') {
                                            if (s[21] == 't') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___BUILTIN_C23_VA_START;
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
                      } else if (s[10] == 'f') {
                        if (s[11] == 'm') {
                          if (s[12] == 'a') {
                            if (s[13] == 'x') {
                              if (s[14] == 'i') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'u') {
                                    if (s[17] == 'm') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'u') {
                                            if (s[21] == 'm') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___BUILTIN_FMAXIMUM_NUM;
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          } else if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'i') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'u') {
                                    if (s[17] == 'm') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'u') {
                                            if (s[21] == 'm') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___BUILTIN_FMINIMUM_NUM;
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
                      } else if (s[10] == 'o') {
                        if (s[11] == 'p') {
                          if (s[12] == 'e') {
                            if (s[13] == 'r') {
                              if (s[14] == 'a') {
                                if (s[15] == 't') {
                                  if (s[16] == 'o') {
                                    if (s[17] == 'r') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'e') {
                                            if (s[21] == 'w') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___BUILTIN_OPERATOR_NEW;
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
        }
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'a') {
                                          if (s[20] == 'd') {
                                            if (s[21] == 'd') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___C11_ATOMIC_FETCH_ADD;
                                            }
                                          } else if (s[20] == 'n') {
                                            if (s[21] == 'd') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___C11_ATOMIC_FETCH_AND;
                                            }
                                          }
                                        } else if (s[19] == 's') {
                                          if (s[20] == 'u') {
                                            if (s[21] == 'b') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___C11_ATOMIC_FETCH_SUB;
                                            }
                                          }
                                        } else if (s[19] == 'x') {
                                          if (s[20] == 'o') {
                                            if (s[21] == 'r') {
                                              return cxx::BuiltinFunctionKind::
                                                  T___C11_ATOMIC_FETCH_XOR;
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
        }
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction23(const char* s)
    -> cxx::BuiltinFunctionKind {
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
                      if (s[10] == 'f') {
                        if (s[11] == 'm') {
                          if (s[12] == 'a') {
                            if (s[13] == 'x') {
                              if (s[14] == 'i') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'u') {
                                    if (s[17] == 'm') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'u') {
                                            if (s[21] == 'm') {
                                              if (s[22] == 'f') {
                                                return cxx::BuiltinFunctionKind::
                                                    T___BUILTIN_FMAXIMUM_NUMF;
                                              } else if (s[22] == 'l') {
                                                return cxx::BuiltinFunctionKind::
                                                    T___BUILTIN_FMAXIMUM_NUML;
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
                          } else if (s[12] == 'i') {
                            if (s[13] == 'n') {
                              if (s[14] == 'i') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'u') {
                                    if (s[17] == 'm') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'u') {
                                            if (s[21] == 'm') {
                                              if (s[22] == 'f') {
                                                return cxx::BuiltinFunctionKind::
                                                    T___BUILTIN_FMINIMUM_NUMF;
                                              } else if (s[22] == 'l') {
                                                return cxx::BuiltinFunctionKind::
                                                    T___BUILTIN_FMINIMUM_NUML;
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
          }
        }
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'f') {
                              if (s[14] == 'e') {
                                if (s[15] == 't') {
                                  if (s[16] == 'c') {
                                    if (s[17] == 'h') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 'a') {
                                            if (s[21] == 'n') {
                                              if (s[22] == 'd') {
                                                return cxx::BuiltinFunctionKind::
                                                    T___C11_ATOMIC_FETCH_NAND;
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
          }
        }
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction24(const char* s)
    -> cxx::BuiltinFunctionKind {
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
                      if (s[10] == 'a') {
                        if (s[11] == 's') {
                          if (s[12] == 's') {
                            if (s[13] == 'u') {
                              if (s[14] == 'm') {
                                if (s[15] == 'e') {
                                  if (s[16] == '_') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'l') {
                                        if (s[19] == 'i') {
                                          if (s[20] == 'g') {
                                            if (s[21] == 'n') {
                                              if (s[22] == 'e') {
                                                if (s[23] == 'd') {
                                                  return cxx::BuiltinFunctionKind::
                                                      T___BUILTIN_ASSUME_ALIGNED;
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
            }
          }
        }
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction25(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'a') {
                      if (s[10] == 'l') {
                        if (s[11] == 'w') {
                          if (s[12] == 'a') {
                            if (s[13] == 'y') {
                              if (s[14] == 's') {
                                if (s[15] == '_') {
                                  if (s[16] == 'l') {
                                    if (s[17] == 'o') {
                                      if (s[18] == 'c') {
                                        if (s[19] == 'k') {
                                          if (s[20] == '_') {
                                            if (s[21] == 'f') {
                                              if (s[22] == 'r') {
                                                if (s[23] == 'e') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___ATOMIC_ALWAYS_LOCK_FREE;
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
                    } else if (s[9] == 'c') {
                      if (s[10] == 'o') {
                        if (s[11] == 'm') {
                          if (s[12] == 'p') {
                            if (s[13] == 'a') {
                              if (s[14] == 'r') {
                                if (s[15] == 'e') {
                                  if (s[16] == '_') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'x') {
                                        if (s[19] == 'c') {
                                          if (s[20] == 'h') {
                                            if (s[21] == 'a') {
                                              if (s[22] == 'n') {
                                                if (s[23] == 'g') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___ATOMIC_COMPARE_EXCHANGE;
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
              }
            }
          }
        }
      } else if (s[2] == 'b') {
        if (s[3] == 'u') {
          if (s[4] == 'i') {
            if (s[5] == 'l') {
              if (s[6] == 't') {
                if (s[7] == 'i') {
                  if (s[8] == 'n') {
                    if (s[9] == '_') {
                      if (s[10] == 'o') {
                        if (s[11] == 'p') {
                          if (s[12] == 'e') {
                            if (s[13] == 'r') {
                              if (s[14] == 'a') {
                                if (s[15] == 't') {
                                  if (s[16] == 'o') {
                                    if (s[17] == 'r') {
                                      if (s[18] == '_') {
                                        if (s[19] == 'd') {
                                          if (s[20] == 'e') {
                                            if (s[21] == 'l') {
                                              if (s[22] == 'e') {
                                                if (s[23] == 't') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___BUILTIN_OPERATOR_DELETE;
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
              }
            }
          }
        }
      } else if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'i') {
                              if (s[14] == 's') {
                                if (s[15] == '_') {
                                  if (s[16] == 'l') {
                                    if (s[17] == 'o') {
                                      if (s[18] == 'c') {
                                        if (s[19] == 'k') {
                                          if (s[20] == '_') {
                                            if (s[21] == 'f') {
                                              if (s[22] == 'r') {
                                                if (s[23] == 'e') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___C11_ATOMIC_IS_LOCK_FREE;
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
                            } else if (s[13] == 's') {
                              if (s[14] == 'i') {
                                if (s[15] == 'g') {
                                  if (s[16] == 'n') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'l') {
                                        if (s[19] == '_') {
                                          if (s[20] == 'f') {
                                            if (s[21] == 'e') {
                                              if (s[22] == 'n') {
                                                if (s[23] == 'c') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___C11_ATOMIC_SIGNAL_FENCE;
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
                            } else if (s[13] == 't') {
                              if (s[14] == 'h') {
                                if (s[15] == 'r') {
                                  if (s[16] == 'e') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'd') {
                                        if (s[19] == '_') {
                                          if (s[20] == 'f') {
                                            if (s[21] == 'e') {
                                              if (s[22] == 'n') {
                                                if (s[23] == 'c') {
                                                  if (s[24] == 'e') {
                                                    return cxx::BuiltinFunctionKind::
                                                        T___C11_ATOMIC_THREAD_FENCE;
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
              }
            }
          }
        }
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction27(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'a') {
        if (s[3] == 't') {
          if (s[4] == 'o') {
            if (s[5] == 'm') {
              if (s[6] == 'i') {
                if (s[7] == 'c') {
                  if (s[8] == '_') {
                    if (s[9] == 'c') {
                      if (s[10] == 'o') {
                        if (s[11] == 'm') {
                          if (s[12] == 'p') {
                            if (s[13] == 'a') {
                              if (s[14] == 'r') {
                                if (s[15] == 'e') {
                                  if (s[16] == '_') {
                                    if (s[17] == 'e') {
                                      if (s[18] == 'x') {
                                        if (s[19] == 'c') {
                                          if (s[20] == 'h') {
                                            if (s[21] == 'a') {
                                              if (s[22] == 'n') {
                                                if (s[23] == 'g') {
                                                  if (s[24] == 'e') {
                                                    if (s[25] == '_') {
                                                      if (s[26] == 'n') {
                                                        return cxx::
                                                            BuiltinFunctionKind::
                                                                T___ATOMIC_COMPARE_EXCHANGE_N;
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
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction31(const char* s)
    -> cxx::BuiltinFunctionKind {
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
                      if (s[10] == 'i') {
                        if (s[11] == 's') {
                          if (s[12] == '_') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 'n') {
                                  if (s[16] == 's') {
                                    if (s[17] == 't') {
                                      if (s[18] == 'a') {
                                        if (s[19] == 'n') {
                                          if (s[20] == 't') {
                                            if (s[21] == '_') {
                                              if (s[22] == 'e') {
                                                if (s[23] == 'v') {
                                                  if (s[24] == 'a') {
                                                    if (s[25] == 'l') {
                                                      if (s[26] == 'u') {
                                                        if (s[27] == 'a') {
                                                          if (s[28] == 't') {
                                                            if (s[29] == 'e') {
                                                              if (s[30] ==
                                                                  'd') {
                                                                return cxx::
                                                                    BuiltinFunctionKind::
                                                                        T___BUILTIN_IS_CONSTANT_EVALUATED;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction34(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'p') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'e') {
                                          if (s[20] == '_') {
                                            if (s[21] == 'e') {
                                              if (s[22] == 'x') {
                                                if (s[23] == 'c') {
                                                  if (s[24] == 'h') {
                                                    if (s[25] == 'a') {
                                                      if (s[26] == 'n') {
                                                        if (s[27] == 'g') {
                                                          if (s[28] == 'e') {
                                                            if (s[29] == '_') {
                                                              if (s[30] ==
                                                                  'w') {
                                                                if (s[31] ==
                                                                    'e') {
                                                                  if (s[32] ==
                                                                      'a') {
                                                                    if (s[33] ==
                                                                        'k') {
                                                                      return cxx::
                                                                          BuiltinFunctionKind::
                                                                              T___C11_ATOMIC_COMPARE_EXCHANGE_WEAK;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static inline auto classifyBuiltinFunction36(const char* s)
    -> cxx::BuiltinFunctionKind {
  if (s[0] == '_') {
    if (s[1] == '_') {
      if (s[2] == 'c') {
        if (s[3] == '1') {
          if (s[4] == '1') {
            if (s[5] == '_') {
              if (s[6] == 'a') {
                if (s[7] == 't') {
                  if (s[8] == 'o') {
                    if (s[9] == 'm') {
                      if (s[10] == 'i') {
                        if (s[11] == 'c') {
                          if (s[12] == '_') {
                            if (s[13] == 'c') {
                              if (s[14] == 'o') {
                                if (s[15] == 'm') {
                                  if (s[16] == 'p') {
                                    if (s[17] == 'a') {
                                      if (s[18] == 'r') {
                                        if (s[19] == 'e') {
                                          if (s[20] == '_') {
                                            if (s[21] == 'e') {
                                              if (s[22] == 'x') {
                                                if (s[23] == 'c') {
                                                  if (s[24] == 'h') {
                                                    if (s[25] == 'a') {
                                                      if (s[26] == 'n') {
                                                        if (s[27] == 'g') {
                                                          if (s[28] == 'e') {
                                                            if (s[29] == '_') {
                                                              if (s[30] ==
                                                                  's') {
                                                                if (s[31] ==
                                                                    't') {
                                                                  if (s[32] ==
                                                                      'r') {
                                                                    if (s[33] ==
                                                                        'o') {
                                                                      if (s[34] ==
                                                                          'n') {
                                                                        if (s[35] ==
                                                                            'g') {
                                                                          return cxx::
                                                                              BuiltinFunctionKind::
                                                                                  T___C11_ATOMIC_COMPARE_EXCHANGE_STRONG;
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
  return cxx::BuiltinFunctionKind::T_NONE;
}

static auto classifyBuiltinFunction(const char* s, int n)
    -> cxx::BuiltinFunctionKind {
  switch (n) {
    case 13:
      return classifyBuiltinFunction13(s);
    case 14:
      return classifyBuiltinFunction14(s);
    case 15:
      return classifyBuiltinFunction15(s);
    case 16:
      return classifyBuiltinFunction16(s);
    case 17:
      return classifyBuiltinFunction17(s);
    case 18:
      return classifyBuiltinFunction18(s);
    case 19:
      return classifyBuiltinFunction19(s);
    case 20:
      return classifyBuiltinFunction20(s);
    case 21:
      return classifyBuiltinFunction21(s);
    case 22:
      return classifyBuiltinFunction22(s);
    case 23:
      return classifyBuiltinFunction23(s);
    case 24:
      return classifyBuiltinFunction24(s);
    case 25:
      return classifyBuiltinFunction25(s);
    case 27:
      return classifyBuiltinFunction27(s);
    case 31:
      return classifyBuiltinFunction31(s);
    case 34:
      return classifyBuiltinFunction34(s);
    case 36:
      return classifyBuiltinFunction36(s);
    default:
      return cxx::BuiltinFunctionKind::T_NONE;
  }  // switch
}