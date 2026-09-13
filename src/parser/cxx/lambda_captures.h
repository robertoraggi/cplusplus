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

#include <cxx/ast.h>
#include <cxx/symbols_fwd.h>

namespace cxx {

[[nodiscard]] inline auto capture_initializer_slot(LambdaCaptureAST* capture)
    -> ExpressionAST** {
  if (auto simple = ast_cast<SimpleLambdaCaptureAST>(capture))
    return &simple->initializer;
  if (auto ref = ast_cast<RefLambdaCaptureAST>(capture))
    return &ref->initializer;
  if (auto self = ast_cast<ThisLambdaCaptureAST>(capture))
    return &self->initializer;
  if (auto initCapture = ast_cast<InitLambdaCaptureAST>(capture))
    return &initCapture->initializer;
  if (auto refInitCapture = ast_cast<RefInitLambdaCaptureAST>(capture))
    return &refInitCapture->initializer;
  return nullptr;
}

[[nodiscard]] inline auto capture_initializer(LambdaCaptureAST* capture)
    -> ExpressionAST* {
  auto slot = capture_initializer_slot(capture);
  return slot ? *slot : nullptr;
}

[[nodiscard]] inline auto capture_field_slot(LambdaCaptureAST* capture)
    -> FieldSymbol** {
  if (auto simple = ast_cast<SimpleLambdaCaptureAST>(capture))
    return &simple->symbol;
  if (auto ref = ast_cast<RefLambdaCaptureAST>(capture)) return &ref->symbol;
  if (auto self = ast_cast<ThisLambdaCaptureAST>(capture)) return &self->symbol;
  if (auto deref = ast_cast<DerefThisLambdaCaptureAST>(capture))
    return &deref->symbol;
  if (auto initCapture = ast_cast<InitLambdaCaptureAST>(capture))
    return &initCapture->symbol;
  if (auto refInitCapture = ast_cast<RefInitLambdaCaptureAST>(capture))
    return &refInitCapture->symbol;
  return nullptr;
}

[[nodiscard]] inline auto capture_field(LambdaCaptureAST* capture)
    -> FieldSymbol* {
  auto slot = capture_field_slot(capture);
  return slot ? *slot : nullptr;
}

[[nodiscard]] inline auto capture_identifier(LambdaCaptureAST* capture)
    -> const Identifier* {
  if (auto simple = ast_cast<SimpleLambdaCaptureAST>(capture))
    return simple->identifier;
  if (auto ref = ast_cast<RefLambdaCaptureAST>(capture)) return ref->identifier;
  if (auto initCapture = ast_cast<InitLambdaCaptureAST>(capture))
    return initCapture->identifier;
  if (auto refInitCapture = ast_cast<RefInitLambdaCaptureAST>(capture))
    return refInitCapture->identifier;
  return nullptr;
}

[[nodiscard]] inline auto is_pack_capture(LambdaCaptureAST* capture) -> bool {
  if (auto simple = ast_cast<SimpleLambdaCaptureAST>(capture))
    return bool(simple->ellipsisLoc);
  if (auto ref = ast_cast<RefLambdaCaptureAST>(capture))
    return bool(ref->ellipsisLoc);
  if (auto initCapture = ast_cast<InitLambdaCaptureAST>(capture))
    return bool(initCapture->ellipsisLoc);
  if (auto refInitCapture = ast_cast<RefInitLambdaCaptureAST>(capture))
    return bool(refInitCapture->ellipsisLoc);
  return false;
}

}  // namespace cxx
