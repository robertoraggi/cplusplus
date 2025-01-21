// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

namespace cxx {
class BasicBlock;
class ExpressionAST;
}  // namespace cxx

namespace cxx::ir {
class BasicBlock;
}

namespace cxx {
struct Codegen {
  void cg_condition(ExpressionAST* ast, ir::BasicBlock* iftrue,
                    ir::BasicBlock* iffalse);
};
}  // namespace cxx

// clang-format off
// CHECK:namespace
// CHECK:  namespace cxx
// CHECK:    class BasicBlock
// CHECK:    class ExpressionAST
// CHECK:    namespace ir
// CHECK:      class BasicBlock
// CHECK:    class Codegen
// CHECK:      function void cg_condition(cxx::ExpressionAST*, ir::BasicBlock*, ir::BasicBlock*)
