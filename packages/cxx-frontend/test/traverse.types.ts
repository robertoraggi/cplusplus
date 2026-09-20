import * as S from "cxx-frontend/model";
import { NodePath, traverse, walk, type Visitor } from "cxx-frontend/traverse";

declare const unit: S.TranslationUnitAST;

traverse(unit, {
  NamespaceDefinition(path) {
    const node: S.NamespaceDefinitionAST = path.node;
    const name: string | undefined = path.node.identifier?.name;
    (void node, name);
  },
});

traverse(unit, {
  Expression(path) {
    const node: S.ExpressionAST = path.node;
    const type: S.Type | undefined = path.node.type;
    (void node, type);
  },
});

traverse(unit, {
  enter(path) {
    const kind: S.ASTKind = path.node.kind;
    void kind;
  },
  exit(path) {
    const kind: S.ASTKind = path.node.kind;
    void kind;
  },
});

traverse(unit, {
  enter(path) {
    if (path.isNamespaceDefinition()) {
      const identifier: S.Identifier | undefined = path.node.identifier;
      void identifier;
    }
    if (path.isFunctionDefinition()) {
      const body: S.FunctionBodyAST | undefined = path.node.functionBody;
      void body;
    }
  },
});

for (const path of walk(unit)) {
  if (path.isParameterDeclaration()) {
    const symbol: S.ParameterSymbol | undefined = path.node.symbol;
    void symbol;
  }
}

const counted: { count: number } = traverse(
  unit,
  {
    FunctionDefinition(_path, state) {
      state.count += 1;
    },
  },
  { count: 0 },
);
void counted;

const visitor: Visitor<{ names: string[] }> = {
  NamespaceDefinition: {
    enter(path, state) {
      state.names.push(path.node.identifier?.name ?? "");
    },
  },
};
void visitor;

const root: NodePath<S.TranslationUnitAST> = new NodePath(unit);
for (const child of root) {
  const parent: S.AST | undefined = child.parent;
  const key: string | number = child.key;
  (void parent, key);
}
void root.traverse(visitor, { names: [] });
void [...root.descendants()];
void [...root.ancestors()];
void root.findParent((path) => path.isDeclaration());

const declaration: NodePath<S.DeclarationAST> | undefined = [
  ...root.children(),
].find((path): path is NodePath<S.DeclarationAST> => path.isDeclaration());
void declaration;

const name: S.ASTKind = root.node.kind;
void name;
