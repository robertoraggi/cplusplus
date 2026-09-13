import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { loadModel } from "../src/parseModel.ts";
import { astFromModel } from "../src/astFromModel.ts";
import { normalizeModel } from "../src/modelSnapshot.ts";

const source = fs.readFileSync(
  new URL("../semantic-model.json", import.meta.url),
  "utf8",
);
const index = loadModel(source);

test("AST generation uses the snapshot and preserves declaration spellings", () => {
  const ast = astFromModel(index);
  assert.ok(ast.nodes.length > 200);
  const field = ast.nodes
    .find((n) => n.name === "CaseStatementAST")
    .members.find((m) => m.name === "caseValue");
  assert.equal(field.type, "std::int64_t");
  assert.equal(field.initializer, "0");
  const attributes = ast.baseMembers.get("AttributeSpecifierAST");
  assert.equal(attributes[0].type, "AttributeMap");
  assert.equal(attributes[0].cv, "const");
  assert.equal(attributes[0].ptrOps, "*");
  const node = ast.nodes.find((n) => n.name === "ClassSpecifierAST");
  assert.equal(
    node.members.find((m) => m.name === "declarationList").kind,
    "node-list",
  );
  assert.equal(node.members.find((m) => m.name === "classLoc").kind, "token");
});

test("model locations are repository relative and normalization preserves order", () => {
  assert.ok(!source.includes("/Users/"));
  const model = {
    enums: [],
    classes: [
      { name: "B", location: { file: "/checkout/src/b.h" } },
      { name: "A", location: { file: "/checkout/src/a.h" } },
    ],
  };
  const normalized = normalizeModel(model, "/checkout");
  assert.deepEqual(
    normalized.classes.map((c) => c.name),
    ["B", "A"],
  );
  assert.equal(normalized.classes[0].location.file, "src/b.h");
  assert.equal(model.classes[0].location.file, "/checkout/src/b.h");
});
