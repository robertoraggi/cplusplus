import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { gen_reflection } from "../src/gen_reflection.ts";
import { loadModel } from "../src/parseModel.ts";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const source = fs.readFileSync(
  new URL("../semantic-model.json", import.meta.url),
  "utf8",
);

function bases(text) {
  return [
    ...text.matchAll(/(?:constexpr int|const) (\w+SlotBase)\s*=\s*([^;]+);/g),
  ].map(([, name, expression]) => [name, expression.replace(/\s+/g, " ")]);
}

test("reflection slots use matching class bases and isolate field additions", () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "cxx-reflection-"));
  try {
    fs.mkdirSync(path.join(output, "src/js/cxx"), { recursive: true });
    fs.mkdirSync(path.join(output, "packages/cxx-frontend/src"), {
      recursive: true,
    });
    fs.symlinkSync(
      path.join(root, "node_modules"),
      path.join(output, "node_modules"),
    );
    const generate = (model) => {
      gen_reflection(loadModel(JSON.stringify(model)), output);
      const cpp = fs.readFileSync(
        path.join(output, "src/js/cxx/reflection.cc"),
        "utf8",
      );
      const ts = fs.readFileSync(
        path.join(output, "packages/cxx-frontend/src/Semantic.ts"),
        "utf8",
      );
      assert.deepEqual(bases(cpp), bases(ts));
      assert.ok(bases(cpp).length > 200);
      assert.doesNotMatch(cpp, /case \d+:\s*\{\s*auto self/);
      assert.doesNotMatch(ts, /cxx\.read\w+\(this\.handle, \d+\)/);
      return { cpp, ts };
    };
    const model = JSON.parse(source);
    const before = generate(model);
    const node = model.classes.find(
      (c) => c.unqualifiedName === "FunctionDeclaratorChunkAST",
    );
    node.fields.push({
      ...node.fields.find((f) => f.name === "isFinal"),
      name: "extraFlag",
    });
    const after = generate(model);
    // Only this class's accessors and the next base declaration may change.
    for (const language of ["cpp", "ts"]) {
      const unrelated = (text) =>
        text
          .split("\n")
          .filter(
            (line) =>
              /SlotBase \+ \d+/.test(line) &&
              !line.includes("FunctionDeclaratorChunkASTSlotBase"),
          );
      assert.deepEqual(unrelated(after[language]), unrelated(before[language]));
    }
    const beforeBases = new Map(bases(before.cpp));
    const changed = bases(after.cpp).filter(
      ([name, value]) => beforeBases.get(name) !== value,
    );
    assert.equal(changed.length, 1);
    assert.match(changed[0][1], /^FunctionDeclaratorChunkASTSlotBase \+ \d+$/);
  } finally {
    fs.rmSync(output, { recursive: true, force: true });
  }
});
