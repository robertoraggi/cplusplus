import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadModel } from "../src/parseModel.ts";
import { PlanBuilder } from "../src/semanticPlan.ts";
import { gen_semantic_codec } from "../src/gen_semantic_codec.ts";

const index = loadModel(
  fs.readFileSync(new URL("../semantic-model.json", import.meta.url), "utf8"),
);

test("semantic decoding derives nested container types from their accessor", () => {
  const plan = new PlanBuilder(index).build();
  assert.deepEqual(plan.diagnostics, []);
  const address = plan.structs.find(
    (entity) => entity.short === "ConstAddress",
  );
  const field = address.fields.find((field) => field.name === "subobjectPath_");
  assert.equal(field.typeExpression, "$->subobjectPath()");

  const output = fs.mkdtempSync(path.join(os.tmpdir(), "cxx-semantic-"));
  try {
    const sourceOutput = path.join(output, "semantic_codec.cc");
    gen_semantic_codec({
      plan,
      headerOutput: path.join(output, "semantic_codec.h"),
      sourceOutput,
    });
    const source = fs.readFileSync(sourceOutput, "utf8");
    const decoder = source.match(
      /void SemanticDecoder::readcxxConstAddress\([^]*?\n\}/,
    )[0];
    assert.match(
      decoder,
      /std::remove_cvref_t<decltype\(self->subobjectPath\(\)\)>/,
    );
    assert.match(decoder, /decltype\(value\d+\)::value_type element\d+/);
    assert.doesNotMatch(decoder, /unsigned long/);
  } finally {
    fs.rmSync(output, { recursive: true, force: true });
  }
});

test("semantic decoding keeps owning types when accessors return pointers or views", () => {
  const plan = new PlanBuilder(index).build();
  const parameter = plan.symbols.find(
    (entity) => entity.short === "ParameterSymbol",
  );
  const pending = parameter.fields.find(
    (field) => field.name === "pendingDefaultArgument_",
  );
  assert.equal(pending.wire.k, "unique");
  assert.equal(pending.typeExpression, undefined);
  assert.equal(pending.cppType, "std::unique_ptr<cxx::PendingInstantiation>");

  const scope = plan.symbolBases.find(
    (entity) => entity.short === "ScopeSymbol",
  );
  const members = scope.fields.find((field) => field.name === "members_");
  assert.equal(members.typeExpression, undefined);
  assert.equal(members.cppType, "std::vector<cxx::Symbol*>");
});
