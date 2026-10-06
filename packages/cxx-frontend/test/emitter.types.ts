import {
  TraceEmitter,
  type DebugEmitterDelegate,
  type DebugDerivedKind,
  type DebugTypeRef,
  type EmitterDelegate,
  type ParameterAbi,
  type ParameterAbiKind,
} from "cxx-frontend";

const emitter: EmitterDelegate = new TraceEmitter();
const debug: DebugEmitterDelegate | undefined = emitter.debug;
const types: readonly DebugTypeRef[] = [1, 2];
const kind: DebugDerivedKind = "MemberPointer";
const extension: ParameterAbiKind = "ZeroExtend";
const resultAbi: ParameterAbi = {
  kind: extension,
  indirectType: 0,
  alignment: 0,
};
void resultAbi;

if (debug) {
  const unit = debug.compileUnit({
    file: "example.cc",
    directory: "/src",
    isCxx: true,
  });
  const type = debug.basicType({
    name: "int",
    sizeInBits: 32,
    encoding: "Signed",
  });
  debug.subroutineType(types);
  debug.derivedType({
    kind,
    baseType: type,
    sizeInBits: 64,
    alignInBits: 64,
    offsetInBits: 0,
    name: "",
    classType: 0,
  });
  debug.compositeType({
    kind: "Structure",
    name: "Record",
    location: { file: "example.cc", line: 1, column: 1 },
    scope: unit,
    baseType: 0,
    sizeInBits: 32,
    alignInBits: 32,
    elements: types,
    isScopedEnum: false,
  });
}

for (const record of new TraceEmitter().debug.types.values()) {
  if (record.kind === "Composite") {
    const elements: readonly DebugTypeRef[] = record.info.elements;
    void elements;
  }
  if (record.kind === "Basic") {
    const encoding: string = record.info.encoding;
    void encoding;
  }
}
