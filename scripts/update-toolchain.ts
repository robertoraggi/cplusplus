import { execFile } from "node:child_process";

const machines = ["aarch64", "x86_64", "wasm32"] as const;
const oses = ["linux", "macosx", "windows", "wasi"] as const;
const compilers = ["c23", "c++26"];

type Machine = (typeof machines)[number];
type OS = (typeof oses)[number];
type Compiler = (typeof compilers)[number];
type Config = [machine: Machine, os: OS, compiler: Compiler];

const configs: Config[] = [
  ["aarch64", "linux", "c++26"],
  ["aarch64", "linux", "c23"],
  ["aarch64", "macosx", "c++26"],
  ["aarch64", "macosx", "c23"],
  ["aarch64", "windows", "c++26"],
  ["aarch64", "windows", "c23"],
  ["wasm32", "wasi", "c++26"],
  ["wasm32", "wasi", "c23"],
  ["x86_64", "linux", "c++26"],
  ["x86_64", "linux", "c23"],
  ["x86_64", "macosx", "c++26"],
  ["x86_64", "macosx", "c23"],
  ["x86_64", "windows", "c++26"],
  ["x86_64", "windows", "c23"],
];

async function main() {
  const { predefinedMacrosByConfig } = await getConfigPredefinedMacros();

  const getMacros = (config: Config): Set<string> => {
    const configKey = config.join("-");
    const macros = predefinedMacrosByConfig.get(configKey);
    if (!macros) {
      throw new Error(`No predefined macros found for config: ${configKey}`);
    }
    return macros;
  };

  const {
    commonMacros: linuxCommonMacros,
    machineMacros: linuxMachineMacros,
    cMacros: linuxCMacros,
    cxxMacros: linuxCxxMacros,
  } = genToolchain({
    os: "linux",
    machineVariants: ["aarch64", "x86_64"],
    configs,
    getMacros,
  });

  const {
    commonMacros: macosxCommonMacros,
    machineMacros: macosxMachineMacros,
    cMacros: macosxCMacros,
    cxxMacros: macosxCxxMacros,
  } = genToolchain({
    os: "macosx",
    machineVariants: ["aarch64", "x86_64"],
    configs,
    getMacros,
  });

  const {
    commonMacros: windowsCommonMacros,
    machineMacros: windowsMachineMacros,
    cMacros: windowsCMacros,
    cxxMacros: windowsCxxMacros,
  } = genToolchain({
    os: "windows",
    machineVariants: ["aarch64", "x86_64"],
    configs,
    getMacros,
  });

  const {
    commonMacros: wasiCommonMacros,
    machineMacros: wasiMachineMacros,
    cMacros: wasiCMacros,
    cxxMacros: wasiCxxMacros,
  } = genToolchain({
    os: "wasi",
    machineVariants: ["wasm32"],
    configs,
    getMacros,
  });

  const commonMacrosByOS: Map<OS, Set<string>> = new Map([
    ["linux", linuxCommonMacros],
    ["macosx", macosxCommonMacros],
    ["windows", windowsCommonMacros],
    ["wasi", wasiCommonMacros],
  ]);

  const machineMacrosByOS: Map<OS, Map<Machine, Set<string>>> = new Map([
    ["linux", linuxMachineMacros],
    ["macosx", macosxMachineMacros],
    ["windows", windowsMachineMacros],
    ["wasi", wasiMachineMacros],
  ]);

  const cCompilerMacrosByOS: Map<OS, Set<string>> = new Map([
    ["linux", linuxCMacros],
    ["macosx", macosxCMacros],
    ["windows", windowsCMacros],
    ["wasi", wasiCMacros],
  ]);

  const cxxCompilerMacrosByOS: Map<OS, Set<string>> = new Map([
    ["linux", linuxCxxMacros],
    ["macosx", macosxCxxMacros],
    ["windows", windowsCxxMacros],
    ["wasi", wasiCxxMacros],
  ]);

  const commonMacros = intersection(commonMacrosByOS.values());
  const cCompilerCommonMacros = intersection(cCompilerMacrosByOS.values());
  const cxxCompilerCommonMacros = intersection(cxxCompilerMacrosByOS.values());

  oses.forEach((os) => {
    commonMacrosByOS.set(
      os,
      commonMacrosByOS.get(os)!.difference(commonMacros)
    );

    cCompilerMacrosByOS.set(
      os,
      cCompilerMacrosByOS.get(os)!.difference(cCompilerCommonMacros)
    );

    cxxCompilerMacrosByOS.set(
      os,
      cxxCompilerMacrosByOS.get(os)!.difference(cxxCompilerCommonMacros)
    );
  });

  const out: string[] = [];
  const emit = (text: string = "") => out.push(text);

  emitMacros({
    common: true,
    macros: commonMacros,
    emit,
  });

  emitMacros({
    common: true,
    macros: cCompilerCommonMacros,
    compiler: "c23",
    emit,
  });

  emitMacros({
    common: true,
    macros: cxxCompilerCommonMacros,
    compiler: "c++26",
    emit,
  });

  commonMacrosByOS.forEach((macros, os) => {
    emitMacros({
      common: true,
      macros,
      os,
      emit,
    });
  });

  machineMacrosByOS.forEach((macrosByMachine, os) => {
    macrosByMachine.forEach((macros, machine) => {
      emitMacros({
        common: false,
        macros,
        os,
        machine,
        emit,
      });
    });
  });

  cCompilerMacrosByOS.forEach((macros, os) => {
    emitMacros({
      common: false,
      macros,
      os,
      compiler: "c23",
      emit,
    });
  });

  cxxCompilerMacrosByOS.forEach((macros, os) => {
    emitMacros({
      common: false,
      macros,
      os,
      compiler: "c++26",
      emit,
    });
  });

  console.log(out.join("\n"));
}

function intersection(sets: Iterable<Set<string>>): Set<string> {
  let result: Set<string> | undefined;
  for (const set of sets) {
    if (!result) {
      result = new Set(set);
    } else {
      result = result.intersection(set);
    }
  }
  return result ?? new Set();
}

function genToolchain({
  os,
  configs,
  machineVariants,
  getMacros,
}: {
  os: OS;
  configs: Config[];
  machineVariants: readonly Machine[];
  getMacros: (config: Config) => Set<string>;
}) {
  const os_configs = configs.filter((config) => config[1] === os);

  let commonMacros: Set<string> | undefined;
  os_configs.forEach((config) => {
    const macros = getMacros(config);
    if (!commonMacros) {
      commonMacros = new Set(macros);
    } else {
      commonMacros = commonMacros.intersection(macros);
    }
  });

  if (!commonMacros) {
    throw new Error(`No common macros found for OS: ${os}`);
  }

  let [first_machine, second_machine] = machineVariants;

  if (!first_machine) {
    throw new Error(`No first machine found for OS: ${os}`);
  }

  if (!second_machine) {
    second_machine = first_machine;
  }

  let machine1_c = getMacros([first_machine, os, "c23"]);
  let machine1_cxx = getMacros([first_machine, os, "c++26"]);
  let machine2_c = getMacros([second_machine, os, "c23"]);
  let machine2_cxx = getMacros([second_machine, os, "c++26"]);

  machine1_c = machine1_c.difference(commonMacros);
  machine1_cxx = machine1_cxx.difference(commonMacros);
  machine2_c = machine2_c.difference(commonMacros);
  machine2_cxx = machine2_cxx.difference(commonMacros);

  const machine_diff = machine1_c.symmetricDifference(machine2_c);
  const lang_diff = machine1_c.symmetricDifference(machine1_cxx);
  const first_machine_macros = machine1_c.intersection(machine_diff);
  const second_machine_macros = machine2_c.intersection(machine_diff);
  const cMacros = machine1_c.intersection(lang_diff);
  const cxxMacros = machine1_cxx.intersection(lang_diff);

  const machineMacros = new Map<Machine, Set<string>>();
  machineMacros.set(first_machine, first_machine_macros);
  if (first_machine !== second_machine) {
    machineMacros.set(second_machine, second_machine_macros);
  }

  return {
    commonMacros,
    machineMacros,
    cMacros,
    cxxMacros,
  };
}

function emitMacros({
  macros,
  common,
  machine,
  os,
  compiler,
  emit,
}: {
  macros: Set<string>;
  common?: boolean;
  machine?: Machine;
  os?: OS;
  compiler?: Compiler;
  emit: (text?: string) => void;
}) {
  let func = "void Toolchain::add";
  if (common) func += "Common";
  if (os != undefined) func += `${osToString(os)}`;
  if (machine != undefined) func += `${machineToString(machine)}`;
  if (compiler != undefined) func += `${compilerToString(compiler)}`;
  func += "Macros() {";
  emit();
  emit(func);
  macros.forEach((macro) => {
    let { name, value } = parseMacro(macro);
    if (name === "__VERSION__") {
      value = '\\"cxx-frontend\\"';
    }
    emit(`  defineMacro("${name}", "${value}");`);
  });
  emit("}");
}

async function getConfigPredefinedMacros() {
  const predefinedMacrosByConfig = new Map<string, Set<string>>();

  for (const config of configs) {
    const configKey = config.join("-");
    const predefinedMacros = await getPredefinedMacros({ config });
    predefinedMacrosByConfig.set(configKey, predefinedMacros);
  }

  return {
    predefinedMacrosByConfig,
  };
}

async function getPredefinedMacros({
  config,
}: {
  config: Config;
}): Promise<Set<string>> {
  const [machine, os, compiler] = config;

  const lang = compiler.startsWith("c++") ? "c++" : "c";
  const target = `${machine}-${os}`;

  return new Promise((resolve, reject) => {
    execFile(
      "/opt/homebrew/opt/llvm/bin/clang",
      [
        `--target=${target}`,
        "-E",
        "-dM",
        `-x${lang}`,
        `-std=${compiler}`,
        "-fno-blocks",
        "/dev/null",
      ],
      (err, stdout) => {
        if (err) {
          reject(err);
          return;
        }

        const predefinedMacros = stdout
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);

        resolve(new Set(predefinedMacros));
      }
    );
  });
}

function osToString(os: OS): string {
  switch (os) {
    case "linux":
      return "Linux";
    case "macosx":
      return "MacOS";
    case "windows":
      return "Windows";
    case "wasi":
      return "WASI";
    default:
      throw new Error(`Unknown OS: ${os}`);
  }
}

function compilerToString(compiler: Compiler): string {
  switch (compiler) {
    case "c23":
      return "C23";
    case "c++26":
      return "Cxx26";
    default:
      throw new Error(`Unknown compiler: ${compiler}`);
  }
}

function machineToString(machine: Machine): string {
  switch (machine) {
    case "aarch64":
      return "AArch64";
    case "x86_64":
      return "X86_64";
    case "wasm32":
      return "Wasm32";
    default:
      throw new Error(`Unknown machine: ${machine}`);
  }
}

function parseMacro(macro: string) {
  const rx = /^#define\s+([^\s]+)\s*(.*)$/;
  const match = macro.match(rx);

  if (!match) {
    throw new Error(`Invalid macro format: ${macro}`);
  }

  function escaped(value: string) {
    return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
  }

  const [, name, value] = match;

  if (!name) {
    throw new Error(`Macro name is empty in: ${macro}`);
  }

  return {
    name,
    value: escaped(value!),
  };
}

await main();
