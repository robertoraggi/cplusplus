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

function configMatches({
  config,
  machine,
  os,
  compiler,
}: {
  config: Config;
  machine?: Machine;
  os?: OS;
  compiler?: Compiler;
}): boolean {
  const [configMachine, configOS, configCompiler] = config;
  return (
    (machine ? configMachine === machine : true) &&
    (os ? configOS === os : true) &&
    (compiler ? configCompiler === compiler : true)
  );
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
      "clang",
      [
        `--target=${target}`,
        "-E",
        "-dM",
        `-x${lang}`,
        `-std=${compiler}`,
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

async function main() {
  const predefinedMacrosByConfig: Record<string, Set<string>> = {};
  for (const config of configs) {
    const predefinedMacros = await getPredefinedMacros({ config });
    predefinedMacrosByConfig[config.join("-")] = predefinedMacros;
    console.log(
      `Config: ${config.join("-")} has ${predefinedMacros.size} predefined macros.`
    );
  }

  machines.forEach((machine) => {
    const machineConfigs = configs.filter((config) =>
      configMatches({ config, machine })
    );
    const predefinedMacrosForMachine = machineConfigs.map((config) => {
      return predefinedMacrosByConfig[config.join("-")];
    });
    return intersection(...predefinedMacrosForMachine);
  });
}

function intersection<T>(...sets: Set<T>[]): Set<T> {
  if (sets.length === 0) return new Set();
  return sets.reduce((acc, set) => {
    const intersection = new Set<T>();
    for (const item of set) {
      if (acc.has(item)) {
        intersection.add(item);
      }
    }
    return intersection;
  }, sets[0]);
}

await main();
