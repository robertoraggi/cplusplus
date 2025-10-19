import { $, path, fs, glob } from "zx";
import {
  parse,
  printParseErrorCode,
  modify,
  applyEdits,
  type ParseError,
} from "jsonc-parser";

$.verbose = true;

const workspacePath = path.join(__dirname, "../");
console.log(`workspace path: ${workspacePath}`);

const unitTestsPath = path.join(workspacePath, "tests/unit_tests");

const ccFiles = await glob(`${unitTestsPath}/**/*.cc`);
console.log(ccFiles);

function makeLaunchConfig({ file }: { file: string }) {
  const kind = path.basename(path.dirname(file));
  const name = path.basename(file);

  return {
    name: `${name} [${kind}]`,
    type: "lldb-dap",
    request: "launch",
    program: "${workspaceRoot}/build/src/frontend/cxx",
    args: [path.relative(workspacePath, file)],
    env: [],
    cwd: "${workspaceRoot}",
  };
}

const testLaunchConfigs = ccFiles.map((file) => makeLaunchConfig({ file }));

console.log(testLaunchConfigs);

const launchConfigPath = path.join(workspacePath, ".vscode/launch.json");
const launchConfigContent = await fs.readFile(launchConfigPath, "utf-8");

// Parse JSON with comments
let errors: ParseError[] = [];
const launchConfig = parse(launchConfigContent, errors, {
  allowTrailingComma: true,
});

if (errors.length > 0) {
  errors.forEach((error) => {
    console.error(
      `Error: ${printParseErrorCode(error.error)} at offset ${error.offset}`
    );
  });
  throw new Error("Failed to parse launch.json");
}

// Filter out old test configurations
const existingConfigs = launchConfig.configurations || [];
const filteredConfigs = existingConfigs.filter(
  (config: any) => !config.name.match(/ \[.*\]$/)
);

// Append new configurations
const updatedConfigs = filteredConfigs.concat(testLaunchConfigs);

const edits = modify(launchConfigContent, ["configurations"], updatedConfigs, {
  formattingOptions: {},
});
const updatedLaunchConfigContent = applyEdits(launchConfigContent, edits);

// Write back to launch.json
await fs.writeFile(launchConfigPath, updatedLaunchConfigContent);

console.log("Updated launch.json with new test configurations.");
