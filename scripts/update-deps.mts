import process from "node:process";
import { readFile } from "node:fs/promises";
import { execFile } from "node:child_process";

const PACKAGES = [
  "packages/cxx-frontend/package.json",
  "packages/cxx-gen-ast/package.json",
  "packages/cxx-gen-lsp/package.json",
  "packages/cxx-playground/package.json",
  "templates/cxx-browser-esm-vite/package.json",
  "templates/cxx-parse-esm/package.json",
  "templates/cxx-parse/package.json",
  "package.json",
];

interface PackageDeps {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}

function packagesAtLatestVersion(deps: Record<string, string> | undefined) {
  if (!deps) return [];
  return Object.keys(deps).map((dep) => `${dep}@latest`);
}

async function updatePackages({
  packages = PACKAGES,
}: { packages?: string[] } = {}) {
  for (const packagePath of packages) {
    const content = await readFile(packagePath, "utf-8");
    const packageJson = JSON.parse(content);
    const { dependencies, devDependencies } = packageJson as PackageDeps;

    // update the dev dependencies to the latest version
    const devDeps = packagesAtLatestVersion(devDependencies);
    if (devDeps.length) {
      console.log(`Updating devDependencies in ${packagePath}:`, devDeps);
      await execFile("npm", ["install", ...devDeps, "-D"], {
        cwd: process.cwd(),
      });
    }

    // update the dependencies to the latest version
    const deps = packagesAtLatestVersion(dependencies);
    if (deps.length) {
      console.log(`Updating dependencies in ${packagePath}:`, deps);
      await execFile("npm", ["install", ...deps], {
        cwd: process.cwd(),
      });
    }
  }
}

updatePackages().catch((error) => {
  console.error(error);
  process.exit(1);
});
