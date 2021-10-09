import typescript from "@rollup/plugin-typescript";
import { terser } from "rollup-plugin-terser";

export default {
    input: "src/index.ts",

    output: [
        {
            file: "dist/cxx-frontend.js",
            format: "esm",
            sourcemap: true,
        },
        {
            file: "dist/cxx-frontend.min.js",
            format: "esm",
            sourcemap: true,
            plugins: [terser()],
        },
    ],

    plugins: [
        typescript({ tsconfig: "./tsconfig.json" }),
    ]
};
