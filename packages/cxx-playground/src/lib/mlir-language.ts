import * as monaco from "monaco-editor"

export function registerMlirLanguage() {
  if (monaco.languages.getLanguages().some((lang) => lang.id === "mlir")) {
    return
  }

  monaco.languages.register({ id: "mlir" })

  monaco.languages.setMonarchTokensProvider("mlir", {
    defaultToken: "",
    tokenPostfix: ".mlir",

    keywords: [
      "module",
      "func",
      "return",
      "attributes",
      "loc",
      "dense",
      "unit",
      "true",
      "false",
      "yield",
      "step",
      "to",
      "for",
      "if",
      "else",
    ],

    typeKeywords: [
      "i1",
      "i8",
      "i16",
      "i32",
      "i64",
      "f16",
      "f32",
      "f64",
      "index",
      "none",
      "memref",
      "tensor",
      "vector",
      "tuple",
    ],

    tokenizer: {
      root: [
        // Comments
        [/\/\/.*$/, "comment"],

        [/[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*/, "keyword"],

        [/@[a-zA-Z_][a-zA-Z0-9_$]*/, "identifier"],

        [/%[a-zA-Z0-9_$]+/, "variable"],

        [/\^[a-zA-Z0-9_$]+/, "metatag"],

        [/![a-zA-Z_][a-zA-Z0-9_.<>!]*/, "type"],

        [/\b(i1|i8|i16|i32|i64|f16|f32|f64|index|none)\b/, "type"],

        [
          /[a-zA-Z_][a-zA-Z0-9_]*/,
          {
            cases: {
              "@keywords": "keyword",
              "@typeKeywords": "type",
              "@default": "identifier",
            },
          },
        ],

        [/"([^"\\]|\\.)*"/, "string"],

        [/\b\d+(\.\d+)?\b/, "number"],
        [/\b0x[0-9a-fA-F]+\b/, "number"],

        [/->/, "operator"],
        [/[:=,(){}<>[\]]/, "delimiter"],
      ],
    },
  })

  // Language configuration (brackets, comments)
  monaco.languages.setLanguageConfiguration("mlir", {
    comments: {
      lineComment: "//",
    },
    brackets: [
      ["{", "}"],
      ["[", "]"],
      ["(", ")"],
      ["<", ">"],
    ],
    autoClosingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
      { open: "<", close: ">" },
      { open: '"', close: '"' },
    ],
  })
}
