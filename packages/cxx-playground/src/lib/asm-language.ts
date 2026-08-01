import * as monaco from "monaco-editor"

export function registerAsmLanguage() {
  if (monaco.languages.getLanguages().some((lang) => lang.id === "asm")) {
    return
  }

  monaco.languages.register({ id: "asm" })

  monaco.languages.setMonarchTokensProvider("asm", {
    defaultToken: "",
    tokenPostfix: ".asm",
    ignoreCase: false,

    tokenizer: {
      root: [
        [/[#;].*$/, "comment"],
        [/\/\/.*$/, "comment"],

        [/^\s*[a-zA-Z_.$][\w.$]*(?=:)/, "type"],

        [/\.[a-zA-Z_][\w]*/, "keyword"],

        [/\b[a-z][a-z0-9_]*\.[a-z][\w]*\b/, "keyword"],

        [/%[a-zA-Z0-9_]+/, "variable"],

        [/"([^"\\]|\\.)*"/, "string"],

        [/-?\b0x[0-9a-fA-F]+\b/, "number"],
        [/-?\b\d+(\.\d+)?\b/, "number"],

        [/[a-zA-Z_.$][\w.$]*/, "identifier"],

        [/[(),:=]/, "delimiter"],
      ],
    },
  })

  monaco.languages.setLanguageConfiguration("asm", {
    comments: {
      lineComment: "#",
    },
    brackets: [["(", ")"]],
    autoClosingPairs: [
      { open: "(", close: ")" },
      { open: '"', close: '"' },
    ],
  })
}
