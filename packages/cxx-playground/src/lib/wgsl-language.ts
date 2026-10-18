import * as monaco from "monaco-editor"

export function registerWgslLanguage() {
  if (monaco.languages.getLanguages().some((lang) => lang.id === "wgsl")) {
    return
  }

  monaco.languages.register({ id: "wgsl" })

  monaco.languages.setMonarchTokensProvider("wgsl", {
    defaultToken: "",
    tokenPostfix: ".wgsl",

    keywords: [
      "alias",
      "break",
      "case",
      "const",
      "continue",
      "continuing",
      "default",
      "discard",
      "else",
      "enable",
      "fn",
      "for",
      "if",
      "let",
      "loop",
      "override",
      "requires",
      "return",
      "struct",
      "switch",
      "var",
      "while",
    ],

    types: [
      "array",
      "atomic",
      "bool",
      "f16",
      "f32",
      "i32",
      "mat2x2",
      "mat3x3",
      "mat4x4",
      "ptr",
      "sampler",
      "u32",
      "vec2",
      "vec3",
      "vec4",
      "vec2f",
      "vec3f",
      "vec4f",
      "vec2u",
      "vec3u",
      "vec4u",
      "vec2i",
      "vec3i",
      "vec4i",
    ],

    tokenizer: {
      root: [
        [/\/\/.*$/, "comment"],
        [/@[a-zA-Z_]\w*/, "annotation"],
        [/\b(true|false)\b/, "keyword"],
        [/-?0[xX][0-9a-fA-F.]+(?:[pP][-+]?\d+)?[fhiu]?/, "number"],
        [/-?\d+\.\d*(?:[eE][-+]?\d+)?[fh]?/, "number"],
        [/-?\d+[fhiu]?/, "number"],
        [
          /[a-zA-Z_]\w*/,
          {
            cases: {
              "@keywords": "keyword",
              "@types": "type",
              "@default": "identifier",
            },
          },
        ],
        [/[{}()[\]<>]/, "@brackets"],
        [/[;,.:]/, "delimiter"],
      ],
    },
  })

  monaco.languages.setLanguageConfiguration("wgsl", {
    comments: { lineComment: "//" },
    brackets: [
      ["{", "}"],
      ["[", "]"],
      ["(", ")"],
    ],
  })
}
