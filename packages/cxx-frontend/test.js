const { Parser } = require(".");
const fs = require("fs");
const process = require("process");

const path = process.argv[2];
const source = fs.readFileSync(path).toString();

const parser = new Parser({ path, source });

parser.parse();

console.log("diagnostics", parser.getDiagnostics());

console.log("ast", parser.getAST());

parser.dispose();
