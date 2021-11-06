# Changelog

### [1.1.5](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.4...v1.1.5) (2021-11-06)


### Bug Fixes

* **ci:** Add support for multi-config cmake projects ([4ae48c3](https://www.github.com/robertoraggi/cplusplus/commit/4ae48c3dbb07f6388685ebe9711a0e1e29db716d))
* **parser:** Parse initialized function pointers ([00fb809](https://www.github.com/robertoraggi/cplusplus/commit/00fb809d2fb6d830da505789225e727a38de2df8))
* **parser:** Validate the reported diagnostic messages ([950f1d1](https://www.github.com/robertoraggi/cplusplus/commit/950f1d1ded9b460d6907ffb4e4c3549b4a999c0b)), closes [#38](https://www.github.com/robertoraggi/cplusplus/issues/38)
* **preproc:** Fix macro redefinition ([9d5b07f](https://www.github.com/robertoraggi/cplusplus/commit/9d5b07f6f0384e467bfa8c21510d860abec4aede))

### [1.1.4](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.3...v1.1.4) (2021-10-30)


### Bug Fixes

* **parser:** Create AST for simple for-range declarations ([3e51ae7](https://www.github.com/robertoraggi/cplusplus/commit/3e51ae74215bc7a39bfa07c0f7ca9043578acad1))
* **preproc:** Add support for comment handlers. ([95c5f08](https://www.github.com/robertoraggi/cplusplus/commit/95c5f086bfab9b127be4ae1937c0264891a0dbff))
* **preproc:** Avoid resolving files when access to the fs is not allowed ([d5d74b7](https://www.github.com/robertoraggi/cplusplus/commit/d5d74b71bf33604113214e809ae9a52b94c2dae5))

### [1.1.3](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.2...v1.1.3) (2021-10-29)


### Bug Fixes

* **parser:** Fix parsing of placeholder type specifiers ([3b3f273](https://www.github.com/robertoraggi/cplusplus/commit/3b3f273108faf7e463390e13f1460fe83e0d1f6f))
* **parser:** Update the system search paths of the mocOS toolchain ([85d9226](https://www.github.com/robertoraggi/cplusplus/commit/85d92268b02e92d1016fa37d44713bee7677cfef))

### [1.1.2](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.1...v1.1.2) (2021-10-27)


### Bug Fixes

* **parser:** Create AST for modules ([ce61309](https://www.github.com/robertoraggi/cplusplus/commit/ce6130902314e36773cd67007b9e284929235a9d))
* **parser:** Create AST for qualified type names ([cd6c437](https://www.github.com/robertoraggi/cplusplus/commit/cd6c4373eea614c85747b8200f495a0171a9fed4))
* **parser:** Store the location of the export token of compound export declarations ([81a4bfd](https://www.github.com/robertoraggi/cplusplus/commit/81a4bfd11ef69e43d397ba580fc6b5f26778f0a1))

### [1.1.1](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.0...v1.1.1) (2021-10-25)


### Bug Fixes

* **docs:** Add example to README.md ([255ba0b](https://www.github.com/robertoraggi/cplusplus/commit/255ba0b87eb209d93957b6b852ce3bb595a85e4e))
* **js:** Add missing .js extension in import directive ([9a48ade](https://www.github.com/robertoraggi/cplusplus/commit/9a48adeefa374084ecc34e9c1d7f070c38504ddc))

## [1.1.0](https://www.github.com/robertoraggi/cplusplus/compare/v1.0.0...v1.1.0) (2021-10-25)


### Features

* **js:** Add TokenKind and Token.getKind() ([f1f11a8](https://www.github.com/robertoraggi/cplusplus/commit/f1f11a8d442e8f489a826f5540ccdeeac96567f4))
* **js:** Added ASTCursor ([146b166](https://www.github.com/robertoraggi/cplusplus/commit/146b166aa886e0b5468e4314bee75f3541d1272a))


### Bug Fixes

* **js:** Extend ASTSlot to return the kind and number of slots ([4d3f313](https://www.github.com/robertoraggi/cplusplus/commit/4d3f3134b795dc44d72efedc8cb2a4105b40f2a6))
* **js:** Made AST.getEndLocation() inclusive ([5849ef3](https://www.github.com/robertoraggi/cplusplus/commit/5849ef39579ea2adea6b3bd3bdc76f6574768d74))
* **parser:** Store the location of the if-token in the AST ([050a877](https://www.github.com/robertoraggi/cplusplus/commit/050a87794c6954946132fe4d7bd0ae2bdd0b4d1e))
* **parser:** Test for invalid source locations ([878a4aa](https://www.github.com/robertoraggi/cplusplus/commit/878a4aaedc68e74d2b2bd4f92f018a32e314489c))

## 1.0.0 (2021-10-23)


### Features

* **json:** Improved support to convert the AST to JSON ([d15d466](https://www.github.com/robertoraggi/cplusplus/commit/d15d4669629a7cc0347ea7ed60697b55cbd44523))
* **js:** Use rollup to build the JavaScript bindings. ([b99eb55](https://www.github.com/robertoraggi/cplusplus/commit/b99eb5570c2551302a1488d8b85e26ba29bb4bfc))


### Bug Fixes

* **docs:** Update the example code in the README.md file ([459e837](https://www.github.com/robertoraggi/cplusplus/commit/459e83795b3bee552f192792e5126c7efa74a9c2))
* **js:** Build with NO_DYNAMIC_EXECUTION to allow using the library in trusted browser environments ([97eb589](https://www.github.com/robertoraggi/cplusplus/commit/97eb5899d93e582a9c0b42e70a58fc87af45142e))
* **lexer:** Scan the characters using the UTF-8 unchecked API ([654e227](https://www.github.com/robertoraggi/cplusplus/commit/654e2275cec722517fc9bd72324bd3f7c45baf51))
* **parser:** Create the AST for concept requirements ([5600bb4](https://www.github.com/robertoraggi/cplusplus/commit/5600bb4ec47b8e3a3f6a28ebc4d9474c85ebc849))
* **parser:** Create the AST for requires-clause ([23aa6a3](https://www.github.com/robertoraggi/cplusplus/commit/23aa6a3cf0f6b0dcee81a714b085a1f340ee6b43))
* **parser:** Fix parsing of parameters in template declarations ([1dc66ce](https://www.github.com/robertoraggi/cplusplus/commit/1dc66cea50922c637870555fca6e71b47b3f33e0))
* **parser:** Store the AST of the template parameters ([f0da7a0](https://www.github.com/robertoraggi/cplusplus/commit/f0da7a0146f69aabff04635a5c8c0edcd4ce4e5b))
* **preproc:** Access the hidesets using transparent comparisons ([96b484d](https://www.github.com/robertoraggi/cplusplus/commit/96b484d578f21be9b930ce765cfb7368beb3cb39))
* **preproc:** Reduced the number of the temporary tokens ([d092959](https://www.github.com/robertoraggi/cplusplus/commit/d092959ab9964b72aff8ace8da8b7e456369498d))
* **preproc:** Trim space characters at the end of the text lines ([03e2518](https://www.github.com/robertoraggi/cplusplus/commit/03e25183ebdb8c10958fff1a7426ec717084762a))
