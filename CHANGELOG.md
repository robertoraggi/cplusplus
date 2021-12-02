# Changelog

### [1.1.13](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.12...v1.1.13) (2021-12-02)


### Bug Fixes

* **parser:** Add checked/setChecked to the AST nodes ([52501b5](https://www.github.com/robertoraggi/cplusplus/commit/52501b548d1d7eeba1381007ce4eb43d7eef6025))
* **parser:** Implemented `__is_arithmetic` ([3e0c377](https://www.github.com/robertoraggi/cplusplus/commit/3e0c377ac3ef3c90fbf022c32fc4cac868a851ff))
* **parser:** Implemented `__is_compound` ([37b90ee](https://www.github.com/robertoraggi/cplusplus/commit/37b90ee794c336070ebbb5927f9897c11e2f701a))
* **parser:** Implemented `__is_floating_point` ([a165026](https://www.github.com/robertoraggi/cplusplus/commit/a1650269ee89d31faaf4b3be8c1712b3008a3004))
* **parser:** Implemented `__is_fundamental` ([4ef0374](https://www.github.com/robertoraggi/cplusplus/commit/4ef03745022e51ed51eded17e79667af38d215a3))
* **parser:** Implemented `__is_integral` ([1891675](https://www.github.com/robertoraggi/cplusplus/commit/1891675a8b3cbd29948a1b8094313adb18ecb67a))
* **parser:** Implemented `__is_object` ([b60f382](https://www.github.com/robertoraggi/cplusplus/commit/b60f382d4d484a02247ed71885eed551cecbe19d))
* **parser:** Implemented `__is_scalar` ([52d64d3](https://www.github.com/robertoraggi/cplusplus/commit/52d64d36572eea0d5c117c7561ee302dbf90ade7))

### [1.1.12](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.11...v1.1.12) (2021-11-28)


### Bug Fixes

* **parser:** Add AST nodes for unary and binary type traits ([3c76fa3](https://www.github.com/robertoraggi/cplusplus/commit/3c76fa328fa40dbc454bb169892371d94708b1f3))
* **parser:** Create AST node for built-in type traits ([5b7b0e3](https://www.github.com/robertoraggi/cplusplus/commit/5b7b0e39e6fe2c2a7317664910b9db731e8b43d4))
* **parser:** Fix `sizeof` of bool type ([46041b8](https://www.github.com/robertoraggi/cplusplus/commit/46041b860c9d81d0c6e6b4da8787d8cb0b570181))
* **parser:** Fix type of `nullptr` literals ([3e8b8be](https://www.github.com/robertoraggi/cplusplus/commit/3e8b8bedcd579dcb0129b778ed9b4d784ded7520))
* **parser:** Implemented `__is_function__` ([4d3e5da](https://www.github.com/robertoraggi/cplusplus/commit/4d3e5dab37563fa66b948f057757d28d54a077e4))
* **parser:** Implemented `__is_lvalue_reference`, `__is_rvalue_reference` and `__is_reference` ([d148471](https://www.github.com/robertoraggi/cplusplus/commit/d1484710aa049178b525d5af3b3a3c26eafc4bb3))
* **parser:** Implemented `__is_member_object_pointer` ([eeeece2](https://www.github.com/robertoraggi/cplusplus/commit/eeeece243072abf804290657edacc639b1e1c378))
* **parser:** Implemented `is_const` and `is_volatile` type traits ([918a680](https://www.github.com/robertoraggi/cplusplus/commit/918a680e6b94fce7d5e423b24a8162370be7c135))
* **parser:** Implemented the `__is_class` and the `__is_union` type traits ([bae22f3](https://www.github.com/robertoraggi/cplusplus/commit/bae22f39047f76a931ce79e7dffc92af8e65465b))
* **parser:** Implemented the `__is_enum` and the `__is_scoped_enum` type traits ([f9a05f2](https://www.github.com/robertoraggi/cplusplus/commit/f9a05f26b9df8441486451acf49fd816d55673f5))
* **parser:** Implemented the `__is_null_pointer` type traits ([1a4837f](https://www.github.com/robertoraggi/cplusplus/commit/1a4837f70a22e46f00a3bb689df5fa4cd043a987))
* **parser:** Implemented the `__is_pointer` type traits ([5f62b9a](https://www.github.com/robertoraggi/cplusplus/commit/5f62b9ab92eee196b777591bc9b58d9365c4ce32))
* **parser:** Implemented the `__is_signed` and the `__is_unsigned` type traits ([1f23b9f](https://www.github.com/robertoraggi/cplusplus/commit/1f23b9fece1d8b623e390bc792589fd68f95927e))
* **parser:** Implemented the `__is_void` type traits ([951b03a](https://www.github.com/robertoraggi/cplusplus/commit/951b03a7cbcfc8a8b55aa595835c037fe24c0a62))
* **preproc:** Remove newline from the diagnostic message raised from `#warning` directives ([a708e3c](https://www.github.com/robertoraggi/cplusplus/commit/a708e3c41a07a402112ba92e1cbcd8b68ea2f387))

### [1.1.11](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.10...v1.1.11) (2021-11-25)


### Bug Fixes

* **js:** Build the npm package using emsdk 2.0.34 ([10e384e](https://www.github.com/robertoraggi/cplusplus/commit/10e384e2c2bf32a18711602a427abaa9f3c6a2d0))

### [1.1.10](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.9...v1.1.10) (2021-11-25)


### Bug Fixes

* **parser:** Add support for the `__is_same_as` built-in function ([f80aa30](https://www.github.com/robertoraggi/cplusplus/commit/f80aa30db5cdbfbe6d5949e3d9c13218bb78e55a))
* **parser:** Add support for type aliases ([6934355](https://www.github.com/robertoraggi/cplusplus/commit/69343553df13ebcfb9a30743841ff0ea5a91d719))
* **parser:** Create the AST of using directives ([d43c447](https://www.github.com/robertoraggi/cplusplus/commit/d43c44731eb85898fefd1065afaefeb918a1716d))
* **parser:** Fix the type of logical expressions ([c9258dc](https://www.github.com/robertoraggi/cplusplus/commit/c9258dc82dcb88ef0f5d0002f0b75e48b8223f5e))
* **parser:** Initial support for `sizeof` and `alignof` expressions ([bd6a255](https://www.github.com/robertoraggi/cplusplus/commit/bd6a255ea7d93a535ad2a0fbe19710b2624b7fd7))
* **parser:** Initial support for the `decltype` specifier ([a24978c](https://www.github.com/robertoraggi/cplusplus/commit/a24978c4c62eef88bfe557383f4e26de3464b324))
* **parser:** Set type of conditional expressions ([bcf8776](https://www.github.com/robertoraggi/cplusplus/commit/bcf877687a087bf68cda4a26f787d4a07bc22bcf))

### [1.1.9](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.8...v1.1.9) (2021-11-22)


### Bug Fixes

* **ci:** Enable release-please action ([95245b0](https://www.github.com/robertoraggi/cplusplus/commit/95245b057cc3399b09cc21ef3c18aa83b5956aef))
* **cli:** Add `-fsyntax-only` to the cli ([50332b2](https://www.github.com/robertoraggi/cplusplus/commit/50332b2f90d32b19bfe1ecc421cc628c51be839e))
* **cli:** Tag the code generation options as experimental ([33439bc](https://www.github.com/robertoraggi/cplusplus/commit/33439bc3b67e1323feb7ac4d61ac67cf58afbb16))
* **ir:** Generate code for the logical_or expressions ([c1c0131](https://www.github.com/robertoraggi/cplusplus/commit/c1c013185a08e01c849e7201efc922b00a15773c))
* **js:** Add an example showing how to check the syntax of a C++ document ([3568e92](https://www.github.com/robertoraggi/cplusplus/commit/3568e92b1097de24a79604ac3d58b8c1768d5eae))
* **js:** Export the symbols defined in Token.js ([7d68898](https://www.github.com/robertoraggi/cplusplus/commit/7d68898984e846949b65bf813d9e914c2a3c1597))
* **parser:** Resolve type names ([798f475](https://www.github.com/robertoraggi/cplusplus/commit/798f475a91d0c54cebabb052851357505f09b223))

### [1.1.8](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.7...v1.1.8) (2021-11-14)


### Bug Fixes

* **parser:** Add classes to represent the utf string literals ([ec3575e](https://www.github.com/robertoraggi/cplusplus/commit/ec3575e082598d6c739a9ffe94e33c5f06253d26))
* **parser:** Add placeholder types ([da8de74](https://www.github.com/robertoraggi/cplusplus/commit/da8de7437a38a706f911babe24c0e791cf61b924))
* **parser:** Add symbol printer ([f65b25c](https://www.github.com/robertoraggi/cplusplus/commit/f65b25ccfc5000f135b3cee2cb37af4783ed121d))
* **parser:** Clean up Type ([feed0ae](https://www.github.com/robertoraggi/cplusplus/commit/feed0ae3dda1b333cfe4b168d08c89377b2c2a4f))
* **parser:** Remove qualifiers from pointer type ([e5f56c2](https://www.github.com/robertoraggi/cplusplus/commit/e5f56c28ab97b4e834d6cd0cb969c41d90a77655))
* **parser:** Support alternative operator spelling ([eab8b86](https://www.github.com/robertoraggi/cplusplus/commit/eab8b86189d2274ff168e66a513dc563e4030557))

### [1.1.7](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.6...v1.1.7) (2021-11-11)


### Bug Fixes

* **cmake:** Add cmake install targets ([b1e0853](https://www.github.com/robertoraggi/cplusplus/commit/b1e08535a013b0c16d6cc062a5f9dcbc33339701))
* **preproc:** Allow multiple definitions of the same macro ([2927a6e](https://www.github.com/robertoraggi/cplusplus/commit/2927a6e4dea375eed5bc841b4783ef245ace7996))

### [1.1.6](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.5...v1.1.6) (2021-11-09)


### Bug Fixes

* **preproc:** Add support for the __COUNTER__ built-in macro ([ecf2be8](https://www.github.com/robertoraggi/cplusplus/commit/ecf2be861146a48c02d6e4be68a984053756ef2c))
* **preproc:** Add support for the __DATE__ and __TIME__ built-in macros ([a0015a7](https://www.github.com/robertoraggi/cplusplus/commit/a0015a7dc146b2d4abe935031881f26ebc619e08))
* **preproc:** Add support for the __FILE__ built-in macro ([23be3a2](https://www.github.com/robertoraggi/cplusplus/commit/23be3a2b85b2f69f4de6a85da64e1e7658ac4250))
* **preproc:** Add support for the __LINE__ built-in macro ([ac7abf9](https://www.github.com/robertoraggi/cplusplus/commit/ac7abf9b0eab60c9c393f6d150b246db96840538))

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
