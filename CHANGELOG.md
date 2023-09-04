# Changelog

## [1.1.21](https://github.com/robertoraggi/cplusplus/compare/v1.1.20...v1.1.21) (2023-09-04)


### Bug Fixes

* AST creation of equal initializers ([9a224d5](https://github.com/robertoraggi/cplusplus/commit/9a224d5642e522adf5d1eb0e43a217811e31cafe))
* Expose TokenKind attributes to JavaScript ([335824c](https://github.com/robertoraggi/cplusplus/commit/335824c9a18f22f74add893290c4e8a56c83de37))

## [1.1.20](https://github.com/robertoraggi/cplusplus/compare/v1.1.19...v1.1.20) (2023-08-28)


### Bug Fixes

* Add AST node for await expressions ([76acaa8](https://github.com/robertoraggi/cplusplus/commit/76acaa86b72903027fd971ddbecf85e0db446afa))
* Add AST node for bitfield declarations ([a0d0b15](https://github.com/robertoraggi/cplusplus/commit/a0d0b15fcd16f7e086b84656f487e5d9aa7914dd))
* Add AST node for deduction guides ([89aa17e](https://github.com/robertoraggi/cplusplus/commit/89aa17eee4fbdc9544a576b5997212fe806952dc))
* Add AST node for new declarators ([4e9134e](https://github.com/robertoraggi/cplusplus/commit/4e9134e14400a44917e902e92c49f023636f62ad))
* Add AST node for new expressions ([e20770e](https://github.com/robertoraggi/cplusplus/commit/e20770ec6026abedbcbf9aa6010be395f613783a))
* Add AST node for noexcept specifiers ([588c08a](https://github.com/robertoraggi/cplusplus/commit/588c08a3cad7c79450562a8a8b86d18703cb98f4))
* Add AST node for the pack expansions ([3fa2c5b](https://github.com/robertoraggi/cplusplus/commit/3fa2c5bcf4ee4d0ca1bfbf38c2d234b145a5ebf6))
* Add AST node for yield expressions ([aacbed0](https://github.com/robertoraggi/cplusplus/commit/aacbed0ab239f7e1653617e77d4a5df99f454c1b))
* Add AST nodes to represent parameter packs ([f182718](https://github.com/robertoraggi/cplusplus/commit/f182718df7a14d2d0c2f80ca9346e51c8bba787f))
* Don't print bool attributes with default values ([e681866](https://github.com/robertoraggi/cplusplus/commit/e6818668132d59f2929e5c05b12ac867d14714d8))
* Expose the AST attributes to the JS API ([658afb7](https://github.com/robertoraggi/cplusplus/commit/658afb74f14871a9540b164c4917e25b491903b7))
* Parse of deduction guide in class scopes ([b99baa5](https://github.com/robertoraggi/cplusplus/commit/b99baa55febb00711be967753bb8e56d0c5fad3e))
* Renamed GCCAttribute to GccAttribute ([977ee14](https://github.com/robertoraggi/cplusplus/commit/977ee14636bd7309f7a209ef899f4f00c8d5da2d))
* Set the declaration of the for range statement ([50a4800](https://github.com/robertoraggi/cplusplus/commit/50a4800d0b85485dc518384950909305e43d4767))
* Set the variadic attribute of function prototypes ([8306e86](https://github.com/robertoraggi/cplusplus/commit/8306e86a661a8ecd6051c18e974d61c55047004c))
* Set the virt specifier attributes ([51d0e8f](https://github.com/robertoraggi/cplusplus/commit/51d0e8feec905a51b69629f98f6e02c696823789))

## [1.1.19](https://github.com/robertoraggi/cplusplus/compare/v1.1.18...v1.1.19) (2023-08-22)


### Bug Fixes

* Add AST for designated initializers ([f36c3f4](https://github.com/robertoraggi/cplusplus/commit/f36c3f4000c64bae5872721f4e0458854e82000d))
* Add AST node for underlying type specifier ([c716cc0](https://github.com/robertoraggi/cplusplus/commit/c716cc0dbe5011b836b75d8d7be1614b5cc6398f))
* Add AST node for using enum declaration ([bf5f22a](https://github.com/robertoraggi/cplusplus/commit/bf5f22a2421fc221c7d295be688c7de11904fb5b))
* Add missing attributes to the enumerator AST node ([bd2c9cd](https://github.com/robertoraggi/cplusplus/commit/bd2c9cd4d1a851e0b86432756af8c34ba76fb01d))
* Add structured binding declaration AST nodes ([3cbbc6a](https://github.com/robertoraggi/cplusplus/commit/3cbbc6aee2201c7d91db7a488c7ae87a39f9f832))
* Dump the AST in plain text instead of JSON ([ec19f4f](https://github.com/robertoraggi/cplusplus/commit/ec19f4fcabdca597aaef5f71915527682509cb71))
* Parse of class virt specifiers ([97cb51e](https://github.com/robertoraggi/cplusplus/commit/97cb51e3dc09875cdcd3e9e59e0a073455a1a146))
* Parse of inline for namespace definitions ([80a34cf](https://github.com/robertoraggi/cplusplus/commit/80a34cfd252db8e3a6b0b4ef459b0620cf910c3a))
* Print TokenKind in AST dump ([0583d39](https://github.com/robertoraggi/cplusplus/commit/0583d39b2510497d2aa0b2fe6d9df6cd98c36c01))
* Represent intiializers as expressions in the AST ([629ca3c](https://github.com/robertoraggi/cplusplus/commit/629ca3c31b3475a89f94cc1f6bcf9a9c25fe3f18))
* Set access and virt specifier to the base specifier ast nodes ([613a070](https://github.com/robertoraggi/cplusplus/commit/613a0709b34e25a5b28ddec470d2577c375dad32))
* Set access specifier of access declaration AST nodes ([dc7a545](https://github.com/robertoraggi/cplusplus/commit/dc7a545a2a3c614168a38c868ae3d67ba1367e1a))
* Set class key of class specifier AST nodes ([f610253](https://github.com/robertoraggi/cplusplus/commit/f61025375f9c49cb6f5c25a453266be52b51c981))
* Set class key of elaborated type specifiers ([a8db107](https://github.com/robertoraggi/cplusplus/commit/a8db10739bf97b379d4bb76846c532392406b872))
* Set location of the atomic type specifier ([8d8e07b](https://github.com/robertoraggi/cplusplus/commit/8d8e07becea096c21b556913755224ea4cf36283))
* Set the name of ctor member initializers ([35fdf21](https://github.com/robertoraggi/cplusplus/commit/35fdf21b1cab38ab142d3fdd5cc2614d39319123))
* stddef.h: add missing typedefs ([3ea986f](https://github.com/robertoraggi/cplusplus/commit/3ea986f83093275fddcc4889cab04235162a2ffa))
* Store the namespace name in the AST ([5df694b](https://github.com/robertoraggi/cplusplus/commit/5df694b0a49aeb4d6783c8859d459f85125fc8fd))
* Store the value of the boolean literals in the AST ([a2fbad8](https://github.com/robertoraggi/cplusplus/commit/a2fbad8614b80c1991407b60251d4bdb11c97983))

## [1.1.18](https://github.com/robertoraggi/cplusplus/compare/v1.1.17...v1.1.18) (2023-08-14)


### Bug Fixes

* Temporary workaround to parse post increments [#118](https://github.com/robertoraggi/cplusplus/issues/118) ([29611ee](https://github.com/robertoraggi/cplusplus/commit/29611ee98a78f250e60b8a0ede2a28ddfc061211))

## [1.1.17](https://github.com/robertoraggi/cplusplus/compare/v1.1.16...v1.1.17) (2023-07-12)


### Bug Fixes

* Add AST for the C++ attributes ([aedc6a7](https://github.com/robertoraggi/cplusplus/commit/aedc6a7ff4433e6294dc6c53fe4dcf99f31a8689))
* Add cxx-gen-ast ([4030a30](https://github.com/robertoraggi/cplusplus/commit/4030a30b2c2fa76ed1bc5f51e9c87695526a0afe))
* Add decltype(auto) type ([2a8ef93](https://github.com/robertoraggi/cplusplus/commit/2a8ef93a5e939cd03286587926261c7de15fb977))
* Build the emscripten bindings with EXPORT_ES6=1 ([#106](https://github.com/robertoraggi/cplusplus/issues/106)) ([91926d7](https://github.com/robertoraggi/cplusplus/commit/91926d7c185a0e1be18f78a59512ae79c16a6c66))
* Encode common literals ([c8c3d84](https://github.com/robertoraggi/cplusplus/commit/c8c3d84dbd0dec63632e11bc8bb6d620668cff32))
* Encode source locations ([58f7719](https://github.com/robertoraggi/cplusplus/commit/58f77198193e4f2f627f724bc6d0012dac18d43d))
* Expose the Lexer API to TypeScript ([1982d22](https://github.com/robertoraggi/cplusplus/commit/1982d222147091495501b343a342eba035fb28ea))
* Expose the preprocessor API to TypeScript ([90e1f8a](https://github.com/robertoraggi/cplusplus/commit/90e1f8a33792bd09b70d9e023d1b988684db5439))
* Expose TranslationUnit to TypeScript ([7792714](https://github.com/robertoraggi/cplusplus/commit/7792714f7758a17850428cf143a0534636baebfe))
* Implement __has_include_next ([3d59a07](https://github.com/robertoraggi/cplusplus/commit/3d59a073c91442343406441988d96646c52945ad))
* Initial work on the flatbuffers based serializer ([#89](https://github.com/robertoraggi/cplusplus/issues/89)) ([740e678](https://github.com/robertoraggi/cplusplus/commit/740e67811ccf5fe762f8b753c53ab3ed8487c2c5))
* Merge adjacent string literals ([80d5b9f](https://github.com/robertoraggi/cplusplus/commit/80d5b9fc85d44e92522734d3438f388a6aaf6d79))
* Print array types ([a9ae0d2](https://github.com/robertoraggi/cplusplus/commit/a9ae0d2f5978015f3089e99ad70b785b4e088d51))
* Print of const pointers ([f7b2be8](https://github.com/robertoraggi/cplusplus/commit/f7b2be8c05aa9170706f0fc37895a5828152ef49))
* Print of function types ([72d6c30](https://github.com/robertoraggi/cplusplus/commit/72d6c30d80ed6a4716d0bd925ceee6cb951d4c16))
* Renamed -fserialize-ast to -emit-ast ([ea24578](https://github.com/robertoraggi/cplusplus/commit/ea245783b4ea4b894a1c982211943603b85d2684))
* Serialize the identifiers ([c430188](https://github.com/robertoraggi/cplusplus/commit/c4301885e3d13b4822a788a5e8662c33aa3bc633))

## [1.1.16](https://github.com/robertoraggi/cplusplus/compare/v1.1.15...v1.1.16) (2023-06-11)


### Bug Fixes

* Add option to install the WASI sysroot ([301700a](https://github.com/robertoraggi/cplusplus/commit/301700a009c9a2bdeb299bc9472c06ce34385fa5))
* Add option to specify the path to the wasi sysroot ([a642d6b](https://github.com/robertoraggi/cplusplus/commit/a642d6b5f61df676405a56bef8f03a690775f96a))
* Build the emscripten bindings with WASM_BIGINT=1 ([4c4e4c9](https://github.com/robertoraggi/cplusplus/commit/4c4e4c99d799747593c7347b004db38600346c0e))
* Classify int64 and int128 integer types ([c40a7fc](https://github.com/robertoraggi/cplusplus/commit/c40a7fcb5401c35214439910c240271b1ec27e6f))
* Import wasi sysroot as an external dependency ([414e699](https://github.com/robertoraggi/cplusplus/commit/414e699afca324fcd7d9375d29d3ba6fcc5385dd))
* Link the wasi sysroot in the build directory ([e82a63d](https://github.com/robertoraggi/cplusplus/commit/e82a63d3c975ef324b7986484ebb769a52471a5f))
* Pretty print the JSON output AST ([70d8119](https://github.com/robertoraggi/cplusplus/commit/70d8119f6b6d2c14e796f8e89a0d612f07efdc81))
* Reduce size of the emscriten wasm binary ([3122c4d](https://github.com/robertoraggi/cplusplus/commit/3122c4d69bffa8d06a8455b602b8efad80e76830))

### [1.1.15](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.14...v1.1.15) (2023-06-04)


### Bug Fixes

* Add APIs to set/get the current working directory ([d0e7328](https://www.github.com/robertoraggi/cplusplus/commit/d0e732875ace72ab0e8b6f46ab621f3be403a3f6))
* Add support for Xcode 14 ([a8896dd](https://www.github.com/robertoraggi/cplusplus/commit/a8896dd2c6cba95cff5fa5b278fabde7d26ae50c))
* Add TemplateParameterList ([6ee556c](https://www.github.com/robertoraggi/cplusplus/commit/6ee556cbde7bee380f5ba19bb4de2d81e2f34656))
* Add wasm32-wasi toolchain definitions ([2656bc7](https://www.github.com/robertoraggi/cplusplus/commit/2656bc71aedb1523afb7505859fa97c9bf36d792))
* Build using wasm-sdk ([1f9bb71](https://www.github.com/robertoraggi/cplusplus/commit/1f9bb712ab00b12582ff82cb6312a22687123700))
* Build when C++ exceptions are not available ([8fb8cca](https://www.github.com/robertoraggi/cplusplus/commit/8fb8cca9f11edcb8ad1b16355b9806f8e6cd9f9b))
* Clean up parsing of template declarations ([99a45ae](https://www.github.com/robertoraggi/cplusplus/commit/99a45ae55a2d70f2c6c491029ab8c1e9d67cee02))
* Create symbols for the template type parameters ([9566308](https://www.github.com/robertoraggi/cplusplus/commit/9566308cdeb7df10be5e3939b9a2a8f411214ee1))
* Link with the LLVM libraries ([0948526](https://www.github.com/robertoraggi/cplusplus/commit/0948526bfdc0dd22b44e7c342583dd72f56fce4b))
* Modernize the code ([9ea5f41](https://www.github.com/robertoraggi/cplusplus/commit/9ea5f41eae3bd4620edc244592ac3968f1117c02))
* Optimize the size of the wasi binary ([ac3dd33](https://www.github.com/robertoraggi/cplusplus/commit/ac3dd33e540bd13b49b4daa0118db2b7e54a9e60))
* Path to macOS toolchain ([f5078d6](https://www.github.com/robertoraggi/cplusplus/commit/f5078d6a517cc8d5e73120b994a9aef0fe014046))
* **preproc:** Output GNU stile output directives ([4a97d38](https://www.github.com/robertoraggi/cplusplus/commit/4a97d38c0fb2e8a846438116d9edcacae2c15708))
* Remove filesystem from the public API ([cac00e3](https://www.github.com/robertoraggi/cplusplus/commit/cac00e30101fde6e962c21040f8e757c2472e0ed))
* Removed TypenamePackTypeParameterAST ([2b7989a](https://www.github.com/robertoraggi/cplusplus/commit/2b7989a5db315d1d99f2a08e03c18376b2f332fe))
* Use the utf8 unchecked api ([328ce98](https://www.github.com/robertoraggi/cplusplus/commit/328ce98fb6c7a7802168bb22d498acd71e21b1d7))

### [1.1.14](https://www.github.com/robertoraggi/cplusplus/compare/v1.1.13...v1.1.14) (2022-02-26)


### Bug Fixes

* Build with emscripten 3.1.4 ([4751aa7](https://www.github.com/robertoraggi/cplusplus/commit/4751aa7cd056dede1246835a433bcf7edbc9dc2e))
* **parser:** Create symbols for the template declarations ([0da8b46](https://www.github.com/robertoraggi/cplusplus/commit/0da8b461c4b0dd8ef8b5a9c7a43be9388a051ee0))
* Restructured the code base ([9e405a7](https://www.github.com/robertoraggi/cplusplus/commit/9e405a7a9e742566ac3e3695ebe092b37ff8252c))

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
