# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2](https://github.com/Hmbown/rlmagents/compare/harbor==0.0.1...harbor==0.0.2) (2026-02-19)


### Features

* **cli:** add langsmith sandbox integration ([#1077](https://github.com/Hmbown/rlmagents/issues/1077)) ([7d17be0](https://github.com/Hmbown/rlmagents/commit/7d17be00b59e586c55517eaca281342e1a6559ff))
* **cli:** model switcher & arbitrary chat model support ([#1127](https://github.com/Hmbown/rlmagents/issues/1127)) ([28fc311](https://github.com/Hmbown/rlmagents/commit/28fc311da37881257e409149022f0717f78013ef))
* **cli:** Support VertexAI provider on the CLI ([#828](https://github.com/Hmbown/rlmagents/issues/828)) ([338da18](https://github.com/Hmbown/rlmagents/commit/338da186dadcb127dda4f5b2e94e246c0c00008d))
* **infra:** decouple deepagents deps and wire multi-package releases ([#2](https://github.com/Hmbown/rlmagents/issues/2)) ([582d446](https://github.com/Hmbown/rlmagents/commit/582d446c84d79e5fd768edda8fe161ab2327c68f))
* **infra:** ensure dep group version match for CLI ([#1316](https://github.com/Hmbown/rlmagents/issues/1316)) ([db05de1](https://github.com/Hmbown/rlmagents/commit/db05de1b0c92208b9752f3f03fa5fa54813ab4ef))


### Bug Fixes

* add missing error message for terminal process group in `HarborSandbox` ([#873](https://github.com/Hmbown/rlmagents/issues/873)) ([90c5381](https://github.com/Hmbown/rlmagents/commit/90c538190e8fa3827e847a2ad9018aeca36795e4))
* **cli:** improve clipboard copy/paste on macOS ([#960](https://github.com/Hmbown/rlmagents/issues/960)) ([3e1c604](https://github.com/Hmbown/rlmagents/commit/3e1c604474bd98ce1e0ac802df6fb049dd049682))
* **cli:** make `pyperclip` hard dep ([#985](https://github.com/Hmbown/rlmagents/issues/985)) ([0f5d4ad](https://github.com/Hmbown/rlmagents/commit/0f5d4ad9e63d415c9b80cd15fa0f89fc2f91357b)), closes [#960](https://github.com/Hmbown/rlmagents/issues/960)
* **cli:** revert, improve clipboard copy/paste on macOS ([#964](https://github.com/Hmbown/rlmagents/issues/964)) ([4991992](https://github.com/Hmbown/rlmagents/commit/4991992a5a60fd9588e2110b46440337affc80da))
* **deepagents:** refactor summarization middleware ([#1138](https://github.com/Hmbown/rlmagents/issues/1138)) ([e87001e](https://github.com/Hmbown/rlmagents/commit/e87001eace2852c2df47095ffd2611f09fdda2f5))
* **harbor:** accommodate harbor 0.1.33 ([#840](https://github.com/Hmbown/rlmagents/issues/840)) ([cbafb67](https://github.com/Hmbown/rlmagents/commit/cbafb6781818e68f608519174f27145c3cef9d0b))
* **harbor:** Command Injection via Unescaped file_path in Shell Echo Statements (awrite) ([#1323](https://github.com/Hmbown/rlmagents/issues/1323)) ([625a9ff](https://github.com/Hmbown/rlmagents/commit/625a9ff8467f617ea79790992b4e6d38c5c12636))
* **harbor:** CVE-2026-0994 ([#1033](https://github.com/Hmbown/rlmagents/issues/1033)) ([1c772b3](https://github.com/Hmbown/rlmagents/commit/1c772b3f89e4c6d9e4a12b8e148b3ae7bc5f2c5c))
* **harbor:** CVE-2026-24486 ([#1032](https://github.com/Hmbown/rlmagents/issues/1032)) ([1181551](https://github.com/Hmbown/rlmagents/commit/1181551fda5008c0e7407079cba95cd883390ab2))
* **harbor:** integrate deepseek runtime fixes and tests ([7384372](https://github.com/Hmbown/rlmagents/commit/738437217e3e05465ed9c56407d89d67b5b23b28))
* **harbor:** read example ID from dataset ([#858](https://github.com/Hmbown/rlmagents/issues/858)) ([9dcc7ce](https://github.com/Hmbown/rlmagents/commit/9dcc7cec34f2c1141410608d929364041431ad67))
* import rules ([#763](https://github.com/Hmbown/rlmagents/issues/763)) ([2c54297](https://github.com/Hmbown/rlmagents/commit/2c54297054ae68b2a98673d2f58d44652815467b))
* **infra:** change `release-please` component ([#1002](https://github.com/Hmbown/rlmagents/issues/1002)) ([cb572b9](https://github.com/Hmbown/rlmagents/commit/cb572b941f94b910cc5b5a49b93f246cd0eb02fa))
* **sdk,acp,harbor:** don't allow relative imports ([#882](https://github.com/Hmbown/rlmagents/issues/882)) ([acc8e3d](https://github.com/Hmbown/rlmagents/commit/acc8e3ddc496c106c833b667833c4a91f28db86e))
* **sdk:** BaseSandbox.ls_info() to return absolute paths ([#797](https://github.com/Hmbown/rlmagents/issues/797)) ([0234039](https://github.com/Hmbown/rlmagents/commit/0234039d49339929ccd5c74f7afe8ba48dc7928c))
* **sdk:** handle large payloads and headless container execution ([#872](https://github.com/Hmbown/rlmagents/issues/872)) ([f21ee57](https://github.com/Hmbown/rlmagents/commit/f21ee57df3f9946b1c9b2a0351496692f23e48d7))


### Reverted Changes

* **deepagents:** refactor summarization middleware ([#1172](https://github.com/Hmbown/rlmagents/issues/1172)) ([621c2be](https://github.com/Hmbown/rlmagents/commit/621c2be76a36df805f4c48991b6262a5a4ea8717))

## [Unreleased]
