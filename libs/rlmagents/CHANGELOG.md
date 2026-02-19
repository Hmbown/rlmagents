# Changelog

## [0.0.3](https://github.com/Hmbown/rlmagents/compare/rlmagents==0.0.2...rlmagents==0.0.3) (2026-02-19)

### Features

* bundle the CLI/TUI into `rlmagents` so `pip install rlmagents` provides the `rlmagents` command

## [0.0.2](https://github.com/Hmbown/rlmagents/compare/rlmagents==0.0.1...rlmagents==0.0.2) (2026-02-19)


### Features

* **cli:** add rlmagents harness mode and deepseek/minimax config ([eee0d77](https://github.com/Hmbown/rlmagents/commit/eee0d770a94318009000be08c9eba043d794cc6b))
* **cli:** make rlmagents primary command, harness, and branding ([99c8930](https://github.com/Hmbown/rlmagents/commit/99c893062318bb2933fd41877075fac105c98b7c))
* **infra:** decouple deepagents deps and wire multi-package releases ([#2](https://github.com/Hmbown/rlmagents/issues/2)) ([582d446](https://github.com/Hmbown/rlmagents/commit/582d446c84d79e5fd768edda8fe161ab2327c68f))
* **rlmagents:** add bootstrap and dogfood examples ([2105bc2](https://github.com/Hmbown/rlmagents/commit/2105bc2a83dd43ce92c7a8dd8dbee493d7987b81))
* **rlmagents:** add coding response schema defaults ([4d23084](https://github.com/Hmbown/rlmagents/commit/4d23084a4b8efd0cf57f43acfb52b37dc79aebe9))
* **rlmagents:** add context compaction and evidence provenance ([ab0fc3a](https://github.com/Hmbown/rlmagents/commit/ab0fc3a899b05f7e64229a170ecb91fd678521b9))
* **rlmagents:** add create_agent backend compatibility shim ([4b4f284](https://github.com/Hmbown/rlmagents/commit/4b4f28441a57e3bfcb4a52ca9dfbc416c29842d3))
* **rlmagents:** add standalone RLM-enhanced agent package ([2f2b1af](https://github.com/Hmbown/rlmagents/commit/2f2b1af69787a5e954d6b891781e630c9e25c3ab))
* **rlmagents:** add terminal-bench scenarios and launch checklist ([8c74d09](https://github.com/Hmbown/rlmagents/commit/8c74d09d7aee6efdd701656eec3fc2ae7a3cf304))
* **rlmagents:** auto-load cli [@file](https://github.com/file) mentions into rlm contexts ([af1cd85](https://github.com/Hmbown/rlmagents/commit/af1cd85fbc0d9fd19461a1cefb807eb64d16655b))
* **rlmagents:** default sub-query api to primary model ([cc71e28](https://github.com/Hmbown/rlmagents/commit/cc71e28990c5a70e315399c4bce6d84c10dbbb21))
* **rlmagents:** enforce edit discipline and execution guardrails ([b44b518](https://github.com/Hmbown/rlmagents/commit/b44b518de8eb1057e09ad5566502020b328ce76b))
* **rlmagents:** harden bootstrap model initialization retries ([931dece](https://github.com/Hmbown/rlmagents/commit/931dece8d7729ea62a37da8b2006c5b92620544e))
* **rlmagents:** standalone agent harness with incorporated middleware ([042a4b8](https://github.com/Hmbown/rlmagents/commit/042a4b86edc08e5bfb120002ea7cbe7f5aa7ebef))


### Bug Fixes

* **harbor:** integrate deepseek runtime fixes and tests ([7384372](https://github.com/Hmbown/rlmagents/commit/738437217e3e05465ed9c56407d89d67b5b23b28))

## [0.1.0] - 2025-02-16

### Added
- Initial release of rlmagents
- `create_rlm_agent()` factory function
- 23 RLM tools for context isolation and evidence tracking
- RLM middleware with auto-load for large tool results
- Cross-context search capability
- Sandboxed Python REPL with 100+ helpers
- Recipe pipeline support
- Comprehensive documentation
