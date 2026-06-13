UEFI environment evaluation
===========================

"Figure 1: cartpole as expected; breakout ain't" - may add third

Note(s)
-------

real-time edit of sim or level of environment?

<details><summary>May 8th, 2026</summary>

Why isn't pufferlib's trained breakout environment evaluating as expected?

Same checkpoint, but different outcome; yea? (may need to edit captures; --local and --uefi-kernel - for side-by-side comparison)

Feature Idea: `--local` playback in `--uefi-kernel` (instead of step & render)

Save 10000 frames to C array for showcase with runtime loop; perhaps render with a noise filter to emphasize prerecorded status - enabling `--local` and `--uefi-kernel` environment eval comparison. The case of cartpole is different, as the visual differences may seem non-distinguisable; playback allows determining how close `--local` and `--uefi-kernel` really are - frame-by-frame, perhaps once kernel input is feasible.

How accessible is raylib's framebuffer? Without access, perhaps save uefi-kernel playback for replay in local OS; preference during analysis as OS-less, though - pure observation.

Pufferlib has `--slowly`, but file format parsing/decode isn't preferable at present commit - 610dcc9; may need memcpy instead of pixel-by-pixel decode and draw - raw format needs to match Graphics Output Protocol ARGB framebuffer.

</details>

Citation needed: https://github.com/PufferAI/PufferLib/pulls?q=--float

> The atmosphere itself is a filter, acoustically speaking - distance to lightning would need to be simulated, yes? (not that optical attentuation doesn't occur)
> — Jason Garner ([env idea](https://robertsspaceindustries.com/spectrum/community/SC/lobby/38230/message/65933990) or not RL?)

TODO: side-by-side evaluation, eg. with and without --float instead of `./build.sh env --uefi-kernel --float` producing separate `.elf`.

Issue(s)
--------

[fe7cce2](https://github.com/elevatorguy/tmp/commit/fe7cce2)

---

![figure](https://pufferai.github.io/source/resource/header.png)

[![Discord](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
