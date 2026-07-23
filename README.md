UEFI environment evaluation
===========================

"Figure 1: cartpole as expected; breakout ain't"

Acknowledgement(s)
------------------

- [No Later Than June 1, 2026](https://pi.dev/models/opencode/big-pickle)

Note(s)
-------

Why isn't pufferlib's trained breakout environment evaluating as expected? Same checkpoint, but different outcome.

---

real-time edit of sim or level of environment?

---

Citation needed: https://github.com/PufferAI/PufferLib/pulls?q=--float

> The atmosphere itself is a filter, acoustically speaking - distance to lightning would need to be simulated, yes? (not that optical attenuation doesn't occur)
> — [Jason G.](https://robertsspaceindustries.com/citizens/Perceus) in [`#sc-testing-chat`](https://robertsspaceindustries.com/spectrum/community/SC/lobby/38230/message/65933990)

---

Perhaps side-by-side evaluation, eg. with and without --float instead of `./build.sh env --uefi-kernel --float` producing separate `.elf`.

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
