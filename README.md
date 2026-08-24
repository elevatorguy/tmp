UEFI environment evaluation
===========================
[insert video here]

"Figure 1: cartpole as expected; breakout ain't"

(pufferlib [4.0](https://github.com/PufferAI/PufferLib/blob/4.0/README.md#original))

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
