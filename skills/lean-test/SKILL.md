---
name: lean-test
description: Use when adding or repairing tests, verification, or acceptance evidence; make the smallest meaningful regression independent.
---

# Lean test

1. Write a focused failing regression for the observed behavior when practical.
2. Make it pass with the smallest implementation change.
3. Keep independent acceptance criteria and include malformed/negative input.
4. Run the focused test and the existing relevant suite.
5. Report exact commands, results, scope, and anything not exercised.
