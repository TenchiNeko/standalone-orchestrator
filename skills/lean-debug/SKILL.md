---
name: lean-debug
description: Use when debugging a reproducible failure, boundary, or unexpected runtime result; trace evidence before changing code.
---

# Lean debug

1. Reproduce the reported behavior with the smallest safe fixture.
2. Trace the shared boundary and its callers; separate observations from
   hypotheses.
3. Change one supported cause at a time, keeping a rollback.
4. Verify the failure regression, the intended behavior, and nearby callers.
5. Record uncertainty and any untested external path.
