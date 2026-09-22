# Optional Sharp-Spark remote worker

The optional `worker_delegate` capability runs `peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF` on the separate `homeserver` GPU host. Qwen on Cortana remains the executive, repository owner, writer, and verifier.

The worker receives only bounded structured prompts and has read-only access to the active repository through tools executed on Cortana. It cannot edit, run arbitrary shell, browse, access credentials, commit, or push. `scout`, `debugger`, and `reviewer` are serial roles with a 14-step/120-second budget. The Qwen-facing report is capped at 3,200 bytes (normally well below 800 tokens); worker metadata only is retained under `~/.local/state/opencode/sharp-spark-worker`.

The service is private to the homeserver Tailscale address (`100.81.200.82:18090`) and uses a dedicated API key stored outside Git. The worker model and llama.cpp image live under `/home/brandon/sharp-spark` on homeserver. The selected initial configuration is Q5_K_XL, 16K context, q8_0 K/V, one slot, Jinja enabled, and full GPU offload without forced Flash Attention.

Operations on homeserver:

```text
systemctl --user daemon-reload
systemctl --user enable --now sharp-spark-worker.service
systemctl --user status sharp-spark-worker.service
journalctl --user -u sharp-spark-worker.service -f
systemctl --user stop sharp-spark-worker.service
systemctl --user disable sharp-spark-worker.service
```

The user service is kept available across logouts by the one-time user-manager
setting `loginctl enable-linger brandon`; verify with `loginctl show-user brandon
-p Linger`. Rollback is `systemctl --user disable --now
sharp-spark-worker.service`; leave existing Ollama and unrelated
llama.cpp/Docker services untouched. Removing the `worker_delegate` registration
and restoring the prior v2 source/config leaves Qwen's local path unchanged.
