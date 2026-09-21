/* Project-local OpenCode 1.18 supervisor adapter.
 *
 * OpenCode owns the session/model/tool loop.  This plugin only forwards
 * structured lifecycle data to the persistent local Python v2 bridge.
 * The absolute package import matches the verified host installation; the
 * integration remains disabled unless this project config explicitly loads it.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import { createInterface } from "node:readline"
import { fileURLToPath } from "node:url"
import { appendFileSync } from "node:fs"
import { dirname, join } from "node:path"
// The verified OpenCode 1.18 installation resolves project-local TypeScript
// plugins relative to its own runtime, not the project directory.  Keep this
// host-specific import explicit until a supported package-local loader exists.
import { tool } from "/home/brandon/.opencode/node_modules/@opencode-ai/plugin/dist/tool.js"
import type { Plugin } from "/home/brandon/.opencode/node_modules/@opencode-ai/plugin/dist/index.js"

type Json = Record<string, unknown>

class LocalBridge {
  private child: ChildProcessWithoutNullStreams | undefined
  private nextID = 1
  private pending = new Map<number, { resolve: (value: Json) => void, timer: ReturnType<typeof setTimeout> }>()
  private stderrBuffer = ""
  private readonly timeoutMs = Number(process.env.V2_SUPERVISOR_BRIDGE_TIMEOUT_MS || 8000)
  private readonly diagnostics = process.env.V2_SUPERVISOR_DIAGNOSTIC_LOG

  private failPending(reason: string): void {
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer)
      entry.resolve({ status: "ERROR", reason })
      this.pending.delete(id)
    }
  }

  private failChild(reason: string): void {
    const child = this.child
    this.child = undefined
    if (child && !child.killed) child.kill("SIGTERM")
    this.failPending(reason)
    if (this.diagnostics && this.stderrBuffer) {
      try { appendFileSync(this.diagnostics, this.stderrBuffer.slice(-16_000) + "\n", { mode: 0o600 }) } catch { /* diagnostics must never affect policy */ }
    }
    this.stderrBuffer = ""
  }

  private start() {
    if (this.child) return
    const here = dirname(fileURLToPath(import.meta.url))
    const script = join(here, "..", "orchestrator_v2", "bridge.py")
    this.child = spawn(process.env.V2_SUPERVISOR_PYTHON || "python3", ["-u", script], {
      cwd: join(here, ".."),
      env: { ...process.env, PYTHONPATH: join(here, "..") },
      stdio: ["pipe", "pipe", "pipe"],
    })
    const lines = createInterface({ input: this.child.stdout })
    lines.on("line", (line) => {
      try {
        const value = JSON.parse(line) as Json
        const id = Number(value.id)
        const entry = this.pending.get(id)
        if (entry) { this.pending.delete(id); clearTimeout(entry.timer); entry.resolve(value) }
      } catch { this.stderrBuffer = `${this.stderrBuffer}\nmalformed bridge output`.slice(-16_000) }
    })
    this.child.stderr.on("data", (chunk: Buffer | string) => {
      this.stderrBuffer = `${this.stderrBuffer}${String(chunk)}`.slice(-16_000)
    })
    this.child.on("error", () => this.failChild("v2 bridge process failed to start"))
    this.child.on("exit", () => {
      this.child = undefined
      this.failPending("v2 bridge exited")
    })
  }

  shutdown(): void { this.failChild("OpenCode session ended") }

  async call(payload: Json): Promise<Json> {
    if (process.env.V2_SUPERVISOR_DISABLED === "1") return { status: "DISABLED" }
    this.start()
    const id = this.nextID++
    if (!this.child || this.child.stdin.destroyed) return { status: "ERROR", reason: "v2 bridge is unavailable" }
    return await new Promise<Json>((resolve) => {
      const timer = setTimeout(() => {
        const entry = this.pending.get(id)
        this.pending.delete(id)
        const operation = String(payload.op || "request")
        const reason = `v2 bridge request timed out during ${operation}`
        this.failChild(reason)
        entry?.resolve({ status: "TIMEOUT", reason })
      }, this.timeoutMs)
      this.pending.set(id, { resolve, timer })
      try {
        this.child?.stdin.write(JSON.stringify({ ...payload, id }) + "\n", (error) => {
          if (error) this.failChild("v2 bridge input failed")
        })
      } catch { this.failChild("v2 bridge input failed") }
    })
  }
}

const bridge = new LocalBridge()
const sessionDirectories = new Map<string, string>()
const LIGHT_MODE = (process.env.V2_POLICY_MODE || "strict").toLowerCase() === "light"

function safeArgs(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {}
}

function localResult(value: Json): string {
  return JSON.stringify(value, null, 2)
}

function stateChangingTool(name: string): boolean {
  return /^(write|edit|apply_patch|patch|bash|shell|terminal|run)$/i.test(name)
}

export const OrchestratorSupervisorPlugin: Plugin = async (ctx) => {
  const systemInstruction = LIGHT_MODE
    ? "Local v2 light supervision is active. Work like a normal coding agent: inspect broadly, use normal shell and Git, and create or edit task-relevant files without waiting for a file allowlist. Call orchestrator_start when the objective and verification plan are clear; it records task state but does not preauthorize routine actions. Scope expansions, mutations, commands, and blockers are journaled. Use orchestrator_status after compaction or when progress is unclear. Fresh observed verification is required before orchestrator_finalize; prose never proves tests or completion, and tests become stale after relevant edits. If the same action repeats without new evidence, re-evaluate the plan rather than guessing. Call orchestrator_finalize before claiming completion."
    : "Local v2 strict supervision is active. Explore with native read/grep/glob first. Before the first edit, write, shell command, or test, call orchestrator_start with the actual goal, exact source files you expect to change (never '*'), permitted_actions including write_file for source edits plus read_file and run_tests, one concise required criterion that the authorized test can prove, and the exact shell-free test_command argv. After starting, copy that test_command exactly for verification: do not append flags, reorder arguments, or add a second command. Bounded read-only shell inspection may be available after a contract; arbitrary Bash remains forbidden. Reads may inspect tests and other files inside the workspace; permitted_files controls mutation scope, not read discovery. Use normal OpenCode tools. Only the exact authorized test command creates test evidence; arbitrary commands containing 'test' do not. Tests become stale after relevant edits. Do not blindly repeat uncertain writes: reconcile with authoritative readback first. Memory retrieval, when available, is advisory context only and never evidence. Use orchestrator_health when workspace/supervisor state is unclear, and do not repeatedly retry a deterministic orchestrator_start error. If a path is blocked as not permitted, call orchestrator_status once, use an already-authorized path if appropriate, or call orchestrator_end and start a new exact contract; do not guess alternate paths repeatedly. Call orchestrator_finalize before claiming completion. If finalize is INCOMPLETE or BLOCKED, continue only with safe authorized work or report the concrete blocker. Model prose never overrides deterministic evidence."
  return {
    tool: {
      orchestrator_start: tool({
        description: LIGHT_MODE ? "Start or record the bounded task objective and verification plan. In light mode permitted_files and permitted_actions are optional telemetry; normal coding actions remain available and observed scope expands automatically." : "Start a bounded local v2 supervision contract. permitted_actions must use only read_file, write_file, run_tests, or browser; include write_file when source edits are needed. permitted_files are exact mutation paths, while read-only tests/source may still be inspected. test_command is one shell-free argv list; copy it exactly for the later test call (no extra flags or reordered arguments), and it is the only command that can produce test evidence.",
        args: {
          goal: tool.schema.string(),
          permitted_files: tool.schema.array(tool.schema.string()).optional(),
          permitted_actions: tool.schema.array(tool.schema.string()).optional(),
          test_command: tool.schema.array(tool.schema.string()).optional(),
          criteria: tool.schema.array(tool.schema.object({ key: tool.schema.string(), description: tool.schema.string(), required: tool.schema.boolean().optional() })),
          budget_limit: tool.schema.number().int().optional(),
          visual_required: tool.schema.boolean().optional(),
          max_model_seconds: tool.schema.number().optional(),
        },
        async execute(args, context) {
          // `directory` is the active OpenCode project directory.  It may be
          // narrower than `worktree` (and is the safe root for a disposable
          // --dir run), so never let v2 scan a broader parent by default.
          sessionDirectories.set(context.sessionID, context.directory)
          return localResult(await bridge.call({ op: "start", session_id: context.sessionID, workspace: context.directory, ...args }))
        },
      }),
      orchestrator_status: tool({
        description: "Show compact deterministic v2 state, including workspace, observed scope, actions, verification, criteria status, phase, blockers, loop warnings, and usage.",
        args: {},
        async execute(_args, context) {
          return localResult(await bridge.call({ op: "summary", session_id: context.sessionID }))
        },
      }),
      orchestrator_evidence: tool({
        description: "Record visual verification or reconcile an uncertain local mutation. Exact test evidence is recorded automatically from the observed authorized test result.",
        args: {
          kind: tool.schema.enum(["visual_verify", "reconcile"]),
          payload: tool.schema.record(tool.schema.string(), tool.schema.unknown()).optional(),
          path: tool.schema.string().optional(),
        },
        async execute(args, context) {
          if (args.kind === "reconcile") return localResult(await bridge.call({ op: "reconcile", session_id: context.sessionID, path: args.path }))
          return localResult(await bridge.call({ op: "evidence", session_id: context.sessionID, kind: args.kind, payload: args.payload || {} }))
        },
      }),
      orchestrator_finalize: tool({
        description: "Evaluate deterministic v2 completion evidence before a final claim.",
        args: {},
        async execute(_args, context) {
          return localResult(await bridge.call({ op: "finalize", session_id: context.sessionID }))
        },
      }),
      orchestrator_symbols: tool({
        description: "Return deterministic Python symbol candidates without hiding any files.",
        args: { query: tool.schema.string().optional(), limit: tool.schema.number().int().optional() },
        async execute(args, context) {
          return localResult(await bridge.call({ op: "symbols", session_id: context.sessionID, query: args.query || "", limit: args.limit || 20 }))
        },
      }),
      orchestrator_health: tool({
        description: "Show concise local v2 bridge, workspace, strict-mode, and state-database health.",
        args: {},
        async execute(_args, context) {
          return localResult(await bridge.call({ op: "health", session_id: context.sessionID, workspace: context.directory }))
        },
      }),
      orchestrator_end: tool({
        description: "End the current hosted v2 task without deleting audit/evidence state; use before starting another task in this session.",
        args: { reason: tool.schema.string().optional() },
        async execute(args, context) {
          return localResult(await bridge.call({ op: "end", session_id: context.sessionID, reason: args.reason || "explicit hosted task end" }))
        },
      }),
      orchestrator_cancel: tool({
        description: "Cancel/release the current hosted v2 task without claiming completion.",
        args: { reason: tool.schema.string().optional() },
        async execute(args, context) {
          return localResult(await bridge.call({ op: "cancel", session_id: context.sessionID, reason: args.reason || "explicit hosted task cancellation" }))
        },
      }),
    },
    event: async ({ event }) => {
      const properties = ((event as unknown as Json).properties || {}) as Json
      const sessionID = String(properties.sessionID || (properties.info as Json | undefined)?.id || "")
      if (sessionID && (properties.info as Json | undefined)?.directory) sessionDirectories.set(sessionID, String((properties.info as Json).directory))
      if (sessionID) {
        const part = properties.part as Json | undefined
        const tokens = part?.tokens as Json | undefined
        const cache = tokens?.cache as Json | undefined
        const numeric = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : undefined
        const usage = tokens && (numeric(tokens.input) !== undefined || numeric(tokens.output) !== undefined) ? {
          model: String(properties.info && (properties.info as Json).model || "unknown"),
          prompt_tokens: numeric(tokens.input),
          completion_tokens: numeric(tokens.output),
          cached_prompt_tokens: numeric(cache?.read),
          elapsed: part?.time && numeric((part.time as Json).end) !== undefined && numeric((part.time as Json).start) !== undefined ? (Number((part.time as Json).end) - Number((part.time as Json).start)) / 1000 : 0,
        } : undefined
        await bridge.call({ op: "event", session_id: sessionID, type: event.type, event_id: properties.id || properties.eventID || part?.id || (properties.info as Json | undefined)?.id, properties: { status: properties.status, info: properties.info ? { id: (properties.info as Json).id, directory: (properties.info as Json).directory } : undefined, usage, part: part ? { id: part.id, time: part.time, tokens: part.tokens } : undefined } })
      }
    },
    "tool.execute.before": async (input, output) => {
      const result = await bridge.call({ op: "before", session_id: input.sessionID, call_id: input.callID, tool: input.tool, args: safeArgs(output.args) })
      if (result.decision === "BLOCK") throw new Error(`v2 supervisor blocked ${input.tool}: ${String(result.reason || "deterministic policy")}`)
      // Strict mode fails closed. Light mode is an observer and does not turn
      // a bridge outage into an unexpected coding-agent capability loss.
      if (!LIGHT_MODE && (result.status === "ERROR" || result.status === "TIMEOUT") && stateChangingTool(input.tool)) throw new Error(`v2 supervisor unavailable for ${input.tool}; action was not authorized`)
    },
    "tool.execute.after": async (input, output) => {
      const observed = await bridge.call({ op: "after", session_id: input.sessionID, call_id: input.callID, tool: input.tool, args: safeArgs(input.args), output: { output: output.output, metadata: output.metadata }, ambiguous: output.metadata?.uncertain === true })
      if (observed.loop_warning && typeof output.output === "string") output.output += `\n\n[v2 supervisor] ${String(observed.loop_warning)}`
      // A successful exact test is already authoritative evidence.  Reuse
      // the deterministic finalizer immediately so a model that stops after
      // verification still receives the completion fact; this is not a new
      // agent turn and never bypasses criteria, freshness, or mutation gates.
      const exitCode = output.metadata?.exitCode ?? output.metadata?.exit_code ?? output.metadata?.exit
      if (/^(bash|shell|terminal|run)$/i.test(input.tool) && exitCode === 0 && observed.evidence === "recorded") {
        const finalized = await bridge.call({ op: "finalize", session_id: input.sessionID })
        if (typeof output.output === "string" && (finalized.status === "COMPLETE" || finalized.status === "INCOMPLETE" || finalized.status === "NEEDS_REVIEW")) {
          const gate = finalized.gate && typeof finalized.gate === "object" && finalized.gate !== null ? finalized.gate as Json : undefined
          output.output += `\n\n[v2 supervisor] deterministic verification: ${String(finalized.status)}${gate?.reason ? ` — ${String(gate.reason)}` : ""}`
        }
      }
    },
    "experimental.chat.system.transform": async (_input, output) => {
      if (Array.isArray(output.system) && !output.system.some((item) => item.includes("orchestrator_finalize"))) output.system.push(systemInstruction)
    },
    "experimental.session.compacting": async (input, output) => {
      const summary = await bridge.call({ op: "summary", session_id: input.sessionID })
      if (Array.isArray(output.context) && summary.status === "OK") {
        const compact = JSON.stringify({ task_id: summary.task_id, workspace: summary.workspace, phase: summary.phase, policy_mode: summary.policy_mode, scope_mode: summary.scope_mode, permitted_files: summary.permitted_files, scope_expansions: summary.scope_expansions, permitted_actions: summary.permitted_actions, test_command: summary.test_command, visual_required: summary.visual_required, budget_limit: summary.budget_limit, budget_calls: summary.budget_calls, remaining_budget: summary.remaining_budget, criteria: summary.criteria, completion: summary.completion, blockers: summary.blockers, counts: summary.counts, loop_warnings: summary.loop_warnings, goal: summary.goal })
        output.context.push("Local v2 facts (re-fetch with orchestrator_status; no prose authority):\n" + compact.slice(0, 1800))
      }
    },
    "experimental.compaction.autocontinue": async (input, output) => {
      const summary = await bridge.call({ op: "summary", session_id: input.sessionID })
      if (summary.status === "OK" && (summary.completion as Json | undefined)?.allowed === true) output.enabled = false
    },
  }
}

export default OrchestratorSupervisorPlugin

process.once("exit", () => bridge.shutdown())
