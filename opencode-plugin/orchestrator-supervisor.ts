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
import { dirname, join } from "node:path"
import { tool } from "/home/brandon/.opencode/node_modules/@opencode-ai/plugin/dist/tool.js"
import type { Plugin } from "/home/brandon/.opencode/node_modules/@opencode-ai/plugin/dist/index.js"

type Json = Record<string, unknown>

class LocalBridge {
  private child: ChildProcessWithoutNullStreams | undefined
  private nextID = 1
  private pending = new Map<number, (value: Json) => void>()

  private start() {
    if (this.child) return
    const here = dirname(fileURLToPath(import.meta.url))
    const script = join(here, "..", "orchestrator_v2", "bridge.py")
    this.child = spawn(process.env.V2_SUPERVISOR_PYTHON || "python3", ["-u", script], {
      cwd: join(here, ".."),
      env: { ...process.env, PYTHONPATH: join(here, "..") },
      stdio: ["pipe", "pipe", "pipe"],
    })
    // Do not keep `opencode run` alive solely because the optional bridge is
    // idle. A TUI/server process still owns the bridge while it is alive.
    this.child.unref()
    this.child.stdin.unref?.()
    this.child.stdout.unref?.()
    this.child.stderr.unref?.()
    const lines = createInterface({ input: this.child.stdout })
    lines.on("line", (line) => {
      try {
        const value = JSON.parse(line) as Json
        const id = Number(value.id)
        const resolve = this.pending.get(id)
        if (resolve) { this.pending.delete(id); resolve(value) }
      } catch { /* malformed bridge output is ignored; host remains usable */ }
    })
    this.child.on("exit", () => {
      this.child = undefined
      for (const resolve of this.pending.values()) resolve({ status: "ERROR", reason: "v2 bridge exited" })
      this.pending.clear()
    })
  }

  async call(payload: Json): Promise<Json> {
    if (process.env.V2_SUPERVISOR_DISABLED === "1") return { status: "DISABLED" }
    this.start()
    const id = this.nextID++
    return await new Promise<Json>((resolve) => {
      this.pending.set(id, resolve)
      this.child?.stdin.write(JSON.stringify({ ...payload, id }) + "\n")
    })
  }
}

const bridge = new LocalBridge()
const sessionDirectories = new Map<string, string>()

function safeArgs(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {}
}

function localResult(value: Json): string {
  return JSON.stringify(value, null, 2)
}

export const OrchestratorSupervisorPlugin: Plugin = async (ctx) => {
  const systemInstruction = "v2 supervises this task locally. Use normal OpenCode tools, do not blindly repeat uncertain writes, and remember tests become stale after relevant edits. Use orchestrator_status when state is unclear and call orchestrator_finalize before claiming completion. Finalize results are deterministic evidence; model prose cannot override INCOMPLETE or BLOCKED."
  return {
    tool: {
      orchestrator_start: tool({
        description: "Start a bounded local v2 supervision contract for this OpenCode session.",
        args: {
          goal: tool.schema.string(),
          permitted_files: tool.schema.array(tool.schema.string()),
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
        description: "Show compact deterministic v2 task state, evidence blockers, and usage.",
        args: {},
        async execute(_args, context) {
          return localResult(await bridge.call({ op: "summary", session_id: context.sessionID }))
        },
      }),
      orchestrator_evidence: tool({
        description: "Record one bounded deterministic evidence item or reconcile an uncertain local mutation.",
        args: {
          kind: tool.schema.enum(["test", "visual_verify", "reconcile"]),
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
        await bridge.call({ op: "event", session_id: sessionID, type: event.type, properties: { status: properties.status, info: properties.info ? { id: (properties.info as Json).id, directory: (properties.info as Json).directory } : undefined, usage } })
      }
    },
    "tool.execute.before": async (input, output) => {
      const result = await bridge.call({ op: "before", session_id: input.sessionID, call_id: input.callID, tool: input.tool, args: safeArgs(output.args) })
      if (result.decision === "BLOCK") throw new Error(`v2 supervisor blocked ${input.tool}: ${String(result.reason || "deterministic policy")}`)
    },
    "tool.execute.after": async (input, output) => {
      await bridge.call({ op: "after", session_id: input.sessionID, call_id: input.callID, tool: input.tool, args: safeArgs(input.args), output: { output: output.output, metadata: output.metadata }, ambiguous: output.metadata?.uncertain === true })
    },
    "experimental.chat.system.transform": async (_input, output) => {
      if (Array.isArray(output.system) && !output.system.some((item) => item.includes("orchestrator_finalize"))) output.system.push(systemInstruction)
    },
    "experimental.session.compacting": async (input, output) => {
      const summary = await bridge.call({ op: "summary", session_id: input.sessionID })
      if (Array.isArray(output.context)) output.context.push("Local v2 state summary (facts only):\n" + localResult(summary))
    },
  }
}

export default OrchestratorSupervisorPlugin
