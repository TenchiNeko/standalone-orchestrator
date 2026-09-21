#!/usr/bin/env node

// OpenCode 1.18.31 has no native MCP per-tool visibility setting: permissions
// deny calls but do not remove schemas. This adapter preserves the configured
// local command argv, cwd, environment, errors, pagination, and lifecycle
// while filtering only tools/list. It is deliberately not a new MCP router.
const command = JSON.parse(process.env.V2_AGENTMEMORY_MCP_COMMAND || "null")
if (!Array.isArray(command) || command.length === 0 || command.some((item) => typeof item !== "string")) {
  process.stderr.write("v2 AgentMemory filter: missing configured command argv\n")
  process.exit(2)
}
let configuredAllowed = [
  "memory_smart_search",
  "memory_recall",
  "memory_save",
  "memory_lesson_recall",
  "memory_lesson_save",
]
try {
  const parsed = JSON.parse(process.env.V2_AGENTMEMORY_ALLOWED_TOOLS || "null")
  if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) configuredAllowed = parsed
} catch { /* defaults are intentionally fixed and small */ }
const allowed = new Set(configuredAllowed)
const child = (await import("node:child_process")).spawn(command[0], command.slice(1), {
  cwd: process.env.V2_AGENTMEMORY_MCP_CWD || process.cwd(),
  env: process.env,
  stdio: ["pipe", "pipe", "inherit"],
})
const methods = new Map()
let input = ""
let output = ""
const MAX_BUFFER = 2 * 1024 * 1024

function write(message) {
  process.stdout.write(JSON.stringify(message) + "\n")
}

function forward(line) {
  if (!line.trim()) return
  let message
  try { message = JSON.parse(line) } catch { return }
  if (message.id !== undefined && message.method) methods.set(String(message.id), message.method)
  child.stdin.write(JSON.stringify(message) + "\n")
}

function filterResponse(message) {
  const method = message.id === undefined ? undefined : methods.get(String(message.id))
  if (method === "tools/list" && Array.isArray(message.result?.tools)) {
    message.result.tools = message.result.tools.filter((tool) => allowed.has(tool.name))
    methods.delete(String(message.id))
  }
  return message
}

process.stdin.on("data", (chunk) => {
  input += chunk.toString()
  if (input.length > MAX_BUFFER) { process.stderr.write("v2 AgentMemory filter: request buffer exceeded limit\n"); child.kill("SIGTERM"); process.exitCode = 1; return }
  let newline
  while ((newline = input.indexOf("\n")) >= 0) {
    const line = input.slice(0, newline)
    input = input.slice(newline + 1)
    forward(line)
  }
})
child.stdout.on("data", (chunk) => {
  output += chunk.toString()
  if (output.length > MAX_BUFFER) { process.stderr.write("v2 AgentMemory filter: response buffer exceeded limit\n"); child.kill("SIGTERM"); process.exitCode = 1; return }
  let newline
  while ((newline = output.indexOf("\n")) >= 0) {
    const line = output.slice(0, newline)
    output = output.slice(newline + 1)
    try { write(filterResponse(JSON.parse(line))) } catch { /* child diagnostics stay on stderr */ }
  }
})
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)))
process.on("SIGTERM", () => child.kill("SIGTERM"))
process.on("SIGINT", () => child.kill("SIGINT"))
