#!/usr/bin/env node

// Keep the useful memory search/save surface without sending AgentMemory's
// entire 54-tool registry on every coding request.  This is a transport
// filter only: calls still go to the configured AgentMemory server.
const child = (await import("node:child_process")).spawn(
  process.env.AGENTMEMORY_MCP_BIN || "agentmemory-mcp",
  [],
  { env: process.env, stdio: ["pipe", "pipe", "inherit"] },
)

const allowed = new Set([
  "memory_save",
  "memory_recall",
  "memory_smart_search",
  "memory_sessions",
  "memory_lesson_save",
  "memory_lesson_recall",
  "memory_consolidate",
  "memory_diagnose",
  "memory_reflect",
])
const methods = new Map()
let input = ""
let output = ""

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
  let newline
  while ((newline = input.indexOf("\n")) >= 0) {
    const line = input.slice(0, newline)
    input = input.slice(newline + 1)
    forward(line)
  }
})
child.stdout.on("data", (chunk) => {
  output += chunk.toString()
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
