#!/usr/bin/env node

// OpenCode 1.18.31 has no native MCP per-tool visibility setting: permissions
// deny calls but do not remove schemas. This adapter preserves the configured
// local command argv, cwd, environment, errors, pagination, and lifecycle
// while filtering only the five daily tools and bounding retrieval traffic.
// It is deliberately not a new MCP router.
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
const retrievalTools = new Set(["memory_recall", "memory_smart_search", "memory_lesson_recall"])
const MAX_BUFFER = 2 * 1024 * 1024
const MAX_PENDING_REQUESTS = 1024
const MAX_RETRIEVAL_BYTES = 4096
const TRUNCATION_MARKER = "[memory result truncated by v2 lean-context guard; narrow the query for more]"

const child = (await import("node:child_process")).spawn(command[0], command.slice(1), {
  cwd: process.env.V2_AGENTMEMORY_MCP_CWD || process.cwd(),
  env: process.env,
  stdio: ["pipe", "pipe", "inherit"],
})

// The key includes the JSON-RPC id type so numeric 1 and string "1" cannot
// collide. Entries are removed when their response arrives and capped for a
// misbehaving peer that never sends responses.
const requests = new Map()
function idKey(id) { return `${typeof id}:${JSON.stringify(id)}` }
function rememberRequest(message) {
  if (message.id === undefined || !message.method) return
  const meta = {
    method: message.method,
    toolName: message.method === "tools/call" && typeof message.params?.name === "string"
      ? message.params.name
      : undefined,
  }
  requests.set(idKey(message.id), meta)
  while (requests.size > MAX_PENDING_REQUESTS) requests.delete(requests.keys().next().value)
}
function takeRequest(message) {
  if (message.id === undefined) return undefined
  const key = idKey(message.id)
  const meta = requests.get(key)
  requests.delete(key)
  return meta
}

function positiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}
function boundedPositive(value, maximum, fallback) {
  return positiveNumber(value) ? Math.min(maximum, value) : fallback
}
function boundedExpandIds(value) {
  if (typeof value === "string") {
    return value.split(",").map((item) => item.trim()).filter(Boolean).slice(0, 3).join(",")
  }
  // The installed schema uses a comma-separated string. This defensive branch
  // avoids forwarding an unexpectedly large array if a future MCP changes it.
  if (Array.isArray(value)) return value.slice(0, 3)
  return value
}
function clampRetrievalRequest(message) {
  if (message.method !== "tools/call" || typeof message.params?.name !== "string") return message
  const toolName = message.params.name
  if (!retrievalTools.has(toolName) || !message.params.arguments || typeof message.params.arguments !== "object") return message
  const args = { ...message.params.arguments }
  if (toolName === "memory_recall") {
    args.limit = boundedPositive(args.limit, 3, 3)
    args.format = "compact"
    args.token_budget = boundedPositive(args.token_budget, 900, 900)
  } else if (toolName === "memory_smart_search") {
    args.limit = boundedPositive(args.limit, 3, 3)
    if (args.expandIds !== undefined) args.expandIds = boundedExpandIds(args.expandIds)
  } else if (toolName === "memory_lesson_recall") {
    args.limit = boundedPositive(args.limit, 3, 3)
  }
  return { ...message, params: { ...message.params, arguments: args } }
}

function jsonBytes(value) { return Buffer.byteLength(JSON.stringify(value)) }
function visiblePayload(result) {
  const payload = { content: result?.content }
  if (result && Object.prototype.hasOwnProperty.call(result, "structuredContent")) payload.structuredContent = result.structuredContent
  return payload
}
function retrievalSource(result) {
  const pieces = []
  if (Array.isArray(result?.content)) {
    for (const item of result.content) {
      if (typeof item?.text === "string") pieces.push(item.text)
      else if (typeof item?.data === "string") pieces.push(item.data)
    }
  }
  if (result && Object.prototype.hasOwnProperty.call(result, "structuredContent")) {
    try { pieces.push(JSON.stringify(result.structuredContent)) } catch { /* use available content */ }
  }
  return pieces.join("\n")
}
function reducedRetrievalResult(result) {
  const marker = `\n${TRUNCATION_MARKER}`
  const source = retrievalSource(result)
  const base = { content: [{ type: "text", text: "" }] }
  if (result && Object.prototype.hasOwnProperty.call(result, "isError")) base.isError = result.isError
  const fits = (text) => {
    const candidate = { ...base, content: [{ type: "text", text }] }
    if (result && Object.prototype.hasOwnProperty.call(result, "structuredContent")) candidate.structuredContent = { truncated: true, notice: TRUNCATION_MARKER }
    return jsonBytes(visiblePayload(candidate)) <= MAX_RETRIEVAL_BYTES
  }
  let low = 0
  let high = Array.from(source).length
  let best = TRUNCATION_MARKER
  if (!fits(best)) best = "[memory result truncated]"
  const characters = Array.from(source)
  while (low <= high) {
    const middle = Math.floor((low + high) / 2)
    // Array.from preserves Unicode code points, so the marker and prefix are
    // always valid UTF-8 when JSON.stringify serializes the candidate.
    const candidate = characters.slice(0, middle).join("") + marker
    if (fits(candidate)) {
      best = candidate
      low = middle + 1
    } else high = middle - 1
  }
  const bounded = { ...base, content: [{ type: "text", text: best }] }
  if (result && Object.prototype.hasOwnProperty.call(result, "structuredContent")) bounded.structuredContent = { truncated: true, notice: TRUNCATION_MARKER }
  return bounded
}
function boundRetrievalResult(result) {
  if (!result || typeof result !== "object") return result
  return jsonBytes(visiblePayload(result)) <= MAX_RETRIEVAL_BYTES ? result : reducedRetrievalResult(result)
}

function write(message) { process.stdout.write(JSON.stringify(message) + "\n") }
function forward(line) {
  if (!line.trim()) return
  let message
  try { message = JSON.parse(line) } catch { return }
  if (!message || typeof message !== "object" || Array.isArray(message)) return
  const outbound = clampRetrievalRequest(message)
  rememberRequest(outbound)
  try { child.stdin.write(JSON.stringify(outbound) + "\n") } catch { /* child lifecycle owns transport errors */ }
}
function filterResponse(message) {
  if (!message || typeof message !== "object" || Array.isArray(message)) return message
  const meta = takeRequest(message)
  if (meta?.method === "tools/list" && Array.isArray(message.result?.tools)) {
    message.result.tools = message.result.tools.filter((tool) => allowed.has(tool.name))
  } else if (meta?.method === "tools/call" && retrievalTools.has(meta.toolName) && message.result) {
    message.result = boundRetrievalResult(message.result)
  }
  return message
}

let input = ""
let output = ""
process.stdin.on("data", (chunk) => {
  input += chunk.toString()
  if (input.length > MAX_BUFFER) {
    process.stderr.write("v2 AgentMemory filter: request buffer exceeded limit\n")
    child.kill("SIGTERM")
    process.exitCode = 1
    return
  }
  let newline
  while ((newline = input.indexOf("\n")) >= 0) {
    const line = input.slice(0, newline)
    input = input.slice(newline + 1)
    forward(line)
  }
})
child.stdout.on("data", (chunk) => {
  output += chunk.toString()
  if (output.length > MAX_BUFFER) {
    process.stderr.write("v2 AgentMemory filter: response buffer exceeded limit\n")
    child.kill("SIGTERM")
    process.exitCode = 1
    return
  }
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
