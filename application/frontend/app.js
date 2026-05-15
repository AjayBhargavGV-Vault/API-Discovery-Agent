const API_BASE = "http://127.0.0.1:8080";

const state = {
  mode: "chat",
  conversationId: null,
  busy: false,
  lastResponse: null,
};

const elements = {
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  includeTrace: document.querySelector("#includeTrace"),
  topK: document.querySelector("#topK"),
  conversationLabel: document.querySelector("#conversationLabel"),
  clearChat: document.querySelector("#clearChat"),
  apiDot: document.querySelector("#apiDot"),
  apiStatus: document.querySelector("#apiStatus"),
  modelList: document.querySelector("#modelList"),
  refreshStatus: document.querySelector("#refreshStatus"),
  sourcesPane: document.querySelector("#sourcesPane"),
  tracePane: document.querySelector("#tracePane"),
  rawPane: document.querySelector("#rawPane"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setBusy(busy) {
  state.busy = busy;
  elements.send.disabled = busy;
  elements.input.disabled = busy;
  elements.send.textContent = busy ? "…" : "↑";
}

function renderEmpty() {
  elements.messages.innerHTML = `
    <div class="empty-state">
      <h3>Ask the API catalog.</h3>
      <p>Find Stripe endpoints, compare schemas, inspect related operations, or ask for implementation examples backed by the local Milvus index and Ollama models.</p>
    </div>
  `;
}

function appendMessage(role, text, meta = "") {
  if (elements.messages.querySelector(".empty-state")) {
    elements.messages.innerHTML = "";
  }
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.innerHTML = `
    <div class="bubble">${escapeHtml(text)}</div>
    ${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ""}
  `;
  elements.messages.append(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function updateConversationLabel() {
  elements.conversationLabel.textContent = state.conversationId
    ? `Conversation ${state.conversationId.slice(0, 8)}`
    : "New conversation";
}

function normalizeDetail(value) {
  if (value === null || value === undefined || value === "") return "None";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function renderDetails(data) {
  state.lastResponse = data;
  const sources = data?.sources || [];
  const trace = data?.agent_trace || [];

  elements.sourcesPane.innerHTML = sources.length
    ? sources.map((source, index) => `
        <div class="detail-item">
          <strong>Source ${index + 1}</strong>
          <span>${escapeHtml(normalizeDetail(source))}</span>
        </div>
      `).join("")
    : `<div class="detail-item">No sources returned.</div>`;

  elements.tracePane.innerHTML = trace.length
    ? trace.map((item, index) => `
        <div class="detail-item">
          <strong>${escapeHtml(item.agent_name || item.task_id || `Step ${index + 1}`)}</strong>
          <span>${escapeHtml(normalizeDetail(item))}</span>
        </div>
      `).join("")
    : `<div class="detail-item">No trace returned.</div>`;

  elements.rawPane.textContent = JSON.stringify(data || {}, null, 2);
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function sendMessage(message) {
  setBusy(true);
  appendMessage("user", message);
  try {
    const topK = Number(elements.topK.value || 5);
    const data = state.mode === "chat"
      ? await requestJson("/api/chat", {
          method: "POST",
          body: JSON.stringify({
            user_message: message,
            conversation_id: state.conversationId,
            top_k: topK,
            include_agent_trace: elements.includeTrace.checked,
          }),
        })
      : await requestJson("/api/search", {
          method: "POST",
          body: JSON.stringify({
            query: message,
            max_graph_expansions: 0,
          }),
        });

    if (data.conversation_id) {
      state.conversationId = data.conversation_id;
      updateConversationLabel();
    }
    appendMessage("assistant", data.answer || "No answer returned.", data.guardrail_status || "Search result");
    renderDetails(data);
  } catch (error) {
    appendMessage("assistant", `Request failed: ${error.message}`, "Error");
  } finally {
    setBusy(false);
    elements.input.focus();
  }
}

async function refreshStatus() {
  elements.apiStatus.textContent = "Checking API";
  elements.apiDot.className = "status-dot";
  try {
    const data = await requestJson("/api/chat/health");
    elements.apiDot.className = "status-dot ok";
    elements.apiStatus.textContent = data.chat_engine === "ready" ? "Chat engine ready" : data.chat_engine;
    const models = data.models || {};
    elements.modelList.innerHTML = Object.entries(models).map(([name, model]) => `
      <div class="model-pill">
        <strong>${escapeHtml(name)}</strong>
        <span>${escapeHtml(model)}</span>
      </div>
    `).join("");
  } catch (error) {
    elements.apiDot.className = "status-dot bad";
    elements.apiStatus.textContent = "API unavailable";
    elements.modelList.innerHTML = `<div class="model-pill"><span>${escapeHtml(error.message)}</span></div>`;
  }
}

function autoresize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 160)}px`;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message || state.busy) return;
  elements.input.value = "";
  autoresize();
  sendMessage(message);
});

elements.input.addEventListener("input", autoresize);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll("[data-mode]").forEach((item) => item.classList.toggle("active", item === button));
  });
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt;
    autoresize();
    elements.input.focus();
  });
});

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((item) => item.classList.toggle("active", item === button));
    const active = button.dataset.tab;
    elements.sourcesPane.classList.toggle("hidden", active !== "sources");
    elements.tracePane.classList.toggle("hidden", active !== "trace");
    elements.rawPane.classList.toggle("hidden", active !== "raw");
  });
});

elements.clearChat.addEventListener("click", () => {
  state.conversationId = null;
  state.lastResponse = null;
  updateConversationLabel();
  renderEmpty();
  renderDetails(null);
});

elements.refreshStatus.addEventListener("click", refreshStatus);

renderEmpty();
renderDetails(null);
refreshStatus();
