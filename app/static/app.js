const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

async function fetchHealth() {
  const badge = $("health");
  try {
    const response = await fetch("/health");
    const data = await response.json();
    const ok = data.status === "ok";
    badge.textContent = ok ? "依赖正常" : "依赖异常";
    badge.className = `badge ${ok ? "ok" : "warn"}`;
    badge.title = (data.dependencies || [])
      .map((dep) => `${dep.name}: ${dep.ok ? "ok" : dep.detail}`)
      .join("\n");
  } catch (error) {
    badge.textContent = "服务不可达";
    badge.className = "badge warn";
  }
}

async function refreshDocs() {
  const list = $("doc-list");
  try {
    const response = await fetch("/api/v1/documents?limit=20");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    list.innerHTML = "";
    if (!data.items.length) {
      list.innerHTML = '<li class="empty">暂无文档，先上传一份试试</li>';
      return;
    }
    for (const doc of data.items) {
      const item = document.createElement("li");
      const info = document.createElement("span");
      info.innerHTML = `${escapeHtml(doc.filename)}<small>${doc.status} · 子块 ${doc.child_count} · 向量 ${doc.vector_count}</small>`;
      const del = document.createElement("button");
      del.className = "ghost";
      del.textContent = "删除";
      del.onclick = async () => {
        if (!confirm(`删除 ${doc.filename}？`)) return;
        await fetch(`/api/v1/documents/${doc.id}`, { method: "DELETE" });
        refreshDocs();
      };
      item.append(info, del);
      list.appendChild(item);
    }
  } catch (error) {
    list.innerHTML = `<li class="empty">加载失败：${escapeHtml(error.message)}</li>`;
  }
}

async function pollDocument(docId) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const response = await fetch(`/api/v1/documents/${docId}`);
    if (response.ok) {
      const doc = await response.json();
      if (doc.status === "ready") {
        $("upload-status").textContent = `已入库：${doc.filename}（子块 ${doc.child_count}，向量 ${doc.vector_count}）`;
        refreshDocs();
        return;
      }
      if (doc.status === "failed") {
        $("upload-status").textContent = `处理失败：${doc.error || "未知错误"}`;
        refreshDocs();
        return;
      }
      $("upload-status").textContent = `处理中：${doc.status}`;
    }
    await sleep(800);
  }
  $("upload-status").textContent = "处理超时，请稍后在文档列表查看状态";
}

function setupUpload() {
  $("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = $("file").files[0];
    if (!file) return;
    const button = $("upload-btn");
    button.disabled = true;
    $("upload-status").textContent = "上传中…";
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/v1/documents", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail;
        $("upload-status").textContent = `上传失败：${typeof detail === "string" ? detail : JSON.stringify(detail)}`;
        return;
      }
      $("upload-status").textContent = "已提交，等待解析与向量化…";
      await pollDocument(payload.id);
      $("file").value = "";
    } catch (error) {
      $("upload-status").textContent = `上传失败：${error.message}`;
    } finally {
      button.disabled = false;
    }
  });
}

function renderCitations(citations) {
  const list = $("citations");
  list.innerHTML = "";
  for (const citation of citations) {
    const label = [citation.filename, citation.section, citation.page != null ? `第 ${citation.page} 页` : ""]
      .filter(Boolean)
      .join(" · ");
    const item = document.createElement("li");
    item.innerHTML = `<b>[${citation.index}]</b> ${escapeHtml(label)}`;
    list.appendChild(item);
  }
}

function parseEvent(raw) {
  let eventName = "message";
  const dataLines = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event: eventName, data: JSON.parse(dataLines.join("\n")) };
  } catch (error) {
    return null;
  }
}

const SESSION_KEY = "rag-demo-session";

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}`;
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function resetPipeline() {
  const routeChip = $("route-chip");
  routeChip.textContent = "分析中…";
  routeChip.className = "chip";
  const rewriteChip = $("rewrite-chip");
  rewriteChip.textContent = "";
  rewriteChip.className = "chip muted";
  $("pipeline-detail").textContent = "";
}

function handleAgentEvent(name, data) {
  const routeChip = $("route-chip");
  const rewriteChip = $("rewrite-chip");
  const detail = $("pipeline-detail");

  if (name === "start") {
    detail.textContent = `会话 ${String(data.session_id || "").slice(0, 8)}…`;
  } else if (name === "condense") {
    detail.textContent = `指代消解：${data.search_query}`;
  } else if (name === "route") {
    const labels = { vector: "混合检索", direct: "直接回答", refuse: "礼貌拒答" };
    routeChip.textContent = labels[data.route] || data.route;
    routeChip.className = `chip ${data.route === "vector" ? "ok" : "warn"}`;
    detail.textContent = data.reason || "";
  } else if (name === "retrieval") {
    detail.textContent = `召回 ${data.count} 条（${data.query}）`;
  } else if (name === "critique") {
    detail.textContent = `自检：${data.relevant ? "相关" : "不相关"}${data.reason ? `（${data.reason}）` : ""}`;
  } else if (name === "rewrite") {
    rewriteChip.textContent = `第 ${data.rewrite_count} 次改写`;
    rewriteChip.className = "chip warn visible";
    detail.textContent = `改为：${data.query}`;
  }
}

function setupChat() {
  $("chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("question").value.trim();
    if (!question) return;
    const button = $("chat-btn");
    button.disabled = true;
    $("answer").textContent = "";
    $("citations").innerHTML = "";
    $("chat-status").textContent = "处理中…";
    resetPipeline();

    try {
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: getSessionId() }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = payload.detail;
        $("chat-status").textContent = `请求失败：${typeof detail === "string" ? detail : JSON.stringify(detail)}`;
        return;
      }

      let answer = "";
      let buffer = "";
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        let index;
        while ((index = buffer.indexOf("\n\n")) >= 0) {
          const parsed = parseEvent(buffer.slice(0, index));
          buffer = buffer.slice(index + 2);
          if (!parsed) continue;
          handleAgentEvent(parsed.event, parsed.data);

          if (parsed.event === "token") {
            answer += parsed.data.text;
            $("answer").textContent = answer;
          } else if (parsed.event === "done") {
            const timings = parsed.data.timings || {};
            const ttft = timings.ttft_ms != null ? `${timings.ttft_ms} ms` : "—";
            $("chat-status").textContent = `首字 ${ttft} · 总耗时 ${timings.total_ms ?? "—"} ms · 改写 ${parsed.data.rewrite_count ?? 0} 次`;
            renderCitations(parsed.data.citations || []);
            if (!answer) $("answer").textContent = parsed.data.answer || "";
          } else if (parsed.event === "error") {
            $("chat-status").textContent = `错误：${parsed.data.message}`;
          }
        }
      }
    } catch (error) {
      $("chat-status").textContent = `请求失败：${error.message}`;
    } finally {
      button.disabled = false;
    }
  });
}

fetchHealth();
refreshDocs();
setupUpload();
setupChat();
