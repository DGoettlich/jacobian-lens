def page() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>J-lens UI</title>
  <style>
    body { margin: 24px; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; }
    h1 { margin: 0 0 18px; font-size: 24px; }
    .row { display: flex; gap: 10px; align-items: flex-start; margin: 10px 0; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    label { font-weight: 600; }
    input, textarea, button, select { font: inherit; }
    input { width: 360px; }
    input.choice { width: 180px; }
    textarea { width: 600px; height: 64px; }
    button { padding: 6px 10px; border: 1px solid #d0d7de; background: white; border-radius: 6px; cursor: pointer; }
    button.primary { background: #0969da; border-color: #0969da; color: white; }
    button.stop { background: #cf222e; border-color: #cf222e; color: white; }
    button:disabled { opacity: 0.5; cursor: default; }
    .choice-row { align-items: center; }
    .focus { display: flex; align-items: center; gap: 6px; font-weight: 600; }
    .tokens { max-width: 780px; line-height: 1.9; }
    .tok { padding: 2px 4px; margin: 1px; border-radius: 2px; display: inline-block; }
    .answer { outline: 2px solid #1f2328; }
    .a0 { background: #d8ccff; }
    .a1 { background: #d7f5d0; }
    .a2 { background: #ffe8bd; }
    .a3 { background: #ffc9c9; }
    .a4 { background: #bfe8ff; }
    .bad { color: #b42318; }
    .ok { color: #067647; }
    #status { min-height: 20px; margin-top: 10px; }
    .panel { margin-top: 18px; padding-top: 16px; border-top: 1px solid #d0d7de; }
    .panel h2 { margin: 0 0 10px; font-size: 18px; }
    input.small { width: 72px; }
    .prompt-tokens, .layer-row, .swap-result { margin: 10px 0; }
    .layer-row { display: grid; grid-template-columns: 54px 1fr; gap: 10px; align-items: start; }
    .layer-label { color: #57606a; padding-top: 5px; }
    .token-chip { margin: 2px; padding: 4px 6px; border: 1px solid #d0d7de; border-radius: 6px; background: white; cursor: pointer; }
    .token-chip:hover { border-color: #0969da; }
    .token-chip.source { border-color: #cf222e; background: #ffebe9; }
    .token-chip.target { border-color: #0969da; background: #ddf4ff; }
    .token-score { color: #57606a; font-size: 12px; margin-left: 4px; }
    .picked { font-weight: 600; min-width: 180px; }
    .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; max-width: 900px; }
    .result-col h3 { margin: 8px 0; font-size: 15px; }
    .result-token { display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid #d8dee4; padding: 4px 0; }
    iframe { width: 100%; height: 860px; border: 1px solid #d0d7de; margin-top: 18px; }
  </style>
</head>
<body>
  <h1>J-lens UI</h1>

  <div class="row">
    <div class="field">
      <label>Model</label>
      <input id="model" value="uzh-echist-org/Ranke-4B-1913">
    </div>
    <div class="field">
      <label>Architecture model / tokenizer</label>
      <input id="architecture" value="">
    </div>
  </div>

  <div class="row">
    <div class="field">
      <label>Lens repo</label>
      <input id="lens-repo" value="history-llms/jlenses">
    </div>
    <div class="field">
      <label>Lens file</label>
      <input id="lens-file" value="Ranke-4B-1913.pt">
    </div>
  </div>

  <div class="row">
    <div class="field">
      <label>Question</label>
      <textarea id="question">The capital of France is</textarea>
    </div>
  </div>

  <div id="choices"></div>

  <div class="row">
    <button id="add">+ choice</button>
    <label class="focus">Focus <select id="active-choice"></select></label>
    <button id="serve" class="primary">Serve</button>
    <button id="stop" class="stop" disabled>Stop</button>
    <button id="submit" disabled>Submit</button>
    <button id="export" disabled>Export</button>
    <button id="mode">Token IDs</button>
  </div>

  <div id="status"></div>

  <div class="panel">
    <h2>Distribution / Swap</h2>
    <div class="row">
      <label>Top K <input id="top-k" class="small" type="number" value="10" min="1" max="50"></label>
      <button id="distribution" disabled>Load distribution</button>
      <label>Layer <select id="swap-layer"></select></label>
      <label>Strength <input id="strength" class="small" type="number" value="1" min="0" step="0.1"></label>
      <button id="swap" disabled>Swap</button>
      <button id="clear-swap" disabled>Clear</button>
    </div>
    <div class="row">
      <div class="picked">Source: <span id="source-token">none</span></div>
      <div class="picked">Target: <span id="target-token">none</span></div>
    </div>
    <div id="distribution-view"></div>
    <div id="swap-view"></div>
  </div>

  <iframe id="report"></iframe>

<script>
let tokenMode = "text";
let lastHtml = null;
let tokenTimer = null;
let swapPick = {source: null, target: null, layer: null};

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function baseBody() {
  return {
    model: document.querySelector("#model").value.trim(),
    architecture_model: document.querySelector("#architecture").value.trim(),
    lens_repo: document.querySelector("#lens-repo").value.trim(),
    lens_file: document.querySelector("#lens-file").value.trim(),
  };
}

function choices() {
  return [...document.querySelectorAll("input.choice")]
    .map(x => x.value.trim())
    .filter(Boolean);
}

function fullBody() {
  return {
    ...baseBody(),
    question: document.querySelector("#question").value,
    choices: choices(),
    active_choice: document.querySelector("#active-choice").value || null,
  };
}

function topK() {
  return Number(document.querySelector("#top-k").value || 10);
}

function choiceRow(value) {
  const div = document.createElement("div");
  div.className = "row choice-row";
  div.innerHTML = `
    <input class="choice" value="${esc(value)}">
    <button class="remove">x</button>
    <div class="tokens"></div>
  `;
  div.querySelector(".remove").onclick = () => {
    div.remove();
    updateFocusChoices();
    scheduleTokenize();
  };
  div.querySelector(".choice").oninput = () => {
    updateFocusChoices();
    scheduleTokenize();
  };
  return div;
}

function updateFocusChoices() {
  const select = document.querySelector("#active-choice");
  const selected = select.value;
  const opts = choices();
  select.innerHTML = opts.map(x => `<option value="${esc(x)}">${esc(x)}</option>`).join("");
  select.value = opts.includes(selected) ? selected : (opts[0] || "");
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {}
  if (!res.ok) throw new Error(data.detail || text || res.statusText);
  return data;
}

function paint(row, data) {
  const spans = data.spans.map((t, i) => {
    const text = tokenMode === "ids" ? t.id : esc(t.text);
    const answer = t.answer ? "answer" : "";
    return `<span class="tok a${i % 5} ${answer}">${text}</span>`;
  }).join("");
  const status = data.single_token
    ? `<span class="ok">single answer token: ${data.answer_ids[0]}</span>`
    : `<span class="bad">answer tokens: [${data.answer_ids.join(", ")}]</span>`;
  row.querySelector(".tokens").innerHTML = `${spans}<br>${status}`;
}

async function tokenize() {
  if (!choices().length) return;
  const data = await post("/api/tokenize", fullBody());
  [...document.querySelectorAll(".choice-row")].forEach((row, i) => {
    if (data.rows[i]) paint(row, data.rows[i]);
  });
}

function scheduleTokenize() {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(tokenize, 250);
}

document.querySelector("#add").onclick = () => {
  document.querySelector("#choices").appendChild(choiceRow(""));
  updateFocusChoices();
};

document.querySelector("#serve").onclick = async () => {
  document.querySelector("#status").textContent = "Loading model/lens on Modal. First 4B load can take a few minutes.";
  try {
    await post("/api/serve", baseBody());
    document.querySelector("#serve").disabled = true;
    document.querySelector("#stop").disabled = false;
    document.querySelector("#submit").disabled = false;
    document.querySelector("#distribution").disabled = false;
    document.querySelector("#status").textContent = "Ready";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

document.querySelector("#stop").onclick = async () => {
  try {
    await post("/api/stop", baseBody());
    document.querySelector("#serve").disabled = false;
    document.querySelector("#stop").disabled = true;
    document.querySelector("#submit").disabled = true;
    document.querySelector("#distribution").disabled = true;
    document.querySelector("#swap").disabled = true;
    document.querySelector("#status").textContent = "Stopped";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

function tokenLabel(token) {
  return `${token.text} (${token.id})`;
}

function updateSwapControls() {
  document.querySelector("#source-token").textContent = swapPick.source ? tokenLabel(swapPick.source) : "none";
  document.querySelector("#target-token").textContent = swapPick.target ? tokenLabel(swapPick.target) : "none";
  document.querySelector("#clear-swap").disabled = !swapPick.source && !swapPick.target;
  document.querySelector("#swap").disabled = !(swapPick.source && swapPick.target && document.querySelector("#swap-layer").value);
}

function pickSwapToken(token, layer) {
  if (!swapPick.source || swapPick.target) {
    swapPick = {source: token, target: null, layer};
    document.querySelector("#swap-layer").value = layer;
  } else {
    swapPick.target = token;
  }
  paintPickedTokens();
  updateSwapControls();
}

function paintPickedTokens() {
  document.querySelectorAll(".token-chip").forEach(button => {
    button.classList.toggle("source", swapPick.source && Number(button.dataset.id) === swapPick.source.id);
    button.classList.toggle("target", swapPick.target && Number(button.dataset.id) === swapPick.target.id);
  });
}

function tokenButton(token, layer) {
  return `<button class="token-chip" data-id="${token.id}" data-text="${esc(token.text)}" data-layer="${layer}">
    ${esc(token.text)} <span class="token-score">${token.id} · ${token.score.toFixed(2)}</span>
  </button>`;
}

function renderDistribution(data) {
  const layers = data.layers.map(row => row.layer);
  document.querySelector("#swap-layer").innerHTML = layers.map(layer => `<option value="${layer}">${layer}</option>`).join("");
  if (swapPick.layer && layers.includes(swapPick.layer)) {
    document.querySelector("#swap-layer").value = swapPick.layer;
  }

  const prompt = data.prompt_tokens
    .map((token, i) => `<span class="tok a${i % 5}">${esc(token.text)}</span>`)
    .join("");
  const layerRows = data.layers.map(row => `
    <div class="layer-row">
      <div class="layer-label">L${row.layer}</div>
      <div>${row.tokens.map(token => tokenButton(token, row.layer)).join("")}</div>
    </div>
  `).join("");

  document.querySelector("#distribution-view").innerHTML = `
    <div class="prompt-tokens">${prompt}</div>
    ${layerRows}
  `;
  document.querySelectorAll(".token-chip").forEach(button => {
    button.onclick = () => pickSwapToken(
      {id: Number(button.dataset.id), text: button.dataset.text},
      Number(button.dataset.layer),
    );
  });
  paintPickedTokens();
  updateSwapControls();
}

function renderSwapResult(data) {
  function col(title, rows) {
    return `<div class="result-col">
      <h3>${title}</h3>
      ${rows.map(token => `
        <div class="result-token">
          <span>${esc(token.text)} <span class="token-score">${token.id}</span></span>
          <span>${token.score.toFixed(2)}</span>
        </div>
      `).join("")}
    </div>`;
  }
  document.querySelector("#swap-view").innerHTML = `
    <div class="result-grid">
      ${col("Baseline", data.baseline)}
      ${col("Swapped", data.swapped)}
    </div>
  `;
}

document.querySelector("#distribution").onclick = async () => {
  document.querySelector("#status").textContent = "Reading layer distributions";
  try {
    const data = await post("/api/distribution", {...fullBody(), top_k: topK()});
    renderDistribution(data);
    document.querySelector("#status").textContent = "Distribution loaded";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

document.querySelector("#swap").onclick = async () => {
  document.querySelector("#status").textContent = "Running swap";
  try {
    const data = await post("/api/swap", {
      ...fullBody(),
      source_token_id: swapPick.source.id,
      target_token_id: swapPick.target.id,
      layer: Number(document.querySelector("#swap-layer").value),
      strength: Number(document.querySelector("#strength").value || 1),
      top_k: topK(),
    });
    renderSwapResult(data);
    document.querySelector("#status").textContent = "Swap done";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

document.querySelector("#clear-swap").onclick = () => {
  swapPick = {source: null, target: null, layer: null};
  paintPickedTokens();
  updateSwapControls();
};

document.querySelector("#submit").onclick = async () => {
  document.querySelector("#status").textContent = "Running";
  try {
    const data = await post("/api/run", fullBody());
    lastHtml = data.html;
    document.querySelector("#report").srcdoc = lastHtml;
    document.querySelector("#export").disabled = false;
    document.querySelector("#status").textContent = "Done";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

document.querySelector("#export").onclick = () => {
  const blob = new Blob([lastHtml], {type: "text/html"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "jlens-report.html";
  a.click();
  URL.revokeObjectURL(a.href);
};

document.querySelector("#mode").onclick = () => {
  tokenMode = tokenMode === "text" ? "ids" : "text";
  document.querySelector("#mode").textContent = tokenMode === "text" ? "Token IDs" : "Text";
  tokenize();
};

document.querySelector("#question").oninput = scheduleTokenize;
document.querySelector("#model").onchange = scheduleTokenize;
document.querySelector("#architecture").onchange = scheduleTokenize;

["Paris", "London", "Berlin"].forEach(x => {
  document.querySelector("#choices").appendChild(choiceRow(x));
});
updateFocusChoices();
scheduleTokenize();
updateSwapControls();
</script>
</body>
</html>"""
