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
    .hint { color: #57606a; margin: -4px 0 12px; max-width: 860px; }
    input.small { width: 72px; }
    .intervention-row { display: grid; grid-template-columns: 180px 1fr; gap: 12px; align-items: start; margin: 10px 0; }
    .intervention-row input { width: 170px; }
    .token-check { line-height: 1.9; min-height: 38px; padding: 8px; background: #f6f8fa; border-radius: 6px; }
    .check-title { color: #57606a; font-size: 12px; margin-bottom: 4px; }
    .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; max-width: 900px; }
    .result-col h3 { margin: 8px 0; font-size: 15px; }
    .generated { white-space: pre-wrap; border: 1px solid #d0d7de; border-radius: 6px; padding: 10px; min-height: 70px; }
    .generated-token { margin: 2px; padding: 3px 5px; border-radius: 4px; display: inline-block; background: #ddf4ff; }
    .muted { color: #57606a; }
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
    <h2>Generate With Intervention</h2>
    <div class="hint">
      Source and target are only tokenization checks for <code>question + " " + token</code>.
      Generation starts from the question alone; generated tokens are appended separately below.
    </div>
    <div class="intervention-row">
      <label>Source token <input id="source" value="Paris"></label>
      <div id="source-check" class="token-check"></div>
    </div>
    <div class="intervention-row">
      <label>Target token <input id="target" value="London"></label>
      <div id="target-check" class="token-check"></div>
    </div>
    <div class="row">
      <label>Strength <input id="strength" class="small" type="number" value="1" min="0" step="0.1"></label>
      <label>Max tokens <input id="max-tokens" class="small" type="number" value="16" min="1" max="64"></label>
      <button id="swap" disabled>Swap</button>
      <button id="steer" disabled>Steer</button>
    </div>
    <div id="generation-view"></div>
  </div>

  <iframe id="report"></iframe>

<script>
let tokenMode = "text";
let lastHtml = null;
let tokenTimer = null;
let interventionTimer = null;

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
  row.querySelector(".tokens").innerHTML = tokenizationHtml(data);
}

function tokenizationHtml(data) {
  const spans = data.spans.map((t, i) => {
    const text = tokenMode === "ids" ? t.id : esc(t.text);
    const answer = t.answer ? "answer" : "";
    return `<span class="tok a${i % 5} ${answer}">${text}</span>`;
  }).join("");
  const status = data.single_token
    ? `<span class="ok">single answer token: ${data.answer_ids[0]}</span>`
    : `<span class="bad">answer tokens: [${data.answer_ids.join(", ")}]</span>`;
  return `<div class="check-title">Tokenization check only: question + candidate</div>${spans}<br>${status}`;
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
  scheduleInterventionCheck();
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
    document.querySelector("#swap").disabled = false;
    document.querySelector("#steer").disabled = false;
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
    document.querySelector("#swap").disabled = true;
    document.querySelector("#steer").disabled = true;
    document.querySelector("#status").textContent = "Stopped";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
};

async function checkInterventionTokens() {
  const source = document.querySelector("#source").value.trim();
  const target = document.querySelector("#target").value.trim();
  const terms = [];
  const slots = [];
  document.querySelector("#source-check").innerHTML = "";
  document.querySelector("#target-check").innerHTML = "";
  if (source) {
    terms.push(source);
    slots.push(document.querySelector("#source-check"));
  }
  if (target) {
    terms.push(target);
    slots.push(document.querySelector("#target-check"));
  }
  if (!terms.length) return;

  const data = await post("/api/tokenize", {...fullBody(), choices: terms});
  data.rows.forEach((row, i) => {
    slots[i].innerHTML = tokenizationHtml(row);
  });
}

function scheduleInterventionCheck() {
  clearTimeout(interventionTimer);
  interventionTimer = setTimeout(checkInterventionTokens, 250);
}

function renderGeneration(data, label) {
  function col(title, generated) {
    return `<div class="result-col">
      <h3>${title}</h3>
      <div class="generated">${generated.tokens.map(token => (
        `<span class="generated-token" title="${token.id}">${esc(token.text)}</span>`
      )).join("")}</div>
      <div class="muted">${generated.tokens.map(token => token.id).join(" ")}</div>
    </div>`;
  }
  document.querySelector("#generation-view").innerHTML = `
    <div class="muted">${label}; edited layers ${data.layers[0]}-${data.layers[data.layers.length - 1]}</div>
    <div class="result-grid">
      ${col("Baseline", data.baseline)}
      ${col("Intervened", data.intervened)}
    </div>
  `;
}

async function generate(mode) {
  document.querySelector("#status").textContent = mode === "swap" ? "Generating swap" : "Generating steer";
  try {
    const data = await post("/api/generate", {
      ...fullBody(),
      mode,
      source: document.querySelector("#source").value.trim(),
      target: document.querySelector("#target").value.trim(),
      strength: Number(document.querySelector("#strength").value || 1),
      max_tokens: Number(document.querySelector("#max-tokens").value || 16),
    });
    renderGeneration(data, mode === "swap" ? "Swap" : "Steer");
    document.querySelector("#status").textContent = "Done";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
}

document.querySelector("#swap").onclick = () => generate("swap");
document.querySelector("#steer").onclick = () => generate("steer");

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
  checkInterventionTokens();
};

document.querySelector("#question").oninput = scheduleTokenize;
document.querySelector("#model").onchange = scheduleTokenize;
document.querySelector("#architecture").onchange = scheduleTokenize;
document.querySelector("#source").oninput = scheduleInterventionCheck;
document.querySelector("#target").oninput = scheduleInterventionCheck;

["Paris", "London", "Berlin"].forEach(x => {
  document.querySelector("#choices").appendChild(choiceRow(x));
});
updateFocusChoices();
scheduleTokenize();
</script>
</body>
</html>"""
