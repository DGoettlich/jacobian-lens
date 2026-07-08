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
  <iframe id="report"></iframe>

<script>
let tokenMode = "text";
let lastHtml = null;
let tokenTimer = null;

function esc(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
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
    document.querySelector("#status").textContent = "Stopped";
  } catch (err) {
    document.querySelector("#status").textContent = err.message;
  }
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
</script>
</body>
</html>"""
