const pdf = document.getElementById("pdf");
const q = document.getElementById("q");

const btnIngest = document.getElementById("btnIngest");
const btnAsk = document.getElementById("btnAsk");
const btnExample = document.getElementById("btnExample");
const btnClear = document.getElementById("btnClear");

const statusEl = document.getElementById("status");
const answerText = document.getElementById("answerText");
const sourcesList = document.getElementById("sourcesList");

function setBusy(isBusy){
  btnIngest.disabled = isBusy;
  btnAsk.disabled = isBusy;
  btnExample.disabled = isBusy;
  btnClear.disabled = isBusy;
}

function setStatus(msg){ statusEl.textContent = msg || ""; }

function showAnswer(text){
  answerText.textContent = text || "";
}

function showSources(arr){
  sourcesList.innerHTML = "";
  (arr || []).forEach(s => {
    const li = document.createElement("li");
    li.textContent = s;
    sourcesList.appendChild(li);
  });
}

btnExample.onclick = () => {
  q.value = "What is the title of the document?";
  q.focus();
};

btnClear.onclick = () => {
  setStatus("");
  showAnswer("");
  showSources([]);
  q.value = "";
  pdf.value = "";
};

btnIngest.onclick = async () => {
  const f = pdf.files[0];
  if (!f) { setStatus("Choose a PDF first."); return; }

  setBusy(true);
  setStatus("Uploading PDF and building index...");
  showAnswer("");
  showSources([]);

  try{
    const fd = new FormData();
    fd.append("file", f);

    const r = await fetch("/ingest", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());

    const data = await r.json().catch(() => null);
    setStatus(data ? `Ingested: ${data.saved_as}` : "Ingest complete.");
  } catch (e){
    setStatus("Ingest error: " + e.message);
  } finally{
    setBusy(false);
  }
};

btnAsk.onclick = async () => {
  const question = q.value.trim();
  if (!question) { setStatus("Type a question first."); return; }

  setBusy(true);
  setStatus("Retrieving context + generating answer...");
  showAnswer("");
  showSources([]);

  try{
    const r = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!r.ok) throw new Error(await r.text());

    const data = await r.json();
    showAnswer(data.answer || "");
    showSources(data.sources || []);
    setStatus("Done.");
  } catch (e){
    setStatus("Query error: " + e.message);
  } finally{
    setBusy(false);
  }
};

// Enter key triggers Ask
q.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") btnAsk.click();
});
