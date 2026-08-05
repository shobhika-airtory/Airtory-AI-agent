const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const refreshBtn = document.getElementById("refresh-btn");

let sessionId = localStorage.getItem("airtory_session_id") || null;

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function showGreeting() {
  addMessage(
    "Heyy! 👋 I'm the Airtory Creative Agent, and I'm here to help you bring your ad to life. " +
      "Just tell me what you'd like to build — e.g. \"Create a Quiz-n-Win ad for the Domino's " +
      "campaign\" — and I'll walk you through the rest, step by step.",
    "bot"
  );
}

showGreeting();

async function refreshMemory() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Refreshing…";

  try {
    // Actually wipe the backend's memory of this conversation (the
    // SESSIONS/WIZARDS dicts in main.py) -- not just a cosmetic page
    // reload. This is a real network call, not window.location.reload().
    if (sessionId) {
      await fetch(`/reset?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
      });
    }
  } catch (err) {
    console.error("Failed to reset session on the server:", err);
    // Even if the network call fails, still wipe the frontend below so
    // the button never leaves the user stuck -- worst case the old
    // session lingers harmlessly in server memory, but the user gets a
    // fresh chat either way.
  }

  localStorage.removeItem("airtory_session_id");
  sessionId = null;

  // Wipe the chat window directly instead of reloading the page. This
  // guarantees the reset actually happens the instant it's clicked --
  // no dependency on a page reload picking up the latest cached script.
  chatWindow.innerHTML = "";
  showGreeting();

  refreshBtn.disabled = false;
  refreshBtn.textContent = "↻ Refresh";
}

refreshBtn.addEventListener("click", refreshMemory);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  const statusMsg = addMessage("Uploading image…", "thinking");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload-image", { method: "POST", body: formData });
    const data = await res.json();

    statusMsg.remove();

    if (!res.ok) {
      addMessage(`Upload failed: ${data.detail || "unknown error"}`, "bot");
      return;
    }

    const url = data.url;
    const start = chatInput.selectionStart;
    const end = chatInput.selectionEnd;
    const current = chatInput.value;
    chatInput.value = current.slice(0, start) + url + current.slice(end);
    chatInput.focus();
    chatInput.selectionStart = chatInput.selectionEnd = start + url.length;
    chatInput.dispatchEvent(new Event("input"));
  } catch (err) {
    statusMsg.remove();
    addMessage(`Upload failed: ${err.message}`, "bot");
  }

  fileInput.value = "";
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  chatInput.value = "";
  chatInput.style.height = "auto";

  const thinking = addMessage("thinking…", "thinking");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem("airtory_session_id", sessionId);

    thinking.remove();
    addMessage(data.reply, "bot");
  } catch (err) {
    thinking.remove();
    addMessage(`Something went wrong: ${err.message}`, "bot");
  }
});