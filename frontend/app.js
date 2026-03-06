const API_URL = "http://127.0.0.1:8000/query";

const chatWindow = document.getElementById("chat-window");
const form = document.getElementById("query-form");
const textarea = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(role, text, contexts) {
  const message = document.createElement("div");
  message.className = `message message-${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar avatar-${role}`;
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = `bubble bubble-${role}`;
  bubble.innerText = text;

  if (role === "assistant" && Array.isArray(contexts) && contexts.length > 0) {
    const contextsEl = document.createElement("div");
    contextsEl.className = "contexts";
    const label = document.createElement("div");
    label.textContent = "Answer grounded in:";
    contextsEl.appendChild(label);

    contexts.slice(0, 4).forEach((ctx) => {
      const pill = document.createElement("span");
      pill.className = "context-pill";

      const page = ctx.page !== null && ctx.page !== undefined ? ctx.page + 1 : "?";
      const pageSpan = document.createElement("span");
      pageSpan.textContent = `Page ${page}`;
      pill.appendChild(pageSpan);

      contextsEl.appendChild(pill);
    });

    bubble.appendChild(contextsEl);
  }

  message.appendChild(avatar);
  message.appendChild(bubble);
  chatWindow.appendChild(message);

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setLoading(isLoading) {
  sendBtn.disabled = isLoading;
  sendBtn.querySelector(".send-label").textContent = isLoading ? "Thinking…" : "Ask";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;

  appendMessage("user", question);
  textarea.value = "";
  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    const answer = data.answer || "I couldn’t generate an answer.";
    const contexts = data.contexts || [];
    appendMessage("assistant", answer, contexts);
  } catch (error) {
    console.error(error);
    appendMessage(
      "assistant",
      "Something went wrong while contacting the RAG backend. Please ensure the API is running on http://localhost:8000 and try again."
    );
  } finally {
    setLoading(false);
  }
});

textarea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

