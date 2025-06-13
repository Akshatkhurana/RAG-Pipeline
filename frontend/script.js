async function sendQuery() {
  const question = document.getElementById("query").value;

  const response = await fetch("http://localhost:5000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const data = await response.json();
  console.log("Backend response:", data);  // ✅ Debug log

  document.getElementById("response").innerText =
    data.answer || "Error retrieving answer";
}
