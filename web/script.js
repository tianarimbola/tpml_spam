const form = document.getElementById('predict-form');
const textInput = document.getElementById('text');
const resultBox = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text) {
    resultBox.innerHTML = '<div class="muted">Veuillez entrer un message.</div>';
    return;
  }

  resultBox.innerHTML = '<div class="muted">Prédiction en cours…</div>';
  form.querySelector('button').disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Erreur serveur');

    const labelClass = data.is_spam ? 'label-spam' : 'label-ham';

    resultBox.innerHTML = `
      <div><strong>Résultat:</strong> <span class="${labelClass}">${data.label}</span></div>
      <div class="muted">Confiance: ${data.confidence_percent.toFixed(2)}%</div>
      <div style="margin-top:8px"><strong>Message:</strong> ${escapeHtml(data.message)}</div>
    `;
  } catch (err) {
    resultBox.innerHTML = `<div class="muted">Erreur: ${escapeHtml(err.message)}</div>`;
  } finally {
    form.querySelector('button').disabled = false;
  }
});

function escapeHtml(str){
  return str.replace(/[&<>"']/g, (s)=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"
  }[s]));
}
