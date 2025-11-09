export default function App() {
  const html = `
    <div style="font-family: ui-sans-serif, system-ui; max-width: 720px; margin: 40px auto; padding: 16px;">
      <h1>AI-Powered Medical Image Analysis</h1>
      <p>Upload a chest X-ray (PNG/JPG). The API returns Normal vs Pneumonia and an optional Grad-CAM heatmap.</p>
      <div style="margin: 16px 0;">
        <input id="file" type="file" accept="image/*" />
        <label style="margin-left: 8px;"><input id="gradcam" type="checkbox" checked /> With Grad-CAM</label>
        <button id="btn" style="margin-left: 8px;">Predict</button>
      </div>
      <div id="status"></div>
      <div id="result" style="margin-top: 16px;"></div>
    </div>
  `
  setTimeout(() => {
    const btn = document.getElementById('btn')
    const fileEl = document.getElementById('file')
    const gradEl = document.getElementById('gradcam')
    const status = document.getElementById('status')
    const result = document.getElementById('result')
    const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/predict'

    btn.onclick = async () => {
      if (!fileEl.files?.[0]) { alert('Choose an image'); return; }
      status.textContent = 'Uploading...'
      result.innerHTML = ''
      const form = new FormData()
      form.append('file', fileEl.files[0])
      form.append('with_gradcam', gradEl.checked ? 'true' : 'false')
      try {
        const resp = await fetch(API_URL, { method: 'POST', body: form })
        const data = await resp.json()
        status.textContent = ''
        const prob = (data.probability * 100).toFixed(2)
        let html = `<h3>Prediction: ${data.predicted_class} (${prob}%)</h3>`
        if (data.gradcam_png_base64) {
          html += `<img style="max-width: 100%; border: 1px solid #ddd" src="data:image/png;base64,${data.gradcam_png_base64}" />`
        }
        result.innerHTML = html
      } catch (e) {
        console.error(e)
        status.textContent = 'Error calling API. Check console.'
      }
    }
  })
  return html
}
