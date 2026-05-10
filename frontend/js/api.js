// Lightweight fetch wrappers + NDJSON streaming helper.
const API = (() => {
  async function jsonGet(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} → ${r.status}`);
    return r.json();
  }

  async function jsonPost(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${url} → ${r.status}`);
    return r.json();
  }

  async function uploadCsv(file, probType) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('prob_type', probType);
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`upload → ${r.status}`);
    return r.json();
  }

  // Stream NDJSON: yields each parsed JSON object as it arrives.
  async function streamNdjson(url, body, onMessage) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) throw new Error(`${url} → ${r.status}`);

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) {
          try { onMessage(JSON.parse(line)); }
          catch (e) { console.warn('bad ndjson line', line, e); }
        }
      }
    }
    if (buf.trim()) {
      try { onMessage(JSON.parse(buf.trim())); } catch {}
    }
  }

  return { jsonGet, jsonPost, uploadCsv, streamNdjson };
})();
