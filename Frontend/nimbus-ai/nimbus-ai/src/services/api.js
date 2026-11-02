// services/api.js
// Base API configuration and utilities

const rawBase =
  import.meta.env.VITE_API_URL ||
  "https://llm-efficiency-backend.onrender.com";
const API_BASE_URL = String(rawBase).replace(/\/+$/, "");

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
    console.log("API Service initialized with URL:", this.baseURL);
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 600000);
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: controller.signal,
      ...options,
    };
    try {
      const response = await fetch(url, config);
      clearTimeout(timeout);
      let body;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      if (!response.ok || body?.success === false) {
        throw new Error(
          body?.error || body?.detail || `The server could not complete the request (HTTP ${response.status}).`
        );
      }
      if (!body) {
        throw new Error('The server returned an invalid response. Please retry.');
      }
      return body;
    } catch (error) {
      clearTimeout(timeout);
      console.error('API request failed:', error);
      if (error.name === 'AbortError') {
        throw new Error('The response took too long. The server may be waking up; please retry.');
      }
      throw error;
    }
  }

  async get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  }

  async post(endpoint, data) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async stream(endpoint, data, handlers = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 600000);
    let completedResponse = null;
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      if (!response.ok) {
        let body = null;
        try { body = await response.json(); } catch { /* handled below */ }
        throw new Error(body?.error || body?.detail || `The server could not complete the request (HTTP ${response.status}).`);
      }
      if (!response.body) throw new Error('The server did not open a response stream. Please retry.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const handleEvent = (frame) => {
        const line = frame.split(/\r?\n/).find((item) => item.startsWith('data:'));
        if (!line) return;
        let event;
        try { event = JSON.parse(line.slice(5).trim()); }
        catch { throw new Error('The server sent an invalid streaming response. Please retry.'); }

        if (event.type === 'progress') handlers.onProgress?.(event.message || 'Working…');
        if (event.type === 'delta') handlers.onDelta?.(event.text || '');
        if (event.type === 'error') throw new Error(event.message || 'The model could not complete this request.');
        if (event.type === 'complete') completedResponse = event.response;
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || '';
        frames.forEach(handleEvent);
        if (done) break;
      }
      if (buffer.trim()) handleEvent(buffer);
      if (!completedResponse) throw new Error('The response stream closed before the model finished. Please retry.');
      return completedResponse;
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('This request exceeded the 10-minute response window. Shorten the request or try Gemini Flash.');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  async put(endpoint, data) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
}

export default new ApiService();
