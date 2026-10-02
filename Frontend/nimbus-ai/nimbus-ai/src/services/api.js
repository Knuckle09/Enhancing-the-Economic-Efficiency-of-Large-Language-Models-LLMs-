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
    const timeout = setTimeout(() => controller.abort(), 120000);
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
