// services/api.js
// Base API configuration and utilities

const rawBase =
  import.meta.env.VITE_API_URL ||
  "https://enhancing-the-economic-efficiency-of.onrender.com";
const API_BASE_URL = String(rawBase).replace(/\/+$/, "");

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
    console.log("API Service initialized with URL:", this.baseURL);
  }

async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
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
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      clearTimeout(timeout);
      console.error('API request failed:', error);
      if (error.name === 'AbortError') {
        throw new Error('Server is waking up, please try again in a moment.');
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
