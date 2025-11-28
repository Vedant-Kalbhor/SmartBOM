import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const authApi = axios.create({
  baseURL: API_BASE_URL,
});

authApi.interceptors.request.use((config) => {
  const token = localStorage.getItem("smartbom_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export function signup(payload) {
  // payload: { email, full_name, password }
  return authApi.post("/auth/signup", payload);
}

export function login({ email, password }) {
  // JSON body, matches UserLogin in main.py
  return authApi.post("/auth/login", { email, password });
}

export function getCurrentUser() {
  return authApi.get("/auth/me");
}
