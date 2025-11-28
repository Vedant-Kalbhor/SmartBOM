import React, { createContext, useContext, useEffect, useState } from "react";
import { getCurrentUser, login as apiLogin, signup as apiSignup } from "../services/auth";
import { message } from "antd";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [initializing, setInitializing] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("smartbom_token");
    if (!token) {
      setInitializing(false);
      return;
    }

    getCurrentUser()
      .then((res) => {
        setUser(res.data);
      })
      .catch(() => {
        localStorage.removeItem("smartbom_token");
      })
      .finally(() => setInitializing(false));
  }, []);

  const handleLogin = async ({ email, password }) => {
    try {
      const res = await apiLogin({ email, password });
      const token = res.data.access_token;
      localStorage.setItem("smartbom_token", token);

      const me = await getCurrentUser();
      setUser(me.data);
      message.success("Logged in successfully");
      return true;
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Login failed");
      return false;
    }
  };

  const handleSignup = async ({ email, full_name, password }) => {
    try {
      await apiSignup({ email, full_name, password });
      message.success("Signup successful. You can now login.");
      return true;
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Signup failed");
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem("smartbom_token");
    setUser(null);
    message.info("Logged out");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        initializing,
        login: handleLogin,
        signup: handleSignup,
        logout,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
