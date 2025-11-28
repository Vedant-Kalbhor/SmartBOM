import React from "react";
import { Button, Card, Form, Input, Typography } from "antd";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const { Title, Text } = Typography;

const LoginPage = () => {
  const [form] = Form.useForm();
  const { login } = useAuth();
  const navigate = useNavigate();

  const onFinish = async (values) => {
    const ok = await login(values);
    if (ok) {
      navigate("/"); // or /dashboard
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#f5f5f5",
      }}
    >
      <Card style={{ width: 400 }}>
        <Title level={3} style={{ textAlign: "center", marginBottom: 24 }}>
          SmartBOM Login
        </Title>

        <Form form={form} layout="vertical" onFinish={onFinish}>
          <Form.Item
            label="Email"
            name="email"
            rules={[
              { required: true, message: "Please enter your email" },
              { type: "email", message: "Invalid email" },
            ]}
          >
            <Input placeholder="you@example.com" />
          </Form.Item>

          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: "Please enter your password" }]}
          >
            <Input.Password placeholder="••••••••" />
          </Form.Item>

          <Form.Item style={{ marginBottom: 12 }}>
            <Button type="primary" htmlType="submit" block>
              Login
            </Button>
          </Form.Item>
        </Form>

        <Text type="secondary">
          Don&apos;t have an account? <Link to="/signup">Sign up</Link>
        </Text>
      </Card>
    </div>
  );
};

export default LoginPage;
