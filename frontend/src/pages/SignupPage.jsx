import React from "react";
import { Button, Card, Form, Input, Typography } from "antd";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const { Title, Text } = Typography;

const SignupPage = () => {
  const [form] = Form.useForm();
  const { signup } = useAuth();
  const navigate = useNavigate();

  const onFinish = async (values) => {
    const ok = await signup(values);
    if (ok) {
      navigate("/login");
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
          Create Account
        </Title>

        <Form form={form} layout="vertical" onFinish={onFinish}>
          <Form.Item
            label="Full Name"
            name="full_name"
            rules={[{ required: true, message: "Please enter your name" }]}
          >
            <Input placeholder="Your name" />
          </Form.Item>

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
            rules={[
              { required: true, message: "Please set a password" },
              { min: 6, message: "Password should be at least 6 characters" },
            ]}
          >
            <Input.Password placeholder="••••••••" />
          </Form.Item>

          <Form.Item style={{ marginBottom: 12 }}>
            <Button type="primary" htmlType="submit" block>
              Sign up
            </Button>
          </Form.Item>
        </Form>

        <Text type="secondary">
          Already have an account? <Link to="/login">Login</Link>
        </Text>
      </Card>
    </div>
  );
};

export default SignupPage;
