import { Link } from "react-router-dom";
import "../styles/auth.css";

function Login() {
  return (
    <div className="auth-container">
      <div className="auth-card">

        <h1>Login</h1>

        <form>

          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              placeholder="Enter your email"
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
            />
          </div>

          <button className="auth-btn" type="submit">
            Login
          </button>

        </form>

        <p className="bottom-text">
          Don't have an account?{" "}
          <Link to="/signup">Sign Up</Link>
        </p>

      </div>
    </div>
  );
}

export default Login;