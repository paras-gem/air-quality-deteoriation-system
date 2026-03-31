import { useState, useEffect } from "react";
import { Wind, Menu, X } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

const navLinks = [
  { label: "Home", href: "/" },
  { label: "Predict", href: "/predict" },
  { label: "Health", href: "/health" },
  { label: "Alerts", href: "/alerts" },
  { label: "About", href: "/about" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`navbar ${scrolled ? "navbar-scrolled" : ""}`}>
      <div className="navbar-container">
        {/* Logo */}
        <Link
          to="/"
          className="nav-logo"
          onClick={() => setMobileOpen(false)}
        >
          <div className="nav-logo-icon">
            <Wind />
          </div>
          <span className="nav-logo-text">
            <span className="nav-logo-text-p1">Air</span>
            <span className="nav-logo-text-p2">Q</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="nav-links-desktop">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.href;
            return (
              <Link
                key={link.label}
                to={link.href}
                className={`nav-link ${isActive ? "active" : ""}`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* CTA */}
        <Link
          to="/predict"
          className="nav-cta-desktop glow-btn"
        >
          Run a Prediction
        </Link>

        {/* Mobile toggle */}
        <button
          className="nav-mobile-toggle"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="nav-mobile-menu">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.href;
            return (
              <Link
                key={link.label}
                to={link.href}
                onClick={() => setMobileOpen(false)}
                className={`nav-link-mobile ${isActive ? "active" : ""}`}
              >
                {link.label}
              </Link>
            )
          })}
          <Link
            to="/predict"
            onClick={() => setMobileOpen(false)}
            className="nav-cta-mobile"
          >
            Run a Prediction
          </Link>
        </div>
      )}
    </header>
  );
}
