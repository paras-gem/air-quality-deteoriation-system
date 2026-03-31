"use client";

import { Wind } from "lucide-react";

const footerLinks = [
  { label: "About", href: "#about" },
  { label: "Project Details", href: "#project" },
  { label: "Developer Credits", href: "#credits" },
];

export default function Footer() {
  const scrollTo = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault();
    const id = href.replace("#", "");
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-layout">
          {/* Brand */}
          <div className="footer-brand">
            <div className="footer-logo-icon">
              <Wind />
            </div>
            <span className="footer-brand-title">
              <span className="footer-brand-p1">Air</span>
              <span className="footer-brand-p2">Q</span>
            </span>
            <span className="footer-brand-subtitle">AI Air Quality Prediction</span>
          </div>

          {/* Links */}
          <nav className="footer-nav">
            {footerLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                onClick={(e) => scrollTo(e, link.href)}
                className="footer-link"
              >
                {link.label}
              </a>
            ))}
          </nav>

          {/* Copyright */}
          <p className="footer-copyright">
            © 2024 AirQ. Developed by{" "}
            <span>Pratham</span> · BCA, Christ University
          </p>
        </div>
      </div>
    </footer>
  );
}
