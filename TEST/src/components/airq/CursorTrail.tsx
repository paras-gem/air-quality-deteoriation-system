"use client";

import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  alpha: number;
  radius: number;
}

export default function CursorTrail() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animFrameRef = useRef<number>(0);
  const aqiRef = useRef<number>(300); // Default to preview max smoke at 300 AQI

  useEffect(() => {
    const handleAqi = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail?.aqi) {
        aqiRef.current = customEvent.detail.aqi;
      }
    };
    window.addEventListener("aqi-update", handleAqi);

    const canvas = canvasRef.current;
    const cursor = cursorRef.current;
    if (!canvas || !cursor) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      // MASSIVE OPTIMIZATION: Render at 50% resolution to skip 75% of GPU blur calculation
      canvas.width = window.innerWidth / 2;
      canvas.height = window.innerHeight / 2;
      ctx.setTransform(0.5, 0, 0, 0.5, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    let lastX = 0;
    let lastY = 0;

    const onMove = (e: MouseEvent) => {
      if (cursor) {
        cursor.style.left = `${e.clientX}px`;
        cursor.style.top = `${e.clientY}px`;
      }

      const dist = Math.hypot(e.clientX - lastX, e.clientY - lastY);
      if (dist < 15) return; // Performance optimization: Throttle spammy spawns

      lastX = e.clientX;
      lastY = e.clientY;

      // Calculate smoke thickness based on AQI
      // AQI 30 -> 1.0x density. AQI 300+ -> max 10.0x density!
      const densityMultiplier = Math.max(0.2, Math.min(10.0, aqiRef.current / 30));
      // Drastically reduced spawn count to fix lag, compensated by larger sizes below
      const spawnCount = Math.ceil(2.5 * densityMultiplier);

      // Spawn soft grey smoke puffs
      for (let i = 0; i < spawnCount; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 2.0 + 0.8;
        particlesRef.current.push({
          x: e.clientX + (Math.random() - 0.5) * 20,
          y: e.clientY + (Math.random() - 0.5) * 20,
          vx: Math.cos(angle) * speed + (Math.random() - 0.5),
          vy: Math.sin(angle) * speed * 0.4,
          // Higher opacity to compensate for fewer particles
          alpha: (Math.random() * 0.15 + 0.05) * (densityMultiplier * 1.2),
          // Huge particles to cover more area with less resources
          radius: Math.random() * 70 + 40,
        });
      }
    };

    window.addEventListener("mousemove", onMove);

    const animate = () => {
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset for clearRect
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(0.5, 0, 0, 0.5, 0, 0); // Re-apply 50% scaling

      // Optimization: Cull particles faster to save memory
      particlesRef.current = particlesRef.current.filter((p) => p.alpha > 0.02);

      for (const p of particlesRef.current) {
        p.x += p.vx + Math.sin(p.alpha * 15) * 0.4; // Swirl horizontally
        p.y += p.vy;
        p.vy -= 0.05; // Hot smoke rises up!
        p.vx *= 0.95; // Air resistance slows horizontal spreading
        p.alpha *= 0.94; // Fade out naturally
        p.radius *= 1.03; // Smoke expands strongly as it rises

        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius);
        grad.addColorStop(0, `rgba(180, 190, 200, ${p.alpha})`);
        grad.addColorStop(0.3, `rgba(160, 170, 180, ${p.alpha * 0.4})`);
        grad.addColorStop(1, `rgba(160, 170, 180, 0)`);

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  return (
    <>
      <canvas ref={canvasRef} id="smoke-canvas" style={{ filter: 'blur(12px)', opacity: 0.85 }} />
      <div ref={cursorRef} className="custom-cursor" />
    </>
  );
}
