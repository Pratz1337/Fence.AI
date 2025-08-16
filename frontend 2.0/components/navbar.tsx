"use client";

import { motion } from "framer-motion";
import { Shield, Search } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrolled
          ? "bg-background/80 backdrop-blur-lg border-b"
          : "bg-transparent"
      )}
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center space-x-2">
            <Shield className="w-8 h-8" />
            <span className="font-space font-bold text-xl">GuardAI</span>
          </Link>
          
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/protect"
              className="flex items-center space-x-2 hover:text-primary transition-colors"
            >
              <Shield className="w-4 h-4" />
              <span>Protect</span>
            </Link>
            <Link
              href="/detect"
              className="flex items-center space-x-2 hover:text-primary transition-colors"
            >
              <Search className="w-4 h-4" />
              <span>Detect</span>
            </Link>
          </div>
        </div>
      </div>
    </motion.nav>
  );
}