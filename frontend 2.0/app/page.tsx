"use client"

import { useRef, useEffect } from "react"
import { Navbar } from "@/components/navbar"
import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { Footer } from "@/components/footer"
import { ScrollSection } from "@/components/scroll-section"
import { RevealSection } from "@/components/reveal-section"

export default function Home() {
  const sectionsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Smooth scroll behavior for the entire page
    document.documentElement.style.scrollBehavior = "smooth"

    return () => {
      document.documentElement.style.scrollBehavior = "auto"
    }
  }, [])

  return (
    <main className="min-h-screen relative">
      <Navbar />

      {/* Hero with parallax effect */}
      <div className="relative h-screen overflow-hidden">
        <HeroSection />
      </div>

      {/* Scrolling sections container */}
      <div ref={sectionsRef} className="relative z-10">
        {/* Features section with reveal animation */}
        <RevealSection>
          <FeaturesSection />
        </RevealSection>

        {/* Additional stacked sections */}
        <ScrollSection
          bgColor="bg-primary/5"
          title="Our Mission"
          description="We're dedicated to creating exceptional experiences that transform how people interact with technology."
        />

        <ScrollSection
          bgColor="bg-secondary/10"
          title="Our Process"
          description="From concept to completion, we follow a rigorous process to ensure every project exceeds expectations."
          reversed
        />

        <ScrollSection
          bgColor="bg-accent/5"
          title="Our Results"
          description="Our clients see measurable improvements in engagement, conversion, and satisfaction."
        />
      </div>

      <Footer />
    </main>
  )
}

