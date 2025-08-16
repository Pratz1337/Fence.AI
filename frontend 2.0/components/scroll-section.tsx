"use client"

import type React from "react"

import { useRef, useEffect, useState } from "react"
import { motion, useScroll, useTransform } from "framer-motion"

interface ScrollSectionProps {
  title: string
  description: string
  bgColor?: string
  children?: React.ReactNode
  reversed?: boolean
}

export function ScrollSection({
  title,
  description,
  bgColor = "bg-background",
  children,
  reversed = false,
}: ScrollSectionProps) {
  const sectionRef = useRef<HTMLDivElement>(null)
  const [isVisible, setIsVisible] = useState(false)

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  })

  const opacity = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 1, 1, 0])
  const y = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [100, 0, 0, -100])

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting)
      },
      { threshold: 0.1 },
    )

    if (sectionRef.current) {
      observer.observe(sectionRef.current)
    }

    return () => {
      if (sectionRef.current) {
        observer.unobserve(sectionRef.current)
      }
    }
  }, [])

  return (
    <section
      ref={sectionRef}
      className={`min-h-screen flex items-center justify-center py-20 ${bgColor} overflow-hidden`}
    >
      <div className="container mx-auto px-4">
        <div className={`flex flex-col md:flex-row items-center gap-8 ${reversed ? "md:flex-row-reverse" : ""}`}>
          <motion.div className="flex-1" style={{ opacity, y }}>
            <h2 className="text-4xl md:text-5xl font-bold mb-4">{title}</h2>
            <p className="text-lg md:text-xl text-muted-foreground mb-6">{description}</p>
            {children}
          </motion.div>

          <motion.div
            className="flex-1 flex justify-center"
            initial={{ opacity: 0, x: reversed ? -100 : 100 }}
            animate={isVisible ? { opacity: 1, x: 0 } : { opacity: 0, x: reversed ? -100 : 100 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="relative w-full aspect-square max-w-md rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 shadow-xl">
              {/* Placeholder for image or content */}
              <div className="absolute inset-0 flex items-center justify-center text-4xl font-bold text-primary/40">
                {title.split(" ")[0]}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

