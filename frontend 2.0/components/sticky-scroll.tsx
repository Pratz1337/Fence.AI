"use client"

import { useRef, useState, useEffect } from "react"
import { motion, useScroll } from "framer-motion"

interface StickyScrollProps {
  sections: {
    title: string
    description: string
    image?: string
  }[]
}

export function StickyScroll({ sections }: StickyScrollProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [activeSection, setActiveSection] = useState(0)

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  })

  const sectionHeight = 100 / sections.length

  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange((value) => {
      const sectionIndex = Math.min(Math.floor(value * sections.length), sections.length - 1)
      setActiveSection(sectionIndex)
    })

    return () => unsubscribe()
  }, [scrollYProgress, sections.length])

  return (
    <div
      ref={containerRef}
      className="relative h-[400vh]" // Height based on number of sections
    >
      <div className="sticky top-0 h-screen flex items-center overflow-hidden">
        <div className="container mx-auto px-4 grid grid-cols-1 md:grid-cols-2 gap-8 h-full">
          <div className="flex flex-col justify-center">
            {sections.map((section, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 50 }}
                animate={{
                  opacity: activeSection === index ? 1 : 0.3,
                  y: activeSection === index ? 0 : 50,
                }}
                transition={{ duration: 0.5 }}
                className="mb-8"
              >
                <h3 className="text-3xl font-bold mb-2">{section.title}</h3>
                <p className="text-muted-foreground">{section.description}</p>
              </motion.div>
            ))}
          </div>

          <div className="flex items-center justify-center">
            <div className="relative w-full aspect-video bg-muted rounded-xl overflow-hidden">
              {sections.map((section, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0 }}
                  animate={{
                    opacity: activeSection === index ? 1 : 0,
                  }}
                  transition={{ duration: 0.5 }}
                  className="absolute inset-0 flex items-center justify-center"
                >
                  {section.image ? (
                    <img
                      src={section.image || "/placeholder.svg"}
                      alt={section.title}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-4xl font-bold text-primary/20">Section {index + 1}</div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

