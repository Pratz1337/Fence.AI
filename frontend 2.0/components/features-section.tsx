"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { Shield, Search, Lock, AlertTriangle, CheckCircle, Zap } from "lucide-react";

const features = [
  {
    icon: Shield,
    title: "Advanced Protection",
    description: "Protect your media content with state-of-the-art AI algorithms",
  },
  {
    icon: Search,
    title: "Deep Detection",
    description: "Identify manipulated content with high accuracy and precision",
  },
  {
    icon: Lock,
    title: "Secure Storage",
    description: "Your media is stored with enterprise-grade encryption",
  },
  {
    icon: AlertTriangle,
    title: "Early Warning",
    description: "Get instant alerts when potential deepfakes are detected",
  },
  {
    icon: CheckCircle,
    title: "Verified Content",
    description: "Ensure your content remains authentic and unaltered",
  },
  {
    icon: Zap,
    title: "Real-time Analysis",
    description: "Process and analyze media content in real-time",
  },
];

export function FeaturesSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section ref={ref} className="py-24 bg-secondary/50">
      <div className="container mx-auto px-4">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-3xl md:text-4xl font-space font-bold text-center mb-16"
        >
          Powerful Features
        </motion.h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="p-6 rounded-lg bg-card hover:bg-card/80 transition-colors"
            >
              <feature.icon className="w-10 h-10 mb-4 text-primary" />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}