"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { Shield, Search } from "lucide-react";
import TrueFocus from "@/components/TextAnimations/TrueFocus/TrueFocus";
import { Ripple } from "@/components/magicui/ripple";
import { FlipText } from "@/components/magicui/flip-text";
import { ShimmerButton } from "@/components/magicui/shimmer-button";
import { cn } from "@/lib/utils";

// ElegantShape component
function ElegantShape({
  className,
  delay = 0,
  width = 400,
  height = 100,
  rotate = 0,
  gradient = "from-white/[0.08]",
}: {
  className?: string;
  delay?: number;
  width?: number;
  height?: number;
  rotate?: number;
  gradient?: string;
}) {
  return (
    <motion.div
      initial={{
        opacity: 0,
        y: -150,
        rotate: rotate - 15,
      }}
      animate={{
        opacity: 1,
        y: 0,
        rotate: rotate,
      }}
      transition={{
        duration: 2.4,
        delay,
        ease: [0.23, 0.86, 0.39, 0.96],
        opacity: { duration: 1.2 },
      }}
      className={cn("absolute", className)}
    >
      <motion.div
        animate={{
          y: [0, 15, 0],
        }}
        transition={{
          duration: 12,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
        style={{
          width,
          height,
        }}
        className="relative"
      >
        <div
          className={cn(
            "absolute inset-0 rounded-full",
            "bg-gradient-to-r to-transparent",
            gradient,
            "backdrop-blur-[2px] border-2 border-white/[0.15]",
            "shadow-[0_8px_32px_0_rgba(255,255,255,0.1)]",
            "after:absolute after:inset-0 after:rounded-full",
            "after:bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.2),transparent_70%)]"
          )}
        />
      </motion.div>
    </motion.div>
  );
}

export function HeroSection() {
  const router = useRouter();

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-[#030303]">
      {/* Gradient backgrounds */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.05] via-transparent to-rose-500/[0.05] blur-3xl" />
      
      {/* Ripple effect */}
      <div className="absolute inset-0">
        <Ripple />
      </div>

      {/* Elegant shapes */}
      <div className="absolute inset-0 overflow-hidden">
        <ElegantShape
          delay={0.3}
          width={600}
          height={140}
          rotate={12}
          gradient="from-indigo-500/[0.25]"
          className="left-[-10%] md:left-[-5%] top-[15%] md:top-[20%]"
        />

        <ElegantShape
          delay={0.5}
          width={500}
          height={120}
          rotate={-15}
          gradient="from-rose-500/[0.25]"
          className="right-[-5%] md:right-[0%] top-[70%] md:top-[75%]"
        />

        <ElegantShape
          delay={0.7}
          width={150}
          height={40}
          rotate={-25}
          gradient="from-cyan-500/[0.25]"
          className="left-[20%] md:left-[25%] top-[5%] md:top-[10%]"
        />

        <ElegantShape
          delay={0.45}
          width={400}
          height={100}
          rotate={30}
          gradient="from-emerald-500/[0.25]"
          className="right-[25%] md:right-[10%] top-[5%] md:top-[10%]"
        />

        <ElegantShape
          delay={0.55}
          width={350}
          height={90}
          rotate={-20}
          gradient="from-purple-500/[0.25]"
          className="left-[10%] md:left-[10%] bottom-[15%] md:bottom-[25%]"
        />

        <ElegantShape
          delay={0.65}
          width={250}
          height={70}
          rotate={15}
          gradient="from-pink-500/[0.25]"
          className="right-[40%] md:right-[45%] bottom-[15%] md:bottom-[15%]"
        />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="flex flex-col items-center gap-4"
        >
          <FlipText
            className="text-[3.9375rem] md:text-[6.5625rem] font-bold"
            duration={0.5}
          >
            Fence AI
          </FlipText>

          <TrueFocus
            sentence="Protect Detect"
            manualMode={false}
            blurAmount={3}
            borderColor="#dc2626"
            glowColor="rgba(220, 38, 38, 0.6)"
            animationDuration={0.3}
            pauseBetweenAnimations={2}
          />
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto mt-6"
        >
          Defeating AI with AI
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="flex flex-col md:flex-row items-center justify-center gap-4"
        >
          <ShimmerButton className="w-auto px-6 py-3 rounded-md">
            <button
              onClick={() => router.push("/protect")}
              className="text-xl font-semibold text-white bg-transparent hover:bg-primary/20 transition-all duration-300"
            >
              <Shield className="w-5 h-5 mr-2 inline-block" />
              Protect
            </button>
          </ShimmerButton>

          <ShimmerButton className="w-auto px-6 py-3 rounded-md">
            <button
              onClick={() => router.push("/detect")}
              className="text-xl font-semibold text-white bg-transparent hover:bg-primary/20 transition-all duration-300"
            >
              <Search className="w-5 h-5 mr-2 inline-block" />
              Detect
            </button>
          </ShimmerButton>
        </motion.div>
      </div>

      {/* Overlay gradient */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#030303] via-transparent to-[#030303]/80 pointer-events-none" />
    </div>
  );
}
