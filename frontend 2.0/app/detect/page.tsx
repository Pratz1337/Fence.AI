"use client"

import type React from "react"
import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import ReactJson from "react-json-view"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import {
  Search,
  Upload,
  FileImage,
  Shield,
  AlertTriangle,
  CheckCircle,
  Info,
  X,
  Loader2,
  Gauge,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile: File) => {
    // Check if file is an image or video
    if (!selectedFile.type.startsWith("image/") && !selectedFile.type.startsWith("video/")) {
      setError("Please upload an image or video file (JPG, PNG, WEBP, MP4, AVI, MOV)")
      return
    }

    setFile(selectedFile)
    setError(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(selectedFile)
  }

  const analyzeMedia = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setAnalysisProgress(0)
    setResults(null)
    setError(null)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setAnalysisProgress((prev) => {
        if (prev >= 95) {
          clearInterval(progressInterval)
          return prev
        }
        return prev + Math.random() * 15
      })
    }, 300)

    try {
      const formData = new FormData()
      formData.append("file", file)

      // Determine endpoint based on file type
      const endpoint = file.type.startsWith("video/")
        ? "https://5000-01jnecfjebarp3wa2fmvx8m6es.cloudspaces.litng.ai/detect_video"
        : "https://5000-01jnecfjebarp3wa2fmvx8m6es.cloudspaces.litng.ai/detect_image"

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)

      if (!response.ok) {
        throw new Error("Analysis failed. Please try again.")
      }

      const data = await response.json()
      setAnalysisProgress(100)

      // Short delay to show 100% progress
      setTimeout(() => {
        if (file.type.startsWith("image/")) {
          // Handle image results
          if (!data || (typeof data === "string" && data.includes("No image data received"))) {
            setError("Detection failed: No valid image data received")
            setIsAnalyzing(false)
            return
          }

          if (!data.deepfake) {
            setError("Detection failed: Could not analyze the image")
            setIsAnalyzing(false)
            return
          }
        }
        // For both image and video results
        setResults(data)
        setIsAnalyzing(false)
      }, 500)
    } catch (err) {
      clearInterval(progressInterval)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setFile(null)
    setPreview(null)
    setResults(null)
    setError(null)
    setIsAnalyzing(false)
    setAnalysisProgress(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  // Determine if the image is fake or real based on results
  const getResultStatus = () => {
    if (!results) return null

    // Handle new array format
    if (Array.isArray(results.deepfake)) {
      // Find the object with the highest score
      interface DeepfakeResult {
        label: string
        score: number
      }

      const highestScoreResult: DeepfakeResult = results.deepfake.reduce(
        (prev: DeepfakeResult, current: DeepfakeResult) => (prev.score > current.score ? prev : current),
        { label: "", score: Number.NEGATIVE_INFINITY },
      )

      if (highestScoreResult.label) {
        return highestScoreResult.label.toLowerCase()
      }
      return "unknown"
    }

    // Handle old string format for backwards compatibility
    const deepfakeResult = results.deepfake || ""
    if (deepfakeResult.startsWith("Fake")) {
      return "fake"
    } else if (deepfakeResult.startsWith("Real")) {
      return "real"
    }
    return "unknown"
  }

  // Extract confidence score from results
  const getConfidenceScore = () => {
    if (!results || !results.deepfake) return null

    if (typeof results.deepfake === "string") {
      const match = results.deepfake.match(/\d+\.\d+/)
      if (match) {
        const score = Number.parseFloat(match[0])
        // Convert to percentage between 0-100
        return Math.abs(score) * 100
      }
    } else if (Array.isArray(results.deepfake)) {
      // Handle new array format
      const highestScoreResult = results.deepfake.reduce(
        (prev: any, current: any) => (prev.score > current.score ? prev : current),
        { label: "", score: Number.NEGATIVE_INFINITY },
      )
      return highestScoreResult.score * 100
    }

    return null
  }

  const resultStatus = getResultStatus()
  const confidenceScore = getConfidenceScore()

  const VideoResultsDisplay = () => {
    if (!results || !results.video_analysis) return null

    const { video_analysis, lip_sync_analysis } = results
    const fakeFramesPercentage = (video_analysis.fake_frames_detected / video_analysis.total_frames_analyzed) * 100

    return (
      <div className="flex flex-col">
        <div
          className={cn(
            "p-6 text-center",
            fakeFramesPercentage > 30
              ? "bg-destructive/10"
              : fakeFramesPercentage > 10
                ? "bg-amber-500/10"
                : "bg-green-500/10",
          )}
        >
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 bg-background">
            {fakeFramesPercentage > 30 ? (
              <AlertTriangle className="w-8 h-8 text-destructive" />
            ) : fakeFramesPercentage > 10 ? (
              <Info className="w-8 h-8 text-amber-500" />
            ) : (
              <CheckCircle className="w-8 h-8 text-green-500" />
            )}
          </div>

          <h2 className="text-3xl font-bold mb-2">
            {fakeFramesPercentage > 30
              ? "Deepfake Detected"
              : fakeFramesPercentage > 10
                ? "Possible Manipulation"
                : "Likely Authentic"}
          </h2>

          <p className="text-muted-foreground max-w-2xl mx-auto">
            {fakeFramesPercentage > 30
              ? "Our AI has detected significant signs of manipulation in this video."
              : fakeFramesPercentage > 10
                ? "Our AI has detected some potential signs of manipulation in this video."
                : "Our AI analysis indicates this is likely an authentic video without significant signs of deepfake manipulation."}
          </p>

          <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background">
            <span className="text-sm font-medium">Fake Frames:</span>
            <Badge
              variant={fakeFramesPercentage > 30 ? "destructive" : fakeFramesPercentage > 10 ? "outline" : "default"}
            >
              {video_analysis.fake_frames_detected} / {video_analysis.total_frames_analyzed} (
              {Math.round(fakeFramesPercentage)}%)
            </Badge>
          </div>
        </div>

        <div className="flex flex-col md:flex-row">
          <div className="w-full md:w-1/2 p-6 border-r border-b">
            <div className="aspect-video max-h-[400px] relative rounded-lg overflow-hidden border mb-4">
              {preview && <video src={preview} controls className="w-full h-full object-cover" />}

              {fakeFramesPercentage > 30 && (
                <div className="absolute top-2 right-2">
                  <Badge variant="destructive" className="text-xs px-2 py-1">
                    FAKE
                  </Badge>
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-2 mt-4">
              <Button variant="outline" size="sm" onClick={resetAnalysis} className="gap-2">
                <X className="w-4 h-4" />
                New Analysis
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={generateVideoReport}
                className="gap-2"
                disabled={isGeneratingReport}
              >
                {isGeneratingReport ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                Generate Report
              </Button>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="secondary" size="sm" className="gap-2 ml-auto">
                      <Shield className="w-4 h-4" />
                      Protect Your Media
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Add protection to your own media</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          <div className="w-full md:w-1/2 p-6">
            <Tabs defaultValue="summary">
              <TabsList className="w-full mb-4">
                <TabsTrigger value="summary" className="flex-1">
                  Summary
                </TabsTrigger>
                <TabsTrigger value="frames" className="flex-1">
                  Frame Analysis
                </TabsTrigger>
                <TabsTrigger value="lipsync" className="flex-1">
                  Lip Sync
                </TabsTrigger>
                <TabsTrigger value="technical" className="flex-1">
                  Technical Details
                </TabsTrigger>
              </TabsList>

              <TabsContent value="summary" className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Analysis Summary</h3>
                  <p className="text-sm text-muted-foreground">
                    {fakeFramesPercentage > 30
                      ? "This video shows significant signs of AI manipulation consistent with deepfake technology. Multiple frames were flagged as potentially fake."
                      : fakeFramesPercentage > 10
                        ? "This video shows some signs of potential manipulation. A small number of frames were flagged as suspicious."
                        : "This video appears to be authentic. Our analysis found natural patterns and consistent features throughout the video."}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-muted">
                    <h4 className="text-sm font-medium mb-1">Video Duration</h4>
                    <p className="text-xs text-muted-foreground">{video_analysis.duration}</p>
                  </div>

                  <div className="p-4 rounded-lg bg-muted">
                    <h4 className="text-sm font-medium mb-1">Frames Analyzed</h4>
                    <p className="text-xs text-muted-foreground">{video_analysis.total_frames_analyzed}</p>
                  </div>

                  <div className="p-4 rounded-lg bg-muted">
                    <h4 className="text-sm font-medium mb-1">Fake Frames</h4>
                    <p
                      className={cn(
                        "text-xs",
                        fakeFramesPercentage > 30
                          ? "text-destructive"
                          : fakeFramesPercentage > 10
                            ? "text-amber-500"
                            : "text-green-500",
                      )}
                    >
                      {video_analysis.fake_frames_detected} ({Math.round(fakeFramesPercentage)}%)
                    </p>
                  </div>

                  <div className="p-4 rounded-lg bg-muted">
                    <h4 className="text-sm font-medium mb-1">Lip Sync Analysis</h4>
                    <p className="text-xs text-muted-foreground">
                      {typeof lip_sync_analysis === "object" && lip_sync_analysis.error
                        ? "Analysis failed"
                        : typeof lip_sync_analysis === "object" && lip_sync_analysis.fake_probability !== undefined
                          ? lip_sync_analysis.fake_probability > 0.5
                            ? "Potential mismatch detected"
                            : "No issues detected"
                          : typeof lip_sync_analysis === "number"
                            ? lip_sync_analysis < 0.5
                              ? "Potential mismatch detected"
                              : "No issues detected"
                            : "Analysis unavailable"}
                    </p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="frames" className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Frame-by-Frame Analysis</h3>
                  <p className="text-sm text-muted-foreground">Detailed analysis of individual video frames.</p>
                </div>

                <div className="border rounded-md overflow-hidden">
                  <div className="grid grid-cols-12 bg-muted p-2 text-xs font-medium">
                    <div className="col-span-1">Frame</div>
                    <div className="col-span-3">Timestamp</div>
                    <div className="col-span-2">Result</div>
                    <div className="col-span-3">Score</div>
                    <div className="col-span-3">Status</div>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    {video_analysis.results.map((frame: any, index: any) => (
                      <div
                        key={index}
                        className={cn(
                          "grid grid-cols-12 p-2 text-xs border-t",
                          frame.is_fake ? "bg-destructive/5" : "",
                        )}
                      >
                        <div className="col-span-1">{frame.frame_number}</div>
                        <div className="col-span-3">{frame.timestamp}</div>
                        <div className="col-span-2">{frame.prediction}</div>
                        <div className="col-span-3">{frame.fusion_score ? frame.fusion_score.toFixed(4) : "N/A"}</div>
                        <div className="col-span-3">
                          {frame.is_fake ? (
                            <span className="inline-flex items-center text-destructive">
                              <AlertTriangle className="w-3 h-3 mr-1" /> Suspicious
                            </span>
                          ) : (
                            <span className="inline-flex items-center text-green-500">
                              <CheckCircle className="w-3 h-3 mr-1" /> OK
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="lipsync" className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Lip Synchronization Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Analysis of audio-visual synchronization to detect potential deepfakes.
                  </p>
                </div>

                {typeof lip_sync_analysis === "object" && lip_sync_analysis.error ? (
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm">{lip_sync_analysis.description || lip_sync_analysis.error}</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-muted">
                        <h4 className="text-sm font-medium mb-1">Real Probability</h4>
                        <div className="flex items-center">
                          {typeof lip_sync_analysis === "object" ? (
                            <>
                              <Progress value={lip_sync_analysis.real_probability * 100} className="h-2 flex-1 mr-2" />
                              <span className="text-xs">{Math.round(lip_sync_analysis.real_probability * 100)}%</span>
                            </>
                          ) : (
                            <>
                              <Progress
                                value={typeof lip_sync_analysis === "number" ? lip_sync_analysis * 100 : 0}
                                className="h-2 flex-1 mr-2"
                              />
                              <span className="text-xs">
                                {typeof lip_sync_analysis === "number" ? Math.round(lip_sync_analysis * 100) : 0}%
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="p-4 rounded-lg bg-muted">
                        <h4 className="text-sm font-medium mb-1">Fake Probability</h4>
                        <div className="flex items-center">
                          {typeof lip_sync_analysis === "object" ? (
                            <>
                              <Progress value={lip_sync_analysis.fake_probability * 100} className="h-2 flex-1 mr-2" />
                              <span className="text-xs">{Math.round(lip_sync_analysis.fake_probability * 100)}%</span>
                            </>
                          ) : (
                            <>
                              <Progress
                                value={typeof lip_sync_analysis === "number" ? (1 - lip_sync_analysis) * 100 : 0}
                                className="h-2 flex-1 mr-2"
                              />
                              <span className="text-xs">
                                {typeof lip_sync_analysis === "number" ? Math.round((1 - lip_sync_analysis) * 100) : 0}%
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-muted">
                      <h4 className="text-sm font-medium mb-1">Analysis Result</h4>
                      <p className="text-sm">
                        {typeof lip_sync_analysis === "object"
                          ? lip_sync_analysis.description ||
                            (lip_sync_analysis.fake_probability > 0.5
                              ? "Potential lip sync mismatch detected, suggesting possible manipulation."
                              : "No significant lip sync issues detected.")
                          : typeof lip_sync_analysis === "number"
                            ? lip_sync_analysis < 0.5
                              ? "Potential lip sync mismatch detected, suggesting possible manipulation."
                              : "No significant lip sync issues detected."
                            : "Analysis unavailable"}
                      </p>
                    </div>

                    {typeof lip_sync_analysis === "object" && lip_sync_analysis.processing_time_seconds && (
                      <div className="p-4 rounded-lg bg-muted">
                        <h4 className="text-sm font-medium mb-1">Processing Time</h4>
                        <p className="text-sm">{lip_sync_analysis.processing_time_seconds} seconds</p>
                      </div>
                    )}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="technical" className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Technical Details</h3>
                  <p className="text-sm text-muted-foreground">
                    Technical information about the video and analysis process.
                  </p>
                </div>

                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="item-1">
                    <AccordionTrigger>Video Properties</AccordionTrigger>
                    <AccordionContent>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>File Type:</div>
                        <div>{file?.type || "Unknown"}</div>
                        <div>File Size:</div>
                        <div>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Unknown"}</div>
                        <div>Frame Rate:</div>
                        <div>{video_analysis.fps} FPS</div>
                        <div>Duration:</div>
                        <div>{video_analysis.duration}</div>
                        <div>Frames Analyzed:</div>
                        <div>{video_analysis.total_frames_analyzed}</div>
                        <div>Frame Interval:</div>
                        <div>{video_analysis.frame_interval} second(s)</div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="item-2">
                    <AccordionTrigger>Detection Method</AccordionTrigger>
                    <AccordionContent>
                      <p className="text-sm text-muted-foreground">
                        Our system uses a multi-stage approach to detect deepfakes:
                      </p>
                      <ul className="text-sm text-muted-foreground space-y-2 mt-2">
                        <li>• Frame-by-frame analysis for visual inconsistencies</li>
                        <li>• Lip synchronization analysis to detect audio-visual mismatches</li>
                        <li>• Facial feature consistency checks across frames</li>
                        <li>• Temporal coherence analysis</li>
                      </ul>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="item-3">
                    <AccordionTrigger>Raw Analysis Data</AccordionTrigger>
                    <AccordionContent>
                      <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
                        {JSON.stringify(results, null, 2)}
                      </pre>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    )
  }

  const generateReport = async () => {
    if (!results) return

    setIsGeneratingReport(true)

    try {
      const formData = new FormData()
      if (file) {
        formData.append("file", file)
      }

      formData.append("investigator_name", "AI Detection System")
      formData.append("analysis_results", JSON.stringify(results))

      const response = await fetch("https://5000-01jnecfjebarp3wa2fmvx8m6es.cloudspaces.litng.ai/generate_report", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to generate report")
      }

      // Get the blob from the response
      const blob = await response.blob()

      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob)

      // Create a temporary link element
      const a = document.createElement("a")
      a.href = url
      a.download = `DeepfakeReport_${Date.now()}.pdf`
      document.body.appendChild(a)

      // Click the link to download the file
      a.click()

      // Clean up
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error("Error generating report:", err)
      setError(err instanceof Error ? err.message : "Failed to generate report")
    } finally {
      setIsGeneratingReport(false)
    }
  }

  const generateVideoReport = async () => {
    if (!results || !results.video_analysis) return

    setIsGeneratingReport(true);

    try {
      // Prepare the data for video report generation
      const reportData = {
        analysis_results: results,  // Now contains all the enhanced data
        investigator_name: "AI Detection System",
        case_number: `VIDEO-${Date.now().toString(36).toUpperCase()}`,
      };

      // Send request to backend for video report generation
      const response = await fetch(
        "https://5000-01jnecfjebarp3wa2fmvx8m6es.cloudspaces.litng.ai/generate_video_report",
        {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(reportData),
      });

      if (!response.ok) {
        throw new Error("Failed to generate video report");
      }

      // Get the blob from the response
      const blob = await response.blob();

      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob);

      // Create a temporary link element
      const a = document.createElement("a");
      a.href = url;
      a.download = `VideoDeepfakeReport_${Date.now()}.pdf`;
      document.body.appendChild(a);

      // Click the link to download the file
      a.click();

      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error("Error generating video report:", err);
      setError(err instanceof Error ? err.message : "Failed to generate video report");
    } finally {
      setIsGeneratingReport(false);
    }
  }

  return (
    <main className="min-h-screen flex flex-col bg-gradient-to-b from-background to-background/95">
      <Navbar />

      <div className="flex-1 container mx-auto px-4 py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-5xl mx-auto"
        >
          <div className="flex items-center space-x-4 mb-6">
            <div className="p-3 rounded-full bg-primary/10">
              <Search className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-4xl font-space font-bold">Deepfake Detection</h1>
              <p className="text-muted-foreground">Analyze images for AI manipulation</p>
            </div>
          </div>

          <Card className="mb-8 overflow-hidden border-2">
            <CardContent className="p-0">
              <AnimatePresence mode="wait">
                {!file && !results ? (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <motion.div
                      className={cn(
                        "p-10 transition-all duration-300",
                        isDragging ? "bg-primary/10 border-primary" : "bg-card",
                      )}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                    >
                      <motion.div
                        className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-12 text-center"
                        initial={{ borderColor: "rgba(0,0,0,0.1)" }}
                        animate={{
                          borderColor: isDragging ? "rgba(var(--primary-rgb), 0.5)" : "rgba(0,0,0,0.1)",
                          boxShadow: isDragging ? "0 0 15px rgba(var(--primary-rgb), 0.2)" : "none",
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        <input
                          type="file"
                          className="hidden"
                          id="file-upload"
                          ref={fileInputRef}
                          onChange={handleFileChange}
                          accept="image/*,video/*"
                        />
                        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                          <motion.div
                            className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-4"
                            whileHover={{ scale: 1.05, backgroundColor: "rgba(var(--primary-rgb), 0.15)" }}
                            whileTap={{ scale: 0.95 }}
                          >
                            <Upload className="w-10 h-10 text-primary" />
                          </motion.div>
                          <h3 className="text-xl font-medium mb-2">
                            {isDragging ? "Drop your file here" : "Drag and drop your file here"}
                          </h3>
                          <p className="text-sm text-muted-foreground mb-6 max-w-md">
                            Upload an image or video to analyze it for potential deepfake manipulation. We support JPG,
                            PNG, WEBP for images and MP4, AVI, MOV for videos.
                          </p>
                          <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                            <Button size="lg" className="gap-2">
                              <FileImage className="w-4 h-4" />
                              Select File
                            </Button>
                          </motion.div>
                        </label>
                      </motion.div>
                    </motion.div>

                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md text-destructive flex items-center gap-2"
                      >
                        <AlertTriangle className="w-5 h-5" />
                        <span>{error}</span>
                      </motion.div>
                    )}
                  </motion.div>
                ) : isAnalyzing ? (
                  <motion.div
                    key="analyzing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="p-10"
                  >
                    <div className="flex flex-col md:flex-row gap-8 items-center">
                      <div className="w-full md:w-1/2 aspect-square md:aspect-auto md:max-h-[400px] relative rounded-lg overflow-hidden border">
                        {preview && file?.type.startsWith("video/") ? (
                          <video src={preview} className="w-full h-full object-cover" controls />
                        ) : (
                          preview && (
                            <img
                              src={preview || "/placeholder.svg"}
                              alt="Preview"
                              className="w-full h-full object-cover"
                            />
                          )
                        )}
                        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center p-6">
                          <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
                          <h3 className="text-xl font-medium mb-2">
                          {file?.type.startsWith("video/") ? "Analyzing Video" : "Analyzing Image"}
                          </h3>
                          <p className="text-sm text-muted-foreground mb-4 text-center">
                          Our AI is examining the {file?.type.startsWith("video/") ? "video" : "image"} for signs of manipulation. This may take a few moments.
                          </p>
                          <div className="w-full max-w-md">
                          <Progress value={analysisProgress} className="h-2" />
                          <p className="text-xs text-right mt-1 text-muted-foreground">
                            {Math.round(analysisProgress)}%
                          </p>
                          </div>
                        </div>
                      </div>

                      <div className="w-full md:w-1/2 space-y-6">
                        <div className="space-y-2">
                          <h3 className="text-xl font-medium">Detection Process</h3>
                          <p className="text-sm text-muted-foreground">
                            Our advanced AI model is analyzing your image through multiple detection layers:
                          </p>
                        </div>

                        <div className="space-y-3">
                          <motion.div
                            className="flex items-center gap-3"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 }}
                          >
                            <motion.div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                analysisProgress > 20
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                              animate={{
                                scale: analysisProgress > 20 ? [1, 1.2, 1] : 1,
                                backgroundColor: analysisProgress > 20 ? "var(--primary)" : "hsl(var(--muted))",
                              }}
                              transition={{ duration: 0.3 }}
                            >
                              {analysisProgress > 20 ? "✓" : "1"}
                            </motion.div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  analysisProgress > 20 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Facial Analysis
                              </p>
                            </div>
                          </motion.div>

                          <motion.div
                            className="flex items-center gap-3"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.2 }}
                          >
                            <motion.div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                analysisProgress > 50
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                              animate={{
                                scale: analysisProgress > 50 ? [1, 1.2, 1] : 1,
                                backgroundColor: analysisProgress > 50 ? "var(--primary)" : "hsl(var(--muted))",
                              }}
                              transition={{ duration: 0.3 }}
                            >
                              {analysisProgress > 50 ? "✓" : "2"}
                            </motion.div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  analysisProgress > 50 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Pattern Recognition
                              </p>
                            </div>
                          </motion.div>

                          <motion.div
                            className="flex items-center gap-3"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.3 }}
                          >
                            <motion.div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                analysisProgress > 80
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                              animate={{
                                scale: analysisProgress > 80 ? [1, 1.2, 1] : 1,
                                backgroundColor: analysisProgress > 80 ? "var(--primary)" : "hsl(var(--muted))",
                              }}
                              transition={{ duration: 0.3 }}
                            >
                              {analysisProgress > 80 ? "✓" : "3"}
                            </motion.div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  analysisProgress > 80 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Metadata Verification
                              </p>
                            </div>
                          </motion.div>

                          <motion.div
                            className="flex items-center gap-3"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.4 }}
                          >
                            <motion.div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                analysisProgress >= 100
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                              animate={{
                                scale: analysisProgress >= 100 ? [1, 1.2, 1] : 1,
                                backgroundColor: analysisProgress >= 100 ? "var(--primary)" : "hsl(var(--muted))",
                              }}
                              transition={{ duration: 0.3 }}
                            >
                              {analysisProgress >= 100 ? "✓" : "4"}
                            </motion.div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  analysisProgress >= 100 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Final Assessment
                              </p>
                            </div>
                          </motion.div>
                        </div>

                        <Button variant="outline" className="w-full" onClick={resetAnalysis}>
                          Cancel Analysis
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                ) : results ? (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="p-0"
                  >
                    {!results || (results.video_analysis === undefined && !results.deepfake) ? (
                      <div className="p-10 text-center">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 bg-destructive/10">
                          <AlertTriangle className="w-8 h-8 text-destructive" />
                        </div>

                        <h2 className="text-3xl font-bold mb-2">Detection Failed</h2>

                        <p className="text-muted-foreground max-w-2xl mx-auto mb-6">
                          We couldn't properly analyze this {file?.type.startsWith("video/") ? "video" : "image"}. This
                          could be due to:
                        </p>

                        <ul className="text-left max-w-md mx-auto mb-8 space-y-2 text-sm text-muted-foreground">
                          <li className="flex items-start gap-2">
                            <span className="text-destructive mt-0.5">•</span>
                            <span>
                              No {file?.type.startsWith("video/") ? "analyzable content" : "face detected"} in the{" "}
                              {file?.type.startsWith("video/") ? "video" : "image"}
                            </span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-destructive mt-0.5">•</span>
                            <span>File format not properly supported</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-destructive mt-0.5">•</span>
                            <span>Server processing error</span>
                          </li>
                        </ul>

                        <Button variant="outline" onClick={resetAnalysis} className="gap-2">
                          <X className="w-4 h-4" />
                          Try Another {file?.type.startsWith("video/") ? "Video" : "Image"}
                        </Button>
                      </div>
                    ) : results.video_analysis ? (
                      <VideoResultsDisplay />
                    ) : (
                      // Original image results display
                      <div className="flex flex-col">
                        <div
                          className={cn(
                            "p-6 text-center",
                            resultStatus === "fake"
                              ? "bg-destructive/10"
                              : resultStatus === "real"
                                ? "bg-green-500/10"
                                : "bg-amber-500/10",
                          )}
                        >
                          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 bg-background">
                            {resultStatus === "fake" ? (
                              <AlertTriangle className="w-8 h-8 text-destructive" />
                            ) : resultStatus === "real" ? (
                              <CheckCircle className="w-8 h-8 text-green-500" />
                            ) : (
                              <Info className="w-8 h-8 text-amber-500" />
                            )}
                          </div>

                          <h2 className="text-3xl font-bold mb-2">
                            {resultStatus === "fake"
                              ? "Deepfake Detected"
                              : resultStatus === "real"
                                ? "Authentic Image"
                                : "Analysis Inconclusive"}
                          </h2>

                          <p className="text-muted-foreground max-w-2xl mx-auto">
                            {resultStatus === "fake"
                              ? "Our AI has detected signs of manipulation in this image. It appears to be artificially generated or modified."
                              : resultStatus === "real"
                                ? "Our AI analysis indicates this is likely an authentic image without signs of deepfake manipulation."
                                : "Our AI couldn't determine with confidence whether this image is real or fake."}
                          </p>

                          {confidenceScore !== null && (
                            <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background">
                              <span className="text-sm font-medium">Confidence:</span>
                              <motion.div
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                transition={{ delay: 0.3, type: "spring" }}
                              >
                                <Badge
                                  variant={
                                    resultStatus === "fake"
                                      ? "destructive"
                                      : resultStatus === "real"
                                        ? "default"
                                        : "outline"
                                  }
                                >
                                  {Math.round(confidenceScore)}%
                                </Badge>
                              </motion.div>
                            </div>
                          )}
                        </div>

                        <div className="flex flex-col md:flex-row">
                          <div className="w-full md:w-1/2 p-6 border-r border-b">
                            <div className="space-y-4">
                              <div className="aspect-square max-h-[400px] relative rounded-lg overflow-hidden border">
                                {preview && (
                                  <img
                                    src={preview || "/placeholder.svg"}
                                    alt="Analyzed image"
                                    className="w-full h-full object-cover"
                                  />
                                )}

                                {resultStatus === "fake" && (
                                  <div className="absolute top-2 right-2">
                                    <Badge variant="destructive" className="text-xs px-2 py-1">
                                      FAKE
                                    </Badge>
                                  </div>
                                )}
                              </div>

                              {results?.segmented && (
                                <div className="space-y-4 mt-4">
                                  <h4 className="text-sm font-medium">AI Visualizations</h4>

                                  {results?.segmented?.LIME?.overlay && (
                                    <div>
                                      <h5 className="text-xs font-medium mb-1 text-primary">LIME Analysis</h5>
                                      <div className="aspect-square max-h-[250px] rounded-lg overflow-hidden border">
                                        <img
                                          src={`${results?.segmented?.LIME?.overlay}`}
                                          alt="LIME visualization"
                                          className="w-full h-full object-cover"
                                        />
                                      </div>
                                      <p className="text-xs text-muted-foreground mt-1">
                                        LIME highlights regions that influenced the detection decision
                                      </p>
                                    </div>
                                  )}

                                  {results?.segmented?.["GradCAM++"]?.overlay && (
                                    <div>
                                      <h5 className="text-xs font-medium mb-1 text-primary">GradCAM++ Analysis</h5>
                                      <div className="aspect-square max-h-[250px] rounded-lg overflow-hidden border">
                                        <img
                                          src={`${results?.segmented?.["GradCAM++"]?.overlay}`}
                                          alt="GradCAM++ visualization"
                                          className="w-full h-full object-cover"
                                        />
                                      </div>
                                      <p className="text-xs text-muted-foreground mt-1">
                                        GradCAM++ highlights regions of potential manipulation
                                      </p>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>

                            <div className="flex flex-wrap gap-2 justify-between">
                              <Button variant="outline" size="sm" onClick={resetAnalysis} className="gap-2">
                                <X className="w-4 h-4" />
                                New Analysis
                              </Button>

                              <div className="flex gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={generateReport}
                                  className="gap-2"
                                  disabled={isGeneratingReport}
                                >
                                  {isGeneratingReport ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                  ) : (
                                    <FileText className="w-4 h-4" />
                                  )}
                                  Generate Report
                                </Button>

                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button variant="secondary" size="sm" className="gap-2">
                                        <Shield className="w-4 h-4" />
                                        Protect Your Media
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p>Add protection to your own media</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                            </div>
                          </div>

                          <div className="w-full md:w-1/2 p-6">
                            <Tabs defaultValue="summary">
                              <TabsList className="w-full mb-4">
                                <TabsTrigger value="summary" className="flex-1">
                                  Summary
                                </TabsTrigger>
                                <TabsTrigger value="technical" className="flex-1">
                                  Technical Details
                                </TabsTrigger>
                                <TabsTrigger value="metadata" className="flex-1">
                                  Metadata
                                </TabsTrigger>
                                <TabsTrigger value="report" className="flex-1">
                                  Analysis Report
                                </TabsTrigger>
                              </TabsList>

                              <AnimatePresence mode="wait">
                                {/* Wrap each TabsContent with motion.div */}
                                <motion.div
                                  key="tab-content"
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -10 }}
                                  transition={{ duration: 0.2 }}
                                >
                                  {/* Your TabsContent components here */}
                                  <TabsContent value="summary" className="space-y-4">
                                    <div className="space-y-2">
                                      <h3 className="text-lg font-medium">Analysis Summary</h3>
                                      {!results || !results.deepfake ? (
                                        <div className="p-4 bg-destructive/10 rounded-lg text-destructive flex items-center gap-2">
                                          <AlertTriangle className="w-5 h-5" />
                                          <p className="text-sm">
                                            Detection failed. No valid results were returned from the analysis.
                                          </p>
                                        </div>
                                      ) : (
                                        <p className="text-sm text-muted-foreground">
                                          {resultStatus === "fake"
                                            ? "This image shows signs of AI manipulation consistent with deepfake technology. The analysis detected inconsistencies in facial features, lighting, and texture patterns."
                                            : resultStatus === "real"
                                              ? "This image appears to be authentic. Our analysis found natural patterns and consistent features throughout the image."
                                              : "The analysis was unable to conclusively determine if this image is real or fake. There may be some unusual patterns, but they're not definitive indicators of manipulation."}
                                        </p>
                                      )}
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                      <div className="p-4 rounded-lg bg-muted">
                                        <h4 className="text-sm font-medium mb-1">File Type</h4>
                                        <p className="text-xs text-muted-foreground">{file?.type || "Unknown"}</p>
                                      </div>

                                      <div className="p-4 rounded-lg bg-muted">
                                        <h4 className="text-sm font-medium mb-1">File Size</h4>
                                        <p className="text-xs text-muted-foreground">
                                          {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Unknown"}
                                        </p>
                                      </div>

                                      <div className="p-4 rounded-lg bg-muted">
                                        <h4 className="text-sm font-medium mb-1">Analysis Result</h4>
                                        <p
                                          className={cn(
                                            "text-xs",
                                            !results || !results.deepfake
                                              ? "text-destructive"
                                              : resultStatus === "fake"
                                                ? "text-destructive"
                                                : resultStatus === "real"
                                                  ? "text-green-500"
                                                  : "text-amber-500",
                                          )}
                                        >
                                          {!results || !results.deepfake
                                            ? "Detection Failed"
                                            : resultStatus === "fake"
                                              ? "Deepfake Detected"
                                              : resultStatus === "real"
                                                ? "Authentic Image"
                                                : "Inconclusive"}
                                        </p>
                                      </div>

                                      <div className="p-4 rounded-lg bg-muted">
                                        <h4 className="text-sm font-medium mb-1">Confidence Score</h4>
                                        <p className="text-xs text-muted-foreground">
                                          {confidenceScore !== null ? `${Math.round(confidenceScore)}%` : "N/A"}
                                        </p>
                                      </div>
                                    </div>

                                    {Array.isArray(results?.deepfake) && (
                                      <motion.div
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.2 }}
                                        className="p-4 rounded-lg bg-muted"
                                      >
                                        <h4 className="text-sm font-medium mb-2">Detailed Model Results</h4>
                                        <div className="space-y-2">
                                          {results.deepfake.map((result:any, index:any) => (
                                            <div key={index} className="flex justify-between items-center">
                                              <span className="text-xs">{result.label}</span>
                                              <div className="flex-1 mx-4">
                                                <Progress value={result.score * 100} className="h-1" />
                                              </div>
                                              <span className="text-xs font-medium">
                                                {(result.score * 100).toFixed(1)}%
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      </motion.div>
                                    )}
                                  </TabsContent>

                                  <TabsContent value="technical" className="space-y-4">
                                    <div className="space-y-2">
                                      <h3 className="text-lg font-medium">Technical Analysis</h3>
                                      <p className="text-sm text-muted-foreground">
                                        Detailed technical information about the analysis process and findings.
                                      </p>
                                    </div>

                                    <Accordion type="single" collapsible className="w-full">
                                      <AccordionItem value="item-1">
                                        <AccordionTrigger>Detection Method</AccordionTrigger>
                                        <AccordionContent>
                                          <p className="text-sm text-muted-foreground">
                                            Our system uses a fusion-based approach that combines multiple AI models to
                                            detect inconsistencies in images. The primary detection results were:
                                          </p>
                                          {Array.isArray(results?.deepfake) ? (
                                            <div className="mt-2 space-y-2">
                                              {results.deepfake.map((result :any, index:any) => (
                                                <div
                                                  key={index}
                                                  className="flex justify-between text-sm p-2 bg-muted/50 rounded-md"
                                                >
                                                  <span>{result.label}</span>
                                                  <span className="font-medium">
                                                    {(result.score * 100).toFixed(2)}%
                                                  </span>
                                                </div>
                                              ))}
                                            </div>
                                          ) : (
                                            <p className="text-sm mt-2">{results?.deepfake || "N/A"}</p>
                                          )}
                                        </AccordionContent>
                                      </AccordionItem>

                                      <AccordionItem value="item-2">
                                        <AccordionTrigger>Analysis Factors</AccordionTrigger>
                                        <AccordionContent>
                                          <ul className="text-sm text-muted-foreground space-y-2">
                                            <li>• Facial feature consistency</li>
                                            <li>• Texture and pattern analysis</li>
                                            <li>• Lighting and shadow consistency</li>
                                            <li>• Metadata verification</li>
                                          </ul>
                                        </AccordionContent>
                                      </AccordionItem>

                                      <AccordionItem value="item-3">
                                        <AccordionTrigger>Raw Detection Data</AccordionTrigger>
                                        <AccordionContent>
                                          <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
                                            {JSON.stringify(results, null, 2)}
                                          </pre>
                                        </AccordionContent>
                                      </AccordionItem>
                                    </Accordion>
                                  </TabsContent>

                                  <TabsContent value="metadata" className="space-y-4">
                                    <div className="space-y-2">
                                      <h3 className="text-lg font-medium">Image Metadata</h3>
                                      <p className="text-sm text-muted-foreground">
                                        Metadata can provide clues about an image's authenticity and origin.
                                      </p>
                                    </div>

                                    {results?.manifest && typeof results.manifest === "object" ? (
                                      <div className="bg-muted rounded-lg p-4 max-h-80 overflow-auto text-white">
                                        <ReactJson src={results.manifest} theme="monokai" />
                                      </div>
                                    ) : (
                                      <div className="bg-muted rounded-lg p-4 text-center">
                                        <p className="text-sm text-muted-foreground">No metadata available</p>
                                      </div>
                                    )}
                                  </TabsContent>

                                  <TabsContent value="report" className="space-y-4">
                                    <div className="space-y-2">
                                      <h3 className="text-lg font-medium">AI Analysis Report</h3>
                                      <p className="text-sm text-muted-foreground">
                                        Comprehensive AI analysis of the image with detailed findings.
                                      </p>
                                    </div>

                                    {results?.report ? (
                                      <div className="space-y-4">
                                        {/* Visual Content Analysis */}
                                        <Card className="overflow-hidden">
                                          <CardHeader className="bg-muted/30 py-3">
                                            <CardTitle className="text-base flex items-center gap-2">
                                              <Search className="h-4 w-4" />
                                              Visual Content Analysis
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line">
                                              {results.report["Visual Content Analysis"]}
                                            </div>
                                          </CardContent>
                                        </Card>

                                        {/* Anomaly Detection */}
                                        <Card className="overflow-hidden">
                                          <CardHeader className="bg-muted/30 py-3">
                                            <CardTitle className="text-base flex items-center gap-2">
                                              <AlertTriangle className="h-4 w-4" />
                                              Anomaly Detection
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line">
                                              {results.report["Anomaly Detection"]}
                                            </div>
                                          </CardContent>
                                        </Card>

                                        {/* Text Extraction */}
                                        <Card className="overflow-hidden">
                                          <CardHeader className="bg-muted/30 py-3">
                                            <CardTitle className="text-base flex items-center gap-2">
                                              <FileImage className="h-4 w-4" />
                                              Text Extraction
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line">
                                              {results.report["Text Extraction"]}
                                            </div>
                                          </CardContent>
                                        </Card>

                                        {/* Deep Learning Evaluation */}
                                        <Card className="overflow-hidden">
                                          <CardHeader className="bg-muted/30 py-3">
                                            <CardTitle className="text-base flex items-center gap-2">
                                              <Gauge className="h-4 w-4" />
                                              Deep Learning Evaluation
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line">
                                              {results.report["Deep Learning Model Evaluation"]}
                                            </div>
                                          </CardContent>
                                        </Card>

                                        {/* Additional Context */}
                                        <Card className="overflow-hidden">
                                          <CardHeader className="bg-muted/30 py-3">
                                            <CardTitle className="text-base flex items-center gap-2">
                                              <Info className="h-4 w-4" />
                                              Additional Context
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line">
                                              {results.report["Additional Context"]}
                                            </div>
                                          </CardContent>
                                        </Card>

                                        {/* Final Verdict */}
                                        <Card
                                          className={cn(
                                            resultStatus === "fake"
                                              ? "bg-destructive/10 border-destructive/30"
                                              : resultStatus === "real"
                                                ? "bg-green-500/10 border-green-500/30"
                                                : "bg-amber-500/10 border-amber-500/30",
                                            "overflow-hidden",
                                          )}
                                        >
                                          <CardHeader
                                            className={cn(
                                              resultStatus === "fake"
                                                ? "bg-destructive/20"
                                                : resultStatus === "real"
                                                  ? "bg-green-500/20"
                                                  : "bg-amber-500/20",
                                              "py-3",
                                            )}
                                          >
                                            <CardTitle className="text-base flex items-center gap-2">
                                              {resultStatus === "fake" ? (
                                                <AlertTriangle className="h-4 w-4 text-destructive" />
                                              ) : resultStatus === "real" ? (
                                                <CheckCircle className="h-4 w-4 text-green-500" />
                                              ) : (
                                                <Info className="h-4 w-4 text-amber-500" />
                                              )}
                                              Final Summary and Verdict
                                            </CardTitle>
                                          </CardHeader>
                                          <CardContent className="p-4">
                                            <div className="text-sm whitespace-pre-line font-medium">
                                              {results.report["Final Summary and Verdict"]}
                                            </div>
                                          </CardContent>
                                        </Card>
                                      </div>
                                    ) : (
                                      <div className="bg-muted rounded-lg p-4 text-center">
                                        <p className="text-sm text-muted-foreground">No analysis report available</p>
                                      </div>
                                    )}
                                  </TabsContent>
                                </motion.div>
                              </AnimatePresence>
                            </Tabs>
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </CardContent>

            {file && !results && !isAnalyzing && (
              <CardFooter className="p-6 border-t bg-muted/50">
                <div className="flex flex-col sm:flex-row w-full gap-4 items-center">
                  <div className="flex-1 flex items-center gap-4">
                    <div className="w-16 h-16 rounded-md overflow-hidden border bg-muted">
                      {preview && (
                        <img src={preview || "/placeholder.svg"} alt="Preview" className="w-full h-full object-cover" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium truncate max-w-[200px] sm:max-w-xs">{file?.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : ""}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2 w-full sm:w-auto">
                    <Button variant="outline" className="flex-1 sm:flex-initial" onClick={resetAnalysis}>
                      Cancel
                    </Button>
                    <Button className="flex-1 sm:flex-initial gap-2" onClick={analyzeMedia}>
                      <Search className="w-4 h-4" />
                      Analyze {file?.type.startsWith("video/") ? "Video" : "Image"}
                    </Button>
                  </div>
                </div>

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md text-destructive flex items-center gap-2"
                  >
                    <AlertTriangle className="w-5 h-5" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </CardFooter>
            )}
          </Card>

          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, staggerChildren: 0.1 }}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-primary" />
                    How It Works
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Our deepfake detection system uses advanced AI to analyze images for signs of manipulation. The
                    system examines facial features, lighting, and texture patterns to identify inconsistencies.
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileImage className="w-5 h-5 text-primary" />
                    Supported Formats
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">JPG</Badge>
                    <Badge variant="outline">JPEG</Badge>
                    <Badge variant="outline">PNG</Badge>
                    <Badge variant="outline">WEBP</Badge>
                    <Badge variant="outline">MP4</Badge>
                    <Badge variant="outline">MOV</Badge>
                    <Badge variant="outline">AVI</Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Info className="w-5 h-5 text-primary" />
                    Privacy
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Your uploads are processed securely and not stored permanently. We respect your privacy and do not
                    share your data with third parties.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>

      <Footer />
    </main>
  )
}

