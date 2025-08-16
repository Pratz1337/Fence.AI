"use client"

import type React from "react"

import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { Shield, Upload, FileImage, AlertTriangle, CheckCircle, X, Loader2, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export default function ProtectPage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processProgress, setProcessProgress] = useState(0)
  const [protectedImage, setProtectedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

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
    // Check if file is an image, audio, or video
    if (
      !selectedFile.type.startsWith("image/") &&
      !selectedFile.type.startsWith("audio/") &&
      !selectedFile.type.startsWith("video/")
    ) {
      setError("Please upload an image, audio, or video file")
      return
    }

    setFile(selectedFile)
    setError(null)

    // Create preview for images
    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreview(e.target?.result as string)
      }
      reader.readAsDataURL(selectedFile)
    } else {
      // For audio/video, just set a placeholder
      setPreview(null)
    }
  }

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = (error) => reject(error)
    })
  }

  const protectFile = async () => {
    if (!file) return

    setIsProcessing(true)
    setProcessProgress(0)
    setProtectedImage(null)
    setError(null)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProcessProgress((prev) => {
        if (prev >= 95) {
          clearInterval(progressInterval)
          return prev
        }
        return prev + Math.random() * 15
      })
    }, 300)

    try {
      if (file.type.startsWith("image/")) {
        const base64 = await convertToBase64(file)
        const payload = JSON.stringify({
          prompt: base64.split("base64,")[1],
        })

        // Simulate API call
        setTimeout(async () => {
          try {
            // In a real implementation, this would be an actual API call
            // const response = await fetch("https://api.cortex.cerebrium.ai/v4/p-5df54339/image-protection/predict", {
            //   method: "POST",
            //   headers: {
            //     "Authorization": "Bearer your-token-here",
            //     "Content-Type": "application/json",
            //   },
            //   body: payload
            // })
            // const data = await response.json()
            // const imageData = data?.result

            // For demo purposes, we'll just use the original image
            const imageData = base64.split("base64,")[1]
            setProtectedImage(`data:image/png;base64,${imageData}`)
            clearInterval(progressInterval)
            setProcessProgress(100)

            // Short delay to show 100% progress
            setTimeout(() => {
              setIsProcessing(false)
            }, 500)
          } catch (err) {
            throw err
          }
        }, 2000)
      } else if (file.type.startsWith("audio/")) {
        // Simulate audio protection
        setTimeout(() => {
          clearInterval(progressInterval)
          setProcessProgress(100)

          // Short delay to show 100% progress
          setTimeout(() => {
            setIsProcessing(false)
            // In a real implementation, this would download the protected audio
          }, 500)
        }, 2000)
      } else if (file.type.startsWith("video/")) {
        // Simulate video protection
        setTimeout(() => {
          clearInterval(progressInterval)
          setProcessProgress(100)

          // Short delay to show 100% progress
          setTimeout(() => {
            setIsProcessing(false)
            // In a real implementation, this would download the protected video
          }, 500)
        }, 3000)
      }
    } catch (err) {
      clearInterval(progressInterval)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
      setIsProcessing(false)
    }
  }

  const downloadProtectedFile = () => {
    if (!protectedImage) return

    const link = document.createElement("a")
    link.href = protectedImage
    link.download = `protected_${file?.name || "image.png"}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const resetProcess = () => {
    setFile(null)
    setPreview(null)
    setProtectedImage(null)
    setError(null)
    setIsProcessing(false)
    setProcessProgress(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
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
              <Shield className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-4xl font-space font-bold">Deepfake Protection</h1>
              <p className="text-muted-foreground">Protect your media from AI manipulation</p>
            </div>
          </div>

          <Card className="mb-8 overflow-hidden border-2">
            <CardContent className="p-0">
              <AnimatePresence mode="wait">
                {!file && !protectedImage ? (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div
                      className={cn(
                        "p-10 transition-all duration-300",
                        isDragging ? "bg-primary/10 border-primary" : "bg-card",
                      )}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-12 text-center">
                        <input
                          type="file"
                          className="hidden"
                          id="file-upload"
                          ref={fileInputRef}
                          onChange={handleFileChange}
                          accept="image/*,audio/*,video/*"
                        />
                        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                          <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                            <Upload className="w-10 h-10 text-primary" />
                          </div>
                          <h3 className="text-xl font-medium mb-2">
                            {isDragging ? "Drop your file here" : "Drag and drop your file here"}
                          </h3>
                          <p className="text-sm text-muted-foreground mb-6 max-w-md">
                            Upload your media to protect it against deepfake manipulation. We support images, audio, and
                            video files.
                          </p>
                          <Button size="lg" className="gap-2">
                            <FileImage className="w-4 h-4" />
                            Select File
                          </Button>
                        </label>
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
                    </div>
                  </motion.div>
                ) : isProcessing ? (
                  <motion.div
                    key="processing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="p-10"
                  >
                    <div className="flex flex-col md:flex-row gap-8 items-center">
                      <div className="w-full md:w-1/2 aspect-square md:aspect-auto md:max-h-[400px] relative rounded-lg overflow-hidden border">
                        {preview ? (
                          <img
                            src={preview || "/placeholder.svg"}
                            alt="Preview"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-muted flex items-center justify-center">
                            <FileImage className="w-16 h-16 text-muted-foreground" />
                          </div>
                        )}
                        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center p-6">
                          <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
                          <h3 className="text-xl font-medium mb-2">Processing Media</h3>
                          <p className="text-sm text-muted-foreground mb-4 text-center">
                            Our AI is adding protection to your{" "}
                            {file?.type.startsWith("image/")
                              ? "image"
                              : file?.type.startsWith("audio/")
                                ? "audio"
                                : "video"}
                            . This may take a few moments.
                          </p>
                          <div className="w-full max-w-md">
                            <Progress value={processProgress} className="h-2" />
                            <p className="text-xs text-right mt-1 text-muted-foreground">
                              {Math.round(processProgress)}%
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="w-full md:w-1/2 space-y-6">
                        <div className="space-y-2">
                          <h3 className="text-xl font-medium">Protection Process</h3>
                          <p className="text-sm text-muted-foreground">
                            Our advanced AI model is adding protection to your media through multiple layers:
                          </p>
                        </div>

                        <div className="space-y-3">
                          <div className="flex items-center gap-3">
                            <div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                processProgress > 20
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                            >
                              {processProgress > 20 ? "✓" : "1"}
                            </div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  processProgress > 20 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Media Analysis
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-3">
                            <div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                processProgress > 50
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                            >
                              {processProgress > 50 ? "✓" : "2"}
                            </div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  processProgress > 50 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Randomizing Embeddings
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-3">
                            <div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                processProgress > 80
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                            >
                              {processProgress > 80 ? "✓" : "3"}
                            </div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  processProgress > 80 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Encryption
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-3">
                            <div
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs",
                                processProgress >= 100
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                            >
                              {processProgress >= 100 ? "✓" : "4"}
                            </div>
                            <div className="flex-1">
                              <p
                                className={cn(
                                  "text-sm font-medium",
                                  processProgress >= 100 ? "text-primary" : "text-muted-foreground",
                                )}
                              >
                                Final Processing
                              </p>
                            </div>
                          </div>
                        </div>

                        <Button variant="outline" className="w-full" onClick={resetProcess}>
                          Cancel Process
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                ) : protectedImage ? (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="p-0"
                  >
                    <div className="p-6 text-center bg-green-500/10">
                      <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 bg-background">
                        <CheckCircle className="w-8 h-8 text-green-500" />
                      </div>

                      <h2 className="text-3xl font-bold mb-2">Protection Complete</h2>

                      <p className="text-muted-foreground max-w-2xl mx-auto">
                        Your media has been successfully protected against deepfake manipulation. You can now download
                        the protected file.
                      </p>

                      <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background">
                        <span className="text-sm font-medium">Status:</span>
                        <Badge variant="default">Protected</Badge>
                      </div>
                    </div>

                    <div className="flex flex-col md:flex-row">
                      <div className="w-full md:w-1/2 p-6 border-r border-b">
                        <div className="aspect-square max-h-[400px] relative rounded-lg overflow-hidden border mb-4">
                          {protectedImage ? (
                            <img
                              src={protectedImage || "/placeholder.svg"}
                              alt="Protected media"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full bg-muted flex items-center justify-center">
                              <FileImage className="w-16 h-16 text-muted-foreground" />
                            </div>
                          )}

                          <div className="absolute top-2 right-2">
                            <Badge variant="default" className="text-xs px-2 py-1">
                              PROTECTED
                            </Badge>
                          </div>
                        </div>

                        <div className="flex justify-between">
                          <Button variant="outline" size="sm" onClick={resetProcess} className="gap-2">
                            <X className="w-4 h-4" />
                            New Protection
                          </Button>

                          <Button variant="default" size="sm" onClick={downloadProtectedFile} className="gap-2">
                            <Download className="w-4 h-4" />
                            Download Protected File
                          </Button>
                        </div>
                      </div>

                      <div className="w-full md:w-1/2 p-6">
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <h3 className="text-lg font-medium">Protection Summary</h3>
                            <p className="text-sm text-muted-foreground">
                              Your media has been protected with our advanced AI technology. This protection helps
                              identify if your content is used to create deepfakes or manipulated in any way.
                            </p>
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
                              <h4 className="text-sm font-medium mb-1">Protection Status</h4>
                              <p className="text-xs text-green-500">Protected</p>
                            </div>

                            <div className="p-4 rounded-lg bg-muted">
                              <h4 className="text-sm font-medium mb-1">Protection Level</h4>
                              <p className="text-xs text-muted-foreground">High</p>
                            </div>
                          </div>

                          <div className="p-4 rounded-lg bg-muted">
                            <h4 className="text-sm font-medium mb-2">What happens next?</h4>
                            <ul className="text-xs text-muted-foreground space-y-1">
                              <li>• Download your protected media file</li>
                              <li>• Use it as you normally would</li>
                              <li>• If someone tries to use it for deepfakes, our protection will help identify it</li>
                              <li>• The protection is invisible to the human eye but detectable by our systems</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </CardContent>

            {file && !protectedImage && !isProcessing && (
              <CardFooter className="p-6 border-t bg-muted/50">
                <div className="flex flex-col sm:flex-row w-full gap-4 items-center">
                  <div className="flex-1 flex items-center gap-4">
                    <div className="w-16 h-16 rounded-md overflow-hidden border bg-muted">
                      {preview ? (
                        <img src={preview || "/placeholder.svg"} alt="Preview" className="w-full h-full object-cover" />
                      ) : (
                        <FileImage className="w-full h-full p-2 text-muted-foreground" />
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
                    <Button variant="outline" className="flex-1 sm:flex-initial" onClick={resetProcess}>
                      Cancel
                    </Button>
                    <Button className="flex-1 sm:flex-initial gap-2" onClick={protectFile}>
                      <Shield className="w-4 h-4" />
                      Protect Media
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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium flex items-center gap-2 mb-2">
                  <Shield className="w-5 h-5 text-primary" />
                  How It Works
                </h3>
                <p className="text-sm text-muted-foreground">
                  Our protection system uses advanced AI to embed invisible watermarks in your media. These watermarks
                  help identify if your content is used to create deepfakes.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium flex items-center gap-2 mb-2">
                  <FileImage className="w-5 h-5 text-primary" />
                  Supported Formats
                </h3>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Images: JPG, PNG, WebP</p>
                  <p className="text-sm text-muted-foreground">Audio: MP3, WAV</p>
                  <p className="text-sm text-muted-foreground">Video: MP4, WebM</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-primary" />
                  File Requirements
                </h3>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Maximum file size: 100MB</p>
                  <p className="text-sm text-muted-foreground">Maximum video length: 5 minutes</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      </div>

      <Footer />
    </main>
  )
}

