import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, AlertCircle, ArrowLeft, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Link } from "react-router-dom";
import { analyzeRetinaImage } from "@/actions/scan";

const ScanPage = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    localStorage.clear();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    // Check if file is an image
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }

    setFile(selectedFile);

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an image to upload");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Convert file to FormData
      const formData = new FormData();
      formData.append("image", file);

      // Call server action to process the image
      const result = await analyzeRetinaImage(formData);

      // Store the result in localStorage
      localStorage.setItem("scanResult", JSON.stringify(result));

      // Navigate to the results page
      navigate(`/scan/results`);
    } catch (err) {
      setError(
        "An error occurred while processing the image. Please try again."
      );
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="w-full p-10">
      <div className="mb-8">
        <Link to="/">
          <Button variant="ghost" className="pl-0">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </Link>
      </div>

      <div className="flex flex-col space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-teal-800">
            Retina Scan Analysis
          </h1>
          <p className="mt-2 text-gray-600">
            Upload a high-quality retina image for diabetes detection analysis.
          </p>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Card className="border-teal-100">
          <CardContent className="p-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid gap-6 sm:grid-cols-2">
                <div className="flex flex-col space-y-4">
                  <div className="flex flex-col space-y-2">
                    <label
                      htmlFor="image-upload"
                      className="text-sm font-medium text-gray-700"
                    >
                      Upload Retina Image
                    </label>
                    <div className="flex items-center justify-center w-full">
                      <label
                        htmlFor="image-upload"
                        className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 border-gray-300"
                      >
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <Upload className="w-10 h-10 mb-3 text-gray-400" />
                          <p className="mb-2 text-sm text-gray-500">
                            <span className="font-semibold">
                              Click to upload
                            </span>{" "}
                            or drag and drop
                          </p>
                          <p className="text-xs text-gray-500">
                            PNG, JPG or JPEG (max. 10MB)
                          </p>
                        </div>
                        <input
                          id="image-upload"
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={handleFileChange}
                          disabled={isProcessing}
                        />
                      </label>
                    </div>
                  </div>
                  <Button
                    type="submit"
                    className="w-full"
                    disabled={!file || isProcessing}
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      "Analyze Image"
                    )}
                  </Button>
                </div>
                <div className="flex flex-col space-y-2">
                  <span className="text-sm font-medium text-gray-700">
                    Image Preview
                  </span>
                  <div className="flex items-center justify-center w-full h-64 bg-gray-100 rounded-lg border border-gray-200">
                    {preview ? (
                      <img
                        src={preview || "/placeholder.svg"}
                        alt="Retina scan preview"
                        className="h-full w-full object-contain rounded-lg"
                      />
                    ) : (
                      <p className="text-sm text-gray-500">No image selected</p>
                    )}
                  </div>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>

        <div className="bg-teal-50 p-6 rounded-lg">
          <h2 className="text-lg font-medium text-teal-800 mb-2">
            Guidelines for Best Results
          </h2>
          <ul className="list-disc pl-5 space-y-1 text-gray-600">
            <li>Upload high-resolution images for better accuracy</li>
            <li>
              Ensure the retina is clearly visible and centered in the image
            </li>
            <li>Images should be well-lit with minimal glare</li>
            <li>Supported formats: JPG, JPEG, PNG</li>
            <li>Maximum file size: 10MB</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ScanPage;
