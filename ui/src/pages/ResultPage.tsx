import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  AlertTriangle,
  Download,
  Share2,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

interface AnalysisResult {
  grade: number; // or `grade_label` as string depending on how you want to use it
  grade_label: string; // Add the grade label
  classification_confidence: number;
  segmentation_class: number;
  timestamp: string; // Assuming the timestamp is returned as a string
  recommendedAction: string; // If the backend provides a recommendation
//   features?: Feature[]; // Optional, because not every result may have features
}

// interface Feature {
//   name: string;
//   present: boolean;
//   severity: string;
// }

const ResultsPage = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Retrieve the scan result from localStorage
    const storedResult = localStorage.getItem("scanResult");

    if (!storedResult) {
      setError("No analysis data found. Please perform a scan first.");
      setIsLoading(false);
      return;
    }

    try {
      const parsedData: AnalysisResult = JSON.parse(storedResult);
      setResult(parsedData);
    } catch (err) {
      console.error("Error parsing data:", err);
      setError("Failed to load analysis results. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle Download Report
  const handleDownloadReport = () => {
    if (!result) return;

    // Create a simple text report
    const reportContent = `
  RetinaScan MD Analysis Report
  -----------------------------
  Date: ${new Date(result.timestamp).toLocaleString()}
  
  RESULT: ${result.grade_label || result.grade}  // Use grade_label or grade
  Confidence: ${result.classification_confidence * 100}%
  
 
  
  Recommended Action:
  ${result.recommendedAction || "N/A"}
      `.trim();

    // Create a blob and download link
    const blob = new Blob([reportContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `retina-scan-report-${
      new Date().toISOString().split("T")[0]
    }.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Only calculate these if result exists
  const grade = result?.grade || 0; // Default to 0 if no grade
  const gradeLabel = result?.grade_label || "No Diabetic Retinopathy";
  const confidenceValue = result ? result.classification_confidence * 100 : 0;

  // Determine the color based on the grade
  const getGradeColor = (grade: number) => {
    switch (grade) {
      case 0:
        return {
          bg: "bg-green-50",
          text: "text-green-700",
          border: "border-green-300",
        };
      case 1:
        return {
          bg: "bg-yellow-50",
          text: "text-yellow-700",
          border: "border-yellow-300",
        };
      case 2:
        return {
          bg: "bg-orange-50",
          text: "text-orange-700",
          border: "border-orange-300",
        };
      case 3:
        return {
          bg: "bg-red-50",
          text: "text-red-700",
          border: "border-red-300",
        };
      case 4:
        return {
          bg: "bg-red-50",
          text: "text-red-800",
          border: "border-red-300",
        };
      default:
        return {
          bg: "bg-gray-50",
          text: "text-gray-700",
          border: "border-gray-300",
        };
    }
  };

  const { bg, text, border } = getGradeColor(grade);

  return (
    <div className="w-full py-10 px-15">
      <div className="mb-8">
        <Link to="/scan" onClick={() => localStorage.clear()}>
          <Button variant="ghost" className="pl-0">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Scan
          </Button>
        </Link>
      </div>

      <div className="flex flex-col space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-teal-800">
            Analysis Results
          </h1>
          <p className="mt-2 text-gray-600">
            Review the AI-powered analysis of your retina scan.
          </p>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-700"></div>
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : result ? (
          <>
            <Card className={`border-2 ${border}`}>
              <CardHeader className={`${bg}`}>
                <CardTitle className={`flex items-center text-xl ${text}`}>
                  {gradeLabel}
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-6">
                  <div>
                    <h2
                      className={`text-2xl font-semibold ${
                        grade === 0
                          ? "text-green-700"
                          : grade === 1
                          ? "text-yellow-700"
                          : grade === 2
                          ? "text-orange-700"
                          : grade >= 3
                          ? "text-red-700"
                          : "text-gray-700"
                      }`}
                    >
                      Diabetes Class: {grade}
                    </h2>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">
                        Confidence Score
                      </span>
                      <span className="text-sm font-medium">
                        {confidenceValue.toFixed(0)}%
                      </span>
                    </div>
                    <Progress value={confidenceValue} className={`h-2 ${bg}`} />
                  </div>

                  {/* <div>
                    <h3 className="text-lg font-medium mb-3">
                      Detected Features
                    </h3>
                    <div className="grid gap-3 sm:grid-cols-2">
                      {result.features && result.features.length > 0 ? (
                        result.features.map((feature, index) => (
                          <div
                            key={index}
                            className={`p-3 rounded-md border ${
                              feature.present
                                ? "bg-amber-50 border-amber-200"
                                : "bg-gray-50 border-gray-200"
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium">
                                {feature.name}
                              </span>
                              {feature.present ? (
                                <span className="text-amber-600 text-sm font-medium">
                                  {feature.severity}
                                </span>
                              ) : (
                                <XCircle className="h-4 w-4 text-gray-400" />
                              )}
                            </div>
                          </div>
                        ))
                      ) : (
                        <p>No features detected.</p>
                      )}
                    </div>
                  </div> */}

                  <div className={`p-4 rounded-md ${bg} border ${border}`}>
                    <h3 className="text-lg font-medium mb-2">
                      Recommended Action
                    </h3>
                    <p className={text}>
                      {result.recommendedAction || "No recommendation provided"}
                    </p>
                  </div>

                  <div className="text-sm text-gray-500">
                    Analysis completed on{" "}
                    {new Date(result.timestamp).toLocaleString()}
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <Button
                      onClick={handleDownloadReport}
                      variant="outline"
                      className="flex items-center"
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Download Report
                    </Button>
                    <Button variant="outline" className="flex items-center">
                      <Share2 className="mr-2 h-4 w-4" />
                      Share with Specialist
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
              <h2 className="text-lg font-medium mb-3">Important Notice</h2>
              <p className="text-gray-600 mb-4">
                This analysis is provided as a clinical decision support tool
                only. Final diagnosis should always be made by a qualified
                healthcare professional based on a comprehensive clinical
                assessment.
              </p>
              <p className="text-gray-600">
                If diabetic retinopathy is detected, prompt referral to an
                ophthalmologist is recommended for further evaluation and
                management.
              </p>
            </div>
          </>
        ) : (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>No Data</AlertTitle>
            <AlertDescription>
              No analysis data available. Please perform a scan first.
            </AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
