import { ArrowRight, FileImage, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Link } from "react-router-dom";
import retina from "@/assets/image.png";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-white">
        <div className="flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0 px-10">
          <div className="flex gap-2 items-center text-xl font-bold text-teal-700">
            <Shield className="h-6 w-6" />
            <span>RetinaScan MD</span>
          </div>
          <div className="flex flex-1 items-center justify-end space-x-4">
            <nav className="flex items-center space-x-2">
              <Link to="/">
                <Button variant="ghost">Home</Button>
              </Link>
              <Link to="/about">
                <Button variant="ghost">About</Button>
              </Link>
              <Link to="/scan">
                <Button>
                  Start Scan
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </nav>
          </div>
        </div>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-white to-teal-50">
          <div className=" px-4 md:px-10">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-teal-800">
                    Clinical Decision Support for Diabetes Detection
                  </h1>
                  <p className="max-w-[600px] text-gray-600 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                    Upload retina images to detect early signs of diabetes using
                    our advanced AI-powered analysis system.
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link to="/scan">
                    <Button size="lg" className="px-8">
                      Start Scanning
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                  <Link to="/about">
                    <Button size="lg" variant="outline" className="px-8">
                      Learn More
                    </Button>
                  </Link>
                </div>
              </div>
              <div className="mx-auto lg:mr-0 flex items-center justify-center">
                <div className="rounded-lg overflow-hidden shadow-xl">
                  <img
                    alt="Retina scan visualization"
                    className="aspect-video object-cover w-full max-w-[600px]"
                    src={retina}
                  />
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-white">
          <div className=" px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-teal-800">
                  How It Works
                </h2>
                <p className="max-w-[900px] text-gray-600 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Our system uses advanced Convolutional Neural Networks to
                  analyze retina images and detect signs of diabetes.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3 lg:gap-12 mt-12">
              <Card className="border-teal-100 shadow-sm">
                <CardContent className="p-6 flex flex-col items-center text-center space-y-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-teal-100 text-teal-700">
                    <FileImage className="h-8 w-8" />
                  </div>
                  <h3 className="text-xl font-bold text-teal-800">Upload</h3>
                  <p className="text-gray-600">
                    Upload high-quality retina images through our secure
                    interface.
                  </p>
                </CardContent>
              </Card>
              <Card className="border-teal-100 shadow-sm">
                <CardContent className="p-6 flex flex-col items-center text-center space-y-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-teal-100 text-teal-700">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-8 w-8"
                    >
                      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                      <path d="m9 12 2 2 4-4" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-teal-800">Analyze</h3>
                  <p className="text-gray-600">
                    Our AI model processes the image to identify diabetic
                    markers.
                  </p>
                </CardContent>
              </Card>
              <Card className="border-teal-100 shadow-sm">
                <CardContent className="p-6 flex flex-col items-center text-center space-y-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-teal-100 text-teal-700">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-8 w-8"
                    >
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-teal-800">Results</h3>
                  <p className="text-gray-600">
                    Receive clear, actionable results to support clinical
                    decisions.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </main>
      <footer className="w-full border-t bg-teal-50 py-6">
        <div className=" flex flex-col items-center justify-center gap-4 text-center md:flex-row md:gap-8 md:text-left">
          <p className="text-sm text-gray-500">
            © 2025 RetinaScan MD. All rights reserved.
          </p>
          <div className="flex gap-4">
            <Link to="/terms" className="text-sm text-gray-500 hover:underline">
              Terms
            </Link>
            <Link
              to="/privacy"
              className="text-sm text-gray-500 hover:underline"
            >
              Privacy
            </Link>
            <Link
              to="/contact"
              className="text-sm text-gray-500 hover:underline"
            >
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
