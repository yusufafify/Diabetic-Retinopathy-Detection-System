import axios from "axios";

export const analyzeRetinaImage = async (formData: FormData) => {
  const response = await axios.post(
    `http://localhost:5000/api/analyze`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );
  return response.data;
};
