import React, { useState } from "react";
import axios from "axios";

function FileUpload() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("image/")) {
      setFile(selectedFile);

      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
      setError(null);
    } else {
      setError("Please select a valid image file (e.g., JPG, PNG).");
      setFile(null);
      setImagePreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data && response.data.predicted_class) {
        setPrediction(response.data);
        setError(null);
      } else {
        setError("Unexpected response from the server.");
      }
    } catch (error) {
      setError("Error during file upload or analysis.");
      console.error(error);
    }
  };

  return (
    <div
      style={{
        backgroundColor: "#e9f5f8",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "'Roboto', sans-serif",
        padding: "20px",
        color: "#0a4b56",
      }}
    >
      <div
        style={{
          backgroundColor: "#e0f2f1",
          borderRadius: "12px",
          boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
          width: "100%",
          maxWidth: "450px",
          padding: "30px",
          textAlign: "center",
          border: "1px solid #b2dfdb",
        }}
      >
        <h1 style={{ fontSize: "1.8rem", fontWeight: "700", color: "#004d40" }}>
          Diabetic Retinopathy Screening
        </h1>
        <form
          onSubmit={handleSubmit}
          style={{
            marginTop: "20px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <label
            htmlFor="file-upload"
            style={{
              marginBottom: "10px",
              fontSize: "1rem",
              fontWeight: "500",
              color: "#00695c",
            }}
          >
            Choose Image for Upload
          </label>
          <input
            id="file-upload"
            type="file"
            onChange={handleFileChange}
            style={{
              padding: "10px",
              border: "1px solid #4db6ac",
              borderRadius: "6px",
              width: "90%",
              fontSize: "0.9rem",
              marginBottom: "15px",
            }}
          />
          <button
            type="submit"
            style={{
              padding: "12px 20px",
              fontSize: "1rem",
              backgroundColor: "#00897b",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              transition: "background-color 0.3s ease",
            }}
            onMouseOver={(e) => (e.target.style.backgroundColor = "#00695c")}
            onMouseOut={(e) => (e.target.style.backgroundColor = "#00897b")}
          >
            Submit
          </button>
        </form>

        {error && (
          <div style={{ color: "#d32f2f", marginTop: "15px" }}>{error}</div>
        )}

        {imagePreview && (
          <div style={{ marginTop: "20px" }}>
            <h3 style={{ fontSize: "1.2rem", fontWeight: "600" }}>
              Image Preview
            </h3>
            <img
              src={imagePreview}
              alt="Uploaded"
              style={{
                maxWidth: "100%",
                maxHeight: "200px",
                borderRadius: "8px",
                objectFit: "cover",
                marginTop: "10px",
                border: "1px solid #a7ffeb",
              }}
            />
          </div>
        )}

        {prediction && (
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              borderRadius: "8px",
              border: "1px solid #004d40",
              backgroundColor: "#e0f7fa",
            }}
          >
            <h3
              style={{
                fontSize: "1.2rem",
                fontWeight: "700",
                color: "#00796b",
              }}
            >
              Analysis Results
            </h3>
            <p>
              <strong>Diagnosis Class:</strong> {prediction.predicted_class}
            </p>
            <p>
              <strong>Confidence Level:</strong>{" "}
              {(
                prediction.confidence_scores[prediction.predicted_class] * 100
              ).toFixed(2)}
              %
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUpload;
