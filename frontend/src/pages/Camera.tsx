import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import CameraCapture from '../components/CameraCapture';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface PredictionResult {
  is_fake: boolean;
  confidence: number;
  raw_score: number;
}

const Camera: React.FC = () => {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleCapture = async (videoBlob: Blob) => {
    const formData = new FormData();
    formData.append('video', videoBlob, 'camera-capture.webm');
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post<PredictionResult>('http://localhost:5000/predict', formData);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const confidencePercent = result ? result.confidence * 100 : 0;
  const confidenceColor = result ? (result.is_fake ? '#dc3545' : '#28a745') : '#6c757d';

  return (
    <Container className="py-5">
      <Row className="justify-content-center">
        <Col lg={8}>
          <Card className="shadow-lg border-0 rounded-4">
            <div className="bg-gradient-primary p-4 text-white text-center">
              <h2 className="fw-bold">📸 Live Camera Analysis</h2>
              <p>Allow camera access, record a short video (3s), and check for deepfakes instantly.</p>
            </div>
            <Card.Body className="p-4">
              <CameraCapture onCapture={handleCapture} isAnalyzing={loading} />
              {result && (
                <div className="mt-4 p-4 rounded-4 bg-light text-center">
                  <div style={{ width: 150, height: 150, margin: '0 auto' }}>
                    <CircularProgressbar value={confidencePercent} text={`${Math.round(confidencePercent)}%`} styles={buildStyles({ textSize: '20px', pathColor: confidenceColor, textColor: confidenceColor, trailColor: '#e9ecef' })} />
                  </div>
                  <h2 className={result.is_fake ? 'text-danger' : 'text-success'}>{result.is_fake ? '🚨 DEEPFAKE DETECTED' : '✅ VIDEO APPEARS REAL'}</h2>
                  <p className="lead">Confidence: {confidencePercent.toFixed(2)}%</p>
                </div>
              )}
              {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Camera;