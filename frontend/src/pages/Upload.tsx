import React, { useState, useEffect, useRef } from 'react';
import api from "../services/api";
import { Container, Row, Col, Card, Button, Alert, Spinner, ProgressBar, Form } from 'react-bootstrap';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import './Upload.css'
interface PredictionResult {
  success: boolean;
  is_fake: boolean;
  label: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  file_name?: string;
  file_type?: string;
  processing_time?: number;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  test_samples: number;
}

const Upload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<'video' | 'image' | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [showFrameDetails, setShowFrameDetails] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.get('/metrics')
      .then(res => setMetrics(res.data.metrics))
      .catch(err => console.error('Could not load metrics:', err));
}, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setFileType('video');
      } else if (file.type.startsWith('image/')) {
        setFileType('image');
      } else {
        setError('Please upload a video or image file');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append(fileType === 'image' ? 'image' : 'video', selectedFile);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const endpoint = "/predict";
      const response = await api.post(endpoint, formData, {
    headers: {
        "Content-Type": "multipart/form-data",
    },
});
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
          <Card className="shadow-lg border-0 rounded-4 overflow-hidden mb-4">
            <div className="bg-gradient-primary p-4 text-white text-center">
              <h1 className="display-5 fw-bold">Upload & Analyze</h1>
              <p className="lead mb-0">Upload a video or image to check its authenticity</p>
            </div>
            <Card.Body className="p-4">
              <div
                className="dropzone"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  if (file) {
                    if (file.type.startsWith('video/')) setFileType('video');
                    else if (file.type.startsWith('image/')) setFileType('image');
                    else { setError('Please upload a video or image file'); return; }
                    setSelectedFile(file);
                    setPreviewUrl(URL.createObjectURL(file));
                    setResult(null);
                    setError(null);
                  }
                }}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="video/*,image/*"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <i className="bi bi-cloud-upload fs-1"></i>
                <p className="mt-2 mb-0">Drag & drop a video or image, or click to select</p>
                <small className="text-muted">MP4, AVI, MOV, JPG, PNG, WEBP</small>
              </div>

              {previewUrl && (
                <div className="mt-4">
                  {fileType === 'video' ? (
                    <video className="w-100 rounded-3 shadow-sm" controls src={previewUrl} style={{ maxHeight: '400px' }} />
                  ) : (
                    <img src={previewUrl} alt="Preview" className="w-100 rounded-3 shadow-sm" style={{ maxHeight: '400px', objectFit: 'contain' }} />
                  )}
                </div>
              )}

              {selectedFile && !loading && !result && (
                <div className="text-center mt-4">
                  <Button variant="primary" size="lg" onClick={handleUpload} className="px-5 rounded-pill">
                    Analyze {fileType === 'image' ? 'Image' : 'Video'}
                  </Button>
                </div>
              )}

              {loading && (
                <div className="text-center mt-4">
                  <Spinner animation="border" variant="primary" />
                  <p className="mt-2">Analyzing... this may take a moment.</p>
                  <ProgressBar animated now={100} className="w-50 mx-auto" />
                </div>
              )}

              {error && <Alert variant="danger" className="mt-4">{error}</Alert>}

              {result && (
                <div className="mt-4 p-4 rounded-4 bg-light text-center">
                  <div className="d-flex justify-content-center mb-3">
                    <div style={{ width: 150, height: 150 }}>
                      <CircularProgressbar
                        value={confidencePercent}
                        text={`${Math.round(confidencePercent)}%`}
                        styles={buildStyles({
                          textSize: '20px',
                          pathColor: confidenceColor,
                          textColor: confidenceColor,
                          trailColor: '#e9ecef'
                        })}
                      />
                    </div>
                  </div>
                  <h2 className={result.is_fake ? 'text-danger' : 'text-success'}>
                    {result.is_fake ? '🚨 DEEPFAKE DETECTED' : '✅ MEDIA APPEARS REAL'}
                  </h2>
                  <p className="lead">Confidence: {confidencePercent.toFixed(2)}%</p>
                  <p className="text-muted">Raw score: {result.raw_score.toFixed(4)} (0=Real, 1=Fake)</p>
                  {fileType === 'video' && result.frame_scores && (
                    <Button variant="outline-secondary" onClick={() => setShowFrameDetails(!showFrameDetails)} className="mt-2">
                      {showFrameDetails ? 'Hide Frame Details' : 'Show Frame Details'}
                    </Button>
                  )}
                  <Button variant="outline-secondary" onClick={() => { setSelectedFile(null); setPreviewUrl(null); setResult(null); setFileType(null); }} className="mt-2 ms-2">
                    Test Another
                  </Button>
                </div>
              )}

              {showFrameDetails && result && result.frame_scores && (
                <div className="mt-4">
                  <h5>Frame‑by‑Frame Analysis</h5>
                  <p className="text-muted small">Green = Real, Red = Fake. The model analyzes each frame independently.</p>
                  <div className="d-flex flex-wrap" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {result.frame_scores.map((score, idx) => (
                      <div
                        key={idx}
                        className="m-1 p-1 text-center"
                        style={{
                          width: '30px',
                          height: '30px',
                          borderRadius: '4px',
                          background: `rgba(255, 0, 0, ${score})`
                        }}
                        title={`Frame ${idx + 1}: ${Math.round(score * 100)}% fake`}
                      >
                        <small style={{ color: score > 0.5 ? 'white' : 'black', fontSize: '8px' }}>
                          {idx + 1}
                        </small>
                      </div>
                    ))}
                  </div>
                  {result.suspicious_frames && result.suspicious_frames.length > 0 && (
                    <div className="mt-3">
                      <Alert variant="warning">
                        <i className="bi bi-exclamation-triangle me-2"></i>
                        <strong>Suspicious frames:</strong> {result.suspicious_frames.map(f => `Frame ${f + 1}`).join(', ')}
                        <br />
                        <small>These frames show high manipulation probability (70%).</small>
                      </Alert>
                    </div>
                  )}
                  {result.audio_features && result.audio_features.has_audio && (
                    <div className="mt-3">
                      <Alert variant={result.audio_features.is_suspicious ? 'warning' : 'info'}>
                        <i className="bi bi-music-note me-2"></i>
                        <strong>Audio Analysis:</strong>{' '}
                        {result.audio_features.is_suspicious ? 'Unusual audio patterns detected.' : 'Audio appears normal.'}
                        <br />
                        <small>RMS: {result.audio_features.rms?.toFixed(4)} | Spectral Centroid: {result.audio_features.spectral_centroid?.toFixed(0)}</small>
                      </Alert>
                    </div>
                  )}
                </div>
              )}
            </Card.Body>
          </Card>

          {metrics && (
            <Card className="shadow-sm border-0 rounded-4">
              <Card.Body>
                <h4 className="mb-3">Model Performance (on test set)</h4>
                <Row>
                  <Col md={3} sm={6} className="mb-3">
                    <div className="metric-card text-center">
                      <h6 className="text-muted">Accuracy</h6>
                      <h3 className="text-primary">{(metrics.accuracy * 100).toFixed(1)}%</h3>
                    </div>
                  </Col>
                  <Col md={3} sm={6} className="mb-3">
                    <div className="metric-card text-center">
                      <h6 className="text-muted">Precision</h6>
                      <h3 className="text-success">{(metrics.precision * 100).toFixed(1)}%</h3>
                    </div>
                  </Col>
                  <Col md={3} sm={6} className="mb-3">
                    <div className="metric-card text-center">
                      <h6 className="text-muted">Recall</h6>
                      <h3 className="text-info">{(metrics.recall * 100).toFixed(1)}%</h3>
                    </div>
                  </Col>
                  <Col md={3} sm={6} className="mb-3">
                    <div className="metric-card text-center">
                      <h6 className="text-muted">F1 Score</h6>
                      <h3 className="text-warning">{(metrics.f1_score * 100).toFixed(1)}%</h3>
                    </div>
                  </Col>
                </Row>
                <p className="text-muted text-center mt-2">Evaluated on {metrics.test_samples} test videos</p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default Upload;
