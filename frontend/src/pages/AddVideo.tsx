import React, { useState, useRef } from 'react';
import api from '../services/api';
import { Container, Row, Col, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';
import './AddVideo.css'
const AddVideo: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<{ is_fake: boolean; confidence: number } | null>(null);
  const [needsConfirmation, setNeedsConfirmation] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setMessage(null);
      setError(null);
      setDetectionResult(null);
      setNeedsConfirmation(false);
    } else {
      setError('Please upload a valid video file (MP4, AVI, MOV)');
      event.target.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a video');
      return;
    }
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('confirmed', 'false');

    setUploading(true);
    setMessage(null);
    setError(null);
    setNeedsConfirmation(false);

    try {
      const response = await api.post('/add_video_auto', formData);
      if (response.data.needs_confirmation) {
        setNeedsConfirmation(true);
        setDetectionResult({
          is_fake: response.data.is_fake,
          confidence: response.data.confidence
        });
      } else {
        setMessage(response.data.message);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setUploading(false);
    }
  };

  const handleConfirm = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('confirmed', 'true');

    setUploading(true);
    setMessage(null);
    setError(null);

    try {
      const response = await api.post('/add_video_auto', formData);
      setMessage(response.data.message);
      setSelectedFile(null);
      setDetectionResult(null);
      setNeedsConfirmation(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setError(err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDropZoneClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Container className="py-5">
      <Row className="justify-content-center">
        <Col lg={8}>
          <Card className="shadow-lg border-0 rounded-4">
            <div className="bg-gradient-primary p-4 text-white text-center">
              <h1 className="display-5 fw-bold">📤 Add Video to Dataset</h1>
              <p className="lead mb-0">Upload a video – our AI will detect if it's real or fake, then you confirm.</p>
            </div>
            <Card.Body className="p-4">
              <Form>
                <div
                  className="dropzone p-5 text-center rounded-4 border-2 border-dashed border-secondary"
                  style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                  onClick={handleDropZoneClick}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const file = e.dataTransfer.files[0];
                    if (file && file.type.startsWith('video/')) {
                      setSelectedFile(file);
                      setMessage(null);
                      setError(null);
                      setDetectionResult(null);
                      setNeedsConfirmation(false);
                    } else {
                      setError('Please upload a valid video file (MP4, AVI, MOV)');
                    }
                  }}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="video/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                  <i className="bi bi-cloud-upload fs-1"></i>
                  <p className="mt-2 mb-0">Drag & drop a video, or click to select</p>
                  <small className="text-muted">MP4, AVI, MOV</small>
                </div>
                {selectedFile && (
                  <div className="mt-3">
                    <p><strong>Selected:</strong> {selectedFile.name}</p>
                  </div>
                )}
                
                {!needsConfirmation && !message && (
                  <Button
                    variant="primary"
                    onClick={handleAnalyze}
                    className="mt-3 w-100"
                    disabled={uploading || !selectedFile}
                  >
                    {uploading ? <Spinner animation="border" size="sm" /> : '🔍 Analyze & Detect'}
                  </Button>
                )}

                {needsConfirmation && detectionResult && (
                  <div className="mt-3 p-3 rounded-3 bg-light">
                    <Alert variant={detectionResult.is_fake ? 'danger' : 'success'}>
                      <h5>
                        {detectionResult.is_fake 
                          ? '🚨 AI detects this as FAKE' 
                          : '✅ AI detects this as REAL'}
                      </h5>
                      <p>Confidence: {(detectionResult.confidence * 100).toFixed(2)}%</p>
                      <p className="mb-0">Do you want to save this video as <strong>{detectionResult.is_fake ? 'FAKE' : 'REAL'}</strong>?</p>
                    </Alert>
                    <div className="d-flex gap-2">
                      <Button variant="success" onClick={handleConfirm} disabled={uploading} className="flex-1">
                        {uploading ? <Spinner animation="border" size="sm" /> : '✅ Confirm & Save'}
                      </Button>
                      <Button variant="secondary" onClick={() => {
                        setNeedsConfirmation(false);
                        setDetectionResult(null);
                        setSelectedFile(null);
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }} className="flex-1">
                        ❌ Cancel
                      </Button>
                    </div>
                  </div>
                )}

                {message && <Alert variant="success" className="mt-3">{message}</Alert>}
                {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
              </Form>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default AddVideo;
