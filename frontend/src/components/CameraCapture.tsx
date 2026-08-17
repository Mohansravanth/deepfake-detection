import React, { useRef, useState, useCallback } from 'react';
import { Button, Alert, Spinner } from 'react-bootstrap';

interface CameraCaptureProps {
  onCapture: (videoBlob: Blob) => void;
  isAnalyzing?: boolean;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, isAnalyzing = false }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        mediaStreamRef.current = stream;
        setHasPermission(true);
        setError(null);
      }
    } catch (err) {
      setError('Unable to access camera. Please check permissions.');
      console.error(err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setHasPermission(false);
  }, []);

  const captureVideo = useCallback(() => {
    if (!videoRef.current || !videoRef.current.srcObject) return;

    const stream = videoRef.current.srcObject as MediaStream;
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks: BlobPart[] = [];

    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      onCapture(blob);
    };

    mediaRecorder.start();
    setTimeout(() => {
      mediaRecorder.stop();
    }, 3000); // record 3 seconds
  }, [onCapture]);

  return (
    <div className="text-center">
      {!hasPermission ? (
        <Button variant="primary" onClick={startCamera} className="mb-3">
          📷 Start Camera
        </Button>
      ) : (
        <div>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: '100%', maxHeight: '400px', borderRadius: '12px' }}
          />
          <div className="mt-3">
            <Button variant="success" onClick={captureVideo} disabled={isAnalyzing} className="me-2">
              🎥 Capture 3s Video
            </Button>
            <Button variant="secondary" onClick={stopCamera}>
              Stop Camera
            </Button>
          </div>
        </div>
      )}
      {isAnalyzing && <Spinner animation="border" variant="primary" className="mt-3" />}
      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
    </div>
  );
};

export default CameraCapture;