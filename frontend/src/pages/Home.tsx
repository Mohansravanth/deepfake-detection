import React from 'react';
import { Container, Row, Col, Card, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import './Home.css'
const Home: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    { id: 1, title: "Upload & Analyze", desc: "Upload videos or images to detect deepfakes with confidence scores.", icon: "bi-upload", path: "/upload" },
    { id: 2, title: "Frame Analysis", desc: "See which frames are manipulated with frame‑by‑frame heatmaps.", icon: "bi-grid-3x3-gap", path: "/upload" },
    { id: 3, title: "Add to Dataset", desc: "Upload verified videos to expand the dataset (auto‑detected).", icon: "bi-plus-circle", path: "/add-video" },
  ];

  const blogPosts = [
    { id: 1, title: "The Rise of Deepfakes", excerpt: "Deepfakes are becoming more realistic. Learn how they work and why they matter.", date: "May 10, 2025" },
    { id: 2, title: "How to Spot a Deepfake", excerpt: "Tips and tricks to identify AI-generated videos.", date: "May 5, 2025" },
    { id: 3, title: "The Future of AI Forensics", excerpt: "How researchers are fighting back against synthetic media.", date: "April 28, 2025" },
  ];

  return (
    <Container className="py-5">
      <Row className="justify-content-center text-center mb-5">
        <Col lg={8}>
          <span className="badge bg-primary bg-gradient px-4 py-2 mb-3">AI FOR AUTHENTICITY</span>
          <h1 className="display-4 fw-bold">Welcome to <span className="text-primary">Deepfake Detector</span></h1>
          <p className="lead text-muted">Empowering you to distinguish real media from AI‑generated fakes using state‑of‑the‑art deep learning.</p>
          <Button variant="primary" size="lg" className="mt-3 px-5 rounded-pill" onClick={() => navigate('/upload')}>
            Get Started <i className="bi bi-arrow-right ms-2"></i>
          </Button>
        </Col>
      </Row>

      <Row className="mt-5 g-4">
        {features.map(feature => (
          <Col md={4} key={feature.id}>
            <Card className="h-100 shadow-sm border-0 card-hover" onClick={() => navigate(feature.path)} style={{ cursor: 'pointer' }}>
              <Card.Body className="text-center">
                <div className="display-1 text-primary"><i className={`bi ${feature.icon}`}></i></div>
                <Card.Title className="mt-3">{feature.title}</Card.Title>
                <Card.Text className="text-muted">{feature.desc}</Card.Text>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      <Row className="mt-5">
        <Col>
          <h2 className="text-center mb-4">Latest Blog Posts</h2>
        </Col>
      </Row>
      <Row className="g-4">
        {blogPosts.map(post => (
          <Col md={4} key={post.id}>
            <Card className="h-100 shadow-sm border-0 card-hover">
              <Card.Body>
                <Card.Title className="fw-bold">{post.title}</Card.Title>
                <div className="text-muted small mb-2">{post.date}</div>
                <Card.Text>{post.excerpt}</Card.Text>
                <Button variant="outline-primary" size="sm" className="rounded-pill" onClick={() => navigate('/blogs')}>
                  Read More <i className="bi bi-arrow-right ms-1"></i>
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      <Row className="mt-5 text-center">
        <Col>
          <Card className="bg-gradient-primary text-white border-0 p-4">
            <h3>Ready to test a video?</h3>
            <p>Upload a video or image to see our AI in action.</p>
            <Button variant="light" className="mt-2 rounded-pill w-50 mx-auto" onClick={() => navigate('/upload')}>
              Go to Upload <i className="bi bi-arrow-right ms-2"></i>
            </Button>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Home;