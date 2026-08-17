import React, { useState } from 'react';
import { Container, Row, Col, Card, Button, Modal } from 'react-bootstrap';
import './Blogs.css'
const blogsData = [
  {
    id: 1,
    title: "What Are Deepfakes and Why Do They Matter?",
    date: "April 15, 2025",
    summary: "Deepfakes are AI-generated synthetic media that can swap faces, manipulate speech, and create realistic but fake videos. Learn how they work and why detecting them is crucial.",
    icon: "bi-film",
    readTime: "5 min read",
    fullContent: "Deepfakes are AI-generated synthetic media that can swap faces, manipulate speech, and create realistic but fake videos. They use generative adversarial networks (GANs) to produce highly convincing content. The rise of deepfakes poses significant threats to misinformation, identity theft, and personal privacy. Detection techniques rely on analyzing unnatural blinking, inconsistent lighting, and artifacts around face boundaries. Understanding these technologies is the first step in combating their misuse."
  },
  {
    id: 2,
    title: "How Our Deepfake Detection Model Works",
    date: "April 10, 2025",
    summary: "We use a CNN+LSTM architecture trained on thousands of real and fake videos. The CNN extracts spatial features and the LSTM detects temporal inconsistencies.",
    icon: "bi-cpu",
    readTime: "7 min read",
    fullContent: "Our model combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) for temporal analysis. The CNN processes individual frames, capturing details like eyes, mouth, and skin texture. The LSTM then analyzes how these features change across frames, identifying unnatural movements that reveal deepfake manipulation. This dual approach allows the model to achieve high accuracy on diverse datasets."
  },
  {
    id: 3,
    title: "The Ethics of Deepfake Technology",
    date: "April 5, 2025",
    summary: "While deepfakes can be creative, they also pose risks for misinformation, identity theft, and fraud. We discuss ethical guidelines and how to stay safe.",
    icon: "bi-shield-check",
    readTime: "6 min read",
    fullContent: "Deepfakes raise critical ethical questions: who is responsible for content authenticity? How can we balance creative expression with protection against harm? Some guidelines include mandatory watermarking of AI-generated content, education on media literacy, and regulatory frameworks to penalize malicious use. Staying vigilant and critical of digital content is essential."
  },
  {
    id: 4,
    title: "Real‑World Deepfake Incidents",
    date: "March 28, 2025",
    summary: "From celebrity impersonations to political disinformation, deepfakes have already caused harm. We highlight major cases and what we can learn from them.",
    icon: "bi-globe",
    readTime: "8 min read",
    fullContent: "Deepfakes have been used to create non‑consensual pornography, impersonate politicians, and spread fake news. Notable incidents include a deepfake video of a European politician that sparked diplomatic tensions, and fabricated speeches used to manipulate stock markets. These cases underline the urgency of robust detection methods and public awareness."
  },
  {
    id: 5,
    title: "Future of AI Media Forensics",
    date: "March 20, 2025",
    summary: "As deepfakes get better, detection methods must evolve. Explore emerging techniques like frequency domain analysis, GAN fingerprints, and blockchain verification.",
    icon: "bi-rocket",
    readTime: "10 min read",
    fullContent: "The future of deepfake detection lies in multi‑modal analysis, combining visual, audio, and metadata cues. Frequency domain analysis can spot unnatural high‑frequency patterns, while GAN fingerprints reveal unique artifacts left by generative models. Blockchain‑based provenance tracking offers a way to verify content origin. This arms race in AI forensics continues."
  }
];

const Blogs: React.FC = () => {
  const [showModal, setShowModal] = useState(false);
  const [selectedBlog, setSelectedBlog] = useState<any>(null);

  const handleReadMore = (blog: any) => {
    setSelectedBlog(blog);
    setShowModal(true);
  };

  return (
    <Container className="py-5">
      <Row className="text-center mb-5">
        <Col>
          <h1 className="display-4 fw-bold">Deepfake Insights</h1>
          <p className="lead text-muted">Stay informed about the latest in deepfake technology and detection.</p>
        </Col>
      </Row>
      <Row className="g-4">
        {blogsData.map((blog) => (
          <Col md={6} lg={4} key={blog.id}>
            <Card className="h-100 shadow-sm border-0 card-hover">
              <div className="blog-img-placeholder">
                <i className={`bi ${blog.icon}`} style={{ fontSize: '3rem' }}></i>
              </div>
              <Card.Body>
                <Card.Title className="fw-bold">{blog.title}</Card.Title>
                <div className="text-muted small mb-2">{blog.date} • {blog.readTime}</div>
                <Card.Text>{blog.summary}</Card.Text>
                <Button
                  variant="outline-primary"
                  size="sm"
                  className="rounded-pill"
                  onClick={() => handleReadMore(blog)}
                >
                  Read More <i className="bi bi-arrow-right ms-1"></i>
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
        <Modal.Header closeButton>
          <Modal.Title>{selectedBlog?.title}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="text-muted small mb-3">{selectedBlog?.date} • {selectedBlog?.readTime}</div>
          <p>{selectedBlog?.fullContent}</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)}>Close</Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default Blogs;