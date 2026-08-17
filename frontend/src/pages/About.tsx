import React, { useState } from 'react';
import { Container, Row, Col, Card, Image } from 'react-bootstrap';
import './About.css'
const About: React.FC = () => {
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({});

  const developers = [
    {
      id: 'mohan',
      name: "Mohan Sravanth Varma",
      role: "Full Stack Developer & AI Engineer",
      email: "mohansravanthvarma123@gmail.com",
      photo: "/mohan.jpeg",
      description: "Designed and built the entire deepfake detection system, including the CNN+LSTM model, Flask backend API, and React frontend with all features.",
      github: "https://github.com/mohansravanth",
      linkedin: "https://linkedin.com/in/mohansravanth"
    },
    {
      id: 'rahul',
      name: "Rahul Sai",
      role: "Backend & AI Engineer",
      email: "rahulsai9866@gmail.com",
      photo: "/rahul.jpeg",
      description: "Assisted with model optimization, dataset preprocessing, and integration of the image and video analysis pipelines.",
      github: "https://github.com/rahulsai",
      linkedin: "https://linkedin.com/in/rahulsai"
    },
    {
      id: 'vedanth',
      name: "Vedanth Reddy",
      role: "Frontend & UI/UX Designer",
      email: "",
      photo: "/vedanth.jpeg",
      description: "Contributed to UI design, frontend components, and user experience improvements across the platform.",
      github: "https://github.com/vedanthreddy",
      linkedin: "https://linkedin.com/in/vedanthreddy"
    }
  ];

  const getAvatarUrl = (name: string) => {
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&size=150&background=4f46e5&color=fff&bold=true`;
  };

  const handleImageError = (id: string) => {
    setImageErrors(prev => ({ ...prev, [id]: true }));
  };

  const getImageSrc = (dev: typeof developers[0]) => {
    if (imageErrors[dev.id]) {
      return getAvatarUrl(dev.name);
    }
    return dev.photo;
  };

  return (
    <Container className="py-5">
      <div className="text-center mb-5">
        <span className="badge bg-primary bg-gradient px-4 py-2 mb-3">MEET THE TEAM</span>
        <h1 className="display-4 fw-bold">About <span className="text-primary">Us</span></h1>
        <p className="text-muted fs-5">The passionate minds behind the Deepfake Detector</p>
      </div>

      <Row className="g-4 justify-content-center">
        {developers.map((dev) => (
          <Col md={4} key={dev.id}>
            <Card className="h-100 shadow-sm border-0 text-center card-hover">
              <Card.Body>
                <Image
                  src={getImageSrc(dev)}
                  width="160"
                  height="160"
                  className="rounded-3 mb-3 border border-2"
                  style={{ objectFit: 'cover' }}
                  alt={dev.name}
                />
                <h5 className="fw-bold">{dev.name}</h5>
                <p className="text-primary fw-semibold">{dev.role}</p>
                {dev.email && (
                  <p className="small">
                    <a href={`mailto:${dev.email}`} className="text-decoration-none">
                      <i className="bi bi-envelope me-1"></i>{dev.email}
                    </a>
                  </p>
                )}
                <div className="d-flex justify-content-center gap-2 mb-2">
                  {dev.github && (
                    <a href={dev.github} target="_blank" rel="noopener noreferrer" className="btn btn-outline-secondary btn-sm">
                      <i className="bi bi-github"></i>
                    </a>
                  )}
                  {dev.linkedin && (
                    <a href={dev.linkedin} target="_blank" rel="noopener noreferrer" className="btn btn-outline-primary btn-sm">
                      <i className="bi bi-linkedin"></i>
                    </a>
                  )}
                </div>
                <p className="text-muted small">{dev.description}</p>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      <Row className="mt-5">
        <Col>
          <Card className="bg-primary bg-gradient text-white border-0 p-4 text-center">
            <h4 className="fw-bold">Project Overview</h4>
            <p className="mb-3">
              This project was developed as a minor project at <strong>MIT Manipal</strong>.
              The system uses a <strong>CNN+LSTM architecture</strong> trained on the SDFVD 2.0 dataset
              to detect deepfake videos with high accuracy.
            </p>
            <div className="d-flex flex-wrap justify-content-center gap-2">
              <span className="badge bg-light text-dark">TensorFlow</span>
              <span className="badge bg-light text-dark">Flask</span>
              <span className="badge bg-light text-dark">React</span>
              <span className="badge bg-light text-dark">TypeScript</span>
              <span className="badge bg-light text-dark">Bootstrap</span>
            </div>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default About;