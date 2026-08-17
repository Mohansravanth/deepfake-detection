import React from 'react';
import { Link } from 'react-router-dom';
import { Navbar as BootstrapNavbar, Nav, Container } from 'react-bootstrap';
import './Navbar.css'
const Navbar: React.FC = () => {
  return (
    <BootstrapNavbar expand="lg" className="navbar-custom" variant="dark">
      <Container>
        <BootstrapNavbar.Brand as={Link} to="/">
          <i className="bi bi-shield-check me-2"></i>Deepfake Detector
        </BootstrapNavbar.Brand>
        <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BootstrapNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link as={Link} to="/"><i className="bi bi-house me-1"></i>Home</Nav.Link>
            <Nav.Link as={Link} to="/upload"><i className="bi bi-upload me-1"></i>Upload</Nav.Link>
            <Nav.Link as={Link} to="/blogs"><i className="bi bi-newspaper me-1"></i>Blogs</Nav.Link>
            <Nav.Link as={Link} to="/add-video"><i className="bi bi-plus-circle me-1"></i>Add Video</Nav.Link>
            <Nav.Link as={Link} to="/about"><i className="bi bi-people me-1"></i>About</Nav.Link>
          </Nav>
        </BootstrapNavbar.Collapse>
      </Container>
    </BootstrapNavbar>
  );
};

export default Navbar;