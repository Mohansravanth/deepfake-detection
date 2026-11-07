import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="footer mt-auto">
      <div className="container text-center">
        <p className="mb-1">&copy; 2025 Deepfake Detector | AI for Truth & Authenticity</p>
        <small>
          Built with <i className="bi bi-heart-fill text-danger"></i> at MIT Manipal
        </small>
      </div>
    </footer>
  );
};

export default Footer;