import axios from 'axios';

// Resolution order:
//   1. window.API_URL  - runtime override, edit public/config.js after deploy
//   2. REACT_APP_API_URL - build-time env var (Vercel dashboard, .env files)
//   3. http://localhost:5000 - local Flask dev server
const API_URL = (
  (typeof window !== 'undefined' && window.API_URL) ||
  process.env.REACT_APP_API_URL ||
  'http://localhost:5000'
).replace(/\/+$/, '');

const api = axios.create({
  baseURL: API_URL,
});

export default api;
