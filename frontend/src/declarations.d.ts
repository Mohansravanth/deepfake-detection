declare module '*.css' {
  const content: { [className: string]: string };
  export default content;
}

interface Window {
  API_URL?: string;
}