# Bitflow

A React-based image dithering application that converts images to bitmap format using various dithering algorithms.

## Features

- Upload PNG, JPG, or WEBP images
- Adjust image size and threshold
- Multiple dithering methods:
  - Standard thresholding
  - Floyd-Steinberg
  - Atkinson
  - Jarvis-Judice-Ninke
  - Stucki
  - Bayer matrix (2x2, 4x4, 8x8)
  - Clustered 4x4
  - Random dithering

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/PawiX25/Bitflow
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view the application.

## Usage

1. Upload an image using the UPLOAD button
2. Adjust the SIZE slider to change the output dimensions
3. Modify the THRESHOLD value to control the dithering intensity
4. Select a DITHERING METHOD from the dropdown
5. View the processed bitmap image in the main panel

## Technologies

- React with TypeScript
- Tailwind CSS for styling
- HTML5 Canvas for image processing