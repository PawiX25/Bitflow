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
- **Animated Effects** - Add dynamic animations to your dithered images:
  - **Glitch** - Random pixel distortion with RGB shift
  - **Scanline** - CRT monitor-style moving scan lines
  - **Pulse** - Subtle breathing/zoom effect
  - **Wave** - Wavy horizontal distortion
  - **Matrix** - Binary code rain overlay (Matrix-style)
  - **VHS** - Retro VHS tape artifacts and chromatic aberration
- Adjustable animation intensity
- Before/After split view comparison
- Pan and zoom controls
- Export processed images

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
5. **Choose an ANIMATION effect** to add dynamic visual effects (optional)
6. Adjust the **INTENSITY slider** to control animation strength
7. Toggle "Show Before/After" to compare original and processed images
8. Use mouse wheel to zoom and drag to pan
9. Click EXPORT to save the processed image

## Technologies

- React with TypeScript
- Tailwind CSS for styling
- HTML5 Canvas for image processing