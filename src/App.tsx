
import React, { useState, useRef, useEffect } from 'react';

const clamp = (v: number, min = 0, max = 255) => Math.max(min, Math.min(max, v));

const generateBayer = (n: number): number[][] => {
  if (n === 2) return [
    [0, 2],
    [3, 1],
  ];
  const half = n / 2;
  const prev = generateBayer(half);
  const res: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let y = 0; y < half; y++) {
    for (let x = 0; x < half; x++) {
      const v = prev[y][x];
      res[y][x] = 4 * v;
      res[y][x + half] = 4 * v + 2;
      res[y + half][x] = 4 * v + 3;
      res[y + half][x + half] = 4 * v + 1;
    }
  }
  return res;
};

const orderedDither = (imageData: ImageData, w: number, h: number, matrix: number[][]) => {
  const d = imageData.data;
  const mH = matrix.length;
  const mW = matrix[0].length;
  const n2 = mH * mW;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const originalAlpha = d[i + 3];
      const threshold = ((matrix[y % mH][x % mW] + 0.5) * 255) / n2;
      const v = d[i] > threshold ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = originalAlpha;
    }
  }
  return imageData;
};

type DiffusionKernel = { x: number; y: number; w: number }[];
const diffuse = (imageData: ImageData, w: number, h: number, kernel: DiffusionKernel, divisor: number) => {
  const d = imageData.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const oldP = d[i];
      const originalAlpha = d[i + 3];
      const newP = oldP > 128 ? 255 : 0;
      const err = oldP - newP;
      d[i] = d[i + 1] = d[i + 2] = newP;
      d[i + 3] = originalAlpha;
      for (const k of kernel) {
        const nx = x + k.x;
        const ny = y + k.y;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          const ni = (ny * w + nx) * 4;
          const nv = d[ni] + (err * k.w) / divisor;
          d[ni] = d[ni + 1] = d[ni + 2] = clamp(nv);
        }
      }
    }
  }
  return imageData;
};

const Dithering = {
  grayscale: (imageData: ImageData) => {
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i];
      const g = d[i + 1];
      const b = d[i + 2];
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      d[i] = d[i + 1] = d[i + 2] = y;
    }
    return imageData;
  },

  threshold: (imageData: ImageData, threshold: number) => {
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      const y = d[i];
      const originalAlpha = d[i + 3];
      const value = y > threshold ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = value;
      d[i + 3] = originalAlpha;
    }
    return imageData;
  },

  floydSteinberg: (imageData: ImageData, w: number, h: number) =>
    diffuse(
      imageData,
      w,
      h,
      [
        { x: 1, y: 0, w: 7 },
        { x: -1, y: 1, w: 3 },
        { x: 0, y: 1, w: 5 },
        { x: 1, y: 1, w: 1 },
      ],
      16,
    ),

  atkinson: (imageData: ImageData, w: number, h: number) =>
    diffuse(
      imageData,
      w,
      h,
      [
        { x: 1, y: 0, w: 1 },
        { x: 2, y: 0, w: 1 },
        { x: -1, y: 1, w: 1 },
        { x: 0, y: 1, w: 1 },
        { x: 1, y: 1, w: 1 },
        { x: 0, y: 2, w: 1 },
      ],
      8,
    ),

  jjn: (imageData: ImageData, w: number, h: number) =>
    diffuse(
      imageData,
      w,
      h,
      [
        { x: 1, y: 0, w: 7 },
        { x: 2, y: 0, w: 5 },
        { x: -2, y: 1, w: 3 },
        { x: -1, y: 1, w: 5 },
        { x: 0, y: 1, w: 7 },
        { x: 1, y: 1, w: 5 },
        { x: 2, y: 1, w: 3 },
        { x: -2, y: 2, w: 1 },
        { x: -1, y: 2, w: 3 },
        { x: 0, y: 2, w: 5 },
        { x: 1, y: 2, w: 3 },
        { x: 2, y: 2, w: 1 },
      ],
      48,
    ),

  stucki: (imageData: ImageData, w: number, h: number) =>
    diffuse(
      imageData,
      w,
      h,
      [
        { x: 1, y: 0, w: 8 },
        { x: 2, y: 0, w: 4 },
        { x: -2, y: 1, w: 2 },
        { x: -1, y: 1, w: 4 },
        { x: 0, y: 1, w: 8 },
        { x: 1, y: 1, w: 4 },
        { x: 2, y: 1, w: 2 },
        { x: -2, y: 2, w: 1 },
        { x: -1, y: 2, w: 2 },
        { x: 0, y: 2, w: 4 },
        { x: 1, y: 2, w: 2 },
        { x: 2, y: 2, w: 1 },
      ],
      42,
    ),

  bayer2: (imageData: ImageData, w: number, h: number) => orderedDither(imageData, w, h, generateBayer(2)),
  bayer4: (imageData: ImageData, w: number, h: number) => orderedDither(imageData, w, h, generateBayer(4)),
  bayer8: (imageData: ImageData, w: number, h: number) => orderedDither(imageData, w, h, generateBayer(8)),

  clustered4x4: (imageData: ImageData, w: number, h: number) =>
    orderedDither(
      imageData,
      w,
      h,
      [
        [15, 8, 9, 16],
        [7, 1, 2, 10],
        [6, 4, 3, 11],
        [14, 13, 12, 5],
      ],
    ),

  random: (imageData: ImageData) => {
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      const noise = Math.random() * 255;
      const v = d[i] > noise ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = 255;
    }
    return imageData;
  },
};

const DITHERING_METHODS = [
  'BITMAP',
  'STRETCH',
  'FLOYD-STEINBERG',
  'ATKINSON',
  'JARVIS-JUDICE-NINKE',
  'STUCKI',
  'BAYER 2X2',
  'BAYER 4X4',
  'BAYER 8X8',
  'CLUSTERED 4X4',
  'RANDOM',
];

function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [size, setSize] = useState<number>(46);
  const [threshold, setThreshold] = useState<number>(46);
  const [ditheringMethod, setDitheringMethod] = useState<string>('BITMAP');
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(new Image());

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result;
        if (typeof result === 'string') {
          setImageSrc(result);
          setZoom(1);
          setPan({ x: 0, y: 0 });
        }
      };
      reader.readAsDataURL(event.target.files[0]);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context || !imageSrc) return;

    imageRef.current.crossOrigin = 'Anonymous';
    imageRef.current.src = imageSrc;
    imageRef.current.onload = () => {
      const img = imageRef.current;
      canvas.width = img.width;
      canvas.height = img.height;
  
      const scale = size / 100;
      const lowWidth = Math.max(1, Math.floor(img.width * scale));
      const lowHeight = Math.max(1, Math.floor(img.height * scale));
      const lowCanvas = document.createElement('canvas');
      lowCanvas.width = lowWidth;
      lowCanvas.height = lowHeight;
      const lowContext = lowCanvas.getContext('2d')!;
  
      lowContext.fillStyle = 'white';
      lowContext.fillRect(0, 0, lowWidth, lowHeight);
  
      lowContext.imageSmoothingEnabled = false;
      lowContext.drawImage(img, 0, 0, lowWidth, lowHeight);
  
      let imageData = lowContext.getImageData(0, 0, lowWidth, lowHeight);
      imageData = Dithering.grayscale(imageData);
  
      switch (ditheringMethod) {
        case 'BITMAP':
          imageData = Dithering.threshold(imageData, threshold * 2.55);
          break;
        case 'STRETCH':
          {
            const d = imageData.data;
            const mid = threshold * 2.55;
            for (let i = 0; i < d.length; i += 4) {
              const y = d[i];
              const originalAlpha = d[i + 3];
              const v = y < mid ? (y * 255) / Math.max(1, mid) : 255 - ((255 - y) * 255) / Math.max(1, 255 - mid);
              d[i] = d[i + 1] = d[i + 2] = clamp(v);
              d[i + 3] = originalAlpha;
            }
            imageData = Dithering.threshold(imageData, mid);
          }
          break;
        case 'FLOYD-STEINBERG':
          imageData = Dithering.floydSteinberg(imageData, lowWidth, lowHeight);
          break;
        case 'ATKINSON':
          imageData = Dithering.atkinson(imageData, lowWidth, lowHeight);
          break;
        case 'JARVIS-JUDICE-NINKE':
          imageData = Dithering.jjn(imageData, lowWidth, lowHeight);
          break;
        case 'STUCKI':
          imageData = Dithering.stucki(imageData, lowWidth, lowHeight);
          break;
        case 'BAYER 2X2':
          imageData = Dithering.bayer2(imageData, lowWidth, lowHeight);
          break;
        case 'BAYER 4X4':
          imageData = Dithering.bayer4(imageData, lowWidth, lowHeight);
          break;
        case 'BAYER 8X8':
          imageData = Dithering.bayer8(imageData, lowWidth, lowHeight);
          break;
        case 'CLUSTERED 4X4':
          imageData = Dithering.clustered4x4(imageData, lowWidth, lowHeight);
          break;
        case 'RANDOM':
          imageData = Dithering.random(imageData);
          break;
        default:
          imageData = Dithering.threshold(imageData, threshold * 2.55);
      }
  
      lowContext.putImageData(imageData, 0, 0);
  
      context.imageSmoothingEnabled = false;
      context.drawImage(lowCanvas, 0, 0, canvas.width, canvas.height);
    };
  }, [imageSrc, size, threshold, ditheringMethod]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom(prev => {
        const newZoom = Math.max(0.1, Math.min(5, prev * delta));
        if (newZoom === 1) {
          setPan({ x: 0, y: 0 });
        }
        return newZoom;
      });
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  return (
    <div className="h-screen bg-white text-black font-mono overflow-hidden">
      <div className="flex flex-row items-stretch gap-4 h-full">
        <div className="w-56 p-3 border border-black flex flex-col self-stretch shrink-0">
          <div className="mb-4">
            <div className="inline-block px-3 py-1 border border-black font-bold text-sm">BITMAP</div>
          </div>

          <div className="mt-6">
            <p className="text-xs">PNG, JPG, WEBP:</p>
            <div className="relative mt-1">
              <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="file-upload" />
              <label htmlFor="file-upload" className="cursor-pointer select-none border border-black bg-black text-white text-xs px-3 py-1 inline-block">UPLOAD</label>
            </div>
          </div>

          <div className="mt-5">
            <label htmlFor="size" className="block text-xs">SIZE:</label>
            <div className="flex items-center gap-2">
              <input id="size" type="range" min="1" max="100" value={size} onChange={(e) => setSize(Number(e.target.value))} className="range" />
              <span className="text-xs w-8 text-right">{size}</span>
            </div>
          </div>

          <div className="mt-4">
            <label htmlFor="threshold" className="block text-xs">THRESHOLD:</label>
            <div className="flex items-center gap-2">
              <input id="threshold" type="range" min="1" max="100" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="range" />
              <span className="text-xs w-8 text-right">{threshold}</span>
            </div>
          </div>

          <div className="mt-4">
            <label htmlFor="dithering" className="block text-xs">DITHERING METHOD:</label>
            <div className="relative mt-1">
              <select id="dithering" value={ditheringMethod} onChange={(e) => setDitheringMethod(e.target.value)} className="w-full border border-black bg-white text-xs px-2 py-1 focus:outline-none">
                {DITHERING_METHODS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div 
          ref={containerRef}
          className="flex-1 p-6 flex items-start justify-center overflow-hidden"
          onMouseDown={(e) => {
            setIsDragging(true);
            setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
          }}
          onMouseMove={(e) => {
            if (isDragging) {
              setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
            }
          }}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
        >
          {imageSrc ? (
            <canvas 
              ref={canvasRef} 
              className="max-w-full h-auto"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center',
                willChange: 'transform',
                imageRendering: 'pixelated'
              }}
            />
          ) : (
            <div className="w-full h-[70vh] border-2 border-dashed border-gray-400 flex items-center justify-center text-xs">
              <p>Upload an image to begin</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
