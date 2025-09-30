
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

type AnimationType = 'none' | 'glitch' | 'scanline' | 'pulse' | 'matrix';

function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [size, setSize] = useState<number>(46);
  const [threshold, setThreshold] = useState<number>(46);
  const [ditheringMethod, setDitheringMethod] = useState<string>('BITMAP');
  const [showSplit, setShowSplit] = useState<boolean>(true);
  const [splitPosition, setSplitPosition] = useState<number>(50);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [isSplitting, setIsSplitting] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [animation, setAnimation] = useState<AnimationType>('none');
  const [animationIntensity, setAnimationIntensity] = useState<number>(50);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(new Image());
  const afterLowCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationStartTimeRef = useRef<number>(Date.now());

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

  const handleExport = () => {
    const afterLowCanvas = afterLowCanvasRef.current;
    const canvas = canvasRef.current;
    if (!afterLowCanvas || !canvas) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = canvas.width;
    exportCanvas.height = canvas.height;
    const exportCtx = exportCanvas.getContext('2d')!;
    exportCtx.imageSmoothingEnabled = false;
    exportCtx.drawImage(afterLowCanvas, 0, 0, exportCanvas.width, exportCanvas.height);

    const link = document.createElement('a');
    link.download = 'dithered-image.png';
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
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

      const beforeCanvas = document.createElement('canvas');
      beforeCanvas.width = canvas.width;
      beforeCanvas.height = canvas.height;
      const beforeCtx = beforeCanvas.getContext('2d')!;
      beforeCtx.fillStyle = 'white';
      beforeCtx.fillRect(0, 0, canvas.width, canvas.height);
      beforeCtx.drawImage(img, 0, 0);
      let beforeImageData = beforeCtx.getImageData(0, 0, canvas.width, canvas.height);
      beforeImageData = Dithering.grayscale(beforeImageData);
      beforeCtx.putImageData(beforeImageData, 0, 0);

      const scale = size / 100;
      const lowWidth = Math.max(1, Math.floor(img.width * scale));
      const lowHeight = Math.max(1, Math.floor(img.height * scale));
      const afterLowCanvas = document.createElement('canvas');
      afterLowCanvas.width = lowWidth;
      afterLowCanvas.height = lowHeight;
      const afterLowCtx = afterLowCanvas.getContext('2d')!;
      afterLowCtx.fillStyle = 'white';
      afterLowCtx.fillRect(0, 0, lowWidth, lowHeight);
      afterLowCtx.imageSmoothingEnabled = false;
      afterLowCtx.drawImage(img, 0, 0, lowWidth, lowHeight);
      let afterImageData = afterLowCtx.getImageData(0, 0, lowWidth, lowHeight);
      afterImageData = Dithering.grayscale(afterImageData);

      switch (ditheringMethod) {
        case 'BITMAP':
          Dithering.threshold(afterImageData, threshold * 2.55);
          break;
        case 'STRETCH':
          {
            const d = afterImageData.data;
            const mid = threshold * 2.55;
            for (let i = 0; i < d.length; i += 4) {
              const y = d[i];
              const originalAlpha = d[i + 3];
              const v = y < mid ? (y * 255) / Math.max(1, mid) : 255 - ((255 - y) * 255) / Math.max(1, 255 - mid);
              d[i] = d[i + 1] = d[i + 2] = clamp(v);
              d[i + 3] = originalAlpha;
            }
            Dithering.threshold(afterImageData, mid);
          }
          break;
        case 'FLOYD-STEINBERG':
          Dithering.floydSteinberg(afterImageData, lowWidth, lowHeight);
          break;
        case 'ATKINSON':
          Dithering.atkinson(afterImageData, lowWidth, lowHeight);
          break;
        case 'JARVIS-JUDICE-NINKE':
          Dithering.jjn(afterImageData, lowWidth, lowHeight);
          break;
        case 'STUCKI':
          Dithering.stucki(afterImageData, lowWidth, lowHeight);
          break;
        case 'BAYER 2X2':
          Dithering.bayer2(afterImageData, lowWidth, lowHeight);
          break;
        case 'BAYER 4X4':
          Dithering.bayer4(afterImageData, lowWidth, lowHeight);
          break;
        case 'BAYER 8X8':
          Dithering.bayer8(afterImageData, lowWidth, lowHeight);
          break;
        case 'CLUSTERED 4X4':
          Dithering.clustered4x4(afterImageData, lowWidth, lowHeight);
          break;
        case 'RANDOM':
          Dithering.random(afterImageData);
          break;
        default:
          Dithering.threshold(afterImageData, threshold * 2.55);
      }

      afterLowCtx.putImageData(afterImageData, 0, 0);
      afterLowCanvasRef.current = afterLowCanvas;

      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, canvas.width, canvas.height);

      if (showSplit) {
        const splitX = canvas.width * (splitPosition / 100);

        if (splitX > 0) {
          const sourceWidth = afterLowCanvas.width * (splitPosition / 100);
          context.drawImage(afterLowCanvas, 0, 0, sourceWidth, afterLowCanvas.height, 0, 0, splitX, canvas.height);
        }

        if (splitX < canvas.width) {
          const sourceX = beforeCanvas.width * (splitPosition / 100);
          const sourceWidth = beforeCanvas.width - sourceX;
          const destWidth = canvas.width - splitX;
          context.drawImage(beforeCanvas, sourceX, 0, sourceWidth, beforeCanvas.height, splitX, 0, destWidth, canvas.height);
        }

        context.fillStyle = 'black';
        context.fillRect(splitX - 0.5, 0, 1, canvas.height);

        const handleHeight = 30;
        const handleWidth = 16;
        const handleY = canvas.height / 2 - handleHeight / 2;

        context.fillStyle = 'white';
        context.fillRect(splitX - handleWidth / 2, handleY, handleWidth, handleHeight);
        context.strokeStyle = 'black';
        context.lineWidth = 1;
        context.strokeRect(splitX - handleWidth / 2, handleY, handleWidth, handleHeight);

        context.fillStyle = 'black';
        context.beginPath();
        context.moveTo(splitX - 2, canvas.height / 2);
        context.lineTo(splitX - 6, canvas.height / 2 - 4);
        context.lineTo(splitX - 6, canvas.height / 2 + 4);
        context.closePath();
        context.fill();
        context.beginPath();
        context.moveTo(splitX + 2, canvas.height / 2);
        context.lineTo(splitX + 6, canvas.height / 2 - 4);
        context.lineTo(splitX + 6, canvas.height / 2 + 4);
        context.closePath();
        context.fill();
      } else {
        context.drawImage(afterLowCanvas, 0, 0, canvas.width, canvas.height);
      }
    };
  }, [imageSrc, size, threshold, ditheringMethod, splitPosition, showSplit]);

  useEffect(() => {
    animationStartTimeRef.current = Date.now();
  }, [animation]);

  useEffect(() => {
    if (animation === 'none') {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const canvas = canvasRef.current;
    const overlay = overlayCanvasRef.current;
    if (!canvas || !overlay) return;

    overlay.width = canvas.width;
    overlay.height = canvas.height;
    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    const intensity = animationIntensity / 100;

    const animate = () => {
      const elapsed = (Date.now() - animationStartTimeRef.current) / 1000;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      if (showSplit) {
        const splitX = overlay.width * (splitPosition / 100);
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, splitX, overlay.height);
        ctx.clip();
      }

      switch (animation) {
        case 'glitch': {
          if (Math.random() < 0.1 * intensity) {
            const sliceHeight = Math.random() * 20 + 5;
            const y = Math.random() * overlay.height;
            const offset = (Math.random() - 0.5) * 20 * intensity;
            ctx.fillStyle = Math.random() > 0.5 ? 'rgba(255,0,0,0.3)' : 'rgba(0,255,255,0.3)';
            ctx.fillRect(offset, y, overlay.width, sliceHeight);
          }
          break;
        }

        case 'scanline': {
          const lineSpeed = 100 * intensity;
          const scanY = (elapsed * lineSpeed) % (overlay.height + 50);
          ctx.fillStyle = 'rgba(255,255,255,0.1)';
          for (let i = 0; i < overlay.height; i += 4) {
            ctx.fillRect(0, i, overlay.width, 2);
          }
          ctx.fillStyle = 'rgba(255,255,255,0.3)';
          ctx.fillRect(0, scanY - 25, overlay.width, 50);
          break;
        }

        case 'pulse': {
          const afterLowCanvas = afterLowCanvasRef.current;
          if (afterLowCanvas) {
            ctx.save();
            ctx.imageSmoothingEnabled = false;
            const scale = 1 + Math.sin(elapsed * 2 * intensity) * 0.02 * intensity;
            ctx.translate(overlay.width / 2, overlay.height / 2);
            ctx.scale(scale, scale);
            ctx.translate(-overlay.width / 2, -overlay.height / 2);
            ctx.drawImage(
              afterLowCanvas,
              0, 0, afterLowCanvas.width, afterLowCanvas.height,
              0, 0, overlay.width, overlay.height
            );
            ctx.restore();
          }
          break;
        }

        case 'matrix': {
          const chars = '01';
          ctx.fillStyle = 'rgba(0,0,0,0.05)';
          ctx.fillRect(0, 0, overlay.width, overlay.height);
          ctx.fillStyle = '#00ff00';
          ctx.font = '12px monospace';
          const changeInterval = 0.15;
          const changeTime = Math.floor(elapsed / changeInterval);
          for (let i = 0; i < overlay.width; i += 20) {
            const seed = changeTime + i;
            const charIndex = Math.floor(Math.sin(seed * 12.9898) * 43758.5453) % 2;
            const char = chars[Math.abs(charIndex)];
            const y = (elapsed * 100 * intensity + i * 20) % overlay.height;
            ctx.fillText(char, i, y);
          }
          break;
        }
      }

      if (showSplit) {
        ctx.restore();

        const splitX = overlay.width * (splitPosition / 100);

        ctx.fillStyle = 'black';
        ctx.fillRect(splitX - 0.5, 0, 1, overlay.height);

        const handleHeight = 30;
        const handleWidth = 16;
        const handleY = overlay.height / 2 - handleHeight / 2;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(splitX - handleWidth / 2, handleY, handleWidth, handleHeight);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.strokeRect(splitX - handleWidth / 2, handleY, handleWidth, handleHeight);

        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.moveTo(splitX - 2, overlay.height / 2);
        ctx.lineTo(splitX - 6, overlay.height / 2 - 4);
        ctx.lineTo(splitX - 6, overlay.height / 2 + 4);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(splitX + 2, overlay.height / 2);
        ctx.lineTo(splitX + 6, overlay.height / 2 - 4);
        ctx.lineTo(splitX + 6, overlay.height / 2 + 4);
        ctx.closePath();
        ctx.fill();
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [animation, animationIntensity, showSplit, splitPosition]);

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
            <div className="flex flex-col items-start">
              <div className="relative">
                <div className="font-bold text-lg tracking-wider">
                  <span className="inline-block border border-black px-2 py-1 bg-black text-white">BIT</span>
                  <span className="inline-block border border-black border-l-0 px-2 py-1 bg-white text-black">FLOW</span>
                </div>
              </div>
              <div className="w-full border-t border-black mt-3 mb-1"></div>
            </div>
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

          <div className="mt-4">
            <label htmlFor="show-split" className="flex items-center gap-2 text-xs cursor-pointer select-none">
              <input 
                id="show-split" 
                type="checkbox" 
                checked={showSplit} 
                onChange={(e) => setShowSplit(e.target.checked)}
                className="sr-only peer" 
              />
              <div className="w-3 h-3 border border-black bg-white peer-checked:bg-black"></div>
              <span>Show Before/After</span>
            </label>
          </div>

          <div className="mt-4">
            <label htmlFor="animation" className="block text-xs">ANIMATION:</label>
            <div className="relative mt-1">
              <select 
                id="animation" 
                value={animation} 
                onChange={(e) => setAnimation(e.target.value as AnimationType)} 
                className="w-full border border-black bg-white text-xs px-2 py-1 focus:outline-none"
              >
                <option value="none">NONE</option>
                <option value="glitch">GLITCH</option>
                <option value="scanline">SCANLINE</option>
                <option value="pulse">PULSE</option>
                <option value="matrix">MATRIX</option>
              </select>
            </div>
          </div>

          {animation !== 'none' && (
            <div className="mt-4">
              <label htmlFor="intensity" className="block text-xs">INTENSITY:</label>
              <div className="flex items-center gap-2">
                <input 
                  id="intensity" 
                  type="range" 
                  min="1" 
                  max="100" 
                  value={animationIntensity} 
                  onChange={(e) => setAnimationIntensity(Number(e.target.value))} 
                  className="range" 
                />
                <span className="text-xs w-8 text-right">{animationIntensity}</span>
              </div>
            </div>
          )}

          <div className="mt-auto pt-4">
            <button onClick={handleExport} className="w-full select-none border border-black bg-black text-white text-xs px-3 py-1">EXPORT</button>
          </div>
        </div>

        <div 
          ref={containerRef}
          className="relative flex-1 p-6 flex items-start justify-center overflow-hidden"
          onMouseDown={(e) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const lineX = rect.width * (splitPosition / 100);

            if (showSplit && Math.abs(x - lineX) < 10) {
              setIsSplitting(true);
            } else {
              setIsDragging(true);
              setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
            }
          }}
          onMouseMove={(e) => {
            if (isSplitting) {
              const canvas = canvasRef.current;
              if (!canvas) return;
              const rect = canvas.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const percent = (x / rect.width) * 100;
              setSplitPosition(Math.max(0, Math.min(100, percent)));
            } else if (isDragging) {
              setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
            }
          }}
          onMouseUp={() => {
            setIsDragging(false);
            setIsSplitting(false);
          }}
          onMouseLeave={() => {
            setIsDragging(false);
            setIsSplitting(false);
          }}
        >
          {imageSrc ? (
            <div className="relative">
              <canvas 
                ref={canvasRef} 
                className="max-w-full h-auto"
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: 'center center',
                  willChange: 'transform',
                  imageRendering: 'pixelated',
                  cursor: isSplitting ? 'ew-resize' : 'grab'
                }}
              />
              <canvas 
                ref={overlayCanvasRef}
                className="absolute top-0 left-0 max-w-full h-auto pointer-events-none"
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: 'center center',
                  willChange: 'transform',
                  imageRendering: 'pixelated',
                  mixBlendMode: animation === 'matrix' ? 'screen' : 'normal'
                }}
              />
            </div>
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
