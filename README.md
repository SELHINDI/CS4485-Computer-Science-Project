# GDP Predictor - Frontend Application Overview

## 🎯 **What This App Does**
This is a **GDP forecasting web application** that predicts economic growth using machine learning models. It provides interactive dashboards, multi-step forecasts, data exploration, and model validation metrics.

## 🏗️ **Architecture & Tech Stack**

### **Core Framework**
- **React 19** + **TypeScript** + **Vite** (fast build tool)
- **React Router v6** for navigation
- **React Query** for API state management and caching
- **Tailwind CSS v4** for styling with custom design system

### **UI Components**
- **Recharts** for interactive charts (line, bar charts)
- **Lucide React** for icons
- **Radix UI** primitives (tooltips, popovers, selects)
- **Custom components** with glassmorphism design

### **Testing & Quality**
- **Vitest** + **Testing Library** for unit/integration tests
- **ESLint** + **TypeScript** for code quality
- **Accessibility** (ARIA labels, keyboard navigation, colorblind-friendly)

## 📱 **Four Main Pages**

### **1. Dashboard (`/`)**
- **Next Quarter GDP** metric card with sparkline
- **Main chart** showing historical GDP + predicted next quarter
- **Mini cards** for Level, Slope, Curvature, Inflation, Unemployment
- **Controls**: Country selector, date picker, unit toggle (Level vs % QoQ)

### **2. Multi-Step Forecasts (`/multi`)**
- **Horizon selection** (2, 3, 4, 8, 12 quarters)
- **Chart** with actual data + dashed prediction lines for each horizon
- **Table** showing horizon → predicted GDP values
- **CSV export** functionality

### **3. Data Explorer (`/explorer`)**
- **Two tabs**: "Yield Curve Proxies" and "Macro"
- **Proxies tab**: Level, Slope, Curvature charts
- **Macro tab**: Inflation, Unemployment, Capacity Utilization, Fed Funds
- **Info tooltips** explaining each metric

### **4. Validation Metrics (`/metrics`)**
- **MSE timeline** chart (one-step predictions over time)
- **Bar chart** comparing average MSE by model (KNN, LR, SARIMAX, ARIMA)
- **Detailed table** with date, y_true, y_pred, abs_error
- **Date range controls** for evaluation period

## 🔧 **Key Features**

### **Global State Management**
```typescript
// Context provides:
- country: string (US, CA, UK, EU)
- asOfDate: string | null (forecast date)
- unitMode: 'level' | 'pct_qoq' (display units)
- horizons: number[] (selected forecast horizons)
- theme: 'light' | 'dark' (with system preference)
```

### **Data Layer**
- **Mock API** with realistic delays (300ms)
- **React Query** hooks with 5-minute cache
- **Error handling** with retry logic
- **Loading states** and skeletons

### **Export Capabilities**
- **CSV export** for all tables
- **PNG export** for charts (using html-to-image)
- **Copy-to-clipboard** for headline numbers

## 🎨 **Design System**

### **Color Palette**
```css
Primary: #5b8def (blue)
Secondary: #7bdcb5 (teal)
Background: #f7f8fc (light) / #0b0f19 (dark)
Neutrals: #1b2333, #2a3347, #e5e7eb
```

### **Components**
- **Cards**: Rounded corners, soft shadows, glassmorphism
- **Charts**: Consistent stroke colors, dashed future predictions
- **Typography**: Inter font, semibold headings
- **Responsive**: Mobile-first, ≥360px width support

## 📁 **File Structure**
```
src/
├── app/
│   ├── components/          # Reusable UI components
│   │   ├── Navbar.tsx      # Navigation with theme toggle
│   │   ├── Controls.tsx    # Date/unit/horizon controls
│   │   ├── MetricCard.tsx  # GDP metric with sparkline
│   │   ├── ForecastChart.tsx # Recharts wrapper
│   │   ├── DataTable.tsx   # Paginated table with CSV export
│   │   ├── States.tsx      # Skeleton/Error/Empty states
│   │   └── InfoTooltip.tsx # Radix tooltip wrapper
│   └── context/
│       └── AppContext.tsx  # Global state management
├── lib/
│   ├── api/
│   │   └── mock.ts         # Mock API responses
│   ├── hooks/
│   │   └── useData.ts      # React Query hooks
│   └── export/
│       └── exportImage.ts   # PNG export utility
├── pages/                  # Route components
│   ├── Dashboard.tsx
│   ├── Multi.tsx
│   ├── Explorer.tsx
│   └── Metrics.tsx
└── main.tsx               # App entry point
```

## 🔌 **API Integration**

### **Expected Backend Endpoints**
```typescript
GET /predict/one-step?country={country}&asof={date}
GET /predict/multi?country={country}&horizons=2,4,8,12&asof={date}
GET /series?country={country}&from={YYYY}&to={YYYY}
GET /metrics?from={YYYY}&to={YYYY}
```

### **Data Shapes**
```typescript
// One-step prediction
{ country: "US", as_of: "2024-12-31", horizon: 1, gdp_pred: 20834.2, unit: "billions_chained_2017" }

// Multi-step predictions
{ country: "US", as_of: "2024-12-31", preds: {"2": 20910.4, "4": 21123.5, "8": 21560.1, "12": 22010.7} }

// Time series data
{ series: { gdp: [{date: "2022-03-31", value: 20120.1}], level: [...], slope: [...], ... } }

// Validation metrics
{ one_step: { avg_mse: {knn: 0.0012, lr: 0.0018, ...}, timeline: [...] }, multi: {...} }
```

## 🚀 **For Backend Team**

### **What You Need to Implement**
1. **Four API endpoints** matching the data shapes above
2. **CORS headers** for localhost:5173
3. **Error responses** (400/500) with JSON error messages
4. **Date format**: YYYY-MM-DD for all date parameters

### **Current Mock Data**
- **Realistic GDP values** (18,000-22,000 range)
- **Random variations** for different countries
- **Time series** with quarterly data points
- **MSE values** for model comparison

### **Performance Requirements**
- **Response time**: <500ms for all endpoints
- **Caching**: 5-minute stale time configured
- **Error handling**: Graceful degradation with retry buttons

## 🧪 **Testing**
```bash
npm test          # Run Vitest tests
npm run test:ui   # Interactive test UI
npm run build     # Production build
npm run preview   # Preview production build
```

## 📦 **Dependencies to Install**
```bash
npm install
```
