import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import axios from 'axios';
import { ClerkExpressWithAuth, StrictAuthProp } from '@clerk/clerk-sdk-node';
import rateLimit from 'express-rate-limit';

// Extend Request type for Clerk
declare global {
  namespace Express {
    interface Request extends StrictAuthProp {}
  }
}

const app = express();
const port = process.env.PORT || 3001;

// Constants from Env
const INTERNAL_SECRET = process.env.INTERNAL_API_SECRET || 'dev_secret';
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';
const LLM_SERVICE_URL = process.env.LLM_SERVICE_URL || 'http://localhost:8002';

// 1. Security: Rate Limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per window
  message: { error: 'Too many requests, please try again later.' }
});

app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(limiter);

// 2. Health Check (Public)
app.get('/health', (req: Request, res: Response) => {
  res.status(200).json({ 
    status: 'ok', 
    service: 'mira-api-gateway',
    timestamp: new Date().toISOString()
  });
});

// 3. Protected Routes (Clerk Auth Required)
app.use(ClerkExpressWithAuth());

// Proxy to ML Service
app.post('/api/ml/:path*', async (req: Request, res: Response) => {
  if (!req.auth.userId) return res.status(401).json({ error: 'Unauthorized' });

  const targetPath = req.params.path + (req.params[0] || '');
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/${targetPath}`, req.body, {
      headers: { 'x-internal-secret': INTERNAL_SECRET }
    });
    res.json(response.data);
  } catch (error: any) {
    res.status(error.response?.status || 500).json({ 
      error: 'ML Service Error', 
      detail: error.response?.data?.detail || 'Unknown error' 
    });
  }
});

// Proxy to LLM Service
app.post('/api/llm/:path*', async (req: Request, res: Response) => {
  if (!req.auth.userId) return res.status(401).json({ error: 'Unauthorized' });

  const targetPath = req.params.path + (req.params[0] || '');
  try {
    const response = await axios.post(`${LLM_SERVICE_URL}/${targetPath}`, req.body, {
      headers: { 'x-internal-secret': INTERNAL_SECRET }
    });
    res.json(response.data);
  } catch (error: any) {
    res.status(error.response?.status || 500).json({ 
      error: 'LLM Service Error', 
      detail: error.response?.data?.detail || 'Unknown error' 
    });
  }
});

app.listen(port, () => {
  console.log(`MIRA API Gateway running on port ${port}`);
});
