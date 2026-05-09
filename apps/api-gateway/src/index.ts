import express from 'express';
import cors from 'cors';
import helmet from 'helmet';

const app = express();
const port = process.env.PORT || 3001;

app.use(helmet());
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', service: 'mira-api-gateway' });
});

app.listen(port, () => {
  console.log(`MIRA API Gateway running on port ${port}`);
});
