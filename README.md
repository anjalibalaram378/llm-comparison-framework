# Comprehensive LLM Comparison & Benchmarking Framework

Production-grade framework for evaluating and comparing 8-10 LLMs across multiple dimensions and domain-specific tasks.

## Overview

Systematically benchmark LLMs on finance, ad tech, and general reasoning tasks. Provides interactive dashboard, automated testing, and comprehensive reports to guide production LLM selection.

## Key Features

### Core Capabilities
- **10 LLMs Tested:** Llama 3 (8B/70B), Mistral, Phi-3, Gemma, GPT-3.5/4, Claude, Gemini, Groq
- **Multi-Domain Benchmarks:** Finance reasoning, ad tech, general knowledge
- **Multi-Dimensional Metrics:** Speed, accuracy, cost, context handling, reliability
- **Automated Testing:** Run 200+ test cases across all models
- **Interactive Dashboard:** Real-time benchmarking with filterable results
- **Production Simulation:** Concurrency, rate limiting, failover testing
- **PDF Reports:** Exportable comparison documents

### Benchmark Suites

#### Finance Tasks (50 questions)
- SEC filing Q&A
- Financial sentiment analysis
- Metric extraction accuracy
- Multi-hop reasoning

#### Ad Tech Tasks (50 questions)
- Ad copy generation quality
- Audience targeting suggestions
- CTR prediction reasoning
- Campaign optimization logic

#### General Benchmarks
- MMLU (reasoning)
- HellaSwag (common sense)
- TruthfulQA (factuality)
- Custom domain tasks

## Tech Stack

### LLMs Tested

**Local/Free (via Ollama):**
1. Llama 3 8B
2. Llama 3 70B
3. Mistral 7B
4. Phi-3 Mini
5. Gemma 2 9B

**API-Based:**
6. GPT-3.5 Turbo (OpenAI)
7. GPT-4 (OpenAI)
8. Claude 3 Haiku (Anthropic)
9. Gemini 1.5 Flash (Google)
10. Groq Llama 3 (Ultra-fast)

### Infrastructure
- **Backend:** FastAPI for benchmark runner
- **Frontend:** React + Recharts + shadcn/ui
- **Database:** PostgreSQL for results storage
- **GPU:** Modal for on-demand heavy models
- **Monitoring:** Prometheus + Grafana
- **Deployment:** Vercel (frontend) + Railway (backend)

## Architecture

```
Benchmark Runner (FastAPI)
    ↓
┌─────────────────────────────────┐
│  Test Suite Executor            │
│  - Finance (50 Q)               │
│  - Ad Tech (50 Q)               │
│  - General (100 Q)              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  LLM Clients (10 models)        │
│  - Ollama (local)               │
│  - OpenAI, Anthropic APIs       │
│  - Google, Groq APIs            │
│  - Modal (serverless GPU)       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Metrics Collection             │
│  - Speed (tok/s, latency, TTFT) │
│  - Quality (accuracy, F1)       │
│  - Cost ($ per 1M tokens)       │
│  - Reliability (error rate)     │
└─────────────────────────────────┘
    ↓
PostgreSQL → Dashboard → Reports
```

## Metrics Measured

### Performance Metrics
- **Tokens per second** (generation speed)
- **Latency** (end-to-end response time)
- **TTFT** (Time to first token)
- **Throughput** (requests per minute)

### Quality Metrics
- **Accuracy** (correct answers %)
- **F1 Score** (precision + recall)
- **Hallucination Rate** (false claims %)
- **Instruction Following** (task adherence)
- **Factuality** (TruthfulQA score)

### Cost Metrics
- **Price per 1M tokens** (input/output)
- **Total benchmark cost**
- **Cost per correct answer**
- **ROI analysis**

### Context Metrics
- **Max context window**
- **Long-context accuracy**
- **Context utilization efficiency**

### Reliability Metrics
- **Error rate** (failed requests %)
- **Timeout frequency**
- **Availability** (uptime %)

## Installation

```bash
# Clone repository
git clone https://github.com/anjalibalaram378/llm-comparison-framework.git
cd llm-comparison-framework

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

# Install Ollama and models
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
ollama pull llama3:70b
ollama pull mistral
ollama pull phi3
ollama pull gemma2:9b

# Environment variables
cp .env.example .env
# Add API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY
```

## Usage

### Run Full Benchmark Suite
```bash
# Start backend
cd backend
python main.py

# Start frontend
cd frontend
npm run dev
```

### Run Specific Benchmarks
```bash
# Finance tasks only
python run_benchmark.py --suite finance --models llama3,gpt-3.5,claude

# Ad tech tasks
python run_benchmark.py --suite adtech --models all

# Speed test only
python run_benchmark.py --metric speed --models groq,gpt-3.5
```

### Generate Reports
```bash
python generate_report.py --format pdf --output results/comparison_report.pdf
```

## Benchmark Results

### Speed Comparison (Coming Soon)
| Model | Tokens/sec | Latency (avg) | TTFT (ms) | Best For |
|-------|-----------|---------------|-----------|----------|
| Groq Llama 3 | TBD | TBD | TBD | Speed |
| Llama 3 8B | TBD | TBD | TBD | Local |
| Llama 3 70B | TBD | TBD | TBD | Quality |
| GPT-3.5 | TBD | TBD | TBD | Baseline |
| GPT-4 | TBD | TBD | TBD | Accuracy |
| Claude Haiku | TBD | TBD | TBD | Value |
| Gemini Flash | TBD | TBD | TBD | Free |

### Accuracy Comparison (Coming Soon)
| Model | Finance (%) | Ad Tech (%) | General (%) | Overall |
|-------|------------|-------------|-------------|---------|
| Llama 3 8B | TBD | TBD | TBD | TBD |
| Llama 3 70B | TBD | TBD | TBD | TBD |
| GPT-3.5 | TBD | TBD | TBD | TBD |
| GPT-4 | TBD | TBD | TBD | TBD |
| Claude Haiku | TBD | TBD | TBD | TBD |
| Gemini Flash | TBD | TBD | TBD | TBD |

### Cost Comparison (Coming Soon)
| Model | Price ($/1M tok) | Benchmark Cost | Cost per Correct | ROI Score |
|-------|-----------------|----------------|------------------|-----------|
| Llama 3 8B | $0 | $0 | $0 | ∞ |
| Llama 3 70B | $0 | $0 | $0 | ∞ |
| GPT-3.5 | $0.50 | TBD | TBD | TBD |
| GPT-4 | $15 | TBD | TBD | TBD |
| Claude Haiku | $0.25 | TBD | TBD | TBD |
| Gemini Flash | $0 | $0 | $0 | ∞ |

## Features

### Interactive Dashboard
- Real-time benchmark execution
- Filterable leaderboard
- Multi-metric sorting
- Cost calculator
- Export to CSV/PDF

### Automated Testing
- Parallel execution across models
- Retry logic for failures
- Progress tracking
- Result caching

### Production Simulation
- Concurrent request handling
- Rate limit testing
- Failover scenarios
- Load testing

## Project Roadmap

- [x] Project setup (Day 26)
- [ ] Implement test runners for 10 LLMs (Day 26-27)
- [ ] Finance task suite (50 questions) (Day 27)
- [ ] Ad tech task suite (50 questions) (Day 27)
- [ ] General benchmarks (MMLU, HellaSwag) (Day 27)
- [ ] Execute all benchmarks (Day 28)
- [ ] Build React dashboard (Day 29)
- [ ] PDF report generation (Day 29)
- [ ] Deploy to Vercel + Railway (Day 30)

## Deployment

**Frontend:** Vercel (https://llm-benchmark.vercel.app)
**Backend:** Railway.app
**Status:** Planned (Starting Day 26/30)

## Use Cases

1. **LLM Selection:** Choose optimal model for production use case
2. **Cost Optimization:** Find best quality/price ratio
3. **Performance Tuning:** Identify speed/accuracy tradeoffs
4. **Research:** Contribute to LLM evaluation science

## Key Insights (To Be Documented)

- Which LLM is best for finance vs ad tech?
- Free vs paid models: is GPT-4 worth 30x the cost?
- Speed vs accuracy tradeoffs
- When to use local vs API models?
- Groq vs OpenAI for production?

## License

MIT

## Author

Built as part of 30-day LLM Portfolio Sprint
Day 26-30 | Jan 7-11, 2026

## Acknowledgments

- Ollama for local LLM serving
- Modal for serverless GPU
- HuggingFace for benchmark datasets
