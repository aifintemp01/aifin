# AI Hedge Fund

An advanced AI-powered hedge fund trading system that combines the expertise of legendary investors with cutting-edge artificial intelligence to analyze stocks and make informed investment decisions.

This project implements a sophisticated trading system that emulates the investment strategies of some of the world's most successful investors. It provides both a user-friendly web interface and a powerful command-line interface for advanced users.

## 🎯 Key Features

### AI Agents
The system employs 29 specialized AI agents, each modeled after legendary investors and advanced investment strategies:

#### Legendary Investor Agents
1. **Aswath Damodaran Agent** - The Dean of Valuation, focusing on story, numbers, and disciplined valuation
2. **Ben Graham Agent** - The godfather of value investing, only buys hidden gems with a margin of safety
3. **Bill Ackman Agent** - An activist investor, takes bold positions and pushes for change
4. **Cathie Wood Agent** - The queen of growth investing, believes in innovation and disruption
5. **Charlie Munger Agent** - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. **Michael Burry Agent** - The Big Short contrarian who hunts for deep value
7. **Mohnish Pabrai Agent** - The Dhandho investor, who looks for doubles at low risk
8. **Peter Lynch Agent** - Practical investor who seeks "ten-baggers" in everyday businesses
9. **Phil Fisher Agent** - Meticulous growth investor who uses deep "scuttlebutt" research
10. **Rakesh Jhunjhunwala Agent** - The Big Bull of India
11. **Stanley Druckenmiller Agent** - Macro legend who hunts for asymmetric opportunities with growth potential
12. **Warren Buffett Agent** - The oracle of Omaha, seeks wonderful companies at a fair price

#### Specialized Strategy Agents
13. **Capital Allocation Agent** - Analyzes how companies allocate capital for maximum returns
14. **Factor Composite Agent** - Combines multiple factor strategies for balanced performance
15. **Fundamentals Agent** - Analyzes fundamental data and generates trading signals
16. **Governance Agent** - Evaluates corporate governance and management quality
17. **Growth Agent** - Focuses on identifying high-growth potential companies
18. **Liquidity Agent** - Assesses stock liquidity and market impact
19. **Macro Exposure Agent** - Analyzes macroeconomic factors and market trends
20. **Momentum Agent** - Identifies and capitalizes on market momentum
21. **News Sentiment Agent** - Analyzes news and social media sentiment
22. **Quality Agent** - Evaluates the quality of businesses and earnings
23. **Risk Manager** - Calculates risk metrics and sets position limits
24. **Sentiment Agent** - Analyzes market sentiment and generates trading signals
25. **Technicals Agent** - Analyzes technical indicators and generates trading signals
26. **Valuation Agent** - Calculates intrinsic value of stocks and generates trading signals
27. **Value Agent** - Identifies undervalued stocks with strong fundamentals
28. **Portfolio Manager** - Makes final trading decisions and generates orders

### Interactive Web Interface
- **Visual Flow Builder**: Create and customize trading strategies using a drag-and-drop interface
- **Real-time Analysis**: Get instant insights from AI agents on selected stocks
- **Backtesting**: Test investment strategies against historical data
- **Portfolio Management**: Track and manage your virtual portfolio performance
- **API Integration**: Connect with multiple LLM providers (OpenAI, GROQ, Anthropic, Ollama) and financial data sources

### Command Line Interface
- **Automation Ready**: Run analyses and backtests from the terminal
- **Scriptable**: Integrate with your existing workflows
- **Customizable**: Specify dates, tickers, and AI models

## 🚀 Quick Start

### Option 1: Web Application (Recommended)

The web application provides a user-friendly interface for building and running AI hedge fund strategies.

#### Prerequisites
- [Node.js](https://nodejs.org/) (includes npm)
- [Python 3](https://python.org/)
- [Poetry](https://python-poetry.org/)

#### One-line Setup
```bash
cd app && ./run.sh  # Mac/Linux
# or
cd app && run.bat  # Windows
```

#### Manual Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/virattt/ai-hedge-fund.git
   cd ai-hedge-fund
   ```

2. **Set up API keys**:
   ```bash
   cp .env.example .env
   # Edit .env file to add your API keys
   ```

3. **Install dependencies**:
   ```bash
   cd app/backend && poetry install
   cd ../frontend && npm install
   ```

4. **Start the application**:
   ```bash
   # Backend (terminal 1)
   cd app/backend && poetry run uvicorn main:app --reload
   
   # Frontend (terminal 2)
   cd app/frontend && npm run dev
   ```

Access the application at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Command Line Interface

#### Quick Run
```bash
# Install dependencies
curl -sSL https://install.python-poetry.org | python3 -
poetry install

# Run AI hedge fund analysis
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run backtest
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# Run with local LLMs using Ollama
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

## 📊 Features in Detail

### Stock Analysis
- **Multi-Agent Analysis**: Each AI agent provides unique perspectives on stocks
- **Comprehensive Reports**: Get detailed investment reports with reasoning
- **Trading Signals**: AI-generated buy/sell/hold recommendations
- **Risk Assessment**: Calculate risk metrics and position limits

### Backtesting
- **Historical Data Analysis**: Test strategies against past market data
- **Performance Metrics**: Track returns, volatility, and risk-adjusted performance
- **Visualization**: Generate charts and graphs to visualize strategy performance

### Portfolio Management
- **Virtual Portfolio**: Build and manage a simulated investment portfolio
- **Position Sizing**: AI-determined optimal position sizes based on risk
- **Performance Tracking**: Monitor portfolio performance over time

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **Python**: Core language
- **Poetry**: Dependency management
- **SQLAlchemy**: Database ORM
- **Alembic**: Database migrations
- **Ollama**: Local LLM integration
- **LangChain**: LLM orchestration

### Frontend
- **React 18**: UI library
- **TypeScript**: Type safety
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Component library
- **ReactFlow**: Flow diagram library
- **Recharts**: Data visualization

### APIs and Data Sources
- **OpenAI API**: GPT-4o, GPT-4o-mini
- **GROQ API**: DeepSeek, Llama3
- **Anthropic API**: Claude models
- **Financial Datasets API**: Stock data
- **12Data API**: Market data
- **Ollama**: Local LLM support (Llama3, Mistral, Gemma)

## 📁 Project Structure

```
ai-hedge-fund/
├── src/                     # Core AI hedge fund logic
│   ├── agents/             # AI agent implementations
│   ├── tools/              # Data and API tools
│   ├── main.py             # CLI entry point
│   └── backtester.py       # Backtesting engine
├── app/                     # Web application
│   ├── backend/            # FastAPI backend
│   ├── frontend/           # React/Vite frontend
│   ├── run.sh              # Mac/Linux run script
│   └── run.bat             # Windows run script
├── .env.example            # Environment variables template
├── poetry.lock             # Poetry dependencies lock file
└── pyproject.toml          # Python project configuration
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the `.env.example` template and add your API keys:

```env
# LLM API Keys (required - at least one)
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Financial Data API Keys
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
TWELVE_DATA_API_KEY=your-twelve-data-api-key

# Ollama Configuration
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

**Note**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key.

## 📈 Usage Examples

### Running Analyses

#### Single Stock Analysis
```bash
poetry run python src/main.py --ticker AAPL
```

#### Multiple Stocks
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA,GOOGL,TSLA
```

#### Date Range Analysis
```bash
poetry run python src/main.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-01
```

#### Local LLMs with Ollama
```bash
# First, start Ollama server and pull a model
ollama pull llama3
poetry run python src/main.py --ticker AAPL --ollama
```

### Backtesting Strategies

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2023-01-01 --end-date 2024-01-01
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Commit your changes**: `git commit -am 'Add some feature'`
4. **Push to the branch**: `git push origin feature/YourFeature`
5. **Create a Pull Request**

Please keep your pull requests small and focused for easier review.

## ⚠️ Important Notice

This project is designed to demonstrate advanced AI trading capabilities. While it provides sophisticated analysis and backtesting features:

- **Trading involves risk**: Past performance does not indicate future results
- **Do your own research**: Always conduct independent analysis before making investment decisions
- **Consult professionals**: Consider seeking advice from qualified financial advisors
- **Start small**: If you choose to use this system with real capital, start with small amounts
- **Risk management**: The system includes risk management features, but they are not guarantees

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Backend Documentation](app/backend/README.md)
- [Frontend Documentation](app/frontend/README.md)
- [Issues Tracker](https://github.com/virattt/ai-hedge-fund/issues)
- [Twitter](https://twitter.com/virattt)

## 📞 Support

If you need help or have questions:

1. Check the [Troubleshooting](app/README.md#troubleshooting) section
2. Search existing [Issues](https://github.com/virattt/ai-hedge-fund/issues)
3. Open a new [Issue](https://github.com/virattt/ai-hedge-fund/issues)
4. Follow updates on [Twitter](https://twitter.com/virattt)

---

**Built with ❤️ using cutting-edge AI technology**
