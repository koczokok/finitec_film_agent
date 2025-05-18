# 🎬 Film Agent

A powerful AI-powered tool for extracting and analyzing film financial data from various sources including local markdown files, Wikipedia, and web searches.

## ✨ Features

- **Multi-source Data Retrieval**: Fetch film financial data from local files, Wikipedia, and web searches
- **Intelligent Search**: Automatically finds the most relevant financial information
- **CLI Interface**: Easy-to-use command line interface for quick lookups
- **Web UI**: Streamlit-based web interface for interactive use
- **Smart Fallback**: Automatically tries different data sources for maximum reliability

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Film
   ```

2. **Set up a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your API keys (Brave Search API, etc.) to the `.env` file

## 🏃‍♂️ Running the Application

### Command Line Interface
```bash
python -m src.cli
```

### Web Interface
```bash
streamlit run src/streamlit_ui.py
```

## 🎯 Usage

The Film Agent can retrieve financial information about films from multiple sources:

1. **Local Database**: Checks pre-populated markdown files in the `movies/` directory
2. **Wikipedia**: Fetches data from Wikipedia pages
3. **Web Search**: Performs web searches as a fallback

## 📊 Agent Output

The agent provides detailed financial information about films in a structured format. Here's what you can expect in the output:

### Standard Output Format
```
Film: [Film Title] (Year)
Budget: $[Amount] million
Profit: $[Amount] million
Data Source: [Data Source]
```



### Error Handling
- When data is incomplete or inconsistent, the agent will mark it as 0
- The agent will automatically try alternative sources if the primary source doesn't have the requested information

## 📂 Project Structure

```
├── movies/               # Pre-populated film data in markdown format
├── src/
│   ├── agent.py         # Core agent logic and data processing
│   ├── cli.py           # Command line interface
│   └── streamlit_ui.py  # Web interface
├── .env.example         # Example environment variables
└── requirements.txt     # Python dependencies
```


