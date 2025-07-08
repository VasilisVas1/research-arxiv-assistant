# 🤖 Multi-Agent Research System

> **Transform any research question into a comprehensive academic report in minutes!**

An intelligent multi-agent system that automatically conducts literature reviews, searches academic papers, and generates professional PDF reports. Perfect for researchers, students, and academics who need quick, thorough research on any topic.

## 🚀 What Makes This Special?

- **🎯 Smart Question Refinement**: Automatically transforms casual queries into well-structured research questions
- **🔍 Intelligent Paper Discovery**: Searches and retrieves relevant academic papers from ArXiv
- **📊 Multi-Agent Coordination**: Specialized AI agents work together seamlessly
- **📄 Professional PDF Output**: Generates publication-ready research reports
- **🎨 Zero Configuration**: Just add your API key and run!

## 🎬 Demo

```python
from main import MultiAgentCoordinator

# Initialize the system
coordinator = MultiAgentCoordinator()

# Ask any research question
result = coordinator.execute_research_pipeline(
    "How does social media affect mental health?",
    output_pdf="social_media_research.pdf"
)

# Get a comprehensive 15-page academic report!
```

## 📋 Features

### 🧠 Intelligent Agent Architecture
- **Research Question Agent**: Formulates and validates research questions
- **Query Expansion Agent**: Breaks down complex topics into manageable subtasks
- **Search Refinement Agent**: Optimizes search terms for academic databases
- **ArXiv Research Agent**: Retrieves and processes academic papers
- **PDF Generation Agent**: Creates professional research reports

### 🔧 Advanced Capabilities
- **Concurrent Processing**: Multi-threaded paper retrieval and analysis
- **Quality Validation**: Automatic research question quality assessment
- **Smart Citations**: Proper academic citation formatting
- **Error Handling**: Robust error recovery and graceful degradation
- **Progress Tracking**: Real-time pipeline status updates

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key (for AI model access)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-agent-research-system.git
   cd multi-agent-research-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Run your first research**
   ```bash
   python example.py
   ```

## 📦 Dependencies

```txt
requests>=2.28.0
python-dotenv>=0.19.0
feedparser>=6.0.0
pdfminer.six>=20220319
fpdf2>=2.5.0
```

## 🎯 Usage Examples

### Basic Research Query
```python
from main import MultiAgentCoordinator

coordinator = MultiAgentCoordinator()

# Simple question
result = coordinator.execute_research_pipeline(
    "What are the effects of climate change on agriculture?"
)
print(result)  # "Enhanced PDF generated: Research_Report.pdf"
```

### Advanced Usage with Custom Output
```python
# More specific research with custom filename
result = coordinator.execute_research_pipeline(
    "How does artificial intelligence impact healthcare diagnostics?",
    output_pdf="AI_Healthcare_Review.pdf"
)

# The system will:
# 1. Refine your question for academic research
# 2. Break it into 4-6 focused subtasks
# 3. Search ArXiv for relevant papers
# 4. Generate a comprehensive literature review
```

### Sample Research Questions
Try these example queries:
- "How does social media use affect adolescent mental health?"
- "What factors influence vaccine hesitancy in different communities?"
- "How does online learning compare to traditional classroom instruction?"
- "What are the environmental impacts of renewable energy adoption?"

## 📊 System Architecture

```
User Query → Research Question Agent → Query Expansion Agent → Search Refinement Agent → ArXiv Research Agent → PDF Generation Agent → Final Report
```

### Agent Responsibilities

| Agent | Function | Output |
|-------|----------|--------|
| 🎯 **Research Question Agent** | Validates and refines user input | Structured research question |
| 🔍 **Query Expansion Agent** | Breaks down into subtasks | JSON research plan |
| 🔧 **Search Refinement Agent** | Optimizes search terms | Enhanced search queries |
| 📚 **ArXiv Research Agent** | Retrieves academic papers | Structured paper data |
| 📄 **PDF Generation Agent** | Creates final report | Professional PDF document |

## 🎨 Output Example

The system generates a professional academic report with:
- **Title Page**: Research question and metadata
- **Abstract**: Comprehensive overview (200-250 words)
- **Literature Review Sections**: 4-6 focused analysis sections
- **Conclusion**: Synthesis and future directions
- **References**: Properly formatted citations

## 🔧 Configuration

### API Setup
Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai) and add it to your `.env` file.

### Customization Options
```python
# Modify search parameters
coordinator.agents["arxiv_research"].call_agent_function(
    "arxiv_research", "search_arxiv", 
    query="your query", 
    max_results=10  # Default: 5
)

# Custom PDF styling
# Edit the PDF class in PDFGenerationAgent for custom formatting
```

## 📈 Performance

- **Processing Time**: 3-5 minutes per research query
- **Paper Retrieval**: Up to 15 papers per subtask
- **Report Length**: 10-20 pages depending on topic complexity
- **Concurrent Processing**: 2 parallel threads for paper retrieval

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs**: Open an issue with detailed reproduction steps
2. **💡 Suggest Features**: Share your ideas for new functionality
3. **📝 Improve Documentation**: Help make the docs even better
4. **🔧 Submit Pull Requests**: Fix bugs or add new features

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/multi-agent-research-system.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push and create a pull request
git push origin feature/amazing-feature
```

## 🐛 Troubleshooting

### Common Issues

**"API Error: Missing 'choices' in response"**
- Check your OpenRouter API key is valid
- Verify your account has sufficient credits

**"Failed to download PDF"**
- Some ArXiv papers may have access restrictions
- The system will continue with available papers

**"Timeout waiting for response"**
- Increase timeout in `send_message_and_wait()` method
- Check your internet connection

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] **Google Scholar Integration**: Expand beyond ArXiv
- [ ] **Citation Analysis**: Advanced citation network analysis
- [ ] **Interactive Web Interface**: Browser-based research tool
- [ ] **Collaboration Features**: Team research capabilities
- [ ] **Export Formats**: Word, LaTeX, and HTML output
- [ ] **Language Support**: Multi-language research capabilities


## 🏆 Acknowledgments

- OpenRouter for providing AI model access
- ArXiv for open academic paper access
- The open-source community for invaluable tools and libraries

---

**⭐ If this project helped you, please give it a star! It helps others discover this tool and motivates continued development.**

Made with ❤️ by [Vasilis Vasiliou](https://vasilisvasileiou.com/)
