# Research ArXiv Assistant

A multi-agent system that takes a research question, searches ArXiv for relevant academic papers, and generates a structured literature review as a PDF report.

---

## How It Works

The system runs five specialized agents in sequence:

1. **Research Question Agent** — Reformulates the user's input into a precise, academically framed research question and scores its quality
2. **Query Expansion Agent** — Decomposes the question into 4–6 focused subtasks
3. **Search Refinement Agent** — Generates targeted search phrases for each subtask using academic terminology
4. **ArXiv Research Agent** — Queries ArXiv, retrieves papers, and extracts full-text content concurrently
5. **PDF Generation Agent** — Synthesizes the retrieved literature into a formatted report with abstract, body sections, conclusion, and references

Agents communicate over an internal message bus. Each agent runs in its own thread and exposes callable functions. The coordinator handles sequencing and passes outputs between stages.

---

## Requirements

- Python 3.8+
- An [OpenRouter](https://openrouter.ai) API key

### Dependencies

```
requests>=2.28.0
python-dotenv>=0.19.0
feedparser>=6.0.0
pdfminer.six>=20220319
fpdf2>=2.5.0
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
```

---

## Setup

```bash
git clone https://github.com/VasilisVas1/research-arxiv-assistant.git
cd research-arxiv-assistant
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-3.5-turbo-0613
```

`OPENROUTER_MODEL` is optional and defaults to `openai/gpt-3.5-turbo-0613` if not set. Any instruction-following model available on OpenRouter will work.

---

## Usage

### Web Interface

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Enter a research question and the pipeline runs with real-time status updates via WebSocket.

### Python API

```python
from main5 import MultiAgentCoordinator

coordinator = MultiAgentCoordinator()
result = coordinator.execute_research_pipeline(
    query="How does social media use affect adolescent mental health?",
    output_pdf="output.pdf"
)
coordinator.shutdown()
```

### Example Queries

```
What are the ethical challenges in applying AI to clinical diagnosis?
How does online learning compare to traditional classroom instruction in higher education?
What factors drive vaccine hesitancy across different demographic groups?
What are the uses of data science in electric vehicle charging load forecasting?
```

---

## Output

Each run produces a PDF report containing:

- Title page with query and generation date
- Abstract (200–250 words)
- 4–6 literature review sections, one per subtask
- Conclusion and future directions
- Reference list with ArXiv URLs

Report length is typically 10–20 pages depending on topic breadth and the number of papers retrieved.

**Processing time:** 3–8 minutes per query, depending on ArXiv response times and the number of subtasks.

---

## Architecture

```
User Input
    └── Research Question Agent      reformulates and validates the query
            └── Query Expansion Agent        generates subtasks as structured JSON
                    └── Search Refinement Agent      produces search phrases per subtask
                            └── ArXiv Research Agent         fetches and extracts papers (concurrent)
                                    └── PDF Generation Agent         writes the final report
```

Inter-agent communication uses a `MessageBus` with blocking `send_message_and_wait` calls and a 5000-second timeout per stage.

---

## Configuration

All configuration is handled through the `.env` file:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o-mini` | Any model available on OpenRouter |

To adjust how many papers are retrieved per search term, modify `max_results` in `ArxivResearchAgent._process_subtask` in `main5.py`:

```python
papers = self._search_arxiv(search_keyword, max_results=3)
```

---

## Known Limitations

- Only searches ArXiv. Papers outside ArXiv (journals, conference proceedings not deposited there) are not retrieved.
- PDF text extraction can fail on scanned or image-based papers.
- If a subtask fails mid-pipeline, the entire run currently fails. Partial results are not saved.
- No caching. Repeated queries on the same topic re-fetch all papers.

---

## Contributing

Pull requests are welcome. For significant changes, open an issue first to discuss the approach.

---

## License

MIT
