import json
import threading
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
from utils import call_openrouter
import requests
from dotenv import load_dotenv
import re
import feedparser
from pdfminer.high_level import extract_text
from io import BytesIO
from urllib.parse import quote
from fpdf import FPDF
import scholarly
from semanticscholar import SemanticScholar
import urllib.parse

load_dotenv()


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    FUNCTION_CALL = "function_call"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


@dataclass
class Message:
    id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None


class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    COMPLETED = "completed"


class BaseAgent:
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.message_bus = None
        self.function_registry = {}
        self.task_queue = queue.Queue()
        self.running = False
        self.thread = None

    def register_function(self, name: str, func: Callable):
        self.function_registry[name] = func

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_messages)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _process_messages(self):
        while self.running:
            try:
                if not self.task_queue.empty():
                    message = self.task_queue.get(timeout=1)
                    self._handle_message(message)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Agent {self.agent_id} error: {e}")

    def _handle_message(self, message: Message):
        if message.message_type == MessageType.FUNCTION_CALL:
            self._handle_function_call(message)
        elif message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)

    def _handle_function_call(self, message: Message):
        func_name = message.payload.get("function_name")
        args = message.payload.get("args", {})

        if func_name in self.function_registry:
            try:
                self.status = AgentStatus.BUSY
                result = self.function_registry[func_name](**args)
                response = Message(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.TASK_RESPONSE,
                    payload={"result": result, "success": True},
                    timestamp=datetime.now(),
                    correlation_id=message.id
                )
                self.message_bus.send_message(response)
                self.status = AgentStatus.IDLE
            except Exception as e:
                error_response = Message(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={"error": str(e), "success": False},
                    timestamp=datetime.now(),
                    correlation_id=message.id
                )
                self.message_bus.send_message(error_response)
                self.status = AgentStatus.ERROR

    def _handle_task_request(self, message: Message):
        pass

    def call_agent_function(self, target_agent_id: str, function_name: str, **kwargs) -> Any:
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=MessageType.FUNCTION_CALL,
            payload={"function_name": function_name, "args": kwargs},
            timestamp=datetime.now()
        )
        return self.message_bus.send_message_and_wait(message)


class MessageBus:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.pending_responses: Dict[str, queue.Queue] = {}

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent
        agent.message_bus = self

    def send_message(self, message: Message):
        if message.receiver_id in self.agents:
            self.agents[message.receiver_id].task_queue.put(message)

        if message.correlation_id and message.correlation_id in self.pending_responses:
            self.pending_responses[message.correlation_id].put(message)

    def send_message_and_wait(self, message: Message, timeout: int = 300) -> Any:
        response_queue = queue.Queue()
        self.pending_responses[message.id] = response_queue
        self.send_message(message)
        try:
            response = response_queue.get(timeout=timeout)
            if response.message_type == MessageType.ERROR:
                raise Exception(response.payload.get("error", "Unknown error"))
            return response.payload.get("result")
        except queue.Empty:
            print(f"Timed out while waiting for response from {message.receiver_id}")
            raise Exception(f"Timeout waiting for response from {message.receiver_id}")
        finally:
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]


class ThesisStructureAgent(BaseAgent):
    def __init__(self):
        super().__init__("thesis_structure", ["create_thesis_outline", "generate_search_strategy"])
        self.register_function("create_comprehensive_outline", self._create_comprehensive_outline)
        self.register_function("generate_search_keywords", self._generate_search_keywords)

    def _create_comprehensive_outline(self, topic: str) -> Dict[str, Any]:
        """Create a comprehensive thesis outline with chapters and sections"""
        prompt = f"""
        You are a PhD advisor creating a comprehensive thesis outline for the topic: "{topic}"

        Create a detailed thesis structure with:
        1. Abstract outline
        2. Introduction chapter with 3-4 sections
        3. Literature Review chapter with 4-5 thematic sections
        4. Methodology chapter with 3-4 sections
        5. Results/Analysis chapter with 3-4 sections
        6. Discussion chapter with 3-4 sections
        7. Conclusion chapter with 2-3 sections

        For each section, provide:
        - Clear title
        - 2-3 key research questions to address
        - Expected word count (total thesis should be 80,000-100,000 words)

        Return as JSON with this structure:
        {{
            "thesis_title": "Comprehensive title for the thesis",
            "abstract": {{"description": "What the abstract should cover", "word_count": 300}},
            "chapters": [
                {{
                    "chapter_number": 1,
                    "title": "Chapter Title",
                    "word_count": 12000,
                    "sections": [
                        {{
                            "section_title": "Section Title",
                            "research_questions": ["Question 1", "Question 2"],
                            "word_count": 4000
                        }}
                    ]
                }}
            ]
        }}
        """

        response = call_openrouter(prompt).strip()
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback structure if JSON parsing fails
            return {
                "thesis_title": f"Comprehensive Analysis of {topic}",
                "abstract": {"description": "Overview of the research", "word_count": 300},
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "Introduction",
                        "word_count": 12000,
                        "sections": [
                            {"section_title": "Background", "research_questions": ["What is the context?"],
                             "word_count": 4000},
                            {"section_title": "Problem Statement", "research_questions": ["What is the problem?"],
                             "word_count": 4000},
                            {"section_title": "Research Objectives", "research_questions": ["What are the goals?"],
                             "word_count": 4000}
                        ]
                    }
                ]
            }

    def _generate_search_keywords(self, thesis_structure: Dict[str, Any]) -> List[str]:
        """Generate comprehensive search keywords based on thesis structure"""
        prompt = f"""
        Based on this thesis structure, generate 50-60 diverse search keywords/phrases that would help find relevant academic papers.

        Thesis: {thesis_structure['thesis_title']}

        Chapters and Sections:
        {json.dumps(thesis_structure['chapters'], indent=2)}

        Generate keywords that cover:
        1. Core concepts and terminology
        2. Related theories and frameworks
        3. Methodological approaches
        4. Specific applications and case studies
        5. Historical perspectives
        6. Future trends and implications
        7. Interdisciplinary connections

        Return a JSON array of 50-60 search terms, each 2-6 words long, suitable for academic databases.
        Example: ["machine learning applications", "neural network architectures", "deep learning algorithms"]
        """

        response = call_openrouter(prompt).strip()
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback keywords
            return [f"{thesis_structure['thesis_title'].lower()}", "research methodology", "literature review"]


class ComprehensiveResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("comprehensive_research", ["multi_source_search", "pdf_extraction"])
        self.register_function("comprehensive_search", self._comprehensive_search)
        self.register_function("extract_pdf_content", self._extract_pdf_content)

    def _search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search ArXiv for papers"""
        try:
            base_url = 'http://export.arxiv.org/api/query?'
            encoded_query = quote(f'all:{query}')
            search_query = f'search_query={encoded_query}&start=0&max_results={max_results}'
            feed = feedparser.parse(base_url + search_query)

            results = []
            for entry in feed.entries:
                pdf_url = next((link.href for link in entry.links if link.type == 'application/pdf'), None)
                results.append({
                    "title": entry.title,
                    "summary": entry.summary,
                    "pdf_url": pdf_url,
                    "authors": [author.name for author in entry.authors],
                    "published": entry.published,
                    "source": "ArXiv",
                    "doi": entry.get('arxiv_doi', ''),
                    "categories": entry.get('tags', [])
                })
            return results
        except Exception as e:
            print(f"ArXiv search error for '{query}': {e}")
            return []

    def _search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        try:
            sch = SemanticScholar()
            results = sch.search_paper(query, limit=max_results)

            papers = []
            for paper in results:
                papers.append({
                    "title": paper.title,
                    "summary": paper.abstract or "",
                    "pdf_url": paper.url,
                    "authors": [author.name for author in paper.authors] if paper.authors else [],
                    "published": str(paper.year) if paper.year else "",
                    "source": "Semantic Scholar",
                    "doi": paper.doi or "",
                    "citation_count": paper.citationCount or 0,
                    "categories": paper.fieldsOfStudy or []
                })
            return papers
        except Exception as e:
            print(f"Semantic Scholar search error for '{query}': {e}")
            return []

    def _search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Google Scholar for papers"""
        try:
            search_query = scholarly.search_pubs(query)
            papers = []

            for i, paper in enumerate(search_query):
                if i >= max_results:
                    break

                papers.append({
                    "title": paper.get('title', ''),
                    "summary": paper.get('abstract', ''),
                    "pdf_url": paper.get('eprint_url', ''),
                    "authors": paper.get('author', []),
                    "published": str(paper.get('year', '')),
                    "source": "Google Scholar",
                    "doi": paper.get('doi', ''),
                    "citation_count": paper.get('num_citations', 0),
                    "categories": []
                })
            return papers
        except Exception as e:
            print(f"Google Scholar search error for '{query}': {e}")
            return []

    def _comprehensive_search(self, keywords: List[str], target_papers: int = 100) -> List[Dict]:
        """Perform comprehensive search across multiple sources"""
        all_papers = []
        papers_per_keyword = max(3, target_papers // len(keywords))

        print(f"Searching for {target_papers} papers across {len(keywords)} keywords...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for keyword in keywords[:20]:  # Limit to first 20 keywords to avoid rate limits
                # Search ArXiv
                futures.append(executor.submit(self._search_arxiv, keyword, papers_per_keyword // 3))
                # Search Semantic Scholar
                futures.append(executor.submit(self._search_semantic_scholar, keyword, papers_per_keyword // 3))
                # Search Google Scholar
                futures.append(executor.submit(self._search_google_scholar, keyword, papers_per_keyword // 3))

            for future in as_completed(futures):
                try:
                    papers = future.result(timeout=60)
                    all_papers.extend(papers)
                except Exception as e:
                    print(f"Search future failed: {e}")

        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()

        for paper in all_papers:
            title_lower = paper['title'].lower().strip()
            if title_lower not in seen_titles and len(title_lower) > 10:
                seen_titles.add(title_lower)
                unique_papers.append(paper)

        print(f"Found {len(unique_papers)} unique papers")
        return unique_papers[:target_papers]

    def _extract_pdf_content(self, pdf_url: str) -> str:
        """Extract text content from PDF"""
        try:
            if not pdf_url or pdf_url == "":
                return "No PDF URL available"

            print(f"Extracting content from: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)

            if response.status_code != 200:
                return f"Failed to download PDF (status {response.status_code})"

            text = extract_text(BytesIO(response.content))
            return text[:10000]  # Return first 10k characters

        except Exception as e:
            return f"Error extracting PDF content: {str(e)}"


class ThesisWritingAgent(BaseAgent):
    def __init__(self):
        super().__init__("thesis_writing", ["write_chapters", "create_bibliography"])
        self.register_function("write_full_thesis", self._write_full_thesis)
        self.register_function("generate_bibliography", self._generate_bibliography)

    def _write_chapter(self, chapter_info: Dict, papers: List[Dict], thesis_context: str) -> str:
        """Write a single chapter based on the outline and available papers"""
        chapter_title = chapter_info['title']
        sections = chapter_info['sections']
        target_words = chapter_info['word_count']

        # Filter papers relevant to this chapter
        relevant_papers = []
        for paper in papers:
            paper_text = f"{paper['title']} {paper['summary']} {paper.get('full_text', '')}"
            if any(keyword.lower() in paper_text.lower() for keyword in chapter_title.split()):
                relevant_papers.append(paper)

        chapter_content = f"# Chapter {chapter_info['chapter_number']}: {chapter_title}\n\n"

        for section in sections:
            section_title = section['section_title']
            research_questions = section['research_questions']
            section_words = section['word_count']

            prompt = f"""
            You are writing a section for a PhD thesis on "{thesis_context}".

            Chapter: {chapter_title}
            Section: {section_title}
            Target word count: {section_words}

            Research questions to address:
            {chr(10).join(f"- {q}" for q in research_questions)}

            Available research papers for reference:
            {chr(10).join(f"- {p['title']} by {', '.join(p['authors'][:3])} ({p['published'][:4]})" for p in relevant_papers[:10])}

            Write a comprehensive academic section that:
            1. Addresses all research questions thoroughly
            2. Synthesizes information from multiple sources
            3. Uses proper academic writing style
            4. Includes in-text citations in (Author, Year) format
            5. Maintains logical flow and coherence
            6. Reaches approximately {section_words} words

            Focus on depth, critical analysis, and original insights.
            """

            section_content = call_openrouter(prompt)
            chapter_content += f"## {section_title}\n\n{section_content}\n\n"

        return chapter_content

    def _write_full_thesis(self, thesis_structure: Dict, papers: List[Dict]) -> str:
        """Write the complete thesis"""
        thesis_title = thesis_structure['thesis_title']

        # Start with title page and abstract
        thesis_content = f"""# {thesis_title}

## Abstract

{self._write_abstract(thesis_structure, papers)}

## Table of Contents

"""

        # Add table of contents
        for chapter in thesis_structure['chapters']:
            thesis_content += f"{chapter['chapter_number']}. {chapter['title']}\n"
            for section in chapter['sections']:
                thesis_content += f"   {section['section_title']}\n"

        thesis_content += "\n---\n\n"

        # Write each chapter
        for chapter in thesis_structure['chapters']:
            print(f"Writing Chapter {chapter['chapter_number']}: {chapter['title']}")
            chapter_content = self._write_chapter(chapter, papers, thesis_title)
            thesis_content += chapter_content + "\n---\n\n"

        # Add bibliography
        thesis_content += "## Bibliography\n\n"
        thesis_content += self._generate_bibliography(papers)

        return thesis_content

    def _write_abstract(self, thesis_structure: Dict, papers: List[Dict]) -> str:
        """Write the thesis abstract"""
        prompt = f"""
        Write a comprehensive PhD thesis abstract (300-400 words) for:

        Title: {thesis_structure['thesis_title']}

        Chapter outline:
        {json.dumps([{'title': ch['title'], 'sections': [s['section_title'] for s in ch['sections']]} for ch in thesis_structure['chapters']], indent=2)}

        The abstract should:
        1. Introduce the research problem and its significance
        2. State the research objectives and questions
        3. Briefly describe the methodology
        4. Summarize key findings and contributions
        5. Discuss implications and future work

        Write in formal academic style appropriate for a PhD thesis.
        """

        return call_openrouter(prompt)

    def _generate_bibliography(self, papers: List[Dict]) -> str:
        """Generate formatted bibliography"""
        bibliography = []

        for paper in papers:
            if not paper.get('title'):
                continue

            authors = ', '.join(paper.get('authors', ['Unknown Author'])[:5])
            year = paper.get('published', 'Unknown')[:4] if paper.get('published') else 'Unknown'
            title = paper.get('title', 'Unknown Title')
            source = paper.get('source', 'Unknown Source')

            if paper.get('doi'):
                citation = f"{authors} ({year}). {title}. {source}. doi:{paper['doi']}"
            elif paper.get('pdf_url'):
                citation = f"{authors} ({year}). {title}. {source}. Retrieved from {paper['pdf_url']}"
            else:
                citation = f"{authors} ({year}). {title}. {source}."

            bibliography.append(citation)

        return '\n\n'.join(sorted(bibliography))


class ThesisPDFGenerator(BaseAgent):
    def __init__(self):
        super().__init__("thesis_pdf", ["generate_thesis_pdf"])
        self.register_function("create_thesis_pdf", self._create_thesis_pdf)

    def _create_thesis_pdf(self, thesis_content: str, output_file: str = "PhD_Thesis.pdf") -> str:
        """Generate a professional PDF thesis"""

        class ThesisPDF(FPDF):
            def __init__(self, title):
                super().__init__()
                self.title = title
                self.chapter_count = 0
                self.set_auto_page_break(auto=True, margin=15)
                try:
                    self.add_font("DejaVu", "", "DejaVuSans-ExtraLight.ttf", uni=True)
                    self.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
                    self.add_font("DejaVu", "I", "DejaVuSansCondensed-Oblique.ttf", uni=True)
                except:
                    pass
                self.set_font("DejaVu", "", 12)

            def header(self):
                if self.page_no() > 1:  # Skip header on first page
                    self.set_font('DejaVu', 'I', 8)
                    self.cell(0, 10, self.title, 0, 0, 'L')
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 1, 'R')
                    self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('DejaVu', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            def chapter_title(self, title):
                self.add_page()
                self.set_font('DejaVu', 'B', 16)
                self.cell(0, 15, title, 0, 1, 'L')
                self.ln(5)

            def section_title(self, title):
                self.set_font('DejaVu', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(3)

            def body_text(self, text):
                self.set_font('DejaVu', '', 11)
                # Clean and split text into paragraphs
                paragraphs = text.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        self.multi_cell(0, 6, paragraph.strip())
                        self.ln(3)

        # Extract title from content
        title_match = re.search(r'^# (.+)$', thesis_content, re.MULTILINE)
        title = title_match.group(1) if title_match else "PhD Thesis"

        pdf = ThesisPDF(title)

        # Split content into sections
        sections = thesis_content.split('\n## ')

        # Title page
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 24)
        pdf.cell(0, 30, title, 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('DejaVu', '', 16)
        pdf.cell(0, 10, 'A PhD Thesis', 0, 1, 'C')
        pdf.ln(10)
        pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')

        # Process each section
        for section in sections[1:]:  # Skip first empty section
            lines = section.split('\n')
            section_title = lines[0]
            content = '\n'.join(lines[1:])

            if section_title.startswith('Chapter ') or section_title in ['Abstract', 'Bibliography']:
                pdf.chapter_title(section_title)
            else:
                pdf.section_title(section_title)

            # Process subsections
            if '### ' in content:
                subsections = content.split('### ')
                pdf.body_text(subsections[0])

                for subsection in subsections[1:]:
                    sub_lines = subsection.split('\n')
                    sub_title = sub_lines[0]
                    sub_content = '\n'.join(sub_lines[1:])

                    pdf.section_title(sub_title)
                    pdf.body_text(sub_content)
            else:
                pdf.body_text(content)

        pdf.output(output_file)
        return f"PhD Thesis PDF generated: {output_file}"


class ThesisCoordinator:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        agents = [
            ThesisStructureAgent(),
            ComprehensiveResearchAgent(),
            ThesisWritingAgent(),
            ThesisPDFGenerator()
        ]

        for agent in agents:
            self.agents[agent.agent_id] = agent
            self.message_bus.register_agent(agent)
            agent.start()

    def conduct_thesis(self, topic: str, target_papers: int = 100, output_file: str = "PhD_Thesis.pdf") -> str:
        """Main thesis generation pipeline"""
        try:
            print(f"🎓 Starting PhD thesis generation for: {topic}")
            print(f"📊 Target papers: {target_papers}")

            # Step 1: Create thesis structure
            print("\n📋 Step 1: Creating comprehensive thesis outline...")
            thesis_structure = self.agents["thesis_structure"].call_agent_function(
                "thesis_structure", "create_comprehensive_outline", topic=topic
            )
            print(f"✅ Thesis outline created: {thesis_structure['thesis_title']}")
            print(f"📚 Total chapters: {len(thesis_structure['chapters'])}")

            # Step 2: Generate search keywords
            print("\n🔍 Step 2: Generating comprehensive search strategy...")
            keywords = self.agents["thesis_structure"].call_agent_function(
                "thesis_structure", "generate_search_keywords", thesis_structure=thesis_structure
            )
            print(f"✅ Generated {len(keywords)} search keywords")

            # Step 3: Comprehensive research
            print(f"\n📖 Step 3: Conducting comprehensive literature search...")
            print("⏳ This may take 10-20 minutes depending on the number of papers...")
            papers = self.agents["comprehensive_research"].call_agent_function(
                "comprehensive_research", "comprehensive_search",
                keywords=keywords, target_papers=target_papers
            )
            print(f"✅ Found {len(papers)} relevant papers")

            # Step 4: Extract PDF content for key papers
            print("\n📄 Step 4: Extracting content from key papers...")
            papers_with_content = []
            for i, paper in enumerate(papers[:50]):  # Limit to top 50 papers for content extraction
                if paper.get('pdf_url'):
                    content = self.agents["comprehensive_research"].call_agent_function(
                        "comprehensive_research", "extract_pdf_content", pdf_url=paper['pdf_url']
                    )
                    paper['full_text'] = content
                papers_with_content.append(paper)
                if i % 10 == 0:
                    print(f"📄 Processed {i + 1}/{min(50, len(papers))} papers")

            # Step 5: Write thesis
            print("\n✍️ Step 5: Writing comprehensive thesis...")
            print("⏳ This may take 15-30 minutes for a full thesis...")
            thesis_content = self.agents["thesis_writing"].call_agent_function(
                "thesis_writing", "write_full_thesis",
                thesis_structure=thesis_structure, papers=papers_with_content
            )
            print("✅ Thesis content generated")

            # Step 6: Generate PDF
            print("\n📄 Step 6: Generating professional PDF...")
            pdf_result = self.agents["thesis_pdf"].call_agent_function(
                "thesis_pdf", "create_thesis_pdf",
                thesis_content=thesis_content, output_file=output_file
            )

            print(f"\n🎉 Thesis generation completed successfully!")
            print(f"📊 Statistics:")
            print(f"   - Total papers found: {len(papers)}")
            print(f"   - Papers with extracted content: {len(papers_with_content)}")
            print(f"   - Total chapters: {len(thesis_structure['chapters'])}")
            print(f"   - Output file: {output_file}")

            return pdf_result

        except Exception as e:
            print(f"❌ Thesis generation failed: {e}")
            raise

    def shutdown(self):
        """Shutdown all agents"""
        for agent in self.agents.values():
            agent.stop()


# Usage example:
if __name__ == "__main__":
    coordinator = ThesisCoordinator()

    # Example usage
    topic = "TikTok as a global cultural transmission tool among Gen Z."
    result = coordinator.conduct_thesis(
        topic=topic,
        target_papers=100,
        output_file="TikTok.pdf"
    )

    print(f"\n{result}")
    coordinator.shutdown()