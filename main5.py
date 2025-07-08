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

load_dotenv()

def format_plan_to_json(raw_text, query):
    lines = raw_text.strip().split("\n")
    subtasks = []
    current_task = None

    for line in lines:
        task_match = re.match(r'^\d+\.\s+\*\*(.+?)\*\*', line)
        if not task_match:
            task_match = re.match(r'^\d+\.\s+(.+)', line)
        bullet_match = re.match(r'^\s*-\s+(.*)', line)
        if task_match:
            if current_task and current_task["details"]:
                subtasks.append(current_task)
            current_task = {"title": task_match.group(1).strip(), "details": []}
        elif bullet_match and current_task:
            current_task["details"].append(bullet_match.group(1).strip())
    if current_task and current_task["details"]:
        subtasks.append(current_task)

    return json.dumps({"query": query, "subtasks": subtasks}, indent=2)


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

    def send_message_and_wait(self, message: Message, timeout: int = 5000) -> Any:
        response_queue = queue.Queue()
        self.pending_responses[message.id] = response_queue
        self.send_message(message)
        try:
            response = response_queue.get(timeout=timeout)
            if response.message_type == MessageType.ERROR:
                raise Exception(response.payload.get("error", "Unknown error"))
            return response.payload.get("result")
        except queue.Empty:
            print(f"Timed out while waiting for response from {message.receiver_id} for function {message.payload.get('function_name', 'unknown')}")
            raise Exception(f"Timeout waiting for response from {message.receiver_id}")
        finally:
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]


class ResearchQuestionAgent(BaseAgent):
    def __init__(self):
        super().__init__("research_question", ["formulate_research_question", "validate_question"])
        self.register_function("formulate_research_question", self._formulate_research_question)
        self.register_function("validate_and_improve", self._validate_and_improve)
        self.register_function("process_user_input", self._process_user_input)

    def _formulate_research_question(self, user_input: str) -> str:
        prompt = f"""
        You are a research methodology expert. Your task is to transform a user's casual question or topic into a well-structured research question that will yield comprehensive academic results.

        User Input: "{user_input}"

        Guidelines for creating a good research question:
        1. Make it specific and focused rather than too broad
        2. Ensure it's researchable with academic sources
        3. Frame it to encourage comprehensive analysis
        4. Include key concepts that will help find relevant literature
        5. Make it neither too narrow (insufficient sources) nor too broad (unfocused)
        6. Consider multiple perspectives or dimensions of the topic

        Transform the user input into ONE clear, well-structured research question that will produce the best academic research results.

        Return ONLY the research question, no explanation or additional text.
        """
        return call_openrouter(prompt).strip()

    def _validate_and_improve(self, research_question: str) -> Dict[str, Any]:
        validation_prompt = f"""
        You are a research quality assessor. Evaluate this research question and provide feedback.

        Research Question: "{research_question}"

        Assess the question on these criteria:
        1. Clarity - Is it clear and unambiguous?
        2. Specificity - Is it focused enough to be manageable?
        3. Researchability - Can it be answered with academic sources?
        4. Scope - Is the scope appropriate (not too broad or narrow)?
        5. Academic value - Will it produce meaningful scholarly insights?

        Provide your response in this exact JSON format:
        {{
            "original_question": "{research_question}",
            "quality_score": [1-10],
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "improved_question": "improved version if needed, or same if good",
            "reasoning": "brief explanation of changes made"
        }}
        """
        response = call_openrouter(validation_prompt).strip()
        try:
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "original_question": research_question,
                "quality_score": 7,
                "strengths": ["Acceptable research question"],
                "weaknesses": ["Could not validate properly"],
                "improved_question": research_question,
                "reasoning": "Validation failed, using original question"
            }

    def _process_user_input(self, user_input: str) -> Dict[str, Any]:
        print(f"Processing user input: {user_input}")
        initial_question = self._formulate_research_question(user_input)
        print(f"Initial research question: {initial_question}")
        validation_result = self._validate_and_improve(initial_question)
        print(f"Quality score: {validation_result['quality_score']}/10")
        if validation_result['quality_score'] < 7:
            print("Question improved based on validation feedback")
            final_question = validation_result['improved_question']
        else:
            final_question = initial_question
        return {
            "user_input": user_input,
            "initial_question": initial_question,
            "final_question": final_question,
            "validation": validation_result,
            "processing_notes": f"Transformed user input into research question with quality score {validation_result['quality_score']}/10"
        }

class QueryExpansionAgent(BaseAgent):
    def __init__(self):
        super().__init__("query_expansion", ["expand_query", "break_down_research", "format_to_json"])
        self.register_function("expand_query", self._expand_query)
        self.register_function("expand_and_format", self._expand_and_format)

    def _expand_query(self, query: str) -> str:
        prompt = f"""
        You are a research planner AI. Given a research question, your job is to break it down into a structured list of subtasks that a team of AI agents can perform to create a comprehensive academic report.

        Each subtask should represent a distinct aspect or dimension of the research question. Create 4-6 subtasks that together will provide a complete understanding of the topic.

        For each subtask:
        - Give it a clear, specific title that indicates what aspect will be explored
        - Include 3-4 bullet points that describe the specific research questions or areas to investigate
        - Make sure the subtasks are complementary and don't overlap significantly
        - Focus on creating subtasks that will lead to finding relevant academic papers

        Research Question: "{query}"

        Return format:
        1. **Subtask Title**
           - Specific research question 1
           - Specific research question 2
           - Specific research question 3
        2. **Another Subtask Title**
           - Specific research question 1
           - Specific research question 2
           - Specific research question 3
        ...
        """
        return call_openrouter(prompt)

    def _expand_and_format(self, query: str) -> str:
        raw_text = self._expand_query(query)
        return format_plan_to_json(raw_text, query)

class SearchRefinementAgent(BaseAgent):
    def __init__(self):
        super().__init__("search_refinement", ["refine_search_terms", "format_json"])
        self.register_function("refine_detail", self._refine_detail)
        self.register_function("reformat_json_with_search_terms", self._reformat_json_with_search_terms)

    def _refine_detail(self, detail: str, main_query: str, subtask_title: str) -> str:
        prompt = f"""
        You are an expert at creating targeted academic search queries. Your task is to create a precise search phrase that will find relevant academic papers.

        CONTEXT:
        - Main Research Question: "{main_query}"
        - Current Subtask Focus: "{subtask_title}"  
        - Specific Detail to Search: "{detail}"

        Create a search phrase that:
        1. Combines key concepts from the main question and specific detail
        2. Uses academic terminology appropriate for scholarly databases
        3. Is specific enough to find relevant papers but broad enough to get results
        4. Focuses on the intersection of the main topic and the specific detail

        Examples of good search phrases:
        - "voter turnout demographic factors United States"
        - "political participation socioeconomic status non-voting"
        - "electoral engagement barriers minority communities"

        Return ONLY the final search phrase with no quotes, explanations, or extra text.
        """

        output = call_openrouter(prompt).strip()
        output = output.replace('"', '').replace("'", "")
        output = output.splitlines()[-1].strip()
        return output

    def _reformat_json_with_search_terms(self, original_json: str) -> str:
        parsed = json.loads(original_json)
        main_query = parsed.get("query", "")
        refined_subtasks = []
        for subtask in parsed.get("subtasks", []):
            subtask_title = subtask["title"]
            refined_details = []
            for detail in subtask.get("details", []):
                try:
                    refined = self._refine_detail(detail, main_query, subtask_title)
                    refined_details.append(refined)
                except Exception as e:
                    print(f"Failed to refine: {detail}\nReason: {e}")
                    refined_details.append(detail)
            refined_subtasks.append({
                "title": subtask["title"],
                "details": subtask["details"],
                "search_keywords": refined_details
            })
        return json.dumps({
            "query": parsed.get("query", ""),
            "subtasks": refined_subtasks
        }, indent=2)


class ArxivResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("arxiv_research", ["search_papers", "extract_content"])
        self.register_function("search_arxiv", self._search_arxiv)
        self.register_function("extract_pdf_text", self._extract_pdf_text)
        self.register_function("arxiv_agent", self._arxiv_agent)

    def _search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
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
                "published": entry.published
            })
        return results

    def _extract_pdf_text(self, pdf_url: str) -> str:
        try:
            print(f"Downloading PDF: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                return f"Failed to download PDF (status {response.status_code})"
            print(f"Extracting text from PDF...")
            text = extract_text(BytesIO(response.content))
            return text[:5000]
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def _arxiv_agent(self, json_input: Dict) -> Dict:
        print(f"ArXiv agent processing {len(json_input.get('subtasks', []))} subtasks")
        output = {
            "query": json_input["query"],
            "subtask_analysis": []
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_subtask = {}
            for subtask in json_input["subtasks"]:
                future = executor.submit(self._process_subtask, subtask)
                future_to_subtask[future] = subtask
            for future in as_completed(future_to_subtask):
                subtask = future_to_subtask[future]
                try:
                    result = future.result(timeout=180)
                    output["subtask_analysis"].append(result)
                    print(f"Completed subtask: {subtask['title']}")
                except Exception as e:
                    print(f"Error processing subtask {subtask['title']}: {e}")
                    error_result = {
                        "title": subtask["title"],
                        "retrieved_documents": [{
                            "error": f"Failed to process subtask: {str(e)}"
                        }]
                    }
                    output["subtask_analysis"].append(error_result)
        return output

    def _process_subtask(self, subtask: Dict) -> Dict:
        print(f"Processing subtask: {subtask['title']}")
        subtask_result = {
            "title": subtask["title"],
            "retrieved_documents": []
        }
        search_keywords = subtask.get("search_keywords", [])[:3]
        for search_keyword in search_keywords:
            try:
                print(f"Searching ArXiv for: {search_keyword}")
                papers = self._search_arxiv(search_keyword, max_results=3)
                for paper in papers:
                    try:
                        full_text = self._extract_pdf_text(paper["pdf_url"])
                    except Exception as e:
                        full_text = f"Error extracting text: {str(e)}"
                    subtask_result["retrieved_documents"].append({
                        "query": search_keyword,
                        "paper_title": paper["title"],
                        "pdf_url": paper["pdf_url"],
                        "authors": paper["authors"],
                        "published": paper["published"],
                        "summary": paper["summary"],
                        "full_text": full_text
                    })
            except Exception as e:
                print(f"Error searching for '{search_keyword}': {e}")
                subtask_result["retrieved_documents"].append({
                    "query": search_keyword,
                    "error": str(e)
                })
        return subtask_result

class PDFGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("pdf_generation", ["generate_report", "create_pdf"])
        self.register_function("generate_research_paper", self._generate_research_paper)

    def _generate_research_paper(self, plan: Dict, results: Dict, output_pdf: str = "Research_Report.pdf") -> str:
        class PDF(FPDF):
            def __init__(self, title):
                super().__init__()
                self.title = title
                try:
                    self.add_font("DejaVu", "", "DejaVuSans-ExtraLight.ttf", uni=True)
                    self.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
                    self.add_font("DejaVu", "I", "DejaVuSansCondensed-Oblique.ttf", uni=True)
                except:
                    pass
                self.set_font("DejaVu", "", 12)

            def footer(self):
                self.set_y(-15)
                self.set_font("DejaVu", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

            def chapter_title(self, num, label):
                self.set_font("DejaVu", "B", 14)
                self.cell(0, 15, f"{num}. {label}", 0, 1)
                self.ln(3)

            def chapter_body(self, body):
                self.set_font("DejaVu", "", 11)
                paragraphs = body.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        self.multi_cell(0, 6, paragraph.strip())
                        self.ln(3)

            def add_section(self, num, title, content):
                self.chapter_title(num, title)
                self.chapter_body(content)

            def add_abstract(self, abstract_text):
                self.set_font("DejaVu", "B", 12)
                self.cell(0, 10, "Abstract", 0, 1)
                self.ln(2)
                self.set_font("DejaVu", "", 11)
                self.multi_cell(0, 6, abstract_text)
                self.ln(5)

            def add_reference_list(self, references):
                self.add_page()
                self.set_font("DejaVu", "B", 14)
                self.cell(0, 15, "References", 0, 1)
                self.ln(5)
                self.set_font("DejaVu", "", 10)
                for i, ref in enumerate(references, 1):
                    self.multi_cell(0, 6, f"[{i}] {ref}")
                    self.ln(2)

        topic = plan["query"]
        pdf = PDF(title=topic)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 18)
        pdf.multi_cell(0, 12, topic, align="C")
        pdf.ln(15)
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 8, "A Comprehensive Literature Review", ln=True, align="C")
        pdf.ln(5)
        pdf.cell(0, 8, "Generated by Multi-Agent Research System", ln=True, align="C")
        pdf.cell(0, 8, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
        pdf.add_page()
        abstract_prompt = f"""
        Based on the research topic "{topic}" and the following subtask areas, write a comprehensive abstract (200-250 words) that:
        1. Introduces the research question and its importance
        2. Briefly describes the scope of the literature review
        3. Summarizes the key areas of investigation
        4. States the main contributions of this review

        Subtask areas to be covered:
        {json.dumps([subtask['title'] for subtask in plan['subtasks']], indent=2)}

        Write in formal academic style with proper flow and coherence.
        """
        abstract = call_openrouter(abstract_prompt)
        pdf.add_abstract(abstract)
        references = []
        section_summaries = []
        for i, subtask in enumerate(plan["subtasks"], 1):
            title = subtask["title"]
            details = subtask["details"]
            matched = next((item for item in results["subtask_analysis"] if item["title"] == title), None)
            papers = matched["retrieved_documents"] if matched else []

            prompt = f"""You are writing a section for an academic literature review on "{topic}".

Section Focus: {title}
Research Questions for this section:
{chr(10).join(f"- {detail}" for detail in details)}

Instructions:
1. Write 600-800 words in formal academic style
2. Create a coherent narrative that flows logically
3. Use topic sentences to introduce main points
4. Include transitions between paragraphs
5. Synthesize findings rather than just summarizing papers
6. Use specific citations with actual author names and years: (Author et al., Year)
7. Do NOT use placeholder citations like (Author, Year) or (Smith, 2020) unless those are actual authors provided
8. If you reference general knowledge not from the papers, don't add citations
9. Structure: Introduction → Main findings → Implications → Conclusion

Available Research Papers:"""

            for paper in papers:
                if paper.get("error"):
                    continue
                text = paper.get("full_text", "")[:2000]
                authors = ", ".join(paper.get("authors", []))
                year = paper.get("published", "")[:4] if paper.get("published") else "Unknown"
                prompt += f"\n\n---\nTitle: {paper.get('paper_title', 'Unknown')}\nAuthors: {authors} ({year})\nSummary: {paper.get('summary', '')}\nExcerpt: {text}..."

            prompt += f"\n\nWrite a well-structured academic section that addresses the research questions and synthesizes the available literature."

            generated_text = call_openrouter(prompt)

            section_summaries.append({
                "title": title,
                "summary": generated_text[:300] + "..."
            })
            for paper in papers:
                if paper.get("error"):
                    continue
                year = paper.get("published", "")[:4] if paper.get("published") else "Unknown"
                authors = paper.get("authors", ["Unknown Author"])
                title_paper = paper.get("paper_title", "Unknown Title")
                url = paper.get("pdf_url", "")
                ref_text = f"{', '.join(authors)} ({year}). {title_paper}. Retrieved from {url}"
                if ref_text not in references:
                    references.append(ref_text)
            pdf.add_section(i, title, generated_text)

        conclusion_prompt = f"""
        Write a comprehensive conclusion (400-500 words) for this literature review on "{topic}".

        The review covered these main areas:
        {json.dumps(section_summaries, indent=2)}

        Your conclusion should:
        1. Synthesize the main findings across all sections
        2. Identify key themes and patterns
        3. Discuss implications for theory and practice
        4. Acknowledge limitations of current research  
        5. Suggest directions for future research
        6. End with a strong closing statement about the significance of this topic

        Write in formal academic style with smooth transitions and logical flow.
        """
        conclusion = call_openrouter(conclusion_prompt)
        pdf.add_section(len(plan["subtasks"]) + 1, "Conclusion and Future Directions", conclusion)
        if references:
            pdf.add_reference_list(references)
        pdf.output(output_pdf)
        return f"Enhanced PDF generated: {output_pdf}"


class MultiAgentCoordinator:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        agents = [
            ResearchQuestionAgent(),
            QueryExpansionAgent(),
            SearchRefinementAgent(),
            ArxivResearchAgent(),
            PDFGenerationAgent()
        ]

        for agent in agents:
            self.agents[agent.agent_id] = agent
            self.message_bus.register_agent(agent)
            agent.start()

    def execute_research_pipeline(self, query: str, output_pdf: str = "Research_Report.pdf") -> str:
        try:
            print(f"Starting research pipeline for: {query}")

            print("Step 0: Formulating research question...")
            question_result = self.agents["research_question"].call_agent_function(
                "research_question", "process_user_input", user_input=query
            )

            research_question = question_result["final_question"]
            print(f"✓ Research question formulated: {research_question}")
            print(f"  Quality score: {question_result['validation']['quality_score']}/10")

            if question_result['validation']['strengths']:
                print(f"  Strengths: {', '.join(question_result['validation']['strengths'])}")
            if question_result['validation']['weaknesses']:
                print(f"  Areas addressed: {', '.join(question_result['validation']['weaknesses'])}")


            print("Step 1: Expanding query and formatting to JSON...")
            plan_json = self.agents["query_expansion"].call_agent_function(
                "query_expansion", "expand_and_format", query=research_question
            )
            print("✓ Query expansion completed")

            print("Step 2: Refining search terms...")
            refined_json = self.agents["search_refinement"].call_agent_function(
                "search_refinement", "reformat_json_with_search_terms",
                original_json=plan_json
            )
            print("✓ Search term refinement completed")

            print("Step 3: Researching papers (this may take several minutes)...")
            research_results = self.agents["arxiv_research"].call_agent_function(
                "arxiv_research", "arxiv_agent",
                json_input=json.loads(refined_json)
            )
            print("✓ Paper research completed")

            print("Step 4: Generating PDF report...")
            pdf_result = self.agents["pdf_generation"].call_agent_function(
                "pdf_generation", "generate_research_paper",
                plan=json.loads(refined_json),
                results=research_results,
                output_pdf=output_pdf
            )
            print("✓ PDF generation completed")

            print("Research pipeline completed successfully!")
            return pdf_result

        except Exception as e:
            print(f"Pipeline failed: {e}")
            print("Attempting graceful shutdown...")
            raise

    def shutdown(self):
        for agent in self.agents.values():
            agent.stop()


#How is artificial intelligence transforming medical diagnosis?
#How does social media use affect adolescent mental health?
#What factors influence vaccine hesitancy in different communities?
#How does online learning compare to traditional classroom instruction?