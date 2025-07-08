from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room
import threading
import json
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
from main5 import MultiAgentCoordinator

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

active_sessions = {}
class WebSocketMultiAgentCoordinator(MultiAgentCoordinator):
    def __init__(self, session_id):
        super().__init__()
        self.session_id = session_id

    def emit_status(self, message, status_type='info'):
        socketio.emit('status_update', {
            'message': message,
            'type': status_type,
            'timestamp': datetime.now().isoformat()
        }, room=self.session_id)

    def emit_agent_status(self, agent_id, status):
        socketio.emit('agent_status', {
            'agent_id': agent_id,
            'status': status
        }, room=self.session_id)

    def emit_pipeline_step(self, step, status):
        socketio.emit('pipeline_step', {
            'step': step,
            'status': status
        }, room=self.session_id)

    def execute_research_pipeline_with_updates(self, query: str, output_pdf: str = None) -> str:
        try:
            if not output_pdf:
                output_pdf = f"research_report_{uuid.uuid4().hex[:8]}.pdf"

            self.emit_status(f"Starting research pipeline for: '{query}'", 'info')

            self.emit_agent_status('agent-researchQuestion', 'active')
            self.emit_pipeline_step(0, 'active')
            self.emit_status("Research Question Formatter Agent started processing...", 'info')
            question_result = self.agents["research_question"].call_agent_function(
                "research_question", "process_user_input", user_input=query
            )
            research_question = question_result["final_question"]
            self.emit_agent_status('agent-researchQuestion', 'completed')
            self.emit_pipeline_step(0, 'completed')
            self.emit_status("Research Question Formatter Agent completed successfully ✓", 'success')

            self.emit_agent_status('agent-query', 'active')
            self.emit_pipeline_step(1, 'active')
            self.emit_status("Query Expansion Agent started processing...", 'info')
            plan_json = self.agents["query_expansion"].call_agent_function(
                "query_expansion", "expand_and_format", query=research_question
            )
            self.emit_agent_status('agent-query', 'completed')
            self.emit_pipeline_step(1, 'completed')
            self.emit_status("Query Expansion Agent completed successfully ✓", 'success')

            self.emit_agent_status('agent-search', 'active')
            self.emit_pipeline_step(2, 'active')
            self.emit_status("Search Refinement Agent started processing...", 'info')
            refined_json = self.agents["search_refinement"].call_agent_function(
                "search_refinement", "reformat_json_with_search_terms",
                original_json=plan_json
            )
            self.emit_agent_status('agent-search', 'completed')
            self.emit_pipeline_step(2, 'completed')
            self.emit_status("Search Refinement Agent completed successfully ✓", 'success')

            self.emit_agent_status('agent-arxiv', 'active')
            self.emit_pipeline_step(3, 'active')
            self.emit_status("ArXiv Research Agent started processing (this may take several minutes)...", 'info')
            research_results = self.agents["arxiv_research"].call_agent_function(
                "arxiv_research", "arxiv_agent",
                json_input=json.loads(refined_json)
            )
            self.emit_agent_status('agent-arxiv', 'completed')
            self.emit_pipeline_step(3, 'completed')
            self.emit_status("ArXiv Research Agent completed successfully ✓", 'success')

            self.emit_agent_status('agent-pdf', 'active')
            self.emit_pipeline_step(4, 'active')
            self.emit_status("PDF Generation Agent started processing...", 'info')
            pdf_result = self.agents["pdf_generation"].call_agent_function(
                "pdf_generation", "generate_research_paper",
                plan=json.loads(refined_json),
                results=research_results,
                output_pdf=output_pdf
            )
            self.emit_agent_status('agent-pdf', 'completed')
            self.emit_pipeline_step(4, 'completed')
            self.emit_status("PDF Generation Agent completed successfully ✓", 'success')

            self.emit_status("Research pipeline completed.", 'success')
            self.emit_status("Generated PDF report ready for download.", 'info')
            socketio.emit('research_complete', {
                'pdf_filename': output_pdf,
                'download_url': f'/download/{output_pdf}'
            }, room=self.session_id)

            return pdf_result

        except Exception as e:
            self.emit_status(f"Pipeline failed: {str(e)}", 'error')
            self.emit_agent_status('agent-pdf', 'error')
            raise


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_research', methods=['POST'])
def start_research():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        session_id = str(uuid.uuid4())
        pdf_filename = f"research_report_{uuid.uuid4().hex[:8]}.pdf"

        active_sessions[session_id] = {
            'query': query,
            'status': 'running',
            'pdf_filename': pdf_filename,
            'started_at': datetime.now()
        }
        def run_research():
            try:
                coordinator = WebSocketMultiAgentCoordinator(session_id)
                result = coordinator.execute_research_pipeline_with_updates(query, pdf_filename)
                active_sessions[session_id]['status'] = 'completed'
                active_sessions[session_id]['result'] = result
            except Exception as e:
                active_sessions[session_id]['status'] = 'error'
                active_sessions[session_id]['error'] = str(e)
            finally:
                coordinator.shutdown()

        thread = threading.Thread(target=run_research)
        thread.daemon = True
        thread.start()
        return jsonify({
            'session_id': session_id,
            'status': 'started',
            'message': 'Research pipeline started'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session_status/<session_id>')
def get_session_status(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(active_sessions[session_id])


@app.route('/download/<filename>')
def download_file(filename):
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(os.getcwd(), safe_filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('join_session')
def handle_join_session(data):
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id})

if __name__ == '__main__':
    print("Starting Multi-Agent Research System...")
    print("Access the web interface at: http://localhost:5000")
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )