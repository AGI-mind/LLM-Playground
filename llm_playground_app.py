# -*- coding: utf-8 -*-
"""
LLM Playground App - Prompt Evaluation Version
Based on llmplayground_streamlit_test_01.py but adapted for prompt evaluation

This app allows users to:
- Configure system and user prompts
- Upload custom evaluation criteria
- Evaluate prompts using multiple LLM models
- View detailed evaluation results and statistics
- Export evaluation data in various formats
"""

import streamlit as st
import os
import json
import time
import re
import traceback
import zipfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from openai import OpenAI
import pandas as pd
import plotly.express as px
import numpy as np

# ------------------------------
# 1. MODEL CONFIGURATION
# ------------------------------
class ModelConfig:
    def __init__(self, id: str, name: str, temperature: float, max_tokens: int):
        self.id = id
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

def generate_mock_evaluations(models, outputs):
    """Generate mock evaluations for testing"""
    criteria = [
        {"name": "coherence", "description": "Coherence evaluation"},
        {"name": "feasibility", "description": "Feasibility evaluation"},
        {"name": "originality", "description": "Originality evaluation"},
        {"name": "pertinence", "description": "Pertinence evaluation"}
    ]
    
    evaluations = {}
    for model in models:
        scores = {}
        for criterion in criteria:
            scores[criterion['name']] = round(np.random.uniform(3, 9), 1)
        scores['feedback'] = f"Mock evaluation for {model.name} with scores: {scores}"
        evaluations[model.id] = scores
    
    return evaluations

# ------------------------------
# 2. PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="LLM Prompt Evaluation Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 3. STYLE FUNCTIONS
# ------------------------------
def apply_styles():
    """Apply global CSS styles"""
    st.markdown("""
    <style>
        /* Progress bar style */
        .stProgress > div > div > div > div {
            background-color: #27ae60 !important;
        }

        /* Layout adaptations */
        .main .block-container {
            max-width: 95%;
            padding-top: 2rem;
        }
        
        /* Custom button styles */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            background-color: #1a5276;
            color: white;
            border: 1px solid #1a5276;
        }
        
        .stButton > button:hover {
            background-color: #154360;
            border-color: #154360;
        }
        
        /* Custom metric styles */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1a5276;
        }
        
        /* Success message styles */
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #c3e6cb;
        }
        
        /* Error message styles */
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #f5c6cb;
        }
        
        /* Info message styles */
        .info-message {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #bee5eb;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# 4. SESSION STATE INITIALIZATION
# ------------------------------
def init_session():
    """Initialize session state"""
    pass

# ------------------------------
# 5. PROGRESS BAR MANAGER
# ------------------------------
class ProgressBarManager:
    """Manages progress bar logic for Streamlit"""
    
    def __init__(self, st_progress_bar_placeholder=None, st_progress_text_placeholder=None):
        self.st_progress_bar = st_progress_bar_placeholder
        self.st_progress_text = st_progress_text_placeholder
        self.current_step = 0
        self.total_steps = 1
        self.current_model = None
        self.phase = None

    def initialize(self, models: List[str]):
        self.current_step = 0
        self.phase = "evaluation_init"
        self.total_steps = len(models)
        self._update(f"Starting prompt evaluation with {len(models)} models")

    def set_model(self, model: str):
        self.current_model = model
        return self

    def set_phase(self, phase: str):
        self.phase = phase
        return self

    def step(self, message: str = None):
        self.current_step = min(self.current_step + 1, self.total_steps)
        if not message:
            message = self._generate_auto_message()
        self._update(message)
        return self

    def _generate_auto_message(self) -> str:
        return f"Evaluating prompt with {self.current_model}"

    def _update(self, message: str):
        if self.st_progress_bar and self.st_progress_text:
            progress_percentage = 0
            if self.total_steps > 0:
                progress_percentage = int((self.current_step / self.total_steps) * 100)

            self.st_progress_bar.progress(progress_percentage)
            self.st_progress_text.markdown(
                f"<div style='text-align:center; color:#666; margin-top:5px'>{message} ({self.current_step}/{self.total_steps}) - {progress_percentage}%</div>",
                unsafe_allow_html=True
            )

    def complete(self):
        self.current_step = self.total_steps
        self._update("Evaluation completed")
        if self.st_progress_bar:
            time.sleep(2)
            self.st_progress_bar.empty()
            if self.st_progress_text:
                self.st_progress_text.empty()

# ------------------------------
# 6. FILE SYSTEM MANAGER
# ------------------------------
class FileSystemManager:
    """Manages file system operations"""
    
    _DEFAULT_DIRS = {
        'base_dir': './data',
        'response_dir': 'data_response',
        'request_dir': 'data_request',
        'config_dir': 'data_config',
        'log_dir': 'data_log'
    }

    def __init__(self, base_dir=None, custom_dirs=None):
        self._default_dirs = self._DEFAULT_DIRS.copy()
        if base_dir:
            self._default_dirs['base_dir'] = base_dir
        if custom_dirs is not None:
            self._default_dirs.update(custom_dirs)
        self._build_paths()
        self._create_dirs()

    def _build_paths(self):
        self.paths = {}
        base = Path(self._default_dirs['base_dir'])
        for key, dir_name in self._default_dirs.items():
            if key == 'base_dir':
                self.paths[key] = str(base)
            else:
                self.paths[key] = str(base / dir_name)

    def _create_dirs(self):
        for dir_key, dir_path in self.paths.items():
            if dir_key != 'base_dir':
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_path(self, dir_key: str) -> str:
        if dir_key not in self.paths:
            raise KeyError(f"Invalid directory key '{dir_key}'. Use one of: {list(self.paths.keys())}")
        return self.paths[dir_key]

    def save_file(self, dir_key: str, filename: str, content: str, mode: str = 'w') -> str:
        target_dir = self.get_path(dir_key)
        filepath = Path(target_dir) / filename
        try:
            with open(filepath, mode, encoding='utf-8') as f:
                f.write(content)
            return str(filepath)
        except Exception as e:
            raise IOError(f"Error saving {filepath}: {str(e)}")

# ------------------------------
# 7. LOGGER MANAGER
# ------------------------------
class LoggerManager:
    """Manages logging operations"""
    
    def __init__(self, file_system_manager: FileSystemManager):
        self.fs_manager = file_system_manager
        self._init_log_directory()

    def _init_log_directory(self):
        self.log_dir = self.fs_manager.get_path('log_dir')
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            test_file = Path(self.log_dir) / 'access_test.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            st.warning(f"Error initializing log directory: {str(e)}. Fallback to ./logs")
            try:
                self.log_dir = "./logs"
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            except:
                self.log_dir = "./logs"
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def _get_log_file_path(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return str(Path(self.log_dir) / f"app_{today}.log")

    def _write_log(self, level: str, message: str, source: str = None, extra: dict = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": str(message),
            "source": source,
            "extra": extra if extra else {}
        }
        try:
            with open(self._get_log_file_path(), 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"FALLBACK LOG (stdout): {json.dumps(log_entry)}")
            print(f"Error writing log to file: {str(e)}")

    def debug(self, message: str, source: str = None): 
        self._write_log("DEBUG", message, source)
    def info(self, message: str, source: str = None): 
        self._write_log("INFO", message, source)
    def warning(self, message: str, source: str = None): 
        self._write_log("WARNING", message, source)
    def error(self, message: str, source: str = None, extra: dict = None): 
        self._write_log("ERROR", message, source, extra)

    def get_today_logs(self):
        log_file = self._get_log_file_path()
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError:
            return "No logs available for today"

# ------------------------------
# 8. OPENROUTER API CLIENT
# ------------------------------
class OpenRouterAPIClient:
    """Handles OpenRouter API interactions"""
    
    def __init__(self, api_key: str, file_system_manager: FileSystemManager, **kwargs):
        self.fs_manager = file_system_manager
        self.logger = LoggerManager(file_system_manager)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.default_params = {
            "temperature": kwargs.get('default_temperature', 0.7),
            "max_tokens": kwargs.get('default_max_tokens', 1500),
            "response_format": kwargs.get('response_format', {"type": "json_object"}),
            "stop": kwargs.get('default_stop', ["\n\n\n"])
        }
        self.extra_headers = kwargs.get('extra_headers', {
            "HTTP-Referer": "https://streamlit.app",
            "X-Title": "LLM_Prompt_Evaluation_App",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Language": "en"
        })
        self._init_paths()

    def _init_paths(self):
        self.response_dir = self.fs_manager.get_path('response_dir')
        self.request_dir = self.fs_manager.get_path('request_dir')
        self.config_dir = self.fs_manager.get_path('config_dir')
        self.log_dir = self.fs_manager.get_path('log_dir')

    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        params = {
            "model": model,
            "messages": messages,
            **{**self.default_params, **kwargs}
        }
        
        # Remove response_format if not JSON for models that don't always support it
        if params.get("response_format", {}).get("type") != "json_object":
            if "response_format" in params:
                del params["response_format"]

        self.logger.debug(f"Sending request to {model} with parameters: { {k:v for k,v in params.items() if k != 'messages'} }", source="OpenRouterAPIClient")
        start_time = time.time()
        try:
            # Ensure API key is properly set
            if not self.client.api_key:
                raise ValueError("API key is not set in the client")
                
            response = self.client.chat.completions.create(**params, extra_headers=self.extra_headers)
            response_data = response.model_dump()
            duration = time.time() - start_time
            self.logger.info(f"Request to {model} completed in {duration:.2f}s", source="OpenRouterAPIClient")
            return response_data
        except Exception as e:
            self.logger.error(
                f"Error in request to {model}",
                source="OpenRouterAPIClient",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "params": {k: str(v) for k, v in params.items()},
                    "api_key_set": bool(self.client.api_key)
                }
            )
            raise

    def evaluate_prompt(self, model: str, system_prompt: str, user_prompt: str, criteria: List[Dict[str, str]], **kwargs):
        """Evaluate a user prompt based on system prompt and criteria"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._create_evaluation_prompt(user_prompt, criteria)}
        ]

        current_request_kwargs = kwargs.copy()
        current_request_kwargs.setdefault('response_format', {"type": "json_object"})

        response = None
        try:
            # Validate API key before making request
            if not self.client.api_key:
                raise ValueError("API key is not set in the client")
                
            response = self.create_chat_completion(
                model=model,
                messages=messages,
                **current_request_kwargs
            )

            # Robust response validation
            if response is None:
                raise ValueError("API call returned None")

            if not isinstance(response, dict):
                raise TypeError(f"API response is not a dictionary. Type received: {type(response).__name__}")

            if 'error' in response and isinstance(response['error'], dict):
                error_details = response['error']
                error_message = error_details.get('message', 'Unknown API error')
                error_code = error_details.get('code', 'N/A')
                raise RuntimeError(f"API returned error: Code={error_code}, Message='{error_message}'")

            if 'choices' not in response:
                raise KeyError("'choices' key missing in API response")

            choices = response['choices']
            if not isinstance(choices, list) or not choices:
                raise ValueError("'choices' list is empty or invalid in API response")

            first_choice = choices[0]
            if not isinstance(first_choice, dict) or 'message' not in first_choice:
                raise KeyError("Invalid structure in first choice")

            message = first_choice['message']
            if not isinstance(message, dict) or 'content' not in message:
                raise KeyError("'content' key missing in message")

            content = message['content']
            is_json_response = False
            parsed_content = content
            
            try:
                parsed_content = json.loads(content)
                is_json_response = True
            except json.JSONDecodeError:
                is_json_response = False

            return {
                "success": True,
                "content": parsed_content,
                "is_json": is_json_response,
                "model": model,
                "raw_response": response
            }
        except Exception as e:
            error_data = {
                "model": model,
                "system_prompt_snippet": system_prompt[:100] + "...",
                "user_prompt_snippet": user_prompt[:100] + "...",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.logger.error(f"Error in evaluate_prompt for {model}", source="OpenRouterAPIClient", extra=error_data)
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "exception_type": type(e).__name__,
                "details": error_data
            }

    def _create_evaluation_prompt(self, user_prompt: str, criteria: List[Dict[str, str]]) -> str:
        """Create evaluation prompt based on criteria"""
        criteria_text = "\n".join([
            f"{i+1}. {c['name']}: {c['description']}"
            for i, c in enumerate(criteria)
        ])
        
        # Define score ranges for each criterion
        score_ranges = {
            "coherence": "0-10",
            "feasibility": "0-6", 
            "originality": "0-8",
            "pertinence": "0-6"
        }
        
        json_template = {
            c['name']: f"<score {score_ranges.get(c['name'], '0-10')}>"
            for c in criteria
        }
        json_template["feedback"] = "<detailed feedback explaining your scores and reasoning>"
        
        return f"""You are an expert evaluator. Please evaluate the following user prompt based on these criteria:

{criteria_text}

User Prompt to evaluate:
{user_prompt}

Please provide your evaluation in the following JSON format:
{json.dumps(json_template, indent=2)}

Important scoring guidelines:
- Coherence: Score 0-10 (10-9: Highly coherent, 8-5: Generally coherent, 4-2: Multiple incoherencies, 1-0: Incoherent)
- Feasibility: Score 0-6 (6-5: Fully feasible, 4: Feasible but lacks detail, 3-2: Questionable, 1-0: Unrealistic)
- Originality: Score 0-8 (8-7: Clearly original, 6-4: Some innovation, 3-2: Limited originality, 1-0: No originality)
- Pertinence: Score 0-6 (6-5: Excellent relevance, 4: Relevant but lacks specificity, 3-2: Weakly connected, 1-0: Poor relevance)

Ensure all scores are numbers within the specified ranges for each criterion.
The feedback should explain your reasoning for each score.

IMPORTANT: Your response must be a valid JSON object with the exact structure shown above."""

# ------------------------------
# 9. EVALUATION ORCHESTRATOR
# ------------------------------
class EvaluationOrchestrator:
    """Orchestrates prompt evaluation across multiple models"""
    
    def __init__(self, api_client: OpenRouterAPIClient, progress_manager: ProgressBarManager = None):
        self.client = api_client
        self.fs_manager = api_client.fs_manager
        self.logger = LoggerManager(self.fs_manager)
        self.progress_manager = progress_manager

    def evaluate_prompts(self, models: List[str], system_prompt: str, user_prompt: str, criteria: List[Dict[str, str]], **kwargs) -> Dict:
        """Evaluate user prompt with multiple models"""
        if self.progress_manager:
            self.progress_manager.initialize(models)

        self.logger.info(
            f"Starting prompt evaluation:\n"
            f"System: {system_prompt[:200]}...\n"
            f"User: {user_prompt[:200]}...\n"
            f"Models: {models}",
            source="EvaluationOrchestrator"
        )

        results = {}
        try:
            for i, model in enumerate(models):
                if self.progress_manager:
                    self.progress_manager.set_model(model).step(f"Evaluating with {model}")

                start_time = time.time()
                raw_response = self.client.evaluate_prompt(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    criteria=criteria,
                    **kwargs
                )
                latency = time.time() - start_time

                # Parse the response using our robust JSON parser
                if raw_response.get("success"):
                    content = raw_response.get("content", "")
                    if isinstance(content, str):
                        # Parse the JSON response
                        parsed_evaluation = extract_json_from_response(content, criteria)
                    else:
                        # Content is already parsed JSON
                        parsed_evaluation = content
                else:
                    # Handle error case
                    parsed_evaluation = {
                        "error": raw_response.get("error", "Unknown error"),
                        "feedback": "Evaluation failed"
                    }

                results[model] = {
                    "response": raw_response,
                    "parsed_evaluation": parsed_evaluation,
                    "latency": latency,
                    "timestamp": datetime.now().isoformat()
                }

            if self.progress_manager:
                self.progress_manager.complete()

            return {
                "success": True,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Error in evaluate_prompts: {str(e)}"
            self.logger.error(error_msg, source="EvaluationOrchestrator", extra={"traceback": traceback.format_exc()})
            if self.progress_manager:
                self.progress_manager._update(f"Error: {str(e)}")
            raise RuntimeError(error_msg) from e

# ------------------------------
# 10. UTILITY FUNCTIONS
# ------------------------------
def parse_uploaded_criteria(uploaded_file) -> List[Dict[str, str]]:
    """Parse evaluation criteria from uploaded file (JSON or CSV)"""
    try:
        if uploaded_file.name.endswith('.json'):
            # Parse JSON file
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            if isinstance(data, list):
                # Direct list of criteria
                return data
            elif isinstance(data, dict) and 'criteria' in data:
                # Wrapped in an object
                return data['criteria']
            else:
                raise ValueError("Invalid JSON format")
                
        elif uploaded_file.name.endswith('.csv'):
            # Parse CSV file
            import pandas as pd
            import io
            
            content = uploaded_file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            
            if 'name' not in df.columns or 'description' not in df.columns:
                raise ValueError("CSV must contain 'name' and 'description' columns")
            
            criteria = []
            for _, row in df.iterrows():
                criteria.append({
                    'name': str(row['name']),
                    'description': str(row['description'])
                })
            return criteria
            
        else:
            raise ValueError("Unsupported file format. Please upload JSON or CSV files.")
            
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")

def extract_json_from_response(response: str, criteria: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract and validate JSON from model response with robust parsing"""
    try:
        # Strategy 1: Try to find and parse JSON directly
        json_pattern = r'\{[\s\S]*\}'
        json_matches = re.findall(json_pattern, response)
        
        if json_matches:
            # Try each potential JSON match
            for json_str in json_matches:
                try:
                    # Clean the JSON string
                    json_str = json_str.strip()
                    json_str = re.sub(r'^```json\s*|\s*```$', '', json_str)
                    json_str = re.sub(r'^```\s*|\s*```$', '', json_str)
                    
                    # Try to fix truncated JSON by finding the last complete key-value pair
                    fixed_json = fix_truncated_json(json_str)
                    
                    data = json.loads(fixed_json)
                    
                    # Check if this is our expected structure
                    if not isinstance(data, dict):
                        continue
                    
                    # Check if we have at least some of our criteria
                    criteria_names = [c['name'] for c in criteria]
                    found_criteria = [name for name in criteria_names if name in data]
                    
                    if len(found_criteria) >= len(criteria_names) * 0.5:  # At least 50% of criteria
                        # Convert any string scores to float
                        for criterion in criteria_names:
                            if criterion in data:
                                score = data[criterion]
                                if isinstance(score, str):
                                    # Handle string numbers like "9", "7.5"
                                    if score.replace('.', '').replace('-', '').isdigit():
                                        data[criterion] = float(score)
                                    else:
                                        # Try to extract number from string
                                        number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                                        if number_match:
                                            data[criterion] = float(number_match.group(1))
                                        else:
                                            data[criterion] = 5.0  # Default
                                elif isinstance(score, (int, float)):
                                    data[criterion] = float(score)
                                else:
                                    data[criterion] = 5.0  # Default
                            else:
                                data[criterion] = 5.0  # Default
                        
                        # Ensure feedback exists
                        if 'feedback' not in data:
                            data['feedback'] = "No feedback provided"
                        
                        return data
                
                except json.JSONDecodeError as e:
                    # Log the JSON decode error for debugging
                    print(f"JSON decode error: {e}")
                    print(f"JSON string: {json_str[:200]}...")
                    continue
        
        # Strategy 2: If no valid JSON found, use simple parsing
        return simple_score_parser(response, criteria)
        
    except Exception as e:
        # Create a default response with all criteria set to 5
        default_data = {c['name']: 5 for c in criteria}
        default_data['feedback'] = f"Error parsing response: {str(e)}"
        return default_data

def fix_truncated_json(json_str: str) -> str:
    """Fix truncated JSON by finding the last complete key-value pair and closing properly."""
    try:
        # First, try to parse as-is
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    # If parsing fails, try to fix truncation
    lines = json_str.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line has a complete key-value pair
        if ':' in line and not line.endswith(','):
            # This might be a complete line, add it
            fixed_lines.append(line)
        elif ':' in line and line.endswith(','):
            # This is a complete line with comma, add it
            fixed_lines.append(line)
        elif line.startswith('"') and ':' in line:
            # This looks like a key-value pair, try to complete it
            if not line.endswith('"') and not line.endswith(','):
                # Try to find the end of the value
                colon_pos = line.find(':')
                if colon_pos != -1:
                    value_part = line[colon_pos + 1:].strip()
                    if value_part.startswith('"') and not value_part.endswith('"'):
                        # Incomplete string value, skip this line
                        continue
                    else:
                        # Complete value, add it
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
    
    # Join the lines and try to close the JSON properly
    fixed_json = '\n'.join(fixed_lines)
    
    # Remove trailing comma if present
    fixed_json = re.sub(r',\s*}', '}', fixed_json)
    
    # If it doesn't end with }, add it
    if not fixed_json.endswith('}'):
        fixed_json += '}'
    
    # Try to parse the fixed JSON
    try:
        json.loads(fixed_json)
        return fixed_json
    except json.JSONDecodeError:
        # If still invalid, return the original and let robust parser handle it
        return json_str

def robust_json_parser(response: str, criteria: List[Dict[str, str]]) -> Dict[str, Any]:
    """Super aggressive parser that searches for criterion names and scores anywhere in the text."""
    criteria_names = [c['name'] for c in criteria]
    result = {}
    
    # Strategy 1: Look for JSON-like patterns with quotes
    for criterion in criteria_names:
        # Pattern 1: "criterion": "score"
        pattern1 = rf'"{criterion}"\s*:\s*["\'](\d+(?:\.\d+)?)["\']'
        match1 = re.search(pattern1, response, re.IGNORECASE)
        
        # Pattern 2: "criterion": score (without quotes)
        pattern2 = rf'"{criterion}"\s*:\s*(\d+(?:\.\d+)?)'
        match2 = re.search(pattern2, response, re.IGNORECASE)
        
        # Pattern 3: criterion: "score" (without quotes around criterion)
        pattern3 = rf'{criterion}\s*:\s*["\'](\d+(?:\.\d+)?)["\']'
        match3 = re.search(pattern3, response, re.IGNORECASE)
        
        # Pattern 4: criterion: score (no quotes anywhere)
        pattern4 = rf'{criterion}\s*:\s*(\d+(?:\.\d+)?)'
        match4 = re.search(pattern4, response, re.IGNORECASE)
        
        # Pattern 5: criterion: score, (with comma)
        pattern5 = rf'{criterion}\s*:\s*(\d+(?:\.\d+)?),'
        match5 = re.search(pattern5, response, re.IGNORECASE)
        
        # Pattern 6: "criterion": score, (with comma)
        pattern6 = rf'"{criterion}"\s*:\s*(\d+(?:\.\d+)?),'
        match6 = re.search(pattern6, response, re.IGNORECASE)
        
        # Use the first match found
        if match1:
            result[criterion] = float(match1.group(1))
        elif match2:
            result[criterion] = float(match2.group(1))
        elif match3:
            result[criterion] = float(match3.group(1))
        elif match4:
            result[criterion] = float(match4.group(1))
        elif match5:
            result[criterion] = float(match5.group(1))
        elif match6:
            result[criterion] = float(match6.group(1))
        else:
            result[criterion] = 5.0  # Default score
    
    # Strategy 2: Look for scores in the text context
    if any(result[c] == 5.0 for c in criteria_names):
        # Try to find scores mentioned in the text
        for criterion in criteria_names:
            if result[criterion] == 5.0:
                # Look for patterns like "clarity score: 9" or "clarity: 9/10"
                score_patterns = [
                    rf'{criterion}\s+score\s*:\s*(\d+(?:\.\d+)?)',
                    rf'{criterion}\s*:\s*(\d+(?:\.\d+)?)/10',
                    rf'{criterion}\s*:\s*(\d+(?:\.\d+)?)',
                    rf'{criterion}\s+is\s+(\d+(?:\.\d+)?)',
                    rf'{criterion}\s+(\d+(?:\.\d+)?)',
                ]
                
                for pattern in score_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        result[criterion] = float(match.group(1))
                        break
    
    # Strategy 3: Look for numbered lists or bullet points
    if any(result[c] == 5.0 for c in criteria_names):
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            for criterion in criteria_names:
                if result[criterion] == 5.0 and criterion.lower() in line.lower():
                    # Look for numbers in the line
                    numbers = re.findall(r'\d+(?:\.\d+)?', line)
                    if numbers:
                        # Take the first number found
                        result[criterion] = float(numbers[0])
    
    # Extract feedback using multiple strategies
    feedback = "No feedback provided"
    
    # Strategy 1: Look for "feedback" field in JSON
    feedback_patterns = [
        r'"feedback"\s*:\s*"([^"]+)"',
        r'"feedback"\s*:\s*"([^"]*)"',
        r'feedback\s*:\s*"([^"]+)"',
        r'feedback\s*:\s*"([^"]*)"',
    ]
    
    for pattern in feedback_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            feedback = match.group(1)
            break
    
    # Strategy 2: If no feedback found, try to extract the last paragraph
    if feedback == "No feedback provided":
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            # Look for the longest paragraph that doesn't contain scores
            longest_paragraph = ""
            for para in paragraphs:
                para = para.strip()
                if len(para) > len(longest_paragraph) and not any(re.search(r'\d+', para) for _ in range(3)):
                    longest_paragraph = para
            
            if longest_paragraph and len(longest_paragraph) > 20:
                feedback = longest_paragraph
    
    result['feedback'] = feedback
    
    return result

def simple_score_parser(response: str, criteria: List[Dict[str, str]]) -> Dict[str, Any]:
    """Super simple parser that just finds criterion names and numbers near them."""
    criteria_names = [c['name'] for c in criteria]
    result = {}
    
    # For each criterion, find it in the text and look for numbers nearby
    for criterion in criteria_names:
        score_found = False
        
        # Find all positions of the criterion name (case insensitive)
        response_lower = response.lower()
        criterion_lower = criterion.lower()
        positions = []
        pos = 0
        while True:
            pos = response_lower.find(criterion_lower, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += 1
        
        # For each position, look for numbers within 100 characters
        for pos in positions:
            start = max(0, pos - 100)
            end = min(len(response), pos + 100)
            search_area = response[start:end]
            
            # Find all numbers in this area
            numbers = re.findall(r'\d+(?:\.\d+)?', search_area)
            if numbers:
                # Take the first reasonable number
                for num_str in numbers:
                    try:
                        score = float(num_str)
                        # Get max score for this criterion
                        max_score = next((c.get('max_score', 10) for c in criteria if c['name'] == criterion), 10)
                        if 0 <= score <= max_score:  # Valid score range for this criterion
                            result[criterion] = score
                            score_found = True
                            break
                    except:
                        continue
                if score_found:
                    break
        
        # If still not found, try to find any number in the entire response
        if not score_found:
            all_numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if all_numbers:
                for num_str in all_numbers:
                    try:
                        score = float(num_str)
                        # Get max score for this criterion
                        max_score = next((c.get('max_score', 10) for c in criteria if c['name'] == criterion), 10)
                        if 0 <= score <= max_score:
                            result[criterion] = score
                            score_found = True
                            break
                    except:
                        continue
        
        # Default score
        if not score_found:
            result[criterion] = 5.0
    
    # Find feedback - just take the longest text block
    feedback = "No feedback provided"
    lines = response.split('\n')
    longest_line = ""
    for line in lines:
        line = line.strip()
        if len(line) > len(longest_line) and not line.startswith('{') and not line.startswith('"'):
            longest_line = line
    
    if len(longest_line) > 20:
        feedback = longest_line
    
    result['feedback'] = feedback
    return result

# ------------------------------
# 11. SESSION STATE INITIALIZATION
# ------------------------------
def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        "api_key_valid": False,
        "api_key_input": "",
        "api_key_loaded_from_secrets": False,
        "available_models": [
            "google/gemini-2.5-flash-lite",
            "openai/chatgpt-4o-latest",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-prover-v2",
            "mistralai/mistral-small-3.1-24b-instruct"
        ],
        "selected_models": [],
        "system_prompt": "You are an expert university professor specializing in Psychology for Well-being, evaluating papers for the \"Lifelong Learning and Empowerment\" course at the Catholic University of the Sacred Heart. Students are required to design psychological interventions for the development of personal resources throughout the life cycle, demonstrating skills in assessment and the enhancement of specific skills.",
        "user_prompt": "Design a comprehensive wellness program for university students that addresses mental health, physical fitness, and academic performance. The program should be innovative, feasible to implement, and directly relevant to student well-being.",
        "evaluation_criteria": [
            {"name": "coherence", "description": "10-9: Highly coherent and logically structured. 8-5: Generally coherent with minor logical gaps. 4-2: Multiple incoherencies impairing comprehension. 1-0: Incoherent, hard or impossible to follow.", "max_score": 10},
            {"name": "feasibility", "description": "6-5: Fully feasible with clear implementation steps. 4: Feasible but lacks implementation detail. 3-2: Questionable feasibility. 1-0: Unrealistic or impractical proposal.", "max_score": 6},
            {"name": "originality", "description": "8-7: Clearly original and creative, beyond classroom examples. 6-4: Some innovative elements. 3-2: Limited originality; derivative. 1-0: No originality; repetitive or banal.", "max_score": 8},
            {"name": "pertinence", "description": "6-5: Excellent relevance to the theme of well-being; includes a clearly defined and innovative skill addressing the issue. 4: Relevant with some lack of specificity. 3-2: General relevance but weakly connected to well-being. 1-0: Poor relevance; unclear or undefined skill.", "max_score": 6}
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
        "evaluation_results": None,
        "fs_manager_instance": None,
        "logger_instance": None,
        "api_client_instance": None,
        "orchestrator_instance": None,
        "progress_bar_placeholder": None,
        "progress_text_placeholder": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ------------------------------
# 12. CACHE RESOURCES
# ------------------------------
@st.cache_resource
def get_fs_manager(base_dir_override=None):
    """Get cached file system manager"""
    effective_base_dir = base_dir_override or "./data"
    return FileSystemManager(base_dir=effective_base_dir)

@st.cache_resource
def get_logger(_fs_manager):
    """Get cached logger"""
    return LoggerManager(_fs_manager)

@st.cache_resource
def get_api_client(api_key, _fs_manager):
    """Get cached API client"""
    if not api_key:
        return None
    try:
        client = OpenRouterAPIClient(api_key=api_key, file_system_manager=_fs_manager)
        # Verify the API key is properly set
        if not client.client.api_key:
            raise ValueError("API key not properly set in client")
        return client
    except Exception as e:
        st.error(f"Error creating API client: {str(e)}")
        return None

@st.cache_resource
def get_orchestrator(_api_client, _progress_manager):
    """Get cached orchestrator"""
    if not _api_client:
        return None
    return EvaluationOrchestrator(api_client=_api_client, progress_manager=_progress_manager)

# ------------------------------
# 13. PAGE FUNCTIONS
# ------------------------------
def setup_page():
    """Setup page for configuration"""
    st.header("Setup Configuration")
    
    # Create two main columns with left column always open
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Left Column Content
    with left_col:
        st.subheader("Prompt Configuration")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=100,
            key="system_prompt_input"
        )
        st.session_state.system_prompt = system_prompt
        
        # File upload for user prompt
        st.write("**Upload User Prompt (Optional)**")
        uploaded_prompt_file = st.file_uploader(
            "Choose a text file with user prompt",
            type=['txt'],
            help="Upload a text file containing the user prompt to be evaluated"
        )
        
        if uploaded_prompt_file is not None:
            try:
                uploaded_prompt_content = uploaded_prompt_file.read().decode('utf-8').strip()
                st.session_state.user_prompt = uploaded_prompt_content
                st.success(f"Successfully loaded user prompt from {uploaded_prompt_file.name}")
            except Exception as e:
                st.error(f"Error loading user prompt: {str(e)}")
        
        user_prompt = st.text_area(
            "User Prompt",
            value=st.session_state.user_prompt,
            height=120,
            key="user_prompt_input"
        )
        st.session_state.user_prompt = user_prompt
        
        st.subheader("Evaluation Criteria")
        
        # Display existing criteria
        for i, criterion in enumerate(st.session_state.evaluation_criteria):
            crit_col1, crit_col2, crit_col3, crit_col4 = st.columns([2, 3, 1, 1])
            with crit_col1:
                new_name = st.text_input(
                    "Criterion Name",
                    value=criterion["name"],
                    key=f"crit_name_{i}"
                )
            with crit_col2:
                new_desc = st.text_input(
                    "Description",
                    value=criterion["description"],
                    key=f"crit_desc_{i}"
                )
            with crit_col3:
                max_score = st.number_input(
                    "Max Score",
                    min_value=1,
                    max_value=20,
                    value=criterion.get("max_score", 10),
                    key=f"crit_max_score_{i}"
                )
            with crit_col4:
                if st.button("🗑️", key=f"remove_crit_{i}"):
                    st.session_state.evaluation_criteria.pop(i)
                    st.rerun()
            
            # Update the criterion in session state
            st.session_state.evaluation_criteria[i] = {
                "name": new_name,
                "description": new_desc,
                "max_score": max_score
            }
        
        # Add new criterion button
        if st.button("➕ Add Criterion"):
            st.session_state.evaluation_criteria.append({
                "name": f"criterion_{len(st.session_state.evaluation_criteria) + 1}",
                "description": "Enter description here",
                "max_score": 10
            })
            st.rerun()
        
        # Store criteria in session state
        if st.session_state.evaluation_criteria:
            st.session_state.parsed_criteria = st.session_state.evaluation_criteria
    
    # Right Column Content
    with right_col:
        st.subheader("API Configuration")
        
        # Try to load API key from secrets first (only once)
        if not st.session_state.get("api_key_loaded_from_secrets", False):
            try:
                secret_api_key = st.secrets.get("OPENROUTER_API_KEY")
                if secret_api_key:
                    st.session_state.api_key_input = secret_api_key
                    st.session_state.api_key_loaded_from_secrets = True
                    st.info("API Key loaded from secrets.", icon="ℹ️")
            except Exception:
                # Silently handle missing secrets file - this is normal
                pass
            finally:
                st.session_state.api_key_loaded_from_secrets = True
        
        api_key = st.text_input(
            "OpenRouter API Key", 
            type="password", 
            value=st.session_state.get("api_key_input", ""),
            key="api_key_input"
        )
        
        # Update validation status based on current input
        if api_key:
            st.session_state.api_key_valid = True
            st.success("API Key entered.", icon="✅")
        else:
            st.session_state.api_key_valid = False
            st.warning("API Key not configured. Enter it to proceed.")
        
        st.info("Get your API key from [openrouter.ai/keys](https://openrouter.ai/keys)")
        
        st.subheader("Models Configuration")
        
        # Initialize models if not exists
        if 'models' not in st.session_state:
            st.session_state.models = [
                ModelConfig('model1', 'google/gemini-2.5-flash-lite', 0.7, 500),
                ModelConfig('model2', 'openai/chatgpt-4o-latest', 0.7, 500),
                ModelConfig('model3', 'anthropic/claude-3.5-sonnet', 0.7, 500),
                ModelConfig('model4', 'deepseek/deepseek-prover-v2', 0.7, 500),
                ModelConfig('model5', 'mistralai/mistral-small-3.1-24b-instruct', 0.7, 500)
            ]
        
        # Model configuration
        for i, model in enumerate(st.session_state.models):
            with st.expander(f"Model {i+1}: {model.name}"):
                new_name = st.selectbox(f"Model {i+1}", st.session_state.available_models, 
                                      index=st.session_state.available_models.index(model.name) if model.name in st.session_state.available_models else 0, 
                                      key=f"model_name_{i}")
                new_temp = st.slider(f"Temperature", 0.0, 1.0, model.temperature, 0.1, key=f"temp_{i}")
                new_tokens = st.slider(f"Max Tokens", 100, 1000, model.max_tokens, 100, key=f"tokens_{i}")
                
                st.session_state.models[i] = ModelConfig(model.id, new_name, new_temp, new_tokens)
                
                if len(st.session_state.models) > 2:
                    if st.button(f"Remove Model {i+1}", key=f"remove_{i}"):
                        st.session_state.models.pop(i)
                        st.rerun()
        
        if st.button("Add Model"):
            new_id = f"model{len(st.session_state.models) + 1}"
            st.session_state.models.append(ModelConfig(new_id, st.session_state.available_models[0], 0.7, 500))
            st.rerun()
    
    # Generate & Evaluate button (full width)
    st.markdown("---")
    if st.button("🚀 Evaluate User Prompt", type="primary", use_container_width=True):
        if not system_prompt or not user_prompt:
            st.error("Please fill all prompt fields!")
            return
        
        if not st.session_state.evaluation_criteria:
            st.error("Please add at least one evaluation criterion!")
            return
        
        if len(st.session_state.models) < 1:
            st.error("Please add at least one model!")
            return
        
        if not st.session_state.api_key_valid:
            st.error("Please enter a valid API key to continue!")
            return
        
        with st.spinner("Evaluating user prompt..."):
            # Real evaluation using orchestrator
            try:
                # Initialize API client and orchestrator if not already done
                if not st.session_state.api_client_instance:
                    st.session_state.api_client_instance = get_api_client(st.session_state.api_key_input, st.session_state.fs_manager_instance)
                
                if not st.session_state.orchestrator_instance:
                    progress_manager_instance = ProgressBarManager(
                        st.session_state.progress_bar_placeholder.progress(0),
                        st.session_state.progress_text_placeholder.markdown("")
                    )
                    st.session_state.orchestrator_instance = get_orchestrator(st.session_state.api_client_instance, progress_manager_instance)

                if not st.session_state.orchestrator_instance:
                    st.error("Error: Could not initialize API client. Please check your API key.")
                    return

                result_data = st.session_state.orchestrator_instance.evaluate_prompts(
                    models=[model.name for model in st.session_state.models],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    criteria=st.session_state.evaluation_criteria,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
                
                st.session_state.evaluation_results = result_data
                st.success("🎉 Evaluation completed successfully!")

            except Exception as e:
                st.error(f"🔥 Critical error during evaluation: {str(e)}")
                if st.session_state.logger_instance:
                    st.session_state.logger_instance.error(f"UI Evaluation Error: {str(e)}", source="StreamlitUI", extra={"traceback": traceback.format_exc()})
        
        st.success("Evaluation complete! Check the Evaluation Matrix page.")

def outputs_page():
    """Model outputs page"""
    st.header("Model Outputs")
    
    if not st.session_state.user_prompt or not st.session_state.system_prompt:
        st.info("No prompts configured yet. Go to Setup to configure the prompts.")
        return
    
    # Check if we have evaluation results
    if not st.session_state.evaluation_results:
        st.info("No model outputs available. Run an evaluation first to see model responses.")
        return
    
    # Display the prompts being evaluated
    st.subheader("Evaluated Prompts")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**System Prompt:**")
        st.write(st.session_state.system_prompt)
    
    with col2:
        st.write("**User Prompt:**")
        st.write(st.session_state.user_prompt)
    
    # Display model outputs
    st.subheader("Model Responses")
    results = st.session_state.evaluation_results.get("results", {})
    
    if results:
        for model_name, model_data in results.items():
            response = model_data.get("response", {})
            latency = model_data.get("latency", "N/A")
            
            with st.expander(f"{model_name} Response", expanded=True):
                # Show latency
                if isinstance(latency, float):
                    latency_str = f"{latency:.2f}s"
                else:
                    latency_str = str(latency)
                st.caption(f"Response time: {latency_str}")
                
                # Show the actual model response
                if response.get("success"):
                    content = response.get("content", "")
                    if response.get("is_json"):
                        # If it's JSON, show both raw and formatted
                        st.write("**Raw Response:**")
                        st.code(content, language="json")
                        
                        # Try to parse and display as structured data
                        try:
                            if isinstance(content, str):
                                parsed_content = json.loads(content)
                            else:
                                parsed_content = content
                            
                            st.write("**Structured Response:**")
                            st.json(parsed_content)
                        except:
                            st.write("**Content:**")
                            st.write(content)
                    else:
                        # If it's plain text
                        st.write("**Response:**")
                        st.text_area("Model Output", content, height=200, disabled=True, key=f"output_{model_name}")
                else:
                    st.error(f"Error for {model_name}: {response.get('error', 'Unknown error')}")
                    if "details" in response:
                        st.json(response["details"])
    else:
        st.info("No model outputs found in the results.")

def evaluation_output_page():
    """Evaluation output page"""
    st.header("Evaluation Output")
    
    if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
        st.warning("No evaluations available. Please run evaluations first.")
        return
    
    # Display the prompts being evaluated
    st.subheader("Evaluated Prompts")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**System Prompt:**")
        st.write(st.session_state.system_prompt)
    
    with col2:
        st.write("**User Prompt:**")
        st.write(st.session_state.user_prompt)
    
    # Create a DataFrame for easier analysis
    eval_data = []
    criteria = [c['name'] for c in st.session_state.evaluation_criteria]
    
    results = st.session_state.evaluation_results.get("results", {})
    for model_name, model_data in results.items():
        parsed_evaluation = model_data.get("parsed_evaluation", {})
        row = {
            'Evaluator': model_name,
            'Feedback': parsed_evaluation.get('feedback', 'No feedback provided')
        }
        
        # Add scores for each criterion
        for criterion in criteria:
            score = parsed_evaluation.get(criterion, "N/A")
            # Handle different score formats consistently
            if isinstance(score, str):
                # Try to convert string to number
                if score.replace('.', '').replace('-', '').isdigit():
                    row[criterion] = float(score)
                else:
                    # Try to extract number from string
                    number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                    if number_match:
                        row[criterion] = float(number_match.group(1))
                    else:
                        row[criterion] = score
            elif isinstance(score, (int, float)):
                row[criterion] = float(score)
            else:
                row[criterion] = score
        
        eval_data.append(row)
    
    if eval_data:
        df = pd.DataFrame(eval_data)
        
        # Calculate average scores for each criterion (raw and normalized)
        avg_scores = {}
        avg_scores_normalized = {}
        for criterion in criteria:
            scores = []
            normalized_scores = []
            max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
            
            for row in eval_data:
                score = row.get(criterion, 0)
                # Handle different score formats consistently
                if isinstance(score, str):
                    # Try to convert string to number
                    if score.replace('.', '').replace('-', '').isdigit():
                        raw_score = float(score)
                        scores.append(raw_score)
                        # Normalize to 0-10 scale
                        normalized_score = (raw_score / max_score) * 10
                        normalized_scores.append(normalized_score)
                    else:
                        # Try to extract number from string
                        number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                        if number_match:
                            raw_score = float(number_match.group(1))
                            scores.append(raw_score)
                            # Normalize to 0-10 scale
                            normalized_score = (raw_score / max_score) * 10
                            normalized_scores.append(normalized_score)
                elif isinstance(score, (int, float)):
                    raw_score = float(score)
                    scores.append(raw_score)
                    # Normalize to 0-10 scale
                    normalized_score = (raw_score / max_score) * 10
                    normalized_scores.append(normalized_score)
            
            if scores:
                avg_scores[criterion] = sum(scores) / len(scores)
                avg_scores_normalized[criterion] = sum(normalized_scores) / len(normalized_scores)
        
        # Display average scores (raw and normalized)
        if avg_scores:
            st.subheader("Average Scores")
            
            # Create two columns for raw and normalized scores
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Raw Scores (Original Scale)**")
                for criterion, score in avg_scores.items():
                    max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
                    st.metric(criterion.title(), f"{score:.2f}/{max_score}")
            
            with col2:
                st.write("**Normalized Scores (0-10 Scale)**")
                for criterion, score in avg_scores_normalized.items():
                    st.metric(criterion.title(), f"{score:.2f}/10")
            
            # Show overall average (normalized)
            if avg_scores_normalized:
                overall_avg = sum(avg_scores_normalized.values()) / len(avg_scores_normalized)
                st.metric("**Overall Average (Normalized)**", f"{overall_avg:.2f}/10")
        
        # Display the full evaluation table
        st.subheader("Detailed Evaluations")
        st.dataframe(df)
    else:
        st.info("No evaluation data available.")

def evaluation_matrix_page():
    """Evaluation matrix page"""
    st.header("Evaluation Matrix")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluations available. Generate evaluations first.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Detailed Results", "Statistics", "Data Export"])
    
    with tab1:
        summary_tab()
    
    with tab2:
        detailed_results_tab()
    
    with tab3:
        statistics_tab()
    
    with tab4:
        export_tab()

def summary_tab():
    """Summary tab content"""
    st.subheader("Model Evaluation Results")
    
    # Get criteria from session state
    criteria = [c['name'] for c in st.session_state.evaluation_criteria]
    
    # Calculate average scores across all models for each criterion (raw and normalized)
    criterion_averages = {}
    criterion_averages_normalized = {}
    results = st.session_state.evaluation_results.get("results", {})
    
    for criterion in criteria:
        scores = []
        normalized_scores = []
        # Get max score for this criterion
        max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
        
        for model_name, model_data in results.items():
            parsed_evaluation = model_data.get("parsed_evaluation", {})
            if criterion in parsed_evaluation:
                score = parsed_evaluation[criterion]
                # Handle different score formats consistently
                if isinstance(score, str):
                    # Try to convert string to number
                    if score.replace('.', '').replace('-', '').isdigit():
                        raw_score = float(score)
                        scores.append(raw_score)
                        # Normalize to 0-10 scale
                        normalized_score = (raw_score / max_score) * 10
                        normalized_scores.append(normalized_score)
                    else:
                        # Try to extract number from string
                        number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                        if number_match:
                            raw_score = float(number_match.group(1))
                            scores.append(raw_score)
                            # Normalize to 0-10 scale
                            normalized_score = (raw_score / max_score) * 10
                            normalized_scores.append(normalized_score)
                elif isinstance(score, (int, float)):
                    raw_score = float(score)
                    scores.append(raw_score)
                    # Normalize to 0-10 scale
                    normalized_score = (raw_score / max_score) * 10
                    normalized_scores.append(normalized_score)
        
        if scores:
            criterion_averages[criterion] = sum(scores) / len(scores)
            criterion_averages_normalized[criterion] = sum(normalized_scores) / len(normalized_scores)
    
    # Display average scores (raw and normalized)
    if criterion_averages:
        st.subheader("Average Scores by Criterion")
        
        # Create two columns for raw and normalized scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Raw Scores (Original Scale)**")
            for criterion, score in criterion_averages.items():
                max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
                st.metric(criterion.title(), f"{score:.2f}/{max_score}")
        
        with col2:
            st.write("**Normalized Scores (0-10 Scale)**")
            for criterion, score in criterion_averages_normalized.items():
                st.metric(criterion.title(), f"{score:.2f}/10")
        
        # Show overall average (normalized)
        if criterion_averages_normalized:
            overall_avg = sum(criterion_averages_normalized.values()) / len(criterion_averages_normalized)
            st.metric("**Overall Average (Normalized)**", f"{overall_avg:.2f}/10")

def detailed_results_tab():
    """Detailed results tab content"""
    st.subheader("Detailed Evaluation Results")
    
    # Get criteria from session state
    criteria = [c['name'] for c in st.session_state.evaluation_criteria]
    results = st.session_state.evaluation_results.get("results", {})
    
    # Display results for each model
    for model_name, model_data in results.items():
        parsed_evaluation = model_data.get("parsed_evaluation", {})
        
        with st.expander(f"{model_name} Evaluation", expanded=True):
            # Display scores (raw and normalized)
            score_cols = st.columns(len(criteria))
            for i, criterion in enumerate(criteria):
                with score_cols[i]:
                    score = parsed_evaluation.get(criterion, "N/A")
                    max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
                    
                    # Handle different score formats consistently
                    if isinstance(score, str):
                        # Try to convert string to number
                        if score.replace('.', '').replace('-', '').isdigit():
                            raw_score = float(score)
                            normalized_score = (raw_score / max_score) * 10
                            st.metric(criterion.title(), f"{raw_score}/{max_score}")
                            st.caption(f"Normalized: {normalized_score:.1f}/10")
                        else:
                            # Try to extract number from string
                            number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                            if number_match:
                                raw_score = float(number_match.group(1))
                                normalized_score = (raw_score / max_score) * 10
                                st.metric(criterion.title(), f"{raw_score}/{max_score}")
                                st.caption(f"Normalized: {normalized_score:.1f}/10")
                            else:
                                st.write(f"**{criterion.title()}:** {score}")
                    elif isinstance(score, (int, float)):
                        raw_score = float(score)
                        normalized_score = (raw_score / max_score) * 10
                        st.metric(criterion.title(), f"{raw_score}/{max_score}")
                        st.caption(f"Normalized: {normalized_score:.1f}/10")
                    else:
                        st.write(f"**{criterion.title()}:** {score}")
            
            # Display feedback
            feedback = parsed_evaluation.get('feedback', 'No feedback provided')
            
            st.write("**Feedback:**")
            st.write(feedback)

def statistics_tab():
    """Statistics tab content"""
    st.subheader("Statistical Analysis")
    
    # Get criteria from session state
    criteria = [c['name'] for c in st.session_state.evaluation_criteria]
    results = st.session_state.evaluation_results.get("results", {})
    
    # Prepare data for analysis (raw and normalized)
    scores_data = []
    normalized_scores_data = []
    
    for model_name, model_data in results.items():
        parsed_evaluation = model_data.get("parsed_evaluation", {})
        model_scores = {'Model': model_name}
        model_scores_normalized = {'Model': model_name}
        
        for criterion in criteria:
            score = parsed_evaluation.get(criterion, 0)
            max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
            
            # Handle different score formats consistently
            if isinstance(score, str):
                # Try to convert string to number
                if score.replace('.', '').replace('-', '').isdigit():
                    raw_score = float(score)
                    normalized_score = (raw_score / max_score) * 10
                    model_scores[criterion] = raw_score
                    model_scores_normalized[criterion] = normalized_score
                else:
                    # Try to extract number from string
                    number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                    if number_match:
                        raw_score = float(number_match.group(1))
                        normalized_score = (raw_score / max_score) * 10
                        model_scores[criterion] = raw_score
                        model_scores_normalized[criterion] = normalized_score
                    else:
                        model_scores[criterion] = 0
                        model_scores_normalized[criterion] = 0
            elif isinstance(score, (int, float)):
                raw_score = float(score)
                normalized_score = (raw_score / max_score) * 10
                model_scores[criterion] = raw_score
                model_scores_normalized[criterion] = normalized_score
            else:
                model_scores[criterion] = 0
                model_scores_normalized[criterion] = 0
        
        scores_data.append(model_scores)
        normalized_scores_data.append(model_scores_normalized)
    
    if scores_data:
        df_stats = pd.DataFrame(scores_data)
        df_stats_normalized = pd.DataFrame(normalized_scores_data)
        
        # Display both raw and normalized data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Scores Analysis")
            # Correlation matrix (if multiple criteria)
            if len(criteria) > 1:
                st.write("**Criteria Correlation (Raw Scores)**")
                corr_matrix = df_stats[criteria].corr()
                fig = px.imshow(corr_matrix, 
                                title="Raw Scores Correlation",
                                color_continuous_scale='RdBu_r',
                                aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.write("**Summary Statistics (Raw Scores)**")
            st.dataframe(df_stats.describe())
        
        with col2:
            st.subheader("Normalized Scores Analysis (0-10 Scale)")
            # Correlation matrix (if multiple criteria)
            if len(criteria) > 1:
                st.write("**Criteria Correlation (Normalized Scores)**")
                corr_matrix_norm = df_stats_normalized[criteria].corr()
                fig = px.imshow(corr_matrix_norm, 
                                title="Normalized Scores Correlation",
                                color_continuous_scale='RdBu_r',
                                aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.write("**Summary Statistics (Normalized Scores)**")
            st.dataframe(df_stats_normalized.describe())
        
        # Overall comparison
        st.subheader("Overall Model Performance Comparison")
        
        # Calculate overall scores for each model
        overall_scores = []
        for model_name, model_data in results.items():
            parsed_evaluation = model_data.get("parsed_evaluation", {})
            total_normalized = 0
            count = 0
            
            for criterion in criteria:
                score = parsed_evaluation.get(criterion, 0)
                max_score = next((c.get('max_score', 10) for c in st.session_state.evaluation_criteria if c['name'] == criterion), 10)
                
                if isinstance(score, (int, float)) or (isinstance(score, str) and score.replace('.', '').replace('-', '').isdigit()):
                    raw_score = float(score) if isinstance(score, (int, float)) else float(score)
                    normalized_score = (raw_score / max_score) * 10
                    total_normalized += normalized_score
                    count += 1
            
            if count > 0:
                overall_scores.append({
                    'Model': model_name,
                    'Overall Score (Normalized)': total_normalized / count
                })
        
        if overall_scores:
            df_overall = pd.DataFrame(overall_scores)
            df_overall = df_overall.sort_values('Overall Score (Normalized)', ascending=False)
            
            fig = px.bar(df_overall, x='Model', y='Overall Score (Normalized)',
                        title="Overall Model Performance (Normalized to 0-10 Scale)",
                        color='Overall Score (Normalized)',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Model Rankings:**")
            st.dataframe(df_overall)

def export_tab():
    """Export tab content"""
    st.subheader("Data Export")
    
    # Get criteria from session state
    criteria = [c['name'] for c in st.session_state.evaluation_criteria]
    results = st.session_state.evaluation_results.get("results", {})
    
    # Prepare CSV data
    csv_data = []
    for model_name, model_data in results.items():
        response = model_data.get("response", {})
        row = {
            'evaluator': model_name,
            'system_prompt': st.session_state.system_prompt,
            'user_prompt': st.session_state.user_prompt
        }
        
        parsed_evaluation = model_data.get("parsed_evaluation", {})
        for criterion in criteria:
            score = parsed_evaluation.get(criterion, 0)
            # Handle different score formats consistently
            if isinstance(score, str):
                # Try to convert string to number
                if score.replace('.', '').replace('-', '').isdigit():
                    row[f'criterion_{criterion}'] = float(score)
                else:
                    # Try to extract number from string
                    number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                    if number_match:
                        row[f'criterion_{criterion}'] = float(number_match.group(1))
                    else:
                        row[f'criterion_{criterion}'] = score
            elif isinstance(score, (int, float)):
                row[f'criterion_{criterion}'] = float(score)
            else:
                row[f'criterion_{criterion}'] = score
        row['feedback'] = parsed_evaluation.get('feedback', '')
        
        csv_data.append(row)
    
    df_export = pd.DataFrame(csv_data)
    
    # Display preview
    st.subheader("Data Preview")
    st.dataframe(df_export)
    
    # Download button
    csv_string = df_export.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_string,
        file_name="prompt_evaluation_results.csv",
        mime="text/csv"
    )
    
    # JSON export
    export_data = {
        'evaluations': results,
        'system_prompt': st.session_state.system_prompt,
        'user_prompt': st.session_state.user_prompt,
        'evaluation_criteria': st.session_state.evaluation_criteria
    }
    json_string = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download JSON",
        data=json_string,
        file_name="prompt_evaluation_results.json",
        mime="application/json"
    )

def data_export_page():
    """Data export page"""
    st.header("Data Export")
    
    if not st.session_state.evaluation_results:
        st.info("No data available for export. Generate evaluations first.")
        return
    
    # Create tabs for different export options
    tab1, tab2, tab3 = st.tabs(["CSV Export", "JSON Export", "Folder Export"])
    
    with tab1:
        st.subheader("CSV Export")
        
        # Prepare CSV data
        csv_data = []
        criteria = [c['name'] for c in st.session_state.evaluation_criteria]
        results = st.session_state.evaluation_results.get("results", {})
        
        for model_name, model_data in results.items():
            parsed_evaluation = model_data.get("parsed_evaluation", {})
            row = {
                'evaluator': model_name,
                'system_prompt': st.session_state.system_prompt,
                'user_prompt': st.session_state.user_prompt
            }
            
            for criterion in criteria:
                score = parsed_evaluation.get(criterion, 0)
                # Handle different score formats consistently
                if isinstance(score, str):
                    # Try to convert string to number
                    if score.replace('.', '').replace('-', '').isdigit():
                        row[f'criterion_{criterion}'] = float(score)
                    else:
                        # Try to extract number from string
                        number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                        if number_match:
                            row[f'criterion_{criterion}'] = float(number_match.group(1))
                        else:
                            row[f'criterion_{criterion}'] = score
                elif isinstance(score, (int, float)):
                    row[f'criterion_{criterion}'] = float(score)
                else:
                    row[f'criterion_{criterion}'] = score
            row['feedback'] = parsed_evaluation.get('feedback', '')
            
            csv_data.append(row)
        
        df_export = pd.DataFrame(csv_data)
        
        # Display preview
        st.subheader("Data Preview")
        st.dataframe(df_export)
        
        # Download button
        csv_string = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_string,
            file_name="prompt_evaluation_results.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("JSON Export")
        
        # Create a single JSON file with all evaluations
        json_data = {
            'evaluations': st.session_state.evaluation_results.get("results", {}),
            'system_prompt': st.session_state.system_prompt,
            'user_prompt': st.session_state.user_prompt,
            'evaluation_criteria': st.session_state.evaluation_criteria
        }
        
        json_string = json.dumps(json_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_string,
            file_name="prompt_evaluation_results.json",
            mime="application/json"
        )
    
    with tab3:
        st.subheader("Folder Export")
        
        # Create a comprehensive export folder with all data
        if st.button("📁 Create Export Folder", type="primary"):
            try:
                # Create timestamp for unique folder name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_folder = f"prompt_evaluation_export_{timestamp}"
                
                # Create the export directory
                os.makedirs(export_folder, exist_ok=True)
                
                # 1. Export CSV data
                csv_data = []
                criteria = [c['name'] for c in st.session_state.evaluation_criteria]
                results = st.session_state.evaluation_results.get("results", {})
                
                for model_name, model_data in results.items():
                    parsed_evaluation = model_data.get("parsed_evaluation", {})
                    row = {
                        'evaluator': model_name,
                        'system_prompt': st.session_state.system_prompt,
                        'user_prompt': st.session_state.user_prompt
                    }
                    
                    for criterion in criteria:
                        score = parsed_evaluation.get(criterion, 0)
                        # Handle different score formats consistently
                        if isinstance(score, str):
                            if score.replace('.', '').replace('-', '').isdigit():
                                row[f'criterion_{criterion}'] = float(score)
                            else:
                                number_match = re.search(r'(\d+(?:\.\d+)?)', score)
                                if number_match:
                                    row[f'criterion_{criterion}'] = float(number_match.group(1))
                                else:
                                    row[f'criterion_{criterion}'] = score
                        elif isinstance(score, (int, float)):
                            row[f'criterion_{criterion}'] = float(score)
                        else:
                            row[f'criterion_{criterion}'] = score
                    
                    row['feedback'] = parsed_evaluation.get('feedback', '')
                    csv_data.append(row)
                
                df_export = pd.DataFrame(csv_data)
                csv_path = os.path.join(export_folder, "evaluation_results.csv")
                df_export.to_csv(csv_path, index=False)
                
                # 2. Export JSON data
                json_data = {
                    'evaluations': results,
                    'system_prompt': st.session_state.system_prompt,
                    'user_prompt': st.session_state.user_prompt,
                    'evaluation_criteria': st.session_state.evaluation_criteria,
                    'export_timestamp': timestamp
                }
                json_path = os.path.join(export_folder, "evaluation_results.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # 3. Export individual model responses
                responses_folder = os.path.join(export_folder, "model_responses")
                os.makedirs(responses_folder, exist_ok=True)
                
                for model_name, model_data in results.items():
                    model_file = os.path.join(responses_folder, f"{model_name.replace('/', '_')}.json")
                    with open(model_file, 'w') as f:
                        json.dump(model_data, f, indent=2)
                
                # 4. Export configuration
                config_data = {
                    'system_prompt': st.session_state.system_prompt,
                    'user_prompt': st.session_state.user_prompt,
                    'evaluation_criteria': st.session_state.evaluation_criteria,
                    'models_used': [model.name for model in st.session_state.models],
                    'export_timestamp': timestamp
                }
                config_path = os.path.join(export_folder, "configuration.json")
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # 5. Create README file
                readme_content = f"""# Prompt Evaluation Export

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents:
- `evaluation_results.csv`: Main results in CSV format
- `evaluation_results.json`: Complete evaluation data in JSON format
- `configuration.json`: Configuration used for the evaluation
- `model_responses/`: Individual model response files
- `README.md`: This file

## Evaluation Summary:
- System Prompt: {st.session_state.system_prompt[:100]}...
- User Prompt: {st.session_state.user_prompt[:100]}...
- Models Evaluated: {len(results)}
- Criteria: {len(criteria)}

## Files:
1. **evaluation_results.csv**: Tabular data with scores and feedback
2. **evaluation_results.json**: Complete structured data
3. **configuration.json**: Evaluation settings and prompts
4. **model_responses/**: Raw responses from each model
"""
                
                readme_path = os.path.join(export_folder, "README.md")
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # 6. Create ZIP file
                zip_path = f"{export_folder}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(export_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, export_folder)
                            zipf.write(file_path, arcname)
                
                # 7. Clean up the folder (keep only ZIP)
                shutil.rmtree(export_folder)
                
                # 8. Provide download
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                st.success(f"✅ Export folder created successfully!")
                st.download_button(
                    label="📥 Download Export Folder (ZIP)",
                    data=zip_data,
                    file_name=f"{export_folder}.zip",
                    mime="application/zip"
                )
                
                # Clean up ZIP file
                os.remove(zip_path)
                
            except Exception as e:
                st.error(f"❌ Error creating export folder: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")
        
        st.info("""
        **Folder Export Features:**
        - 📊 Complete CSV export with all evaluation data
        - 📄 Full JSON export with raw responses
        - ⚙️ Configuration file with all settings
        - 📁 Individual model response files
        - 📖 README with export summary
        - 📦 All files packaged in a single ZIP download
        """)

# ------------------------------
# 13. MAIN APPLICATION
# ------------------------------
def main():
    """Main application function"""
    # Apply styles
    apply_styles()
    
    # Initialize session state
    init_session_state()
    init_session()
    


    # Initialize backend classes
    if st.session_state.fs_manager_instance is None:
        st.session_state.fs_manager_instance = get_fs_manager()

    if st.session_state.logger_instance is None:
        st.session_state.logger_instance = get_logger(st.session_state.fs_manager_instance)

    # Progress bar placeholders
    if st.session_state.progress_bar_placeholder is None:
        st.session_state.progress_bar_placeholder = st.empty()
    if st.session_state.progress_text_placeholder is None:
        st.session_state.progress_text_placeholder = st.empty()

    progress_bar_widget = st.session_state.progress_bar_placeholder.progress(0)
    progress_text_widget = st.session_state.progress_text_placeholder.markdown("")

    # API client and orchestrator will be initialized when needed in setup page

    # Main UI
    st.title("🤖 LLM Prompt Evaluation Playground")
    st.markdown("Evaluate user prompts based on system prompts and evaluation criteria")
    
    # Initialize page if not set
    if 'page' not in st.session_state:
        st.session_state.page = "Setup"
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Add navigation links with icons
    if st.sidebar.button("🏗️ Setup", use_container_width=True):
        st.session_state.page = "Setup"
    if st.sidebar.button("📊 Model Outputs", use_container_width=True):
        st.session_state.page = "Model Outputs"
    if st.sidebar.button("📋 Evaluation Output", use_container_width=True):
        st.session_state.page = "Evaluation Output"
    if st.sidebar.button("📈 Evaluation Matrix", use_container_width=True):
        st.session_state.page = "Evaluation Matrix"
    if st.sidebar.button("💾 Data Export", use_container_width=True):
        st.session_state.page = "Data Export"

    # Display the selected page
    if st.session_state.page == "Setup":
        setup_page()
    elif st.session_state.page == "Model Outputs":
        outputs_page()
    elif st.session_state.page == "Evaluation Output":
        evaluation_output_page()
    elif st.session_state.page == "Evaluation Matrix":
        evaluation_matrix_page()
    elif st.session_state.page == "Data Export":
        data_export_page()

# ------------------------------
# 13. APPLICATION ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.exception(e) 