# LLM Playground

A simple Streamlit app to evaluate user prompts using multiple large language models (LLMs) via OpenRouter. Configure your own evaluation criteria and compare model responses side by side.

## Features
- Evaluate prompts with several LLMs (OpenAI, Google, Anthropic, DeepSeek, Mistral)
- Customize system and user prompts
- Define your own evaluation criteria (with score ranges)
- View and export results (CSV/JSON)

## Quick Start
1. **Clone this repo**
   ```bash
   git clone <repository-url>
   cd LLM-Playground
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Get an OpenRouter API key**
   - Sign up at [openrouter.ai/keys](https://openrouter.ai/keys)
4. **Run the app**
   ```bash
   streamlit run llm_playground_app.py
   ```
5. **Enter your API key in the app**

## Usage
- Set your system prompt (e.g., role or context)
- Enter the user prompt to evaluate
- Adjust or add evaluation criteria (e.g., coherence, originality)
- Select models and parameters
- Click "Evaluate" to compare results
- Download results as CSV or JSON

## App Pages
- **Setup**: Configure prompts, criteria, models, and API key. Start an evaluation.
- **Model Outputs**: View the raw responses from each model.
- **Evaluation Output**: See parsed scores and feedback for each model.
- **Evaluation Matrix**: Compare results, see averages, statistics, and visualizations.
- **Data Export**: Download your results in CSV or JSON format.
  
## Example Evaluation Criteria
- **Coherence** (0-10): (10-9: Highly coherent, 8-5: Generally coherent, 4-2: Multiple incoherencies, 1-0: Incoherent)
- **Feasibility** (0-6): (6-5: Fully feasible, 4: Feasible but lacks detail, 3-2: Questionable, 1-0: Unrealistic)
- **Originality** (0-8): (8-7: Clearly original, 6-4: Some innovation, 3-2: Limited originality, 1-0: No originality)
- **Pertinence** (0-6): (6-5: Excellent relevance, 4: Relevant but lacks specificity, 3-2: Weakly connected, 1-0: Poor relevance)
You can add, remove, or edit criteria in the app.

## Supported Models (via OpenRouter)
- Google Gemini
- OpenAI GPT-4o
- Anthropic Claude
- DeepSeek Prover
- Mistral Instruct

---

*Made for easy, transparent LLM prompt evaluation and comparison.* 
