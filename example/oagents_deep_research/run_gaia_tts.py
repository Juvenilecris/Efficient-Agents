#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import os
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import logging
import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from typing import Dict, List

from scripts.scorer import question_scorer
from scripts.reformulator import prepare_response
from scripts.searcher import SearchTool
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.audio_inspector_tool import AudioInspectorTool
from scripts.visual_inspector_tool import VisualInspectorTool
from scripts.async_web_crawler import (
    CrawlerReadTool,
    CrawlerArchiveSearchTool,
    SimpleCrawler,
)
from scripts.automodel import get_api_model, process_selected_tasks_param, prepare_model_kwargs

from agent_kb.agent_kb_utils import AKBClient, call_model

from smolagents.memory import ActionStep, PlanningStep, TaskStep
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
)

# Import utilities for cost tracking
from smolagents.verify_function import reset_verify_function_cost_tracker, get_cumulative_verify_cost_details
# Assuming OpenAIEmbedding is correctly placed for this import path
from rag.embeddings.openai_embedding import OpenAIEmbedding, EmbeddingModelType 
# Import new cost trackers from visual_qa
from scripts.visual_qa import (
    reset_idefics_hf_tracker, get_cumulative_idefics_hf_details,
    reset_visualizer_gpt4o_tracker, get_cumulative_visualizer_gpt4o_details
)
# Import new cost trackers from audio_inspector_tool
from scripts.audio_inspector_tool import (
    reset_whisper_cost_tracker, get_cumulative_whisper_cost_details
)
# Import new cost trackers from visual_inspector_tool
from scripts.visual_inspector_tool import (
    reset_visual_inspector_gpt4o_tracker, get_cumulative_visual_inspector_gpt4o_details
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil"
]


parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
env_path = os.path.join(parent_dir, '.env')

load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))

logger = logging.getLogger(__name__)

jsonl_lock = threading.Lock()

logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="gpt-4.1")
    parser.add_argument("--model_id_search", type=str, default="gpt-4.1")
    parser.add_argument("--run_name", type=str, default="init_run")
    parser.add_argument("--debug", default=False, action="store_true")
    # infer params
    parser.add_argument('--planning_interval', type=int, default=1, help='Number of rollouts per state.')
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps for ReAct agent.")
    parser.add_argument("--temperature", default=None, type=float, help= "The temperature for llm generation.")
    parser.add_argument('--top_p', default=None, type=float, help="The top_p for llm generation.")
    parser.add_argument('--reflection', action='store_true', default=True, help='Enable reflection')
    # data selection
    parser.add_argument("--split", type=str, default="validation", choices=['validation','test'])
    parser.add_argument("--level", type=str, default="all", choices=["all", "1", "2", "3"])
    parser.add_argument("--selected-tasks", default=['32102e3e-d12a-4209-9163-7b3a104efe5d'], nargs='*', help="Tasks to run: specify single or multiple indices (--selected-tasks 1 or --selected-tasks 1 2 5), a single task ID, or a path to a text file with one task ID per line")
    # search params
    parser.add_argument('--search_tool_reflection', action='store_true', default=False, help='Enable search tool reflection')
    # plan params
    parser.add_argument('--subtask', action='store_true', default=False, help='Enable subtask')
    parser.add_argument('--static_plan', action='store_true', default=False, help='Use static plan')
    parser.add_argument('--dynamic_update_plan', action='store_true', default=False, help='Use dynamic update plan')
    # TTS params
    parser.add_argument('--n_rollouts', type=int, default=1, help='Number of rollouts per state.')
    parser.add_argument('--search_type', type=str, choices=['BON-wise','Beam-Search','Tree-Search','BON','default'], default='default', help='Type of search algorithm to use.')
    parser.add_argument('--reflection_threshold', type=int, default=2, help='Number of rollouts per state.')
    parser.add_argument('--verify_type', type=str, choices=['list-wise','scoring'], default='list-wise', help='Type of search algorithm to use.')
    parser.add_argument('--result_merging_type', type=str, choices=['list-wise','scoring','voting'], default='list-wise', help='Type of search algorithm to use.')
    # memory params
    parser.add_argument('--summary', action='store_true', default=False, help='Summarize the current step memory')
    parser.add_argument('--use_long_term_memory', action='store_true', default=False, help='Use long-term memory')
    parser.add_argument('--retrieve_key_memory', action='store_true', default=False, help='Retrieve key memory')
    # agent_kb params
    parser.add_argument('--agent_kb', action='store_true', default=False, help='Enable knowledge base retrieval')
    parser.add_argument('--retrieval_type', type=str, choices=["text", "semantic", "hybrid"], default="hybrid", help="Type of retrieval method")
    parser.add_argument('--top_k', type=int, default=3, help="Retrieval params top_k")
    parser.add_argument('--model_id_retrieval', type=str, default="gpt-4.1", help="Agent kb model choice")
    
    return parser.parse_args()

def load_gaia_dataset(args):
    eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)[args.split]
    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{args.split}/" + row["file_name"]
        return row

    eval_ds = eval_ds.map(preprocess_file_paths)
    eval_df = pd.DataFrame(eval_ds)
    return eval_df

def create_agent_hierarchy(model: Model, model_search: Model, args, debug=False):
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    text_limit = 100000

    search_types = ['wiki', 'google', 'baidu', 'bing', 'duckduckgo']
    search_tools = [SearchTool(search_type=st, reflection=args.search_tool_reflection) for st in search_types]
    
    WEB_TOOLS = [
        CrawlerReadTool(crawler),
        CrawlerArchiveSearchTool(crawler),
        TextInspectorTool(model, text_limit),
        VisualInspectorTool(model, text_limit), 
        AudioInspectorTool(model, text_limit), 
    ]
    WEB_TOOLS += search_tools
    manager_agent = CodeAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        debug=debug,
        subtask=args.subtask,
        static_plan=args.static_plan,
        dynamic_update_plan=args.dynamic_update_plan,
        reflection=args.reflection,
        reflection_threshold=args.reflection_threshold,
        verify_type=args.verify_type,
        result_merging_type=args.result_merging_type,
        n_rollouts=args.n_rollouts,
        search_type=args.search_type,
        summary=args.summary,
        use_long_term_memory=args.use_long_term_memory,
        retrieve_key_memory=args.retrieve_key_memory,
        
        agent_kb=args.agent_kb,
        top_k=args.top_k,
        retrieval_type=args.retrieval_type,
    )
    return manager_agent

def append_answer(entry: dict, jsonl_file: str, file_lock) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entry) + "\n"
    with file_lock:
        with open(jsonl_file, "a", encoding="utf-8") as fp:
            fp.write(data)
    assert os.path.exists(jsonl_file), "File not found!"
    logger.info("Answer exported to file: {}".format(jsonl_file.resolve()))

def extract_intermediate_steps(agent):

    intermediate_steps = []
    for memory_step in agent.memory.steps:
        memory_step.model_input_messages = None
        step_dict = memory_step.dict()
        if isinstance(memory_step, ActionStep):
            step_dict['step_type'] = 'action'
            step_dict.pop('model_output_message', None)
        elif isinstance(memory_step, TaskStep):
            step_dict['step_type'] = 'task'
        elif isinstance(memory_step, PlanningStep):
            step_dict['step_type'] = 'planning'
            step_dict.pop('model_output_message_facts', None)
            step_dict.pop('model_output_message_plan', None)
        else:
            step_dict['step_type'] = 'unknown'
        intermediate_steps.append(step_dict)
    return intermediate_steps

def student_retrieval_process(example, args, model_id_retrieval, key, url):

    akb_client = AKBClient()
    
    with open("./agent_kb/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    
    student_agent_reason_template = prompts["student_agent_reason"]
    student_agent_refine_template = prompts["student_agent_refine"]
    
    student_reason = student_agent_reason_template.format(
        user_query=example["question"]
    )
    
    retrieval_method = {
        "hybrid": akb_client.hybrid_search,
        "text": akb_client.text_search,
        "semantic": akb_client.semantic_search
    }[args.retrieval_type]
    
    student_retrieval_results = retrieval_method(student_reason, top_k=args.top_k)

    student_retrieval = ""
    for result in student_retrieval_results:
        student_retrieval += "\nSimilar task:\n"
        student_retrieval += result['query']
        student_retrieval += "\nSuggestions:\n"
        student_retrieval += result['agent_experience']
    student_refine = student_agent_refine_template.format(
        knowledge=student_retrieval
    )
    
    student_suggestions = call_model(query=student_refine, model_name=model_id_retrieval, key=key, url=url)
    
    return student_suggestions, retrieval_method, prompts

def teacher_retrieval_process(example, agent, args, retrieval_method, prompts, model_id_retrieval, key, key_search, url, url_search, output):

    intermediate_steps = extract_intermediate_steps(agent)
    
    annotated_example = {
        "question": example["question"],
        "prediction": output,
        "intermediate_steps": intermediate_steps,
    }
    
    teacher_agent_reason_template = prompts["teacher_agent_reson"]
    teacher_agent_refine_template = prompts["teacher_agent_refine"]
    
    teacher_reason = teacher_agent_reason_template.format(
        agent_log=str(annotated_example)
    )
    summary = call_model(query=teacher_reason, model_name=model_id_retrieval, key=key_search, url=url_search)
    
    log_plan = None
    for memory_step in agent.memory.steps:
        if isinstance(memory_step, PlanningStep):
            step_dict = memory_step.dict()
            log_plan = step_dict.get('plan', '')
            break
    
    teacher_retrieval_results = retrieval_method(example["question"] + (log_plan or '') + summary, top_k=args.top_k)
    
    teacher_retrieval = ""
    for result in teacher_retrieval_results:
        teacher_retrieval += "\nSimilar task:\n"
        teacher_retrieval += result['query']
        teacher_retrieval += "\nSuggestions:\n"
        teacher_retrieval += result['agent_experience']
    
    teacher_refine = teacher_agent_refine_template.format(
        knowledge=teacher_retrieval,
        log_summary=summary
    )
    
    teacher_suggestions = call_model(query=teacher_refine, model_id=model_id_retrieval, key=key, url=url)
    
    return teacher_suggestions

def answer_single_question(example, args, model_id, model_id_search, answers_file, debug=False, retrieval=False):

    text_limit = 100000
    model_name, key, url, model_wrapper = get_api_model(model_id)
    model_name_search, key_search, url_search, model_wrapper_search = get_api_model(model_id_search)

    kwargs = prepare_model_kwargs(model_id, args)
    kwargs_search = prepare_model_kwargs(model_id_search, args)

    model = model_wrapper(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key,
        api_base=url,
        **kwargs
    )

    model_search = model_wrapper_search(
        model_name_search,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key_search,
        api_base=url_search,
        **kwargs_search
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    audio_inspection_tool = AudioInspectorTool(model, text_limit)
    visual_inspection_tool = VisualInspectorTool(model, text_limit)

    agent = create_agent_hierarchy(model, model_search, args, debug)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). 
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if retrieval:
            model_name_retrieval = args.model_id_retrieval
            
            student_suggestions, retrieval_method, prompts = student_retrieval_process(
                example, args, model_name_retrieval, key, url
            )
            
            final_result = agent.run(augmented_question, additional_knowledge=student_suggestions)
            # agent_memory = agent.write_memory_to_messages(summary_mode=True)
            # final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
            output = str(final_result)
            
            semantic_match_template = prompts["semantic_match_prompt"]
            
            output_query = semantic_match_template.format(
                question=example["question"],
                prediction=output,
                true_answer=example["true_answer"]
            )
            
            semantic_check = call_model(query=output_query, model_name=model_name_retrieval, key=key_search, url=url_search)
            
            if (not question_scorer(output, example["true_answer"])) and (semantic_check == "false"):
                teacher_suggestions = teacher_retrieval_process(
                    example, agent, args, retrieval_method, prompts,
                    model_name_retrieval, key, key_search, url, url_search, output
                )
                
                final_result = agent.run(augmented_question, additional_knowledge=teacher_suggestions)
                output = str(final_result)
        else:
            final_result = agent.run(augmented_question)
            output = str(final_result)

        intermediate_steps = extract_intermediate_steps(agent)
        
        intermediate_steps_check = [str(step) for step in agent.memory.steps]
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps_check]) else False
        
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        logger.error(f"Error on task {example['task_id']}\n{e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
        
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
    }
    # --- Collect and add cost summary --- 
    all_costs_summary = {}

    # 1. Manager Agent Model cost
    manager_model_cost = {}
    if hasattr(agent.model, "get_cumulative_cost_details"):
        manager_model_cost = agent.model.get_cumulative_cost_details()
    all_costs_summary["manager_agent_model_cost"] = manager_model_cost

    # 2. Search Agent Model cost (if exists)
    search_agent_model_cost = {}
    if agent.managed_agents and "search_agent" in agent.managed_agents:
        search_agent = agent.managed_agents["search_agent"]
        if hasattr(search_agent.model, "get_cumulative_cost_details"):
            search_agent_model_cost = search_agent.model.get_cumulative_cost_details()
    all_costs_summary["search_agent_model_cost"] = search_agent_model_cost
    
    # Aggregate model costs
    total_model_prompt_tokens = manager_model_cost.get("total_prompt_tokens", 0) + search_agent_model_cost.get("total_prompt_tokens", 0)
    total_model_completion_tokens = manager_model_cost.get("total_completion_tokens", 0) + search_agent_model_cost.get("total_completion_tokens", 0)
    total_model_input_cost = manager_model_cost.get("total_input_cost", 0.0) + search_agent_model_cost.get("total_input_cost", 0.0)
    total_model_output_cost = manager_model_cost.get("total_output_cost", 0.0) + search_agent_model_cost.get("total_output_cost", 0.0)
    total_model_cost_val = manager_model_cost.get("total_cost", 0.0) + search_agent_model_cost.get("total_cost", 0.0)
    
    all_costs_summary["aggregated_model_cost"] = {
        "total_prompt_tokens": total_model_prompt_tokens,
        "total_completion_tokens": total_model_completion_tokens,
        "total_tokens": total_model_prompt_tokens + total_model_completion_tokens,
        "total_input_cost": round(total_model_input_cost, 6),
        "total_output_cost": round(total_model_output_cost, 6),
        "total_cost": round(total_model_cost_val, 6)
    }

    # 3. Verify function cost
    verify_cost = get_cumulative_verify_cost_details()
    all_costs_summary["verify_function_cost"] = verify_cost

    # 4. Embedding costs (from tools)
    total_embedding_tokens_agg = 0
    total_embedding_cost_agg = 0.0
    embedding_details_list = []

    # Helper function to process tools from an agent
    def collect_embedding_costs_from_agent_tools(agent_instance, agent_name_prefix=""):
        nonlocal total_embedding_tokens_agg, total_embedding_cost_agg # Allow modification of outer scope variables
        if hasattr(agent_instance, 'tools'):
            for tool_name, tool_instance in agent_instance.tools.items():
                if hasattr(tool_instance, "embedding_model") and tool_instance.embedding_model is not None and hasattr(tool_instance.embedding_model, "get_cumulative_embedding_cost_details"):
                    emb_cost_details = tool_instance.embedding_model.get_cumulative_embedding_cost_details()
                    embedding_details_list.append({f"{agent_name_prefix}{tool_name}": emb_cost_details})
                    total_embedding_tokens_agg += emb_cost_details.get("total_embedding_tokens", 0)
                    total_embedding_cost_agg += emb_cost_details.get("total_embedding_cost", 0.0)
                    # Resetting is handled by agent.run() now, so not needed here per tool after collection

    # Collect from manager agent's tools
    collect_embedding_costs_from_agent_tools(agent, agent_name_prefix="manager_")

    # Collect from search agent's tools (if exists)
    if agent.managed_agents and "search_agent" in agent.managed_agents:
        collect_embedding_costs_from_agent_tools(agent.managed_agents["search_agent"], agent_name_prefix="search_")

    all_costs_summary["embedding_costs_by_tool"] = embedding_details_list
    all_costs_summary["aggregated_embedding_cost"] = {
        "total_embedding_tokens": total_embedding_tokens_agg,
        "total_embedding_cost": round(total_embedding_cost_agg, 6),
    }

    # 5. Whisper (ASR) costs from AudioInspectorTool
    whisper_costs = get_cumulative_whisper_cost_details()
    all_costs_summary["whisper_asr_cost"] = whisper_costs

    # 6. VisualQA tool costs
    idefics_hf_costs = get_cumulative_idefics_hf_details()
    visualizer_gpt4o_costs = get_cumulative_visualizer_gpt4o_details()
    all_costs_summary["visual_qa_idefics_hf_tokens"] = idefics_hf_costs # Only tokens, no direct $ cost here
    all_costs_summary["visual_qa_visualizer_gpt4o_cost"] = visualizer_gpt4o_costs

    # 7. Visual Inspector fallback GPT-4o cost
    visual_inspector_fallback_costs = get_cumulative_visual_inspector_gpt4o_details()
    all_costs_summary["visual_inspector_fallback_gpt4o_cost"] = visual_inspector_fallback_costs

    # --- Aggregate all tokens from all sources ---
    grand_total_prompt_tokens = 0
    grand_total_completion_tokens = 0
    grand_total_overall_tokens = 0 # This will be the sum of all tokens from all components

    # 1. From aggregated_model_cost
    grand_total_prompt_tokens += all_costs_summary.get("aggregated_model_cost", {}).get("total_prompt_tokens", 0)
    grand_total_completion_tokens += all_costs_summary.get("aggregated_model_cost", {}).get("total_completion_tokens", 0)
    grand_total_overall_tokens += all_costs_summary.get("aggregated_model_cost", {}).get("total_tokens", 0)

    # 2. From verify_function_cost
    grand_total_prompt_tokens += all_costs_summary.get("verify_function_cost", {}).get("total_prompt_tokens", 0)
    grand_total_completion_tokens += all_costs_summary.get("verify_function_cost", {}).get("total_completion_tokens", 0)
    # Assuming verify_function_cost might not have a pre-calculated 'total_tokens'
    grand_total_overall_tokens += all_costs_summary.get("verify_function_cost", {}).get("total_prompt_tokens", 0) + \
                                all_costs_summary.get("verify_function_cost", {}).get("total_completion_tokens", 0)

    # 3. From aggregated_embedding_cost
    # Embeddings typically count all processed tokens as 'prompt' or 'total' tokens.
    embedding_tokens = all_costs_summary.get("aggregated_embedding_cost", {}).get("total_embedding_tokens", 0)
    grand_total_prompt_tokens += embedding_tokens # Add to prompt tokens as they are input to the embedding model
    grand_total_overall_tokens += embedding_tokens
    # No completion tokens for embeddings in this context

    # 4. From visual_qa_idefics_hf_tokens
    grand_total_prompt_tokens += all_costs_summary.get("visual_qa_idefics_hf_tokens", {}).get("total_prompt_tokens", 0)
    grand_total_completion_tokens += all_costs_summary.get("visual_qa_idefics_hf_tokens", {}).get("total_completion_tokens", 0)
    grand_total_overall_tokens += all_costs_summary.get("visual_qa_idefics_hf_tokens", {}).get("total_tokens", 0)

    # 5. From visual_qa_visualizer_gpt4o_cost
    grand_total_prompt_tokens += all_costs_summary.get("visual_qa_visualizer_gpt4o_cost", {}).get("total_prompt_tokens", 0)
    grand_total_completion_tokens += all_costs_summary.get("visual_qa_visualizer_gpt4o_cost", {}).get("total_completion_tokens", 0)
    grand_total_overall_tokens += all_costs_summary.get("visual_qa_visualizer_gpt4o_cost", {}).get("total_tokens", 0)
    
    # 6. From visual_inspector_fallback_gpt4o_cost
    grand_total_prompt_tokens += all_costs_summary.get("visual_inspector_fallback_gpt4o_cost", {}).get("total_prompt_tokens", 0)
    grand_total_completion_tokens += all_costs_summary.get("visual_inspector_fallback_gpt4o_cost", {}).get("total_completion_tokens", 0)
    grand_total_overall_tokens += all_costs_summary.get("visual_inspector_fallback_gpt4o_cost", {}).get("total_tokens", 0)

    all_costs_summary["grand_total_task_tokens"] = {
        "total_prompt_tokens": grand_total_prompt_tokens,
        "total_completion_tokens": grand_total_completion_tokens,
        "total_tokens": grand_total_overall_tokens, # This should be the sum of all tokens from all components
    }
    # --- End of token aggregation ---

    # Grand total cost for the task
    grand_total_cost = (
        all_costs_summary["aggregated_model_cost"].get("total_cost", 0.0) +
        all_costs_summary["verify_function_cost"].get("total_cost", 0.0) +
        all_costs_summary["aggregated_embedding_cost"].get("total_embedding_cost", 0.0) +
        all_costs_summary["whisper_asr_cost"].get("total_cost", 0.0) +
        all_costs_summary["visual_qa_visualizer_gpt4o_cost"].get("total_cost", 0.0) +
        all_costs_summary["visual_inspector_fallback_gpt4o_cost"].get("total_cost", 0.0)
        # Note: idefics_hf_costs doesn't have a direct $ cost in this setup
    )
    all_costs_summary["grand_total_task_cost"] = round(grand_total_cost, 6)
    
    logger.info(f"Task {example['task_id']} Cost Summary: {json.dumps(all_costs_summary, indent=2)}")
    annotated_example["cost_summary"] = all_costs_summary
    # --- End of cost collection ---

    # Prepare the main result example, excluding cost_summary initially
    main_annotated_example = {
        "agent_name": model.model_id, # Ensure model_id is the correct attribute for agent's name
        "question": example["question"],
        "augmented_question": augmented_question, # Usually not needed in final slim output
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps, # Usually not needed in final slim output
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time, # Usually not needed in final slim output
        "end_time": end_time, # Usually not needed in final slim output
        "task": example["task"], # Usually not needed in final slim output
        "task_id": example["task_id"],
        "search_agent_actions": agent.managed_agents['search_agent'].task_records, # Usually not needed in final slim output
    }
    
    # Prepare the cost-specific example
    cost_example = {
        "task_id": example["task_id"],
        "task": example["task"],
        "agent_name": model.model_id, # Include agent name for context if needed
        "agent_error": str(exception) if raised_exception else None, # Keep error for context
        "prediction": output, # Keep prediction for context
        "true_answer": example["true_answer"], # Keep true_answer for context
        "cost_summary": all_costs_summary 
    }
    
    append_answer(annotated_example, answers_file, jsonl_lock)


def get_examples_to_answer(answers_file, eval_df, selected_tasks=None, level='all', debug=False) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    try:
        answer_df = pd.read_json(answers_file, lines=True)
        done_questions = answer_df.get("task_id", []).tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.info("Error when loading records: ", e)
        logger.info("No usable records! ▶️ Starting new.")
        done_questions = []

    if level == 'all':
        filtered_df = eval_df
    else:
        filtered_df = eval_df[eval_df['task'] == level]

    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            filtered_df = eval_df.iloc[selected_tasks]
        else:
            filtered_df = eval_df[eval_df['task_id'].isin(selected_tasks)]
    
    if debug:
        done_questions = []
    return [row.to_dict() for idx, row in filtered_df.iterrows() if row["task_id"] not in done_questions]

def main():
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")
    answers_file = f"output/{args.split}/{args.run_name}.jsonl"

    eval_df = load_gaia_dataset(args)

    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    level = args.level
    tasks_to_run = get_examples_to_answer(answers_file, eval_df, selected_tasks, level, args.debug)
    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            answer_single_question(example, args, args.model_id, args.model_id_search, answers_file, args.debug, args.agent_kb)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(answer_single_question, example, args, args.model_id, args.model_id_search, answers_file, args.debug, args.agent_kb)
                for example in tasks_to_run
            ]
            for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()