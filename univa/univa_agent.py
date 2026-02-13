import asyncio
import json
import os
import re
import traceback
import uuid
from pathlib import Path
from dotenv import load_dotenv
import inspect

from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel, Field
import logging

from agno.agent import Agent
from agno.tools.mcp import MCPTools, MultiMCPTools
from agno.db.sqlite import SqliteDb
from univa.memory.tools import get_memory_tools
from univa.memory.context import build_memory_context, format_memory_context

def _init_env():
    base = Path(__file__).resolve().parents[1]
    env_file = base / ".env"
    if not env_file.exists():
        # Proceed even if .env is missing, logic might rely on env vars
        pass
    load_dotenv(dotenv_path=str(env_file), override=False)

_init_env()

from univa.config.config import config
from univa.utils.model_factory import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt(prompt_name: str) -> str:
    prompt_dir = config.get('prompt_dir')
    if not prompt_dir:
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        
    prompt_path = os.path.join(prompt_dir, f"{prompt_name}.txt")
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""

class ReActAgent:
    def __init__(self, mcp_tools: MultiMCPTools, db_file: str = "react_agent.db"):
        self.mcp_tools = mcp_tools
        self.memory_tools = get_memory_tools()
        
        # Load prompt
        base_instructions = load_prompt("react")
        if not base_instructions:
            base_instructions = (
                "You are an intelligent agent capable of using tools to solve problems.\n"
                "Think step-by-step. Use tools when necessary."
            )
        
        # Get model configuration
        # Prioritize 'react_model_*' keys, fallback to 'act_model_*', then 'plan_model_*'
        provider = config.get('react_model_provider', config.get('act_model_provider', config.get('plan_model_provider', 'openai')))
        model_id = config.get('react_model_id', config.get('act_model_id', config.get('plan_model_id', 'gpt-4o')))
        api_key = config.get('react_model_api_key', config.get('act_model_api_key', config.get('plan_model_api_key', '')))
        base_url = config.get('react_model_base_url', config.get('act_model_base_url', config.get('plan_model_base_url', '')))
        extra_params = config.get('react_model_extra_params', config.get('act_model_extra_params', config.get('plan_model_extra_params', '')))
        
        # Format instructions with tool descriptions if needed
        # Agno might handle tool descriptions automatically, but if the prompt has {tools_description}, we should fill it.
        formatted_instructions = base_instructions
        if "{tools_description}" in base_instructions:
            formatted_instructions = base_instructions.format(tools_description=self._get_available_tools_description())
            
        self.agent = Agent(
            name="Univideo ReAct Agent",
            model=create_model(
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                base_url=base_url or None,
                extra_params=extra_params,
            ),
            tools=[mcp_tools, *self.memory_tools],
            instructions=formatted_instructions,
            # agno>=? uses `db=` (not `storage=`) for persistence.
            db=SqliteDb(db_file=db_file),
            # Include recent chat history in the context sent to the model.
            add_history_to_context=True,
            num_history_messages=10,
            markdown=True,
            # enable_react=True # Assuming Agno defaults to ReAct/FunctionCalling with tools
        )
    
    def _get_available_tools_description(self) -> str:
        tools_info = []
        if hasattr(self.mcp_tools, 'tools') and self.mcp_tools.tools:
            for tool in self.mcp_tools.tools:
                tools_info.append(f"- {tool.name}: {tool.description}")
        if self.memory_tools:
            for tool in self.memory_tools:
                tool_name = getattr(tool, "__name__", str(tool))
                tool_doc = (getattr(tool, "__doc__", "") or "").strip().splitlines()
                tool_desc = tool_doc[0].strip() if tool_doc else ""
                tools_info.append(f"- {tool_name}: {tool_desc}")
        return "\n".join(tools_info) if tools_info else "No tools available"

    async def run(self, input_text: str, session_id: str = None, stream: bool = False):
        # agno.Agent.arun has a dual API:
        # - stream=False -> returns an awaitable (coroutine) that yields RunOutput
        # - stream=True  -> returns an async generator (NOT awaitable)
        result = self.agent.arun(input_text, stream=stream, session_id=session_id)
        if inspect.isawaitable(result):
            return await result
        return result

class ReActSystem:
    def __init__(self, mcp_command: List[str], db_file: str = "video_agent_system"):
        mcp_tools_path = config.get('mcp_tools_path') or str(Path(__file__).resolve().parent)

        # Pass through the current environment so MCP tool subprocesses can read API keys
        # loaded from `.env` (e.g. WAVESPEED_API_KEY, LLM_OPENAI_API_KEY).
        mcp_env = dict(os.environ)
        mcp_env.update(
            {
                "PYTHONPATH": mcp_tools_path,
                "CWD": mcp_tools_path,
            }
        )

        self.mcp_tools = MultiMCPTools(
            commands=mcp_command,
            env={
                **mcp_env
            },
            timeout_seconds=600,
            refresh_connection=True
        )
        self.db_file = db_file
        self.agent: Optional[ReActAgent] = None

    def _inject_memory_context(
        self,
        user_request: str,
        project_id: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        pad_sec: float = 8.0,
        max_segments: int = 12,
    ) -> str:
        if not project_id:
            return user_request
        try:
            ctx = build_memory_context(
                project_id=project_id,
                t_start=t_start,
                t_end=t_end,
                pad_sec=pad_sec,
                max_segments=max_segments,
            )
            if not (ctx.get("segments") or ctx.get("entity_states_at_center")):
                return user_request
            ctx_text = format_memory_context(ctx)
            return f"{ctx_text}\n\nUSER_REQUEST:\n{user_request}"
        except Exception as exc:
            logger.warning(f"Failed to build memory context for project_id={project_id}: {exc}")
            return user_request
    
    async def __aenter__(self):
        await self.mcp_tools.connect()
        self.agent = ReActAgent(self.mcp_tools, f"{self.db_file}.db")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mcp_tools.close()
    
    async def execute_task(
        self,
        session_id: str,
        user_request: str,
        project_id: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        pad_sec: float = 8.0,
        max_segments: int = 12,
    ) -> Dict[str, Any]:
        if not self.agent:
            raise RuntimeError("Agent not initialized. Use 'async with' context manager.")

        request_with_ctx = self._inject_memory_context(
            user_request=user_request,
            project_id=project_id,
            t_start=t_start,
            t_end=t_end,
            pad_sec=pad_sec,
            max_segments=max_segments,
        )
        response = await self.agent.run(request_with_ctx, session_id=session_id, stream=False)
        
        return {
            "content": response.content,
            # "response": response # Return full response if needed for debugging
        }
    
    async def execute_task_stream(
        self,
        session_id: str,
        user_request: str,
        project_id: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        pad_sec: float = 8.0,
        max_segments: int = 12,
    ):
        if not self.agent:
            raise RuntimeError("Agent not initialized. Use 'async with' context manager.")
            
        try:
            logger.info(f"[Stream] Starting task stream for session {session_id}")
            yield {'type': 'content', 'content': 'Using ReAct Agent to process request...\n'}
            
            request_with_ctx = self._inject_memory_context(
                user_request=user_request,
                project_id=project_id,
                t_start=t_start,
                t_end=t_end,
                pad_sec=pad_sec,
                max_segments=max_segments,
            )
            # Streaming response from Agno
            stream_gen = await self.agent.run(request_with_ctx, session_id=session_id, stream=True)
            
            async for chunk in stream_gen:
                content = ""
                # Handle different chunk types from Agno
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                
                if content:
                    yield {'type': 'content', 'content': content}
            
            yield {'type': 'finish', 'session_id': session_id}
            
        except Exception as e:
            logger.error(f"[Stream] Error in execute_task_stream: {e}")
            logger.error(traceback.format_exc())
            yield {
                'type': 'error',
                'content': str(e)
            }


async def initialize_global_agents() -> ReActSystem:
    # load MCP server configurations
    config_path = config.get('mcp_servers_config')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            mcp_config = json.load(f)
        
        mcp_servers = mcp_config.get("mcpServers", {})
        logger.info(f"Loaded {len(mcp_servers)} MCP servers from config")
        
        # construct mcp commands
        mcp_commands = []
        for server_name, server_config in mcp_servers.items():
            command = server_config.get("command", "")
            args = server_config.get("args", [])
            
            # full command with args
            full_command = f"{command} {' '.join(args)}"
            
            mcp_commands.append(full_command)
            logger.info(f"Registered MCP server '{server_name}': {full_command}")
        
        if not mcp_commands:
             mcp_commands = ["npx -y @modelcontextprotocol/server-filesystem /tmp"]
        
    except FileNotFoundError:
        logger.warning(f"MCP config file not found: {config_path}, using default")
        mcp_commands = ["npx -y @modelcontextprotocol/server-filesystem /tmp"]
    except Exception as e:
        logger.error(f"Error loading MCP config: {e}, using default")
        mcp_commands = ["npx -y @modelcontextprotocol/server-filesystem /tmp"]
    
    global_system = ReActSystem(mcp_command=mcp_commands)
    await global_system.__aenter__()
    
    logger.info("Global ReActSystem initialized")
    
    return global_system


async def main():
    system = await initialize_global_agents()
    
    try:
        is_tty = os.isatty(0)
        def _c(s: str, code: str) -> str:
            if not is_tty:
                return s
            return f"\033[{code}m{s}\033[0m"

        def _banner() -> None:
            print(_c("UniVA ReAct CLI", "1;36"))
            print(_c("Type /help for commands. Type 'exit' or 'quit' to stop.", "2"))
            print(_c("-" * 64, "2"))

        def _context_line() -> None:
            win = f"{t_start},{t_end}" if t_start is not None and t_end is not None else "none"
            pid = project_id if project_id else "none"
            print(_c(f"[session={session_id}] [project={pid}] [window={win}] [pad={pad_sec}] [maxseg={max_segments}]", "2"))

        _banner()
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        project_id = None
        t_start = None
        t_end = None
        pad_sec = 8.0
        max_segments = 12
        while True:
            try:
                _context_line()
                input_prompt = input(_c(">>> ", "1;32"))
                if input_prompt.lower() in ['exit', 'quit']:
                    break
                
                if not input_prompt.strip():
                    continue
                if input_prompt.startswith("/"):
                    parts = input_prompt.strip().split()
                    cmd = parts[0].lower()
                    if cmd == "/help":
                        print(_c("Commands:", "1"))
                        print("  /session <id>        set session id")
                        print("  /new                 create a new session id")
                        print("  /project <id>        set project id for memory context")
                        print("  /window <t0> <t1>    set focus window seconds")
                        print("  /clearwindow         clear focus window")
                        print("  /pad <sec>           set context pad seconds (default 8.0)")
                        print("  /maxseg <n>          set max segments when no window (default 12)")
                        continue
                    if cmd == "/session" and len(parts) >= 2:
                        session_id = parts[1]
                        print(f"session_id set to {session_id}")
                        continue
                    if cmd == "/new":
                        session_id = f"session_{uuid.uuid4().hex[:8]}"
                        print(f"session_id set to {session_id}")
                        continue
                    if cmd == "/project" and len(parts) >= 2:
                        project_id = parts[1]
                        print(f"project_id set to {project_id}")
                        continue
                    if cmd == "/window" and len(parts) >= 3:
                        try:
                            t_start = float(parts[1])
                            t_end = float(parts[2])
                            print(f"window set to t_start={t_start} t_end={t_end}")
                        except ValueError:
                            print("invalid /window args, expected floats")
                        continue
                    if cmd == "/clearwindow":
                        t_start = None
                        t_end = None
                        print("window cleared")
                        continue
                    if cmd == "/pad" and len(parts) >= 2:
                        try:
                            pad_sec = float(parts[1])
                            print(f"pad_sec set to {pad_sec}")
                        except ValueError:
                            print("invalid /pad arg, expected float")
                        continue
                    if cmd == "/maxseg" and len(parts) >= 2:
                        try:
                            max_segments = int(parts[1])
                            print(f"max_segments set to {max_segments}")
                        except ValueError:
                            print("invalid /maxseg arg, expected int")
                        continue
                    print("unknown command, use /help")
                    continue

                print(_c("-" * 64, "2"))
                print(
                    _c(
                        f"processing: {input_prompt} ... (session_id: {session_id}, project_id: {project_id}, window: {t_start},{t_end})",
                        "2",
                    )
                )
                
                # Use stream for CLI feedback
                print(_c("Output:", "1"))
                async for event in system.execute_task_stream(
                    session_id,
                    input_prompt,
                    project_id=project_id,
                    t_start=t_start,
                    t_end=t_end,
                    pad_sec=pad_sec,
                    max_segments=max_segments,
                ):
                    if event['type'] == 'content':
                        print(event['content'], end="", flush=True)
                    elif event['type'] == 'error':
                        print(f"\nError: {event['content']}")
                
                print("\n" + _c("-" * 64, "2"))

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"error: {e}")
    finally:
        await system.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
