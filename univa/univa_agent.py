import asyncio
import json
import os
import traceback
import uuid
from pathlib import Path
from dotenv import load_dotenv
import inspect
from types import SimpleNamespace

from typing import List, Dict, Any, Optional, Sequence, Callable
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SqliteSaver = None

from univa.memory.tools import get_memory_tools
from univa.memory.context import build_memory_context, format_memory_context
from univa.mcp_tools.video_gen import (
    text2video_gen,
    image2video_gen,
    frame2frame_video_gen,
    merge2videos,
)
from univa.mcp_tools.image_gen import (
    text2image_generate,
    image2image_generate,
    sequential_image_gen,
)
from univa.mcp_tools.video_understanding import vision2text_gen

def _init_env():
    base = Path(__file__).resolve().parents[1]
    env_file = base / ".env"
    if not env_file.exists():
        # Proceed even if .env is missing, logic might rely on env vars
        pass
    load_dotenv(dotenv_path=str(env_file), override=False)

_init_env()

from univa.config.config import config

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


def _builtin_tool_callables() -> List[Callable[..., Any]]:
    return [
        text2video_gen,
        image2video_gen,
        frame2frame_video_gen,
        merge2videos,
        text2image_generate,
        image2image_generate,
        sequential_image_gen,
        vision2text_gen,
    ]


def _parse_extra(extra: Optional[str | Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra.strip():
        try:
            return json.loads(extra)
        except Exception:
            return {}
    return {}


def _create_chat_model(
    provider: str,
    model_id: str,
    api_key: Optional[str],
    base_url: Optional[str],
    extra_params: Optional[str | Dict[str, Any]] = None,
) -> ChatOpenAI:
    p = (provider or "").lower()
    if p not in {"openai", "openai_compatible", "vllm", "sglang", "ollama", "azure", "azure_openai"}:
        logger.warning(f"Provider '{provider}' not explicitly supported; using OpenAI-compatible ChatOpenAI.")

    extra = _parse_extra(extra_params)
    # Pull known top-level args to avoid burying them in model_kwargs.
    temperature = extra.pop("temperature", None)
    top_p = extra.pop("top_p", None)
    max_completion_tokens = extra.pop("max_completion_tokens", None)
    presence_penalty = extra.pop("presence_penalty", None)
    frequency_penalty = extra.pop("frequency_penalty", None)

    return ChatOpenAI(
        model=model_id,
        api_key=api_key or None,
        base_url=base_url or None,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        model_kwargs=extra or {},
    )


def _safe_json(val: Any) -> str:
    try:
        return json.dumps(val, ensure_ascii=True)
    except Exception:
        return str(val)


def _normalize_tool_content(val: Any) -> Optional[Dict[str, Any]]:
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if hasattr(val, "model_dump"):
        try:
            return val.model_dump()
        except Exception:
            pass
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except Exception:
            return {"result": val}
    return {"result": val}


def _summarize_tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "success",
        "output_path",
        "output_url",
        "segment_id",
        "clip_id",
        "last_frame_path",
        "message",
        "error",
        "cached",
    ]
    summary = {k: payload.get(k) for k in keys if payload.get(k) is not None}
    if summary:
        return summary
    if "content" in payload and isinstance(payload["content"], dict):
        return _summarize_tool_result(payload["content"])
    return payload


def _trim_messages(messages: Sequence[Any], max_messages: Optional[int]) -> List[Any]:
    if not messages:
        return []
    if not max_messages or max_messages <= 0:
        return list(messages)

    msgs = list(messages)
    system = []
    if msgs and getattr(msgs[0], "type", None) == "system":
        system = [msgs[0]]
        msgs = msgs[1:]
    if len(msgs) <= max_messages:
        return system + msgs
    return system + msgs[-max_messages:]


class ReActAgent:
    def __init__(self, db_file: str = "react_agent.db"):
        self.memory_tools = get_memory_tools()
        self.tool_callables = _builtin_tool_callables()
        self.num_history_messages = 10

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

        tool_call_limit = config.get("react_tool_call_limit", None)
        try:
            tool_call_limit = int(tool_call_limit) if tool_call_limit is not None else None
        except Exception:
            tool_call_limit = None
        self.tool_call_limit = tool_call_limit

        # Build tools first so we can include descriptions in prompt if needed.
        self.tools = self._build_tools()

        formatted_instructions = base_instructions
        if "{tools_description}" in base_instructions:
            formatted_instructions = base_instructions.format(tools_description=self._get_available_tools_description())

        self.model = _create_chat_model(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or None,
            extra_params=extra_params,
        )

        self.checkpointer = self._get_checkpointer(db_file)

        self.graph = create_react_agent(
            self.model,
            self.tools,
            prompt=formatted_instructions,
            pre_model_hook=self._pre_model_hook,
            checkpointer=self.checkpointer,
        )

    def _get_checkpointer(self, db_file: str):
        if SqliteSaver is not None:
            try:
                return SqliteSaver(db_file)
            except Exception as exc:
                logger.warning(f"Failed to initialize SqliteSaver ({db_file}); falling back to MemorySaver: {exc}")
        return MemorySaver()

    def _pre_model_hook(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        trimmed = _trim_messages(messages, self.num_history_messages)
        if trimmed is messages:
            return state
        return {**state, "messages": trimmed}

    def _wrap_callable(self, func: Callable[..., Any]) -> BaseTool:
        desc = (getattr(func, "__doc__", "") or "").strip().splitlines()
        description = desc[0].strip() if desc else func.__name__
        if inspect.iscoroutinefunction(func):
            return StructuredTool.from_function(
                coroutine=func,
                name=func.__name__,
                description=description or func.__name__,
                parse_docstring=False,
            )
        return StructuredTool.from_function(
            func=func,
            name=func.__name__,
            description=description or func.__name__,
            parse_docstring=False,
        )

    def _build_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []
        for func in self.tool_callables:
            tools.append(self._wrap_callable(func))
        for func in self.memory_tools:
            tools.append(self._wrap_callable(func))
        return tools

    def _get_available_tools_description(self) -> str:
        tools_info = []
        for tool in self.tools:
            tools_info.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_info) if tools_info else "No tools available"

    def _make_config(self, session_id: Optional[str]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"configurable": {"thread_id": session_id or "default"}}
        if self.tool_call_limit is not None:
            cfg["recursion_limit"] = self.tool_call_limit
        return cfg

    async def run(
        self,
        input_text: str,
        session_id: str = None,
        stream: bool = False,
        stream_events: bool = False,
    ):
        config = self._make_config(session_id)
        inputs = {"messages": [HumanMessage(content=input_text)]}
        if stream:
            if stream_events:
                return self.graph.astream_events(inputs, config=config, version="v2")
            return self.graph.astream(inputs, config=config)

        output = await self.graph.ainvoke(inputs, config=config)
        content = self._extract_response_content(output)
        return SimpleNamespace(content=content, output=output)

    async def get_state(self, session_id: str):
        try:
            return await self.graph.aget_state(self._make_config(session_id))
        except Exception:
            return None

    def _extract_response_content(self, output: Dict[str, Any]) -> str:
        messages = output.get("messages", []) if isinstance(output, dict) else []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                payload = _normalize_tool_content(msg.content)
                if payload is None:
                    continue
                summary = _summarize_tool_result(payload)
                return _safe_json(summary)
        return ""


class ReActSystem:
    def __init__(self, db_file: str = "video_agent_system"):
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
        self.agent = ReActAgent(f"{self.db_file}.db")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
    
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
            # Streaming response from LangGraph (with events so we can surface tool calls/results)
            stream_gen = self.agent.run(
                request_with_ctx,
                session_id=session_id,
                stream=True,
                stream_events=True,
            )
            if inspect.isawaitable(stream_gen):
                stream_gen = await stream_gen

            got_model_content = False
            last_tool_summary: Optional[Dict[str, Any]] = None
            reported_tool_summary = False

            async for event in stream_gen:
                content = ""
                event_type = event.get("event") if isinstance(event, dict) else None
                data = event.get("data", {}) if isinstance(event, dict) else {}

                if event_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    chunk_content = getattr(chunk, "content", None) if chunk is not None else None
                    if chunk_content:
                        content = chunk_content
                        got_model_content = True
                elif event_type == "on_chat_model_end":
                    output = data.get("output")
                    output_content = getattr(output, "content", None) if output is not None else None
                    if output_content:
                        content = output_content
                        got_model_content = True
                elif event_type == "on_tool_start":
                    tool_name = event.get("name") or data.get("name")
                    tool_args = data.get("input") or {}
                    if tool_name:
                        yield {'type': 'tool_start', 'name': tool_name, 'args': tool_args}

                elif event_type == "on_tool_end":
                    tool_name = event.get("name") or data.get("name")
                    payload = _normalize_tool_content(data.get("output"))
                    if payload is not None:
                        summary = _summarize_tool_result(payload)
                        last_tool_summary = summary
                        if tool_name:
                            # summary = {"tool": tool_name, **summary} # Optional: include name in summary if needed
                            yield {'type': 'tool_end', 'name': tool_name, 'output': summary}
                        reported_tool_summary = True

                if content:
                    yield {'type': 'content', 'content': content}

            if not last_tool_summary or not (
                last_tool_summary.get("output_path") or last_tool_summary.get("output_url")
            ):
                session_summary = await self._extract_last_tool_summary_from_state(session_id)
                if session_summary:
                    last_tool_summary = session_summary
                    if not reported_tool_summary:
                        yield {
                            'type': 'content',
                            'content': f"\n[tool_result] {_safe_json(last_tool_summary)}\n",
                        }

            if not got_model_content and last_tool_summary:
                yield {
                    'type': 'content',
                    'content': f"\nResult: {_safe_json(last_tool_summary)}\n",
                }
            yield {'type': 'finish', 'session_id': session_id}
            
        except Exception as e:
            logger.error(f"[Stream] Error in execute_task_stream: {e}")
            logger.error(traceback.format_exc())
            yield {
                'type': 'error',
                'content': str(e)
            }

    async def _extract_last_tool_summary_from_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.agent:
            return None
        snapshot = await self.agent.get_state(session_id)
        if not snapshot:
            return None
        messages = snapshot.values.get("messages", []) if hasattr(snapshot, "values") else []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                payload = _normalize_tool_content(msg.content)
                if payload is None:
                    continue
                summary = _summarize_tool_result(payload)
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    summary = {"tool": tool_name, **summary}
                return summary
        return None


async def initialize_global_agents() -> ReActSystem:
    global_system = ReActSystem()
    await global_system.__aenter__()
    
    logger.info("Global ReActSystem initialized")
    
    return global_system


async def main():
    # CLI logging: write to file, keep console clean.
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "cli"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cli.log"
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logging.captureWarnings(True)

    # Import rich here to ensure it's available
    try:
        from rich.console import Console, Group
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text
        from rich.json import JSON
    except ImportError:
        print("Error: 'rich' library not found. Please install it with 'pip install rich'.")
        return

    console = Console()
    system = await initialize_global_agents()
    
    try:
        def _banner() -> None:
            console.print(Panel.fit(
                "[bold cyan]UniVA ReAct CLI[/bold cyan]\n"
                "[dim]Type /help for commands. Type 'exit' or 'quit' to stop.[/dim]",
                border_style="cyan"
            ))

        def _print_context() -> None:
            win = f"{t_start},{t_end}" if t_start is not None and t_end is not None else "none"
            pid = project_id if project_id else "none"
            info = f"[dim]Session: [bold]{session_id}[/] | Project: [bold]{pid}[/] | Window: [bold]{win}[/] | Pad: {pad_sec}s | MaxSeg: {max_segments}[/]"
            console.print(info)

        _banner()
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        project_id = None
        t_start = None
        t_end = None
        pad_sec = 8.0
        max_segments = 12

        while True:
            try:
                _print_context()
                input_prompt = console.input("[bold green]>>> [/]")
                
                if input_prompt.lower() in ['exit', 'quit']:
                    break
                
                if not input_prompt.strip():
                    continue

                if input_prompt.startswith("/"):
                    parts = input_prompt.strip().split()
                    cmd = parts[0].lower()
                    if cmd == "/help":
                        console.print(Panel(
                            "  /session <id>        set session id\n"
                            "  /new                 create a new session id\n"
                            "  /project <id>        set project id for memory context\n"
                            "  /window <t0> <t1>    set focus window seconds\n"
                            "  /clearwindow         clear focus window\n"
                            "  /pad <sec>           set context pad seconds (default 8.0)\n"
                            "  /maxseg <n>          set max segments when no window (default 12)",
                            title="Commands",
                            border_style="yellow"
                        ))
                        continue
                    if cmd == "/session":
                        if len(parts) >= 2:
                            session_id = parts[1]
                            console.print(f"[green]session_id set to {session_id}[/]")
                        else:
                            console.print(f"session_id: {session_id}")
                        continue
                    if cmd == "/new":
                        session_id = f"session_{uuid.uuid4().hex[:8]}"
                        console.print(f"[green]session_id set to {session_id}[/]")
                        continue
                    if cmd == "/project" and len(parts) >= 2:
                        project_id = parts[1]
                        console.print(f"[green]project_id set to {project_id}[/]")
                        continue
                    if cmd == "/window" and len(parts) >= 3:
                        try:
                            t_start = float(parts[1])
                            t_end = float(parts[2])
                            console.print(f"[green]window set to t_start={t_start} t_end={t_end}[/]")
                        except ValueError:
                            console.print("[red]invalid /window args, expected floats[/]")
                        continue
                    if cmd == "/clearwindow":
                        t_start = None
                        t_end = None
                        console.print("[green]window cleared[/]")
                        continue
                    if cmd == "/pad" and len(parts) >= 2:
                        try:
                            pad_sec = float(parts[1])
                            console.print(f"[green]pad_sec set to {pad_sec}[/]")
                        except ValueError:
                            console.print("[red]invalid /pad arg, expected float[/]")
                        continue
                    if cmd == "/maxseg" and len(parts) >= 2:
                        try:
                            max_segments = int(parts[1])
                            console.print(f"[green]max_segments set to {max_segments}[/]")
                        except ValueError:
                            console.print("[red]invalid /maxseg arg, expected int[/]")
                        continue
                    console.print("[red]unknown command, use /help[/]")
                    continue

                console.print(f"[dim]Processing...[/]")
                
                # Renderables accumulator for the chat history of this turn
                renderables = []
                current_text = ""
                active_tool = None
                
                def generate_group():
                    items = list(renderables)
                    if current_text:
                        items.append(Markdown(current_text))
                    if active_tool:
                        items.append(active_tool)
                    return Group(*items)

                with Live(generate_group(), console=console, refresh_per_second=10) as live:
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
                            current_text += event['content']
                            live.update(generate_group())
                        
                        elif event['type'] == 'tool_start':
                            tool_name = event.get('name', 'Unknown Tool')
                            tool_args = event.get('args', {})
                            # Show spinner
                            active_tool = Spinner("dots", text=f"Running [bold cyan]{tool_name}[/]...", style="cyan")
                            live.update(generate_group())
                        
                        elif event['type'] == 'tool_end':
                            # Convert output to a nice panel
                            tool_name = event.get('name', 'Tool')
                            output = event.get('output', {})
                            
                            # Add current text to renderables so it "commits" before the tool output
                            if current_text:
                                renderables.append(Markdown(current_text))
                                current_text = ""
                            
                            # Create a panel for the tool result
                            # Try to format JSON if possible, otherwise string
                            try:
                                json_str = json.dumps(output, indent=2)
                                content = JSON(json_str)
                            except Exception:
                                content = str(output)
                                
                            panel = Panel(
                                content,
                                title=f"[bold magenta]Tool Result: {tool_name}[/]",
                                border_style="magenta",
                                expand=False
                            )
                            renderables.append(panel)
                            active_tool = None
                            live.update(generate_group())

                        elif event['type'] == 'error':
                            console.print(f"[bold red]Error:[/bold red] {event['content']}")
                            break
                            
                console.print() # Newline after turn

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"error: {e}")
                console.print(f"[bold red]System Error:[/bold red] {e}")
                console.print_exception()
    finally:
        await system.__aexit__(None, None, None)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()


    asyncio.run(main())
