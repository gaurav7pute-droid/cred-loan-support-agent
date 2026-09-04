"""
mock_llm.py -- Part 2 Task 7: the required MOCK_LLM for CrewAI, implemented
by extending crewai.llms.base_llm.BaseLLM (CrewAI's documented extension
point for a non-litellm LLM), per the brief's explicit instruction that this
is "more involved than a simple flag."

Both documented pitfalls are addressed explicitly:

  Pitfall 1 (the "Observation:" trap) -- CrewAI's own built-in ReAct
  system-prompt template contains the literal example text
  "Observation: the result of the action". A parser that searches the
  WHOLE conversation for the substring "Observation:" will match that
  template text on the very first call, before any tool has run, and
  silently return placeholder text. Fix used here: `_has_real_observation()`
  looks ONLY at the last message in `messages` -- the message CrewAI itself
  appends after it actually executes a tool -- never at the system prompt,
  which is always an earlier message at that point.

  Pitfall 2 (tool-name substring dispatch) -- picking a tool by checking
  whether e.g. "lookup" is a substring of its name would silently
  misclassify a tool literally named `rag_lookup`. Fix used here:
  `_find_tool_for_arg()` dispatches by inspecting each tool's OWN declared
  JSON-schema `function.parameters.properties` and matches on the argument
  name the calling agent is known to supply ("query" for the Retrieval
  Agent, "record_id" for the Lookup Agent) -- never on the tool's name.

Because this is a MOCK, `supports_function_calling()` returns False so
CrewAI drives the standard text-based ReAct loop (Thought / Action / Action
Input / Observation / Final Answer) against this LLM, which is the code
path the two pitfalls above are actually about.
"""
import ast
import json
import re
from typing import Any, Dict, List, Optional, Union

import env_setup  # noqa: F401 -- must run before crewai is imported (disables telemetry)
from crewai.llms.base_llm import BaseLLM

_RECORD_ID_RE = re.compile(r"\bCRED-\d{4}\b")
_QUERY_MARKER_RE = re.compile(r"QUERY_START>>>(.*?)<<<QUERY_END", re.DOTALL)
# CrewAI's real text-based ReAct executor never populates the `tools=` kwarg
# on BaseLLM.call() (that JSON-schema kwarg is only used on the
# function-calling path, which this MOCK_LLM deliberately opts out of via
# supports_function_calling() == False). Instead it renders each tool's name
# and declared-argument schema directly into the system prompt as plain
# text, e.g.:
#   Tool Name: rag_lookup
#   Tool Arguments: {'query': {'description': ..., 'type': 'str'}}
# so the PITFALL 2 fix below matches tool blocks in that text instead.
_TOOL_BLOCK_RE = re.compile(
    r"Tool Name:\s*(?P<name>\S+)\s*\n\s*Tool Arguments:\s*(?P<args>\{.*?\})\s*\n",
)


class MockCrewLLM(BaseLLM):
    def __init__(self, role: str, model: str = "mock-llm-v1", temperature: Optional[float] = None):
        super().__init__(model=model, temperature=temperature)
        if role not in ("retrieval", "lookup", "composer"):
            raise ValueError(f"MockCrewLLM: unknown role {role!r}")
        self.role = role

    # ------------------------------------------------------------------
    # BaseLLM overrides
    # ------------------------------------------------------------------
    def supports_function_calling(self) -> bool:
        return False  # force the text-based ReAct loop, not JSON tool_calls

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_list(messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return list(messages)

    @staticmethod
    def _has_real_observation(messages: List[Dict[str, str]]) -> bool:
        """PITFALL 1 FIX: only ever look at the LAST message, never scan the
        whole conversation (which would match the system prompt's own
        ReAct template example text)."""
        if not messages:
            return False
        last = messages[-1]
        content = last.get("content", "") or ""
        return last.get("role") != "system" and "Observation:" in content

    @staticmethod
    def _non_system_text(messages: List[Dict[str, str]]) -> str:
        return "\n".join(m.get("content", "") or "" for m in messages if m.get("role") != "system")

    @staticmethod
    def _all_text(messages: List[Dict[str, str]]) -> str:
        return "\n".join(m.get("content", "") or "" for m in messages)

    @staticmethod
    def _find_tool_for_arg(tools: Optional[List[dict]], arg_name: str) -> Optional[str]:
        """PITFALL 2 FIX: dispatch by the tool's declared argument schema,
        never by matching a substring of the tool's name. Handles the
        JSON-schema `tools=` kwarg CrewAI would pass on the function-calling
        path (kept for forward/backward compatibility with other CrewAI
        versions/paths that do populate it)."""
        for t in tools or []:
            try:
                props = t["function"]["parameters"]["properties"]
            except Exception:
                try:
                    props = t["parameters"]["properties"]
                except Exception:
                    continue
            if arg_name in props:
                return t.get("function", t).get("name")
        return None

    @staticmethod
    def _find_tool_for_arg_in_text(text: str, arg_name: str) -> Optional[str]:
        """PITFALL 2 FIX (text-ReAct path): the CrewAI version/execution
        path actually exercised by this project never passes a `tools=`
        JSON-schema kwarg into call() -- it renders each tool's name and
        declared-argument schema into the system prompt as plain text (see
        _TOOL_BLOCK_RE above). Parse those blocks and match on the declared
        argument NAME, never on a substring of the tool's own name (e.g.
        "lookup" must never match a tool literally named `rag_lookup`)."""
        for m in _TOOL_BLOCK_RE.finditer(text):
            try:
                args = ast.literal_eval(m.group("args"))
            except (ValueError, SyntaxError):
                continue
            if isinstance(args, dict) and arg_name in args:
                return m.group("name")
        return None

    @staticmethod
    def _extract_record_id(text: str) -> Optional[str]:
        m = _RECORD_ID_RE.search(text)
        return m.group(0) if m else None

    @staticmethod
    def _extract_last_observation_value(messages: List[Dict[str, str]]) -> str:
        content = messages[-1].get("content", "") or ""
        idx = content.rfind("Observation:")
        return content[idx + len("Observation:"):].strip() if idx >= 0 else content.strip()

    # ------------------------------------------------------------------
    # main entrypoint
    # ------------------------------------------------------------------
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Union[str, Any]:
        msgs = self._as_list(messages)

        if self.role == "retrieval":
            return self._react_step(msgs, tools, arg_name="query", extract_value=self._extract_user_query)
        if self.role == "lookup":
            return self._react_step(msgs, tools, arg_name="record_id", extract_value=self._extract_record_id_or_none)
        return self._compose(msgs)

    def _extract_user_query(self, msgs: List[Dict[str, str]]) -> str:
        text = self._non_system_text(msgs)
        # Prefer the distinctive QUERY_START/...>>>/<<</QUERY_END marker crew.py
        # wraps the real query in -- robust regardless of whatever boilerplate
        # CrewAI's own task/ReAct prompt template adds before or after it.
        m = _QUERY_MARKER_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fallback: first non-empty line (used only if the marker is missing,
        # e.g. when this LLM is driven by something other than crew.py).
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return text.strip() or "policy question"

    def _extract_record_id_or_none(self, msgs: List[Dict[str, str]]) -> str:
        text = self._non_system_text(msgs)
        rid = self._extract_record_id(text)
        return rid or "UNKNOWN"

    def _react_step(self, msgs, tools, arg_name: str, extract_value) -> str:
        if self._has_real_observation(msgs):
            observation = self._extract_last_observation_value(msgs)
            return (
                "Thought: I now have the information I need.\n"
                f"Final Answer: {observation}"
            )

        tool_name = self._find_tool_for_arg(tools, arg_name) or \
            self._find_tool_for_arg_in_text(self._all_text(msgs), arg_name)
        if tool_name is None:
            # no matching tool wired to this agent -- answer directly without a tool call
            return "Thought: No matching tool is available to me.\nFinal Answer: NO_TOOL_AVAILABLE"

        value = extract_value(msgs)
        if arg_name == "record_id" and value == "UNKNOWN":
            return (
                "Thought: The task does not contain a loan application record_id, so there "
                "is nothing to look up.\nFinal Answer: NO_RECORD_ID_PROVIDED"
            )

        action_input = json.dumps({arg_name: value})
        return (
            f"Thought: I should use the {tool_name} tool to answer this.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {action_input}"
        )

    _CONTEXT_SECTION_RE = re.compile(
        r"context you're working with:\s*\n(?P<context>.*?)\n\nBegin!", re.DOTALL
    )

    def _compose(self, msgs: List[Dict[str, str]]) -> str:
        text = self._non_system_text(msgs)
        # CrewAI's task-context prompt wraps the upstream tasks' outputs
        # between "...context you're working with:\n" and "\n\nBegin!" --
        # extract just that section rather than echoing the whole prompt
        # (task description, format instructions, "Begin!" epilogue, etc.)
        # back out as the "final answer".
        m = self._CONTEXT_SECTION_RE.search(text)
        context = m.group("context").strip() if m else text.strip()[-1500:]
        return (
            "Thought: I have the retrieval and lookup results in context.\n"
            f"Final Answer: {context}"
        )
