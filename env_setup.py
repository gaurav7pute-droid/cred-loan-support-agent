"""
env_setup.py -- must be imported (for its side effects) before `crewai` is
imported anywhere in this project.

CrewAI's own telemetry attempts an outbound network call by default when
crew.kickoff() runs, which would silently violate this capstone's
zero-network-access requirement under MOCK_LLM. Setting these two
environment variables before crewai is imported disables it.
"""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
