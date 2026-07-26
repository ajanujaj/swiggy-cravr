"""Run with: `python -m cravr.run` (requires OPENAI_API_KEY)."""

import uuid

if __package__ in (None, ""):
    # Support direct execution: `python cravr/run.py`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cravr.graph import app
else:
    from .graph import app

from langgraph.types import Command


def _print_assistant_reply(result: dict) -> None:
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        assistant_text = getattr(last, "content", str(last))
        source = getattr(last, "name", None) or "assistant"
        print(f"Assistant [{source}]:", assistant_text)
    else:
        print("Assistant: (no response)")


def main() -> None:
    print("Cravr CLI — type a message, or Ctrl-C/Ctrl-D to exit")
    # A checkpointer (configured on the compiled graph) persists state per
    # thread_id, so the conversation no longer needs to be replayed by hand —
    # each turn only sends the new message and the graph remembers the rest.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    first_turn = True

    try:
        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                continue

            if first_turn:
                graph_input = {
                    "messages": [{"role": "user", "content": user_msg}],
                    "intent_profile": {},
                    "context_profile": {},
                    "history_profile": {},
                    "user_id": "USR-001",
                }
                first_turn = False
            else:
                graph_input = {"messages": [{"role": "user", "content": user_msg}]}

            try:
                result = app.invoke(graph_input, config=config)
            except Exception as exc:
                print("Error running graph:", exc)
                continue

            # decision_agent_node pauses the graph with `interrupt()` whenever
            # it needs more information. Ask the specific question it surfaced
            # and resume with the answer until the graph completes normally.
            while result.get("__interrupt__"):
                pending = result["__interrupt__"][0]
                payload = pending.value
                if isinstance(payload, dict):
                    question = payload.get("question")
                    source = payload.get("agent") or "assistant"
                else:
                    question = payload
                    source = "assistant"
                print(f"Assistant [{source}]:", question)
                clarification = input("You: ").strip()
                try:
                    result = app.invoke(Command(resume=clarification), config=config)
                except Exception as exc:
                    print("Error resuming graph:", exc)
                    break

            _print_assistant_reply(result)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
