"""Run with: `python -m cravr.run` (requires OPENAI_API_KEY)."""

import uuid
from langgraph.types import Command


def _load_app():
    if __package__ in (None, ""):
        # Support direct execution: `python cravr/run.py`.
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cravr.graph import app
    else:
        from .graph import app
    return app


def _print_assistant_reply(result: dict) -> None:
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        # Intent collection can end on the user's last clarification -- never echo that back as an assistant reply.
        if getattr(last, "type", None) in {"human", "user"} or getattr(last, "role", None) in {"human", "user"}:
            return
        assistant_text = getattr(last, "content", str(last))
        source = getattr(last, "name", None) or "assistant"
        print(f"Assistant [{source}]:", assistant_text)
    else:
        print("Assistant: (no response)")


def _resume_until_complete(result: dict, config: dict, app) -> dict | None:
    """Display graph questions and resume them until the current turn ends."""
    while result.get("__interrupt__"):
        pending = result["__interrupt__"][0]
        payload = pending.value
        if isinstance(payload, dict):
            question = payload.get("question")
            source = payload.get("agent") or "assistant"
            options = payload.get("options") or []
        else:
            question = payload
            source = "assistant"
            options = []
        print(f"Assistant [{source}]:", question)
        # Numbered so a one-keystroke reply works; agents also accept the value/label, or free text.
        for position, option in enumerate(options, start=1):
            print(f"    {position}. {option.get('label') or option.get('value')}")

        clarification = input("You: ").strip()
        try:
            result = app.invoke(Command(resume=clarification), config=config)
        except Exception as exc:
            print("Error resuming graph:", exc)
            return None
    return result


def main() -> None:
    print("Cravr CLI — type a message, or Ctrl-C/Ctrl-D to exit")
    app = _load_app()
    # A checkpointer persists state per thread_id, so each turn only sends the new message.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        # Start the graph before collecting free text, so intent_discovery's opening question shows immediately.
        try:
            result = app.invoke(
                {
                    # No hardcoded user_id -- Swiggy's OAuth token is the real, single identity.
                    "messages": [],
                    "intent_profile": {},
                    "context_profile": {},
                    "history_profile": {},
                },
                config=config,
            )
        except Exception as exc:
            print("Error starting conversation:", exc)
            result = None

        if result is not None:
            # The opening exchange can resolve all the way to a recommendation via clarification
            # rounds alone, before the free-typed loop below runs -- print it or it's silently dropped.
            result = _resume_until_complete(result, config, app)
            if result is not None:
                _print_assistant_reply(result)

        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                continue

            graph_input = {"messages": [{"role": "user", "content": user_msg}]}

            try:
                result = app.invoke(graph_input, config=config)
            except Exception as exc:
                print("Error running graph:", exc)
                continue

            result = _resume_until_complete(result, config, app)
            if result is None:
                continue

            _print_assistant_reply(result)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
