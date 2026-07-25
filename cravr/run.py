"""Run with: `python -m cravr.run` (requires OPENAI_API_KEY)."""

if __package__ in (None, ""):
    # Support direct execution: `python cravr/run.py`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cravr.graph import app
else:
    from .graph import app


def main() -> None:
    print("Cravr CLI — type a message, or Ctrl-C/Ctrl-D to exit")
    # Keep a short conversation history across turns so the graph has context.
    conversation_messages = []
    try:
        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                continue

            conversation_messages.append({"role": "user", "content": user_msg})

            state = {"messages": conversation_messages,
                     "intent_profile": {},
                     "context_profile": {},
                     "history_profile": {},
                     "user_id": "USR-001"}

            try:
                result = app.invoke(state)
            except Exception as exc:
                print("Error running graph:", exc)
                continue

            # Print the assistant's latest response produced by the graph
            messages = result.get("messages", [])
            if messages:
                assistant_text = getattr(messages[-1], "content", str(messages[-1]))
                print("Assistant:", assistant_text)
                # Add assistant reply to history for next turn
                conversation_messages.append({"role": "assistant", "content": assistant_text})
            else:
                print("Assistant: (no response)")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
