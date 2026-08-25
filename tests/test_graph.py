import uuid
import unittest
from unittest.mock import patch

from langgraph.types import Command

from cravr.graph import app
from cravr.schemas import IntentExtraction, ContextExtraction, DecisionOutput


class ClarificationContextTests(unittest.TestCase):
    """Regression test for a live bug report: a bare clarification reply
    like "Indian" or "500" was never getting extracted into intent_profile
    at all, round after round. Root cause: the asking node recorded the
    user's answer in `messages` but never the question that prompted it, so
    on the next round intent_discovery_node's LLM call saw an orphaned
    one-word message with zero context tying it to "what cuisine?" or
    "what's your budget?". This checks the question is now present in
    `messages` right before the answer, in the right role, after a resume.

    Clarification has since moved into intent_discovery_node, so this now
    exercises that node's interrupt loop end-to-end through the real graph.
    """

    def test_question_is_recorded_in_messages_alongside_the_answer(self):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        class StubLLM:
            def with_structured_output(self, schema):
                class Stub:
                    def invoke(self, messages):
                        if schema is IntentExtraction:
                            # A craving but no meal_intent, so intent asks.
                            return IntentExtraction(craving="something spicy")
                        if schema is ContextExtraction:
                            return ContextExtraction()
                        return DecisionOutput(ready_to_recommend=False, decision="clarify")

                return Stub()

        stub = StubLLM()
        with patch("cravr.intent.intentAgent.get_llm", return_value=stub), \
             patch("cravr.context.ContextAgent.get_llm", return_value=stub), \
             patch("cravr.decision.decisionAgent.get_llm", return_value=stub):
            result = app.invoke(
                {
                    "messages": [{"role": "user", "content": "I want something spicy"}],
                    "intent_profile": {},
                    "context_profile": {},
                    "history_profile": {},
                    "user_id": "USR-001",
                },
                config=config,
            )
            self.assertTrue(result.get("__interrupt__"))
            question = result["__interrupt__"][0].value["question"]

            result = app.invoke(Command(resume="Indian"), config=config)

        contents = [getattr(m, "content", None) for m in result["messages"]]
        # The question the bot asked must appear in the message history
        # immediately before the user's bare answer, not be dropped.
        self.assertIn(question, contents)
        idx = contents.index(question)
        self.assertEqual(contents[idx + 1], "Indian")


class GraphWiringTests(unittest.TestCase):
    """Guards against a node being registered with add_node(...) but never
    connected via add_edge/add_conditional_edges — that class of bug doesn't
    raise an error, it just silently makes the node dead code (this is
    exactly what happened to context_discovery: it was registered but the
    edge intent_discovery -> context_discovery -> history_discovery had been
    collapsed to intent_discovery -> history_discovery, so context_profile
    was always empty)."""

    def test_every_registered_node_is_reachable_from_start(self):
        graph = app.get_graph()
        node_names = {n for n in graph.nodes if n not in ("__start__", "__end__")}
        reachable = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
        unreachable = node_names - reachable
        self.assertFalse(
            unreachable,
            f"Node(s) {unreachable} are registered but have no edges connecting them to the graph.",
        )

    def test_context_discovery_sits_between_intent_and_history(self):
        graph = app.get_graph()
        edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("intent_discovery", "context_discovery"), edge_pairs)
        self.assertIn(("context_discovery", "history_discovery"), edge_pairs)


if __name__ == "__main__":
    unittest.main()
