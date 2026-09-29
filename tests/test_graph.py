import os
os.environ.setdefault("CRAVR_DISHES_FIRST", "0")  # question-flow tests; see conftest.py
os.environ.setdefault("CRAVR_WEATHER", "0")
import uuid
import unittest
from unittest.mock import patch

from langgraph.types import Command

from cravr.graph import app
from cravr.schemas import IntentExtraction


class ClarificationContextTests(unittest.TestCase):
    def test_question_is_recorded_in_messages_alongside_the_answer(self):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        class StubLLM:
            def with_structured_output(self, schema):
                class Stub:
                    def invoke(self, messages):
                        return IntentExtraction(craving_tag="comfort-food")

                return Stub()

        stub = StubLLM()
        with patch("cravr.intent.intentAgent.get_llm", return_value=stub), \
             patch("cravr.options.get_llm", side_effect=RuntimeError("no llm in tests")):
            result = app.invoke(
                {
                    "messages": [{"role": "user", "content": "I'm hungry"}],
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
        self.assertIn(question, contents)
        idx = contents.index(question)
        self.assertEqual(contents[idx + 1], "Indian")


class GraphWiringTests(unittest.TestCase):
    def test_every_registered_node_is_reachable_from_start(self):
        graph = app.get_graph()
        node_names = {n for n in graph.nodes if n not in ("__start__", "__end__")}
        reachable = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
        unreachable = node_names - reachable
        self.assertFalse(
            unreachable,
            f"Node(s) {unreachable} are registered but have no edges connecting them to the graph.",
        )

    def test_user_preference_context_and_intent_are_ordered(self):
        graph = app.get_graph()
        edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("user_preference", "context_discovery"), edge_pairs)
        self.assertIn(("context_discovery", "intent_discovery"), edge_pairs)

    def test_decision_agent_always_reaches_the_recommender(self):
        graph = app.get_graph()
        edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("decision_agent", "recommendation_agent"), edge_pairs)


if __name__ == "__main__":
    unittest.main()
