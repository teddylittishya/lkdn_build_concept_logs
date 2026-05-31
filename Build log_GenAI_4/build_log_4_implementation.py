"""
Build Log: GenAI #4
Implementing the "Snowball" fix from ICLR 2026 Outstanding Paper:
  "LLMs Get Lost In Multi-Turn Conversation"
  Laban, Hayashi, Zhou, Neville — Microsoft Research + Salesforce
  arXiv: 2505.06120

Core finding: LLMs drop ~39% in performance in multi-turn conversations
vs. single-turn, not because they get dumber — but because they anchor
to wrong early answers and never recover ("lost in conversation phenomenon").

The paper's SNOWBALL mitigation: at each turn, re-state ALL prior information
shards so the model always sees the full accumulated context, not just new input.

This implementation wraps any OpenAI-compatible API with a rolling context
accumulator. Drop it in front of your chatbot and watch reliability improve.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI  # pip install openai

# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ConversationShard:
    """A single piece of information revealed across conversation turns."""
    turn: int
    content: str
    is_requirement: bool = True  # mark as task requirement vs. casual chat


@dataclass
class ConversationState:
    """
    Tracks accumulated task requirements across turns.
    Implements the SNOWBALL protocol: each turn includes all prior shards.
    """
    shards: list[ConversationShard] = field(default_factory=list)
    raw_history: list[dict] = field(default_factory=list)
    turn_count: int = 0

    def add_shard(self, content: str, is_requirement: bool = True):
        self.turn_count += 1
        self.shards.append(ConversationShard(
            turn=self.turn_count,
            content=content,
            is_requirement=is_requirement,
        ))

    def build_snowball_context(self) -> str:
        """
        Core of the SNOWBALL mitigation:
        Concatenate ALL accumulated requirements into one clear block.
        This prevents the model from 'forgetting' early turns or anchoring
        to incorrect early attempts.
        """
        requirements = [s for s in self.shards if s.is_requirement]
        if not requirements:
            return ""
        lines = ["[Accumulated task requirements so far:]"]
        for i, shard in enumerate(requirements, 1):
            lines.append(f"  {i}. {shard.content}")
        lines.append("[End of requirements — please respond to ALL of the above.]")
        return "\n".join(lines)

    def get_reliability_score(self, responses: list[str]) -> float:
        """
        Estimates reliability: fraction of responses that are consistent.
        Paper finding: unreliability (not capability loss) drives the 39% drop.
        Simple proxy: check response length stability (verbose = bad sign).
        """
        if len(responses) < 2:
            return 1.0
        lengths = [len(r.split()) for r in responses]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        # High variance in response length → high instability
        normalized_variance = min(variance / (mean_len ** 2 + 1e-9), 1.0)
        return round(1.0 - normalized_variance, 3)


# ── Main Wrapper ───────────────────────────────────────────────────────────────

class SnowballChatAgent:
    """
    A drop-in wrapper that implements the SNOWBALL multi-turn mitigation.
    
    Instead of passing raw user messages, it:
    1. Accumulates all task requirements as shards
    2. Prepends the full snowball context to every new turn
    3. Tracks reliability across the conversation
    
    Usage:
        agent = SnowballChatAgent(model="gpt-4o-mini")
        response = agent.chat("Write a function that sorts a list")
        response = agent.chat("Make it work for strings too")  # shard added
        response = agent.chat("Also handle None values")       # shard added
        # Each call sees ALL prior requirements, not just the new one
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.state = ConversationState()
        self.responses: list[str] = []

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, user_message: str, is_requirement: bool = True) -> str:
        """
        Send a message. If is_requirement=True, it's accumulated as a shard
        and the full snowball context is prepended to the actual API call.
        """
        # 1. Accumulate the shard
        self.state.add_shard(user_message, is_requirement=is_requirement)

        # 2. Build snowball-augmented message
        snowball_ctx = self.state.build_snowball_context()
        augmented_message = f"{snowball_ctx}\n\nLatest instruction: {user_message}"

        # 3. Keep raw history for API continuity (without snowball bloat in display)
        self.state.raw_history.append({"role": "user", "content": augmented_message})

        # 4. Call the API with full accumulated context
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.state.raw_history,
            ],
        )
        reply = response.choices[0].message.content
        self.state.raw_history.append({"role": "assistant", "content": reply})
        self.responses.append(reply)

        if self.verbose:
            reliability = self.state.get_reliability_score(self.responses)
            print(f"\n[Turn {self.state.turn_count}] Reliability score: {reliability}")
            print(f"[Shards accumulated: {len(self.state.shards)}]")
            print(f"Response preview: {reply[:120]}...")

        return reply

    def recap_all(self) -> str:
        """
        Implements the RECAP mitigation from the paper:
        Issue a final recapitulation turn restating all shards, giving the model
        one clean shot at a coherent final answer.
        """
        recap_prompt = (
            "Based on everything discussed so far, please provide a final, "
            "complete, and coherent response that addresses ALL requirements "
            "that have been stated across this conversation."
        )
        return self.chat(recap_prompt, is_requirement=False)

    def get_conversation_report(self) -> dict:
        """Diagnostic report mirroring paper's analysis framework."""
        return {
            "total_turns": self.state.turn_count,
            "requirement_shards": len([s for s in self.state.shards if s.is_requirement]),
            "reliability_score": self.state.get_reliability_score(self.responses),
            "avg_response_length_words": (
                sum(len(r.split()) for r in self.responses) / max(len(self.responses), 1)
            ),
            "shards": [
                {"turn": s.turn, "content": s.content[:80] + "..."}
                for s in self.state.shards
            ],
        }


# ── Baseline: Naive Multi-Turn (No Snowball) ───────────────────────────────────

class NaiveChatAgent:
    """
    Standard multi-turn chat — no snowball accumulation.
    This is what causes the 39% drop the paper documents.
    Each turn only sees the new message + raw prior history.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.history: list[dict] = []
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                *self.history,
            ],
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ── Demo: Sharded Task (Simulating Real User Behavior) ────────────────────────

def run_sharded_demo():
    """
    Simulates the paper's "SHARDED" condition:
    A coding task drip-fed across 4 turns instead of given all at once.
    
    Single-turn equivalent prompt would be:
    "Write a Python function that: sorts a list of dicts by a given key,
     handles string and numeric values, returns None entries last,
     and supports both ascending and descending order."
    """
    print("=" * 60)
    print("SNOWBALL AGENT (with mitigation)")
    print("=" * 60)

    snowball = SnowballChatAgent(
        model="gpt-4o-mini",
        system_prompt="You are an expert Python developer.",
        verbose=True,
    )

    # Shards drip-fed across turns — exactly how real users behave
    snowball.chat("Write a Python function that sorts a list of dicts by a given key.")
    snowball.chat("It should handle both string and numeric values.")
    snowball.chat("None values should always appear last.")
    snowball.chat("Also add support for both ascending and descending order.")

    # Final recap turn — paper's RECAP mitigation
    print("\n[Issuing final RECAP turn...]")
    final = snowball.recap_all()

    print("\n" + "=" * 60)
    print("CONVERSATION REPORT")
    print("=" * 60)
    report = snowball.get_conversation_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("NAIVE AGENT (no mitigation — baseline for comparison)")
    print("=" * 60)

    naive = NaiveChatAgent(model="gpt-4o-mini")
    naive.chat("Write a Python function that sorts a list of dicts by a given key.")
    naive.chat("It should handle both string and numeric values.")
    naive.chat("None values should always appear last.")
    r = naive.chat("Also add support for both ascending and descending order.")
    print(f"Naive agent final response (last turn only):\n{r[:300]}...")
    print("\nNote: Naive agent's final response likely won't address ALL 4 requirements.")
    print("That's the 39% drop the paper quantifies.")


if __name__ == "__main__":
    # Set your key: export OPENAI_API_KEY=sk-...
    # Or pass api_key= to the agent constructors above.
    run_sharded_demo()
