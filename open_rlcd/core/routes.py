"""Route contracts: a choice selects a contract, not a naked name.

A choice head's options are routes (an LLM tier, a retrieval strategy, a hand-off). What an option commits the
application to — which provider class, which data it may receive, which tools, how much context, how many
retries, how the output is verified, what to fall back to — lives in configuration, not in question wording or in
provider-specific code, so a semantic router can never bypass a limit. Deterministic filters remove the options a
request cannot use *before* the head chooses: the model picks among allowed possibilities, it is never asked to
overrule policy.

    routes:                                   # rlcd.yaml
      response_type:                          # a choice head; its options are the keys
        direct_answer:  {llm: small, context_budget_tokens: 2000, max_retries: 1, verify: postflight,
                         fallback: escalate_human}
        summarize:      {llm: large, requires: [llm_large], allowed_data: [public, internal], latency_ms: 4000,
                         verify: postflight, fallback: escalate_human}
        escalate_human: {human: true}
      route:
        graph:  {requires: [graph], fallback: hybrid}
        hybrid: {requires: [vector, graph], fallback: vector}
        vector: {requires: [vector]}

Keys the deterministic filter reads (every other key is the caller's and is returned untouched as the contract):

    enabled: false        the option is never chosen
    requires: [names]     capabilities the option needs; removed when the request's `capabilities` lack one
    allowed_data: [cls]   data classes the option may receive; removed when the request's `sensitivity` is not one
    latency_ms: n         expected latency; removed when above the request's `latency_budget_ms`
    fallback: option      what to run when this option fails at execution: another option of the same head, which
                          must pass the same filters (a fallback never sends data where the primary could not)

Request fields (decide): `capabilities` (what is available right now), `sensitivity` (the data class of this
state), `latency_budget_ms`. Without a field, its filter does not apply. Every option considered is reported with
the reason it was removed, the chosen option carries its contract, and its fallback is resolved against the
filters of the same request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import ContractError

RESERVED = ("enabled", "requires", "allowed_data", "latency_ms", "fallback")


@dataclass(frozen=True)
class Constraints:
    capabilities: frozenset[str] | None = None
    sensitivity: str | None = None
    latency_budget_ms: float | None = None

    @classmethod
    def from_request(cls, request: dict | None) -> Constraints:
        r = request or {}
        caps = r.get("capabilities")
        if caps is not None and (not isinstance(caps, list) or not all(isinstance(c, str) for c in caps)):
            raise ContractError("capabilities must be an array of capability names", field="capabilities")
        sens = r.get("sensitivity")
        if sens is not None and not isinstance(sens, str):
            raise ContractError("sensitivity must be a string (a data class such as public, internal, restricted)",
                                field="sensitivity")
        lat = r.get("latency_budget_ms")
        if lat is not None:
            try:
                lat = float(lat)
            except (TypeError, ValueError):
                raise ContractError("latency_budget_ms must be a number", field="latency_budget_ms") from None
            if lat <= 0:
                raise ContractError("latency_budget_ms must be positive", field="latency_budget_ms")
        return cls(capabilities=frozenset(caps) if caps is not None else None, sensitivity=sens,
                   latency_budget_ms=lat)

    @property
    def active(self) -> bool:
        return self.capabilities is not None or self.sensitivity is not None or self.latency_budget_ms is not None


def _strings(v: Any, where: str) -> tuple[str, ...]:
    if isinstance(v, str):
        return (v,)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{where} must be a list of names")
    return tuple(v)


class RouteBook:
    """The route contracts of a project: {head: {option: contract}}."""

    def __init__(self, routes: dict[str, dict[str, dict]] | None = None):
        self.routes = routes or {}

    @classmethod
    def from_config(cls, section: Any) -> RouteBook:
        if not section:
            return cls({})
        if not isinstance(section, dict):
            raise ValueError("routes maps a choice head to {option: contract}")
        out: dict[str, dict[str, dict]] = {}
        for head, options in section.items():
            if not isinstance(options, dict) or not options:
                raise ValueError(f"routes.{head} maps each option of the head to its contract")
            out[str(head)] = {}
            for opt, c in options.items():
                c = {} if c is None else c
                if not isinstance(c, dict):
                    raise ValueError(f"routes.{head}.{opt} must be a mapping (the contract)")
                if "requires" in c:
                    _strings(c["requires"], f"routes.{head}.{opt}.requires")
                if "allowed_data" in c:
                    _strings(c["allowed_data"], f"routes.{head}.{opt}.allowed_data")
                if "latency_ms" in c and not isinstance(c["latency_ms"], (int, float)):
                    raise ValueError(f"routes.{head}.{opt}.latency_ms must be a number")
                if "enabled" in c and not isinstance(c["enabled"], bool):
                    raise ValueError(f"routes.{head}.{opt}.enabled must be true or false")
                out[str(head)][str(opt)] = dict(c)
            for opt, c in out[str(head)].items():
                fb = c.get("fallback")
                if fb is None:
                    continue
                if fb == opt or fb not in out[str(head)]:
                    raise ValueError(f"routes.{head}.{opt}.fallback must name another option of {head} that has a "
                                     f"contract, got {fb!r}")
            # a fallback chain must end
            for opt in out[str(head)]:
                seen, cur = [opt], out[str(head)][opt].get("fallback")
                while cur is not None:
                    if cur in seen:
                        raise ValueError(f"routes.{head}: fallback cycle {' -> '.join([*seen, cur])}")
                    seen.append(cur)
                    cur = out[str(head)][cur].get("fallback")
        return cls(out)

    def has(self, head: str) -> bool:
        return head in self.routes

    def check_head(self, head: str, spec: Any) -> None:
        """A contract for an option the head does not have is a configuration error worth a clear message."""
        if spec.type != "choice":
            raise ContractError(f"routes.{head}: head '{head}' is a {spec.type} head; contracts belong to the options "
                                f"of a choice head", kind="bad_config", status=500)
        unknown = sorted(set(self.routes.get(head, {})) - set(spec.classes))
        if unknown:
            raise ContractError(f"routes.{head} has contracts for options {unknown} that the head does not have "
                                f"({list(spec.classes)})", kind="bad_config", status=500)

    def contract(self, head: str, option: str | None) -> dict | None:
        if option is None:
            return None
        c = self.routes.get(head, {}).get(option)
        return None if c is None else dict(c)

    def _why_removed(self, c: dict, cons: Constraints) -> str | None:
        if c.get("enabled") is False:
            return "disabled by configuration"
        if cons.capabilities is not None and c.get("requires") is not None:
            missing = [r for r in _strings(c["requires"], "requires") if r not in cons.capabilities]
            if missing:
                return f"needs {', '.join(missing)}, which the request does not list as available"
        if cons.sensitivity is not None and c.get("allowed_data") is not None:
            ok = _strings(c["allowed_data"], "allowed_data")
            if cons.sensitivity not in ok:
                return f"may not receive {cons.sensitivity} data (allowed: {', '.join(ok)})"
        if cons.latency_budget_ms is not None and c.get("latency_ms") is not None \
                and float(c["latency_ms"]) > cons.latency_budget_ms:
            return f"expected latency {c['latency_ms']:g} ms exceeds the budget {cons.latency_budget_ms:g} ms"
        return None

    def filter(self, head: str, options: tuple[str, ...] | list[str], cons: Constraints) -> tuple[list[str], list]:
        """(the options that survive, [{option, kept, reason}] for every option considered). An option without a
        contract has no constraints and survives."""
        considered, kept = [], []
        book = self.routes.get(head, {})
        for o in options:
            c = book.get(o)
            why = self._why_removed(c, cons) if c is not None else None
            considered.append({"option": o, "kept": why is None, "reason": why})
            if why is None:
                kept.append(o)
        return kept, considered

    def resolve(self, head: str, option: str | None, kept: list[str]) -> dict | None:
        """The chosen option's contract and its usable fallback: the first option along the fallback chain that
        survived this request's filters (None when none did)."""
        if option is None or head not in self.routes:
            return None
        c = self.contract(head, option)
        fallback, cur, seen = None, (c or {}).get("fallback"), {option}
        while cur is not None and cur not in seen:
            if cur in kept:
                fallback = cur
                break
            seen.add(cur)
            cur = (self.contract(head, cur) or {}).get("fallback")
        return {"option": option, "contract": c, "fallback": fallback,
                "fallback_contract": self.contract(head, fallback)}
