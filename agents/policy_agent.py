"""
Policy Agent — Insurance Policy Search & Analysis

This agent:
1. READS the Medical Necessity Ledger to understand what the Clinical Agent found
2. Uses those clinical findings to ADAPT its search through policy documents
3. Identifies relevant sections, exception clauses, and authorization pathways
4. Writes matches back to the ledger

KEY DESIGN: The Policy Agent's behavior changes based on what the Clinical Agent wrote.
If the Clinical Agent finds a rare autoimmune condition, the Policy Agent shifts from
searching generic imaging rules to searching for autoimmune exception clauses.
This is the shared memory USP in action.
"""

import json
import re
from typing import Any, Dict, List

from openai import AsyncOpenAI
from models.schemas import AgentSource, Severity, PolicySection
from memory.ledger import MedicalNecessityLedger


POLICY_SYSTEM_PROMPT = """You are a Policy Analysis Agent specializing in navigating 
insurance policy documents to find authorization pathways for medical procedures.

You have been given context from a Clinical Agent about a patient's condition.
Your job is to:
1. Search the policy document for relevant sections
2. Identify exception clauses that apply to this specific clinical situation
3. Find the fastest authorization pathway
4. Identify documentation requirements

You think like both an insurance expert and a patient advocate — finding every 
legitimate pathway to get necessary care approved.

You must respond in valid JSON format only. No markdown, no explanation outside JSON.
"""


class PolicyAgent:
    def __init__(self, client: AsyncOpenAI, ledger: MedicalNecessityLedger):
        self.client = client
        self.ledger = ledger
        self.model = "nvidia/nemotron-3-super-120b-a12b:free"

    async def analyze_policy(self, policy_text: str, plan_name: str) -> dict:
        """
        Main entry point: read the ledger, then search the policy accordingly.
        Always returns a valid object so the coordinator does not crash.
        """
        clinical_context = await self.ledger.get_clinical_context()
        search_hints = await self.ledger.get_context("policy_search_hints") or []
        exception_indicators = await self.ledger.get_context("exception_indicators") or []
        necessity_level = await self.ledger.get_context("clinical_necessity_level") or "UNKNOWN"

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="LEDGER_READ",
            message=(
                f"Reading shared ledger. Clinical necessity: {necessity_level}. "
                f"Adapting search parameters based on {len(search_hints)} clinical hint(s): "
                f"{'; '.join(search_hints[:3]) if search_hints else 'none'}."
            ),
            data={
                "necessity_level": necessity_level,
                "search_hints": search_hints,
                "exception_indicators": exception_indicators,
            },
            tags=["LEDGER_READ", "SEARCH_ADAPTED"],
        )

        matched_sections = await self._search_policy(
            policy_text, clinical_context, search_hints, exception_indicators
        )

        exception_analysis = await self._analyze_exceptions(
            policy_text, clinical_context, matched_sections, exception_indicators
        )

        auth_pathway = await self._determine_pathway(
            clinical_context, matched_sections, exception_analysis
        )

        await self.ledger.write(
            source=AgentSource.LEDGER,
            event_type="POLICY_CONTEXT_COMPLETE",
            message=(
                f"Policy analysis complete. Authorization pathway: "
                f"{auth_pathway.get('recommended_pathway', 'PENDING_MANUAL_REVIEW')}. "
                f"Status: {auth_pathway.get('expected_status', 'PENDING_MANUAL_REVIEW')}. "
                f"Exception clauses found: {len(exception_analysis.get('applicable_exceptions', []))}."
            ),
            data={
                "pathway": auth_pathway,
                "exceptions": exception_analysis,
                "matched_sections_count": len(matched_sections),
            },
            tags=[
                "LEDGER_WRITE",
                "POLICY_COMPLETE",
                auth_pathway.get("expected_status", "PENDING_MANUAL_REVIEW"),
            ],
            severity=Severity.CRITICAL,
        )

        return {
            "matched_sections": matched_sections,
            "exception_analysis": exception_analysis,
            "auth_pathway": auth_pathway,
        }

    def _safe_json_parse(self, text: Any) -> dict:
        """
        Robust JSON parser for LLM output.

        Handles:
        - None responses
        - markdown code fences
        - leading/trailing conversational text
        - trailing commas
        - basic Python literals
        - common delimiter issues in long arrays/objects
        """
        if text is None:
            return {"error": "Empty response from model", "raw_text": None}

        if not isinstance(text, (str, bytes, bytearray)):
            return {
                "error": f"Unexpected response type: {type(text).__name__}",
                "raw_text": str(text),
            }

        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", errors="ignore")

        if not text.strip():
            return {"error": "Empty response from model", "raw_text": text}

        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        repaired = cleaned
        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)

        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        repaired = re.sub(r'(\})(\s*)(\{)', r'\1,\2\3', repaired)
        repaired = re.sub(r'(\])(\s*)(\{)', r'\1,\2\3', repaired)
        repaired = re.sub(r'(\})(\s*)(\")', r'\1,\2\3', repaired)
        repaired = re.sub(r'(\])(\s*)(\")', r'\1,\2\3', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse JSON from model output",
                "parse_error": str(e),
                "raw_text": text,
            }

    async def _call_json_model(
        self,
        prompt: str,
        fallback: dict,
        max_tokens: int = 2000,
    ) -> dict:
        """
        Safe wrapper around model calls.
        Always returns a dict, never raises parsing errors upstream.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            result = dict(fallback)
            result["error"] = f"Model call failed: {e}"
            return result

        result_text = ""
        try:
            result_text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            result_text = ""

        parsed = self._safe_json_parse(result_text)

        if not isinstance(parsed, dict):
            result = dict(fallback)
            result["error"] = "Parsed output was not a JSON object"
            result["raw_text"] = result_text
            return result

        if parsed.get("error"):
            result = dict(fallback)
            result.update({"parse_error": parsed.get("error"), "raw_text": parsed.get("raw_text")})
            return result

        return parsed

    def _safe_score(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _build_policy_chunks(
        self,
        policy_text: str,
        chunk_size: int = 3500,
        overlap: int = 400,
    ) -> List[dict]:
        """
        Splits large policy text into overlapping chunks so the model
        does not get overwhelmed by very long documents.
        """
        if not policy_text:
            return []

        chunks = []
        start = 0
        idx = 1
        text_len = len(policy_text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = policy_text[start:end]
            chunks.append(
                {
                    "chunk_id": f"chunk_{idx}",
                    "start": start,
                    "end": end,
                    "text": chunk_text,
                }
            )
            if end >= text_len:
                break
            start = max(end - overlap, start + 1)
            idx += 1

        return chunks

    async def _search_policy(
        self,
        policy_text: str,
        clinical_context: str,
        search_hints: list[str],
        exception_indicators: list[str],
    ) -> list[dict]:
        """
        Search the policy document with clinically-informed parameters.

        Fixes:
        - avoids sending the full giant policy at once
        - uses chunking for token efficiency
        - uses safe model parsing
        - always returns a valid list
        """
        chunks = self._build_policy_chunks(policy_text)
        if not chunks:
            await self.ledger.write(
                source=AgentSource.POLICY,
                event_type="POLICY_SEARCH",
                message="Policy search skipped because the policy document was empty.",
                data={"sections_found": 0, "exception_clauses": 0},
                tags=["POLICY_SEARCH", "NO_POLICY_TEXT"],
                severity=Severity.NORMAL,
            )
            return []

        all_sections: List[dict] = []

        for chunk in chunks[:8]:
            prompt = f"""Search this insurance policy chunk for sections relevant to the following clinical case.

IMPORTANT: The Clinical Agent has identified these specific search priorities:
{json.dumps(search_hints, indent=2)}

And these clinical indicators that might trigger policy exceptions:
{json.dumps(exception_indicators, indent=2)}

CLINICAL CONTEXT FROM SHARED LEDGER:
{clinical_context}

POLICY CHUNK ID: {chunk["chunk_id"]}
POLICY DOCUMENT CHUNK:
{chunk["text"]}

Find ALL relevant sections in this chunk. Pay special attention to:
- Exception clauses for the specific condition type
- Expedited review pathways
- Medical necessity definitions that match these clinical indicators
- Multi-morbidity or complex condition provisions
- Numeric thresholds mentioned in the policy

Respond with JSON:
{{
    "matched_sections": [
        {{
            "section_id": "string - section number/identifier",
            "title": "string - section title",
            "relevant_text": "string - the specific policy text that applies",
            "relevance_score": 0.0,
            "is_exception_clause": true,
            "match_reason": "string - why this section is relevant",
            "clinical_criteria_matched": ["list of clinical findings"],
            "chunk_id": "{chunk["chunk_id"]}"
        }}
    ],
    "search_strategy_used": "string"
}}"""

            fallback = {
                "matched_sections": [],
                "search_strategy_used": "fallback_chunk_scan",
            }

            result = await self._call_json_model(prompt, fallback, max_tokens=1400)
            sections = result.get("matched_sections", [])

            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict):
                        section.setdefault("chunk_id", chunk["chunk_id"])
                        all_sections.append(section)

        deduped = []
        seen = set()
        for section in all_sections:
            key = (
                str(section.get("section_id", "")).strip(),
                str(section.get("title", "")).strip(),
                str(section.get("relevant_text", "")).strip()[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(section)

        deduped = sorted(
            deduped,
            key=lambda s: self._safe_score(s.get("relevance_score", 0)),
            reverse=True,
        )[:12]

        exception_count = sum(1 for s in deduped if s.get("is_exception_clause"))

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="POLICY_SEARCH",
            message=(
                f"Policy search complete. Found {len(deduped)} relevant section(s), "
                f"including {exception_count} exception clause(s). "
                f"Strategy: chunked policy search guided by clinical hints."
            ),
            data={"sections_found": len(deduped), "exception_clauses": exception_count},
            tags=["POLICY_SEARCH", "SECTIONS_FOUND"],
            severity=Severity.HIGH if exception_count > 0 else Severity.NORMAL,
        )

        for section in deduped[:3]:
            tag_list = ["SECTION_MATCH"]
            if section.get("is_exception_clause"):
                tag_list.append("EXCEPTION_CLAUSE")

            await self.ledger.write(
                source=AgentSource.POLICY,
                event_type="SECTION_MATCH",
                message=(
                    f"§{section.get('section_id', '?')} — {section.get('title', 'Untitled')}: "
                    f"{section.get('match_reason', 'Relevant policy section identified')}"
                ),
                data={
                    "section_id": section.get("section_id"),
                    "relevance_score": section.get("relevance_score"),
                    "is_exception": section.get("is_exception_clause"),
                    "chunk_id": section.get("chunk_id"),
                    "text_preview": str(section.get("relevant_text", ""))[:200],
                },
                tags=tag_list,
                severity=Severity.CRITICAL if section.get("is_exception_clause") else Severity.NORMAL,
            )

        return deduped

    async def _analyze_exceptions(
        self,
        policy_text: str,
        clinical_context: str,
        matched_sections: list[dict],
        exception_indicators: list[str],
    ) -> dict:
        """
        Deep-dive into exception clauses.

        Fix:
        - no more blind policy_text[:5000]
        - only passes relevant_text from matched exception sections
        """
        exception_sections = [
            s for s in matched_sections
            if isinstance(s, dict) and s.get("is_exception_clause")
        ]

        if not exception_sections:
            return {
                "applicable_exceptions": [],
                "best_exception_pathway": "None",
                "recommendation": "No exception clauses found; standard authorization pathway applies.",
            }

        focused_policy_text = "\n\n".join(
            [
                f"SECTION {s.get('section_id', '?')} — {s.get('title', 'Untitled')}\n"
                f"{s.get('relevant_text', '')}"
                for s in exception_sections
            ]
        )

        prompt = f"""Perform a detailed analysis of whether this patient qualifies for the exception clauses found.

CLINICAL CONTEXT:
{clinical_context}

CLINICAL EXCEPTION INDICATORS:
{json.dumps(exception_indicators, indent=2)}

EXCEPTION CLAUSES FOUND:
{json.dumps(exception_sections, indent=2)}

RELEVANT POLICY TEXT ONLY:
{focused_policy_text}

For each exception clause, determine:
1. Does the patient meet ALL required criteria?
2. What specific evidence satisfies each criterion?
3. Is any documentation missing?

Respond with JSON:
{{
    "applicable_exceptions": [
        {{
            "section_id": "string",
            "title": "string",
            "all_criteria_met": true,
            "criteria_evaluation": [
                {{
                    "criterion": "string",
                    "met": true,
                    "evidence": "string"
                }}
            ],
            "missing_documentation": ["list of anything still needed"],
            "confidence": 0.0
        }}
    ],
    "best_exception_pathway": "string",
    "recommendation": "string"
}}"""

        fallback = {
            "applicable_exceptions": [],
            "best_exception_pathway": "Pending Manual Review",
            "recommendation": "Unable to reliably evaluate exception clauses; manual review recommended.",
        }

        result = await self._call_json_model(prompt, fallback, max_tokens=1600)

        result.setdefault("applicable_exceptions", [])
        result.setdefault("best_exception_pathway", "Pending Manual Review")
        result.setdefault(
            "recommendation",
            "Unable to reliably evaluate exception clauses; manual review recommended.",
        )

        if not isinstance(result["applicable_exceptions"], list):
            result["applicable_exceptions"] = []

        qualifying = [
            e for e in result["applicable_exceptions"]
            if isinstance(e, dict) and e.get("all_criteria_met")
        ]

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="EXCEPTION_ANALYSIS",
            message=(
                f"Exception analysis: {len(qualifying)} of {len(result.get('applicable_exceptions', []))} "
                f"exception clause(s) fully satisfied. "
                f"Best pathway: {result.get('best_exception_pathway', 'Pending Manual Review')}."
            ),
            data={
                "qualifying_exceptions": len(qualifying),
                "best_pathway": result.get("best_exception_pathway"),
            },
            tags=["EXCEPTION_ANALYSIS", "CRITERIA_MET" if qualifying else "CRITERIA_PARTIAL"],
            severity=Severity.CRITICAL if qualifying else Severity.HIGH,
        )

        return result

    async def _determine_pathway(
        self,
        clinical_context: str,
        matched_sections: list[dict],
        exception_analysis: dict,
    ) -> dict:
        """
        Determine the final authorization pathway.

        Fix:
        - safe parsing
        - default fallback object instead of red-screen failure
        """
        prompt = f"""Based on the complete analysis, determine the authorization pathway.

CLINICAL CONTEXT:
{clinical_context}

MATCHED POLICY SECTIONS:
{json.dumps(matched_sections[:5], indent=2)}

EXCEPTION ANALYSIS:
{json.dumps(exception_analysis, indent=2)}

Respond with JSON:
{{
    "recommended_pathway": "string",
    "expected_status": "AUTO_APPROVED | EXPEDITED_REVIEW | STANDARD_REVIEW | LIKELY_DENIED | PENDING_MANUAL_REVIEW",
    "estimated_processing_time": "string",
    "confidence_score": 0.0,
    "reasoning": "string",
    "documentation_checklist": [
        {{
            "item": "string",
            "status": "AVAILABLE | NEEDED | OPTIONAL",
            "source": "string"
        }}
    ],
    "admin_cost_savings_estimate": "string",
    "appeal_risk": "LOW | MEDIUM | HIGH",
    "alternative_pathways": ["list of backup pathways if primary is rejected"]
}}"""

        fallback = {
            "recommended_pathway": "Pending Manual Review",
            "expected_status": "PENDING_MANUAL_REVIEW",
            "estimated_processing_time": "Unknown",
            "confidence_score": 0.0,
            "reasoning": "Model output was unavailable or malformed, so manual review is required.",
            "documentation_checklist": [],
            "admin_cost_savings_estimate": "Unknown",
            "appeal_risk": "MEDIUM",
            "alternative_pathways": ["STANDARD_REVIEW"],
        }

        result = await self._call_json_model(prompt, fallback, max_tokens=1400)

        result.setdefault("recommended_pathway", "Pending Manual Review")
        result.setdefault("expected_status", "PENDING_MANUAL_REVIEW")
        result.setdefault("estimated_processing_time", "Unknown")
        result.setdefault("confidence_score", 0.0)
        result.setdefault(
            "reasoning",
            "Model output was unavailable or malformed, so manual review is required.",
        )
        result.setdefault("documentation_checklist", [])
        result.setdefault("admin_cost_savings_estimate", "Unknown")
        result.setdefault("appeal_risk", "MEDIUM")
        result.setdefault("alternative_pathways", ["STANDARD_REVIEW"])

        if not isinstance(result["documentation_checklist"], list):
            result["documentation_checklist"] = []
        if not isinstance(result["alternative_pathways"], list):
            result["alternative_pathways"] = ["STANDARD_REVIEW"]

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="PATHWAY_DETERMINATION",
            message=(
                f"Authorization pathway determined: {result.get('recommended_pathway', 'Pending Manual Review')}. "
                f"Expected status: {result.get('expected_status', 'PENDING_MANUAL_REVIEW')}. "
                f"Processing time: {result.get('estimated_processing_time', 'Unknown')}. "
                f"Confidence: {self._safe_score(result.get('confidence_score', 0)):.0%}."
            ),
            data=result,
            tags=["PATHWAY", result.get("expected_status", "PENDING_MANUAL_REVIEW")],
            severity=Severity.CRITICAL,
        )

        return result