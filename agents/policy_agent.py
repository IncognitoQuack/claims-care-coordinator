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
from anthropic import AsyncAnthropic
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
    def __init__(self, client: AsyncAnthropic, ledger: MedicalNecessityLedger):
        self.client = client
        self.ledger = ledger
        self.model = "claude-sonnet-4-20250514"

    async def analyze_policy(self, policy_text: str, plan_name: str) -> dict:
        """
        Main entry point: read the ledger, then search the policy accordingly.
        """
        # Step 1: Read the shared ledger to understand clinical context
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
                f"{'; '.join(search_hints[:3])}."
            ),
            data={
                "necessity_level": necessity_level,
                "search_hints": search_hints,
                "exception_indicators": exception_indicators,
            },
            tags=["LEDGER_READ", "SEARCH_ADAPTED"],
        )

        # Step 2: Search policy for relevant sections
        matched_sections = await self._search_policy(
            policy_text, clinical_context, search_hints, exception_indicators
        )

        # Step 3: Deep analysis of exception clauses
        exception_analysis = await self._analyze_exceptions(
            policy_text, clinical_context, matched_sections, exception_indicators
        )

        # Step 4: Determine authorization pathway
        auth_pathway = await self._determine_pathway(
            clinical_context, matched_sections, exception_analysis
        )

        # Step 5: Write final policy findings to ledger
        await self.ledger.write(
            source=AgentSource.LEDGER,
            event_type="POLICY_CONTEXT_COMPLETE",
            message=(
                f"Policy analysis complete. Authorization pathway: {auth_pathway.get('recommended_pathway', 'UNKNOWN')}. "
                f"Status: {auth_pathway.get('expected_status', 'PENDING')}. "
                f"Exception clauses found: {len(exception_analysis.get('applicable_exceptions', []))}."
            ),
            data={
                "pathway": auth_pathway,
                "exceptions": exception_analysis,
                "matched_sections_count": len(matched_sections),
            },
            tags=["LEDGER_WRITE", "POLICY_COMPLETE", auth_pathway.get("expected_status", "PENDING")],
            severity=Severity.CRITICAL,
        )

        return {
            "matched_sections": matched_sections,
            "exception_analysis": exception_analysis,
            "auth_pathway": auth_pathway,
        }

    async def _search_policy(
        self,
        policy_text: str,
        clinical_context: str,
        search_hints: list[str],
        exception_indicators: list[str],
    ) -> list[dict]:
        """Search the policy document with clinically-informed parameters."""
        prompt = f"""Search this insurance policy document for sections relevant to the following clinical case.

IMPORTANT: The Clinical Agent has identified these specific search priorities:
{json.dumps(search_hints, indent=2)}

And these clinical indicators that might trigger policy exceptions:
{json.dumps(exception_indicators, indent=2)}

CLINICAL CONTEXT FROM SHARED LEDGER:
{clinical_context}

POLICY DOCUMENT:
{policy_text}

Find ALL relevant sections. Pay special attention to:
- Exception clauses for the specific condition type
- Expedited review pathways
- Medical necessity definitions that match these clinical indicators
- Multi-morbidity or complex condition provisions
- Any sections where the clinical indicators meet specific numeric thresholds mentioned in the policy

Respond with JSON:
{{
    "matched_sections": [
        {{
            "section_id": "string - section number/identifier",
            "title": "string - section title",
            "relevant_text": "string - the specific policy text that applies (quote exactly)",
            "relevance_score": 0.0-1.0,
            "is_exception_clause": true/false,
            "match_reason": "string - why this section is relevant to THIS specific case",
            "clinical_criteria_matched": ["list of specific clinical findings that match this section's requirements"]
        }}
    ],
    "search_strategy_used": "string - describe how the clinical hints changed your search approach"
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=POLICY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"matched_sections": [], "error": "Parse failed"}

        sections = result.get("matched_sections", [])
        exception_count = sum(1 for s in sections if s.get("is_exception_clause"))

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="POLICY_SEARCH",
            message=(
                f"Policy search complete. Found {len(sections)} relevant section(s), "
                f"including {exception_count} exception clause(s). "
                f"Strategy: {result.get('search_strategy_used', 'standard')}."
            ),
            data={"sections_found": len(sections), "exception_clauses": exception_count},
            tags=["POLICY_SEARCH", "SECTIONS_FOUND"],
            severity=Severity.HIGH if exception_count > 0 else Severity.NORMAL,
        )

        # Write individual section matches
        for section in sorted(sections, key=lambda s: s.get("relevance_score", 0), reverse=True)[:3]:
            tag_list = ["SECTION_MATCH"]
            if section.get("is_exception_clause"):
                tag_list.append("EXCEPTION_CLAUSE")

            await self.ledger.write(
                source=AgentSource.POLICY,
                event_type="SECTION_MATCH",
                message=(
                    f"§{section.get('section_id', '?')} — {section.get('title', 'Untitled')}: "
                    f"{section.get('match_reason', '')}"
                ),
                data={
                    "section_id": section.get("section_id"),
                    "relevance_score": section.get("relevance_score"),
                    "is_exception": section.get("is_exception_clause"),
                    "text_preview": section.get("relevant_text", "")[:200],
                },
                tags=tag_list,
                severity=Severity.CRITICAL if section.get("is_exception_clause") else Severity.NORMAL,
            )

        return sections

    async def _analyze_exceptions(
        self,
        policy_text: str,
        clinical_context: str,
        matched_sections: list[dict],
        exception_indicators: list[str],
    ) -> dict:
        """Deep-dive into exception clauses to see if clinical criteria are met."""
        exception_sections = [s for s in matched_sections if s.get("is_exception_clause")]

        if not exception_sections:
            return {
                "applicable_exceptions": [],
                "recommendation": "No exception clauses found; standard authorization pathway applies.",
            }

        prompt = f"""Perform a detailed analysis of whether this patient qualifies for the exception clauses found.

CLINICAL CONTEXT:
{clinical_context}

CLINICAL EXCEPTION INDICATORS:
{json.dumps(exception_indicators, indent=2)}

EXCEPTION CLAUSES FOUND:
{json.dumps(exception_sections, indent=2)}

FULL POLICY TEXT (for cross-referencing):
{policy_text[:5000]}

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
            "all_criteria_met": true/false,
            "criteria_evaluation": [
                {{
                    "criterion": "string - what the policy requires",
                    "met": true/false,
                    "evidence": "string - specific clinical evidence that satisfies this"
                }}
            ],
            "missing_documentation": ["list of anything still needed"],
            "confidence": 0.0-1.0
        }}
    ],
    "best_exception_pathway": "string - which exception clause is the strongest match",
    "recommendation": "string - overall recommendation"
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=POLICY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"applicable_exceptions": []}

        qualifying = [e for e in result.get("applicable_exceptions", []) if e.get("all_criteria_met")]

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="EXCEPTION_ANALYSIS",
            message=(
                f"Exception analysis: {len(qualifying)} of {len(result.get('applicable_exceptions', []))} "
                f"exception clause(s) fully satisfied. "
                f"Best pathway: {result.get('best_exception_pathway', 'None')}."
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
        """Determine the final authorization pathway."""
        prompt = f"""Based on the complete analysis, determine the authorization pathway.

CLINICAL CONTEXT:
{clinical_context}

MATCHED POLICY SECTIONS:
{json.dumps(matched_sections[:5], indent=2)}

EXCEPTION ANALYSIS:
{json.dumps(exception_analysis, indent=2)}

Respond with JSON:
{{
    "recommended_pathway": "string - the specific policy pathway (e.g., 'Autoimmune Exception §7.3')",
    "expected_status": "AUTO_APPROVED | EXPEDITED_REVIEW | STANDARD_REVIEW | LIKELY_DENIED",
    "estimated_processing_time": "string - e.g., 'Instant', '24-48 hours', '5-7 business days'",
    "confidence_score": 0.0-1.0,
    "reasoning": "string - 2-3 sentence explanation of why this pathway applies",
    "documentation_checklist": [
        {{
            "item": "string - what document/information is needed",
            "status": "AVAILABLE | NEEDED | OPTIONAL",
            "source": "string - where to get it (EHR, physician, lab)"
        }}
    ],
    "admin_cost_savings_estimate": "string - estimated admin savings vs manual process",
    "appeal_risk": "LOW | MEDIUM | HIGH",
    "alternative_pathways": ["list of backup pathways if primary is rejected"]
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=POLICY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"error": "Parse failed"}

        await self.ledger.write(
            source=AgentSource.POLICY,
            event_type="PATHWAY_DETERMINATION",
            message=(
                f"Authorization pathway determined: {result.get('recommended_pathway', 'UNKNOWN')}. "
                f"Expected status: {result.get('expected_status', 'UNKNOWN')}. "
                f"Processing time: {result.get('estimated_processing_time', 'Unknown')}. "
                f"Confidence: {result.get('confidence_score', 0):.0%}."
            ),
            data=result,
            tags=["PATHWAY", result.get("expected_status", "UNKNOWN")],
            severity=Severity.CRITICAL,
        )

        return result
