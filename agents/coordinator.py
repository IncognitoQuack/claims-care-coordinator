"""
Claims Coordinator — Orchestrates Clinical + Policy Agents

Manages the end-to-end flow:
1. Clinical Agent analyzes EHR data
2. Findings are written to the Medical Necessity Ledger (shared memory)
3. Policy Agent reads the ledger and adapts its search
4. Final authorization determination is produced

All events are streamed via SSE for real-time UI updates.
"""

import json
import asyncio
from anthropic import AsyncAnthropic
from models.schemas import AgentSource, Severity, EHRData, AuthorizationResult
from memory.ledger import MedicalNecessityLedger
from agents.clinical_agent import ClinicalAgent
from agents.policy_agent import PolicyAgent


class ClaimsCoordinator:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.ledger = MedicalNecessityLedger()
        self.clinical_agent = ClinicalAgent(self.client, self.ledger)
        self.policy_agent = PolicyAgent(self.client, self.ledger)

    async def process_claim(
        self, ehr: EHRData, policy_text: str, plan_name: str = "Insurance Plan"
    ) -> dict:
        """
        Process a claim end-to-end. Returns the complete result.
        Subscribe to self.ledger for real-time events.
        """
        # Clear previous state
        await self.ledger.clear()

        # System start
        await self.ledger.write(
            source=AgentSource.SYSTEM,
            event_type="PROCESS_START",
            message=f"Claims & Care Coordinator initiated for {ehr.patient_name}. Starting dual-agent analysis.",
            tags=["SYSTEM", "START"],
        )

        # Phase 1: Clinical Agent
        await self.ledger.write(
            source=AgentSource.SYSTEM,
            event_type="PHASE_CHANGE",
            message="Phase 1: Clinical Agent — Analyzing EHR data",
            data={"phase": "clinical"},
            tags=["PHASE", "CLINICAL"],
        )

        clinical_result = await self.clinical_agent.analyze_ehr(ehr)

        # Phase 2: Policy Agent (reads ledger from Phase 1)
        await self.ledger.write(
            source=AgentSource.SYSTEM,
            event_type="PHASE_CHANGE",
            message="Phase 2: Policy Agent — Searching policy with clinical context from shared ledger",
            data={"phase": "policy"},
            tags=["PHASE", "POLICY"],
        )

        policy_result = await self.policy_agent.analyze_policy(policy_text, plan_name)

        # Phase 3: Final determination
        await self.ledger.write(
            source=AgentSource.SYSTEM,
            event_type="PHASE_CHANGE",
            message="Phase 3: Generating final authorization determination",
            data={"phase": "resolution"},
            tags=["PHASE", "RESOLUTION"],
        )

        final_determination = await self._generate_final_determination(
            ehr, clinical_result, policy_result
        )

        await self.ledger.write(
            source=AgentSource.SYSTEM,
            event_type="PROCESS_COMPLETE",
            message=(
                f"Processing complete. Status: {final_determination.get('status', 'UNKNOWN')}. "
                f"Pathway: {final_determination.get('pathway', 'N/A')}."
            ),
            data=final_determination,
            tags=["SYSTEM", "COMPLETE", final_determination.get("status", "UNKNOWN")],
            severity=Severity.CRITICAL,
        )

        return {
            "clinical_analysis": clinical_result,
            "policy_analysis": policy_result,
            "determination": final_determination,
            "ledger": [e.model_dump() for e in await self.ledger.read_all()],
        }

    async def _generate_final_determination(
        self, ehr: EHRData, clinical_result: dict, policy_result: dict
    ) -> dict:
        """Generate the final, unified authorization determination."""
        ledger_context = await self.ledger.get_full_context()
        auth_pathway = policy_result.get("auth_pathway", {})

        prompt = f"""You are generating the FINAL authorization determination for a prior authorization request.
This determination synthesizes findings from both a Clinical Agent and a Policy Agent.

FULL LEDGER CONTEXT (all agent communications):
{ledger_context}

CLINICAL ANALYSIS SUMMARY:
- Necessity Level: {clinical_result.get('necessity_assessment', {}).get('necessity_level', 'UNKNOWN')}
- Justification: {clinical_result.get('necessity_assessment', {}).get('justification', 'N/A')}

POLICY ANALYSIS SUMMARY:
- Recommended Pathway: {auth_pathway.get('recommended_pathway', 'N/A')}
- Expected Status: {auth_pathway.get('expected_status', 'UNKNOWN')}
- Confidence: {auth_pathway.get('confidence_score', 0)}

PATIENT: {ehr.patient_name}
PROCEDURE: {ehr.requested_procedure.name if ehr.requested_procedure else 'N/A'} ({ehr.requested_procedure.code if ehr.requested_procedure else 'N/A'})

Generate the final determination. Respond with JSON:
{{
    "status": "APPROVED | DENIED | PENDING_REVIEW",
    "pathway": "string - the authorization pathway used",
    "determination_text": "string - 3-4 sentence formal determination that cites specific policy sections and clinical evidence",
    "reasoning": "string - detailed reasoning",
    "confidence_score": 0.0-1.0,
    "estimated_processing_time": "string",
    "admin_cost_savings": "string",
    "documentation_complete": true/false,
    "missing_items": ["list of any missing documentation"],
    "appeal_guidance": "string - if denied, what to do"
}}"""

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system="You are the final decision engine for a medical prior authorization system. Provide clear, well-reasoned determinations in JSON format only.",
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {
                "status": auth_pathway.get("expected_status", "PENDING_REVIEW"),
                "pathway": auth_pathway.get("recommended_pathway", "Standard Review"),
                "determination_text": "Determination generated from policy and clinical analysis.",
            }

        return result
