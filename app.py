import asyncio
import os
import sys
from pathlib import Path
from typing import List, TypedDict

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowConfig,
    FlowManager,
    FlowResult,
)

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class IdentityVerificationResult(FlowResult):
    is_valid: bool


class SymptomCheckResult(FlowResult):
    high_risk_present: bool


class AlertResult(FlowResult):
    alert_raised: bool



async def verify_patient_identity(args: FlowArgs) -> IdentityVerificationResult:
    """Handler for verifying patient identity."""
    patient_data = {
        "patient_id": "CL-P00123",
        "full_name": "Mrs. Kavita Sharma",
        "date_of_birth": "1980-08-12",
        "preferred_language": "Hinglish",
        "phone_number": "+91-98XXXXXX75",
        "diagnosis": "Stage IIIB Breast Cancer",
        "chemo_regimen": "FEC (5-Fluorouracil, Epirubicin, Cyclophosphamide)",
        "cycle_number": 1,
        "day_in_cycle": 7,
        "last_chemo_date": "2025-05-08",
        "high_risk_symptoms_to_check": [
            "Fever >=38°C (100.4 °F)",
            "Severe vomiting (>=4 episodes in 24 h or unable to keep liquids down)",
            "Shortness of breath or chest tightness"
        ],
        "baseline_side_effects_reported": [
            {
                "symptom": "Mild nausea",
                "grade": "1",
                "first_noted_on": "2025-05-10"
            }
        ],
        "current_supportive_medications": [
            "Ondansetron 8 mg PO q8h prn",
            "Pantoprazole 40 mg PO OD"
        ],
        "allergies": "None known",
        "treating_oncologist": "Dr Jaideep Singh",
        "next_followup_visit": "2025-05-20",
        "emergency_contact": {
            "name": "Mr Rajesh Sharma",
            "relationship": "Spouse",
            "phone": "+91-98XXXXXX43"
        }
    }

    full_name = args["full_name"]
    date_of_birth = args["date_of_birth"]

    is_valid = (full_name.lower() == patient_data["full_name"].lower() and date_of_birth == date_of_birth)
    return IdentityVerificationResult(is_valid=is_valid)



async def check_high_risk_symptoms(args: FlowArgs) -> SymptomCheckResult:
    """Handler for checking high-risk symptoms."""
    fever = args["fever"]
    vomiting = args["vomiting"]
    breathlessness = args["breathlessness"]
    any_high_risk = fever or vomiting or breathlessness
    return SymptomCheckResult(high_risk_present=any_high_risk)



async def raise_alert(args: FlowArgs) -> AlertResult:
    """Handler for raising an alert (API call simulation)."""
    symptoms_present = args["symptoms_present"]
    if symptoms_present:
        logger.warning("🚨 HIGH-RISK SYMPTOMS DETECTED! SIMULATING API CALL TO RAISE ALERT 🚨")
    return AlertResult(alert_raised=symptoms_present)



flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a healthcare agent for Tri-County Health Services. Your job is to follow up with patients after their treatment. Be professional and friendly. Always use the available functions.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Start by introducing yourself and asking for the patient's full name and date of birth to verify their identity.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "verify_patient_identity",
                        "handler": verify_patient_identity,
                        "description": "Verify the patient's identity using their full name and date of birth.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "full_name": {
                                    "type": "string",
                                    "description": "The patient's full name.",
                                },
                                "date_of_birth": {
                                    "type": "string",
                                    "description": "The patient's date of birth (YYYY-MM-DD).",
                                },
                            },
                            "required": ["full_name", "date_of_birth"],
                        },
                        "transition_to": "check_symptoms",
                    },
                },
            ],
        },
        "check_symptoms": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a healthcare agent for Tri-County Health Services. Your job is to follow up with patients after their treatment. Be professional and friendly. Always use the available functions.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Check for high-risk symptoms: Fever >=38°C (100.4 °F), Severe vomiting (>=4 episodes in 24 h or unable to keep liquids down), Shortness of breath or chest tightness.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "check_high_risk_symptoms",
                        "handler": check_high_risk_symptoms,
                        "description": "Check if the patient is experiencing any high-risk symptoms.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fever": {
                                    "type": "boolean",
                                    "description": "Whether the patient has a fever >=38°C (100.4 °F).",
                                },
                                "vomiting": {
                                    "type": "boolean",
                                    "description": "Whether the patient has severe vomiting (>=4 episodes in 24 h or unable to keep liquids down).",
                                },
                                "breathlessness": {
                                    "type": "boolean",
                                    "description": "Whether the patient has shortness of breath or chest tightness.",
                                },
                            },
                            "required": ["fever", "vomiting", "breathlessness"],
                        },
                        "transition_to": "handle_symptoms",
                    },
                },
            ],
        },
        "handle_symptoms": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a healthcare agent for Tri-County Health Services. Your job is to follow up with patients after their treatment. Be professional and friendly. Always use the available functions.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "If high-risk symptoms are present, raise a red flag.  If not, conclude the call.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "raise_alert",
                        "handler": raise_alert,
                        "description": "Raise an alert to the on-call medical team.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symptoms_present": {
                                    "type": "boolean",
                                    "description": "Indicates if high-risk symptoms are present.",
                                },
                            },
                            "required": ["symptoms_present"],
                        },
                        "transition_to": "end",
                    },
                },
            ],
            "post_actions": [{"type": "end_conversation"}]
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Conclude the call.",
                }
            ],
            "functions": [],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}



async def main():
    """Main function to set up and run the patient follow-up bot."""
    async with aiohttp.ClientSession() as session:
        room_url = "https://example.daily.co/myroom"
        transport = DailyTransport(
            room_url,
            None,
            "Patient Follow-up Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121", 
            text_filter=MarkdownTextFilter(),
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await flow_manager.initialize()

        runner = PipelineRunner()
        await runner.run(task)



if __name__ == "__main__":
    asyncio.run(main())
