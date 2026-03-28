from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from sakha.models import SakhaAction, SakhaObservation, WardState


class SakhaEnv(EnvClient[SakhaAction, SakhaObservation, State]):
    def _step_payload(self, action: SakhaAction) -> dict:
        return {
            "action_type": action.action_type,
            "patient_id": action.patient_id,
            "medicine_id": action.medicine_id,
            "reason_code": action.reason_code,
        }

    def _parse_result(self, payload: dict) -> StepResult[SakhaObservation]:
        obs_data = payload.get("observation", {})
        obs = SakhaObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            ward_state=WardState.model_validate(obs_data.get("ward_state", {})),
            pending_count=obs_data.get("pending_count", 0),
            time_remaining_minutes=obs_data.get("time_remaining_minutes", 0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
