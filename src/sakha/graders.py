from typing import Any

from sakha.models import SakhaEpisodeMetrics, SakhaObservation


def _compute_timeliness_score(on_time: int, late: int, missed: int) -> float:
    total = on_time + late + missed
    if total == 0:
        return 0.0
    return (on_time + 0.5 * late) / total


def _compute_completion_rate(completed: int, total: int) -> float:
    if total == 0:
        return 1.0
    return min(1.0, completed / total)


def _count_final_metrics(trajectory: list[SakhaObservation]) -> SakhaEpisodeMetrics:
    if not trajectory:
        return SakhaEpisodeMetrics()

    first = trajectory[0]
    last = trajectory[-1]

    if hasattr(last.ward_state, "total_admissions"):
        patients_admitted = last.ward_state.total_admissions
        patients_discharged = last.ward_state.total_discharges
        patients_waiting = len(last.ward_state.waiting_patients)
    else:
        patients_admitted = (
            len(first.ward_state.patients) if hasattr(first.ward_state, "patients") else 0
        )
        patients_discharged = 0
        patients_waiting = 0

    meds_administered = 0
    meds_on_time = 0
    meds_late = 0
    meds_missed = 0
    vitals_checked = 0
    vitals_on_time = 0
    vitals_late = 0
    vitals_missed = 0
    escalations = 0
    escalations_on_time = 0
    escalations_late = 0
    missed_escalations = 0
    deteriorations_handled = 0
    deteriorations_detected_early = 0
    documents_completed = 0
    handovers_completed = 0
    conflicts_resolved = 0

    for idx in range(len(trajectory) - 1):
        obs_before = trajectory[idx]
        obs_after = trajectory[idx + 1]

        if not hasattr(obs_before.ward_state, "patients"):
            continue

        before_by_id = {p.patient_id: p for p in obs_before.ward_state.patients}
        after_by_id = {p.patient_id: p for p in obs_after.ward_state.patients}

        for pid, patient_before in before_by_id.items():
            patient_after = after_by_id.get(pid)
            if patient_after is None:
                continue

            if hasattr(patient_before, "medication_schedules") and hasattr(
                patient_after, "medication_schedules"
            ):
                meds_before = sum(
                    1 for m in patient_before.medication_schedules if not m.administered
                )
                meds_after = sum(
                    1 for m in patient_after.medication_schedules if not m.administered
                )
                if meds_after < meds_before:
                    meds_administered += meds_before - meds_after
                    for med in patient_after.medication_schedules:
                        if (
                            med.administered
                            and med.administered_time == obs_after.ward_state.current_step
                        ):
                            if med.administered_time <= med.deadline:
                                meds_on_time += 1
                            else:
                                meds_late += 1

            if hasattr(patient_before, "medications_due") and hasattr(
                patient_after, "medications_due"
            ):
                legacy_before = len(patient_before.medications_due)
                legacy_after = len(patient_after.medications_due)
                if legacy_after < legacy_before:
                    count = legacy_before - legacy_after
                    meds_administered += count
                    meds_on_time += count

            if hasattr(patient_before, "vitals_schedules") and hasattr(
                patient_after, "vitals_schedules"
            ):
                vitals_before = sum(1 for v in patient_before.vitals_schedules if not v.checked)
                vitals_after = sum(1 for v in patient_after.vitals_schedules if not v.checked)
                if vitals_after < vitals_before:
                    vitals_checked += vitals_before - vitals_after
                    for v in patient_after.vitals_schedules:
                        if v.checked and v.checked_time == obs_after.ward_state.current_step:
                            if v.checked_time <= v.deadline:
                                vitals_on_time += 1
                            else:
                                vitals_late += 1

            if hasattr(patient_before, "vitals_due") and hasattr(patient_after, "vitals_due"):
                if patient_before.vitals_due and not patient_after.vitals_due:
                    vitals_checked += 1
                    vitals_on_time += 1

            if patient_before.escalation_level >= 2 and patient_after.escalation_level == 0:
                escalations += 1
                escalations_on_time += 1

            if hasattr(patient_before, "deterioration_events") and hasattr(
                patient_after, "deterioration_events"
            ):
                events_before = {e.event_id: e for e in patient_before.deterioration_events}
                events_after = {e.event_id: e for e in patient_after.deterioration_events}
                for eid, event_after in events_after.items():
                    event_before = events_before.get(eid)
                    if event_before and not event_before.responded and event_after.responded:
                        deteriorations_handled += 1
                    if (
                        event_before
                        and not event_before.early_warning_detected
                        and event_after.early_warning_detected
                    ):
                        deteriorations_detected_early += 1

            if hasattr(patient_before, "last_documented_step") and hasattr(
                patient_after, "last_documented_step"
            ):
                if patient_after.last_documented_step == obs_after.ward_state.current_step:
                    if patient_before.last_documented_step != obs_after.ward_state.current_step:
                        documents_completed += 1

            if hasattr(patient_before, "escalation_level") and hasattr(
                patient_after, "escalation_level"
            ):
                if patient_before.escalation_level > 0 and patient_after.medications_due:
                    if len(patient_after.medications_due) < len(patient_before.medications_due):
                        conflicts_resolved += 1

    final_patients = last.ward_state.patients if hasattr(last.ward_state, "patients") else []
    for patient in final_patients:
        if hasattr(patient, "escalation_level") and patient.escalation_level >= 2:
            missed_escalations += 1
        if hasattr(patient, "deterioration_events"):
            for event in patient.deterioration_events:
                if not event.responded:
                    missed_escalations += 1

        meds_remaining = len(getattr(patient, "medications_due", []))
        meds_remaining += sum(
            1 for m in getattr(patient, "medication_schedules", []) if not m.administered
        )
        if meds_remaining > 0:
            meds_missed += meds_remaining

        vitals_remaining = sum(1 for v in getattr(patient, "vitals_schedules", []) if not v.checked)
        if vitals_remaining > 0:
            vitals_missed += vitals_remaining

    if hasattr(last.ward_state, "nurse"):
        handovers_completed = 1 if last.ward_state.nurse.handover_completed else 0

    return SakhaEpisodeMetrics(
        patients_admitted=patients_admitted,
        patients_discharged=patients_discharged,
        patients_waiting=patients_waiting,
        meds_administered=meds_administered,
        meds_on_time=meds_on_time,
        meds_late=meds_late,
        vitals_checked=vitals_checked,
        vitals_on_time=vitals_on_time,
        vitals_late=vitals_late,
        escalations=escalations,
        escalations_on_time=escalations_on_time,
        escalations_late=escalations_late,
        missed_escalations=missed_escalations,
        deteriorations_handled=deteriorations_handled,
        deteriorations_detected_early=deteriorations_detected_early,
        documents_completed=documents_completed,
        handovers_completed=handovers_completed,
        conflicts_resolved=conflicts_resolved,
    )


def compute_diagnostic_breakdown(trajectory: list[SakhaObservation]) -> dict[str, Any]:
    if not trajectory:
        return {
            "easy_score": 0.0,
            "medium_score": 0.0,
            "hard_score": 0.0,
            "patients_admitted": 0,
            "patients_discharged": 0,
            "meds_administered": 0,
            "vitals_checked": 0,
            "escalations": 0,
            "missed_escalations": 0,
            "handovers_completed": 0,
            "documents_completed": 0,
            "adverse_events_triggered": 0,
            "timeliness_score": 0.0,
            "safety_violation": True,
            "failure_tags": ["empty_trajectory"],
        }

    metrics = _count_final_metrics(trajectory)

    med_timeliness = _compute_timeliness_score(
        metrics.meds_on_time, metrics.meds_late, metrics.meds_missed
    )
    vitals_timeliness = _compute_timeliness_score(
        metrics.vitals_on_time, metrics.vitals_late, metrics.vitals_missed
    )
    escalation_timeliness = _compute_timeliness_score(
        metrics.escalations_on_time, metrics.escalations_late, metrics.missed_escalations
    )

    late_escalation_rate = (
        round(metrics.escalations_late / max(1, metrics.escalations), 4)
        if metrics.escalations > 0
        else 0.0
    )

    action_loop_count = 0
    window_count = 0
    for idx in range(3, len(trajectory)):
        prev = trajectory[idx - 3 : idx]
        if all(obs.pending_count == prev[0].pending_count for obs in prev):
            action_loop_count += 1
        window_count += 1
    loop_ratio = round(action_loop_count / max(1, window_count), 4)

    safety_violation = metrics.missed_escalations > 0

    failure_tags = []
    if safety_violation:
        failure_tags.append("missed_escalation")
    if loop_ratio > 0.35:
        failure_tags.append("looping_policy")
    if metrics.meds_missed > 0:
        failure_tags.append("missed_medications")
    if metrics.vitals_missed > 0:
        failure_tags.append("missed_vitals")

    adverse_events = 0
    for idx in range(len(trajectory) - 1):
        obs_before = trajectory[idx]
        obs_after = trajectory[idx + 1]
        before_by_id = {p.patient_id: p for p in obs_before.ward_state.patients}
        after_by_id = {p.patient_id: p for p in obs_after.ward_state.patients}
        for pid, pb in before_by_id.items():
            pa = after_by_id.get(pid)
            if (
                pa
                and getattr(pb, "escalation_level", 0) < 2
                and getattr(pa, "escalation_level", 0) >= 2
            ):
                adverse_events += 1

    return {
        "easy_score": score_easy_task(trajectory),
        "medium_score": score_medium_task(trajectory),
        "hard_score": score_hard_task(trajectory),
        "patients_admitted": metrics.patients_admitted,
        "patients_discharged": metrics.patients_discharged,
        "meds_administered": metrics.meds_administered,
        "meds_on_time": metrics.meds_on_time,
        "meds_missed": metrics.meds_missed,
        "vitals_checked": metrics.vitals_checked,
        "vitals_on_time": metrics.vitals_on_time,
        "vitals_missed": metrics.vitals_missed,
        "escalations": metrics.escalations,
        "escalations_on_time": metrics.escalations_on_time,
        "escalations_late": metrics.escalations_late,
        "missed_escalations": metrics.missed_escalations,
        "deteriorations_handled": metrics.deteriorations_handled,
        "deteriorations_detected_early": metrics.deteriorations_detected_early,
        "documents_completed": metrics.documents_completed,
        "handovers_completed": metrics.handovers_completed,
        "adverse_events_triggered": adverse_events,
        "timeliness_score": round(
            (med_timeliness + vitals_timeliness + escalation_timeliness) / 3, 4
        ),
        "late_escalation_rate": late_escalation_rate,
        "loop_ratio": loop_ratio,
        "safety_violation": safety_violation,
        "failure_tags": failure_tags,
    }


def score_easy_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    metrics = _count_final_metrics(trajectory)

    med_score = _compute_timeliness_score(
        metrics.meds_on_time, metrics.meds_late, metrics.meds_missed
    )
    vitals_score = _compute_timeliness_score(
        metrics.vitals_on_time, metrics.vitals_late, metrics.vitals_missed
    )

    total_steps = len(trajectory) - 1
    useful_actions = (
        metrics.meds_administered
        + metrics.vitals_checked
        + metrics.escalations
        + metrics.documents_completed
        + metrics.handovers_completed
    )
    efficiency = useful_actions / max(1, total_steps)

    safety_penalty = 0.1 * metrics.missed_escalations
    base = 0.35 * med_score + 0.35 * vitals_score + 0.3 * efficiency
    return round(max(0.0, min(1.0, base - safety_penalty)), 4)


def score_medium_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    metrics = _count_final_metrics(trajectory)

    med_score = _compute_timeliness_score(
        metrics.meds_on_time, metrics.meds_late, metrics.meds_missed
    )
    vitals_score = _compute_timeliness_score(
        metrics.vitals_on_time, metrics.vitals_late, metrics.vitals_missed
    )

    total_patients = max(1, metrics.patients_admitted)
    conflict_score = min(1.0, metrics.conflicts_resolved / max(1, total_patients // 2))

    total_steps = len(trajectory) - 1
    useful_actions = (
        metrics.meds_administered
        + metrics.vitals_checked
        + metrics.escalations
        + metrics.documents_completed
        + metrics.handovers_completed
    )
    efficiency = useful_actions / max(1, total_steps)

    safety_penalty = 0.1 * metrics.missed_escalations
    base = (
        0.35 * med_score
        + 0.3 * vitals_score
        + 0.12 * conflict_score
        + 0.08 * (metrics.patients_discharged / max(1, total_patients))
        + 0.15 * efficiency
    )
    return round(max(0.0, min(1.0, base - safety_penalty)), 4)


def score_hard_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    metrics = _count_final_metrics(trajectory)

    med_score = _compute_timeliness_score(
        metrics.meds_on_time, metrics.meds_late, metrics.meds_missed
    )
    vitals_score = _compute_timeliness_score(
        metrics.vitals_on_time, metrics.vitals_late, metrics.vitals_missed
    )

    total_patients = max(1, metrics.patients_admitted)
    escalation_score = min(1.0, metrics.escalations / max(1, total_patients // 3))
    deterioration_score = min(1.0, metrics.deteriorations_handled / max(1, total_patients // 3))

    doc_score = min(1.0, metrics.documents_completed / max(1, total_patients))
    handover_score = metrics.handovers_quality if metrics.handovers_completed > 0 else 0.5

    total_steps = len(trajectory) - 1
    useful_actions = (
        metrics.meds_administered
        + metrics.vitals_checked
        + metrics.escalations
        + metrics.documents_completed
        + metrics.handovers_completed
    )
    efficiency = useful_actions / max(1, total_steps)

    safety_penalty = 0.15 * metrics.missed_escalations

    base = (
        0.22 * med_score
        + 0.18 * vitals_score
        + 0.13 * escalation_score
        + 0.13 * deterioration_score
        + 0.09 * doc_score
        + 0.09 * handover_score
        + 0.04 * (metrics.patients_discharged / max(1, total_patients))
        + 0.12 * efficiency
    )
    return round(max(0.0, min(1.0, base - safety_penalty)), 4)
