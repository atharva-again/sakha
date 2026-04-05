from sakha.models import PatientState, SakhaObservation


def _final_patients(trajectory: list[SakhaObservation]) -> list[PatientState]:
    if not trajectory:
        return []
    return trajectory[-1].ward_state.patients


def _sum_metric(patients: list[PatientState], field_name: str) -> int:
    return sum(getattr(patient, field_name) for patient in patients)


def _ratio(numerator: int, denominator: int) -> float:
    return min(1.0, numerator / max(1, denominator))


def _action_quality(trajectory: list[SakhaObservation]) -> float:
    if len(trajectory) <= 1:
        return 0.0
    total_steps = len(trajectory) - 1
    productive = sum(
        1
        for observation in trajectory[1:]
        if observation.action_result is not None and observation.action_result.status == "success"
    )
    return _ratio(productive, total_steps)


def score_easy_task(trajectory: list[SakhaObservation]) -> float:
    patients = _final_patients(trajectory)
    if not patients:
        return 0.0

    med_completion = _ratio(
        _sum_metric(patients, "medication_tasks_completed"),
        _sum_metric(patients, "medication_tasks_completed")
        + sum(len(patient.medications_due) > 0 for patient in patients),
    )
    med_timeliness = _ratio(
        _sum_metric(patients, "medication_tasks_on_time"),
        _sum_metric(patients, "medication_tasks_completed"),
    )
    vitals_completion = _ratio(
        _sum_metric(patients, "vitals_tasks_completed"),
        _sum_metric(patients, "vitals_tasks_completed")
        + sum(patient.vitals_due for patient in patients),
    )
    vitals_timeliness = _ratio(
        _sum_metric(patients, "vitals_tasks_on_time"),
        _sum_metric(patients, "vitals_tasks_completed"),
    )
    overdue_penalty = min(0.25, _sum_metric(patients, "overdue_tasks") * 0.03)
    base = (
        0.3 * med_completion
        + 0.25 * med_timeliness
        + 0.2 * vitals_completion
        + 0.15 * vitals_timeliness
        + 0.1 * _action_quality(trajectory)
    )
    return round(max(0.0, min(1.0, base - overdue_penalty)), 4)


def score_medium_task(trajectory: list[SakhaObservation]) -> float:
    patients = _final_patients(trajectory)
    if not patients:
        return 0.0

    routine_score = score_easy_task(trajectory)
    total_incidents = _sum_metric(patients, "critical_incidents_total")
    admissions_completed = _sum_metric(patients, "admissions_completed")
    admissions_on_time = _sum_metric(patients, "admissions_on_time")
    resolved = _sum_metric(patients, "critical_incidents_resolved")
    resolved_on_time = _sum_metric(patients, "critical_incidents_resolved_in_time")
    documented = _sum_metric(patients, "documentation_count")
    alerts = _sum_metric(patients, "doctor_alert_count")
    critical_success = _ratio(resolved, total_incidents)
    critical_timeliness = _ratio(resolved_on_time, total_incidents)
    workflow_score = _ratio(min(documented, alerts), max(1, total_incidents))
    admission_score = _ratio(admissions_on_time, max(1, admissions_completed))
    missed_penalty = min(0.35, _sum_metric(patients, "critical_incidents_missed") * 0.08)
    base = (
        0.4 * routine_score
        + 0.22 * critical_success
        + 0.18 * critical_timeliness
        + 0.1 * workflow_score
        + 0.1 * admission_score
    )
    return round(max(0.0, min(1.0, base - missed_penalty)), 4)


def score_hard_task(trajectory: list[SakhaObservation]) -> float:
    patients = _final_patients(trajectory)
    if not patients:
        return 0.0

    medium_score = score_medium_task(trajectory)
    total_incidents = _sum_metric(patients, "critical_incidents_total")
    admissions_completed = _sum_metric(patients, "admissions_completed")
    admissions_on_time = _sum_metric(patients, "admissions_on_time")
    admissions_total = sum(1 for patient in patients if patient.admission_step > 0)
    resolved = _sum_metric(patients, "critical_incidents_resolved")
    resolved_on_time = _sum_metric(patients, "critical_incidents_resolved_in_time")
    discharge_score = _ratio(
        sum(1 for patient in patients if patient.discharge_prepared),
        max(1, len(patients) // 3),
    )
    review_score = _ratio(
        _sum_metric(patients, "reviews_completed"),
        max(len(patients), len(trajectory) // 6),
    )
    backlog_control = _ratio(resolved, total_incidents)
    admission_score = _ratio(admissions_on_time, max(1, admissions_completed))
    overdue_penalty = min(0.08, _sum_metric(patients, "overdue_tasks") * 0.001)
    workload_penalty = min(
        0.12,
        max(0, total_incidents - 4) * 0.015 + max(0, admissions_total - 2) * 0.005,
    )
    base = (
        0.3 * medium_score
        + 0.2 * _ratio(resolved, total_incidents)
        + 0.1 * _ratio(resolved_on_time, total_incidents)
        + 0.1 * discharge_score
        + 0.05 * review_score
        + 0.1 * backlog_control
        + 0.05 * admission_score
    )
    return round(max(0.0, min(1.0, base - overdue_penalty - workload_penalty)), 4)
