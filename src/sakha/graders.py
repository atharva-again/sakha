from sakha.models import SakhaEpisodeMetrics, SakhaObservation


def _compute_med_score(meds_administered: int, total_patients: int) -> float:
    return min(1.0, meds_administered / max(1, total_patients))


def _compute_vitals_score(vitals_checked: int, total_patients: int) -> float:
    return min(1.0, vitals_checked / max(1, total_patients))


def _count_final_metrics(trajectory: list[SakhaObservation]) -> SakhaEpisodeMetrics:
    if not trajectory:
        return SakhaEpisodeMetrics()

    final_obs = trajectory[-1]
    initial_patients = trajectory[0].ward_state.patients
    final_patients = final_obs.ward_state.patients

    meds = 0
    vitals = 0
    escalations = 0
    missed_escalations = 0
    conflicts_resolved = 0
    deteriorations_handled = 0

    # Calculate medication administration and vitals checks
    for i, initial_p in enumerate(initial_patients):
        final_p = final_patients[i] if i < len(final_patients) else initial_p
        initial_meds = len(initial_p.medications_due)
        final_meds = len(final_p.medications_due)
        if final_meds < initial_meds:
            meds += initial_meds - final_meds
        if not initial_p.vitals_due and final_p.vitals_due:
            continue
        if initial_p.vitals_due and not final_p.vitals_due:
            vitals += 1

    # Calculate escalations (when escalation level goes from >=2 to 0)
    escalations = sum(
        1
        for p in final_patients
        if p.escalation_level == 0
        and any(orig.escalation_level >= 2 for orig in initial_patients if orig.bed_id == p.bed_id)
    )

    # Calculate missed escalations: count patients who had escalation_level >= 2 at any point but never got escalated
    # We need to check if any patient ever had escalation_level >= 2 but final escalation_level >= 2 (meaning never resolved)
    for i, initial_p in enumerate(initial_patients):
        final_p = final_patients[i] if i < len(final_patients) else initial_p
        # Check if patient ever had escalation >= 2 during the trajectory
        ever_had_escalation = False
        for obs in trajectory:
            patient_in_obs = next(
                (p for p in obs.ward_state.patients if p.bed_id == initial_p.bed_id),
                None,
            )
            if patient_in_obs and patient_in_obs.escalation_level >= 2:
                ever_had_escalation = True
                break
        # If they ever had escalation >= 2 but still have it >= 2 at end, it was missed
        if ever_had_escalation and final_p.escalation_level >= 2:
            missed_escalations += 1

    # Calculate conflicts resolved: medicine given to patient with escalation_level > 0
    # We need to check each step for medicine administration to patients with escalation > 0
    for step_idx in range(len(trajectory) - 1):
        obs_before = trajectory[step_idx]
        obs_after = trajectory[step_idx + 1]
        # Find patients who got medicine between these observations
        for patient_before in obs_before.ward_state.patients:
            patient_after = next(
                (p for p in obs_after.ward_state.patients if p.bed_id == patient_before.bed_id),
                None,
            )
            if patient_after:
                # Check if medicine was administered (medications_due decreased)
                if len(patient_after.medications_due) < len(patient_before.medications_due):
                    # And patient had escalation > 0 before the medicine was given
                    if patient_before.escalation_level > 0:
                        conflicts_resolved += 1

    # Calculate deteriorations handled: vitals checked on patient with escalation_level >= 2
    for step_idx in range(len(trajectory) - 1):
        obs_before = trajectory[step_idx]
        obs_after = trajectory[step_idx + 1]
        # Find patients who got vitals checked between these observations
        for patient_before in obs_before.ward_state.patients:
            patient_after = next(
                (p for p in obs_after.ward_state.patients if p.bed_id == patient_before.bed_id),
                None,
            )
            if patient_after:
                # Check if vitals were checked (vitals_due went from True to False)
                if patient_before.vitals_due and not patient_after.vitals_due:
                    # And patient had escalation >= 2 before vitals were checked
                    if patient_before.escalation_level >= 2:
                        deteriorations_handled += 1

    return SakhaEpisodeMetrics(
        meds_administered=meds,
        vitals_checked=vitals,
        escalations=escalations,
        missed_escalations=missed_escalations,
        conflicts_resolved=conflicts_resolved,
        deteriorations_handled=deteriorations_handled,
    )


def score_easy_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    total_patients = len(trajectory[0].ward_state.patients)
    if total_patients == 0:
        return 0.0

    final = _count_final_metrics(trajectory)
    med_score = _compute_med_score(final.meds_administered, total_patients)
    vitals_score = _compute_vitals_score(final.vitals_checked, total_patients)

    return round(0.5 * med_score + 0.5 * vitals_score, 4)


def score_medium_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    total_patients = len(trajectory[0].ward_state.patients)
    if total_patients == 0:
        return 0.0

    final = _count_final_metrics(trajectory)
    med_score = _compute_med_score(final.meds_administered, total_patients)
    vitals_score = _compute_vitals_score(final.vitals_checked, total_patients)
    conflict_score = min(1.0, final.conflicts_resolved / max(1, total_patients // 2))

    return round(0.4 * med_score + 0.3 * vitals_score + 0.3 * conflict_score, 4)


def score_hard_task(trajectory: list[SakhaObservation]) -> float:
    if not trajectory:
        return 0.0

    total_patients = len(trajectory[0].ward_state.patients)
    if total_patients == 0:
        return 0.0

    final = _count_final_metrics(trajectory)
    med_score = _compute_med_score(final.meds_administered, total_patients)
    vitals_score = _compute_vitals_score(final.vitals_checked, total_patients)
    escalation_score = min(1.0, final.escalations / max(1, total_patients // 4))
    deterioration_score = min(1.0, final.deteriorations_handled / max(1, total_patients // 3))
    penalty = final.missed_escalations * 0.05

    base = (
        0.3 * med_score + 0.2 * vitals_score + 0.25 * escalation_score + 0.25 * deterioration_score
    )
    return round(max(0.0, base - penalty), 4)
