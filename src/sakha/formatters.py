from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class StatusIcons:
    OK = "✓"
    FAIL = "✗"
    SKIP = "!"

    @staticmethod
    def for_score(score: float) -> str:
        if score >= 0.5:
            return StatusIcons.OK
        elif score >= 0.3:
            return StatusIcons.SKIP + " "
        else:
            return StatusIcons.FAIL

    @staticmethod
    def for_status(status: str) -> str:
        if status == "success":
            return StatusIcons.OK
        elif status == "no_effect":
            return StatusIcons.SKIP + " "
        else:
            return StatusIcons.FAIL


class StatusLabels:
    DONE = "✓"
    INCOMPLETE = "✗"


class ScoreBar:
    @staticmethod
    def make(score: float, length: int = 20) -> str:
        filled = int(score * length)
        if score >= 0.5:
            return "▓" * filled + "░" * (length - filled)
        elif score >= 0.3:
            return "▒" * filled + "░" * (length - filled)
        else:
            return "░" * length


@dataclass
class StepData:
    task: str
    episode: int
    step_num: int
    action_name: str
    patient_id: int | None
    reward: float
    status: str
    done: bool


@dataclass
class EpisodeResult:
    task: str
    episode: int
    seed: int
    score: float
    steps: int
    done: bool
    runtime_seconds: float
    critical_incidents_missed: int


class Formatter(ABC):
    @abstractmethod
    def start_episode(
        self,
        task: str,
        episode: int,
        seed: int,
        patient_count: int,
        max_steps: int,
        mode: str,
    ) -> None:
        pass

    @abstractmethod
    def step(self, data: StepData) -> None:
        pass

    @abstractmethod
    def end_episode(self, result: EpisodeResult) -> None:
        pass

    @abstractmethod
    def summary(self, results: list[EpisodeResult]) -> None:
        pass


class CompactFormatter(Formatter):
    def start_episode(
        self,
        task: str,
        episode: int,
        seed: int,
        patient_count: int,
        max_steps: int,
        mode: str,
    ) -> None:
        mode_value = "deterministic" if mode == "deterministic" else "llm"
        print(
            f"[START] task={task} episode={episode} seed={seed} "
            f"mode={mode_value} max_steps={max_steps}",
            flush=True,
        )

    def step(self, data: StepData) -> None:
        patient_id_str = str(data.patient_id) if data.patient_id is not None else "None"
        reward_str = f"{data.reward:.4f}"
        done_str = str(data.done).lower()
        print(
            f"[STEP] task={data.task} episode={data.episode} step={data.step_num} "
            f"action={data.action_name} patient_id={patient_id_str} "
            f"reward={reward_str} done={done_str} status={data.status}",
            flush=True,
        )

    def end_episode(self, result: EpisodeResult) -> None:
        print(
            f"[END] task={result.task} episode={result.episode} seed={result.seed} "
            f"score={result.score:.4f} steps={result.steps} done={str(result.done).lower()}",
            flush=True,
        )

    def summary(self, results: list[EpisodeResult]) -> None:
        task_scores: dict[str, list[float]] = {}
        for r in results:
            task_scores.setdefault(r.task, []).append(r.score)

        print("\n" + "=" * 40)
        print(" SCORES BY TASK")
        print("=" * 40)
        for task, scores in task_scores.items():
            avg = sum(scores) / len(scores)
            print(f"  {task:<8} {avg:>6.4f}  (over {len(scores)} episodes)")

        print("\n" + "=" * 40)
        print(" FULL RESULTS")
        print("=" * 40)
        for r in results:
            done_icon = "✓" if r.done else "✗"
            incidents = (
                f" | {r.critical_incidents_missed} missed" if r.critical_incidents_missed else ""
            )
            print(
                f"  {r.task:<8} seed={r.seed:<4} score={r.score:.4f} "
                f"steps={r.steps:<2} {done_icon}{incidents}"
            )


class PrettyFormatter(Formatter):
    def start_episode(
        self,
        task: str,
        episode: int,
        seed: int,
        patient_count: int,
        max_steps: int,
        mode: str,
    ) -> None:
        task_label = task.upper()
        print()
        print("┌" + "─" * 56 + "┐")
        print(f"│ {task_label:^50} │")
        print(f"│    Patients: {patient_count:<43} │")
        print(f"│    Seed: {seed:<48} │")
        print("└" + "─" * 56 + "┘")

    def step(self, data: StepData) -> None:
        patient_str = f"#{data.patient_id}" if data.patient_id else "—"
        reward_str = f"{data.reward:+.2f}" if data.reward != 0 else "0.00"
        status_icon = StatusIcons.for_status(data.status)
        action_name = data.action_name
        print(
            f"  Step {data.step_num:>2}: {action_name:<18} "
            f"patient={patient_str:<3} {status_icon}  reward={reward_str}"
        )

    def end_episode(self, result: EpisodeResult) -> None:
        score_icon = StatusIcons.for_score(result.score)
        score_str = f"{result.score:.4f}"

        if result.critical_incidents_missed:
            incidents_str = f" | {result.critical_incidents_missed} incidents missed"
        else:
            incidents_str = ""
        print(f"  → Score: {score_str} ({score_icon} complete){incidents_str}")

    def summary(self, results: list[EpisodeResult]) -> None:
        print("\n" + "═" * 50)
        print("          SAKHA BENCHMARK RESULTS")
        print("═" * 50)

        task_scores: dict[str, list[float]] = {}
        for r in results:
            task_scores.setdefault(r.task, []).append(r.score)

        for task, scores in task_scores.items():
            avg = sum(scores) / len(scores)
            bar = ScoreBar.make(avg)
            task_label = task.upper()
            print(f"\n  {task_label}")
            print(f"    Avg Score: {avg:.4f}  [{bar}]")
            print(f"    Episodes:  {len(scores)}")

        print("\n" + "─" * 50)
        for r in results:
            done_icon = StatusIcons.for_score(r.score)
            if r.critical_incidents_missed:
                incidents = f" ({r.critical_incidents_missed} missed)"
            else:
                incidents = ""
            print(
                f"  {r.task:<8} seed={r.seed:<4} | "
                f"score={r.score:.4f} | {done_icon} | "
                f"{r.steps} steps | {r.runtime_seconds:.3f}s{incidents}"
            )
        print("─" * 50)


class JSONFormatter(Formatter):
    def __init__(self, output_file: str | None = None):
        self.output_file = output_file
        self.results: list[dict] = []

    def start_episode(
        self,
        task: str,
        episode: int,
        seed: int,
        patient_count: int,
        max_steps: int,
        mode: str,
    ) -> None:
        pass

    def step(self, data: StepData) -> None:
        pass

    def end_episode(self, result: EpisodeResult) -> None:
        self.results.append(
            {
                "task": result.task,
                "episode": result.episode,
                "seed": result.seed,
                "grader_score": result.score,
                "steps": result.steps,
                "done": result.done,
                "runtime_seconds": result.runtime_seconds,
                "critical_incidents_missed": result.critical_incidents_missed,
            }
        )

    def summary(self, results: list[EpisodeResult]) -> None:
        import json

        output = {"episodes": self.results}
        if self.output_file:
            Path(self.output_file).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2))


def get_formatter(name: str = "compact", output_file: str | None = None) -> Formatter:
    formatters = {
        "compact": CompactFormatter,
        "pretty": PrettyFormatter,
        "json": JSONFormatter,
    }
    formatter_cls = formatters.get(name, CompactFormatter)
    if output_file and name == "json":
        return formatter_cls(output_file)
    return formatter_cls()
