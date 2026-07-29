import argparse
from datetime import datetime
import importlib
import json
import os
import sys

from isaaclab.app import AppLauncher

MAX_INPROC_RESTARTS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--env_cfg_type",
    type=str,
    required=True,
    help="config file name for evaluation",
)
parser.add_argument("--device_id", type=int, required=True, help="the device id for current process")
parser.add_argument(
    "--policy_name",
    type=str,
    required=True,
    help="XPolicyLab module name for deployment",
)
parser.add_argument("--port", type=int, required=True, help="the port for the policy WebSocket server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="IP address or hostname of the policy server. Defaults to localhost.",
)
parser.add_argument(
    "--protocol",
    choices=("ws",),
    default="ws",
    help=(
        "Env-to-policy transport. 'ws' is the default WebSocket protocol "
        "(msgpack frames over ws://host:port); also set as protocol: ws in deploy.yml."
    ),
)
parser.add_argument(
    "--policy_runtime",
    choices=("xpolicy_ws_v0", "robodojo_policy_v1"),
    default="xpolicy_ws_v0",
    help="Application-level policy API; independent from the WebSocket transport.",
)
parser.add_argument(
    "--action_type",
    type=str,
    default="",
    help="Policy action-space label used for runtime compatibility checks.",
)
parser.add_argument(
    "--policy_seed",
    type=int,
    default=None,
    help="Episode sampling seed for policy-v1; defaults to --seed.",
)
parser.add_argument("--policy_connect_timeout_s", type=float, default=30.0)
parser.add_argument("--policy_request_timeout_s", type=float, default=600.0)
parser.add_argument("--policy_close_timeout_s", type=float, default=10.0)
parser.add_argument(
    "--policy_server_url",
    type=str,
    default="",
    help=(
        "Full WebSocket URL for the policy server (e.g. ws://127.0.0.1:9999). "
        "Built from --host and --port when omitted."
    ),
)
parser.add_argument(
    "--additional_info",
    type=str,
    required=True,
    help="additional information for the evaluation",
)


parser.add_argument("--seed", type=int, required=True, help="policy seed for eval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.policy_seed is None:
    args_cli.policy_seed = args_cli.seed
if args_cli.policy_runtime == "robodojo_policy_v1":
    if not 0 <= args_cli.policy_seed <= (1 << 32) - 1:
        parser.error("--policy_seed must be in [0, 2^32 - 1] for policy-v1")
    if args_cli.env_cfg_type != "arx_x5":
        parser.error("policy-v1 currently requires --env_cfg_type arx_x5")
    if args_cli.action_type != "joint":
        parser.error("policy-v1 currently requires --action_type joint")
    for timeout_name in (
        "policy_connect_timeout_s",
        "policy_request_timeout_s",
        "policy_close_timeout_s",
    ):
        if getattr(args_cli, timeout_name) <= 0:
            parser.error(f"--{timeout_name} must be positive")

# Safe to import before AppLauncher: env is a namespace package (no __init__)
# and GLOBAL_CONFIGS only imports os, so this pulls in no app-dependent code.
from env.global_configs import BENCHMARK, ROOT_DIR

task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


class PhysXBrokenError(Exception):
    pass


class PhysXFatalError(Exception):
    pass


def get_monitor():
    return None


def _physx_monitor_needed(task_name) -> bool:
    """Enable the PhysX log monitor only for tasks whose Config declares a
    non-empty `Articulation` section (those bodies trigger the PhysX
    "Invalid PhysX transform" / CUDA failures we recover from). Read with a
    lightweight yaml load before AppLauncher; any failure falls back to
    enabled (fail-safe).
    """
    cfg_path = task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name)
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("Articulation"))
    except Exception:
        return True


# Fix the run id for this entire eval invocation (across all in-process
# os.execv self-restarts and bash-level retries). When eval_policy.sh
# launches us it exports ROBODOJO_RUN_ID up-front; if a developer runs
# main.py directly we generate one here and propagate it via the
# environment so subsequent execv calls see the same value.
if not os.environ.get("ROBODOJO_RUN_ID"):
    os.environ["ROBODOJO_RUN_ID"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}"

enable_monitor = _physx_monitor_needed(args_cli.task_name)
print(f"[main] PhysX monitor enabled={enable_monitor} (task={args_cli.task_name})")
if enable_monitor:
    # Start before AppLauncher so Kit inherits the redirected stdout/stderr fds.
    from src.eval_client.physx_warning_monitor import (
        PhysXBrokenError,
        PhysXFatalError,
        get_monitor,
    )

    get_monitor().start(enabled=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omegaconf import OmegaConf

from env.global_configs import *
from src.eval_client.eval_env import create_eval_env
from src.eval_client.intervention_loop import (
    InterventionAcceptedAndExit,
    InterventionDiscardedAndExit,
    InterventionRejected,
    InterventionSavedForRetry,
)
from src.eval_client.lerobot_stream_recorder import (
    LeRobotStreamStartupError,
    close_lerobot_stream_session,
)
from src.eval_client.observation_loop import ObservationAdvance, ObservationExit
from src.eval_client.piperx_bridge_client import (
    PiperXBridgeError,
    close_piperx_bridge_session,
)
from src.eval_client.policy_runtime import PolicyClientError, ResetReason
from utils.cluttered_generator import UnStableError
from utils.load_file import load_yaml
from utils.pipeline_utils import *

BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)


def _load_policy_deploy(policy_name):
    deploy_yml_path = os.path.join(ROOT_DIR, "XPolicyLab", "policy", policy_name, "deploy.yml")
    return load_yaml(deploy_yml_path) if os.path.isfile(deploy_yml_path) else {}


def _resume_manifest_path(eval_cfg, run_id):
    """Mirror of EvalEnv.resume_manifest_path() so we can load BEFORE
    constructing the env. Keeping the layout aligned avoids drift between
    the writer and reader paths.
    """
    return os.path.join(
        "eval_result",
        BENCHMARK,
        eval_cfg["task_name"],
        eval_cfg["policy_name"],
        eval_cfg["config_name"],
        f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        f"_resume_{run_id}.json",
    )


def _load_resume_manifest(eval_cfg, run_id):
    """Return parsed manifest dict, or None if no resume is in progress."""
    path = _resume_manifest_path(eval_cfg, run_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"[main] failed to load resume manifest at {path}: {e}; ignoring.")
        return None
    print(
        f"[main] resuming from manifest {path} "
        f"(success={data.get('success_nums')} fail={data.get('fail_nums')} "
        f"completed={len(data.get('completed_layout_ids') or [])} "
        f"abandoned={len(data.get('abandoned_layout_ids') or [])} "
        f"restart_count={data.get('restart_count', 0)})"
    )
    return data


def _delete_resume_manifest(env):
    """Best-effort cleanup at normal completion. Failure is non-fatal."""
    try:
        path = env.resume_manifest_path()
    except Exception:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
            print(f"[main] removed resume manifest {path} (eval completed)")
    except Exception as e:
        print(f"[main] failed to unlink resume manifest {path}: {e}")


def _close_model_client(env):
    """Best-effort graceful close for policy communication."""
    try:
        model_client = getattr(env, "model_client", None)
        close = getattr(model_client, "close", None)
        if callable(close):
            close()
    except Exception as e:
        print(f"[main] failed to close model client: {e}")


def _restart_or_exit(env, simulation_app, fatal_msg):
    """Persist progress and either os.execv-restart or sys.exit(99).

    Bounded by ROBODOJO_FATAL_RESTART_COUNT env var so a persistent hardware
    failure cannot put us in an infinite loop. The bash retry loop in
    eval_policy.sh provides a second layer of bounded restarts.
    """
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0")) + 1
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        print(f"[FATAL] persist_resume_manifest failed: {e}")
    print(
        f"[FATAL] PhysX kernel failure detected: {fatal_msg}; persisted manifest. "
        f"In-process restart attempt {restart_count}/{MAX_INPROC_RESTARTS}."
    )
    # Release the strict policy server's single-model lease before execv.
    # Otherwise the replacement process can race the server's disconnect
    # cleanup and receive a terminal session_busy response during HELLO.
    _close_model_client(env)
    close_piperx_bridge_session()
    try:
        simulation_app.close()
    except Exception:
        pass
    if restart_count <= MAX_INPROC_RESTARTS:
        os.environ["ROBODOJO_FATAL_RESTART_COUNT"] = str(restart_count)
        print(f"[FATAL] os.execv self-restart with run_id={os.environ.get('ROBODOJO_RUN_ID')}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    print(f"[FATAL] in-process restart cap reached ({MAX_INPROC_RESTARTS}); exiting with rc=99 for bash-level retry.")
    sys.exit(99)


def _exit_for_shell_restart(env, fatal_msg):
    """Persist progress, then let eval_policy.sh restart a fresh process."""
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0"))
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        print(f"[FATAL] persist_resume_manifest failed: {e}")
    print(f"[FATAL] PhysX requested shell-level restart: {fatal_msg}; exiting with rc=99 for bash-level retry.")
    # os._exit skips normal cleanup; explicitly close the socket so the next
    # shell-launched client does not inherit a still-held Kai0 policy lease.
    _close_model_client(env)
    close_piperx_bridge_session()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(99)


def main():
    """Assemble the env config, build the eval env, and run the eval loop with
    PhysX crash/resume recovery until the requested episode count is reached.
    """
    task_name = args_cli.task_name
    num_envs = args_cli.num_envs
    policy_runtime = args_cli.policy_runtime
    control_mode = os.environ.get("ROBODOJO_CONTROL_MODE", "policy").strip().lower()
    if control_mode not in {
        "policy",
        "keyboard_intervention",
        "keyboard_observe",
        "piperx_sim_dagger",
    }:
        raise ValueError(
            "ROBODOJO_CONTROL_MODE must be 'policy', 'keyboard_intervention', "
            "'keyboard_observe', or 'piperx_sim_dagger', "
            f"got {control_mode!r}."
        )
    operator_driven = control_mode in {"keyboard_intervention", "piperx_sim_dagger"}
    observation_mode = control_mode == "keyboard_observe"
    if control_mode in {"keyboard_intervention", "keyboard_observe"}:
        if policy_runtime == "xpolicy_ws_v0" and args_cli.policy_name != "Pi_05":
            raise ValueError("Interactive keyboard modes are currently validated only for policy_name=Pi_05.")
        launcher_headless = bool(getattr(app_launcher, "_headless", getattr(args_cli, "headless", False)))
        if launcher_headless:
            raise ValueError(
                "Interactive keyboard mode needs the Isaac Sim window. Set ROBODOJO_HEADLESS=0, "
                "HEADLESS=0, and LIVESTREAM=0, then keep that window focused while operating."
            )
        if num_envs != 1:
            print(f"[main] {control_mode} forces num_envs {num_envs} -> 1")
            num_envs = 1
    if control_mode == "piperx_sim_dagger":
        if policy_runtime != "robodojo_policy_v1":
            raise ValueError("piperx_sim_dagger requires --policy_runtime robodojo_policy_v1")
        launcher_headless = bool(
            getattr(app_launcher, "_headless", getattr(args_cli, "headless", False))
        )
        if launcher_headless:
            raise ValueError(
                "piperx_sim_dagger requires the live Isaac Sim window; disable --headless"
            )
        if num_envs != 1:
            print(f"[main] {control_mode} forces num_envs {num_envs} -> 1")
            num_envs = 1
    if policy_runtime == "robodojo_policy_v1" and num_envs != 1:
        print(f"[main] policy-v1 forces num_envs {num_envs} -> 1")
        num_envs = 1
    eval_cfg_name = args_cli.env_cfg_type
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, eval_cfg_name + ".yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = num_envs
    eval_cfg["device_id"] = args_cli.device_id
    policy_deploy_cfg = (
        _load_policy_deploy(args_cli.policy_name)
        if policy_runtime == "xpolicy_ws_v0"
        else {}
    )
    eval_batch = (
        bool(policy_deploy_cfg.get("eval_batch", False))
        if policy_runtime == "xpolicy_ws_v0"
        else False
    )
    eval_cfg["eval_batch"] = eval_batch
    eval_cfg["policy_name"] = args_cli.policy_name
    eval_cfg["additional_info"] = args_cli.additional_info
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = enable_monitor
    eval_cfg["control_mode"] = control_mode
    eval_cfg["operator_driven"] = operator_driven
    eval_cfg["observation_mode"] = observation_mode
    eval_cfg["policy_runtime"] = policy_runtime

    deploy_cfg = {}
    deploy_cfg["policy_name"] = args_cli.policy_name
    deploy_cfg["port"] = args_cli.port
    deploy_cfg["host"] = args_cli.host
    deploy_cfg["protocol"] = args_cli.protocol
    deploy_cfg["policy_runtime"] = policy_runtime
    deploy_cfg["policy_server_url"] = args_cli.policy_server_url or f"ws://{args_cli.host}:{args_cli.port}"
    deploy_cfg["policy_seed"] = args_cli.policy_seed
    deploy_cfg["policy_connect_timeout_s"] = args_cli.policy_connect_timeout_s
    deploy_cfg["policy_request_timeout_s"] = args_cli.policy_request_timeout_s
    deploy_cfg["policy_close_timeout_s"] = args_cli.policy_close_timeout_s
    deploy_cfg["evaluation_id"] = os.environ["ROBODOJO_RUN_ID"]
    deploy_cfg["trial_id"] = f"{task_name}-{os.environ['ROBODOJO_RUN_ID']}"
    deploy_cfg["action_case_id"] = f"{task_name}_case"
    deploy_cfg["repeat_index"] = None
    for key in ("ws_keepalive", "ws_ping_interval_s", "ws_ping_timeout_s", "ws_request_timeout_s"):
        if key in policy_deploy_cfg:
            deploy_cfg[key] = policy_deploy_cfg[key]
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "scene",
                    eval_cfg["config"]["scene"] + ".yml",
                )
            ),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "camera",
                    eval_cfg["config"]["camera"] + ".yml",
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "robot",
                    eval_cfg["config"]["robot"] + ".yml",
                )
            ),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    capped_num_envs = resolve_random_task_num_envs(task_name, num_envs, env_cfg.sim)
    if capped_num_envs != num_envs:
        print(
            f"[main] Random task {task_name}: num_envs capped "
            f"{num_envs} -> {capped_num_envs} "
        )
    num_envs = capped_num_envs
    if not eval_batch and num_envs != 1:
        print(
            f"[main] eval_batch=false for policy runtime {policy_runtime}; "
            f"forcing num_envs {num_envs} -> 1"
        )
        num_envs = 1
    eval_cfg["num_envs"] = num_envs
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
    if policy_runtime == "robodojo_policy_v1":
        collect_freq = float(eval_cfg["observation"].get("collect_freq", 0))
        if collect_freq != 25.0:
            raise ValueError(
                "policy-v1 ARX X5 profile requires observation.collect_freq=25, "
                f"got {collect_freq}",
            )

    if os.environ.get("EVAL_NUM") and not operator_driven:
        _env_eval_num = os.environ.get("EVAL_NUM")
        if str(_env_eval_num).lower() != "native":
            eval_num = min(int(_env_eval_num), int(eval_num))
    eval_cfg["eval_num"] = eval_num

    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )

    env_cfg.sim.seed = [0 for _ in range(num_envs)]
    run_id = os.environ["ROBODOJO_RUN_ID"]
    resume_state = _load_resume_manifest(eval_cfg, run_id)
    env = create_eval_env(env_cfg, simulation_app, resume_state=resume_state)
    eval_time = env.success_nums + env.fail_nums
    if operator_driven:
        env.env_seeds = env.seed_manager.get_cyclic_seeds(max_count=1)
    elif eval_time >= eval_num:
        # Already complete on resume - nothing left to do.
        env.env_seeds = None
    else:
        env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
    operator_stop_requested = False
    operator_fatal_error = None
    observed_count = 0
    while env.env_seeds is not None:
        retry_round = False
        if enable_monitor:
            get_monitor().reset()
        bad_envs = None
        try:
            env.reset(seed=env.env_seeds)
            env.run_eval()
            env.seed_manager.eval_step()
            if observation_mode:
                observed_count += 1

        except PhysXFatalError as e:
            # Unrecoverable: GPU/CUDA context is dead. Persist progress
            # and re-exec (or sys.exit(99) for bash to restart).
            if not enable_monitor:
                raise
            if get_monitor().requires_shell_restart():
                _exit_for_shell_restart(env, str(e))
            _restart_or_exit(env, simulation_app, str(e))

        except PhysXBrokenError as e:
            # Monitor caught the warning in time.
            bad_envs = sorted(e.broken_envs)

        except UnStableError:
            if operator_driven:
                print("[Intervention] unstable reset has no candidate data; advancing to the next layout.")
            env.seed_manager.eval_step()
        except InterventionRejected:
            print("[Intervention] rejected attempt does not count; retrying the same layout.")
            env.set_next_policy_reset_reason(ResetReason.OPERATOR_RETRY)
            env.close()
            retry_round = True
        except InterventionSavedForRetry as request:
            print(
                "[Intervention] saved attempt does not consume the layout; "
                f"resetting the same layout. file={request.saved_path}"
            )
            env.set_next_policy_reset_reason(ResetReason.OPERATOR_RETRY)
            env.close()
            retry_round = True
        except InterventionAcceptedAndExit as request:
            print(
                "[Intervention] accepted final episode; closing collection session. "
                f"episode={getattr(request, 'saved_path', '')}"
            )
            env.seed_manager.eval_step()
            operator_stop_requested = True
        except InterventionDiscardedAndExit:
            print("[Intervention] discarded final attempt; closing collection session.")
            operator_stop_requested = True
        except ObservationAdvance:
            env.seed_manager.eval_step()
            observed_count += 1
            print(f"[Observer] observed {observed_count}/{eval_num}; loading the next layout.")
        except ObservationExit:
            print(f"[Observer] exit requested after {observed_count}/{eval_num} completed layout(s).")
            operator_stop_requested = True
        except LeRobotStreamStartupError as e:
            # Retrying cannot repair a bad codec, incompatible existing
            # dataset, missing environment, or a second writer holding the
            # dataset lock.  Stop cleanly instead of reloading Isaac forever.
            print(f"[Intervention][FATAL] {e}", flush=True)
            env.close()
            operator_fatal_error = e
            operator_stop_requested = True
        except PiperXBridgeError as e:
            # Hardware/protocol ambiguity is never replayable: the bridge has
            # already entered fail-closed hold/disable and the staged episode
            # was discarded by the control loop.
            print(f"[PiPER-X DAgger][FATAL] {e}", flush=True)
            env.close()
            operator_fatal_error = e
            operator_stop_requested = True
        except KeyboardInterrupt as e:
            print("[main] interrupted by operator; closing the current run.", flush=True)
            operator_fatal_error = e
            operator_stop_requested = True
        except PolicyClientError as e:
            print(
                f"[PolicyV1][FATAL] policy session cannot be replayed or reused: {e}",
                flush=True,
            )
            operator_fatal_error = e
            operator_stop_requested = True
        except Exception as e:
            import traceback

            print(
                f"[Eval] unhandled exception during reset/run_eval: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            if enable_monitor and get_monitor().is_fatal():
                fatal_msg = get_monitor().get_fatal_message() or str(e)
                if get_monitor().requires_shell_restart():
                    _exit_for_shell_restart(env, fatal_msg)
                _restart_or_exit(env, simulation_app, fatal_msg)
            bad = {i for i in get_monitor().get_broken_envs() if i < env.num_envs} if enable_monitor else set()
            if bad:
                print(
                    f"[PhysX] downstream exception {type(e).__name__} with "
                    f"monitor broken_envs={sorted(bad)}; treating as PhysX break."
                )
                bad_envs = sorted(bad)
            else:
                if operator_driven:
                    print(
                        "[Intervention] current candidate was discarded after an error; "
                        "retrying the same layout."
                    )
                    env.set_next_policy_reset_reason(ResetReason.SIMULATOR_RECOVERY)
                    env.close()
                    retry_round = True
                elif policy_runtime == "robodojo_policy_v1":
                    print(
                        "[PolicyV1][FATAL] local policy-v1 integration error; "
                        "the current session will not be reused.",
                        flush=True,
                    )
                    operator_fatal_error = e
                    operator_stop_requested = True
                else:
                    env.seed_manager.eval_step()

        if operator_stop_requested:
            break

        if bad_envs is not None:
            # Abandon broken-env seeds, refill from the seed queue, and
            # retry this round iff there is at least one real seed left.
            bad_seeds = env.get_seeds_for_envs(bad_envs)
            env.abandoned_seeds.update(bad_seeds)

            replacements = env.seed_manager.get_seeds(max_count=len(bad_envs)) or []
            bad_env_set = set(bad_envs)
            new_batch = [None] * env.num_envs
            for env_idx, seed in env.current_env_seed_map.items():
                if env_idx not in bad_env_set:
                    new_batch[env_idx] = seed
            for k, env_idx in enumerate(bad_envs):
                if k < len(replacements):
                    new_batch[env_idx] = replacements[k]
            env.env_seeds = new_batch

            real_remaining = sum(1 for s in env.env_seeds if s is not None)
            print(
                f"[PhysX] broken envs={bad_envs} -> abandon seeds={sorted(bad_seeds)}; "
                f"refill from queue={replacements}; new batch={env.env_seeds}; "
                f"real_remaining={real_remaining}"
            )
            env.set_next_policy_reset_reason(ResetReason.SIMULATOR_RECOVERY)
            env.close()
            if real_remaining == 0:
                print("[PhysX] no real seeds remaining in this batch, advancing.")
                retry_round = False
            else:
                retry_round = True

        if retry_round:
            continue

        if observation_mode:
            print(f"[Observer] progress: {observed_count}/{eval_num}")
            if observed_count >= eval_num:
                print(f"[Observer] completed requested {eval_num} layout(s); exiting.")
                break
        else:
            print(f"Success nums: {env.success_nums}, Fail nums: {env.fail_nums}, Unstable nums: {env.unstable_nums}")
        eval_time = env.success_nums + env.fail_nums
        if not operator_driven and eval_time >= eval_num:
            break

        if operator_driven:
            env.env_seeds = env.seed_manager.get_cyclic_seeds(max_count=1)
        elif observation_mode:
            env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - observed_count)
        else:
            env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
        if env.env_seeds is None:
            print("No saved layouts are available; exiting.")
            break
        if operator_driven:
            print(
                "[Intervention] loading "
                f"layout={env.env_seeds[0]} cycle={env.seed_manager.cycle_index}"
            )

        env.close()

    if operator_fatal_error is None:
        _delete_resume_manifest(env)
    else:
        try:
            env.persist_resume_manifest()
        except Exception as e:
            print(f"[main] failed to preserve resume manifest: {e}")
    close_lerobot_stream_session()
    close_piperx_bridge_session()
    _close_model_client(env)
    env.close()
    simulation_app.close()
    if operator_fatal_error is not None:
        raise operator_fatal_error


if __name__ == "__main__":
    main()
