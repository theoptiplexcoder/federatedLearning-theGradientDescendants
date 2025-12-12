# run_fedmd_multi_rounds.py
"""
Multi-round FedMD controller.

Behaviour per round:
  1) Optionally regenerate seeds (currently generates only on round=1)
  2) Run each client: calls client/run_client.py with appropriate args
  3) Aggregate logits on server
  4) Run distillation on server to produce global_v{round}.pth

You can adjust `clients` list below or pass a --clients-json config path.
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fedmd_controller")

PROJECT_ROOT = Path(__file__).resolve().parent
SEEDS_SCRIPT = PROJECT_ROOT / "server" / "seeds_generator.py"
CLIENT_RUN = PROJECT_ROOT / "client" / "run_client.py"
AGGREGATE = PROJECT_ROOT / "server" / "aggregate_logits.py"
DISTILL = PROJECT_ROOT / "server" / "distill.py"
INIT_GLOBAL = PROJECT_ROOT / "global_model" / "init_global_model.py"

DEFAULT_CLIENTS = [
    # Example clients: adjust local_model_path to your actual saved local models
    {"client_id": "C1", "client_type": "segmentation", "local_model_path": "client/local_polyp.pth"},
    {"client_id": "C2", "client_type": "classification", "local_model_path": "client/local_lung.pth"}
]


def run_cmd(cmd: List[str]):
    logger.info("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        raise SystemExit(result.returncode)


def run_clients(clients, round_number, dp_clip=1.0, noise_multiplier=0.5):
    for c in clients:
        cmd = [
            sys.executable,  # use same python interpreter
            str(CLIENT_RUN),
            "--client_id", c["client_id"],
            "--client_type", c["client_type"],
            "--local_model_path", c["local_model_path"],
            "--round", str(round_number),
            "--dp_clip", str(dp_clip),
            "--noise_multiplier", str(noise_multiplier)
        ]
        run_cmd(cmd)


def main(num_rounds: int = 3, clients=None, dp_clip=1.0, noise_multiplier=0.5, regenerate_seeds_on_first_round=True):
    clients = clients or DEFAULT_CLIENTS

    # initialize global model if missing
    init_path = PROJECT_ROOT / "global_models" / "global_v0.pth"
    if not init_path.exists():
        logger.info("No global_v0 found — creating initial global model.")
        run_cmd([sys.executable, str(INIT_GLOBAL)])

    # Run rounds
    for r in range(1, num_rounds + 1):
        logger.info(f"\n=== START ROUND {r} ===")

        # generate seeds on round 1 (or optionally every round if desired)
        if r == 1 and regenerate_seeds_on_first_round:
            logger.info("Generating seeds for round 1")
            run_cmd([sys.executable, str(SEEDS_SCRIPT)])

        # Run all clients (sequentially). You can parallelize by launching them in the background if desired.
        logger.info("Running clients...")
        run_clients(clients, r, dp_clip=dp_clip, noise_multiplier=noise_multiplier)

        # Aggregate
        logger.info("Aggregating logits on server (TEE-sim simulated)...")
        run_cmd([sys.executable, str(AGGREGATE), "--round", str(r)])

        # Distill / update global model
        logger.info("Distilling global model...")
        run_cmd([sys.executable, str(DISTILL), "--round", str(r), "--epochs", "5"])

        logger.info(f"=== END ROUND {r} ===\n")

    logger.info("Finished all rounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--clients_json", type=str, default=None,
                        help="Optional path to a JSON file listing clients (overrides DEFAULT_CLIENTS).")
    parser.add_argument("--dp_clip", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.5)
    args = parser.parse_args()

    clients = None
    if args.clients_json:
        clients_path = Path(args.clients_json)
        if not clients_path.exists():
            raise FileNotFoundError(f"Clients JSON not found: {clients_path}")
        clients = json.loads(clients_path.read_text())

    main(num_rounds=args.num_rounds, clients=clients, dp_clip=args.dp_clip, noise_multiplier=args.noise_multiplier)
