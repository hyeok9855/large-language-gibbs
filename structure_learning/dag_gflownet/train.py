import numpy as np
import optax
import networkx as nx
import pickle
import jax
from tqdm import trange
from numpy.random import default_rng

from structure_learning.dag_gflownet.env import GFlowNetDAGEnv
from structure_learning.dag_gflownet.gflownet import DAGGFlowNet
from structure_learning.dag_gflownet.utils.replay_buffer import ReplayBuffer
from structure_learning.dag_gflownet.utils.gflownet import posterior_estimate

from structure_learning.utils.eval_utils import expected_shd, expected_edges, threshold_metrics
from structure_learning.utils.score_utils import get_scorer
from structure_learning.utils.misc_utils import STRUCTURE_LEARNING_DIR


OUTPUT_FOLDER = STRUCTURE_LEARNING_DIR / "results"


def main(args):
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    scorer, data, graph = get_scorer(args, rng=rng)

    # Create the output folder
    dirname = args.graph if args.graph != "bn" else args.dataset_name
    output_folder = OUTPUT_FOLDER / dirname / f"n{len(data)}" / args.exp_name
    # If results.json already exists, raise an error
    if (output_folder / "results.json").exists():
        raise FileExistsError(f"Results already exist for {output_folder}")
    output_folder.mkdir(exist_ok=True, parents=True)

    # Create the environment
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs, scorer=scorer, num_workers=args.num_workers, context=args.mp_context
    )

    # Create the replay buffer
    replay = ReplayBuffer(args.replay_capacity, num_variables=env.num_variables)

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(delta=args.delta, update_target_every=args.update_target_every)
    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey, optimizer, replay.dummy["adjacency"], replay.dummy["mask"]
    )
    exploration_schedule = jax.jit(
        optax.linear_schedule(
            init_value=0.0,
            end_value=1.0 - args.min_exploration,
            transition_steps=args.num_iterations // 2,
            transition_begin=args.prefill,
        )
    )

    # Try-finally block to ensure that the environment is closed even if an error occurs
    try:
        # Training loop
        indices = None
        observations = env.reset()
        with trange(
            args.prefill + args.num_iterations, desc="Training", dynamic_ncols=True
        ) as pbar:
            for iteration in pbar:
                # Sample actions, execute them, and save transitions in the replay buffer
                epsilon = exploration_schedule(iteration)
                actions, key, logs = gflownet.act(params.online, key, observations, epsilon)
                next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
                indices = replay.add(
                    observations,
                    actions,
                    logs["is_exploration"],
                    next_observations,
                    delta_scores,
                    dones,
                    prev_indices=indices,
                )
                observations = next_observations

                if iteration >= args.prefill:
                    # Update the parameters of the GFlowNet
                    samples = replay.sample(batch_size=args.batch_size, rng=rng)
                    params, state, logs = gflownet.step(params, state, samples)

                    pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

        # Evaluate the posterior estimate
        posterior, _ = posterior_estimate(
            gflownet,
            params.online,
            env,
            key,
            num_samples=args.num_samples_posterior,
        )

        # Compute the metrics
        ground_truth = nx.to_numpy_array(graph, weight=None)  # type: ignore
        results = {
            "expected_shd": expected_shd(posterior, ground_truth),
            "expected_edges": expected_edges(posterior),
            **threshold_metrics(posterior, ground_truth),
        }
        print(f"Expected SHD: {results['expected_shd']:.6f}")
        print(f"Expected edges: {results['expected_edges']:.6f}")
        print(f"ROC AUC: {results['roc_auc']:.6f}")

        full_posterior = None
        exact_posterior = None
        if env.num_variables <= 5:
            from structure_learning.dag_gflownet.utils.exact_gflownet import (
                posterior_exact,
            )
            from structure_learning.dag_gflownet.utils.exhaustive import (
                get_full_posterior,
                get_gfn_exact_posterior,
            )
            from structure_learning.dag_gflownet.utils.jsd import (
                jensen_shannon_divergence,
            )

            full_posterior = get_full_posterior(data, scorer, verbose=True)
            exact_posterior = get_gfn_exact_posterior(
                posterior_exact(gflownet, params, data.columns)
            )

            jsd = jensen_shannon_divergence(full_posterior, exact_posterior)
            results["jsd"] = jsd
            print(f"Jensen-Shannon divergence: {jsd:.6f}")

        # Save model, data & results
        with open(output_folder / "arguments.json", "w") as f:
            json.dump(vars(args), f, default=str, indent=4)
        data.to_csv(output_folder / "data.csv")
        with open(output_folder / "graph.pkl", "wb") as f:
            pickle.dump(graph, f)
        # io.save(output_folder / "model.npz", params=params.online)
        # replay.save(output_folder / "replay_buffer.npz")
        np.save(output_folder / "posterior.npy", posterior)
        with open(output_folder / "results.json", "w") as f:
            json.dump(results, f, default=list, indent=4)

        if env.num_variables <= 5:
            assert full_posterior is not None and exact_posterior is not None
            full_posterior.save(output_folder / "full_posterior.npz")
            exact_posterior.save(output_folder / "exact_posterior.npz")

    finally:
        env.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description="DAG-GFlowNet for Strucure Learning.")

    # Environment
    environment = parser.add_argument_group("Environment")
    environment.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: %(default)s)",
    )
    environment.add_argument(
        "--scorer_kwargs", type=json.loads, default="{}", help="Arguments of the scorer."
    )
    environment.add_argument(
        "--prior",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "erdos_renyi",
            "edge",
            "fair",
            "llm_data",
            "llm_edge_matrix_bernoulli",
            "llm_edge_matrix_l1",
        ],
        help="Prior over graphs (default: %(default)s)",
    )
    environment.add_argument(
        "--prior_kwargs", type=json.loads, default="{}", help="Arguments of the prior over graphs."
    )

    # Optimization
    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: %(default)s)"
    )
    optimization.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Value of delta for Huber loss (default: %(default)s)",
    )
    optimization.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: %(default)s)"
    )
    optimization.add_argument(
        "--num_iterations",
        type=int,
        default=100_000,
        help="Number of iterations (default: %(default)s)",
    )

    # Replay buffer
    replay = parser.add_argument_group("Replay Buffer")
    replay.add_argument(
        "--replay_capacity",
        type=int,
        default=100_000,
        help="Capacity of the replay buffer (default: %(default)s)",
    )
    replay.add_argument(
        "--prefill",
        type=int,
        default=1000,
        help="Number of iterations with a random policy to prefill "
        "the replay buffer (default: %(default)s)",
    )

    # Exploration
    exploration = parser.add_argument_group("Exploration")
    exploration.add_argument(
        "--min_exploration",
        type=float,
        default=0.1,
        help="Minimum value of epsilon-exploration (default: %(default)s)",
    )
    exploration.add_argument(
        "--update_epsilon_every",
        type=int,
        default=10,
        help="Frequency of update for epsilon (default: %(default)s)",
    )

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--num_samples_posterior",
        type=int,
        default=1000,
        help="Number of samples for the posterior estimate (default: %(default)s)",
    )
    misc.add_argument(
        "--update_target_every",
        type=int,
        default=1000,
        help="Frequency of update for the target network (default: %(default)s)",
    )
    misc.add_argument("--seed", type=int, default=0, help="Random seed (default: %(default)s)")
    misc.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers (default: %(default)s)"
    )
    misc.add_argument(
        "--mp_context",
        type=str,
        default="spawn",
        help="Multiprocessing context (default: %(default)s)",
    )
    misc.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment directory name (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(help="Type of graph", dest="graph")

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser("erdos_renyi_lingauss")
    er_lingauss.add_argument("--num_variables", type=int, required=True, help="Number of variables")
    er_lingauss.add_argument("--num_edges", type=int, required=True, help="Average number of edges")
    er_lingauss.add_argument("--num_samples", type=int, required=True, help="Number of samples")

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser("sachs_continuous")

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser("sachs_interventional")

    # bnlearn or bnrep models
    bnlearn = subparsers.add_parser("bn")
    bnlearn.add_argument("--dataset_name", type=str, required=True, help="Name of the model")
    bnlearn.add_argument("--data_path", type=str, default=None, help="Path to the data")
    bnlearn.add_argument("--num_samples", type=int, default=None, help="Number of samples")
    bnlearn.add_argument(
        "--data_seed", type=int, default=None, help="Random seed for data generation"
    )

    args = parser.parse_args()

    main(args)
