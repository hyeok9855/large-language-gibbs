import json
import pickle
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pgmpy.example_models import load_model
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models.DiscreteBayesianNetwork import DiscreteBayesianNetwork

from structure_learning.utils.bn_utils import (
    get_feature_renames,
    get_state_renames,
    get_true_domains,
)

STRUCTURE_LEARNING_DIR = Path(__file__).parent


if __name__ == "__main__":
    argparse = ArgumentParser(description="Get a dataset from pgmpy")
    argparse.add_argument("--datasets", nargs="+", required=True)
    argparse.add_argument("--n_samples", type=int, default=100)
    argparse.add_argument("--seed", type=int, default=42)
    args = argparse.parse_args()

    if args.datasets is None:
        raise ValueError("datasets should be provided.")

    for dataset_name in args.datasets:
        pgmpy_model = load_model(dataset_name)

        base_dir = STRUCTURE_LEARNING_DIR / "datasets" / dataset_name.replace("/", "_")
        base_dir.mkdir(parents=True, exist_ok=True)

        assert isinstance(pgmpy_model, DiscreteBayesianNetwork), (
            "This script supports only DiscreteBayesianNetwork models for now."
        )

        # download the .Rd file
        filename = dataset_name.split("/")[-1]
        if not (base_dir / f"{filename}.Rd").exists():
            url = "https://raw.githubusercontent.com"
            if "bnlearn" in dataset_name:
                url = f"{url}/cran/bnlearn/refs/heads/master/man/{filename}.Rd"
            elif "bnrep" in dataset_name:
                url = f"{url}/manueleleonelli/bnRep/refs/heads/master/man/{filename}.Rd"
            else:
                raise ValueError(f"Unknown model name: {dataset_name}")

            with urllib.request.urlopen(url) as response:
                rd_file = response.read().decode("utf-8")
                with open(base_dir / f"{filename}.Rd", "w") as f:
                    f.write(rd_file)

        if not (base_dir / "meta_data.json").exists():
            domains = get_true_domains(pgmpy_model, dataset_name)
            meta_data = {
                "dataset_description": "null",  # !! The description should be added by the user !!
                "field": "null",  # !! The field should be added by the user !!
                "features": {
                    feature_name: {
                        "description": "null",  # !! The description should be added by the user !!
                        "schema": {"type": "string", "enum": domains[feature_name]},
                    }
                    for feature_name in pgmpy_model.nodes()
                },
            }

            for raw_name, llm_name in get_feature_renames(dataset_name).items():
                meta_data["features"][llm_name] = meta_data["features"].pop(raw_name)

            with open(base_dir / "meta_data.json", "w") as f:
                json.dump(meta_data, f, indent=4)

        if not (base_dir / f"data_n{args.n_samples}_sd{args.seed}.csv").exists():
            # Generate data and save as a CSV file
            df = pgmpy_model.simulate(n_samples=args.n_samples, seed=args.seed)

            state_renames = get_state_renames(dataset_name)
            if state_renames:
                for col in df.columns:
                    df[col] = df[col].cat.rename_categories(state_renames)

            df = df[list(pgmpy_model.nodes())]
            df.to_csv(base_dir / f"data_n{args.n_samples}_sd{args.seed}.csv", index=False)

        if not (base_dir / "gt_graph.csv").exists():
            # Save the ground truth graph as a CSV file
            edges = '"Cause","Effect"\n'
            for e in pgmpy_model.edges():
                edges += f'"{e[0]}","{e[1]}"\n'
            with open(base_dir / "gt_graph.csv", "w") as f:
                f.write(edges)

        if not (base_dir / "cpds.pkl").exists():
            # Save CPDs as a pickle file (dictionary of pandas objects for each node)
            state_renames = get_state_renames(dataset_name)

            def cpd_to_dataframe(cpd: TabularCPD) -> pd.DataFrame | pd.Series:
                node = cpd.variable
                try:
                    df_or_series = cpd.to_dataframe()
                    if state_renames:
                        df_or_series = df_or_series.rename(  # type: ignore
                            columns=state_renames, index=state_renames
                        )
                except ValueError:  # This happens for the root nodes.
                    df_or_series = pd.Series(
                        cpd.values, index=pd.Index(cpd.state_names[node], name=node)
                    )
                    if state_renames:
                        df_or_series = df_or_series.rename(index=state_renames)

                return df_or_series

            cpds = {cpd.variable: cpd_to_dataframe(cpd) for cpd in pgmpy_model.cpds}
            with open(base_dir / "cpds.pkl", "wb") as f:
                pickle.dump(cpds, f)

        try:
            if not (base_dir / "graph.png").exists():
                # Save the graph as a PNG file
                graphviz_model = pgmpy_model.to_graphviz()
                graphviz_model.draw(base_dir / "graph.png", prog="dot")
        except ImportError as e:
            print(f"Error when generating graph.png: {e}")
            print("Skipping graph.png generation...")
