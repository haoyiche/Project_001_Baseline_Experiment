import yaml
import sys
from datetime import datetime
def load_config(config_path):
    with open(config_path, 'r',encoding = "utf-8") as file:
        config = yaml.safe_load(file)

    return config

def run_experiment(config):
    results = {}
    print("==============================")
    print("Experiment Start")
    print("==============================")

    print("Experiment",config['experiment_name'])

    print("learning_rate:", config['learning_rate'])
    print("batch_size:", config['batch_size'])
    print("epochs:", config['epochs'])

    results["experiment"] = config["experiment_name"]

    results["learning_rate"] = config["learning_rate"]

    results["batch_size"] = config["batch_size"]

    results["epochs"] = config["epochs"]


    # 模拟实验指标
    results["score"] = config["epochs"] * 2
    return results


def save_results(results, path):
    with open(path, 'w', encoding = "utf-8") as f:
        f.write("===============================\n")
        f.write("Experiment Results\n")
        f.write("===============================\n")
        f.write(f"Time: {datetime.now()}\n\n")
        for key,value in results.items():
            f.write(f"{key}: {value}\n")



if __name__ == "__main__":
    config_path = sys.argv[1]

    config = load_config(config_path)

    results = run_experiment(config)

    
    print("==============================")
    print("Experiment Start")
    print("==============================")


    for key, value in results.items():
        print(f"{key}: {value}")

    save_results(results, "results/exp001_results.txt")

    print("\nResults saved!")
