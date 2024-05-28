import argparse
import evaluate
import logging
import torch
import numpy as np
from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_device():
    """Determine the available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def empty_cache(device):
    """Empty cache for the specified device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            from torch.mps import empty_cache as mps_empty_cache
            mps_empty_cache()
        except ImportError:
            logger.warning("mps_empty_cache() function is not available!")


def load_model_and_tokenizer(model_name, device):
    """Load the appropriate model and tokenizer based on the model name."""
    if model_name == "llama2":
        model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=2).to(device)
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    elif model_name == "llama3":
        model = LlamaForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B", num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    else:
        raise ValueError("Invalid model name. Choose 'llama2' or 'llama3'.")
    return model, tokenizer


def evaluate_model(model_name, dataset, metric, device, num_iterations):
    """Evaluate the model over a specified number of iterations."""
    results = []

    for _ in tqdm(range(num_iterations), desc="Evaluating"):
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        result = evaluate.evaluator("text-classification").compute(
            model_or_pipeline=model,
            tokenizer=tokenizer,
            data=dataset["validation"],
            input_column="code",
            label_column="label",
            metric=metric,
            label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
            device=device.index,
        )
        results.append(result["accuracy"])

        # Unload the model to free up memory
        del model
        empty_cache(device)

    return results


def calculate_statistics(results):
    """Calculate mean accuracy and standard deviation."""
    mean_accuracy = np.mean(results)
    std_deviation = np.std(results)
    return mean_accuracy, std_deviation


def run(model_name, num_iterations):
    """Main function to run the evaluation."""
    device = get_device()
    logger.info(f"Using device: {device}")

    dataset = load_dataset("aholovko/pep8_indentation_compliance")
    metric = evaluate.load("accuracy")

    results = evaluate_model(model_name, dataset, metric, device, num_iterations)
    mean_accuracy, std_deviation = calculate_statistics(results)

    logger.info(f"Results: {results}")
    logger.info(f"Mean Accuracy: {mean_accuracy}")
    logger.info(f"Standard Deviation: {std_deviation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Evaluation for Llama 2 and Llama 3")
    parser.add_argument("--model", type=str, required=True, choices=["llama2", "llama3"], help="Model to evaluate")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    run(args.model, args.iterations)
