import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

DATASET_NAME = "aholovko/pep8_indentation_compliance"
LLAMA_2_MODEL_ID = "meta-llama/Llama-2-7b-hf"
LLAMA_3_MODEL_ID = "meta-llama/Meta-Llama-3-8B"


def _get_model_id(model_type):
    if model_type == "llama2":
        return LLAMA_2_MODEL_ID
    elif model_type == "llama3":
        return LLAMA_2_MODEL_ID
    else:
        raise ValueError("Unsupported model type specified")


class LlamaIndentationComplianceTrainer:
    def __init__(self, model_type, epochs, batch_size, learning_rate,
                 lora_r=12, lora_alpha=32, lora_dropout=0.1):
        self.model_type = model_type
        self.model_id = _get_model_id(model_type)
        self.dataset_name = DATASET_NAME
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Load and preprocess dataset
        self.indent_ds = load_dataset(self.dataset_name)
        self.id2label, self.label2id = self._create_label_mappings()
        self.tokenizer = self._initialize_tokenizer()
        self.indent_ds = self.indent_ds.map(self._tokenize_text, batched=True)

        # Define train and validation datasets
        self.train_dataset = self.indent_ds["train"]
        self.eval_dataset = self.indent_ds["validation"]

        # Initialize model
        self.model = self._initialize_model()
        self.model = self._apply_lora(self.model)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.metric = evaluate.load("accuracy")

        # Training arguments and trainer setup
        self.training_args = self._create_training_args()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = self._create_trainer()

    def _create_label_mappings(self):
        features = self.indent_ds["train"].features
        id2label = {idx: features["label"].int2str(idx) for idx in range(2)}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id

    def _initialize_tokenizer(self):
        if self.model_id == LLAMA_2_MODEL_ID:
            tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif self.model_id == LLAMA_3_MODEL_ID:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            return tokenizer
        else:
            raise ValueError("Unsupported model type specified")

    def _tokenize_text(self, examples):
        return self.tokenizer(examples["code"], truncation=True, max_length=512, padding='longest')

    def _initialize_model(self):
        return LlamaForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=2,
            device_map='mps',
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def _apply_lora(self, model):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def _create_training_args(self):
        return TrainingArguments(
            output_dir="llama-tuned",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_steps=len(self.train_dataset) // self.batch_size,
            save_strategy="no",
            load_best_model_at_end=False,
            push_to_hub=False,
        )

    def _create_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self):
        self.trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama model for indentation compliance classification")
    parser.add_argument("--model", type=str, required=True, choices=["llama2", "llama3"], help="Specify the model")
    parser.add_argument('--epochs', type=int, required=True, help="Specify the number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Specify the batch size for training")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Specify the learning rate for training")

    args = parser.parse_args()

    trainer = LlamaIndentationComplianceTrainer(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    trainer.train()
