import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from typing import Any, Dict, Optional, Tuple


def check_compliance(code_snippet: str) -> bool:
    """Check if the code snippet complies with the following PEP8 rules: E101, E111, E112."""
    if dfa_e101(code_snippet) == "E101":
        return False
    if dfa_e111(code_snippet) == "E111":
        return False
    if dfa_e112(code_snippet) == "E112":
        return False
    return True


def dfa_e101(code_snippet):
    state = 'START'

    for char in code_snippet:
        if state == 'START':
            if char == ' ':
                state = 'SPACE'
            elif char == '\t':
                state = 'TAB'
        elif state == 'SPACE':
            if char == '\t':
                return "E101"
            elif char == '\n':
                state = 'START'
            elif char != ' ':
                state = 'CODE'
        elif state == 'TAB':
            if char == ' ':
                return "E101"
            elif char == '\n':
                state = 'START'
            elif char != '\t':
                state = 'CODE'
        elif state == 'CODE':
            if char == '\n':
                state = 'START'

    return ""


def dfa_e111(code_snippet):
    lines = code_snippet.split('\n')

    for line in lines:
        # count leading spaces
        leading_spaces = len(line) - len(line.lstrip(' '))
        # if there are leading spaces, check if they are a multiple of four
        if leading_spaces > 0 and leading_spaces % 4 != 0:
            return "E111"

    return ""


def dfa_e112(code_snippet):
    lines = code_snippet.split('\n')
    indent_required = False

    for line in lines:
        stripped_line = line.strip()

        # Check if the line is a statement that requires an indented block
        if any(stripped_line.startswith(keyword) and stripped_line.endswith(':') for keyword in
               ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with', 'elif', 'else']):
            indent_required = True
        elif stripped_line and not stripped_line.startswith('#'):  # Skip empty lines and comments
            if indent_required:
                # Check if the next line is indented
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces == 0:
                    return "E112"
                indent_required = False

    return ""


class DFATrainer(Trainer):
    """DFA trainer with custom loss function."""

    def __init__(self, penalty_value: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.penalty_value = penalty_value

    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, Any], return_outputs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the loss for the model, including DFA penalties."""
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if logits is None or labels is None:
            raise ValueError("Logits and labels must be present in the inputs")

        class_weights = torch.tensor([1.0, 1.0], device=logits.device)  # dataset is balanced so no need for adjustments
        loss_func = CrossEntropyLoss(weight=class_weights)

        loss = loss_func(logits, labels)

        input_ids = inputs.get("input_ids")
        code_snippets = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Create an array indicating compliance (1 for compliant, 0 for non-compliant)
        compliance_array = torch.tensor([1 if check_compliance(code_snippet) else 0 for code_snippet in code_snippets],
                                        dtype=torch.float, device=logits.device)

        # Compute penalty for non-compliance using the compliance array
        non_compliance_penalty = self.penalty_value * (1 - compliance_array).sum()

        # Normalize the penalty by the batch size to ensure it scales appropriately
        non_compliance_penalty /= len(code_snippets)

        # Add the penalty to the loss
        loss += non_compliance_penalty

        loss += non_compliance_penalty

        return (loss, outputs) if return_outputs else loss
