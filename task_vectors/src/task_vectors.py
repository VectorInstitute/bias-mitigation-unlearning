import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.utils import cached_file
import pdb
from tqdm import tqdm
class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint).state_dict()
                finetuned_state_dict = AutoModelForCausalLM.from_pretrained(finetuned_checkpoint).state_dict()
                #unbiased_state_dict = AutoModelForCausalLM.from_pretrained("").state_dict()
                pdb.set_trace()
                self.vector = {}
                for key in tqdm(pretrained_state_dict):
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in tqdm(pretrained_state_dict):
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--finetuned_model_path', type=str, required=True)
    parser.add_argument('--scaling_coef', type=float, required=True)
    parser.add_argument('--model_save_path', type = str, required=True)
    args = parser.parse_args()

    vector = TaskVector(args.pretrained_model_path, args.finetuned_model_path)
    print("TaskVector created")
    neg_tv = -vector
    unbiased_model = neg_tv.apply_to(args.pretrained_model_path, args.scaling_coef)
    print("unbiased_model created")
    unbiased_model.save_pretrained(args.model_save_path)
    print("unbiased_model saved")