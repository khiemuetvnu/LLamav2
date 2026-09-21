import torch
from pathlib import Path
from inference import LLaMa
from speculative_sampling import Speculative_Sampling
from specinfer import SpecInfer

if __name__ == '__main__':

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Can you tell me about the best football player in the world?"
        ]
    
    # Model P
    current_dir = Path(__file__).parent
    model_p = LLaMa.build(
        checkpoints_dir=str(current_dir / 'tinyllama-1.1b'),
        tokenizer_path=str(current_dir / 'tokenizer.model'),
        load_model=True,
        max_seq_len = 2048,
        max_batch_size=len(prompts),
        device=device
    )

    # Model Q
    model_q = LLaMa.build(
        checkpoints_dir=str(current_dir / 'llama-160m'),
        tokenizer_path=str(current_dir / 'tokenizer.model'),
        load_model=True,
        max_seq_len = 2048,
        max_batch_size=len(prompts),
        device=device
    )

    # SpecInfer
    torch.manual_seed(42)
    SpecInfer = SpecInfer(model_p, model_q)
    out_tokens_0, out_texts_0 = SpecInfer.text_completion(prompts, [2,2,1], max_gen_len= 512)
    assert len(out_texts_0) == len(prompts)
    for i in range(len(out_texts_0)):
        print(f'{out_texts_0[i]}')
        print('-' * 150)
