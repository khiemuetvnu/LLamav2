import torch
from pathlib import Path
from inference import LLaMa
from speculative_decoding import Speculative_Decoding

if __name__ == '__main__':

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Is chatgpt gay?"
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

    model_p_1 = LLaMa.build(
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

    # Vanilla Decoding
    torch.manual_seed(0)
    out_tokens_0, out_texts_0 = model_p_1.text_completion(
        prompts,
        temperature= 0.8,
        max_gen_len= 256
    )
    assert len(out_texts_0) == len(prompts)
    for i in range(len(out_texts_0)):
        print(f'{out_texts_0[i]}')
        print('-' * 150)


    spec_decoding = Speculative_Decoding(model_p, model_q)
    # Speculative Decoding
    torch.manual_seed(0)
    out_tokens_1, out_texts_1 = spec_decoding.text_completion(
        prompts,
        gamma = 5,
        temperature = 0.8,
        max_gen_len= 256
    )
    assert len(out_texts_1) == len(prompts)
    for i in range(len(out_texts_1)):
        print(f'{out_texts_1[i]}')
        print('-' * 150)