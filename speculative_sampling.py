from typing import Optional
import torch
from tqdm import tqdm
from inference import LLaMa
import time

class Speculative_Sampling:
    def __init__(self, Mp: LLaMa, Mq: LLaMa, eps: float = 1e-4):
        self.Mp = Mp
        self.Mq = Mq    
        self.args = self.Mq.args
        self.eps = eps
        self.tokenizer = self.Mq.tokenizer

    def text_completion(self, prompts: list[str], gamma: int = 3, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 3, max_gen_len: Optional[int] = None):
        device = self.args.device
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # Convert each prompts into tokens
        prompt_tokens = self.tokenizer.encode(prompts, out_type = int, add_bos = True, add_eos = False)
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # Create the list that will contrain the generated tokens, along with the initital prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype = torch.long, device = device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype = torch.long, device = device)
        
        eos_reached = torch.tensor([False] * batch_size, device = device)
        prompt_tokens_mask = tokens != pad_id #True if the prompt token, False otherwise

        # Time benchmark.
        start = time.time()

        # Filling Prompt for both Model:
        for cur_pos in tqdm(range(1, max_prompt_len), desc = "Processing Prompts"):
            with torch.no_grad():
                #Mq
                logits_q = self.Mq.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos-1)
                #Mp
                logits_p = self.Mp.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos-1)

                if temperature > 0:
                    probs = torch.softmax(logits_p[:, -1] / temperature, dim = -1)
                    next_token = self.Mq._sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits_p[:, -1], dim = -1)

                next_token = next_token.reshape(-1)
                next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token
                eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

        # Speculative Decoding
        process_bar = tqdm(total = total_len - max_prompt_len, desc = "Generating Tokens")
        cur_pos = max_prompt_len
        while cur_pos < total_len and not eos_reached.all():
            #Gamma_actual
            Gamma_actual = min(gamma, total_len - cur_pos)

            # Mq distribution
            Mq_distribution = torch.full((batch_size, Gamma_actual, self.args.vocab_size), 0.0, dtype = torch.float, device = device)
            
            with torch.no_grad():
                for i in range(Gamma_actual):
                    logits_q = self.Mq.model.forward(tokens[:, cur_pos + i - 1:cur_pos + i], cur_pos + i - 1)
                    probs = torch.softmax(logits_q[:, -1] / (temperature + self.eps), dim = -1)
                    Mq_distribution[:, i] = probs

                    if temperature > 0:
                        next_token = self.Mq._sample_top_p(probs, top_p)
                    else:
                        next_token = torch.argmax(logits_q[:, -1], dim = -1)

                    next_token = next_token.reshape(-1)
                    tokens[:, cur_pos + i] = next_token
            
            # Mp Distribution and making a new adjusted distribution
            with torch.no_grad():
                logits_p = self.Mp.model.forward(tokens[:, cur_pos - 1: cur_pos + Gamma_actual], cur_pos - 1)
                Mp_distribution = torch.softmax(logits_p / (temperature + self.eps), dim = -1)
                accepted_count = 0
                for i in range(Gamma_actual):
                    draft_tokens = tokens[:, cur_pos + i] #(B, )

                    p = Mp_distribution[:, i].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1) #(B,)
                    q = Mq_distribution[:, i].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1) #(B,)

                    ratio = p.div(q + self.eps)
                    u = torch.rand(batch_size, device = device)

                    rejected = u >= ratio

                    if rejected.any():
                        resample_probs = torch.relu(Mp_distribution[:, i] - Mq_distribution[:, i])
                        resample_probs_sum = resample_probs.sum(dim = -1, keepdim = True).clamp_min(self.eps)

                        resample_probs.div_(resample_probs_sum)

                        if temperature > 0:
                            new_tokens = self.Mq._sample_top_p(resample_probs, top_p)
                        else:
                            new_tokens = torch.argmax(resample_probs, dim = -1)
                        # Attention!
                        new_tokens = new_tokens.reshape(-1)
                        tokens[:, cur_pos + i] = torch.where(rejected, new_tokens, draft_tokens)

                        break
                    
                    else:
                        accepted_count += 1
                
                # Đoạn này sẽ hơi khó hiểu một tí là tại sao lại làm như dưới, ví dụ như cur_pos = 10, thì giai đoạn sinh draft token ta chuyền các token 9, 10, 11.
                # Nếu Accept hết tất cả token mà mô hình Mq sinh thì tất nhiên về mặt trực giác là thiếu token thứ 12 trong KV cache nên ta phải forward nó.
                # Lúc này ta chưa cần forward token thứ 13 là token mà Mp đã sinh mà để KV cache mà ta sẽ nhảy curpos nên vị trí 14.
                # Khi tiếp tục sinh draft token, để đoán token thứ 14, mô hình sẽ nôi token thứ 13 trong tokens ra, như vậy vừa sinh được token nháp mà vừa lưu KV cache. 
                if accepted_count == Gamma_actual and cur_pos + accepted_count < total_len:
                    final_probs = Mp_distribution[:, Gamma_actual]
                    if temperature > 0:
                        next_token = self.Mq._sample_top_p(final_probs, top_p)
                    else:
                        next_token = torch.argmax(final_probs, dim = -1)
                    # Attention!
                    next_token = next_token.reshape(-1)
                    tokens[:, cur_pos + accepted_count] = next_token

                    self.Mq.model.forward(tokens[:, cur_pos + i - 1: cur_pos + i], cur_pos + i - 1)

                generated_tokens = min(accepted_count + 1, total_len - cur_pos)

                for i in range(generated_tokens):
                    eos_reached |= (tokens[:, cur_pos + i] == self.tokenizer.eos_id())
                process_bar.update(generated_tokens)
                cur_pos += generated_tokens
        process_bar.close()

        print(f"Generating tokens in {time.time() - start:.2f} seconds")

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)



                    
                    


        
