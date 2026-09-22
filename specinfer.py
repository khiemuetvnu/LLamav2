from typing import Optional
import torch
from tqdm import tqdm
from inference import LLaMa
import time



class SpecInfer():
    def __init__(self, Mp: LLaMa, Mq: LLaMa, eps: float = 1e-4):
        self.Mp = Mp
        self.Mq = Mq
        self.args = self.Mq.args
        self.eps = eps
        self.tokenizer = self.Mq.tokenizer

    def _build_tree_mask(self, exp_cfg, device):
        parent_indices = [-1]
        prev_start, prev_count = 0, 1
        for value in exp_cfg:
            for p_idx in range(prev_count):
                for _ in range (value):
                    parent_indices.append(prev_start + p_idx)
            prev_start += prev_count
            prev_count *= value
        
        n = len(parent_indices)
        mask = torch.full((n, n), float("-inf"), device=device)
        for i in range(n):
            cur = i
            while cur != -1:
                mask[i, cur] = 0.0
                cur = parent_indices[cur]
        return mask, parent_indices

    
    def text_completion(self, prompts: list[str], exp_cfg: list[int], temperature: float = 0.6, top_p: float = 0.9, top_k: int = 3, max_gen_len: Optional[int] = None):
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
    
        process_bar = tqdm(total = total_len - max_prompt_len, desc = "Generating Tokens")
        tree_mask, parent_indices = self._build_tree_mask(exp_cfg, device)
        #SpecInfer
        cur_pos = max_prompt_len
        while cur_pos < total_len and not eos_reached.all():
            #Drafting
            #Example exp_config = [3, 2]
            draft_tokens = []
            draft_tokens_vocab_probs = []
            logical_positions = []

            draft_tokens.append(tokens[:, cur_pos - 1:cur_pos]) # (B, 1)
            #draft_tokens_vocab_probs.append(torch.zeros((batch_size, 1, self.args.vocab_size), device = device))

            logical_positions.append(cur_pos - 1)
            branching = 1
            tracking_index = 0
            for value in exp_cfg:
                branching *= value
                for _ in range(branching):
                    logical_positions.append(cur_pos + tracking_index)
                tracking_index += 1
            del branching, tracking_index

            prev_start, prev_count = 0, 1
            for level, value in enumerate(exp_cfg):
                prev_tokens = torch.cat(draft_tokens[0 : prev_start + prev_count], dim=1) #(B, all_current_draft_token)
                
                pos_tensor = torch.tensor(logical_positions[0 : prev_start + prev_count], device = device) #(all_current_draft_token, )
                
                with torch.no_grad():
                    logits_q = self.Mq.model.forward(tokens = prev_tokens, start_pos = cur_pos - 1, 
                                                    logical_pos = pos_tensor, mask = tree_mask[: prev_start + prev_count, : prev_start + prev_count], 
                                                    use_cache = False) #(B, all_draft_token, vocab_size)
                    
                    sub_probs_q = torch.softmax(logits_q[:, prev_start: prev_start + prev_count], dim = -1) #(B, prev_count, vocab_size)
                    prob_value, token_idx = torch.topk(sub_probs_q, value, dim = -1) #(B, prev_count, value)
                    for p_idx in range(prev_count):
                        for v in range(value):
                            draft_tokens.append(token_idx[:, p_idx, v : v+1])  # (B, 1)
                            #draft_tokens_vocab_probs.append() #(B, 1)
                    prev_start += prev_count
                    prev_count *= value
            
            #Evaluation
            O = torch.cat(draft_tokens[:], dim = 1) # (B, all_draft_token)

            logits_p = self.Mp.model.forward(tokens = O, start_pos = cur_pos - 1, 
                                            logical_pos = torch.tensor(logical_positions, device = device), mask = tree_mask, 
                                            use_cache = False) #(B, all_draft_token, vocab_size)

            probs_p = torch.softmax(logits_p, dim = -1) #(B, all_draft_token, vocab_size)

            # alignment
            N_vocab_probs = torch.zeros_like(probs_p) #(B, all_draft_token, vocab_size)
            for i, p in enumerate(parent_indices):
                if p != -1:
                    N_vocab_probs[:, i] = probs_p[:, p]

            # Accept and Reject
            V = self.verify_greedy(O, N_vocab_probs, parent_indices) # (B, accepted_token)
            if V is None:
                actual_accepted_token = 1
                tokens[:, cur_pos : cur_pos + actual_accepted_token] = self.Mp._sample_top_k(probs_p[:, 0], k = top_k)
                self.Mq.model.forward(tokens[:, cur_pos - 1 : cur_pos + actual_accepted_token - 1], start_pos = cur_pos - 1)
                self.Mp.model.forward(tokens[:, cur_pos - 1 : cur_pos + actual_accepted_token - 1], start_pos = cur_pos - 1)
            else:
                actual_accepted_token = min(total_len - cur_pos, V.shape[-1])
                tokens[:, cur_pos : cur_pos + actual_accepted_token] = V[:, : actual_accepted_token] # (B, accepted_token)
                self.Mq.model.forward(tokens[:, cur_pos - 1 : cur_pos + actual_accepted_token - 1], start_pos = cur_pos - 1)
                self.Mp.model.forward(tokens[:, cur_pos - 1 : cur_pos + actual_accepted_token - 1], start_pos = cur_pos - 1)

            for i in range(actual_accepted_token):
                eos_reached |= (tokens[:, cur_pos + i] == self.tokenizer.eos_id())
            process_bar.update(actual_accepted_token)
            cur_pos += actual_accepted_token    
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


    def verify_greedy(self, O: torch.tensor, N_vocab_probs: torch.tensor, parent_index: list):
        # O (B, all_draft_token)
        # N (B, all_draft_token, vocab_size)
        # parent_index (all_draft_token, )
        root = 0
        V = []
        keeping = True
        while keeping:
            keeping = False
            for index, p_index in enumerate(parent_index):
                if p_index == root and (O[:, index] == torch.argmax(N_vocab_probs[:, index], dim = -1)).all():
                    keeping = True
                    V.append(O[:, index : index+1]) #(B, 1)
                    root = index 
                    break
        if len(V) != 0:
            return torch.cat(V[:], dim = 1) # (B, accepted_token)
        return None

    def verify_stochastic(self, O: torch.tensor, N_vocab_probs: torch.tensor, parent_index: list):

        return None


            
                

