import torch
import numpy as np
from torch.nn.functional import log_softmax, softmax

class Beam:
    def __init__(
        self,
        beam_size=8,
        min_length=0,
        n_top=1,
        ranker=None,
        start_token_id=1,
        end_token_id=2,
    ):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker
        self.end_token_id = end_token_id
        self.top_sentence_ended = False
        self.prev_ks = []
        
        # Initialize with LongTensor
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]
        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []
        self.finished = []
        self.n_top = n_top
        self.actual_beam_size = beam_size

    def get_current_state(self):
        """Get beam state as LongTensor with consistent size"""
        if not self.next_ys:
            return torch.LongTensor([[self.start_token_id]])
        
        # Use the size of the most recent next_ys
        current_size = self.next_ys[-1].size(0)
        
        # Ensure all tensors have the same size
        padded_ys = []
        for ys in self.next_ys:
            if ys.size(0) < current_size:
                # Pad with the last token
                padding = ys[-1].unsqueeze(0).repeat(current_size - ys.size(0))
                padded_ys.append(torch.cat([ys, padding]))
            elif ys.size(0) > current_size:
                # Truncate to current size
                padded_ys.append(ys[:current_size])
            else:
                padded_ys.append(ys)
        
        return torch.stack(padded_ys).long()

    def advance(self, next_log_probs):
        vocab_size = next_log_probs.size(-1)
        current_beam_size = next_log_probs.size(0)
        
        # Adjust current_scores to match the current beam size
        if self.current_scores.size(0) != current_beam_size:
            if self.current_scores.size(0) > current_beam_size:
                # Truncate if current_scores is larger
                self.current_scores = self.current_scores[:current_beam_size]
            else:
                # Pad if current_scores is smaller (fill with worst score)
                padding_size = current_beam_size - self.current_scores.size(0)
                worst_score = self.current_scores.min().item() - 1.0
                padding = torch.full((padding_size,), worst_score, dtype=self.current_scores.dtype)
                self.current_scores = torch.cat([self.current_scores, padding])
        
        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
        else:
            beam_scores = next_log_probs[0] if next_log_probs.dim() > 1 else next_log_probs
        
        flat_beam_scores = beam_scores.view(-1)
        
        # Don't exceed available elements
        self.actual_beam_size = min(self.beam_size, flat_beam_scores.size(0))
        best_scores, best_scores_id = flat_beam_scores.topk(self.actual_beam_size, 0, True, True)
        
        self.all_scores.append(self.current_scores[:self.actual_beam_size])
        self.current_scores = best_scores
        
        prev_k = best_scores_id // vocab_size
        next_y = best_scores_id - prev_k * vocab_size
        
        self.prev_ks.append(prev_k.long())
        self.next_ys.append(next_y.long())
        
        # Check for finished sequences
        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self.end_token_id:
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        
        if len(self.finished) >= self.n_top:
            self.finished.sort(key=lambda a: -a[0])
            for s, t, k in self.finished[:self.n_top]:
                if t > self.min_length:
                    self.top_sentence_ended = True
    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            current_beam_size = self.current_scores.size(0)
            while len(self.finished) < minimum and i < current_beam_size:
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1
        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hypothesis(self, timestep, k):
        hypothesis = []
        current_k = int(k) if isinstance(k, torch.Tensor) else k  # ← Ensure k is int
        
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            next_ys_size = self.next_ys[j+1].size(0)
            if current_k >= next_ys_size:
                current_k = next_ys_size - 1
            
            token = self.next_ys[j+1][current_k].item()  # ← Always use .item()
            hypothesis.append(int(token))  # ← Ensure it's Python int
            
            prev_ks_size = self.prev_ks[j].size(0)
            if current_k >= prev_ks_size:
                current_k = prev_ks_size - 1
            else:
                current_k = int(self.prev_ks[j][current_k].item())  # ← Convert to int
        
        return hypothesis[::-1]

def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()

    beam = Beam(
        beam_size=beam_size,
        min_length=0,
        n_top=candidates,
        ranker=None,
        start_token_id=sos_token,
        end_token_id=eos_token,
    )

    with torch.no_grad():
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):
            tgt_inp = beam.get_current_state()
            tgt_inp = tgt_inp.transpose(0, 1).to(device)
            
            assert tgt_inp.dtype == torch.long, f"Expected long tensor, got {tgt_inp.dtype}"
            
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypotheses = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypotheses.append(hypothesis)

    # Ensure we always return a list of integers
     # At the end of beamsearch function:
    if hypotheses and hypotheses[0] is not None:
        result = hypotheses[0]
        
        # Convert to Python list of ints
        if isinstance(result, np.ndarray):
            result = result.tolist()
        elif isinstance(result, (int, np.integer)):
            result = [int(result)]
        elif isinstance(result, list):
            result = [int(token) for token in result if token is not None]
        else:
            result = []
        
        # Remove SOS/EOS tokens
        if result and result[0] == sos_token:
            result = result[1:]
        if result and result[-1] == eos_token:
            result = result[:-1]
        
        return result  # Python list
    else:
        return []  # Python list
def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()
    device = img.device
    src = img

    with torch.no_grad():
        # Use the full model pipeline
        src = model.cnn(src)
        src = src.transpose(0, 1)
        src = model.project(src)
        memories = model.transformer.forward_encoder(src)

    sents = []
    for i in range(src.size(0)):
        memory = model.transformer.get_memory(memories, i)
        sent = beamsearch(
            memory,
            model,
            device,
            beam_size,
            candidates,
            max_seq_length,
            sos_token,
            eos_token,
        )
        # Ensure sent is a Python list, not numpy array
        if isinstance(sent, np.ndarray):
            sent = sent.tolist()
        elif not isinstance(sent, list):
            sent = [sent] if sent is not None else []
        
        sents.append(sent)

    # Don't convert to numpy array - return as list of lists
    return sents  # Return Python lists, not numpy array
def translate_beam_search(
    img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2
):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        src = src.transpose(0, 1)  # Add this line
        src = model.project(src)   # Add this line to apply the projection
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(
            memory,
            model,
            device,
            beam_size,
            candidates,
            max_seq_length,
            sos_token,
            eos_token,
        )

    return sent


def translate_beam_search(
    img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2
):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(
            memory,
            model,
            device,
            beam_size,
            candidates,
            max_seq_length,
            sos_token,
            eos_token,
        )

    return sent




def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
        ):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to("cpu")

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs