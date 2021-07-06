import torch
from time import time
from datetime import timedelta

from cytoolz import concat


def sc_validate(agent, abstractor, loader, entity=False, bert=False):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            greedy_inputs = []
            for idx, raw_arts in enumerate(art_batch):
                greedy, sample, log_probs = agent(raw_arts, sample_time=1, validate=True)
                if entity or bert:
                    raw_arts = raw_arts[0]
                # sample = sample[0]
                # log_probs = log_probs[0]
                greedy_sents = [raw_arts[ind] for ind in greedy]
                # greedy_sents = list(concat(greedy_sents))
                greedy_sents = [word for sent in greedy_sents for word in sent]
                greedy_inputs.append(greedy_sents)
            with torch.no_grad():
                greedy_outputs = abstractor(greedy_inputs)
            greedy_abstracts = []
            for greedy_sents in greedy_outputs:
                greedy_sents = sent_tokenize(' '.join(greedy_sents))
                greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
                greedy_abstracts.append(greedy_sents)
            for idx, greedy_sents in enumerate(greedy_abstracts):
                abss = abs_batch[idx]
                bs = compute_rouge_n(list(concat(greedy_sents)), list(concat(abss)))
                avg_reward += bs
                i += 1
    avg_reward /= (i / 100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time() - start)), avg_reward))
    return {'reward': avg_reward}


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices) - 1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]
            all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j + n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i / 100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time() - start)), avg_reward))
    return {'reward': avg_reward}


def train_step(self, sample_time=1):
        torch.autograd.set_detect_anomaly(True)

        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]

        def sum_id2word(raw_article_sents, decs, attns, id2word):
            if self._bert:
                dec_sents = []
                for i, raw_words in enumerate(raw_article_sents):
                    dec = []
                    for id_, attn in zip(decs, attns):
                        if id_[i] == self._end:
                            break
                        elif id_[i] == self._unk:
                            dec.append(argmax(raw_words, attn[i]))
                        else:
                            dec.append(id2word[id_[i].item()])
                    dec_sents.append(dec)
            else:
                dec_sents = []
                for i, raw_words in enumerate(raw_article_sents):
                    dec = []
                    for id_, attn in zip(decs, attns):
                        if id_[i] == self._end:
                            break
                        elif id_[i] == self._unk:
                            dec.append(argmax(raw_words, attn[i]))
                        else:
                            dec.append(id2word[id_[i].item()])
                    dec_sents.append(dec)
            return dec_sents

        def pack_seq(seq_list):
            return torch.cat([_.unsqueeze(1) for _ in seq_list], 1)

        # forward pass of model
        self._net.train()
        # self._net.zero_grad()
        total_loss = None
        for i in range(self._accumulate_g_step):
            fw_args, bw_args = next(self._batches)
            raw_articles = bw_args[0]
            id2word = bw_args[1]
            raw_targets = bw_args[2]
            if self._reward_fn is not None:
                questions = bw_args[3]
            targets = bw_args[4]

            # encode
            # attention, init_dec_states, nodes = self._net.encode_general(*fw_args)
            # fw_args += (attention, init_dec_states, nodes)
            # _init_dec_states = ((init_dec_states[0][0].clone(), init_dec_states[0][1].clone()), init_dec_states[1].clone())
            with torch.no_grad():
                # g_fw_args = fw_args + (attention, _init_dec_states, nodes, False)
                # greedies, greedy_attns = self._net.rl_step(*g_fw_args)
                greedies, greedy_attns = self._net.greedy(*fw_args)
            greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns, id2word)
            bl_scores = []
            if self._reward_fn is not None:
                bl_reward_scores = []
                bl_reward_inputs = []
            if self._local_coh_fun is not None:
                bl_local_coh_scores = []
            for baseline, target in zip(greedy_sents, raw_targets):
                if self._bert:
                    text = ''.join(baseline)
                    baseline = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                                 errors=self._tokenizer.errors)
                    baseline = baseline.strip().lower().split(' ')
                    text = ''.join(target)
                    target = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                               errors=self._tokenizer.errors)
                    target = target.strip().lower().split(' ')

                bss = sent_tokenize(' '.join(baseline))
                tgs = sent_tokenize(' '.join(target))
                if self._reward_fn is not None:
                    bl_reward_inputs.append(bss)
                if self._local_coh_fun is not None:
                    local_coh_score = self._local_coh_fun(bss)
                    bl_local_coh_scores.append(local_coh_score)
                bss = [bs.split(' ') for bs in bss]
                tgs = [tg.split(' ') for tg in tgs]

                # bl_score = compute_rouge_l_summ(bss, tgs)
                bl_score = (self._w8[2] * compute_rouge_l_summ(bss, tgs) + \
                            self._w8[0] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1) + \
                            self._w8[1] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
                bl_scores.append(bl_score)
            bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)

            # sample
            # s_fw_args = fw_args + (attention, init_dec_states, nodes, True)
            # samples, sample_attns, seqLogProbs = self._net.rl_step(*s_fw_args)
            fw_args += (self._ml_loss,)
            if self._ml_loss:
                samples, sample_attns, seqLogProbs, ml_logit = self._net.sample(*fw_args)
            else:
                samples, sample_attns, seqLogProbs = self._net.sample(*fw_args)
            sample_sents = sum_id2word(raw_articles, samples, sample_attns, id2word)
            sp_seqs = pack_seq(samples)
            _masks = (sp_seqs > PAD).float()
            sp_seqLogProb = pack_seq(seqLogProbs)
            # loss_nll = - sp_seqLogProb.squeeze(2)
            loss_nll = - sp_seqLogProb.squeeze(2) * _masks.detach().type_as(sp_seqLogProb)
            sp_scores = []
            if self._reward_fn is not None:
                sp_reward_inputs = []
            for sample, target in zip(sample_sents, raw_targets):
                if self._bert:
                    text = ''.join(sample)
                    sample = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                               errors=self._tokenizer.errors)
                    sample = sample.strip().lower().split(' ')
                    text = ''.join(target)
                    target = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                               errors=self._tokenizer.errors)
                    target = target.strip().lower().split(' ')

                sps = sent_tokenize(' '.join(sample))
                tgs = sent_tokenize(' '.join(target))
                if self._reward_fn is not None:
                    sp_reward_inputs.append(sps)

                sps = [sp.split(' ') for sp in sps]
                tgs = [tg.split(' ') for tg in tgs]
                # sp_score = compute_rouge_l_summ(sps, tgs)
                sp_score = (self._w8[2] * compute_rouge_l_summ(sps, tgs) + \
                            self._w8[0] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=1) + \
                            self._w8[1] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=2))
                sp_scores.append(sp_score)
            sp_scores = torch.tensor(sp_scores, dtype=torch.float32, device=greedy_attns[0].device)
            if self._reward_fn is not None:
                sp_reward_scores, bl_reward_scores = self._reward_fn.score_two_seqs(questions, sp_reward_inputs,
                                                                                    bl_reward_inputs)
                sp_reward_scores = torch.tensor(sp_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)
                bl_reward_scores = torch.tensor(bl_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)

            reward = sp_scores.view(-1, 1) - bl_scores.view(-1, 1)
            if self._reward_fn is not None:
                reward += self._reward_w8 * (sp_reward_scores.view(-1, 1) - bl_reward_scores.view(-1, 1))
            reward.requires_grad_(False)

            loss = reward.contiguous().detach() * loss_nll
            loss = loss.sum()
            full_length = _masks.data.float().sum()
            loss = loss / full_length
            if self._ml_loss:
                ml_loss = self._ml_criterion(ml_logit, targets)
                loss += self._ml_loss_w8 * ml_loss.mean()
            # if total_loss is None:
            #     total_loss = loss
            # else:
            #     total_loss += loss

            loss = loss / self._accumulate_g_step
            loss.backward()

        log_dict = {}
        if self._reward_fn is not None:
            log_dict['reward'] = bl_scores.mean().item()
            log_dict['question_reward'] = bl_reward_scores.mean().item()
            log_dict['sample_question_reward'] = sp_reward_scores.mean().item()
            log_dict['sample_reward'] = sp_scores.mean().item()
        else:
            log_dict['reward'] = bl_scores.mean().item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())

        self._opt.step()
        self._net.zero_grad()
        # torch.cuda.empty_cache()

        return log_dict
