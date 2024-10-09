from typing import List, Tuple
import os
import csv
import torch
import torch.nn as nn
import torch.utils.data as data
import math
import random
import pandas as pd
from tqdm import tqdm
from consts import PAD_VALUE
from utils import read_jsonl
from datasets import load_dataset, concatenate_datasets

def save_sampled_sents(filename: str, sents: List[str]): 
    dir = os.path.split(filename)[0]
    if dir: 
        os.makedirs(dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f: 
        for sent in sents: 
            if sent: 
                f.write(sent)
                f.write('\n')

def load_sampled_sents(filename: str) -> List[str]: 
    with open(filename, 'r', encoding='utf-8') as f: 
        sents = [line.strip() for line in f if line]

    return sents

def load_sampled_sents_2(filename: str) -> List[Tuple]: 
    with open(filename, 'r', encoding='utf-8') as f: 
        examples = [tuple(line.strip().split('\t')) for line in f if line]

    return examples 

def load_wg_stats(filename): 
    pcts = {}
    with open(filename, 'r', encoding='utf-8') as f: 
        for line in f.readlines()[1:]: 
            if line: 
                split_line = line.strip().split('\t')
                pcts[split_line[0]] = float(split_line[1]) # using the bergsma proportion
    return pcts

def load_sents_wg(filename: str) -> List[Tuple]: 
    sents = []
    occs = []
    with open(filename, 'r', encoding='utf-8') as f: 
        for line in f.readlines()[1:]: 
            if line: 
                split_line = line.strip().split('\t')
                sents.append(split_line[1])
                occs.append(split_line[0].split('.')[0])

    sent_triples = []
    trip_occ = []
    for i in range(0, len(sents), 3): 
        sent_triples.append((sents[i], sents[i+1], sents[i+2]))
        trip_occ.append(occs[i])
    
    return sent_triples, trip_occ

def load_sents_crows(filename: str): 
    with open(filename, 'r', encoding='utf-8', newline='') as csvfile: 
        csv_reader = csv.reader(csvfile)
        lines = list(csv_reader)
    sent_pairs = [(row[1], row[2]) for row in lines]
    return sent_pairs


class StereoSetDataset(data.Dataset): 
    def __init__(self, dataset_file, tokenizer, max_len=64, dev=True, proportion_dev=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        examples = load_sampled_sents_2(dataset_file)
        
        tokenized_mask = tokenizer.mask_token_id
        tokenized_unk = tokenizer.unk_token_id

        sents, target_words = [list(unzipped) for unzipped in zip(*examples)]

        tokenized_sents = [tokenizer(sent, add_special_tokens=False)['input_ids'][:max_len] for sent in sents]
        tokenized_target_words = [tokenizer(target, add_special_tokens=False)['input_ids'] for target in target_words]

        inds = [-1]*len(tokenized_target_words)
        for i, (tokenized_sent, tokenized_target_word) in enumerate(zip(tokenized_sents, tokenized_target_words)): 
            if len(tokenized_target_word)!=1: 
                inds[i] = -2
                continue
            
            tokenized_target = tokenized_target_word[0]

            if tokenized_target==tokenized_unk: 
                continue

            token_indices = [i for i, token in enumerate(tokenized_sent) if token==tokenized_target]
            if len(token_indices)!=1: 
                inds[i] = -3
                continue
            
            inds[i] = token_indices[0]
        
        for i in range(2, len(inds), 3): 
            if inds[i]<0 or inds[i-1]<0 or inds[i-2]<0: 
                inds[i] = -4
                inds[i-1] = -4
                inds[i-2] = -4


        kept_sents = [tokenized_sent for i, tokenized_sent in enumerate(tokenized_sents) if inds[i]>=0]
        kept_inds = [ind for ind in inds if ind>=0]
        kept_target_tokens = [tokens[0] for i, tokens in enumerate(tokenized_target_words) if inds[i]>=0]

        for sent, ind in zip(kept_sents, kept_inds): 
            sent[ind] = tokenized_mask

        examples = list(zip(kept_sents, kept_inds, kept_target_tokens))
        example_count = len(examples)//3
        if example_count*3!=len(examples): 
            raise ValueError('malformed stereoset input file?')
        cutoff = math.floor(example_count * proportion_dev)*3

        if dev: 
            self.examples = examples[:cutoff]
        else: 
            self.examples = examples[cutoff:]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sents, inds, target_tokens = [list(unzipped) for unzipped in zip(*batch)]

            seqs = [torch.tensor(sent) for sent in sents]
            inds = torch.LongTensor(inds)
            target_tokens = torch.LongTensor(target_tokens)

            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0
            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE

            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class CrowsDataset(data.Dataset): 
    def __init__(self, dataset_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        examples = load_sents_crows(dataset_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
        self.tokenized_unk = tokenizer('[UNK]', add_special_tokens=False)['input_ids'][0]

        sent_pairs = []
        inds = []
        target_token_pairs = []
        skipped_count = 0
        for first_sent, second_sent in examples: 
            ind, (t_first, first), (t_second, second) = self._tokenize_and_find_diff(first_sent, second_sent)
            if ind is None: 
                skipped_count += 1
                continue
            sent_pairs.append((t_first, t_second))
            inds.append(ind)
            target_token_pairs.append((first, second))
        print(f'Skipped {skipped_count}/{len(examples)}')

        for (first, second), ind in zip(sent_pairs, inds): 
            first[ind] = tokenized_mask
            second[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))

    def _tokenize_and_find_diff(self, first, second): 
        tokenized_first = self.tokenizer(first, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_second = self.tokenizer(second, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_first)==len(tokenized_second): 
            found_i, found_f, found_s = None, None, None
            for i, (f, s) in enumerate(zip(tokenized_first, tokenized_second)): 
                if f==s: 
                    continue
                else: 
                    if found_i is not None: # only for single token diffs
                        return None, (None, None), (None, None)
                    found_i = i
                    found_f = f
                    found_s = s
            if found_i is None or found_f==self.tokenized_unk or found_s==self.tokenized_unk: 
                return None, (None, None), (None, None)
            return i, (tokenized_first, found_f), (tokenized_second, found_s)
        return None, (None, None), (None, None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]

            seqs = []
            for first_sent, second_sent in sent_pairs: 
                seqs.append(torch.tensor(first_sent))
                seqs.append(torch.tensor(second_sent))
            full_inds = []
            for ind in inds: 
                full_inds.append(ind)
                full_inds.append(ind)
            inds = torch.LongTensor(full_inds)
            target_tokens = []
            for first_target, second_target in target_token_pairs: 
                target_tokens.append(first_target)
                target_tokens.append(second_target)
            target_tokens = torch.LongTensor(target_tokens)

            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0
            
            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE
            
            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class WGDataset(data.Dataset): 
    def __init__(self, dataset_file, stats_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        sent_triples, occupations = load_sents_wg(dataset_file)
        stat_pcts = load_wg_stats(stats_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

        sent_pairs = []
        inds = []
        target_token_pairs = []
        for sent_trip, occ in zip(sent_triples, occupations): 
            male, female, neutral = sent_trip
            ind, (t_male, m), (t_female, f), (t_neutral, n) = self._tokenize_and_find_diff(male, female, neutral)
            if ind is None: 
                continue
            if stat_pcts[occ]<50: 
                disadvantaged = t_female
                disadvantaged_token = f
                advantaged = t_male
                advantaged_token = m
            else: 
                disadvantaged = t_male
                disadvantaged_token = m
                advantaged = t_female
                advantaged_token = f
            sent_pairs.append((disadvantaged, advantaged))
            inds.append(ind)
            target_token_pairs.append((disadvantaged_token, advantaged_token))

        for (disadvantaged, advantaged), ind in zip(sent_pairs, inds): 
            disadvantaged[ind] = tokenized_mask
            advantaged[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))
    
    def _tokenize_and_find_diff(self, male, female, neutral): 
        tokenized_male = self.tokenizer(male, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_female = self.tokenizer(female, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_neutral = self.tokenizer(neutral, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_male)==len(tokenized_female)==len(tokenized_neutral): 
            for i, (m, f, n) in enumerate(zip(tokenized_male, tokenized_female, tokenized_neutral)): 
                if m==f==n: 
                    continue
                else: 
                    return i, (tokenized_male, m), (tokenized_female, f), (tokenized_neutral, n)
        return None, None, None, None

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]
            disadvantaged_sents, advantaged_sents = [list(unzipped) for unzipped in zip(*sent_pairs)]
            disadvantaged_targets, advantaged_targets = [list(unzipped) for unzipped in zip(*target_token_pairs)]

            disadvantaged_seqs = [torch.tensor(sent) for sent in disadvantaged_sents]
            advantaged_seqs = [torch.tensor(sent) for sent in advantaged_sents]
            inds = torch.LongTensor(inds)
            disadvantaged_target_tokens = torch.LongTensor(disadvantaged_targets)
            advantaged_target_tokens = torch.LongTensor(advantaged_targets)

            padded_disadvantaged_seqs = nn.utils.rnn.pad_sequence(disadvantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_advantaged_seqs = nn.utils.rnn.pad_sequence(advantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            disadvantaged_attention_masks = torch.ones_like(padded_disadvantaged_seqs)
            advantaged_attention_masks = torch.ones_like(padded_advantaged_seqs)
            disadvantaged_attention_masks[padded_disadvantaged_seqs==tokenizer.pad_token_id] = 0
            advantaged_attention_masks[padded_advantaged_seqs==tokenizer.pad_token_id] = 0
            
            disadvantaged_labels = padded_disadvantaged_seqs.detach().clone()
            disadvantaged_labels[padded_disadvantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            advantaged_labels = padded_advantaged_seqs.detach().clone()
            advantaged_labels[padded_advantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            
            return (padded_disadvantaged_seqs, padded_advantaged_seqs), \
                    (disadvantaged_attention_masks, advantaged_attention_masks), \
                    inds, \
                    (disadvantaged_target_tokens, advantaged_target_tokens), \
                    (disadvantaged_labels, advantaged_labels)
        return collate_batch


BBQ_CFGS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", 
            "Race_ethnicity", "Religion", "SES", "Sexual_orientation"] # TODO - Include cross domain: "Race_x_gender", "Race_x_SES"

def load_sents_bbq(dataset_dir: str, cfgs: list=["Age"], sample=False):
    # TODO - train and test split
    cfg_datasets = []
    for cfg in cfgs:
        cfg_dataset = read_jsonl(os.path.join(dataset_dir, f'{cfg}.jsonl'))
        if sample:
            # Sample frac_sample fraction of data for test run
            frac_sample = 0.05
            cfg_dataset = random.sample(cfg_dataset, int(frac_sample*len(cfg_dataset)))
        cfg_datasets.append(pd.DataFrame({"category": cfg, "data": cfg_dataset}))
    dataset = pd.concat(cfg_datasets)

    # Read BBQ metadata for target index
    metadata = pd.read_csv(os.path.join(dataset_dir, 'bbq_metadata.csv'))
    # Keep first entry to remove duplicates for "Gender_identity" category - since they have same target_loc
    metadata = metadata.drop_duplicates(subset=["category", "example_id"], keep="first")
    metadata = metadata[["category", "example_id", "target_loc"]]

    # Add merge cols and merge with dataset
    dataset["example_id"] = dataset["data"].apply(lambda x: x["example_id"])
    print(dataset.shape)
    dataset = dataset.merge(right=metadata, on=["category", "example_id"], how="left")
    print(dataset.shape)
    assert dataset[["category", "example_id"]].drop_duplicates().shape[0] == dataset.shape[0]
    # Remove samples with empty target_loc
    print(f"Dropping {dataset.isnull().sum()['target_loc']} samples due to unknown target index")
    dataset.dropna(subset=["target_loc"], inplace=True)
    print(dataset.shape)
    assert dataset.isnull().sum().sum() == 0
    
    # Based on approach [1] (Simple)
    ANS_FORMAT_1 = 'Choose among the following two options: A: {option_A}; B: {option_B}. Answer: Option {ans}'
    ANS_OPTIONS = ['A', 'B']
    sent_pairs = []
    categories = []
    elms = []
    # sample_printed = False
    for _, row in dataset.iterrows():
        # Only process ambig examples
        elm = row['data']
        # TODO - Also experiment with only nonneg question polarity (positive questions)
        if elm['context_condition'] == 'ambig':
            # Identify stereotyped answer
            stereo_ans_key = f'ans{int(row["target_loc"])}'
            antistereo_ans_key = list(set.difference(set([f'ans{idx}' for idx in range(3)]), set([stereo_ans_key, f'ans{elm["label"]}'])))[0]
            # Random shuffle to ensure, adv and disadv answers dont get assigned to the same option for every pair
            ans_keys = [stereo_ans_key, antistereo_ans_key]
            random.shuffle(ans_keys)
            stereo_sent = f'{elm["context"].strip()} {elm["question"].strip()} ' + \
                ANS_FORMAT_1.format(option_A=elm[ans_keys[0]], option_B=elm[ans_keys[1]], ans=ANS_OPTIONS[ans_keys.index(stereo_ans_key)])
            antistereo_sent = f'{elm["context"].strip()} {elm["question"].strip()} ' + \
                ANS_FORMAT_1.format(option_A=elm[ans_keys[0]], option_B=elm[ans_keys[1]], ans=ANS_OPTIONS[ans_keys.index(antistereo_ans_key)])
            # if (row['category'] == "Gender_identity") and (not sample_printed):
            #     print(stereo_sent)
            #     print(antistereo_sent)
            #     sample_printed = True
            # Stereotyped sentence is always advantaged
            sent_pairs.append(((antistereo_sent, "disadvantaged"), (stereo_sent, "advantaged")))
            categories.append(row['category'])
            elms.append(row)

    print("# sentence pairs across protected groups:\n")
    print(pd.Series(categories).value_counts())

    if not sample:
        assert len(sent_pairs) == len(dataset)//2, f"No. of pairs ({len(sent_pairs)}) is not half of whole dataset ({len(dataset)//2})"
    assert len(sent_pairs) == len(categories)
    
    # # Count distribution of options for adv, disadv sents - The output should be as given below - ALMOST UNIFORM [REMOVE]
    # # Counter({'A': 7840, 'B': 7838})
    # # Counter({'B': 7840, 'A': 7838})
    # disadv_option = []
    # adv_option = []
    # for (disadv, adv) in sent_pairs:
    #     disadv_option.append(disadv[0].split(' ')[-1])
    #     adv_option.append(adv[0].split(' ')[-1])
    # from collections import Counter
    # print(Counter(disadv_option))
    # print(Counter(adv_option))

    # # Code snippet to view samples [REMOVE]
    # print(dataset.shape[0])
    # print(len(sent_pairs))
    # random.seed(41)
    # sample_id = random.randint(0, len(sent_pairs)-1)
    # print(sent_pairs[sample_id])
    # print(elms[sample_id].to_dict())

    return sent_pairs, categories

                
class BBQDataset(data.Dataset):
    # ------------------ #
    # For some examples the length of the tokenized target str is different, 
    # hence we can go ahead with two choices for formatting the dataset:
    #   [1] (Simple) Assign each answer a letter choice A, B and ask the model to choose, 
    #   this makes it simple since the adv and disadv will only differ in a single token
    #   [2] (Complex) Aggregate logits for tokens of those target str which result in more than one token when tokenized
    # Lets test method with first approach and then move to the second approach if it shows promise
    # ------------------ #

    def __init__(self, dataset_name, tokenizer, max_len=115, sample=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        sent_tup_pairs, categories = load_sents_bbq(dataset_name, cfgs=BBQ_CFGS, sample=sample)

        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

        # # Max tokenized length (vanilla): 93, hence max_len is set to 100
        # # Max tokenized length (with extra instruction from approach 1): 114, hence max_len is set to 115
        # # Get max length
        # tokenized_len = []
        # for (disadv, adv) in sent_tup_pairs:
        #     tokenized_len.append(len(self.tokenizer(disadv[0], add_special_tokens=False)['input_ids']))
        #     tokenized_len.append(len(self.tokenizer(adv[0], add_special_tokens=False)['input_ids']))
        # print(f"Max tokenized length: {max(tokenized_len)}")

        sent_pairs = []
        inds = []
        target_token_pairs = []
        mismatch_count = 0
        for (disadv, adv) in tqdm(sent_tup_pairs, desc="Tokenizing and finding target index"):
            assert ((disadv[1]=='disadvantaged') and (adv[1]=='advantaged')), "Incorrect adv/disadv labels"

            ind, (disadvantaged, disadvantaged_token), (advantaged, advantaged_token) = self._tokenize_and_find_diff(disadv[0], adv[0])
            if ind is None:
                # Disadv and adv sentence tokens length mismatched
                mismatch_count += 1
                continue

            sent_pairs.append((disadvantaged, advantaged))
            inds.append(ind)
            target_token_pairs.append((disadvantaged_token, advantaged_token))

        assert mismatch_count == 0, f'Some sent pairs have adv/disadv tokens length mismatched: {mismatch_count}'

        # No need to do this for LM, since the target token is the last token
        # # Replace target token with MASK token
        # for (disadvantaged, advantaged), ind in zip(sent_pairs, inds): 
        #     disadvantaged[ind] = tokenized_mask
        #     advantaged[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))

    def _tokenize_and_find_diff(self, disadv, adv): 
        tokenized_disadv = self.tokenizer(disadv, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_adv = self.tokenizer(adv, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_disadv)==len(tokenized_adv): 
            for i, (d, a) in enumerate(zip(tokenized_disadv, tokenized_adv)): 
                if d==a: 
                    continue
                else: 
                    return i, (tokenized_disadv, d), (tokenized_adv, a)
        print(disadv, tokenized_disadv)
        print(adv, tokenized_adv)
        return None, (None, None), (None, None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]
            disadvantaged_sents, advantaged_sents = [list(unzipped) for unzipped in zip(*sent_pairs)]
            disadvantaged_targets, advantaged_targets = [list(unzipped) for unzipped in zip(*target_token_pairs)]

            disadvantaged_seqs = [torch.tensor(sent) for sent in disadvantaged_sents]
            advantaged_seqs = [torch.tensor(sent) for sent in advantaged_sents]
            inds = torch.LongTensor(inds)
            disadvantaged_target_tokens = torch.LongTensor(disadvantaged_targets)
            advantaged_target_tokens = torch.LongTensor(advantaged_targets)

            padded_disadvantaged_seqs = nn.utils.rnn.pad_sequence(disadvantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_advantaged_seqs = nn.utils.rnn.pad_sequence(advantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            disadvantaged_attention_masks = torch.ones_like(padded_disadvantaged_seqs)
            advantaged_attention_masks = torch.ones_like(padded_advantaged_seqs)
            disadvantaged_attention_masks[padded_disadvantaged_seqs==tokenizer.pad_token_id] = 0
            advantaged_attention_masks[padded_advantaged_seqs==tokenizer.pad_token_id] = 0
            
            disadvantaged_labels = padded_disadvantaged_seqs.detach().clone()
            disadvantaged_labels[padded_disadvantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            advantaged_labels = padded_advantaged_seqs.detach().clone()
            advantaged_labels[padded_advantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            
            return (padded_disadvantaged_seqs, padded_advantaged_seqs), \
                    (disadvantaged_attention_masks, advantaged_attention_masks), \
                    inds, \
                    (disadvantaged_target_tokens, advantaged_target_tokens), \
                    (disadvantaged_labels, advantaged_labels)
        return collate_batch


